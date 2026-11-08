from std.memory import UnsafePointer, alloc
from std.math import abs, exp, cos, sin
import std.time
from diffusion_kernels import (
    build_active_canvas_map,
    update_locked_mask,
    gather_active,
    scatter_active_sampler_outputs,
    q4_gemm_l1_2x4,
)


def apply_rope_test(
    x: UnsafePointer[Float16, MutAnyOrigin],
    num_heads: Int,
    head_dim: Int,
    pos: Int,
):
    var half_dim = head_dim // 2
    for h in range(num_heads):
        var h_ptr = x + h * head_dim
        for d in range(half_dim):
            var freq = Float32(pos) * Float32(0.001) / Float32(d + 1)
            var c = cos(freq)
            var s = sin(freq)
            var x1 = Float32(h_ptr[d])
            var x2 = Float32(h_ptr[d + half_dim])
            h_ptr[d] = Float16(x1 * c - x2 * s)
            h_ptr[d + half_dim] = Float16(x2 * c + x1 * s)


def softmax_test(x: UnsafePointer[Float32, MutAnyOrigin], n: Int):
    var max_v = x[0]
    for i in range(1, n):
        if x[i] > max_v:
            max_v = x[i]
    var sum_e: Float32 = 0.0
    for i in range(n):
        var e = exp(x[i] - max_v)
        x[i] = e
        sum_e += e
    var inv = Float32(1.0) / (sum_e + 1e-10)
    for i in range(n):
        x[i] = x[i] * inv


def main() raises:
    print("=== Test & Benchmark: Fast-dLLM / dKV-Cache Active-Canvas Compaction (llm-mojo-2tc.3) ===")
    comptime CANVAS_LEN = 256
    comptime NUM_PROMPT = 16
    comptime SEQ_LEN = NUM_PROMPT + CANVAS_LEN
    comptime EMBED_DIM = 2816
    comptime NUM_HEADS = 16
    comptime NUM_KV_HEADS = 8
    comptime HEAD_DIM = 256
    comptime Q_SIZE = NUM_HEADS * HEAD_DIM
    comptime KV_STRIDE = NUM_KV_HEADS * HEAD_DIM

    # 1. Test update_locked_mask & build_active_canvas_map
    var eb_mask = alloc[Bool](CANVAS_LEN)
    var locked_mask = alloc[Bool](CANVAS_LEN)
    var history = alloc[Int](2 * CANVAS_LEN)
    var token_entropy = alloc[Float32](CANVAS_LEN)
    var active_canvas_idx = alloc[Int](CANVAS_LEN)

    for t in range(CANVAS_LEN):
        locked_mask[t] = False
        # Lock 3 out of every 4 tokens (192 locked, 64 active)
        if (t % 4) != 3:
            eb_mask[t] = True
            history[t] = 100 + t
            history[CANVAS_LEN + t] = 100 + t
            token_entropy[t] = 0.08
        else:
            eb_mask[t] = False
            history[t] = 100 + t
            history[CANVAS_LEN + t] = 200 + t
            token_entropy[t] = 2.5

    var num_locked = update_locked_mask(
        eb_mask, history, token_entropy, locked_mask, CANVAS_LEN, 2
    )
    var active_len = build_active_canvas_map(
        locked_mask, active_canvas_idx, CANVAS_LEN
    )
    print("  [Lock & Map] num_locked =", num_locked, "| active_len =", active_len)
    if num_locked != 192 or active_len != 64:
        raise Error("Expected num_locked=192 and active_len=64!")
    for b in range(active_len):
        if active_canvas_idx[b] != b * 4 + 3:
            raise Error("Unexpected active_canvas_idx mapping!")

    # 2. Test exact equivalence of full 256-token evaluation vs compacted 64-token evaluation
    # Allocate full and compacted buffers
    var full_norm = alloc[Float32](CANVAS_LEN * EMBED_DIM)
    var compact_norm = alloc[Float32](CANVAS_LEN * EMBED_DIM)
    for t in range(CANVAS_LEN):
        for i in range(EMBED_DIM):
            full_norm[t * EMBED_DIM + i] = Float32(((t * 17 + i * 13) % 97) - 48) * 0.02

    for b in range(active_len):
        var orig_t = active_canvas_idx[b]
        for i in range(EMBED_DIM):
            compact_norm[b * EMBED_DIM + i] = full_norm[orig_t * EMBED_DIM + i]

    var wq_packed = alloc[UInt8](Q_SIZE * (EMBED_DIM // 2))
    var wq_scales = alloc[Float32](Q_SIZE * (EMBED_DIM // 32))
    var wk_packed = alloc[UInt8](KV_STRIDE * (EMBED_DIM // 2))
    var wk_scales = alloc[Float32](KV_STRIDE * (EMBED_DIM // 32))
    var wv_packed = alloc[UInt8](KV_STRIDE * (EMBED_DIM // 2))
    var wv_scales = alloc[Float32](KV_STRIDE * (EMBED_DIM // 32))
    var wo_packed = alloc[UInt8](EMBED_DIM * (Q_SIZE // 2))
    var wo_scales = alloc[Float32](EMBED_DIM * (Q_SIZE // 32))

    for i in range(Q_SIZE * (EMBED_DIM // 2)):
        wq_packed[i] = UInt8((i * 31 + 7) & 0xFF)
    for i in range(Q_SIZE * (EMBED_DIM // 32)):
        wq_scales[i] = Float32(((i * 19) % 13) + 1) * 0.005
    for i in range(KV_STRIDE * (EMBED_DIM // 2)):
        wk_packed[i] = UInt8((i * 23 + 11) & 0xFF)
        wv_packed[i] = UInt8((i * 29 + 5) & 0xFF)
    for i in range(KV_STRIDE * (EMBED_DIM // 32)):
        wk_scales[i] = Float32(((i * 17) % 11) + 1) * 0.005
        wv_scales[i] = Float32(((i * 13) % 11) + 1) * 0.005
    for i in range(EMBED_DIM * (Q_SIZE // 2)):
        wo_packed[i] = UInt8((i * 37 + 3) & 0xFF)
    for i in range(EMBED_DIM * (Q_SIZE // 32)):
        wo_scales[i] = Float32(((i * 7) % 9) + 1) * 0.005

    var k_cache_ref = alloc[Float32](SEQ_LEN * KV_STRIDE)
    var v_cache_ref = alloc[Float32](SEQ_LEN * KV_STRIDE)
    var k_cache_opt = alloc[Float32](SEQ_LEN * KV_STRIDE)
    var v_cache_opt = alloc[Float32](SEQ_LEN * KV_STRIDE)

    # Populate prompt KV cache in both ref and opt
    for p in range(NUM_PROMPT * KV_STRIDE):
        var val_k = Float32((p % 41) - 20) * 0.03
        var val_v = Float32((p % 37) - 18) * 0.03
        k_cache_ref[p] = val_k
        k_cache_opt[p] = val_k
        v_cache_ref[p] = val_v
        v_cache_opt[p] = val_v

    var full_q = alloc[Float16](CANVAS_LEN * Q_SIZE)
    var full_k = alloc[Float16](CANVAS_LEN * KV_STRIDE)
    var full_v = alloc[Float16](CANVAS_LEN * KV_STRIDE)
    var full_attn_out = alloc[Float16](CANVAS_LEN * Q_SIZE)
    var full_proj_out = alloc[Float32](CANVAS_LEN * EMBED_DIM)

    var compact_q = alloc[Float16](CANVAS_LEN * Q_SIZE)
    var compact_k = alloc[Float16](CANVAS_LEN * KV_STRIDE)
    var compact_v = alloc[Float16](CANVAS_LEN * KV_STRIDE)
    var compact_attn_out = alloc[Float16](CANVAS_LEN * Q_SIZE)
    var compact_proj_out = alloc[Float32](CANVAS_LEN * EMBED_DIM)
    var scores_buf = alloc[Float32](SEQ_LEN)

    # Full 256-token pass (populates k_cache_ref / v_cache_ref and also pre-populates locked slots in k_cache_opt / v_cache_opt)
    var t0_full = std.time.perf_counter()
    q4_gemm_l1_2x4[DType.float32, DType.float16](wq_packed, wq_scales, full_norm, full_q, CANVAS_LEN, EMBED_DIM, Q_SIZE, 8)
    q4_gemm_l1_2x4[DType.float32, DType.float16](wk_packed, wk_scales, full_norm, full_k, CANVAS_LEN, EMBED_DIM, KV_STRIDE, 8)
    q4_gemm_l1_2x4[DType.float32, DType.float16](wv_packed, wv_scales, full_norm, full_v, CANVAS_LEN, EMBED_DIM, KV_STRIDE, 8)

    for t in range(CANVAS_LEN):
        var pos = NUM_PROMPT + t
        var q_ptr = full_q + t * Q_SIZE
        var k_ptr = full_k + t * KV_STRIDE
        var v_ptr = full_v + t * KV_STRIDE
        apply_rope_test(q_ptr, NUM_HEADS, HEAD_DIM, pos)
        apply_rope_test(k_ptr, NUM_KV_HEADS, HEAD_DIM, pos)
        for i in range(0, KV_STRIDE, 8):
            k_cache_ref.store[8](pos * KV_STRIDE + i, k_ptr.load[8](i).cast[DType.float32]())
            v_cache_ref.store[8](pos * KV_STRIDE + i, v_ptr.load[8](i).cast[DType.float32]())
            if locked_mask[t]:
                # Simulate locked tokens already stored in k_cache_opt / v_cache_opt from the step they locked
                k_cache_opt.store[8](pos * KV_STRIDE + i, k_ptr.load[8](i).cast[DType.float32]())
                v_cache_opt.store[8](pos * KV_STRIDE + i, v_ptr.load[8](i).cast[DType.float32]())

    var ratio = NUM_HEADS // NUM_KV_HEADS
    var num_simd_lanes = HEAD_DIM // 8
    for t in range(CANVAS_LEN):
        var q_load = full_q + t * Q_SIZE
        var attn_out_ptr = full_attn_out + t * Q_SIZE
        for h in range(NUM_HEADS):
            var h_off = h * HEAD_DIM
            var g_off = (h // ratio) * HEAD_DIM
            for tt in range(SEQ_LEN):
                var sc = SIMD[DType.float32, 8](0.0)
                for d in range(num_simd_lanes):
                    sc += q_load.load[8](h_off + d * 8).cast[DType.float32]() * k_cache_ref.load[8](tt * KV_STRIDE + g_off + d * 8)
                scores_buf[tt] = sc.reduce_add()
            softmax_test(scores_buf, SEQ_LEN)
            for d in range(num_simd_lanes):
                var acc = SIMD[DType.float32, 8](0.0)
                for tt in range(SEQ_LEN):
                    acc += SIMD[DType.float32, 8](scores_buf[tt]) * v_cache_ref.load[8](tt * KV_STRIDE + g_off + d * 8)
                attn_out_ptr.store[8](h_off + d * 8, acc.cast[DType.float16]())

    q4_gemm_l1_2x4[DType.float16, DType.float32](wo_packed, wo_scales, full_attn_out, full_proj_out, CANVAS_LEN, Q_SIZE, EMBED_DIM, 8)
    var dt_full_ms = (std.time.perf_counter() - t0_full) * 1000.0

    # Compacted 64-token pass (Fast-dLLM / dKV-Cache)
    var t0_opt = std.time.perf_counter()
    q4_gemm_l1_2x4[DType.float32, DType.float16](wq_packed, wq_scales, compact_norm, compact_q, active_len, EMBED_DIM, Q_SIZE, 8)
    q4_gemm_l1_2x4[DType.float32, DType.float16](wk_packed, wk_scales, compact_norm, compact_k, active_len, EMBED_DIM, KV_STRIDE, 8)
    q4_gemm_l1_2x4[DType.float32, DType.float16](wv_packed, wv_scales, compact_norm, compact_v, active_len, EMBED_DIM, KV_STRIDE, 8)

    for b in range(active_len):
        var orig_t = active_canvas_idx[b]
        var pos = NUM_PROMPT + orig_t
        var q_ptr = compact_q + b * Q_SIZE
        var k_ptr = compact_k + b * KV_STRIDE
        var v_ptr = compact_v + b * KV_STRIDE
        apply_rope_test(q_ptr, NUM_HEADS, HEAD_DIM, pos)
        apply_rope_test(k_ptr, NUM_KV_HEADS, HEAD_DIM, pos)
        for i in range(0, KV_STRIDE, 8):
            k_cache_opt.store[8](pos * KV_STRIDE + i, k_ptr.load[8](i).cast[DType.float32]())
            v_cache_opt.store[8](pos * KV_STRIDE + i, v_ptr.load[8](i).cast[DType.float32]())

    for b in range(active_len):
        var q_load = compact_q + b * Q_SIZE
        var attn_out_ptr = compact_attn_out + b * Q_SIZE
        for h in range(NUM_HEADS):
            var h_off = h * HEAD_DIM
            var g_off = (h // ratio) * HEAD_DIM
            for tt in range(SEQ_LEN):
                var sc = SIMD[DType.float32, 8](0.0)
                for d in range(num_simd_lanes):
                    sc += q_load.load[8](h_off + d * 8).cast[DType.float32]() * k_cache_opt.load[8](tt * KV_STRIDE + g_off + d * 8)
                scores_buf[tt] = sc.reduce_add()
            softmax_test(scores_buf, SEQ_LEN)
            for d in range(num_simd_lanes):
                var acc = SIMD[DType.float32, 8](0.0)
                for tt in range(SEQ_LEN):
                    acc += SIMD[DType.float32, 8](scores_buf[tt]) * v_cache_opt.load[8](tt * KV_STRIDE + g_off + d * 8)
                attn_out_ptr.store[8](h_off + d * 8, acc.cast[DType.float16]())

    q4_gemm_l1_2x4[DType.float16, DType.float32](wo_packed, wo_scales, compact_attn_out, compact_proj_out, active_len, Q_SIZE, EMBED_DIM, 8)
    var dt_opt_ms = (std.time.perf_counter() - t0_opt) * 1000.0

    # Verify exact numerical identity on all 64 active tokens
    var max_err: Float32 = 0.0
    for b in range(active_len):
        var orig_t = active_canvas_idx[b]
        for i in range(EMBED_DIM):
            var d = abs(full_proj_out[orig_t * EMBED_DIM + i] - compact_proj_out[b * EMBED_DIM + i])
            if d > max_err:
                max_err = d

    print("  [Compacted Attention + QKV + Wo (64/256 active)] max_err =", max_err, "| Full 256: ", dt_full_ms, "ms -> Compacted 64: ", dt_opt_ms, "ms (", dt_full_ms / dt_opt_ms, "x speedup)")
    if max_err > 1e-5:
        raise Error("Compacted active-canvas evaluation diverged from full evaluation!")

    eb_mask.free()
    locked_mask.free()
    history.free()
    token_entropy.free()
    active_canvas_idx.free()
    full_norm.free()
    compact_norm.free()
    wq_packed.free()
    wq_scales.free()
    wk_packed.free()
    wk_scales.free()
    wv_packed.free()
    wv_scales.free()
    wo_packed.free()
    wo_scales.free()
    k_cache_ref.free()
    v_cache_ref.free()
    k_cache_opt.free()
    v_cache_opt.free()
    full_q.free()
    full_k.free()
    full_v.free()
    full_attn_out.free()
    full_proj_out.free()
    compact_q.free()
    compact_k.free()
    compact_v.free()
    compact_attn_out.free()
    compact_proj_out.free()
    scores_buf.free()
    print("=== ALL TESTS PASSED! ===")
