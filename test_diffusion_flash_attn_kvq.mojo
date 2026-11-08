from std.memory import UnsafePointer, alloc
from std.collections import InlineArray
from std.math import sqrt, exp, cos, sin, abs
from std.algorithm import parallelize
import std.time
from diffusion_kernels import (
    rms_norm,
    rms_norm_noscale,
    prep_qkv_norm_rope_kvq_batch,
    flash_dllm_bidirectional_attn,
)


@always_inline
def ref_apply_rope(
    vec: UnsafePointer[Float16, MutAnyOrigin],
    num_heads: Int,
    head_dim: Int,
    pos: Int,
    is_global: Bool,
):
    var half_dim = head_dim // 2
    var base: Float32 = Float32(1000000.0) if is_global else Float32(10000.0)
    var rope_dim = 512 if is_global else 256
    for h in range(num_heads):
        var h_offset = h * head_dim
        for i in range(half_dim):
            var theta: Float32 = 0.0
            if (not is_global) or (i < 64):
                theta = Float32(pos) / (base ** (Float32(2 * i) / Float32(rope_dim)))
            var cos_theta = cos(theta)
            var sin_theta = sin(theta)
            var val1 = Float32(vec[h_offset + i])
            var val2 = Float32(vec[h_offset + half_dim + i])
            vec[h_offset + i] = Float16(val1 * cos_theta - val2 * sin_theta)
            vec[h_offset + half_dim + i] = Float16(val2 * cos_theta + val1 * sin_theta)


@always_inline
def ref_softmax(scores: UnsafePointer[Float32, MutAnyOrigin], size: Int):
    var max_val = scores[0]
    for i in range(1, size):
        if scores[i] > max_val:
            max_val = scores[i]
    var sum_exp: Float32 = 0.0
    for i in range(size):
        scores[i] = exp(scores[i] - max_val)
        sum_exp += scores[i]
    for i in range(size):
        scores[i] /= (sum_exp + 1e-10)


def main() raises:
    print("=== Test & Benchmark: Flash-dLLM + dLLM-KVQ Bidirectional Attention (llm-mojo-2tc.9) ===")
    comptime CANVAS_LEN = 256
    comptime PROMPT_LEN = 16
    comptime SEQ_LEN = PROMPT_LEN + CANVAS_LEN
    comptime MAX_SEQ_LEN = 512
    comptime MAX_Q_SIZE = 16 * 512
    comptime MAX_KV_SIZE = 8 * 256

    var active_idx = alloc[Int](CANVAS_LEN)
    for b in range(CANVAS_LEN):
        active_idx[b] = b

    var q_ref = alloc[Float16](CANVAS_LEN * MAX_Q_SIZE)
    var k_ref = alloc[Float16](CANVAS_LEN * MAX_KV_SIZE)
    var v_ref = alloc[Float16](CANVAS_LEN * MAX_KV_SIZE)
    var q_opt = alloc[Float16](CANVAS_LEN * MAX_Q_SIZE)
    var k_opt = alloc[Float16](CANVAS_LEN * MAX_KV_SIZE)
    var v_opt = alloc[Float16](CANVAS_LEN * MAX_KV_SIZE)

    var q_norm_w = alloc[Float32](512)
    var k_norm_w = alloc[Float32](512)
    for i in range(512):
        q_norm_w[i] = Float32(0.8 + Float32(i % 7) * 0.05)
        k_norm_w[i] = Float32(0.9 + Float32(i % 5) * 0.04)

    var k_cache_f32 = alloc[Float32](MAX_SEQ_LEN * MAX_KV_SIZE)
    var v_cache_f32 = alloc[Float32](MAX_SEQ_LEN * MAX_KV_SIZE)
    var k_cache_f16 = alloc[Float16](MAX_SEQ_LEN * MAX_KV_SIZE)
    var v_cache_f16 = alloc[Float16](MAX_SEQ_LEN * MAX_KV_SIZE)

    for i in range(MAX_SEQ_LEN * MAX_KV_SIZE):
        var val = Float16(Float32((i % 29) - 14) * 0.04)
        k_cache_f32[i] = Float32(val)
        v_cache_f32[i] = Float32(val) * 0.5
        k_cache_f16[i] = val
        v_cache_f16[i] = Float16(Float32(val) * 0.5)

    for i in range(CANVAS_LEN * MAX_Q_SIZE):
        var val = Float16(Float32((i % 37) - 18) * 0.05)
        q_ref[i] = val
        q_opt[i] = val
    for i in range(CANVAS_LEN * MAX_KV_SIZE):
        var val_k = Float16(Float32((i % 31) - 15) * 0.05)
        var val_v = Float16(Float32((i % 23) - 11) * 0.05)
        k_ref[i] = val_k
        k_opt[i] = val_k
        v_ref[i] = val_v
        v_opt[i] = val_v

    var out_ref = alloc[Float16](CANVAS_LEN * MAX_Q_SIZE)
    var out_opt = alloc[Float16](CANVAS_LEN * MAX_Q_SIZE)
    var scores_buf = alloc[Float32](CANVAS_LEN * MAX_SEQ_LEN)

    # ------------------------------------------------------------------
    # Test 1: Sliding Layer (16 Q heads, 8 KV heads, dim=256, ratio=2)
    # ------------------------------------------------------------------
    var t0_ref = std.time.perf_counter()
    for b in range(CANVAS_LEN):
        var orig_t = active_idx[b]
        var pos = PROMPT_LEN + orig_t
        var q_store = q_ref + b * MAX_Q_SIZE
        var k_store = k_ref + b * MAX_KV_SIZE
        var v_store = v_ref + b * MAX_KV_SIZE
        for h in range(16):
            rms_norm[in_type=DType.float16, out_type=DType.float16](q_store + h * 256, q_norm_w, q_store + h * 256, 256)
        for h in range(8):
            rms_norm[in_type=DType.float16, out_type=DType.float16](k_store + h * 256, k_norm_w, k_store + h * 256, 256)
        for h in range(8):
            rms_norm_noscale[in_type=DType.float16, out_type=DType.float16](v_store + h * 256, v_store + h * 256, 256)
        ref_apply_rope(q_store, 16, 256, pos, False)
        ref_apply_rope(k_store, 8, 256, pos, False)
        var step_off = pos * 2048
        for i in range(0, 2048, 8):
            k_cache_f32.store[8](step_off + i, k_store.load[8](i).cast[DType.float32]())
            v_cache_f32.store[8](step_off + i, v_store.load[8](i).cast[DType.float32]())

    def ref_attn_sliding(b: Int) capturing -> None:
        var q_load = q_ref + b * MAX_Q_SIZE
        var attn_out_ptr = out_ref + b * MAX_Q_SIZE
        var scores_t = scores_buf + b * MAX_SEQ_LEN
        var q_regs = InlineArray[SIMD[DType.float32, 8], 32](uninitialized=True)
        var acc_regs = InlineArray[SIMD[DType.float32, 8], 32](uninitialized=True)
        for h in range(16):
            var h_offset = h * 256
            var g_offset = (h // 2) * 256
            for d in range(32):
                q_regs[d] = q_load.load[8](h_offset + d * 8).cast[DType.float32]()
            for tt in range(SEQ_LEN):
                var t_off = tt * 2048 + g_offset
                var sv = SIMD[DType.float32, 8](0.0)
                for d in range(32):
                    sv += q_regs[d] * k_cache_f32.load[8](t_off + d * 8)
                scores_t[tt] = sv.reduce_add()
            ref_softmax(scores_t, SEQ_LEN)
            for d in range(32):
                acc_regs[d] = SIMD[DType.float32, 8](0.0)
            for tt in range(SEQ_LEN):
                var t_off = tt * 2048 + g_offset
                var s_vec = SIMD[DType.float32, 8](scores_t[tt])
                for d in range(32):
                    acc_regs[d] += s_vec * v_cache_f32.load[8](t_off + d * 8)
            for d in range(32):
                attn_out_ptr.store[8](h_offset + d * 8, acc_regs[d].cast[DType.float16]())

    parallelize[ref_attn_sliding](CANVAS_LEN)
    var dt_ref_ms = (std.time.perf_counter() - t0_ref) * 1000.0

    var t0_opt = std.time.perf_counter()
    prep_qkv_norm_rope_kvq_batch(
        q_opt, k_opt, v_opt, q_norm_w, k_norm_w, active_idx, k_cache_f16, v_cache_f16,
        CANVAS_LEN, PROMPT_LEN, 16, 8, 256, 2048, 0, MAX_Q_SIZE, MAX_KV_SIZE, False
    )
    flash_dllm_bidirectional_attn(
        q_opt, k_cache_f16, v_cache_f16, out_opt,
        CANVAS_LEN, SEQ_LEN, 16, 8, 256, 2048, 0, MAX_Q_SIZE
    )
    var dt_opt_ms = (std.time.perf_counter() - t0_opt) * 1000.0

    var max_err_sliding: Float32 = 0.0
    for b in range(CANVAS_LEN):
        for i in range(16 * 256):
            var diff = abs(Float32(out_ref[b * MAX_Q_SIZE + i]) - Float32(out_opt[b * MAX_Q_SIZE + i]))
            if diff > max_err_sliding:
                max_err_sliding = diff

    print("  [Sliding Layer 16Q/8KV dim=256] max_err =", max_err_sliding, "| Ref:", dt_ref_ms, "ms -> Flash-dLLM + KVQ:", dt_opt_ms, "ms (", dt_ref_ms / dt_opt_ms, "x speedup)")
    if max_err_sliding > 1e-3:
        raise Error("Sliding layer Flash-dLLM + dLLM-KVQ exceeded 1e-3 error tolerance!")

    # ------------------------------------------------------------------
    # Test 2: Odd active_len (active_len = 63) on Sliding Layer
    # ------------------------------------------------------------------
    flash_dllm_bidirectional_attn(
        q_opt, k_cache_f16, v_cache_f16, out_opt,
        63, SEQ_LEN, 16, 8, 256, 2048, 0, MAX_Q_SIZE
    )
    var max_err_odd: Float32 = 0.0
    for b in range(63):
        for i in range(16 * 256):
            var diff = abs(Float32(out_ref[b * MAX_Q_SIZE + i]) - Float32(out_opt[b * MAX_Q_SIZE + i]))
            if diff > max_err_odd:
                max_err_odd = diff
    print("  [Odd Active Canvas active_len=63] max_err =", max_err_odd)
    if max_err_odd > 1e-3:
        raise Error("Odd active_len Flash-dLLM failed!")

    # ------------------------------------------------------------------
    # Test 3: Global Layer (16 Q heads, 2 KV heads, dim=512, ratio=8)
    # ------------------------------------------------------------------
    for b in range(CANVAS_LEN):
        var orig_t = active_idx[b]
        var pos = PROMPT_LEN + orig_t
        var q_store = q_ref + b * MAX_Q_SIZE
        var k_store = k_ref + b * MAX_KV_SIZE
        var v_store = v_ref + b * MAX_KV_SIZE
        for h in range(16):
            rms_norm[in_type=DType.float16, out_type=DType.float16](q_store + h * 512, q_norm_w, q_store + h * 512, 512)
        for h in range(2):
            rms_norm[in_type=DType.float16, out_type=DType.float16](k_store + h * 512, k_norm_w, k_store + h * 512, 512)
        for h in range(2):
            rms_norm_noscale[in_type=DType.float16, out_type=DType.float16](v_store + h * 512, v_store + h * 512, 512)
        ref_apply_rope(q_store, 16, 512, pos, True)
        ref_apply_rope(k_store, 2, 512, pos, True)
        var step_off = pos * 1024
        for i in range(0, 1024, 8):
            k_cache_f32.store[8](step_off + i, k_store.load[8](i).cast[DType.float32]())
            v_cache_f32.store[8](step_off + i, v_store.load[8](i).cast[DType.float32]())

    def ref_attn_global(b: Int) capturing -> None:
        var q_load = q_ref + b * MAX_Q_SIZE
        var attn_out_ptr = out_ref + b * MAX_Q_SIZE
        var scores_t = scores_buf + b * MAX_SEQ_LEN
        var q_regs = InlineArray[SIMD[DType.float32, 8], 64](uninitialized=True)
        var acc_regs = InlineArray[SIMD[DType.float32, 8], 64](uninitialized=True)
        for h in range(16):
            var h_offset = h * 512
            var g_offset = (h // 8) * 512
            for d in range(64):
                q_regs[d] = q_load.load[8](h_offset + d * 8).cast[DType.float32]()
            for tt in range(SEQ_LEN):
                var t_off = tt * 1024 + g_offset
                var sv = SIMD[DType.float32, 8](0.0)
                for d in range(64):
                    sv += q_regs[d] * k_cache_f32.load[8](t_off + d * 8)
                scores_t[tt] = sv.reduce_add()
            ref_softmax(scores_t, SEQ_LEN)
            for d in range(64):
                acc_regs[d] = SIMD[DType.float32, 8](0.0)
            for tt in range(SEQ_LEN):
                var t_off = tt * 1024 + g_offset
                var s_vec = SIMD[DType.float32, 8](scores_t[tt])
                for d in range(64):
                    acc_regs[d] += s_vec * v_cache_f32.load[8](t_off + d * 8)
            for d in range(64):
                attn_out_ptr.store[8](h_offset + d * 8, acc_regs[d].cast[DType.float16]())

    parallelize[ref_attn_global](CANVAS_LEN)

    prep_qkv_norm_rope_kvq_batch(
        q_opt, k_opt, v_opt, q_norm_w, k_norm_w, active_idx, k_cache_f16, v_cache_f16,
        CANVAS_LEN, PROMPT_LEN, 16, 2, 512, 1024, 0, MAX_Q_SIZE, MAX_KV_SIZE, True
    )
    flash_dllm_bidirectional_attn(
        q_opt, k_cache_f16, v_cache_f16, out_opt,
        CANVAS_LEN, SEQ_LEN, 16, 2, 512, 1024, 0, MAX_Q_SIZE
    )

    var max_err_global: Float32 = 0.0
    for b in range(CANVAS_LEN):
        for i in range(16 * 512):
            var diff = abs(Float32(out_ref[b * MAX_Q_SIZE + i]) - Float32(out_opt[b * MAX_Q_SIZE + i]))
            if diff > max_err_global:
                max_err_global = diff

    print("  [Global Layer 16Q/2KV dim=512] max_err =", max_err_global)
    if max_err_global > 1e-3:
        raise Error("Global layer Flash-dLLM + dLLM-KVQ exceeded 1e-3 error tolerance!")

    active_idx.free()
    q_ref.free()
    k_ref.free()
    v_ref.free()
    q_opt.free()
    k_opt.free()
    v_opt.free()
    q_norm_w.free()
    k_norm_w.free()
    k_cache_f32.free()
    v_cache_f32.free()
    k_cache_f16.free()
    v_cache_f16.free()
    out_ref.free()
    out_opt.free()
    scores_buf.free()
    print("=== ALL TESTS PASSED! ===")
