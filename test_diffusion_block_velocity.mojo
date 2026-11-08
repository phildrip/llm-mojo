from std.memory import UnsafePointer, alloc
from std.math import sqrt, abs
import std.time
from diffusion_kernels import (
    build_block_active_canvas_map,
    extrapolate_sc_velocity_batch,
    prep_qkv_norm_rope_kvq_batch,
    flash_dllm_bidirectional_attn,
)


def main() raises:
    print("=== Test & Benchmark: Block-Diffusion Mode & 2nd-Order Velocity Extrapolation (llm-mojo-2tc.6) ===")
    comptime CANVAS_LEN = 256
    comptime BLOCK_SIZE = 64
    comptime EMBED_DIM = 2816
    comptime PROMPT_LEN = 16
    comptime MAX_Q_SIZE = 16 * 512
    comptime MAX_KV_SIZE = 8 * 256

    var locked_mask = alloc[Bool](CANVAS_LEN)
    var active_canvas_idx = alloc[Int](CANVAS_LEN)
    for t in range(CANVAS_LEN):
        locked_mask[t] = False

    # 1. Test 4 sequential 64-token blocks and intra-block locking
    for blk in range(4):
        var b_start = blk * BLOCK_SIZE
        var b_end = b_start + BLOCK_SIZE
        var a_len = build_block_active_canvas_map(locked_mask, active_canvas_idx, b_start, b_end)
        if a_len != BLOCK_SIZE or active_canvas_idx[0] != b_start or active_canvas_idx[a_len - 1] != b_end - 1:
            raise Error("Block active canvas map returned wrong range!")
        # Lock 16 tokens inside this block
        for t in range(b_start, b_start + 16):
            locked_mask[t] = True
        var a_len_after = build_block_active_canvas_map(locked_mask, active_canvas_idx, b_start, b_end)
        if a_len_after != 48 or active_canvas_idx[0] != b_start + 16:
            raise Error("Intra-block locking failed!")
        # Lock rest of block before moving to next block
        for t in range(b_start, b_end):
            locked_mask[t] = True

    # 2. Test 2nd-Order Latent Velocity Extrapolation (Adams-Bashforth + Norm Conservation + Trust-Region Guard)
    for t in range(CANVAS_LEN):
        locked_mask[t] = False
    var sc_embeds = alloc[Float32](CANVAS_LEN * EMBED_DIM)
    var prev_sc_embeds = alloc[Float32](CANVAS_LEN * EMBED_DIM)
    for i in range(CANVAS_LEN * EMBED_DIM):
        sc_embeds[i] = Float32((i % 29) - 14) * 0.05
        prev_sc_embeds[i] = 0.0

    # Step 1: initializes prev_sc_embeds
    extrapolate_sc_velocity_batch(
        sc_embeds, prev_sc_embeds, locked_mask, 0, BLOCK_SIZE, EMBED_DIM, 1, 1
    )

    # Step 2: Token 0 has smooth velocity; Token 1 has a discontinuous jump (||v|| > 0.85 ||s||)
    var orig_norm_t0_sq: Float32 = 0.0
    for d in range(EMBED_DIM):
        var s0 = prev_sc_embeds[0 * EMBED_DIM + d] + Float32(0.02)
        sc_embeds[0 * EMBED_DIM + d] = s0
        orig_norm_t0_sq += s0 * s0
        # Token 1 flips sign discontinuously -> velocity norm = 2 * ||s|| > 0.85 * ||s||
        sc_embeds[1 * EMBED_DIM + d] = -prev_sc_embeds[1 * EMBED_DIM + d]

    var t1_before_d0 = sc_embeds[1 * EMBED_DIM + 0]
    extrapolate_sc_velocity_batch(
        sc_embeds, prev_sc_embeds, locked_mask, 0, BLOCK_SIZE, EMBED_DIM, 2, 2
    )

    var new_norm_t0_sq: Float32 = 0.0
    for d in range(EMBED_DIM):
        var v = sc_embeds[0 * EMBED_DIM + d]
        new_norm_t0_sq += v * v
    var norm_rel_err = abs(sqrt(new_norm_t0_sq) - sqrt(orig_norm_t0_sq)) / sqrt(orig_norm_t0_sq)
    var jump_diff = abs(sc_embeds[1 * EMBED_DIM + 0] - t1_before_d0)
    print("  [Velocity Extrapolation] Smooth token L2 norm rel_err =", norm_rel_err, "| Discontinuous flip guard diff =", jump_diff)
    if norm_rel_err > 1e-5 or jump_diff > 0.0:
        raise Error("Velocity extrapolation norm conservation or trust-region guard failed!")

    # 3. Benchmark 64-token Block-Diffusion step vs 256-token Monolithic step (QKV Prep + Flash-dLLM Attention)
    for t in range(CANVAS_LEN):
        active_canvas_idx[t] = t
    var q_buf = alloc[Float16](CANVAS_LEN * MAX_Q_SIZE)
    var k_buf = alloc[Float16](CANVAS_LEN * MAX_KV_SIZE)
    var v_buf = alloc[Float16](CANVAS_LEN * MAX_KV_SIZE)
    var out_buf = alloc[Float16](CANVAS_LEN * MAX_Q_SIZE)
    var k_cache = alloc[Float16](512 * MAX_KV_SIZE)
    var v_cache = alloc[Float16](512 * MAX_KV_SIZE)
    var q_norm_w = alloc[Float32](512)
    var k_norm_w = alloc[Float32](512)
    for i in range(512):
        q_norm_w[i] = 1.0
        k_norm_w[i] = 1.0
    for i in range(CANVAS_LEN * MAX_Q_SIZE):
        q_buf[i] = Float16(0.02)
    for i in range(CANVAS_LEN * MAX_KV_SIZE):
        k_buf[i] = Float16(0.02)
        v_buf[i] = Float16(0.02)
    for i in range(512 * MAX_KV_SIZE):
        k_cache[i] = Float16(0.02)
        v_cache[i] = Float16(0.02)

    var t0_mono = std.time.perf_counter()
    prep_qkv_norm_rope_kvq_batch(
        q_buf, k_buf, v_buf, q_norm_w, k_norm_w, active_canvas_idx, k_cache, v_cache,
        256, PROMPT_LEN, 16, 8, 256, 2048, 0, MAX_Q_SIZE, MAX_KV_SIZE, False
    )
    flash_dllm_bidirectional_attn(
        q_buf, k_cache, v_cache, out_buf,
        256, PROMPT_LEN + 256, 16, 8, 256, 2048, 0, MAX_Q_SIZE
    )
    var dt_mono_ms = (std.time.perf_counter() - t0_mono) * 1000.0

    var t0_blk = std.time.perf_counter()
    prep_qkv_norm_rope_kvq_batch(
        q_buf, k_buf, v_buf, q_norm_w, k_norm_w, active_canvas_idx, k_cache, v_cache,
        64, PROMPT_LEN, 16, 8, 256, 2048, 0, MAX_Q_SIZE, MAX_KV_SIZE, False
    )
    flash_dllm_bidirectional_attn(
        q_buf, k_cache, v_cache, out_buf,
        64, PROMPT_LEN + 64, 16, 8, 256, 2048, 0, MAX_Q_SIZE
    )
    var dt_blk_ms = (std.time.perf_counter() - t0_blk) * 1000.0

    print("  [Block-64 vs Monolithic-256 Step] Monolithic 256:", dt_mono_ms, "ms -> Block-64 (seq_len=80):", dt_blk_ms, "ms (", dt_mono_ms / dt_blk_ms, "x per-step speedup)")
    if dt_blk_ms >= dt_mono_ms:
        raise Error("Expected Block-64 step to be faster than Monolithic-256 step!")

    locked_mask.free()
    active_canvas_idx.free()
    sc_embeds.free()
    prev_sc_embeds.free()
    q_buf.free()
    k_buf.free()
    v_buf.free()
    out_buf.free()
    k_cache.free()
    v_cache.free()
    q_norm_w.free()
    k_norm_w.free()
    print("=== ALL TESTS PASSED! ===")
