from std.memory import alloc
from std.collections import InlineArray
from std.math import cos, sin, exp, abs
import std.time
from attention import paged_attention_layer, apply_rope, apply_rope_qk, ring_row, _eval_head_range
from lsa_core import lsa_sparse_paged_attention


def ref_apply_rope(
    vec: UnsafePointer[Float32, MutUntrackedOrigin],
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
            var val1 = vec[h_offset + i]
            var val2 = vec[h_offset + half_dim + i]
            vec[h_offset + i] = val1 * cos_theta - val2 * sin_theta
            vec[h_offset + half_dim + i] = val2 * cos_theta + val1 * sin_theta


def ref_paged_attention_layer(
    q: UnsafePointer[Float32, _],
    k_pool: UnsafePointer[Float32, _],
    v_pool: UnsafePointer[Float32, _],
    block_table: List[Int],
    block_size: Int,
    pos: Int,
    num_heads: Int,
    num_kv_heads: Int,
    head_dim: Int,
    window: Int,
    scores: UnsafePointer[Float32, MutUntrackedOrigin],
    rows: UnsafePointer[Int, MutUntrackedOrigin],
    attn_out: UnsafePointer[Float32, MutUntrackedOrigin],
    ring_size: Int = 0,
    ring_base: Int = 0,
):
    var kv_stride = num_kv_heads * head_dim
    var ratio = num_heads // num_kv_heads
    var start_t = 0
    if window > 0 and pos - window + 1 > 0:
        start_t = pos - window + 1
    var n = pos + 1 - start_t
    if ring_size > 0:
        for i in range(n):
            var t = start_t + i
            rows[i] = (ring_base + t % ring_size) * kv_stride
    else:
        for i in range(n):
            var t = start_t + i
            rows[i] = (block_table[t // block_size] * block_size + t % block_size) * kv_stride

    for g in range(num_kv_heads):
        var g_offset = g * head_dim
        var h_first = g * ratio
        for i in range(n):
            var k_off = rows[i] + g_offset
            for hh in range(ratio):
                var h = h_first + hh
                var h_offset = h * head_dim
                var score_vec = SIMD[DType.float32, 8](0.0)
                for d in range(0, head_dim, 8):
                    score_vec = score_vec + q.load[8](h_offset + d) * k_pool.load[8](k_off + d)
                scores[h * n + i] = score_vec.reduce_add()

        for hh in range(ratio):
            var h = h_first + hh
            var s_ptr = scores + h * n
            var max_val = s_ptr[0]
            for i in range(1, n):
                if s_ptr[i] > max_val:
                    max_val = s_ptr[i]
            var sum_exp: Float32 = 0.0
            for i in range(n):
                s_ptr[i] = exp(s_ptr[i] - max_val)
                sum_exp += s_ptr[i]
            for i in range(n):
                s_ptr[i] /= (sum_exp + 1e-10)
            var h_offset = h * head_dim
            for d in range(0, head_dim, 8):
                attn_out.store[8](h_offset + d, SIMD[DType.float32, 8](0.0))

        for i in range(n):
            var v_off = rows[i] + g_offset
            for hh in range(ratio):
                var h = h_first + hh
                var h_offset = h * head_dim
                var alpha_vec = SIMD[DType.float32, 8](scores[h * n + i])
                for d in range(0, head_dim, 8):
                    attn_out.store[8](
                        h_offset + d,
                        attn_out.load[8](h_offset + d) + alpha_vec * v_pool.load[8](v_off + d),
                    )


def main() raises:
    print("==================================================================")
    print("   TDD & BENCHMARK: Fast Trig-Hoisted RoPE & 8-Core GQA Attention ")
    print("==================================================================")

    # Test 1: RoPE across pos = 0, 47, 1023, 1024, 100000 (SWA & Global, apply_rope & apply_rope_qk)
    var q_ref = alloc[Float32](16 * 512)
    var k_ref = alloc[Float32](8 * 512)
    var q_opt = alloc[Float32](16 * 512)
    var k_opt = alloc[Float32](8 * 512)

    var rope_positions: InlineArray[Int, 5] = [0, 47, 1023, 1024, 100000]
    var rope_swa_err: Float32 = 0.0
    var rope_glob_err: Float32 = 0.0

    for p_idx in range(5):
        var p = rope_positions[p_idx]
        for i in range(16 * 512):
            var v = Float32((i % 41) - 20) * 0.05
            q_ref[i] = v
            q_opt[i] = v
        for i in range(8 * 512):
            var v = Float32((i % 37) - 18) * 0.05
            k_ref[i] = v
            k_opt[i] = v

        ref_apply_rope(q_ref, 16, 256, p, False)
        ref_apply_rope(k_ref, 8, 256, p, False)
        apply_rope_qk(q_opt, k_opt, 16, 8, 256, p, False)
        for i in range(16 * 256):
            var d = abs(q_ref[i] - q_opt[i])
            if d > rope_swa_err:
                rope_swa_err = d
        for i in range(8 * 256):
            var d = abs(k_ref[i] - k_opt[i])
            if d > rope_swa_err:
                rope_swa_err = d

        for i in range(16 * 512):
            var v = Float32((i % 41) - 20) * 0.05
            q_ref[i] = v
            q_opt[i] = v
        for i in range(2 * 512):
            var v = Float32((i % 37) - 18) * 0.05
            k_ref[i] = v
            k_opt[i] = v

        ref_apply_rope(q_ref, 16, 512, p, True)
        ref_apply_rope(k_ref, 2, 512, p, True)
        if p_idx % 2 == 0:
            apply_rope_qk(q_opt, k_opt, 16, 2, 512, p, True)
        else:
            apply_rope(q_opt, 16, 512, p, True)
            apply_rope(k_opt, 2, 512, p, True)
        for i in range(16 * 512):
            var d = abs(q_ref[i] - q_opt[i])
            if d > rope_glob_err:
                rope_glob_err = d
        for i in range(2 * 512):
            var d = abs(k_ref[i] - k_opt[i])
            if d > rope_glob_err:
                rope_glob_err = d

    # Benchmark 30 layers of RoPE (25 SWA + 5 Global)
    var rope_iters = 200
    var tr0 = std.time.perf_counter()
    for _ in range(rope_iters):
        for l in range(30):
            var is_g = (l % 6 == 5)
            var hd = 512 if is_g else 256
            var kvh = 2 if is_g else 8
            ref_apply_rope(q_ref, 16, hd, 47, is_g)
            ref_apply_rope(k_ref, kvh, hd, 47, is_g)
    var tr1 = std.time.perf_counter()
    for _ in range(rope_iters):
        for l in range(30):
            var is_g = (l % 6 == 5)
            var hd = 512 if is_g else 256
            var kvh = 2 if is_g else 8
            apply_rope_qk(q_opt, k_opt, 16, kvh, hd, 47, is_g)
    var tr2 = std.time.perf_counter()

    var rope_ref_ms = (tr1 - tr0) * 1000.0 / Float64(rope_iters)
    var rope_opt_ms = (tr2 - tr1) * 1000.0 / Float64(rope_iters)
    print("  [Test 1: 30-Layer RoPE] SWA Err:", rope_swa_err, "| Global Err:", rope_glob_err)
    print("    Scalar Per-Head RoPE (30 layers): ", rope_ref_ms, "ms/token")
    print("    Trig-Hoisted SIMD RoPE (30 layers):", rope_opt_ms, "ms/token")
    print("    RoPE Speedup:                     ", rope_ref_ms / rope_opt_ms, "x")
    if rope_swa_err != 0.0 or rope_glob_err != 0.0:
        raise Error("RoPE accuracy mismatch!")

    # Test 2: 8-Core GQA Group-Parallel Attention across pos=0, 62, 63, 511, 2047, ring wrap, odd ratio, and LSA
    var max_pos = 2048
    var block_size = 16
    var num_blocks = max_pos // block_size
    var block_table = List[Int]()
    for b in range(num_blocks):
        block_table.append(b)

    var k_pool = alloc[Float32](max_pos * 8 * 256)
    var v_pool = alloc[Float32](max_pos * 8 * 256)
    for i in range(max_pos * 8 * 256):
        k_pool[i] = Float32((i % 29) - 14) * 0.02
        v_pool[i] = Float32((i % 31) - 15) * 0.02

    var scores_ref = alloc[Float32](16 * max_pos)
    var scores_opt = alloc[Float32](16 * max_pos)
    var rows_ref = alloc[Int](max_pos)
    var rows_opt = alloc[Int](max_pos)
    var out_ref = alloc[Float32](16 * 512)
    var out_opt = alloc[Float32](16 * 512)

    var test_positions: InlineArray[Int, 4] = [0, 62, 63, 511]
    var max_attn_err: Float32 = 0.0
    for p_idx in range(4):
        var p = test_positions[p_idx]
        ref_paged_attention_layer(q_opt, k_pool, v_pool, block_table, block_size, p, 16, 8, 256, 1024, scores_ref, rows_ref, out_ref, 512, 0)
        paged_attention_layer(q_opt, k_pool, v_pool, block_table, block_size, p, 16, 8, 256, 1024, scores_opt, rows_opt, out_opt, 512, 0)
        for i in range(16 * 256):
            var d = abs(out_ref[i] - out_opt[i])
            if d > max_attn_err:
                max_attn_err = d

    # Ring wrap test: pos=600, ring_size=512, window=256 (start_t=345 > 0)
    ref_paged_attention_layer(q_opt, k_pool, v_pool, block_table, block_size, 600, 16, 8, 256, 256, scores_ref, rows_ref, out_ref, 512, 0)
    paged_attention_layer(q_opt, k_pool, v_pool, block_table, block_size, 600, 16, 8, 256, 256, scores_opt, rows_opt, out_opt, 512, 0)
    for i in range(16 * 256):
        var d = abs(out_ref[i] - out_opt[i])
        if d > max_attn_err:
            max_attn_err = d

    # Odd ratio test: 12q / 4kv (ratio=3 -> hits pair + single-head tail in _eval_head_range)
    ref_paged_attention_layer(q_opt, k_pool, v_pool, block_table, block_size, 127, 12, 4, 256, 1024, scores_ref, rows_ref, out_ref, 512, 0)
    paged_attention_layer(q_opt, k_pool, v_pool, block_table, block_size, 127, 12, 4, 256, 1024, scores_opt, rows_opt, out_opt, 512, 0)
    for i in range(12 * 256):
        var d = abs(out_ref[i] - out_opt[i])
        if d > max_attn_err:
            max_attn_err = d

    # Global pos=511 and pos=2047 (16q / 2kv, head_dim=512)
    ref_paged_attention_layer(q_opt, k_pool, v_pool, block_table, block_size, 511, 16, 2, 512, 0, scores_ref, rows_ref, out_ref, 0, 0)
    paged_attention_layer(q_opt, k_pool, v_pool, block_table, block_size, 511, 16, 2, 512, 0, scores_opt, rows_opt, out_opt, 0, 0)
    var attn_err_glob: Float32 = 0.0
    for i in range(16 * 512):
        var d = abs(out_ref[i] - out_opt[i])
        if d > attn_err_glob:
            attn_err_glob = d

    ref_paged_attention_layer(q_opt, k_pool, v_pool, block_table, block_size, 2047, 16, 2, 512, 0, scores_ref, rows_ref, out_ref, 0, 0)
    paged_attention_layer(q_opt, k_pool, v_pool, block_table, block_size, 2047, 16, 2, 512, 0, scores_opt, rows_opt, out_opt, 0, 0)
    for i in range(16 * 512):
        var d = abs(out_ref[i] - out_opt[i])
        if d > attn_err_glob:
            attn_err_glob = d

    # LSA Sparse Attention: test n_sel == 0 (empty) and n_sel == 128 (dense prefix equivalence)
    var empty_sel = List[Int]()
    for i in range(16 * 512):
        out_opt[i] = 99.0
    lsa_sparse_paged_attention(q_opt, k_pool, v_pool, block_table, block_size, empty_sel, 16, 2, 512, scores_opt, rows_opt, out_opt)
    for i in range(16 * 512):
        if out_opt[i] != 0.0:
            raise Error("Expected zeroed output for empty LSA selection!")
    _eval_head_range(q_opt, k_pool, v_pool, rows_opt, scores_opt, out_opt, 0, 0, 0, 2, 512)
    for i in range(2 * 512):
        if out_opt[i] != 0.0:
            raise Error("Expected zeroed output for _eval_head_range with n=0!")

    var full_sel_128 = List[Int]()
    for t in range(128):
        full_sel_128.append(t)
    ref_paged_attention_layer(q_opt, k_pool, v_pool, block_table, block_size, 127, 16, 2, 512, 0, scores_ref, rows_ref, out_ref, 0, 0)
    lsa_sparse_paged_attention(q_opt, k_pool, v_pool, block_table, block_size, full_sel_128, 16, 2, 512, scores_opt, rows_opt, out_opt)
    var lsa_err_128: Float32 = 0.0
    for i in range(16 * 512):
        var d = abs(out_ref[i] - out_opt[i])
        if d > lsa_err_128:
            lsa_err_128 = d

    var attn_iters = 40
    var ta0 = std.time.perf_counter()
    for _ in range(attn_iters):
        for l in range(30):
            var is_g = (l % 6 == 5)
            if is_g:
                ref_paged_attention_layer(q_opt, k_pool, v_pool, block_table, block_size, 511, 16, 2, 512, 0, scores_ref, rows_ref, out_ref, 0, 0)
            else:
                ref_paged_attention_layer(q_opt, k_pool, v_pool, block_table, block_size, 511, 16, 8, 256, 1024, scores_ref, rows_ref, out_ref, 512, 0)
    var ta1 = std.time.perf_counter()
    for _ in range(attn_iters):
        for l in range(30):
            var is_g = (l % 6 == 5)
            if is_g:
                paged_attention_layer(q_opt, k_pool, v_pool, block_table, block_size, 511, 16, 2, 512, 0, scores_opt, rows_opt, out_opt, 0, 0)
            else:
                paged_attention_layer(q_opt, k_pool, v_pool, block_table, block_size, 511, 16, 8, 256, 1024, scores_opt, rows_opt, out_opt, 512, 0)
    var ta2 = std.time.perf_counter()

    var attn_ref_ms = (ta1 - ta0) * 1000.0 / Float64(attn_iters)
    var attn_opt_ms = (ta2 - ta1) * 1000.0 / Float64(attn_iters)
    print("  [Test 2: 30-Layer Attention] Max SWA/Ring/Odd Err:", max_attn_err, "| Global Err:", attn_err_glob, "| LSA Err:", lsa_err_128)
    print("    Single-Core Attention (pos=511): ", attn_ref_ms, "ms/token")
    print("    8-Core GQA Attention (pos=511):  ", attn_opt_ms, "ms/token")
    print("    Attention Speedup (pos=511):     ", attn_ref_ms / attn_opt_ms, "x")

    q_ref.free()
    k_ref.free()
    q_opt.free()
    k_opt.free()
    k_pool.free()
    v_pool.free()
    scores_ref.free()
    scores_opt.free()
    rows_ref.free()
    rows_opt.free()
    out_ref.free()
    out_opt.free()

    if max_attn_err != 0.0 or attn_err_glob != 0.0 or lsa_err_128 != 0.0:
        raise Error("Expected bit-exact 0.0 error on parallel GQA and LSA attention!")
    print("  [PASS] Trig-Hoisted SIMD RoPE & 8-Core GQA Attention Bit-Exactness Verified!")
