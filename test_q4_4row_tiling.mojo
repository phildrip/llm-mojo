from std.memory import alloc
from std.collections import InlineArray
from std.algorithm import parallelize
from std.math import abs
import std.time
from moe_kernels import (
    q4_matmul_vec,
    gelu_and_mul_moe,
    run_moe_experts_stage1_8core,
    run_moe_experts_stage2_8core,
)


def scalar_q4_oracle(
    W_packed: UnsafePointer[UInt8, _],
    scales: UnsafePointer[Float32, _],
    x: UnsafePointer[Float32, _],
    out_ptr: UnsafePointer[Float32, MutUntrackedOrigin],
    c_in: Int,
    c_out: Int,
):
    """Independent scalar dequantization oracle for Q4_0 matmul."""
    var c_in_packed = c_in // 2
    var c_in_blocks = c_in // 32
    for r in range(c_out):
        var row_sum: Float32 = 0.0
        for b in range(c_in_blocks):
            var sc = scales[r * c_in_blocks + b]
            var blk_sum: Float32 = 0.0
            for j in range(16):
                var byte = Int(W_packed[r * c_in_packed + b * 16 + j])
                var w_lo = Float32((byte & 15) - 8)
                var w_hi = Float32((byte >> 4) - 8)
                blk_sum += w_lo * x[b * 32 + 2 * j] + w_hi * x[b * 32 + 2 * j + 1]
            row_sum += blk_sum * sc
        out_ptr[r] = row_sum


def q4_matmul_vec_2row_heap(
    W_packed: UnsafePointer[UInt8, _],
    scales: UnsafePointer[Float32, _],
    x: UnsafePointer[Float32, _],
    out_ptr: UnsafePointer[Float32, MutUntrackedOrigin],
    c_in: Int,
    c_out: Int,
    num_tasks: Int,
):
    var tasks = max(1, min(num_tasks, c_out))
    var chunk_size = c_out // tasks
    var c_in_packed = c_in // 2
    var c_in_blocks = c_in // 32
    var x_deint = alloc[Float32](c_in)
    var x_even = x_deint
    var x_odd = x_deint + c_in_packed
    for b in range(c_in_blocks):
        var v0 = x.load[16](b * 32)
        var v1 = x.load[16](b * 32 + 16)
        x_even.store[16](b * 16, v0.deinterleave()[0].join(v1.deinterleave()[0]))
        x_odd.store[16](b * 16, v0.deinterleave()[1].join(v1.deinterleave()[1]))

    def worker(t: Int) capturing -> None:
        var start_r = t * chunk_size
        var end_r = (t + 1) * chunk_size if t < tasks - 1 else c_out
        var r = start_r
        while r < end_r - 1:
            var acc0 = SIMD[DType.float32, 16](0.0)
            var acc1 = SIMD[DType.float32, 16](0.0)
            var rp0 = r * c_in_packed
            var rs0 = r * c_in_blocks
            var rp1 = (r + 1) * c_in_packed
            var rs1 = (r + 1) * c_in_blocks
            for b in range(c_in_blocks):
                var xe = x_even.load[16](b * 16)
                var xo = x_odd.load[16](b * 16)
                var sc0 = scales[rs0 + b]
                var b0 = W_packed.load[16](rp0 + b * 16)
                var we0 = ((b0 & 15).cast[DType.int8]() - 8).cast[DType.float32]()
                var wo0 = ((b0 >> 4).cast[DType.int8]() - 8).cast[DType.float32]()
                acc0 = acc0 + (we0 * xe + wo0 * xo) * sc0

                var sc1 = scales[rs1 + b]
                var b1 = W_packed.load[16](rp1 + b * 16)
                var we1 = ((b1 & 15).cast[DType.int8]() - 8).cast[DType.float32]()
                var wo1 = ((b1 >> 4).cast[DType.int8]() - 8).cast[DType.float32]()
                acc1 = acc1 + (we1 * xe + wo1 * xo) * sc1
            out_ptr[r] = acc0.reduce_add()
            out_ptr[r + 1] = acc1.reduce_add()
            r += 2
        while r < end_r:
            var acc = SIMD[DType.float32, 16](0.0)
            var rp = r * c_in_packed
            var rs = r * c_in_blocks
            for b in range(c_in_blocks):
                var sc = scales[rs + b]
                var bytes = W_packed.load[16](rp + b * 16)
                var we = ((bytes & 15).cast[DType.int8]() - 8).cast[DType.float32]()
                var wo = ((bytes >> 4).cast[DType.int8]() - 8).cast[DType.float32]()
                acc = acc + (we * x_even.load[16](b * 16) + wo * x_odd.load[16](b * 16)) * sc
            out_ptr[r] = acc.reduce_add()
            r += 1

    parallelize[worker](tasks)
    x_deint.free()


def main() raises:
    print("==================================================================")
    print("   TDD & BENCHMARK: 4-Row Register Tiling & Zero-Alloc Kernels    ")
    print("==================================================================")
    var c_in = 2816
    var c_out = 8192
    var w_packed = alloc[UInt8](c_out * (c_in // 2))
    var w_scales = alloc[Float32](c_out * (c_in // 32))
    var x = alloc[Float32](c_in)
    var out_ref = alloc[Float32](c_out)
    var out_opt = alloc[Float32](c_out)
    var out_scalar = alloc[Float32](128)

    for i in range(c_in):
        x[i] = Float32((i % 31) - 15) * 0.02
    for i in range(c_out * (c_in // 2)):
        w_packed[i] = UInt8((i * 13 + 7) & 255)
    for i in range(c_out * (c_in // 32)):
        w_scales[i] = Float32((i % 19) + 1) * 0.002

    # Test 1: Independent scalar oracle check (first 67 rows, non-multiple of 4)
    scalar_q4_oracle(w_packed, w_scales, x, out_scalar, c_in, 67)
    q4_matmul_vec(w_packed, w_scales, x, out_opt, c_in, 67, 8)
    var scalar_err: Float32 = 0.0
    for i in range(67):
        var d = abs(out_scalar[i] - out_opt[i])
        if d > scalar_err:
            scalar_err = d
    print("  [Test 1: Independent Scalar Oracle c_out=67] Max Error:", scalar_err)
    if scalar_err > 1e-4:
        raise Error("Scalar oracle mismatch: " + String(scalar_err))

    # Test 2: Warmed-up benchmark & bit-exact equality vs 2-row baseline on Wq (2816 -> 8192)
    var iters = 100
    for _ in range(10):
        q4_matmul_vec_2row_heap(w_packed, w_scales, x, out_ref, c_in, c_out, 8)
        q4_matmul_vec(w_packed, w_scales, x, out_opt, c_in, c_out, 8)

    var t0 = std.time.perf_counter()
    for _ in range(iters):
        q4_matmul_vec_2row_heap(w_packed, w_scales, x, out_ref, c_in, c_out, 8)
    var t1 = std.time.perf_counter()
    for _ in range(iters):
        q4_matmul_vec(w_packed, w_scales, x, out_opt, c_in, c_out, 8)
    var t2 = std.time.perf_counter()

    var max_err: Float32 = 0.0
    for i in range(c_out):
        var d = abs(out_ref[i] - out_opt[i])
        if d > max_err:
            max_err = d

    var ref_ms = (t1 - t0) * 1000.0 / Float64(iters)
    var opt_ms = (t2 - t1) * 1000.0 / Float64(iters)
    print("  [Test 2: Wq 2816->8192] Max Error:", max_err)
    print("    2-Row + Heap Alloc:       ", ref_ms, "ms")
    print("    2-Row + Stack Zero-Alloc: ", opt_ms, "ms")
    print("    Speedup:                  ", ref_ms / opt_ms, "x")
    if max_err != 0.0:
        raise Error("Expected bit-exact 0.0 error vs 2-row baseline, got " + String(max_err))

    # Test 3: Remainder c_out=707 + Heap fallback path c_in=8224 (> 8192)
    q4_matmul_vec_2row_heap(w_packed, w_scales, x, out_ref, c_in, 707, 8)
    q4_matmul_vec(w_packed, w_scales, x, out_opt, c_in, 707, 8)
    for i in range(707):
        if out_ref[i] != out_opt[i]:
            raise Error("Remainder c_out=707 bit-exact mismatch at row " + String(i))

    var big_cin = 8224
    var big_cout = 19
    var big_w = alloc[UInt8](big_cout * (big_cin // 2))
    var big_s = alloc[Float32](big_cout * (big_cin // 32))
    var big_x = alloc[Float32](big_cin)
    for i in range(big_cin):
        big_x[i] = Float32((i % 23) - 11) * 0.01
    for i in range(big_cout * (big_cin // 2)):
        big_w[i] = UInt8((i * 7 + 3) & 255)
    for i in range(big_cout * (big_cin // 32)):
        big_s[i] = Float32((i % 13) + 1) * 0.001
    q4_matmul_vec_2row_heap(big_w, big_s, big_x, out_ref, big_cin, big_cout, 8)
    q4_matmul_vec(big_w, big_s, big_x, out_opt, big_cin, big_cout, 8)
    for i in range(big_cout):
        if out_ref[i] != out_opt[i]:
            raise Error("Heap fallback c_in=8224 mismatch at row " + String(i))
    big_w.free()
    big_s.free()
    big_x.free()
    print("  [Test 3: Remainder c_out=707 & Heap Fallback c_in=8224]: PASS (0.0 error)")

    # Test 4: MoE Stage 1 & Stage 2 edge cases (chunk % 4 in {1,2,3}, active_count=0,1,8, num_tasks > rows)
    var test_edim = 64
    var test_mdim = 32
    var test_gu_rows = test_mdim * 2
    var gu_p = alloc[UInt8](8 * test_gu_rows * (test_edim // 2))
    var gu_s = alloc[Float32](8 * test_gu_rows * (test_edim // 32))
    var dn_p = alloc[UInt8](8 * test_edim * (test_mdim // 2))
    var dn_s = alloc[Float32](8 * test_edim * (test_mdim // 32))
    for i in range(8 * test_gu_rows * (test_edim // 2)):
        gu_p[i] = UInt8((i * 11 + 5) & 255)
    for i in range(8 * test_gu_rows * (test_edim // 32)):
        gu_s[i] = Float32((i % 9) + 1) * 0.01
    for i in range(8 * test_edim * (test_mdim // 2)):
        dn_p[i] = UInt8((i * 19 + 1) & 255)
    for i in range(8 * test_edim * (test_mdim // 32)):
        dn_s[i] = Float32((i % 7) + 1) * 0.01

    var act_idx = alloc[Int](8)
    var act_w = alloc[Float32](8)
    for i in range(8):
        act_idx[i] = i
        act_w[i] = Float32(0.125)
    var m_deint = alloc[Float32](test_edim)
    var gu_out = alloc[Float32](8 * test_gu_rows)
    var act_buf = alloc[Float32](8 * test_mdim)
    var ad_buf = alloc[Float32](8 * test_mdim)
    var m_out_ref = alloc[Float32](test_edim)
    var m_out_opt = alloc[Float32](test_edim)

    # Test active_count = 0 (should zero moe_out)
    run_moe_experts_stage1_8core(gu_p, gu_s, x, m_deint, act_idx, 0, gu_out, act_buf, test_edim, test_mdim, 8)
    run_moe_experts_stage2_8core(dn_p, dn_s, act_buf, ad_buf, act_idx, act_w, 0, m_out_opt, test_edim, test_mdim, 8)
    for i in range(test_edim):
        if m_out_opt[i] != 0.0:
            raise Error("Expected 0.0 for active_count=0")

    # Test active_count = 8 with num_tasks = 7 (chunk = 64 // 7 = 9, 9 % 4 == 1) and num_tasks = 128 (> rows)
    run_moe_experts_stage1_8core(gu_p, gu_s, x, m_deint, act_idx, 8, gu_out, act_buf, test_edim, test_mdim, 1)
    run_moe_experts_stage2_8core(dn_p, dn_s, act_buf, ad_buf, act_idx, act_w, 8, m_out_ref, test_edim, test_mdim, 1)

    run_moe_experts_stage1_8core(gu_p, gu_s, x, m_deint, act_idx, 8, gu_out, act_buf, test_edim, test_mdim, 7)
    run_moe_experts_stage2_8core(dn_p, dn_s, act_buf, ad_buf, act_idx, act_w, 8, m_out_opt, test_edim, test_mdim, 7)
    for i in range(test_edim):
        if abs(m_out_ref[i] - m_out_opt[i]) > 1e-6:
            raise Error("Mismatch for chunk % 4 == 1 (num_tasks=7)")

    run_moe_experts_stage1_8core(gu_p, gu_s, x, m_deint, act_idx, 8, gu_out, act_buf, test_edim, test_mdim, 128)
    run_moe_experts_stage2_8core(dn_p, dn_s, act_buf, ad_buf, act_idx, act_w, 8, m_out_opt, test_edim, test_mdim, 128)
    for i in range(test_edim):
        if abs(m_out_ref[i] - m_out_opt[i]) > 1e-6:
            raise Error("Mismatch for num_tasks > rows (num_tasks=128)")
    print("  [Test 4: MoE Stage 1/2 Edge Cases (active_count=0,8, chunk%4=1, tasks>rows)]: PASS")

    gu_p.free()
    gu_s.free()
    dn_p.free()
    dn_s.free()
    act_idx.free()
    act_w.free()
    m_deint.free()
    gu_out.free()
    act_buf.free()
    ad_buf.free()
    m_out_ref.free()
    m_out_opt.free()
    w_packed.free()
    w_scales.free()
    x.free()
    out_ref.free()
    out_opt.free()
    out_scalar.free()
    print("  [PASS] All 4-Row Register Tiling & Zero-Alloc Tests Verified!")
