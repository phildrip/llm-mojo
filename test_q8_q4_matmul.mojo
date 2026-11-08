from std.memory import alloc
from std.collections import InlineArray
from std.time import perf_counter_ns
from std.math import abs, exp, round
from std.algorithm import parallelize
from moe_kernels import (
    q4_matmul_vec,
    q4_matmul_batch5,
    compute_subbatch_set_cover_moe_weights,
    run_moe_coalesced_core,
    gelu_and_mul_moe,
)


@always_inline
def run_moe_coalesced_fast(
    sub_b: Int,
    experts_gate_up_packed: UnsafePointer[UInt8, _],
    experts_gate_up_scales: UnsafePointer[Float32, _],
    experts_down_packed: UnsafePointer[UInt8, _],
    experts_down_scales: UnsafePointer[Float32, _],
    batch_in_deint: UnsafePointer[Float32, _],
    tok_active_count: UnsafePointer[Int, _],
    tok_active_idx: UnsafePointer[Int, _],
    tok_active_weights: UnsafePointer[Float32, _],
    bucket_counts: UnsafePointer[Int, MutUntrackedOrigin],
    bucket_toks: UnsafePointer[Int, MutUntrackedOrigin],
    bucket_weights: UnsafePointer[Float32, MutUntrackedOrigin],
    unique_experts: UnsafePointer[Int, MutUntrackedOrigin],
    unique_base_slot: UnsafePointer[Int, MutUntrackedOrigin],
    gu_out_scratch: UnsafePointer[Float32, MutUntrackedOrigin],
    act_scratch: UnsafePointer[Float32, MutUntrackedOrigin],
    act_deint_scratch: UnsafePointer[Float32, MutUntrackedOrigin],
    batch_moe_out: UnsafePointer[Float32, MutUntrackedOrigin],
    embed_dim: Int = 2816,
    expert_mlp_dim: Int = 704,
    num_experts: Int = 128,
    num_tasks: Int = 8,
):
    var half_in = embed_dim // 2
    var c_in_blocks = embed_dim // 32
    var gate_up_rows = expert_mlp_dim * 2
    var down_half_in = expert_mlp_dim // 2
    var down_blocks = expert_mlp_dim // 32
    var tasks = max(1, num_tasks)

    for e in range(num_experts):
        bucket_counts[e] = 0
    for b in range(sub_b):
        var ac = tok_active_count[b]
        for k in range(ac):
            var e = tok_active_idx[b * 8 + k]
            var w = tok_active_weights[b * 8 + k]
            var c = bucket_counts[e]
            bucket_toks[e * 5 + c] = b
            bucket_weights[e * 5 + c] = w
            bucket_counts[e] = c + 1

    var num_unique = 0
    var total_pairs = 0
    for e in range(num_experts):
        if bucket_counts[e] > 0:
            unique_experts[num_unique] = e
            unique_base_slot[num_unique] = total_pairs
            total_pairs += bucket_counts[e]
            num_unique += 1

    def coal_s1(t: Int) capturing -> None:
        var chunk = gate_up_rows // tasks
        var sr = t * chunk
        var er = (t + 1) * chunk if t < tasks - 1 else gate_up_rows
        for u in range(num_unique):
            var e = unique_experts[u]
            var m_count = bucket_counts[e]
            var base_slot = unique_base_slot[u]
            var gu_p = experts_gate_up_packed + e * (gate_up_rows * half_in)
            var gu_s = experts_gate_up_scales + e * (gate_up_rows * c_in_blocks)
            var b_t0 = bucket_toks[e * 5 + 0]
            if m_count == 1:
                var in_e0 = batch_in_deint + b_t0 * embed_dim
                var in_o0 = in_e0 + half_in
                var out0 = gu_out_scratch + base_slot * gate_up_rows
                var r = sr
                while r + 1 < er:
                    var rp0 = r * half_in
                    var rs0 = r * c_in_blocks
                    var rp1 = (r + 1) * half_in
                    var rs1 = (r + 1) * c_in_blocks
                    var acc0 = SIMD[DType.float32, 16](0.0)
                    var acc1 = SIMD[DType.float32, 16](0.0)
                    for blk in range(c_in_blocks):
                        var xe0 = in_e0.load[16](blk * 16)
                        var xo0 = in_o0.load[16](blk * 16)
                        var sc0 = gu_s[rs0 + blk]
                        var bytes0 = gu_p.load[16](rp0 + blk * 16)
                        var we0 = ((bytes0 & 15).cast[DType.int8]() - 8).cast[DType.float32]() * sc0
                        var wo0 = ((bytes0 >> 4).cast[DType.int8]() - 8).cast[DType.float32]() * sc0
                        acc0 = we0.fma(xe0, wo0.fma(xo0, acc0))

                        var sc1 = gu_s[rs1 + blk]
                        var bytes1 = gu_p.load[16](rp1 + blk * 16)
                        var we1 = ((bytes1 & 15).cast[DType.int8]() - 8).cast[DType.float32]() * sc1
                        var wo1 = ((bytes1 >> 4).cast[DType.int8]() - 8).cast[DType.float32]() * sc1
                        acc1 = we1.fma(xe0, wo1.fma(xo0, acc1))
                    out0[r] = acc0.reduce_add()
                    out0[r + 1] = acc1.reduce_add()
                    r += 2
                while r < er:
                    var rp0 = r * half_in
                    var rs0 = r * c_in_blocks
                    var acc0 = SIMD[DType.float32, 16](0.0)
                    for blk in range(c_in_blocks):
                        var xe0 = in_e0.load[16](blk * 16)
                        var xo0 = in_o0.load[16](blk * 16)
                        var sc0 = gu_s[rs0 + blk]
                        var bytes0 = gu_p.load[16](rp0 + blk * 16)
                        var we0 = ((bytes0 & 15).cast[DType.int8]() - 8).cast[DType.float32]() * sc0
                        var wo0 = ((bytes0 >> 4).cast[DType.int8]() - 8).cast[DType.float32]() * sc0
                        acc0 = we0.fma(xe0, wo0.fma(xo0, acc0))
                    out0[r] = acc0.reduce_add()
                    r += 1
            elif m_count == 2:
                var b_t1 = bucket_toks[e * 5 + 1]
                var in_e0 = batch_in_deint + b_t0 * embed_dim
                var in_o0 = in_e0 + half_in
                var in_e1 = batch_in_deint + b_t1 * embed_dim
                var in_o1 = in_e1 + half_in
                var out0 = gu_out_scratch + base_slot * gate_up_rows
                var out1 = gu_out_scratch + (base_slot + 1) * gate_up_rows
                var r = sr
                while r + 1 < er:
                    var rp0 = r * half_in
                    var rs0 = r * c_in_blocks
                    var rp1 = (r + 1) * half_in
                    var rs1 = (r + 1) * c_in_blocks
                    var acc0_r0 = SIMD[DType.float32, 16](0.0)
                    var acc1_r0 = SIMD[DType.float32, 16](0.0)
                    var acc0_r1 = SIMD[DType.float32, 16](0.0)
                    var acc1_r1 = SIMD[DType.float32, 16](0.0)
                    for blk in range(c_in_blocks):
                        var xe0 = in_e0.load[16](blk * 16)
                        var xo0 = in_o0.load[16](blk * 16)
                        var xe1 = in_e1.load[16](blk * 16)
                        var xo1 = in_o1.load[16](blk * 16)

                        var sc0 = gu_s[rs0 + blk]
                        var bytes0 = gu_p.load[16](rp0 + blk * 16)
                        var we0 = ((bytes0 & 15).cast[DType.int8]() - 8).cast[DType.float32]() * sc0
                        var wo0 = ((bytes0 >> 4).cast[DType.int8]() - 8).cast[DType.float32]() * sc0
                        acc0_r0 = we0.fma(xe0, wo0.fma(xo0, acc0_r0))
                        acc1_r0 = we0.fma(xe1, wo0.fma(xo1, acc1_r0))

                        var sc1 = gu_s[rs1 + blk]
                        var bytes1 = gu_p.load[16](rp1 + blk * 16)
                        var we1 = ((bytes1 & 15).cast[DType.int8]() - 8).cast[DType.float32]() * sc1
                        var wo1 = ((bytes1 >> 4).cast[DType.int8]() - 8).cast[DType.float32]() * sc1
                        acc0_r1 = we1.fma(xe0, wo1.fma(xo0, acc0_r1))
                        acc1_r1 = we1.fma(xe1, wo1.fma(xo1, acc1_r1))
                    out0[r] = acc0_r0.reduce_add()
                    out1[r] = acc1_r0.reduce_add()
                    out0[r + 1] = acc0_r1.reduce_add()
                    out1[r + 1] = acc1_r1.reduce_add()
                    r += 2
                while r < er:
                    var rp = r * half_in
                    var rs = r * c_in_blocks
                    var acc0 = SIMD[DType.float32, 16](0.0)
                    var acc1 = SIMD[DType.float32, 16](0.0)
                    for blk in range(c_in_blocks):
                        var sc = gu_s[rs + blk]
                        var bytes = gu_p.load[16](rp + blk * 16)
                        var we = ((bytes & 15).cast[DType.int8]() - 8).cast[DType.float32]() * sc
                        var wo = ((bytes >> 4).cast[DType.int8]() - 8).cast[DType.float32]() * sc
                        acc0 = we.fma(in_e0.load[16](blk * 16), wo.fma(in_o0.load[16](blk * 16), acc0))
                        acc1 = we.fma(in_e1.load[16](blk * 16), wo.fma(in_o1.load[16](blk * 16), acc1))
                    out0[r] = acc0.reduce_add()
                    out1[r] = acc1.reduce_add()
                    r += 1
            elif m_count == 3:
                var b_t1 = bucket_toks[e * 5 + 1]
                var b_t2 = bucket_toks[e * 5 + 2]
                var in_e0 = batch_in_deint + b_t0 * embed_dim
                var in_o0 = in_e0 + half_in
                var in_e1 = batch_in_deint + b_t1 * embed_dim
                var in_o1 = in_e1 + half_in
                var in_e2 = batch_in_deint + b_t2 * embed_dim
                var in_o2 = in_e2 + half_in
                for r in range(sr, er):
                    var rp = r * half_in
                    var rs = r * c_in_blocks
                    var acc0 = SIMD[DType.float32, 16](0.0)
                    var acc1 = SIMD[DType.float32, 16](0.0)
                    var acc2 = SIMD[DType.float32, 16](0.0)
                    for blk in range(c_in_blocks):
                        var sc = gu_s[rs + blk]
                        var bytes = gu_p.load[16](rp + blk * 16)
                        var we = ((bytes & 15).cast[DType.int8]() - 8).cast[DType.float32]() * sc
                        var wo = ((bytes >> 4).cast[DType.int8]() - 8).cast[DType.float32]() * sc
                        acc0 = we.fma(in_e0.load[16](blk * 16), wo.fma(in_o0.load[16](blk * 16), acc0))
                        acc1 = we.fma(in_e1.load[16](blk * 16), wo.fma(in_o1.load[16](blk * 16), acc1))
                        acc2 = we.fma(in_e2.load[16](blk * 16), wo.fma(in_o2.load[16](blk * 16), acc2))
                    gu_out_scratch[base_slot * gate_up_rows + r] = acc0.reduce_add()
                    gu_out_scratch[(base_slot + 1) * gate_up_rows + r] = acc1.reduce_add()
                    gu_out_scratch[(base_slot + 2) * gate_up_rows + r] = acc2.reduce_add()
            elif m_count == 4:
                var b_t1 = bucket_toks[e * 5 + 1]
                var b_t2 = bucket_toks[e * 5 + 2]
                var b_t3 = bucket_toks[e * 5 + 3]
                var in_e0 = batch_in_deint + b_t0 * embed_dim
                var in_o0 = in_e0 + half_in
                var in_e1 = batch_in_deint + b_t1 * embed_dim
                var in_o1 = in_e1 + half_in
                var in_e2 = batch_in_deint + b_t2 * embed_dim
                var in_o2 = in_e2 + half_in
                var in_e3 = batch_in_deint + b_t3 * embed_dim
                var in_o3 = in_e3 + half_in
                for r in range(sr, er):
                    var rp = r * half_in
                    var rs = r * c_in_blocks
                    var acc0 = SIMD[DType.float32, 16](0.0)
                    var acc1 = SIMD[DType.float32, 16](0.0)
                    var acc2 = SIMD[DType.float32, 16](0.0)
                    var acc3 = SIMD[DType.float32, 16](0.0)
                    for blk in range(c_in_blocks):
                        var sc = gu_s[rs + blk]
                        var bytes = gu_p.load[16](rp + blk * 16)
                        var we = ((bytes & 15).cast[DType.int8]() - 8).cast[DType.float32]() * sc
                        var wo = ((bytes >> 4).cast[DType.int8]() - 8).cast[DType.float32]() * sc
                        acc0 = we.fma(in_e0.load[16](blk * 16), wo.fma(in_o0.load[16](blk * 16), acc0))
                        acc1 = we.fma(in_e1.load[16](blk * 16), wo.fma(in_o1.load[16](blk * 16), acc1))
                        acc2 = we.fma(in_e2.load[16](blk * 16), wo.fma(in_o2.load[16](blk * 16), acc2))
                        acc3 = we.fma(in_e3.load[16](blk * 16), wo.fma(in_o3.load[16](blk * 16), acc3))
                    gu_out_scratch[base_slot * gate_up_rows + r] = acc0.reduce_add()
                    gu_out_scratch[(base_slot + 1) * gate_up_rows + r] = acc1.reduce_add()
                    gu_out_scratch[(base_slot + 2) * gate_up_rows + r] = acc2.reduce_add()
                    gu_out_scratch[(base_slot + 3) * gate_up_rows + r] = acc3.reduce_add()
            else:
                var b_t1 = bucket_toks[e * 5 + 1]
                var b_t2 = bucket_toks[e * 5 + 2]
                var b_t3 = bucket_toks[e * 5 + 3]
                var b_t4 = bucket_toks[e * 5 + 4]
                var in_e0 = batch_in_deint + b_t0 * embed_dim
                var in_o0 = in_e0 + half_in
                var in_e1 = batch_in_deint + b_t1 * embed_dim
                var in_o1 = in_e1 + half_in
                var in_e2 = batch_in_deint + b_t2 * embed_dim
                var in_o2 = in_e2 + half_in
                var in_e3 = batch_in_deint + b_t3 * embed_dim
                var in_o3 = in_e3 + half_in
                var in_e4 = batch_in_deint + b_t4 * embed_dim
                var in_o4 = in_e4 + half_in
                for r in range(sr, er):
                    var rp = r * half_in
                    var rs = r * c_in_blocks
                    var acc0 = SIMD[DType.float32, 16](0.0)
                    var acc1 = SIMD[DType.float32, 16](0.0)
                    var acc2 = SIMD[DType.float32, 16](0.0)
                    var acc3 = SIMD[DType.float32, 16](0.0)
                    var acc4 = SIMD[DType.float32, 16](0.0)
                    for blk in range(c_in_blocks):
                        var sc = gu_s[rs + blk]
                        var bytes = gu_p.load[16](rp + blk * 16)
                        var we = ((bytes & 15).cast[DType.int8]() - 8).cast[DType.float32]() * sc
                        var wo = ((bytes >> 4).cast[DType.int8]() - 8).cast[DType.float32]() * sc
                        acc0 = we.fma(in_e0.load[16](blk * 16), wo.fma(in_o0.load[16](blk * 16), acc0))
                        acc1 = we.fma(in_e1.load[16](blk * 16), wo.fma(in_o1.load[16](blk * 16), acc1))
                        acc2 = we.fma(in_e2.load[16](blk * 16), wo.fma(in_o2.load[16](blk * 16), acc2))
                        acc3 = we.fma(in_e3.load[16](blk * 16), wo.fma(in_o3.load[16](blk * 16), acc3))
                        acc4 = we.fma(in_e4.load[16](blk * 16), wo.fma(in_o4.load[16](blk * 16), acc4))
                    gu_out_scratch[base_slot * gate_up_rows + r] = acc0.reduce_add()
                    gu_out_scratch[(base_slot + 1) * gate_up_rows + r] = acc1.reduce_add()
                    gu_out_scratch[(base_slot + 2) * gate_up_rows + r] = acc2.reduce_add()
                    gu_out_scratch[(base_slot + 3) * gate_up_rows + r] = acc3.reduce_add()
                    gu_out_scratch[(base_slot + 4) * gate_up_rows + r] = acc4.reduce_add()

    parallelize[coal_s1](tasks)

    # Fused In-Register GeGLU + Dual-Space Deinterleave (eliminates intermediate act_scratch stores/loads!)
    for p in range(total_pairs):
        var gu_gate = gu_out_scratch + p * gate_up_rows
        var gu_up = gu_gate + expert_mlp_dim
        var ad_e = act_deint_scratch + p * expert_mlp_dim
        var ad_o = ad_e + down_half_in
        for blk in range(down_blocks):
            var g0 = gu_gate.load[16](blk * 32)
            var g1 = gu_gate.load[16](blk * 32 + 16)
            var u0 = gu_up.load[16](blk * 32)
            var u1 = gu_up.load[16](blk * 32 + 16)
            var sig0 = SIMD[DType.float32, 16](1.0) / (
                SIMD[DType.float32, 16](1.0) + exp(SIMD[DType.float32, 16](-1.702) * g0)
            )
            var sig1 = SIMD[DType.float32, 16](1.0) / (
                SIMD[DType.float32, 16](1.0) + exp(SIMD[DType.float32, 16](-1.702) * g1)
            )
            var v0 = (g0 * sig0) * u0
            var v1 = (g1 * sig1) * u1
            ad_e.store[16](blk * 16, v0.deinterleave()[0].join(v1.deinterleave()[0]))
            ad_o.store[16](blk * 16, v0.deinterleave()[1].join(v1.deinterleave()[1]))

    def coal_s2(t: Int) capturing -> None:
        var chunk = embed_dim // tasks
        var sr = t * chunk
        var er = (t + 1) * chunk if t < tasks - 1 else embed_dim
        for b in range(sub_b):
            var out_b = batch_moe_out + b * embed_dim
            for r in range(sr, er):
                out_b[r] = 0.0
        for u in range(num_unique):
            var e = unique_experts[u]
            var m_count = bucket_counts[e]
            var base_slot = unique_base_slot[u]
            var dn_p = experts_down_packed + e * (embed_dim * down_half_in)
            var dn_s = experts_down_scales + e * (embed_dim * down_blocks)
            var b_t0 = bucket_toks[e * 5 + 0]
            var w0 = bucket_weights[e * 5 + 0]
            if m_count == 1:
                var ae0_ptr = act_deint_scratch + base_slot * expert_mlp_dim
                var ao0_ptr = ae0_ptr + down_half_in
                var out0_ptr = batch_moe_out + b_t0 * embed_dim
                var r = sr
                while r + 1 < er:
                    var rp0 = r * down_half_in
                    var rs0 = r * down_blocks
                    var rp1 = (r + 1) * down_half_in
                    var rs1 = (r + 1) * down_blocks
                    var acc0 = SIMD[DType.float32, 16](0.0)
                    var acc1 = SIMD[DType.float32, 16](0.0)
                    for blk in range(down_blocks):
                        var ae0 = ae0_ptr.load[16](blk * 16)
                        var ao0 = ao0_ptr.load[16](blk * 16)
                        var sc0 = dn_s[rs0 + blk]
                        var bytes0 = dn_p.load[16](rp0 + blk * 16)
                        var we0 = ((bytes0 & 15).cast[DType.int8]() - 8).cast[DType.float32]() * sc0
                        var wo0 = ((bytes0 >> 4).cast[DType.int8]() - 8).cast[DType.float32]() * sc0
                        acc0 = we0.fma(ae0, wo0.fma(ao0, acc0))

                        var sc1 = dn_s[rs1 + blk]
                        var bytes1 = dn_p.load[16](rp1 + blk * 16)
                        var we1 = ((bytes1 & 15).cast[DType.int8]() - 8).cast[DType.float32]() * sc1
                        var wo1 = ((bytes1 >> 4).cast[DType.int8]() - 8).cast[DType.float32]() * sc1
                        acc1 = we1.fma(ae0, wo1.fma(ao0, acc1))
                    out0_ptr[r] += acc0.reduce_add() * w0
                    out0_ptr[r + 1] += acc1.reduce_add() * w0
                    r += 2
                while r < er:
                    var rp0 = r * down_half_in
                    var rs0 = r * down_blocks
                    var acc0 = SIMD[DType.float32, 16](0.0)
                    for blk in range(down_blocks):
                        var ae0 = ae0_ptr.load[16](blk * 16)
                        var ao0 = ao0_ptr.load[16](blk * 16)
                        var sc0 = dn_s[rs0 + blk]
                        var bytes0 = dn_p.load[16](rp0 + blk * 16)
                        var we0 = ((bytes0 & 15).cast[DType.int8]() - 8).cast[DType.float32]() * sc0
                        var wo0 = ((bytes0 >> 4).cast[DType.int8]() - 8).cast[DType.float32]() * sc0
                        acc0 = we0.fma(ae0, wo0.fma(ao0, acc0))
                    out0_ptr[r] += acc0.reduce_add() * w0
                    r += 1
            elif m_count == 2:
                var b_t1 = bucket_toks[e * 5 + 1]
                var w1 = bucket_weights[e * 5 + 1]
                var ae0_ptr = act_deint_scratch + base_slot * expert_mlp_dim
                var ao0_ptr = ae0_ptr + down_half_in
                var ae1_ptr = act_deint_scratch + (base_slot + 1) * expert_mlp_dim
                var ao1_ptr = ae1_ptr + down_half_in
                var out0_ptr = batch_moe_out + b_t0 * embed_dim
                var out1_ptr = batch_moe_out + b_t1 * embed_dim
                var r = sr
                while r + 1 < er:
                    var rp0 = r * down_half_in
                    var rs0 = r * down_blocks
                    var rp1 = (r + 1) * down_half_in
                    var rs1 = (r + 1) * down_blocks
                    var acc0_r0 = SIMD[DType.float32, 16](0.0)
                    var acc1_r0 = SIMD[DType.float32, 16](0.0)
                    var acc0_r1 = SIMD[DType.float32, 16](0.0)
                    var acc1_r1 = SIMD[DType.float32, 16](0.0)
                    for blk in range(down_blocks):
                        var ae0 = ae0_ptr.load[16](blk * 16)
                        var ao0 = ao0_ptr.load[16](blk * 16)
                        var ae1 = ae1_ptr.load[16](blk * 16)
                        var ao1 = ao1_ptr.load[16](blk * 16)

                        var sc0 = dn_s[rs0 + blk]
                        var bytes0 = dn_p.load[16](rp0 + blk * 16)
                        var we0 = ((bytes0 & 15).cast[DType.int8]() - 8).cast[DType.float32]() * sc0
                        var wo0 = ((bytes0 >> 4).cast[DType.int8]() - 8).cast[DType.float32]() * sc0
                        acc0_r0 = we0.fma(ae0, wo0.fma(ao0, acc0_r0))
                        acc1_r0 = we0.fma(ae1, wo0.fma(ao1, acc1_r0))

                        var sc1 = dn_s[rs1 + blk]
                        var bytes1 = dn_p.load[16](rp1 + blk * 16)
                        var we1 = ((bytes1 & 15).cast[DType.int8]() - 8).cast[DType.float32]() * sc1
                        var wo1 = ((bytes1 >> 4).cast[DType.int8]() - 8).cast[DType.float32]() * sc1
                        acc0_r1 = we1.fma(ae0, wo1.fma(ao0, acc0_r1))
                        acc1_r1 = we1.fma(ae1, wo1.fma(ao1, acc1_r1))
                    out0_ptr[r] += acc0_r0.reduce_add() * w0
                    out1_ptr[r] += acc1_r0.reduce_add() * w1
                    out0_ptr[r + 1] += acc0_r1.reduce_add() * w0
                    out1_ptr[r + 1] += acc1_r1.reduce_add() * w1
                    r += 2
                while r < er:
                    var rp = r * down_half_in
                    var rs = r * down_blocks
                    var acc0 = SIMD[DType.float32, 16](0.0)
                    var acc1 = SIMD[DType.float32, 16](0.0)
                    for blk in range(down_blocks):
                        var sc = dn_s[rs + blk]
                        var bytes = dn_p.load[16](rp + blk * 16)
                        var we = ((bytes & 15).cast[DType.int8]() - 8).cast[DType.float32]() * sc
                        var wo = ((bytes >> 4).cast[DType.int8]() - 8).cast[DType.float32]() * sc
                        acc0 = we.fma(ae0_ptr.load[16](blk * 16), wo.fma(ao0_ptr.load[16](blk * 16), acc0))
                        acc1 = we.fma(ae1_ptr.load[16](blk * 16), wo.fma(ao1_ptr.load[16](blk * 16), acc1))
                    out0_ptr[r] += acc0.reduce_add() * w0
                    out1_ptr[r] += acc1.reduce_add() * w1
                    r += 1
            elif m_count == 3:
                var b_t1 = bucket_toks[e * 5 + 1]
                var b_t2 = bucket_toks[e * 5 + 2]
                var w1 = bucket_weights[e * 5 + 1]
                var w2 = bucket_weights[e * 5 + 2]
                var ae0_ptr = act_deint_scratch + base_slot * expert_mlp_dim
                var ao0_ptr = ae0_ptr + down_half_in
                var ae1_ptr = act_deint_scratch + (base_slot + 1) * expert_mlp_dim
                var ao1_ptr = ae1_ptr + down_half_in
                var ae2_ptr = act_deint_scratch + (base_slot + 2) * expert_mlp_dim
                var ao2_ptr = ae2_ptr + down_half_in
                for r in range(sr, er):
                    var rp = r * down_half_in
                    var rs = r * down_blocks
                    var acc0 = SIMD[DType.float32, 16](0.0)
                    var acc1 = SIMD[DType.float32, 16](0.0)
                    var acc2 = SIMD[DType.float32, 16](0.0)
                    for blk in range(down_blocks):
                        var sc = dn_s[rs + blk]
                        var bytes = dn_p.load[16](rp + blk * 16)
                        var we = ((bytes & 15).cast[DType.int8]() - 8).cast[DType.float32]() * sc
                        var wo = ((bytes >> 4).cast[DType.int8]() - 8).cast[DType.float32]() * sc
                        acc0 = we.fma(ae0_ptr.load[16](blk * 16), wo.fma(ao0_ptr.load[16](blk * 16), acc0))
                        acc1 = we.fma(ae1_ptr.load[16](blk * 16), wo.fma(ao1_ptr.load[16](blk * 16), acc1))
                        acc2 = we.fma(ae2_ptr.load[16](blk * 16), wo.fma(ao2_ptr.load[16](blk * 16), acc2))
                    batch_moe_out[b_t0 * embed_dim + r] += acc0.reduce_add() * w0
                    batch_moe_out[b_t1 * embed_dim + r] += acc1.reduce_add() * w1
                    batch_moe_out[b_t2 * embed_dim + r] += acc2.reduce_add() * w2
            elif m_count == 4:
                var b_t1 = bucket_toks[e * 5 + 1]
                var b_t2 = bucket_toks[e * 5 + 2]
                var b_t3 = bucket_toks[e * 5 + 3]
                var w1 = bucket_weights[e * 5 + 1]
                var w2 = bucket_weights[e * 5 + 2]
                var w3 = bucket_weights[e * 5 + 3]
                var ae0_ptr = act_deint_scratch + base_slot * expert_mlp_dim
                var ao0_ptr = ae0_ptr + down_half_in
                var ae1_ptr = act_deint_scratch + (base_slot + 1) * expert_mlp_dim
                var ao1_ptr = ae1_ptr + down_half_in
                var ae2_ptr = act_deint_scratch + (base_slot + 2) * expert_mlp_dim
                var ao2_ptr = ae2_ptr + down_half_in
                var ae3_ptr = act_deint_scratch + (base_slot + 3) * expert_mlp_dim
                var ao3_ptr = ae3_ptr + down_half_in
                for r in range(sr, er):
                    var rp = r * down_half_in
                    var rs = r * down_blocks
                    var acc0 = SIMD[DType.float32, 16](0.0)
                    var acc1 = SIMD[DType.float32, 16](0.0)
                    var acc2 = SIMD[DType.float32, 16](0.0)
                    var acc3 = SIMD[DType.float32, 16](0.0)
                    for blk in range(down_blocks):
                        var sc = dn_s[rs + blk]
                        var bytes = dn_p.load[16](rp + blk * 16)
                        var we = ((bytes & 15).cast[DType.int8]() - 8).cast[DType.float32]() * sc
                        var wo = ((bytes >> 4).cast[DType.int8]() - 8).cast[DType.float32]() * sc
                        acc0 = we.fma(ae0_ptr.load[16](blk * 16), wo.fma(ao0_ptr.load[16](blk * 16), acc0))
                        acc1 = we.fma(ae1_ptr.load[16](blk * 16), wo.fma(ao1_ptr.load[16](blk * 16), acc1))
                        acc2 = we.fma(ae2_ptr.load[16](blk * 16), wo.fma(ao2_ptr.load[16](blk * 16), acc2))
                        acc3 = we.fma(ae3_ptr.load[16](blk * 16), wo.fma(ao3_ptr.load[16](blk * 16), acc3))
                    batch_moe_out[b_t0 * embed_dim + r] += acc0.reduce_add() * w0
                    batch_moe_out[b_t1 * embed_dim + r] += acc1.reduce_add() * w1
                    batch_moe_out[b_t2 * embed_dim + r] += acc2.reduce_add() * w2
                    batch_moe_out[b_t3 * embed_dim + r] += acc3.reduce_add() * w3
            else:
                var b_t1 = bucket_toks[e * 5 + 1]
                var b_t2 = bucket_toks[e * 5 + 2]
                var b_t3 = bucket_toks[e * 5 + 3]
                var b_t4 = bucket_toks[e * 5 + 4]
                var w1 = bucket_weights[e * 5 + 1]
                var w2 = bucket_weights[e * 5 + 2]
                var w3 = bucket_weights[e * 5 + 3]
                var w4 = bucket_weights[e * 5 + 4]
                var ae0_ptr = act_deint_scratch + base_slot * expert_mlp_dim
                var ao0_ptr = ae0_ptr + down_half_in
                var ae1_ptr = act_deint_scratch + (base_slot + 1) * expert_mlp_dim
                var ao1_ptr = ae1_ptr + down_half_in
                var ae2_ptr = act_deint_scratch + (base_slot + 2) * expert_mlp_dim
                var ao2_ptr = ae2_ptr + down_half_in
                var ae3_ptr = act_deint_scratch + (base_slot + 3) * expert_mlp_dim
                var ao3_ptr = ae3_ptr + down_half_in
                var ae4_ptr = act_deint_scratch + (base_slot + 4) * expert_mlp_dim
                var ao4_ptr = ae4_ptr + down_half_in
                for r in range(sr, er):
                    var rp = r * down_half_in
                    var rs = r * down_blocks
                    var acc0 = SIMD[DType.float32, 16](0.0)
                    var acc1 = SIMD[DType.float32, 16](0.0)
                    var acc2 = SIMD[DType.float32, 16](0.0)
                    var acc3 = SIMD[DType.float32, 16](0.0)
                    var acc4 = SIMD[DType.float32, 16](0.0)
                    for blk in range(down_blocks):
                        var sc = dn_s[rs + blk]
                        var bytes = dn_p.load[16](rp + blk * 16)
                        var we = ((bytes & 15).cast[DType.int8]() - 8).cast[DType.float32]() * sc
                        var wo = ((bytes >> 4).cast[DType.int8]() - 8).cast[DType.float32]() * sc
                        acc0 = we.fma(ae0_ptr.load[16](blk * 16), wo.fma(ao0_ptr.load[16](blk * 16), acc0))
                        acc1 = we.fma(ae1_ptr.load[16](blk * 16), wo.fma(ao1_ptr.load[16](blk * 16), acc1))
                        acc2 = we.fma(ae2_ptr.load[16](blk * 16), wo.fma(ao2_ptr.load[16](blk * 16), acc2))
                        acc3 = we.fma(ae3_ptr.load[16](blk * 16), wo.fma(ao3_ptr.load[16](blk * 16), acc3))
                        acc4 = we.fma(ae4_ptr.load[16](blk * 16), wo.fma(ao4_ptr.load[16](blk * 16), acc4))
                    batch_moe_out[b_t0 * embed_dim + r] += acc0.reduce_add() * w0
                    batch_moe_out[b_t1 * embed_dim + r] += acc1.reduce_add() * w1
                    batch_moe_out[b_t2 * embed_dim + r] += acc2.reduce_add() * w2
                    batch_moe_out[b_t3 * embed_dim + r] += acc3.reduce_add() * w3
                    batch_moe_out[b_t4 * embed_dim + r] += acc4.reduce_add() * w4

    parallelize[coal_s2](tasks)


def main() raises:
    print("=== TDD Benchmark: 2-Row Tiled + Hoisted-Scale FMA + Fused GeGLU MoE (llm-mojo-xr6.4) ===")
    var sub_b = 5
    var embed_dim = 2816
    var expert_mlp_dim = 704
    var num_experts = 128
    var half_in = embed_dim // 2
    var c_in_blocks = embed_dim // 32
    var gate_up_rows = expert_mlp_dim * 2
    var down_half_in = expert_mlp_dim // 2
    var down_blocks = expert_mlp_dim // 32

    var gu_p = alloc[UInt8](num_experts * gate_up_rows * half_in)
    var gu_s = alloc[Float32](num_experts * gate_up_rows * c_in_blocks)
    var dn_p = alloc[UInt8](num_experts * embed_dim * down_half_in)
    var dn_s = alloc[Float32](num_experts * embed_dim * down_blocks)

    for i in range(num_experts * gate_up_rows * half_in):
        gu_p[i] = UInt8((i * 73 + 19) & 255)
    for i in range(num_experts * gate_up_rows * c_in_blocks):
        gu_s[i] = Float32(0.01)
    for i in range(num_experts * embed_dim * down_half_in):
        dn_p[i] = UInt8((i * 41 + 7) & 255)
    for i in range(num_experts * embed_dim * down_blocks):
        dn_s[i] = Float32(0.01)

    var batch_in_deint = alloc[Float32](sub_b * embed_dim)
    for i in range(sub_b * embed_dim):
        batch_in_deint[i] = Float32((i % 29) - 14) * Float32(0.05)

    var tok_active_count = alloc[Int](sub_b)
    var tok_active_idx = alloc[Int](sub_b * 8)
    var tok_active_weights = alloc[Float32](sub_b * 8)
    # Create a realistic mix of m_count = 1, 2, 3, 4, 5
    var exps: InlineArray[Int, 40] = [
        10, 20, 30, 40, 50, 60, 70, 80,
        10, 20, 30, 41, 51, 61, 71, 81,
        10, 20, 41, 42, 52, 62, 72, 82,
        10, 30, 41, 42, 53, 63, 73, 83,
        10, 20, 30, 40, 53, 64, 74, 84,
    ]
    for b in range(sub_b):
        tok_active_count[b] = 6
        for k in range(6):
            tok_active_idx[b * 8 + k] = exps[b * 8 + k]
            tok_active_weights[b * 8 + k] = Float32(1.0 / 6.0)

    var bucket_counts = alloc[Int](num_experts)
    var bucket_toks = alloc[Int](num_experts * 5)
    var bucket_weights = alloc[Float32](num_experts * 5)
    var unique_experts = alloc[Int](num_experts)
    var unique_base_slot = alloc[Int](num_experts)
    var gu_out_scratch = alloc[Float32](40 * gate_up_rows)
    var act_scratch = alloc[Float32](40 * expert_mlp_dim)
    var act_deint_scratch = alloc[Float32](40 * expert_mlp_dim)
    var out_ref = alloc[Float32](sub_b * embed_dim)
    var out_fast = alloc[Float32](sub_b * embed_dim)

    run_moe_coalesced_core(
        sub_b, gu_p, gu_s, dn_p, dn_s, batch_in_deint,
        tok_active_count, tok_active_idx, tok_active_weights,
        bucket_counts, bucket_toks, bucket_weights, unique_experts, unique_base_slot,
        gu_out_scratch, act_scratch, act_deint_scratch, out_ref,
        embed_dim, expert_mlp_dim, num_experts, 8
    )

    run_moe_coalesced_fast(
        sub_b, gu_p, gu_s, dn_p, dn_s, batch_in_deint,
        tok_active_count, tok_active_idx, tok_active_weights,
        bucket_counts, bucket_toks, bucket_weights, unique_experts, unique_base_slot,
        gu_out_scratch, act_scratch, act_deint_scratch, out_fast,
        embed_dim, expert_mlp_dim, num_experts, 8
    )

    var max_diff: Float32 = 0.0
    for i in range(sub_b * embed_dim):
        var d = abs(out_ref[i] - out_fast[i])
        if d > max_diff:
            max_diff = d
    print("  Max diff (Fast vs Ref):", max_diff)
    if max_diff > 1e-4:
        raise Error("FAIL: max_diff > 1e-4")

    var iters = 100
    var t0 = perf_counter_ns()
    for _ in range(iters):
        run_moe_coalesced_core(
            sub_b, gu_p, gu_s, dn_p, dn_s, batch_in_deint,
            tok_active_count, tok_active_idx, tok_active_weights,
            bucket_counts, bucket_toks, bucket_weights, unique_experts, unique_base_slot,
            gu_out_scratch, act_scratch, act_deint_scratch, out_ref,
            embed_dim, expert_mlp_dim, num_experts, 8
        )
    var t1 = perf_counter_ns()

    var t2 = perf_counter_ns()
    for _ in range(iters):
        run_moe_coalesced_fast(
            sub_b, gu_p, gu_s, dn_p, dn_s, batch_in_deint,
            tok_active_count, tok_active_idx, tok_active_weights,
            bucket_counts, bucket_toks, bucket_weights, unique_experts, unique_base_slot,
            gu_out_scratch, act_scratch, act_deint_scratch, out_fast,
            embed_dim, expert_mlp_dim, num_experts, 8
        )
    var t3 = perf_counter_ns()

    var ref_ms = Float64(t1 - t0) / 1e6 / Float64(iters)
    var fast_ms = Float64(t3 - t2) / 1e6 / Float64(iters)
    print("  Ref Coalesced MoE: ", ref_ms, "ms/layer (", ref_ms * 30.0, "ms / 30 layers)")
    print("  Fast Coalesced MoE:", fast_ms, "ms/layer (", fast_ms * 30.0, "ms / 30 layers)")
    print("  Speedup:           ", ref_ms / fast_ms, "x")

    gu_p.free()
    gu_s.free()
    dn_p.free()
    dn_s.free()
    batch_in_deint.free()
    tok_active_count.free()
    tok_active_idx.free()
    tok_active_weights.free()
    bucket_counts.free()
    bucket_toks.free()
    bucket_weights.free()
    unique_experts.free()
    unique_base_slot.free()
    gu_out_scratch.free()
    act_scratch.free()
    act_deint_scratch.free()
    out_ref.free()
    out_fast.free()
