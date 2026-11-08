from std.memory import alloc
from std.collections import InlineArray
from std.math import sqrt, exp, tanh, log
from std.algorithm import parallelize


@always_inline
def rms_norm[
    in_type: DType = DType.float32,
    out_type: DType = DType.float32,
](
    x: UnsafePointer[Scalar[in_type], _],
    w: UnsafePointer[Float32, _],
    out_ptr: UnsafePointer[Scalar[out_type], MutAnyOrigin],
    size: Int,
):
    var acc = SIMD[DType.float32, 8](0.0)
    for i in range(0, size, 8):
        var v = x.load[8](i).cast[DType.float32]()
        acc = v.fma(v, acc)
    var sum_sq = acc.reduce_add()
    var inv_rms = 1.0 / sqrt(sum_sq / Float32(size) + 1e-6)
    var inv_rms_vec = SIMD[DType.float32, 8](inv_rms)
    for i in range(0, size, 8):
        var v = x.load[8](i).cast[DType.float32]()
        var weight = w.load[8](i)
        out_ptr.store[8](i, (v * inv_rms_vec * weight).cast[out_type]())


@always_inline
def rms_norm_noscale[
    in_type: DType = DType.float32,
    out_type: DType = DType.float32,
](
    x: UnsafePointer[Scalar[in_type], _],
    out_ptr: UnsafePointer[Scalar[out_type], MutAnyOrigin],
    size: Int,
):
    var acc = SIMD[DType.float32, 8](0.0)
    for i in range(0, size, 8):
        var v = x.load[8](i).cast[DType.float32]()
        acc = v.fma(v, acc)
    var sum_sq = acc.reduce_add()
    var inv_rms = 1.0 / sqrt(sum_sq / Float32(size) + 1e-6)
    var inv_rms_vec = SIMD[DType.float32, 8](inv_rms)
    for i in range(0, size, 8):
        var v = x.load[8](i).cast[DType.float32]()
        out_ptr.store[8](i, (v * inv_rms_vec).cast[out_type]())


@always_inline
def deinterleave_batch_to_f32[
    in_type: DType = DType.float16
](
    src: UnsafePointer[Scalar[in_type], _],
    dst_deint: UnsafePointer[Float32, MutAnyOrigin],
    B: Int,
    c_in: Int,
    in_stride: Int = -1,
):
    """Converts [B, c_in] activations (Float16 or Float32) into Dual-Space Deinterleaved Float32 layout
    where each 32-element block is stored as [16 even floats, 16 odd floats]."""
    var x_stride = in_stride if in_stride >= 0 else c_in
    var c_in_blocks = c_in // 32
    for b in range(B):
        var src_b = src + b * x_stride
        var dst_b = dst_deint + b * c_in
        for blk in range(c_in_blocks):
            var v0 = src_b.load[16](blk * 32).cast[DType.float32]()
            var v1 = src_b.load[16](blk * 32 + 16).cast[DType.float32]()
            var ev = v0.deinterleave()[0].join(v1.deinterleave()[0])
            var od = v0.deinterleave()[1].join(v1.deinterleave()[1])
            dst_b.store[16](blk * 32, ev)
            dst_b.store[16](blk * 32 + 16, od)


@always_inline
def gelu_and_mul_to_deinterleaved_f32[
    in_type: DType = DType.float32
](
    gate_up: UnsafePointer[Scalar[in_type], _],
    dst_deint: UnsafePointer[Float32, MutAnyOrigin],
    B: Int,
    dim: Int,
):
    """Computes GeGLU on [B, 2*dim] (gate = [0..dim-1], up = [dim..2*dim-1]) and writes
    directly into Dual-Space Deinterleaved Float32 [B, dim] in a single pass."""
    var num_blocks = dim // 32
    for b in range(B):
        var g_ptr = gate_up + b * (dim * 2)
        var u_ptr = g_ptr + dim
        var d_ptr = dst_deint + b * dim
        for blk in range(num_blocks):
            var off = blk * 32
            var g0 = g_ptr.load[16](off).cast[DType.float32]()
            var u0 = u_ptr.load[16](off).cast[DType.float32]()
            var sig0 = 1.0 / (1.0 + exp(SIMD[DType.float32, 16](-1.702) * g0))
            var act0 = g0 * sig0 * u0

            var g1 = g_ptr.load[16](off + 16).cast[DType.float32]()
            var u1 = u_ptr.load[16](off + 16).cast[DType.float32]()
            var sig1 = 1.0 / (1.0 + exp(SIMD[DType.float32, 16](-1.702) * g1))
            var act1 = g1 * sig1 * u1

            d_ptr.store[16](off, act0.deinterleave()[0].join(act1.deinterleave()[0]))
            d_ptr.store[16](off + 16, act0.deinterleave()[1].join(act1.deinterleave()[1]))


@always_inline
def q4_moe_expert_single_thread(
    W_packed: UnsafePointer[UInt8, _],
    scales: UnsafePointer[Float32, _],
    X_deint: UnsafePointer[Float32, _],
    out_ptr: UnsafePointer[Float32, MutAnyOrigin],
    B: Int,
    c_in: Int,
    c_out: Int,
    w_deint_scratch: UnsafePointer[Float32, MutAnyOrigin],
):
    """Single-thread L1-resident Dual-Space Deinterleaved Q4 GEMM for one MoE expert.
    Uses SIMD[DType.float32, 8] (1 YMM register per accumulator) with 2-row x 4-token
    register blocking (8 acc + 2 weight + 4 input = 14 YMM registers, zero spills!)."""
    var c_in_packed = c_in // 2
    var c_in_blocks = c_in // 32
    var w_deint0 = w_deint_scratch
    var w_deint1 = w_deint_scratch + c_in

    var r = 0
    while r + 1 < c_out:
        var rp0 = r * c_in_packed
        var rs0 = r * c_in_blocks
        var rp1 = (r + 1) * c_in_packed
        var rs1 = (r + 1) * c_in_blocks

        if B <= 4:
            # Direct in-register deinterleaved unpacking without L1 scratch roundtrip
            var a0_0 = SIMD[DType.float32, 8](0.0)
            var a0_1 = SIMD[DType.float32, 8](0.0)
            var a0_2 = SIMD[DType.float32, 8](0.0)
            var a0_3 = SIMD[DType.float32, 8](0.0)
            var a1_0 = SIMD[DType.float32, 8](0.0)
            var a1_1 = SIMD[DType.float32, 8](0.0)
            var a1_2 = SIMD[DType.float32, 8](0.0)
            var a1_3 = SIMD[DType.float32, 8](0.0)

            for blk in range(c_in_blocks):
                var sc0 = scales[rs0 + blk]
                var sc1 = scales[rs1 + blk]
                var x_off = blk * 32

                # Sub-block 0 (first 8 bytes = 8 even + 8 odd)
                var b0_lo = W_packed.load[8](rp0 + blk * 16)
                var we0 = ((b0_lo & 15).cast[DType.int8]() - 8).cast[DType.float32]() * sc0
                var wo0 = ((b0_lo >> 4).cast[DType.int8]() - 8).cast[DType.float32]() * sc0

                var b1_lo = W_packed.load[8](rp1 + blk * 16)
                var we1 = ((b1_lo & 15).cast[DType.int8]() - 8).cast[DType.float32]() * sc1
                var wo1 = ((b1_lo >> 4).cast[DType.int8]() - 8).cast[DType.float32]() * sc1

                var xe0 = X_deint.load[8](0 * c_in + x_off)
                var xo0 = X_deint.load[8](0 * c_in + x_off + 16)
                a0_0 = we0.fma(xe0, wo0.fma(xo0, a0_0))
                a1_0 = we1.fma(xe0, wo1.fma(xo0, a1_0))
                if B > 1:
                    var xe1 = X_deint.load[8](1 * c_in + x_off)
                    var xo1 = X_deint.load[8](1 * c_in + x_off + 16)
                    a0_1 = we0.fma(xe1, wo0.fma(xo1, a0_1))
                    a1_1 = we1.fma(xe1, wo1.fma(xo1, a1_1))
                if B > 2:
                    var xe2 = X_deint.load[8](2 * c_in + x_off)
                    var xo2 = X_deint.load[8](2 * c_in + x_off + 16)
                    a0_2 = we0.fma(xe2, wo0.fma(xo2, a0_2))
                    a1_2 = we1.fma(xe2, wo1.fma(xo2, a1_2))
                if B > 3:
                    var xe3 = X_deint.load[8](3 * c_in + x_off)
                    var xo3 = X_deint.load[8](3 * c_in + x_off + 16)
                    a0_3 = we0.fma(xe3, wo0.fma(xo3, a0_3))
                    a1_3 = we1.fma(xe3, wo1.fma(xo3, a1_3))

                # Sub-block 1 (second 8 bytes = 8 even + 8 odd)
                var b0_hi = W_packed.load[8](rp0 + blk * 16 + 8)
                we0 = ((b0_hi & 15).cast[DType.int8]() - 8).cast[DType.float32]() * sc0
                wo0 = ((b0_hi >> 4).cast[DType.int8]() - 8).cast[DType.float32]() * sc0

                var b1_hi = W_packed.load[8](rp1 + blk * 16 + 8)
                we1 = ((b1_hi & 15).cast[DType.int8]() - 8).cast[DType.float32]() * sc1
                wo1 = ((b1_hi >> 4).cast[DType.int8]() - 8).cast[DType.float32]() * sc1

                xe0 = X_deint.load[8](0 * c_in + x_off + 8)
                xo0 = X_deint.load[8](0 * c_in + x_off + 24)
                a0_0 = we0.fma(xe0, wo0.fma(xo0, a0_0))
                a1_0 = we1.fma(xe0, wo1.fma(xo0, a1_0))
                if B > 1:
                    var xe1 = X_deint.load[8](1 * c_in + x_off + 8)
                    var xo1 = X_deint.load[8](1 * c_in + x_off + 24)
                    a0_1 = we0.fma(xe1, wo0.fma(xo1, a0_1))
                    a1_1 = we1.fma(xe1, wo1.fma(xo1, a1_1))
                if B > 2:
                    var xe2 = X_deint.load[8](2 * c_in + x_off + 8)
                    var xo2 = X_deint.load[8](2 * c_in + x_off + 24)
                    a0_2 = we0.fma(xe2, wo0.fma(xo2, a0_2))
                    a1_2 = we1.fma(xe2, wo1.fma(xo2, a1_2))
                if B > 3:
                    var xe3 = X_deint.load[8](3 * c_in + x_off + 8)
                    var xo3 = X_deint.load[8](3 * c_in + x_off + 24)
                    a0_3 = we0.fma(xe3, wo0.fma(xo3, a0_3))
                    a1_3 = we1.fma(xe3, wo1.fma(xo3, a1_3))

            out_ptr[0 * c_out + r] = a0_0.reduce_add()
            out_ptr[0 * c_out + r + 1] = a1_0.reduce_add()
            if B > 1:
                out_ptr[1 * c_out + r] = a0_1.reduce_add()
                out_ptr[1 * c_out + r + 1] = a1_1.reduce_add()
            if B > 2:
                out_ptr[2 * c_out + r] = a0_2.reduce_add()
                out_ptr[2 * c_out + r + 1] = a1_2.reduce_add()
            if B > 3:
                out_ptr[3 * c_out + r] = a0_3.reduce_add()
                out_ptr[3 * c_out + r + 1] = a1_3.reduce_add()
        else:
            # Unpack 2 rows into L1 scratch once, then 2x4 SIMD8 FMA (14 YMM regs, 0 spills!)
            for blk in range(c_in_blocks):
                var sc0 = scales[rs0 + blk]
                var b0 = W_packed.load[16](rp0 + blk * 16)
                w_deint0.store[16](blk * 32, ((b0 & 15).cast[DType.int8]() - 8).cast[DType.float32]() * sc0)
                w_deint0.store[16](blk * 32 + 16, ((b0 >> 4).cast[DType.int8]() - 8).cast[DType.float32]() * sc0)

                var sc1 = scales[rs1 + blk]
                var b1 = W_packed.load[16](rp1 + blk * 16)
                w_deint1.store[16](blk * 32, ((b1 & 15).cast[DType.int8]() - 8).cast[DType.float32]() * sc1)
                w_deint1.store[16](blk * 32 + 16, ((b1 >> 4).cast[DType.int8]() - 8).cast[DType.float32]() * sc1)

            var b = 0
            while b + 3 < B:
                var a0_0 = SIMD[DType.float32, 8](0.0)
                var a0_1 = SIMD[DType.float32, 8](0.0)
                var a0_2 = SIMD[DType.float32, 8](0.0)
                var a0_3 = SIMD[DType.float32, 8](0.0)
                var a1_0 = SIMD[DType.float32, 8](0.0)
                var a1_1 = SIMD[DType.float32, 8](0.0)
                var a1_2 = SIMD[DType.float32, 8](0.0)
                var a1_3 = SIMD[DType.float32, 8](0.0)
                var xb0 = (b + 0) * c_in
                var xb1 = (b + 1) * c_in
                var xb2 = (b + 2) * c_in
                var xb3 = (b + 3) * c_in
                for k in range(0, c_in, 8):
                    var w0 = w_deint0.load[8](k)
                    var w1 = w_deint1.load[8](k)
                    var x0 = X_deint.load[8](xb0 + k)
                    var x1 = X_deint.load[8](xb1 + k)
                    var x2 = X_deint.load[8](xb2 + k)
                    var x3 = X_deint.load[8](xb3 + k)
                    a0_0 = w0.fma(x0, a0_0)
                    a0_1 = w0.fma(x1, a0_1)
                    a0_2 = w0.fma(x2, a0_2)
                    a0_3 = w0.fma(x3, a0_3)
                    a1_0 = w1.fma(x0, a1_0)
                    a1_1 = w1.fma(x1, a1_1)
                    a1_2 = w1.fma(x2, a1_2)
                    a1_3 = w1.fma(x3, a1_3)
                out_ptr[(b + 0) * c_out + r] = a0_0.reduce_add()
                out_ptr[(b + 1) * c_out + r] = a0_1.reduce_add()
                out_ptr[(b + 2) * c_out + r] = a0_2.reduce_add()
                out_ptr[(b + 3) * c_out + r] = a0_3.reduce_add()
                out_ptr[(b + 0) * c_out + r + 1] = a1_0.reduce_add()
                out_ptr[(b + 1) * c_out + r + 1] = a1_1.reduce_add()
                out_ptr[(b + 2) * c_out + r + 1] = a1_2.reduce_add()
                out_ptr[(b + 3) * c_out + r + 1] = a1_3.reduce_add()
                b += 4
            while b < B:
                var a0 = SIMD[DType.float32, 8](0.0)
                var a1 = SIMD[DType.float32, 8](0.0)
                var xb = b * c_in
                for k in range(0, c_in, 8):
                    var x0 = X_deint.load[8](xb + k)
                    a0 = w_deint0.load[8](k).fma(x0, a0)
                    a1 = w_deint1.load[8](k).fma(x0, a1)
                out_ptr[b * c_out + r] = a0.reduce_add()
                out_ptr[b * c_out + r + 1] = a1.reduce_add()
                b += 1
        r += 2


def run_diffusion_moe_experts_parallel(
    experts_gate_up_packed: UnsafePointer[UInt8, _],
    experts_gate_up_scales: UnsafePointer[Float32, _],
    experts_down_packed: UnsafePointer[UInt8, _],
    experts_down_scales: UnsafePointer[Float32, _],
    canvas_moe_in: UnsafePointer[Float16, _],
    canvas_top_idx: UnsafePointer[Int, _],
    canvas_top_weights: UnsafePointer[Float32, _],
    canvas_moe_out: UnsafePointer[Float16, MutAnyOrigin],
    B: Int,
    embed_dim: Int,
    expert_mlp_dim: Int,
    top_k_experts: Int,
    num_tasks: Int,
    prealloc_deint_x: UnsafePointer[Float32, MutAnyOrigin],
    prealloc_work_buf: UnsafePointer[Float32, MutAnyOrigin],
    prealloc_down_out: UnsafePointer[Float32, MutAnyOrigin],
):
    """Expert-parallel Dual-Space Deinterleaved MoE execution for diffusion canvas.
    Replaces 256 per-expert thread-pool barriers and single-threaded gather/GeGLU/scatter
    with a single parallel dispatch across experts + parallel token accumulation.
    Uses caller-provided pre-allocated buffers for zero heap allocations (PASID safety)."""
    var expert_counts = InlineArray[Int, 128](uninitialized=True)
    for e in range(128):
        expert_counts[e] = 0

    for t in range(B):
        var top_w = canvas_top_weights + t * top_k_experts
        var top_i = canvas_top_idx + t * top_k_experts
        for s in range(top_k_experts):
            if top_w[s] > 0.0:
                expert_counts[top_i[s]] += 1

    var expert_offsets = InlineArray[Int, 128](uninitialized=True)
    var write_offsets = InlineArray[Int, 128](uninitialized=True)
    var active_experts = InlineArray[Int, 128](uninitialized=True)
    var num_active_experts = 0
    var total_assignments = 0
    for e in range(128):
        expert_offsets[e] = total_assignments
        write_offsets[e] = total_assignments
        var cnt = expert_counts[e]
        if cnt > 0:
            active_experts[num_active_experts] = e
            num_active_experts += 1
            total_assignments += cnt

    var assignments_t = InlineArray[Int, 2048](uninitialized=True)
    var assignments_slot = InlineArray[Int, 2048](uninitialized=True)
    for i in range(B * top_k_experts):
        assignments_slot[i] = -1

    for t in range(B):
        var top_w = canvas_top_weights + t * top_k_experts
        var top_i = canvas_top_idx + t * top_k_experts
        for s in range(top_k_experts):
            if top_w[s] > 0.0:
                var e = top_i[s]
                var wp = write_offsets[e]
                assignments_t[wp] = t
                assignments_slot[t * top_k_experts + s] = wp
                write_offsets[e] = wp + 1

    # 1. Pre-deinterleave all B token inputs in Float32 once
    var canvas_in_deint = prealloc_deint_x
    deinterleave_batch_to_f32[DType.float16](canvas_moe_in, canvas_in_deint, B, embed_dim)

    # 2. Flat output buffer for all active assignments [total_assignments, embed_dim]
    var all_down_out = prealloc_down_out

    var gate_up_dim = expert_mlp_dim * 2
    var gu_packed_stride = gate_up_dim * embed_dim // 2
    var gu_scales_stride = gate_up_dim * embed_dim // 32
    var dn_packed_stride = embed_dim * expert_mlp_dim // 2
    var dn_scales_stride = embed_dim * expert_mlp_dim // 32

    var tasks = num_tasks
    if tasks > num_active_experts:
        tasks = num_active_experts
    if tasks < 1:
        tasks = 1

    var work_stride = B * embed_dim + B * gate_up_dim + B * expert_mlp_dim + embed_dim * 2
    var work_base = prealloc_work_buf

    def expert_worker(worker_id: Int) capturing -> None:
        var w_base = work_base + worker_id * work_stride
        var local_x = w_base
        var local_gu = local_x + B * embed_dim
        var local_act_deint = local_gu + B * gate_up_dim
        var local_w_scratch = local_act_deint + B * expert_mlp_dim

        # Interleaved stride scheduling across active experts for load balance
        var idx = worker_id
        while idx < num_active_experts:
            var e = active_experts[idx]
            var cnt = expert_counts[e]
            var off = expert_offsets[e]

            # Gather deinterleaved inputs for expert e
            for i in range(cnt):
                var t = assignments_t[off + i]
                var src = canvas_in_deint + t * embed_dim
                var dst = local_x + i * embed_dim
                for j in range(0, embed_dim, 8):
                    dst.store[8](j, src.load[8](j))

            # Gate + Up projection
            q4_moe_expert_single_thread(
                experts_gate_up_packed + e * gu_packed_stride,
                experts_gate_up_scales + e * gu_scales_stride,
                local_x,
                local_gu,
                cnt,
                embed_dim,
                gate_up_dim,
                local_w_scratch,
            )

            # Fused GeGLU + deinterleave
            gelu_and_mul_to_deinterleaved_f32[DType.float32](
                local_gu,
                local_act_deint,
                cnt,
                expert_mlp_dim,
            )

            # Down projection directly into all_down_out[off .. off + cnt]
            q4_moe_expert_single_thread(
                experts_down_packed + e * dn_packed_stride,
                experts_down_scales + e * dn_scales_stride,
                local_act_deint,
                all_down_out + off * embed_dim,
                cnt,
                expert_mlp_dim,
                embed_dim,
                local_w_scratch,
            )

            idx += tasks

    if num_active_experts > 0:
        if tasks == 1:
            expert_worker(0)
        else:
            parallelize[expert_worker](tasks)

    # 3. Accumulate weighted expert outputs per token in Float32, then cast once to Float16
    for t in range(B):
        var dst = canvas_moe_out + t * embed_dim
        var top_w = canvas_top_weights + t * top_k_experts
        for j in range(0, embed_dim, 8):
            var sum_vec = SIMD[DType.float32, 8](0.0)
            for s in range(top_k_experts):
                var wp = assignments_slot[t * top_k_experts + s]
                if wp >= 0:
                    var w_vec = SIMD[DType.float32, 8](top_w[s])
                    var src = all_down_out + wp * embed_dim
                    sum_vec = src.load[8](j).fma(w_vec, sum_vec)
            dst.store[8](j, sum_vec.cast[DType.float16]())


def run_diffusion_moe_experts_parallel(
    experts_gate_up_packed: UnsafePointer[UInt8, _],
    experts_gate_up_scales: UnsafePointer[Float32, _],
    experts_down_packed: UnsafePointer[UInt8, _],
    experts_down_scales: UnsafePointer[Float32, _],
    canvas_moe_in: UnsafePointer[Float16, _],
    canvas_top_idx: UnsafePointer[Int, _],
    canvas_top_weights: UnsafePointer[Float32, _],
    canvas_moe_out: UnsafePointer[Float16, MutAnyOrigin],
    B: Int,
    embed_dim: Int,
    expert_mlp_dim: Int,
    top_k_experts: Int,
    num_tasks: Int,
):
    """Convenience overload that allocates temporary scratch buffers when KFD waves are not active."""
    var gate_up_dim = expert_mlp_dim * 2
    var work_stride = B * embed_dim + B * gate_up_dim + B * expert_mlp_dim + embed_dim * 2
    var tasks = num_tasks if num_tasks >= 1 else 1
    var deint_x = alloc[Float32](B * embed_dim)
    var work_buf = alloc[Float32](tasks * work_stride)
    var down_out = alloc[Float32]((B * top_k_experts if B * top_k_experts > 0 else 1) * embed_dim)
    run_diffusion_moe_experts_parallel(
        experts_gate_up_packed,
        experts_gate_up_scales,
        experts_down_packed,
        experts_down_scales,
        canvas_moe_in,
        canvas_top_idx,
        canvas_top_weights,
        canvas_moe_out,
        B,
        embed_dim,
        expert_mlp_dim,
        top_k_experts,
        num_tasks,
        deint_x,
        work_buf,
        down_out,
    )
    down_out.free()
    work_buf.free()
    deint_x.free()


@always_inline
def q4_moe_gemm_deinterleaved_small_b[
    MAX_B: Int,
    out_type: DType = DType.float32,
](
    W_packed: UnsafePointer[UInt8, _],
    scales: UnsafePointer[Float32, _],
    X_deint: UnsafePointer[Float32, _],
    out_ptr: UnsafePointer[Scalar[out_type], MutAnyOrigin],
    B: Int,
    c_in: Int,
    c_out: Int,
    num_tasks: Int,
    out_stride: Int = -1,
):
    var tasks = num_tasks
    if tasks > c_out:
        tasks = c_out
    if tasks < 1:
        tasks = 1
    var o_stride = out_stride if out_stride >= 0 else c_out
    var chunk_size = ((c_out // tasks + 1) // 2) * 2
    if chunk_size < 2:
        chunk_size = 2
    var c_in_packed = c_in // 2
    var c_in_blocks = c_in // 32

    def worker(t: Int) capturing -> None:
        var start_r = t * chunk_size
        if start_r >= c_out:
            return
        var end_r = (t + 1) * chunk_size if t < tasks - 1 else c_out
        if end_r > c_out:
            end_r = c_out

        var acc0 = InlineArray[SIMD[DType.float32, 16], MAX_B](uninitialized=True)
        var acc1 = InlineArray[SIMD[DType.float32, 16], MAX_B](uninitialized=True)

        var r = start_r
        while r + 1 < end_r:
            var rp0 = r * c_in_packed
            var rs0 = r * c_in_blocks
            var rp1 = (r + 1) * c_in_packed
            var rs1 = (r + 1) * c_in_blocks

            comptime for i in range(MAX_B):
                acc0[i] = SIMD[DType.float32, 16](0.0)
                acc1[i] = SIMD[DType.float32, 16](0.0)

            for blk in range(c_in_blocks):
                var sc0 = scales[rs0 + blk]
                var b0 = W_packed.load[16](rp0 + blk * 16)
                var we0 = ((b0 & 15).cast[DType.int8]() - 8).cast[DType.float32]() * sc0
                var wo0 = ((b0 >> 4).cast[DType.int8]() - 8).cast[DType.float32]() * sc0

                var sc1 = scales[rs1 + blk]
                var b1 = W_packed.load[16](rp1 + blk * 16)
                var we1 = ((b1 & 15).cast[DType.int8]() - 8).cast[DType.float32]() * sc1
                var wo1 = ((b1 >> 4).cast[DType.int8]() - 8).cast[DType.float32]() * sc1

                var x_off = blk * 32
                comptime for i in range(MAX_B):
                    if i < B:
                        var xb = i * c_in + x_off
                        var xe = X_deint.load[16](xb)
                        var xo = X_deint.load[16](xb + 16)
                        acc0[i] = we0.fma(xe, wo0.fma(xo, acc0[i]))
                        acc1[i] = we1.fma(xe, wo1.fma(xo, acc1[i]))

            comptime for i in range(MAX_B):
                if i < B:
                    out_ptr[i * o_stride + r] = acc0[i].reduce_add().cast[out_type]()
                    out_ptr[i * o_stride + r + 1] = acc1[i].reduce_add().cast[out_type]()
            r += 2

        while r < end_r:
            var rp0 = r * c_in_packed
            var rs0 = r * c_in_blocks
            comptime for i in range(MAX_B):
                acc0[i] = SIMD[DType.float32, 16](0.0)
            for blk in range(c_in_blocks):
                var sc0 = scales[rs0 + blk]
                var b0 = W_packed.load[16](rp0 + blk * 16)
                var we0 = ((b0 & 15).cast[DType.int8]() - 8).cast[DType.float32]() * sc0
                var wo0 = ((b0 >> 4).cast[DType.int8]() - 8).cast[DType.float32]() * sc0
                var x_off = blk * 32
                comptime for i in range(MAX_B):
                    if i < B:
                        var xb = i * c_in + x_off
                        var xe = X_deint.load[16](xb)
                        var xo = X_deint.load[16](xb + 16)
                        acc0[i] = we0.fma(xe, wo0.fma(xo, acc0[i]))
            comptime for i in range(MAX_B):
                if i < B:
                    out_ptr[i * o_stride + r] = acc0[i].reduce_add().cast[out_type]()
            r += 1

    if tasks == 1:
        worker(0)
    else:
        parallelize[worker](tasks)


@always_inline
def q4_moe_gemm_deinterleaved[
    out_type: DType = DType.float32
](
    W_packed: UnsafePointer[UInt8, _],
    scales: UnsafePointer[Float32, _],
    X_deint: UnsafePointer[Float32, _],
    out_ptr: UnsafePointer[Scalar[out_type], MutAnyOrigin],
    B: Int,
    c_in: Int,
    c_out: Int,
    num_tasks: Int,
    out_stride: Int = -1,
):
    if B <= 0:
        return
    if B == 1:
        q4_moe_gemm_deinterleaved_small_b[1, out_type](W_packed, scales, X_deint, out_ptr, B, c_in, c_out, num_tasks, out_stride)
    elif B == 2:
        q4_moe_gemm_deinterleaved_small_b[2, out_type](W_packed, scales, X_deint, out_ptr, B, c_in, c_out, num_tasks, out_stride)
    elif B <= 4:
        q4_moe_gemm_deinterleaved_small_b[4, out_type](W_packed, scales, X_deint, out_ptr, B, c_in, c_out, num_tasks, out_stride)
    elif B <= 6:
        q4_moe_gemm_deinterleaved_small_b[6, out_type](W_packed, scales, X_deint, out_ptr, B, c_in, c_out, num_tasks, out_stride)
    else:
        # For B > 6, unpack 2 rows in deinterleaved layout into L1 and tile B by 4
        var tasks = num_tasks
        if tasks > c_out:
            tasks = c_out
        if tasks < 1:
            tasks = 1
        var o_stride = out_stride if out_stride >= 0 else c_out
        var chunk_size = ((c_out // tasks + 1) // 2) * 2
        if chunk_size < 2:
            chunk_size = 2
        var c_in_packed = c_in // 2
        var c_in_blocks = c_in // 32

        def worker_large(t: Int) capturing -> None:
            var start_r = t * chunk_size
            if start_r >= c_out:
                return
            var end_r = (t + 1) * chunk_size if t < tasks - 1 else c_out
            if end_r > c_out:
                end_r = c_out

            var w_deint0 = alloc[Float32](c_in)
            var w_deint1 = alloc[Float32](c_in)

            var r = start_r
            while r + 1 < end_r:
                var rp0 = r * c_in_packed
                var rs0 = r * c_in_blocks
                var rp1 = (r + 1) * c_in_packed
                var rs1 = (r + 1) * c_in_blocks
                for blk in range(c_in_blocks):
                    var sc0 = scales[rs0 + blk]
                    var b0 = W_packed.load[16](rp0 + blk * 16)
                    w_deint0.store[16](blk * 32, ((b0 & 15).cast[DType.int8]() - 8).cast[DType.float32]() * sc0)
                    w_deint0.store[16](blk * 32 + 16, ((b0 >> 4).cast[DType.int8]() - 8).cast[DType.float32]() * sc0)

                    var sc1 = scales[rs1 + blk]
                    var b1 = W_packed.load[16](rp1 + blk * 16)
                    w_deint1.store[16](blk * 32, ((b1 & 15).cast[DType.int8]() - 8).cast[DType.float32]() * sc1)
                    w_deint1.store[16](blk * 32 + 16, ((b1 >> 4).cast[DType.int8]() - 8).cast[DType.float32]() * sc1)

                var b = 0
                while b + 2 < B:
                    var a0_0 = SIMD[DType.float32, 16](0.0)
                    var a0_1 = SIMD[DType.float32, 16](0.0)
                    var a0_2 = SIMD[DType.float32, 16](0.0)
                    var a1_0 = SIMD[DType.float32, 16](0.0)
                    var a1_1 = SIMD[DType.float32, 16](0.0)
                    var a1_2 = SIMD[DType.float32, 16](0.0)
                    var xb0 = (b + 0) * c_in
                    var xb1 = (b + 1) * c_in
                    var xb2 = (b + 2) * c_in
                    for k in range(0, c_in, 16):
                        var w0 = w_deint0.load[16](k)
                        var w1 = w_deint1.load[16](k)
                        var x0 = X_deint.load[16](xb0 + k)
                        var x1 = X_deint.load[16](xb1 + k)
                        var x2 = X_deint.load[16](xb2 + k)
                        a0_0 = w0.fma(x0, a0_0)
                        a0_1 = w0.fma(x1, a0_1)
                        a0_2 = w0.fma(x2, a0_2)
                        a1_0 = w1.fma(x0, a1_0)
                        a1_1 = w1.fma(x1, a1_1)
                        a1_2 = w1.fma(x2, a1_2)
                    out_ptr[(b + 0) * o_stride + r] = a0_0.reduce_add().cast[out_type]()
                    out_ptr[(b + 1) * o_stride + r] = a0_1.reduce_add().cast[out_type]()
                    out_ptr[(b + 2) * o_stride + r] = a0_2.reduce_add().cast[out_type]()
                    out_ptr[(b + 0) * o_stride + r + 1] = a1_0.reduce_add().cast[out_type]()
                    out_ptr[(b + 1) * o_stride + r + 1] = a1_1.reduce_add().cast[out_type]()
                    out_ptr[(b + 2) * o_stride + r + 1] = a1_2.reduce_add().cast[out_type]()
                    b += 3
                while b < B:
                    var a0 = SIMD[DType.float32, 16](0.0)
                    var a1 = SIMD[DType.float32, 16](0.0)
                    var xb = b * c_in
                    for k in range(0, c_in, 16):
                        var x0 = X_deint.load[16](xb + k)
                        a0 = w_deint0.load[16](k).fma(x0, a0)
                        a1 = w_deint1.load[16](k).fma(x0, a1)
                    out_ptr[b * o_stride + r] = a0.reduce_add().cast[out_type]()
                    out_ptr[b * o_stride + r + 1] = a1.reduce_add().cast[out_type]()
                    b += 1
                r += 2

            while r < end_r:
                var rp0 = r * c_in_packed
                var rs0 = r * c_in_blocks
                for blk in range(c_in_blocks):
                    var sc0 = scales[rs0 + blk]
                    var b0 = W_packed.load[16](rp0 + blk * 16)
                    w_deint0.store[16](blk * 32, ((b0 & 15).cast[DType.int8]() - 8).cast[DType.float32]() * sc0)
                    w_deint0.store[16](blk * 32 + 16, ((b0 >> 4).cast[DType.int8]() - 8).cast[DType.float32]() * sc0)
                for b in range(B):
                    var a0 = SIMD[DType.float32, 16](0.0)
                    var xb = b * c_in
                    for k in range(0, c_in, 16):
                        a0 = w_deint0.load[16](k).fma(X_deint.load[16](xb + k), a0)
                    out_ptr[b * o_stride + r] = a0.reduce_add().cast[out_type]()
                r += 1

            w_deint0.free()
            w_deint1.free()

        if tasks == 1:
            worker_large(0)
        else:
            parallelize[worker_large](tasks)


@always_inline
def unpack_16bytes(
    bytes: SIMD[DType.uint8, 16],
    mut w0: SIMD[DType.float32, 8],
    mut w1: SIMD[DType.float32, 8],
    mut w2: SIMD[DType.float32, 8],
    mut w3: SIMD[DType.float32, 8],
):
    var repeated_01 = bytes.shuffle[0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7]()
    var shifted_01 = repeated_01 >> SIMD[DType.uint8, 16](0, 4, 0, 4, 0, 4, 0, 4, 0, 4, 0, 4, 0, 4, 0, 4)
    var masked_01 = (shifted_01 & 15).cast[DType.int8]() - 8

    var repeated_23 = bytes.shuffle[8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13, 14, 14, 15, 15]()
    var shifted_23 = repeated_23 >> SIMD[DType.uint8, 16](0, 4, 0, 4, 0, 4, 0, 4, 0, 4, 0, 4, 0, 4, 0, 4)
    var masked_23 = (shifted_23 & 15).cast[DType.int8]() - 8

    w0 = masked_01.slice[8]().cast[DType.float32]()
    w1 = masked_01.shift_left[8]().slice[8]().cast[DType.float32]()
    w2 = masked_23.slice[8]().cast[DType.float32]()
    w3 = masked_23.shift_left[8]().slice[8]().cast[DType.float32]()


@always_inline
def unpack_row_prescaled(
    row_packed: UnsafePointer[UInt8, _],
    row_scales: UnsafePointer[Float32, _],
    w_buf: UnsafePointer[Float32, MutAnyOrigin],
    c_in_blocks: Int,
):
    for blk in range(c_in_blocks):
        var scale_vec = SIMD[DType.float32, 8](row_scales[blk])
        var bytes = row_packed.load[16](blk * 16)
        var w0 = SIMD[DType.float32, 8]()
        var w1 = SIMD[DType.float32, 8]()
        var w2 = SIMD[DType.float32, 8]()
        var w3 = SIMD[DType.float32, 8]()
        unpack_16bytes(bytes, w0, w1, w2, w3)
        var base = blk * 32
        w_buf.store[8](base, w0 * scale_vec)
        w_buf.store[8](base + 8, w1 * scale_vec)
        w_buf.store[8](base + 16, w2 * scale_vec)
        w_buf.store[8](base + 24, w3 * scale_vec)


# L1-Unpacked 2-Row x 4-Batch FMA Q4 GEMM kernel.
# Unpacks each Q4 row ONCE into a thread-local L1 buffer (instead of B/8 = 32 times
# in q4_gemv_tiled), then uses 2x4 register blocking (8 YMM accumulators + 2 weight
# regs + 4 input regs = 14 YMM registers) with chained FMA across all B tokens.
@always_inline
def q4_gemm_l1_2x4[
    in_type: DType = DType.float32,
    out_type: DType = DType.float32,
](
    W_packed: UnsafePointer[UInt8, _],
    scales: UnsafePointer[Float32, _],
    X: UnsafePointer[Scalar[in_type], _],
    out_ptr: UnsafePointer[Scalar[out_type], MutAnyOrigin],
    B: Int,
    c_in: Int,
    c_out: Int,
    num_tasks: Int,
    in_stride: Int = -1,
    out_stride: Int = -1,
):
    var tasks = num_tasks
    if tasks > c_out:
        tasks = c_out
    if tasks < 1:
        tasks = 1
    var x_stride = in_stride if in_stride >= 0 else c_in
    var o_stride = out_stride if out_stride >= 0 else c_out

    var chunk_size = ((c_out // tasks + 1) // 2) * 2
    if chunk_size < 2:
        chunk_size = 2
    var c_in_packed = c_in // 2
    var c_in_blocks = c_in // 32

    def worker(t: Int) capturing -> None:
        var start_r = t * chunk_size
        if start_r >= c_out:
            return
        var end_r = (t + 1) * chunk_size if t < tasks - 1 else c_out
        if end_r > c_out:
            end_r = c_out

        var w_buf0 = alloc[Float32](c_in)
        var w_buf1 = alloc[Float32](c_in)
        var w_buf2 = alloc[Float32](c_in)
        var w_buf3 = alloc[Float32](c_in)

        var r = start_r
        while r + 3 < end_r:
            unpack_row_prescaled(
                W_packed + (r + 0) * c_in_packed,
                scales + (r + 0) * c_in_blocks,
                w_buf0,
                c_in_blocks,
            )
            unpack_row_prescaled(
                W_packed + (r + 1) * c_in_packed,
                scales + (r + 1) * c_in_blocks,
                w_buf1,
                c_in_blocks,
            )
            unpack_row_prescaled(
                W_packed + (r + 2) * c_in_packed,
                scales + (r + 2) * c_in_blocks,
                w_buf2,
                c_in_blocks,
            )
            unpack_row_prescaled(
                W_packed + (r + 3) * c_in_packed,
                scales + (r + 3) * c_in_blocks,
                w_buf3,
                c_in_blocks,
            )

            var b = 0
            while b + 2 < B:
                var acc0_0 = SIMD[DType.float32, 8](0.0)
                var acc0_1 = SIMD[DType.float32, 8](0.0)
                var acc0_2 = SIMD[DType.float32, 8](0.0)
                var acc1_0 = SIMD[DType.float32, 8](0.0)
                var acc1_1 = SIMD[DType.float32, 8](0.0)
                var acc1_2 = SIMD[DType.float32, 8](0.0)
                var acc2_0 = SIMD[DType.float32, 8](0.0)
                var acc2_1 = SIMD[DType.float32, 8](0.0)
                var acc2_2 = SIMD[DType.float32, 8](0.0)
                var acc3_0 = SIMD[DType.float32, 8](0.0)
                var acc3_1 = SIMD[DType.float32, 8](0.0)
                var acc3_2 = SIMD[DType.float32, 8](0.0)

                var xb0 = (b + 0) * x_stride
                var xb1 = (b + 1) * x_stride
                var xb2 = (b + 2) * x_stride

                for k in range(0, c_in, 8):
                    var w0 = w_buf0.load[8](k)
                    var w1 = w_buf1.load[8](k)
                    var w2 = w_buf2.load[8](k)
                    var w3 = w_buf3.load[8](k)

                    var x0 = X.load[8](xb0 + k).cast[DType.float32]()
                    acc0_0 = w0.fma(x0, acc0_0)
                    acc1_0 = w1.fma(x0, acc1_0)
                    acc2_0 = w2.fma(x0, acc2_0)
                    acc3_0 = w3.fma(x0, acc3_0)

                    var x1 = X.load[8](xb1 + k).cast[DType.float32]()
                    acc0_1 = w0.fma(x1, acc0_1)
                    acc1_1 = w1.fma(x1, acc1_1)
                    acc2_1 = w2.fma(x1, acc2_1)
                    acc3_1 = w3.fma(x1, acc3_1)

                    var x2 = X.load[8](xb2 + k).cast[DType.float32]()
                    acc0_2 = w0.fma(x2, acc0_2)
                    acc1_2 = w1.fma(x2, acc1_2)
                    acc2_2 = w2.fma(x2, acc2_2)
                    acc3_2 = w3.fma(x2, acc3_2)

                out_ptr[(b + 0) * o_stride + r + 0] = acc0_0.reduce_add().cast[out_type]()
                out_ptr[(b + 1) * o_stride + r + 0] = acc0_1.reduce_add().cast[out_type]()
                out_ptr[(b + 2) * o_stride + r + 0] = acc0_2.reduce_add().cast[out_type]()
                out_ptr[(b + 0) * o_stride + r + 1] = acc1_0.reduce_add().cast[out_type]()
                out_ptr[(b + 1) * o_stride + r + 1] = acc1_1.reduce_add().cast[out_type]()
                out_ptr[(b + 2) * o_stride + r + 1] = acc1_2.reduce_add().cast[out_type]()
                out_ptr[(b + 0) * o_stride + r + 2] = acc2_0.reduce_add().cast[out_type]()
                out_ptr[(b + 1) * o_stride + r + 2] = acc2_1.reduce_add().cast[out_type]()
                out_ptr[(b + 2) * o_stride + r + 2] = acc2_2.reduce_add().cast[out_type]()
                out_ptr[(b + 0) * o_stride + r + 3] = acc3_0.reduce_add().cast[out_type]()
                out_ptr[(b + 1) * o_stride + r + 3] = acc3_1.reduce_add().cast[out_type]()
                out_ptr[(b + 2) * o_stride + r + 3] = acc3_2.reduce_add().cast[out_type]()
                b += 3

            while b < B:
                var acc0 = SIMD[DType.float32, 8](0.0)
                var acc1 = SIMD[DType.float32, 8](0.0)
                var acc2 = SIMD[DType.float32, 8](0.0)
                var acc3 = SIMD[DType.float32, 8](0.0)
                var xb = b * x_stride
                for k in range(0, c_in, 8):
                    var x0 = X.load[8](xb + k).cast[DType.float32]()
                    acc0 = w_buf0.load[8](k).fma(x0, acc0)
                    acc1 = w_buf1.load[8](k).fma(x0, acc1)
                    acc2 = w_buf2.load[8](k).fma(x0, acc2)
                    acc3 = w_buf3.load[8](k).fma(x0, acc3)
                out_ptr[b * o_stride + r + 0] = acc0.reduce_add().cast[out_type]()
                out_ptr[b * o_stride + r + 1] = acc1.reduce_add().cast[out_type]()
                out_ptr[b * o_stride + r + 2] = acc2.reduce_add().cast[out_type]()
                out_ptr[b * o_stride + r + 3] = acc3.reduce_add().cast[out_type]()
                b += 1

            r += 4

        while r < end_r:
            unpack_row_prescaled(
                W_packed + r * c_in_packed,
                scales + r * c_in_blocks,
                w_buf0,
                c_in_blocks,
            )
            for b in range(B):
                var acc0 = SIMD[DType.float32, 8](0.0)
                var xb = b * x_stride
                for k in range(0, c_in, 8):
                    acc0 = w_buf0.load[8](k).fma(
                        X.load[8](xb + k).cast[DType.float32](), acc0
                    )
                out_ptr[b * o_stride + r] = acc0.reduce_add().cast[out_type]()
            r += 1

        w_buf0.free()
        w_buf1.free()
        w_buf2.free()
        w_buf3.free()

    if tasks == 1:
        worker(0)
    else:
        parallelize[worker](tasks)


# Fused Dual-Matrix (Gate + Up + GeGLU) L1-Unpacked 1-Row-Pair x 4-Batch FMA kernel.
# Computes out_gated[b, r] = gelu(W_gate[r] @ X[b]) * (W_up[r] @ X[b]) in a single
# parallel dispatch, eliminating intermediate gate/up memory roundtrips and barrier!
@always_inline
def q4_fused_gate_up_geglu_l1[
    in_type: DType = DType.float32,
    out_type: DType = DType.float32,
](
    W_gate_packed: UnsafePointer[UInt8, _],
    W_gate_scales: UnsafePointer[Float32, _],
    W_up_packed: UnsafePointer[UInt8, _],
    W_up_scales: UnsafePointer[Float32, _],
    X: UnsafePointer[Scalar[in_type], _],
    out_gated: UnsafePointer[Scalar[out_type], MutAnyOrigin],
    B: Int,
    c_in: Int,
    c_out: Int,
    num_tasks: Int,
):
    var tasks = num_tasks
    if tasks > c_out:
        tasks = c_out
    if tasks < 1:
        tasks = 1

    var chunk_size = c_out // tasks
    var c_in_packed = c_in // 2
    var c_in_blocks = c_in // 32

    def worker(t: Int) capturing -> None:
        var start_r = t * chunk_size
        var end_r = (t + 1) * chunk_size if t < tasks - 1 else c_out

        var w_gate0 = alloc[Float32](c_in)
        var w_gate1 = alloc[Float32](c_in)
        var w_up0 = alloc[Float32](c_in)
        var w_up1 = alloc[Float32](c_in)

        var r = start_r
        while r + 1 < end_r:
            var r0 = r
            var r1 = r + 1
            unpack_row_prescaled(
                W_gate_packed + r0 * c_in_packed,
                W_gate_scales + r0 * c_in_blocks,
                w_gate0,
                c_in_blocks,
            )
            unpack_row_prescaled(
                W_gate_packed + r1 * c_in_packed,
                W_gate_scales + r1 * c_in_blocks,
                w_gate1,
                c_in_blocks,
            )
            unpack_row_prescaled(
                W_up_packed + r0 * c_in_packed,
                W_up_scales + r0 * c_in_blocks,
                w_up0,
                c_in_blocks,
            )
            unpack_row_prescaled(
                W_up_packed + r1 * c_in_packed,
                W_up_scales + r1 * c_in_blocks,
                w_up1,
                c_in_blocks,
            )

            var b = 0
            while b + 2 < B:
                var g0_0 = SIMD[DType.float32, 8](0.0)
                var g0_1 = SIMD[DType.float32, 8](0.0)
                var g0_2 = SIMD[DType.float32, 8](0.0)
                var u0_0 = SIMD[DType.float32, 8](0.0)
                var u0_1 = SIMD[DType.float32, 8](0.0)
                var u0_2 = SIMD[DType.float32, 8](0.0)
                var g1_0 = SIMD[DType.float32, 8](0.0)
                var g1_1 = SIMD[DType.float32, 8](0.0)
                var g1_2 = SIMD[DType.float32, 8](0.0)
                var u1_0 = SIMD[DType.float32, 8](0.0)
                var u1_1 = SIMD[DType.float32, 8](0.0)
                var u1_2 = SIMD[DType.float32, 8](0.0)

                var xb0 = (b + 0) * c_in
                var xb1 = (b + 1) * c_in
                var xb2 = (b + 2) * c_in

                for k in range(0, c_in, 8):
                    var wg0 = w_gate0.load[8](k)
                    var wu0 = w_up0.load[8](k)
                    var wg1 = w_gate1.load[8](k)
                    var wu1 = w_up1.load[8](k)

                    var x0 = X.load[8](xb0 + k).cast[DType.float32]()
                    g0_0 = wg0.fma(x0, g0_0)
                    u0_0 = wu0.fma(x0, u0_0)
                    g1_0 = wg1.fma(x0, g1_0)
                    u1_0 = wu1.fma(x0, u1_0)

                    var x1 = X.load[8](xb1 + k).cast[DType.float32]()
                    g0_1 = wg0.fma(x1, g0_1)
                    u0_1 = wu0.fma(x1, u0_1)
                    g1_1 = wg1.fma(x1, g1_1)
                    u1_1 = wu1.fma(x1, u1_1)

                    var x2 = X.load[8](xb2 + k).cast[DType.float32]()
                    g0_2 = wg0.fma(x2, g0_2)
                    u0_2 = wu0.fma(x2, u0_2)
                    g1_2 = wg1.fma(x2, g1_2)
                    u1_2 = wu1.fma(x2, u1_2)

                var gate_v = SIMD[DType.float32, 8](
                    g0_0.reduce_add(), g0_1.reduce_add(), g0_2.reduce_add(),
                    g1_0.reduce_add(), g1_1.reduce_add(), g1_2.reduce_add(),
                    0.0, 0.0,
                )
                var up_v = SIMD[DType.float32, 8](
                    u0_0.reduce_add(), u0_1.reduce_add(), u0_2.reduce_add(),
                    u1_0.reduce_add(), u1_1.reduce_add(), u1_2.reduce_add(),
                    0.0, 0.0,
                )
                var sig_v = 1.0 / (1.0 + exp(SIMD[DType.float32, 8](-1.702) * gate_v))
                var act_v = (gate_v * sig_v * up_v).cast[out_type]()
                out_gated[(b + 0) * c_out + r0] = act_v[0]
                out_gated[(b + 1) * c_out + r0] = act_v[1]
                out_gated[(b + 2) * c_out + r0] = act_v[2]
                out_gated[(b + 0) * c_out + r1] = act_v[3]
                out_gated[(b + 1) * c_out + r1] = act_v[4]
                out_gated[(b + 2) * c_out + r1] = act_v[5]
                b += 3

            while b < B:
                var g0 = SIMD[DType.float32, 8](0.0)
                var u0 = SIMD[DType.float32, 8](0.0)
                var g1 = SIMD[DType.float32, 8](0.0)
                var u1 = SIMD[DType.float32, 8](0.0)
                var xb = b * c_in
                for k in range(0, c_in, 8):
                    var x0 = X.load[8](xb + k).cast[DType.float32]()
                    g0 = w_gate0.load[8](k).fma(x0, g0)
                    u0 = w_up0.load[8](k).fma(x0, u0)
                    g1 = w_gate1.load[8](k).fma(x0, g1)
                    u1 = w_up1.load[8](k).fma(x0, u1)
                var gv0 = g0.reduce_add()
                var uv0 = u0.reduce_add()
                var gv1 = g1.reduce_add()
                var uv1 = u1.reduce_add()
                out_gated[b * c_out + r0] = Scalar[out_type](
                    gv0 * (1.0 / (1.0 + exp(Float32(-1.702) * gv0))) * uv0
                )
                out_gated[b * c_out + r1] = Scalar[out_type](
                    gv1 * (1.0 / (1.0 + exp(Float32(-1.702) * gv1))) * uv1
                )
                b += 1

            r += 2

        while r < end_r:
            unpack_row_prescaled(
                W_gate_packed + r * c_in_packed,
                W_gate_scales + r * c_in_blocks,
                w_gate0,
                c_in_blocks,
            )
            unpack_row_prescaled(
                W_up_packed + r * c_in_packed,
                W_up_scales + r * c_in_blocks,
                w_up0,
                c_in_blocks,
            )
            for b in range(B):
                var g0 = SIMD[DType.float32, 8](0.0)
                var u0 = SIMD[DType.float32, 8](0.0)
                var xb = b * c_in
                for k in range(0, c_in, 8):
                    var x0 = X.load[8](xb + k).cast[DType.float32]()
                    g0 = w_gate0.load[8](k).fma(x0, g0)
                    u0 = w_up0.load[8](k).fma(x0, u0)
                var gv = g0.reduce_add()
                var uv = u0.reduce_add()
                var sig = 1.0 / (1.0 + exp(Float32(-1.702) * gv))
                out_gated[b * c_out + r] = Scalar[out_type](gv * sig * uv)
            r += 1

        w_gate0.free()
        w_gate1.free()
        w_up0.free()
        w_up1.free()

    if tasks == 1:
        worker(0)
    else:
        parallelize[worker](tasks)


# Batched Self-Conditioning MLP Forward (DiffusionGemma SC path, llm-mojo-2tc.1).
# Reads SC Q4 weights (9 MB) ONCE across all B canvas tokens instead of B=256 times!
def sc_forward_batch[
    inputs_embeds_type: DType = DType.float16,
    sc_out_type: DType = DType.float16,
](
    inputs_embeds: UnsafePointer[Scalar[inputs_embeds_type], _],
    soft_embeds: UnsafePointer[Float32, _],
    sc_gate_packed: UnsafePointer[UInt8, _],
    sc_up_packed: UnsafePointer[UInt8, _],
    sc_down_packed: UnsafePointer[UInt8, _],
    sc_gate_scales: UnsafePointer[Float32, _],
    sc_up_scales: UnsafePointer[Float32, _],
    sc_down_scales: UnsafePointer[Float32, _],
    sc_pre_norm: UnsafePointer[Float32, _],
    buf_normed: UnsafePointer[Float32, MutAnyOrigin],
    buf_gated: UnsafePointer[Float32, MutAnyOrigin],
    buf_down: UnsafePointer[Float32, MutAnyOrigin],
    sc_out: UnsafePointer[Scalar[sc_out_type], MutAnyOrigin],
    B: Int,
    embed_dim: Int,
    sc_size: Int,
    num_tasks: Int,
):
    # 1. Batched pre_norm on soft_embeds: [B, embed_dim]
    for b in range(B):
        rms_norm[DType.float32, DType.float32](
            soft_embeds + b * embed_dim,
            sc_pre_norm,
            buf_normed + b * embed_dim,
            embed_dim,
        )

    # 2. Fused batched gate_proj + up_proj + GeGLU: [B, embed_dim] -> [B, sc_size]
    q4_fused_gate_up_geglu_l1[DType.float32, DType.float32](
        sc_gate_packed,
        sc_gate_scales,
        sc_up_packed,
        sc_up_scales,
        buf_normed,
        buf_gated,
        B,
        embed_dim,
        sc_size,
        num_tasks,
    )

    # 3. Batched down_proj: [B, sc_size] -> [B, embed_dim]
    q4_gemm_l1_2x4[DType.float32, DType.float32](
        sc_down_packed,
        sc_down_scales,
        buf_gated,
        buf_down,
        B,
        sc_size,
        embed_dim,
        num_tasks,
    )

    # 4. Residual + post_norm_noscale in Float32 precision, then cast to sc_out_type
    for b in range(B):
        var in_b = inputs_embeds + b * embed_dim
        var down_b = buf_down + b * embed_dim
        var out_b = sc_out + b * embed_dim
        var acc = SIMD[DType.float32, 8](0.0)
        for i in range(0, embed_dim, 8):
            var val = in_b.load[8](i).cast[DType.float32]() + down_b.load[8](i)
            down_b.store[8](i, val)
            acc = val.fma(val, acc)
        var inv_rms = 1.0 / sqrt(acc.reduce_add() / Float32(embed_dim) + 1e-6)
        var inv_rms_vec = SIMD[DType.float32, 8](inv_rms)
        for i in range(0, embed_dim, 8):
            out_b.store[8](i, (down_b.load[8](i) * inv_rms_vec).cast[sc_out_type]())


# Sparse Top-32 Soft-Embedding Accumulation (llm-mojo-2tc.1).
# Accumulates soft_embeds = (top32_probs @ embed_weight[top32_idx]) * prompt_scale
# reading at most 32 Q4 rows instead of 262,144 / 32,768 rows.
@always_inline
def probs_to_soft_embed_top32[
    TOP_K: Int = 32
](
    top_idx: UnsafePointer[Int, _],
    top_probs: UnsafePointer[Float32, _],
    embed_packed: UnsafePointer[UInt8, _],
    embed_scales: UnsafePointer[Float32, _],
    sc_out: UnsafePointer[Float32, MutAnyOrigin],
    embed_dim: Int,
    prompt_scale: Float32 = 1.0,
):
    for d in range(0, embed_dim, 8):
        sc_out.store[8](d, SIMD[DType.float32, 8](0.0))

    var num_blocks = embed_dim // 32
    var row_packed_stride = embed_dim // 2
    var row_scales_stride = embed_dim // 32

    for k in range(TOP_K):
        var p = top_probs[k]
        if p < 1e-8:
            continue
        var r = top_idx[k]
        if r < 0:
            continue
        var p_scaled = p * prompt_scale
        var row_packed_offset = r * row_packed_stride
        var row_scales_offset = r * row_scales_stride
        for b in range(num_blocks):
            var scale = embed_scales[row_scales_offset + b]
            var block_packed_offset = row_packed_offset + b * 16
            var block_out_offset = b * 32

            var bytes = embed_packed.load[16](block_packed_offset)
            var w0 = SIMD[DType.float32, 8]()
            var w1 = SIMD[DType.float32, 8]()
            var w2 = SIMD[DType.float32, 8]()
            var w3 = SIMD[DType.float32, 8]()
            unpack_16bytes(bytes, w0, w1, w2, w3)

            var p_scale_vec = SIMD[DType.float32, 8](p_scaled * scale)
            sc_out.store[8](
                block_out_offset,
                w0.fma(p_scale_vec, sc_out.load[8](block_out_offset)),
            )
            sc_out.store[8](
                block_out_offset + 8,
                w1.fma(p_scale_vec, sc_out.load[8](block_out_offset + 8)),
            )
            sc_out.store[8](
                block_out_offset + 16,
                w2.fma(p_scale_vec, sc_out.load[8](block_out_offset + 16)),
            )
            sc_out.store[8](
                block_out_offset + 24,
                w3.fma(p_scale_vec, sc_out.load[8](block_out_offset + 24)),
            )


# Fast SIMD Top-32 reservoir + monotonic softcap + cumulant-tail entropy + sparse soft-embed.
# Replaces full-vocab scalar tanh/exp/softmax + full-vocab probs_to_soft_embed.
def sample_and_soft_embed_top32_batch[
    TOP_K: Int = 32
](
    canvas_logits: UnsafePointer[Float32, _],
    rand_u: UnsafePointer[Float32, _],
    embed_packed: UnsafePointer[UInt8, _],
    embed_scales: UnsafePointer[Float32, _],
    token_entropy: UnsafePointer[Float32, MutAnyOrigin],
    argmax_tokens: UnsafePointer[Int, MutAnyOrigin],
    new_tokens: UnsafePointer[Int, MutAnyOrigin],
    sc_embeds: UnsafePointer[Float32, MutAnyOrigin],
    B: Int,
    vocab_size: Int,
    embed_dim: Int,
    temp: Float32,
    prompt_scale: Float32,
    num_tasks: Int = 8,
):
    def token_worker(t: Int) capturing -> None:
        var logits_t = canvas_logits + t * vocab_size
        var top_vals = InlineArray[Float32, TOP_K](uninitialized=True)
        var top_idxs = InlineArray[Int, TOP_K](uninitialized=True)
        var top_probs = InlineArray[Float32, TOP_K](uninitialized=True)

        # Initialize reservoir with first TOP_K elements and find initial min
        var min_val: Float32 = logits_t[0]
        var min_pos: Int = 0
        for k in range(TOP_K):
            var v = logits_t[k]
            top_vals[k] = v
            top_idxs[k] = k
            if v < min_val:
                min_val = v
                min_pos = k

        # SIMD8 scan over remaining vocabulary:
        # 1. Fast vector max rejection: only inspect lanes when vec.reduce_max() > min_val
        # 2. Accumulate sum and sum_sq for 2nd-order cumulant tail partition estimate
        var sum_vec = SIMD[DType.float32, 8](0.0)
        var sum_sq_vec = SIMD[DType.float32, 8](0.0)
        for k in range(0, TOP_K, 8):
            var v0 = logits_t.load[8](k)
            sum_vec += v0
            sum_sq_vec = v0.fma(v0, sum_sq_vec)

        var v_idx = TOP_K
        var v_vec_end = (vocab_size // 8) * 8
        while v_idx < v_vec_end:
            var vec = logits_t.load[8](v_idx)
            sum_vec += vec
            sum_sq_vec = vec.fma(vec, sum_sq_vec)
            if vec.reduce_max() > min_val:
                for lane in range(8):
                    var val = vec[lane]
                    if val > min_val:
                        top_vals[min_pos] = val
                        top_idxs[min_pos] = v_idx + lane
                        # Recompute min_val and min_pos across TOP_K
                        min_val = top_vals[0]
                        min_pos = 0
                        for j in range(1, TOP_K):
                            if top_vals[j] < min_val:
                                min_val = top_vals[j]
                                min_pos = j
            v_idx += 8

        var tail_sum: Float32 = sum_vec.reduce_add()
        var tail_sum_sq: Float32 = sum_sq_vec.reduce_add()
        while v_idx < vocab_size:
            var val = logits_t[v_idx]
            tail_sum += val
            tail_sum_sq += val * val
            if val > min_val:
                top_vals[min_pos] = val
                top_idxs[min_pos] = v_idx
                min_val = top_vals[0]
                min_pos = 0
                for j in range(1, TOP_K):
                    if top_vals[j] < min_val:
                        min_val = top_vals[j]
                        min_pos = j
            v_idx += 1

        # Sort TOP_K descending so top_idxs[0] is argmax and largest probs come first
        for i in range(TOP_K):
            var best = i
            for j in range(i + 1, TOP_K):
                if top_vals[j] > top_vals[best] or (
                    top_vals[j] == top_vals[best] and top_idxs[j] < top_idxs[best]
                ):
                    best = j
            if best != i:
                var tv = top_vals[i]
                top_vals[i] = top_vals[best]
                top_vals[best] = tv
                var ti = top_idxs[i]
                top_idxs[i] = top_idxs[best]
                top_idxs[best] = ti

        argmax_tokens[t] = top_idxs[0]

        # Subtract top-K raw logits from total sum / sum_sq to get tail moments
        var top_raw_sum: Float32 = 0.0
        var top_raw_sum_sq: Float32 = 0.0
        for k in range(TOP_K):
            var z = top_vals[k]
            top_raw_sum += z
            top_raw_sum_sq += z * z
            # Apply monotonic softcap ONLY to TOP_K logits
            top_vals[k] = (Float32(30.0) * tanh(z / Float32(30.0))) / temp

        var max_sc_logit = top_vals[0]
        var min_sc_logit = top_vals[TOP_K - 1]

        var sum_exp_top: Float32 = 0.0
        var weighted_diff_top: Float32 = 0.0
        for k in range(TOP_K):
            var diff = max_sc_logit - top_vals[k]
            var e_val = exp(-diff)
            top_probs[k] = e_val
            sum_exp_top += e_val
            weighted_diff_top += e_val * diff

        # Cumulant tail partition estimate for exact entropy across both peaked & flat regimes
        var n_tail = Float32(vocab_size - TOP_K)
        var z_tail: Float32 = 0.0
        var diff_tail: Float32 = 0.0
        if n_tail > 0.0:
            var mu_raw = (tail_sum - top_raw_sum) / n_tail
            var var_raw = (tail_sum_sq - top_raw_sum_sq) / n_tail - mu_raw * mu_raw
            if var_raw < 0.0:
                var_raw = 0.0
            var th_mu = tanh(mu_raw / Float32(30.0))
            var mu_y = (Float32(30.0) * th_mu) / temp
            var slope_y = (Float32(1.0) - th_mu * th_mu) / temp
            var var_y = slope_y * slope_y * var_raw
            var eff_y = mu_y + Float32(0.5) * var_y
            if eff_y > min_sc_logit:
                eff_y = min_sc_logit
            diff_tail = max_sc_logit - eff_y
            if diff_tail < 80.0:
                z_tail = n_tail * exp(-diff_tail)

        var z_total = sum_exp_top + z_tail
        var ent = log(z_total + 1e-30) + (
            weighted_diff_top + z_tail * diff_tail
        ) / (z_total + 1e-30)
        if ent < 0.0:
            ent = 0.0
        token_entropy[t] = ent

        # Renormalize top-K probabilities for sampling and sparse soft-embedding
        var inv_sum_top = Float32(1.0) / (sum_exp_top + 1e-30)
        for k in range(TOP_K):
            top_probs[k] = top_probs[k] * inv_sum_top

        # Sample from top-K distribution
        var u = rand_u[t]
        var cumsum: Float32 = 0.0
        var sampled = top_idxs[0]
        for k in range(TOP_K):
            cumsum += top_probs[k]
            if cumsum >= u:
                sampled = top_idxs[k]
                break
        new_tokens[t] = sampled

        # Sparse Top-32 soft embedding accumulation
        var top_idxs_ptr = UnsafePointer[Int, MutAnyOrigin](
            unsafe_from_address=Int( UnsafePointer(to=top_idxs) )
        )
        var top_probs_ptr = UnsafePointer[Float32, MutAnyOrigin](
            unsafe_from_address=Int( UnsafePointer(to=top_probs) )
        )
        probs_to_soft_embed_top32[TOP_K](
            top_idxs_ptr,
            top_probs_ptr,
            embed_packed,
            embed_scales,
            sc_embeds + t * embed_dim,
            embed_dim,
            prompt_scale,
        )

    if num_tasks <= 1 or B <= 1:
        for t in range(B):
            token_worker(t)
    else:
        parallelize[token_worker](B)
