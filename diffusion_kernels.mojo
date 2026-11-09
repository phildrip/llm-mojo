from std.memory import alloc
from std.collections import InlineArray
from std.math import sqrt, exp, tanh, log, cos, sin
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
    Uses 256-bit AVX2 SIMD[DType.float32, 8] with 4-row x 3-token register blocking
    (12 YMM accumulators + 4 YMM weights/inputs = 16 YMM registers, zero stack spills on Zen 3!)."""
    var c_in_packed = c_in // 2
    var c_in_blocks = c_in // 32

    var w_arr2 = InlineArray[Float32, 2816](uninitialized=True)
    var w_arr3 = InlineArray[Float32, 2816](uninitialized=True)
    var w_deint0 = w_deint_scratch
    var w_deint1 = w_deint_scratch + c_in
    var w_deint2 = UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=Int(UnsafePointer(to=w_arr2)))
    var w_deint3 = UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=Int(UnsafePointer(to=w_arr3)))

    var r = 0
    while r + 3 < c_out:
        var rp0 = (r + 0) * c_in_packed
        var rs0 = (r + 0) * c_in_blocks
        var rp1 = (r + 1) * c_in_packed
        var rs1 = (r + 1) * c_in_blocks
        var rp2 = (r + 2) * c_in_packed
        var rs2 = (r + 2) * c_in_blocks
        var rp3 = (r + 3) * c_in_packed
        var rs3 = (r + 3) * c_in_blocks

        for blk in range(c_in_blocks):
            var off = blk * 32
            var sc0 = scales[rs0 + blk]
            var b0 = W_packed.load[16](rp0 + blk * 16)
            w_deint0.store[16](off, ((b0 & 15).cast[DType.int8]() - 8).cast[DType.float32]() * sc0)
            w_deint0.store[16](off + 16, ((b0 >> 4).cast[DType.int8]() - 8).cast[DType.float32]() * sc0)

            var sc1 = scales[rs1 + blk]
            var b1 = W_packed.load[16](rp1 + blk * 16)
            w_deint1.store[16](off, ((b1 & 15).cast[DType.int8]() - 8).cast[DType.float32]() * sc1)
            w_deint1.store[16](off + 16, ((b1 >> 4).cast[DType.int8]() - 8).cast[DType.float32]() * sc1)

            var sc2 = scales[rs2 + blk]
            var b2 = W_packed.load[16](rp2 + blk * 16)
            w_deint2.store[16](off, ((b2 & 15).cast[DType.int8]() - 8).cast[DType.float32]() * sc2)
            w_deint2.store[16](off + 16, ((b2 >> 4).cast[DType.int8]() - 8).cast[DType.float32]() * sc2)

            var sc3 = scales[rs3 + blk]
            var b3 = W_packed.load[16](rp3 + blk * 16)
            w_deint3.store[16](off, ((b3 & 15).cast[DType.int8]() - 8).cast[DType.float32]() * sc3)
            w_deint3.store[16](off + 16, ((b3 >> 4).cast[DType.int8]() - 8).cast[DType.float32]() * sc3)

        var b = 0
        while b + 2 < B:
            var a0_0 = SIMD[DType.float32, 8](0.0)
            var a0_1 = SIMD[DType.float32, 8](0.0)
            var a0_2 = SIMD[DType.float32, 8](0.0)
            var a1_0 = SIMD[DType.float32, 8](0.0)
            var a1_1 = SIMD[DType.float32, 8](0.0)
            var a1_2 = SIMD[DType.float32, 8](0.0)
            var a2_0 = SIMD[DType.float32, 8](0.0)
            var a2_1 = SIMD[DType.float32, 8](0.0)
            var a2_2 = SIMD[DType.float32, 8](0.0)
            var a3_0 = SIMD[DType.float32, 8](0.0)
            var a3_1 = SIMD[DType.float32, 8](0.0)
            var a3_2 = SIMD[DType.float32, 8](0.0)
            var xb0 = (b + 0) * c_in
            var xb1 = (b + 1) * c_in
            var xb2 = (b + 2) * c_in
            for k in range(0, c_in, 8):
                var w0 = w_deint0.load[8](k)
                var w1 = w_deint1.load[8](k)
                var w2 = w_deint2.load[8](k)
                var w3 = w_deint3.load[8](k)
                var x0 = X_deint.load[8](xb0 + k)
                a0_0 = w0.fma(x0, a0_0)
                a1_0 = w1.fma(x0, a1_0)
                a2_0 = w2.fma(x0, a2_0)
                a3_0 = w3.fma(x0, a3_0)
                var x1 = X_deint.load[8](xb1 + k)
                a0_1 = w0.fma(x1, a0_1)
                a1_1 = w1.fma(x1, a1_1)
                a2_1 = w2.fma(x1, a2_1)
                a3_1 = w3.fma(x1, a3_1)
                var x2 = X_deint.load[8](xb2 + k)
                a0_2 = w0.fma(x2, a0_2)
                a1_2 = w1.fma(x2, a1_2)
                a2_2 = w2.fma(x2, a2_2)
                a3_2 = w3.fma(x2, a3_2)
            out_ptr[(b + 0) * c_out + r + 0] = a0_0.reduce_add()
            out_ptr[(b + 1) * c_out + r + 0] = a0_1.reduce_add()
            out_ptr[(b + 2) * c_out + r + 0] = a0_2.reduce_add()
            out_ptr[(b + 0) * c_out + r + 1] = a1_0.reduce_add()
            out_ptr[(b + 1) * c_out + r + 1] = a1_1.reduce_add()
            out_ptr[(b + 2) * c_out + r + 1] = a1_2.reduce_add()
            out_ptr[(b + 0) * c_out + r + 2] = a2_0.reduce_add()
            out_ptr[(b + 1) * c_out + r + 2] = a2_1.reduce_add()
            out_ptr[(b + 2) * c_out + r + 2] = a2_2.reduce_add()
            out_ptr[(b + 0) * c_out + r + 3] = a3_0.reduce_add()
            out_ptr[(b + 1) * c_out + r + 3] = a3_1.reduce_add()
            out_ptr[(b + 2) * c_out + r + 3] = a3_2.reduce_add()
            b += 3
        while b < B:
            var a0 = SIMD[DType.float32, 8](0.0)
            var a1 = SIMD[DType.float32, 8](0.0)
            var a2 = SIMD[DType.float32, 8](0.0)
            var a3 = SIMD[DType.float32, 8](0.0)
            var xb = b * c_in
            for k in range(0, c_in, 8):
                var x0 = X_deint.load[8](xb + k)
                a0 = w_deint0.load[8](k).fma(x0, a0)
                a1 = w_deint1.load[8](k).fma(x0, a1)
                a2 = w_deint2.load[8](k).fma(x0, a2)
                a3 = w_deint3.load[8](k).fma(x0, a3)
            out_ptr[b * c_out + r + 0] = a0.reduce_add()
            out_ptr[b * c_out + r + 1] = a1.reduce_add()
            out_ptr[b * c_out + r + 2] = a2.reduce_add()
            out_ptr[b * c_out + r + 3] = a3.reduce_add()
            b += 1
        r += 4


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
    var acc_slots_ptr = UnsafePointer[Int, MutAnyOrigin](unsafe_from_address=Int(UnsafePointer(to=assignments_slot)))

    def accum_token_worker(t: Int) capturing -> None:
        var dst = canvas_moe_out + t * embed_dim
        var top_w = canvas_top_weights + t * top_k_experts
        var slot_base = t * top_k_experts
        for j in range(0, embed_dim, 16):
            var sum_vec = SIMD[DType.float32, 16](0.0)
            for s in range(top_k_experts):
                var wp = acc_slots_ptr[slot_base + s]
                if wp >= 0:
                    var w_vec = SIMD[DType.float32, 16](top_w[s])
                    var src = all_down_out + wp * embed_dim
                    sum_vec = src.load[16](j).fma(w_vec, sum_vec)
            dst.store[16](j, sum_vec.cast[DType.float16]())

    if B <= 1:
        if B == 1:
            accum_token_worker(0)
    else:
        parallelize[accum_token_worker](B)


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

            var w_arr0 = InlineArray[Float32, 8192](uninitialized=True)
            var w_arr1 = InlineArray[Float32, 8192](uninitialized=True)
            var w_deint0 = UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=Int(UnsafePointer(to=w_arr0)))
            var w_deint1 = UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=Int(UnsafePointer(to=w_arr1)))

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

            _ = w_arr0[0]
            _ = w_arr1[0]

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


@always_inline
def unpack_row_deinterleaved_prescaled(
    row_packed: UnsafePointer[UInt8, _],
    row_scales: UnsafePointer[Float32, _],
    w_deint: UnsafePointer[Float32, MutAnyOrigin],
    c_in_blocks: Int,
):
    """Shuffle-free Dual-Space Q4 row unpacker: unpacks low nibbles (even indices) into
    `[off .. off+15]` and high nibbles (odd indices) into `[off+16 .. off+31]` with zero
    vector shuffles or shifts across lanes."""
    for blk in range(c_in_blocks):
        var off = blk * 32
        var sc = row_scales[blk]
        var b = row_packed.load[16](blk * 16)
        w_deint.store[16](off, ((b & 15).cast[DType.int8]() - 8).cast[DType.float32]() * sc)
        w_deint.store[16](off + 16, ((b >> 4).cast[DType.int8]() - 8).cast[DType.float32]() * sc)


# L1-Unpacked 4-Row x 3-Batch FMA Q4 GEMM kernel with Dual-Space Shuffle-Free X_deint support (llm-mojo-yj4.2).
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
    pre_deint_X: UnsafePointer[Float32, _] = UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=Int(0)),
):
    if B <= 0 or c_in <= 0 or c_out <= 0:
        return
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

    var x_stack = InlineArray[Float32, 65536](uninitialized=True)
    var use_deint = Int(pre_deint_X) != 0 and c_in <= 8192 and (c_in % 32 == 0)
    var x_deint_ptr = UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=Int(pre_deint_X))
    if not use_deint and c_in <= 8192 and (c_in % 32 == 0) and B * c_in <= 65536:
        x_deint_ptr = UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=Int(UnsafePointer(to=x_stack)))
        deinterleave_batch_to_f32[in_type](X, x_deint_ptr, B, c_in, x_stride)
        use_deint = True

    def worker(t: Int) capturing -> None:
        var start_r = t * chunk_size
        if start_r >= c_out:
            return
        var end_r = (t + 1) * chunk_size if t < tasks - 1 else c_out
        if end_r > c_out:
            end_r = c_out

        var w_arr0 = InlineArray[Float32, 8192](uninitialized=True)
        var w_arr1 = InlineArray[Float32, 8192](uninitialized=True)
        var w_arr2 = InlineArray[Float32, 8192](uninitialized=True)
        var w_arr3 = InlineArray[Float32, 8192](uninitialized=True)
        var w_buf0 = UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=Int(UnsafePointer(to=w_arr0)))
        var w_buf1 = UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=Int(UnsafePointer(to=w_arr1)))
        var w_buf2 = UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=Int(UnsafePointer(to=w_arr2)))
        var w_buf3 = UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=Int(UnsafePointer(to=w_arr3)))

        var r = start_r
        if use_deint:
            while r + 3 < end_r:
                unpack_row_deinterleaved_prescaled(
                    W_packed + (r + 0) * c_in_packed, scales + (r + 0) * c_in_blocks, w_buf0, c_in_blocks
                )
                unpack_row_deinterleaved_prescaled(
                    W_packed + (r + 1) * c_in_packed, scales + (r + 1) * c_in_blocks, w_buf1, c_in_blocks
                )
                unpack_row_deinterleaved_prescaled(
                    W_packed + (r + 2) * c_in_packed, scales + (r + 2) * c_in_blocks, w_buf2, c_in_blocks
                )
                unpack_row_deinterleaved_prescaled(
                    W_packed + (r + 3) * c_in_packed, scales + (r + 3) * c_in_blocks, w_buf3, c_in_blocks
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

                    var xb0 = (b + 0) * c_in
                    var xb1 = (b + 1) * c_in
                    var xb2 = (b + 2) * c_in

                    for k in range(0, c_in, 8):
                        var w0 = w_buf0.load[8](k)
                        var w1 = w_buf1.load[8](k)
                        var w2 = w_buf2.load[8](k)
                        var w3 = w_buf3.load[8](k)

                        var x0 = x_deint_ptr.load[8](xb0 + k)
                        acc0_0 = w0.fma(x0, acc0_0)
                        acc1_0 = w1.fma(x0, acc1_0)
                        acc2_0 = w2.fma(x0, acc2_0)
                        acc3_0 = w3.fma(x0, acc3_0)

                        var x1 = x_deint_ptr.load[8](xb1 + k)
                        acc0_1 = w0.fma(x1, acc0_1)
                        acc1_1 = w1.fma(x1, acc1_1)
                        acc2_1 = w2.fma(x1, acc2_1)
                        acc3_1 = w3.fma(x1, acc3_1)

                        var x2 = x_deint_ptr.load[8](xb2 + k)
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
                    var xb = b * c_in
                    for k in range(0, c_in, 8):
                        var x0 = x_deint_ptr.load[8](xb + k)
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
                unpack_row_deinterleaved_prescaled(
                    W_packed + r * c_in_packed, scales + r * c_in_blocks, w_buf0, c_in_blocks
                )
                for b in range(B):
                    var acc0 = SIMD[DType.float32, 8](0.0)
                    var xb = b * c_in
                    for k in range(0, c_in, 8):
                        acc0 = w_buf0.load[8](k).fma(x_deint_ptr.load[8](xb + k), acc0)
                    out_ptr[b * o_stride + r] = acc0.reduce_add().cast[out_type]()
                r += 1
            _ = w_arr0[0]
            _ = w_arr1[0]
            _ = w_arr2[0]
            _ = w_arr3[0]
            return
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

    if tasks == 1:
        worker(0)
    else:
        parallelize[worker](tasks)
    _ = x_stack[0]


@always_inline
def q4_lm_head_two_stage_sieve[
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
    pre_deint_X: UnsafePointer[Float32, _] = UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=Int(0)),
    forced_cand_tokens: UnsafePointer[Int, _] = UnsafePointer[Int, MutAnyOrigin](unsafe_from_address=Int(0)),
    num_forced_cands: Int = 0,
    coarse_blocks: Int = 8,
    span_keep_frac: Float32 = 0.45,
):
    """Two-Stage Coarse-to-Fine Vocab Sieve for LM Head Projection (llm-mojo-yj4.4):
    - Stage 1: Branchless 4x3 AVX2 linear scan over the first `coarse_blocks` Q4 blocks (`coarse_dim = coarse_blocks * 32`).
    - Compaction: Selects rows exceeding `max_c[b] - span_keep_frac * (max_c[b] - min_c[b])` for any active token `b`,
      unioned with `forced_cand_tokens` (e.g., previous-step Top-32 / argmax / top2 canvas tokens). Non-candidates
      are masked to `-1e30` so incomplete coarse sums can never pollute downstream Top-32 reservoir sampling.
    - Stage 2: Branchless 4x3 AVX2 FMA over the remaining `fine_blocks` Q4 blocks ONLY for the compacted `cand_rows`."""
    if B <= 0 or c_in <= 0 or c_out <= 0:
        return
    var c_in_blocks = c_in // 32
    if (
        coarse_blocks <= 0
        or coarse_blocks >= c_in_blocks
        or c_out <= 64
        or B > 256
        or span_keep_frac >= Float32(1.0)
        or c_in > 8192
        or (c_in % 32 != 0)
    ):
        q4_gemm_l1_2x4[in_type, out_type](
            W_packed, scales, X, out_ptr, B, c_in, c_out, num_tasks, in_stride, out_stride, pre_deint_X
        )
        return

    var tasks = num_tasks
    if tasks > c_out:
        tasks = c_out
    if tasks < 1:
        tasks = 1
    var x_stride = in_stride if in_stride >= 0 else c_in
    var o_stride = out_stride if out_stride >= 0 else c_out

    var chunk_size = ((c_out // tasks + 3) // 4) * 4
    if chunk_size < 4:
        chunk_size = 4
    var c_in_packed = c_in // 2
    var coarse_dim = coarse_blocks * 32
    var coarse_packed = coarse_dim // 2
    var fine_blocks = c_in_blocks - coarse_blocks
    var fine_dim = c_in - coarse_dim

    var x_stack = InlineArray[Float32, 65536](uninitialized=True)
    var use_deint = Int(pre_deint_X) != 0
    var x_deint_ptr = UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=Int(pre_deint_X))
    if not use_deint:
        if B * c_in <= 65536:
            x_deint_ptr = UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=Int(UnsafePointer(to=x_stack)))
            deinterleave_batch_to_f32[in_type](X, x_deint_ptr, B, c_in, x_stride)
        else:
            q4_gemm_l1_2x4[in_type, out_type](
                W_packed, scales, X, out_ptr, B, c_in, c_out, num_tasks, in_stride, out_stride, pre_deint_X
            )
            _ = x_stack[0]
            return

    var has_forced = Int(forced_cand_tokens) != 0 and num_forced_cands > 0
    var pruned_sentinel = Float32(-65500.0) if out_type == DType.float16 else Float32(-1e30)
    var tile_max = ((16384 // B) // 4) * 4
    if tile_max > 256:
        tile_max = 256
    if tile_max < 4:
        tile_max = 4

    def worker(t: Int) capturing -> None:
        var start_r = t * chunk_size
        if start_r >= c_out:
            return
        var end_r = (t + 1) * chunk_size if t < tasks - 1 else c_out
        if end_r > c_out:
            end_r = c_out
        var chunk_rows = end_r - start_r

        var w_arr0 = InlineArray[Float32, 8192](uninitialized=True)
        var w_arr1 = InlineArray[Float32, 8192](uninitialized=True)
        var w_arr2 = InlineArray[Float32, 8192](uninitialized=True)
        var w_arr3 = InlineArray[Float32, 8192](uninitialized=True)
        var w_buf0 = UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=Int(UnsafePointer(to=w_arr0)))
        var w_buf1 = UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=Int(UnsafePointer(to=w_arr1)))
        var w_buf2 = UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=Int(UnsafePointer(to=w_arr2)))
        var w_buf3 = UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=Int(UnsafePointer(to=w_arr3)))

        # L2-resident stack scratch buffers (zero heap allocations for AMD KFD PASID safety)
        var partial_arr = InlineArray[Float32, 16384](uninitialized=True)
        var row_extrap_arr = InlineArray[Float32, 256](uninitialized=True)
        var cum_max_arr = InlineArray[Float32, 256](uninitialized=True)
        var cum_min_arr = InlineArray[Float32, 256](uninitialized=True)
        var cum_sum_arr = InlineArray[Float32, 256](uninitialized=True)
        var cum_sq_arr = InlineArray[Float32, 256](uninitialized=True)
        var cum_valid_arr = InlineArray[Int, 256](uninitialized=True)
        var thresh_arr = InlineArray[Float32, 256](uninitialized=True)
        var is_cand_arr = InlineArray[Bool, 256](uninitialized=True)
        var cand_rows_arr = InlineArray[Int, 256](uninitialized=True)

        var partial_buf = UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=Int(UnsafePointer(to=partial_arr)))
        var row_extrap = UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=Int(UnsafePointer(to=row_extrap_arr)))
        var max_c = UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=Int(UnsafePointer(to=cum_max_arr)))
        var min_c = UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=Int(UnsafePointer(to=cum_min_arr)))
        var cum_sum_v = UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=Int(UnsafePointer(to=cum_sum_arr)))
        var cum_sum_sq = UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=Int(UnsafePointer(to=cum_sq_arr)))
        var cum_valid = UnsafePointer[Int, MutAnyOrigin](unsafe_from_address=Int(UnsafePointer(to=cum_valid_arr)))
        var thresh = UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=Int(UnsafePointer(to=thresh_arr)))
        var is_cand = UnsafePointer[Bool, MutAnyOrigin](unsafe_from_address=Int(UnsafePointer(to=is_cand_arr)))
        var cand_rows = UnsafePointer[Int, MutAnyOrigin](unsafe_from_address=Int(UnsafePointer(to=cand_rows_arr)))

        for b in range(B):
            max_c[b] = -1e30
            min_c[b] = 1e30
            cum_sum_v[b] = 0.0
            cum_sum_sq[b] = 0.0
            cum_valid[b] = 0

        var max_extrap = (Float32(c_in_blocks) / Float32(coarse_blocks)) * Float32(4.0)
        var min_per_worker = (32 + tasks - 1) // tasks
        var min_worker_cands = min_per_worker if chunk_rows >= min_per_worker else chunk_rows
        var worker_total_cands = 0

        var tile_start = start_r
        while tile_start < end_r:
            var tile_end = tile_start + tile_max
            if tile_end > end_r:
                tile_end = end_r
            var tile_rows = tile_end - tile_start

            for lr in range(tile_rows):
                var r_scales = scales + (tile_start + lr) * c_in_blocks
                var s_coarse: Float32 = 0.0
                var s_total: Float32 = 0.0
                for blk in range(coarse_blocks):
                    var sc = r_scales[blk]
                    var a_sc = sc if sc >= 0.0 else -sc
                    s_coarse += a_sc
                    s_total += a_sc
                for blk in range(coarse_blocks, c_in_blocks):
                    var sc = r_scales[blk]
                    var a_sc = sc if sc >= 0.0 else -sc
                    s_total += a_sc
                if s_coarse <= Float32(1e-5) * s_total and s_total > Float32(1e-7):
                    # Row has negligible coarse energy and significant fine energy: force Stage 2 evaluation
                    # and set row_extrap=-1.0 as sentinel so it does not skew Stage 1 statistics
                    row_extrap[lr] = Float32(-1.0)
                else:
                    var ext = s_total / (s_coarse + Float32(1e-12)) if s_coarse > Float32(1e-12) else Float32(1.0)
                    if ext > max_extrap:
                        ext = max_extrap
                    row_extrap[lr] = ext

            # Stage 1: Branchless 4x3 Coarse-Prefix Scan over [0 .. coarse_dim) for current L2 super-tile
            var r = tile_start
            while r + 3 < tile_end:
                var lr = r - tile_start
                unpack_row_deinterleaved_prescaled(
                    W_packed + (r + 0) * c_in_packed, scales + (r + 0) * c_in_blocks, w_buf0, coarse_blocks
                )
                unpack_row_deinterleaved_prescaled(
                    W_packed + (r + 1) * c_in_packed, scales + (r + 1) * c_in_blocks, w_buf1, coarse_blocks
                )
                unpack_row_deinterleaved_prescaled(
                    W_packed + (r + 2) * c_in_packed, scales + (r + 2) * c_in_blocks, w_buf2, coarse_blocks
                )
                unpack_row_deinterleaved_prescaled(
                    W_packed + (r + 3) * c_in_packed, scales + (r + 3) * c_in_blocks, w_buf3, coarse_blocks
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

                    var xb0 = (b + 0) * c_in
                    var xb1 = (b + 1) * c_in
                    var xb2 = (b + 2) * c_in

                    for k in range(0, coarse_dim, 8):
                        var w0 = w_buf0.load[8](k)
                        var w1 = w_buf1.load[8](k)
                        var w2 = w_buf2.load[8](k)
                        var w3 = w_buf3.load[8](k)

                        var x0 = x_deint_ptr.load[8](xb0 + k)
                        acc0_0 = w0.fma(x0, acc0_0)
                        acc1_0 = w1.fma(x0, acc1_0)
                        acc2_0 = w2.fma(x0, acc2_0)
                        acc3_0 = w3.fma(x0, acc3_0)

                        var x1 = x_deint_ptr.load[8](xb1 + k)
                        acc0_1 = w0.fma(x1, acc0_1)
                        acc1_1 = w1.fma(x1, acc1_1)
                        acc2_1 = w2.fma(x1, acc2_1)
                        acc3_1 = w3.fma(x1, acc3_1)

                        var x2 = x_deint_ptr.load[8](xb2 + k)
                        acc0_2 = w0.fma(x2, acc0_2)
                        acc1_2 = w1.fma(x2, acc1_2)
                        acc2_2 = w2.fma(x2, acc2_2)
                        acc3_2 = w3.fma(x2, acc3_2)

                    partial_buf[(lr + 0) * B + b + 0] = acc0_0.reduce_add()
                    partial_buf[(lr + 0) * B + b + 1] = acc0_1.reduce_add()
                    partial_buf[(lr + 0) * B + b + 2] = acc0_2.reduce_add()
                    partial_buf[(lr + 1) * B + b + 0] = acc1_0.reduce_add()
                    partial_buf[(lr + 1) * B + b + 1] = acc1_1.reduce_add()
                    partial_buf[(lr + 1) * B + b + 2] = acc1_2.reduce_add()
                    partial_buf[(lr + 2) * B + b + 0] = acc2_0.reduce_add()
                    partial_buf[(lr + 2) * B + b + 1] = acc2_1.reduce_add()
                    partial_buf[(lr + 2) * B + b + 2] = acc2_2.reduce_add()
                    partial_buf[(lr + 3) * B + b + 0] = acc3_0.reduce_add()
                    partial_buf[(lr + 3) * B + b + 1] = acc3_1.reduce_add()
                    partial_buf[(lr + 3) * B + b + 2] = acc3_2.reduce_add()
                    b += 3

                while b < B:
                    var acc0 = SIMD[DType.float32, 8](0.0)
                    var acc1 = SIMD[DType.float32, 8](0.0)
                    var acc2 = SIMD[DType.float32, 8](0.0)
                    var acc3 = SIMD[DType.float32, 8](0.0)
                    var xb = b * c_in
                    for k in range(0, coarse_dim, 8):
                        var x0 = x_deint_ptr.load[8](xb + k)
                        acc0 = w_buf0.load[8](k).fma(x0, acc0)
                        acc1 = w_buf1.load[8](k).fma(x0, acc1)
                        acc2 = w_buf2.load[8](k).fma(x0, acc2)
                        acc3 = w_buf3.load[8](k).fma(x0, acc3)
                    partial_buf[(lr + 0) * B + b] = acc0.reduce_add()
                    partial_buf[(lr + 1) * B + b] = acc1.reduce_add()
                    partial_buf[(lr + 2) * B + b] = acc2.reduce_add()
                    partial_buf[(lr + 3) * B + b] = acc3.reduce_add()
                    b += 1

                r += 4

            while r < tile_end:
                var lr = r - tile_start
                unpack_row_deinterleaved_prescaled(
                    W_packed + r * c_in_packed, scales + r * c_in_blocks, w_buf0, coarse_blocks
                )
                for b in range(B):
                    var acc0 = SIMD[DType.float32, 8](0.0)
                    var xb = b * c_in
                    for k in range(0, coarse_dim, 8):
                        acc0 = w_buf0.load[8](k).fma(x_deint_ptr.load[8](xb + k), acc0)
                    partial_buf[lr * B + b] = acc0.reduce_add()
                r += 1

            # Update cumulative per-token scale-extrapolated coarse statistics across seen rows
            for b in range(B):
                var sum_v = cum_sum_v[b]
                var sum_sq = cum_sum_sq[b]
                var valid_rows = cum_valid[b]
                for lr in range(tile_rows):
                    var ext = row_extrap[lr]
                    if ext < Float32(0.0):
                        continue
                    var v = partial_buf[lr * B + b] * ext
                    sum_v += v
                    sum_sq += v * v
                    valid_rows += 1
                    if v > max_c[b]:
                        max_c[b] = v
                    if v < min_c[b]:
                        min_c[b] = v
                cum_sum_v[b] = sum_v
                cum_sum_sq[b] = sum_sq
                cum_valid[b] = valid_rows
                var span = max_c[b] - min_c[b]
                if valid_rows == 0 or span < Float32(1e-6):
                    # Token b has zero coarse variance; do not let a single zero-coarse token force all rows on
                    thresh[b] = Float32(1e30)
                else:
                    var mean_v = sum_v / Float32(valid_rows)
                    var var_v = (sum_sq / Float32(valid_rows)) - mean_v * mean_v
                    if var_v < Float32(0.0):
                        var_v = Float32(0.0)
                    var std_v = sqrt(var_v + Float32(1e-12))
                    var z_cut = (Float32(0.75) - span_keep_frac) * Float32(3.8)
                    var z_thresh = mean_v + z_cut * std_v
                    if z_thresh > max_c[b]:
                        z_thresh = max_c[b]
                    thresh[b] = z_thresh

            # Mark candidate rows and union with forced_cand_tokens
            for lr in range(tile_rows):
                var p_row = partial_buf + lr * B
                var ext = row_extrap[lr]
                if ext < Float32(0.0):
                    is_cand[lr] = True
                    continue
                var keep = False
                for b in range(B):
                    if p_row[b] * ext >= thresh[b]:
                        keep = True
                        break
                is_cand[lr] = keep

            if has_forced:
                for f in range(num_forced_cands):
                    var tok = forced_cand_tokens[f]
                    if tok >= tile_start and tok < tile_end:
                        is_cand[tok - tile_start] = True

            var num_cands = 0
            for lr in range(tile_rows):
                if is_cand[lr]:
                    cand_rows[num_cands] = lr
                    num_cands += 1

            # Guarantee at least min_worker_cands across the worker's tiles
            if worker_total_cands + num_cands < min_worker_cands:
                var needed = min_worker_cands - worker_total_cands
                if needed > tile_rows:
                    needed = tile_rows
                for lr in range(tile_rows):
                    if not is_cand[lr]:
                        is_cand[lr] = True
                        cand_rows[num_cands] = lr
                        num_cands += 1
                        if num_cands >= needed:
                            break
            worker_total_cands += num_cands

            for lr in range(tile_rows):
                if not is_cand[lr]:
                    var row_idx = tile_start + lr
                    for b in range(B):
                        out_ptr[b * o_stride + row_idx] = Scalar[out_type](pruned_sentinel)

            # Stage 2: Branchless 4x3 Fine-Suffix GEMM over [coarse_dim .. c_in) ONLY on compacted cand_rows
            var c = 0
            while c + 3 < num_cands:
                var lr0 = cand_rows[c + 0]
                var lr1 = cand_rows[c + 1]
                var lr2 = cand_rows[c + 2]
                var lr3 = cand_rows[c + 3]
                var r0 = tile_start + lr0
                var r1 = tile_start + lr1
                var r2 = tile_start + lr2
                var r3 = tile_start + lr3

                unpack_row_deinterleaved_prescaled(
                    W_packed + r0 * c_in_packed + coarse_packed, scales + r0 * c_in_blocks + coarse_blocks, w_buf0, fine_blocks
                )
                unpack_row_deinterleaved_prescaled(
                    W_packed + r1 * c_in_packed + coarse_packed, scales + r1 * c_in_blocks + coarse_blocks, w_buf1, fine_blocks
                )
                unpack_row_deinterleaved_prescaled(
                    W_packed + r2 * c_in_packed + coarse_packed, scales + r2 * c_in_blocks + coarse_blocks, w_buf2, fine_blocks
                )
                unpack_row_deinterleaved_prescaled(
                    W_packed + r3 * c_in_packed + coarse_packed, scales + r3 * c_in_blocks + coarse_blocks, w_buf3, fine_blocks
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

                    var xb0 = (b + 0) * c_in + coarse_dim
                    var xb1 = (b + 1) * c_in + coarse_dim
                    var xb2 = (b + 2) * c_in + coarse_dim

                    for k in range(0, fine_dim, 8):
                        var w0 = w_buf0.load[8](k)
                        var w1 = w_buf1.load[8](k)
                        var w2 = w_buf2.load[8](k)
                        var w3 = w_buf3.load[8](k)

                        var x0 = x_deint_ptr.load[8](xb0 + k)
                        acc0_0 = w0.fma(x0, acc0_0)
                        acc1_0 = w1.fma(x0, acc1_0)
                        acc2_0 = w2.fma(x0, acc2_0)
                        acc3_0 = w3.fma(x0, acc3_0)

                        var x1 = x_deint_ptr.load[8](xb1 + k)
                        acc0_1 = w0.fma(x1, acc0_1)
                        acc1_1 = w1.fma(x1, acc1_1)
                        acc2_1 = w2.fma(x1, acc2_1)
                        acc3_1 = w3.fma(x1, acc3_1)

                        var x2 = x_deint_ptr.load[8](xb2 + k)
                        acc0_2 = w0.fma(x2, acc0_2)
                        acc1_2 = w1.fma(x2, acc1_2)
                        acc2_2 = w2.fma(x2, acc2_2)
                        acc3_2 = w3.fma(x2, acc3_2)

                    out_ptr[(b + 0) * o_stride + r0] = (partial_buf[lr0 * B + b + 0] + acc0_0.reduce_add()).cast[out_type]()
                    out_ptr[(b + 1) * o_stride + r0] = (partial_buf[lr0 * B + b + 1] + acc0_1.reduce_add()).cast[out_type]()
                    out_ptr[(b + 2) * o_stride + r0] = (partial_buf[lr0 * B + b + 2] + acc0_2.reduce_add()).cast[out_type]()
                    out_ptr[(b + 0) * o_stride + r1] = (partial_buf[lr1 * B + b + 0] + acc1_0.reduce_add()).cast[out_type]()
                    out_ptr[(b + 1) * o_stride + r1] = (partial_buf[lr1 * B + b + 1] + acc1_1.reduce_add()).cast[out_type]()
                    out_ptr[(b + 2) * o_stride + r1] = (partial_buf[lr1 * B + b + 2] + acc1_2.reduce_add()).cast[out_type]()
                    out_ptr[(b + 0) * o_stride + r2] = (partial_buf[lr2 * B + b + 0] + acc2_0.reduce_add()).cast[out_type]()
                    out_ptr[(b + 1) * o_stride + r2] = (partial_buf[lr2 * B + b + 1] + acc2_1.reduce_add()).cast[out_type]()
                    out_ptr[(b + 2) * o_stride + r2] = (partial_buf[lr2 * B + b + 2] + acc2_2.reduce_add()).cast[out_type]()
                    out_ptr[(b + 0) * o_stride + r3] = (partial_buf[lr3 * B + b + 0] + acc3_0.reduce_add()).cast[out_type]()
                    out_ptr[(b + 1) * o_stride + r3] = (partial_buf[lr3 * B + b + 1] + acc3_1.reduce_add()).cast[out_type]()
                    out_ptr[(b + 2) * o_stride + r3] = (partial_buf[lr3 * B + b + 2] + acc3_2.reduce_add()).cast[out_type]()
                    b += 3

                while b < B:
                    var acc0 = SIMD[DType.float32, 8](0.0)
                    var acc1 = SIMD[DType.float32, 8](0.0)
                    var acc2 = SIMD[DType.float32, 8](0.0)
                    var acc3 = SIMD[DType.float32, 8](0.0)
                    var xb = b * c_in + coarse_dim
                    for k in range(0, fine_dim, 8):
                        var x0 = x_deint_ptr.load[8](xb + k)
                        acc0 = w_buf0.load[8](k).fma(x0, acc0)
                        acc1 = w_buf1.load[8](k).fma(x0, acc1)
                        acc2 = w_buf2.load[8](k).fma(x0, acc2)
                        acc3 = w_buf3.load[8](k).fma(x0, acc3)
                    out_ptr[b * o_stride + r0] = (partial_buf[lr0 * B + b] + acc0.reduce_add()).cast[out_type]()
                    out_ptr[b * o_stride + r1] = (partial_buf[lr1 * B + b] + acc1.reduce_add()).cast[out_type]()
                    out_ptr[b * o_stride + r2] = (partial_buf[lr2 * B + b] + acc2.reduce_add()).cast[out_type]()
                    out_ptr[b * o_stride + r3] = (partial_buf[lr3 * B + b] + acc3.reduce_add()).cast[out_type]()
                    b += 1

                c += 4

            while c < num_cands:
                var lr0 = cand_rows[c]
                var r0 = tile_start + lr0
                unpack_row_deinterleaved_prescaled(
                    W_packed + r0 * c_in_packed + coarse_packed, scales + r0 * c_in_blocks + coarse_blocks, w_buf0, fine_blocks
                )
                for b in range(B):
                    var acc0 = SIMD[DType.float32, 8](0.0)
                    var xb = b * c_in + coarse_dim
                    for k in range(0, fine_dim, 8):
                        acc0 = w_buf0.load[8](k).fma(x_deint_ptr.load[8](xb + k), acc0)
                    out_ptr[b * o_stride + r0] = (partial_buf[lr0 * B + b] + acc0.reduce_add()).cast[out_type]()
                c += 1

            tile_start = tile_end

        _ = partial_arr[0]
        _ = row_extrap_arr[0]
        _ = cum_max_arr[0]
        _ = cum_min_arr[0]
        _ = cum_sum_arr[0]
        _ = cum_sq_arr[0]
        _ = cum_valid_arr[0]
        _ = thresh_arr[0]
        _ = is_cand_arr[0]
        _ = cand_rows_arr[0]
        _ = w_arr0[0]
        _ = w_arr1[0]
        _ = w_arr2[0]
        _ = w_arr3[0]

    if tasks == 1:
        worker(0)
    else:
        parallelize[worker](tasks)
    _ = x_stack[0]


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
    pre_deint_X: UnsafePointer[Float32, _] = UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=Int(0)),
):
    if B <= 0 or c_in <= 0 or c_out <= 0:
        return
    var tasks = num_tasks
    if tasks > c_out:
        tasks = c_out
    if tasks < 1:
        tasks = 1

    var chunk_size = c_out // tasks
    var c_in_packed = c_in // 2
    var c_in_blocks = c_in // 32

    var x_stack = InlineArray[Float32, 65536](uninitialized=True)
    var use_deint = Int(pre_deint_X) != 0 and c_in <= 4096 and (c_in % 32 == 0) and B <= 256
    var x_deint_ptr = UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=Int(pre_deint_X))
    if not use_deint and c_in <= 4096 and (c_in % 32 == 0) and B <= 256 and B * c_in <= 65536:
        x_deint_ptr = UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=Int(UnsafePointer(to=x_stack)))
        deinterleave_batch_to_f32[in_type](X, x_deint_ptr, B, c_in)
        use_deint = True

    def worker(t: Int) capturing -> None:
        var start_r = t * chunk_size
        var end_r = (t + 1) * chunk_size if t < tasks - 1 else c_out

        var wg_arr0 = InlineArray[Float32, 4096](uninitialized=True)
        var wg_arr1 = InlineArray[Float32, 4096](uninitialized=True)
        var wu_arr0 = InlineArray[Float32, 4096](uninitialized=True)
        var wu_arr1 = InlineArray[Float32, 4096](uninitialized=True)
        var w_gate0 = UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=Int(UnsafePointer(to=wg_arr0)))
        var w_gate1 = UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=Int(UnsafePointer(to=wg_arr1)))
        var w_up0 = UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=Int(UnsafePointer(to=wu_arr0)))
        var w_up1 = UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=Int(UnsafePointer(to=wu_arr1)))

        var r = start_r
        if use_deint:
            # Stream-separated 4-row x 3-batch GeGLU tile (llm-mojo-yj4.5):
            # Pass 1 unpacks 4 contiguous rows of W_gate into w_gate0..w_up1 and computes gate_tile[4 * B].
            # Pass 2 unpacks 4 contiguous rows of W_up into the same L1 buffers and fuses GeGLU!
            var gate_stack = InlineArray[Float32, 1024](uninitialized=True)
            var gate_tile = UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=Int(UnsafePointer(to=gate_stack)))

            while r + 3 < end_r:
                var r0 = r
                var r1 = r + 1
                var r2 = r + 2
                var r3 = r + 3
                # Contiguous Stream 1: 4 rows of W_gate
                unpack_row_deinterleaved_prescaled(
                    W_gate_packed + r0 * c_in_packed, W_gate_scales + r0 * c_in_blocks, w_gate0, c_in_blocks
                )
                unpack_row_deinterleaved_prescaled(
                    W_gate_packed + r1 * c_in_packed, W_gate_scales + r1 * c_in_blocks, w_gate1, c_in_blocks
                )
                unpack_row_deinterleaved_prescaled(
                    W_gate_packed + r2 * c_in_packed, W_gate_scales + r2 * c_in_blocks, w_up0, c_in_blocks
                )
                unpack_row_deinterleaved_prescaled(
                    W_gate_packed + r3 * c_in_packed, W_gate_scales + r3 * c_in_blocks, w_up1, c_in_blocks
                )

                var b = 0
                while b + 2 < B:
                    var g0_0 = SIMD[DType.float32, 8](0.0)
                    var g0_1 = SIMD[DType.float32, 8](0.0)
                    var g0_2 = SIMD[DType.float32, 8](0.0)
                    var g1_0 = SIMD[DType.float32, 8](0.0)
                    var g1_1 = SIMD[DType.float32, 8](0.0)
                    var g1_2 = SIMD[DType.float32, 8](0.0)
                    var g2_0 = SIMD[DType.float32, 8](0.0)
                    var g2_1 = SIMD[DType.float32, 8](0.0)
                    var g2_2 = SIMD[DType.float32, 8](0.0)
                    var g3_0 = SIMD[DType.float32, 8](0.0)
                    var g3_1 = SIMD[DType.float32, 8](0.0)
                    var g3_2 = SIMD[DType.float32, 8](0.0)

                    var xb0 = (b + 0) * c_in
                    var xb1 = (b + 1) * c_in
                    var xb2 = (b + 2) * c_in
                    for k in range(0, c_in, 8):
                        var wg0 = w_gate0.load[8](k)
                        var wg1 = w_gate1.load[8](k)
                        var wg2 = w_up0.load[8](k)
                        var wg3 = w_up1.load[8](k)

                        var x0 = x_deint_ptr.load[8](xb0 + k)
                        g0_0 = wg0.fma(x0, g0_0)
                        g1_0 = wg1.fma(x0, g1_0)
                        g2_0 = wg2.fma(x0, g2_0)
                        g3_0 = wg3.fma(x0, g3_0)

                        var x1 = x_deint_ptr.load[8](xb1 + k)
                        g0_1 = wg0.fma(x1, g0_1)
                        g1_1 = wg1.fma(x1, g1_1)
                        g2_1 = wg2.fma(x1, g2_1)
                        g3_1 = wg3.fma(x1, g3_1)

                        var x2 = x_deint_ptr.load[8](xb2 + k)
                        g0_2 = wg0.fma(x2, g0_2)
                        g1_2 = wg1.fma(x2, g1_2)
                        g2_2 = wg2.fma(x2, g2_2)
                        g3_2 = wg3.fma(x2, g3_2)

                    gate_tile[(b + 0) * 4 + 0] = g0_0.reduce_add()
                    gate_tile[(b + 1) * 4 + 0] = g0_1.reduce_add()
                    gate_tile[(b + 2) * 4 + 0] = g0_2.reduce_add()
                    gate_tile[(b + 0) * 4 + 1] = g1_0.reduce_add()
                    gate_tile[(b + 1) * 4 + 1] = g1_1.reduce_add()
                    gate_tile[(b + 2) * 4 + 1] = g1_2.reduce_add()
                    gate_tile[(b + 0) * 4 + 2] = g2_0.reduce_add()
                    gate_tile[(b + 1) * 4 + 2] = g2_1.reduce_add()
                    gate_tile[(b + 2) * 4 + 2] = g2_2.reduce_add()
                    gate_tile[(b + 0) * 4 + 3] = g3_0.reduce_add()
                    gate_tile[(b + 1) * 4 + 3] = g3_1.reduce_add()
                    gate_tile[(b + 2) * 4 + 3] = g3_2.reduce_add()
                    b += 3

                while b < B:
                    var g0 = SIMD[DType.float32, 8](0.0)
                    var g1 = SIMD[DType.float32, 8](0.0)
                    var g2 = SIMD[DType.float32, 8](0.0)
                    var g3 = SIMD[DType.float32, 8](0.0)
                    var xb = b * c_in
                    for k in range(0, c_in, 8):
                        var x0 = x_deint_ptr.load[8](xb + k)
                        g0 = w_gate0.load[8](k).fma(x0, g0)
                        g1 = w_gate1.load[8](k).fma(x0, g1)
                        g2 = w_up0.load[8](k).fma(x0, g2)
                        g3 = w_up1.load[8](k).fma(x0, g3)
                    gate_tile[b * 4 + 0] = g0.reduce_add()
                    gate_tile[b * 4 + 1] = g1.reduce_add()
                    gate_tile[b * 4 + 2] = g2.reduce_add()
                    gate_tile[b * 4 + 3] = g3.reduce_add()
                    b += 1

                # Contiguous Stream 2: 4 rows of W_up + fused GeGLU
                unpack_row_deinterleaved_prescaled(
                    W_up_packed + r0 * c_in_packed, W_up_scales + r0 * c_in_blocks, w_gate0, c_in_blocks
                )
                unpack_row_deinterleaved_prescaled(
                    W_up_packed + r1 * c_in_packed, W_up_scales + r1 * c_in_blocks, w_gate1, c_in_blocks
                )
                unpack_row_deinterleaved_prescaled(
                    W_up_packed + r2 * c_in_packed, W_up_scales + r2 * c_in_blocks, w_up0, c_in_blocks
                )
                unpack_row_deinterleaved_prescaled(
                    W_up_packed + r3 * c_in_packed, W_up_scales + r3 * c_in_blocks, w_up1, c_in_blocks
                )

                b = 0
                while b + 2 < B:
                    var u0_0 = SIMD[DType.float32, 8](0.0)
                    var u0_1 = SIMD[DType.float32, 8](0.0)
                    var u0_2 = SIMD[DType.float32, 8](0.0)
                    var u1_0 = SIMD[DType.float32, 8](0.0)
                    var u1_1 = SIMD[DType.float32, 8](0.0)
                    var u1_2 = SIMD[DType.float32, 8](0.0)
                    var u2_0 = SIMD[DType.float32, 8](0.0)
                    var u2_1 = SIMD[DType.float32, 8](0.0)
                    var u2_2 = SIMD[DType.float32, 8](0.0)
                    var u3_0 = SIMD[DType.float32, 8](0.0)
                    var u3_1 = SIMD[DType.float32, 8](0.0)
                    var u3_2 = SIMD[DType.float32, 8](0.0)

                    var xb0 = (b + 0) * c_in
                    var xb1 = (b + 1) * c_in
                    var xb2 = (b + 2) * c_in
                    for k in range(0, c_in, 8):
                        var wu0 = w_gate0.load[8](k)
                        var wu1 = w_gate1.load[8](k)
                        var wu2 = w_up0.load[8](k)
                        var wu3 = w_up1.load[8](k)

                        var x0 = x_deint_ptr.load[8](xb0 + k)
                        u0_0 = wu0.fma(x0, u0_0)
                        u1_0 = wu1.fma(x0, u1_0)
                        u2_0 = wu2.fma(x0, u2_0)
                        u3_0 = wu3.fma(x0, u3_0)

                        var x1 = x_deint_ptr.load[8](xb1 + k)
                        u0_1 = wu0.fma(x1, u0_1)
                        u1_1 = wu1.fma(x1, u1_1)
                        u2_1 = wu2.fma(x1, u2_1)
                        u3_1 = wu3.fma(x1, u3_1)

                        var x2 = x_deint_ptr.load[8](xb2 + k)
                        u0_2 = wu0.fma(x2, u0_2)
                        u1_2 = wu1.fma(x2, u1_2)
                        u2_2 = wu2.fma(x2, u2_2)
                        u3_2 = wu3.fma(x2, u3_2)

                    var gate_v0 = SIMD[DType.float32, 4](
                        gate_tile[(b + 0) * 4 + 0], gate_tile[(b + 0) * 4 + 1],
                        gate_tile[(b + 0) * 4 + 2], gate_tile[(b + 0) * 4 + 3],
                    )
                    var up_v0 = SIMD[DType.float32, 4](
                        u0_0.reduce_add(), u1_0.reduce_add(), u2_0.reduce_add(), u3_0.reduce_add()
                    )
                    var act_v0 = (gate_v0 * (1.0 / (1.0 + exp(SIMD[DType.float32, 4](-1.702) * gate_v0))) * up_v0).cast[out_type]()
                    out_gated.store((b + 0) * c_out + r0, act_v0)

                    var gate_v1 = SIMD[DType.float32, 4](
                        gate_tile[(b + 1) * 4 + 0], gate_tile[(b + 1) * 4 + 1],
                        gate_tile[(b + 1) * 4 + 2], gate_tile[(b + 1) * 4 + 3],
                    )
                    var up_v1 = SIMD[DType.float32, 4](
                        u0_1.reduce_add(), u1_1.reduce_add(), u2_1.reduce_add(), u3_1.reduce_add()
                    )
                    var act_v1 = (gate_v1 * (1.0 / (1.0 + exp(SIMD[DType.float32, 4](-1.702) * gate_v1))) * up_v1).cast[out_type]()
                    out_gated.store((b + 1) * c_out + r0, act_v1)

                    var gate_v2 = SIMD[DType.float32, 4](
                        gate_tile[(b + 2) * 4 + 0], gate_tile[(b + 2) * 4 + 1],
                        gate_tile[(b + 2) * 4 + 2], gate_tile[(b + 2) * 4 + 3],
                    )
                    var up_v2 = SIMD[DType.float32, 4](
                        u0_2.reduce_add(), u1_2.reduce_add(), u2_2.reduce_add(), u3_2.reduce_add()
                    )
                    var act_v2 = (gate_v2 * (1.0 / (1.0 + exp(SIMD[DType.float32, 4](-1.702) * gate_v2))) * up_v2).cast[out_type]()
                    out_gated.store((b + 2) * c_out + r0, act_v2)
                    b += 3

                while b < B:
                    var u0 = SIMD[DType.float32, 8](0.0)
                    var u1 = SIMD[DType.float32, 8](0.0)
                    var u2 = SIMD[DType.float32, 8](0.0)
                    var u3 = SIMD[DType.float32, 8](0.0)
                    var xb = b * c_in
                    for k in range(0, c_in, 8):
                        var x0 = x_deint_ptr.load[8](xb + k)
                        u0 = w_gate0.load[8](k).fma(x0, u0)
                        u1 = w_gate1.load[8](k).fma(x0, u1)
                        u2 = w_up0.load[8](k).fma(x0, u2)
                        u3 = w_up1.load[8](k).fma(x0, u3)
                    var gate_vb = SIMD[DType.float32, 4](
                        gate_tile[b * 4 + 0], gate_tile[b * 4 + 1], gate_tile[b * 4 + 2], gate_tile[b * 4 + 3]
                    )
                    var up_vb = SIMD[DType.float32, 4](
                        u0.reduce_add(), u1.reduce_add(), u2.reduce_add(), u3.reduce_add()
                    )
                    var act_vb = (gate_vb * (1.0 / (1.0 + exp(SIMD[DType.float32, 4](-1.702) * gate_vb))) * up_vb).cast[out_type]()
                    out_gated.store(b * c_out + r0, act_vb)
                    b += 1

                r += 4

            while r < end_r:
                unpack_row_deinterleaved_prescaled(
                    W_gate_packed + r * c_in_packed, W_gate_scales + r * c_in_blocks, w_gate0, c_in_blocks
                )
                unpack_row_deinterleaved_prescaled(
                    W_up_packed + r * c_in_packed, W_up_scales + r * c_in_blocks, w_up0, c_in_blocks
                )
                for b in range(B):
                    var g0 = SIMD[DType.float32, 8](0.0)
                    var u0 = SIMD[DType.float32, 8](0.0)
                    var xb = b * c_in
                    for k in range(0, c_in, 8):
                        var x0 = x_deint_ptr.load[8](xb + k)
                        g0 = w_gate0.load[8](k).fma(x0, g0)
                        u0 = w_up0.load[8](k).fma(x0, u0)
                    var gv = g0.reduce_add()
                    var uv = u0.reduce_add()
                    var sig = 1.0 / (1.0 + exp(Float32(-1.702) * gv))
                    out_gated[b * c_out + r] = Scalar[out_type](gv * sig * uv)
                r += 1
            _ = gate_stack[0]
            _ = wg_arr0[0]
            _ = wg_arr1[0]
            _ = wu_arr0[0]
            _ = wu_arr1[0]
            return
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

    if tasks == 1:
        worker(0)
    else:
        parallelize[worker](tasks)
    _ = x_stack[0]


# Branchless Pre-Compacted Block-Sparse W_down Q4 GEMM (llm-mojo-yj4.5).
# Pre-scans 32-element Q4 blocks of X across all B tokens once before the parallel c_out loop,
# builds active_blks[0..num_active_blks-1], compacts X into contiguous deinterleaved active blocks,
# and executes 100% branchless 4x3 AVX2 FMA over only the active blocks.
@always_inline
def q4_gemm_block_sparse_down[
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
    in_stride: Int = 0,
    out_stride: Int = 0,
    pre_deint_X: UnsafePointer[Float32, _] = UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=Int(0)),
    eps_blk: Float32 = 1e-6,
):
    if B <= 0 or c_in <= 0 or c_out <= 0:
        return
    var x_stride = in_stride if in_stride > 0 else c_in
    var o_stride = out_stride if out_stride > 0 else c_out
    if c_in > 4096 or (c_in % 32 != 0):
        q4_gemm_l1_2x4[in_type, out_type](
            W_packed, scales, X, out_ptr, B, c_in, c_out, num_tasks, in_stride, out_stride, pre_deint_X
        )
        return

    var c_in_blocks = c_in // 32
    var c_in_packed = c_in // 2
    var act_blk_arr = InlineArray[Int, 128](uninitialized=True)
    var active_blks = UnsafePointer[Int, MutAnyOrigin](unsafe_from_address=Int(UnsafePointer(to=act_blk_arr)))
    var num_active_blks = 0
    var has_pre_deint = Int(pre_deint_X) != 0

    for blk in range(c_in_blocks):
        var blk_off = blk * 32
        var is_active = False
        for b in range(B):
            var max_v: Float32
            if has_pre_deint:
                var xb = b * c_in + blk_off
                var m0 = abs(pre_deint_X.load[8](xb + 0))
                var m1 = abs(pre_deint_X.load[8](xb + 8))
                var m2 = abs(pre_deint_X.load[8](xb + 16))
                var m3 = abs(pre_deint_X.load[8](xb + 24))
                max_v = max(max(m0, m1), max(m2, m3)).reduce_max()
            else:
                var xb = b * x_stride + blk_off
                var m0 = abs(X.load[8](xb + 0).cast[DType.float32]())
                var m1 = abs(X.load[8](xb + 8).cast[DType.float32]())
                var m2 = abs(X.load[8](xb + 16).cast[DType.float32]())
                var m3 = abs(X.load[8](xb + 24).cast[DType.float32]())
                max_v = max(max(m0, m1), max(m2, m3)).reduce_max()
            if max_v > eps_blk:
                is_active = True
                break
        if is_active:
            active_blks[num_active_blks] = blk
            num_active_blks += 1

    # Fast path 1: All blocks across the batch are zero
    if num_active_blks == 0:
        for b in range(B):
            for r in range(c_out):
                out_ptr[b * o_stride + r] = Scalar[out_type](0.0)
        _ = act_blk_arr[0]
        return

    # Fast path 2: All blocks are active (dense)
    if num_active_blks == c_in_blocks:
        q4_gemm_l1_2x4[in_type, out_type](
            W_packed, scales, X, out_ptr, B, c_in, c_out, num_tasks, in_stride, out_stride, pre_deint_X
        )
        _ = act_blk_arr[0]
        return

    var active_dim = num_active_blks * 32
    var x_stack = InlineArray[Float32, 65536](uninitialized=True)
    var x_compact = UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=Int(UnsafePointer(to=x_stack)))
    if B * active_dim > 65536:
        if has_pre_deint:
            # In-place monotonic compaction in pre_deint_X (b * active_dim + ab * 32 <= b * c_in + active_blks[ab] * 32)
            x_compact = UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=Int(pre_deint_X))
        else:
            q4_gemm_l1_2x4[in_type, out_type](
                W_packed, scales, X, out_ptr, B, c_in, c_out, num_tasks, in_stride, out_stride, pre_deint_X
            )
            _ = act_blk_arr[0]
            return

    for b in range(B):
        var dst_row = x_compact + b * active_dim
        if has_pre_deint:
            var src_row = pre_deint_X + b * c_in
            for ab in range(num_active_blks):
                var blk_off = active_blks[ab] * 32
                var d_off = ab * 32
                dst_row.store(d_off + 0, src_row.load[16](blk_off + 0))
                dst_row.store(d_off + 16, src_row.load[16](blk_off + 16))
        else:
            var src_row = X + b * x_stride
            for ab in range(num_active_blks):
                var blk_off = active_blks[ab] * 32
                var d_off = ab * 32
                var v0 = src_row.load[16](blk_off).cast[DType.float32]()
                var v1 = src_row.load[16](blk_off + 16).cast[DType.float32]()
                dst_row.store(d_off + 0, v0.deinterleave()[0].join(v1.deinterleave()[0]))
                dst_row.store(d_off + 16, v0.deinterleave()[1].join(v1.deinterleave()[1]))

    var tasks = num_tasks
    if tasks > c_out:
        tasks = c_out
    if tasks < 1:
        tasks = 1
    var chunk_size = c_out // tasks

    def worker(t: Int) capturing -> None:
        var start_r = t * chunk_size
        var end_r = (t + 1) * chunk_size if t < tasks - 1 else c_out

        var w_arr0 = InlineArray[Float32, 4096](uninitialized=True)
        var w_arr1 = InlineArray[Float32, 4096](uninitialized=True)
        var w_arr2 = InlineArray[Float32, 4096](uninitialized=True)
        var w_arr3 = InlineArray[Float32, 4096](uninitialized=True)
        var w_buf0 = UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=Int(UnsafePointer(to=w_arr0)))
        var w_buf1 = UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=Int(UnsafePointer(to=w_arr1)))
        var w_buf2 = UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=Int(UnsafePointer(to=w_arr2)))
        var w_buf3 = UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=Int(UnsafePointer(to=w_arr3)))

        var r = start_r
        while r + 3 < end_r:
            var r0 = r
            var r1 = r + 1
            var r2 = r + 2
            var r3 = r + 3
            for ab in range(num_active_blks):
                var blk = active_blks[ab]
                var p_off = blk * 16
                var d_off = ab * 32
                unpack_row_deinterleaved_prescaled(
                    W_packed + r0 * c_in_packed + p_off, scales + r0 * c_in_blocks + blk, w_buf0 + d_off, 1
                )
                unpack_row_deinterleaved_prescaled(
                    W_packed + r1 * c_in_packed + p_off, scales + r1 * c_in_blocks + blk, w_buf1 + d_off, 1
                )
                unpack_row_deinterleaved_prescaled(
                    W_packed + r2 * c_in_packed + p_off, scales + r2 * c_in_blocks + blk, w_buf2 + d_off, 1
                )
                unpack_row_deinterleaved_prescaled(
                    W_packed + r3 * c_in_packed + p_off, scales + r3 * c_in_blocks + blk, w_buf3 + d_off, 1
                )

            var b = 0
            while b + 2 < B:
                var acc00 = SIMD[DType.float32, 8](0.0)
                var acc01 = SIMD[DType.float32, 8](0.0)
                var acc02 = SIMD[DType.float32, 8](0.0)
                var acc10 = SIMD[DType.float32, 8](0.0)
                var acc11 = SIMD[DType.float32, 8](0.0)
                var acc12 = SIMD[DType.float32, 8](0.0)
                var acc20 = SIMD[DType.float32, 8](0.0)
                var acc21 = SIMD[DType.float32, 8](0.0)
                var acc22 = SIMD[DType.float32, 8](0.0)
                var acc30 = SIMD[DType.float32, 8](0.0)
                var acc31 = SIMD[DType.float32, 8](0.0)
                var acc32 = SIMD[DType.float32, 8](0.0)

                var xb0 = (b + 0) * active_dim
                var xb1 = (b + 1) * active_dim
                var xb2 = (b + 2) * active_dim
                for k in range(0, active_dim, 8):
                    var w0 = w_buf0.load[8](k)
                    var w1 = w_buf1.load[8](k)
                    var w2 = w_buf2.load[8](k)
                    var w3 = w_buf3.load[8](k)

                    var x0 = x_compact.load[8](xb0 + k)
                    acc00 = w0.fma(x0, acc00)
                    acc10 = w1.fma(x0, acc10)
                    acc20 = w2.fma(x0, acc20)
                    acc30 = w3.fma(x0, acc30)

                    var x1 = x_compact.load[8](xb1 + k)
                    acc01 = w0.fma(x1, acc01)
                    acc11 = w1.fma(x1, acc11)
                    acc21 = w2.fma(x1, acc21)
                    acc31 = w3.fma(x1, acc31)

                    var x2 = x_compact.load[8](xb2 + k)
                    acc02 = w0.fma(x2, acc02)
                    acc12 = w1.fma(x2, acc12)
                    acc22 = w2.fma(x2, acc22)
                    acc32 = w3.fma(x2, acc32)

                out_ptr.store(
                    (b + 0) * o_stride + r0,
                    SIMD[DType.float32, 4](
                        acc00.reduce_add(), acc10.reduce_add(), acc20.reduce_add(), acc30.reduce_add()
                    ).cast[out_type](),
                )
                out_ptr.store(
                    (b + 1) * o_stride + r0,
                    SIMD[DType.float32, 4](
                        acc01.reduce_add(), acc11.reduce_add(), acc21.reduce_add(), acc31.reduce_add()
                    ).cast[out_type](),
                )
                out_ptr.store(
                    (b + 2) * o_stride + r0,
                    SIMD[DType.float32, 4](
                        acc02.reduce_add(), acc12.reduce_add(), acc22.reduce_add(), acc32.reduce_add()
                    ).cast[out_type](),
                )
                b += 3

            while b < B:
                var acc0 = SIMD[DType.float32, 8](0.0)
                var acc1 = SIMD[DType.float32, 8](0.0)
                var acc2 = SIMD[DType.float32, 8](0.0)
                var acc3 = SIMD[DType.float32, 8](0.0)
                var xb = b * active_dim
                for k in range(0, active_dim, 8):
                    var x0 = x_compact.load[8](xb + k)
                    acc0 = w_buf0.load[8](k).fma(x0, acc0)
                    acc1 = w_buf1.load[8](k).fma(x0, acc1)
                    acc2 = w_buf2.load[8](k).fma(x0, acc2)
                    acc3 = w_buf3.load[8](k).fma(x0, acc3)
                out_ptr.store(
                    b * o_stride + r0,
                    SIMD[DType.float32, 4](
                        acc0.reduce_add(), acc1.reduce_add(), acc2.reduce_add(), acc3.reduce_add()
                    ).cast[out_type](),
                )
                b += 1

            r += 4

        while r < end_r:
            var r0 = r
            for ab in range(num_active_blks):
                var blk = active_blks[ab]
                unpack_row_deinterleaved_prescaled(
                    W_packed + r0 * c_in_packed + blk * 16, scales + r0 * c_in_blocks + blk, w_buf0 + ab * 32, 1
                )
            for b in range(B):
                var acc0 = SIMD[DType.float32, 8](0.0)
                var xb = b * active_dim
                for k in range(0, active_dim, 8):
                    acc0 = w_buf0.load[8](k).fma(x_compact.load[8](xb + k), acc0)
                out_ptr[b * o_stride + r0] = acc0.reduce_add().cast[out_type]()
            r += 1

        _ = w_arr0[0]
        _ = w_arr1[0]
        _ = w_arr2[0]
        _ = w_arr3[0]

    if tasks == 1:
        worker(0)
    else:
        parallelize[worker](tasks)
    _ = x_stack[0]
    _ = act_blk_arr[0]


# Router-Subspace FFN Bypass for Stationary Canvas Tokens (llm-mojo-yj4.6).
# Uses the already-computed 128-dim MoE router logits r_curr[b] and top-2 expert IDs as a
# low-rank subspace monitor. Bypasses accepted tokens (eb_mask[orig_t] == True) whose top-2
# expert IDs match prev_top2[orig_t] and whose relative router logit drift <= tau_ffn.
# Updates r_prev[orig_t] and prev_top2[orig_t] ONLY when token b is evaluated, preventing
# slow cumulative drift across consecutive steps from escaping detection!
@always_inline
def router_subspace_ffn_filter(
    r_curr: UnsafePointer[Float32, _],
    r_prev: UnsafePointer[Float32, MutAnyOrigin],
    top_idx: UnsafePointer[Int, _],
    prev_top2: UnsafePointer[Int, MutAnyOrigin],
    eb_mask: UnsafePointer[Bool, _],
    active_canvas_idx: UnsafePointer[Int, _],
    ffn_keep_b: UnsafePointer[Int, MutAnyOrigin],
    B: Int,
    num_experts: Int,
    top_k: Int,
    step: Int,
    tau_ffn: Float32 = 0.05,
    top_weights: UnsafePointer[Float32, MutAnyOrigin] = UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=Int(0)),
) -> Int:
    if B <= 0:
        return 0
    var has_eb = Int(eb_mask) != 0
    var has_idx = Int(active_canvas_idx) != 0
    var has_tw = Int(top_weights) != 0
    var active_count = 0
    var tau_sq = tau_ffn * tau_ffn

    for b in range(B):
        var orig_t = active_canvas_idx[b] if has_idx else b
        var is_accepted = eb_mask[orig_t] if has_eb else True
        var bypass = False

        if step >= 1 and is_accepted and top_k >= 2 and num_experts > 0:
            var e0 = top_idx[b * top_k + 0]
            var e1 = top_idx[b * top_k + 1]
            var p0 = prev_top2[orig_t * 2 + 0]
            var p1 = prev_top2[orig_t * 2 + 1]
            if p0 >= 0 and p1 >= 0 and e0 == p0 and e1 == p1:
                var diff_sq: Float32 = 0.0
                var norm_sq: Float32 = 0.0
                var curr_row = r_curr + b * num_experts
                var prev_row = r_prev + orig_t * num_experts
                var e = 0
                var v_diff = SIMD[DType.float32, 8](0.0)
                var v_norm = SIMD[DType.float32, 8](0.0)
                while e + 7 < num_experts:
                    var vc = curr_row.load[8](e)
                    var vp = prev_row.load[8](e)
                    var vd = vc - vp
                    v_diff = vd.fma(vd, v_diff)
                    v_norm = vp.fma(vp, v_norm)
                    e += 8
                diff_sq = v_diff.reduce_add()
                norm_sq = v_norm.reduce_add()
                while e < num_experts:
                    var vc = curr_row[e]
                    var vp = prev_row[e]
                    var vd = vc - vp
                    diff_sq += vd * vd
                    norm_sq += vp * vp
                    e += 1
                if norm_sq > 1e-12:
                    bypass = diff_sq <= tau_sq * norm_sq
                else:
                    bypass = diff_sq <= 1e-12

        if bypass:
            if has_tw:
                for s in range(top_k):
                    top_weights[b * top_k + s] = 0.0
        else:
            ffn_keep_b[active_count] = b
            active_count += 1
            if top_k >= 1:
                prev_top2[orig_t * 2 + 0] = top_idx[b * top_k + 0]
            if top_k >= 2:
                prev_top2[orig_t * 2 + 1] = top_idx[b * top_k + 1]
            var curr_row = r_curr + b * num_experts
            var prev_row = r_prev + orig_t * num_experts
            for e in range(num_experts):
                prev_row[e] = curr_row[e]

    return active_count


# In-place cache sync for router-subspace MoE bypass:
# When bypassed tokens had their top_weights zeroed before run_diffusion_moe_experts_parallel,
# evaluated tokens already sit in moe_out[b] (and are saved to moe_prev_cache[orig_t]), while
# bypassed tokens restore their cached output from moe_prev_cache[orig_t] into moe_out[b]!
@always_inline
def sync_moe_subspace_cache[
    dtype: DType = DType.float16
](
    moe_out: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    moe_prev_cache: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    active_canvas_idx: UnsafePointer[Int, _],
    ffn_keep_b: UnsafePointer[Int, _],
    B: Int,
    ffn_active_len: Int,
    dim: Int,
):
    if B <= 0 or dim <= 0:
        return
    var has_idx = Int(active_canvas_idx) != 0
    var cur_idx = 0
    for b in range(B):
        var orig_t = active_canvas_idx[b] if has_idx else b
        var row = moe_out + b * dim
        var cache_row = moe_prev_cache + orig_t * dim
        if cur_idx < ffn_active_len and ffn_keep_b[cur_idx] == b:
            var d = 0
            while d + 7 < dim:
                cache_row.store(d, row.load[8](d))
                d += 8
            while d < dim:
                cache_row[d] = row[d]
                d += 1
            cur_idx += 1
        else:
            var d = 0
            while d + 7 < dim:
                row.store(d, cache_row.load[8](d))
                d += 8
            while d < dim:
                row[d] = cache_row[d]
                d += 1


# Scatters newly evaluated FFN/MoE outputs for ffn_keep_b[0..ffn_active_len-1] into ffn_full_out
# and updates ffn_prev_cache[orig_t], while restoring ffn_prev_cache[orig_t] for bypassed tokens!
@always_inline
def scatter_and_restore_ffn_subspace[
    dtype: DType = DType.float16
](
    ffn_sub_out: UnsafePointer[Scalar[dtype], _],
    ffn_prev_cache: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    ffn_full_out: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    active_canvas_idx: UnsafePointer[Int, _],
    ffn_keep_b: UnsafePointer[Int, _],
    B: Int,
    ffn_active_len: Int,
    dim: Int,
):
    if B <= 0 or dim <= 0:
        return
    var has_idx = Int(active_canvas_idx) != 0
    var cur_idx = 0
    for b in range(B):
        var orig_t = active_canvas_idx[b] if has_idx else b
        var dst_row = ffn_full_out + b * dim
        var cache_row = ffn_prev_cache + orig_t * dim
        if cur_idx < ffn_active_len and ffn_keep_b[cur_idx] == b:
            var src_row = ffn_sub_out + cur_idx * dim
            var d = 0
            while d + 7 < dim:
                var v = src_row.load[8](d)
                dst_row.store(d, v)
                cache_row.store(d, v)
                d += 8
            while d < dim:
                var v = src_row[d]
                dst_row[d] = v
                cache_row[d] = v
                d += 1
            cur_idx += 1
        else:
            var d = 0
            while d + 7 < dim:
                dst_row.store(d, cache_row.load[8](d))
                d += 8
            while d < dim:
                dst_row[d] = cache_row[d]
                d += 1


# Detects Global Attention heads that act as pure BOS sinks across all B active canvas tokens:
# sink_mask[h] = True iff max_{b, d} |attn_out[b, h, d] - v_bos[h // kv_group_size, d]| <= eps_sink.
@always_inline
def detect_bos_sink_heads[
    in_type: DType = DType.float16
](
    attn_out: UnsafePointer[Scalar[in_type], _],
    v_bos: UnsafePointer[Scalar[in_type], _],
    sink_mask: UnsafePointer[Bool, MutAnyOrigin],
    B: Int,
    num_heads: Int,
    head_dim: Int,
    kv_group_size: Int = 1,
    eps_sink: Float32 = 1e-3,
) -> Int:
    if B <= 0 or num_heads <= 0 or head_dim <= 0:
        return 0
    var c_in = num_heads * head_dim
    var grp = kv_group_size if kv_group_size > 0 else 1
    var num_sinks = 0
    for h in range(num_heads):
        var kv_h = h // grp
        var v_row = v_bos + kv_h * head_dim
        var is_sink = True
        for b in range(B):
            var a_row = attn_out + b * c_in + h * head_dim
            var d = 0
            var max_diff: Float32 = 0.0
            while d + 7 < head_dim:
                var va = a_row.load[8](d).cast[DType.float32]()
                var vb = v_row.load[8](d).cast[DType.float32]()
                var m = abs(va - vb).reduce_max()
                if m > max_diff:
                    max_diff = m
                d += 8
            while d < head_dim:
                var m = abs(Float32(a_row[d]) - Float32(v_row[d]))
                if m > max_diff:
                    max_diff = m
                d += 1
            if max_diff > eps_sink:
                is_sink = False
                break
        sink_mask[h] = is_sink
        if is_sink:
            num_sinks += 1
    return num_sinks


# BOS-Sink Head Precomputation & Wo Slice Bypass on Global Attention Layers (llm-mojo-yj4.7).
# Partitions heads into sink_heads (sink_mask[h] == True, where attn_out[b, h] == v_bos[h] for all b)
# and active_heads (sink_mask[h] == False). Computes the sink contribution c_sink[r] ONCE per row
# against v_bos, and runs the B-token 4x3 FMA only over the compacted non-sink head columns!
@always_inline
def q4_wo_bos_sink_bypass[
    in_type: DType = DType.float16,
    out_type: DType = DType.float32,
](
    Wo_packed: UnsafePointer[UInt8, _],
    Wo_scales: UnsafePointer[Float32, _],
    attn_out: UnsafePointer[Scalar[in_type], _],
    v_bos: UnsafePointer[Scalar[in_type], _],
    sink_mask: UnsafePointer[Bool, _],
    out_ptr: UnsafePointer[Scalar[out_type], MutAnyOrigin],
    B: Int,
    num_heads: Int,
    head_dim: Int,
    c_out: Int,
    num_tasks: Int,
    kv_group_size: Int = 1,
    prealloc_act_deint: UnsafePointer[Float32, MutAnyOrigin] = UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=Int(0)),
):
    if B <= 0 or num_heads <= 0 or head_dim <= 0 or c_out <= 0:
        return
    var c_in = num_heads * head_dim
    var grp = kv_group_size if kv_group_size > 0 else 1
    if Int(sink_mask) == 0 or Int(v_bos) == 0 or (head_dim % 32 != 0) or num_heads > 64 or c_in > 8192:
        q4_gemm_l1_2x4[in_type, out_type](
            Wo_packed, Wo_scales, attn_out, out_ptr, B, c_in, c_out, num_tasks
        )
        return

    var sink_arr = InlineArray[Int, 64](uninitialized=True)
    var act_arr = InlineArray[Int, 64](uninitialized=True)
    var sink_heads = UnsafePointer[Int, MutAnyOrigin](unsafe_from_address=Int(UnsafePointer(to=sink_arr)))
    var act_heads = UnsafePointer[Int, MutAnyOrigin](unsafe_from_address=Int(UnsafePointer(to=act_arr)))
    var num_sink = 0
    var num_act = 0
    for h in range(num_heads):
        if sink_mask[h]:
            sink_heads[num_sink] = h
            num_sink += 1
        else:
            act_heads[num_act] = h
            num_act += 1

    if num_sink == 0:
        q4_gemm_l1_2x4[in_type, out_type](
            Wo_packed, Wo_scales, attn_out, out_ptr, B, c_in, c_out, num_tasks
        )
        _ = sink_arr[0]
        _ = act_arr[0]
        return

    var sink_dim = num_sink * head_dim
    var act_dim = num_act * head_dim
    var head_blocks = head_dim // 32
    var head_packed = head_dim // 2
    var c_in_blocks = c_in // 32
    var c_in_packed = c_in // 2

    # Pre-deinterleave v_bos for sink heads (1 vector of length sink_dim)
    var v_sink_arr = InlineArray[Float32, 8192](uninitialized=True)
    var v_sink_deint = UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=Int(UnsafePointer(to=v_sink_arr)))
    for sh in range(num_sink):
        var h = sink_heads[sh]
        var kv_h = h // grp
        deinterleave_batch_to_f32[in_type](v_bos + kv_h * head_dim, v_sink_deint + sh * head_dim, 1, head_dim)

    # Pre-deinterleave attn_out for active (non-sink) heads across all B tokens
    var x_act_stack = InlineArray[Float32, 65536](uninitialized=True)
    var x_act_deint = UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=Int(UnsafePointer(to=x_act_stack)))
    if num_act > 0 and B * act_dim > 65536:
        if Int(prealloc_act_deint) != 0:
            x_act_deint = prealloc_act_deint
        else:
            q4_gemm_l1_2x4[in_type, out_type](
                Wo_packed, Wo_scales, attn_out, out_ptr, B, c_in, c_out, num_tasks
            )
            _ = v_sink_arr[0]
            _ = sink_arr[0]
            _ = act_arr[0]
            return

    if num_act > 0:
        for b in range(B):
            var src_b = attn_out + b * c_in
            var dst_b = x_act_deint + b * act_dim
            for ah in range(num_act):
                var h = act_heads[ah]
                deinterleave_batch_to_f32[in_type](src_b + h * head_dim, dst_b + ah * head_dim, 1, head_dim)

    var tasks = num_tasks
    if tasks > c_out:
        tasks = c_out
    if tasks < 1:
        tasks = 1
    var chunk_size = c_out // tasks

    def worker(t: Int) capturing -> None:
        var start_r = t * chunk_size
        var end_r = (t + 1) * chunk_size if t < tasks - 1 else c_out

        var w_arr0 = InlineArray[Float32, 8192](uninitialized=True)
        var w_arr1 = InlineArray[Float32, 8192](uninitialized=True)
        var w_arr2 = InlineArray[Float32, 8192](uninitialized=True)
        var w_arr3 = InlineArray[Float32, 8192](uninitialized=True)
        var w_buf0 = UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=Int(UnsafePointer(to=w_arr0)))
        var w_buf1 = UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=Int(UnsafePointer(to=w_arr1)))
        var w_buf2 = UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=Int(UnsafePointer(to=w_arr2)))
        var w_buf3 = UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=Int(UnsafePointer(to=w_arr3)))

        var r = start_r
        while r + 3 < end_r:
            var r0 = r
            var r1 = r + 1
            var r2 = r + 2
            var r3 = r + 3

            # Step 1: Compute 1-token BOS-sink bias c_sink0..3 for rows r0..r3
            for sh in range(num_sink):
                var h = sink_heads[sh]
                var p_off = h * head_packed
                var s_off = h * head_blocks
                var d_off = sh * head_dim
                unpack_row_deinterleaved_prescaled(
                    Wo_packed + r0 * c_in_packed + p_off, Wo_scales + r0 * c_in_blocks + s_off, w_buf0 + d_off, head_blocks
                )
                unpack_row_deinterleaved_prescaled(
                    Wo_packed + r1 * c_in_packed + p_off, Wo_scales + r1 * c_in_blocks + s_off, w_buf1 + d_off, head_blocks
                )
                unpack_row_deinterleaved_prescaled(
                    Wo_packed + r2 * c_in_packed + p_off, Wo_scales + r2 * c_in_blocks + s_off, w_buf2 + d_off, head_blocks
                )
                unpack_row_deinterleaved_prescaled(
                    Wo_packed + r3 * c_in_packed + p_off, Wo_scales + r3 * c_in_blocks + s_off, w_buf3 + d_off, head_blocks
                )

            var s0 = SIMD[DType.float32, 8](0.0)
            var s1 = SIMD[DType.float32, 8](0.0)
            var s2 = SIMD[DType.float32, 8](0.0)
            var s3 = SIMD[DType.float32, 8](0.0)
            for k in range(0, sink_dim, 8):
                var vs = v_sink_deint.load[8](k)
                s0 = w_buf0.load[8](k).fma(vs, s0)
                s1 = w_buf1.load[8](k).fma(vs, s1)
                s2 = w_buf2.load[8](k).fma(vs, s2)
                s3 = w_buf3.load[8](k).fma(vs, s3)
            var c_sink_v = SIMD[DType.float32, 4](
                s0.reduce_add(), s1.reduce_add(), s2.reduce_add(), s3.reduce_add()
            )

            if num_act == 0:
                var out_v = c_sink_v.cast[out_type]()
                for b in range(B):
                    out_ptr.store(b * c_out + r0, out_v)
                r += 4
                continue

            # Step 2: Unpack only non-sink head slices of Wo and run 4x3 FMA across B tokens
            for ah in range(num_act):
                var h = act_heads[ah]
                var p_off = h * head_packed
                var s_off = h * head_blocks
                var d_off = ah * head_dim
                unpack_row_deinterleaved_prescaled(
                    Wo_packed + r0 * c_in_packed + p_off, Wo_scales + r0 * c_in_blocks + s_off, w_buf0 + d_off, head_blocks
                )
                unpack_row_deinterleaved_prescaled(
                    Wo_packed + r1 * c_in_packed + p_off, Wo_scales + r1 * c_in_blocks + s_off, w_buf1 + d_off, head_blocks
                )
                unpack_row_deinterleaved_prescaled(
                    Wo_packed + r2 * c_in_packed + p_off, Wo_scales + r2 * c_in_blocks + s_off, w_buf2 + d_off, head_blocks
                )
                unpack_row_deinterleaved_prescaled(
                    Wo_packed + r3 * c_in_packed + p_off, Wo_scales + r3 * c_in_blocks + s_off, w_buf3 + d_off, head_blocks
                )

            var b = 0
            while b + 2 < B:
                var acc00 = SIMD[DType.float32, 8](0.0)
                var acc01 = SIMD[DType.float32, 8](0.0)
                var acc02 = SIMD[DType.float32, 8](0.0)
                var acc10 = SIMD[DType.float32, 8](0.0)
                var acc11 = SIMD[DType.float32, 8](0.0)
                var acc12 = SIMD[DType.float32, 8](0.0)
                var acc20 = SIMD[DType.float32, 8](0.0)
                var acc21 = SIMD[DType.float32, 8](0.0)
                var acc22 = SIMD[DType.float32, 8](0.0)
                var acc30 = SIMD[DType.float32, 8](0.0)
                var acc31 = SIMD[DType.float32, 8](0.0)
                var acc32 = SIMD[DType.float32, 8](0.0)

                var xb0 = (b + 0) * act_dim
                var xb1 = (b + 1) * act_dim
                var xb2 = (b + 2) * act_dim
                for k in range(0, act_dim, 8):
                    var w0 = w_buf0.load[8](k)
                    var w1 = w_buf1.load[8](k)
                    var w2 = w_buf2.load[8](k)
                    var w3 = w_buf3.load[8](k)

                    var x0 = x_act_deint.load[8](xb0 + k)
                    acc00 = w0.fma(x0, acc00)
                    acc10 = w1.fma(x0, acc10)
                    acc20 = w2.fma(x0, acc20)
                    acc30 = w3.fma(x0, acc30)

                    var x1 = x_act_deint.load[8](xb1 + k)
                    acc01 = w0.fma(x1, acc01)
                    acc11 = w1.fma(x1, acc11)
                    acc21 = w2.fma(x1, acc21)
                    acc31 = w3.fma(x1, acc31)

                    var x2 = x_act_deint.load[8](xb2 + k)
                    acc02 = w0.fma(x2, acc02)
                    acc12 = w1.fma(x2, acc12)
                    acc22 = w2.fma(x2, acc22)
                    acc32 = w3.fma(x2, acc32)

                var v0 = SIMD[DType.float32, 4](
                    acc00.reduce_add(), acc10.reduce_add(), acc20.reduce_add(), acc30.reduce_add()
                ) + c_sink_v
                var v1 = SIMD[DType.float32, 4](
                    acc01.reduce_add(), acc11.reduce_add(), acc21.reduce_add(), acc31.reduce_add()
                ) + c_sink_v
                var v2 = SIMD[DType.float32, 4](
                    acc02.reduce_add(), acc12.reduce_add(), acc22.reduce_add(), acc32.reduce_add()
                ) + c_sink_v
                out_ptr.store((b + 0) * c_out + r0, v0.cast[out_type]())
                out_ptr.store((b + 1) * c_out + r0, v1.cast[out_type]())
                out_ptr.store((b + 2) * c_out + r0, v2.cast[out_type]())
                b += 3

            while b < B:
                var acc0 = SIMD[DType.float32, 8](0.0)
                var acc1 = SIMD[DType.float32, 8](0.0)
                var acc2 = SIMD[DType.float32, 8](0.0)
                var acc3 = SIMD[DType.float32, 8](0.0)
                var xb = b * act_dim
                for k in range(0, act_dim, 8):
                    var x0 = x_act_deint.load[8](xb + k)
                    acc0 = w_buf0.load[8](k).fma(x0, acc0)
                    acc1 = w_buf1.load[8](k).fma(x0, acc1)
                    acc2 = w_buf2.load[8](k).fma(x0, acc2)
                    acc3 = w_buf3.load[8](k).fma(x0, acc3)
                var vb = SIMD[DType.float32, 4](
                    acc0.reduce_add(), acc1.reduce_add(), acc2.reduce_add(), acc3.reduce_add()
                ) + c_sink_v
                out_ptr.store(b * c_out + r0, vb.cast[out_type]())
                b += 1

            r += 4

        while r < end_r:
            var r0 = r
            for sh in range(num_sink):
                var h = sink_heads[sh]
                unpack_row_deinterleaved_prescaled(
                    Wo_packed + r0 * c_in_packed + h * head_packed,
                    Wo_scales + r0 * c_in_blocks + h * head_blocks,
                    w_buf0 + sh * head_dim,
                    head_blocks,
                )
            var s0 = SIMD[DType.float32, 8](0.0)
            for k in range(0, sink_dim, 8):
                s0 = w_buf0.load[8](k).fma(v_sink_deint.load[8](k), s0)
            var c_sink0 = s0.reduce_add()

            if num_act == 0:
                for b in range(B):
                    out_ptr[b * c_out + r0] = c_sink0.cast[out_type]()
                r += 1
                continue

            for ah in range(num_act):
                var h = act_heads[ah]
                unpack_row_deinterleaved_prescaled(
                    Wo_packed + r0 * c_in_packed + h * head_packed,
                    Wo_scales + r0 * c_in_blocks + h * head_blocks,
                    w_buf0 + ah * head_dim,
                    head_blocks,
                )
            for b in range(B):
                var acc0 = SIMD[DType.float32, 8](0.0)
                var xb = b * act_dim
                for k in range(0, act_dim, 8):
                    acc0 = w_buf0.load[8](k).fma(x_act_deint.load[8](xb + k), acc0)
                out_ptr[b * c_out + r0] = (acc0.reduce_add() + c_sink0).cast[out_type]()
            r += 1

        _ = w_arr0[0]
        _ = w_arr1[0]
        _ = w_arr2[0]
        _ = w_arr3[0]

    if tasks == 1:
        worker(0)
    else:
        parallelize[worker](tasks)
    _ = x_act_stack[0]
    _ = v_sink_arr[0]
    _ = sink_arr[0]
    _ = act_arr[0]


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
    pre_deint_X: UnsafePointer[Float32, MutAnyOrigin] = UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=Int(0)),
):
    # 1. Batched pre_norm on soft_embeds: [B, embed_dim]
    for b in range(B):
        rms_norm[DType.float32, DType.float32](
            soft_embeds + b * embed_dim,
            sc_pre_norm,
            buf_normed + b * embed_dim,
            embed_dim,
        )

    var has_pre_deint = Int(pre_deint_X) != 0 and (embed_dim % 32 == 0) and (sc_size % 32 == 0)
    if has_pre_deint:
        deinterleave_batch_to_f32[DType.float32](buf_normed, pre_deint_X, B, embed_dim)

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
        pre_deint_X=pre_deint_X if has_pre_deint else UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=Int(0)),
    )

    if has_pre_deint:
        deinterleave_batch_to_f32[DType.float32](buf_gated, pre_deint_X, B, sc_size)

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
        pre_deint_X=pre_deint_X if has_pre_deint else UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=Int(0)),
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
    top2_tokens: UnsafePointer[Int, MutAnyOrigin] = UnsafePointer[Int, MutAnyOrigin](unsafe_from_address=Int(0)),
):
    var has_top2 = Int(top2_tokens) != 0

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

        # SIMD8 scan over remaining vocabulary with fast vector max rejection:
        # only inspect individual lanes when vec.reduce_max() > min_val
        var v_idx = TOP_K
        var v_vec_end = (vocab_size // 8) * 8
        while v_idx < v_vec_end:
            var vec = logits_t.load[8](v_idx)
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

        while v_idx < vocab_size:
            var val = logits_t[v_idx]
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
        if has_top2:
            top2_tokens[t] = top_idxs[1] if TOP_K > 1 else top_idxs[0]

        for k in range(TOP_K):
            var z = top_vals[k]
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

        # Smooth, continuous Zipfian continuation tail estimate from the K-th rank decay rate:
        # - When flat (diff_tail -> 0), w_flat -> 1 and eff_tail_count -> (vocab_size - TOP_K) smoothly.
        # - When peaked (diff_tail >= 0.4), w_flat -> 0 and eff_tail_count -> min(64, 1 / slope_k),
        #   immune to -30.0 <unused*> outlier explosion in the tail.
        var diff_tail = max_sc_logit - min_sc_logit
        var z_tail: Float32 = 0.0
        if diff_tail < Float32(20.0):
            var slope_k = (top_vals[TOP_K // 2] - min_sc_logit) / Float32(TOP_K - TOP_K // 2)
            if slope_k < Float32(0.05):
                slope_k = Float32(0.05)
            var base_tail = Float32(1.0) / slope_k
            if base_tail > Float32(64.0):
                base_tail = Float32(64.0)
            var w_flat = exp(-diff_tail * Float32(25.0))
            var eff_tail_count = w_flat * Float32(vocab_size - TOP_K) + (Float32(1.0) - w_flat) * base_tail
            z_tail = eff_tail_count * exp(-diff_tail)

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


@always_inline
def resolve_adjacent_duplicate_tokens(
    sorted_idx: UnsafePointer[Int, _],
    eb_mask: UnsafePointer[Bool, _],
    locked_mask: UnsafePointer[Bool, _],
    argmax_tokens: UnsafePointer[Int, MutAnyOrigin],
    new_tokens: UnsafePointer[Int, MutAnyOrigin],
    top2_tokens: UnsafePointer[Int, _],
    block_start: Int,
    block_end: Int,
):
    """In ascending entropy order `sorted_idx[0 .. cur_block_len - 1]`, checks whether an unlocked
    position `orig_idx` predicts the exact same word token (`>= 256`) as a locked or lower-entropy
    adjacent neighbor (`orig_idx - 1` or `orig_idx + 1`). When an adjacent duplicate occurs,
    substitutes the 2nd-best candidate `top2_tokens[orig_idx]` into `argmax_tokens[orig_idx]`
    and `new_tokens[orig_idx]` so neither intermediate steps nor final `argmax_canvas` decode
    can emit adjacent word stutters (e.g. 'Building Building' or 'coin coin')."""
    var cur_block_len = block_end - block_start
    var resolved_mask = InlineArray[Bool, 256](fill=False)
    for t in range(block_start, block_end):
        if locked_mask[t] or eb_mask[t]:
            resolved_mask[t] = True

    for i in range(cur_block_len):
        var orig_idx = sorted_idx[i]
        if locked_mask[orig_idx]:
            resolved_mask[orig_idx] = True
            continue
        var cand_arg = argmax_tokens[orig_idx]
        var cand_new = new_tokens[orig_idx]
        var alt_tok = top2_tokens[orig_idx]
        if alt_tok >= 0:
            if cand_arg >= 256:
                var left_resolved = (orig_idx > 0 and locked_mask[orig_idx - 1]) or (
                    orig_idx > block_start and resolved_mask[orig_idx - 1]
                )
                var right_resolved = orig_idx + 1 < block_end and resolved_mask[orig_idx + 1]
                var dup_arg = False
                if left_resolved and argmax_tokens[orig_idx - 1] == cand_arg:
                    dup_arg = True
                if right_resolved and argmax_tokens[orig_idx + 1] == cand_arg:
                    dup_arg = True
                if dup_arg and alt_tok != cand_arg:
                    var alt_clash_arg = (
                        (left_resolved and argmax_tokens[orig_idx - 1] == alt_tok)
                        or (right_resolved and argmax_tokens[orig_idx + 1] == alt_tok)
                    )
                    if not alt_clash_arg:
                        argmax_tokens[orig_idx] = alt_tok
            if cand_new >= 256:
                var left_resolved = (orig_idx > 0 and locked_mask[orig_idx - 1]) or (
                    orig_idx > block_start and resolved_mask[orig_idx - 1]
                )
                var right_resolved = orig_idx + 1 < block_end and resolved_mask[orig_idx + 1]
                var dup_new = False
                if left_resolved and new_tokens[orig_idx - 1] == cand_new:
                    dup_new = True
                if right_resolved and new_tokens[orig_idx + 1] == cand_new:
                    dup_new = True
                if dup_new and alt_tok != cand_new:
                    var alt_clash_new = (
                        (left_resolved and new_tokens[orig_idx - 1] == alt_tok)
                        or (right_resolved and new_tokens[orig_idx + 1] == alt_tok)
                    )
                    if not alt_clash_new:
                        new_tokens[orig_idx] = alt_tok
        resolved_mask[orig_idx] = True


# ==============================================================================
# 3. Fast-dLLM / dKV-Cache: Progressive Active-Canvas Compaction (llm-mojo-2tc.3)
# ==============================================================================


@always_inline
def build_active_canvas_map(
    locked_mask: UnsafePointer[Bool, _],
    active_canvas_idx: UnsafePointer[Int, MutAnyOrigin],
    canvas_len: Int,
) -> Int:
    """Builds a compact index table `active_canvas_idx[0 .. active_len - 1]` containing
    the canvas indices `t` of all unconverged (`not locked_mask[t]`) tokens."""
    var active_len = 0
    for t in range(canvas_len):
        if not locked_mask[t]:
            active_canvas_idx[active_len] = t
            active_len += 1
    return active_len


@always_inline
def update_locked_mask(
    eb_mask: UnsafePointer[Bool, _],
    history: UnsafePointer[Int, _],
    token_entropy: UnsafePointer[Float32, _],
    locked_mask: UnsafePointer[Bool, MutAnyOrigin],
    canvas_len: Int,
    hist_len: Int,
    mean_ent: Float32 = 0.15,
    freeze_entropy_thresh: Float32 = 0.02,
    max_stable_entropy: Float32 = 0.08,
    block_start: Int = 0,
    block_end: Int = -1,
) -> Int:
    """Updates `locked_mask[t]` using Fast-dLLM / dKV-Cache token freezing rules with
    anti-stutter verification:
    A token `t` in `[block_start, eff_end)` transitions to locked when `mean_ent <= 0.30`,
    `eb_mask[t] == True`, `hist_len >= 2`, `history[t] == history[canvas_len + t]`,
    `token_entropy[t] <= max_stable_entropy`, and `t` does not duplicate an adjacent token."""
    var eff_end = block_end if block_end >= 0 else canvas_len
    var num_locked = 0
    for t in range(block_start, eff_end):
        if locked_mask[t]:
            num_locked += 1
        elif mean_ent <= Float32(0.30) and hist_len >= 2 and eb_mask[t] and token_entropy[t] <= max_stable_entropy:
            var self_stable = history[t] == history[canvas_len + t]
            if not self_stable:
                continue
            var no_left_dup = (t == 0) or (
                history[t] != history[t - 1] and history[canvas_len + t] != history[canvas_len + t - 1]
            )
            var no_right_dup = (t + 1 >= eff_end) or (
                history[t] != history[t + 1] and history[canvas_len + t] != history[canvas_len + t + 1]
            )
            if no_left_dup and no_right_dup:
                locked_mask[t] = True
                num_locked += 1
    return num_locked


@always_inline
def gather_active[
    type: DType = DType.float32
](
    src_full: UnsafePointer[Scalar[type], _],
    active_canvas_idx: UnsafePointer[Int, _],
    dst_compact: UnsafePointer[Scalar[type], MutAnyOrigin],
    active_len: Int,
    dim: Int,
):
    """Gathers `[canvas_len, dim]` rows at `active_canvas_idx[0..active_len-1]`
    into contiguous `[active_len, dim]` buffer `dst_compact`."""
    for b in range(active_len):
        var orig_t = active_canvas_idx[b]
        var s_ptr = src_full + orig_t * dim
        var d_ptr = dst_compact + b * dim
        for i in range(0, dim, 8):
            d_ptr.store[8](i, s_ptr.load[8](i))


@always_inline
def scatter_active_sampler_outputs[
    sc_type: DType = DType.float32
](
    active_canvas_idx: UnsafePointer[Int, _],
    compact_entropy: UnsafePointer[Float32, _],
    compact_argmax: UnsafePointer[Int, _],
    compact_new_tokens: UnsafePointer[Int, _],
    compact_sc_embeds: UnsafePointer[Scalar[sc_type], _],
    token_entropy: UnsafePointer[Float32, MutAnyOrigin],
    argmax_tokens: UnsafePointer[Int, MutAnyOrigin],
    new_tokens: UnsafePointer[Int, MutAnyOrigin],
    sc_embeds: UnsafePointer[Scalar[sc_type], MutAnyOrigin],
    locked_mask: UnsafePointer[Bool, _],
    canvas_tokens: UnsafePointer[Int, _],
    active_len: Int,
    canvas_len: Int,
    embed_dim: Int,
    compact_top2: UnsafePointer[Int, _] = UnsafePointer[Int, MutAnyOrigin](unsafe_from_address=Int(0)),
    top2_tokens: UnsafePointer[Int, MutAnyOrigin] = UnsafePointer[Int, MutAnyOrigin](unsafe_from_address=Int(0)),
):
    """Scatters sampler outputs for the `active_len` unconverged tokens back into their
    full canvas slots `orig_t = active_canvas_idx[b]`, while smoothly cooling locked tokens'
    entropy (`*= 0.65`) and syncing any AdaCache-skipped unlocked tokens' `new_tokens[t]`
    with their current `canvas_tokens[t]`."""
    var has_top2 = Int(compact_top2) != 0 and Int(top2_tokens) != 0
    for t in range(canvas_len):
        if locked_mask[t]:
            token_entropy[t] = token_entropy[t] * Float32(0.65)
            argmax_tokens[t] = canvas_tokens[t]
            new_tokens[t] = canvas_tokens[t]
        else:
            new_tokens[t] = canvas_tokens[t]

    for b in range(active_len):
        var orig_t = active_canvas_idx[b]
        token_entropy[orig_t] = compact_entropy[b]
        argmax_tokens[orig_t] = compact_argmax[b]
        new_tokens[orig_t] = compact_new_tokens[b]
        if has_top2:
            top2_tokens[orig_t] = compact_top2[b]
        var s_ptr = compact_sc_embeds + b * embed_dim
        var d_ptr = sc_embeds + orig_t * embed_dim
        for i in range(0, embed_dim, 8):
            d_ptr.store[8](i, s_ptr.load[8](i))


# ==============================================================================
# 4. TeaCache (CVPR 2025) & TaylorSeer (ICCV 2025) Feature Forecasting (2tc.4/7)
# ==============================================================================


@always_inline
def teacache_eval_gate(
    h_curr: UnsafePointer[Float16, _],
    h_prev_full: UnsafePointer[Float16, MutAnyOrigin],
    active_canvas_idx: UnsafePointer[Int, _],
    active_len: Int,
    embed_dim: Int,
    temp: Float32,
    full_steps_count: Int,
    consec_skips: Int,
    accum_fluctuation: UnsafePointer[Float32, MutAnyOrigin],
    teacache_thresh: Float32 = 0.20,
    max_consec_skips: Int = 2,
) -> Bool:
    """Evaluates the TeaCache (CVPR 2025) polynomial timestep-modulated relative L1 gate
    at the probe layer (layer 5).
    Returns `True` if middle layers (6..22) should be skipped and forecasted via TaylorSeer,
    or `False` if all 30 layers should be evaluated (updating `h_prev_full` and resetting
    `accum_fluctuation[0] = 0.0`). Requires `full_steps_count >= 2` so both self-conditioning
    and 1st-order Taylor velocity `d1_D` are initialized before forecasting."""
    if full_steps_count < 2:
        for b in range(active_len):
            var orig_t = active_canvas_idx[b]
            var s_ptr = h_curr + b * embed_dim
            var d_ptr = h_prev_full + orig_t * embed_dim
            for i in range(0, embed_dim, 8):
                d_ptr.store[8](i, s_ptr.load[8](i))
        accum_fluctuation[0] = 0.0
        return False

    # Compute unpooled per-token normalized L2 fluctuation sum_b ||c_b - p_b||_2^2 / sum_b (||c_b||_2^2 + ||p_b||_2^2)
    # so any re-noised token immediately prevents premature mid-layer skipping.
    var diff_sq_vec = SIMD[DType.float32, 8](0.0)
    var sum_sq_vec = SIMD[DType.float32, 8](0.0)
    for b in range(active_len):
        var orig_t = active_canvas_idx[b]
        var c_ptr = h_curr + b * embed_dim
        var p_ptr = h_prev_full + orig_t * embed_dim
        for i in range(0, embed_dim, 8):
            var c_v = c_ptr.load[8](i).cast[DType.float32]()
            var p_v = p_ptr.load[8](i).cast[DType.float32]()
            var d_v = c_v - p_v
            diff_sq_vec = d_v.fma(d_v, diff_sq_vec)
            sum_sq_vec = c_v.fma(c_v, p_v.fma(p_v, sum_sq_vec))

    var raw_l1 = diff_sq_vec.reduce_add() / (sum_sq_vec.reduce_add() + Float32(1e-6))

    # Normalized trajectory progress u in [0, 1] for temp in [0.8 -> 0.4]
    var u = (Float32(0.8) - temp) * Float32(2.5)
    if u < 0.0:
        u = 0.0
    if u > 1.0:
        u = 1.0
    var du = u - Float32(0.5)
    # Polynomial timestep modulation: gentler accumulation during coarse high-noise phase (u -> 0),
    # stricter sensitivity during final token lock-in (u -> 1)
    var poly_mod = Float32(0.45) + Float32(0.30) * du * du + Float32(0.25) * u
    var new_accum = accum_fluctuation[0] + poly_mod * raw_l1

    if new_accum < teacache_thresh and consec_skips < max_consec_skips:
        accum_fluctuation[0] = new_accum
        return True

    # Full evaluation triggered: update reference probe state and reset accumulator
    for b in range(active_len):
        var orig_t = active_canvas_idx[b]
        var s_ptr = h_curr + b * embed_dim
        var d_ptr = h_prev_full + orig_t * embed_dim
        for i in range(0, embed_dim, 8):
            d_ptr.store[8](i, s_ptr.load[8](i))
    accum_fluctuation[0] = 0.0
    return False


@always_inline
def taylorseer_record_full_step(
    h_start: UnsafePointer[Float16, _],
    h_end: UnsafePointer[Float16, _],
    active_canvas_idx: UnsafePointer[Int, _],
    taylor_D: UnsafePointer[Float32, MutAnyOrigin],
    taylor_d1: UnsafePointer[Float32, MutAnyOrigin],
    taylor_d2: UnsafePointer[Float32, MutAnyOrigin],
    active_len: Int,
    embed_dim: Int,
    full_steps_count: Int,
):
    """Records the middle-block residual `D_k = h_end - h_start` and updates backward
    finite differences `d1_D = D_k - D_{k-1}` and `d2_D = d1_D - d1_{k-1}` in Float32
    (eliminating FP16 finite-difference cancellation) for all active canvas tokens."""
    var zero_f32 = SIMD[DType.float32, 8](0.0)
    for b in range(active_len):
        var orig_t = active_canvas_idx[b]
        var s_ptr = h_start + b * embed_dim
        var e_ptr = h_end + b * embed_dim
        var d_ptr = taylor_D + orig_t * embed_dim
        var d1_ptr = taylor_d1 + orig_t * embed_dim
        var d2_ptr = taylor_d2 + orig_t * embed_dim

        if full_steps_count == 0:
            for i in range(0, embed_dim, 8):
                var d_new = e_ptr.load[8](i).cast[DType.float32]() - s_ptr.load[8](i).cast[DType.float32]()
                d_ptr.store[8](i, d_new)
                d1_ptr.store[8](i, zero_f32)
                d2_ptr.store[8](i, zero_f32)
        elif full_steps_count == 1:
            for i in range(0, embed_dim, 8):
                var d_new = e_ptr.load[8](i).cast[DType.float32]() - s_ptr.load[8](i).cast[DType.float32]()
                var d_old = d_ptr.load[8](i)
                var d1_new = d_new - d_old
                d_ptr.store[8](i, d_new)
                d1_ptr.store[8](i, d1_new)
                d2_ptr.store[8](i, zero_f32)
        else:
            for i in range(0, embed_dim, 8):
                var d_new = e_ptr.load[8](i).cast[DType.float32]() - s_ptr.load[8](i).cast[DType.float32]()
                var d_old = d_ptr.load[8](i)
                var d1_old = d1_ptr.load[8](i)
                var d1_new = d_new - d_old
                var d2_new = d1_new - d1_old
                d_ptr.store[8](i, d_new)
                d1_ptr.store[8](i, d1_new)
                d2_ptr.store[8](i, d2_new)


@always_inline
def taylorseer_forecast_skipped_step(
    canvas_hidden: UnsafePointer[Float16, MutAnyOrigin],
    active_canvas_idx: UnsafePointer[Int, _],
    taylor_D: UnsafePointer[Float32, _],
    taylor_d1: UnsafePointer[Float32, _],
    taylor_d2: UnsafePointer[Float32, _],
    active_len: Int,
    embed_dim: Int,
    full_steps_count: Int,
    step_offset_m: Int,
    max_vel_ratio: Float32 = 0.65,
    max_acc_ratio: Float32 = 0.85,
):
    """Forecasts the middle-block output `h^(22)_{k+m} = h^(5)_{k+m} + D_hat(k+m)` using
    Newton's backward finite-difference Taylor expansion in Float32:
      `D_hat(k + m) = D_k + m * d1_D + 0.5 * m * (m + 1) * d2_D`
    with per-token confidence-gated curvature damping to prevent overshoot on discontinuous tokens."""
    var m_f32 = Float32(step_offset_m)
    var max_vel_sq = max_vel_ratio * max_vel_ratio
    var max_acc_sq = max_acc_ratio * max_acc_ratio

    for b in range(active_len):
        var orig_t = active_canvas_idx[b]
        var h_ptr = canvas_hidden + b * embed_dim
        var d_ptr = taylor_D + orig_t * embed_dim
        var d1_ptr = taylor_d1 + orig_t * embed_dim
        var d2_ptr = taylor_d2 + orig_t * embed_dim

        var c1: Float32 = 0.0
        var c2: Float32 = 0.0

        if full_steps_count >= 2:
            var norm_d_sq = SIMD[DType.float32, 8](0.0)
            var norm_d1_sq = SIMD[DType.float32, 8](0.0)
            var norm_d2_sq = SIMD[DType.float32, 8](0.0)
            for i in range(0, embed_dim, 8):
                var dv = d_ptr.load[8](i)
                var d1v = d1_ptr.load[8](i)
                var d2v = d2_ptr.load[8](i)
                norm_d_sq = dv.fma(dv, norm_d_sq)
                norm_d1_sq = d1v.fma(d1v, norm_d1_sq)
                norm_d2_sq = d2v.fma(d2v, norm_d2_sq)

            var d_sq = norm_d_sq.reduce_add()
            var d1_sq = norm_d1_sq.reduce_add()
            var d2_sq = norm_d2_sq.reduce_add()

            if d1_sq <= max_vel_sq * (d_sq + Float32(1e-6)):
                c1 = m_f32
                if full_steps_count >= 3 and d2_sq <= max_acc_sq * (d1_sq + Float32(1e-6)):
                    c2 = Float32(0.5) * m_f32 * (m_f32 + Float32(1.0))

        var c1_vec = SIMD[DType.float32, 8](c1)
        var c2_vec = SIMD[DType.float32, 8](c2)
        var f16_max = SIMD[DType.float32, 8](65500.0)
        var f16_min = SIMD[DType.float32, 8](-65500.0)
        for i in range(0, embed_dim, 8):
            var hv = h_ptr.load[8](i).cast[DType.float32]()
            var dv = d_ptr.load[8](i)
            var d1v = d1_ptr.load[8](i)
            var d2v = d2_ptr.load[8](i)
            var forecast = hv + dv + c1_vec * d1v + c2_vec * d2v
            forecast = forecast.clamp(f16_min, f16_max)
            h_ptr.store[8](i, forecast.cast[DType.float16]())


@always_inline
def taylorseer_forecast_skipped_step(
    canvas_hidden: UnsafePointer[Float16, MutAnyOrigin],
    h_prev_full: UnsafePointer[Float16, _],
    active_canvas_idx: UnsafePointer[Int, _],
    taylor_D: UnsafePointer[Float32, _],
    taylor_d1: UnsafePointer[Float32, _],
    taylor_d2: UnsafePointer[Float32, _],
    active_len: Int,
    embed_dim: Int,
    full_steps_count: Int,
    step_offset_m: Int,
    max_vel_ratio: Float32 = 0.65,
    max_acc_ratio: Float32 = 0.85,
):
    """Scale-aware TaylorSeer forecast that trust-region bounds the early-layer re-noising
    perturbation `h^(5)_{k+m} - h^(5)_k` relative to the target layer-22 norm `||h^(22)_k||_2`."""
    var m_f32 = Float32(step_offset_m)
    var max_vel_sq = max_vel_ratio * max_vel_ratio
    var max_acc_sq = max_acc_ratio * max_acc_ratio
    var f16_max = SIMD[DType.float32, 8](65500.0)
    var f16_min = SIMD[DType.float32, 8](-65500.0)

    for b in range(active_len):
        var orig_t = active_canvas_idx[b]
        var h_ptr = canvas_hidden + b * embed_dim
        var hp_ptr = h_prev_full + orig_t * embed_dim
        var d_ptr = taylor_D + orig_t * embed_dim
        var d1_ptr = taylor_d1 + orig_t * embed_dim
        var d2_ptr = taylor_d2 + orig_t * embed_dim

        var norm_h22_sq = SIMD[DType.float32, 8](0.0)
        var norm_dh5_sq = SIMD[DType.float32, 8](0.0)
        var norm_d_sq = SIMD[DType.float32, 8](0.0)
        var norm_d1_sq = SIMD[DType.float32, 8](0.0)
        var norm_d2_sq = SIMD[DType.float32, 8](0.0)

        for i in range(0, embed_dim, 8):
            var hv = h_ptr.load[8](i).cast[DType.float32]()
            var hpv = hp_ptr.load[8](i).cast[DType.float32]()
            var dv = d_ptr.load[8](i)
            var h22_prev = hpv + dv
            var dh5 = hv - hpv
            norm_h22_sq = h22_prev.fma(h22_prev, norm_h22_sq)
            norm_dh5_sq = dh5.fma(dh5, norm_dh5_sq)
            if full_steps_count >= 2:
                var d1v = d1_ptr.load[8](i)
                var d2v = d2_ptr.load[8](i)
                norm_d_sq = dv.fma(dv, norm_d_sq)
                norm_d1_sq = d1v.fma(d1v, norm_d1_sq)
                norm_d2_sq = d2v.fma(d2v, norm_d2_sq)

        var h22_norm = sqrt(norm_h22_sq.reduce_add() + Float32(1e-6))
        var dh5_norm = sqrt(norm_dh5_sq.reduce_add() + Float32(1e-6))
        var gamma: Float32 = 1.0
        if dh5_norm > Float32(0.15) * h22_norm:
            gamma = (Float32(0.15) * h22_norm) / dh5_norm

        var c1: Float32 = 0.0
        var c2: Float32 = 0.0
        if full_steps_count >= 2:
            var d_sq = norm_d_sq.reduce_add()
            var d1_sq = norm_d1_sq.reduce_add()
            var d2_sq = norm_d2_sq.reduce_add()
            if d1_sq <= max_vel_sq * (d_sq + Float32(1e-6)):
                c1 = m_f32
                if full_steps_count >= 3 and d2_sq <= max_acc_sq * (d1_sq + Float32(1e-6)):
                    c2 = Float32(0.5) * m_f32 * (m_f32 + Float32(1.0))

        var gamma_vec = SIMD[DType.float32, 8](gamma)
        var c1_vec = SIMD[DType.float32, 8](c1)
        var c2_vec = SIMD[DType.float32, 8](c2)
        for i in range(0, embed_dim, 8):
            var hv = h_ptr.load[8](i).cast[DType.float32]()
            var hpv = hp_ptr.load[8](i).cast[DType.float32]()
            var dv = d_ptr.load[8](i)
            var d1v = d1_ptr.load[8](i)
            var d2v = d2_ptr.load[8](i)
            var forecast = (hpv + dv) + gamma_vec * (hv - hpv) + c1_vec * d1v + c2_vec * d2v
            forecast = forecast.clamp(f16_min, f16_max)
            h_ptr.store[8](i, forecast.cast[DType.float16]())


# ==============================================================================
# 5. Noise-Scheduled MoE LOD & Subbatch Set-Cover (llm-mojo-2tc.5)
# ==============================================================================


@always_inline
def compute_moe_lod_k(denoise_step: Int, mean_ent: Float32) -> Int:
    """Returns active Top-K expert budget (full Top-8 across all denoising steps to
    preserve 26B MoE expert mixture fidelity)."""
    return 8


@always_inline
def compute_moe_lod_top_p(denoise_step: Int, mean_ent: Float32) -> Float32:
    """Returns nucleus cumulative probability threshold `moe_top_p` for MoE routing."""
    return Float32(1.0)


@always_inline
def apply_moe_lod_and_set_cover(
    canvas_top_idx: UnsafePointer[Int, _],
    canvas_top_val: UnsafePointer[Float32, _],
    router_per_expert_scale: UnsafePointer[Float32, _],
    canvas_top_weights: UnsafePointer[Float32, MutAnyOrigin],
    B: Int,
    top_k_experts: Int,
    active_top_k: Int,
    moe_top_p: Float32,
    eb_mask: UnsafePointer[Bool, _] = UnsafePointer[Bool, MutAnyOrigin](unsafe_from_address=Int(0)),
    active_canvas_idx: UnsafePointer[Int, _] = UnsafePointer[Int, MutAnyOrigin](unsafe_from_address=Int(0)),
    unaccepted_top_k: Int = 8,
):
    """Computes Gemma-4 / DiffusionGemma router expert weights for the active canvas tokens:
    - When `eb_mask` is provided (llm-mojo-yj4.3), already-accepted tokens (`eb_mask[orig_t] == True`)
      retain their full `active_top_k` (Top-8) budget with unmodified `norm_p >= 0.05` weights, while
      unaccepted tokens (`eb_mask[orig_t] == False`, whose input was just replaced by random noise)
      use `unaccepted_top_k` (Top-2) energy-conserved routing.
    - Applied before expert bucket grouping so the inner expert GEMM has zero added branches."""
    var eff_k = active_top_k if active_top_k <= top_k_experts else top_k_experts
    if eff_k < 1:
        eff_k = 1
    var unacc_k = unaccepted_top_k if unaccepted_top_k <= eff_k else eff_k
    if unacc_k < 1:
        unacc_k = 1
    var has_eb = Int(eb_mask) != 0
    var has_map = Int(active_canvas_idx) != 0

    for b in range(B):
        var orig_t = active_canvas_idx[b] if has_map else b
        var is_accepted = eb_mask[orig_t] if has_eb else True
        var tok_k = eff_k if is_accepted else unacc_k

        var top_i = canvas_top_idx + b * top_k_experts
        var top_v = canvas_top_val + b * top_k_experts
        var top_w = canvas_top_weights + b * top_k_experts

        var sum_denom: Float32 = 0.0
        var denom_slots = tok_k
        for s in range(denom_slots):
            if top_i[s] >= 0:
                sum_denom += top_v[s]
        var inv_sum = Float32(1.0) / (sum_denom + Float32(1e-10))

        for s in range(top_k_experts):
            var idx = top_i[s]
            var norm_p = top_v[s] * inv_sum
            if s < tok_k and idx >= 0 and idx < 128 and norm_p >= Float32(0.05):
                top_w[s] = norm_p * router_per_expert_scale[idx]
            else:
                top_w[s] = 0.0


# ==============================================================================
# 6. Flash-dLLM + dLLM-KVQ: Fused Parallel QKV Prep, Hoisted RoPE & Head-Group
#    Tiled FP16 Bidirectional Attention (llm-mojo-2tc.9)
# ==============================================================================


@always_inline
def _softmax_inline(scores: UnsafePointer[Float32, MutAnyOrigin], size: Int):
    var max_val = scores[0]
    for i in range(1, size):
        if scores[i] > max_val:
            max_val = scores[i]
    var sum_exp: Float32 = 0.0
    for i in range(size):
        var e = exp(scores[i] - max_val)
        scores[i] = e
        sum_exp += e
    var inv_sum = Float32(1.0) / (sum_exp + Float32(1e-10))
    for i in range(size):
        scores[i] = scores[i] * inv_sum


@always_inline
def prep_qkv_norm_rope_kvq_batch(
    canvas_q: UnsafePointer[Float16, MutAnyOrigin],
    canvas_k: UnsafePointer[Float16, MutAnyOrigin],
    canvas_v: UnsafePointer[Float16, MutAnyOrigin],
    q_norm_weight: UnsafePointer[Float32, _],
    k_norm_weight: UnsafePointer[Float32, _],
    active_canvas_idx: UnsafePointer[Int, _],
    k_cache: UnsafePointer[Float16, MutAnyOrigin],
    v_cache: UnsafePointer[Float16, MutAnyOrigin],
    active_len: Int,
    num_prompt_tokens: Int,
    layer_heads: Int,
    layer_kv_heads: Int,
    layer_head_dim: Int,
    kv_stride: Int,
    layer_cache_offset: Int,
    max_q_size: Int,
    max_kv_size: Int,
    is_global: Bool,
):
    """Parallelized Q/K/V head RMSNorm + Hoisted SIMD8 RoPE (computing cos/sin once
    per token `b` instead of `layer_heads + layer_kv_heads` times, and rotating only
    the active 64 pairs on global layers) + direct FP16 `dLLM-KVQ` cache scatter."""
    var half_dim = layer_head_dim // 2
    var rot_half = 64 if is_global else half_dim
    var rot_vecs = rot_half // 8
    var base: Float32 = Float32(1000000.0) if is_global else Float32(10000.0)
    var inv_rope_dim = Float32(1.0) / Float32(512 if is_global else 256)

    def prep_worker(b: Int) capturing -> None:
        var orig_t = active_canvas_idx[b]
        var pos = num_prompt_tokens + orig_t
        var pos_f = Float32(pos)
        var q_store = canvas_q + b * max_q_size
        var k_store = canvas_k + b * max_kv_size
        var v_store = canvas_v + b * max_kv_size

        for h in range(layer_heads):
            rms_norm[in_type=DType.float16, out_type=DType.float16](
                q_store + h * layer_head_dim, q_norm_weight, q_store + h * layer_head_dim, layer_head_dim
            )
        for h in range(layer_kv_heads):
            rms_norm[in_type=DType.float16, out_type=DType.float16](
                k_store + h * layer_head_dim, k_norm_weight, k_store + h * layer_head_dim, layer_head_dim
            )
        for h in range(layer_kv_heads):
            rms_norm_noscale[in_type=DType.float16, out_type=DType.float16](
                v_store + h * layer_head_dim, v_store + h * layer_head_dim, layer_head_dim
            )

        # Precompute cos/sin SIMD8 tables once for token position `pos`
        var cos_vecs = InlineArray[SIMD[DType.float32, 8], 16](uninitialized=True)
        var sin_vecs = InlineArray[SIMD[DType.float32, 8], 16](uninitialized=True)
        for v_idx in range(rot_vecs):
            var base_i = v_idx * 8
            var c_vec = SIMD[DType.float32, 8](0.0)
            var s_vec = SIMD[DType.float32, 8](0.0)
            comptime for lane in range(8):
                var i = base_i + lane
                var theta = pos_f / (base ** (Float32(2 * i) * inv_rope_dim))
                c_vec[lane] = cos(theta)
                s_vec[lane] = sin(theta)
            cos_vecs[v_idx] = c_vec
            sin_vecs[v_idx] = s_vec

        # Apply SIMD8 RoPE across all Q heads and K heads using precomputed cos_vecs / sin_vecs
        for h in range(layer_heads):
            var h_ptr = q_store + h * layer_head_dim
            for v_idx in range(rot_vecs):
                var off = v_idx * 8
                var c_v = cos_vecs[v_idx]
                var s_v = sin_vecs[v_idx]
                var v1 = h_ptr.load[8](off).cast[DType.float32]()
                var v2 = h_ptr.load[8](half_dim + off).cast[DType.float32]()
                h_ptr.store[8](off, (v1 * c_v - v2 * s_v).cast[DType.float16]())
                h_ptr.store[8](half_dim + off, (v2 * c_v + v1 * s_v).cast[DType.float16]())

        for h in range(layer_kv_heads):
            var h_ptr = k_store + h * layer_head_dim
            for v_idx in range(rot_vecs):
                var off = v_idx * 8
                var c_v = cos_vecs[v_idx]
                var s_v = sin_vecs[v_idx]
                var v1 = h_ptr.load[8](off).cast[DType.float32]()
                var v2 = h_ptr.load[8](half_dim + off).cast[DType.float32]()
                h_ptr.store[8](off, (v1 * c_v - v2 * s_v).cast[DType.float16]())
                h_ptr.store[8](half_dim + off, (v2 * c_v + v1 * s_v).cast[DType.float16]())

        # Scatter FP16 K and V directly into dLLM-KVQ cache
        var step_cache_offset = layer_cache_offset + pos * kv_stride
        for i in range(0, kv_stride, 16):
            k_cache.store[16](step_cache_offset + i, k_store.load[16](i))
            v_cache.store[16](step_cache_offset + i, v_store.load[16](i))

    parallelize[prep_worker](active_len)


@always_inline
def flash_dllm_bidirectional_attn(
    canvas_q: UnsafePointer[Float16, _],
    k_cache: UnsafePointer[Float16, _],
    v_cache: UnsafePointer[Float16, _],
    canvas_attn_out: UnsafePointer[Float16, MutAnyOrigin],
    active_len: Int,
    seq_len: Int,
    layer_heads: Int,
    layer_kv_heads: Int,
    layer_head_dim: Int,
    kv_stride: Int,
    layer_cache_offset: Int,
    max_q_size: Int,
):
    """Flash-dLLM Head-Group & Query-Tile Fused Bidirectional Attention over FP16 `dLLM-KVQ`:
      - On sliding/local layers (`ratio == 2`, `head_dim == 256`), pairs 2 query tokens `(b0, b1)`
        with the 2 query heads `(2g, 2g+1)` sharing KV head `g` to evaluate **4 queries per KV load**.
      - On global layers (`ratio == 8`, `head_dim == 512`), groups 4 query heads `(8g + r .. 8g + r + 3)`
        sharing KV head `g` to evaluate **4 queries per KV load**.
      - Cuts KV cache memory traffic by **8x** (2x from FP16 `dLLM-KVQ` + 4x from 4-query fusion)."""
    var ratio = layer_heads // layer_kv_heads
    var num_simd_lanes = layer_head_dim // 8

    if ratio == 2 and num_simd_lanes == 32:
        var num_pairs = (active_len + 1) // 2

        def sliding_pair_worker(pair_idx: Int) capturing -> None:
            var b0 = pair_idx * 2
            var b1 = b0 + 1
            var q0_base = canvas_q + b0 * max_q_size
            var out0_base = canvas_attn_out + b0 * max_q_size

            var q00 = InlineArray[SIMD[DType.float32, 8], 32](uninitialized=True)
            var q01 = InlineArray[SIMD[DType.float32, 8], 32](uninitialized=True)
            var acc00 = InlineArray[SIMD[DType.float32, 8], 32](uninitialized=True)
            var acc01 = InlineArray[SIMD[DType.float32, 8], 32](uninitialized=True)
            var scores00 = InlineArray[Float32, 512](uninitialized=True)
            var scores01 = InlineArray[Float32, 512](uninitialized=True)
            var s00_ptr = UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=Int( UnsafePointer(to=scores00)))
            var s01_ptr = UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=Int( UnsafePointer(to=scores01)))

            if b1 < active_len:
                var q1_base = canvas_q + b1 * max_q_size
                var out1_base = canvas_attn_out + b1 * max_q_size
                var q10 = InlineArray[SIMD[DType.float32, 8], 32](uninitialized=True)
                var q11 = InlineArray[SIMD[DType.float32, 8], 32](uninitialized=True)
                var acc10 = InlineArray[SIMD[DType.float32, 8], 32](uninitialized=True)
                var acc11 = InlineArray[SIMD[DType.float32, 8], 32](uninitialized=True)
                var scores10 = InlineArray[Float32, 512](uninitialized=True)
                var scores11 = InlineArray[Float32, 512](uninitialized=True)
                var s10_ptr = UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=Int( UnsafePointer(to=scores10)))
                var s11_ptr = UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=Int( UnsafePointer(to=scores11)))

                for g in range(layer_kv_heads):
                    var h0_off = (g * 2) * 256
                    var h1_off = (g * 2 + 1) * 256
                    var g_offset = g * 256

                    for d in range(32):
                        q00[d] = q0_base.load[8](h0_off + d * 8).cast[DType.float32]()
                        q01[d] = q0_base.load[8](h1_off + d * 8).cast[DType.float32]()
                        q10[d] = q1_base.load[8](h0_off + d * 8).cast[DType.float32]()
                        q11[d] = q1_base.load[8](h1_off + d * 8).cast[DType.float32]()

                    for tt in range(seq_len):
                        var t_cache_offset = layer_cache_offset + tt * kv_stride + g_offset
                        var v00 = SIMD[DType.float32, 8](0.0)
                        var v01 = SIMD[DType.float32, 8](0.0)
                        var v10 = SIMD[DType.float32, 8](0.0)
                        var v11 = SIMD[DType.float32, 8](0.0)
                        for d in range(32):
                            var k_vec = k_cache.load[8](t_cache_offset + d * 8).cast[DType.float32]()
                            v00 = q00[d].fma(k_vec, v00)
                            v01 = q01[d].fma(k_vec, v01)
                            v10 = q10[d].fma(k_vec, v10)
                            v11 = q11[d].fma(k_vec, v11)
                        s00_ptr[tt] = v00.reduce_add()
                        s01_ptr[tt] = v01.reduce_add()
                        s10_ptr[tt] = v10.reduce_add()
                        s11_ptr[tt] = v11.reduce_add()

                    _softmax_inline(s00_ptr, seq_len)
                    _softmax_inline(s01_ptr, seq_len)
                    _softmax_inline(s10_ptr, seq_len)
                    _softmax_inline(s11_ptr, seq_len)

                    for d in range(32):
                        acc00[d] = SIMD[DType.float32, 8](0.0)
                        acc01[d] = SIMD[DType.float32, 8](0.0)
                        acc10[d] = SIMD[DType.float32, 8](0.0)
                        acc11[d] = SIMD[DType.float32, 8](0.0)

                    for tt in range(seq_len):
                        var t_cache_offset = layer_cache_offset + tt * kv_stride + g_offset
                        var w00 = SIMD[DType.float32, 8](s00_ptr[tt])
                        var w01 = SIMD[DType.float32, 8](s01_ptr[tt])
                        var w10 = SIMD[DType.float32, 8](s10_ptr[tt])
                        var w11 = SIMD[DType.float32, 8](s11_ptr[tt])
                        for d in range(32):
                            var v_vec = v_cache.load[8](t_cache_offset + d * 8).cast[DType.float32]()
                            acc00[d] = w00.fma(v_vec, acc00[d])
                            acc01[d] = w01.fma(v_vec, acc01[d])
                            acc10[d] = w10.fma(v_vec, acc10[d])
                            acc11[d] = w11.fma(v_vec, acc11[d])

                    for d in range(32):
                        out0_base.store[8](h0_off + d * 8, acc00[d].cast[DType.float16]())
                        out0_base.store[8](h1_off + d * 8, acc01[d].cast[DType.float16]())
                        out1_base.store[8](h0_off + d * 8, acc10[d].cast[DType.float16]())
                        out1_base.store[8](h1_off + d * 8, acc11[d].cast[DType.float16]())
            else:
                for g in range(layer_kv_heads):
                    var h0_off = (g * 2) * 256
                    var h1_off = (g * 2 + 1) * 256
                    var g_offset = g * 256

                    for d in range(32):
                        q00[d] = q0_base.load[8](h0_off + d * 8).cast[DType.float32]()
                        q01[d] = q0_base.load[8](h1_off + d * 8).cast[DType.float32]()

                    for tt in range(seq_len):
                        var t_cache_offset = layer_cache_offset + tt * kv_stride + g_offset
                        var v00 = SIMD[DType.float32, 8](0.0)
                        var v01 = SIMD[DType.float32, 8](0.0)
                        for d in range(32):
                            var k_vec = k_cache.load[8](t_cache_offset + d * 8).cast[DType.float32]()
                            v00 = q00[d].fma(k_vec, v00)
                            v01 = q01[d].fma(k_vec, v01)
                        s00_ptr[tt] = v00.reduce_add()
                        s01_ptr[tt] = v01.reduce_add()

                    _softmax_inline(s00_ptr, seq_len)
                    _softmax_inline(s01_ptr, seq_len)

                    for d in range(32):
                        acc00[d] = SIMD[DType.float32, 8](0.0)
                        acc01[d] = SIMD[DType.float32, 8](0.0)

                    for tt in range(seq_len):
                        var t_cache_offset = layer_cache_offset + tt * kv_stride + g_offset
                        var w00 = SIMD[DType.float32, 8](s00_ptr[tt])
                        var w01 = SIMD[DType.float32, 8](s01_ptr[tt])
                        for d in range(32):
                            var v_vec = v_cache.load[8](t_cache_offset + d * 8).cast[DType.float32]()
                            acc00[d] = w00.fma(v_vec, acc00[d])
                            acc01[d] = w01.fma(v_vec, acc01[d])

                    for d in range(32):
                        out0_base.store[8](h0_off + d * 8, acc00[d].cast[DType.float16]())
                        out0_base.store[8](h1_off + d * 8, acc01[d].cast[DType.float16]())

        parallelize[sliding_pair_worker](num_pairs)

    elif ratio == 8 and num_simd_lanes == 64:
        def global_token_worker(b: Int) capturing -> None:
            var q_base = canvas_q + b * max_q_size
            var out_base = canvas_attn_out + b * max_q_size

            var q0 = InlineArray[SIMD[DType.float32, 8], 64](uninitialized=True)
            var q1 = InlineArray[SIMD[DType.float32, 8], 64](uninitialized=True)
            var q2 = InlineArray[SIMD[DType.float32, 8], 64](uninitialized=True)
            var q3 = InlineArray[SIMD[DType.float32, 8], 64](uninitialized=True)
            var acc0 = InlineArray[SIMD[DType.float32, 8], 64](uninitialized=True)
            var acc1 = InlineArray[SIMD[DType.float32, 8], 64](uninitialized=True)
            var acc2 = InlineArray[SIMD[DType.float32, 8], 64](uninitialized=True)
            var acc3 = InlineArray[SIMD[DType.float32, 8], 64](uninitialized=True)
            var scores0 = InlineArray[Float32, 512](uninitialized=True)
            var scores1 = InlineArray[Float32, 512](uninitialized=True)
            var scores2 = InlineArray[Float32, 512](uninitialized=True)
            var scores3 = InlineArray[Float32, 512](uninitialized=True)
            var s0_ptr = UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=Int(UnsafePointer(to=scores0)))
            var s1_ptr = UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=Int(UnsafePointer(to=scores1)))
            var s2_ptr = UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=Int(UnsafePointer(to=scores2)))
            var s3_ptr = UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=Int(UnsafePointer(to=scores3)))

            for g in range(layer_kv_heads):
                var g_offset = g * 512
                for sub in range(2):
                    var h_base = g * 8 + sub * 4
                    var h0_off = (h_base + 0) * 512
                    var h1_off = (h_base + 1) * 512
                    var h2_off = (h_base + 2) * 512
                    var h3_off = (h_base + 3) * 512

                    for d in range(64):
                        q0[d] = q_base.load[8](h0_off + d * 8).cast[DType.float32]()
                        q1[d] = q_base.load[8](h1_off + d * 8).cast[DType.float32]()
                        q2[d] = q_base.load[8](h2_off + d * 8).cast[DType.float32]()
                        q3[d] = q_base.load[8](h3_off + d * 8).cast[DType.float32]()

                    for tt in range(seq_len):
                        var t_cache_offset = layer_cache_offset + tt * kv_stride + g_offset
                        var v0 = SIMD[DType.float32, 8](0.0)
                        var v1 = SIMD[DType.float32, 8](0.0)
                        var v2 = SIMD[DType.float32, 8](0.0)
                        var v3 = SIMD[DType.float32, 8](0.0)
                        for d in range(64):
                            var k_vec = k_cache.load[8](t_cache_offset + d * 8).cast[DType.float32]()
                            v0 = q0[d].fma(k_vec, v0)
                            v1 = q1[d].fma(k_vec, v1)
                            v2 = q2[d].fma(k_vec, v2)
                            v3 = q3[d].fma(k_vec, v3)
                        s0_ptr[tt] = v0.reduce_add()
                        s1_ptr[tt] = v1.reduce_add()
                        s2_ptr[tt] = v2.reduce_add()
                        s3_ptr[tt] = v3.reduce_add()

                    _softmax_inline(s0_ptr, seq_len)
                    _softmax_inline(s1_ptr, seq_len)
                    _softmax_inline(s2_ptr, seq_len)
                    _softmax_inline(s3_ptr, seq_len)

                    for d in range(64):
                        acc0[d] = SIMD[DType.float32, 8](0.0)
                        acc1[d] = SIMD[DType.float32, 8](0.0)
                        acc2[d] = SIMD[DType.float32, 8](0.0)
                        acc3[d] = SIMD[DType.float32, 8](0.0)

                    for tt in range(seq_len):
                        var t_cache_offset = layer_cache_offset + tt * kv_stride + g_offset
                        var w0 = SIMD[DType.float32, 8](s0_ptr[tt])
                        var w1 = SIMD[DType.float32, 8](s1_ptr[tt])
                        var w2 = SIMD[DType.float32, 8](s2_ptr[tt])
                        var w3 = SIMD[DType.float32, 8](s3_ptr[tt])
                        for d in range(64):
                            var v_vec = v_cache.load[8](t_cache_offset + d * 8).cast[DType.float32]()
                            acc0[d] = w0.fma(v_vec, acc0[d])
                            acc1[d] = w1.fma(v_vec, acc1[d])
                            acc2[d] = w2.fma(v_vec, acc2[d])
                            acc3[d] = w3.fma(v_vec, acc3[d])

                    for d in range(64):
                        out_base.store[8](h0_off + d * 8, acc0[d].cast[DType.float16]())
                        out_base.store[8](h1_off + d * 8, acc1[d].cast[DType.float16]())
                        out_base.store[8](h2_off + d * 8, acc2[d].cast[DType.float16]())
                        out_base.store[8](h3_off + d * 8, acc3[d].cast[DType.float16]())

        parallelize[global_token_worker](active_len)
    else:
        def generic_attn_worker(b: Int) capturing -> None:
            var q_load = canvas_q + b * max_q_size
            var attn_out_ptr = canvas_attn_out + b * max_q_size
            var scores = InlineArray[Float32, 512](uninitialized=True)
            var s_ptr = UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=Int(UnsafePointer(to=scores)))
            var q_regs = InlineArray[SIMD[DType.float32, 8], 64](uninitialized=True)
            var acc_regs = InlineArray[SIMD[DType.float32, 8], 64](uninitialized=True)

            for h in range(layer_heads):
                var h_offset = h * layer_head_dim
                var g = h // ratio
                var g_offset = g * layer_head_dim
                for d in range(num_simd_lanes):
                    q_regs[d] = q_load.load[8](h_offset + d * 8).cast[DType.float32]()
                for tt in range(seq_len):
                    var t_cache_offset = layer_cache_offset + tt * kv_stride + g_offset
                    var score_vec = SIMD[DType.float32, 8](0.0)
                    for d in range(num_simd_lanes):
                        score_vec = q_regs[d].fma(k_cache.load[8](t_cache_offset + d * 8).cast[DType.float32](), score_vec)
                    s_ptr[tt] = score_vec.reduce_add()
                _softmax_inline(s_ptr, seq_len)
                for d in range(num_simd_lanes):
                    acc_regs[d] = SIMD[DType.float32, 8](0.0)
                for tt in range(seq_len):
                    var t_cache_offset = layer_cache_offset + tt * kv_stride + g_offset
                    var s_vec = SIMD[DType.float32, 8](s_ptr[tt])
                    for d in range(num_simd_lanes):
                        acc_regs[d] = s_vec.fma(v_cache.load[8](t_cache_offset + d * 8).cast[DType.float32](), acc_regs[d])
                for d in range(num_simd_lanes):
                    attn_out_ptr.store[8](h_offset + d * 8, acc_regs[d].cast[DType.float16]())

        parallelize[generic_attn_worker](active_len)


# ==============================================================================
# 7. AdaCache (ICCV 2025) + dLLM-Cache (2025): Per-Token Feature-Similarity
#    Partial Canvas Updates with Motion Regularization (llm-mojo-2tc.8)
# ==============================================================================


@always_inline
def adacache_filter_after_layer1(
    canvas_hidden: UnsafePointer[Float16, MutAnyOrigin],
    adacache_h1_prev: UnsafePointer[Float16, MutAnyOrigin],
    active_canvas_idx: UnsafePointer[Int, MutAnyOrigin],
    adacache_skip_streak: UnsafePointer[Int, MutAnyOrigin],
    eb_mask: UnsafePointer[Bool, _],
    active_len: Int,
    embed_dim: Int,
    denoise_step: Int,
    k_slow: Int = 2,
    tau_ada: Float32 = 0.015,
) -> Int:
    """Evaluates per-token normalized squared-L2 hidden-state fluctuation after Layer 1
    (`l == 1`). Crucially, unaccepted tokens (`not eb_mask[orig_t]`, whose input token ID
    was just randomly re-noised) are ALWAYS forced to evaluate all 30 layers + logit projection;
    only previously accepted quasi-static tokens (`eb_mask[orig_t] == True` with
    `raw_fluct < tau_ada`) are eligible for Layer 2..29 skipping."""
    if active_len <= 16 or denoise_step < 2:
        for b in range(active_len):
            var orig_t = active_canvas_idx[b]
            var h_curr = canvas_hidden + b * embed_dim
            var h_prev = adacache_h1_prev + orig_t * embed_dim
            for i in range(0, embed_dim, 16):
                h_prev.store[16](i, h_curr.load[16](i))
            adacache_skip_streak[orig_t] = 0
        return active_len

    var max_consec_skip = (k_slow - 1) if k_slow > 1 else 0
    var keep_mask = InlineArray[Bool, 256](uninitialized=True)
    var fluct = InlineArray[Float32, 256](uninitialized=True)
    var num_keep = 0

    for b in range(active_len):
        var orig_t = active_canvas_idx[b]
        var h_curr = canvas_hidden + b * embed_dim
        var h_prev = adacache_h1_prev + orig_t * embed_dim

        var diff_sq_vec = SIMD[DType.float32, 8](0.0)
        var norm_sq_vec = SIMD[DType.float32, 8](0.0)
        for i in range(0, embed_dim, 8):
            var c = h_curr.load[8](i).cast[DType.float32]()
            var p = h_prev.load[8](i).cast[DType.float32]()
            var d = c - p
            diff_sq_vec = d.fma(d, diff_sq_vec)
            norm_sq_vec = c.fma(c, p.fma(p, norm_sq_vec))

        var raw_fluct = diff_sq_vec.reduce_add() / (norm_sq_vec.reduce_add() + Float32(1e-8))
        fluct[b] = raw_fluct
        var streak = adacache_skip_streak[orig_t]

        # Never skip a re-noised/unaccepted token (`not eb_mask[orig_t]`)!
        if (not eb_mask[orig_t]) or (streak >= max_consec_skip) or (raw_fluct >= tau_ada):
            keep_mask[b] = True
            num_keep += 1
        else:
            keep_mask[b] = False

    if num_keep >= active_len:
        for b in range(active_len):
            var orig_t = active_canvas_idx[b]
            var h_curr = canvas_hidden + b * embed_dim
            var h_prev = adacache_h1_prev + orig_t * embed_dim
            for i in range(0, embed_dim, 16):
                h_prev.store[16](i, h_curr.load[16](i))
            adacache_skip_streak[orig_t] = 0
        return active_len

    # Round target_active up to a multiple of 16 for iGPU 16x16 matmul tiles
    var target_active = ((num_keep + 15) // 16) * 16
    if target_active < 16:
        target_active = 16
    if target_active >= active_len:
        for b in range(active_len):
            var orig_t = active_canvas_idx[b]
            var h_curr = canvas_hidden + b * embed_dim
            var h_prev = adacache_h1_prev + orig_t * embed_dim
            for i in range(0, embed_dim, 16):
                h_prev.store[16](i, h_curr.load[16](i))
            adacache_skip_streak[orig_t] = 0
        return active_len

    while num_keep < target_active:
        var best_b = -1
        var best_f: Float32 = -1.0
        for b in range(active_len):
            if not keep_mask[b] and fluct[b] > best_f:
                best_f = fluct[b]
                best_b = b
        if best_b < 0:
            break
        keep_mask[best_b] = True
        num_keep += 1

    # In-place monotonic compaction of canvas_hidden and active_canvas_idx.
    # Only update adacache_h1_prev[orig_t] for KEPT tokens so skipped tokens preserve
    # cumulative drift relative to the last step where Layers 2..29 were evaluated!
    var write_idx = 0
    for b in range(active_len):
        var orig_t = active_canvas_idx[b]
        if keep_mask[b]:
            adacache_skip_streak[orig_t] = 0
            var src_ptr = canvas_hidden + b * embed_dim
            var h_prev = adacache_h1_prev + orig_t * embed_dim
            for i in range(0, embed_dim, 16):
                h_prev.store[16](i, src_ptr.load[16](i))
            if write_idx != b:
                active_canvas_idx[write_idx] = orig_t
                var dst_ptr = canvas_hidden + write_idx * embed_dim
                for i in range(0, embed_dim, 16):
                    dst_ptr.store[16](i, src_ptr.load[16](i))
            write_idx += 1
        else:
            adacache_skip_streak[orig_t] += 1

    return write_idx


# ==============================================================================
# 8. Block-Diffusion Canvas Mode (BD3-LM / Fast-dLLM v2) & 2nd-Order Latent
#    Velocity Extrapolation (llm-mojo-2tc.6)
# ==============================================================================


@always_inline
def build_block_active_canvas_map(
    locked_mask: UnsafePointer[Bool, _],
    active_canvas_idx: UnsafePointer[Int, MutAnyOrigin],
    block_start: Int,
    block_end: Int,
) -> Int:
    """Builds the active canvas index map `active_canvas_idx[0..active_len-1]` restricted
    to unlocked positions within the semi-autoregressive block `[block_start, block_end)`."""
    var active_len = 0
    for t in range(block_start, block_end):
        if not locked_mask[t]:
            active_canvas_idx[active_len] = t
            active_len += 1
    return active_len


@always_inline
def extrapolate_sc_velocity_batch(
    sc_embeds: UnsafePointer[Float32, MutAnyOrigin],
    prev_sc_embeds: UnsafePointer[Float32, MutAnyOrigin],
    locked_mask: UnsafePointer[Bool, _],
    block_start: Int,
    block_end: Int,
    embed_dim: Int,
    steps_executed: Int,
    skip_amount: Int,
    max_vel_ratio: Float32 = 0.85,
):
    """Applies 2nd-order Adams-Bashforth latent velocity extrapolation on `sc_embeds`:
        `s_extrap = normalize_norm(s_curr + beta * (s_curr - s_prev))`
    with a discontinuous-flip trust-region guard (`||s_curr - s_prev||_2 <= max_vel_ratio * ||s_curr||_2`)
    and exact L2 norm conservation (`||s_extrap||_2 == ||s_curr||_2`)."""
    if steps_executed <= 1:
        for t in range(block_start, block_end):
            var s_ptr = sc_embeds + t * embed_dim
            var p_ptr = prev_sc_embeds + t * embed_dim
            for i in range(0, embed_dim, 8):
                p_ptr.store[8](i, s_ptr.load[8](i))
        return

    var beta = Float32(0.12) * Float32(skip_amount)
    if beta > Float32(0.35):
        beta = Float32(0.35)
    var max_vel_sq = max_vel_ratio * max_vel_ratio
    var beta_vec = SIMD[DType.float32, 8](beta)

    for t in range(block_start, block_end):
        var s_ptr = sc_embeds + t * embed_dim
        var p_ptr = prev_sc_embeds + t * embed_dim
        if locked_mask[t] or beta <= Float32(0.0):
            for i in range(0, embed_dim, 8):
                p_ptr.store[8](i, s_ptr.load[8](i))
            continue

        var norm_s_sq_vec = SIMD[DType.float32, 8](0.0)
        var norm_v_sq_vec = SIMD[DType.float32, 8](0.0)
        for i in range(0, embed_dim, 8):
            var s_v = s_ptr.load[8](i)
            var p_v = p_ptr.load[8](i)
            var v_v = s_v - p_v
            norm_s_sq_vec = s_v.fma(s_v, norm_s_sq_vec)
            norm_v_sq_vec = v_v.fma(v_v, norm_v_sq_vec)

        var norm_s_sq = norm_s_sq_vec.reduce_add()
        var norm_v_sq = norm_v_sq_vec.reduce_add()

        if norm_s_sq > Float32(1e-8) and norm_v_sq <= max_vel_sq * norm_s_sq:
            var norm_ext_sq_vec = SIMD[DType.float32, 8](0.0)
            for i in range(0, embed_dim, 8):
                var s_v = s_ptr.load[8](i)
                var p_v = p_ptr.load[8](i)
                var ext_v = s_v + beta_vec * (s_v - p_v)
                norm_ext_sq_vec = ext_v.fma(ext_v, norm_ext_sq_vec)
            var rescale = sqrt(norm_s_sq / (norm_ext_sq_vec.reduce_add() + Float32(1e-10)))
            var rescale_vec = SIMD[DType.float32, 8](rescale)
            for i in range(0, embed_dim, 8):
                var s_v = s_ptr.load[8](i)
                var p_v = p_ptr.load[8](i)
                var ext_v = (s_v + beta_vec * (s_v - p_v)) * rescale_vec
                p_ptr.store[8](i, s_v)
                s_ptr.store[8](i, ext_v)
        else:
            for i in range(0, embed_dim, 8):
                p_ptr.store[8](i, s_ptr.load[8](i))






