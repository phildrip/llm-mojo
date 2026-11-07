from std.algorithm import parallelize
from std.collections import InlineArray
from std.memory import alloc
from std.math import exp


def q4_matmul_vec(
    W_packed: UnsafePointer[UInt8, _],
    scales: UnsafePointer[Float32, _],
    x: UnsafePointer[Float32, _],
    out_ptr: UnsafePointer[Float32, MutUntrackedOrigin],
    c_in: Int,
    c_out: Int,
    num_tasks: Int,
):
    """
    Zero-alloc Dual-Space Deinterleaved Q4_0 matrix-vector multiplication with 4-row register blocking.
    Uses a stack InlineArray[Float32, 8192] for c_in <= 8192 (zero heap alloc/free per call).
    """
    debug_assert(c_in % 32 == 0, "q4_matmul_vec requires c_in to be a multiple of 32")
    if c_out <= 0:
        return
    var tasks = num_tasks
    if c_out <= 256:
        tasks = 1
    elif c_out <= 1024:
        tasks = 2 if tasks >= 2 else 1
    elif c_out <= 2048:
        tasks = 4 if tasks >= 4 else tasks
    if tasks > c_out:
        tasks = c_out
    if tasks < 1:
        tasks = 1
    var chunk_size = c_out // tasks
    var c_in_packed = c_in // 2
    var c_in_blocks = c_in // 32

    var stack_deint = InlineArray[Float32, 8192](uninitialized=True)
    stack_deint[0] = 0.0
    var x_deint = UnsafePointer[Float32, MutUntrackedOrigin](
        unsafe_from_address=Int(stack_deint.unsafe_ptr())
    ) if c_in <= 8192 else alloc[Float32](c_in)
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
        while r + 1 < end_r:
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
            var r_offset_packed = r * c_in_packed
            var r_offset_scales = r * c_in_blocks
            for b in range(c_in_blocks):
                var scale = scales[r_offset_scales + b]
                var bytes = W_packed.load[16](r_offset_packed + b * 16)
                var we = ((bytes & 15).cast[DType.int8]() - 8).cast[DType.float32]()
                var wo = ((bytes >> 4).cast[DType.int8]() - 8).cast[DType.float32]()
                acc = acc + (we * x_even.load[16](b * 16) + wo * x_odd.load[16](b * 16)) * scale
            out_ptr[r] = acc.reduce_add()
            r += 1

    if tasks == 1:
        worker(0)
    else:
        parallelize[worker](tasks)

    # Pin stack_deint lifetime past parallelize[worker] so Mojo's ASAP destructor cannot
    # reclaim the 32 KB stack buffer while worker threads are reading from x_even/x_odd.
    var _keep_alive = stack_deint[0]
    if _keep_alive > 1e38:
        out_ptr[0] += 0.0
    if c_in > 8192:
        x_deint.free()


@always_inline
def gelu_and_mul_moe(
    h1: UnsafePointer[Float32, _],
    h2: UnsafePointer[Float32, _],
    out_ptr: UnsafePointer[Float32, MutUntrackedOrigin],
    size: Int,
):
    for i in range(0, size, 8):
        var x1 = h1.load[8](i)
        var sig = SIMD[DType.float32, 8](1.0) / (
            SIMD[DType.float32, 8](1.0) + exp(SIMD[DType.float32, 8](-1.702) * x1)
        )
        var gelu = x1 * sig
        out_ptr.store[8](i, gelu * h2.load[8](i))


@always_inline
def compute_moe_lod_weights(
    top_idx: UnsafePointer[Int, _],
    top_val: UnsafePointer[Float32, _],
    per_expert_scale: UnsafePointer[Float32, _],
    moe_top_p: Float32,
    active_idx: UnsafePointer[Int, MutUntrackedOrigin],
    active_weights: UnsafePointer[Float32, MutUntrackedOrigin],
    top_k: Int = 8,
) -> Int:
    """Clark (1976) Hierarchical LOD / Arvo & Kirk (1990) Russian-Roulette Top-p MoE pruning."""
    var sum_top_val: Float32 = 0.0
    for i in range(top_k):
        sum_top_val += top_val[i]
    var inv_sum = Float32(1.0) / (sum_top_val + Float32(1e-10))

    # Guard against NaN or >= 1.0 -> lossless default threshold
    var safe_top_p = moe_top_p if (moe_top_p == moe_top_p) else Float32(1.0)
    if safe_top_p >= Float32(1.0):
        var count = 0
        for i in range(top_k):
            var norm_prob = top_val[i] * inv_sum
            if norm_prob >= Float32(0.05) or i == 0:
                var idx = top_idx[i]
                active_idx[count] = idx
                active_weights[count] = norm_prob * per_expert_scale[idx]
                count += 1
        return count

    var cum_prob: Float32 = 0.0
    var count = 0
    for i in range(top_k):
        var norm_prob = top_val[i] * inv_sum
        if i > 0 and (norm_prob < Float32(0.05) or cum_prob >= safe_top_p):
            break
        active_idx[count] = top_idx[i]
        active_weights[count] = norm_prob
        cum_prob += norm_prob
        count += 1

    # Energy-preserving renormalization across retained experts
    var renorm_factor = Float32(1.0) / (cum_prob + Float32(1e-10))
    for i in range(count):
        var idx = active_idx[i]
        active_weights[i] = (active_weights[i] * renorm_factor) * per_expert_scale[idx]
    return count


@always_inline
def run_moe_experts_stage1_8core(
    experts_gate_up_packed: UnsafePointer[UInt8, _],
    experts_gate_up_scales: UnsafePointer[Float32, _],
    moe_in: UnsafePointer[Float32, _],
    moe_in_deint: UnsafePointer[Float32, MutUntrackedOrigin],
    active_idx: UnsafePointer[Int, _],
    active_count: Int,
    gate_up_out_all: UnsafePointer[Float32, MutUntrackedOrigin],
    act_all: UnsafePointer[Float32, MutUntrackedOrigin],
    embed_dim: Int = 2816,
    expert_mlp_dim: Int = 704,
    num_tasks: Int = 8,
):
    """Row-sliced Stage 1 MoE execution: computes gate + up + fused GeLU across all active experts."""
    var half_in = embed_dim // 2
    var c_in_blocks = embed_dim // 32
    var xe_ptr = moe_in_deint
    var xo_ptr = moe_in_deint + half_in
    for b in range(c_in_blocks):
        var v0 = moe_in.load[16](b * 32)
        var v1 = moe_in.load[16](b * 32 + 16)
        xe_ptr.store[16](b * 16, v0.deinterleave()[0].join(v1.deinterleave()[0]))
        xo_ptr.store[16](b * 16, v0.deinterleave()[1].join(v1.deinterleave()[1]))

    var gate_up_rows = expert_mlp_dim * 2
    var tasks = max(1, min(num_tasks, gate_up_rows))

    def stage1_worker(t: Int) capturing -> None:
        var chunk = gate_up_rows // tasks
        var sr = t * chunk
        var er = (t + 1) * chunk if t < tasks - 1 else gate_up_rows
        for exp_slot in range(active_count):
            var e = active_idx[exp_slot]
            var gu_p = experts_gate_up_packed + e * (gate_up_rows * half_in)
            var gu_s = experts_gate_up_scales + e * (gate_up_rows * c_in_blocks)
            var gu_out = gate_up_out_all + exp_slot * gate_up_rows

            var r = sr
            while r + 1 < er:
                var acc0 = SIMD[DType.float32, 16](0.0)
                var acc1 = SIMD[DType.float32, 16](0.0)
                var rp0 = r * half_in
                var rs0 = r * c_in_blocks
                var rp1 = (r + 1) * half_in
                var rs1 = (r + 1) * c_in_blocks
                for b in range(c_in_blocks):
                    var xe = xe_ptr.load[16](b * 16)
                    var xo = xo_ptr.load[16](b * 16)
                    var sc0 = gu_s[rs0 + b]
                    var b0 = gu_p.load[16](rp0 + b * 16)
                    var we0 = ((b0 & 15).cast[DType.int8]() - 8).cast[DType.float32]()
                    var wo0 = ((b0 >> 4).cast[DType.int8]() - 8).cast[DType.float32]()
                    acc0 = acc0 + (we0 * xe + wo0 * xo) * sc0

                    var sc1 = gu_s[rs1 + b]
                    var b1 = gu_p.load[16](rp1 + b * 16)
                    var we1 = ((b1 & 15).cast[DType.int8]() - 8).cast[DType.float32]()
                    var wo1 = ((b1 >> 4).cast[DType.int8]() - 8).cast[DType.float32]()
                    acc1 = acc1 + (we1 * xe + wo1 * xo) * sc1
                gu_out[r] = acc0.reduce_add()
                gu_out[r + 1] = acc1.reduce_add()
                r += 2
            while r < er:
                var acc0 = SIMD[DType.float32, 16](0.0)
                var rp0 = r * half_in
                var rs0 = r * c_in_blocks
                for b in range(c_in_blocks):
                    var xe = xe_ptr.load[16](b * 16)
                    var xo = xo_ptr.load[16](b * 16)
                    var sc0 = gu_s[rs0 + b]
                    var b0 = gu_p.load[16](rp0 + b * 16)
                    var we0 = ((b0 & 15).cast[DType.int8]() - 8).cast[DType.float32]()
                    var wo0 = ((b0 >> 4).cast[DType.int8]() - 8).cast[DType.float32]()
                    acc0 = acc0 + (we0 * xe + wo0 * xo) * sc0
                gu_out[r] = acc0.reduce_add()
                r += 1

    parallelize[stage1_worker](tasks)
    for exp_slot in range(active_count):
        var gu_out = gate_up_out_all + exp_slot * gate_up_rows
        var act_ptr = act_all + exp_slot * expert_mlp_dim
        gelu_and_mul_moe(gu_out, gu_out + expert_mlp_dim, act_ptr, expert_mlp_dim)


@always_inline
def run_moe_experts_stage2_8core(
    experts_down_packed: UnsafePointer[UInt8, _],
    experts_down_scales: UnsafePointer[Float32, _],
    act_all: UnsafePointer[Float32, _],
    act_deint: UnsafePointer[Float32, MutUntrackedOrigin],
    active_idx: UnsafePointer[Int, _],
    active_weights: UnsafePointer[Float32, _],
    active_count: Int,
    moe_out: UnsafePointer[Float32, MutUntrackedOrigin],
    embed_dim: Int = 2816,
    expert_mlp_dim: Int = 704,
    num_tasks: Int = 8,
):
    """Row-sliced Stage 2 MoE execution: computes experts_down and fuses weighted accumulation into moe_out."""
    var c_in_packed = expert_mlp_dim // 2
    var c_in_blocks = expert_mlp_dim // 32
    var tasks = max(1, min(num_tasks, embed_dim))

    for exp_slot in range(active_count):
        var act_ptr = act_all + exp_slot * expert_mlp_dim
        var ad_e = act_deint + exp_slot * expert_mlp_dim
        var ad_o = ad_e + c_in_packed
        for b in range(c_in_blocks):
            var v0 = act_ptr.load[16](b * 32)
            var v1 = act_ptr.load[16](b * 32 + 16)
            ad_e.store[16](b * 16, v0.deinterleave()[0].join(v1.deinterleave()[0]))
            ad_o.store[16](b * 16, v0.deinterleave()[1].join(v1.deinterleave()[1]))

    def stage2_worker(t: Int) capturing -> None:
        var chunk = embed_dim // tasks
        var sr = t * chunk
        var er = (t + 1) * chunk if t < tasks - 1 else embed_dim
        for r in range(sr, er):
            moe_out[r] = 0.0
        for exp_slot in range(active_count):
            var e = active_idx[exp_slot]
            var w_e = active_weights[exp_slot]
            var dn_p = experts_down_packed + e * (embed_dim * c_in_packed)
            var dn_s = experts_down_scales + e * (embed_dim * c_in_blocks)
            var ad_e = act_deint + exp_slot * expert_mlp_dim
            var ad_o = ad_e + c_in_packed
            var r = sr
            while r + 1 < er:
                var acc0 = SIMD[DType.float32, 16](0.0)
                var acc1 = SIMD[DType.float32, 16](0.0)
                var rp0 = r * c_in_packed
                var rs0 = r * c_in_blocks
                var rp1 = (r + 1) * c_in_packed
                var rs1 = (r + 1) * c_in_blocks
                for b in range(c_in_blocks):
                    var xe = ad_e.load[16](b * 16)
                    var xo = ad_o.load[16](b * 16)
                    var sc0 = dn_s[rs0 + b]
                    var b0 = dn_p.load[16](rp0 + b * 16)
                    var we0 = ((b0 & 15).cast[DType.int8]() - 8).cast[DType.float32]()
                    var wo0 = ((b0 >> 4).cast[DType.int8]() - 8).cast[DType.float32]()
                    acc0 = acc0 + (we0 * xe + wo0 * xo) * sc0

                    var sc1 = dn_s[rs1 + b]
                    var b1 = dn_p.load[16](rp1 + b * 16)
                    var we1 = ((b1 & 15).cast[DType.int8]() - 8).cast[DType.float32]()
                    var wo1 = ((b1 >> 4).cast[DType.int8]() - 8).cast[DType.float32]()
                    acc1 = acc1 + (we1 * xe + wo1 * xo) * sc1
                moe_out[r] += acc0.reduce_add() * w_e
                moe_out[r + 1] += acc1.reduce_add() * w_e
                r += 2
            while r < er:
                var acc0 = SIMD[DType.float32, 16](0.0)
                var rp0 = r * c_in_packed
                var rs0 = r * c_in_blocks
                for b in range(c_in_blocks):
                    var xe = ad_e.load[16](b * 16)
                    var xo = ad_o.load[16](b * 16)
                    var sc0 = dn_s[rs0 + b]
                    var b0 = dn_p.load[16](rp0 + b * 16)
                    var we0 = ((b0 & 15).cast[DType.int8]() - 8).cast[DType.float32]()
                    var wo0 = ((b0 >> 4).cast[DType.int8]() - 8).cast[DType.float32]()
                    acc0 = acc0 + (we0 * xe + wo0 * xo) * sc0
                moe_out[r] += acc0.reduce_add() * w_e
                r += 1

    parallelize[stage2_worker](tasks)


@always_inline
def run_moe_coalesced_core(
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
    """
    Hoare (1961) Bucket-Sorted Coalesced MoE evaluation across 1..5 tokens simultaneously.
    Uses identical (we * xe + wo * xo) * sc FP32 association to single-token MoE kernels.
    """
    debug_assert(sub_b >= 1 and sub_b <= 5, "run_moe_coalesced_core requires 1 <= sub_b <= 5")
    var half_in = embed_dim // 2
    var c_in_blocks = embed_dim // 32
    var gate_up_rows = expert_mlp_dim * 2
    var down_half_in = expert_mlp_dim // 2
    var down_blocks = expert_mlp_dim // 32
    var tasks = max(1, num_tasks)

    # 1. Hoare Bucket-Sort (Inverted Index): group (token b, weight w) by expert ID e
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

    # 2. Coalesced Stage 1: Load each unique expert's gate_up weights ONCE from RAM
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
            var b_t1 = bucket_toks[e * 5 + 1] if m_count > 1 else 0
            var b_t2 = bucket_toks[e * 5 + 2] if m_count > 2 else 0
            var b_t3 = bucket_toks[e * 5 + 3] if m_count > 3 else 0
            var b_t4 = bucket_toks[e * 5 + 4] if m_count > 4 else 0
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
                    var we = ((bytes & 15).cast[DType.int8]() - 8).cast[DType.float32]()
                    var wo = ((bytes >> 4).cast[DType.int8]() - 8).cast[DType.float32]()
                    var xe0 = (batch_in_deint + b_t0 * embed_dim).load[16](blk * 16)
                    var xo0 = (batch_in_deint + b_t0 * embed_dim + half_in).load[16](blk * 16)
                    acc0 = acc0 + (we * xe0 + wo * xo0) * sc
                    if m_count > 1:
                        var xe1 = (batch_in_deint + b_t1 * embed_dim).load[16](blk * 16)
                        var xo1 = (batch_in_deint + b_t1 * embed_dim + half_in).load[16](blk * 16)
                        acc1 = acc1 + (we * xe1 + wo * xo1) * sc
                    if m_count > 2:
                        var xe2 = (batch_in_deint + b_t2 * embed_dim).load[16](blk * 16)
                        var xo2 = (batch_in_deint + b_t2 * embed_dim + half_in).load[16](blk * 16)
                        acc2 = acc2 + (we * xe2 + wo * xo2) * sc
                    if m_count > 3:
                        var xe3 = (batch_in_deint + b_t3 * embed_dim).load[16](blk * 16)
                        var xo3 = (batch_in_deint + b_t3 * embed_dim + half_in).load[16](blk * 16)
                        acc3 = acc3 + (we * xe3 + wo * xo3) * sc
                    if m_count > 4:
                        var xe4 = (batch_in_deint + b_t4 * embed_dim).load[16](blk * 16)
                        var xo4 = (batch_in_deint + b_t4 * embed_dim + half_in).load[16](blk * 16)
                        acc4 = acc4 + (we * xe4 + wo * xo4) * sc
                gu_out_scratch[base_slot * gate_up_rows + r] = acc0.reduce_add()
                if m_count > 1:
                    gu_out_scratch[(base_slot + 1) * gate_up_rows + r] = acc1.reduce_add()
                if m_count > 2:
                    gu_out_scratch[(base_slot + 2) * gate_up_rows + r] = acc2.reduce_add()
                if m_count > 3:
                    gu_out_scratch[(base_slot + 3) * gate_up_rows + r] = acc3.reduce_add()
                if m_count > 4:
                    gu_out_scratch[(base_slot + 4) * gate_up_rows + r] = acc4.reduce_add()

    parallelize[coal_s1](tasks)

    for p in range(total_pairs):
        var gu_out = gu_out_scratch + p * gate_up_rows
        var act_ptr = act_scratch + p * expert_mlp_dim
        gelu_and_mul_moe(gu_out, gu_out + expert_mlp_dim, act_ptr, expert_mlp_dim)
        var ad_e = act_deint_scratch + p * expert_mlp_dim
        var ad_o = ad_e + down_half_in
        for blk in range(down_blocks):
            var v0 = act_ptr.load[16](blk * 32)
            var v1 = act_ptr.load[16](blk * 32 + 16)
            ad_e.store[16](blk * 16, v0.deinterleave()[0].join(v1.deinterleave()[0]))
            ad_o.store[16](blk * 16, v0.deinterleave()[1].join(v1.deinterleave()[1]))

    # 3. Coalesced Stage 2: Load each unique expert's down weights ONCE from RAM
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
            var b_t1 = bucket_toks[e * 5 + 1] if m_count > 1 else 0
            var b_t2 = bucket_toks[e * 5 + 2] if m_count > 2 else 0
            var b_t3 = bucket_toks[e * 5 + 3] if m_count > 3 else 0
            var b_t4 = bucket_toks[e * 5 + 4] if m_count > 4 else 0
            var w0 = bucket_weights[e * 5 + 0]
            var w1 = bucket_weights[e * 5 + 1] if m_count > 1 else Float32(0.0)
            var w2 = bucket_weights[e * 5 + 2] if m_count > 2 else Float32(0.0)
            var w3 = bucket_weights[e * 5 + 3] if m_count > 3 else Float32(0.0)
            var w4 = bucket_weights[e * 5 + 4] if m_count > 4 else Float32(0.0)
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
                    var we = ((bytes & 15).cast[DType.int8]() - 8).cast[DType.float32]()
                    var wo = ((bytes >> 4).cast[DType.int8]() - 8).cast[DType.float32]()
                    var ae0 = (act_deint_scratch + base_slot * expert_mlp_dim).load[16](blk * 16)
                    var ao0 = (act_deint_scratch + base_slot * expert_mlp_dim + down_half_in).load[16](blk * 16)
                    acc0 = acc0 + (we * ae0 + wo * ao0) * sc
                    if m_count > 1:
                        var ae1 = (act_deint_scratch + (base_slot + 1) * expert_mlp_dim).load[16](blk * 16)
                        var ao1 = (act_deint_scratch + (base_slot + 1) * expert_mlp_dim + down_half_in).load[16](blk * 16)
                        acc1 = acc1 + (we * ae1 + wo * ao1) * sc
                    if m_count > 2:
                        var ae2 = (act_deint_scratch + (base_slot + 2) * expert_mlp_dim).load[16](blk * 16)
                        var ao2 = (act_deint_scratch + (base_slot + 2) * expert_mlp_dim + down_half_in).load[16](blk * 16)
                        acc2 = acc2 + (we * ae2 + wo * ao2) * sc
                    if m_count > 3:
                        var ae3 = (act_deint_scratch + (base_slot + 3) * expert_mlp_dim).load[16](blk * 16)
                        var ao3 = (act_deint_scratch + (base_slot + 3) * expert_mlp_dim + down_half_in).load[16](blk * 16)
                        acc3 = acc3 + (we * ae3 + wo * ao3) * sc
                    if m_count > 4:
                        var ae4 = (act_deint_scratch + (base_slot + 4) * expert_mlp_dim).load[16](blk * 16)
                        var ao4 = (act_deint_scratch + (base_slot + 4) * expert_mlp_dim + down_half_in).load[16](blk * 16)
                        acc4 = acc4 + (we * ae4 + wo * ao4) * sc
                batch_moe_out[b_t0 * embed_dim + r] += acc0.reduce_add() * w0
                if m_count > 1:
                    batch_moe_out[b_t1 * embed_dim + r] += acc1.reduce_add() * w1
                if m_count > 2:
                    batch_moe_out[b_t2 * embed_dim + r] += acc2.reduce_add() * w2
                if m_count > 3:
                    batch_moe_out[b_t3 * embed_dim + r] += acc3.reduce_add() * w3
                if m_count > 4:
                    batch_moe_out[b_t4 * embed_dim + r] += acc4.reduce_add() * w4

    parallelize[coal_s2](tasks)
