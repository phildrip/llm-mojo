from std.memory import alloc
from std.math import sqrt, exp, abs
import std.time
from lsa_core import LSALayerIndex, LSA_DIM, LSA_BUCKET_CAP
from moe_kernels import (
    run_moe_experts_stage1_8core,
    run_moe_experts_stage2_8core,
    run_moe_coalesced_core,
)


def main() raises:
    print("==================================================================")
    print("   TDD & BENCHMARK: Hoare Bucket-Sorted MoE Coalescing & Rollback ")
    print("==================================================================")

    # Test 1: LSALayerIndex.rollback_to across a 512-token Ward cluster block boundary + exact 512 boundary
    var idx_clean = LSALayerIndex(max_capacity=2048)
    var idx_spec = LSALayerIndex(max_capacity=2048)
    var k_buf = alloc[Float32](LSA_DIM)

    for pos in range(515):
        for d in range(LSA_DIM):
            k_buf[d] = Float32(((pos * 13 + d * 7) % 31) - 15) * 0.05
        idx_clean.append_key(pos, k_buf)
        if pos < 510:
            idx_spec.append_key(pos, k_buf)

    # Append 5 bogus speculative draft tokens at 510..514 (crossing 511 -> closing block 0!)
    for pos in range(510, 515):
        for d in range(LSA_DIM):
            k_buf[d] = Float32(((pos * 97 + d * 3) % 29) - 14) * 0.05
        idx_spec.append_key(pos, k_buf)

    if idx_spec.closed_blocks != 1:
        raise Error("Expected idx_spec.closed_blocks == 1 after crossing 511")

    # Rollback to 510 and replay true tokens 510..514
    idx_spec.rollback_to(510)
    if idx_spec.closed_blocks != 0 or idx_spec.total_tokens_appended != 510:
        raise Error("Rollback to 510 failed to undo closed block 0!")

    for pos in range(510, 515):
        for d in range(LSA_DIM):
            k_buf[d] = Float32(((pos * 13 + d * 7) % 31) - 15) * 0.05
        idx_spec.append_key(pos, k_buf)

    if idx_spec.closed_blocks != idx_clean.closed_blocks or idx_spec.total_tokens_appended != idx_clean.total_tokens_appended:
        raise Error("Replayed LSA index state mismatch after rollback!")
    for b in range(32):
        if idx_spec.sizes[b] != idx_clean.sizes[b]:
            raise Error("Bucket size mismatch at bucket " + String(b))
        for m in range(idx_clean.sizes[b]):
            if idx_spec.members[b * LSA_BUCKET_CAP + m] != idx_clean.members[b * LSA_BUCKET_CAP + m]:
                raise Error("Bucket member mismatch at bucket " + String(b))
        for d in range(LSA_DIM):
            if abs(idx_spec.centroids[b * LSA_DIM + d] - idx_clean.centroids[b * LSA_DIM + d]) > 1e-6:
                raise Error("Bucket centroid mismatch at bucket " + String(b))

    # Also test rollback_to(512) (exact block boundary: block 0 stays closed, tokens 512..514 discarded)
    idx_spec.rollback_to(512)
    if idx_spec.closed_blocks != 1 or idx_spec.total_tokens_appended != 512:
        raise Error("Rollback to exact 512 boundary should preserve closed block 0!")
    idx_clean.free()
    idx_spec.free()
    print("  [Test 1] LSALayerIndex.rollback_to (cross-boundary + exact-512, members & centroids): PASS")

    # Test 2: Hoare Bucket-Sorted Multi-Token MoE Expert Coalescing (batch_b = 5)
    var batch_b = 5
    var embed_dim = 2816
    var expert_mlp_dim = 704
    var gate_up_rows = expert_mlp_dim * 2
    var num_experts = 16
    var half_in = embed_dim // 2
    var c_in_blocks = embed_dim // 32
    var down_half_in = expert_mlp_dim // 2
    var down_blocks = expert_mlp_dim // 32

    var gate_up_packed = alloc[UInt8](num_experts * gate_up_rows * half_in)
    var gate_up_scales = alloc[Float32](num_experts * gate_up_rows * c_in_blocks)
    var down_packed = alloc[UInt8](num_experts * embed_dim * down_half_in)
    var down_scales = alloc[Float32](num_experts * embed_dim * down_blocks)

    for i in range(num_experts * gate_up_rows * half_in):
        gate_up_packed[i] = UInt8((i * 17 + 5) & 255)
    for i in range(num_experts * gate_up_rows * c_in_blocks):
        gate_up_scales[i] = Float32((i % 19) + 1) * 0.002
    for i in range(num_experts * embed_dim * down_half_in):
        down_packed[i] = UInt8((i * 23 + 11) & 255)
    for i in range(num_experts * embed_dim * down_blocks):
        down_scales[i] = Float32((i % 17) + 1) * 0.002

    var batch_moe_in = alloc[Float32](batch_b * embed_dim)
    for i in range(batch_b * embed_dim):
        batch_moe_in[i] = Float32((i % 37) - 18) * 0.02

    # Each of the 5 tokens selects 6 correlated experts (realistic overlap across consecutive tokens)
    var tok_active_count = alloc[Int](batch_b)
    var tok_active_idx = alloc[Int](batch_b * 8)
    var tok_active_weights = alloc[Float32](batch_b * 8)
    for b in range(batch_b):
        tok_active_count[b] = 6
        for k in range(6):
            tok_active_idx[b * 8 + k] = (k * 2 + (b // 2)) % num_experts
            tok_active_weights[b * 8 + k] = Float32(1.0 / 6.0)

    var ref_out = alloc[Float32](batch_b * embed_dim)
    var opt_out = alloc[Float32](batch_b * embed_dim)
    var batch_in_deint = alloc[Float32](batch_b * embed_dim)
    var gu_out_scratch = alloc[Float32](8 * batch_b * gate_up_rows)
    var act_scratch = alloc[Float32](8 * batch_b * expert_mlp_dim)
    var act_deint_scratch = alloc[Float32](8 * batch_b * expert_mlp_dim)
    var bucket_counts = alloc[Int](128)
    var bucket_toks = alloc[Int](128 * 5)
    var bucket_weights = alloc[Float32](128 * 5)
    var unique_experts = alloc[Int](128)
    var unique_base_slot = alloc[Int](128)

    var iters = 30
    var t0 = std.time.perf_counter()
    for _ in range(iters):
        for b in range(batch_b):
            run_moe_experts_stage1_8core(
                gate_up_packed,
                gate_up_scales,
                batch_moe_in + b * embed_dim,
                batch_in_deint + b * embed_dim,
                tok_active_idx + b * 8,
                tok_active_count[b],
                gu_out_scratch,
                act_scratch,
                embed_dim,
                expert_mlp_dim,
                8,
            )
            run_moe_experts_stage2_8core(
                down_packed,
                down_scales,
                act_scratch,
                act_deint_scratch,
                tok_active_idx + b * 8,
                tok_active_weights + b * 8,
                tok_active_count[b],
                ref_out + b * embed_dim,
                embed_dim,
                expert_mlp_dim,
                8,
            )
    var t1 = std.time.perf_counter()

    for _ in range(iters):
        for b in range(batch_b):
            var x_b = batch_moe_in + b * embed_dim
            var xe_ptr = batch_in_deint + b * embed_dim
            var xo_ptr = xe_ptr + half_in
            for blk in range(c_in_blocks):
                var v0 = x_b.load[16](blk * 32)
                var v1 = x_b.load[16](blk * 32 + 16)
                xe_ptr.store[16](blk * 16, v0.deinterleave()[0].join(v1.deinterleave()[0]))
                xo_ptr.store[16](blk * 16, v0.deinterleave()[1].join(v1.deinterleave()[1]))

        run_moe_coalesced_core(
            batch_b,
            gate_up_packed,
            gate_up_scales,
            down_packed,
            down_scales,
            batch_in_deint,
            tok_active_count,
            tok_active_idx,
            tok_active_weights,
            bucket_counts,
            bucket_toks,
            bucket_weights,
            unique_experts,
            unique_base_slot,
            gu_out_scratch,
            act_scratch,
            act_deint_scratch,
            opt_out,
            embed_dim,
            expert_mlp_dim,
            num_experts,
            8,
        )
    var t2 = std.time.perf_counter()

    var max_err: Float32 = 0.0
    for i in range(batch_b * embed_dim):
        var diff = abs(ref_out[i] - opt_out[i])
        if diff > max_err:
            max_err = diff

    var ref_ms = (t1 - t0) * 1000.0 / Float64(iters)
    var opt_ms = (t2 - t1) * 1000.0 / Float64(iters)
    print("  [Test 2] Batch=5 Coalesced MoE Max Error:", max_err)
    print("  Sequential 5x Token-by-Token: ", ref_ms, "ms/layer")
    print("  Hoare Bucket-Sorted Coalesced:", opt_ms, "ms/layer")
    print("  Speedup factor:               ", ref_ms / opt_ms, "x")

    if max_err > 1e-6:
        raise Error("Coalesced MoE output mismatch: " + String(max_err))
    print("  [PASS] Hoare Bucket-Sorted MoE Coalescing & LSA Rollback Verified!")

    k_buf.free()
    gate_up_packed.free()
    gate_up_scales.free()
    down_packed.free()
    down_scales.free()
    batch_moe_in.free()
    tok_active_count.free()
    tok_active_idx.free()
    tok_active_weights.free()
    ref_out.free()
    opt_out.free()
    batch_in_deint.free()
    gu_out_scratch.free()
    act_scratch.free()
    act_deint_scratch.free()
    bucket_counts.free()
    bucket_toks.free()
    bucket_weights.free()
    unique_experts.free()
    unique_base_slot.free()
