from std.memory import alloc
from std.sys.info import num_physical_cores
from std.math import abs
from moe_kernels import (
    compute_subbatch_set_cover_moe_weights,
    run_moe_coalesced_core,
    run_moe_coalesced_stage1,
    run_moe_coalesced_stage2,
)


def test_split_moe_equivalence() raises:
    print("=== TDD Test: Two-Phase Split Coalesced MoE Equivalence (llm-mojo-xr6.3) ===")
    var num_tasks = num_physical_cores()
    var sub_b = 5
    var top_k = 8
    var num_experts = 128
    var embed_dim = 2816
    var expert_mlp_dim = 704
    var gate_up_rows = expert_mlp_dim * 2
    var half_in = embed_dim // 2
    var c_in_blocks = embed_dim // 32
    var down_half_in = expert_mlp_dim // 2
    var down_blocks = expert_mlp_dim // 32

    var all_top_idx = alloc[Int](sub_b * top_k)
    var all_top_val = alloc[Float32](sub_b * top_k)
    var per_expert_scale = alloc[Float32](num_experts)
    for e in range(num_experts):
        per_expert_scale[e] = Float32(0.85 + 0.3 * Float64((e * 19) % 29) / 29.0)

    for b in range(sub_b):
        for k in range(top_k):
            all_top_idx[b * 8 + k] = (b * 7 + k * 13) % num_experts
            all_top_val[b * 8 + k] = Float32(0.30 - 0.032 * Float64(k))

    var sc_counts = alloc[Int](sub_b)
    var sc_idx = alloc[Int](sub_b * top_k)
    var sc_weights = alloc[Float32](sub_b * top_k)
    compute_subbatch_set_cover_moe_weights(
        sub_b, all_top_idx, all_top_val, per_expert_scale, Float32(1.0),
        sc_counts, sc_idx, sc_weights, num_experts, 8,
    )

    var gu_p = alloc[UInt8](num_experts * gate_up_rows * half_in)
    var gu_s = alloc[Float32](num_experts * gate_up_rows * c_in_blocks)
    var dn_p = alloc[UInt8](num_experts * embed_dim * down_half_in)
    var dn_s = alloc[Float32](num_experts * embed_dim * down_blocks)
    for i in range(num_experts * gate_up_rows * half_in):
        gu_p[i] = UInt8((i * 11 + 5) & 255)
    for i in range(num_experts * gate_up_rows * c_in_blocks):
        gu_s[i] = Float32(0.012)
    for i in range(num_experts * embed_dim * down_half_in):
        dn_p[i] = UInt8((i * 19 + 9) & 255)
    for i in range(num_experts * embed_dim * down_blocks):
        dn_s[i] = Float32(0.011)

    var batch_in_deint = alloc[Float32](sub_b * embed_dim)
    for i in range(sub_b * embed_dim):
        batch_in_deint[i] = Float32(((i * 29) % 23) - 11) * 0.04

    var bucket_counts = alloc[Int](num_experts)
    var bucket_toks = alloc[Int](num_experts * 5)
    var bucket_weights = alloc[Float32](num_experts * 5)
    var unique_experts = alloc[Int](num_experts)
    var unique_base_slot = alloc[Int](num_experts)
    var gu_out_scratch = alloc[Float32](40 * gate_up_rows)
    var act_scratch = alloc[Float32](40 * expert_mlp_dim)
    var act_deint_scratch = alloc[Float32](40 * expert_mlp_dim)
    var out_mono = alloc[Float32](sub_b * embed_dim)
    var out_split = alloc[Float32](sub_b * embed_dim)

    run_moe_coalesced_core(
        sub_b, gu_p, gu_s, dn_p, dn_s, batch_in_deint,
        sc_counts, sc_idx, sc_weights,
        bucket_counts, bucket_toks, bucket_weights,
        unique_experts, unique_base_slot,
        gu_out_scratch, act_scratch, act_deint_scratch, out_mono,
        embed_dim, expert_mlp_dim, num_experts, num_tasks,
    )

    var num_unique = run_moe_coalesced_stage1(
        sub_b, gu_p, gu_s, batch_in_deint,
        sc_counts, sc_idx, sc_weights,
        bucket_counts, bucket_toks, bucket_weights,
        unique_experts, unique_base_slot,
        gu_out_scratch, act_scratch, act_deint_scratch,
        embed_dim, expert_mlp_dim, num_experts, num_tasks,
    )
    run_moe_coalesced_stage2(
        sub_b, num_unique, dn_p, dn_s,
        bucket_counts, bucket_toks, bucket_weights,
        unique_experts, unique_base_slot,
        act_deint_scratch, out_split,
        embed_dim, expert_mlp_dim, num_tasks,
    )

    for i in range(sub_b * embed_dim):
        if out_mono[i] != out_split[i]:
            raise Error("Mismatch between monolithic and split MoE at index " + String(i))

    print("  [PASS] Split Stage 1 + Stage 2 (num_unique=" + String(num_unique) + ") is 100% bit-for-bit identical to monolithic MoE!")

    all_top_idx.free()
    all_top_val.free()
    per_expert_scale.free()
    sc_counts.free()
    sc_idx.free()
    sc_weights.free()
    gu_p.free()
    gu_s.free()
    dn_p.free()
    dn_s.free()
    batch_in_deint.free()
    bucket_counts.free()
    bucket_toks.free()
    bucket_weights.free()
    unique_experts.free()
    unique_base_slot.free()
    gu_out_scratch.free()
    act_scratch.free()
    act_deint_scratch.free()
    out_mono.free()
    out_split.free()
    print("=== PASS ===")


def main() raises:
    test_split_moe_equivalence()
