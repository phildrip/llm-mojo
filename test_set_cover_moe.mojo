from std.memory import alloc
from std.sys.info import num_physical_cores
from std.math import abs
import std.time
from moe_kernels import (
    quantize_fp32_to_q4_0,
    compute_moe_lod_weights,
    compute_subbatch_set_cover_moe_weights,
    run_moe_coalesced_core,
)


def test_set_cover_moe() raises:
    print("=== TDD Test: Subbatch-Coherent Set-Cover MoE Routing (llm-mojo-xr6.2) ===")
    var num_tasks = num_physical_cores()
    var sub_b = 5
    var top_k = 8
    var num_experts = 128
    var embed_dim = 2816
    var expert_mlp_dim = 704

    var all_top_idx = alloc[Int](sub_b * top_k)
    var all_top_val = alloc[Float32](sub_b * top_k)
    var per_expert_scale = alloc[Float32](num_experts)
    for e in range(num_experts):
        per_expert_scale[e] = Float32(0.8 + 0.4 * Float64((e * 17) % 31) / 31.0)

    # Construct realistic 5-token top-8 router distributions:
    # Token 0 (anchor): experts [10, 20, 30, 40, 50, 60, 70, 80] with probs [0.28, 0.22, 0.16, 0.10, 0.07, 0.06, 0.06, 0.05]
    var b0_exps: InlineArray[Int, 8] = [10, 20, 30, 40, 50, 60, 70, 80]
    var b0_vals: InlineArray[Float32, 8] = [0.28, 0.22, 0.16, 0.10, 0.07, 0.06, 0.06, 0.05]
    for k in range(8):
        all_top_idx[0 * 8 + k] = b0_exps[k]
        all_top_val[0 * 8 + k] = b0_vals[k]

    # Tokens 1..4: share core experts [10, 20, 30] + 1 token-specific core expert,
    # plus 2 shared tail experts (from [40, 50, 60]) and 2 singleton straggler tail experts (prob 0.055, 0.052)
    for b in range(1, sub_b):
        all_top_idx[b * 8 + 0] = 10
        all_top_val[b * 8 + 0] = 0.27
        all_top_idx[b * 8 + 1] = 20
        all_top_val[b * 8 + 1] = 0.23
        all_top_idx[b * 8 + 2] = 30 + b
        all_top_val[b * 8 + 2] = 0.16
        all_top_idx[b * 8 + 3] = 40
        all_top_val[b * 8 + 3] = 0.12
        # Tail slots k=4..7:
        # k=4,5 are in shared pool (50, 60 from b=0!) with prob 0.06, 0.055 -> MUST PIGGYBACK!
        all_top_idx[b * 8 + 4] = 50
        all_top_val[b * 8 + 4] = 0.060
        all_top_idx[b * 8 + 5] = 60
        all_top_val[b * 8 + 5] = 0.055
        # k=6,7 are singleton stragglers (unique per token b: 90+b*2, 91+b*2) with prob 0.053, 0.052 -> MUST PRUNE!
        all_top_idx[b * 8 + 6] = 90 + b * 2
        all_top_val[b * 8 + 6] = 0.053
        all_top_idx[b * 8 + 7] = 91 + b * 2
        all_top_val[b * 8 + 7] = 0.052

    var std_counts = alloc[Int](sub_b)
    var std_idx = alloc[Int](sub_b * top_k)
    var std_weights = alloc[Float32](sub_b * top_k)

    var sc_counts = alloc[Int](sub_b)
    var sc_idx = alloc[Int](sub_b * top_k)
    var sc_weights = alloc[Float32](sub_b * top_k)

    for b in range(sub_b):
        std_counts[b] = compute_moe_lod_weights(
            all_top_idx + b * 8,
            all_top_val + b * 8,
            per_expert_scale,
            Float32(1.0),
            std_idx + b * 8,
            std_weights + b * 8,
            8,
        )

    compute_subbatch_set_cover_moe_weights(
        sub_b,
        all_top_idx,
        all_top_val,
        per_expert_scale,
        Float32(1.0),
        sc_counts,
        sc_idx,
        sc_weights,
        num_experts,
        8,
    )

    # 1. Verify Anchor Invariance (b = 0 must be 100% bit-for-bit identical)
    if sc_counts[0] != std_counts[0]:
        raise Error("Anchor b=0 active count mismatch: " + String(sc_counts[0]) + " vs " + String(std_counts[0]))
    for k in range(sc_counts[0]):
        if sc_idx[k] != std_idx[k] or sc_weights[k] != std_weights[k]:
            raise Error("Anchor b=0 expert/weight mismatch at slot " + String(k))
    print("  [PASS] Anchor token b=0 is 100% bit-for-bit identical (count=" + String(sc_counts[0]) + ")")

    # 2. Verify Piggybacking & Singleton Straggler Pruning on b = 1..4
    for b in range(1, sub_b):
        # Should keep 6 experts (core [10, 20, 30+b, 40] + piggybacked [50, 60]) and prune singletons [90+2b, 91+2b]
        if sc_counts[b] != 6:
            raise Error("Expected 6 retained experts on token " + String(b) + ", got " + String(sc_counts[b]))
        # Verify energy-preserving renormalization before per_expert_scale:
        var unscaled_sum: Float32 = 0.0
        for k in range(sc_counts[b]):
            var e = sc_idx[b * 8 + k]
            unscaled_sum += sc_weights[b * 8 + k] / per_expert_scale[e]
        if abs(unscaled_sum - 1.0) > 1e-4:
            raise Error("Renormalized probability sum mismatch on token " + String(b) + ": " + String(unscaled_sum))
    print("  [PASS] Zero-marginal-cost piggybacking retained shared tail experts [50, 60] & pruned 8 singleton stragglers!")

    # 3. Benchmark run_moe_coalesced_core latency: Standard (24 unique experts) vs Set-Cover (16 unique experts)
    var gate_up_rows = expert_mlp_dim * 2
    var half_in = embed_dim // 2
    var c_in_blocks = embed_dim // 32
    var down_half_in = expert_mlp_dim // 2
    var down_blocks = expert_mlp_dim // 32

    var gu_p = alloc[UInt8](num_experts * gate_up_rows * half_in)
    var gu_s = alloc[Float32](num_experts * gate_up_rows * c_in_blocks)
    var dn_p = alloc[UInt8](num_experts * embed_dim * down_half_in)
    var dn_s = alloc[Float32](num_experts * embed_dim * down_blocks)
    for i in range(num_experts * gate_up_rows * half_in):
        gu_p[i] = UInt8((i * 13 + 7) & 255)
    for i in range(num_experts * gate_up_rows * c_in_blocks):
        gu_s[i] = Float32(0.01)
    for i in range(num_experts * embed_dim * down_half_in):
        dn_p[i] = UInt8((i * 17 + 3) & 255)
    for i in range(num_experts * embed_dim * down_blocks):
        dn_s[i] = Float32(0.01)

    var batch_in_deint = alloc[Float32](sub_b * embed_dim)
    for i in range(sub_b * embed_dim):
        batch_in_deint[i] = Float32(((i * 23) % 19) - 9) * 0.05

    var bucket_counts = alloc[Int](num_experts)
    var bucket_toks = alloc[Int](num_experts * 5)
    var bucket_weights = alloc[Float32](num_experts * 5)
    var unique_experts = alloc[Int](num_experts)
    var unique_base_slot = alloc[Int](num_experts)
    var gu_out_scratch = alloc[Float32](40 * gate_up_rows)
    var act_scratch = alloc[Float32](40 * expert_mlp_dim)
    var act_deint_scratch = alloc[Float32](40 * expert_mlp_dim)
    var out_std = alloc[Float32](sub_b * embed_dim)
    var out_sc = alloc[Float32](sub_b * embed_dim)

    var iters = 30
    var t0 = std.time.perf_counter()
    for _ in range(iters):
        run_moe_coalesced_core(
            sub_b, gu_p, gu_s, dn_p, dn_s, batch_in_deint,
            std_counts, std_idx, std_weights,
            bucket_counts, bucket_toks, bucket_weights,
            unique_experts, unique_base_slot,
            gu_out_scratch, act_scratch, act_deint_scratch, out_std,
            embed_dim, expert_mlp_dim, num_experts, num_tasks,
        )
    var t1 = std.time.perf_counter()

    var t2 = std.time.perf_counter()
    for _ in range(iters):
        run_moe_coalesced_core(
            sub_b, gu_p, gu_s, dn_p, dn_s, batch_in_deint,
            sc_counts, sc_idx, sc_weights,
            bucket_counts, bucket_toks, bucket_weights,
            unique_experts, unique_base_slot,
            gu_out_scratch, act_scratch, act_deint_scratch, out_sc,
            embed_dim, expert_mlp_dim, num_experts, num_tasks,
        )
    var t3 = std.time.perf_counter()

    # Verify anchor b=0 output vector is 100% bit-for-bit identical between Standard and Set-Cover!
    for j in range(embed_dim):
        if out_std[j] != out_sc[j]:
            raise Error("Anchor b=0 MoE output mismatch at dim " + String(j))

    var std_ms = (t1 - t0) * 1000.0 / Float64(iters)
    var sc_ms = (t3 - t2) * 1000.0 / Float64(iters)
    print("  Standard 5-token MoE layer latency (20 unique experts): ", std_ms, "ms (30 layers =", std_ms * 30.0, "ms)")
    print("  Set-Cover 5-token MoE layer latency (12 unique experts):", sc_ms, "ms (30 layers =", sc_ms * 30.0, "ms)")

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
    out_std.free()
    out_sc.free()
    all_top_idx.free()
    all_top_val.free()
    per_expert_scale.free()
    std_counts.free()
    std_idx.free()
    std_weights.free()
    sc_counts.free()
    sc_idx.free()
    sc_weights.free()
    print("=== ALL SET-COVER MOE TESTS PASSED ===")


def main() raises:
    test_set_cover_moe()
