from std.memory import alloc
from std.algorithm import parallelize
from std.math import sqrt, exp, abs
import std.time
from moe_kernels import (
    gelu_and_mul_moe,
    compute_moe_lod_weights,
    run_moe_experts_stage1_8core,
    run_moe_experts_stage2_8core,
)


def main() raises:
    print("==================================================================")
    print("   TDD & BENCHMARK: Hierarchical LOD Top-p & 8-Core MoE Slicing   ")
    print("==================================================================")

    # Test 1: Verify LOD Top-p routing & energy renormalization
    var top_idx = alloc[Int](8)
    var top_val = alloc[Float32](8)
    var per_scale = alloc[Float32](128)
    var act_idx = alloc[Int](8)
    var act_w = alloc[Float32](8)

    for i in range(128):
        per_scale[i] = 1.0

    # Synthetic sorted router probabilities: [0.42, 0.25, 0.15, 0.08, 0.04, 0.03, 0.02, 0.01]
    top_idx[0] = 10; top_val[0] = 0.42
    top_idx[1] = 25; top_val[1] = 0.25
    top_idx[2] = 3;  top_val[2] = 0.15
    top_idx[3] = 77; top_val[3] = 0.08
    top_idx[4] = 12; top_val[4] = 0.04
    top_idx[5] = 99; top_val[5] = 0.03
    top_idx[6] = 41; top_val[6] = 0.02
    top_idx[7] = 8;  top_val[7] = 0.01

    var c_default = compute_moe_lod_weights(top_idx, top_val, per_scale, 1.0, act_idx, act_w, 8)
    print("  [Test 1a] moe_top_p=1.0 active experts:", c_default, "(expected 4 with prob >= 0.05)")
    if c_default != 4:
        raise Error("Expected 4 active experts for moe_top_p=1.0, got " + String(c_default))

    var c_80 = compute_moe_lod_weights(top_idx, top_val, per_scale, 0.80, act_idx, act_w, 8)
    var w_sum_80: Float32 = 0.0
    for i in range(c_80):
        w_sum_80 += act_w[i]
    print("  [Test 1b] moe_top_p=0.80 active experts:", c_80, "(expected 3: 0.42+0.25+0.15=0.82), renorm sum:", w_sum_80)
    if c_80 != 3 or abs(w_sum_80 - 1.0) > 1e-5:
        raise Error("LOD Top-p=0.80 failed count or renormalization check!")

    # Test 2: Compare 4-thread per-expert baseline vs production 8-core row-sliced MoE kernels
    var embed_dim = 2816
    var expert_mlp_dim = 704
    var num_test_experts = 8
    var gate_up_rows = expert_mlp_dim * 2
    var gate_up_packed = alloc[UInt8](num_test_experts * gate_up_rows * (embed_dim // 2))
    var gate_up_scales = alloc[Float32](num_test_experts * gate_up_rows * (embed_dim // 32))
    var down_packed = alloc[UInt8](num_test_experts * embed_dim * (expert_mlp_dim // 2))
    var down_scales = alloc[Float32](num_test_experts * embed_dim * (expert_mlp_dim // 32))
    var moe_in = alloc[Float32](embed_dim)

    for i in range(embed_dim):
        moe_in[i] = Float32((i % 31) - 15) * 0.03
    for i in range(num_test_experts * gate_up_rows * (embed_dim // 2)):
        gate_up_packed[i] = UInt8((i * 13 + 7) & 255)
    for i in range(num_test_experts * gate_up_rows * (embed_dim // 32)):
        gate_up_scales[i] = Float32(((i % 19) + 1)) * 0.002
    for i in range(num_test_experts * embed_dim * (expert_mlp_dim // 2)):
        down_packed[i] = UInt8((i * 29 + 3) & 255)
    for i in range(num_test_experts * embed_dim * (expert_mlp_dim // 32)):
        down_scales[i] = Float32(((i % 17) + 1)) * 0.002

    for i in range(5):
        act_idx[i] = i
        act_w[i] = Float32(0.20)
    var active_count = 5

    var ref_moe_out = alloc[Float32](embed_dim)
    var opt_moe_out = alloc[Float32](embed_dim)
    var gate_up_out_all = alloc[Float32](8 * gate_up_rows)
    var act_all = alloc[Float32](8 * expert_mlp_dim)
    var down_out_all = alloc[Float32](8 * embed_dim)
    var x_deint = alloc[Float32](embed_dim)
    var act_deint = alloc[Float32](8 * expert_mlp_dim)

    var iters = 40
    var t0 = std.time.perf_counter()
    for _ in range(iters):
        for b in range(embed_dim // 32):
            var v0 = moe_in.load[16](b * 32)
            var v1 = moe_in.load[16](b * 32 + 16)
            x_deint.store[16](b * 16, v0.deinterleave()[0].join(v1.deinterleave()[0]))
            (x_deint + embed_dim // 2).store[16](b * 16, v0.deinterleave()[1].join(v1.deinterleave()[1]))

        def ref_worker(exp_slot: Int) capturing -> None:
            if exp_slot >= active_count:
                return
            var e = act_idx[exp_slot]
            var gu_p = gate_up_packed + e * (gate_up_rows * embed_dim // 2)
            var gu_s = gate_up_scales + e * (gate_up_rows * embed_dim // 32)
            var gu_out = gate_up_out_all + exp_slot * gate_up_rows
            var xe_ptr = x_deint
            var xo_ptr = x_deint + embed_dim // 2
            for r in range(gate_up_rows):
                var acc = SIMD[DType.float32, 16](0.0)
                var rp = r * (embed_dim // 2)
                var rs = r * (embed_dim // 32)
                for b in range(embed_dim // 32):
                    var sc = gu_s[rs + b]
                    var bytes = gu_p.load[16](rp + b * 16)
                    var we = ((bytes & 15).cast[DType.int8]() - 8).cast[DType.float32]()
                    var wo = ((bytes >> 4).cast[DType.int8]() - 8).cast[DType.float32]()
                    acc = acc + (we * xe_ptr.load[16](b * 16) + wo * xo_ptr.load[16](b * 16)) * sc
                gu_out[r] = acc.reduce_add()
            var act_ptr = act_all + exp_slot * expert_mlp_dim
            gelu_and_mul_moe(gu_out, gu_out + expert_mlp_dim, act_ptr, expert_mlp_dim)

            var ad_e = act_deint + exp_slot * expert_mlp_dim
            var ad_o = ad_e + expert_mlp_dim // 2
            for b in range(expert_mlp_dim // 32):
                var v0 = act_ptr.load[16](b * 32)
                var v1 = act_ptr.load[16](b * 32 + 16)
                ad_e.store[16](b * 16, v0.deinterleave()[0].join(v1.deinterleave()[0]))
                ad_o.store[16](b * 16, v0.deinterleave()[1].join(v1.deinterleave()[1]))

            var dn_p = down_packed + e * (embed_dim * expert_mlp_dim // 2)
            var dn_s = down_scales + e * (embed_dim * expert_mlp_dim // 32)
            var dn_out = down_out_all + exp_slot * embed_dim
            for r in range(embed_dim):
                var acc = SIMD[DType.float32, 16](0.0)
                var rp = r * (expert_mlp_dim // 2)
                var rs = r * (expert_mlp_dim // 32)
                for b in range(expert_mlp_dim // 32):
                    var sc = dn_s[rs + b]
                    var bytes = dn_p.load[16](rp + b * 16)
                    var we = ((bytes & 15).cast[DType.int8]() - 8).cast[DType.float32]()
                    var wo = ((bytes >> 4).cast[DType.int8]() - 8).cast[DType.float32]()
                    acc = acc + (we * ad_e.load[16](b * 16) + wo * ad_o.load[16](b * 16)) * sc
                dn_out[r] = acc.reduce_add()

        parallelize[ref_worker](4)
        def ref_worker_2(h: Int) capturing -> None:
            ref_worker(h + 4)
        parallelize[ref_worker_2](4)

        for j in range(embed_dim):
            var s: Float32 = 0.0
            for exp_slot in range(active_count):
                s += down_out_all[exp_slot * embed_dim + j] * act_w[exp_slot]
            ref_moe_out[j] = s
    var t1 = std.time.perf_counter()

    for _ in range(iters):
        run_moe_experts_stage1_8core(
            gate_up_packed,
            gate_up_scales,
            moe_in,
            x_deint,
            act_idx,
            active_count,
            gate_up_out_all,
            act_all,
            embed_dim,
            expert_mlp_dim,
            8,
        )
        run_moe_experts_stage2_8core(
            down_packed,
            down_scales,
            act_all,
            act_deint,
            act_idx,
            act_w,
            active_count,
            opt_moe_out,
            embed_dim,
            expert_mlp_dim,
            8,
        )
    var t2 = std.time.perf_counter()

    var max_err: Float32 = 0.0
    for j in range(embed_dim):
        var diff = abs(ref_moe_out[j] - opt_moe_out[j])
        if diff > max_err:
            max_err = diff

    var ref_ms = (t1 - t0) * 1000.0 / Float64(iters)
    var opt_ms = (t2 - t1) * 1000.0 / Float64(iters)
    print("  [Test 2] 5-Expert MoE Layer Max Error:", max_err)
    print("  4-Thread Split Baseline:  ", ref_ms, "ms/layer (", ref_ms * 30.0, "ms / 30 layers)")
    print("  8-Core Row-Sliced Coalesc:", opt_ms, "ms/layer (", opt_ms * 30.0, "ms / 30 layers)")
    print("  Speedup factor:           ", ref_ms / opt_ms, "x")

    if max_err > 1e-5:
        raise Error("MoE 8-core row-sliced coalesced output mismatch: " + String(max_err))
    print("  [PASS] Hierarchical LOD Top-p & 8-Core Coalesced MoE Verified!")

    top_idx.free()
    top_val.free()
    per_scale.free()
    act_idx.free()
    act_w.free()
    gate_up_packed.free()
    gate_up_scales.free()
    down_packed.free()
    down_scales.free()
    moe_in.free()
    ref_moe_out.free()
    opt_moe_out.free()
    gate_up_out_all.free()
    act_all.free()
    down_out_all.free()
    x_deint.free()
    act_deint.free()
