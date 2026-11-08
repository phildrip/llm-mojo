from std.memory import UnsafePointer, alloc
from std.math import abs
import std.time
from diffusion_kernels import (
    compute_moe_lod_k,
    compute_moe_lod_top_p,
    apply_moe_lod_and_set_cover,
    run_diffusion_moe_experts_parallel,
)


def main() raises:
    print("=== Test & Benchmark: Noise-Scheduled MoE LOD & Subbatch Set-Cover (llm-mojo-2tc.5) ===")
    comptime CANVAS_LEN = 256
    comptime EMBED_DIM = 2816
    comptime EXPERT_MLP_DIM = 704
    comptime TOP_K = 8

    # 1. Test LOD Schedule transitions
    var k0 = compute_moe_lod_k(0, 10.2)
    var p0 = compute_moe_lod_top_p(0, 10.2)
    var k1 = compute_moe_lod_k(3, 1.5)
    var p1 = compute_moe_lod_top_p(3, 1.5)
    var k2 = compute_moe_lod_k(8, 0.25)
    var p2 = compute_moe_lod_top_p(8, 0.25)
    print("  [LOD Schedule] Phase 1 (step=0, ent=10.2): K=", k0, "top_p=", p0, "| Phase 2 (ent=1.5): K=", k1, "top_p=", p1, "| Phase 3 (ent=0.25): K=", k2, "top_p=", p2)
    if k0 != 2 or k1 != 4 or k2 != 8:
        raise Error("MoE LOD schedule returned unexpected Top-K!")

    # 2. Test apply_moe_lod_and_set_cover energy conservation and singleton secondary-slot pruning
    var top_idx = alloc[Int](CANVAS_LEN * TOP_K)
    var top_val = alloc[Float32](CANVAS_LEN * TOP_K)
    var top_weights_base = alloc[Float32](CANVAS_LEN * TOP_K)
    var top_weights_lod = alloc[Float32](CANVAS_LEN * TOP_K)
    var router_scale = alloc[Float32](128)
    for e in range(128):
        router_scale[e] = Float32(1.0 + Float32(e % 5) * 0.1)

    for b in range(CANVAS_LEN):
        for s in range(TOP_K):
            # Slot 0 clusters on experts 0..31; secondary slots spread across 0..127 with some singletons
            if s == 0:
                top_idx[b * TOP_K + s] = b % 32
                top_val[b * TOP_K + s] = Float32(0.45)
            elif s == 1 and b == 0:
                # Expert 127 is a singleton on token 0, slot 1!
                top_idx[b * TOP_K + s] = 127
                top_val[b * TOP_K + s] = Float32(0.18)
            else:
                top_idx[b * TOP_K + s] = (b * 3 + s * 7) % 96
                top_val[b * TOP_K + s] = Float32(0.55) / Float32(TOP_K - 1)

    # Compute baseline Top-8 weights
    for b in range(CANVAS_LEN):
        var sum_v: Float32 = 0.0
        for s in range(TOP_K):
            sum_v += top_val[b * TOP_K + s]
        for s in range(TOP_K):
            var idx = top_idx[b * TOP_K + s]
            top_weights_base[b * TOP_K + s] = (top_val[b * TOP_K + s] / sum_v) * router_scale[idx]

    # Compute Phase-1 LOD (K=2, top_p=0.65) + Set-Cover weights
    apply_moe_lod_and_set_cover(
        top_idx, top_val, router_scale, top_weights_lod, CANVAS_LEN, TOP_K, 2, 0.65
    )

    # Verify singleton expert 127 on (b=0, s=1) was pruned and folded into slot 0!
    if top_weights_lod[0 * TOP_K + 1] != 0.0:
        raise Error("Singleton secondary expert 127 on (b=0, s=1) was not pruned!")
    var expected_b0_s0 = Float32(1.0) * router_scale[top_idx[0]]
    if abs(top_weights_lod[0 * TOP_K + 0] - expected_b0_s0) > 1e-5:
        raise Error("Folded singleton probability mass did not conserve energy on slot 0!")

    # 3. Benchmark Full 128-Expert MoE Layer: Top-8 Baseline vs Phase-1 LOD (K=2)
    comptime GU_DIM = EXPERT_MLP_DIM * 2
    var gu_packed = alloc[UInt8](128 * GU_DIM * (EMBED_DIM // 2))
    var gu_scales = alloc[Float32](128 * GU_DIM * (EMBED_DIM // 32))
    var dn_packed = alloc[UInt8](128 * EMBED_DIM * (EXPERT_MLP_DIM // 2))
    var dn_scales = alloc[Float32](128 * EMBED_DIM * (EXPERT_MLP_DIM // 32))
    for i in range(128 * GU_DIM * (EMBED_DIM // 2)):
        gu_packed[i] = UInt8((i * 17 + 5) & 0xFF)
    for i in range(128 * GU_DIM * (EMBED_DIM // 32)):
        gu_scales[i] = 0.01
    for i in range(128 * EMBED_DIM * (EXPERT_MLP_DIM // 2)):
        dn_packed[i] = UInt8((i * 23 + 3) & 0xFF)
    for i in range(128 * EMBED_DIM * (EXPERT_MLP_DIM // 32)):
        dn_scales[i] = 0.01

    var moe_in = alloc[Float16](CANVAS_LEN * EMBED_DIM)
    var moe_out = alloc[Float16](CANVAS_LEN * EMBED_DIM)
    for i in range(CANVAS_LEN * EMBED_DIM):
        moe_in[i] = Float16(Float32((i % 31) - 15) * 0.05)

    var t0_top8 = std.time.perf_counter()
    run_diffusion_moe_experts_parallel(
        gu_packed, gu_scales, dn_packed, dn_scales, moe_in, top_idx, top_weights_base, moe_out,
        CANVAS_LEN, EMBED_DIM, EXPERT_MLP_DIM, TOP_K, 8
    )
    var dt_top8_ms = (std.time.perf_counter() - t0_top8) * 1000.0

    var t0_lod2 = std.time.perf_counter()
    run_diffusion_moe_experts_parallel(
        gu_packed, gu_scales, dn_packed, dn_scales, moe_in, top_idx, top_weights_lod, moe_out,
        CANVAS_LEN, EMBED_DIM, EXPERT_MLP_DIM, TOP_K, 8
    )
    var dt_lod2_ms = (std.time.perf_counter() - t0_lod2) * 1000.0

    print("  [128-Expert MoE Benchmark] Top-8 Baseline:", dt_top8_ms, "ms -> Phase-1 LOD (K=2 + Set-Cover):", dt_lod2_ms, "ms (", dt_top8_ms / dt_lod2_ms, "x speedup)")
    if dt_lod2_ms >= dt_top8_ms:
        raise Error("Expected Phase-1 MoE LOD to be faster than Top-8 baseline!")

    top_idx.free()
    top_val.free()
    top_weights_base.free()
    top_weights_lod.free()
    router_scale.free()
    gu_packed.free()
    gu_scales.free()
    dn_packed.free()
    dn_scales.free()
    moe_in.free()
    moe_out.free()
    print("=== ALL TESTS PASSED! ===")
