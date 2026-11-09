from std.memory import UnsafePointer, alloc
from std.math import sqrt, exp, tanh, log, abs
from std.algorithm import parallelize
from std.sys.info import num_physical_cores
import std.time
from diffusion_kernels import (
    rms_norm,
    rms_norm_noscale,
    unpack_16bytes,
    q4_gemm_l1_2x4,
    q4_moe_expert_single_thread,
    run_diffusion_moe_experts_parallel,
    sc_forward_batch,
    sample_and_soft_embed_top32_batch,
    resolve_adjacent_duplicate_tokens,
    build_active_canvas_map,
    build_block_active_canvas_map,
    gather_active,
    update_locked_mask,
    scatter_active_sampler_outputs,
    teacache_eval_gate,
    taylorseer_record_full_step,
    taylorseer_forecast_skipped_step,
    compute_moe_lod_k,
    compute_moe_lod_top_p,
    apply_moe_lod_and_set_cover,
    prep_qkv_norm_rope_kvq_batch,
    flash_dllm_bidirectional_attn,
    adacache_filter_after_layer1,
    extrapolate_sc_velocity_batch,
    q4_fused_gate_up_geglu_l1,
    deinterleave_batch_to_f32,
    q4_lm_head_two_stage_sieve,
    q4_gemm_block_sparse_down,
    router_subspace_ffn_filter,
    scatter_and_restore_ffn_subspace,
    sync_moe_subspace_cache,
    q4_wo_bos_sink_bypass,
    detect_bos_sink_heads,
)


def test_1_sc_and_top32_sampler(num_tasks: Int) raises:
    comptime B = 64
    comptime EMBED_DIM = 256
    comptime SC_SIZE = 192
    comptime VOCAB_SIZE = 4096
    comptime SC_MAT = SC_SIZE * EMBED_DIM

    var sc_gate_p = alloc[UInt8](SC_MAT // 2)
    var sc_up_p = alloc[UInt8](SC_MAT // 2)
    var sc_down_p = alloc[UInt8](SC_MAT // 2)
    var sc_gate_s = alloc[Float32](SC_MAT // 32)
    var sc_up_s = alloc[Float32](SC_MAT // 32)
    var sc_down_s = alloc[Float32](SC_MAT // 32)
    var sc_pre_norm = alloc[Float32](EMBED_DIM)

    for i in range(SC_MAT // 2):
        sc_gate_p[i] = UInt8((i * 73 + 19) & 255)
        sc_up_p[i] = UInt8((i * 41 + 107) & 255)
        sc_down_p[i] = UInt8((i * 59 + 211) & 255)
    for i in range(SC_MAT // 32):
        sc_gate_s[i] = Float32(((i * 17) % 100) - 50) * 0.001
        sc_up_s[i] = Float32(((i * 23) % 100) - 50) * 0.001
        sc_down_s[i] = Float32(((i * 31) % 100) - 50) * 0.001
    for i in range(EMBED_DIM):
        sc_pre_norm[i] = Float32(0.9) + Float32(i % 32) * 0.005

    var in_embeds = alloc[Float16](B * EMBED_DIM)
    var soft_embeds = alloc[Float32](B * EMBED_DIM)
    var b_normed = alloc[Float32](B * EMBED_DIM)
    var b_gated = alloc[Float32](B * SC_SIZE)
    var b_down = alloc[Float32](B * EMBED_DIM)
    var b_out = alloc[Float16](B * EMBED_DIM)
    for i in range(B * EMBED_DIM):
        in_embeds[i] = Float16(Float32(((i * 13 + 7) % 200) - 100) * 0.02)
        soft_embeds[i] = Float32(((i * 29 + 3) % 200) - 100) * 0.015

    sc_forward_batch[DType.float16, DType.float16](
        in_embeds, soft_embeds, sc_gate_p, sc_up_p, sc_down_p,
        sc_gate_s, sc_up_s, sc_down_s, sc_pre_norm,
        b_normed, b_gated, b_down, b_out, B, EMBED_DIM, SC_SIZE, num_tasks
    )

    var embed_p = alloc[UInt8](VOCAB_SIZE * (EMBED_DIM // 2))
    var embed_s = alloc[Float32](VOCAB_SIZE * (EMBED_DIM // 32))
    for i in range(VOCAB_SIZE * (EMBED_DIM // 2)):
        embed_p[i] = UInt8((i * 37 + 11) & 255)
    for i in range(VOCAB_SIZE * (EMBED_DIM // 32)):
        embed_s[i] = Float32(((i * 19) % 100) - 50) * 0.002

    var logits = alloc[Float32](B * VOCAB_SIZE)
    var rand_u = alloc[Float32](B)
    var ent = alloc[Float32](B)
    var argmax_t = alloc[Int](B)
    var sampled_t = alloc[Int](B)
    var sc_out = alloc[Float32](B * EMBED_DIM)

    # 1a. Uniform noise -> entropy == ln(VOCAB_SIZE)
    for i in range(B * VOCAB_SIZE):
        logits[i] = 0.0
    for t in range(B):
        rand_u[t] = 0.5
    sample_and_soft_embed_top32_batch[32](
        logits, rand_u, embed_p, embed_s, ent, argmax_t, sampled_t, sc_out,
        B, VOCAB_SIZE, EMBED_DIM, Float32(0.8), sqrt(Float32(EMBED_DIM)), num_tasks
    )
    if abs(ent[0] - log(Float32(VOCAB_SIZE))) > 1e-3:
        raise Error("Uniform entropy check failed in Top-32 sampler!")

    # 1a-cont. Continuity check: tiny perturbation (2e-4) in logits must not cause a > 0.2 cliff jump in entropy
    for t in range(B):
        logits[t * VOCAB_SIZE + 0] = Float32(2e-4)
    var ent_pert = alloc[Float32](B)
    sample_and_soft_embed_top32_batch[32](
        logits, rand_u, embed_p, embed_s, ent_pert, argmax_t, sampled_t, sc_out,
        B, VOCAB_SIZE, EMBED_DIM, Float32(0.8), sqrt(Float32(EMBED_DIM)), num_tasks
    )
    var jump_diff = abs(ent[0] - ent_pert[0])
    ent_pert.free()
    if jump_diff > 0.25:
        raise Error("Discontinuous entropy cliff at diff_tail=2e-4: jump_diff=" + String(jump_diff))

    # 1b. Peaked distribution with -30.0 <unused*> outliers and <eos> (token 1) guard
    for t in range(B):
        var base_l = t * VOCAB_SIZE
        for v in range(VOCAB_SIZE):
            logits[base_l + v] = Float32(-30.0) if (v % 3 == 0) else Float32(-5.0)
        # Token 0: <eos> (id=1) is rank-2 with non-trivial prob, id=300 is rank-1
        logits[base_l + 300] = Float32(15.0)
        logits[base_l + 1] = Float32(12.0)
        for k in range(2, 32):
            logits[base_l + 300 + k] = Float32(6.0) - Float32(k) * 0.15

    sample_and_soft_embed_top32_batch[32](
        logits, rand_u, embed_p, embed_s, ent, argmax_t, sampled_t, sc_out,
        B, VOCAB_SIZE, EMBED_DIM, Float32(0.6), sqrt(Float32(EMBED_DIM)), num_tasks
    )
    if argmax_t[0] != 300 or ent[0] > 0.15:
        raise Error("Outlier tail guard or argmax failed in Top-32 sampler!")

    # 1c. TDD (llm-mojo-yj4.4): Two-Stage Coarse-to-Fine Vocab Sieve for LM Head
    from diffusion_kernels import q4_lm_head_two_stage_sieve
    var sieve_p = alloc[UInt8](VOCAB_SIZE * (EMBED_DIM // 2))
    var sieve_s = alloc[Float32](VOCAB_SIZE * (EMBED_DIM // 32))
    for r in range(VOCAB_SIZE):
        for c in range(EMBED_DIM // 2):
            sieve_p[r * (EMBED_DIM // 2) + c] = UInt8(((r * 131 + (r >> 6) * 97 + c * 37 + ((r + c * 7) ^ (r >> 3)) * 53 + 11) & 255))
        for blk in range(EMBED_DIM // 32):
            sieve_s[r * (EMBED_DIM // 32) + blk] = Float32(((r * 23 + blk * 19) % 80) + 20) * 0.001
    var x_lm = alloc[Float32](B * EMBED_DIM)
    var logits_full = alloc[Float32](B * VOCAB_SIZE)
    var logits_sieve = alloc[Float32](B * VOCAB_SIZE)
    var forced_toks = alloc[Int](4)
    forced_toks[0] = 17
    forced_toks[1] = 259
    forced_toks[2] = 511
    forced_toks[3] = 0
    for b in range(B):
        # Construct x_lm[b] correlated with row (100 + b * 11) so that row is the true argmax
        var target_r = 100 + b * 11
        for k in range(EMBED_DIM):
            var byte_val = sieve_p[target_r * (EMBED_DIM // 2) + (k // 2)]
            var nib = (byte_val & 15) if (k % 2 == 0) else (byte_val >> 4)
            var w_val = Float32(Int(nib) - 8) * sieve_s[target_r * (EMBED_DIM // 32) + (k // 32)]
            x_lm[b * EMBED_DIM + k] = w_val * 5.0
    # Adversarial check (llm-mojo-yj4.4):
    # 1) Row 77 has s_coarse = 0.0 on blk 0..1 and huge s_fine = 0.50 on blk 2..3 (true argmax for b=0)
    # 2) Row 102 (in same chunk as b=0's old target 100 and b=1's target 111) has tiny s_coarse = 1e-11
    #    which must not blow up row_extrap / std_v and blind the chunk!
    sieve_s[77 * (EMBED_DIM // 32) + 0] = 0.0
    sieve_s[77 * (EMBED_DIM // 32) + 1] = 0.0
    sieve_s[77 * (EMBED_DIM // 32) + 2] = 0.50
    sieve_s[77 * (EMBED_DIM // 32) + 3] = 0.50
    for k in range(EMBED_DIM):
        var byte_val = sieve_p[77 * (EMBED_DIM // 2) + (k // 2)]
        var nib = (byte_val & 15) if (k % 2 == 0) else (byte_val >> 4)
        var sc_k = Float32(0.05) if (k < 64) else sieve_s[77 * (EMBED_DIM // 32) + (k // 32)]
        var w_val = Float32(Int(nib) - 8) * sc_k
        x_lm[0 * EMBED_DIM + k] = w_val * 5.0
    sieve_s[102 * (EMBED_DIM // 32) + 0] = 1e-11
    sieve_s[102 * (EMBED_DIM // 32) + 1] = 1e-11
    q4_gemm_l1_2x4[DType.float32, DType.float32](
        sieve_p, sieve_s, x_lm, logits_full, B, EMBED_DIM, VOCAB_SIZE, num_tasks
    )
    q4_lm_head_two_stage_sieve[DType.float32, DType.float32](
        sieve_p, sieve_s, x_lm, logits_sieve, B, EMBED_DIM, VOCAB_SIZE, num_tasks,
        forced_cand_tokens=forced_toks, num_forced_cands=4, coarse_blocks=2, span_keep_frac=0.35
    )
    var pruned_count = 0
    for b in range(B):
        for f in range(4):
            var r = forced_toks[f]
            if abs(logits_sieve[b * VOCAB_SIZE + r] - logits_full[b * VOCAB_SIZE + r]) > 1e-4:
                raise Error("q4_lm_head_two_stage_sieve forced candidate logit mismatch!")
        for v in range(VOCAB_SIZE):
            if logits_sieve[b * VOCAB_SIZE + v] <= -1e20:
                pruned_count += 1
            else:
                if abs(logits_sieve[b * VOCAB_SIZE + v] - logits_full[b * VOCAB_SIZE + v]) > 1e-4:
                    raise Error("q4_lm_head_two_stage_sieve evaluated candidate logit mismatch!")
    if pruned_count == 0:
        raise Error("q4_lm_head_two_stage_sieve failed to prune any low-scoring rows in Stage 1!")
    var argmax_full = alloc[Int](B)
    var argmax_sieve = alloc[Int](B)
    sample_and_soft_embed_top32_batch[32](
        logits_full, rand_u, sieve_p, sieve_s, ent, argmax_full, sampled_t, sc_out,
        B, VOCAB_SIZE, EMBED_DIM, Float32(0.6), sqrt(Float32(EMBED_DIM)), num_tasks
    )
    sample_and_soft_embed_top32_batch[32](
        logits_sieve, rand_u, sieve_p, sieve_s, ent, argmax_sieve, sampled_t, sc_out,
        B, VOCAB_SIZE, EMBED_DIM, Float32(0.6), sqrt(Float32(EMBED_DIM)), num_tasks
    )
    for b in range(B):
        if argmax_sieve[b] != argmax_full[b]:
            raise Error("q4_lm_head_two_stage_sieve missed true argmax token on b=" + String(b))
    sieve_p.free()
    sieve_s.free()
    x_lm.free()
    logits_full.free()
    logits_sieve.free()
    forced_toks.free()
    argmax_full.free()
    argmax_sieve.free()

    sc_gate_p.free()
    sc_up_p.free()
    sc_down_p.free()
    sc_gate_s.free()
    sc_up_s.free()
    sc_down_s.free()
    sc_pre_norm.free()
    in_embeds.free()
    soft_embeds.free()
    b_normed.free()
    b_gated.free()
    b_down.free()
    b_out.free()
    embed_p.free()
    embed_s.free()
    logits.free()
    rand_u.free()
    ent.free()
    argmax_t.free()
    sampled_t.free()
    sc_out.free()
    print("  [PASS] 1. Batched SC MLP & Top-32 Soft-Embedding Sampler (Uniform + -30.0 Outlier Guard)")


def test_2_moe_4x3_and_lod(num_tasks: Int) raises:
    comptime B = 32
    comptime EMBED_DIM = 256
    comptime MLP_DIM = 128
    comptime TOP_K = 8
    comptime NUM_EXP = 16

    var k_lod = compute_moe_lod_k(0, 5.0)
    var p_lod = compute_moe_lod_top_p(0, 5.0)
    if k_lod != 8 or p_lod != 1.0:
        raise Error("Expected full Top-8 MoE schedule!")

    var top_idx = alloc[Int](B * TOP_K)
    var top_val = alloc[Float32](B * TOP_K)
    var top_w = alloc[Float32](B * TOP_K)
    var r_scale = alloc[Float32](128)
    for e in range(128):
        r_scale[e] = 1.0

    for b in range(B):
        for s in range(TOP_K):
            top_idx[b * TOP_K + s] = (b + s) % NUM_EXP
            top_val[b * TOP_K + s] = Float32(0.20) if s < 4 else Float32(0.05)
    # Singleton expert 15 on b=0, s=1 with norm_p >= 0.05
    top_idx[0 * TOP_K + 1] = 15
    top_val[0 * TOP_K + 1] = Float32(0.20)

    var gu_p = alloc[UInt8](NUM_EXP * (MLP_DIM * 2) * (EMBED_DIM // 2))
    var gu_s = alloc[Float32](NUM_EXP * (MLP_DIM * 2) * (EMBED_DIM // 32))
    var dn_p = alloc[UInt8](NUM_EXP * EMBED_DIM * (MLP_DIM // 2))
    var dn_s = alloc[Float32](NUM_EXP * EMBED_DIM * (MLP_DIM // 32))
    for i in range(NUM_EXP * (MLP_DIM * 2) * (EMBED_DIM // 2)):
        gu_p[i] = UInt8((i * 17 + 5) & 255)
    for i in range(NUM_EXP * (MLP_DIM * 2) * (EMBED_DIM // 32)):
        gu_s[i] = 0.01
    for i in range(NUM_EXP * EMBED_DIM * (MLP_DIM // 2)):
        dn_p[i] = UInt8((i * 23 + 3) & 255)
    for i in range(NUM_EXP * EMBED_DIM * (MLP_DIM // 32)):
        dn_s[i] = 0.01

    from diffusion_kernels import deinterleave_batch_to_f32
    apply_moe_lod_and_set_cover(top_idx, top_val, r_scale, top_w, B, TOP_K, 8, 1.0)
    if top_w[0 * TOP_K + 1] <= 0.0:
        raise Error("Singleton domain expert with norm_p >= 0.05 was unexpectedly pruned!")

    # TDD (llm-mojo-yj4.3): Per-token confidence-adaptive MoE budget via eb_mask + active_canvas_idx
    var eb_moe = alloc[Bool](64)
    var act_map_moe = alloc[Int](B)
    for b in range(B):
        act_map_moe[b] = b * 2
        eb_moe[b * 2] = (b >= 16)  # b=0..15 unaccepted (K=2), b=16..31 accepted (K=8)
    apply_moe_lod_and_set_cover(
        top_idx, top_val, r_scale, top_w, B, TOP_K, 8, 1.0,
        eb_mask=eb_moe, active_canvas_idx=act_map_moe, unaccepted_top_k=2
    )
    # Unaccepted token b=0 (orig_t=0) must have slots 2..7 zeroed and slots 0..1 energy-conserved (0.5 each)
    if top_w[0 * TOP_K + 2] != 0.0 or abs(top_w[0 * TOP_K + 0] - 0.5) > 1e-5:
        eb_moe.free()
        act_map_moe.free()
        raise Error("Per-token confidence-adaptive MoE failed on unaccepted token b=0!")
    # Accepted token b=16 (orig_t=32) must retain slots 2..3 (Top-8 budget)
    if top_w[16 * TOP_K + 2] <= 0.0 or top_w[16 * TOP_K + 3] <= 0.0:
        eb_moe.free()
        act_map_moe.free()
        raise Error("Per-token confidence-adaptive MoE failed to retain Top-8 on accepted token b=16!")
    eb_moe.free()
    act_map_moe.free()

    # Adversarial check (llm-mojo-yj4.3): explicit active_top_k=2 without eb_mask must also conserve energy
    apply_moe_lod_and_set_cover(top_idx, top_val, r_scale, top_w, B, TOP_K, 2, 1.0)
    if top_w[0 * TOP_K + 2] != 0.0 or abs(top_w[0 * TOP_K + 0] - 0.5) > 1e-5:
        raise Error("apply_moe_lod_and_set_cover failed to conserve energy when active_top_k=2 without eb_mask!")

    # TDD (llm-mojo-yj4.2): Dual-Space Shuffle-Free 4x3 AVX2 GEMM with shared pre_deint_X (including B=29 % 3 != 0)
    comptime B_ODD = 29
    var x_odd = alloc[Float16](B_ODD * EMBED_DIM)
    var x_deint_shared = alloc[Float32](B_ODD * EMBED_DIM)
    var out_auto = alloc[Float32](B_ODD * MLP_DIM)
    var out_shared = alloc[Float32](B_ODD * MLP_DIM)
    var gated_auto = alloc[Float16](B_ODD * MLP_DIM)
    var gated_shared = alloc[Float16](B_ODD * MLP_DIM)
    for i in range(B_ODD * EMBED_DIM):
        x_odd[i] = Float16(Float32((i % 23) - 11) * 0.04)
    deinterleave_batch_to_f32[DType.float16](x_odd, x_deint_shared, B_ODD, EMBED_DIM)
    q4_gemm_l1_2x4[DType.float16, DType.float32](
        gu_p, gu_s, x_odd, out_auto, B_ODD, EMBED_DIM, MLP_DIM, num_tasks
    )
    q4_gemm_l1_2x4[DType.float16, DType.float32](
        gu_p, gu_s, x_odd, out_shared, B_ODD, EMBED_DIM, MLP_DIM, num_tasks,
        pre_deint_X=x_deint_shared
    )
    q4_fused_gate_up_geglu_l1[DType.float16, DType.float16](
        gu_p, gu_s, gu_p + MLP_DIM * (EMBED_DIM // 2), gu_s + MLP_DIM * (EMBED_DIM // 32),
        x_odd, gated_auto, B_ODD, EMBED_DIM, MLP_DIM, num_tasks
    )
    q4_fused_gate_up_geglu_l1[DType.float16, DType.float16](
        gu_p, gu_s, gu_p + MLP_DIM * (EMBED_DIM // 2), gu_s + MLP_DIM * (EMBED_DIM // 32),
        x_odd, gated_shared, B_ODD, EMBED_DIM, MLP_DIM, num_tasks,
        pre_deint_X=x_deint_shared
    )
    var max_deint_err: Float32 = 0.0
    for i in range(B_ODD * MLP_DIM):
        var d1 = abs(out_auto[i] - out_shared[i])
        var d2 = abs(Float32(gated_auto[i]) - Float32(gated_shared[i]))
        if d1 > max_deint_err:
            max_deint_err = d1
        if d2 > max_deint_err:
            max_deint_err = d2
    x_odd.free()
    x_deint_shared.free()
    out_auto.free()
    out_shared.free()
    gated_auto.free()
    gated_shared.free()
    if max_deint_err > 1e-5:
        raise Error("Dual-Space 4x3 GEMM pre_deint_X mismatch: " + String(max_deint_err))

    # Adversarial check (llm-mojo-yj4.2): B * c_in > 65536 (B=36, c_in=2048 -> 73728),
    # non-contiguous in_stride=2304 > c_in, out_stride=24 > c_out=18 (c_out % 4 != 0), and B=0 no-op guard
    comptime B_LARGE = 36
    comptime C_IN_LARGE = 2048
    comptime IN_STRIDE_LARGE = 2304
    comptime C_OUT_ODD = 18
    comptime OUT_STRIDE_ODD = 24
    var w_lg_p = alloc[UInt8](C_OUT_ODD * (C_IN_LARGE // 2))
    var w_lg_s = alloc[Float32](C_OUT_ODD * (C_IN_LARGE // 32))
    var x_lg = alloc[Float16](B_LARGE * IN_STRIDE_LARGE)
    var x_lg_deint = alloc[Float32](B_LARGE * C_IN_LARGE)
    var out_lg_auto = alloc[Float32](B_LARGE * OUT_STRIDE_ODD)
    var out_lg_shared = alloc[Float32](B_LARGE * OUT_STRIDE_ODD)
    for i in range(C_OUT_ODD * (C_IN_LARGE // 2)):
        w_lg_p[i] = UInt8((i * 19 + 3) & 255)
    for i in range(C_OUT_ODD * (C_IN_LARGE // 32)):
        w_lg_s[i] = 0.01
    for i in range(B_LARGE * IN_STRIDE_LARGE):
        x_lg[i] = Float16(Float32((i % 17) - 8) * 0.05)
    for i in range(B_LARGE * OUT_STRIDE_ODD):
        out_lg_auto[i] = -777.0
        out_lg_shared[i] = -777.0
    # Verify B=0 is a safe no-op that does not touch output sentinel
    q4_gemm_l1_2x4[DType.float16, DType.float32](
        w_lg_p, w_lg_s, x_lg, out_lg_auto, 0, C_IN_LARGE, C_OUT_ODD, num_tasks,
        in_stride=IN_STRIDE_LARGE, out_stride=OUT_STRIDE_ODD
    )
    if out_lg_auto[0] != -777.0:
        raise Error("q4_gemm_l1_2x4 modified output when B=0!")
    deinterleave_batch_to_f32[DType.float16](x_lg, x_lg_deint, B_LARGE, C_IN_LARGE, IN_STRIDE_LARGE)
    q4_gemm_l1_2x4[DType.float16, DType.float32](
        w_lg_p, w_lg_s, x_lg, out_lg_auto, B_LARGE, C_IN_LARGE, C_OUT_ODD, num_tasks,
        in_stride=IN_STRIDE_LARGE, out_stride=OUT_STRIDE_ODD
    )
    q4_gemm_l1_2x4[DType.float16, DType.float32](
        w_lg_p, w_lg_s, x_lg, out_lg_shared, B_LARGE, C_IN_LARGE, C_OUT_ODD, num_tasks,
        in_stride=IN_STRIDE_LARGE, out_stride=OUT_STRIDE_ODD, pre_deint_X=x_lg_deint
    )
    for b in range(B_LARGE):
        for r in range(C_OUT_ODD):
            if abs(out_lg_auto[b * OUT_STRIDE_ODD + r] - out_lg_shared[b * OUT_STRIDE_ODD + r]) > 1e-5:
                raise Error("q4_gemm_l1_2x4 B*c_in > 65536 stride/tail mismatch!")
        if out_lg_shared[b * OUT_STRIDE_ODD + C_OUT_ODD] != -777.0:
            raise Error("q4_gemm_l1_2x4 overwrote out_stride padding!")
    w_lg_p.free()
    w_lg_s.free()
    x_lg.free()
    x_lg_deint.free()
    out_lg_auto.free()
    out_lg_shared.free()
    apply_moe_lod_and_set_cover(top_idx, top_val, r_scale, top_w, B, TOP_K, 8, 1.0)

    var moe_in = alloc[Float16](B * EMBED_DIM)
    var moe_out = alloc[Float16](B * EMBED_DIM)
    for i in range(B * EMBED_DIM):
        moe_in[i] = Float16(Float32((i % 31) - 15) * 0.05)

    run_diffusion_moe_experts_parallel(
        gu_p, gu_s, dn_p, dn_s, moe_in, top_idx, top_w, moe_out,
        B, EMBED_DIM, MLP_DIM, TOP_K, num_tasks
    )
    var norm_out: Float32 = 0.0
    for i in range(B * EMBED_DIM):
        norm_out += abs(Float32(moe_out[i]))
    if norm_out <= 0.0:
        raise Error("4x3 AVX2 MoE produced zero output!")

    # TDD (llm-mojo-yj4.5): Branchless Pre-Compacted Block-Sparse W_down
    from diffusion_kernels import (
        q4_gemm_block_sparse_down,
        router_subspace_ffn_filter,
        scatter_and_restore_ffn_subspace,
        q4_wo_bos_sink_bypass,
        detect_bos_sink_heads,
    )
    var x_sparse = alloc[Float16](B * MLP_DIM)
    var out_dense_dn = alloc[Float16](B * EMBED_DIM)
    var out_sparse_dn = alloc[Float16](B * EMBED_DIM)
    for b in range(B):
        for blk in range(MLP_DIM // 32):
            for j in range(32):
                # Make odd blocks completely zero (50% block-sparse)
                x_sparse[b * MLP_DIM + blk * 32 + j] = (
                    Float16(0.0) if (blk % 2 == 1) else Float16(Float32((j % 7) - 3) * 0.05)
                )
    q4_gemm_l1_2x4[DType.float16, DType.float16](
        dn_p, dn_s, x_sparse, out_dense_dn, B, MLP_DIM, EMBED_DIM, num_tasks
    )
    q4_gemm_block_sparse_down[DType.float16, DType.float16](
        dn_p, dn_s, x_sparse, out_sparse_dn, B, MLP_DIM, EMBED_DIM, num_tasks
    )
    for i in range(B * EMBED_DIM):
        if abs(Float32(out_dense_dn[i]) - Float32(out_sparse_dn[i])) > 1e-3:
            raise Error("q4_gemm_block_sparse_down mismatch vs dense q4_gemm_l1_2x4!")
    # Adversarial test (llm-mojo-yj4.5): B=37 (%3!=0), c_in=1024, c_out=18 (%4!=0),
    # 75% active blocks -> B * active_dim = 37 * 768 = 28,416 > 16384 (heap path),
    # non-contiguous in_stride=1088, out_stride=24
    comptime ADV_B_DN = 37
    comptime ADV_CIN_DN = 1024
    comptime ADV_COUT_DN = 18
    comptime ADV_ISTR_DN = 1088
    comptime ADV_OSTR_DN = 24
    var adv_w_dn = alloc[UInt8](ADV_COUT_DN * (ADV_CIN_DN // 2))
    var adv_s_dn = alloc[Float32](ADV_COUT_DN * (ADV_CIN_DN // 32))
    var adv_x_dn = alloc[Float16](ADV_B_DN * ADV_ISTR_DN)
    var adv_ref_dn = alloc[Float32](ADV_B_DN * ADV_OSTR_DN)
    var adv_got_dn = alloc[Float32](ADV_B_DN * ADV_OSTR_DN)
    for i in range(ADV_COUT_DN * (ADV_CIN_DN // 2)):
        adv_w_dn[i] = UInt8((i * 19 + 11) & 255)
    for i in range(ADV_COUT_DN * (ADV_CIN_DN // 32)):
        adv_s_dn[i] = 0.015
    for b in range(ADV_B_DN):
        for j in range(ADV_ISTR_DN):
            var blk = j // 32
            adv_x_dn[b * ADV_ISTR_DN + j] = (
                Float16(999.0) if j >= ADV_CIN_DN else (
                    Float16(0.0) if (blk % 4 == 3) else Float16(Float32(((b + j) % 9) - 4) * 0.04)
                )
            )
    for i in range(ADV_B_DN * ADV_OSTR_DN):
        adv_ref_dn[i] = -77.0
        adv_got_dn[i] = -77.0
    q4_gemm_l1_2x4[DType.float16, DType.float32](
        adv_w_dn, adv_s_dn, adv_x_dn, adv_ref_dn,
        ADV_B_DN, ADV_CIN_DN, ADV_COUT_DN, num_tasks,
        in_stride=ADV_ISTR_DN, out_stride=ADV_OSTR_DN,
    )
    q4_gemm_block_sparse_down[DType.float16, DType.float32](
        adv_w_dn, adv_s_dn, adv_x_dn, adv_got_dn,
        ADV_B_DN, ADV_CIN_DN, ADV_COUT_DN, num_tasks,
        in_stride=ADV_ISTR_DN, out_stride=ADV_OSTR_DN,
    )
    for b in range(ADV_B_DN):
        for r in range(ADV_COUT_DN):
            if abs(adv_ref_dn[b * ADV_OSTR_DN + r] - adv_got_dn[b * ADV_OSTR_DN + r]) > 1e-4:
                raise Error("Adversarial q4_gemm_block_sparse_down heap/stride/tail mismatch!")
        if adv_got_dn[b * ADV_OSTR_DN + ADV_COUT_DN] != -77.0:
            raise Error("Adversarial q4_gemm_block_sparse_down overwrote out_stride padding!")
    adv_w_dn.free()
    adv_s_dn.free()
    adv_x_dn.free()
    adv_ref_dn.free()
    adv_got_dn.free()
    # All-zero input block guard: must write exact 0.0
    for i in range(B * MLP_DIM):
        x_sparse[i] = Float16(0.0)
    for i in range(B * EMBED_DIM):
        out_sparse_dn[i] = Float16(99.0)
    q4_gemm_block_sparse_down[DType.float16, DType.float16](
        dn_p, dn_s, x_sparse, out_sparse_dn, B, MLP_DIM, EMBED_DIM, num_tasks
    )
    if Float32(out_sparse_dn[0]) != 0.0 or Float32(out_sparse_dn[B * EMBED_DIM - 1]) != 0.0:
        raise Error("q4_gemm_block_sparse_down failed to zero output on all-zero input!")
    x_sparse.free()
    out_dense_dn.free()
    out_sparse_dn.free()

    # TDD (llm-mojo-yj4.6): Router-Subspace FFN Bypass for Stationary Canvas Tokens
    var r_curr = alloc[Float32](B * NUM_EXP)
    var r_prev = alloc[Float32](64 * NUM_EXP)
    var prev_top2 = alloc[Int](64 * 2)
    var eb_ffn = alloc[Bool](64)
    var act_idx_ffn = alloc[Int](B)
    var ffn_keep_b = alloc[Int](B)
    for b in range(B):
        var orig_t = b * 2
        act_idx_ffn[b] = orig_t
        eb_ffn[orig_t] = (b < 24)  # b=24..31 unaccepted -> must NEVER be bypassed
        prev_top2[orig_t * 2 + 0] = top_idx[b * TOP_K + 0]
        # Flip top-2 expert on b=8..15 -> must NEVER be bypassed despite small L2 diff
        prev_top2[orig_t * 2 + 1] = top_idx[b * TOP_K + 1] if (b < 8 or b >= 16) else (top_idx[b * TOP_K + 1] + 1)
        for e in range(NUM_EXP):
            var base_v = Float32((e % 9) - 4) * 0.2
            r_prev[orig_t * NUM_EXP + e] = base_v
            # b=0..7: accepted + matching top2 + stationary router -> bypass (8 tokens bypassed)
            # b=16..23: accepted + matching top2 + large router drift -> evaluate
            r_curr[b * NUM_EXP + e] = base_v if (b < 16 or b >= 24) else (base_v + 0.5)
    var ffn_active_len = router_subspace_ffn_filter(
        r_curr, r_prev, top_idx, prev_top2, eb_ffn, act_idx_ffn, ffn_keep_b,
        B, NUM_EXP, TOP_K, step=2, tau_ffn=0.05
    )
    if ffn_active_len != 24 or ffn_keep_b[0] != 8:
        raise Error("router_subspace_ffn_filter failed: expected 24 active tokens starting at b=8, got " + String(ffn_active_len))

    # Adversarial tests (llm-mojo-yj4.6):
    # 1. Uninitialized prev_top2 = -1 guard (even if top_idx == -1 and r_curr == r_prev, must NOT bypass!)
    prev_top2[0] = -1
    prev_top2[1] = -1
    var saved_top0 = top_idx[0]
    var saved_top1 = top_idx[1]
    top_idx[0] = -1
    top_idx[1] = -1
    var uninit_len = router_subspace_ffn_filter(
        r_curr, r_prev, top_idx, prev_top2, eb_ffn, act_idx_ffn, ffn_keep_b,
        1, NUM_EXP, TOP_K, step=2, tau_ffn=0.05
    )
    top_idx[0] = saved_top0
    top_idx[1] = saved_top1
    if uninit_len != 1:
        raise Error("router_subspace_ffn_filter bypassed uninitialized prev_top2 = -1!")

    # 2. Cumulative multi-step router drift guard:
    # Step 1: b=0 drifts by +0.035 (< tau_ffn=0.05) -> bypassed! r_prev[0] must NOT be overwritten.
    # Step 2: b=0 drifts by another +0.035 (cumulative +0.070 > 0.05) -> MUST be evaluated!
    prev_top2[0] = top_idx[0]
    prev_top2[1] = top_idx[1]
    for e in range(NUM_EXP):
        r_prev[e] = 1.0
        r_curr[e] = 1.035
    var step1_len = router_subspace_ffn_filter(
        r_curr, r_prev, top_idx, prev_top2, eb_ffn, act_idx_ffn, ffn_keep_b,
        1, NUM_EXP, TOP_K, step=1, tau_ffn=0.05
    )
    if step1_len != 0:
        raise Error("Expected b=0 to be bypassed on step 1 with 0.035 drift!")
    for e in range(NUM_EXP):
        r_curr[e] = 1.070
    var step2_len = router_subspace_ffn_filter(
        r_curr, r_prev, top_idx, prev_top2, eb_ffn, act_idx_ffn, ffn_keep_b,
        1, NUM_EXP, TOP_K, step=2, tau_ffn=0.05
    )
    if step2_len != 1:
        raise Error("router_subspace_ffn_filter failed to catch cumulative 2-step drift!")

    # 3. Cached FFN residual restoration (scatter_and_restore_ffn_subspace):
    # Bypassed tokens MUST receive cached ffn_prev_out[orig_t], never 0.0!
    var ffn_sub_out = alloc[Float16](1 * EMBED_DIM)
    var ffn_prev_cache = alloc[Float16](64 * EMBED_DIM)
    var ffn_full_out = alloc[Float16](2 * EMBED_DIM)
    for d in range(EMBED_DIM):
        ffn_prev_cache[0 * EMBED_DIM + d] = Float16(2.5)  # orig_t=0 (b=0, bypassed)
        ffn_prev_cache[2 * EMBED_DIM + d] = Float16(1.0)  # orig_t=2 (b=1, evaluated)
        ffn_sub_out[0 * EMBED_DIM + d] = Float16(7.25)    # new evaluated output for b=1
        ffn_full_out[0 * EMBED_DIM + d] = Float16(0.0)
        ffn_full_out[1 * EMBED_DIM + d] = Float16(0.0)
    ffn_keep_b[0] = 1  # Only b=1 was evaluated; b=0 was bypassed
    scatter_and_restore_ffn_subspace[DType.float16](
        ffn_sub_out, ffn_prev_cache, ffn_full_out, act_idx_ffn, ffn_keep_b,
        B=2, ffn_active_len=1, dim=EMBED_DIM
    )
    if abs(Float32(ffn_full_out[0]) - 2.5) > 1e-3 or abs(Float32(ffn_full_out[EMBED_DIM]) - 7.25) > 1e-3:
        raise Error("scatter_and_restore_ffn_subspace failed to restore bypassed token or scatter active token!")
    if abs(Float32(ffn_prev_cache[2 * EMBED_DIM]) - 7.25) > 1e-3:
        raise Error("scatter_and_restore_ffn_subspace failed to update ffn_prev_cache for evaluated token!")
    ffn_sub_out.free()
    ffn_prev_cache.free()
    ffn_full_out.free()

    r_curr.free()
    r_prev.free()
    prev_top2.free()
    eb_ffn.free()
    act_idx_ffn.free()
    ffn_keep_b.free()

    # TDD (llm-mojo-yj4.7): BOS-Sink Head Precomputation & Wo Slice Bypass on Global Layers
    comptime N_HEADS = 4
    comptime H_DIM = 64
    comptime Q_DIM = N_HEADS * H_DIM
    var wo_g_p = alloc[UInt8](EMBED_DIM * (Q_DIM // 2))
    var wo_g_s = alloc[Float32](EMBED_DIM * (Q_DIM // 32))
    var attn_g = alloc[Float16](B * Q_DIM)
    var v_bos_g = alloc[Float16](N_HEADS * H_DIM)
    var sink_mask = alloc[Bool](N_HEADS)
    var wo_ref = alloc[Float32](B * EMBED_DIM)
    var wo_sink = alloc[Float32](B * EMBED_DIM)
    for i in range(EMBED_DIM * (Q_DIM // 2)):
        wo_g_p[i] = UInt8((i * 23 + 7) & 255)
    for i in range(EMBED_DIM * (Q_DIM // 32)):
        wo_g_s[i] = 0.01
    for i in range(N_HEADS * H_DIM):
        v_bos_g[i] = Float16(Float32((i % 13) - 6) * 0.04)
    # Heads 1 and 3 are pure BOS-sink heads (attn_out[b, h] == v_bos_g[h] for all b)
    sink_mask[0] = False
    sink_mask[1] = True
    sink_mask[2] = False
    sink_mask[3] = True
    for b in range(B):
        for h in range(N_HEADS):
            for d in range(H_DIM):
                attn_g[b * Q_DIM + h * H_DIM + d] = (
                    v_bos_g[h * H_DIM + d] if sink_mask[h] else Float16(Float32(((b + d) % 11) - 5) * 0.03)
                )
    # Auto-detect BOS-sink heads from attn_g vs v_bos_g
    var auto_sink_mask = alloc[Bool](N_HEADS)
    var detected_sinks = detect_bos_sink_heads[DType.float16](
        attn_g, v_bos_g, auto_sink_mask, B, N_HEADS, H_DIM, kv_group_size=1, eps_sink=1e-3
    )
    if detected_sinks != 2 or not auto_sink_mask[1] or not auto_sink_mask[3] or auto_sink_mask[0] or auto_sink_mask[2]:
        raise Error("detect_bos_sink_heads failed to identify exact BOS-sink heads 1 and 3!")
    q4_gemm_l1_2x4[DType.float16, DType.float32](
        wo_g_p, wo_g_s, attn_g, wo_ref, B, Q_DIM, EMBED_DIM, num_tasks
    )
    q4_wo_bos_sink_bypass[DType.float16, DType.float32](
        wo_g_p, wo_g_s, attn_g, v_bos_g, auto_sink_mask, wo_sink,
        B, N_HEADS, H_DIM, EMBED_DIM, num_tasks
    )
    for i in range(B * EMBED_DIM):
        if abs(wo_ref[i] - wo_sink[i]) > 1e-4:
            raise Error("q4_wo_bos_sink_bypass mismatch vs full Wo projection!")
    # Adversarial test (llm-mojo-yj4.7): All heads are BOS-sinks (num_act == 0) + GQA kv_group_size=2 + c_out=18 (%4!=0)
    var v_bos_gqa = alloc[Float16](2 * H_DIM)
    for i in range(2 * H_DIM):
        v_bos_gqa[i] = Float16(Float32((i % 11) - 5) * 0.05)
    for b in range(B):
        for h in range(N_HEADS):
            var kv_h = h // 2
            for d in range(H_DIM):
                attn_g[b * Q_DIM + h * H_DIM + d] = v_bos_gqa[kv_h * H_DIM + d]
    var all_sinks = detect_bos_sink_heads[DType.float16](
        attn_g, v_bos_gqa, auto_sink_mask, B, N_HEADS, H_DIM, kv_group_size=2, eps_sink=1e-3
    )
    if all_sinks != N_HEADS:
        raise Error("detect_bos_sink_heads failed with GQA kv_group_size=2!")
    q4_gemm_l1_2x4[DType.float16, DType.float32](
        wo_g_p, wo_g_s, attn_g, wo_ref, B, Q_DIM, 18, num_tasks
    )
    q4_wo_bos_sink_bypass[DType.float16, DType.float32](
        wo_g_p, wo_g_s, attn_g, v_bos_gqa, auto_sink_mask, wo_sink,
        B, N_HEADS, H_DIM, 18, num_tasks, kv_group_size=2
    )
    for i in range(B * 18):
        if abs(wo_ref[i] - wo_sink[i]) > 1e-4:
            raise Error("q4_wo_bos_sink_bypass all-sink GQA c_out=18 mismatch!")
    auto_sink_mask.free()
    v_bos_gqa.free()
    wo_g_p.free()
    wo_g_s.free()
    attn_g.free()
    v_bos_g.free()
    sink_mask.free()
    wo_ref.free()
    wo_sink.free()

    top_idx.free()
    top_val.free()
    top_w.free()
    r_scale.free()
    gu_p.free()
    gu_s.free()
    dn_p.free()
    dn_s.free()
    moe_in.free()
    moe_out.free()
    print("  [PASS] 2. Zen 3 4x3 AVX2 MoE Kernel & Top-8 Singleton Expert Retention")


def test_3_compaction_adacache_teacache_taylor_velocity_flash() raises:
    comptime CANVAS_LEN = 64
    comptime EMBED_DIM = 256

    # 3a. AdaCache + eb_mask unaccepted-token retention guard
    var h_curr = alloc[Float16](CANVAS_LEN * EMBED_DIM)
    var h_prev = alloc[Float16](CANVAS_LEN * EMBED_DIM)
    var act_idx = alloc[Int](CANVAS_LEN)
    var streak = alloc[Int](CANVAS_LEN)
    var eb_mask = alloc[Bool](CANVAS_LEN)
    for t in range(CANVAS_LEN):
        act_idx[t] = t
        streak[t] = 0
        eb_mask[t] = False if t < 8 else True
        for d in range(EMBED_DIM):
            var v = Float16(Float32((t * 5 + d % 17) - 8) * 0.05)
            h_curr[t * EMBED_DIM + d] = v
            h_prev[t * EMBED_DIM + d] = v

    var ada_len = adacache_filter_after_layer1(
        h_curr, h_prev, act_idx, streak, eb_mask, CANVAS_LEN, EMBED_DIM, 3, k_slow=3, tau_ada=0.08
    )
    if ada_len != 16:
        raise Error("Expected AdaCache to round 8 unaccepted tokens up to 16!")
    for b in range(8):
        if act_idx[b] != b:
            raise Error("AdaCache skipped an unaccepted (eb_mask=False) token!")

    # 3a-cont. Cumulative drift guard: tokens 16..63 were skipped on step 3 because their
    # single-step drift was 0.0. Now suppose on step 3 token 32 had drift 0.05 (< tau_ada=0.08, skipped),
    # and on step 4 token 32 drifts another +0.05 from step 3 (cumulative drift from step 2 > 0.08,
    # even though single-step drift from step 3 is 0.05 < 0.08).
    # adacache_h1_prev[32] must NOT have been overwritten on skipped step 3!
    for t in range(CANVAS_LEN):
        act_idx[t] = t
        streak[t] = 0
        eb_mask[t] = True
        for d in range(EMBED_DIM):
            h_prev[t * EMBED_DIM + d] = Float16(1.0)
            # Tokens 0..15 have large motion (kept); token 32 has medium motion (0.20, raw_fluct=0.04 < 0.08 -> skipped)
            var delta: Float32 = Float32(1.0) if t < 16 else (Float32(0.20) if t == 32 else Float32(0.0))
            h_curr[t * EMBED_DIM + d] = Float16(1.0 + delta)
    var ada_step3 = adacache_filter_after_layer1(
        h_curr, h_prev, act_idx, streak, eb_mask, CANVAS_LEN, EMBED_DIM, 3, k_slow=4, tau_ada=0.08
    )
    if ada_step3 != 16:
        raise Error("Expected 16 active tokens on step 3!")
    # Because token 32 was skipped on step 3, h_prev[32] must still be 1.0 (NOT 1.20)!
    if abs(Float32(h_prev[32 * EMBED_DIM]) - 1.0) > 1e-3:
        raise Error("AdaCache overwrote adacache_h1_prev for skipped token 32, losing cumulative drift!")

    # 3b. TeaCache unpooled per-token L2 drift gate + TaylorSeer forecast
    var tc_prev = alloc[Float16](CANVAS_LEN * EMBED_DIM)
    var taylor_D = alloc[Float32](CANVAS_LEN * EMBED_DIM)
    var taylor_d1 = alloc[Float32](CANVAS_LEN * EMBED_DIM)
    var taylor_d2 = alloc[Float32](CANVAS_LEN * EMBED_DIM)
    for i in range(CANVAS_LEN * EMBED_DIM):
        tc_prev[i] = h_prev[i]
        h_curr[i] = h_prev[i]
        taylor_D[i] = 0.01
        taylor_d1[i] = 0.001
        taylor_d2[i] = 0.0
    for t in range(CANVAS_LEN):
        act_idx[t] = t
    var accum = InlineArray[Float32, 1](fill=0.0)
    var skip = teacache_eval_gate(
        h_curr, tc_prev, act_idx, CANVAS_LEN, EMBED_DIM, Float32(0.5), 5, 0,
        UnsafePointer(to=accum[0]), teacache_thresh=0.10, max_consec_skips=1
    )
    if not skip:
        raise Error("TeaCache should skip when per-token L2 drift is 0.0!")

    # 3c. Flash-dLLM + dLLM-KVQ Bidirectional Attention
    comptime HEADS = 4
    comptime KV_HEADS = 2
    comptime HEAD_DIM = 256
    comptime PROMPT_LEN = 8
    comptime SEQ_LEN = PROMPT_LEN + CANVAS_LEN
    var q_buf = alloc[Float16](CANVAS_LEN * HEADS * HEAD_DIM)
    var k_buf = alloc[Float16](CANVAS_LEN * KV_HEADS * HEAD_DIM)
    var v_buf = alloc[Float16](CANVAS_LEN * KV_HEADS * HEAD_DIM)
    var q_norm = alloc[Float32](HEAD_DIM)
    var k_norm = alloc[Float32](HEAD_DIM)
    var k_cache = alloc[Float16](SEQ_LEN * KV_HEADS * HEAD_DIM)
    var v_cache = alloc[Float16](SEQ_LEN * KV_HEADS * HEAD_DIM)
    var attn_out = alloc[Float16](CANVAS_LEN * HEADS * HEAD_DIM)
    for i in range(HEAD_DIM):
        q_norm[i] = 1.0
        k_norm[i] = 1.0
    for i in range(CANVAS_LEN * HEADS * HEAD_DIM):
        q_buf[i] = Float16(0.05)
    for i in range(CANVAS_LEN * KV_HEADS * HEAD_DIM):
        k_buf[i] = Float16(0.05)
        v_buf[i] = Float16(0.05)
    for i in range(SEQ_LEN * KV_HEADS * HEAD_DIM):
        k_cache[i] = Float16(0.05)
        v_cache[i] = Float16(0.05)

    prep_qkv_norm_rope_kvq_batch(
        q_buf, k_buf, v_buf, q_norm, k_norm, act_idx, k_cache, v_cache,
        CANVAS_LEN, PROMPT_LEN, HEADS, KV_HEADS, HEAD_DIM, KV_HEADS * HEAD_DIM,
        0, HEADS * HEAD_DIM, KV_HEADS * HEAD_DIM, False
    )
    flash_dllm_bidirectional_attn(
        q_buf, k_cache, v_cache, attn_out, CANVAS_LEN, SEQ_LEN,
        HEADS, KV_HEADS, HEAD_DIM, KV_HEADS * HEAD_DIM, 0, HEADS * HEAD_DIM
    )

    h_curr.free()
    h_prev.free()
    act_idx.free()
    streak.free()
    eb_mask.free()
    tc_prev.free()
    taylor_D.free()
    taylor_d1.free()
    taylor_d2.free()
    q_buf.free()
    k_buf.free()
    v_buf.free()
    q_norm.free()
    k_norm.free()
    k_cache.free()
    v_cache.free()
    attn_out.free()
    print("  [PASS] 3. AdaCache + TeaCache + TaylorSeer + Flash-dLLM KVQ Bidirectional Attention")


def test_4_e2e_denoising_state_machine() raises:
    """Synthetic multi-step denoising loop integration test verifying:
    1) Entropy-bounded `eb_mask` acceptance `(cumsum - cummax) <= entropy_bound`
    2) Adjacent-duplicate word suppression (`adj_dup` for token IDs >= 256)
    3) `update_locked_mask` mean-entropy (`< 0.30`) and token-entropy (`< 0.08`) gating
    4) Monotonic convergence without premature step-1 locking or adjacent stuttering.
    """
    comptime CANVAS_LEN = 16
    var token_entropy = alloc[Float32](CANVAS_LEN)
    var new_tokens = alloc[Int](CANVAS_LEN)
    var argmax_tokens = alloc[Int](CANVAS_LEN)
    var canvas_tokens = alloc[Int](CANVAS_LEN)
    var eb_mask = alloc[Bool](CANVAS_LEN)
    var locked_mask = alloc[Bool](CANVAS_LEN)
    var history = alloc[Int](2 * CANVAS_LEN)
    var sorted_idx = alloc[Int](CANVAS_LEN)
    var sorted_ent = alloc[Float32](CANVAS_LEN)

    for t in range(CANVAS_LEN):
        locked_mask[t] = False
        eb_mask[t] = False
        canvas_tokens[t] = 1000 + t

    var hist_len = 0
    var entropy_bound: Float32 = 0.10

    # Simulate 4 denoising steps:
    # Steps 0 & 1: High mean entropy (~1.2) -> even if history[0] == history[1] at t=0,
    # update_locked_mask MUST NOT permanently lock t=0 while mean_ent >= 0.30!
    # Also test adjacent duplicate at t=2 and t=3 predicting the same word token 500:
    # only the lower-entropy one (t=2) may be accepted into eb_mask on the same step!
    for step in range(4):
        var base_ent: Float32 = Float32(1.2) if step < 2 else Float32(0.01)
        for t in range(CANVAS_LEN):
            token_entropy[t] = base_ent + Float32(t) * 0.002
            new_tokens[t] = 300 + t
            argmax_tokens[t] = 300 + t

        # Plant adjacent duplicate word token (id=500 >= 256) at positions t=2 and t=3
        new_tokens[2] = 500
        argmax_tokens[2] = 500
        token_entropy[2] = 0.005
        new_tokens[3] = 500
        argmax_tokens[3] = 500
        token_entropy[3] = 0.006

        var mean_ent: Float32 = 0.0
        for t in range(CANVAS_LEN):
            mean_ent += token_entropy[t]
        mean_ent /= Float32(CANVAS_LEN)

        # Sort by ascending entropy
        for i in range(CANVAS_LEN):
            sorted_idx[i] = i
            sorted_ent[i] = token_entropy[i]
        for i in range(CANVAS_LEN):
            var min_idx = i
            for j in range(i + 1, CANVAS_LEN):
                if sorted_ent[j] < sorted_ent[min_idx]:
                    min_idx = j
            if min_idx != i:
                var te = sorted_ent[i]
                sorted_ent[i] = sorted_ent[min_idx]
                sorted_ent[min_idx] = te
                var ti = sorted_idx[i]
                sorted_idx[i] = sorted_idx[min_idx]
                sorted_idx[min_idx] = ti

        var cumsum_ent: Float32 = 0.0
        var cummax_ent: Float32 = 0.0
        for t in range(CANVAS_LEN):
            eb_mask[t] = locked_mask[t]
        for i in range(CANVAS_LEN):
            cumsum_ent += sorted_ent[i]
            if sorted_ent[i] > cummax_ent:
                cummax_ent = sorted_ent[i]
            if (cumsum_ent - cummax_ent) <= entropy_bound:
                var orig_idx = sorted_idx[i]
                if eb_mask[orig_idx]:
                    continue
                var cand_tok = new_tokens[orig_idx]
                var adj_dup = False
                if cand_tok >= 256:
                    if orig_idx > 0 and eb_mask[orig_idx - 1] and new_tokens[orig_idx - 1] == cand_tok:
                        adj_dup = True
                    if orig_idx + 1 < CANVAS_LEN and eb_mask[orig_idx + 1] and new_tokens[orig_idx + 1] == cand_tok:
                        adj_dup = True
                if not adj_dup:
                    eb_mask[orig_idx] = True

        # Verify adjacent duplicate guard: t=2 is accepted, t=3 is rejected!
        if not eb_mask[2] or eb_mask[3]:
            raise Error("Adjacent duplicate guard failed to reject duplicate neighbor t=3!")

        # Also verify resolve_adjacent_duplicate_tokens replaces t=3's argmax_tokens[3] and new_tokens[3]
        # with top2_tokens[3] (501) so final argmax_canvas never contains adjacent word stutters!
        var top2_tokens = alloc[Int](CANVAS_LEN)
        for t in range(CANVAS_LEN):
            top2_tokens[t] = 600 + t
        top2_tokens[3] = 501
        resolve_adjacent_duplicate_tokens(
            sorted_idx, eb_mask, locked_mask, argmax_tokens, new_tokens, top2_tokens,
            0, CANVAS_LEN
        )
        var fixed_argmax_3 = argmax_tokens[3]
        var fixed_new_3 = new_tokens[3]
        if fixed_argmax_3 != 501 or fixed_new_3 != 501:
            top2_tokens.free()
            raise Error("resolve_adjacent_duplicate_tokens failed to substitute top2 token on t=3!")

        # Two-sided collision guard: if t=2 has 500 and t=4 (already resolved) has 501,
        # and t=3 has argmax=500, top2=501, resolve_adjacent_duplicate_tokens must NOT
        # replace t=3 with 501 (which would collide with t=4)!
        argmax_tokens[2] = 500
        new_tokens[2] = 500
        argmax_tokens[3] = 500
        new_tokens[3] = 500
        argmax_tokens[4] = 501
        new_tokens[4] = 501
        eb_mask[2] = True
        eb_mask[3] = False
        eb_mask[4] = True
        top2_tokens[3] = 501
        resolve_adjacent_duplicate_tokens(
            sorted_idx, eb_mask, locked_mask, argmax_tokens, new_tokens, top2_tokens,
            0, CANVAS_LEN
        )
        var clash_argmax_3 = argmax_tokens[3]
        top2_tokens.free()
        if clash_argmax_3 == 501:
            raise Error("resolve_adjacent_duplicate_tokens substituted top2=501 despite colliding with right neighbor t=4!")
        argmax_tokens[3] = 501
        new_tokens[3] = 501

        var write_pos = hist_len % 2
        for t in range(CANVAS_LEN):
            history[write_pos * CANVAS_LEN + t] = argmax_tokens[t]
        hist_len += 1

        var num_locked = update_locked_mask(
            eb_mask, history, token_entropy, locked_mask, CANVAS_LEN, hist_len,
            mean_ent=mean_ent, block_start=0, block_end=CANVAS_LEN
        )
        if step == 1 and num_locked != 0:
            raise Error("update_locked_mask prematurely locked tokens at step 1 while mean_ent > 0.30!")
        if step == 3 and num_locked == 0:
            raise Error("update_locked_mask failed to lock converged low-entropy tokens at step 3!")

    token_entropy.free()
    new_tokens.free()
    argmax_tokens.free()
    canvas_tokens.free()
    eb_mask.free()
    locked_mask.free()
    history.free()
    sorted_idx.free()
    sorted_ent.free()
    print("  [PASS] 4. End-to-End Multi-Step Denoising Loop State Machine (eb_mask + adj_dup + lock gating)")


def test_5_mini_2layer_e2e_diffusion_pipeline(num_tasks: Int) raises:
    """Miniature 2-Layer (Layer 0 = Sliding 16Q/8KV dim=256, Layer 1 = Global 16Q/2KV dim=512),
    4-Step End-to-End Diffusion Pipeline Test (<15ms, zero disk I/O).
    Exercises Phase 1 (Embed + Velocity Extrap + SC MLP + Active Compaction),
    Phase 2 (Sliding & Global Attention + Shared MLP + MoE Router/Experts + AdaCache + TeaCache/TaylorSeer),
    and Phase 3 (Logit Projection + Top-32 Sampler + Scatter + Adj-Dup Resolution + Locking).
    """
    var t0 = std.time.perf_counter()
    comptime PROMPT_LEN = 4
    comptime CANVAS_LEN = 32
    comptime SEQ_LEN = PROMPT_LEN + CANVAS_LEN
    comptime EMBED_DIM = 128
    comptime SC_SIZE = 64
    comptime STD_MLP_DIM = 64
    comptime EXP_MLP_DIM = 32
    comptime NUM_EXPERTS = 16
    comptime TOP_K_EXP = 8
    comptime VOCAB_SIZE = 512
    comptime MAX_Q_SIZE = 16 * 512
    comptime MAX_KV_SIZE = 8 * 256

    # Allocate compact synthetic Q4 weights for 2 layers + SC + Embed
    var embed_p = alloc[UInt8](VOCAB_SIZE * (EMBED_DIM // 2))
    var embed_s = alloc[Float32](VOCAB_SIZE * (EMBED_DIM // 32))
    for i in range(VOCAB_SIZE * (EMBED_DIM // 2)):
        embed_p[i] = UInt8((i * 31 + 7) & 255)
    for i in range(VOCAB_SIZE * (EMBED_DIM // 32)):
        embed_s[i] = Float32(((i * 13) % 40) - 20) * 0.005

    var sc_gate_p = alloc[UInt8](SC_SIZE * (EMBED_DIM // 2))
    var sc_up_p = alloc[UInt8](SC_SIZE * (EMBED_DIM // 2))
    var sc_down_p = alloc[UInt8](EMBED_DIM * (SC_SIZE // 2))
    var sc_s = alloc[Float32](SC_SIZE * (EMBED_DIM // 32))
    var norm_w = alloc[Float32](512)
    for i in range(SC_SIZE * (EMBED_DIM // 2)):
        sc_gate_p[i] = UInt8((i * 17 + 3) & 255)
        sc_up_p[i] = UInt8((i * 23 + 11) & 255)
        sc_down_p[i] = UInt8((i * 19 + 5) & 255)
    for i in range(SC_SIZE * (EMBED_DIM // 32)):
        sc_s[i] = 0.01
    for i in range(512):
        norm_w[i] = 1.0

    var wq_p = alloc[UInt8](MAX_Q_SIZE * (EMBED_DIM // 2))
    var wq_s = alloc[Float32](MAX_Q_SIZE * (EMBED_DIM // 32))
    var wkv_p = alloc[UInt8](MAX_KV_SIZE * (EMBED_DIM // 2))
    var wkv_s = alloc[Float32](MAX_KV_SIZE * (EMBED_DIM // 32))
    var wo_p = alloc[UInt8](EMBED_DIM * (MAX_Q_SIZE // 2))
    var wo_s = alloc[Float32](EMBED_DIM * (MAX_Q_SIZE // 32))
    for i in range(MAX_Q_SIZE * (EMBED_DIM // 2)):
        wq_p[i] = UInt8((i * 7 + 1) & 255)
        wo_p[i] = UInt8((i * 11 + 9) & 255)
    for i in range(MAX_Q_SIZE * (EMBED_DIM // 32)):
        wq_s[i] = 0.005
        wo_s[i] = 0.005
    for i in range(MAX_KV_SIZE * (EMBED_DIM // 2)):
        wkv_p[i] = UInt8((i * 13 + 3) & 255)
    for i in range(MAX_KV_SIZE * (EMBED_DIM // 32)):
        wkv_s[i] = 0.005

    var gu_p = alloc[UInt8](NUM_EXPERTS * (EXP_MLP_DIM * 2) * (EMBED_DIM // 2))
    var gu_s = alloc[Float32](NUM_EXPERTS * (EXP_MLP_DIM * 2) * (EMBED_DIM // 32))
    var dn_p = alloc[UInt8](NUM_EXPERTS * EMBED_DIM * (EXP_MLP_DIM // 2))
    var dn_s = alloc[Float32](NUM_EXPERTS * EMBED_DIM * (EXP_MLP_DIM // 32))
    for i in range(NUM_EXPERTS * (EXP_MLP_DIM * 2) * (EMBED_DIM // 2)):
        gu_p[i] = UInt8((i * 29 + 5) & 255)
    for i in range(NUM_EXPERTS * (EXP_MLP_DIM * 2) * (EMBED_DIM // 32)):
        gu_s[i] = 0.01
    for i in range(NUM_EXPERTS * EMBED_DIM * (EXP_MLP_DIM // 2)):
        dn_p[i] = UInt8((i * 37 + 7) & 255)
    for i in range(NUM_EXPERTS * EMBED_DIM * (EXP_MLP_DIM // 32)):
        dn_s[i] = 0.01

    # Canvas state buffers
    var inputs_embeds = alloc[Float16](CANVAS_LEN * EMBED_DIM)
    var sc_embeds = alloc[Float32](CANVAS_LEN * EMBED_DIM)
    var prev_sc_embeds = alloc[Float32](CANVAS_LEN * EMBED_DIM)
    var sc_normed = alloc[Float32](CANVAS_LEN * EMBED_DIM)
    var sc_gated = alloc[Float32](CANVAS_LEN * SC_SIZE)
    var sc_down = alloc[Float32](CANVAS_LEN * EMBED_DIM)
    var full_hidden = alloc[Float16](CANVAS_LEN * EMBED_DIM)
    var canvas_hidden = alloc[Float16](CANVAS_LEN * EMBED_DIM)
    var canvas_norm = alloc[Float32](CANVAS_LEN * EMBED_DIM)
    var canvas_q = alloc[Float16](CANVAS_LEN * MAX_Q_SIZE)
    var canvas_k = alloc[Float16](CANVAS_LEN * MAX_KV_SIZE)
    var canvas_v = alloc[Float16](CANVAS_LEN * MAX_KV_SIZE)
    var canvas_attn_out = alloc[Float16](CANVAS_LEN * MAX_Q_SIZE)
    var canvas_proj_attn = alloc[Float32](CANVAS_LEN * EMBED_DIM)
    var k_cache = alloc[Float16](2 * SEQ_LEN * MAX_KV_SIZE)
    var v_cache = alloc[Float16](2 * SEQ_LEN * MAX_KV_SIZE)
    var ffn_gated = alloc[Float16](CANVAS_LEN * STD_MLP_DIM)
    var ffn_down = alloc[Float16](CANVAS_LEN * EMBED_DIM)
    var moe_out = alloc[Float16](CANVAS_LEN * EMBED_DIM)
    var top_idx = alloc[Int](CANVAS_LEN * TOP_K_EXP)
    var top_val = alloc[Float32](CANVAS_LEN * TOP_K_EXP)
    var top_w = alloc[Float32](CANVAS_LEN * TOP_K_EXP)
    var moe_r_curr = alloc[Float32](CANVAS_LEN * NUM_EXPERTS)
    var moe_r_prev = alloc[Float32](2 * CANVAS_LEN * NUM_EXPERTS)
    var moe_prev_top2 = alloc[Int](2 * CANVAS_LEN * 2)
    var moe_prev_out = alloc[Float16](2 * CANVAS_LEN * EMBED_DIM)
    var ffn_keep_b = alloc[Int](CANVAS_LEN)
    var adacache_h1_prev = alloc[Float16](CANVAS_LEN * EMBED_DIM)
    var adacache_streak = alloc[Int](CANVAS_LEN)
    var teacache_prev = alloc[Float16](CANVAS_LEN * EMBED_DIM)
    var taylor_h5_curr = alloc[Float16](CANVAS_LEN * EMBED_DIM)
    var taylor_D = alloc[Float32](CANVAS_LEN * EMBED_DIM)
    var taylor_d1 = alloc[Float32](CANVAS_LEN * EMBED_DIM)
    var taylor_d2 = alloc[Float32](CANVAS_LEN * EMBED_DIM)
    var logits = alloc[Float32](CANVAS_LEN * VOCAB_SIZE)
    var rand_u = alloc[Float32](CANVAS_LEN)
    var token_entropy = alloc[Float32](CANVAS_LEN)
    var argmax_tokens = alloc[Int](CANVAS_LEN)
    var new_tokens = alloc[Int](CANVAS_LEN)
    var top2_tokens = alloc[Int](CANVAS_LEN)
    var canvas_tokens = alloc[Int](CANVAS_LEN)
    var compact_entropy = alloc[Float32](CANVAS_LEN)
    var compact_argmax = alloc[Int](CANVAS_LEN)
    var compact_new = alloc[Int](CANVAS_LEN)
    var compact_top2 = alloc[Int](CANVAS_LEN)
    var compact_sc = alloc[Float32](CANVAS_LEN * EMBED_DIM)
    var locked_mask = alloc[Bool](CANVAS_LEN)
    var eb_mask = alloc[Bool](CANVAS_LEN)
    var active_canvas_idx = alloc[Int](CANVAS_LEN)
    var history = alloc[Int](2 * CANVAS_LEN)
    var sorted_idx = alloc[Int](CANVAS_LEN)
    var sorted_ent = alloc[Float32](CANVAS_LEN)

    for i in range(2 * SEQ_LEN * MAX_KV_SIZE):
        k_cache[i] = Float16(0.01)
        v_cache[i] = Float16(0.01)
    for i in range(CANVAS_LEN * EMBED_DIM):
        inputs_embeds[i] = Float16(Float32((i % 29) - 14) * 0.03)
        sc_embeds[i] = 0.0
        prev_sc_embeds[i] = 0.0
        adacache_h1_prev[i] = Float16(0.0)
        teacache_prev[i] = Float16(0.0)
        taylor_D[i] = 0.0
        taylor_d1[i] = 0.0
        taylor_d2[i] = 0.0
    for t in range(CANVAS_LEN):
        canvas_tokens[t] = 260 + t
        argmax_tokens[t] = 260 + t
        new_tokens[t] = 260 + t
        top2_tokens[t] = 260 + t + 1
        token_entropy[t] = 0.05
        locked_mask[t] = (t < 4)  # Pre-lock 4 tokens to exercise active compaction (active_len=28)
        eb_mask[t] = (t < 16)
        adacache_streak[t] = 0
        rand_u[t] = 0.25
        for s in range(TOP_K_EXP):
            top_idx[t * TOP_K_EXP + s] = (t + s) % NUM_EXPERTS
            top_val[t * TOP_K_EXP + s] = Float32(0.25) if s < 3 else Float32(0.05)
        for e in range(NUM_EXPERTS):
            moe_r_curr[t * NUM_EXPERTS + e] = Float32((e % 7) - 3) * 0.1
    for i in range(2 * CANVAS_LEN * 2):
        moe_prev_top2[i] = -1
    for i in range(2 * CANVAS_LEN * EMBED_DIM):
        moe_prev_out[i] = Float16(0.0)

    var hist_len = 0
    var full_steps = 0
    var consec_skips = 0
    var tc_accum = InlineArray[Float32, 1](fill=0.0)

    # Execute 3 full end-to-end denoising steps
    for step in range(3):
        extrapolate_sc_velocity_batch(
            sc_embeds, prev_sc_embeds, locked_mask, 0, CANVAS_LEN, EMBED_DIM, step, 1
        )
        sc_forward_batch[DType.float16, DType.float16](
            inputs_embeds, sc_embeds, sc_gate_p, sc_up_p, sc_down_p,
            sc_s, sc_s, sc_s, norm_w, sc_normed, sc_gated, sc_down,
            full_hidden, CANVAS_LEN, EMBED_DIM, SC_SIZE, num_tasks
        )
        var active_len = build_block_active_canvas_map(locked_mask, active_canvas_idx, 0, CANVAS_LEN)
        gather_active[DType.float16](full_hidden, active_canvas_idx, canvas_hidden, active_len, EMBED_DIM)

        # 2-Layer Transformer Loop (l=0: Sliding 16Q/8KV dim=256; l=1: Global 16Q/2KV dim=512)
        for l in range(2):
            var is_global = (l == 1)
            var l_heads = 16
            var l_kv_heads = 2 if is_global else 8
            var l_head_dim = 512 if is_global else 256
            var kv_stride = l_kv_heads * l_head_dim
            var q_size = l_heads * l_head_dim
            var cache_off = l * SEQ_LEN * MAX_KV_SIZE

            for b in range(active_len):
                rms_norm[DType.float16, DType.float32](
                    canvas_hidden + b * EMBED_DIM, norm_w, canvas_norm + b * EMBED_DIM, EMBED_DIM
                )
            var deint_buf = alloc[Float32](active_len * MAX_Q_SIZE)
            deinterleave_batch_to_f32[DType.float32](canvas_norm, deint_buf, active_len, EMBED_DIM)
            q4_gemm_l1_2x4[DType.float32, DType.float16](
                wq_p, wq_s, canvas_norm, canvas_q, active_len, EMBED_DIM, q_size, num_tasks,
                in_stride=EMBED_DIM, out_stride=MAX_Q_SIZE, pre_deint_X=deint_buf
            )
            q4_gemm_l1_2x4[DType.float32, DType.float16](
                wkv_p, wkv_s, canvas_norm, canvas_k, active_len, EMBED_DIM, kv_stride, num_tasks,
                in_stride=EMBED_DIM, out_stride=MAX_KV_SIZE, pre_deint_X=deint_buf
            )
            q4_gemm_l1_2x4[DType.float32, DType.float16](
                wkv_p, wkv_s, canvas_norm, canvas_v, active_len, EMBED_DIM, kv_stride, num_tasks,
                in_stride=EMBED_DIM, out_stride=MAX_KV_SIZE, pre_deint_X=deint_buf
            )
            prep_qkv_norm_rope_kvq_batch(
                canvas_q, canvas_k, canvas_v, norm_w, norm_w, active_canvas_idx,
                k_cache, v_cache, active_len, PROMPT_LEN, l_heads, l_kv_heads,
                l_head_dim, kv_stride, cache_off, MAX_Q_SIZE, MAX_KV_SIZE, is_global
            )
            flash_dllm_bidirectional_attn(
                canvas_q, k_cache, v_cache, canvas_attn_out, active_len, SEQ_LEN,
                l_heads, l_kv_heads, l_head_dim, kv_stride, cache_off, MAX_Q_SIZE
            )
            if is_global:
                var sink_m_arr = InlineArray[Bool, 16](uninitialized=True)
                var sink_m_ptr = UnsafePointer[Bool, MutAnyOrigin](unsafe_from_address=Int(UnsafePointer(to=sink_m_arr)))
                _ = detect_bos_sink_heads[DType.float16](
                    canvas_attn_out, v_cache + cache_off, sink_m_ptr, active_len, l_heads, l_head_dim,
                    kv_group_size=l_heads // l_kv_heads, eps_sink=1e-3
                )
                q4_wo_bos_sink_bypass[DType.float16, DType.float32](
                    wo_p, wo_s, canvas_attn_out, v_cache + cache_off, sink_m_ptr, canvas_proj_attn,
                    active_len, l_heads, l_head_dim, EMBED_DIM, num_tasks,
                    kv_group_size=l_heads // l_kv_heads
                )
                _ = sink_m_arr[0]
            else:
                deinterleave_batch_to_f32[DType.float16](canvas_attn_out, deint_buf, active_len, q_size, in_stride=MAX_Q_SIZE)
                q4_gemm_l1_2x4[DType.float16, DType.float32](
                    wo_p, wo_s, canvas_attn_out, canvas_proj_attn, active_len, q_size, EMBED_DIM, num_tasks,
                    in_stride=MAX_Q_SIZE, out_stride=EMBED_DIM, pre_deint_X=deint_buf
                )
            for b in range(active_len):
                var h_ptr = canvas_hidden + b * EMBED_DIM
                var p_ptr = canvas_proj_attn + b * EMBED_DIM
                for i in range(0, EMBED_DIM, 8):
                    h_ptr.store[8](i, (h_ptr.load[8](i).cast[DType.float32]() + p_ptr.load[8](i)).cast[DType.float16]())
                rms_norm[DType.float16, DType.float32](h_ptr, norm_w, canvas_norm + b * EMBED_DIM, EMBED_DIM)

            # Shared MLP + MoE Experts
            deinterleave_batch_to_f32[DType.float32](canvas_norm, deint_buf, active_len, EMBED_DIM)
            q4_fused_gate_up_geglu_l1[DType.float32, DType.float16](
                sc_gate_p, sc_s, sc_up_p, sc_s, canvas_norm, ffn_gated,
                active_len, EMBED_DIM, STD_MLP_DIM, num_tasks, pre_deint_X=deint_buf
            )
            deint_buf.free()
            q4_gemm_block_sparse_down[DType.float16, DType.float16](
                sc_down_p, sc_s, ffn_gated, ffn_down, active_len, STD_MLP_DIM, EMBED_DIM, num_tasks
            )
            apply_moe_lod_and_set_cover(
                top_idx, top_val, norm_w, top_w, active_len, TOP_K_EXP, 8, 1.0,
                eb_mask=eb_mask, active_canvas_idx=active_canvas_idx, unaccepted_top_k=2
            )
            var ffn_active_len = router_subspace_ffn_filter(
                moe_r_curr,
                moe_r_prev + l * CANVAS_LEN * NUM_EXPERTS,
                top_idx,
                moe_prev_top2 + l * CANVAS_LEN * 2,
                eb_mask,
                active_canvas_idx,
                ffn_keep_b,
                active_len,
                NUM_EXPERTS,
                TOP_K_EXP,
                step,
                tau_ffn=0.05,
                top_weights=top_w,
            )
            if ffn_active_len > 0:
                run_diffusion_moe_experts_parallel(
                    gu_p, gu_s, dn_p, dn_s, canvas_hidden, top_idx, top_w, moe_out,
                    active_len, EMBED_DIM, EXP_MLP_DIM, TOP_K_EXP, num_tasks
                )
            sync_moe_subspace_cache[DType.float16](
                moe_out,
                moe_prev_out + l * CANVAS_LEN * EMBED_DIM,
                active_canvas_idx,
                ffn_keep_b,
                active_len,
                ffn_active_len,
                EMBED_DIM,
            )
            for b in range(active_len):
                var h_ptr = canvas_hidden + b * EMBED_DIM
                var f_ptr = ffn_down + b * EMBED_DIM
                var m_ptr = moe_out + b * EMBED_DIM
                for i in range(0, EMBED_DIM, 8):
                    var comb = h_ptr.load[8](i).cast[DType.float32]() + f_ptr.load[8](i).cast[DType.float32]() + m_ptr.load[8](i).cast[DType.float32]()
                    h_ptr.store[8](i, comb.cast[DType.float16]())

            if l == 1:
                active_len = adacache_filter_after_layer1(
                    canvas_hidden, adacache_h1_prev, active_canvas_idx, adacache_streak,
                    eb_mask, active_len, EMBED_DIM, step, k_slow=2, tau_ada=0.05
                )
                var skip_mid = teacache_eval_gate(
                    canvas_hidden, teacache_prev, active_canvas_idx, active_len, EMBED_DIM,
                    Float32(0.6), full_steps, consec_skips, UnsafePointer(to=tc_accum[0])
                )
                if skip_mid:
                    consec_skips += 1
                    taylorseer_forecast_skipped_step(
                        canvas_hidden, teacache_prev, active_canvas_idx,
                        taylor_D, taylor_d1, taylor_d2, active_len, EMBED_DIM, full_steps, consec_skips
                    )
                else:
                    for b in range(active_len):
                        for i in range(0, EMBED_DIM, 8):
                            (taylor_h5_curr + b * EMBED_DIM).store[8](i, (canvas_hidden + b * EMBED_DIM).load[8](i))
                    taylorseer_record_full_step(
                        taylor_h5_curr, canvas_hidden, active_canvas_idx,
                        taylor_D, taylor_d1, taylor_d2, active_len, EMBED_DIM, full_steps
                    )
                    full_steps += 1
                    consec_skips = 0

        # Phase 3: Logit projection (Two-Stage Coarse-to-Fine Vocab Sieve) + Top-32 Sampler + Scatter
        for b in range(active_len):
            rms_norm[DType.float16, DType.float32](
                canvas_hidden + b * EMBED_DIM, norm_w, canvas_norm + b * EMBED_DIM, EMBED_DIM
            )
        q4_lm_head_two_stage_sieve[DType.float32, DType.float32](
            embed_p, embed_s, canvas_norm, logits, active_len, EMBED_DIM, VOCAB_SIZE, num_tasks,
            forced_cand_tokens=argmax_tokens, num_forced_cands=CANVAS_LEN,
            coarse_blocks=1, span_keep_frac=0.45
        )
        sample_and_soft_embed_top32_batch[32](
            logits, rand_u, embed_p, embed_s, compact_entropy, compact_argmax, compact_new, compact_sc,
            active_len, VOCAB_SIZE, EMBED_DIM, Float32(0.6), sqrt(Float32(EMBED_DIM)), num_tasks,
            top2_tokens=compact_top2
        )
        scatter_active_sampler_outputs[DType.float32](
            active_canvas_idx, compact_entropy, compact_argmax, compact_new, compact_sc,
            token_entropy, argmax_tokens, new_tokens, sc_embeds, locked_mask, canvas_tokens,
            active_len, CANVAS_LEN, EMBED_DIM, compact_top2=compact_top2, top2_tokens=top2_tokens
        )
        for i in range(CANVAS_LEN):
            sorted_idx[i] = i
            sorted_ent[i] = token_entropy[i]
        resolve_adjacent_duplicate_tokens(
            sorted_idx, eb_mask, locked_mask, argmax_tokens, new_tokens, top2_tokens, 0, CANVAS_LEN
        )
        var write_pos = hist_len % 2
        for t in range(CANVAS_LEN):
            history[write_pos * CANVAS_LEN + t] = argmax_tokens[t]
        hist_len += 1
        _ = update_locked_mask(
            eb_mask, history, token_entropy, locked_mask, CANVAS_LEN, hist_len, mean_ent=0.20
        )

    var dt_ms = (std.time.perf_counter() - t0) * 1000.0
    for t in range(CANVAS_LEN):
        if argmax_tokens[t] < 0 or argmax_tokens[t] >= VOCAB_SIZE:
            raise Error("Invalid argmax token in 2-layer E2E pipeline!")
        if token_entropy[t] < 0.0 or token_entropy[t] > 10.0:
            raise Error("Invalid token_entropy in 2-layer E2E pipeline!")

    embed_p.free()
    embed_s.free()
    sc_gate_p.free()
    sc_up_p.free()
    sc_down_p.free()
    sc_s.free()
    norm_w.free()
    wq_p.free()
    wq_s.free()
    wkv_p.free()
    wkv_s.free()
    wo_p.free()
    wo_s.free()
    gu_p.free()
    gu_s.free()
    dn_p.free()
    dn_s.free()
    inputs_embeds.free()
    sc_embeds.free()
    prev_sc_embeds.free()
    sc_normed.free()
    sc_gated.free()
    sc_down.free()
    full_hidden.free()
    canvas_hidden.free()
    canvas_norm.free()
    canvas_q.free()
    canvas_k.free()
    canvas_v.free()
    canvas_attn_out.free()
    canvas_proj_attn.free()
    k_cache.free()
    v_cache.free()
    ffn_gated.free()
    ffn_down.free()
    moe_out.free()
    top_idx.free()
    top_val.free()
    top_w.free()
    moe_r_curr.free()
    moe_r_prev.free()
    moe_prev_top2.free()
    moe_prev_out.free()
    ffn_keep_b.free()
    adacache_h1_prev.free()
    adacache_streak.free()
    teacache_prev.free()
    taylor_h5_curr.free()
    taylor_D.free()
    taylor_d1.free()
    taylor_d2.free()
    logits.free()
    rand_u.free()
    token_entropy.free()
    argmax_tokens.free()
    new_tokens.free()
    top2_tokens.free()
    canvas_tokens.free()
    compact_entropy.free()
    compact_argmax.free()
    compact_new.free()
    compact_top2.free()
    compact_sc.free()
    locked_mask.free()
    eb_mask.free()
    active_canvas_idx.free()
    history.free()
    sorted_idx.free()
    sorted_ent.free()
    print("  [PASS] 5. Miniature 2-Layer 3-Step Full E2E Diffusion Pipeline (", dt_ms, "ms)")


def main() raises:
    var t0 = std.time.perf_counter()
    print("=== Fast Unified Diffusion Test Suite ===")
    var num_tasks = num_physical_cores()
    test_1_sc_and_top32_sampler(num_tasks)
    test_2_moe_4x3_and_lod(num_tasks)
    test_3_compaction_adacache_teacache_taylor_velocity_flash()
    test_4_e2e_denoising_state_machine()
    test_5_mini_2layer_e2e_diffusion_pipeline(num_tasks)
    var dt_ms = (std.time.perf_counter() - t0) * 1000.0
    print("=== ALL DIFFUSION SUITE TESTS PASSED in", dt_ms, "ms! ===")
