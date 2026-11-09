from std.memory import alloc
from std.math import sqrt, exp, tanh, log, abs
from std.algorithm import parallelize
from std.sys.info import num_physical_cores
import std.time
from diffusion_kernels import (
    rms_norm,
    rms_norm_noscale,
    unpack_16bytes,
    q4_gemm_l1_2x4,
    sc_forward_batch,
    probs_to_soft_embed_top32,
    sample_and_soft_embed_top32_batch,
)


@always_inline
def ref_q4_gemv_cpu(
    W_packed: UnsafePointer[UInt8, _],
    scales: UnsafePointer[Float32, _],
    x: UnsafePointer[Float32, _],
    out_ptr: UnsafePointer[Float32, MutAnyOrigin],
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

        for r in range(start_r, end_r):
            var acc = SIMD[DType.float32, 8](0.0)
            var r_offset_packed = r * c_in_packed
            var r_offset_scales = r * c_in_blocks

            for b in range(c_in_blocks):
                var scale = scales[r_offset_scales + b]
                var block_p = r_offset_packed + b * 16
                var x_offset = b * 32

                var bytes = W_packed.load[16](block_p)
                var w0 = SIMD[DType.float32, 8]()
                var w1 = SIMD[DType.float32, 8]()
                var w2 = SIMD[DType.float32, 8]()
                var w3 = SIMD[DType.float32, 8]()
                unpack_16bytes(bytes, w0, w1, w2, w3)

                acc = acc + (w0 * x.load[8](x_offset)) * scale
                acc = acc + (w1 * x.load[8](x_offset + 8)) * scale
                acc = acc + (w2 * x.load[8](x_offset + 16)) * scale
                acc = acc + (w3 * x.load[8](x_offset + 24)) * scale

            out_ptr[r] = acc.reduce_add()

    if tasks == 1:
        worker(0)
    else:
        parallelize[worker](tasks)


@always_inline
def ref_gelu_and_mul(
    h1: UnsafePointer[Float32, _],
    h2: UnsafePointer[Float32, _],
    out_ptr: UnsafePointer[Float32, MutAnyOrigin],
    size: Int,
):
    for i in range(0, size, 8):
        var x1 = h1.load[8](i)
        var sig = 1.0 / (1.0 + exp(SIMD[DType.float32, 8](-1.702) * x1))
        var gelu = x1 * sig
        var h2_val = h2.load[8](i)
        out_ptr.store[8](i, gelu * h2_val)


def ref_sc_forward(
    inputs_embeds: UnsafePointer[Float16, _],
    soft_embeds: UnsafePointer[Float32, _],
    sc_gate_packed: UnsafePointer[UInt8, _],
    sc_up_packed: UnsafePointer[UInt8, _],
    sc_down_packed: UnsafePointer[UInt8, _],
    sc_gate_scales: UnsafePointer[Float32, _],
    sc_up_scales: UnsafePointer[Float32, _],
    sc_down_scales: UnsafePointer[Float32, _],
    sc_pre_norm: UnsafePointer[Float32, _],
    buf_normed: UnsafePointer[Float32, MutAnyOrigin],
    buf_gate: UnsafePointer[Float32, MutAnyOrigin],
    buf_up: UnsafePointer[Float32, MutAnyOrigin],
    buf_gated: UnsafePointer[Float32, MutAnyOrigin],
    buf_down: UnsafePointer[Float32, MutAnyOrigin],
    sc_out: UnsafePointer[Float32, MutAnyOrigin],
    embed_dim: Int,
    sc_size: Int,
    num_tasks: Int,
):
    rms_norm(soft_embeds, sc_pre_norm, buf_normed, embed_dim)
    ref_q4_gemv_cpu(sc_gate_packed, sc_gate_scales, buf_normed, buf_gate, embed_dim, sc_size, num_tasks)
    ref_q4_gemv_cpu(sc_up_packed, sc_up_scales, buf_normed, buf_up, embed_dim, sc_size, num_tasks)
    ref_gelu_and_mul(buf_gate, buf_up, buf_gated, sc_size)
    ref_q4_gemv_cpu(sc_down_packed, sc_down_scales, buf_gated, buf_down, sc_size, embed_dim, num_tasks)
    for i in range(0, embed_dim, 8):
        var val = inputs_embeds.load[8](i).cast[DType.float32]() + buf_down.load[8](i)
        sc_out.store[8](i, val)
    rms_norm_noscale(sc_out, sc_out, embed_dim)


def ref_probs_to_soft_embed(
    probs: UnsafePointer[Float32, _],
    embed_packed: UnsafePointer[UInt8, _],
    embed_scales: UnsafePointer[Float32, _],
    sc_out: UnsafePointer[Float32, MutAnyOrigin],
    vocab_size: Int,
    embed_dim: Int,
):
    for d in range(0, embed_dim, 8):
        sc_out.store[8](d, SIMD[DType.float32, 8](0.0))
    var num_blocks = embed_dim // 32
    for r in range(vocab_size):
        var p = probs[r]
        if p < 1e-8:
            continue
        var row_packed_offset = r * (embed_dim // 2)
        var row_scales_offset = r * (embed_dim // 32)
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

            var p_scale_vec = SIMD[DType.float32, 8](p * scale)
            sc_out.store[8](block_out_offset, sc_out.load[8](block_out_offset) + w0 * p_scale_vec)
            sc_out.store[8](block_out_offset + 8, sc_out.load[8](block_out_offset + 8) + w1 * p_scale_vec)
            sc_out.store[8](block_out_offset + 16, sc_out.load[8](block_out_offset + 16) + w2 * p_scale_vec)
            sc_out.store[8](block_out_offset + 24, sc_out.load[8](block_out_offset + 24) + w3 * p_scale_vec)


def cosine_sim(a: UnsafePointer[Float32, _], b: UnsafePointer[Float32, _], n: Int) -> Float32:
    var dot: Float32 = 0.0
    var na: Float32 = 0.0
    var nb: Float32 = 0.0
    for i in range(n):
        dot += a[i] * b[i]
        na += a[i] * a[i]
        nb += b[i] * b[i]
    return dot / (sqrt(na * nb) + 1e-12)


def main() raises:
    print("=== Test & Benchmark: Batched SC MLP & Top-32 Soft-Embedding Sampler (llm-mojo-2tc.1) ===")
    var num_tasks = num_physical_cores()

    # 1. Test sc_forward_batch vs 256 calls to ref_sc_forward
    comptime B = 256
    comptime EMBED_DIM = 2816
    comptime SC_SIZE = 2112
    comptime SC_MAT_SIZE = SC_SIZE * EMBED_DIM
    comptime SC_PACKED = SC_MAT_SIZE // 2
    comptime SC_SCALES = SC_MAT_SIZE // 32

    var sc_gate_packed = alloc[UInt8](SC_PACKED)
    var sc_up_packed = alloc[UInt8](SC_PACKED)
    var sc_down_packed = alloc[UInt8](SC_PACKED)
    var sc_gate_scales = alloc[Float32](SC_SCALES)
    var sc_up_scales = alloc[Float32](SC_SCALES)
    var sc_down_scales = alloc[Float32](SC_SCALES)
    var sc_pre_norm = alloc[Float32](EMBED_DIM)

    for i in range(SC_PACKED):
        sc_gate_packed[i] = UInt8((i * 73 + 19) & 255)
        sc_up_packed[i] = UInt8((i * 41 + 107) & 255)
        sc_down_packed[i] = UInt8((i * 59 + 211) & 255)
    for i in range(SC_SCALES):
        sc_gate_scales[i] = Float32(((i * 17) % 100) - 50) * 0.0005
        sc_up_scales[i] = Float32(((i * 23) % 100) - 50) * 0.0005
        sc_down_scales[i] = Float32(((i * 31) % 100) - 50) * 0.0005
    for i in range(EMBED_DIM):
        sc_pre_norm[i] = Float32(0.8) + Float32(i % 64) * 0.005

    var inputs_embeds = alloc[Float16](B * EMBED_DIM)
    var soft_embeds = alloc[Float32](B * EMBED_DIM)
    var ref_out_f32 = alloc[Float32](B * EMBED_DIM)
    var batch_out_f32 = alloc[Float32](B * EMBED_DIM)
    var ref_out_f16 = alloc[Float16](B * EMBED_DIM)
    var batch_out_f16 = alloc[Float16](B * EMBED_DIM)

    for i in range(B * EMBED_DIM):
        inputs_embeds[i] = Float16(Float32(((i * 13 + 7) % 200) - 100) * 0.02)
        soft_embeds[i] = Float32(((i * 29 + 3) % 200) - 100) * 0.015

    var single_normed = alloc[Float32](EMBED_DIM)
    var single_gate = alloc[Float32](SC_SIZE)
    var single_up = alloc[Float32](SC_SIZE)
    var single_gated = alloc[Float32](SC_SIZE)
    var single_down = alloc[Float32](EMBED_DIM)
    var single_out = alloc[Float32](EMBED_DIM)

    var batch_normed = alloc[Float32](B * EMBED_DIM)
    var batch_gated = alloc[Float32](B * SC_SIZE)
    var batch_down = alloc[Float32](B * EMBED_DIM)

    # Run reference 256-iteration single-token loop
    var t0 = std.time.perf_counter()
    for t in range(B):
        ref_sc_forward(
            inputs_embeds + t * EMBED_DIM,
            soft_embeds + t * EMBED_DIM,
            sc_gate_packed,
            sc_up_packed,
            sc_down_packed,
            sc_gate_scales,
            sc_up_scales,
            sc_down_scales,
            sc_pre_norm,
            single_normed,
            single_gate,
            single_up,
            single_gated,
            single_down,
            single_out,
            EMBED_DIM,
            SC_SIZE,
            num_tasks,
        )
        for i in range(EMBED_DIM):
            ref_out_f32[t * EMBED_DIM + i] = single_out[i]
            ref_out_f16[t * EMBED_DIM + i] = Float16(single_out[i])
    var t_ref_sc = (std.time.perf_counter() - t0) * 1000.0

    # Run batched sc_forward_batch (Float32 output)
    t0 = std.time.perf_counter()
    sc_forward_batch[DType.float16, DType.float32](
        inputs_embeds,
        soft_embeds,
        sc_gate_packed,
        sc_up_packed,
        sc_down_packed,
        sc_gate_scales,
        sc_up_scales,
        sc_down_scales,
        sc_pre_norm,
        batch_normed,
        batch_gated,
        batch_down,
        batch_out_f32,
        B,
        EMBED_DIM,
        SC_SIZE,
        num_tasks,
    )
    var t_batch_sc = (std.time.perf_counter() - t0) * 1000.0

    # Also test in-place Float16 output
    sc_forward_batch[DType.float16, DType.float16](
        inputs_embeds,
        soft_embeds,
        sc_gate_packed,
        sc_up_packed,
        sc_down_packed,
        sc_gate_scales,
        sc_up_scales,
        sc_down_scales,
        sc_pre_norm,
        batch_normed,
        batch_gated,
        batch_down,
        batch_out_f16,
        B,
        EMBED_DIM,
        SC_SIZE,
        num_tasks,
    )

    var max_diff_f32: Float32 = 0.0
    var max_diff_f16: Float32 = 0.0
    for i in range(B * EMBED_DIM):
        var d32 = abs(ref_out_f32[i] - batch_out_f32[i])
        if d32 > max_diff_f32:
            max_diff_f32 = d32
        var d16 = abs(Float32(ref_out_f16[i]) - Float32(batch_out_f16[i]))
        if d16 > max_diff_f16:
            max_diff_f16 = d16

    print(
        "  [SC MLP] max_diff_f32 =",
        max_diff_f32,
        "| max_diff_f16 =",
        max_diff_f16,
        "| Ref 256x single:",
        t_ref_sc,
        "ms -> Batched:",
        t_batch_sc,
        "ms (",
        t_ref_sc / t_batch_sc,
        "x speedup)",
    )
    if max_diff_f32 > 1e-5:
        raise Error("sc_forward_batch f32 error exceeded 1e-5!")
    if max_diff_f16 > 2e-3:
        raise Error("sc_forward_batch f16 error exceeded 2e-3!")

    # 2. Test Top-32 Soft-Embedding Sampler vs Full-Vocab Sampler
    comptime VOCAB_SIZE = 32768
    var embed_packed = alloc[UInt8](VOCAB_SIZE * (EMBED_DIM // 2))
    var embed_scales = alloc[Float32](VOCAB_SIZE * (EMBED_DIM // 32))
    for i in range(VOCAB_SIZE * (EMBED_DIM // 2)):
        embed_packed[i] = UInt8((i * 37 + 11) & 255)
    for i in range(VOCAB_SIZE * (EMBED_DIM // 32)):
        embed_scales[i] = Float32(((i * 19) % 100) - 50) * 0.001

    var canvas_logits_raw = alloc[Float32](B * VOCAB_SIZE)
    var canvas_logits_ref = alloc[Float32](B * VOCAB_SIZE)
    var canvas_probs_ref = alloc[Float32](B * VOCAB_SIZE)
    var rand_u = alloc[Float32](B)

    for t in range(B):
        rand_u[t] = Float32((t * 7 + 3) % 100) * 0.01
        var base_l = t * VOCAB_SIZE
        # Background noise logits in [-6, -2]
        for v in range(VOCAB_SIZE):
            canvas_logits_raw[base_l + v] = Float32(-6.0) + Float32((v * 17 + t * 31) % 400) * 0.01
        # Plant a realistic Zipfian / peaked cluster of ~16 strong candidate tokens in [2.0, 14.0]
        for k in range(16):
            var hot_v = (t * 197 + k * 997 + 13) % VOCAB_SIZE
            canvas_logits_raw[base_l + hot_v] = Float32(14.0) - Float32(k) * 0.75

    var ref_entropy = alloc[Float32](B)
    var ref_argmax = alloc[Int](B)
    var ref_sampled = alloc[Int](B)
    var ref_sc_embeds = alloc[Float32](B * EMBED_DIM)

    var fast_entropy = alloc[Float32](B)
    var fast_argmax = alloc[Int](B)
    var fast_sampled = alloc[Int](B)
    var fast_sc_embeds = alloc[Float32](B * EMBED_DIM)

    var prompt_scale = sqrt(Float32(EMBED_DIM))
    var temps = InlineArray[Float32, 3](uninitialized=True)
    temps[0] = 0.4
    temps[1] = 0.6
    temps[2] = 0.8

    for ti in range(3):
        var temp = temps[ti]
        for i in range(B * VOCAB_SIZE):
            canvas_logits_ref[i] = canvas_logits_raw[i]

        var ts0 = std.time.perf_counter()
        def ref_sampler_worker(t: Int) capturing -> None:
            var logits_t = canvas_logits_ref + t * VOCAB_SIZE
            var probs_t = canvas_probs_ref + t * VOCAB_SIZE
            for v in range(VOCAB_SIZE):
                logits_t[v] = (Float32(30.0) * tanh(logits_t[v] / Float32(30.0))) / temp
            var max_logit = logits_t[0]
            var t_argmax = 0
            for v in range(1, VOCAB_SIZE):
                if logits_t[v] > max_logit:
                    max_logit = logits_t[v]
                    t_argmax = v
            ref_argmax[t] = t_argmax
            var sum_exp: Float32 = 0.0
            for v in range(VOCAB_SIZE):
                var lp = exp(logits_t[v] - max_logit)
                probs_t[v] = lp
                sum_exp += lp
            var log_sum = log(sum_exp + 1e-30)
            var ent: Float32 = 0.0
            for v in range(VOCAB_SIZE):
                var p = probs_t[v] / (sum_exp + 1e-30)
                probs_t[v] = p
                if p > 1e-12:
                    ent -= p * (logits_t[v] - max_logit - log_sum)
            ref_entropy[t] = ent
            var u = rand_u[t]
            var cumsum: Float32 = 0.0
            var sampled = t_argmax
            for v in range(VOCAB_SIZE):
                cumsum += probs_t[v]
                if cumsum >= u:
                    sampled = v
                    break
            ref_sampled[t] = sampled
            ref_probs_to_soft_embed(
                probs_t,
                embed_packed,
                embed_scales,
                ref_sc_embeds + t * EMBED_DIM,
                VOCAB_SIZE,
                EMBED_DIM,
            )
            var sc_t = ref_sc_embeds + t * EMBED_DIM
            var p_vec = SIMD[DType.float32, 8](prompt_scale)
            for i in range(0, EMBED_DIM, 8):
                sc_t.store[8](i, sc_t.load[8](i) * p_vec)

        parallelize[ref_sampler_worker](B)
        var t_ref_samp = (std.time.perf_counter() - ts0) * 1000.0

        ts0 = std.time.perf_counter()
        sample_and_soft_embed_top32_batch[32](
            canvas_logits_raw,
            rand_u,
            embed_packed,
            embed_scales,
            fast_entropy,
            fast_argmax,
            fast_sampled,
            fast_sc_embeds,
            B,
            VOCAB_SIZE,
            EMBED_DIM,
            temp,
            prompt_scale,
            num_tasks,
        )
        var t_fast_samp = (std.time.perf_counter() - ts0) * 1000.0

        var min_cos: Float32 = 1.0
        var max_ent_diff: Float32 = 0.0
        var argmax_matches = 0
        for t in range(B):
            if ref_argmax[t] == fast_argmax[t]:
                argmax_matches += 1
            var cs = cosine_sim(
                ref_sc_embeds + t * EMBED_DIM,
                fast_sc_embeds + t * EMBED_DIM,
                EMBED_DIM,
            )
            if cs < min_cos:
                min_cos = cs
            var ed = abs(ref_entropy[t] - fast_entropy[t])
            if ed > max_ent_diff:
                max_ent_diff = ed
            if fast_sampled[t] < 0 or fast_sampled[t] >= VOCAB_SIZE:
                raise Error("Invalid sampled token index!")

        print(
            "  [Sampler T=",
            temp,
            "] argmax_match =",
            argmax_matches,
            "/",
            B,
            "| min_cosine_sim =",
            min_cos,
            "| max_ent_diff =",
            max_ent_diff,
            "| Ref:",
            t_ref_samp,
            "ms -> Top-32:",
            t_fast_samp,
            "ms (",
            t_ref_samp / t_fast_samp,
            "x speedup)",
        )
        if argmax_matches != B:
            raise Error("Argmax mismatch in sample_and_soft_embed_top32_batch!")
        if min_cos < 0.9995:
            raise Error("Cosine similarity below 0.9995 threshold!")

    # 3. Adversarial test: Flat / uniform logits regime (step 0 noise)
    for i in range(B * VOCAB_SIZE):
        canvas_logits_raw[i] = 0.0
    var t_u0 = std.time.perf_counter()
    sample_and_soft_embed_top32_batch[32](
        canvas_logits_raw,
        rand_u,
        embed_packed,
        embed_scales,
        fast_entropy,
        fast_argmax,
        fast_sampled,
        fast_sc_embeds,
        B,
        VOCAB_SIZE,
        EMBED_DIM,
        Float32(0.8),
        prompt_scale,
        num_tasks,
    )
    var t_fast_uniform = (std.time.perf_counter() - t_u0) * 1000.0
    var expected_uniform_ent = log(Float32(VOCAB_SIZE))
    var uniform_ent_diff = abs(fast_entropy[0] - expected_uniform_ent)

    # Measure ref_sampler_worker on step-0 uniform/high-entropy logits
    for i in range(B * VOCAB_SIZE):
        canvas_logits_ref[i] = 0.0
    var t_u_ref0 = std.time.perf_counter()
    def ref_uniform_worker(t: Int) capturing -> None:
        var logits_t = canvas_logits_ref + t * VOCAB_SIZE
        var probs_t = canvas_probs_ref + t * VOCAB_SIZE
        for v in range(VOCAB_SIZE):
            logits_t[v] = (Float32(30.0) * tanh(logits_t[v] / Float32(30.0))) / Float32(0.8)
        var max_logit = logits_t[0]
        for v in range(1, VOCAB_SIZE):
            if logits_t[v] > max_logit:
                max_logit = logits_t[v]
        var sum_exp: Float32 = 0.0
        for v in range(VOCAB_SIZE):
            var lp = exp(logits_t[v] - max_logit)
            probs_t[v] = lp
            sum_exp += lp
        for v in range(VOCAB_SIZE):
            probs_t[v] = probs_t[v] / (sum_exp + 1e-30)
        ref_probs_to_soft_embed(
            probs_t,
            embed_packed,
            embed_scales,
            ref_sc_embeds + t * EMBED_DIM,
            VOCAB_SIZE,
            EMBED_DIM,
        )
    parallelize[ref_uniform_worker](B)
    var t_ref_uniform = (std.time.perf_counter() - t_u_ref0) * 1000.0

    print(
        "  [Step-0 High-Entropy] fast_entropy =",
        fast_entropy[0],
        "| expected ln(V) =",
        expected_uniform_ent,
        "| diff =",
        uniform_ent_diff,
        "| Ref step-0:",
        t_ref_uniform,
        "ms -> Top-32 step-0:",
        t_fast_uniform,
        "ms (",
        t_ref_uniform / t_fast_uniform,
        "x speedup)",
    )
    if uniform_ent_diff > 1e-3:
        raise Error("Uniform noise entropy cumulant estimate failed!")

    # 4. Regression test: Peaked distribution with -30.0 <unused*> outlier logits in the tail
    # Verify that negative outlier logits do NOT blow up fast_entropy to ln(V) ~ 10.4!
    for t in range(B):
        var base_l = t * VOCAB_SIZE
        for v in range(VOCAB_SIZE):
            canvas_logits_raw[base_l + v] = Float32(-30.0) if (v % 4 == 0) else Float32(-4.0)
        canvas_logits_raw[base_l + 300] = Float32(15.0)
        for k in range(1, 32):
            canvas_logits_raw[base_l + 300 + k] = Float32(8.0) - Float32(k) * 0.15

    sample_and_soft_embed_top32_batch[32](
        canvas_logits_raw,
        rand_u,
        embed_packed,
        embed_scales,
        fast_entropy,
        fast_argmax,
        fast_sampled,
        fast_sc_embeds,
        B,
        VOCAB_SIZE,
        EMBED_DIM,
        Float32(0.6),
        prompt_scale,
        num_tasks,
    )
    print("  [Outlier Tail Guard] peaked + (-30.0 outliers) fast_entropy =", fast_entropy[0])
    if fast_entropy[0] > 0.05:
        raise Error("Negative tail outliers (-30.0) inflated Top-32 token entropy!")

    sc_gate_packed.free()
    sc_up_packed.free()
    sc_down_packed.free()
    sc_gate_scales.free()
    sc_up_scales.free()
    sc_down_scales.free()
    sc_pre_norm.free()
    inputs_embeds.free()
    soft_embeds.free()
    ref_out_f32.free()
    batch_out_f32.free()
    ref_out_f16.free()
    batch_out_f16.free()
    single_normed.free()
    single_gate.free()
    single_up.free()
    single_gated.free()
    single_down.free()
    single_out.free()
    batch_normed.free()
    batch_gated.free()
    batch_down.free()
    embed_packed.free()
    embed_scales.free()
    canvas_logits_raw.free()
    canvas_logits_ref.free()
    canvas_probs_ref.free()
    rand_u.free()
    ref_entropy.free()
    ref_argmax.free()
    ref_sampled.free()
    ref_sc_embeds.free()
    fast_entropy.free()
    fast_argmax.free()
    fast_sampled.free()
    fast_sc_embeds.free()
    print("=== ALL TESTS PASSED! ===")
