from std.memory import alloc
import std.time
from std.math import sqrt, abs
from q4_embed import Q4Embedding


def main() raises:
    print("==================================================================")
    print("   TDD & BENCHMARK: Early-Z Prefix Culling + Monotonic tanh PVS   ")
    print("==================================================================")
    var embed = Q4Embedding("gemma4_moe_embed_q4.bin", 262144, 2816)

    var num_tasks = 8
    var x = alloc[Float32](2816)
    var tmp_vec = alloc[Float32](2816)
    var ref_max_l = alloc[Float32](num_tasks)
    var ref_toks = alloc[Int](num_tasks)
    var opt_max_l = alloc[Float32](num_tasks)
    var opt_toks = alloc[Int](num_tasks)

    var num_trials = 60
    var matches = 0
    var total_ref_ms: Float64 = 0.0
    var total_opt_ms: Float64 = 0.0

    for trial in range(num_trials):
        # Construct realistic + adversarial close-tie hidden states x
        var tok_a = (trial * 4099 + 17) % 262144
        var tok_b = (trial * 1031 + 5005) % 262144
        embed.reconstruct(tok_a, x)
        embed.reconstruct(tok_b, tmp_vec)
        var w_a: Float32 = 0.501 if (trial % 3 == 0) else 0.65
        var w_b: Float32 = 0.499 if (trial % 3 == 0) else 0.35
        for i in range(2816):
            var noise = Float32(((i * 17 + trial * 31) % 29) - 14) * 0.02
            x[i] = x[i] * w_a + tmp_vec[i] * w_b + noise

        var t0 = std.time.perf_counter()
        embed.project_logits_parallel_reference(x, ref_max_l, ref_toks, num_tasks)
        var t1 = std.time.perf_counter()
        embed.project_logits_parallel(x, opt_max_l, opt_toks, num_tasks)
        var t2 = std.time.perf_counter()

        total_ref_ms += (t1 - t0) * 1000.0
        total_opt_ms += (t2 - t1) * 1000.0

        var best_ref_l: Float32 = -999999.0
        var best_ref_tok = 0
        var best_opt_l: Float32 = -999999.0
        var best_opt_tok = 0
        for t in range(num_tasks):
            if ref_max_l[t] > best_ref_l:
                best_ref_l = ref_max_l[t]
                best_ref_tok = ref_toks[t]
            if opt_max_l[t] > best_opt_l:
                best_opt_l = opt_max_l[t]
                best_opt_tok = opt_toks[t]

        if best_ref_tok == best_opt_tok and abs(best_ref_l - best_opt_l) < 1e-5:
            matches += 1
        else:
            print(
                "  [MISMATCH] trial",
                trial,
                "ref_tok:",
                best_ref_tok,
                "opt_tok:",
                best_opt_tok,
                "ref_l:",
                best_ref_l,
                "opt_l:",
                best_opt_l,
            )

    var avg_ref_ms = total_ref_ms / Float64(num_trials)
    var avg_opt_ms = total_opt_ms / Float64(num_trials)
    var speedup = avg_ref_ms / avg_opt_ms

    print("  Trials matched:       ", matches, "/", num_trials)
    print("  Reference avg time:   ", avg_ref_ms, "ms/token")
    print("  Early-Z PVS avg time: ", avg_opt_ms, "ms/token")
    print("  Speedup factor:       ", speedup, "x")

    x.free()
    tmp_vec.free()
    ref_max_l.free()
    ref_toks.free()
    opt_max_l.free()
    opt_toks.free()
    embed.free()

    if matches != num_trials:
        raise Error("Early-Z PVS logit culling failed recall test: " + String(matches) + "/" + String(num_trials))
    print("  [PASS] 100% Top-1 Token & Logit Identity Verified!")
