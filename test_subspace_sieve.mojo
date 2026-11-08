from std.memory import alloc
from std.algorithm import parallelize
from std.sys.info import num_physical_cores
from std.math import sqrt, sin, cos, tanh
import std.time
from q4_embed import Q4Embedding
from moe_kernels import quantize_fp32_to_q4_0


def test_sieve_correctness_and_speed() raises:
    print("=== TDD Test: Contiguous Prefix/Tail Subspace Sieve (llm-mojo-xr6.1) ===")
    var num_tasks = num_physical_cores()
    var vocab_size = 262144
    var embed_dim = 1024

    var emb = Q4Embedding(vocab_size, embed_dim)
    var half_dim = embed_dim // 2
    var num_blocks = embed_dim // 32
    var packed_ptr = emb.packed_weights
    var scales_ptr = emb.scales

    def init_worker(t: Int) capturing -> None:
        var row_fp32 = alloc[Float32](embed_dim)
        var chunk = vocab_size // num_tasks
        var sr = t * chunk
        var er = (t + 1) * chunk if t < num_tasks - 1 else vocab_size
        for r in range(sr, er):
            var scale_r = Float32(0.35 + 0.75 * Float64((r * 73 + (r >> 8)) % 100) / 100.0)
            if r == 42123 or r == 198765:
                scale_r = 1.45
            for j in range(embed_dim):
                var h = ((r + 1) * 1664525 + (j + 1) * 1013904223) ^ ((r * 31337) >> 4) ^ (j * 65537)
                var v = Float32((h & 255) - 128) * 0.008 * scale_r
                row_fp32[j] = v
            quantize_fp32_to_q4_0(
                row_fp32,
                packed_ptr + r * half_dim,
                scales_ptr + r * num_blocks,
                embed_dim,
                1,
            )
        row_fp32.free()

    parallelize[init_worker](num_tasks)
    emb.finalize_tail_bounds()

    var x_batch = alloc[Float32](5 * embed_dim)
    # Create 5 query vectors where token 0 aligns with row 42123, etc., scaled so raw dot product ~ 15..25
    for b in range(5):
        var target_r = 42123 if b == 0 else (10000 + b * 11111)
        emb.reconstruct(target_r, x_batch + b * embed_dim)
        for j in range(embed_dim):
            x_batch[b * embed_dim + j] = x_batch[b * embed_dim + j] * 0.025 + Float32(((b * 37 + j * 13) % 19) - 9) * 0.002

    var ref_logits = alloc[Float32](num_tasks)
    var ref_tokens = alloc[Int](num_tasks)
    var fast_logits = alloc[Float32](num_tasks)
    var fast_tokens = alloc[Int](num_tasks)

    # 1. Verify single-token project_logits_parallel vs reference
    for b in range(5):
        var x_ptr = x_batch + b * embed_dim
        emb.project_logits_parallel_reference(x_ptr, ref_logits, ref_tokens, num_tasks)
        var ref_best_l: Float32 = -1e30
        var ref_best_t: Int = 0
        for t in range(num_tasks):
            if ref_logits[t] > ref_best_l:
                ref_best_l = ref_logits[t]
                ref_best_t = ref_tokens[t]

        emb.project_logits_parallel(x_ptr, fast_logits, fast_tokens, num_tasks, softcap=True)
        var fast_best_l: Float32 = -1e30
        var fast_best_t: Int = 0
        for t in range(num_tasks):
            if fast_logits[t] > fast_best_l:
                fast_best_l = fast_logits[t]
                fast_best_t = fast_tokens[t]

        var diff = abs(ref_best_l - fast_best_l)
        print("  Token", b, "-> ref_tok:", ref_best_t, "fast_tok:", fast_best_t, "logit_diff:", diff)
        if ref_best_t != fast_best_t or diff > 1e-3:
            raise Error("Single-token sieve mismatch on token " + String(b))

    # 2. Verify batch5 project_logits_batch5_parallel vs reference
    var seed_toks = alloc[Int](5)
    var out_toks = alloc[Int](5)
    for b in range(5):
        seed_toks[b] = 42123 if b == 0 else (10000 + b * 11111)
    emb.project_logits_batch5_parallel(x_batch, seed_toks, out_toks, num_tasks, softcap=True)
    for b in range(5):
        var x_ptr = x_batch + b * embed_dim
        emb.project_logits_parallel_reference(x_ptr, ref_logits, ref_tokens, num_tasks)
        var ref_best_l: Float32 = -1e30
        var ref_best_t: Int = 0
        for t in range(num_tasks):
            if ref_logits[t] > ref_best_l:
                ref_best_l = ref_logits[t]
                ref_best_t = ref_tokens[t]
        if out_toks[b] != ref_best_t:
            raise Error("Batch5 sieve mismatch on token " + String(b))

    # 3. Benchmark 4-step single-token (MTP style) + 1-step batch5
    var iters = 25
    var t0 = std.time.perf_counter()
    for _ in range(iters):
        for b in range(4):
            emb.project_logits_parallel(x_batch + b * embed_dim, fast_logits, fast_tokens, num_tasks, softcap=False)
    var t1 = std.time.perf_counter()
    var single_ms = (t1 - t0) * 1000.0 / Float64(iters)

    var t2 = std.time.perf_counter()
    for _ in range(iters):
        emb.project_logits_batch5_parallel(x_batch, seed_toks, out_toks, num_tasks, softcap=True)
    var t3 = std.time.perf_counter()
    var batch5_ms = (t3 - t2) * 1000.0 / Float64(iters)

    print("  4x Single-Token (MTP 262k x 1024) latency:", single_ms, "ms")
    print("  1x Batch-5 (262k x 1024) latency:         ", batch5_ms, "ms")
    x_batch.free()
    emb.free()

    # Test real gemma4_moe_embed_q4.bin (262,144 x 2816)
    var real_emb = Q4Embedding("gemma4_moe_embed_q4.bin", 262144, 2816)
    var real_x = alloc[Float32](5 * real_emb.embed_dim)
    for b in range(5):
        var t_id = 1000 + b * 5000
        real_emb.reconstruct(t_id, real_x + b * real_emb.embed_dim)
        var row_sq: Float32 = 0.0
        for j in range(real_emb.embed_dim):
            var v = real_x[b * real_emb.embed_dim + j]
            row_sq += v * v
        var row_rms = sqrt(row_sq / Float32(real_emb.embed_dim))
        if b == 0:
            print("  gemma4_moe_embed_q4 row 1000 rms:", row_rms)
        for j in range(real_emb.embed_dim):
            real_x[b * real_emb.embed_dim + j] = real_x[b * real_emb.embed_dim + j] + Float32(((b * 37 + j * 13) % 19) - 9) * (row_rms * 0.25)
        seed_toks[b] = t_id

    # Measure actual max tail cosine similarity across all 262,144 rows of gemma4_moe_embed_q4.bin
    var nb = real_emb.embed_dim // 32
    var b25 = nb // 4
    var hd = real_emb.embed_dim // 2
    var xe_test = real_emb.scratch_x_even
    var xo_test = real_emb.scratch_x_odd
    var sq_tail: Float32 = 0.0
    var sq_full: Float32 = 0.0
    for b in range(nb):
        var v0 = real_x.load[16](b * 32)
        var v1 = real_x.load[16](b * 32 + 16)
        var ev = v0.deinterleave()[0].join(v1.deinterleave()[0])
        var od = v0.deinterleave()[1].join(v1.deinterleave()[1])
        xe_test.store[16](b * 16, ev)
        xo_test.store[16](b * 16, od)
        var bsq = (ev * ev + od * od).reduce_add()
        sq_full += bsq
        if b >= b25:
            sq_tail += bsq
    var x_tail_rms_val = sqrt(sq_tail)
    var winner_full = real_emb._eval_row_full(1000, xe_test, xo_test, nb)
    var max_tail_cos_all: Float32 = 0.0
    var max_tail_cos_nonwin: Float32 = 0.0
    for r in range(real_emb.vocab_size):
        var acc_p = SIMD[DType.float32, 16](0.0)
        var acc_t = SIMD[DType.float32, 16](0.0)
        var rp = r * hd
        var rs = r * nb
        for b in range(b25):
            var sc = real_emb.scales[rs + b]
            var by = real_emb.packed_weights.load[16](rp + b * 16)
            var we = ((by & 15).cast[DType.int8]() - 8).cast[DType.float32]()
            var wo = ((by >> 4).cast[DType.int8]() - 8).cast[DType.float32]()
            acc_p = acc_p + (we * xe_test.load[16](b * 16) + wo * xo_test.load[16](b * 16)) * sc
        for b in range(b25, nb):
            var sc = real_emb.scales[rs + b]
            var by = real_emb.packed_weights.load[16](rp + b * 16)
            var we = ((by & 15).cast[DType.int8]() - 8).cast[DType.float32]()
            var wo = ((by >> 4).cast[DType.int8]() - 8).cast[DType.float32]()
            acc_t = acc_t + (we * xe_test.load[16](b * 16) + wo * xo_test.load[16](b * 16)) * sc
        var t_dot = acc_t.reduce_add()
        var denom = real_emb.row_tail_bound_25[r] * x_tail_rms_val
        if denom > 1e-6:
            var t_cos = t_dot / denom
            if t_cos > max_tail_cos_all:
                max_tail_cos_all = t_cos
            if r != 1000 and t_cos > max_tail_cos_nonwin:
                max_tail_cos_nonwin = t_cos
    print("  Winner full dot:", winner_full, "max_tail_cos (all):", max_tail_cos_all, "max_tail_cos (r != 1000):", max_tail_cos_nonwin)

    real_emb.project_logits_batch5_parallel(real_x, seed_toks, out_toks, num_tasks, softcap=True)
    var tr0 = std.time.perf_counter()
    for _ in range(10):
        real_emb.project_logits_parallel(real_x, fast_logits, fast_tokens, num_tasks, softcap=True)
    var tr1 = std.time.perf_counter()
    var tr2 = std.time.perf_counter()
    for _ in range(10):
        real_emb.project_logits_batch5_parallel(real_x, seed_toks, out_toks, num_tasks, softcap=True)
    var tr3 = std.time.perf_counter()
    print("  Real 1x Target (262k x 2816) single latency:", (tr1 - tr0) * 100.0, "ms")
    print("  Real 1x Target (262k x 2816) batch5 latency:", (tr3 - tr2) * 100.0, "ms")
    real_x.free()
    real_emb.free()

    seed_toks.free()
    out_toks.free()
    ref_logits.free()
    ref_tokens.free()
    fast_logits.free()
    fast_tokens.free()
    print("=== PASS ===")


def main() raises:
    test_sieve_correctness_and_speed()
