from std.memory import alloc
from std.math import sqrt, abs
from std.algorithm import parallelize
from std.sys.info import num_physical_cores
import std.time

from moe_kernels import quantize_fp32_to_q4_0, q4_matmul_vec, q4_matmul_batch5
from q4_embed import Q4Embedding


@always_inline
def fp32_gemv(
    mat: UnsafePointer[Float32, _],
    vec: UnsafePointer[Float32, _],
    out_ptr: UnsafePointer[Float32, MutUntrackedOrigin],
    rows: Int,
    cols: Int,
    num_tasks: Int,
):
    var tasks = num_tasks
    if tasks > rows:
        tasks = rows
    if tasks < 1:
        tasks = 1
    var chunk_size = rows // tasks

    def worker(t: Int) capturing -> None:
        var start_r = t * chunk_size
        var end_r = (t + 1) * chunk_size if t < tasks - 1 else rows
        for r in range(start_r, end_r):
            var acc = SIMD[DType.float32, 16](0.0)
            var row_offset = r * cols
            for c in range(0, cols, 16):
                acc = acc + mat.load[16](row_offset + c) * vec.load[16](c)
            out_ptr[r] = acc.reduce_add()

    if tasks == 1:
        worker(0)
    else:
        parallelize[worker](tasks)


def cosine_sim(a: UnsafePointer[Float32, _], b: UnsafePointer[Float32, _], n: Int) -> Float32:
    var dot: Float32 = 0.0
    var na: Float32 = 0.0
    var nb: Float32 = 0.0
    for i in range(n):
        dot += a[i] * b[i]
        na += a[i] * a[i]
        nb += b[i] * b[i]
    return dot / (sqrt(na) * sqrt(nb) + 1e-12)


def main() raises:
    print("=================================================================")
    print("  TDD: Q4_0 Quantized MTP Speculative Draft Engine + Fast Argmax ")
    print("=================================================================")
    var num_tasks = max(1, num_physical_cores())
    print("Detected physical cores:", num_tasks)

    # 1. Test quantize_fp32_to_q4_0 + q4_matmul_vec vs fp32_gemv on MTP shapes
    var rows = 1024
    var cols = 5632
    var n_elem = rows * cols
    var w_fp32 = alloc[Float32](n_elem)
    var w_packed = alloc[UInt8](n_elem // 2)
    var w_scales = alloc[Float32](n_elem // 32)
    var x = alloc[Float32](cols)
    var out_fp32 = alloc[Float32](rows)
    var out_q4 = alloc[Float32](rows)

    for i in range(n_elem):
        w_fp32[i] = Float32(((i * 73 + 19) % 251) - 125) * 0.008
    for i in range(cols):
        x[i] = Float32(((i * 31 + 7) % 199) - 99) * 0.01

    quantize_fp32_to_q4_0(w_fp32, w_packed, w_scales, n_elem, num_tasks)
    fp32_gemv(w_fp32, x, out_fp32, rows, cols, num_tasks)
    q4_matmul_vec(w_packed, w_scales, x, out_q4, cols, rows, num_tasks)

    var cs = cosine_sim(out_fp32, out_q4, rows)
    print("[Test 1] pre_proj [1024, 5632] Q4_0 vs FP32 cosine similarity:", cs)
    if cs < 0.995:
        raise Error("FAILURE: Cosine similarity below 0.995: " + String(cs))

    w_fp32.free()
    w_packed.free()
    w_scales.free()
    x.free()
    out_fp32.free()
    out_q4.free()

    # 2. Test in-memory Q4Embedding + project_logits_parallel(softcap=False) vs FP32 argmax
    # Including large logits (> 400.0) that would saturate 30 * tanh(x / 30) == 30.0 in Float32!
    var vocab_test = 16384
    var dim_test = 1024
    var lm_n = vocab_test * dim_test
    var lm_fp32 = alloc[Float32](lm_n)
    var h_vec = alloc[Float32](dim_test)
    var logits_fp32 = alloc[Float32](vocab_test)

    for i in range(lm_n):
        lm_fp32[i] = Float32(((i * 41 + 13) % 127) - 63) * 0.01
    for j in range(dim_test):
        h_vec[j] = 1.0

    # Plant two strong rows at r=3000 (score ~ 450) and r=12345 (score ~ 680)
    # Note: both 450/30 = 15 and 680/30 = 22.6 have tanh(.) == 1.0 in Float32!
    # So softcap=True would tie at 30.0 and pick r=3000, whereas softcap=False must pick r=12345!
    for j in range(dim_test):
        lm_fp32[3000 * dim_test + j] = 0.45
        lm_fp32[12345 * dim_test + j] = 0.68

    var q4_lm = Q4Embedding(vocab_test, dim_test)
    quantize_fp32_to_q4_0(lm_fp32, q4_lm.packed_weights, q4_lm.scales, lm_n, num_tasks)
    q4_lm.finalize_tail_bounds()

    fp32_gemv(lm_fp32, h_vec, logits_fp32, vocab_test, dim_test, num_tasks)
    var fp32_best_tok = 0
    var fp32_best_val = logits_fp32[0]
    for r in range(1, vocab_test):
        if logits_fp32[r] > fp32_best_val:
            fp32_best_val = logits_fp32[r]
            fp32_best_tok = r

    var local_max_l = alloc[Float32](num_tasks)
    var local_next_t = alloc[Int](num_tasks)
    q4_lm.project_logits_parallel(h_vec, local_max_l, local_next_t, num_tasks, softcap=False)
    var q4_best_tok = 0
    var q4_best_val: Float32 = -1e30
    for t in range(num_tasks):
        if local_max_l[t] > q4_best_val:
            q4_best_val = local_max_l[t]
            q4_best_tok = local_next_t[t]

    print(
        "[Test 2] Un-softcapped argmax: FP32 tok =",
        fp32_best_tok,
        "(val =",
        fp32_best_val,
        ") | Q4_0 PVS tok =",
        q4_best_tok,
        "(val =",
        q4_best_val,
        ")",
    )
    if fp32_best_tok != 12345 or q4_best_tok != 12345:
        raise Error("FAILURE: Expected winner token 12345, got FP32=" + String(fp32_best_tok) + " Q4=" + String(q4_best_tok))

    q4_lm.free()
    lm_fp32.free()
    logits_fp32.free()

    # 3. Full-scale [262144, 1024] MTP LM Head + FFN [8192, 1024] / [1024, 8192] Benchmark (Q4_0 vs FP32)
    var full_vocab = 262144
    var full_lm_q4 = Q4Embedding(full_vocab, dim_test)
    var total_bytes_packed = full_vocab * (dim_test // 2)
    var total_blocks_lm = full_vocab * (dim_test // 32)
    for i in range(total_bytes_packed):
        full_lm_q4.packed_weights[i] = UInt8((i * 17 + 5) & 255)
    for i in range(total_blocks_lm):
        full_lm_q4.scales[i] = Float32(((i * 13) % 97) + 1) * 0.001
    full_lm_q4.finalize_tail_bounds()

    # Plant winning token at 77777
    for b in range(dim_test // 32):
        full_lm_q4.scales[77777 * (dim_test // 32) + b] = 0.25
        for k in range(16):
            full_lm_q4.packed_weights[77777 * (dim_test // 2) + b * 16 + k] = UInt8(15 | (15 << 4))
    full_lm_q4.finalize_tail_bounds()

    # Allocate a single FP32 chunk [65536, 1024] (256 MB) and run 4x to measure exact 262,144-row FP32 bandwidth cost
    # without needing an extra 1 GB allocation alongside the test
    var chunk_vocab = 65536
    var fp32_chunk = alloc[Float32](chunk_vocab * dim_test)
    var fp32_logits_full = alloc[Float32](full_vocab)
    for i in range(chunk_vocab * dim_test):
        fp32_chunk[i] = Float32(i & 63) * 0.001

    var iters = 10
    # Warmup
    full_lm_q4.project_logits_parallel(h_vec, local_max_l, local_next_t, num_tasks, softcap=False)
    for c in range(4):
        fp32_gemv(fp32_chunk, h_vec, fp32_logits_full + c * chunk_vocab, chunk_vocab, dim_test, num_tasks)

    var t0 = std.time.perf_counter()
    var dummy_tok = 0
    for _ in range(iters):
        for c in range(4):
            fp32_gemv(fp32_chunk, h_vec, fp32_logits_full + c * chunk_vocab, chunk_vocab, dim_test, num_tasks)
        var max_v = fp32_logits_full[0]
        var max_i = 0
        for r in range(1, full_vocab):
            if fp32_logits_full[r] > max_v:
                max_v = fp32_logits_full[r]
                max_i = r
        dummy_tok += max_i
    var fp32_lm_ms = (std.time.perf_counter() - t0) * 1000.0 / Float64(iters)

    var t1 = std.time.perf_counter()
    var q4_winner = 0
    for _ in range(iters):
        full_lm_q4.project_logits_parallel(h_vec, local_max_l, local_next_t, num_tasks, softcap=False)
        var best_v: Float32 = -1e30
        var best_i = 0
        for t in range(num_tasks):
            if local_max_l[t] > best_v:
                best_v = local_max_l[t]
                best_i = local_next_t[t]
        q4_winner = best_i
    var q4_lm_ms = (std.time.perf_counter() - t1) * 1000.0 / Float64(iters)

    print("[Test 3] 262,144 x 1,024 MTP LM Head Latency:")
    print("  FP32 GEMV + Linear Scan : ", fp32_lm_ms, "ms/step")
    print("  Q4_0 PVS Fast Argmax    : ", q4_lm_ms, "ms/step (winner =", q4_winner, ")")
    print("  Speedup                 : ", fp32_lm_ms / q4_lm_ms, "x")
    if q4_winner != 77777 or dummy_tok < 0:
        raise Error("FAILURE: Expected winner 77777, got " + String(q4_winner))

    # 3b. Test project_logits_batch5_parallel vs 5x project_logits_parallel
    var b5_h = alloc[Float32](5 * dim_test)
    var b5_seeds = alloc[Int](5)
    var b5_ref_toks = alloc[Int](5)
    var b5_got_toks = alloc[Int](5)
    for tok in range(5):
        b5_seeds[tok] = 77777
        for j in range(dim_test):
            b5_h[tok * dim_test + j] = Float32(((j * 17 + tok * 31) % 97) + 1) * 0.01

    for tok in range(5):
        full_lm_q4.last_winner_tok[0] = b5_seeds[tok]
        full_lm_q4.project_logits_parallel(b5_h + tok * dim_test, local_max_l, local_next_t, num_tasks, softcap=True)
        var best_v: Float32 = -1e30
        var best_i = 0
        for t in range(num_tasks):
            if local_max_l[t] > best_v or (local_max_l[t] == best_v and local_next_t[t] < best_i):
                best_v = local_max_l[t]
                best_i = local_next_t[t]
        b5_ref_toks[tok] = best_i

    full_lm_q4.project_logits_batch5_parallel(b5_h, b5_seeds, b5_got_toks, num_tasks, softcap=True)
    for tok in range(5):
        if b5_ref_toks[tok] != b5_got_toks[tok]:
            raise Error(
                "FAILURE: project_logits_batch5_parallel mismatch at tok "
                + String(tok)
                + ": ref="
                + String(b5_ref_toks[tok])
                + " got="
                + String(b5_got_toks[tok])
            )
    var t_b5_0 = std.time.perf_counter()
    for _ in range(iters):
        for tok in range(5):
            full_lm_q4.last_winner_tok[0] = b5_seeds[tok]
            full_lm_q4.project_logits_parallel(b5_h + tok * dim_test, local_max_l, local_next_t, num_tasks, softcap=True)
    var seq5_lm_ms = (std.time.perf_counter() - t_b5_0) * 1000.0 / Float64(iters)

    var t_b5_1 = std.time.perf_counter()
    for _ in range(iters):
        full_lm_q4.project_logits_batch5_parallel(b5_h, b5_seeds, b5_got_toks, num_tasks, softcap=True)
    var coal5_lm_ms = (std.time.perf_counter() - t_b5_1) * 1000.0 / Float64(iters)
    print(
        "[Test 3b] 5-Token Coalesced LM Head Verification:",
        seq5_lm_ms,
        "ms ->",
        coal5_lm_ms,
        "ms (",
        seq5_lm_ms / coal5_lm_ms,
        "x speedup, 5/5 exact match)",
    )

    b5_h.free()
    b5_seeds.free()
    b5_ref_toks.free()
    b5_got_toks.free()

    fp32_chunk.free()
    fp32_logits_full.free()
    full_lm_q4.free()
    h_vec.free()
    local_max_l.free()
    local_next_t.free()

    # 4. Test q4_matmul_batch5 vs 5x q4_matmul_vec on [4096, 2816] (wq/wo shape)
    var b5_in = 2816
    var b5_out = 4096
    var b5_p = alloc[UInt8](b5_out * (b5_in // 2))
    var b5_s = alloc[Float32](b5_out * (b5_in // 32))
    var b5_x = alloc[Float32](5 * b5_in)
    var b5_ref = alloc[Float32](5 * b5_out)
    var b5_got = alloc[Float32](5 * b5_out)
    for i in range(b5_out * (b5_in // 2)):
        b5_p[i] = UInt8((i * 29 + 11) & 255)
    for i in range(b5_out * (b5_in // 32)):
        b5_s[i] = Float32(((i * 19) % 101) + 1) * 0.002
    for i in range(5 * b5_in):
        b5_x[i] = Float32(((i * 37 + 3) % 199) - 99) * 0.01

    for b in range(5):
        q4_matmul_vec(b5_p, b5_s, b5_x + b * b5_in, b5_ref + b * b5_out, b5_in, b5_out, num_tasks)
    q4_matmul_batch5(b5_p, b5_s, b5_x, b5_got, b5_in, b5_out, num_tasks)

    var max_b5_err: Float32 = 0.0
    for i in range(5 * b5_out):
        var d = abs(b5_ref[i] - b5_got[i])
        if d > max_b5_err:
            max_b5_err = d
    print("[Test 4] q4_matmul_batch5 [4096, 2816] max error:", max_b5_err)
    if max_b5_err > 1e-4:
        raise Error("FAILURE: q4_matmul_batch5 error too high: " + String(max_b5_err))

    b5_p.free()
    b5_s.free()
    b5_x.free()
    b5_ref.free()
    b5_got.free()
    print("ALL TESTS PASSED!")
