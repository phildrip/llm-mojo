from std.memory import alloc
from std.collections import InlineArray
from std.time import perf_counter_ns
from std.math import abs
from moe_kernels import (
    q4_matmul_vec,
    q4_matmul_batchN,
    compute_adaptive_draft_steps,
    quantize_fp32_to_q4_0,
)
from q4_embed import Q4Embedding


def main() raises:
    print("=== TDD Test: Adaptive Draft Horizon & Variable-Batch Kernels (llm-mojo-xr6.5) ===")

    # 1. Test compute_adaptive_draft_steps
    if compute_adaptive_draft_steps(1, 4) != 0:
        raise Error("FAIL: rem_tokens=1 should require 0 draft steps")
    if compute_adaptive_draft_steps(2, 4) != 1:
        raise Error("FAIL: rem_tokens=2 should require 1 draft step")
    if compute_adaptive_draft_steps(3, 4) != 2:
        raise Error("FAIL: rem_tokens=3 should require 2 draft steps")
    if compute_adaptive_draft_steps(4, 4) != 3:
        raise Error("FAIL: rem_tokens=4 should require 3 draft steps")
    if compute_adaptive_draft_steps(5, 4) != 4:
        raise Error("FAIL: rem_tokens=5 should require 4 draft steps")
    if compute_adaptive_draft_steps(100, 4) != 4:
        raise Error("FAIL: rem_tokens=100 should require 4 draft steps")
    print("  [PASS] compute_adaptive_draft_steps verified for rem_tokens = 1..100")

    # 2. Test q4_matmul_batchN for B = 1, 2, 3, 4, 5
    var c_in = 2816
    var c_out = 1024
    var c_in_packed = c_in // 2
    var c_in_blocks = c_in // 32

    var W_packed = alloc[UInt8](c_out * c_in_packed)
    var scales = alloc[Float32](c_out * c_in_blocks)
    var x_batch = alloc[Float32](5 * c_in)
    var out_ref = alloc[Float32](5 * c_out)
    var out_bn = alloc[Float32](5 * c_out)

    for i in range(c_out * c_in_packed):
        W_packed[i] = UInt8((i * 73 + 19) & 255)
    for i in range(c_out * c_in_blocks):
        scales[i] = Float32(0.01) + Float32(i % 13) * Float32(0.002)
    for i in range(5 * c_in):
        x_batch[i] = Float32((i % 29) - 14) * Float32(0.11)

    for b in range(5):
        q4_matmul_vec(W_packed, scales, x_batch + b * c_in, out_ref + b * c_out, c_in, c_out, 8)

    for batch_size in range(1, 6):
        for i in range(batch_size * c_out):
            out_bn[i] = -999.0
        q4_matmul_batchN(batch_size, W_packed, scales, x_batch, out_bn, c_in, c_out, 8)
        var max_diff: Float32 = 0.0
        for i in range(batch_size * c_out):
            var d = abs(out_ref[i] - out_bn[i])
            if d > max_diff:
                max_diff = d
        if max_diff > 1e-4:
            raise Error("FAIL: q4_matmul_batchN mismatch for batch_size=" + String(batch_size))
        print("  [PASS] q4_matmul_batchN(batch_size=" + String(batch_size) + ") max_diff =", max_diff)

    # 3. Test Q4Embedding.project_logits_batchN_parallel for B = 1, 2, 3, 4, 5
    var vocab_size = 4096
    var embed_dim = 2816
    var embed = Q4Embedding(vocab_size, embed_dim)
    var raw_w = alloc[Float32](vocab_size * embed_dim)
    for i in range(vocab_size * embed_dim):
        raw_w[i] = Float32(((i * 37 + 11) % 97) - 48) * Float32(0.02)
    quantize_fp32_to_q4_0(raw_w, embed.packed_weights, embed.scales, vocab_size * embed_dim, 8)
    embed.finalize_tail_bounds()
    raw_w.free()

    var ref_toks = InlineArray[Int, 5](uninitialized=True)
    var lmax = alloc[Float32](8)
    var ltok = alloc[Int](8)
    for b in range(5):
        embed.project_logits_parallel(x_batch + b * embed_dim, lmax, ltok, 8, softcap=True)
        ref_toks[b] = embed.last_winner_tok[0]

    var seeds = InlineArray[Int, 5](uninitialized=True)
    var out_toks = InlineArray[Int, 5](uninitialized=True)
    var seeds_ptr = UnsafePointer[Int, MutUntrackedOrigin](unsafe_from_address=Int(UnsafePointer(to=seeds)))
    var out_toks_ptr = UnsafePointer[Int, MutUntrackedOrigin](unsafe_from_address=Int(UnsafePointer(to=out_toks)))

    for batch_size in range(1, 6):
        for b in range(batch_size):
            seeds_ptr[b] = ref_toks[b]
            out_toks_ptr[b] = -1
        embed.project_logits_batchN_parallel(batch_size, x_batch, seeds_ptr, out_toks_ptr, 8, softcap=True)
        for b in range(batch_size):
            if out_toks_ptr[b] != ref_toks[b]:
                raise Error(
                    "FAIL: project_logits_batchN_parallel mismatch at batch_size="
                    + String(batch_size)
                    + " b="
                    + String(b)
                )
        print("  [PASS] project_logits_batchN_parallel(batch_size=" + String(batch_size) + ") 100% exact match!")

    lmax.free()
    ltok.free()
    embed.free()
    W_packed.free()
    scales.free()
    x_batch.free()
    out_ref.free()
    out_bn.free()
    print("=== PASS ===")
