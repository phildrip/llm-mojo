from std.math import exp

# =============================================================================
# Shared paged attention kernel for the Gemma-4 AR engines.
#
# Replaces the three duplicated per-head loops in
# model_speculative_moe_batched.mojo. Differences from that loop:
#   * Optional sliding window (`window` keys including the current token).
#   * KV-group-major: each K/V row is fetched from the cache once per KV head
#     and reused (from L1) by every query head in the group, instead of being
#     re-streamed from DRAM once per query head.
#   * Physical row offsets are resolved once per token, not once per head.
#
# With window <= 0 the result is bit-identical to the old loop: every head's
# dot products, softmax and value accumulation run in the same order with the
# same SIMD expressions.
#
# No 1/sqrt(head_dim) scaling: Gemma-4 uses QK-norm with attention scale 1.0.
# =============================================================================


@always_inline
def attention_window_start(pos: Int, window: Int) -> Int:
    """First key position attended by the query at `pos`.

    `window` counts keys including the current token; `window <= 0` is full causal.
    """
    if window <= 0:
        return 0
    var start = pos - window + 1
    return start if start > 0 else 0


@always_inline
def _softmax(scores: UnsafePointer[Float32, MutUntrackedOrigin], size: Int):
    # Kept identical to the engine's `softmax` so the no-window path stays bit-exact.
    var max_val = scores[0]
    for i in range(1, size):
        if scores[i] > max_val:
            max_val = scores[i]
    var sum_exp: Float32 = 0.0
    for i in range(size):
        scores[i] = exp(scores[i] - max_val)
        sum_exp += scores[i]
    for i in range(size):
        scores[i] /= (sum_exp + 1e-10)


def paged_attention_layer(
    q: UnsafePointer[Float32, _],
    k_pool: UnsafePointer[Float32, _],
    v_pool: UnsafePointer[Float32, _],
    block_table: List[Int],
    block_size: Int,
    pos: Int,
    num_heads: Int,
    num_kv_heads: Int,
    head_dim: Int,
    window: Int,
    scores: UnsafePointer[Float32, MutUntrackedOrigin],
    rows: UnsafePointer[Int, MutUntrackedOrigin],
    attn_out: UnsafePointer[Float32, MutUntrackedOrigin],
):
    """Single-query attention over a paged KV cache.

    Args:
        q: Query heads [num_heads * head_dim], post-norm, post-RoPE.
        k_pool: Key cache rows [physical_row * kv_stride], kv_stride = num_kv_heads * head_dim.
        v_pool: Value cache rows, same layout as k_pool.
        block_table: Logical block -> physical block for this sequence.
        block_size: Tokens per block.
        pos: Position of the current query (its K/V must already be written).
        num_heads: Query heads.
        num_kv_heads: KV heads; must divide num_heads.
        head_dim: Head dimension; multiple of 8.
        window: Keys attended including the current token; <= 0 for full causal.
        scores: Scratch, at least num_heads * (number of attended keys) floats.
        rows: Scratch, at least (number of attended keys) ints.
        attn_out: Output [num_heads * head_dim].
    """
    var kv_stride = num_kv_heads * head_dim
    var ratio = num_heads // num_kv_heads
    var start_t = attention_window_start(pos, window)
    var n = pos + 1 - start_t
    debug_assert(num_kv_heads > 0 and num_heads % num_kv_heads == 0, "num_kv_heads must divide num_heads")
    debug_assert(head_dim % 8 == 0, "head_dim must be a multiple of 8")
    debug_assert(pos >= 0 and pos // block_size < len(block_table), "pos beyond the sequence's block table")

    # Resolve physical cache offsets once per attended token.
    for i in range(n):
        var t = start_t + i
        rows[i] = (block_table[t // block_size] * block_size + t % block_size) * kv_stride

    for g in range(num_kv_heads):
        var g_offset = g * head_dim
        var h_first = g * ratio

        # Scores: one pass over the group's K rows, reused by every head in the group.
        for i in range(n):
            var k_off = rows[i] + g_offset
            for hh in range(ratio):
                var h = h_first + hh
                var h_offset = h * head_dim
                var score_vec = SIMD[DType.float32, 8](0.0)
                for d in range(0, head_dim, 8):
                    score_vec = score_vec + q.load[8](h_offset + d) * k_pool.load[8](k_off + d)
                scores[h * n + i] = score_vec.reduce_add()

        for hh in range(ratio):
            var h = h_first + hh
            _softmax(scores + h * n, n)
            var h_offset = h * head_dim
            for d in range(0, head_dim, 8):
                attn_out.store[8](h_offset + d, SIMD[DType.float32, 8](0.0))

        # Values: one pass over the group's V rows.
        for i in range(n):
            var v_off = rows[i] + g_offset
            for hh in range(ratio):
                var h = h_first + hh
                var h_offset = h * head_dim
                var alpha_vec = SIMD[DType.float32, 8](scores[h * n + i])
                for d in range(0, head_dim, 8):
                    attn_out.store[8](
                        h_offset + d,
                        attn_out.load[8](h_offset + d) + alpha_vec * v_pool.load[8](v_off + d)
                    )
