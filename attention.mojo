from std.math import exp, cos, sin
from std.collections import InlineArray
from std.algorithm import parallelize
from std.os import abort

# =============================================================================
# Shared paged attention & trig-hoisted SIMD RoPE kernels for Gemma-4 AR engines.
#
#   * Trig-Hoisted SIMD RoPE (`apply_rope` / `apply_rope_qk`): computes cos/sin
#     once per layer instead of 24 redundant times across Q and K heads, skips
#     the 384 position-free dimensions on global layers, and applies rotations
#     with SIMD[Float32, 8].
#   * KV-group-major + 8-Core GQA Parallel Attention (`paged_attention_layer`):
#     loads each K/V row once per query-head pair and parallelizes across all
#     query-head pairs (`num_heads // 2`) when context length `n >= PARALLEL_ATTN_MIN_N`
#     while preserving 0.0 bit-exactness.
#     NOTE: Callers must invoke `paged_attention_layer` from a serial context
#     (never nested inside another `parallelize` worker).
# =============================================================================

comptime PARALLEL_ATTN_MIN_N: Int = 64


@always_inline
def apply_rope(
    vec: UnsafePointer[Float32, MutUntrackedOrigin],
    num_heads: Int,
    head_dim: Int,
    pos: Int,
    is_global: Bool,
):
    var half_dim = head_dim // 2
    var rotary_half = min(64, half_dim) if is_global else half_dim
    debug_assert(rotary_half <= 256, "rotary_half exceeds 256-float stack trig table")
    var base: Float32 = Float32(1000000.0) if is_global else Float32(10000.0)
    var rope_dim = Float32(512 if is_global else 256)
    var pos_f = Float32(pos)

    var cos_arr = InlineArray[Float32, 256](uninitialized=True)
    var sin_arr = InlineArray[Float32, 256](uninitialized=True)
    var cos_ptr = cos_arr.unsafe_ptr()
    var sin_ptr = sin_arr.unsafe_ptr()

    for i in range(rotary_half):
        var theta = pos_f / (base ** (Float32(2 * i) / rope_dim))
        cos_ptr[i] = cos(theta)
        sin_ptr[i] = sin(theta)

    var vec_limit = (rotary_half // 8) * 8
    for h in range(num_heads):
        var h_off = h * head_dim
        var h_off2 = h_off + half_dim
        for i in range(0, vec_limit, 8):
            var c = cos_ptr.load[8](i)
            var s = sin_ptr.load[8](i)
            var v1 = vec.load[8](h_off + i)
            var v2 = vec.load[8](h_off2 + i)
            vec.store[8](h_off + i, v1 * c - v2 * s)
            vec.store[8](h_off2 + i, v2 * c + v1 * s)
        for i in range(vec_limit, rotary_half):
            var c = cos_ptr[i]
            var s = sin_ptr[i]
            var v1 = vec[h_off + i]
            var v2 = vec[h_off2 + i]
            vec[h_off + i] = v1 * c - v2 * s
            vec[h_off2 + i] = v2 * c + v1 * s


@always_inline
def apply_rope_qk(
    q: UnsafePointer[Float32, MutUntrackedOrigin],
    k: UnsafePointer[Float32, MutUntrackedOrigin],
    num_q_heads: Int,
    num_kv_heads: Int,
    head_dim: Int,
    pos: Int,
    is_global: Bool,
):
    """Computes rotary sin/cos once and rotates both Q and K heads with SIMD[Float32, 8]."""
    var half_dim = head_dim // 2
    var rotary_half = min(64, half_dim) if is_global else half_dim
    debug_assert(rotary_half <= 256, "rotary_half exceeds 256-float stack trig table")
    var base: Float32 = Float32(1000000.0) if is_global else Float32(10000.0)
    var rope_dim = Float32(512 if is_global else 256)
    var pos_f = Float32(pos)

    var cos_arr = InlineArray[Float32, 256](uninitialized=True)
    var sin_arr = InlineArray[Float32, 256](uninitialized=True)
    var cos_ptr = cos_arr.unsafe_ptr()
    var sin_ptr = sin_arr.unsafe_ptr()

    for i in range(rotary_half):
        var theta = pos_f / (base ** (Float32(2 * i) / rope_dim))
        cos_ptr[i] = cos(theta)
        sin_ptr[i] = sin(theta)

    var vec_limit = (rotary_half // 8) * 8
    for h in range(num_q_heads):
        var h_off = h * head_dim
        var h_off2 = h_off + half_dim
        for i in range(0, vec_limit, 8):
            var c = cos_ptr.load[8](i)
            var s = sin_ptr.load[8](i)
            var v1 = q.load[8](h_off + i)
            var v2 = q.load[8](h_off2 + i)
            q.store[8](h_off + i, v1 * c - v2 * s)
            q.store[8](h_off2 + i, v2 * c + v1 * s)
        for i in range(vec_limit, rotary_half):
            var c = cos_ptr[i]
            var s = sin_ptr[i]
            var v1 = q[h_off + i]
            var v2 = q[h_off2 + i]
            q[h_off + i] = v1 * c - v2 * s
            q[h_off2 + i] = v2 * c + v1 * s

    for h in range(num_kv_heads):
        var h_off = h * head_dim
        var h_off2 = h_off + half_dim
        for i in range(0, vec_limit, 8):
            var c = cos_ptr.load[8](i)
            var s = sin_ptr.load[8](i)
            var v1 = k.load[8](h_off + i)
            var v2 = k.load[8](h_off2 + i)
            k.store[8](h_off + i, v1 * c - v2 * s)
            k.store[8](h_off2 + i, v2 * c + v1 * s)
        for i in range(vec_limit, rotary_half):
            var c = cos_ptr[i]
            var s = sin_ptr[i]
            var v1 = k[h_off + i]
            var v2 = k[h_off2 + i]
            k[h_off + i] = v1 * c - v2 * s
            k[h_off2 + i] = v2 * c + v1 * s


@always_inline
def attention_window_start(pos: Int, window: Int) -> Int:
    """First key position attended by the query at `pos`."""
    if window <= 0:
        return 0
    var start = pos - window + 1
    return start if start > 0 else 0


@always_inline
def ring_row(slot: Int, ring_size: Int, t: Int) -> Int:
    """Physical row of token `t` in sequence slot `slot` of a ring-buffer KV pool."""
    debug_assert(slot >= 0 and ring_size > 0 and t >= 0, "invalid ring address")
    return slot * ring_size + t % ring_size


@always_inline
def _softmax(scores: UnsafePointer[Float32, MutUntrackedOrigin], size: Int):
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


@always_inline
def _eval_head_range(
    q: UnsafePointer[Float32, _],
    k_pool: UnsafePointer[Float32, _],
    v_pool: UnsafePointer[Float32, _],
    rows: UnsafePointer[Int, _],
    scores: UnsafePointer[Float32, MutUntrackedOrigin],
    attn_out: UnsafePointer[Float32, MutUntrackedOrigin],
    n: Int,
    g_offset: Int,
    h_start: Int,
    h_end: Int,
    head_dim: Int,
):
    """Evaluates attention for query heads [h_start, h_end) sharing KV group offset g_offset."""
    if n <= 0:
        for h in range(h_start, h_end):
            var h_off = h * head_dim
            for d in range(0, head_dim, 8):
                attn_out.store[8](h_off + d, SIMD[DType.float32, 8](0.0))
        return

    # 1. Scores: pair-wise query heads share each K row load from L1
    for i in range(n):
        var k_off = rows[i] + g_offset
        var h = h_start
        while h + 1 < h_end:
            var h0_off = h * head_dim
            var h1_off = (h + 1) * head_dim
            var s0 = SIMD[DType.float32, 8](0.0)
            var s1 = SIMD[DType.float32, 8](0.0)
            for d in range(0, head_dim, 8):
                var kv = k_pool.load[8](k_off + d)
                s0 = s0 + q.load[8](h0_off + d) * kv
                s1 = s1 + q.load[8](h1_off + d) * kv
            scores[h * n + i] = s0.reduce_add()
            scores[(h + 1) * n + i] = s1.reduce_add()
            h += 2
        while h < h_end:
            var h_off = h * head_dim
            var s0 = SIMD[DType.float32, 8](0.0)
            for d in range(0, head_dim, 8):
                s0 = s0 + q.load[8](h_off + d) * k_pool.load[8](k_off + d)
            scores[h * n + i] = s0.reduce_add()
            h += 1

    # 2. Softmax and zero-init output per head
    for h in range(h_start, h_end):
        _softmax(scores + h * n, n)
        var h_off = h * head_dim
        for d in range(0, head_dim, 8):
            attn_out.store[8](h_off + d, SIMD[DType.float32, 8](0.0))

    # 3. Values: pair-wise query heads share each V row load from L1
    for i in range(n):
        var v_off = rows[i] + g_offset
        var h = h_start
        while h + 1 < h_end:
            var h0_off = h * head_dim
            var h1_off = (h + 1) * head_dim
            var a0 = SIMD[DType.float32, 8](scores[h * n + i])
            var a1 = SIMD[DType.float32, 8](scores[(h + 1) * n + i])
            for d in range(0, head_dim, 8):
                var vv = v_pool.load[8](v_off + d)
                attn_out.store[8](h0_off + d, attn_out.load[8](h0_off + d) + a0 * vv)
                attn_out.store[8](h1_off + d, attn_out.load[8](h1_off + d) + a1 * vv)
            h += 2
        while h < h_end:
            var h_off = h * head_dim
            var a0 = SIMD[DType.float32, 8](scores[h * n + i])
            for d in range(0, head_dim, 8):
                attn_out.store[8](h_off + d, attn_out.load[8](h_off + d) + a0 * v_pool.load[8](v_off + d))
            h += 1


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
    ring_size: Int = 0,
    ring_base: Int = 0,
):
    var kv_stride = num_kv_heads * head_dim
    var ratio = num_heads // num_kv_heads
    var start_t = attention_window_start(pos, window)
    var n = pos + 1 - start_t
    debug_assert(num_kv_heads > 0 and num_heads % num_kv_heads == 0, "num_kv_heads must divide num_heads")
    debug_assert(head_dim % 8 == 0, "head_dim must be a multiple of 8")

    if ring_size > 0:
        if n > ring_size:
            abort("paged_attention_layer: ring_size smaller than the attended window")
        for i in range(n):
            var t = start_t + i
            rows[i] = (ring_base + t % ring_size) * kv_stride
    else:
        debug_assert(pos >= 0 and pos // block_size < len(block_table), "pos beyond the sequence's block table")
        for i in range(n):
            var t = start_t + i
            rows[i] = (block_table[t // block_size] * block_size + t % block_size) * kv_stride

    if n >= PARALLEL_ATTN_MIN_N and num_heads >= 4 and (num_heads % 2 == 0) and (ratio % 2 == 0):
        # Parallel GQA execution: each task t in 0..(num_heads // 2) evaluates query heads [2*t, 2*t+2)
        # Note: since ratio % 2 == 0, both heads 2*t and 2*t+1 belong to the same KV group g = (2*t) // ratio.
        var num_pairs = num_heads // 2
        def gqa_pair_worker(t: Int) capturing -> None:
            var h_start = t * 2
            var g = h_start // ratio
            _eval_head_range(
                q, k_pool, v_pool, rows, scores, attn_out,
                n, g * head_dim, h_start, h_start + 2, head_dim,
            )
        parallelize[gqa_pair_worker](num_pairs)
    else:
        for g in range(num_kv_heads):
            var h_first = g * ratio
            _eval_head_range(
                q, k_pool, v_pool, rows, scores, attn_out,
                n, g * head_dim, h_first, h_first + ratio, head_dim,
            )
