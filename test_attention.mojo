from std.memory import alloc
from std.math import exp
from std.testing import assert_equal, assert_true
from attention import paged_attention_layer, attention_window_start

# =============================================================================
# Phase 0 tests for the shared paged attention kernel (attention.mojo).
#
# Oracles:
#   1. legacy_paged_attention  - verbatim copy of the loop that existed in
#      model_speculative_moe_batched.mojo (3 copies) before Phase 0. With no
#      window, the new kernel must match it BIT-FOR-BIT.
#   2. reference_attention_f64 - naive Float64 attention with an explicit
#      window, used to check windowed results to a tolerance.
# =============================================================================

comptime BLOCK_SIZE = 16
comptime NUM_BLOCKS = 160          # 2560 tokens of capacity
comptime CAPACITY = NUM_BLOCKS * BLOCK_SIZE
comptime GUARD = 64                # canary floats after every scratch buffer
comptime CANARY: Float32 = -12345.678


# -----------------------------------------------------------------------------
# Deterministic test data
# -----------------------------------------------------------------------------
struct Lcg:
    var state: UInt64

    def __init__(out self, seed: UInt64):
        self.state = seed * 6364136223846793005 + 1442695040888963407

    def next_f32(mut self) -> Float32:
        # Uniform in [-1, 1)
        self.state = self.state * 6364136223846793005 + 1442695040888963407
        var bits = (self.state >> 40) & 0xFFFFFF
        return Float32(bits) / Float32(0x800000) - 1.0


def fill_random(p: UnsafePointer[Float32, MutUntrackedOrigin], n: Int, scale: Float32, mut rng: Lcg):
    for i in range(n):
        p[i] = rng.next_f32() * scale


def shuffled_block_table(num_logical: Int) -> List[Int]:
    # Non-contiguous physical blocks so paging bugs show up. 37 is coprime with NUM_BLOCKS.
    var table = List[Int]()
    for i in range(num_logical):
        table.append((i * 37 + 11) % NUM_BLOCKS)
    return table^


# -----------------------------------------------------------------------------
# Oracle 1: the pre-Phase-0 engine loop, copied verbatim (attn_scores sized
# correctly here so the oracle itself does not overflow).
# -----------------------------------------------------------------------------
@always_inline
def legacy_softmax(scores: UnsafePointer[Float32, MutUntrackedOrigin], size: Int):
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


def legacy_paged_attention(
    q: UnsafePointer[Float32, MutUntrackedOrigin],
    k_pool: UnsafePointer[Float32, MutUntrackedOrigin],
    v_pool: UnsafePointer[Float32, MutUntrackedOrigin],
    block_table: List[Int],
    block_size: Int,
    pos: Int,
    layer_heads: Int,
    layer_kv_heads: Int,
    layer_head_dim: Int,
    attn_scores: UnsafePointer[Float32, MutUntrackedOrigin],
    attn_out: UnsafePointer[Float32, MutUntrackedOrigin],
):
    var kv_stride = layer_kv_heads * layer_head_dim
    var ratio = layer_heads // layer_kv_heads
    for h in range(layer_heads):
        var h_offset = h * layer_head_dim
        var g = h // ratio
        var g_offset = g * layer_head_dim
        for t in range(pos + 1):
            var t_logical_blk = t // block_size
            var t_blk_offset = t % block_size
            var t_phys_blk_id = block_table[t_logical_blk]
            var t_cache_offset = (t_phys_blk_id * block_size + t_blk_offset) * kv_stride + g_offset

            var score_vec = SIMD[DType.float32, 8](0.0)
            for d in range(0, layer_head_dim, 8):
                score_vec = score_vec + q.load[8](h_offset + d) * k_pool.load[8](t_cache_offset + d)
            attn_scores[t] = score_vec.reduce_add()

        legacy_softmax(attn_scores, pos + 1)

        for d in range(0, layer_head_dim, 8):
            attn_out.store[8](h_offset + d, SIMD[DType.float32, 8](0.0))

        for t in range(pos + 1):
            var t_logical_blk = t // block_size
            var t_blk_offset = t % block_size
            var t_phys_blk_id = block_table[t_logical_blk]
            var t_cache_offset = (t_phys_blk_id * block_size + t_blk_offset) * kv_stride + g_offset

            var alpha_vec = SIMD[DType.float32, 8](attn_scores[t])
            for d in range(0, layer_head_dim, 8):
                attn_out.store[8](
                    h_offset + d,
                    attn_out.load[8](h_offset + d) + alpha_vec * v_pool.load[8](t_cache_offset + d)
                )


# -----------------------------------------------------------------------------
# Oracle 2: naive Float64 reference with an explicit inclusive window start.
# -----------------------------------------------------------------------------
def reference_attention_f64(
    q: UnsafePointer[Float32, MutUntrackedOrigin],
    k_pool: UnsafePointer[Float32, MutUntrackedOrigin],
    v_pool: UnsafePointer[Float32, MutUntrackedOrigin],
    block_table: List[Int],
    pos: Int,
    start_t: Int,
    heads: Int,
    kv_heads: Int,
    head_dim: Int,
    out_ref: UnsafePointer[Float64, MutUntrackedOrigin],
):
    var kv_stride = kv_heads * head_dim
    var ratio = heads // kv_heads
    var n = pos + 1 - start_t
    var s = alloc[Float64](n)
    for h in range(heads):
        var g = h // ratio
        var mx: Float64 = -1e300
        for i in range(n):
            var t = start_t + i
            var row = (block_table[t // BLOCK_SIZE] * BLOCK_SIZE + t % BLOCK_SIZE) * kv_stride + g * head_dim
            var acc: Float64 = 0.0
            for d in range(head_dim):
                acc += Float64(q[h * head_dim + d]) * Float64(k_pool[row + d])
            s[i] = acc
            if acc > mx:
                mx = acc
        var tot: Float64 = 0.0
        for i in range(n):
            s[i] = exp(s[i] - mx)
            tot += s[i]
        for d in range(head_dim):
            out_ref[h * head_dim + d] = 0.0
        for i in range(n):
            var t = start_t + i
            var row = (block_table[t // BLOCK_SIZE] * BLOCK_SIZE + t % BLOCK_SIZE) * kv_stride + g * head_dim
            var w = s[i] / tot
            for d in range(head_dim):
                out_ref[h * head_dim + d] += w * Float64(v_pool[row + d])
    s.free()


# -----------------------------------------------------------------------------
# Fixture: pools + scratch with canaries
# -----------------------------------------------------------------------------
struct Fixture:
    var heads: Int
    var kv_heads: Int
    var head_dim: Int
    var kv_stride: Int
    var q: UnsafePointer[Float32, MutUntrackedOrigin]
    var k_pool: UnsafePointer[Float32, MutUntrackedOrigin]
    var v_pool: UnsafePointer[Float32, MutUntrackedOrigin]
    var scores: UnsafePointer[Float32, MutUntrackedOrigin]
    var rows: UnsafePointer[Int, MutUntrackedOrigin]
    var out_new: UnsafePointer[Float32, MutUntrackedOrigin]
    var out_old: UnsafePointer[Float32, MutUntrackedOrigin]
    var out_ref: UnsafePointer[Float64, MutUntrackedOrigin]
    var legacy_scores: UnsafePointer[Float32, MutUntrackedOrigin]
    var table: List[Int]

    def __init__(out self, heads: Int, kv_heads: Int, head_dim: Int, seed: UInt64, k_scale: Float32):
        self.heads = heads
        self.kv_heads = kv_heads
        self.head_dim = head_dim
        self.kv_stride = kv_heads * head_dim
        var rng = Lcg(seed)
        self.q = alloc[Float32](heads * head_dim)
        self.k_pool = alloc[Float32](CAPACITY * self.kv_stride)
        self.v_pool = alloc[Float32](CAPACITY * self.kv_stride)
        fill_random(self.q, heads * head_dim, 1.0, rng)
        fill_random(self.k_pool, CAPACITY * self.kv_stride, k_scale, rng)
        fill_random(self.v_pool, CAPACITY * self.kv_stride, 1.0, rng)
        # Scratch sized exactly to the contract (heads * CAPACITY, CAPACITY) + guard canaries
        self.scores = alloc[Float32](heads * CAPACITY + GUARD)
        self.rows = alloc[Int](CAPACITY + GUARD)
        for i in range(GUARD):
            self.scores[heads * CAPACITY + i] = CANARY
            self.rows[CAPACITY + i] = -777
        self.out_new = alloc[Float32](heads * head_dim + GUARD)
        for i in range(GUARD):
            self.out_new[heads * head_dim + i] = CANARY
        self.out_old = alloc[Float32](heads * head_dim)
        self.out_ref = alloc[Float64](heads * head_dim)
        self.legacy_scores = alloc[Float32](CAPACITY)
        self.table = shuffled_block_table(NUM_BLOCKS)

    def run_new(self, pos: Int, window: Int):
        paged_attention_layer(
            self.q, self.k_pool, self.v_pool, self.table, BLOCK_SIZE, pos,
            self.heads, self.kv_heads, self.head_dim, window,
            self.scores, self.rows, self.out_new,
        )

    def run_legacy(self, pos: Int):
        legacy_paged_attention(
            self.q, self.k_pool, self.v_pool, self.table, BLOCK_SIZE, pos,
            self.heads, self.kv_heads, self.head_dim, self.legacy_scores, self.out_old,
        )

    def run_reference(self, pos: Int, start_t: Int):
        reference_attention_f64(
            self.q, self.k_pool, self.v_pool, self.table, pos, start_t,
            self.heads, self.kv_heads, self.head_dim, self.out_ref,
        )

    def check_canaries(self) raises:
        for i in range(GUARD):
            assert_equal(self.scores[self.heads * CAPACITY + i], CANARY, "scores scratch overflow")
            assert_equal(self.rows[CAPACITY + i], -777, "rows scratch overflow")
            assert_equal(self.out_new[self.heads * self.head_dim + i], CANARY, "attn_out overflow")

    def free(self):
        self.q.free()
        self.k_pool.free()
        self.v_pool.free()
        self.scores.free()
        self.rows.free()
        self.out_new.free()
        self.out_old.free()
        self.out_ref.free()
        self.legacy_scores.free()


def assert_bit_exact(fx: Fixture, label: String) raises:
    var n = fx.heads * fx.head_dim
    var mismatches = 0
    for i in range(n):
        if fx.out_new[i] != fx.out_old[i]:
            mismatches += 1
    if mismatches > 0:
        print("  [FAIL]", label, "mismatches:", mismatches, "of", n)
    assert_equal(mismatches, 0, label + ": new kernel must be bit-exact with legacy loop")


def max_rel_err_vs_ref(fx: Fixture) -> Float64:
    var worst: Float64 = 0.0
    for i in range(fx.heads * fx.head_dim):
        var d = abs(Float64(fx.out_new[i]) - fx.out_ref[i])
        var r = d / (abs(fx.out_ref[i]) + 1e-3)
        if r > worst:
            worst = r
    return worst


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------
def test_window_start() raises:
    print("test_window_start")
    # window <= 0 means full causal
    assert_equal(attention_window_start(0, 0), 0)
    assert_equal(attention_window_start(5000, 0), 0)
    assert_equal(attention_window_start(5000, -1), 0)
    # window of 1024 keys INCLUDING the current token: t >= pos - 1023
    assert_equal(attention_window_start(0, 1024), 0)
    assert_equal(attention_window_start(1022, 1024), 0)
    assert_equal(attention_window_start(1023, 1024), 0)
    assert_equal(attention_window_start(1024, 1024), 1)
    assert_equal(attention_window_start(2047, 1024), 1024)
    assert_equal(attention_window_start(10, 1), 10)


def test_full_bit_exact(heads: Int, kv_heads: Int, head_dim: Int, label: String) raises:
    print("test_full_bit_exact", label)
    var fx = Fixture(heads, kv_heads, head_dim, 42, 0.25)
    var positions = [0, 1, 15, 16, 17, 99, 100, 101, 777, 1023, 1024, CAPACITY - 1]
    for p in positions:
        fx.run_legacy(p)
        fx.run_new(p, 0)
        assert_bit_exact(fx, label + " pos=" + String(p))
        fx.check_canaries()
    fx.free()


def test_window_is_noop_when_short() raises:
    print("test_window_is_noop_when_short")
    var fx = Fixture(16, 8, 256, 7, 0.25)
    var positions = [0, 5, 512, 1022, 1023]
    for p in positions:
        fx.run_legacy(p)
        fx.run_new(p, 1024)
        assert_bit_exact(fx, "short window pos=" + String(p))
        fx.check_canaries()
    fx.free()


def test_window_matches_reference() raises:
    print("test_window_matches_reference")
    var fx = Fixture(16, 8, 256, 99, 0.25)
    var positions = [1024, 1025, 1500, 2047, CAPACITY - 1]
    for p in positions:
        fx.run_new(p, 1024)
        fx.run_reference(p, p - 1023)
        var err = max_rel_err_vs_ref(fx)
        assert_true(err < 1e-4, "windowed attention vs f64 reference, pos=" + String(p) + " err=" + String(err))
        fx.check_canaries()
    fx.free()


def test_window_boundary_is_exact() raises:
    # Adversarial: plant one dominant key exactly at the window edge, and one just outside.
    # The key at t = pos - 1023 must dominate; the key at t = pos - 1024 must have zero effect.
    print("test_window_boundary_is_exact")
    var heads = 16
    var kv_heads = 8
    var head_dim = 256
    var fx = Fixture(heads, kv_heads, head_dim, 3, 0.01)
    var kv_stride = kv_heads * head_dim
    var pos = 2000
    var inside_t = pos - 1023
    var outside_t = pos - 1024

    def row_of(t: Int) {read fx, read kv_stride} -> Int:
        return (fx.table[t // BLOCK_SIZE] * BLOCK_SIZE + t % BLOCK_SIZE) * kv_stride

    # Make the outside key extremely aligned with every query head and give it a huge value.
    for g in range(kv_heads):
        for d in range(head_dim):
            fx.k_pool[row_of(outside_t) + g * head_dim + d] = 0.0
            fx.v_pool[row_of(outside_t) + g * head_dim + d] = 1000.0
    for h in range(heads):
        var g = h // (heads // kv_heads)
        for d in range(head_dim):
            fx.k_pool[row_of(outside_t) + g * head_dim + d] += fx.q[h * head_dim + d] * 4.0

    fx.run_new(pos, 1024)
    for i in range(heads * head_dim):
        assert_true(abs(fx.out_new[i]) < 10.0, "key outside the window leaked into the output")

    # Now copy the same planted key to the inside edge: it must dominate.
    for g in range(kv_heads):
        for d in range(head_dim):
            fx.k_pool[row_of(inside_t) + g * head_dim + d] = fx.k_pool[row_of(outside_t) + g * head_dim + d]
            fx.v_pool[row_of(inside_t) + g * head_dim + d] = 1000.0
    fx.run_new(pos, 1024)
    var dominated = 0
    for i in range(heads * head_dim):
        if fx.out_new[i] > 900.0:
            dominated += 1
    assert_equal(dominated, heads * head_dim, "key at the inside window edge must be attended")
    fx.free()


def test_large_scores_stable() raises:
    # Gemma uses no 1/sqrt(d) scaling, so raw scores can be large. Must not produce NaN/Inf.
    print("test_large_scores_stable")
    var fx = Fixture(16, 2, 512, 11, 4.0)
    var positions = [0, 300, CAPACITY - 1]
    for p in positions:
        fx.run_legacy(p)
        fx.run_new(p, 0)
        assert_bit_exact(fx, "large scores pos=" + String(p))
        for i in range(fx.heads * fx.head_dim):
            var x = fx.out_new[i]
            assert_true(x == x and abs(x) < 1e30, "NaN/Inf in output")
    fx.free()


def run_tight(fx: Fixture, pos: Int, window: Int) raises:
    """Run the kernel with scratch sized EXACTLY to the contract (heads*n, n) plus canaries.

    Catches stride/overflow bugs that stay inside the generously sized fixture buffers.
    Result is copied into fx.out_new.
    """
    var start_t = attention_window_start(pos, window)
    var n = pos + 1 - start_t
    var scores = alloc[Float32](fx.heads * n + GUARD)
    var rows = alloc[Int](n + GUARD)
    for i in range(GUARD):
        scores[fx.heads * n + i] = CANARY
        rows[n + i] = -777
    paged_attention_layer(
        fx.q, fx.k_pool, fx.v_pool, fx.table, BLOCK_SIZE, pos,
        fx.heads, fx.kv_heads, fx.head_dim, window,
        scores, rows, fx.out_new,
    )
    for i in range(GUARD):
        assert_equal(scores[fx.heads * n + i], CANARY, "tight scores scratch overflow at pos=" + String(pos))
        assert_equal(rows[n + i], -777, "tight rows scratch overflow at pos=" + String(pos))
    scores.free()
    rows.free()


def test_tight_scratch_contract() raises:
    print("test_tight_scratch_contract")
    var geoms = [(16, 2, 512), (16, 8, 256), (8, 8, 64)]
    for geom in geoms:
        var fx = Fixture(geom[0], geom[1], geom[2], 5, 0.25)
        var positions = [0, 1, 1023, 1024, 1500, CAPACITY - 1]
        for p in positions:
            for w in [0, 1, 7, 1024]:
                run_tight(fx, p, w)
        fx.free()


def test_ratio_one_and_seeds() raises:
    # ratio = 1 (num_kv_heads == num_heads) exercises a single-member head group; extra seeds.
    print("test_ratio_one_and_seeds")
    for seed in [1, 2]:
        var fx = Fixture(8, 8, 64, UInt64(seed), 0.5)
        var positions = [0, 33, CAPACITY - 1]
        for p in positions:
            fx.run_legacy(p)
            fx.run_new(p, 0)
            assert_bit_exact(fx, "ratio1 seed=" + String(seed) + " pos=" + String(p))
            fx.check_canaries()
        fx.free()
    var fx2 = Fixture(16, 2, 512, 1234, 0.25)
    for p in [5, 640, CAPACITY - 1]:
        fx2.run_legacy(p)
        fx2.run_new(p, 0)
        assert_bit_exact(fx2, "global seed=1234 pos=" + String(p))
    fx2.free()


def test_window_block_aligned_edge() raises:
    # Window start exactly on a block boundary (start % 16 == 0), and the current token itself
    # must be attended (planting the dominant key at t = pos).
    print("test_window_block_aligned_edge")
    var heads = 16
    var kv_heads = 8
    var head_dim = 256
    var kv_stride = kv_heads * head_dim
    var fx = Fixture(heads, kv_heads, head_dim, 21, 0.01)
    var pos = 1024 + 16 * 3 + 1023          # start_t = 1072, a multiple of 16
    assert_equal(attention_window_start(pos, 1024) % BLOCK_SIZE, 0)
    assert_true(pos < CAPACITY)

    def row_of(t: Int) {read fx, read kv_stride} -> Int:
        return (fx.table[t // BLOCK_SIZE] * BLOCK_SIZE + t % BLOCK_SIZE) * kv_stride

    def plant(t: Int) {read fx, read kv_stride, read heads, read kv_heads, read head_dim, read row_of}:
        for g in range(kv_heads):
            for d in range(head_dim):
                fx.k_pool[row_of(t) + g * head_dim + d] = 0.0
                fx.v_pool[row_of(t) + g * head_dim + d] = 1000.0
        for h in range(heads):
            var g = h // (heads // kv_heads)
            for d in range(head_dim):
                fx.k_pool[row_of(t) + g * head_dim + d] += fx.q[h * head_dim + d] * 4.0

    var start_t = attention_window_start(pos, 1024)
    # Just outside, on the previous block's last slot: no effect.
    plant(start_t - 1)
    run_tight(fx, pos, 1024)
    for i in range(heads * head_dim):
        assert_true(abs(fx.out_new[i]) < 10.0, "key at start_t-1 (block edge) leaked")
    # Inside edge, first slot of a block: dominates.
    plant(start_t)
    run_tight(fx, pos, 1024)
    for i in range(heads * head_dim):
        assert_true(fx.out_new[i] > 900.0, "key at block-aligned start_t not attended")
    fx.free()

    # Current token (t = pos) is attended.
    var fx3 = Fixture(heads, kv_heads, head_dim, 22, 0.01)
    var pos3 = 1777

    def row3(t: Int) {read fx3, read kv_stride} -> Int:
        return (fx3.table[t // BLOCK_SIZE] * BLOCK_SIZE + t % BLOCK_SIZE) * kv_stride

    for g in range(kv_heads):
        for d in range(head_dim):
            fx3.k_pool[row3(pos3) + g * head_dim + d] = 0.0
            fx3.v_pool[row3(pos3) + g * head_dim + d] = 1000.0
    for h in range(heads):
        var g = h // (heads // kv_heads)
        for d in range(head_dim):
            fx3.k_pool[row3(pos3) + g * head_dim + d] += fx3.q[h * head_dim + d] * 4.0
    run_tight(fx3, pos3, 1024)
    for i in range(heads * head_dim):
        assert_true(fx3.out_new[i] > 900.0, "current token (t = pos) not attended")
    fx3.free()


def main() raises:
    print("=========================================================")
    print("        PHASE 0: SHARED PAGED ATTENTION KERNEL TESTS     ")
    print("=========================================================")
    test_window_start()
    test_full_bit_exact(16, 2, 512, "global(16q/2kv/512)")
    test_full_bit_exact(16, 8, 256, "sliding(16q/8kv/256)")
    test_window_is_noop_when_short()
    test_window_matches_reference()
    test_window_boundary_is_exact()
    test_large_scores_stable()
    test_tight_scratch_contract()
    test_ratio_one_and_seeds()
    test_window_block_aligned_edge()
    print("ALL ATTENTION TESTS PASSED")
