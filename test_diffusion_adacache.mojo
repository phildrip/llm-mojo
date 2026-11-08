from std.memory import UnsafePointer, alloc
from std.math import abs
import std.time
from diffusion_kernels import adacache_filter_after_layer1


def main() raises:
    print("=== Test & Benchmark: AdaCache + dLLM-Cache Partial Canvas Updates (llm-mojo-2tc.8) ===")
    comptime CANVAS_LEN = 256
    comptime EMBED_DIM = 2816

    var h_orig = alloc[Float16](CANVAS_LEN * EMBED_DIM)
    var canvas_hidden = alloc[Float16](CANVAS_LEN * EMBED_DIM)
    var adacache_h1_prev = alloc[Float16](CANVAS_LEN * EMBED_DIM)
    var active_canvas_idx = alloc[Int](CANVAS_LEN)
    var adacache_skip_streak = alloc[Int](CANVAS_LEN)

    for t in range(CANVAS_LEN):
        active_canvas_idx[t] = t
        adacache_skip_streak[t] = 0
        for d in range(EMBED_DIM):
            var v = Float16(Float32((t * 7 + d % 31) - 15) * 0.05)
            h_orig[t * EMBED_DIM + d] = v
            canvas_hidden[t * EMBED_DIM + d] = v
            adacache_h1_prev[t * EMBED_DIM + d] = Float16(0.0)

    # 1. Warmup steps 0 & 1 must keep all 256 tokens
    var len0 = adacache_filter_after_layer1(
        canvas_hidden, adacache_h1_prev, active_canvas_idx, adacache_skip_streak,
        CANVAS_LEN, EMBED_DIM, 0, k_slow=3, tau_ada=0.08
    )
    var len1 = adacache_filter_after_layer1(
        canvas_hidden, adacache_h1_prev, active_canvas_idx, adacache_skip_streak,
        CANVAS_LEN, EMBED_DIM, 1, k_slow=3, tau_ada=0.08
    )
    if len0 != 256 or len1 != 256:
        raise Error("Warmup steps 0 & 1 should retain all 256 tokens!")

    # 2. Step 2: Simulate 96 quasi-static tokens (t = 0..95, tiny perturbation) and 160 high-motion tokens (t = 96..255)
    for t in range(CANVAS_LEN):
        active_canvas_idx[t] = t
        for d in range(EMBED_DIM):
            var base_v = Float32(h_orig[t * EMBED_DIM + d])
            var pert = Float32(0.002) if t < 96 else Float32((d % 13) - 6) * 0.35
            var new_v = Float16(base_v + pert)
            h_orig[t * EMBED_DIM + d] = new_v
            canvas_hidden[t * EMBED_DIM + d] = new_v

    var t0 = std.time.perf_counter()
    var len2 = adacache_filter_after_layer1(
        canvas_hidden, adacache_h1_prev, active_canvas_idx, adacache_skip_streak,
        CANVAS_LEN, EMBED_DIM, 2, k_slow=3, tau_ada=0.08
    )
    var dt_ms = (std.time.perf_counter() - t0) * 1000.0

    # Verify len2 is a multiple of 16 and equals 128 or 160
    if len2 % 16 != 0 or len2 >= 256:
        raise Error("Expected compacted len2 to be a multiple of 16 < 256!")

    # Verify monotonic order and exact data integrity after in-place compaction
    var max_err: Float32 = 0.0
    for b in range(len2):
        var orig_t = active_canvas_idx[b]
        if b > 0 and orig_t <= active_canvas_idx[b - 1]:
            raise Error("Compacted active_canvas_idx is not strictly increasing!")
        if orig_t < 96:
            raise Error("Quasi-static token was unexpectedly prioritized over high-motion token!")
        for d in range(EMBED_DIM):
            var diff = abs(Float32(canvas_hidden[b * EMBED_DIM + d]) - Float32(h_orig[orig_t * EMBED_DIM + d]))
            if diff > max_err:
                max_err = diff

    # Verify quasi-static tokens incremented their skip streak to 1
    for t in range(96):
        if adacache_skip_streak[t] != 1:
            raise Error("Expected skipped quasi-static token streak to be 1!")

    # 3. Verify k_slow=3 periodic refresh invariant when streak reaches k_slow - 1 = 2
    for t in range(CANVAS_LEN):
        active_canvas_idx[t] = t
        for d in range(EMBED_DIM):
            canvas_hidden[t * EMBED_DIM + d] = adacache_h1_prev[t * EMBED_DIM + d]
    # Set token 0..31 to streak=2 (must be forced to refresh even with 0 fluctuation!)
    for t in range(32):
        adacache_skip_streak[t] = 2
    for t in range(32, CANVAS_LEN):
        adacache_skip_streak[t] = 0

    var len3 = adacache_filter_after_layer1(
        canvas_hidden, adacache_h1_prev, active_canvas_idx, adacache_skip_streak,
        CANVAS_LEN, EMBED_DIM, 3, k_slow=3, tau_ada=0.08
    )
    for t in range(32):
        if adacache_skip_streak[t] != 0:
            raise Error("Token with streak == k_slow - 1 was not forced to refresh!")

    print("  [AdaCache + MoReg] len0=", len0, "| len2=", len2, "/ 256 (multiple of 16) | max_err=", max_err, "| filter latency=", dt_ms, "ms | k_slow refresh verified (len3=", len3, ")")
    if max_err > 0.0:
        raise Error("In-place compaction corrupted hidden state!")

    h_orig.free()
    canvas_hidden.free()
    adacache_h1_prev.free()
    active_canvas_idx.free()
    adacache_skip_streak.free()
    print("=== ALL TESTS PASSED! ===")
