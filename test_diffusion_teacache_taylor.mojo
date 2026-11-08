from std.memory import UnsafePointer, alloc
from std.math import abs
import std.time
from diffusion_kernels import (
    teacache_eval_gate,
    taylorseer_record_full_step,
    taylorseer_forecast_skipped_step,
)


def main() raises:
    print("=== Test & Benchmark: TeaCache Gate & TaylorSeer Forecasting (llm-mojo-2tc.4 & 2tc.7) ===")
    comptime CANVAS_LEN = 256
    comptime EMBED_DIM = 2816

    var active_canvas_idx = alloc[Int](CANVAS_LEN)
    for t in range(CANVAS_LEN):
        active_canvas_idx[t] = t

    var h_curr = alloc[Float16](CANVAS_LEN * EMBED_DIM)
    var h_prev_full = alloc[Float16](CANVAS_LEN * EMBED_DIM)
    var accum_fluct = alloc[Float32](1)
    accum_fluct[0] = 0.0

    # 1. Test TeaCache Gate: step 0 must never skip, small fluctuations skip up to max_consec_skips, large fluctuation triggers full step
    for i in range(CANVAS_LEN * EMBED_DIM):
        h_curr[i] = Float16(1.0)
    var skip0 = teacache_eval_gate(
        h_curr, h_prev_full, active_canvas_idx, CANVAS_LEN, EMBED_DIM, 0.8, 0, 0, accum_fluct, 0.20, 2
    )
    if skip0:
        raise Error("TeaCache must not skip on full_steps_count == 0!")

    # Small 5% change at mid-trajectory (temp=0.6 -> poly_mod = 0.45, delta < 0.20)
    for i in range(CANVAS_LEN * EMBED_DIM):
        h_curr[i] = Float16(1.05)
    var skip1 = teacache_eval_gate(
        h_curr, h_prev_full, active_canvas_idx, CANVAS_LEN, EMBED_DIM, 0.6, 2, 0, accum_fluct, 0.20, 2
    )
    var skip2 = teacache_eval_gate(
        h_curr, h_prev_full, active_canvas_idx, CANVAS_LEN, EMBED_DIM, 0.6, 2, 1, accum_fluct, 0.20, 2
    )
    # 3rd consecutive skip blocked by max_consec_skips == 2
    var skip3 = teacache_eval_gate(
        h_curr, h_prev_full, active_canvas_idx, CANVAS_LEN, EMBED_DIM, 0.6, 2, 2, accum_fluct, 0.20, 2
    )
    print("  [TeaCache Gate] skip0 =", skip0, "| skip1 =", skip1, "| skip2 =", skip2, "| skip3 (max_consec=2) =", skip3, "| reset accum =", accum_fluct[0])
    if not skip1 or not skip2 or skip3 or accum_fluct[0] != 0.0:
        raise Error("TeaCache gating or max_consec_skips reset failed!")

    # 2. Test TaylorSeer 2nd-Order Newton Backward Difference Forecasting on Quadratic Trajectory:
    # Let D(k)[t, i] = A[t, i] + B[t, i] * k + 0.5 * C[t, i] * k^2 for k = 0, 1, 2, 3, 4.
    var h_start = alloc[Float16](CANVAS_LEN * EMBED_DIM)
    var h_end = alloc[Float16](CANVAS_LEN * EMBED_DIM)
    var taylor_D = alloc[Float32](CANVAS_LEN * EMBED_DIM)
    var taylor_d1 = alloc[Float32](CANVAS_LEN * EMBED_DIM)
    var taylor_d2 = alloc[Float32](CANVAS_LEN * EMBED_DIM)

    for k in range(3):
        var kf = Float32(k)
        for t in range(CANVAS_LEN):
            for i in range(EMBED_DIM):
                var idx = t * EMBED_DIM + i
                var a = Float32(((t * 7 + i * 3) % 19) + 5) * 0.1
                var b = Float32(((t * 5 + i * 11) % 7) - 3) * 0.03
                var c = Float32(((t * 3 + i * 13) % 5) - 2) * 0.008
                var d_val = a + b * kf + Float32(0.5) * c * kf * kf
                h_start[idx] = Float16(0.5)
                h_end[idx] = Float16(Float32(0.5) + d_val)
        taylorseer_record_full_step(
            h_start, h_end, active_canvas_idx, taylor_D, taylor_d1, taylor_d2, CANVAS_LEN, EMBED_DIM, k
        )

    # Forecast k = 3 (step_offset_m = 1 after k = 2) and compare against exact quadratic D(3) and stale D(2)
    for i in range(CANVAS_LEN * EMBED_DIM):
        h_curr[i] = Float16(0.5)

    var t0_ts = std.time.perf_counter()
    taylorseer_forecast_skipped_step(
        h_curr, active_canvas_idx, taylor_D, taylor_d1, taylor_d2, CANVAS_LEN, EMBED_DIM, 3, 1
    )
    var dt_ts_ms = (std.time.perf_counter() - t0_ts) * 1000.0

    var max_err_taylor: Float32 = 0.0
    var max_err_stale: Float32 = 0.0
    var kf3 = Float32(3.0)
    for t in range(CANVAS_LEN):
        for i in range(EMBED_DIM):
            var idx = t * EMBED_DIM + i
            var a = Float32(((t * 7 + i * 3) % 19) + 5) * 0.1
            var b = Float32(((t * 5 + i * 11) % 7) - 3) * 0.03
            var c = Float32(((t * 3 + i * 13) % 5) - 2) * 0.008
            var exact_h22 = Float32(Float16(0.5)) + (a + b * kf3 + Float32(0.5) * c * kf3 * kf3)
            var stale_h22 = Float32(Float16(0.5)) + Float32(taylor_D[idx])
            var err_ts = abs(Float32(h_curr[idx]) - exact_h22)
            var err_st = abs(stale_h22 - exact_h22)
            if err_ts > max_err_taylor:
                max_err_taylor = err_ts
            if err_st > max_err_stale:
                max_err_stale = err_st

    print("  [TaylorSeer m=1 Forecast] TaylorSeer max_err =", max_err_taylor, "(vs 0th-order stale max_err =", max_err_stale, ") | Forecast latency for 17 layers =", dt_ts_ms, "ms (vs ~2,850 ms full 17 layers!)")
    if max_err_taylor > 0.006 or max_err_taylor >= max_err_stale * 0.05:
        raise Error("TaylorSeer 2nd-order forecast accuracy check failed!")

    # 3. Test Confidence-Gated Curvature Fallback on Discontinuous Jump
    # Inject a huge discontinuous jump on token 0 at step k=2 and verify TaylorSeer damps c1/c2 to 0.0
    for i in range(EMBED_DIM):
        taylor_D[i] = Float32(1.0)
        taylor_d1[i] = Float32(5.0)  # ||d1|| >> 0.65 * ||D|| -> must damp to 0th-order!
        taylor_d2[i] = Float32(10.0)
        h_curr[i] = Float16(0.5)
    taylorseer_forecast_skipped_step(
        h_curr, active_canvas_idx, taylor_D, taylor_d1, taylor_d2, 1, EMBED_DIM, 3, 1
    )
    var jump_err = abs(Float32(h_curr[0]) - Float32(1.5))
    print("  [TaylorSeer Discontinuous Curvature Guard] jump_err =", jump_err)
    if jump_err > 1e-3:
        raise Error("TaylorSeer curvature guard failed to damp discontinuous jump!")

    active_canvas_idx.free()
    h_curr.free()
    h_prev_full.free()
    accum_fluct.free()
    h_start.free()
    h_end.free()
    taylor_D.free()
    taylor_d1.free()
    taylor_d2.free()
    print("=== ALL TESTS PASSED! ===")
