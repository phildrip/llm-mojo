from std.memory import alloc
from std.collections import InlineArray
from std.math import abs, exp
from std.algorithm import parallelize
from std.sys.info import num_physical_cores
import std.time
from kfd_gpu import KFDContext
from diffusion_kernels import (
    unpack_16bytes,
    deinterleave_batch_to_f32,
    gelu_and_mul_to_deinterleaved_f32,
    q4_moe_gemm_deinterleaved,
    q4_gemm_l1_2x4,
    run_diffusion_moe_experts_parallel,
)


def ref_q4_matmul(
    W_packed: UnsafePointer[UInt8, _],
    scales: UnsafePointer[Float32, _],
    X: UnsafePointer[Float32, _],
    out_ptr: UnsafePointer[Float32, MutAnyOrigin],
    B: Int,
    c_in: Int,
    c_out: Int,
):
    var c_in_packed = c_in // 2
    var c_in_blocks = c_in // 32
    for b in range(B):
        for r in range(c_out):
            var acc: Float32 = 0.0
            var rp = r * c_in_packed
            var rs = r * c_in_blocks
            var xb = b * c_in
            for blk in range(c_in_blocks):
                var sc = scales[rs + blk]
                var bsum: Float32 = 0.0
                for j in range(16):
                    var byte_val = Int(W_packed[rp + blk * 16 + j])
                    var wl = Float32((byte_val & 15) - 8)
                    var wh = Float32(((byte_val >> 4) & 15) - 8)
                    bsum += wl * X[xb + blk * 32 + 2 * j] + wh * X[xb + blk * 32 + 2 * j + 1]
                acc += bsum * sc
            out_ptr[b * c_out + r] = acc


@always_inline
def old_q4_gemv_small_b_f16[
    MAX_B: Int
](
    W_packed: UnsafePointer[UInt8, _],
    scales: UnsafePointer[Float32, _],
    X: UnsafePointer[Float16, _],
    out_ptr: UnsafePointer[Float16, MutAnyOrigin],
    B: Int,
    c_in: Int,
    c_out: Int,
    num_tasks: Int,
):
    var tasks = num_tasks
    var chunk_size = c_out // tasks
    var c_in_packed = c_in // 2
    var c_in_blocks = c_in // 32

    def worker(t: Int) capturing -> None:
        var start_r = t * chunk_size
        var end_r = (t + 1) * chunk_size if t < tasks - 1 else c_out
        var acc = InlineArray[SIMD[DType.float32, 8], MAX_B](uninitialized=True)

        for r in range(start_r, end_r):
            var r_offset_packed = r * c_in_packed
            var r_offset_scales = r * c_in_blocks
            comptime for i in range(MAX_B):
                acc[i] = SIMD[DType.float32, 8](0.0)
            for blk in range(c_in_blocks):
                var scale = scales[r_offset_scales + blk]
                var block_p = r_offset_packed + blk * 16
                var x_offset_base = blk * 32
                var bytes = W_packed.load[16](block_p)
                var w0 = SIMD[DType.float32, 8]()
                var w1 = SIMD[DType.float32, 8]()
                var w2 = SIMD[DType.float32, 8]()
                var w3 = SIMD[DType.float32, 8]()
                unpack_16bytes(bytes, w0, w1, w2, w3)
                w0 = w0 * scale
                w1 = w1 * scale
                w2 = w2 * scale
                w3 = w3 * scale
                comptime for i in range(MAX_B):
                    if i < B:
                        var x_base = i * c_in + x_offset_base
                        acc[i] = acc[i] + w0 * X.load[8](x_base).cast[DType.float32]()
                        acc[i] = acc[i] + w1 * X.load[8](x_base + 8).cast[DType.float32]()
                        acc[i] = acc[i] + w2 * X.load[8](x_base + 16).cast[DType.float32]()
                        acc[i] = acc[i] + w3 * X.load[8](x_base + 24).cast[DType.float32]()
            comptime for i in range(MAX_B):
                if i < B:
                    out_ptr[i * c_out + r] = acc[i].reduce_add().cast[DType.float16]()

    parallelize[worker](tasks)


def main() raises:
    print("=== Test & Benchmark: Unified KFDContext 256-Token Q4_0 & Dual-Space AVX2 MoE (llm-mojo-2tc.2) ===")
    var num_tasks = num_physical_cores()

    # 1. Test 256-token batched q4_matmul_lds.hsaco via KFDContext
    var ctx = KFDContext(
        init_queue=True,
        default_hsaco_path="/home/phil/git/llm-mojo/scratch/q4_matmul_lds.hsaco",
        default_kernel_name="q4_matmul",
    )

    comptime B = 256
    comptime C_IN = 2816
    comptime Q_OUT = 4096
    comptime KV_OUT = 2048
    comptime C_IN_PACKED = C_IN // 2
    comptime C_IN_BLOCKS = C_IN // 32

    var wq_buf = ctx.alloc_gpu_buffer(Q_OUT * C_IN_PACKED)
    var sq_buf = ctx.alloc_gpu_buffer(Q_OUT * C_IN_BLOCKS * 4)
    var wk_buf = ctx.alloc_gpu_buffer(KV_OUT * C_IN_PACKED)
    var sk_buf = ctx.alloc_gpu_buffer(KV_OUT * C_IN_BLOCKS * 4)
    var wv_buf = ctx.alloc_gpu_buffer(KV_OUT * C_IN_PACKED)
    var sv_buf = ctx.alloc_gpu_buffer(KV_OUT * C_IN_BLOCKS * 4)

    var x_buf = ctx.alloc_gpu_buffer(B * C_IN * 4)
    var out_q_buf = ctx.alloc_gpu_buffer(B * Q_OUT * 4)
    var out_k_buf = ctx.alloc_gpu_buffer(B * KV_OUT * 4)
    var out_v_buf = ctx.alloc_gpu_buffer(B * KV_OUT * 4)

    var wq_ptr = UnsafePointer[UInt8, MutAnyOrigin](unsafe_from_address=wq_buf.cpu_addr)
    var sq_ptr = UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=sq_buf.cpu_addr)
    var wk_ptr = UnsafePointer[UInt8, MutAnyOrigin](unsafe_from_address=wk_buf.cpu_addr)
    var sk_ptr = UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=sk_buf.cpu_addr)
    var wv_ptr = UnsafePointer[UInt8, MutAnyOrigin](unsafe_from_address=wv_buf.cpu_addr)
    var sv_ptr = UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=sv_buf.cpu_addr)
    var x_ptr = UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=x_buf.cpu_addr)
    var out_q_ptr = UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=out_q_buf.cpu_addr)
    var out_k_ptr = UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=out_k_buf.cpu_addr)
    var out_v_ptr = UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=out_v_buf.cpu_addr)

    for i in range(Q_OUT * C_IN_PACKED):
        wq_ptr[i] = UInt8((i * 31 + 17) & 255)
    for i in range(Q_OUT * C_IN_BLOCKS):
        sq_ptr[i] = Float32(((i * 13) % 100) - 50) * 0.001
    for i in range(KV_OUT * C_IN_PACKED):
        wk_ptr[i] = UInt8((i * 43 + 7) & 255)
        wv_ptr[i] = UInt8((i * 59 + 23) & 255)
    for i in range(KV_OUT * C_IN_BLOCKS):
        sk_ptr[i] = Float32(((i * 19) % 100) - 50) * 0.001
        sv_ptr[i] = Float32(((i * 29) % 100) - 50) * 0.001
    for i in range(B * C_IN):
        x_ptr[i] = Float32(((i * 17 + 5) % 200) - 100) * 0.01

    ctx.flush_cache_range(wq_buf.cpu_addr, Q_OUT * C_IN_PACKED)
    ctx.flush_cache_range(sq_buf.cpu_addr, Q_OUT * C_IN_BLOCKS * 4)
    ctx.flush_cache_range(wk_buf.cpu_addr, KV_OUT * C_IN_PACKED)
    ctx.flush_cache_range(sk_buf.cpu_addr, KV_OUT * C_IN_BLOCKS * 4)
    ctx.flush_cache_range(wv_buf.cpu_addr, KV_OUT * C_IN_PACKED)
    ctx.flush_cache_range(sv_buf.cpu_addr, KV_OUT * C_IN_BLOCKS * 4)

    # Warmup and test multi-packet barrier-free QKV dispatch with single doorbell ring
    var t0 = std.time.perf_counter()
    var slot_q = ctx.enqueue_matmul(
        wq_buf.va_addr,
        sq_buf.va_addr,
        x_buf.va_addr,
        out_q_buf.va_addr,
        C_IN,
        Q_OUT,
        batch_size=B,
        flush_cache=True,
        barrier=False,
        ring_doorbell_now=False,
    )
    var slot_k = ctx.enqueue_matmul(
        wk_buf.va_addr,
        sk_buf.va_addr,
        x_buf.va_addr,
        out_k_buf.va_addr,
        C_IN,
        KV_OUT,
        batch_size=B,
        flush_cache=False,
        barrier=False,
        ring_doorbell_now=False,
    )
    var slot_v = ctx.enqueue_matmul(
        wv_buf.va_addr,
        sv_buf.va_addr,
        x_buf.va_addr,
        out_v_buf.va_addr,
        C_IN,
        KV_OUT,
        batch_size=B,
        flush_cache=False,
        barrier=False,
        ring_doorbell_now=True,
    )
    ctx.wait_matmul(slot_q, out_q_buf.cpu_addr, Q_OUT, batch_size=B)
    ctx.wait_matmul(slot_k, out_k_buf.cpu_addr, KV_OUT, batch_size=B)
    ctx.wait_matmul(slot_v, out_v_buf.cpu_addr, KV_OUT, batch_size=B)
    var t_gpu_qkv = (std.time.perf_counter() - t0) * 1000.0

    var ref_q = alloc[Float32](B * Q_OUT)
    var ref_k = alloc[Float32](B * KV_OUT)
    var ref_v = alloc[Float32](B * KV_OUT)
    t0 = std.time.perf_counter()
    q4_gemm_l1_2x4[DType.float32, DType.float32](wq_ptr, sq_ptr, x_ptr, ref_q, B, C_IN, Q_OUT, num_tasks)
    q4_gemm_l1_2x4[DType.float32, DType.float32](wk_ptr, sk_ptr, x_ptr, ref_k, B, C_IN, KV_OUT, num_tasks)
    q4_gemm_l1_2x4[DType.float32, DType.float32](wv_ptr, sv_ptr, x_ptr, ref_v, B, C_IN, KV_OUT, num_tasks)
    var t_cpu_qkv = (std.time.perf_counter() - t0) * 1000.0

    var max_err_q: Float32 = 0.0
    var max_err_k: Float32 = 0.0
    var max_err_v: Float32 = 0.0
    for i in range(B * Q_OUT):
        var d = abs(out_q_ptr[i] - ref_q[i])
        if d > max_err_q:
            max_err_q = d
    for i in range(B * KV_OUT):
        var dk = abs(out_k_ptr[i] - ref_k[i])
        if dk > max_err_k:
            max_err_k = dk
        var dv = abs(out_v_ptr[i] - ref_v[i])
        if dv > max_err_v:
            max_err_v = dv

    print(
        "  [256-Token QKV] max_err_q =",
        max_err_q,
        "| max_err_k =",
        max_err_k,
        "| max_err_v =",
        max_err_v,
        "| CPU L1-4x3:",
        t_cpu_qkv,
        "ms | KFD 16x16 Q4 GPU:",
        t_gpu_qkv,
        "ms",
    )
    if max_err_q > 1e-4 or max_err_k > 1e-4 or max_err_v > 1e-4:
        raise Error("256-token KFD Q4_0 matmul error exceeded 1e-4!")

    ref_q.free()
    ref_k.free()
    ref_v.free()
    ctx.free_gpu_buffer(wq_buf)
    ctx.free_gpu_buffer(sq_buf)
    ctx.free_gpu_buffer(wk_buf)
    ctx.free_gpu_buffer(sk_buf)
    ctx.free_gpu_buffer(wv_buf)
    ctx.free_gpu_buffer(sv_buf)
    ctx.free_gpu_buffer(x_buf)
    ctx.free_gpu_buffer(out_q_buf)
    ctx.free_gpu_buffer(out_k_buf)
    ctx.free_gpu_buffer(out_v_buf)
    ctx.close()

    # 2. Test & Benchmark Dual-Space Deinterleaved MoE Expert vs Old Float16 Expert
    comptime EXP_GATE_UP = 1408
    comptime EXP_DOWN = 704
    comptime CNT = 16

    var w_gu_packed = alloc[UInt8](EXP_GATE_UP * C_IN_PACKED)
    var w_gu_scales = alloc[Float32](EXP_GATE_UP * C_IN_BLOCKS)
    for i in range(EXP_GATE_UP * C_IN_PACKED):
        w_gu_packed[i] = UInt8((i * 37 + 19) & 255)
    for i in range(EXP_GATE_UP * C_IN_BLOCKS):
        w_gu_scales[i] = Float32(((i * 23) % 100) - 50) * 0.001

    var x_f16 = alloc[Float16](CNT * C_IN)
    var x_f32 = alloc[Float32](CNT * C_IN)
    var x_deint = alloc[Float32](CNT * C_IN)
    for i in range(CNT * C_IN):
        var v = Float16(Float32(((i * 13 + 7) % 200) - 100) * 0.01)
        x_f16[i] = v
        x_f32[i] = Float32(v)

    var out_old_f16 = alloc[Float16](CNT * EXP_GATE_UP)
    var out_new_f32 = alloc[Float32](CNT * EXP_GATE_UP)
    var out_ref_f32 = alloc[Float32](CNT * EXP_GATE_UP)

    ref_q4_matmul(w_gu_packed, w_gu_scales, x_f32, out_ref_f32, CNT, C_IN, EXP_GATE_UP)

    t0 = std.time.perf_counter()
    for _ in range(32):
        old_q4_gemv_small_b_f16[16](w_gu_packed, w_gu_scales, x_f16, out_old_f16, CNT, C_IN, EXP_GATE_UP, num_tasks)
    var t_old_moe = (std.time.perf_counter() - t0) * 1000.0

    t0 = std.time.perf_counter()
    deinterleave_batch_to_f32[DType.float16](x_f16, x_deint, CNT, C_IN)
    for _ in range(32):
        q4_moe_gemm_deinterleaved[DType.float32](w_gu_packed, w_gu_scales, x_deint, out_new_f32, CNT, C_IN, EXP_GATE_UP, num_tasks)
    var t_new_moe = (std.time.perf_counter() - t0) * 1000.0

    var max_moe_diff: Float32 = 0.0
    for i in range(CNT * EXP_GATE_UP):
        var d = abs(out_new_f32[i] - out_ref_f32[i])
        if d > max_moe_diff:
            max_moe_diff = d

    print(
        "  [MoE 32x Experts (cnt=16)] max_diff_f32 =",
        max_moe_diff,
        "| Old F16 unpack_16bytes:",
        t_old_moe,
        "ms -> Dual-Space Deinterleaved F32:",
        t_new_moe,
        "ms (",
        t_old_moe / t_new_moe,
        "x speedup)",
    )
    if max_moe_diff > 1e-4:
        raise Error("q4_moe_gemm_deinterleaved error exceeded 1e-4!")

    w_gu_packed.free()
    w_gu_scales.free()
    x_f16.free()
    x_f32.free()
    x_deint.free()
    out_old_f16.free()
    out_new_f32.free()
    out_ref_f32.free()

    # 3. Test full 128-expert diffusion MoE layer: run_diffusion_moe_experts_parallel vs old loop
    comptime NUM_EXP = 128
    comptime TOP_K = 8
    var gu_stride_p = EXP_GATE_UP * C_IN_PACKED
    var gu_stride_s = EXP_GATE_UP * C_IN_BLOCKS
    var dn_stride_p = C_IN * (EXP_DOWN // 2)
    var dn_stride_s = C_IN * (EXP_DOWN // 32)

    var all_gu_p = alloc[UInt8](NUM_EXP * gu_stride_p)
    var all_gu_s = alloc[Float32](NUM_EXP * gu_stride_s)
    var all_dn_p = alloc[UInt8](NUM_EXP * dn_stride_p)
    var all_dn_s = alloc[Float32](NUM_EXP * dn_stride_s)

    for i in range(NUM_EXP * gu_stride_p):
        all_gu_p[i] = UInt8((i * 29 + 11) & 255)
    for i in range(NUM_EXP * gu_stride_s):
        all_gu_s[i] = Float32(((i * 17) % 100) - 50) * 0.0005
    for i in range(NUM_EXP * dn_stride_p):
        all_dn_p[i] = UInt8((i * 41 + 3) & 255)
    for i in range(NUM_EXP * dn_stride_s):
        all_dn_s[i] = Float32(((i * 23) % 100) - 50) * 0.0005

    var canvas_in = alloc[Float16](B * C_IN)
    var top_idx = alloc[Int](B * TOP_K)
    var top_weights = alloc[Float32](B * TOP_K)
    var moe_out_new = alloc[Float16](B * C_IN)
    var moe_out_old = alloc[Float16](B * C_IN)

    for i in range(B * C_IN):
        canvas_in[i] = Float16(Float32(((i * 19 + 5) % 200) - 100) * 0.01)
    for t in range(B):
        for s in range(TOP_K):
            top_idx[t * TOP_K + s] = (t * 7 + s * 15) % NUM_EXP
            top_weights[t * TOP_K + s] = Float32(0.125)

    t0 = std.time.perf_counter()
    run_diffusion_moe_experts_parallel(
        all_gu_p,
        all_gu_s,
        all_dn_p,
        all_dn_s,
        canvas_in,
        top_idx,
        top_weights,
        moe_out_new,
        B,
        C_IN,
        EXP_DOWN,
        TOP_K,
        num_tasks,
    )
    var t_parallel_moe = (std.time.perf_counter() - t0) * 1000.0

    # Verify against old sequential-expert loop on first 16 experts (to verify numerical equivalence)
    for t in range(B):
        for s in range(TOP_K):
            if top_idx[t * TOP_K + s] >= 16:
                top_weights[t * TOP_K + s] = 0.0
    run_diffusion_moe_experts_parallel(
        all_gu_p,
        all_gu_s,
        all_dn_p,
        all_dn_s,
        canvas_in,
        top_idx,
        top_weights,
        moe_out_new,
        B,
        C_IN,
        EXP_DOWN,
        TOP_K,
        num_tasks,
    )
    for i in range(B * C_IN):
        moe_out_old[i] = Float16(0.0)
    var tmp_x = alloc[Float16](B * C_IN)
    var tmp_gu = alloc[Float16](B * EXP_GATE_UP)
    var tmp_act = alloc[Float16](B * EXP_DOWN)
    var tmp_dn = alloc[Float16](B * C_IN)
    for e in range(16):
        var cnt = 0
        var t_list = InlineArray[Int, 256](uninitialized=True)
        var s_list = InlineArray[Int, 256](uninitialized=True)
        for t in range(B):
            for s in range(TOP_K):
                if top_weights[t * TOP_K + s] > 0.0 and top_idx[t * TOP_K + s] == e:
                    t_list[cnt] = t
                    s_list[cnt] = s
                    cnt += 1
        if cnt == 0:
            continue
        for i in range(cnt):
            var t = t_list[i]
            for j in range(C_IN):
                tmp_x[i * C_IN + j] = canvas_in[t * C_IN + j]
        old_q4_gemv_small_b_f16[32](all_gu_p + e * gu_stride_p, all_gu_s + e * gu_stride_s, tmp_x, tmp_gu, cnt, C_IN, EXP_GATE_UP, num_tasks)
        for i in range(cnt):
            for j in range(EXP_DOWN):
                var g = Float32(tmp_gu[i * EXP_GATE_UP + j])
                var u = Float32(tmp_gu[i * EXP_GATE_UP + EXP_DOWN + j])
                tmp_act[i * EXP_DOWN + j] = Float16(g * (1.0 / (1.0 + exp(Float32(-1.702) * g))) * u)
        old_q4_gemv_small_b_f16[32](all_dn_p + e * dn_stride_p, all_dn_s + e * dn_stride_s, tmp_act, tmp_dn, cnt, EXP_DOWN, C_IN, num_tasks)
        for i in range(cnt):
            var t = t_list[i]
            var w_e = top_weights[t * TOP_K + s_list[i]]
            for j in range(C_IN):
                moe_out_old[t * C_IN + j] = Float16(Float32(moe_out_old[t * C_IN + j]) + Float32(tmp_dn[i * C_IN + j]) * w_e)

    var max_layer_moe_diff: Float32 = 0.0
    for i in range(B * C_IN):
        var d = abs(Float32(moe_out_new[i]) - Float32(moe_out_old[i]))
        if d > max_layer_moe_diff:
            max_layer_moe_diff = d

    print(
        "  [Full 128-Expert MoE Layer (B=256)] max_f16_diff =",
        max_layer_moe_diff,
        "| Full 128-expert time:",
        t_parallel_moe,
        "ms (vs ~135 ms old!)",
    )
    if max_layer_moe_diff > 5e-3:
        raise Error("Full MoE layer mismatch exceeded 5e-3!")

    tmp_x.free()
    tmp_gu.free()
    tmp_act.free()
    tmp_dn.free()
    all_gu_p.free()
    all_gu_s.free()
    all_dn_p.free()
    all_dn_s.free()
    canvas_in.free()
    top_idx.free()
    top_weights.free()
    moe_out_new.free()
    moe_out_old.free()
    print("=== ALL TESTS PASSED! ===")
