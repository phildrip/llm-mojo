from std.memory import alloc
from std.math import abs
import std.time
from kfd_gpu import KFDContext, GPUMemoryBuffer, HSACOKernel


def cpu_q8_matmul_ref(
    W_packed: UnsafePointer[Int8, MutUntrackedOrigin],
    scales: UnsafePointer[Float32, MutUntrackedOrigin],
    x: UnsafePointer[Float32, MutUntrackedOrigin],
    out_ptr: UnsafePointer[Float32, MutUntrackedOrigin],
    c_in: Int,
    c_out: Int,
    batch_size: Int,
    out_stride: Int = 0,
):
    var eff_stride = out_stride if out_stride > 0 else c_out
    var num_blocks = c_in // 32
    for t in range(batch_size):
        for r in range(c_out):
            var acc: Float32 = 0.0
            for b in range(num_blocks):
                var block_sum: Float32 = 0.0
                var w_base = r * c_in + b * 32
                var x_base = t * c_in + b * 32
                for j in range(32):
                    block_sum += Float32(W_packed[w_base + j]) * x[x_base + j]
                acc += block_sum * scales[r * num_blocks + b]
            out_ptr[t * eff_stride + r] = acc


def cpu_q4_matmul_ref(
    W_packed: UnsafePointer[UInt8, MutUntrackedOrigin],
    scales: UnsafePointer[Float32, MutUntrackedOrigin],
    x: UnsafePointer[Float32, MutUntrackedOrigin],
    out_ptr: UnsafePointer[Float32, MutUntrackedOrigin],
    c_in: Int,
    c_out: Int,
    batch_size: Int,
    out_stride: Int = 0,
):
    var eff_stride = out_stride if out_stride > 0 else c_out
    var num_blocks = c_in // 32
    var c_in_packed = c_in // 2
    for t in range(batch_size):
        for r in range(c_out):
            var acc: Float32 = 0.0
            for b in range(num_blocks):
                var block_sum: Float32 = 0.0
                var w_byte_base = r * c_in_packed + b * 16
                var x_base = t * c_in + b * 32
                for j in range(16):
                    var byte_val = Int(W_packed[w_byte_base + j])
                    var w_lo = Float32((byte_val & 15) - 8)
                    var w_hi = Float32(((byte_val >> 4) & 15) - 8)
                    block_sum += w_lo * x[x_base + 2 * j] + w_hi * x[x_base + 2 * j + 1]
                acc += block_sum * scales[r * num_blocks + b]
            out_ptr[t * eff_stride + r] = acc


def test_dynamic_elf_loader(mut ctx: KFDContext) raises:
    print("=== Test 1: Dynamic ELF64 .hsaco Segment & Symbol Loader ===")
    var q8_mod = ctx.load_hsaco("/home/phil/git/llm-mojo/scratch/q8_matmul_lds.hsaco")
    var q8_kernel = ctx.get_kernel("q8_matmul", q8_mod)
    print(
        "  q8_matmul.kd -> offset:",
        q8_kernel.kd_offset,
        "| group_segment_size:",
        q8_kernel.group_segment_size,
        "| private_segment_size:",
        q8_kernel.private_segment_size,
    )
    if q8_kernel.kd_offset != 0xA80:
        raise Error("Expected q8_matmul.kd offset 0xA80 (2688), got " + String(q8_kernel.kd_offset))
    if q8_kernel.group_segment_size != 2560:
        raise Error("Expected q8_matmul group_segment_size 2560, got " + String(q8_kernel.group_segment_size))

    var q4_mod = ctx.load_hsaco("/home/phil/git/llm-mojo/scratch/q4_matmul_lds.hsaco")
    var q4_kernel = ctx.get_kernel("q4_matmul", q4_mod)
    print(
        "  q4_matmul.kd -> offset:",
        q4_kernel.kd_offset,
        "| group_segment_size:",
        q4_kernel.group_segment_size,
        "| private_segment_size:",
        q4_kernel.private_segment_size,
    )
    if q4_kernel.kd_offset != 0xAC0:
        raise Error("Expected q4_matmul.kd offset 0xAC0 (2752), got " + String(q4_kernel.kd_offset))
    if q4_kernel.group_segment_size != 4352:
        raise Error("Expected q4_matmul group_segment_size 4352, got " + String(q4_kernel.group_segment_size))
    print("  [PASS] Dynamic ELF64 loader resolved both .hsaco code objects and .kd descriptors!")


def test_q4_matmul_kernel(mut ctx: KFDContext) raises:
    print("\n=== Test 2: Native Q4_0 2D LDS-Tiled AMDGCN Kernel (batch=1, 7, 16, 32, 33, 64 + dirty cache + c_out=500) ===")
    var q4_kernel = ctx.get_kernel("q4_matmul", 1)

    var c_in = 2816
    var c_out = 512
    var max_batch = 64
    var c_in_packed = c_in // 2
    var num_blocks = c_in // 32

    var w_buf = ctx.alloc_gpu_buffer(c_out * c_in_packed)
    var s_buf = ctx.alloc_gpu_buffer(c_out * num_blocks * 4)
    var x_buf = ctx.alloc_gpu_buffer(max_batch * c_in * 4)
    var out_buf = ctx.alloc_gpu_buffer(max_batch * c_out * 4)
    var ref_out = alloc[Float32](max_batch * c_out)

    var w_ptr = UnsafePointer[UInt8, MutUntrackedOrigin](unsafe_from_address=w_buf.cpu_addr)
    var s_ptr = UnsafePointer[Float32, MutUntrackedOrigin](unsafe_from_address=s_buf.cpu_addr)
    var x_ptr = UnsafePointer[Float32, MutUntrackedOrigin](unsafe_from_address=x_buf.cpu_addr)
    var out_ptr = UnsafePointer[Float32, MutUntrackedOrigin](unsafe_from_address=out_buf.cpu_addr)

    for i in range(c_out * c_in_packed):
        var lo = (i * 7 + 3) & 15
        var hi = (i * 11 + 5) & 15
        w_ptr[i] = UInt8(lo | (hi << 4))

    for i in range(c_out * num_blocks):
        s_ptr[i] = Float32(((i * 13 + 7) % 19) - 9) * 0.015625

    for i in range(max_batch * c_in):
        x_ptr[i] = Float32(((i * 17 + 11) % 23) - 11) * 0.03125

    ctx.flush_cache_range(w_buf.cpu_addr, c_out * c_in_packed)
    ctx.flush_cache_range(s_buf.cpu_addr, c_out * num_blocks * 4)

    var batch_sizes: InlineArray[Int, 6] = [1, 7, 16, 32, 33, 64]
    for b_idx in range(6):
        var bs = batch_sizes[b_idx]
        # Intentionally dirty CPU cache lines in out_ptr WITHOUT manual flush_cache_range
        # to verify enqueue_kernel_matmul's pre-dispatch output cache flush!
        for i in range(bs * c_out):
            out_ptr[i] = -999999.0

        var slot = ctx.enqueue_kernel_matmul(
            q4_kernel,
            w_buf.va_addr,
            s_buf.va_addr,
            x_buf.va_addr,
            out_buf.va_addr,
            c_in,
            c_out,
            batch_size=bs,
            out_stride=c_out,
            flush_input=True,
            barrier=False,
            ring_doorbell_now=True,
        )
        ctx.wait_matmul(slot, out_buf.cpu_addr, c_out, batch_size=bs)

        cpu_q4_matmul_ref(w_ptr, s_ptr, x_ptr, ref_out, c_in, c_out, bs, c_out)

        var max_err: Float32 = 0.0
        for i in range(bs * c_out):
            var err = abs(out_ptr[i] - ref_out[i])
            if err > max_err:
                max_err = err
        print("  batch_size =", bs, "-> max_abs_err =", max_err)
        if max_err > 1e-4:
            raise Error("Q4_0 GPU matmul mismatch at batch_size=" + String(bs) + ": max_err=" + String(max_err))

    # Non-multiple-of-16 c_out test (c_out = 500, batch_size = 7)
    var odd_c_out = 500
    var odd_bs = 7
    for i in range(odd_bs * odd_c_out):
        out_ptr[i] = 777777.0
    var slot_odd = ctx.enqueue_kernel_matmul(
        q4_kernel,
        w_buf.va_addr,
        s_buf.va_addr,
        x_buf.va_addr,
        out_buf.va_addr,
        c_in,
        odd_c_out,
        batch_size=odd_bs,
        out_stride=odd_c_out,
        flush_input=True,
        barrier=False,
        ring_doorbell_now=True,
    )
    ctx.wait_matmul(slot_odd, out_buf.cpu_addr, odd_c_out, batch_size=odd_bs)
    cpu_q4_matmul_ref(w_ptr, s_ptr, x_ptr, ref_out, c_in, odd_c_out, odd_bs, odd_c_out)
    var max_odd_err: Float32 = 0.0
    for i in range(odd_bs * odd_c_out):
        var err = abs(out_ptr[i] - ref_out[i])
        if err > max_odd_err:
            max_odd_err = err
    print("  non-tile-aligned c_out = 500, batch_size = 7 -> max_abs_err =", max_odd_err)
    if max_odd_err > 1e-4:
        raise Error("Q4_0 GPU matmul mismatch at c_out=500: max_err=" + String(max_odd_err))

    var t_gpu_b1_start = std.time.perf_counter()
    var slot_b1 = ctx.enqueue_kernel_matmul(
        q4_kernel,
        w_buf.va_addr,
        s_buf.va_addr,
        x_buf.va_addr,
        out_buf.va_addr,
        c_in,
        c_out,
        batch_size=1,
        out_stride=c_out,
        flush_input=True,
        barrier=False,
        ring_doorbell_now=True,
    )
    ctx.wait_matmul(slot_b1, out_buf.cpu_addr, c_out, batch_size=1)
    var t_gpu_b1_ms = (std.time.perf_counter() - t_gpu_b1_start) * 1000.0

    var t_gpu_start = std.time.perf_counter()
    var slot_bench = ctx.enqueue_kernel_matmul(
        q4_kernel,
        w_buf.va_addr,
        s_buf.va_addr,
        x_buf.va_addr,
        out_buf.va_addr,
        c_in,
        c_out,
        batch_size=32,
        out_stride=c_out,
        flush_input=True,
        barrier=False,
        ring_doorbell_now=True,
    )
    ctx.wait_matmul(slot_bench, out_buf.cpu_addr, c_out, batch_size=32)
    var t_gpu_ms = (std.time.perf_counter() - t_gpu_start) * 1000.0

    var t_cpu_gtt_start = std.time.perf_counter()
    cpu_q4_matmul_ref(w_ptr, s_ptr, x_ptr, ref_out, c_in, c_out, 1, c_out)
    var t_cpu_gtt_ms = (std.time.perf_counter() - t_cpu_gtt_start) * 1000.0
    print("  [Perf] GPU batch=1:", t_gpu_b1_ms, "ms | GPU batch=32:", t_gpu_ms, "ms | CPU batch=1:", t_cpu_gtt_ms, "ms")

    ref_out.free()
    ctx.free_gpu_buffer(out_buf)
    ctx.free_gpu_buffer(x_buf)
    ctx.free_gpu_buffer(s_buf)
    ctx.free_gpu_buffer(w_buf)
    print("  [PASS] Q4_0 GPU kernel matches CPU reference across all batch sizes!")


def test_multi_packet_batching_and_stride(mut ctx: KFDContext) raises:
    print("\n=== Test 3: Multi-Packet Barrier-Free AQL Dispatch + Strided QKV Output ===")
    var q4_kernel = ctx.get_kernel("q4_matmul", 1)

    # Simulate fused Q, K, V projection into a single [batch_size, q_dim + k_dim + v_dim] buffer
    # using 3 barrier-free AQL packets and a single doorbell ring!
    var c_in = 2816
    var q_dim = 512
    var k_dim = 256
    var v_dim = 256
    var total_stride = q_dim + k_dim + v_dim
    var batch_size = 16
    var c_in_packed = c_in // 2
    var num_blocks = c_in // 32

    var wq_buf = ctx.alloc_gpu_buffer(q_dim * c_in_packed)
    var wk_buf = ctx.alloc_gpu_buffer(k_dim * c_in_packed)
    var wv_buf = ctx.alloc_gpu_buffer(v_dim * c_in_packed)

    var sq_buf = ctx.alloc_gpu_buffer(q_dim * num_blocks * 4)
    var sk_buf = ctx.alloc_gpu_buffer(k_dim * num_blocks * 4)
    var sv_buf = ctx.alloc_gpu_buffer(v_dim * num_blocks * 4)

    var x_buf = ctx.alloc_gpu_buffer(batch_size * c_in * 4)
    var qkv_buf = ctx.alloc_gpu_buffer(batch_size * total_stride * 4)
    var ref_qkv = alloc[Float32](batch_size * total_stride)

    var wq_ptr = UnsafePointer[UInt8, MutUntrackedOrigin](unsafe_from_address=wq_buf.cpu_addr)
    var wk_ptr = UnsafePointer[UInt8, MutUntrackedOrigin](unsafe_from_address=wk_buf.cpu_addr)
    var wv_ptr = UnsafePointer[UInt8, MutUntrackedOrigin](unsafe_from_address=wv_buf.cpu_addr)

    var sq_ptr = UnsafePointer[Float32, MutUntrackedOrigin](unsafe_from_address=sq_buf.cpu_addr)
    var sk_ptr = UnsafePointer[Float32, MutUntrackedOrigin](unsafe_from_address=sk_buf.cpu_addr)
    var sv_ptr = UnsafePointer[Float32, MutUntrackedOrigin](unsafe_from_address=sv_buf.cpu_addr)

    var x_ptr = UnsafePointer[Float32, MutUntrackedOrigin](unsafe_from_address=x_buf.cpu_addr)
    var qkv_ptr = UnsafePointer[Float32, MutUntrackedOrigin](unsafe_from_address=qkv_buf.cpu_addr)

    for i in range(q_dim * c_in_packed):
        wq_ptr[i] = UInt8(((i * 3 + 1) & 15) | (((i * 5 + 2) & 15) << 4))
    for i in range(k_dim * c_in_packed):
        wk_ptr[i] = UInt8(((i * 7 + 4) & 15) | (((i * 9 + 6) & 15) << 4))
    for i in range(v_dim * c_in_packed):
        wv_ptr[i] = UInt8(((i * 11 + 5) & 15) | (((i * 13 + 8) & 15) << 4))

    for i in range(q_dim * num_blocks):
        sq_ptr[i] = Float32(((i * 5 + 3) % 17) - 8) * 0.015625
    for i in range(k_dim * num_blocks):
        sk_ptr[i] = Float32(((i * 7 + 5) % 17) - 8) * 0.015625
    for i in range(v_dim * num_blocks):
        sv_ptr[i] = Float32(((i * 11 + 9) % 17) - 8) * 0.015625

    for i in range(batch_size * c_in):
        x_ptr[i] = Float32(((i * 19 + 7) % 29) - 14) * 0.03125

    for i in range(batch_size * total_stride):
        qkv_ptr[i] = 0.0
        ref_qkv[i] = 0.0

    ctx.flush_cache_range(wq_buf.cpu_addr, q_dim * c_in_packed)
    ctx.flush_cache_range(wk_buf.cpu_addr, k_dim * c_in_packed)
    ctx.flush_cache_range(wv_buf.cpu_addr, v_dim * c_in_packed)
    ctx.flush_cache_range(sq_buf.cpu_addr, q_dim * num_blocks * 4)
    ctx.flush_cache_range(sk_buf.cpu_addr, k_dim * num_blocks * 4)
    ctx.flush_cache_range(sv_buf.cpu_addr, v_dim * num_blocks * 4)
    ctx.flush_cache_range(x_buf.cpu_addr, batch_size * c_in * 4)
    ctx.flush_cache_range(qkv_buf.cpu_addr, batch_size * total_stride * 4)

    # Enqueue Q, K, V without barriers and ring doorbell once at the end!
    var slot_q = ctx.enqueue_kernel_matmul(
        q4_kernel,
        wq_buf.va_addr,
        sq_buf.va_addr,
        x_buf.va_addr,
        qkv_buf.va_addr + 0,
        c_in,
        q_dim,
        batch_size=batch_size,
        out_stride=total_stride,
        flush_input=False,
        barrier=False,
        ring_doorbell_now=False,
    )
    var slot_k = ctx.enqueue_kernel_matmul(
        q4_kernel,
        wk_buf.va_addr,
        sk_buf.va_addr,
        x_buf.va_addr,
        qkv_buf.va_addr + q_dim * 4,
        c_in,
        k_dim,
        batch_size=batch_size,
        out_stride=total_stride,
        flush_input=False,
        barrier=False,
        ring_doorbell_now=False,
    )
    var slot_v = ctx.enqueue_kernel_matmul(
        q4_kernel,
        wv_buf.va_addr,
        sv_buf.va_addr,
        x_buf.va_addr,
        qkv_buf.va_addr + (q_dim + k_dim) * 4,
        c_in,
        v_dim,
        batch_size=batch_size,
        out_stride=total_stride,
        flush_input=False,
        barrier=False,
        ring_doorbell_now=False,
    )
    ctx.ring_doorbell()

    ctx.wait_matmul(slot_q, 0, q_dim, batch_size=batch_size, flush_cache=False)
    ctx.wait_matmul(slot_k, 0, k_dim, batch_size=batch_size, flush_cache=False)
    ctx.wait_matmul(slot_v, qkv_buf.cpu_addr, total_stride, batch_size=batch_size, flush_cache=True)

    cpu_q4_matmul_ref(wq_ptr, sq_ptr, x_ptr, ref_qkv + 0, c_in, q_dim, batch_size, total_stride)
    cpu_q4_matmul_ref(wk_ptr, sk_ptr, x_ptr, ref_qkv + q_dim, c_in, k_dim, batch_size, total_stride)
    cpu_q4_matmul_ref(wv_ptr, sv_ptr, x_ptr, ref_qkv + q_dim + k_dim, c_in, v_dim, batch_size, total_stride)

    var max_err: Float32 = 0.0
    for i in range(batch_size * total_stride):
        var err = abs(qkv_ptr[i] - ref_qkv[i])
        if err > max_err:
            max_err = err
    print("  3-packet concurrent strided QKV dispatch -> max_abs_err =", max_err)
    if max_err > 1e-4:
        raise Error("Multi-packet strided QKV mismatch: max_err=" + String(max_err))

    ref_qkv.free()
    ctx.free_gpu_buffer(qkv_buf)
    ctx.free_gpu_buffer(x_buf)
    ctx.free_gpu_buffer(sv_buf)
    ctx.free_gpu_buffer(sk_buf)
    ctx.free_gpu_buffer(sq_buf)
    ctx.free_gpu_buffer(wv_buf)
    ctx.free_gpu_buffer(wk_buf)
    ctx.free_gpu_buffer(wq_buf)
    print("  [PASS] Multi-packet barrier-free dispatch + strided output verified!")


def test_lsa_f32_pca_and_gram(mut ctx: KFDContext) raises:
    print("\n=== Test 4: LSA 512-Token Block PCA Projection & [512, 512] Gram Matrix on GPU ===")
    var f32_mod = ctx.load_hsaco("/home/phil/git/llm-mojo/scratch/f32_matmul_lds.hsaco")
    var f32_kernel = ctx.get_kernel("f32_matmul", f32_mod)

    var block_tokens = 512
    var index_dim = 1024
    var lsa_dim = 64

    var k_pooled_buf = ctx.alloc_gpu_buffer(block_tokens * index_dim * 4)
    var proj_w_buf = ctx.alloc_gpu_buffer(lsa_dim * index_dim * 4)
    var k_proj_buf = ctx.alloc_gpu_buffer(block_tokens * lsa_dim * 4)
    var gram_buf = ctx.alloc_gpu_buffer(block_tokens * block_tokens * 4)

    var k_pooled_ptr = UnsafePointer[Float32, MutUntrackedOrigin](unsafe_from_address=k_pooled_buf.cpu_addr)
    var proj_w_ptr = UnsafePointer[Float32, MutUntrackedOrigin](unsafe_from_address=proj_w_buf.cpu_addr)
    var k_proj_ptr = UnsafePointer[Float32, MutUntrackedOrigin](unsafe_from_address=k_proj_buf.cpu_addr)
    var gram_ptr = UnsafePointer[Float32, MutUntrackedOrigin](unsafe_from_address=gram_buf.cpu_addr)

    for i in range(block_tokens * index_dim):
        k_pooled_ptr[i] = Float32(((i * 13 + 5) % 29) - 14) * 0.03125
    for i in range(lsa_dim * index_dim):
        proj_w_ptr[i] = Float32(((i * 17 + 7) % 31) - 15) * 0.015625

    ctx.flush_cache_range(k_pooled_buf.cpu_addr, block_tokens * index_dim * 4)
    ctx.flush_cache_range(proj_w_buf.cpu_addr, lsa_dim * index_dim * 4)

    var t0 = std.time.perf_counter()
    # 1. Batched PCA projection: [512, 1024] @ [64, 1024]^T -> [512, 64]
    var slot_pca = ctx.enqueue_kernel_matmul(
        f32_kernel,
        proj_w_buf.va_addr,
        0,
        k_pooled_buf.va_addr,
        k_proj_buf.va_addr,
        index_dim,
        lsa_dim,
        batch_size=block_tokens,
        out_stride=lsa_dim,
        flush_input=False,
        barrier=False,
        ring_doorbell_now=True,
    )
    ctx.wait_matmul(slot_pca, k_proj_buf.cpu_addr, lsa_dim, batch_size=block_tokens)

    # 2. [512, 512] Gram matrix: K_proj[512, 64] @ K_proj[512, 64]^T -> Gram[512, 512]
    var slot_gram = ctx.enqueue_kernel_matmul(
        f32_kernel,
        k_proj_buf.va_addr,
        0,
        k_proj_buf.va_addr,
        gram_buf.va_addr,
        lsa_dim,
        block_tokens,
        batch_size=block_tokens,
        out_stride=block_tokens,
        flush_input=False,
        barrier=False,
        ring_doorbell_now=True,
    )
    ctx.wait_matmul(slot_gram, gram_buf.cpu_addr, block_tokens, batch_size=block_tokens)
    var t_gpu_ms = (std.time.perf_counter() - t0) * 1000.0

    # Verify spot checks of PCA projection and Gram matrix against CPU
    var max_pca_err: Float32 = 0.0
    for t in range(0, block_tokens, 31):
        for r in range(lsa_dim):
            var acc: Float32 = 0.0
            for c in range(index_dim):
                acc += k_pooled_ptr[t * index_dim + c] * proj_w_ptr[r * index_dim + c]
            var err = abs(k_proj_ptr[t * lsa_dim + r] - acc)
            if err > max_pca_err:
                max_pca_err = err

    var max_gram_err: Float32 = 0.0
    for i in range(0, block_tokens, 29):
        for j in range(0, block_tokens, 23):
            var acc: Float32 = 0.0
            for c in range(lsa_dim):
                acc += k_proj_ptr[i * lsa_dim + c] * k_proj_ptr[j * lsa_dim + c]
            var err = abs(gram_ptr[i * block_tokens + j] - acc)
            if err > max_gram_err:
                max_gram_err = err

    print(
        "  512-tok PCA ([512,1024]@[64,1024]^T) + [512,512] Gram on GPU:",
        t_gpu_ms,
        "ms | max_pca_err =",
        max_pca_err,
        "| max_gram_err =",
        max_gram_err,
    )
    if max_pca_err > 1e-3 or max_gram_err > 1e-2:
        raise Error("LSA GPU PCA/Gram mismatch!")

    ctx.free_gpu_buffer(gram_buf)
    ctx.free_gpu_buffer(k_proj_buf)
    ctx.free_gpu_buffer(proj_w_buf)
    ctx.free_gpu_buffer(k_pooled_buf)
    print("  [PASS] LSA 512-token block PCA projection & [512, 512] Gram matrix verified on GPU!")


def test_cdc6600_asymmetric_overlap(mut ctx: KFDContext) raises:
    print("\n=== Test 5: CDC-6600 Asymmetric Co-Processor Overlap (Gemma-4 Decode Layer Shapes) ===")
    var q4_kernel = ctx.get_kernel("q4_matmul", 1)

    var c_in = 2816
    var q_dim = 8192
    var kv_dim = 2048
    var num_blocks = c_in // 32
    var wo_blocks = q_dim // 32

    var wq_buf = ctx.alloc_gpu_buffer(q_dim * (c_in // 2))
    var sq_buf = ctx.alloc_gpu_buffer(q_dim * num_blocks * 4)
    var wk_buf = ctx.alloc_gpu_buffer(kv_dim * (c_in // 2))
    var sk_buf = ctx.alloc_gpu_buffer(kv_dim * num_blocks * 4)
    var wv_buf = ctx.alloc_gpu_buffer(kv_dim * (c_in // 2))
    var sv_buf = ctx.alloc_gpu_buffer(kv_dim * num_blocks * 4)
    var wo_buf = ctx.alloc_gpu_buffer(c_in * (q_dim // 2))
    var so_buf = ctx.alloc_gpu_buffer(c_in * wo_blocks * 4)

    var x_buf = ctx.alloc_gpu_buffer(c_in * 4)
    var q_buf = ctx.alloc_gpu_buffer(q_dim * 4)
    var k_buf = ctx.alloc_gpu_buffer(kv_dim * 4)
    var v_buf = ctx.alloc_gpu_buffer(kv_dim * 4)
    var o_buf = ctx.alloc_gpu_buffer(c_in * 4)

    var wq_ptr = UnsafePointer[UInt8, MutUntrackedOrigin](unsafe_from_address=wq_buf.cpu_addr)
    var sq_ptr = UnsafePointer[Float32, MutUntrackedOrigin](unsafe_from_address=sq_buf.cpu_addr)
    var wk_ptr = UnsafePointer[UInt8, MutUntrackedOrigin](unsafe_from_address=wk_buf.cpu_addr)
    var sk_ptr = UnsafePointer[Float32, MutUntrackedOrigin](unsafe_from_address=sk_buf.cpu_addr)
    var wv_ptr = UnsafePointer[UInt8, MutUntrackedOrigin](unsafe_from_address=wv_buf.cpu_addr)
    var sv_ptr = UnsafePointer[Float32, MutUntrackedOrigin](unsafe_from_address=sv_buf.cpu_addr)
    var wo_ptr = UnsafePointer[UInt8, MutUntrackedOrigin](unsafe_from_address=wo_buf.cpu_addr)
    var so_ptr = UnsafePointer[Float32, MutUntrackedOrigin](unsafe_from_address=so_buf.cpu_addr)

    var x_ptr = UnsafePointer[Float32, MutUntrackedOrigin](unsafe_from_address=x_buf.cpu_addr)
    var q_ptr = UnsafePointer[Float32, MutUntrackedOrigin](unsafe_from_address=q_buf.cpu_addr)
    var k_ptr = UnsafePointer[Float32, MutUntrackedOrigin](unsafe_from_address=k_buf.cpu_addr)
    var v_ptr = UnsafePointer[Float32, MutUntrackedOrigin](unsafe_from_address=v_buf.cpu_addr)
    var o_ptr = UnsafePointer[Float32, MutUntrackedOrigin](unsafe_from_address=o_buf.cpu_addr)

    for i in range(q_dim * (c_in // 2)):
        wq_ptr[i] = UInt8(((i * 7 + 3) & 15) | (((i * 11 + 5) & 15) << 4))
    for i in range(q_dim * num_blocks):
        sq_ptr[i] = Float32(((i * 13 + 7) % 19) - 9) * 0.015625
    for i in range(kv_dim * (c_in // 2)):
        wk_ptr[i] = UInt8(((i * 5 + 1) & 15) | (((i * 9 + 3) & 15) << 4))
        wv_ptr[i] = UInt8(((i * 3 + 7) & 15) | (((i * 13 + 2) & 15) << 4))
    for i in range(kv_dim * num_blocks):
        sk_ptr[i] = Float32(((i * 11 + 3) % 17) - 8) * 0.015625
        sv_ptr[i] = Float32(((i * 7 + 5) % 23) - 11) * 0.015625
    for i in range(c_in * (q_dim // 2)):
        wo_ptr[i] = UInt8(((i * 9 + 1) & 15) | (((i * 5 + 4) & 15) << 4))
    for i in range(c_in * wo_blocks):
        so_ptr[i] = Float32(((i * 17 + 9) % 21) - 10) * 0.015625
    for i in range(c_in):
        x_ptr[i] = Float32(((i * 19 + 3) % 27) - 13) * 0.03125

    ctx.flush_cache_range(wq_buf.cpu_addr, q_dim * (c_in // 2))
    ctx.flush_cache_range(sq_buf.cpu_addr, q_dim * num_blocks * 4)
    ctx.flush_cache_range(wk_buf.cpu_addr, kv_dim * (c_in // 2))
    ctx.flush_cache_range(sk_buf.cpu_addr, kv_dim * num_blocks * 4)
    ctx.flush_cache_range(wv_buf.cpu_addr, kv_dim * (c_in // 2))
    ctx.flush_cache_range(sv_buf.cpu_addr, kv_dim * num_blocks * 4)
    ctx.flush_cache_range(wo_buf.cpu_addr, c_in * (q_dim // 2))
    ctx.flush_cache_range(so_buf.cpu_addr, c_in * wo_blocks * 4)
    ctx.flush_cache_range(x_buf.cpu_addr, c_in * 4)

    # 1. All-iGPU 3-packet QKV dispatch (batch=1)
    var t0 = std.time.perf_counter()
    var sq0 = ctx.enqueue_kernel_matmul(
        q4_kernel, wq_buf.va_addr, sq_buf.va_addr, x_buf.va_addr, q_buf.va_addr,
        c_in, q_dim, batch_size=1, flush_input=True, barrier=False, ring_doorbell_now=False,
    )
    var sk0 = ctx.enqueue_kernel_matmul(
        q4_kernel, wk_buf.va_addr, sk_buf.va_addr, x_buf.va_addr, k_buf.va_addr,
        c_in, kv_dim, batch_size=1, flush_input=False, barrier=False, ring_doorbell_now=False,
    )
    var sv0 = ctx.enqueue_kernel_matmul(
        q4_kernel, wv_buf.va_addr, sv_buf.va_addr, x_buf.va_addr, v_buf.va_addr,
        c_in, kv_dim, batch_size=1, flush_input=False, barrier=False, ring_doorbell_now=False,
    )
    ctx.ring_doorbell()
    ctx.wait_matmul(sq0, q_buf.cpu_addr, q_dim, batch_size=1)
    ctx.wait_matmul(sk0, k_buf.cpu_addr, kv_dim, batch_size=1)
    ctx.wait_matmul(sv0, v_buf.cpu_addr, kv_dim, batch_size=1)
    var t_igpu_qkv_ms = (std.time.perf_counter() - t0) * 1000.0

    # 2. Wo (8192 -> 2816, batch=1) on iGPU
    var t1 = std.time.perf_counter()
    var so0 = ctx.enqueue_kernel_matmul(
        q4_kernel, wo_buf.va_addr, so_buf.va_addr, q_buf.va_addr, o_buf.va_addr,
        q_dim, c_in, batch_size=1, flush_input=True, barrier=False, ring_doorbell_now=True,
    )
    ctx.wait_matmul(so0, o_buf.cpu_addr, c_in, batch_size=1)
    var t_igpu_wo_ms = (std.time.perf_counter() - t1) * 1000.0

    # Verify accuracy against CPU reference
    var ref_q = alloc[Float32](q_dim)
    var ref_k = alloc[Float32](kv_dim)
    var ref_v = alloc[Float32](kv_dim)
    var ref_o = alloc[Float32](c_in)
    cpu_q4_matmul_ref(wq_ptr, sq_ptr, x_ptr, ref_q, c_in, q_dim, 1, q_dim)
    cpu_q4_matmul_ref(wk_ptr, sk_ptr, x_ptr, ref_k, c_in, kv_dim, 1, kv_dim)
    cpu_q4_matmul_ref(wv_ptr, sv_ptr, x_ptr, ref_v, c_in, kv_dim, 1, kv_dim)
    cpu_q4_matmul_ref(wo_ptr, so_ptr, q_ptr, ref_o, q_dim, c_in, 1, c_in)

    var max_err: Float32 = 0.0
    for i in range(kv_dim):
        var ek = abs(k_ptr[i] - ref_k[i])
        var ev = abs(v_ptr[i] - ref_v[i])
        if ek > max_err:
            max_err = ek
        if ev > max_err:
            max_err = ev
    for i in range(c_in):
        var err = abs(o_ptr[i] - ref_o[i])
        if err > max_err:
            max_err = err

    # 3. Test CDC-6600 Co-Processor Split on Wo (960 rows on iGPU || 1856 rows on CPU)
    var gpu_wo_rows = 960
    var cpu_wo_rows = c_in - gpu_wo_rows
    for i in range(c_in):
        o_ptr[i] = -999.0
    var t2 = std.time.perf_counter()
    var slot_wo_split = ctx.enqueue_kernel_matmul(
        q4_kernel, wo_buf.va_addr, so_buf.va_addr, q_buf.va_addr, o_buf.va_addr,
        q_dim, gpu_wo_rows, batch_size=1, flush_input=True, barrier=False, ring_doorbell_now=True,
    )
    # CPU concurrently computes rows [960..2816) on disjoint 64-byte cache lines
    cpu_q4_matmul_ref(
        wo_ptr + gpu_wo_rows * (q_dim // 2),
        so_ptr + gpu_wo_rows * wo_blocks,
        q_ptr,
        o_ptr + gpu_wo_rows,
        q_dim,
        cpu_wo_rows,
        1,
        cpu_wo_rows,
    )
    ctx.wait_matmul(slot_wo_split, o_buf.cpu_addr, gpu_wo_rows, batch_size=1)
    var t_split_wo_ms = (std.time.perf_counter() - t2) * 1000.0

    var max_split_err: Float32 = 0.0
    for i in range(c_in):
        var err = abs(o_ptr[i] - ref_o[i])
        if err > max_split_err:
            max_split_err = err

    ref_q.free()
    ref_k.free()
    ref_v.free()
    ref_o.free()

    print(
        "  [Gemma-4 Layer Shapes] iGPU QKV:",
        t_igpu_qkv_ms,
        "ms | iGPU Wo:",
        t_igpu_wo_ms,
        "ms | Split CPU+iGPU Wo:",
        t_split_wo_ms,
        "ms | max_err =",
        max_err,
        "| max_split_err =",
        max_split_err,
    )
    if max_err > 1e-4 or max_split_err > 1e-4:
        raise Error("Wo mismatch on iGPU: " + String(max_err) + " / " + String(max_split_err))

    ctx.free_gpu_buffer(o_buf)
    ctx.free_gpu_buffer(v_buf)
    ctx.free_gpu_buffer(k_buf)
    ctx.free_gpu_buffer(q_buf)
    ctx.free_gpu_buffer(x_buf)
    ctx.free_gpu_buffer(so_buf)
    ctx.free_gpu_buffer(wo_buf)
    ctx.free_gpu_buffer(sv_buf)
    ctx.free_gpu_buffer(wv_buf)
    ctx.free_gpu_buffer(sk_buf)
    ctx.free_gpu_buffer(wk_buf)
    ctx.free_gpu_buffer(sq_buf)
    ctx.free_gpu_buffer(wq_buf)
    print("  [PASS] Gemma-4 decode layer shapes verified on KFD iGPU!")


def main() raises:
    var ctx = KFDContext(init_queue=True)
    test_dynamic_elf_loader(ctx)
    test_q4_matmul_kernel(ctx)
    test_multi_packet_batching_and_stride(ctx)
    test_lsa_f32_pca_and_gram(ctx)
    test_cdc6600_asymmetric_overlap(ctx)
    print("\nALL KFD GPU RUNTIME, Q4 SHADER & LSA F32 SHADER TESTS PASSED!")
