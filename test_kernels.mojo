from std.memory import alloc
from compress import Float32BlockCompressor
from std.math import sqrt, exp, cos, sin
from std.testing import assert_equal, assert_true, assert_false
from std.ffi import OwnedDLHandle
from std.python import Python
from std.algorithm import parallelize
from std.sys.info import num_physical_cores

# -----------------------------------------------------------------------------
# Kernels to Test (exactly matching model.mojo)
# -----------------------------------------------------------------------------
@always_inline
def rms_norm(x: UnsafePointer[Float32, _], w: UnsafePointer[Float32, _], out_ptr: UnsafePointer[Float32, MutExternalOrigin], size: Int):
    var acc = SIMD[DType.float32, 8](0.0)
    for i in range(0, size, 8):
        var v = x.load[8](i)
        acc = acc + v * v
    var sum_sq = acc.reduce_add()
    var inv_rms = 1.0 / sqrt(sum_sq / Float32(size) + 1e-6)
    for i in range(0, size, 8):
        var v = x.load[8](i)
        var weight = w.load[8](i)
        out_ptr.store[8](i, v * inv_rms * weight)

@always_inline
def rms_norm_noscale(x: UnsafePointer[Float32, _], out_ptr: UnsafePointer[Float32, MutExternalOrigin], size: Int):
    var acc = SIMD[DType.float32, 8](0.0)
    for i in range(0, size, 8):
        var v = x.load[8](i)
        acc = acc + v * v
    var sum_sq = acc.reduce_add()
    var inv_rms = 1.0 / sqrt(sum_sq / Float32(size) + 1e-6)
    for i in range(0, size, 8):
        var v = x.load[8](i)
        out_ptr.store[8](i, v * inv_rms)

@always_inline
def fast_tanh(x: Float32) -> Float32:
    if x > 15.0:
        return 1.0
    elif x < -15.0:
        return -1.0
    var e2x = exp(2.0 * x)
    return (e2x - 1.0) / (e2x + 1.0)

@always_inline
def gelu_and_mul(h1: UnsafePointer[Float32, _], h2: UnsafePointer[Float32, _], out_ptr: UnsafePointer[Float32, MutExternalOrigin], size: Int):
    for i in range(0, size, 8):
        var v1 = h1.load[8](i)
        var v2 = h2.load[8](i)
        for lane in range(8):
            var x1 = v1[lane]
            var x2 = v2[lane]
            var tanh_arg = 0.79788456 * x1 * (1.0 + 0.044715 * x1 * x1)
            var gelu = 0.5 * x1 * (1.0 + fast_tanh(tanh_arg))
            out_ptr[i + lane] = gelu * x2

@always_inline
def apply_rope(vec: UnsafePointer[Float32, MutExternalOrigin], num_heads: Int, head_dim: Int, pos: Int, is_global: Bool):
    var half_dim = head_dim // 2
    var base: Float32 = Float32(1000000.0) if is_global else Float32(10000.0)
    var rope_dim = 512 if is_global else 256
    
    for h in range(num_heads):
        var h_offset = h * head_dim
        for i in range(half_dim):
            var theta: Float32 = 0.0
            if (not is_global) or (i < 64):
                theta = Float32(pos) / (base ** (Float32(2 * i) / Float32(rope_dim)))
                
            var cos_theta = cos(theta)
            var sin_theta = sin(theta)
            
            var val1 = vec[h_offset + i]
            var val2 = vec[h_offset + half_dim + i]
            
            vec[h_offset + i] = val1 * cos_theta - val2 * sin_theta
            vec[h_offset + half_dim + i] = val2 * cos_theta + val1 * sin_theta

@always_inline
def softmax(scores: UnsafePointer[Float32, MutExternalOrigin], size: Int):
    var max_val = scores[0]
    for i in range(1, size):
        if scores[i] > max_val:
            max_val = scores[i]
            
    var sum_exp: Float32 = 0.0
    for i in range(size):
        scores[i] = exp(scores[i] - max_val)
        sum_exp += scores[i]
        
    for i in range(size):
        scores[i] /= sum_exp

@always_inline
def matmul_vec(W: UnsafePointer[Float32, _], x: UnsafePointer[Float32, _], out_ptr: UnsafePointer[Float32, MutExternalOrigin], c_in: Int, c_out: Int):
    var num_tasks = num_physical_cores()
    var chunk_size = c_out // num_tasks
    
    def worker(t: Int) capturing -> None:
        var start_r = t * chunk_size
        var end_r = (t + 1) * chunk_size if t < num_tasks - 1 else c_out
        
        for r in range(start_r, end_r):
            var acc0 = SIMD[DType.float32, 8](0.0)
            var acc1 = SIMD[DType.float32, 8](0.0)
            var acc2 = SIMD[DType.float32, 8](0.0)
            var acc3 = SIMD[DType.float32, 8](0.0)
            var r_offset = r * c_in
            for c in range(0, c_in, 32):
                acc0 = acc0 + W.load[8](r_offset + c)      * x.load[8](c)
                acc1 = acc1 + W.load[8](r_offset + c + 8)  * x.load[8](c + 8)
                acc2 = acc2 + W.load[8](r_offset + c + 16) * x.load[8](c + 16)
                acc3 = acc3 + W.load[8](r_offset + c + 24) * x.load[8](c + 24)
            out_ptr[r] = (acc0 + acc1 + acc2 + acc3).reduce_add()
            
    parallelize[worker](num_tasks)

@always_inline
def unpack_2bytes(b0: UInt8, b1: UInt8) -> SIMD[DType.float32, 8]:
    return SIMD[DType.float32, 8](
        Float32(Int(b0 & 3) - 1),
        Float32(Int((b0 >> 2) & 3) - 1),
        Float32(Int((b0 >> 4) & 3) - 1),
        Float32(Int((b0 >> 6) & 3) - 1),
        Float32(Int(b1 & 3) - 1),
        Float32(Int((b1 >> 2) & 3) - 1),
        Float32(Int((b1 >> 4) & 3) - 1),
        Float32(Int((b1 >> 6) & 3) - 1)
    )

@always_inline
def ternary_matmul_vec(
    W_packed: UnsafePointer[UInt8, _],
    scales: UnsafePointer[Float32, _],
    x: UnsafePointer[Float32, _],
    out_ptr: UnsafePointer[Float32, MutExternalOrigin],
    c_in: Int,
    c_out: Int,
):
    var num_tasks = num_physical_cores()
    var chunk_size = c_out // num_tasks
    var c_in_packed = c_in // 4
    
    def worker(t: Int) capturing -> None:
        var start_r = t * chunk_size
        var end_r = (t + 1) * chunk_size if t < num_tasks - 1 else c_out
        
        for r in range(start_r, end_r):
            var acc0 = SIMD[DType.float32, 8](0.0)
            var acc1 = SIMD[DType.float32, 8](0.0)
            var acc2 = SIMD[DType.float32, 8](0.0)
            var acc3 = SIMD[DType.float32, 8](0.0)
            
            var r_offset_packed = r * c_in_packed
            for c in range(0, c_in_packed, 8):
                var p = W_packed.load[8](r_offset_packed + c)
                
                var w0 = unpack_2bytes(p[0], p[1])
                var w1 = unpack_2bytes(p[2], p[3])
                var w2 = unpack_2bytes(p[4], p[5])
                var w3 = unpack_2bytes(p[6], p[7])
                
                var x_offset = c * 4
                acc0 = acc0 + w0 * x.load[8](x_offset)
                acc1 = acc1 + w1 * x.load[8](x_offset + 8)
                acc2 = acc2 + w2 * x.load[8](x_offset + 16)
                acc3 = acc3 + w3 * x.load[8](x_offset + 24)
                
            var row_sum = (acc0 + acc1 + acc2 + acc3).reduce_add()
            out_ptr[r] = row_sum * scales[r]
            
    parallelize[worker](num_tasks)

@always_inline
def q8_matmul_w2_block_column(
    W_packed: UnsafePointer[Int8, _],
    x: UnsafePointer[Float32, _],
    out_ptr: UnsafePointer[Float32, MutExternalOrigin],
    c_in: Int,
    c_out: Int,
    num_tasks: Int,
    c_memcpy: fn(Int, Int, Int) -> Int,
    w2_scratch: UnsafePointer[Int8, MutExternalOrigin],
    threshold: Float32 = 1e-4,
):
    var c_in_blocks = c_in // 32
    
    var active_blocks = List[Int]()
    for b in range(c_in_blocks):
        var active = False
        var x_offset = b * 32
        for i in range(32):
            if abs(x[x_offset + i]) > threshold:
                active = True
                break
        if active:
            active_blocks.append(b)
            
    var num_active = len(active_blocks)
    if num_active == 0:
        for r in range(c_out):
            out_ptr[r] = 0.0
        return
        
    var chunk_size = c_out // num_tasks
    
    # Copy active blocks from uncached GTT memory to CPU cached memory
    for idx in range(num_active):
        var b = active_blocks[idx]
        _ = c_memcpy((w2_scratch + idx * 139264).__int__(), (W_packed + b * 139264).__int__(), 139264)
    
    def sparse_worker(t: Int) capturing -> None:
        var start_r = t * chunk_size
        var end_r = (t + 1) * chunk_size if t < num_tasks - 1 else c_out
        
        for r in range(start_r, end_r):
            var acc = SIMD[DType.float32, 8](0.0)
            
            for idx in range(num_active):
                var block_base = w2_scratch + idx * 139264
                var block_scales = (block_base + 122880).bitcast[Float32]()
                
                var scale = block_scales[r]
                var block_p = block_base + r * 32
                
                var b = active_blocks[idx]
                var x_offset = b * 32
                
                var w0 = block_p.load[8]().cast[DType.float32]()
                var w1 = block_p.load[8](8).cast[DType.float32]()
                var w2 = block_p.load[8](16).cast[DType.float32]()
                var w3 = block_p.load[8](24).cast[DType.float32]()
                
                var dot = w0 * x.load[8](x_offset) + w1 * x.load[8](x_offset + 8) + w2 * x.load[8](x_offset + 16) + w3 * x.load[8](x_offset + 24)
                acc = acc + dot * scale
                
            out_ptr[r] = acc.reduce_add()
            
    parallelize[sparse_worker](num_tasks)

# -----------------------------------------------------------------------------
# Unit Test Suite
# -----------------------------------------------------------------------------
def test_rms_norm() raises:
    print("Running test_rms_norm...")
    var np = Python.import_module("numpy")
    var size = 128
    
    # 1. Setup random inputs
    var x_np = np.random.randn(size).astype(np.float32)
    var w_np = np.random.randn(size).astype(np.float32) * 0.1
    
    var x_ptr = alloc[Float32](size)
    var w_ptr = alloc[Float32](size)
    var out_ptr = alloc[Float32](size)
    
    for i in range(size):
        x_ptr[i] = Float32(Float64(py=x_np[i]))
        w_ptr[i] = Float32(Float64(py=w_np[i]))
        
    # 2. Run Mojo kernel
    rms_norm(x_ptr, w_ptr, out_ptr, size)
    
    # 3. Run NumPy reference
    var variance = np.mean(x_np ** 2)
    var inv_rms = 1.0 / np.sqrt(variance + 1e-6)
    var expected_np = x_np * inv_rms * w_np
    
    # 4. Compare outputs
    for i in range(size):
        var expected = Float32(Float64(py=expected_np[i]))
        var actual = out_ptr[i]
        var diff = abs(expected - actual)
        if diff > 1e-5:
            print("Mismatch at index", i, "Expected:", expected, "Actual:", actual)
            raise Error("RMSNorm output mismatch")
            
    x_ptr.free()
    w_ptr.free()
    out_ptr.free()
    print("test_rms_norm passed!")

def test_rms_norm_noscale() raises:
    print("Running test_rms_norm_noscale...")
    var np = Python.import_module("numpy")
    var size = 128
    
    var x_np = np.random.randn(size).astype(np.float32)
    var x_ptr = alloc[Float32](size)
    var out_ptr = alloc[Float32](size)
    
    for i in range(size):
        x_ptr[i] = Float32(Float64(py=x_np[i]))
        
    rms_norm_noscale(x_ptr, out_ptr, size)
    
    var variance = np.mean(x_np ** 2)
    var inv_rms = 1.0 / np.sqrt(variance + 1e-6)
    var expected_np = x_np * inv_rms
    
    for i in range(size):
        var expected = Float32(Float64(py=expected_np[i]))
        var actual = out_ptr[i]
        var diff = abs(expected - actual)
        if diff > 1e-5:
            print("Mismatch at index", i, "Expected:", expected, "Actual:", actual)
            raise Error("RMSNorm noscale output mismatch")
            
    x_ptr.free()
    out_ptr.free()
    print("test_rms_norm_noscale passed!")

def test_apply_rope_sliding() raises:
    print("Running test_apply_rope_sliding...")
    var np = Python.import_module("numpy")
    var num_heads = 4
    var head_dim = 256
    var total_size = num_heads * head_dim
    var pos = 3
    
    var vec_np = np.random.randn(total_size).astype(np.float32)
    var vec_ptr = alloc[Float32](total_size)
    
    for i in range(total_size):
        vec_ptr[i] = Float32(Float64(py=vec_np[i]))
        
    # Run Mojo
    apply_rope(vec_ptr, num_heads, head_dim, pos, False)
    
    # Run Reference Python split-half RoPE
    var expected_np = vec_np.copy().reshape(num_heads, head_dim)
    var half_dim = head_dim // 2
    var base = 10000.0
    var rope_dim = 256
    
    for h in range(num_heads):
        for i in range(half_dim):
            var theta = Float64(pos) / (base ** (2.0 * Float64(i) / Float64(rope_dim)))
            var cos_t = Float64(py=np.cos(theta))
            var sin_t = Float64(py=np.sin(theta))
            
            var val1 = Float64(py=expected_np[h, i])
            var val2 = Float64(py=expected_np[h, half_dim + i])
            
            expected_np[h, i] = val1 * cos_t - val2 * sin_t
            expected_np[h, half_dim + i] = val2 * cos_t + val1 * sin_t
            
    expected_flat = expected_np.flatten()
    for i in range(total_size):
        var expected = Float32(Float64(py=expected_flat[i]))
        var actual = vec_ptr[i]
        var diff = abs(expected - actual)
        if diff > 1e-5:
            print("Mismatch at index", i, "Expected:", expected, "Actual:", actual)
            raise Error("RoPE sliding mismatch")
            
    vec_ptr.free()
    print("test_apply_rope_sliding passed!")

def test_apply_rope_global() raises:
    print("Running test_apply_rope_global...")
    var np = Python.import_module("numpy")
    var num_heads = 2
    var head_dim = 512
    var total_size = num_heads * head_dim
    var pos = 5
    
    var vec_np = np.random.randn(total_size).astype(np.float32)
    var vec_ptr = alloc[Float32](total_size)
    
    for i in range(total_size):
        vec_ptr[i] = Float32(Float64(py=vec_np[i]))
        
    apply_rope(vec_ptr, num_heads, head_dim, pos, True)
    
    # Run Reference Python split-half RoPE with proportional frequency
    var expected_np = vec_np.copy().reshape(num_heads, head_dim)
    var half_dim = head_dim // 2
    var base = 1000000.0
    var rope_dim = 512
    
    for h in range(num_heads):
        for i in range(half_dim):
            var theta = 0.0
            if i < 64:
                theta = Float64(pos) / (base ** (2.0 * Float64(i) / Float64(rope_dim)))
            var cos_t = Float64(py=np.cos(theta))
            var sin_t = Float64(py=np.sin(theta))
            
            var val1 = Float64(py=expected_np[h, i])
            var val2 = Float64(py=expected_np[h, half_dim + i])
            
            expected_np[h, i] = val1 * cos_t - val2 * sin_t
            expected_np[h, half_dim + i] = val2 * cos_t + val1 * sin_t
            
    expected_flat = expected_np.flatten()
    for i in range(total_size):
        var expected = Float32(Float64(py=expected_flat[i]))
        var actual = vec_ptr[i]
        var diff = abs(expected - actual)
        if diff > 1e-5:
            print("Mismatch at index", i, "Expected:", expected, "Actual:", actual)
            raise Error("RoPE global mismatch")
            
    vec_ptr.free()
    print("test_apply_rope_global passed!")

def test_softmax() raises:
    print("Running test_softmax...")
    var np = Python.import_module("numpy")
    var size = 16
    
    var x_np = np.random.randn(size).astype(np.float32)
    var x_ptr = alloc[Float32](size)
    
    for i in range(size):
        x_ptr[i] = Float32(Float64(py=x_np[i]))
        
    softmax(x_ptr, size)
    
    # NumPy reference softmax
    var max_val = np.max(x_np)
    var exp_vals = np.exp(x_np - max_val)
    var expected_np = exp_vals / np.sum(exp_vals)
    
    for i in range(size):
        var expected = Float32(Float64(py=expected_np[i]))
        var actual = x_ptr[i]
        var diff = abs(expected - actual)
        if diff > 1e-5:
            print("Mismatch at index", i, "Expected:", expected, "Actual:", actual)
            raise Error("Softmax output mismatch")
            
    x_ptr.free()
    print("test_softmax passed!")

def test_gelu_and_mul() raises:
    print("Running test_gelu_and_mul...")
    var np = Python.import_module("numpy")
    var size = 128
    
    var h1_np = np.random.randn(size).astype(np.float32)
    var h2_np = np.random.randn(size).astype(np.float32)
    
    var h1_ptr = alloc[Float32](size)
    var h2_ptr = alloc[Float32](size)
    var out_ptr = alloc[Float32](size)
    
    for i in range(size):
        h1_ptr[i] = Float32(Float64(py=h1_np[i]))
        h2_ptr[i] = Float32(Float64(py=h2_np[i]))
        
    gelu_and_mul(h1_ptr, h2_ptr, out_ptr, size)
    
    # NumPy reference for gelu_pytorch_tanh
    var sqrt_2_pi = np.sqrt(2.0 / np.pi)
    var tanh_arg = sqrt_2_pi * h1_np * (1.0 + 0.044715 * h1_np ** 2)
    var gelu_np = 0.5 * h1_np * (1.0 + np.tanh(tanh_arg))
    var expected_np = gelu_np * h2_np
    
    for i in range(size):
        var expected = Float32(Float64(py=expected_np[i]))
        var actual = out_ptr[i]
        var diff = abs(expected - actual)
        if diff > 1e-5:
            print("Mismatch at index", i, "Expected:", expected, "Actual:", actual)
            raise Error("GELU and Mul output mismatch")
            
    h1_ptr.free()
    h2_ptr.free()
    out_ptr.free()
    print("test_gelu_and_mul passed!")

def test_logit_softcapping() raises:
    print("Running test_logit_softcapping...")
    var np = Python.import_module("numpy")
    var cap: Float32 = 30.0
    
    # Test values across a wide range (including extreme values to test tanh saturation)
    var py_list = Python.evaluate("[-100.0, -50.0, -30.0, -10.0, 0.0, 10.0, 30.0, 50.0, 100.0]")
    var test_vals_np = np.array(py_list, dtype=np.float32)
    
    for i in range(9):
        var val = Float32(Float64(py=test_vals_np[i]))
        var actual = cap * fast_tanh(val / cap)
        
        # NumPy reference: cap * np.tanh(val / cap)
        var expected = Float32(Float64(py=cap * np.float32(np.tanh(Float64(val / cap)))))
        var diff = abs(expected - actual)
        if diff > 1e-4:
            print("Mismatch for value", val, "Expected:", expected, "Actual:", actual)
            raise Error("Logit soft-capping output mismatch")
            
    print("test_logit_softcapping passed!")

def test_block_compression() raises:
    print("Running test_block_compression...")
    var length = 16
    var src_ptr = alloc[Float32](length)
    var decomp_ptr = alloc[Float32](length)
    var comp_ptr = alloc[UInt8](length * 5) # generous buffer
    
    # --- Case 1: All values compressible (small exponent differences) ---
    # We populate with standard floats. Their exponents are close.
    src_ptr[0] = 1.0
    src_ptr[1] = 1.5
    src_ptr[2] = 2.0
    src_ptr[3] = 0.5
    src_ptr[4] = 0.75
    src_ptr[5] = 1.25
    src_ptr[6] = 1.875
    src_ptr[7] = 2.5
    src_ptr[8] = -1.0
    src_ptr[9] = -1.5
    src_ptr[10] = -2.0
    src_ptr[11] = -0.5
    src_ptr[12] = -0.75
    src_ptr[13] = -1.25
    src_ptr[14] = -1.875
    src_ptr[15] = -2.5
    
    # Compress
    var comp_bytes_1 = Float32BlockCompressor.compress(src_ptr, comp_ptr, length)
    # Since length is 16 and all blocks are compressed, size should be 2 * 30 = 60 bytes
    assert_equal(comp_bytes_1, 60)
    
    # Decompress
    Float32BlockCompressor.decompress(comp_ptr, decomp_ptr, length)
    
    # Verify exact losslessness
    for i in range(length):
        assert_equal(src_ptr[i], decomp_ptr[i])
        
    # --- Case 2: Incompressible block (forces uncompressed fallback) ---
    # Elements in the second block (idx 8 to 15) have huge exponent difference
    src_ptr[8] = 1.0
    src_ptr[9] = 1e30 # Huge exponent difference relative to src_ptr[8] (base_exponent)
    
    # Zero out decomp_ptr
    for i in range(length):
        decomp_ptr[i] = 0.0
        
    var comp_bytes_2 = Float32BlockCompressor.compress(src_ptr, comp_ptr, length)
    # Block 1: compressed (30 bytes), Block 2: uncompressed (33 bytes) -> Total 63 bytes
    assert_equal(comp_bytes_2, 63)
    
    # Decompress
    Float32BlockCompressor.decompress(comp_ptr, decomp_ptr, length)
    
    # Verify exact losslessness
    for i in range(length):
        assert_equal(src_ptr[i], decomp_ptr[i])
        
    src_ptr.free()
    decomp_ptr.free()
    comp_ptr.free()
    print("test_block_compression passed!")

def test_matmul_vec() raises:
    print("Running test_matmul_vec...")
    var np = Python.import_module("numpy")
    var c_in = 3840
    var c_out = 128
    
    # 1. Setup random inputs (c_in must be a multiple of 32 for our unrolled loop)
    var w_np = np.random.randn(c_out, c_in).astype(np.float32) * 0.01
    var x_np = np.random.randn(c_in).astype(np.float32)
    
    var w_ptr = alloc[Float32](c_out * c_in)
    var x_ptr = alloc[Float32](c_in)
    var out_ptr = alloc[Float32](c_out)
    
    for r in range(c_out):
        for c in range(c_in):
            w_ptr[r * c_in + c] = Float32(Float64(py=w_np[r, c]))
            
    for c in range(c_in):
        x_ptr[c] = Float32(Float64(py=x_np[c]))
        
    # 2. Run Mojo kernel
    matmul_vec(w_ptr, x_ptr, out_ptr, c_in, c_out)
    
    # 3. Run NumPy reference
    var expected_np = np.dot(w_np, x_np)
    
    # 4. Compare outputs
    for r in range(c_out):
        var expected = Float32(Float64(py=expected_np[r]))
        var actual = out_ptr[r]
        var diff = abs(expected - actual)
        if diff > 1e-4:
            print("Mismatch at row", r, "Expected:", expected, "Actual:", actual)
            raise Error("matmul_vec output mismatch")
            
    w_ptr.free()
    x_ptr.free()
    out_ptr.free()
    print("test_matmul_vec passed!")

def test_ternary_matmul_vec() raises:
    print("Running test_ternary_matmul_vec...")
    var np = Python.import_module("numpy")
    var c_in = 3840
    var c_out = 128
    var c_in_packed = c_in // 4
    
    # 1. Setup random inputs in Float32
    var w_np = np.random.randn(c_out, c_in).astype(np.float32) * 0.05
    var x_np = np.random.randn(c_in).astype(np.float32)
    
    # 2. Do row-wise (absmean) quantization and packing in Python (vectorized)
    var scales_np = np.mean(np.abs(w_np), axis=1).reshape(c_out, 1)
    scales_np = np.maximum(scales_np, 1e-5)
    
    var w_quant_np = np.round(w_np / scales_np)
    w_quant_np = np.clip(w_quant_np, -1.0, 1.0).astype(np.int32)
    
    # Map {-1, 0, 1} to {0, 1, 2}
    var w_mapped_np = w_quant_np + 1
    
    # Pack 4 ternary weights per byte using .take(index, axis=2) and arithmetic to avoid slice parser issues
    var w_reshaped_np = w_mapped_np.reshape(c_out, c_in_packed, 4)
    var w0 = w_reshaped_np.take(0, axis=2)
    var w1 = w_reshaped_np.take(1, axis=2)
    var w2 = w_reshaped_np.take(2, axis=2)
    var w3 = w_reshaped_np.take(3, axis=2)
    var packed_np = (w0 + w1 * 4 + w2 * 16 + w3 * 64).astype(np.uint8)
    
    # 3. Reference quantized Float32 matrix (W_quant * scale)
    var w_ref_np = w_quant_np.astype(np.float32) * scales_np
    var expected_np = np.dot(w_ref_np, x_np)
    
    # 4. Allocate Mojo memory & transfer values
    var w_packed_ptr = alloc[UInt8](c_out * c_in_packed)
    var scales_ptr = alloc[Float32](c_out)
    var x_ptr = alloc[Float32](c_in)
    var out_ptr = alloc[Float32](c_out)
    
    for r in range(c_out):
        for c in range(c_in_packed):
            w_packed_ptr[r * c_in_packed + c] = UInt8(Int(py=packed_np[r, c]))
            
    for r in range(c_out):
        scales_ptr[r] = Float32(Float64(py=scales_np[r, 0]))
        
    for c in range(c_in):
        x_ptr[c] = Float32(Float64(py=x_np[c]))
        
    # 5. Run Mojo quantized ternary kernel
    ternary_matmul_vec(w_packed_ptr, scales_ptr, x_ptr, out_ptr, c_in, c_out)
    
    # 6. Verify correct matrix-vector multiplication results
    for r in range(c_out):
        var expected = Float32(Float64(py=expected_np[r]))
        var actual = out_ptr[r]
        var diff = abs(expected - actual)
        if diff > 1e-4:
            print("Mismatch at row", r, "Expected:", expected, "Actual:", actual)
            raise Error("ternary_matmul_vec output mismatch")
            
    w_packed_ptr.free()
    scales_ptr.free()
    x_ptr.free()
    out_ptr.free()
    print("test_ternary_matmul_vec passed!")

def test_attention_no_scaling() raises:
    print("Running test_attention_no_scaling...")
    var np = Python.import_module("numpy")
    var head_dim = 256
    var pos = 7
    
    # 1. Setup random inputs
    var q_np = np.random.randn(head_dim).astype(np.float32)
    var k_np = np.random.randn(pos + 1, head_dim).astype(np.float32)
    
    var q_ptr = alloc[Float32](head_dim)
    var k_ptr = alloc[Float32]((pos + 1) * head_dim)
    var scores_ptr = alloc[Float32](pos + 1)
    
    for d in range(head_dim):
        q_ptr[d] = Float32(Float64(py=q_np[d]))
        
    for t in range(pos + 1):
        for d in range(head_dim):
            k_ptr[t * head_dim + d] = Float32(Float64(py=k_np[t, d]))
            
    # 2. Run the exact Gemma 4 attention dot-product loop (NO scaling)
    for t in range(pos + 1):
        var score: Float32 = 0.0
        for d in range(head_dim):
            score += q_ptr[d] * k_ptr[t * head_dim + d]
        scores_ptr[t] = score
        
    # 3. Compute reference unscaled scores in NumPy
    var expected_np = np.dot(k_np, q_np)
    
    # 4. Compute reference SCALED scores in NumPy (what other models do, e.g. Llama)
    var scale = 1.0 / sqrt(Float64(head_dim))
    var expected_scaled_np = expected_np * scale
    
    # 5. Assert that Mojo matches unscaled scores and does NOT match scaled scores
    for t in range(pos + 1):
        var expected_unscaled = Float32(Float64(py=expected_np[t]))
        var expected_scaled = Float32(Float64(py=expected_scaled_np[t]))
        var actual = scores_ptr[t]
        
        # Verify it matches the unscaled computation
        var diff_unscaled = abs(expected_unscaled - actual)
        if diff_unscaled > 1e-4:
            print("Mismatch at t", t, "Expected unscaled:", expected_unscaled, "Actual:", actual)
            raise Error("Attention unscaled score mismatch")
            
        # Verify it is significantly different from the scaled computation
        var diff_scaled = abs(expected_scaled - actual)
        if diff_scaled < 1e-2:
            print("Warning: actual score is too close to scaled score at t", t, "Actual:", actual, "Scaled:", expected_scaled)
            raise Error("Attention score was scaled! Regression detected.")
            
    q_ptr.free()
    k_ptr.free()
    scores_ptr.free()
    print("test_attention_no_scaling passed!")

def test_q8_matmul_w2_block_column() raises:
    print("Running test_q8_matmul_w2_block_column...")
    var np = Python.import_module("numpy")
    var libc = OwnedDLHandle("libc.so.6")
    var c_memcpy = libc.get_function[fn(Int, Int, Int) -> Int]("memcpy")
    
    var c_in = 3840
    var c_out = 3840
    var c_in_blocks = c_in // 32
    var block_size = 139264
    
    # 1. Setup random inputs in Float32
    var x_np = np.zeros(c_in, dtype=np.float32)
    # Make only 3 blocks active
    var active_indices: List[Int] = [5, 42, 100]
    for idx in range(len(active_indices)):
        var b = active_indices[idx]
        var block_vals = np.random.randn(32).astype(np.float32)
        # Avoid zero by adding 1.0 or similar to guarantee active status
        block_vals = block_vals + np.sign(block_vals) * 0.1
        for i in range(32):
            x_np[b * 32 + i] = block_vals[i]
            
    # Setup random weights and scales for W_packed
    var w_blocks_np = np.random.randint(-128, 127, size=c_in_blocks * c_out * 32).reshape(c_in_blocks, c_out, 32).astype(np.int8)
    var scale_blocks_np = np.random.randn(c_in_blocks, c_out).astype(np.float32) * 0.05
    
    # 2. Compute reference outputs using active blocks in Python
    var expected_np = np.zeros(c_out, dtype=np.float32)
    for idx in range(len(active_indices)):
        var b = active_indices[idx]
        var w_b = w_blocks_np[b].astype(np.float32) # (c_out, 32)
        var s_b = scale_blocks_np[b] # (c_out,)
        var x_b = x_np[b * 32 : (b + 1) * 32] # (32,)
        
        var block_dot = np.dot(w_b, x_b) # (c_out,)
        expected_np = expected_np + block_dot * s_b
        
    # 3. Allocate Mojo memory
    var w_packed_ptr = alloc[Int8](c_in_blocks * block_size)
    var x_ptr = alloc[Float32](c_in)
    var out_ptr = alloc[Float32](c_out)
    var w2_scratch = alloc[Int8](c_in_blocks * block_size)
    
    # 4. Transfer values to Mojo memory
    for c in range(c_in):
        x_ptr[c] = Float32(Float64(py=x_np[c]))
        
    for b in range(c_in_blocks):
        var block_base = w_packed_ptr + b * block_size
        # Copy weights
        var w_flat = w_blocks_np[b].flatten()
        for r_c in range(c_out * 32):
            block_base[r_c] = Int8(Int(py=w_flat[r_c]))
            
        # Copy scales
        var scales_base = (block_base + 122880).bitcast[Float32]()
        var s_flat = scale_blocks_np[b]
        for r in range(c_out):
            scales_base[r] = Float32(Float64(py=s_flat[r]))
            
    # 5. Run Mojo sparse matmul kernel
    var num_tasks = num_physical_cores()
    q8_matmul_w2_block_column(w_packed_ptr, x_ptr, out_ptr, c_in, c_out, num_tasks, c_memcpy, w2_scratch, Float32(1e-4))
    
    # 6. Verify correct matrix-vector multiplication results
    for r in range(c_out):
        var expected = Float32(Float64(py=expected_np[r]))
        var actual = out_ptr[r]
        var diff = abs(expected - actual)
        if diff > 1e-3:
            print("Mismatch at row", r, "Expected:", expected, "Actual:", actual)
            raise Error("q8_matmul_w2_block_column output mismatch")
            
    w_packed_ptr.free()
    x_ptr.free()
    out_ptr.free()
    w2_scratch.free()
    print("test_q8_matmul_w2_block_column passed!")

def main() raises:
    print("=========================================================")
    print("               GEMMA 4 KERNEL TEST SUITE                 ")
    print("=========================================================")
    test_rms_norm()
    test_rms_norm_noscale()
    test_apply_rope_sliding()
    test_apply_rope_global()
    test_softmax()
    test_gelu_and_mul()
    test_logit_softcapping()
    test_block_compression()
    test_matmul_vec()
    test_ternary_matmul_vec()
    test_attention_no_scaling()
    test_q8_matmul_w2_block_column()
    print("=========================================================")
    print("        ALL KERNEL MATH TESTS PASSED SUCCESSFULLY!       ")
    print("=========================================================")
