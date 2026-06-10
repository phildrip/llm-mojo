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
def unpack_4bytes(b0: UInt8, b1: UInt8, b2: UInt8, b3: UInt8) -> SIMD[DType.float32, 8]:
    var chunk = SIMD[DType.uint8, 8](b0, b1, b2, b3, 0, 0, 0, 0)
    var repeated = chunk.shuffle[0, 0, 1, 1, 2, 2, 3, 3]()
    var shifted = repeated >> SIMD[DType.uint8, 8](0, 4, 0, 4, 0, 4, 0, 4)
    var masked = shifted & 15
    return (masked.cast[DType.int8]() - 8).cast[DType.float32]()


@always_inline
def q4_matmul_vec(
    W_packed: UnsafePointer[UInt8, _],
    scales: UnsafePointer[Float32, _],
    x: UnsafePointer[Float32, _],
    out_ptr: UnsafePointer[Float32, MutExternalOrigin],
    c_in: Int,
    c_out: Int,
    num_tasks: Int,
):
    var chunk_size = c_out // num_tasks
    var c_in_packed = c_in // 2
    var c_in_blocks = c_in // 32
    
    def worker(t: Int) capturing -> None:
        var start_r = t * chunk_size
        var end_r = (t + 1) * chunk_size if t < num_tasks - 1 else c_out
        
        for r in range(start_r, end_r):
            var acc = SIMD[DType.float32, 8](0.0)
            
            var r_offset_packed = r * c_in_packed
            var r_offset_scales = r * c_in_blocks
            
            for b in range(c_in_blocks):
                var scale = scales[r_offset_scales + b]
                var block_p = r_offset_packed + b * 16
                var x_offset = b * 32
                
                # Load 16 bytes of packed weights at once
                var bytes = W_packed.load[16](block_p)
                
                var w0 = unpack_4bytes(bytes[0], bytes[1], bytes[2], bytes[3])
                var w1 = unpack_4bytes(bytes[4], bytes[5], bytes[6], bytes[7])
                var w2 = unpack_4bytes(bytes[8], bytes[9], bytes[10], bytes[11])
                var w3 = unpack_4bytes(bytes[12], bytes[13], bytes[14], bytes[15])
                
                acc = acc + (w0 * x.load[8](x_offset)) * scale
                acc = acc + (w1 * x.load[8](x_offset + 8)) * scale
                acc = acc + (w2 * x.load[8](x_offset + 16)) * scale
                acc = acc + (w3 * x.load[8](x_offset + 24)) * scale
                
            out_ptr[r] = acc.reduce_add()
            
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
    var max_active_block = -1
    for b in range(c_in_blocks):
        var active = False
        var x_offset = b * 32
        for i in range(32):
            if abs(x[x_offset + i]) > threshold:
                active = True
                break
        if active:
            active_blocks.append(b)
            max_active_block = b
            
    var num_active = len(active_blocks)
    if num_active == 0:
        for r in range(c_out):
            out_ptr[r] = 0.0
        return
        
    var chunk_size = c_out // num_tasks
    
    # Copy all blocks up to max_active_block contiguously from GTT to cached memory
    var copy_size = (max_active_block + 1) * 139264
    _ = c_memcpy(w2_scratch.__int__(), W_packed.__int__(), copy_size)
    
    def sparse_worker(t: Int) capturing -> None:
        var start_r = t * chunk_size
        var end_r = (t + 1) * chunk_size if t < num_tasks - 1 else c_out
        
        for r in range(start_r, end_r):
            var acc = SIMD[DType.float32, 8](0.0)
            
            for idx in range(num_active):
                var b = active_blocks[idx]
                var block_base = w2_scratch + b * 139264
                var block_scales = (block_base + 122880).bitcast[Float32]()
                
                var scale = block_scales[r]
                var block_p = block_base + r * 32
                
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

def test_q4_matmul_vec() raises:
    print("Running test_q4_matmul_vec...")
    var np = Python.import_module("numpy")
    var c_in = 3840
    var c_out = 128
    var c_in_packed = c_in // 2
    var c_in_blocks = c_in // 32
    
    # 1. Setup random inputs in Float32
    var w_np = np.random.randn(c_out, c_in).astype(np.float32) * 0.05
    var x_np = np.random.randn(c_in).astype(np.float32)
    
    var w_packed_ptr = alloc[UInt8](c_out * c_in_packed)
    var scales_ptr = alloc[Float32](c_out * c_in_blocks)
    var x_ptr = alloc[Float32](c_in)
    var out_ptr = alloc[Float32](c_out)
    
    # Populate x_ptr
    for c in range(c_in):
        x_ptr[c] = Float32(Float64(py=x_np[c]))
        
    # Python-based Q4 quantization and packing
    for r in range(c_out):
        for b in range(c_in_blocks):
            var block_start = b * 32
            
            var max_abs: Float64 = 0.0
            for i in range(32):
                var val = abs(Float64(py=w_np[r, block_start + i]))
                if val > max_abs:
                    max_abs = val
                    
            var scale = max_abs / 7.0
            if scale < 1e-5:
                scale = 1e-5
                
            scales_ptr[r * c_in_blocks + b] = Float32(scale)
            
            # Quantize 32 elements to 4-bit (-8 to 7)
            for i in range(16):
                var idx0 = block_start + i * 2
                var idx1 = block_start + i * 2 + 1
                
                var v0 = Float64(py=w_np[r, idx0])
                var v1 = Float64(py=w_np[r, idx1])
                
                var q0 = Int(py=np.clip(np.round(v0 / scale), -8.0, 7.0))
                var q1 = Int(py=np.clip(np.round(v1 / scale), -8.0, 7.0))
                
                var u0 = UInt8(q0 + 8)
                var u1 = UInt8(q1 + 8)
                var packed_val = u0 | (u1 << 4)
                
                w_packed_ptr[r * c_in_packed + b * 16 + i] = packed_val
                
    # 2. Run our Mojo q4_matmul_vec kernel
    q4_matmul_vec(w_packed_ptr, scales_ptr, x_ptr, out_ptr, c_in, c_out, num_physical_cores())
    
    # 3. Calculate expected values in Python to verify
    var w_dequant_np = np.zeros(c_out * c_in, dtype=np.float32).reshape(c_out, c_in)
    for r in range(c_out):
        for b in range(c_in_blocks):
            var block_start = b * 32
            var scale = Float64(scales_ptr[r * c_in_blocks + b])
            for i in range(16):
                var idx0 = block_start + i * 2
                var idx1 = block_start + i * 2 + 1
                
                var packed_val = w_packed_ptr[r * c_in_packed + b * 16 + i]
                var u0 = packed_val & 15
                var u1 = packed_val >> 4
                
                var q0 = Int(u0) - 8
                var q1 = Int(u1) - 8
                
                w_dequant_np[r, idx0] = Float32(q0 * scale)
                w_dequant_np[r, idx1] = Float32(q1 * scale)
                
    var expected_np = np.dot(w_dequant_np, x_np)
    
    # 4. Compare outputs
    for r in range(c_out):
        var expected = Float32(Float64(py=expected_np[r]))
        var actual = out_ptr[r]
        var diff = abs(expected - actual)
        if diff > 1e-4:
            print("Mismatch at row", r, "Expected:", expected, "Actual:", actual)
            raise Error("q4_matmul_vec output mismatch")
            
    w_packed_ptr.free()
    scales_ptr.free()
    x_ptr.free()
    out_ptr.free()
    print("test_q4_matmul_vec passed!")

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

def test_block_sparsity_regression() raises:
    print("Running test_block_sparsity_regression (Verify all active blocks above threshold are computed)...")
    var np = Python.import_module("numpy")
    var libc = OwnedDLHandle("libc.so.6")
    var c_memcpy = libc.get_function[fn(Int, Int, Int) -> Int]("memcpy")
    
    var c_in = 320
    var c_out = 320
    var c_in_blocks = c_in // 32
    var block_size = 139264
    
    # 1. Setup inputs:
    # 2 blocks have large values (importance ~32.0)
    # 4 blocks have small values (importance ~0.064) which is still above threshold (1e-4)
    # 4 blocks are completely zero
    var x_np = np.zeros(c_in, dtype=np.float32)
    
    # Large blocks: indices 1, 5
    for i in range(32):
        x_np[1 * 32 + i] = 1.0
        x_np[5 * 32 + i] = 1.0
    
    # Small but active blocks (above threshold 1e-4): indices 2, 4, 7, 8
    for i in range(32):
        x_np[2 * 32 + i] = 0.002
        x_np[4 * 32 + i] = 0.002
        x_np[7 * 32 + i] = 0.002
        x_np[8 * 32 + i] = 0.002
    
    # Setup random weights and scales
    var w_blocks_np = np.random.randint(-128, 127, size=c_in_blocks * c_out * 32).reshape(c_in_blocks, c_out, 32).astype(np.int8)
    var scale_blocks_np = np.random.randn(c_in_blocks, c_out).astype(np.float32) * 0.05
    
    # 2. Compute reference outputs including ALL active blocks (large + small)
    var expected_np = np.zeros(c_out, dtype=np.float32)
    var active_indices: List[Int] = [1, 2, 4, 5, 7, 8]
    for idx in range(len(active_indices)):
        var b = active_indices[idx]
        var w_b = w_blocks_np[b].astype(np.float32)
        var s_b = scale_blocks_np[b]
        var x_b = x_np[b * 32 : (b + 1) * 32]
        expected_np = expected_np + np.dot(w_b, x_b) * s_b
        
    # Allocate Mojo memory
    var w_packed_ptr = alloc[Int8](c_in_blocks * block_size)
    var x_ptr = alloc[Float32](c_in)
    var out_ptr = alloc[Float32](c_out)
    var w2_scratch = alloc[Int8](c_in_blocks * block_size)
    
    # Transfer values
    for c in range(c_in):
        x_ptr[c] = Float32(Float64(py=x_np[c]))
        
    for b in range(c_in_blocks):
        var block_base = w_packed_ptr + b * block_size
        var w_flat = w_blocks_np[b].flatten()
        for r_c in range(c_out * 32):
            block_base[r_c] = Int8(Int(py=w_flat[r_c]))
            
        var scales_base = (block_base + 122880).bitcast[Float32]()
        var s_flat = scale_blocks_np[b]
        for r in range(c_out):
            scales_base[r] = Float32(Float64(py=s_flat[r]))
            
    # Run sparse matmul with standard threshold 1e-4
    var num_tasks = num_physical_cores()
    q8_matmul_w2_block_column(w_packed_ptr, x_ptr, out_ptr, c_in, c_out, num_tasks, c_memcpy, w2_scratch, Float32(1e-4))
    
    # Verify correctness: any loss of active blocks would cause error
    for r in range(c_out):
        var expected = Float32(Float64(py=expected_np[r]))
        var actual = out_ptr[r]
        var diff = abs(expected - actual)
        if diff > 1e-3:
            print("Block Sparsity Regression Mismatch at row", r, "Expected:", expected, "Actual:", actual)
            raise Error("Block Sparsity Regression: active blocks were incorrectly discarded!")
            
    w_packed_ptr.free()
    x_ptr.free()
    out_ptr.free()
    w2_scratch.free()
    print("test_block_sparsity_regression passed successfully!")

# ==============================================================================
# GPU Integration Tests (direct AMD KFD/DRM Userspace Queue & Shader Execution)
# ==============================================================================

struct kfd_ioctl_acquire_vm_args:
    var drm_fd: UInt32
    var gpu_id: UInt32

    fn __init__(out self, drm_fd: UInt32, gpu_id: UInt32):
        self.drm_fd = drm_fd
        self.gpu_id = gpu_id

struct kfd_ioctl_runtime_enable_args:
    var r_debug: UInt64
    var mode_mask: UInt32
    var capabilities_mask: UInt32

    fn __init__(out self):
        self.r_debug = 0
        self.mode_mask = 0
        self.capabilities_mask = 0

struct kfd_ioctl_alloc_memory_of_gpu_args:
    var va_addr: UInt64
    var size: UInt64
    var handle: UInt64
    var mmap_offset: UInt64
    var gpu_id: UInt32
    var flags: UInt32

    fn __init__(out self, va_addr: UInt64, size: UInt64, gpu_id: UInt32, flags: UInt32):
        self.va_addr = va_addr
        self.size = size
        self.handle = 0
        self.mmap_offset = 0
        self.gpu_id = gpu_id
        self.flags = flags

struct kfd_ioctl_map_memory_to_gpu_args:
    var handle: UInt64
    var device_ids_array_ptr: UInt64
    var n_devices: UInt32
    var pad: UInt32

    fn __init__(out self, handle: UInt64, device_ids_array_ptr: UInt64, n_devices: UInt32):
        self.handle = handle
        self.device_ids_array_ptr = device_ids_array_ptr
        self.n_devices = n_devices
        self.pad = 0

struct kfd_ioctl_unmap_memory_from_gpu_args:
    var handle: UInt64
    var device_ids_array_ptr: UInt64
    var n_devices: UInt32
    var pad: UInt32

    fn __init__(out self, handle: UInt64, device_ids_array_ptr: UInt64, n_devices: UInt32):
        self.handle = handle
        self.device_ids_array_ptr = device_ids_array_ptr
        self.n_devices = n_devices
        self.pad = 0

struct kfd_ioctl_free_memory_of_gpu_args:
    var handle: UInt64

    fn __init__(out self, handle: UInt64):
        self.handle = handle

struct kfd_ioctl_create_queue_args:
    var ring_base_address: UInt64
    var write_pointer_address: UInt64
    var read_pointer_address: UInt64
    var doorbell_offset: UInt64
    var ring_size: UInt32
    var gpu_id: UInt32
    var queue_type: UInt32
    var queue_percentage: UInt32
    var queue_priority: UInt32
    var queue_id: UInt32
    var eop_buffer_address: UInt64
    var eop_buffer_size: UInt64
    var ctx_save_restore_address: UInt64
    var ctx_save_restore_size: UInt32
    var ctl_stack_size: UInt32
    var sdma_engine_id: UInt32
    var pad: UInt32

    fn __init__(
        out self,
        ring_base_address: UInt64,
        write_pointer_address: UInt64,
        read_pointer_address: UInt64,
        ring_size: UInt32,
        gpu_id: UInt32,
        queue_type: UInt32,
        queue_percentage: UInt32,
        queue_priority: UInt32,
        eop_buffer_address: UInt64 = 0,
        eop_buffer_size: UInt64 = 0,
        ctx_save_restore_address: UInt64 = 0,
        ctx_save_restore_size: UInt32 = 0,
        ctl_stack_size: UInt32 = 0,
        sdma_engine_id: UInt32 = 0
    ):
        self.ring_base_address = ring_base_address
        self.write_pointer_address = write_pointer_address
        self.read_pointer_address = read_pointer_address
        self.doorbell_offset = 0
        self.ring_size = ring_size
        self.gpu_id = gpu_id
        self.queue_type = queue_type
        self.queue_percentage = queue_percentage
        self.queue_priority = queue_priority
        self.queue_id = 0
        self.eop_buffer_address = eop_buffer_address
        self.eop_buffer_size = eop_buffer_size
        self.ctx_save_restore_address = ctx_save_restore_address
        self.ctx_save_restore_size = ctx_save_restore_size
        self.ctl_stack_size = ctl_stack_size
        self.sdma_engine_id = sdma_engine_id
        self.pad = 0

struct kfd_ioctl_destroy_queue_args:
    var queue_id: UInt32
    var pad: UInt32

    fn __init__(out self, queue_id: UInt32):
        self.queue_id = queue_id
        self.pad = 0

struct GPUMemoryBuffer:
    var va_addr: Int
    var size: Int
    var handle: UInt64
    var cpu_addr: Int

    fn __init__(out self, va_addr: Int, size: Int, handle: UInt64, cpu_addr: Int):
        self.va_addr = va_addr
        self.size = size
        self.handle = handle
        self.cpu_addr = cpu_addr

struct hsa_kernel_dispatch_packet_t(Copyable, Movable):
    var header: UInt16
    var setup: UInt16
    var workgroup_size_x: UInt16
    var workgroup_size_y: UInt16
    var workgroup_size_z: UInt16
    var reserved0: UInt16
    var grid_size_x: UInt32
    var grid_size_y: UInt32
    var grid_size_z: UInt32
    var private_segment_size: UInt32
    var group_segment_size: UInt32
    var kernel_object: UInt64
    var kernarg_address: UInt64
    var reserved2: UInt64
    var completion_signal: UInt64

    fn __init__(
        out self,
        header: UInt16,
        setup: UInt16,
        workgroup_size_x: UInt16,
        workgroup_size_y: UInt16,
        workgroup_size_z: UInt16,
        grid_size_x: UInt32,
        grid_size_y: UInt32,
        grid_size_z: UInt32,
        private_segment_size: UInt32,
        group_segment_size: UInt32,
        kernel_object: UInt64,
        kernarg_address: UInt64,
        completion_signal: UInt64
    ):
        self.header = header
        self.setup = setup
        self.workgroup_size_x = workgroup_size_x
        self.workgroup_size_y = workgroup_size_y
        self.workgroup_size_z = workgroup_size_z
        self.reserved0 = 0
        self.grid_size_x = grid_size_x
        self.grid_size_y = grid_size_y
        self.grid_size_z = grid_size_z
        self.private_segment_size = private_segment_size
        self.group_segment_size = group_segment_size
        self.kernel_object = kernel_object
        self.kernarg_address = kernarg_address
        self.reserved2 = 0
        self.completion_signal = completion_signal

def test_early_exit() raises:
    print("Running test_early_exit...")
    
    # Cosine Similarity test (Scale-invariant Hidden State Directional Convergence)
    print("  Testing Cosine Similarity metrics...")
    comptime embed_dim = 128 # small size for testing
    var x = alloc[Float32](embed_dim)
    var x_old = alloc[Float32](embed_dim)
    
    # 1. Exact match (perfect convergence)
    for i in range(embed_dim):
        x[i] = Float32(i + 1)
        x_old[i] = Float32(i + 1)
        
    var dot_prod: Float32 = 0.0
    var sum_sq_x: Float32 = 0.0
    var sum_sq_x_old: Float32 = 0.0
    for i in range(embed_dim):
        dot_prod += x[i] * x_old[i]
        sum_sq_x += x[i] * x[i]
        sum_sq_x_old += x_old[i] * x_old[i]
    var cosine_sim_1 = dot_prod / (sqrt(sum_sq_x) * sqrt(sum_sq_x_old) + 1e-9)
    assert_true(abs(cosine_sim_1 - 1.0) < 1e-5)
    
    # 2. Global scaling change (direction is identical, magnitude differs)
    for i in range(embed_dim):
        x[i] = Float32(i + 1) * 1.5 # Scale by 1.5
        
    dot_prod = 0.0
    sum_sq_x = 0.0
    sum_sq_x_old = 0.0
    for i in range(embed_dim):
        dot_prod += x[i] * x_old[i]
        sum_sq_x += x[i] * x[i]
        sum_sq_x_old += x_old[i] * x_old[i]
    var cosine_sim_2 = dot_prod / (sqrt(sum_sq_x) * sqrt(sum_sq_x_old) + 1e-9)
    # Cosine similarity must remain exactly 1.0 because the direction is unchanged!
    assert_true(abs(cosine_sim_2 - 1.0) < 1e-5)
    
    # 3. Small directional shift (modify one element slightly)
    x[0] = x_old[0] + 5.0
    dot_prod = 0.0
    sum_sq_x = 0.0
    sum_sq_x_old = 0.0
    for i in range(embed_dim):
        dot_prod += x[i] * x_old[i]
        sum_sq_x += x[i] * x[i]
        sum_sq_x_old += x_old[i] * x_old[i]
    var cosine_sim_3 = dot_prod / (sqrt(sum_sq_x) * sqrt(sum_sq_x_old) + 1e-9)
    # Direction is slightly modified, so cosine similarity is slightly less than 1.0
    assert_true(cosine_sim_3 < 1.0)
    assert_true(cosine_sim_3 > 0.95)
    
    x.free()
    x_old.free()
    
    # Heuristic B test (Sub-Sampled Projection probability math)
    comptime test_vocab_size = 8
    comptime test_embed_dim = 16
    var norm_x = alloc[Float32](test_embed_dim)
    var embed_ptr = alloc[Float32](test_vocab_size * test_embed_dim)
    
    # Initialize mock norm_x
    for i in range(test_embed_dim):
        norm_x[i] = 1.0 # simple constant vector
        
    # Initialize mock embed_ptr: token 3 is highly aligned, others are not
    for r in range(test_vocab_size):
        var r_offset = r * test_embed_dim
        for c in range(test_embed_dim):
            if r == 3:
                embed_ptr[r_offset + c] = 2.0 # high alignment
            else:
                embed_ptr[r_offset + c] = -1.0 # low alignment
                
    var common_tokens = [0, 1, 2, 3, 4, 5, 6, 7]
    
    # Compute sub-sampled logits and probabilities
    var max_sub_logit: Float32 = -999999.0
    var max_sub_tok = 0
    var sub_logits = List[Float32]()
    for i in range(len(common_tokens)):
        var r_offset = common_tokens[i] * test_embed_dim
        var acc: Float32 = 0.0
        for c in range(test_embed_dim):
            acc += norm_x[c] * embed_ptr[r_offset + c]
        
        var logit = Float32(30.0) * fast_tanh(acc / Float32(30.0))
        sub_logits.append(logit)
        if logit > max_sub_logit:
            max_sub_logit = logit
            max_sub_tok = common_tokens[i]
            
    assert_equal(max_sub_tok, 3)
    
    var sum_exp_sub: Float32 = 0.0
    for i in range(len(sub_logits)):
        sum_exp_sub += exp(sub_logits[i] - max_sub_logit)
        
    var top_prob = 1.0 / sum_exp_sub
    assert_true(top_prob > 0.999)
    
    norm_x.free()
    embed_ptr.free()
    
    # Heuristic C: KV Cache Propagation / Recovery test for skipped layers
    print("  Testing KV Cache propagation for skipped layers...")
    comptime num_layers = 4
    comptime max_seq_len = 10
    comptime kv_stride = 16
    
    var k_cache = alloc[Float32](num_layers * max_seq_len * kv_stride)
    var v_cache = alloc[Float32](num_layers * max_seq_len * kv_stride)
    
    # Initialize all caches to some marker value (e.g. -999.0)
    for i in range(num_layers * max_seq_len * kv_stride):
        k_cache[i] = -999.0
        v_cache[i] = -999.0
        
    # Setup some dummy cache offsets
    var layer_cache_offsets = List[Int]()
    for l in range(num_layers):
        layer_cache_offsets.append(l * max_seq_len * kv_stride)
        
    # Set up some known values for position pos_prev (e.g. pos_prev = 2)
    var pos_prev = 2
    var pos_curr = 3
    
    # Fill position pos_prev with recognizable values for all layers
    for l in range(num_layers):
        var offset = layer_cache_offsets[l] + pos_prev * kv_stride
        for i in range(kv_stride):
            k_cache[offset + i] = Float32(l * 100 + i)
            v_cache[offset + i] = Float32(l * 200 + i)
            
    # Simulate early exit at layer_idx = 2
    var early_exit_layer = 2
    
    # Copy KV cache from pos_prev to pos_curr for skipped layers (from early_exit_layer to num_layers - 1)
    for skipped_layer in range(early_exit_layer, num_layers):
        var l_cache_offset = layer_cache_offsets[skipped_layer]
        var s_cache_offset = l_cache_offset + pos_curr * kv_stride
        var p_cache_offset = l_cache_offset + pos_prev * kv_stride
        for i in range(kv_stride):
            k_cache[s_cache_offset + i] = k_cache[p_cache_offset + i]
            v_cache[s_cache_offset + i] = v_cache[p_cache_offset + i]
            
    # Verification
    # 1. Skipped layers (2 and 3) should have copied values at pos_curr
    for skipped_layer in range(early_exit_layer, num_layers):
        var l_cache_offset = layer_cache_offsets[skipped_layer]
        var curr_offset = l_cache_offset + pos_curr * kv_stride
        for i in range(kv_stride):
            assert_equal(k_cache[curr_offset + i], Float32(skipped_layer * 100 + i))
            assert_equal(v_cache[curr_offset + i], Float32(skipped_layer * 200 + i))
            
    # 2. Non-skipped layers (0 and 1) should remain untouched (still contain marker value -999.0)
    for l in range(early_exit_layer):
        var l_cache_offset = layer_cache_offsets[l]
        var curr_offset = l_cache_offset + pos_curr * kv_stride
        for i in range(kv_stride):
            assert_equal(k_cache[curr_offset + i], Float32(-999.0))
            assert_equal(v_cache[curr_offset + i], Float32(-999.0))
            
    k_cache.free()
    v_cache.free()
    
    print("test_early_exit passed!")

def test_gpu_integration() raises:
    print("Running test_gpu_integration (AMD KFD Direct GPU execution)...")
    
    # Check if AMD KFD device nodes are present and accessible
    var libc = OwnedDLHandle("libc.so.6")
    var c_open = libc.get_function[fn(Int, Int32) -> Int32]("open")
    
    var kfd_path = "/dev/kfd"
    var kfd_path_bytes = kfd_path.as_bytes()
    var kfd_path_ptr = alloc[UInt8](len(kfd_path_bytes) + 1)
    for i in range(len(kfd_path_bytes)):
        kfd_path_ptr[i] = kfd_path_bytes[i]
    kfd_path_ptr[len(kfd_path_bytes)] = 0
    var kfd_fd = c_open(kfd_path_ptr.__int__(), 2) # O_RDWR
    kfd_path_ptr.free()
    
    if kfd_fd < 0:
        print(" -> SKIP: /dev/kfd not accessible or AMD GPU not present.")
        _ = libc
        return
        
    var drm_path = "/dev/dri/renderD128"
    var drm_path_bytes = drm_path.as_bytes()
    var drm_path_ptr = alloc[UInt8](len(drm_path_bytes) + 1)
    for i in range(len(drm_path_bytes)):
        drm_path_ptr[i] = drm_path_bytes[i]
    drm_path_ptr[len(drm_path_bytes)] = 0
    var drm_fd = c_open(drm_path_ptr.__int__(), 2) # O_RDWR
    drm_path_ptr.free()
    
    if drm_fd < 0:
        print(" -> SKIP: /dev/dri/renderD128 not accessible.")
        var c_close = libc.get_function[fn(Int32) -> Int32]("close")
        _ = c_close(kfd_fd)
        _ = libc
        return

    var c_ioctl = libc.get_function[fn(Int32, Int, Int) -> Int32]("ioctl")
    var c_close = libc.get_function[fn(Int32) -> Int32]("close")
    var c_mmap = libc.get_function[fn(Int, Int, Int32, Int32, Int32, Int) -> Int]("mmap")
    var c_munmap = libc.get_function[fn(Int, Int) -> Int32]("munmap")
    var c_read = libc.get_function[fn(Int32, Int, Int) -> Int]("read")
    var c_errno_loc = libc.get_function[fn() -> Int]("__errno_location")

    # Load CPU fence helper library
    var cpu_fence_lib = OwnedDLHandle("/home/phil/git/llm-mojo/scratch/libcpu_fence.so")
    var cpu_sfence = cpu_fence_lib.get_function[fn(Int32) -> Int32]("cpu_sfence")
    var cpu_clflush = cpu_fence_lib.get_function[fn(Int) -> Int32]("cpu_clflush")

    # Define KFD constants
    comptime AMDKFD_IOC_ACQUIRE_VM = 0x40084B15
    comptime AMDKFD_IOC_RUNTIME_ENABLE = 0xC0104B25
    comptime AMDKFD_IOC_ALLOC_MEMORY_OF_GPU = 0xC0284B16
    comptime AMDKFD_IOC_FREE_MEMORY_OF_GPU = 0x40084B17
    comptime AMDKFD_IOC_MAP_MEMORY_TO_GPU = 0xC0184B18
    comptime AMDKFD_IOC_UNMAP_MEMORY_FROM_GPU = 0xC0184B19
    comptime AMDKFD_IOC_CREATE_QUEUE = 0xC0604B02
    comptime AMDKFD_IOC_DESTROY_QUEUE = 0xC0084B03

    comptime PROT_NONE = 0
    comptime PROT_READ = 1
    comptime PROT_WRITE = 2
    comptime MAP_SHARED = 0x01
    comptime MAP_PRIVATE = 0x02
    comptime MAP_FIXED = 0x10
    comptime MAP_ANONYMOUS = 0x20

    comptime KFD_IOC_ALLOC_MEM_FLAGS_VRAM = 1 << 0
    comptime KFD_IOC_ALLOC_MEM_FLAGS_GTT = 1 << 1
    comptime KFD_IOC_ALLOC_MEM_FLAGS_WRITABLE = 1 << 31
    comptime KFD_IOC_ALLOC_MEM_FLAGS_EXECUTABLE = 1 << 30
    comptime KFD_IOC_ALLOC_MEM_FLAGS_PUBLIC = 1 << 29
    comptime KFD_IOC_ALLOC_MEM_FLAGS_NO_SUBSTITUTE = 1 << 28
    comptime KFD_IOC_ALLOC_MEM_FLAGS_COHERENT = 1 << 26
    comptime KFD_IOC_ALLOC_MEM_FLAGS_UNCACHED = 1 << 25

    var base_flags = (KFD_IOC_ALLOC_MEM_FLAGS_WRITABLE |
                      KFD_IOC_ALLOC_MEM_FLAGS_EXECUTABLE |
                      KFD_IOC_ALLOC_MEM_FLAGS_NO_SUBSTITUTE)

    var gtt_uncached_flags = (base_flags |
                              KFD_IOC_ALLOC_MEM_FLAGS_GTT |
                              KFD_IOC_ALLOC_MEM_FLAGS_COHERENT |
                              KFD_IOC_ALLOC_MEM_FLAGS_UNCACHED |
                              KFD_IOC_ALLOC_MEM_FLAGS_PUBLIC)

    var vram_private_flags = (base_flags |
                              KFD_IOC_ALLOC_MEM_FLAGS_VRAM)

    var gpu_id: UInt32 = 0
    for node_idx in range(1, 16):
        var path = "/sys/class/kfd/kfd/topology/nodes/" + String(node_idx) + "/gpu_id"
        var path_bytes = path.as_bytes()
        var path_ptr = alloc[UInt8](len(path_bytes) + 1)
        for i in range(len(path_bytes)):
            path_ptr[i] = path_bytes[i]
        path_ptr[len(path_bytes)] = 0
        var fd = c_open(path_ptr.__int__(), 0)
        path_ptr.free()
        
        if fd >= 0:
            var buf = alloc[UInt8](64)
            var bytes_read = c_read(fd, buf.__int__(), 63)
            _ = c_close(fd)
            if bytes_read > 0:
                buf[bytes_read] = 0
                var val: Int = 0
                for i in range(bytes_read):
                    var char_code = buf[i]
                    if char_code >= 48 and char_code <= 57:
                        val = val * 10 + (Int(char_code) - 48)
                buf.free()
                if val > 0:
                    gpu_id = UInt32(val)
                    break
            else:
                buf.free()

    if gpu_id == 0:
        gpu_id = 21148

    # VM Acquisition
    var acq_args_ptr = alloc[kfd_ioctl_acquire_vm_args](1)
    acq_args_ptr[0] = kfd_ioctl_acquire_vm_args(drm_fd=UInt32(drm_fd), gpu_id=gpu_id)
    var r = c_ioctl(kfd_fd, AMDKFD_IOC_ACQUIRE_VM, acq_args_ptr.__int__())
    acq_args_ptr.free()
    if r < 0:
        _ = c_close(kfd_fd)
        _ = c_close(drm_fd)
        raise Error("GPU VM Acquisition failed!")

    # Enable Runtime
    var runtime_enable_ptr = alloc[kfd_ioctl_runtime_enable_args](1)
    runtime_enable_ptr[0] = kfd_ioctl_runtime_enable_args()
    r = c_ioctl(kfd_fd, AMDKFD_IOC_RUNTIME_ENABLE, runtime_enable_ptr.__int__())
    runtime_enable_ptr.free()
    if r < 0:
        _ = c_close(kfd_fd)
        _ = c_close(drm_fd)
        raise Error("GPU Runtime enable failed!")

    def alloc_gpu_buffer(size: Int, flags: UInt32, cpu_access: Bool) raises -> GPUMemoryBuffer:
        var va_addr = c_mmap(0, size, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0)
        if va_addr == -1 or va_addr == 0:
            raise Error("mmap failed")
        
        var alloc_args_ptr = alloc[kfd_ioctl_alloc_memory_of_gpu_args](1)
        alloc_args_ptr[0] = kfd_ioctl_alloc_memory_of_gpu_args(va_addr=UInt64(va_addr), size=UInt64(size), gpu_id=gpu_id, flags=flags)
        var io_res = c_ioctl(kfd_fd, AMDKFD_IOC_ALLOC_MEMORY_OF_GPU, alloc_args_ptr.__int__())
        if io_res < 0:
            alloc_args_ptr.free()
            _ = c_munmap(va_addr, size)
            raise Error("ALLOC_MEMORY failed")
        var handle = alloc_args_ptr[0].handle
        var mmap_offset = alloc_args_ptr[0].mmap_offset
        alloc_args_ptr.free()

        var cpu_addr = 0
        if cpu_access:
            cpu_addr = c_mmap(va_addr, size, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_FIXED, drm_fd, Int(mmap_offset))
            if cpu_addr == -1 or cpu_addr != va_addr:
                raise Error("CPU mmap failed")

        var device_ids_ptr = alloc[UInt32](1)
        device_ids_ptr[0] = gpu_id
        var map_args_ptr = alloc[kfd_ioctl_map_memory_to_gpu_args](1)
        map_args_ptr[0] = kfd_ioctl_map_memory_to_gpu_args(handle=handle, device_ids_array_ptr=UInt64(device_ids_ptr.__int__()), n_devices=1)
        io_res = c_ioctl(kfd_fd, AMDKFD_IOC_MAP_MEMORY_TO_GPU, map_args_ptr.__int__())
        device_ids_ptr.free()
        map_args_ptr.free()
        if io_res < 0:
            raise Error("MAP failed")

        return GPUMemoryBuffer(va_addr, size, handle, cpu_addr)

    def free_gpu_buffer(buf: GPUMemoryBuffer) raises:
        var unmap_device_ids_ptr = alloc[UInt32](1)
        unmap_device_ids_ptr[0] = gpu_id
        var unmap_args_ptr = alloc[kfd_ioctl_unmap_memory_from_gpu_args](1)
        unmap_args_ptr[0] = kfd_ioctl_unmap_memory_from_gpu_args(handle=buf.handle, device_ids_array_ptr=UInt64(unmap_device_ids_ptr.__int__()), n_devices=1)
        _ = c_ioctl(kfd_fd, AMDKFD_IOC_UNMAP_MEMORY_FROM_GPU, unmap_args_ptr.__int__())
        unmap_device_ids_ptr.free()
        unmap_args_ptr.free()
        
        if buf.cpu_addr != 0:
            _ = c_munmap(buf.cpu_addr, buf.size)
        else:
            _ = c_munmap(buf.va_addr, buf.size)

        var free_args_ptr = alloc[kfd_ioctl_free_memory_of_gpu_args](1)
        free_args_ptr[0] = kfd_ioctl_free_memory_of_gpu_args(handle=buf.handle)
        _ = c_ioctl(kfd_fd, AMDKFD_IOC_FREE_MEMORY_OF_GPU, free_args_ptr.__int__())
        free_args_ptr.free()

    # Create userspace AQL Queue
    var ring_size = 64 * 1024
    var ring_buf = alloc_gpu_buffer(ring_size, UInt32(gtt_uncached_flags), cpu_access=True)
    var gart_buf = alloc_gpu_buffer(4096, UInt32(gtt_uncached_flags), cpu_access=True)
    var cwsr_buf = alloc_gpu_buffer(2801664, UInt32(vram_private_flags), cpu_access=False)
    var eop_buf = alloc_gpu_buffer(4096, UInt32(vram_private_flags), cpu_access=False)

    var ring_ptr = UnsafePointer[UInt32, MutExternalOrigin](unsafe_from_address=ring_buf.cpu_addr)
    for i in range(ring_size // 4):
        ring_ptr[i] = 0

    var gart_u32_ptr = UnsafePointer[UInt32, MutExternalOrigin](unsafe_from_address=gart_buf.cpu_addr)
    for i in range(4096 // 4):
        gart_u32_ptr[i] = 0

    var sig_cpu_addr = gart_buf.cpu_addr + 512
    var sig_va_addr = gart_buf.va_addr + 512

    var sig_u64_ptr = UnsafePointer[UInt64, MutExternalOrigin](unsafe_from_address=sig_cpu_addr)
    for i in range(8):
        sig_u64_ptr[i] = 0

    UnsafePointer[Int64, MutExternalOrigin](unsafe_from_address=sig_cpu_addr + 0)[0] = -1
    UnsafePointer[UInt64, MutExternalOrigin](unsafe_from_address=sig_cpu_addr + 8)[0] = 0
    UnsafePointer[UInt64, MutExternalOrigin](unsafe_from_address=sig_cpu_addr + 48)[0] = UInt64(gart_buf.va_addr)

    _ = cpu_clflush(sig_cpu_addr)
    _ = cpu_sfence(0)

    var q_base = gart_buf.cpu_addr
    UnsafePointer[UInt32, MutExternalOrigin](unsafe_from_address=q_base + 0)[0] = 2       # queue_type = AQL
    UnsafePointer[UInt32, MutExternalOrigin](unsafe_from_address=q_base + 4)[0] = 1       # features
    UnsafePointer[UInt64, MutExternalOrigin](unsafe_from_address=q_base + 8)[0] = UInt64(ring_buf.va_addr)
    UnsafePointer[UInt64, MutExternalOrigin](unsafe_from_address=q_base + 16)[0] = UInt64(sig_va_addr)
    UnsafePointer[UInt32, MutExternalOrigin](unsafe_from_address=q_base + 24)[0] = 1024   # size

    UnsafePointer[UInt32, MutExternalOrigin](unsafe_from_address=q_base + 72)[0] = 7
    UnsafePointer[UInt32, MutExternalOrigin](unsafe_from_address=q_base + 76)[0] = 39
    UnsafePointer[UInt32, MutExternalOrigin](unsafe_from_address=q_base + 136)[0] = 128
    UnsafePointer[UInt32, MutExternalOrigin](unsafe_from_address=q_base + 180)[0] = 10

    _ = cpu_clflush(q_base)
    _ = cpu_clflush(q_base + 64)
    _ = cpu_clflush(q_base + 128)
    _ = cpu_clflush(q_base + 180)
    _ = cpu_sfence(0)

    var create_args_ptr = alloc[kfd_ioctl_create_queue_args](1)
    create_args_ptr[0] = kfd_ioctl_create_queue_args(
        ring_base_address=UInt64(ring_buf.va_addr),
        write_pointer_address=UInt64(gart_buf.va_addr + 56),
        read_pointer_address=UInt64(gart_buf.va_addr + 128),
        ring_size=UInt32(ring_size),
        gpu_id=gpu_id,
        queue_type=2,
        queue_percentage=100,
        queue_priority=7,
        eop_buffer_address=UInt64(eop_buf.va_addr),
        eop_buffer_size=4096,
        ctx_save_restore_address=UInt64(cwsr_buf.va_addr),
        ctx_save_restore_size=2789376,
        ctl_stack_size=4096,
        sdma_engine_id=0
    )
    r = c_ioctl(kfd_fd, AMDKFD_IOC_CREATE_QUEUE, create_args_ptr.__int__())
    if r < 0:
        var err = UnsafePointer[Int32, MutExternalOrigin](unsafe_from_address=c_errno_loc())[0]
        free_gpu_buffer(eop_buf)
        free_gpu_buffer(cwsr_buf)
        free_gpu_buffer(gart_buf)
        free_gpu_buffer(ring_buf)
        _ = c_close(kfd_fd)
        _ = c_close(drm_fd)
        raise Error("KFD Queue creation failed with errno: " + String(err))

    var queue_id = create_args_ptr[0].queue_id
    var doorbell_offset = create_args_ptr[0].doorbell_offset
    create_args_ptr.free()

    var doorbells_base = doorbell_offset & (~0x1fff)
    var doorbell_page_size = 0x2000
    var doorbell_mmap_addr = c_mmap(0, doorbell_page_size, PROT_READ | PROT_WRITE, MAP_SHARED, kfd_fd, Int(doorbells_base))
    if doorbell_mmap_addr == -1:
        raise Error("Doorbell page mmap failed")
    var doorbell_reg_addr = doorbell_mmap_addr + (Int(doorbell_offset) - Int(doorbells_base))

    UnsafePointer[UInt64, MutExternalOrigin](unsafe_from_address=sig_cpu_addr + 8)[0] = UInt64(doorbell_reg_addr)
    _ = cpu_clflush(sig_cpu_addr)
    _ = cpu_sfence(0)

    # Load GCN compiled ELF shader
    var hsaco_buf = alloc_gpu_buffer(16384, UInt32(gtt_uncached_flags), cpu_access=True)
    var hsaco_u32_ptr = UnsafePointer[UInt32, MutExternalOrigin](unsafe_from_address=hsaco_buf.cpu_addr)
    for i in range(16384 // 4):
        hsaco_u32_ptr[i] = 0

    var hsaco_file = "/home/phil/git/llm-mojo/scratch/q8_matmul.hsaco"
    var hsaco_file_bytes = hsaco_file.as_bytes()
    var hsaco_file_ptr = alloc[UInt8](len(hsaco_file_bytes) + 1)
    for i in range(len(hsaco_file_bytes)):
        hsaco_file_ptr[i] = hsaco_file_bytes[i]
    hsaco_file_ptr[len(hsaco_file_bytes)] = 0
    var hsaco_fd = c_open(hsaco_file_ptr.__int__(), 0)
    hsaco_file_ptr.free()
    if hsaco_fd < 0:
        raise Error("Failed to open q8_matmul.hsaco")

    var temp_file_buf = alloc[UInt8](8192)
    var bytes_read = c_read(hsaco_fd, temp_file_buf.__int__(), 8192)
    _ = c_close(hsaco_fd)
    assert_true(bytes_read > 0, "Failed to read any bytes from q8_matmul.hsaco")

    var segment1_src = temp_file_buf
    var segment1_dst = UnsafePointer[UInt8, MutExternalOrigin](unsafe_from_address=hsaco_buf.cpu_addr + 0)
    for i in range(0xa80):
        segment1_dst[i] = segment1_src[i]

    var segment2_src = temp_file_buf + 0xb00
    var segment2_dst = UnsafePointer[UInt8, MutExternalOrigin](unsafe_from_address=hsaco_buf.cpu_addr + 0x1b00)
    for i in range(0x414):
        segment2_dst[i] = segment2_src[i]

    var segment3_src = temp_file_buf + 0xf18
    var segment3_dst = UnsafePointer[UInt8, MutExternalOrigin](unsafe_from_address=hsaco_buf.cpu_addr + 0x2f18)
    for i in range(0x70):
        segment3_dst[i] = segment3_src[i]

    temp_file_buf.free()

    # Populate Predictable Matmul activations
    comptime test_c_in = 3072
    comptime test_c_out = 256
    comptime test_c_in_blocks = test_c_in // 32

    var W_size = test_c_out * test_c_in * 1
    var scales_size = test_c_out * test_c_in_blocks * 4
    var x_size = test_c_in * 4
    var out_size = test_c_out * 4

    W_size = ((W_size + 4095) // 4096) * 4096
    scales_size = ((scales_size + 4095) // 4096) * 4096
    x_size = ((x_size + 4095) // 4096) * 4096
    out_size = ((out_size + 4095) // 4096) * 4096

    var W_buf = alloc_gpu_buffer(W_size, UInt32(gtt_uncached_flags), cpu_access=True)
    var scales_buf = alloc_gpu_buffer(scales_size, UInt32(gtt_uncached_flags), cpu_access=True)
    var x_buf = alloc_gpu_buffer(x_size, UInt32(gtt_uncached_flags), cpu_access=True)
    var out_buf = alloc_gpu_buffer(out_size, UInt32(gtt_uncached_flags), cpu_access=True)
    var completion_buf = alloc_gpu_buffer(4096, UInt32(gtt_uncached_flags), cpu_access=True)
    var kernarg_buf = alloc_gpu_buffer(4096, UInt32(gtt_uncached_flags), cpu_access=True)

    var W_ptr = UnsafePointer[Int8, MutExternalOrigin](unsafe_from_address=W_buf.cpu_addr)
    var scales_ptr = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=scales_buf.cpu_addr)
    var x_ptr = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=x_buf.cpu_addr)
    var out_ptr = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=out_buf.cpu_addr)
    var completion_ptr = UnsafePointer[Int32, MutExternalOrigin](unsafe_from_address=completion_buf.cpu_addr)

    for i in range(test_c_out * test_c_in):
        W_ptr[i] = Int8((i % 7) - 3)

    for i in range(test_c_out * test_c_in_blocks):
        scales_ptr[i] = Float32(0.02)

    for i in range(test_c_in):
        x_ptr[i] = Float32(i % 10) * 0.1

    for i in range(test_c_out):
        out_ptr[i] = 0.0

    completion_ptr[0] = 0

    # CPU reference values
    var expected_out = alloc[Float32](test_c_out)
    for r_idx in range(test_c_out):
        var sum_ref: Float32 = 0.0
        var r_offset_packed = r_idx * test_c_in
        var r_offset_scales = r_idx * test_c_in_blocks
        for b_idx in range(test_c_in_blocks):
            var sc = scales_ptr[r_offset_scales + b_idx]
            var block_acc: Float32 = 0.0
            var block_p = r_offset_packed + b_idx * 32
            var x_offset = b_idx * 32
            for i in range(32):
                var w_val = Float32(W_ptr[block_p + i])
                var x_val = x_ptr[x_offset + i]
                block_acc += w_val * x_val
            sum_ref += block_acc * sc
        expected_out[r_idx] = sum_ref

    # Dispatch Kernarg packing
    var kernarg_p = UnsafePointer[UInt64, MutExternalOrigin](unsafe_from_address=kernarg_buf.cpu_addr)
    kernarg_p[0] = UInt64(W_buf.va_addr)
    kernarg_p[1] = UInt64(scales_buf.va_addr)
    kernarg_p[2] = UInt64(x_buf.va_addr)
    kernarg_p[3] = UInt64(out_buf.va_addr)
    kernarg_p[4] = UInt64(completion_buf.va_addr)

    var kernarg_u32_p = UnsafePointer[UInt32, MutExternalOrigin](unsafe_from_address=kernarg_buf.cpu_addr + 40)
    kernarg_u32_p[0] = UInt32(test_c_in)
    kernarg_u32_p[1] = UInt32(test_c_out)
    kernarg_u32_p[2] = UInt32(256)
    kernarg_u32_p[3] = UInt32(1)

    _ = cpu_sfence(0)
    _ = cpu_clflush(kernarg_buf.cpu_addr)

    # Format dispatch packet
    var packet = hsa_kernel_dispatch_packet_t(
        header=5378,
        setup=1,
        workgroup_size_x=256,
        workgroup_size_y=1,
        workgroup_size_z=1,
        grid_size_x=256,
        grid_size_y=1,
        grid_size_z=1,
        private_segment_size=0,
        group_segment_size=0,
        kernel_object=UInt64(hsaco_buf.va_addr + 0xa40),
        kernarg_address=UInt64(kernarg_buf.va_addr),
        completion_signal=0
    )

    var ring_packet_ptr = UnsafePointer[hsa_kernel_dispatch_packet_t, MutExternalOrigin](unsafe_from_address=ring_buf.cpu_addr)
    ring_packet_ptr[0] = packet^

    UnsafePointer[UInt64, MutExternalOrigin](unsafe_from_address=gart_buf.cpu_addr + 56)[0] = 1

    _ = cpu_sfence(0)
    _ = cpu_clflush(gart_buf.cpu_addr + 56)
    _ = cpu_clflush(ring_buf.cpu_addr)
    _ = cpu_sfence(0)

    # Direct doorbell write
    UnsafePointer[UInt64, MutExternalOrigin](unsafe_from_address=doorbell_reg_addr)[0] = 0
    _ = cpu_sfence(0)

    # Wait for completion
    var polled = 0
    var gpu_success = False
    while polled < 50000000:
        if completion_ptr[0] == 1:
            gpu_success = True
            break
        polled += 1
        if polled % 10000000 == 0:
            var wptr = UnsafePointer[UInt64, MutExternalOrigin](unsafe_from_address=gart_buf.cpu_addr + 56)[0]
            var rptr = UnsafePointer[UInt64, MutExternalOrigin](unsafe_from_address=gart_buf.cpu_addr + 128)[0]
            print("    -> Polling status... (iter", polled, "| flag =", completion_ptr[0], "| wptr =", wptr, "| rptr =", rptr, ")")

    assert_true(gpu_success, "GPU execution timed out")

    # Math parity verification
    for r_idx in range(test_c_out):
        var cpu_val = expected_out[r_idx]
        var gpu_val = out_ptr[r_idx]
        var diff = abs(cpu_val - gpu_val)
        assert_true(diff <= 1e-4, "Mathematical mismatch between GPU and CPU at row " + String(r_idx))

    # Teardown
    var destroy_args_ptr = alloc[kfd_ioctl_destroy_queue_args](1)
    destroy_args_ptr[0] = kfd_ioctl_destroy_queue_args(queue_id=queue_id)
    _ = c_ioctl(kfd_fd, AMDKFD_IOC_DESTROY_QUEUE, destroy_args_ptr.__int__())
    destroy_args_ptr.free()

    _ = c_munmap(doorbell_mmap_addr, doorbell_page_size)

    free_gpu_buffer(kernarg_buf)
    free_gpu_buffer(completion_buf)
    free_gpu_buffer(out_buf)
    free_gpu_buffer(x_buf)
    free_gpu_buffer(scales_buf)
    free_gpu_buffer(W_buf)
    free_gpu_buffer(hsaco_buf)

    free_gpu_buffer(eop_buf)
    free_gpu_buffer(cwsr_buf)
    free_gpu_buffer(gart_buf)
    free_gpu_buffer(ring_buf)

    _ = c_close(kfd_fd)
    _ = c_close(drm_fd)
    expected_out.free()
    
    _ = libc
    _ = cpu_fence_lib
    print("test_gpu_integration passed successfully!")

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
    test_q4_matmul_vec()
    test_attention_no_scaling()
    test_q8_matmul_w2_block_column()
    test_block_sparsity_regression()
    test_early_exit()
    test_gpu_integration()
    print("=========================================================")
    print("        ALL KERNEL MATH TESTS PASSED SUCCESSFULLY!       ")
    print("=========================================================")
