from std.ffi import OwnedDLHandle
from std.memory import alloc
from std.math import sqrt, exp, cos, sin
from std.python import Python
import std.os
from uring import IoUring
from std.algorithm import parallelize
from std.sys.info import num_physical_cores

# -----------------------------------------------------------------------------
# 1. Model Configuration & Memory Offsets
# -----------------------------------------------------------------------------
struct TransformerConfig[
    embed_dim: Int,
    head_dim: Int,
    num_heads: Int,
    num_kv_heads: Int,
    intermediate_dim: Int,
    num_layers: Int,
    vocab_size: Int,
    max_seq_len: Int,
]:
    @always_inline
    def is_global_layer(self, layer_idx: Int) -> Bool:
        return layer_idx % 6 == 5

    @always_inline
    def get_w2_offset_in_layer(self, layer_idx: Int) -> Int:
        var is_global = self.is_global_layer(layer_idx)
        var layer_heads = 16
        var layer_head_dim = 512 if is_global else 256
        var layer_kv_heads = 1 if is_global else 8
        
        var wq_size = layer_heads * layer_head_dim * Self.embed_dim
        var wk_size = layer_kv_heads * layer_head_dim * Self.embed_dim
        var wv_size = 0 if is_global else (layer_kv_heads * layer_head_dim * Self.embed_dim)
        var wo_size = Self.embed_dim * layer_heads * layer_head_dim
        var w1_size = Self.intermediate_dim * Self.embed_dim
        var w3_size = Self.intermediate_dim * Self.embed_dim
        
        var packed_weights_size = wq_size + wk_size + wv_size + wo_size + w1_size + w3_size
        var scales_count = packed_weights_size // 32
        var scales_size_bytes = scales_count * 4
        
        var input_layernorm_size = Self.embed_dim
        var post_attention_layernorm_size = Self.embed_dim
        var pre_feedforward_layernorm_size = Self.embed_dim
        var post_feedforward_layernorm_size = Self.embed_dim
        var q_norm_size = layer_head_dim
        var k_norm_size = layer_head_dim
        var layer_scalar_size = 1
        
        var unchanged_size_bytes = (
            input_layernorm_size + post_attention_layernorm_size +
            pre_feedforward_layernorm_size + post_feedforward_layernorm_size +
            q_norm_size + k_norm_size + layer_scalar_size
        ) * 4
        
        var raw_bytes_size = packed_weights_size + scales_size_bytes + unchanged_size_bytes
        return ((raw_bytes_size + 4095) // 4096) * 4096

    @always_inline
    def get_layer_aligned_size(self, layer_idx: Int) -> Int:
        var base_offset = self.get_w2_offset_in_layer(layer_idx)
        var w2_blocks = Self.intermediate_dim // 32
        var w2_size = w2_blocks * 139264
        return base_offset + w2_size

    def get_layer_offset(self, layer_idx: Int) -> Int:
        var offset = 0
        for i in range(layer_idx):
            offset += self.get_layer_aligned_size(i)
        return offset

    @always_inline
    def get_layer_weights(self, layer_idx: Int, base_ptr: UnsafePointer[Int8, _]) -> Q8LayerWeights:
        var is_global = self.is_global_layer(layer_idx)
        var layer_heads = 16
        var layer_head_dim = 512 if is_global else 256
        var layer_kv_heads = 1 if is_global else 8
        
        var wq_size = layer_heads * layer_head_dim * Self.embed_dim
        var wk_size = layer_kv_heads * layer_head_dim * Self.embed_dim
        var wv_size = 0 if is_global else (layer_kv_heads * layer_head_dim * Self.embed_dim)
        var wo_size = Self.embed_dim * layer_heads * layer_head_dim
        var w1_size = Self.intermediate_dim * Self.embed_dim
        var w3_size = Self.intermediate_dim * Self.embed_dim
        
        var wq_size_bytes = wq_size
        var wk_size_bytes = wk_size
        var wv_size_bytes = wv_size
        var wo_size_bytes = wo_size
        var w1_size_bytes = w1_size
        var w3_size_bytes = w3_size
        
        var c_blocks_q = wq_size // 32
        var c_blocks_k = wk_size // 32
        var c_blocks_v = wv_size // 32
        var c_blocks_o = wo_size // 32
        var c_blocks_w1 = w1_size // 32
        var c_blocks_w3 = w3_size // 32

        # Pointers to packed weights (Int8)
        var wq_packed = base_ptr
        var wk_packed = wq_packed + wq_size_bytes
        var wv_packed = wk_packed + wk_size_bytes # fallback/initialized
        var wo_packed = wk_packed + wk_size_bytes
        if not is_global:
            wv_packed = wk_packed + wk_size_bytes
            wo_packed = wv_packed + wv_size_bytes
            
        var w1_packed = wo_packed + wo_size_bytes
        var w3_packed = w1_packed + w1_size_bytes
        
        # Scales base address: immediately after the end of w3_packed
        var scales_base = (w3_packed + w3_size_bytes).bitcast[Float32]()
        
        var wq_scales = scales_base
        var wk_scales = wq_scales + c_blocks_q
        var wv_scales = wk_scales + c_blocks_k # fallback/initialized
        var wo_scales = wk_scales + c_blocks_k
        if not is_global:
            wv_scales = wk_scales + c_blocks_k
            wo_scales = wv_scales + c_blocks_v
            
        var w1_scales = wo_scales + c_blocks_o
        var w3_scales = w1_scales + c_blocks_w1
        
        # Unchanged Float32 parameters base address: immediately after the end of w3_scales
        var unchanged_base = w3_scales + c_blocks_w3
        
        var input_layernorm = unchanged_base
        var post_attention_layernorm = input_layernorm + Self.embed_dim
        var pre_feedforward_layernorm = post_attention_layernorm + Self.embed_dim
        var post_feedforward_layernorm = pre_feedforward_layernorm + Self.embed_dim
        var q_norm = post_feedforward_layernorm + Self.embed_dim
        var k_norm = q_norm + layer_head_dim
        var layer_scalar = k_norm + layer_head_dim
        
        # Resolve w2 pointers using new layout at end of layer
        var w2_packed = base_ptr + self.get_w2_offset_in_layer(layer_idx)
        var w2_scales = (w2_packed + 122880).bitcast[Float32]()
        
        return Q8LayerWeights(
            UnsafePointer[Int8, MutExternalOrigin](unsafe_from_address=wq_packed.__int__()),
            UnsafePointer[Int8, MutExternalOrigin](unsafe_from_address=wk_packed.__int__()),
            UnsafePointer[Int8, MutExternalOrigin](unsafe_from_address=wv_packed.__int__()),
            UnsafePointer[Int8, MutExternalOrigin](unsafe_from_address=wo_packed.__int__()),
            UnsafePointer[Int8, MutExternalOrigin](unsafe_from_address=w1_packed.__int__()),
            UnsafePointer[Int8, MutExternalOrigin](unsafe_from_address=w2_packed.__int__()),
            UnsafePointer[Int8, MutExternalOrigin](unsafe_from_address=w3_packed.__int__()),
            UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=wq_scales.__int__()),
            UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=wk_scales.__int__()),
            UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=wv_scales.__int__()),
            UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=wo_scales.__int__()),
            UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=w1_scales.__int__()),
            UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=w2_scales.__int__()),
            UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=w3_scales.__int__()),
            UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=input_layernorm.__int__()),
            UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=post_attention_layernorm.__int__()),
            UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=pre_feedforward_layernorm.__int__()),
            UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=post_feedforward_layernorm.__int__()),
            UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=q_norm.__int__()),
            UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=k_norm.__int__()),
            UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=layer_scalar.__int__()),
        )

    def __init__(out self):
        pass

# -----------------------------------------------------------------------------
# 2. Structural Layer Weights Representation
# -----------------------------------------------------------------------------
struct Q8LayerWeights:
    var wq_packed: UnsafePointer[Int8, MutExternalOrigin]
    var wk_packed: UnsafePointer[Int8, MutExternalOrigin]
    var wv_packed: UnsafePointer[Int8, MutExternalOrigin]
    var wo_packed: UnsafePointer[Int8, MutExternalOrigin]
    var w1_packed: UnsafePointer[Int8, MutExternalOrigin]
    var w2_packed: UnsafePointer[Int8, MutExternalOrigin]
    var w3_packed: UnsafePointer[Int8, MutExternalOrigin]

    var wq_scales: UnsafePointer[Float32, MutExternalOrigin]
    var wk_scales: UnsafePointer[Float32, MutExternalOrigin]
    var wv_scales: UnsafePointer[Float32, MutExternalOrigin]
    var wo_scales: UnsafePointer[Float32, MutExternalOrigin]
    var w1_scales: UnsafePointer[Float32, MutExternalOrigin]
    var w2_scales: UnsafePointer[Float32, MutExternalOrigin]
    var w3_scales: UnsafePointer[Float32, MutExternalOrigin]

    var input_layernorm_weight: UnsafePointer[Float32, MutExternalOrigin]
    var post_attention_layernorm_weight: UnsafePointer[Float32, MutExternalOrigin]
    var pre_feedforward_layernorm_weight: UnsafePointer[Float32, MutExternalOrigin]
    var post_feedforward_layernorm_weight: UnsafePointer[Float32, MutExternalOrigin]
    var q_norm_weight: UnsafePointer[Float32, MutExternalOrigin]
    var k_norm_weight: UnsafePointer[Float32, MutExternalOrigin]
    var layer_scalar: UnsafePointer[Float32, MutExternalOrigin]

    def __init__(
        out self,
        wq_packed: UnsafePointer[Int8, MutExternalOrigin],
        wk_packed: UnsafePointer[Int8, MutExternalOrigin],
        wv_packed: UnsafePointer[Int8, MutExternalOrigin],
        wo_packed: UnsafePointer[Int8, MutExternalOrigin],
        w1_packed: UnsafePointer[Int8, MutExternalOrigin],
        w2_packed: UnsafePointer[Int8, MutExternalOrigin],
        w3_packed: UnsafePointer[Int8, MutExternalOrigin],
        wq_scales: UnsafePointer[Float32, MutExternalOrigin],
        wk_scales: UnsafePointer[Float32, MutExternalOrigin],
        wv_scales: UnsafePointer[Float32, MutExternalOrigin],
        wo_scales: UnsafePointer[Float32, MutExternalOrigin],
        w1_scales: UnsafePointer[Float32, MutExternalOrigin],
        w2_scales: UnsafePointer[Float32, MutExternalOrigin],
        w3_scales: UnsafePointer[Float32, MutExternalOrigin],
        input_layernorm: UnsafePointer[Float32, MutExternalOrigin],
        post_attention_layernorm: UnsafePointer[Float32, MutExternalOrigin],
        pre_feedforward_layernorm: UnsafePointer[Float32, MutExternalOrigin],
        post_feedforward_layernorm: UnsafePointer[Float32, MutExternalOrigin],
        q_norm: UnsafePointer[Float32, MutExternalOrigin],
        k_norm: UnsafePointer[Float32, MutExternalOrigin],
        layer_scalar: UnsafePointer[Float32, MutExternalOrigin],
    ):
        self.wq_packed = wq_packed
        self.wk_packed = wk_packed
        self.wv_packed = wv_packed
        self.wo_packed = wo_packed
        self.w1_packed = w1_packed
        self.w2_packed = w2_packed
        self.w3_packed = w3_packed
        self.wq_scales = wq_scales
        self.wk_scales = wk_scales
        self.wv_scales = wv_scales
        self.wo_scales = wo_scales
        self.w1_scales = w1_scales
        self.w2_scales = w2_scales
        self.w3_scales = w3_scales
        self.input_layernorm_weight = input_layernorm
        self.post_attention_layernorm_weight = post_attention_layernorm
        self.pre_feedforward_layernorm_weight = pre_feedforward_layernorm
        self.post_feedforward_layernorm_weight = post_feedforward_layernorm
        self.q_norm_weight = q_norm
        self.k_norm_weight = k_norm
        self.layer_scalar = layer_scalar

# -----------------------------------------------------------------------------
# 3. High-Performance Vectorized Math Kernels
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
def q8_matmul_vec(
    W_packed: UnsafePointer[Int8, _],
    scales: UnsafePointer[Float32, _],
    x: UnsafePointer[Float32, _],
    out_ptr: UnsafePointer[Float32, MutExternalOrigin],
    c_in: Int,
    c_out: Int,
    num_tasks: Int,
    threshold: Float32 = 0.0,
):
    var c_in_blocks = c_in // 32
    
    if threshold > 0.0:
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
        
        def sparse_worker(t: Int) capturing -> None:
            var start_r = t * chunk_size
            var end_r = (t + 1) * chunk_size if t < num_tasks - 1 else c_out
            
            for r in range(start_r, end_r):
                var acc = SIMD[DType.float32, 8](0.0)
                var r_offset_packed = r * c_in
                var r_offset_scales = r * c_in_blocks
                
                for idx in range(num_active):
                    var b = active_blocks[idx]
                    var scale = scales[r_offset_scales + b]
                    var block_p = r_offset_packed + b * 32
                    var x_offset = b * 32
                    
                    var w0 = W_packed.load[8](block_p).cast[DType.float32]()
                    var w1 = W_packed.load[8](block_p + 8).cast[DType.float32]()
                    var w2 = W_packed.load[8](block_p + 16).cast[DType.float32]()
                    var w3 = W_packed.load[8](block_p + 24).cast[DType.float32]()
                    
                    acc = acc + (w0 * x.load[8](x_offset)) * scale
                    acc = acc + (w1 * x.load[8](x_offset + 8)) * scale
                    acc = acc + (w2 * x.load[8](x_offset + 16)) * scale
                    acc = acc + (w3 * x.load[8](x_offset + 24)) * scale
                    
                out_ptr[r] = acc.reduce_add()
                
        parallelize[sparse_worker](num_tasks)
    else:
        var chunk_size = c_out // num_tasks
        
        def dense_worker(t: Int) capturing -> None:
            var start_r = t * chunk_size
            var end_r = (t + 1) * chunk_size if t < num_tasks - 1 else c_out
            
            for r in range(start_r, end_r):
                var acc = SIMD[DType.float32, 8](0.0)
                var r_offset_packed = r * c_in
                var r_offset_scales = r * c_in_blocks
                
                for b in range(c_in_blocks):
                    var scale = scales[r_offset_scales + b]
                    var block_p = r_offset_packed + b * 32
                    var x_offset = b * 32
                    
                    var w0 = W_packed.load[8](block_p).cast[DType.float32]()
                    var w1 = W_packed.load[8](block_p + 8).cast[DType.float32]()
                    var w2 = W_packed.load[8](block_p + 16).cast[DType.float32]()
                    var w3 = W_packed.load[8](block_p + 24).cast[DType.float32]()
                    
                    acc = acc + (w0 * x.load[8](x_offset)) * scale
                    acc = acc + (w1 * x.load[8](x_offset + 8)) * scale
                    acc = acc + (w2 * x.load[8](x_offset + 16)) * scale
                    acc = acc + (w3 * x.load[8](x_offset + 24)) * scale
                    
                out_ptr[r] = acc.reduce_add()
                
        parallelize[dense_worker](num_tasks)

@always_inline
def sort_floats(mut arr: List[Float32]):
    var n = len(arr)
    var gap = n // 2
    while gap > 0:
        for i in range(gap, n):
            var temp = arr[i]
            var j = i
            while j >= gap and arr[j - gap] > temp:
                arr[j] = arr[j - gap]
                j -= gap
            arr[j] = temp
        gap = gap // 2

@always_inline
def q8_matmul_w2_block_column(
    W_packed: UnsafePointer[Int8, _],
    x: UnsafePointer[Float32, _],
    out_ptr: UnsafePointer[Float32, MutExternalOrigin],
    c_in: Int,
    c_out: Int,
    num_tasks: Int,
    top_k_pct: Float32 = 0.30,
):
    var c_in_blocks = c_in // 32
    
    # 1. Compute block importance/energy (sum of absolute values of activations)
    var importance = List[Float32](capacity=c_in_blocks)
    for b in range(c_in_blocks):
        var imp: Float32 = 0.0
        var x_offset = b * 32
        for i in range(32):
            imp += abs(x[x_offset + i])
        importance.append(imp)
        
    # 2. Sort the importances to find the threshold for Top-K blocks
    var sorted_importance = importance.copy()
    sort_floats(sorted_importance)
    
    var top_k_blocks = Int(top_k_pct * Float32(c_in_blocks))
    if top_k_blocks < 1:
        top_k_blocks = 1
    elif top_k_blocks > c_in_blocks:
        top_k_blocks = c_in_blocks
        
    var threshold_val = sorted_importance[c_in_blocks - top_k_blocks]
    
    # 3. Collect active blocks
    var active_blocks = List[Int]()
    for b in range(c_in_blocks):
        if importance[b] >= threshold_val and importance[b] > 0.0:
            active_blocks.append(b)
            
    var num_active = len(active_blocks)
    if num_active == 0:
        for r in range(c_out):
            out_ptr[r] = 0.0
        return
        
    var chunk_size = c_out // num_tasks
    
    # 4. Parallelized CPU multiplication
    def sparse_worker(t: Int) capturing -> None:
        var start_r = t * chunk_size
        var end_r = (t + 1) * chunk_size if t < num_tasks - 1 else c_out
        
        for r in range(start_r, end_r):
            var acc = SIMD[DType.float32, 8](0.0)
            
            for idx in range(num_active):
                var b = active_blocks[idx]
                var block_base = W_packed + b * 139264
                var block_scales = (block_base + 122880).bitcast[Float32]()
                
                var scale = block_scales[r]
                var block_p = block_base + r * 32
                var x_offset = b * 32
                
                var w0 = block_p.load[8]().cast[DType.float32]()
                var w1 = block_p.load[8](8).cast[DType.float32]()
                var w2 = block_p.load[8](16).cast[DType.float32]()
                var w3 = block_p.load[8](24).cast[DType.float32]()
                
                acc = acc + (w0 * x.load[8](x_offset)) * scale
                acc = acc + (w1 * x.load[8](x_offset + 8)) * scale
                acc = acc + (w2 * x.load[8](x_offset + 16)) * scale
                acc = acc + (w3 * x.load[8](x_offset + 24)) * scale
                
            out_ptr[r] = acc.reduce_add()
            
    parallelize[sparse_worker](num_tasks)


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

# Helper to wait for io_uring tags
def wait_for_tag(ring: IoUring, expected_tag: UInt64, out_tag_ptr: UnsafePointer[UInt64, MutExternalOrigin]) raises:
    var attempts = 0
    while True:
        var res = ring.poll_completion(out_tag_ptr.__int__())
        if res >= 0:
            if out_tag_ptr[0] == expected_tag:
                return
        attempts += 1
        if attempts > 1000000000:
            raise Error("Weights paging I/O timed out waiting for layer: " + String(expected_tag))

# -----------------------------------------------------------------------------
# 4. Standalone Test Executable & Story Generator
# -----------------------------------------------------------------------------
def main() raises:
    var num_tasks = num_physical_cores()
    print("=========================================================")
    print("   GEMMA 4 12B-SCALE DEEP TRANSCO-PROCESSOR RUNNER       ")
    print("=========================================================")
    print("Embedding Size:      3840")
    print("Attention Heads:     16 Query Heads / 8 KV Heads (GQA/MQA)")
    print("Model Parameters:    12B Scale Causal Layout")
    print("Kernel Paging I/O:   Linux io_uring (Direct NVMe double-buf)")
    print("---------------------------------------------------------")

    var CONFIG = TransformerConfig[
        embed_dim = 3840,
        head_dim = 256,
        num_heads = 16,
        num_kv_heads = 8,
        intermediate_dim = 15360,
        num_layers = 48,
        vocab_size = 262144,
        max_seq_len = 100
    ]()

    # 1. Initialize FFI and io_uring
    var libc = OwnedDLHandle("libc.so.6")
    var c_open = libc.get_function[def(Int, Int32, Int32) -> Int32]("open")
    var c_close = libc.get_function[def(Int32) -> Int32]("close")
    var posix_memalign = libc.get_function[def(Int, Int, Int) -> Int32]("posix_memalign")
    var free_c = libc.get_function[def(Int) -> NoneType]("free")
    var py = Python.import_module("time")
    var sys = Python.import_module("sys")
    # sys.path.append("/home/phil/git/llm-mojo/.venv/lib/python3.12/site-packages")
    var transformers = Python.import_module("transformers")
    print("Loading Gemma 4 tokenizer...")
    var tokenizer = transformers.AutoTokenizer.from_pretrained("google/gemma-4-12B-it")
    print("Tokenizer loaded successfully!")

    var ring = IoUring(16)
    var out_tag_ptr = alloc[UInt64](1)
    out_tag_ptr[0] = 0

    # 2. Find max layer size and Setup Page-Aligned Layer Weight Buffers (Double-buffering)
    var max_layer_bytes = 0
    for i in range(CONFIG.num_layers):
        var sz = CONFIG.get_layer_aligned_size(i)
        if sz > max_layer_bytes:
            max_layer_bytes = sz

    var memptr_w_a = alloc[Int](1)
    _ = posix_memalign(memptr_w_a.__int__(), 4096, max_layer_bytes)
    var w_a_addr = memptr_w_a[0]
    var w_a_ptr = UnsafePointer[Int8, MutExternalOrigin](unsafe_from_address=w_a_addr)

    var memptr_w_b = alloc[Int](1)
    _ = posix_memalign(memptr_w_b.__int__(), 4096, max_layer_bytes)
    var w_b_addr = memptr_w_b[0]
    var w_b_ptr = UnsafePointer[Int8, MutExternalOrigin](unsafe_from_address=w_b_addr)

    # 3. Open converted real weights file on NVMe with O_DIRECT
    var weights_file = "gemma4_weights_q8_sorted.bin"
    print("Opening converted weights file 'gemma4_weights_q8_sorted.bin' with O_DIRECT read-only...")
    
    var path_bytes = weights_file.as_bytes()
    var path_ptr = alloc[UInt8](len(path_bytes) + 1)
    for i in range(len(path_bytes)):
        path_ptr[i] = path_bytes[i]
    path_ptr[len(path_bytes)] = 0

    var weights_fd = c_open(path_ptr.__int__(), 16384, 0)
    
    if weights_fd < 0:
        print("CRITICAL: Failed to open 'gemma4_weights_q8_sorted.bin' weights file with O_DIRECT. Make sure you converted weights first!")
        return
    else:
        var total_size = CONFIG.get_layer_offset(CONFIG.num_layers)
        print("Weights file opened successfully. Active size: ", total_size, " bytes")

    # Embeddings (F32)
    var embed_file = "gemma4_embed.bin"
    var embed_path_bytes = embed_file.as_bytes()
    var embed_path_ptr = alloc[UInt8](len(embed_path_bytes) + 1)
    for i in range(len(embed_path_bytes)):
        embed_path_ptr[i] = embed_path_bytes[i]
    embed_path_ptr[len(embed_path_bytes)] = 0
    var embed_fd = c_open(embed_path_ptr.__int__(), 0, 0)
    if embed_fd < 0:
        print("CRITICAL: Failed to open 'gemma4_embed.bin' weights file!")
        return

    var embed_size_bytes = CONFIG.vocab_size * CONFIG.embed_dim * 4
    var memptr_embed = alloc[Int](1)
    _ = posix_memalign(memptr_embed.__int__(), 4096, embed_size_bytes)
    var embed_addr = memptr_embed[0]
    var embed_ptr = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=embed_addr)

    var c_read = libc.get_function[def(Int32, Int, Int) -> Int]("read")
    var total_read = 0
    while total_read < embed_size_bytes:
        var bytes_to_read = embed_size_bytes - total_read
        var read_res = c_read(embed_fd, embed_addr + total_read, bytes_to_read)
        if read_res <= 0:
            print("CRITICAL: Failed to read from gemma4_embed.bin!")
            return
        total_read += read_res
    print("Loaded embeddings into memory: " + String(total_read) + " bytes")
    print("---------------------------------------------------------")

    # 3c. Open and Load Real Final Layer Norm weights
    var final_norm_file = "gemma4_final_norm.bin"
    print("Opening final norm weights file 'gemma4_final_norm.bin'...")
    var final_norm_path_bytes = final_norm_file.as_bytes()
    var final_norm_path_ptr = alloc[UInt8](len(final_norm_path_bytes) + 1)
    for i in range(len(final_norm_path_bytes)):
        final_norm_path_ptr[i] = final_norm_path_bytes[i]
    final_norm_path_ptr[len(final_norm_path_bytes)] = 0

    var final_norm_fd = c_open(final_norm_path_ptr.__int__(), 0, 0)
    if final_norm_fd < 0:
        print("CRITICAL: Failed to open 'gemma4_final_norm.bin' weights file!")
        return

    var final_norm_weight = alloc[Float32](CONFIG.embed_dim)
    var final_norm_size_bytes = CONFIG.embed_dim * 4
    total_read = 0
    while total_read < final_norm_size_bytes:
        var bytes_to_read = final_norm_size_bytes - total_read
        var read_res = c_read(final_norm_fd, final_norm_weight.__int__() + total_read, bytes_to_read)
        if read_res <= 0:
            print("CRITICAL: Failed to read from gemma4_final_norm.bin!")
            return
        total_read += read_res
    print("Loaded final norm weights into memory: " + String(total_read) + " bytes")
    print("---------------------------------------------------------")
    _ = c_close(final_norm_fd)
    final_norm_path_ptr.free()

    # 4. Allocate Activations and Cache Buffers
    var x = alloc[Float32](CONFIG.embed_dim)
    var norm_x = alloc[Float32](CONFIG.embed_dim)
    var final_norm_x = alloc[Float32](CONFIG.embed_dim)
    # Use max possible heads (16 query heads with 512 dim, or 8 kv heads with 256 dim) for allocations
    var q = alloc[Float32](8192) # 16 heads * 512 dim
    var k_vec = alloc[Float32](2048) # 8 heads * 256 dim
    var v_vec = alloc[Float32](2048) # 8 heads * 256 dim
    var attn_scores = alloc[Float32](CONFIG.max_seq_len)
    var attn_out = alloc[Float32](8192) # 16 heads * 512 dim
    var proj_attn_out = alloc[Float32](CONFIG.embed_dim)
    var h1 = alloc[Float32](CONFIG.intermediate_dim)
    var h2 = alloc[Float32](CONFIG.intermediate_dim)
    var ffn_out_gate = alloc[Float32](CONFIG.intermediate_dim)
    var ffn_out = alloc[Float32](CONFIG.embed_dim)
    var x_old = alloc[Float32](CONFIG.embed_dim)
    var common_tokens = [
        108, 109, 110, 111,
        235248, 235265, 235292, 235276, 235284,
        476, 502, 518, 544, 570, 601, 612, 634, 642, 673, 696, 713, 728, 759, 798, 811
    ]

    # Precompute layer cache offsets and total cache size
    var layer_cache_offsets = List[Int]()
    var current_cache_offset = 0
    for i in range(CONFIG.num_layers):
        layer_cache_offsets.append(current_cache_offset)
        var is_global = CONFIG.is_global_layer(i)
        var layer_kv_heads = 1 if is_global else 8
        var layer_head_dim = 512 if is_global else 256
        current_cache_offset += CONFIG.max_seq_len * layer_kv_heads * layer_head_dim
    var total_cache_size = current_cache_offset

    # Global continuous KV-cache
    var k_cache = alloc[Float32](total_cache_size)
    var v_cache = alloc[Float32](total_cache_size)
    # Zero out caches
    for i in range(total_cache_size):
        k_cache[i] = 0.0
        v_cache[i] = 0.0

    # 5. Interactive Chat Agent Loop
    print("=========================================================")
    print("   GEMMA 4 12B-SCALE DEEP TRANSCO-PROCESSOR CHAT AGENT   ")
    print("=========================================================")
    print("Embedding Size:      3840")
    print("Attention Heads:     16 Query Heads / 8 KV Heads (GQA/MQA)")
    print("Model Parameters:    12B Scale Q8_0 Layout")
    print("Kernel Paging I/O:   Linux io_uring (Direct NVMe double-buf)")
    print("---------------------------------------------------------")
    print("(Type 'exit' or 'quit' to end the conversation)")

    var py_builtins = Python.import_module("builtins")
    var list_cls = py_builtins.list

    while True:
        print()
        var user_input = py_builtins.input("User > ")
        var user_input_str = String(user_input).strip()
        if user_input_str == "exit" or user_input_str == "quit":
            break
            
        var formatted_prompt = "<bos><|turn>user\n" + user_input_str + "<turn|>\n<|turn>model\n"
        var encoded = tokenizer.encode(formatted_prompt, add_special_tokens=False)
        var num_encoded = len(encoded)
        if num_encoded == 0:
            print("Agent > (Please type something!)")
            continue
            
        print("[Tokenizer] Encoded input into", num_encoded, "real sub-word token IDs:")
        var token_ids_str = String("")
        var prompt_tokens = List[Int]()
        for i in range(num_encoded):
            var tid = Int(py=encoded[i])
            token_ids_str += String(tid) + " "
            prompt_tokens.append(tid)
        print(token_ids_str)
            
        print("Agent > ", end="")
        
        var current_token: Int
        var num_prompt_tokens = len(prompt_tokens)
        var max_steps = num_prompt_tokens + 30
        if max_steps > CONFIG.max_seq_len:
            max_steps = CONFIG.max_seq_len
            
        var scale = sqrt(Float32(CONFIG.embed_dim))
        
        # 1. Prefill phase: process all prompt tokens layer-by-layer
        # This streams/pages each layer weights from NVMe EXACTLY ONCE!
        var t_prefill_start = py.time()
        var prompt_x = alloc[Float32](num_prompt_tokens * CONFIG.embed_dim)
        for pos in range(num_prompt_tokens):
            var token_emb_offset = prompt_tokens[pos] * CONFIG.embed_dim
            var px_offset = pos * CONFIG.embed_dim
            for i in range(CONFIG.embed_dim):
                prompt_x[px_offset + i] = embed_ptr[token_emb_offset + i] * scale
                
        # Load Layer 0 weights before the layer loop (stable base only)
        var tag_0: UInt64 = UInt64(8000)
        var ret_val = ring.submit_read(weights_fd, w_a_ptr.__int__(), CONFIG.get_w2_offset_in_layer(0), 0, tag_0)
        if ret_val < 0:
            print("CRITICAL: submit_read failed with error: ", ret_val)
        wait_for_tag(ring, tag_0, out_tag_ptr)

        for layer_idx in range(CONFIG.num_layers):
            var active_weights_ptr = w_a_ptr if layer_idx % 2 == 0 else w_b_ptr
            var next_weights_ptr = w_b_ptr if layer_idx % 2 == 0 else w_a_ptr
            
            # Prefetch next layer's weights (stable base only!)
            if layer_idx + 1 < CONFIG.num_layers:
                var tag_next: UInt64 = UInt64(8000 + layer_idx + 1)
                var offset_next = CONFIG.get_layer_offset(layer_idx + 1)
                var size_next = CONFIG.get_w2_offset_in_layer(layer_idx + 1)
                _ = ring.submit_read(weights_fd, next_weights_ptr.__int__(), size_next, offset_next, tag_next)
                
            var weights = CONFIG.get_layer_weights(layer_idx, active_weights_ptr)
            var is_global = CONFIG.is_global_layer(layer_idx)
            var layer_heads = 16
            var layer_kv_heads = 1 if is_global else 8
            var layer_head_dim = 512 if is_global else 256
            var kv_stride = layer_kv_heads * layer_head_dim
            var ratio = layer_heads // layer_kv_heads
            
            var next_layer_completed = False
            
            # Loop over all prompt positions
            for pos in range(num_prompt_tokens):
                var px_offset = pos * CONFIG.embed_dim
                var local_x = prompt_x + px_offset
                
                # Setup norm_x
                rms_norm(local_x, weights.input_layernorm_weight, norm_x, CONFIG.embed_dim)
                
                # Compute q, k
                q8_matmul_vec(weights.wq_packed, weights.wq_scales, norm_x, q, CONFIG.embed_dim, layer_heads * layer_head_dim, num_tasks)
                q8_matmul_vec(weights.wk_packed, weights.wk_scales, norm_x, k_vec, CONFIG.embed_dim, layer_kv_heads * layer_head_dim, num_tasks)
                
                if is_global:
                    for i in range(kv_stride):
                        v_vec[i] = k_vec[i]
                else:
                    q8_matmul_vec(weights.wv_packed, weights.wv_scales, norm_x, v_vec, CONFIG.embed_dim, layer_kv_heads * layer_head_dim, num_tasks)
                    
                # Apply head-wise RMS normalization
                for h in range(layer_heads):
                    var head_ptr = q + h * layer_head_dim
                    rms_norm(head_ptr, weights.q_norm_weight, head_ptr, layer_head_dim)
                for h in range(layer_kv_heads):
                    var head_ptr = k_vec + h * layer_head_dim
                    rms_norm(head_ptr, weights.k_norm_weight, head_ptr, layer_head_dim)
                for h in range(layer_kv_heads):
                    var head_ptr = v_vec + h * layer_head_dim
                    rms_norm_noscale(head_ptr, head_ptr, layer_head_dim)
                    
                apply_rope(q, layer_heads, layer_head_dim, pos, is_global)
                apply_rope(k_vec, layer_kv_heads, layer_head_dim, pos, is_global)
                
                var layer_cache_offset = layer_cache_offsets[layer_idx]
                var step_cache_offset = layer_cache_offset + pos * kv_stride
                for i in range(kv_stride):
                    k_cache[step_cache_offset + i] = k_vec[i]
                    v_cache[step_cache_offset + i] = v_vec[i]
                    
                for h in range(layer_heads):
                    var h_offset = h * layer_head_dim
                    var g = h // ratio
                    var g_offset = g * layer_head_dim
                    
                    var scale = 1.0 / sqrt(Float32(layer_head_dim))
                    for t in range(pos + 1):
                        var t_cache_offset = layer_cache_offset + t * kv_stride + g_offset
                        var score: Float32 = 0.0
                        for d in range(layer_head_dim):
                            score += q[h_offset + d] * k_cache[t_cache_offset + d]
                        attn_scores[t] = score
                        
                    softmax(attn_scores, pos + 1)
                    
                    for d in range(layer_head_dim):
                        var val_sum: Float32 = 0.0
                        for t in range(pos + 1):
                            var t_cache_offset = layer_cache_offset + t * kv_stride + g_offset
                            val_sum += attn_scores[t] * v_cache[t_cache_offset + d]
                        attn_out[h_offset + d] = val_sum
                        
                q8_matmul_vec(weights.wo_packed, weights.wo_scales, attn_out, proj_attn_out, layer_heads * layer_head_dim, CONFIG.embed_dim, num_tasks)
                
                # Post-attention normalization on projected attention output
                rms_norm(proj_attn_out, weights.post_attention_layernorm_weight, proj_attn_out, CONFIG.embed_dim)
                for i in range(CONFIG.embed_dim):
                    local_x[i] += proj_attn_out[i]
                    
                # Pre-feedforward normalization
                rms_norm(local_x, weights.pre_feedforward_layernorm_weight, norm_x, CONFIG.embed_dim)
                q8_matmul_vec(weights.w1_packed, weights.w1_scales, norm_x, h1, CONFIG.embed_dim, CONFIG.intermediate_dim, num_tasks)
                q8_matmul_vec(weights.w3_packed, weights.w3_scales, norm_x, h2, CONFIG.embed_dim, CONFIG.intermediate_dim, num_tasks)
                gelu_and_mul(h1, h2, ffn_out_gate, CONFIG.intermediate_dim)
                
                # Contiguous sequential W2 paging via io_uring (prefix reading)
                var num_blocks = CONFIG.intermediate_dim // 32
                var layer_offset = CONFIG.get_layer_offset(layer_idx)
                var w2_offset_in_layer = CONFIG.get_w2_offset_in_layer(layer_idx)
                var tag_base_for_w2: UInt64 = 100000 + UInt64(layer_idx) * 1000 + UInt64(pos) * 1000000
                
                # Scan from right to left to find highest active block N
                var max_active_block: Int = -1
                for b in range(num_blocks - 1, -1, -1):
                    var block_offset_gate = b * 32
                    var block_active = False
                    for c in range(32):
                        if abs(ffn_out_gate[block_offset_gate + c]) > 1e-4:
                            block_active = True
                            break
                    if block_active:
                        max_active_block = b
                        break
                        
                var num_active = 0
                if max_active_block >= 0:
                    num_active = 1
                    var size_to_read = (max_active_block + 1) * 139264
                    var block_offset_in_file = layer_offset + w2_offset_in_layer
                    var dest_mem_addr = weights.w2_packed.__int__()
                    _ = ring.submit_read(weights_fd, dest_mem_addr, size_to_read, block_offset_in_file, tag_base_for_w2)
                    
                # Wait for all active W2 block reads to complete
                var completed_active = 0
                var attempts = 0
                var tag_next: UInt64 = UInt64(8000 + layer_idx + 1)
                
                while completed_active < num_active:
                    var res = ring.poll_completion(out_tag_ptr.__int__())
                    if res >= 0:
                        var t = out_tag_ptr[0]
                        if t == tag_base_for_w2:
                            completed_active += 1
                        elif t == tag_next:
                            next_layer_completed = True
                    attempts += 1
                    if attempts > 1000000000:
                        raise Error("Prefill weights paging I/O timed out waiting for sparse W2 blocks")
                        
                q8_matmul_w2_block_column(weights.w2_packed, ffn_out_gate, ffn_out, CONFIG.intermediate_dim, CONFIG.embed_dim, num_tasks)
                
                # Post-feedforward normalization on MLP output
                rms_norm(ffn_out, weights.post_feedforward_layernorm_weight, ffn_out, CONFIG.embed_dim)
                for i in range(CONFIG.embed_dim):
                    local_x[i] += ffn_out[i]
                    
                # Scale hidden state by layer_scalar
                var scalar = weights.layer_scalar[0]
                for i in range(CONFIG.embed_dim):
                    local_x[i] *= scalar

            if layer_idx + 1 < CONFIG.num_layers:
                if not next_layer_completed:
                    var tag_next_val: UInt64 = UInt64(8000 + layer_idx + 1)
                    wait_for_tag(ring, tag_next_val, out_tag_ptr)

        # Copy the last prompt token's hidden state to x
        var last_px_offset = (num_prompt_tokens - 1) * CONFIG.embed_dim
        for i in range(CONFIG.embed_dim):
            x[i] = prompt_x[last_px_offset + i]
        prompt_x.free()
        
        var t_prefill_end = py.time()
        var prefill_time = Float64(py=(t_prefill_end - t_prefill_start))
        print("\n[Mojo Bench] Prefill completed in", prefill_time, "sec (" + String(Float64(num_prompt_tokens) / prefill_time) + " t/s)")
        
        # 2. Setup parallel projection worker and select first generated token
        rms_norm(x, final_norm_weight, final_norm_x, CONFIG.embed_dim)

        # Uses the pre-detected core count (num_tasks) to compute chunk_size for vocabulary projection
        var chunk_size = CONFIG.vocab_size // num_tasks
        
        var local_max_logits = alloc[Float32](num_tasks)
        var local_next_tokens = alloc[Int](num_tasks)
        
        def proj_worker(t: Int) capturing -> None:
            var start_r = t * chunk_size
            var end_r = (t + 1) * chunk_size if t < num_tasks - 1 else CONFIG.vocab_size
            
            var local_max_l: Float32 = -999999.0
            var local_tok = 0
            
            for r in range(start_r, end_r):
                var r_offset = r * CONFIG.embed_dim
                var acc = SIMD[DType.float32, 8](0.0)
                for c in range(0, CONFIG.embed_dim, 8):
                    acc = acc + final_norm_x.load[8](c) * embed_ptr.load[8](r_offset + c)
                var logit = acc.reduce_add()
                
                # Apply logit soft-capping (30.0 for Gemma 4)
                logit = Float32(30.0) * fast_tanh(logit / Float32(30.0))
                
                if logit > local_max_l:
                    local_max_l = logit
                    local_tok = r
                    
            local_max_logits[t] = local_max_l
            local_next_tokens[t] = local_tok
            
        parallelize[proj_worker](num_tasks)
        
        # Reduce to find global max
        var max_logit: Float32 = -999999.0
        var next_token = 0
        for t in range(num_tasks):
            if local_max_logits[t] > max_logit:
                max_logit = local_max_logits[t]
                next_token = local_next_tokens[t]
                
        # Print the first generated token
        var next_token_list = list_cls()
        _ = next_token_list.append(next_token)
        var decoded = tokenizer.decode(next_token_list)
        py_builtins.print(decoded, end="", flush=True)
        
        current_token = next_token
        
        # 3. Subsequent generation steps
        var tokens_generated = 1
        var early_exits_l12 = 0
        var early_exits_l24 = 0
        var sum_cos_l12: Float32 = 0.0
        var count_l12 = 0
        var min_cos_l12: Float32 = 1.0
        var max_cos_l12: Float32 = -1.0
        var sum_cos_l24: Float32 = 0.0
        var count_l24 = 0
        var min_cos_l24: Float32 = 1.0
        var max_cos_l24: Float32 = -1.0
        var t_gen_start = py.time()
        for pos in range(num_prompt_tokens, max_steps):
            # Layer 0 weight prefetch (stable base only!)
            var tag_0: UInt64 = UInt64(2000)
            _ = ring.submit_read(weights_fd, w_a_ptr.__int__(), CONFIG.get_w2_offset_in_layer(0), 0, tag_0)
            wait_for_tag(ring, tag_0, out_tag_ptr)
            
            # Setup input activations x with the dense embedding of current_token (scaled)
            var token_emb_offset = current_token * CONFIG.embed_dim
            for i in range(CONFIG.embed_dim):
                x[i] = embed_ptr[token_emb_offset + i] * scale
                
            # Run the 40 layers of Transformer forward pass
            var early_exit_triggered = False
            var early_exit_layer = CONFIG.num_layers
            for layer_idx in range(CONFIG.num_layers):
                if layer_idx == 12 or layer_idx == 24:
                    for i in range(CONFIG.embed_dim):
                        x_old[i] = x[i]
                var active_weights_ptr = w_a_ptr if layer_idx % 2 == 0 else w_b_ptr
                var next_weights_ptr = w_b_ptr if layer_idx % 2 == 0 else w_a_ptr
                
                # Prefetch next layer's weights (stable base only!)
                if layer_idx + 1 < CONFIG.num_layers:
                    var tag_next: UInt64 = UInt64(3000 + pos * 10 + layer_idx + 1)
                    var offset_next = CONFIG.get_layer_offset(layer_idx + 1)
                    var size_next = CONFIG.get_w2_offset_in_layer(layer_idx + 1)
                    _ = ring.submit_read(weights_fd, next_weights_ptr.__int__(), size_next, offset_next, tag_next)
                    
                var weights = CONFIG.get_layer_weights(layer_idx, active_weights_ptr)
                var is_global = CONFIG.is_global_layer(layer_idx)
                var next_layer_completed = False
                rms_norm(x, weights.input_layernorm_weight, norm_x, CONFIG.embed_dim)
                
                var layer_heads = 16
                var layer_kv_heads = 1 if is_global else 8
                var layer_head_dim = 512 if is_global else 256
                var kv_stride = layer_kv_heads * layer_head_dim
                
                q8_matmul_vec(weights.wq_packed, weights.wq_scales, norm_x, q, CONFIG.embed_dim, layer_heads * layer_head_dim, num_tasks)
                q8_matmul_vec(weights.wk_packed, weights.wk_scales, norm_x, k_vec, CONFIG.embed_dim, layer_kv_heads * layer_head_dim, num_tasks)
                
                if is_global:
                    for i in range(kv_stride):
                        v_vec[i] = k_vec[i]
                else:
                    q8_matmul_vec(weights.wv_packed, weights.wv_scales, norm_x, v_vec, CONFIG.embed_dim, layer_kv_heads * layer_head_dim, num_tasks)
                    
                # Apply head-wise RMS normalization
                for h in range(layer_heads):
                    var head_ptr = q + h * layer_head_dim
                    rms_norm(head_ptr, weights.q_norm_weight, head_ptr, layer_head_dim)
                for h in range(layer_kv_heads):
                    var head_ptr = k_vec + h * layer_head_dim
                    rms_norm(head_ptr, weights.k_norm_weight, head_ptr, layer_head_dim)
                for h in range(layer_kv_heads):
                    var head_ptr = v_vec + h * layer_head_dim
                    rms_norm_noscale(head_ptr, head_ptr, layer_head_dim)
                    
                apply_rope(q, layer_heads, layer_head_dim, pos, is_global)
                apply_rope(k_vec, layer_kv_heads, layer_head_dim, pos, is_global)
                
                var layer_cache_offset = layer_cache_offsets[layer_idx]
                var step_cache_offset = layer_cache_offset + pos * kv_stride
                for i in range(kv_stride):
                    k_cache[step_cache_offset + i] = k_vec[i]
                    v_cache[step_cache_offset + i] = v_vec[i]
                    
                var ratio = layer_heads // layer_kv_heads
                for h in range(layer_heads):
                    var h_offset = h * layer_head_dim
                    var g = h // ratio
                    var g_offset = g * layer_head_dim
                    
                    var scale = 1.0 / sqrt(Float32(layer_head_dim))
                    for t in range(pos + 1):
                        var t_cache_offset = layer_cache_offset + t * kv_stride + g_offset
                        var score: Float32 = 0.0
                        for d in range(layer_head_dim):
                            score += q[h_offset + d] * k_cache[t_cache_offset + d]
                        attn_scores[t] = score
                        
                    softmax(attn_scores, pos + 1)
                    
                    for d in range(layer_head_dim):
                        var val_sum: Float32 = 0.0
                        for t in range(pos + 1):
                            var t_cache_offset = layer_cache_offset + t * kv_stride + g_offset
                            val_sum += attn_scores[t] * v_cache[t_cache_offset + d]
                        attn_out[h_offset + d] = val_sum
                        
                q8_matmul_vec(weights.wo_packed, weights.wo_scales, attn_out, proj_attn_out, layer_heads * layer_head_dim, CONFIG.embed_dim, num_tasks)
                
                # Post-attention normalization on projected attention output
                rms_norm(proj_attn_out, weights.post_attention_layernorm_weight, proj_attn_out, CONFIG.embed_dim)
                for i in range(CONFIG.embed_dim):
                    x[i] += proj_attn_out[i]
                    
                # Pre-feedforward normalization
                rms_norm(x, weights.pre_feedforward_layernorm_weight, norm_x, CONFIG.embed_dim)
                q8_matmul_vec(weights.w1_packed, weights.w1_scales, norm_x, h1, CONFIG.embed_dim, CONFIG.intermediate_dim, num_tasks)
                q8_matmul_vec(weights.w3_packed, weights.w3_scales, norm_x, h2, CONFIG.embed_dim, CONFIG.intermediate_dim, num_tasks)
                gelu_and_mul(h1, h2, ffn_out_gate, CONFIG.intermediate_dim)
                
                # Contiguous sequential W2 paging via io_uring (prefix reading)
                var num_blocks = CONFIG.intermediate_dim // 32
                var layer_offset = CONFIG.get_layer_offset(layer_idx)
                var w2_offset_in_layer = CONFIG.get_w2_offset_in_layer(layer_idx)
                var tag_base_for_w2: UInt64 = 200000 + UInt64(layer_idx) * 1000
                
                # Scan from right to left to find highest active block N
                var max_active_block: Int = -1
                for b in range(num_blocks - 1, -1, -1):
                    var block_offset_gate = b * 32
                    var block_active = False
                    for c in range(32):
                        if abs(ffn_out_gate[block_offset_gate + c]) > 1e-4:
                            block_active = True
                            break
                    if block_active:
                        max_active_block = b
                        break
                        
                var num_active = 0
                if max_active_block >= 0:
                    num_active = 1
                    var size_to_read = (max_active_block + 1) * 139264
                    var block_offset_in_file = layer_offset + w2_offset_in_layer
                    var dest_mem_addr = weights.w2_packed.__int__()
                    _ = ring.submit_read(weights_fd, dest_mem_addr, size_to_read, block_offset_in_file, tag_base_for_w2)
                    
                # Wait for all active W2 block reads to complete
                var completed_active = 0
                var attempts = 0
                var tag_next: UInt64 = UInt64(3000 + pos * 10 + layer_idx + 1)
                
                while completed_active < num_active:
                    var res = ring.poll_completion(out_tag_ptr.__int__())
                    if res >= 0:
                        var t = out_tag_ptr[0]
                        if t == tag_base_for_w2:
                            completed_active += 1
                        elif t == tag_next:
                            next_layer_completed = True
                    attempts += 1
                    if attempts > 1000000000:
                        raise Error("Decode weights paging I/O timed out waiting for sparse W2 blocks")
                        
                q8_matmul_w2_block_column(weights.w2_packed, ffn_out_gate, ffn_out, CONFIG.intermediate_dim, CONFIG.embed_dim, num_tasks)
                
                # Post-feedforward normalization on MLP output
                rms_norm(ffn_out, weights.post_feedforward_layernorm_weight, ffn_out, CONFIG.embed_dim)
                for i in range(CONFIG.embed_dim):
                    x[i] += ffn_out[i]
                    
                # Scale hidden state by layer_scalar
                var scalar = weights.layer_scalar[0]
                for i in range(CONFIG.embed_dim):
                    x[i] *= scalar
                    
                # Unified Early-Exit Depth Skipping Check (Heuristic C via scale-invariant Cosine Similarity)
                if layer_idx == 12 or layer_idx == 24:
                    var dot_vec = SIMD[DType.float32, 16](0.0)
                    var sq_x_vec = SIMD[DType.float32, 16](0.0)
                    var sq_x_old_vec = SIMD[DType.float32, 16](0.0)
                    
                    for c in range(0, CONFIG.embed_dim, 16):
                        var xv = x.load[16](c)
                        var xov = x_old.load[16](c)
                        dot_vec += xv * xov
                        sq_x_vec += xv * xv
                        sq_x_old_vec += xov * xov
                        
                    var dot_prod = dot_vec.reduce_add()
                    var sum_sq_x = sq_x_vec.reduce_add()
                    var sum_sq_x_old = sq_x_old_vec.reduce_add()
                    
                    var norm_x = sqrt(sum_sq_x)
                    var norm_x_old = sqrt(sum_sq_x_old)
                    var cosine_sim = dot_prod / (norm_x * norm_x_old + 1e-9)
                    
                    # Update real-time stats
                    if layer_idx == 12:
                        sum_cos_l12 += cosine_sim
                        count_l12 += 1
                        if cosine_sim < min_cos_l12: min_cos_l12 = cosine_sim
                        if cosine_sim > max_cos_l12: max_cos_l12 = cosine_sim
                    else:
                        sum_cos_l24 += cosine_sim
                        count_l24 += 1
                        if cosine_sim < min_cos_l24: min_cos_l24 = cosine_sim
                        if cosine_sim > max_cos_l24: max_cos_l24 = cosine_sim
                        
                    var threshold_extreme = Float32(1.01)
                    var threshold_moderate = Float32(1.01)



                    
                    var exit_now = False
                    if cosine_sim > threshold_extreme:
                        exit_now = True
                    elif cosine_sim > threshold_moderate:
                        # Run sub-sampled projection head over common tokens
                        rms_norm(x, final_norm_weight, final_norm_x, CONFIG.embed_dim)
                        
                        var max_sub_logit: Float32 = -999999.0
                        var sub_logits = List[Float32]()
                        for i in range(len(common_tokens)):
                            var r = common_tokens[i]
                            var r_offset = r * CONFIG.embed_dim
                            var acc = SIMD[DType.float32, 8](0.0)
                            for c in range(0, CONFIG.embed_dim, 8):
                                acc = acc + final_norm_x.load[8](c) * embed_ptr.load[8](r_offset + c)
                            var logit = acc.reduce_add()
                            logit = Float32(30.0) * fast_tanh(logit / Float32(30.0))
                            sub_logits.append(logit)
                            if logit > max_sub_logit:
                                max_sub_logit = logit
                                
                        var sum_exp_sub: Float32 = 0.0
                        for i in range(len(sub_logits)):
                            sum_exp_sub += exp(sub_logits[i] - max_sub_logit)
                            
                        var top_prob = 1.0 / sum_exp_sub
                        if top_prob > 0.90:
                            exit_now = True
                            
                    if exit_now:
                        if layer_idx == 12:
                            early_exits_l12 += 1
                        else:
                            early_exits_l24 += 1
                        early_exit_triggered = True
                        early_exit_layer = layer_idx
                        
                        # Wait for the pending prefetch of layer_idx + 1 to complete so io_uring is clean!
                        if layer_idx + 1 < CONFIG.num_layers and not next_layer_completed:
                            var tag_next_val: UInt64 = UInt64(3000 + pos * 10 + layer_idx + 1)
                            wait_for_tag(ring, tag_next_val, out_tag_ptr)
                            
                        # Recover/Propagate KV cache to skipped layers from position pos - 1
                        for skipped_layer in range(layer_idx + 1, CONFIG.num_layers):
                            var is_gl = CONFIG.is_global_layer(skipped_layer)
                            var l_kv_heads = 1 if is_gl else 8
                            var l_head_dim = 512 if is_gl else 256
                            var l_stride = l_kv_heads * l_head_dim
                            var l_cache_offset = layer_cache_offsets[skipped_layer]
                            var s_cache_offset = l_cache_offset + pos * l_stride
                            var p_cache_offset = l_cache_offset + (pos - 1) * l_stride
                            for i in range(l_stride):
                                k_cache[s_cache_offset + i] = k_cache[p_cache_offset + i]
                                v_cache[s_cache_offset + i] = v_cache[p_cache_offset + i]
                        break
                    
                if layer_idx + 1 < CONFIG.num_layers:
                    if not next_layer_completed:
                        var tag_next_val: UInt64 = UInt64(3000 + pos * 10 + layer_idx + 1)
                        wait_for_tag(ring, tag_next_val, out_tag_ptr)
                    
            # Predict the next token
            rms_norm(x, final_norm_weight, final_norm_x, CONFIG.embed_dim)
            
            parallelize[proj_worker](num_tasks)
            
            # Reduce to find global max
            max_logit = -999999.0
            next_token = 0
            for t in range(num_tasks):
                if local_max_logits[t] > max_logit:
                    max_logit = local_max_logits[t]
                    next_token = local_next_tokens[t]
                    
            # Print the generated token in real-time
            var next_token_list_inner = list_cls()
            _ = next_token_list_inner.append(next_token)
            var word = String(tokenizer.decode(next_token_list_inner))
            py_builtins.print(word, end="", flush=True)
            
            if next_token == 1 or next_token == 106: # <eos> or <turn|>
                print()
                break
                
            current_token = next_token
            tokens_generated += 1
            
        var t_gen_end = py.time()
        var gen_time = Float64(py=(t_gen_end - t_gen_start))
        print("\n\n========================================= EARLY-EXIT DIAGNOSTICS =========================================")
        print("  Total Tokens Generated:        ", tokens_generated)
        print("  Early Exits at Layer 12 (32%):  ", early_exits_l12, " (", String(Float64(early_exits_l12) * 100.0 / Float64(tokens_generated)), "%)")
        print("  Early Exits at Layer 24 (62%):  ", early_exits_l24, " (", String(Float64(early_exits_l24) * 100.0 / Float64(tokens_generated)), "%)")
        var avg_layers = (Float64(early_exits_l12) * 13.0 + Float64(early_exits_l24) * 25.0 + Float64(tokens_generated - early_exits_l12 - early_exits_l24) * 40.0) / Float64(tokens_generated)
        print("  Average Layers Executed:       ", avg_layers, " / 40 (", String(avg_layers * 100.0 / 40.0), "%)")
        var bandwidth_saved = (1.0 - (avg_layers / 40.0)) * 100.0
        print("  Estimated DDR Bandwidth Saved:  ", bandwidth_saved, "%")
        print("  --------------------------------------------------------------------------------------------------------")
        var avg_cos_l12 = sum_cos_l12 / Float32(count_l12) if count_l12 > 0 else Float32(0.0)
        var avg_cos_l24 = sum_cos_l24 / Float32(count_l24) if count_l24 > 0 else Float32(0.0)
        print("  Layer 12 Cosine Similarity:     Min: ", String(min_cos_l12 if count_l12 > 0 else Float32(0.0)), " | Max: ", String(max_cos_l12 if count_l12 > 0 else Float32(0.0)), " | Avg: ", String(avg_cos_l12))
        print("  Layer 24 Cosine Similarity:     Min: ", String(min_cos_l24 if count_l24 > 0 else Float32(0.0)), " | Max: ", String(max_cos_l24 if count_l24 > 0 else Float32(0.0)), " | Avg: ", String(avg_cos_l24))
        print("==========================================================================================================")
        print("[Mojo Bench] Generation completed in", gen_time, "sec (" + String(Float64(tokens_generated) / gen_time) + " t/s)")
        
        local_max_logits.free()
        local_next_tokens.free()

    print("\n---------------------------------------------------------")
    print("Inference generation loop complete!")

    # Cleanup open file handles and memory
    free_c(w_a_addr)
    free_c(w_b_addr)
    free_c(embed_addr)
    memptr_w_a.free()
    memptr_w_b.free()
    memptr_embed.free()
    out_tag_ptr.free()
    path_ptr.free()
    embed_path_ptr.free()
    _ = c_close(weights_fd)
    _ = c_close(embed_fd)

    # Deallocate activation memory pointers
    x.free()
    norm_x.free()
    final_norm_x.free()
    final_norm_weight.free()
    q.free()
    k_vec.free()
    v_vec.free()
    attn_scores.free()
    attn_out.free()
    proj_attn_out.free()
    h1.free()
    h2.free()
    ffn_out_gate.free()
    ffn_out.free()
    x_old.free()
    k_cache.free()
    v_cache.free()
