from std.ffi import OwnedDLHandle
from std.memory import alloc
from std.math import sqrt, exp, cos, sin
from std.python import Python
import std.os
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
        var wv_packed = wk_packed + wk_size_bytes
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
        var wv_scales = wk_scales + c_blocks_k
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
def q8_matmul_w2_block_column(
    W_packed: UnsafePointer[Int8, _],
    x: UnsafePointer[Float32, _],
    out_ptr: UnsafePointer[Float32, MutExternalOrigin],
    c_in: Int,
    c_out: Int,
    num_tasks: Int,
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
            
            var v_real = vec[h_offset + i]
            var v_imag = vec[h_offset + half_dim + i]
            
            vec[h_offset + i] = v_real * cos_theta - v_imag * sin_theta
            vec[h_offset + half_dim + i] = v_real * sin_theta + v_imag * cos_theta

@always_inline
def softmax(x: UnsafePointer[Float32, MutExternalOrigin], size: Int):
    var max_val = x[0]
    for i in range(1, size):
        if x[i] > max_val:
            max_val = x[i]
            
    var sum: Float32 = 0.0
    for i in range(size):
        x[i] = exp(x[i] - max_val)
        sum += x[i]
        
    for i in range(size):
        x[i] /= sum

# -----------------------------------------------------------------------------
# 4. Main Entry Point (Memory Resident Runner)
# -----------------------------------------------------------------------------
def main() raises:
    var num_tasks = num_physical_cores()
    if num_tasks > 6:
        num_tasks = 6
    print("=========================================================")
    print("   GEMMA 4 12B-SCALE CPU MEM-RESIDENT RUNNER             ")
    print("=========================================================")
    print("Embedding Size:      3840")
    print("Attention Heads:     16 Query Heads / 8 KV Heads (GQA/MQA)")
    print("Model Parameters:    12B Scale Causal Layout")
    print("Weight Memory Model: Contiguous System RAM Allocation")
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

    # 1. Initialize FFI
    var libc = OwnedDLHandle("libc.so.6")
    var c_open = libc.get_function[def(Int, Int32, Int32) -> Int32]("open")
    var c_close = libc.get_function[def(Int32) -> Int32]("close")
    var posix_memalign = libc.get_function[def(Int, Int, Int) -> Int32]("posix_memalign")
    var free_c = libc.get_function[def(Int) -> NoneType]("free")
    var py = Python.import_module("time")
    var sys = Python.import_module("sys")
    sys.path.append("/home/phil/git/llm-mojo/.venv/lib/python3.12/site-packages")
    var transformers = Python.import_module("transformers")
    print("Loading Gemma 4 tokenizer...")
    var tokenizer = transformers.AutoTokenizer.from_pretrained("google/gemma-4-12B-it")
    print("Tokenizer loaded successfully!")

    # 2. Allocate and load the ENTIRE quantized Q8 weights into RAM
    var total_size = CONFIG.get_layer_offset(CONFIG.num_layers)
    print("Allocating memory for 12.29 GB Q8 weights in contiguous RAM...")
    var memptr_weights = alloc[Int](1)
    var alloc_res = posix_memalign(memptr_weights.__int__(), 4096, total_size)
    if alloc_res != 0:
        print("CRITICAL: Failed to allocate aligned memory for weights!")
        return
        
    var weights_addr = memptr_weights[0]
    var weights_ram_ptr = UnsafePointer[Int8, MutExternalOrigin](unsafe_from_address=weights_addr)

    var weights_file = "gemma4_weights_q8_sorted.bin"
    print("Opening weights file 'gemma4_weights_q8_sorted.bin'...")
    var path_bytes = weights_file.as_bytes()
    var path_ptr = alloc[UInt8](len(path_bytes) + 1)
    for i in range(len(path_bytes)):
        path_ptr[i] = path_bytes[i]
    path_ptr[len(path_bytes)] = 0
    var weights_fd = c_open(path_ptr.__int__(), 0, 0) # 0 is O_RDONLY
    if weights_fd < 0:
        print("CRITICAL: Failed to open 'gemma4_weights_q8_sorted.bin'!")
        return

    var c_read = libc.get_function[def(Int32, Int, Int) -> Int]("read")
    var total_read = 0
    print("Loading weights into system RAM...")
    var t_load_start = py.time()
    while total_read < total_size:
        var bytes_to_read = total_size - total_read
        if bytes_to_read > 1024 * 1024 * 1024:
            bytes_to_read = 1024 * 1024 * 1024 # 1GB chunk
        var read_res = c_read(weights_fd, weights_addr + total_read, bytes_to_read)
        if read_res <= 0:
            print("CRITICAL: Failed to read from weights file!")
            return
        total_read += read_res
        print(" -> Loaded ", total_read // (1024*1024), " MB / ", total_size // (1024*1024), " MB")
        
    var t_load_end = py.time()
    print("Weights loaded into RAM successfully in " + String(Float64(py=(t_load_end - t_load_start))) + " sec!")
    _ = c_close(weights_fd)
    path_ptr.free()

    # 3. Load embeddings
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

    total_read = 0
    while total_read < embed_size_bytes:
        var bytes_to_read = embed_size_bytes - total_read
        var read_res = c_read(embed_fd, embed_addr + total_read, bytes_to_read)
        if read_res <= 0:
            print("CRITICAL: Failed to read from gemma4_embed.bin!")
            return
        total_read += read_res
    print("Loaded embeddings into memory: " + String(total_read) + " bytes")
    _ = c_close(embed_fd)
    embed_path_ptr.free()

    # 4. Open and Load Real Final Layer Norm weights
    var final_norm_file = "gemma4_final_norm.bin"
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

    # 5. Allocate Activations and Cache Buffers
    var x = alloc[Float32](CONFIG.embed_dim)
    var norm_x = alloc[Float32](CONFIG.embed_dim)
    var final_norm_x = alloc[Float32](CONFIG.embed_dim)
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
    for i in range(total_cache_size):
        k_cache[i] = 0.0
        v_cache[i] = 0.0

    # 6. Interactive Chat Agent Loop
    print("=========================================================")
    print("   GEMMA 4 12B-SCALE MEMORY-RESIDENT CHAT AGENT          ")
    print("=========================================================")
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
        
        # Prefill phase: process all prompt tokens
        var t_prefill_start = py.time()
        var prompt_x = alloc[Float32](num_prompt_tokens * CONFIG.embed_dim)
        for pos in range(num_prompt_tokens):
            var token_emb_offset = prompt_tokens[pos] * CONFIG.embed_dim
            var px_offset = pos * CONFIG.embed_dim
            for i in range(CONFIG.embed_dim):
                prompt_x[px_offset + i] = embed_ptr[token_emb_offset + i] * scale

        for layer_idx in range(CONFIG.num_layers):
            # Resolve pointer directly from RAM weights!
            var layer_ptr = weights_ram_ptr + CONFIG.get_layer_offset(layer_idx)
            var weights = CONFIG.get_layer_weights(layer_idx, layer_ptr)
            
            var is_global = CONFIG.is_global_layer(layer_idx)
            var layer_heads = 16
            var layer_kv_heads = 1 if is_global else 8
            var layer_head_dim = 512 if is_global else 256
            var kv_stride = layer_kv_heads * layer_head_dim
            var ratio = layer_heads // layer_kv_heads
            
            for pos in range(num_prompt_tokens):
                var px_offset = pos * CONFIG.embed_dim
                var local_x = prompt_x + px_offset
                
                rms_norm(local_x, weights.input_layernorm_weight, norm_x, CONFIG.embed_dim)
                
                q8_matmul_vec(weights.wq_packed, weights.wq_scales, norm_x, q, CONFIG.embed_dim, layer_heads * layer_head_dim, num_tasks)
                q8_matmul_vec(weights.wk_packed, weights.wk_scales, norm_x, k_vec, CONFIG.embed_dim, layer_kv_heads * layer_head_dim, num_tasks)
                
                if is_global:
                    for i in range(kv_stride):
                        v_vec[i] = k_vec[i]
                else:
                    q8_matmul_vec(weights.wv_packed, weights.wv_scales, norm_x, v_vec, CONFIG.embed_dim, layer_kv_heads * layer_head_dim, num_tasks)
                    
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
                
                rms_norm(proj_attn_out, weights.post_attention_layernorm_weight, proj_attn_out, CONFIG.embed_dim)
                for i in range(CONFIG.embed_dim):
                    local_x[i] += proj_attn_out[i]
                    
                rms_norm(local_x, weights.pre_feedforward_layernorm_weight, norm_x, CONFIG.embed_dim)
                q8_matmul_vec(weights.w1_packed, weights.w1_scales, norm_x, h1, CONFIG.embed_dim, CONFIG.intermediate_dim, num_tasks)
                q8_matmul_vec(weights.w3_packed, weights.w3_scales, norm_x, h2, CONFIG.embed_dim, CONFIG.intermediate_dim, num_tasks)
                gelu_and_mul(h1, h2, ffn_out_gate, CONFIG.intermediate_dim)
                
                # Dynamic computation sparse skipping (avoiding flops on zero columns)
                q8_matmul_w2_block_column(weights.w2_packed, ffn_out_gate, ffn_out, CONFIG.intermediate_dim, CONFIG.embed_dim, num_tasks, Float32(1e-4))
                
                rms_norm(ffn_out, weights.post_feedforward_layernorm_weight, ffn_out, CONFIG.embed_dim)
                for i in range(CONFIG.embed_dim):
                    local_x[i] += ffn_out[i]
                    
                var scalar = weights.layer_scalar[0]
                for i in range(CONFIG.embed_dim):
                    local_x[i] *= scalar

        # Copy last prompt token's hidden state to x
        var last_px_offset = (num_prompt_tokens - 1) * CONFIG.embed_dim
        for i in range(CONFIG.embed_dim):
            x[i] = prompt_x[last_px_offset + i]
        prompt_x.free()
        
        var t_prefill_end = py.time()
        var prefill_time = Float64(py=(t_prefill_end - t_prefill_start))
        print("\n[Mojo Bench] Prefill completed in", prefill_time, "sec (" + String(Float64(num_prompt_tokens) / prefill_time) + " t/s)")
        
        # Project first generated token
        rms_norm(x, final_norm_weight, final_norm_x, CONFIG.embed_dim)
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
                
                # Logit soft-capping
                logit = Float32(30.0) * fast_tanh(logit / Float32(30.0))
                
                if logit > local_max_l:
                    local_max_l = logit
                    local_tok = r
                    
            local_max_logits[t] = local_max_l
            local_next_tokens[t] = local_tok
            
        parallelize[proj_worker](num_tasks)
        
        var max_logit: Float32 = -999999.0
        var next_token = 0
        for t in range(num_tasks):
            if local_max_logits[t] > max_logit:
                max_logit = local_max_logits[t]
                next_token = local_next_tokens[t]
                
        var next_token_list = list_cls()
        _ = next_token_list.append(next_token)
        var decoded = tokenizer.decode(next_token_list)
        py_builtins.print(decoded, end="", flush=True)
        
        current_token = next_token
        
        # Subsequent generation steps
        var tokens_generated = 1
        var t_gen_start = py.time()
        for pos in range(num_prompt_tokens, max_steps):
            var token_emb_offset = current_token * CONFIG.embed_dim
            for i in range(CONFIG.embed_dim):
                x[i] = embed_ptr[token_emb_offset + i] * scale
                
            for layer_idx in range(CONFIG.num_layers):
                # Resolve pointer directly from RAM weights!
                var layer_ptr = weights_ram_ptr + CONFIG.get_layer_offset(layer_idx)
                var weights = CONFIG.get_layer_weights(layer_idx, layer_ptr)
                
                var is_global = CONFIG.is_global_layer(layer_idx)
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
                
                rms_norm(proj_attn_out, weights.post_attention_layernorm_weight, proj_attn_out, CONFIG.embed_dim)
                for i in range(CONFIG.embed_dim):
                    x[i] += proj_attn_out[i]
                    
                rms_norm(x, weights.pre_feedforward_layernorm_weight, norm_x, CONFIG.embed_dim)
                q8_matmul_vec(weights.w1_packed, weights.w1_scales, norm_x, h1, CONFIG.embed_dim, CONFIG.intermediate_dim, num_tasks)
                q8_matmul_vec(weights.w3_packed, weights.w3_scales, norm_x, h2, CONFIG.embed_dim, CONFIG.intermediate_dim, num_tasks)
                gelu_and_mul(h1, h2, ffn_out_gate, CONFIG.intermediate_dim)
                
                # Dynamic computation sparse skipping (avoiding flops on zero columns)
                q8_matmul_w2_block_column(weights.w2_packed, ffn_out_gate, ffn_out, CONFIG.intermediate_dim, CONFIG.embed_dim, num_tasks, Float32(1e-4))
                
                rms_norm(ffn_out, weights.post_feedforward_layernorm_weight, ffn_out, CONFIG.embed_dim)
                for i in range(CONFIG.embed_dim):
                    x[i] += ffn_out[i]
                    
                var scalar = weights.layer_scalar[0]
                for i in range(CONFIG.embed_dim):
                    x[i] *= scalar

            rms_norm(x, final_norm_weight, final_norm_x, CONFIG.embed_dim)
            parallelize[proj_worker](num_tasks)
            
            max_logit = -999999.0
            next_token = 0
            for t in range(num_tasks):
                if local_max_logits[t] > max_logit:
                    max_logit = local_max_logits[t]
                    next_token = local_next_tokens[t]
                    
            var next_token_list_inner = list_cls()
            _ = next_token_list_inner.append(next_token)
            var word = String(tokenizer.decode(next_token_list_inner))
            py_builtins.print(word, end="", flush=True)
            
            if next_token == 1 or next_token == 106:
                print()
                break
                
            current_token = next_token
            tokens_generated += 1
            
        var t_gen_end = py.time()
        var gen_time = Float64(py=(t_gen_end - t_gen_start))
        print("\n[Mojo Bench] Generation completed in", gen_time, "sec (" + String(Float64(tokens_generated) / gen_time) + " t/s)")
        
        local_max_logits.free()
        local_next_tokens.free()

    print("\n---------------------------------------------------------")
    print("Inference generation loop complete!")

    # Cleanup open file handles and memory
    free_c(weights_addr)
    free_c(embed_addr)
    memptr_weights.free()
    memptr_embed.free()

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
    k_cache.free()
    v_cache.free()
