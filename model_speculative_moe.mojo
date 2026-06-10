from std.ffi import OwnedDLHandle
from std.memory import alloc
from std.math import sqrt, exp, cos, sin
from std.python import Python, PythonObject
import std.os
from std.algorithm import parallelize
from std.sys.info import num_physical_cores
from q4_embed import Q4Embedding
import std.time

# -----------------------------------------------------------------------------
# 1. Model Configuration & Memory Offsets
# -----------------------------------------------------------------------------

struct MoEConfig[
    embed_dim: Int = 2816,
    standard_mlp_dim: Int = 2112,
    expert_mlp_dim: Int = 704,
    num_experts: Int = 128,
    top_k_experts: Int = 8,
    num_layers: Int = 30,
    vocab_size: Int = 262144,
    max_seq_len: Int = 100,
]:
    @always_inline
    def is_global_layer(self, layer_idx: Int) -> Bool:
        return layer_idx % 6 == 5

    @always_inline
    def get_layer_aligned_size(self, layer_idx: Int) -> Int:
        var is_global = self.is_global_layer(layer_idx)
        var layer_heads = 16
        var layer_head_dim = 512 if is_global else 256
        var layer_kv_heads = 2 if is_global else 8
        
        var wq_size = layer_heads * layer_head_dim * Self.embed_dim
        var wk_size = layer_kv_heads * layer_head_dim * Self.embed_dim
        var wv_size = 0 if is_global else (layer_kv_heads * layer_head_dim * Self.embed_dim)
        var wo_size = Self.embed_dim * layer_heads * layer_head_dim
        var w_gate_size = Self.standard_mlp_dim * Self.embed_dim
        var w_up_size = Self.standard_mlp_dim * Self.embed_dim
        var w_down_size = Self.embed_dim * Self.standard_mlp_dim
        var w_router_size = Self.num_experts * Self.embed_dim
        var experts_gate_up_size = Self.num_experts * (Self.expert_mlp_dim * 2) * Self.embed_dim
        var experts_down_size = Self.num_experts * Self.embed_dim * Self.expert_mlp_dim
        
        var packed_weights_size = (
            wq_size + wk_size + wv_size + wo_size +
            w_gate_size + w_up_size + w_down_size +
            w_router_size + experts_gate_up_size + experts_down_size
        ) // 2
        
        var scales_count = (
            wq_size + wk_size + wv_size + wo_size +
            w_gate_size + w_up_size + w_down_size +
            w_router_size + experts_gate_up_size + experts_down_size
        ) // 32
        var scales_size_bytes = scales_count * 4
        
        var input_layernorm_size = Self.embed_dim
        var post_attention_layernorm_size = Self.embed_dim
        var pre_feedforward_layernorm_size = Self.embed_dim
        var pre_feedforward_layernorm_2_size = Self.embed_dim
        var post_feedforward_layernorm_size = Self.embed_dim
        var post_feedforward_layernorm_1_size = Self.embed_dim
        var post_feedforward_layernorm_2_size = Self.embed_dim
        
        var q_norm_size = layer_head_dim
        var k_norm_size = layer_head_dim
        var router_scale_size = Self.embed_dim
        var router_per_expert_scale_size = 128
        var layer_scalar_size = 1
        
        var unchanged_size_bytes = (
            input_layernorm_size + post_attention_layernorm_size +
            pre_feedforward_layernorm_size + pre_feedforward_layernorm_2_size +
            post_feedforward_layernorm_size + post_feedforward_layernorm_1_size +
            post_feedforward_layernorm_2_size +
            q_norm_size + k_norm_size +
            router_scale_size + router_per_expert_scale_size + layer_scalar_size
        ) * 4
        
        var raw_bytes_size = packed_weights_size + scales_size_bytes + unchanged_size_bytes
        return ((raw_bytes_size + 4095) // 4096) * 4096

    def get_layer_offset(self, layer_idx: Int) -> Int:
        var offset = 0
        for i in range(layer_idx):
            offset += self.get_layer_aligned_size(i)
        return offset

    @always_inline
    def get_layer_weights(self, layer_idx: Int, base_ptr: UnsafePointer[UInt8, _]) -> MoELayerWeights:
        var is_global = self.is_global_layer(layer_idx)
        var layer_heads = 16
        var layer_head_dim = 512 if is_global else 256
        var layer_kv_heads = 2 if is_global else 8
        
        var wq_size = layer_heads * layer_head_dim * Self.embed_dim
        var wk_size = layer_kv_heads * layer_head_dim * Self.embed_dim
        var wv_size = 0 if is_global else (layer_kv_heads * layer_head_dim * Self.embed_dim)
        var wo_size = Self.embed_dim * layer_heads * layer_head_dim
        var w_gate_size = Self.standard_mlp_dim * Self.embed_dim
        var w_up_size = Self.standard_mlp_dim * Self.embed_dim
        var w_down_size = Self.embed_dim * Self.standard_mlp_dim
        var w_router_size = Self.num_experts * Self.embed_dim
        var experts_gate_up_size = Self.num_experts * (Self.expert_mlp_dim * 2) * Self.embed_dim
        var experts_down_size = Self.num_experts * Self.embed_dim * Self.expert_mlp_dim
        
        var wq_size_bytes = wq_size // 2
        var wk_size_bytes = wk_size // 2
        var wv_size_bytes = wv_size // 2
        var wo_size_bytes = wo_size // 2
        var w_gate_size_bytes = w_gate_size // 2
        var w_up_size_bytes = w_up_size // 2
        var w_down_size_bytes = w_down_size // 2
        var w_router_size_bytes = w_router_size // 2
        var experts_gate_up_size_bytes = experts_gate_up_size // 2
        var experts_down_size_bytes = experts_down_size // 2
        
        var wq_packed = base_ptr
        var wk_packed = wq_packed + wq_size_bytes
        var wv_packed = wk_packed + wk_size_bytes
        var wo_packed = wv_packed + wv_size_bytes
        var w_gate_packed = wo_packed + wo_size_bytes
        var w_up_packed = w_gate_packed + w_gate_size_bytes
        var w_down_packed = w_up_packed + w_up_size_bytes
        var w_router_packed = w_down_packed + w_down_size_bytes
        var experts_gate_up_packed = w_router_packed + w_router_size_bytes
        var experts_down_packed = experts_gate_up_packed + experts_gate_up_size_bytes
        
        var scales_base = (experts_down_packed + experts_down_size_bytes).bitcast[Float32]()
        
        var c_blocks_q = wq_size // 32
        var c_blocks_k = wk_size // 32
        var c_blocks_v = wv_size // 32
        var c_blocks_o = wo_size // 32
        var c_blocks_w_gate = w_gate_size // 32
        var c_blocks_w_up = w_up_size // 32
        var c_blocks_w_down = w_down_size // 32
        var c_blocks_w_router = w_router_size // 32
        var c_blocks_experts_gate_up = experts_gate_up_size // 32
        var c_blocks_experts_down = experts_down_size // 32
        
        var wq_scales = scales_base
        var wk_scales = wq_scales + c_blocks_q
        var wv_scales = wk_scales + c_blocks_k
        var wo_scales = wv_scales + c_blocks_v
        var w_gate_scales = wo_scales + c_blocks_o
        var w_up_scales = w_gate_scales + c_blocks_w_gate
        var w_down_scales = w_up_scales + c_blocks_w_up
        var w_router_scales = w_down_scales + c_blocks_w_down
        var experts_gate_up_scales = w_router_scales + c_blocks_w_router
        var experts_down_scales = experts_gate_up_scales + c_blocks_experts_gate_up
        
        var unchanged_base = experts_down_scales + c_blocks_experts_down
        
        var input_layernorm = unchanged_base
        var post_attention_layernorm = input_layernorm + Self.embed_dim
        var pre_feedforward_layernorm = post_attention_layernorm + Self.embed_dim
        var pre_feedforward_layernorm_2 = pre_feedforward_layernorm + Self.embed_dim
        var post_feedforward_layernorm = pre_feedforward_layernorm_2 + Self.embed_dim
        var post_feedforward_layernorm_1 = post_feedforward_layernorm + Self.embed_dim
        var post_feedforward_layernorm_2 = post_feedforward_layernorm_1 + Self.embed_dim
        var q_norm = post_feedforward_layernorm_2 + Self.embed_dim
        var k_norm = q_norm + layer_head_dim
        var router_scale = k_norm + layer_head_dim
        var router_per_expert_scale = router_scale + Self.embed_dim
        var layer_scalar = router_per_expert_scale + 128
        
        return MoELayerWeights(
            UnsafePointer[UInt8, MutExternalOrigin](unsafe_from_address=wq_packed.__int__()),
            UnsafePointer[UInt8, MutExternalOrigin](unsafe_from_address=wk_packed.__int__()),
            UnsafePointer[UInt8, MutExternalOrigin](unsafe_from_address=wv_packed.__int__()),
            UnsafePointer[UInt8, MutExternalOrigin](unsafe_from_address=wo_packed.__int__()),
            UnsafePointer[UInt8, MutExternalOrigin](unsafe_from_address=w_gate_packed.__int__()),
            UnsafePointer[UInt8, MutExternalOrigin](unsafe_from_address=w_up_packed.__int__()),
            UnsafePointer[UInt8, MutExternalOrigin](unsafe_from_address=w_down_packed.__int__()),
            UnsafePointer[UInt8, MutExternalOrigin](unsafe_from_address=w_router_packed.__int__()),
            UnsafePointer[UInt8, MutExternalOrigin](unsafe_from_address=experts_gate_up_packed.__int__()),
            UnsafePointer[UInt8, MutExternalOrigin](unsafe_from_address=experts_down_packed.__int__()),
            UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=wq_scales.__int__()),
            UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=wk_scales.__int__()),
            UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=wv_scales.__int__()),
            UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=wo_scales.__int__()),
            UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=w_gate_scales.__int__()),
            UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=w_up_scales.__int__()),
            UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=w_down_scales.__int__()),
            UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=w_router_scales.__int__()),
            UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=experts_gate_up_scales.__int__()),
            UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=experts_down_scales.__int__()),
            UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=input_layernorm.__int__()),
            UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=post_attention_layernorm.__int__()),
            UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=pre_feedforward_layernorm.__int__()),
            UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=pre_feedforward_layernorm_2.__int__()),
            UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=post_feedforward_layernorm.__int__()),
            UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=post_feedforward_layernorm_1.__int__()),
            UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=post_feedforward_layernorm_2.__int__()),
            UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=q_norm.__int__()),
            UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=k_norm.__int__()),
            UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=router_scale.__int__()),
            UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=router_per_expert_scale.__int__()),
            UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=layer_scalar.__int__())
        )

    def __init__(out self):
        pass


# -----------------------------------------------------------------------------
# 2. Structural Layer Weights Representation
# -----------------------------------------------------------------------------

struct MoELayerWeights:
    var wq_packed: UnsafePointer[UInt8, MutExternalOrigin]
    var wk_packed: UnsafePointer[UInt8, MutExternalOrigin]
    var wv_packed: UnsafePointer[UInt8, MutExternalOrigin]
    var wo_packed: UnsafePointer[UInt8, MutExternalOrigin]
    var w_gate_packed: UnsafePointer[UInt8, MutExternalOrigin]
    var w_up_packed: UnsafePointer[UInt8, MutExternalOrigin]
    var w_down_packed: UnsafePointer[UInt8, MutExternalOrigin]
    var w_router_packed: UnsafePointer[UInt8, MutExternalOrigin]
    var experts_gate_up_packed: UnsafePointer[UInt8, MutExternalOrigin]
    var experts_down_packed: UnsafePointer[UInt8, MutExternalOrigin]

    var wq_scales: UnsafePointer[Float32, MutExternalOrigin]
    var wk_scales: UnsafePointer[Float32, MutExternalOrigin]
    var wv_scales: UnsafePointer[Float32, MutExternalOrigin]
    var wo_scales: UnsafePointer[Float32, MutExternalOrigin]
    var w_gate_scales: UnsafePointer[Float32, MutExternalOrigin]
    var w_up_scales: UnsafePointer[Float32, MutExternalOrigin]
    var w_down_scales: UnsafePointer[Float32, MutExternalOrigin]
    var w_router_scales: UnsafePointer[Float32, MutExternalOrigin]
    var experts_gate_up_scales: UnsafePointer[Float32, MutExternalOrigin]
    var experts_down_scales: UnsafePointer[Float32, MutExternalOrigin]

    var input_layernorm_weight: UnsafePointer[Float32, MutExternalOrigin]
    var post_attention_layernorm_weight: UnsafePointer[Float32, MutExternalOrigin]
    var pre_feedforward_layernorm_weight: UnsafePointer[Float32, MutExternalOrigin]
    var pre_feedforward_layernorm_2_weight: UnsafePointer[Float32, MutExternalOrigin]
    var post_feedforward_layernorm_weight: UnsafePointer[Float32, MutExternalOrigin]
    var post_feedforward_layernorm_1_weight: UnsafePointer[Float32, MutExternalOrigin]
    var post_feedforward_layernorm_2_weight: UnsafePointer[Float32, MutExternalOrigin]
    var q_norm_weight: UnsafePointer[Float32, MutExternalOrigin]
    var k_norm_weight: UnsafePointer[Float32, MutExternalOrigin]
    var router_scale: UnsafePointer[Float32, MutExternalOrigin]
    var router_per_expert_scale: UnsafePointer[Float32, MutExternalOrigin]
    var layer_scalar: UnsafePointer[Float32, MutExternalOrigin]

    def __init__(
        out self,
        wq_packed: UnsafePointer[UInt8, MutExternalOrigin],
        wk_packed: UnsafePointer[UInt8, MutExternalOrigin],
        wv_packed: UnsafePointer[UInt8, MutExternalOrigin],
        wo_packed: UnsafePointer[UInt8, MutExternalOrigin],
        w_gate_packed: UnsafePointer[UInt8, MutExternalOrigin],
        w_up_packed: UnsafePointer[UInt8, MutExternalOrigin],
        w_down_packed: UnsafePointer[UInt8, MutExternalOrigin],
        w_router_packed: UnsafePointer[UInt8, MutExternalOrigin],
        experts_gate_up_packed: UnsafePointer[UInt8, MutExternalOrigin],
        experts_down_packed: UnsafePointer[UInt8, MutExternalOrigin],
        wq_scales: UnsafePointer[Float32, MutExternalOrigin],
        wk_scales: UnsafePointer[Float32, MutExternalOrigin],
        wv_scales: UnsafePointer[Float32, MutExternalOrigin],
        wo_scales: UnsafePointer[Float32, MutExternalOrigin],
        w_gate_scales: UnsafePointer[Float32, MutExternalOrigin],
        w_up_scales: UnsafePointer[Float32, MutExternalOrigin],
        w_down_scales: UnsafePointer[Float32, MutExternalOrigin],
        w_router_scales: UnsafePointer[Float32, MutExternalOrigin],
        experts_gate_up_scales: UnsafePointer[Float32, MutExternalOrigin],
        experts_down_scales: UnsafePointer[Float32, MutExternalOrigin],
        input_layernorm_weight: UnsafePointer[Float32, MutExternalOrigin],
        post_attention_layernorm_weight: UnsafePointer[Float32, MutExternalOrigin],
        pre_feedforward_layernorm_weight: UnsafePointer[Float32, MutExternalOrigin],
        pre_feedforward_layernorm_2_weight: UnsafePointer[Float32, MutExternalOrigin],
        post_feedforward_layernorm_weight: UnsafePointer[Float32, MutExternalOrigin],
        post_feedforward_layernorm_1_weight: UnsafePointer[Float32, MutExternalOrigin],
        post_feedforward_layernorm_2_weight: UnsafePointer[Float32, MutExternalOrigin],
        q_norm_weight: UnsafePointer[Float32, MutExternalOrigin],
        k_norm_weight: UnsafePointer[Float32, MutExternalOrigin],
        router_scale: UnsafePointer[Float32, MutExternalOrigin],
        router_per_expert_scale: UnsafePointer[Float32, MutExternalOrigin],
        layer_scalar: UnsafePointer[Float32, MutExternalOrigin],
    ):
        self.wq_packed = wq_packed
        self.wk_packed = wk_packed
        self.wv_packed = wv_packed
        self.wo_packed = wo_packed
        self.w_gate_packed = w_gate_packed
        self.w_up_packed = w_up_packed
        self.w_down_packed = w_down_packed
        self.w_router_packed = w_router_packed
        self.experts_gate_up_packed = experts_gate_up_packed
        self.experts_down_packed = experts_down_packed
        self.wq_scales = wq_scales
        self.wk_scales = wk_scales
        self.wv_scales = wv_scales
        self.wo_scales = wo_scales
        self.w_gate_scales = w_gate_scales
        self.w_up_scales = w_up_scales
        self.w_down_scales = w_down_scales
        self.w_router_scales = w_router_scales
        self.experts_gate_up_scales = experts_gate_up_scales
        self.experts_down_scales = experts_down_scales
        self.input_layernorm_weight = input_layernorm_weight
        self.post_attention_layernorm_weight = post_attention_layernorm_weight
        self.pre_feedforward_layernorm_weight = pre_feedforward_layernorm_weight
        self.pre_feedforward_layernorm_2_weight = pre_feedforward_layernorm_2_weight
        self.post_feedforward_layernorm_weight = post_feedforward_layernorm_weight
        self.post_feedforward_layernorm_1_weight = post_feedforward_layernorm_1_weight
        self.post_feedforward_layernorm_2_weight = post_feedforward_layernorm_2_weight
        self.q_norm_weight = q_norm_weight
        self.k_norm_weight = k_norm_weight
        self.router_scale = router_scale
        self.router_per_expert_scale = router_per_expert_scale
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
def unpack_4bytes(b0: UInt8, b1: UInt8, b2: UInt8, b3: UInt8) -> SIMD[DType.float32, 8]:
    var chunk = SIMD[DType.uint8, 8](b0, b1, b2, b3, 0, 0, 0, 0)
    var repeated = chunk.shuffle[0, 0, 1, 1, 2, 2, 3, 3]()
    var shifted = repeated >> SIMD[DType.uint8, 8](0, 4, 0, 4, 0, 4, 0, 4)
    var masked = shifted & 15
    return (masked.cast[DType.int8]() - 8).cast[DType.float32]()


@always_inline
def unpack_16bytes(
    bytes: SIMD[DType.uint8, 16], 
    mut w0: SIMD[DType.float32, 8], 
    mut w1: SIMD[DType.float32, 8], 
    mut w2: SIMD[DType.float32, 8], 
    mut w3: SIMD[DType.float32, 8]
):
    var repeated_01 = bytes.shuffle[0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7]()
    var shifted_01 = repeated_01 >> SIMD[DType.uint8, 16](0, 4, 0, 4, 0, 4, 0, 4, 0, 4, 0, 4, 0, 4, 0, 4)
    var masked_01 = (shifted_01 & 15).cast[DType.int8]() - 8
    
    var repeated_23 = bytes.shuffle[8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13, 14, 14, 15, 15]()
    var shifted_23 = repeated_23 >> SIMD[DType.uint8, 16](0, 4, 0, 4, 0, 4, 0, 4, 0, 4, 0, 4, 0, 4, 0, 4)
    var masked_23 = (shifted_23 & 15).cast[DType.int8]() - 8
    
    w0 = masked_01.slice[8]().cast[DType.float32]()
    w1 = masked_01.shift_left[8]().slice[8]().cast[DType.float32]()
    w2 = masked_23.slice[8]().cast[DType.float32]()
    w3 = masked_23.shift_left[8]().slice[8]().cast[DType.float32]()


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
    var tasks = num_tasks
    if tasks > c_out:
        tasks = c_out
    if tasks < 1:
        tasks = 1
        
    var chunk_size = c_out // tasks
    var c_in_packed = c_in // 2
    var c_in_blocks = c_in // 32
    
    def worker(t: Int) capturing -> None:
        var start_r = t * chunk_size
        var end_r = (t + 1) * chunk_size if t < tasks - 1 else c_out
        
        for r in range(start_r, end_r):
            var acc = SIMD[DType.float32, 8](0.0)
            
            var r_offset_packed = r * c_in_packed
            var r_offset_scales = r * c_in_blocks
            
            for b in range(c_in_blocks):
                var scale = scales[r_offset_scales + b]
                var block_p = r_offset_packed + b * 16
                var x_offset = b * 32
                
                var bytes = W_packed.load[16](block_p)
                
                var w0 = SIMD[DType.float32, 8]()
                var w1 = SIMD[DType.float32, 8]()
                var w2 = SIMD[DType.float32, 8]()
                var w3 = SIMD[DType.float32, 8]()
                unpack_16bytes(bytes, w0, w1, w2, w3)
                
                acc = acc + (w0 * x.load[8](x_offset)) * scale
                acc = acc + (w1 * x.load[8](x_offset + 8)) * scale
                acc = acc + (w2 * x.load[8](x_offset + 16)) * scale
                acc = acc + (w3 * x.load[8](x_offset + 24)) * scale
                
            out_ptr[r] = acc.reduce_add()
            
    if tasks == 1:
        worker(0)
    else:
        parallelize[worker](tasks)

@always_inline
def gelu_and_mul(h1: UnsafePointer[Float32, _], h2: UnsafePointer[Float32, _], out_ptr: UnsafePointer[Float32, MutExternalOrigin], size: Int):
    for i in range(size):
        var x1 = h1[i]
        var sig = 1.0 / (1.0 + exp(-1.702 * x1))
        var gelu = x1 * sig
        out_ptr[i] = gelu * h2[i]

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
        scores[i] /= (sum_exp + 1e-10)

@always_inline
def get_top_k[k: Int](probs: UnsafePointer[Float32, _], top_idx: UnsafePointer[Int, MutExternalOrigin], top_val: UnsafePointer[Float32, MutExternalOrigin], size: Int):
    for i in range(k):
        top_idx[i] = -1
        top_val[i] = -999999.0
        
    for i in range(size):
        var val = probs[i]
        var curr_val = val
        var curr_idx = i
        
        for j in range(k):
            if curr_val > top_val[j]:
                var temp_val = top_val[j]
                var temp_idx = top_idx[j]
                top_val[j] = curr_val
                top_idx[j] = curr_idx
                curr_val = temp_val
                curr_idx = temp_idx


# -----------------------------------------------------------------------------
# 4. Custom Padding Helpers & Main Executive Engine
# -----------------------------------------------------------------------------

def pad_right(s: String, width: Int) -> String:
    var res = s
    while res.byte_length() < width:
        res += " "
    return res

def pad_left(s: String, width: Int) -> String:
    var res = s
    while res.byte_length() < width:
        res = " " + res
    return res

def main() raises:
    var num_tasks = num_physical_cores()
    print("=========================================================")
    print("   GEMMA 4 26B MoE RAM-RESIDENT INFERENCE ENGINE         ")
    print("=========================================================")
    print("Embedding Size:      2816")
    print("Attention Heads:     16 Query Heads / 8 or 2 KV Heads (GQA)")
    print("Number of Layers:    30 Layers (5 global attention layers)")
    print("Number of Experts:   128 Experts (Top-8 Router routing)")
    print("Draft Model State:   Fully RAM-resident Q4_0 (4-bit)")
    print("---------------------------------------------------------")

    var CONFIG = MoEConfig[]()

    # Initialize FFI and libc functions
    var libc = OwnedDLHandle("libc.so.6")
    var c_open = libc.get_function[def(Int, Int32, Int32) -> Int32]("open")
    var c_close = libc.get_function[def(Int32) -> Int32]("close")
    var posix_memalign = libc.get_function[def(Int, Int, Int) -> Int32]("posix_memalign")
    var free_c = libc.get_function[def(Int) -> None]("free")
    var c_read = libc.get_function[def(Int32, Int, Int) -> Int]("read")

    # 1. Allocate and Load MoE Weights (RAM-resident)
    var weights_file = String("gemma4_moe_weights_q4.bin")
    print("Opening MoE weights file '" + weights_file + "'...")
    var path_bytes = weights_file.as_bytes()
    var path_ptr = alloc[UInt8](len(path_bytes) + 1)
    for i in range(len(path_bytes)):
        path_ptr[i] = path_bytes[i]
    path_ptr[len(path_bytes)] = 0
    var weights_fd = c_open(path_ptr.__int__(), 0, 0)
    path_ptr.free()
    if weights_fd < 0:
        print("CRITICAL: Failed to open weights file '" + weights_file + "'!")
        return

    var weights_file_size: Int = 15311790080
    var memptr_weights = alloc[Int](1)
    var err_weights = posix_memalign(memptr_weights.__int__(), 4096, weights_file_size)
    if err_weights != 0:
        print("CRITICAL: Failed to allocate aligned memory for MoE weights (14.3 GB)!")
        _ = c_close(weights_fd)
        memptr_weights.free()
        return
    var base_addr = memptr_weights[0]
    memptr_weights.free()
    var base_ptr = UnsafePointer[UInt8, MutExternalOrigin](unsafe_from_address=base_addr)

    print("Loading MoE weights into RAM (14.26 GB) in chunks...")
    var total_read: Int = 0
    var chunk_bytes: Int = 268435456 # 256 MB chunks
    while total_read < weights_file_size:
        var bytes_to_read = weights_file_size - total_read
        if bytes_to_read > chunk_bytes:
            bytes_to_read = chunk_bytes
        var read_res = c_read(weights_fd, base_addr + total_read, bytes_to_read)
        if read_res <= 0:
            print("CRITICAL: Error reading weights from file!")
            _ = c_close(weights_fd)
            return
        total_read += read_res
        if total_read % (2 * 1024 * 1024 * 1024) < chunk_bytes:
            print("  Loaded: ", total_read // 1024 // 1024 // 1024, " GB / 14 GB...")
    print("Successfully loaded all MoE weights into RAM: ", total_read, " bytes")
    _ = c_close(weights_fd)

    # 2. Allocate and Load Final Layer Norm Weights
    var final_norm_file = String("gemma4_moe_final_norm.bin")
    print("Opening final norm weights file 'gemma4_moe_final_norm.bin'...")
    var final_norm_path_bytes = final_norm_file.as_bytes()
    var final_norm_path_ptr = alloc[UInt8](len(final_norm_path_bytes) + 1)
    for i in range(len(final_norm_path_bytes)):
        final_norm_path_ptr[i] = final_norm_path_bytes[i]
    final_norm_path_ptr[len(final_norm_path_bytes)] = 0
    var final_norm_fd = c_open(final_norm_path_ptr.__int__(), 0, 0)
    final_norm_path_ptr.free()
    if final_norm_fd < 0:
        print("CRITICAL: Failed to open final norm file!")
        return

    var final_norm_weight = alloc[Float32](CONFIG.embed_dim)
    var final_norm_size_bytes = CONFIG.embed_dim * 4
    total_read = 0
    while total_read < final_norm_size_bytes:
        var bytes_to_read = final_norm_size_bytes - total_read
        var read_res = c_read(final_norm_fd, final_norm_weight.__int__() + total_read, bytes_to_read)
        if read_res <= 0:
            print("CRITICAL: Failed to read from final norm weights file!")
            _ = c_close(final_norm_fd)
            final_norm_weight.free()
            return
        total_read += read_res
    _ = c_close(final_norm_fd)
    print("Final norm weights loaded: ", total_read, " bytes")

    # 3. Load Q4 Token Embeddings
    print("Loading Q4 Token Embeddings...")
    var embed = Q4Embedding("gemma4_moe_embed_q4.bin", CONFIG.vocab_size, CONFIG.embed_dim)

    # 4. Allocate continuous KV Cache for 30 layers
    var kv_cache_offsets = List[Int]()
    var current_cache_offset = 0
    for l in range(30):
        kv_cache_offsets.append(current_cache_offset)
        var is_global = CONFIG.is_global_layer(l)
        var kv_heads = 2 if is_global else 8
        var head_dim = 512 if is_global else 256
        current_cache_offset += CONFIG.max_seq_len * kv_heads * head_dim
    print("Allocating KV Cache (", (current_cache_offset * 8) // 1024 // 1024, " MB)...")
    var k_cache = alloc[Float32](current_cache_offset)
    var v_cache = alloc[Float32](current_cache_offset)
    for i in range(current_cache_offset):
        k_cache[i] = 0.0
        v_cache[i] = 0.0

    # 5. Allocate thread-safe reusable activation buffers
    var local_x = alloc[Float32](CONFIG.embed_dim)
    var norm_x = alloc[Float32](CONFIG.embed_dim)
    var scaled_norm_x = alloc[Float32](CONFIG.embed_dim)
    var q = alloc[Float32](8192)
    var k_vec = alloc[Float32](2048)
    var v_vec = alloc[Float32](2048)
    var attn_scores = alloc[Float32](CONFIG.max_seq_len)
    var attn_out = alloc[Float32](8192)
    var proj_attn_out = alloc[Float32](CONFIG.embed_dim)
    
    var shared_h1 = alloc[Float32](CONFIG.standard_mlp_dim)
    var shared_h2 = alloc[Float32](CONFIG.standard_mlp_dim)
    var shared_ffn_out_gate = alloc[Float32](CONFIG.standard_mlp_dim)
    var shared_mlp_out = alloc[Float32](CONFIG.embed_dim)
    var shared_h1_norm = alloc[Float32](CONFIG.embed_dim)
    
    var moe_in = alloc[Float32](CONFIG.embed_dim)
    var moe_out = alloc[Float32](CONFIG.embed_dim)
    var gate_up_out_all = alloc[Float32](CONFIG.top_k_experts * (CONFIG.expert_mlp_dim * 2))
    var act_all = alloc[Float32](CONFIG.top_k_experts * CONFIG.expert_mlp_dim)
    var down_out_all = alloc[Float32](CONFIG.top_k_experts * CONFIG.embed_dim)
    var moe_out_norm = alloc[Float32](CONFIG.embed_dim)
    var combined_ffn = alloc[Float32](CONFIG.embed_dim)
    var ffn_out = alloc[Float32](CONFIG.embed_dim)
    
    var scores = alloc[Float32](128)
    var top_idx = alloc[Int](8)
    var top_val = alloc[Float32](8)
    var top_weights = alloc[Float32](8)
    
    var target_final_norm_x = alloc[Float32](CONFIG.embed_dim)
    var local_max_logits = alloc[Float32](num_tasks)
    var local_next_tokens = alloc[Int](num_tasks)

    var total_experts_evaluated: Int = 0
    var total_experts_skipped: Int = 0

    # 6. Initialize tokenizer
    var py = Python.import_module("time")
    var transformers = Python.import_module("transformers")
    print("Loading AutoTokenizer...")
    var tokenizer = transformers.AutoTokenizer.from_pretrained("google/gemma-4-12B-it")
    var py_builtins = Python.import_module("builtins")
    print("Tokenizer loaded successfully!")

    print("---------------------------------------------------------")
    print("   Chat Agent is ready! Type 'exit' or 'quit' to end.")
    print("---------------------------------------------------------")

    var prompt_scale = sqrt(Float32(CONFIG.embed_dim))

    while True:
        print()
        var user_input_str: String
        try:
            var user_input = py_builtins.input("User > ")
            user_input_str = String(String(user_input).strip())
        except err:
            print("\nError reading input:", err)
            break


        if user_input_str == "exit" or user_input_str == "quit":
            break
        if len(user_input_str) == 0:
            continue


        var prompt_str = String("<bos><|turn>user\n") + user_input_str + String("<turn|>\n<|turn>model\n")
        var encoded = tokenizer.encode(prompt_str, add_special_tokens=False)
        var prompt_tokens = List[Int]()
        for i in range(len(encoded)):
            prompt_tokens.append(Int(py=encoded[i]))

        var num_prompt_tokens = len(prompt_tokens)
        var max_gen_steps = 100
        if num_prompt_tokens + max_gen_steps > CONFIG.max_seq_len:
            max_gen_steps = CONFIG.max_seq_len - num_prompt_tokens

        # Zero-out the KV Cache for the new interaction
        for i in range(current_cache_offset):
            k_cache[i] = 0.0
            v_cache[i] = 0.0

        print("Agent > ", end="", flush=True)

        var pos = 0
        var next_token = 0

        var t_prefill_start = std.time.perf_counter()
        var t_decode_start: Float64 = 0.0

        # Micro-profiler accumulators (in seconds)
        var p_embed_reconstruct: Float64 = 0.0
        var p_attn_norm: Float64 = 0.0
        var p_qkv_proj: Float64 = 0.0
        var p_head_norm_rope_cache: Float64 = 0.0
        var p_attn_dot_product: Float64 = 0.0
        var p_wo_proj: Float64 = 0.0
        var p_shared_mlp: Float64 = 0.0
        var p_moe_router: Float64 = 0.0
        var p_moe_experts: Float64 = 0.0
        var p_moe_combine: Float64 = 0.0
        var p_final_norm: Float64 = 0.0
        var p_logit_proj: Float64 = 0.0
        var p_misc_decode: Float64 = 0.0

        # Main Prefill & Generation Autoregressive Loop
        while pos < num_prompt_tokens + max_gen_steps:
            var t0 = std.time.perf_counter()
            # Reconstruct the token embedding for the current step
            var current_token_id = prompt_tokens[pos] if pos < num_prompt_tokens else next_token

            embed.reconstruct(current_token_id, local_x)
            
            # Scale input embedding by sqrt(embed_dim)
            for i in range(CONFIG.embed_dim):
                local_x[i] *= prompt_scale

            var t1 = std.time.perf_counter()
            if pos >= num_prompt_tokens:
                p_embed_reconstruct += t1 - t0

            # Run layers
            for l in range(30):
                var weights = CONFIG.get_layer_weights(l, base_ptr + CONFIG.get_layer_offset(l))
                var is_global = CONFIG.is_global_layer(l)
                var layer_heads = 16
                var layer_kv_heads = 2 if is_global else 8
                var layer_head_dim = 512 if is_global else 256
                var kv_stride = layer_kv_heads * layer_head_dim
                var ratio = layer_heads // layer_kv_heads

                # 1. Attention Input Norm
                var ta0 = std.time.perf_counter()
                rms_norm(local_x, weights.input_layernorm_weight, norm_x, CONFIG.embed_dim)
                var ta1 = std.time.perf_counter()
                if pos >= num_prompt_tokens:
                    p_attn_norm += ta1 - ta0

                # 2. Attention projections
                ta0 = std.time.perf_counter()
                q4_matmul_vec(weights.wq_packed, weights.wq_scales, norm_x, q, CONFIG.embed_dim, layer_heads * layer_head_dim, num_tasks)
                q4_matmul_vec(weights.wk_packed, weights.wk_scales, norm_x, k_vec, CONFIG.embed_dim, layer_kv_heads * layer_head_dim, num_tasks)
                if is_global:
                    for i in range(kv_stride):
                        v_vec[i] = k_vec[i]
                else:
                    q4_matmul_vec(weights.wv_packed, weights.wv_scales, norm_x, v_vec, CONFIG.embed_dim, layer_kv_heads * layer_head_dim, num_tasks)
                ta1 = std.time.perf_counter()
                if pos >= num_prompt_tokens:
                    p_qkv_proj += ta1 - ta0

                # 3. Head-Normalizations & RoPE & KV cache write
                ta0 = std.time.perf_counter()
                for h in range(layer_heads):
                    var head_ptr = q + h * layer_head_dim
                    rms_norm(head_ptr, weights.q_norm_weight, head_ptr, layer_head_dim)
                for h in range(layer_kv_heads):
                    var head_ptr = k_vec + h * layer_head_dim
                    rms_norm(head_ptr, weights.k_norm_weight, head_ptr, layer_head_dim)
                for h in range(layer_kv_heads):
                    var head_ptr = v_vec + h * layer_head_dim
                    rms_norm_noscale(head_ptr, head_ptr, layer_head_dim)

                # 4. RoPE
                apply_rope(q, layer_heads, layer_head_dim, pos, is_global)
                apply_rope(k_vec, layer_kv_heads, layer_head_dim, pos, is_global)

                # 5. KV Cache write
                var layer_cache_offset = kv_cache_offsets[l]
                var step_cache_offset = layer_cache_offset + pos * kv_stride
                for i in range(kv_stride):
                    k_cache[step_cache_offset + i] = k_vec[i]
                    v_cache[step_cache_offset + i] = v_vec[i]
                ta1 = std.time.perf_counter()
                if pos >= num_prompt_tokens:
                    p_head_norm_rope_cache += ta1 - ta0

                # 6. Dot-product Attention
                ta0 = std.time.perf_counter()
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
                ta1 = std.time.perf_counter()
                if pos >= num_prompt_tokens:
                    p_attn_dot_product += ta1 - ta0

                # 7. Out projection and Attention Residual Add
                ta0 = std.time.perf_counter()
                q4_matmul_vec(weights.wo_packed, weights.wo_scales, attn_out, proj_attn_out, layer_heads * layer_head_dim, CONFIG.embed_dim, num_tasks)
                rms_norm(proj_attn_out, weights.post_attention_layernorm_weight, proj_attn_out, CONFIG.embed_dim)
                for i in range(CONFIG.embed_dim):
                    local_x[i] += proj_attn_out[i]
                ta1 = std.time.perf_counter()
                if pos >= num_prompt_tokens:
                    p_wo_proj += ta1 - ta0

                # 8. Parallel Feedforward: Shared MLP + sparse MoE Experts
                # Path A: Shared MLP
                ta0 = std.time.perf_counter()
                rms_norm(local_x, weights.pre_feedforward_layernorm_weight, norm_x, CONFIG.embed_dim)
                q4_matmul_vec(weights.w_gate_packed, weights.w_gate_scales, norm_x, shared_h1, CONFIG.embed_dim, CONFIG.standard_mlp_dim, num_tasks)
                q4_matmul_vec(weights.w_up_packed, weights.w_up_scales, norm_x, shared_h2, CONFIG.embed_dim, CONFIG.standard_mlp_dim, num_tasks)
                gelu_and_mul(shared_h1, shared_h2, shared_ffn_out_gate, CONFIG.standard_mlp_dim)
                q4_matmul_vec(weights.w_down_packed, weights.w_down_scales, shared_ffn_out_gate, shared_mlp_out, CONFIG.standard_mlp_dim, CONFIG.embed_dim, num_tasks)
                rms_norm(shared_mlp_out, weights.post_feedforward_layernorm_1_weight, shared_h1_norm, CONFIG.embed_dim)
                ta1 = std.time.perf_counter()
                if pos >= num_prompt_tokens:
                    p_shared_mlp += ta1 - ta0

                # Path B: Router and Experts
                ta0 = std.time.perf_counter()
                rms_norm_noscale(local_x, norm_x, CONFIG.embed_dim)
                var r_scale_factor = Float32(1.0 / sqrt(2816.0))
                for j in range(CONFIG.embed_dim):
                    scaled_norm_x[j] = norm_x[j] * weights.router_scale[j] * r_scale_factor
                q4_matmul_vec(weights.w_router_packed, weights.w_router_scales, scaled_norm_x, scores, CONFIG.embed_dim, CONFIG.num_experts, num_tasks)
                softmax(scores, CONFIG.num_experts)
                get_top_k[8](scores, top_idx, top_val, CONFIG.num_experts)
                
                var sum_top_val: Float32 = 0.0
                for i in range(CONFIG.top_k_experts):
                    sum_top_val += top_val[i]
                for i in range(CONFIG.top_k_experts):
                    var idx = top_idx[i]
                    var norm_prob = top_val[i] / (sum_top_val + 1e-10)
                    if norm_prob < 0.05:
                        top_weights[i] = 0.0
                        if pos >= num_prompt_tokens:
                            total_experts_skipped += 1
                    else:
                        top_weights[i] = norm_prob * weights.router_per_expert_scale[idx]
                    if pos >= num_prompt_tokens:
                        total_experts_evaluated += 1

                for i in range(CONFIG.embed_dim):
                    moe_out[i] = 0.0
                rms_norm(local_x, weights.pre_feedforward_layernorm_2_weight, moe_in, CONFIG.embed_dim)
                ta1 = std.time.perf_counter()
                if pos >= num_prompt_tokens:
                    p_moe_router += ta1 - ta0

                ta0 = std.time.perf_counter()
                def expert_worker(exp_idx: Int) capturing -> None:
                    var e = top_idx[exp_idx]
                    var w_e = top_weights[exp_idx]
                    if w_e == 0.0:
                        return

                    var exp_gate_up_out = gate_up_out_all + exp_idx * (CONFIG.expert_mlp_dim * 2)
                    var exp_act = act_all + exp_idx * CONFIG.expert_mlp_dim
                    var exp_down_out = down_out_all + exp_idx * CONFIG.embed_dim

                    var exp_gate_up_packed = weights.experts_gate_up_packed + e * (CONFIG.expert_mlp_dim * 2 * CONFIG.embed_dim // 2)
                    var exp_gate_up_scales = weights.experts_gate_up_scales + e * (CONFIG.expert_mlp_dim * 2 * CONFIG.embed_dim // 32)
                    q4_matmul_vec(exp_gate_up_packed, exp_gate_up_scales, moe_in, exp_gate_up_out, CONFIG.embed_dim, CONFIG.expert_mlp_dim * 2, 1)

                    var gate_ptr = exp_gate_up_out
                    var up_ptr = exp_gate_up_out + CONFIG.expert_mlp_dim
                    gelu_and_mul(gate_ptr, up_ptr, exp_act, CONFIG.expert_mlp_dim)

                    var exp_down_packed = weights.experts_down_packed + e * (CONFIG.embed_dim * CONFIG.expert_mlp_dim // 2)
                    var exp_down_scales = weights.experts_down_scales + e * (CONFIG.embed_dim * CONFIG.expert_mlp_dim // 32)
                    q4_matmul_vec(exp_down_packed, exp_down_scales, exp_act, exp_down_out, CONFIG.expert_mlp_dim, CONFIG.embed_dim, 1)

                parallelize[expert_worker](CONFIG.top_k_experts)
                ta1 = std.time.perf_counter()
                if pos >= num_prompt_tokens:
                    p_moe_experts += ta1 - ta0

                ta0 = std.time.perf_counter()
                for exp_idx in range(CONFIG.top_k_experts):
                    var w_e = top_weights[exp_idx]
                    if w_e == 0.0:
                        continue
                    var exp_down_out = down_out_all + exp_idx * CONFIG.embed_dim
                    for j in range(CONFIG.embed_dim):
                        moe_out[j] += exp_down_out[j] * w_e

                rms_norm(moe_out, weights.post_feedforward_layernorm_2_weight, moe_out_norm, CONFIG.embed_dim)

                # Combine Shared MLP and MoE Path
                for i in range(CONFIG.embed_dim):
                    combined_ffn[i] = shared_h1_norm[i] + moe_out_norm[i]
                rms_norm(combined_ffn, weights.post_feedforward_layernorm_weight, ffn_out, CONFIG.embed_dim)

                # Add to residual and multiply by layer_scalar
                var scalar = weights.layer_scalar[0]
                for i in range(CONFIG.embed_dim):
                    local_x[i] += ffn_out[i]
                    local_x[i] *= scalar
                ta1 = std.time.perf_counter()
                if pos >= num_prompt_tokens:
                    p_moe_combine += ta1 - ta0

            # Only predict and decode tokens after we process prompt or generate
            if pos >= num_prompt_tokens - 1:
                # 1. Final layer norm
                var tb0 = std.time.perf_counter()
                rms_norm(local_x, final_norm_weight, target_final_norm_x, CONFIG.embed_dim)
                var tb1 = std.time.perf_counter()
                if pos >= num_prompt_tokens:
                    p_final_norm += tb1 - tb0

                # 2. Logit projections
                tb0 = std.time.perf_counter()
                embed.project_logits_parallel(target_final_norm_x, local_max_logits, local_next_tokens, num_tasks)
                tb1 = std.time.perf_counter()
                if pos >= num_prompt_tokens:
                    p_logit_proj += tb1 - tb0

                # 3. Miscellaneous & max find & print
                tb0 = std.time.perf_counter()
                var t_max_logit: Float32 = -999999.0
                next_token = 0
                for t in range(num_tasks):
                    if local_max_logits[t] > t_max_logit:
                        t_max_logit = local_max_logits[t]
                        next_token = local_next_tokens[t]

                if next_token == 1 or next_token == 106: # <eos> or <turn|>
                    if t_decode_start == 0.0:
                        t_decode_start = std.time.perf_counter()
                    break

                # Decode and print generated token
                var next_token_list = Python.list()
                next_token_list.append(next_token)
                var decoded = tokenizer.decode(next_token_list)
                py_builtins.print(decoded, end="", flush=True)
                tb1 = std.time.perf_counter()
                if pos >= num_prompt_tokens:
                    p_misc_decode += tb1 - tb0

            pos += 1
            if pos == num_prompt_tokens:
                t_decode_start = std.time.perf_counter()

        var t_end = std.time.perf_counter()
        var prefill_duration = t_decode_start - t_prefill_start
        var decode_duration = t_end - t_decode_start
        var actual_decoded_tokens = pos - num_prompt_tokens

        print("\n---------------------------------------------------------")
        print("Generation complete!")
        if actual_decoded_tokens > 0 and decode_duration > 0:
            var decode_speed = Float64(actual_decoded_tokens) / decode_duration
            print("Benchmark Results:")
            print("  Prefill duration: ", String(prefill_duration * 1000.0), " ms (", String((prefill_duration * 1000.0) / Float64(num_prompt_tokens)), " ms/tok)")
            print("  Decode duration:  ", String(decode_duration * 1000.0), " ms (", String((decode_duration * 1000.0) / Float64(actual_decoded_tokens)), " ms/tok)")
            print("  Decode speed:     ", String(decode_speed), " tokens/sec")
            print("\n---------------------------------------------------------")
            print("Micro-Profiling Report (Decode Phase Only):")
            print("---------------------------------------------------------")
            
            var total_profile_time = p_embed_reconstruct + p_attn_norm + p_qkv_proj + p_head_norm_rope_cache + p_attn_dot_product + p_wo_proj + p_shared_mlp + p_moe_router + p_moe_experts + p_moe_combine + p_final_norm + p_logit_proj + p_misc_decode
            
            def print_row(name: String, time_sec: Float64):
                var pct = (time_sec / total_profile_time) * 100.0 if total_profile_time > 0 else 0.0
                var ms_per_tok = (time_sec * 1000.0) / Float64(actual_decoded_tokens)
                print("  " + pad_right(name, 32) + ": " + pad_left(String(time_sec * 1000.0), 10) + " ms (" + pad_left(String(pct), 6) + "%) [" + pad_left(String(ms_per_tok), 8) + " ms/tok]")

            print_row("Embed Reconstruct", p_embed_reconstruct)
            print_row("Attention Input Norm (RMS)", p_attn_norm)
            print_row("QKV Projections (Matmul)", p_qkv_proj)
            print_row("Head Norm, RoPE & Cache", p_head_norm_rope_cache)
            print_row("Attention Dot-Prod / Softmax", p_attn_dot_product)
            print_row("WO Projection (Matmul)", p_wo_proj)
            print_row("Shared MLP Path (Matmul)", p_shared_mlp)
            print_row("MoE Router Path", p_moe_router)
            print_row("MoE Active Experts (Top-k)", p_moe_experts)
            print_row("MoE Combine & Layer Scale", p_moe_combine)
            print_row("Final RMS Norm", p_final_norm)
            print_row("Logit Projection (LM Head)", p_logit_proj)
            print_row("Misc Decode & Print Overhead", p_misc_decode)
            
            print("  " + pad_right("Total Profiled Time", 32) + ": " + pad_left(String(total_profile_time * 1000.0), 10) + " ms (100.0%) [" + pad_left(String((total_profile_time * 1000.0) / Float64(actual_decoded_tokens)), 8) + " ms/tok]")
            print("---------------------------------------------------------")
            print("Expert Pruning Statistics (Decode Phase Only):")
            print("---------------------------------------------------------")
            print("  Total Experts Evaluated         : " + String(total_experts_evaluated))
            print("  Total Experts Pruned / Skipped  : " + String(total_experts_skipped))
            var skip_pct = (Float64(total_experts_skipped) / Float64(total_experts_evaluated)) * 100.0 if total_experts_evaluated > 0 else 0.0
            print("  Expert Pruning Ratio            : " + String(skip_pct) + "%")
            print("---------------------------------------------------------")

    # Cleanup memory
    free_c(base_addr)
    final_norm_weight.free()
    k_cache.free()
    v_cache.free()
    local_x.free()
    norm_x.free()
    scaled_norm_x.free()
    q.free()
    k_vec.free()
    v_vec.free()
    attn_scores.free()
    attn_out.free()
    proj_attn_out.free()
    shared_h1.free()
    shared_h2.free()
    shared_ffn_out_gate.free()
    shared_mlp_out.free()
    shared_h1_norm.free()
    moe_in.free()
    moe_out.free()
    gate_up_out_all.free()
    act_all.free()
    down_out_all.free()
    moe_out_norm.free()
    combined_ffn.free()
    ffn_out.free()
    scores.free()
    top_idx.free()
    top_val.free()
    top_weights.free()
    target_final_norm_x.free()
    local_max_logits.free()
    local_next_tokens.free()
