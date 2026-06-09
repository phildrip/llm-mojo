from std.ffi import OwnedDLHandle
from std.memory import alloc
from std.math import sqrt, exp, cos, sin
from std.python import Python
import std.os
from uring import IoUring
from std.algorithm import parallelize
from std.sys.info import num_physical_cores
from svd_embed import SvdEmbedding

# -----------------------------------------------------------------------------
# 1. Model Configurations & Memory Offsets
# -----------------------------------------------------------------------------

# Target Model Configuration (Q8_0 layout)
struct TargetConfig[
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

        var wq_packed = base_ptr
        var wk_packed = wq_packed + wq_size_bytes
        var wv_packed = wk_packed + wk_size_bytes
        var wo_packed = wk_packed + wk_size_bytes
        if not is_global:
            wv_packed = wk_packed + wk_size_bytes
            wo_packed = wv_packed + wv_size_bytes
            
        var w1_packed = wo_packed + wo_size_bytes
        var w3_packed = w1_packed + w1_size_bytes
        
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
        
        var unchanged_base = w3_scales + c_blocks_w3
        
        var input_layernorm = unchanged_base
        var post_attention_layernorm = input_layernorm + Self.embed_dim
        var pre_feedforward_layernorm = post_attention_layernorm + Self.embed_dim
        var post_feedforward_layernorm = pre_feedforward_layernorm + Self.embed_dim
        var q_norm = post_feedforward_layernorm + Self.embed_dim
        var k_norm = q_norm + layer_head_dim
        var layer_scalar = k_norm + layer_head_dim
        
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


# Draft Model Configuration (4-bit Q4_0, fully RAM-resident)
struct DraftConfig[
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
    def get_layer_weights(self, layer_idx: Int, base_ptr: UnsafePointer[UInt8, _]) -> Q4LayerWeights:
        var is_global = self.is_global_layer(layer_idx)
        var layer_heads = 16
        var layer_head_dim = 512 if is_global else 256
        var layer_kv_heads = 1 if is_global else 8
        
        var wq_size = layer_heads * layer_head_dim * Self.embed_dim
        var wk_size = layer_kv_heads * layer_head_dim * Self.embed_dim
        var wv_size = 0 if is_global else (layer_kv_heads * layer_head_dim * Self.embed_dim)
        var wo_size = Self.embed_dim * layer_heads * layer_head_dim
        var w1_size = Self.intermediate_dim * Self.embed_dim
        var w2_size = Self.embed_dim * Self.intermediate_dim
        var w3_size = Self.intermediate_dim * Self.embed_dim
        
        var wq_size_bytes = wq_size // 2
        var wk_size_bytes = wk_size // 2
        var wv_size_bytes = wv_size // 2
        var wo_size_bytes = wo_size // 2
        var w1_size_bytes = w1_size // 2
        var w2_size_bytes = w2_size // 2
        var w3_size_bytes = w3_size // 2
        
        var c_blocks_q = wq_size // 32
        var c_blocks_k = wk_size // 32
        var c_blocks_v = wv_size // 32
        var c_blocks_o = wo_size // 32
        var c_blocks_w1 = w1_size // 32
        var c_blocks_w2 = w2_size // 32
        var c_blocks_w3 = w3_size // 32

        # Pointers to packed weights (UInt8)
        var wq_packed = base_ptr
        var wk_packed = wq_packed + wq_size_bytes
        var wv_packed = wk_packed + wk_size_bytes # fallback/initialized
        var wo_packed = wk_packed + wk_size_bytes
        if not is_global:
            wv_packed = wk_packed + wk_size_bytes
            wo_packed = wv_packed + wv_size_bytes
            
        var w1_packed = wo_packed + wo_size_bytes
        var w2_packed = w1_packed + w1_size_bytes
        var w3_packed = w2_packed + w2_size_bytes
        
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
        var w2_scales = w1_scales + c_blocks_w1
        var w3_scales = w2_scales + c_blocks_w2
        
        # Unchanged Float32 parameters base address: immediately after the end of w3_scales
        var unchanged_base = w3_scales + c_blocks_w3
        
        var input_layernorm = unchanged_base
        var post_attention_layernorm = input_layernorm + Self.embed_dim
        var pre_feedforward_layernorm = post_attention_layernorm + Self.embed_dim
        var post_feedforward_layernorm = pre_feedforward_layernorm + Self.embed_dim
        var q_norm = post_feedforward_layernorm + Self.embed_dim
        var k_norm = q_norm + layer_head_dim
        var layer_scalar = k_norm + layer_head_dim
        
        return Q4LayerWeights(
            UnsafePointer[UInt8, MutExternalOrigin](unsafe_from_address=wq_packed.__int__()),
            UnsafePointer[UInt8, MutExternalOrigin](unsafe_from_address=wk_packed.__int__()),
            UnsafePointer[UInt8, MutExternalOrigin](unsafe_from_address=wv_packed.__int__()),
            UnsafePointer[UInt8, MutExternalOrigin](unsafe_from_address=wo_packed.__int__()),
            UnsafePointer[UInt8, MutExternalOrigin](unsafe_from_address=w1_packed.__int__()),
            UnsafePointer[UInt8, MutExternalOrigin](unsafe_from_address=w2_packed.__int__()),
            UnsafePointer[UInt8, MutExternalOrigin](unsafe_from_address=w3_packed.__int__()),
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

    @always_inline
    def get_layer_aligned_size(self, layer_idx: Int) -> Int:
        var is_global = self.is_global_layer(layer_idx)
        var layer_heads = 16
        var layer_head_dim = 512 if is_global else 256
        var layer_kv_heads = 1 if is_global else 8
        
        var wq_size = layer_heads * layer_head_dim * Self.embed_dim
        var wk_size = layer_kv_heads * layer_head_dim * Self.embed_dim
        var wv_size = 0 if is_global else (layer_kv_heads * layer_head_dim * Self.embed_dim)
        var wo_size = Self.embed_dim * layer_heads * layer_head_dim
        var w1_size = Self.intermediate_dim * Self.embed_dim
        var w2_size = Self.embed_dim * Self.intermediate_dim
        var w3_size = Self.intermediate_dim * Self.embed_dim
        
        # Packed Q4 weights: 2 elements per byte
        var packed_weights_size = (wq_size + wk_size + wv_size + wo_size + w1_size + w2_size + w3_size) // 2
        
        # Scale factors (1 per block of 32 elements)
        var scales_count = (wq_size + wk_size + wv_size + wo_size + w1_size + w2_size + w3_size) // 32
        var scales_size_bytes = scales_count * 4
        
        # Float32 unchanged weights
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

    def get_layer_offset(self, layer_idx: Int) -> Int:
        var offset = 0
        for i in range(layer_idx):
            offset += self.get_layer_aligned_size(i)
        return offset

    def __init__(out self):
        pass


# -----------------------------------------------------------------------------
# 2. Structural Layer Weights Representations
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


struct Q4LayerWeights:
    var wq_packed: UnsafePointer[UInt8, MutExternalOrigin]
    var wk_packed: UnsafePointer[UInt8, MutExternalOrigin]
    var wv_packed: UnsafePointer[UInt8, MutExternalOrigin]
    var wo_packed: UnsafePointer[UInt8, MutExternalOrigin]
    var w1_packed: UnsafePointer[UInt8, MutExternalOrigin]
    var w2_packed: UnsafePointer[UInt8, MutExternalOrigin]
    var w3_packed: UnsafePointer[UInt8, MutExternalOrigin]

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
        wq_packed: UnsafePointer[UInt8, MutExternalOrigin],
        wk_packed: UnsafePointer[UInt8, MutExternalOrigin],
        wv_packed: UnsafePointer[UInt8, MutExternalOrigin],
        wo_packed: UnsafePointer[UInt8, MutExternalOrigin],
        w1_packed: UnsafePointer[UInt8, MutExternalOrigin],
        w2_packed: UnsafePointer[UInt8, MutExternalOrigin],
        w3_packed: UnsafePointer[UInt8, MutExternalOrigin],
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


# --- 4-bit Q4_0 Unpacking & Matmul ---
@always_inline
def unpack_4bytes(b0: UInt8, b1: UInt8, b2: UInt8, b3: UInt8) -> SIMD[DType.float32, 8]:
    return SIMD[DType.float32, 8](
        Float32(Int(b0 & 15) - 8),
        Float32(Int((b0 >> 4) & 15) - 8),
        Float32(Int(b1 & 15) - 8),
        Float32(Int((b1 >> 4) & 15) - 8),
        Float32(Int(b2 & 15) - 8),
        Float32(Int((b2 >> 4) & 15) - 8),
        Float32(Int(b3 & 15) - 8),
        Float32(Int((b3 >> 4) & 15) - 8)
    )

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
def fast_tanh(x: Float32) -> Float32:
    if x > 15.0:
        return 1.0
    elif x < -15.0:
        return -1.0
    var e2x = exp(2.0 * x)
    return (e2x - 1.0) / (e2x + 1.0)

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
        if attempts > 30000000:
            raise Error("Weights paging I/O timed out waiting for layer tag: " + String(expected_tag))

# -----------------------------------------------------------------------------
# 4. Main Executive Engine
# -----------------------------------------------------------------------------
def main() raises:
    var num_tasks = num_physical_cores()
    print("=========================================================")
    print("   GEMMA 4 12B SPECULATIVE DECODING ENGINE (Mojo)        ")
    print("=========================================================")
    print("Embedding Size:      3840")
    print("Attention Heads:     16 Query Heads / 8 KV Heads (GQA/MQA)")
    print("Model Parameters:    12B Scale Causal Speculative Layout")
    print("Kernel Paging I/O:   Linux io_uring (Direct NVMe double-buf)")
    print("Draft Model State:   Fully RAM-resident Q4_0 (4-bit)")
    print("Target Model State:  Direct NVMe-paged Q8_0 (Block column)")
    print("---------------------------------------------------------")

    var TARGET_CONFIG = TargetConfig[
        embed_dim = 3840,
        head_dim = 256,
        num_heads = 16,
        num_kv_heads = 8,
        intermediate_dim = 15360,
        num_layers = 48,
        vocab_size = 262144,
        max_seq_len = 100
    ]()

    var DRAFT_CONFIG = DraftConfig[
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
    var c_read = libc.get_function[def(Int32, Int, Int) -> Int]("read")
    
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

    # Setup Page-Aligned Layer Weight Buffers (Double-buffering) for Q8 Target model
    var max_layer_bytes = 0
    for i in range(TARGET_CONFIG.num_layers):
        var sz = TARGET_CONFIG.get_layer_aligned_size(i)
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

    # 2. Open converted target weights file on NVMe with O_DIRECT
    var weights_file = "gemma4_weights_q8_sorted.bin"
    print("Opening converted target weights file '" + weights_file + "' with O_DIRECT read-only...")
    var path_bytes = weights_file.as_bytes()
    var path_ptr = alloc[UInt8](len(path_bytes) + 1)
    for i in range(len(path_bytes)):
        path_ptr[i] = path_bytes[i]
    path_ptr[len(path_bytes)] = 0

    # Flags: O_RDONLY (0) | O_DIRECT (16384) = 16384
    var weights_fd = c_open(path_ptr.__int__(), 16384, 0)
    if weights_fd < 0:
        print("CRITICAL: Failed to open target weights file with O_DIRECT.")
        return
    else:
        var total_size = TARGET_CONFIG.get_layer_offset(TARGET_CONFIG.num_layers)
        print("Target weights file opened successfully. Active size: ", total_size, " bytes")

    # 3. Load SVD embeddings (compressed) and final layer norm
    print("Loading SVD embeddings from gemma4_embed_svd_d32.bin...")
    var svd = SvdEmbedding("gemma4_embed_svd_d32.bin")

    var final_norm_file = "gemma4_final_norm.bin"
    var final_norm_path_bytes = final_norm_file.as_bytes()
    var final_norm_path_ptr = alloc[UInt8](len(final_norm_path_bytes) + 1)
    for i in range(len(final_norm_path_bytes)):
        final_norm_path_ptr[i] = final_norm_path_bytes[i]
    final_norm_path_ptr[len(final_norm_path_bytes)] = 0
    var final_norm_fd = c_open(final_norm_path_ptr.__int__(), 0, 0)
    if final_norm_fd < 0:
        print("CRITICAL: Failed to open 'gemma4_final_norm.bin'!")
        return

    var final_norm_weight = alloc[Float32](TARGET_CONFIG.embed_dim)
    var final_norm_size_bytes = TARGET_CONFIG.embed_dim * 4
    total_read = 0
    while total_read < final_norm_size_bytes:
        var bytes_to_read = final_norm_size_bytes - total_read
        var read_res = c_read(final_norm_fd, final_norm_weight.__int__() + total_read, bytes_to_read)
        if read_res <= 0:
            print("CRITICAL: Failed to read from gemma4_final_norm.bin!")
            return
        total_read += read_res
    print("Loaded final norm weights into memory: ", total_read, " bytes")
    print("---------------------------------------------------------")
    _ = c_close(final_norm_fd)
    final_norm_path_ptr.free()

    # 4. Load Q4 Draft Weights FULLY into memory (RAM-resident)
    var draft_file = "gemma4_weights_q4.bin"
    print("Opening Q4_0 draft weights file '" + draft_file + "'...")
    var draft_path_bytes = draft_file.as_bytes()
    var draft_path_ptr = alloc[UInt8](len(draft_path_bytes) + 1)
    for i in range(len(draft_path_bytes)):
        draft_path_ptr[i] = draft_path_bytes[i]
    draft_path_ptr[len(draft_path_bytes)] = 0
    var draft_fd = c_open(draft_path_ptr.__int__(), 0, 0)
    if draft_fd < 0:
        print("CRITICAL: Failed to open 'gemma4_weights_q4.bin'!")
        return

    var draft_file_size = DRAFT_CONFIG.get_layer_offset(DRAFT_CONFIG.num_layers)
    var memptr_draft = alloc[Int](1)
    _ = posix_memalign(memptr_draft.__int__(), 4096, draft_file_size)
    var draft_base_addr = memptr_draft[0]
    var draft_base_ptr = UnsafePointer[UInt8, MutExternalOrigin](unsafe_from_address=draft_base_addr)

    print("Loading entire Q4_0 weights file into memory (6.82 GB)...")
    total_read = 0
    while total_read < draft_file_size:
        var bytes_to_read = draft_file_size - total_read
        var read_res = c_read(draft_fd, draft_base_addr + total_read, bytes_to_read)
        if read_res <= 0:
            print("CRITICAL: Failed to read from gemma4_weights_q4.bin!")
            return
        total_read += read_res
    print("Loaded Q4 draft weights fully into RAM: ", total_read, " bytes")
    print("---------------------------------------------------------")
    _ = c_close(draft_fd)
    draft_path_ptr.free()

    # 5. Allocate continuous KV caches and activation buffers
    # Precompute layer cache offsets and total cache size
    var layer_cache_offsets = List[Int]()
    var current_cache_offset = 0
    for i in range(TARGET_CONFIG.num_layers):
        layer_cache_offsets.append(current_cache_offset)
        var is_global = TARGET_CONFIG.is_global_layer(i)
        var layer_kv_heads = 1 if is_global else 8
        var layer_head_dim = 512 if is_global else 256
        current_cache_offset += TARGET_CONFIG.max_seq_len * layer_kv_heads * layer_head_dim
    var total_cache_size = current_cache_offset

    # Draft (Ternary) KV Cache
    var draft_k_cache = alloc[Float32](total_cache_size)
    var draft_v_cache = alloc[Float32](total_cache_size)
    
    # Target (Q8_0) KV Cache
    var target_k_cache = alloc[Float32](total_cache_size)
    var target_v_cache = alloc[Float32](total_cache_size)

    for i in range(total_cache_size):
        draft_k_cache[i] = 0.0
        draft_v_cache[i] = 0.0
        target_k_cache[i] = 0.0
        target_v_cache[i] = 0.0

    # Common shared draft activations
    var draft_x = alloc[Float32](DRAFT_CONFIG.embed_dim)
    var draft_norm_x = alloc[Float32](DRAFT_CONFIG.embed_dim)
    var draft_q = alloc[Float32](8192)
    var draft_k_vec = alloc[Float32](2048)
    var draft_v_vec = alloc[Float32](2048)
    var draft_attn_scores = alloc[Float32](DRAFT_CONFIG.max_seq_len)
    var draft_attn_out = alloc[Float32](8192)
    var draft_proj_attn_out = alloc[Float32](DRAFT_CONFIG.embed_dim)
    var draft_h1 = alloc[Float32](DRAFT_CONFIG.intermediate_dim)
    var draft_h2 = alloc[Float32](DRAFT_CONFIG.intermediate_dim)
    var draft_ffn_out_gate = alloc[Float32](DRAFT_CONFIG.intermediate_dim)
    var draft_ffn_out = alloc[Float32](DRAFT_CONFIG.embed_dim)

    # Common target activations (can validation-batch up to K+1 tokens sequentially inside layer loops)
    comptime K_MAX = 5
    var target_x_seq = alloc[Float32]((K_MAX + 1) * TARGET_CONFIG.embed_dim)
    var target_norm_x = alloc[Float32](TARGET_CONFIG.embed_dim)
    var target_q = alloc[Float32](8192)
    var target_k_vec = alloc[Float32](2048)
    var target_v_vec = alloc[Float32](2048)
    var target_attn_scores = alloc[Float32](TARGET_CONFIG.max_seq_len)
    var target_attn_out = alloc[Float32](8192)
    var target_proj_attn_out = alloc[Float32](TARGET_CONFIG.embed_dim)
    var target_h1 = alloc[Float32](TARGET_CONFIG.intermediate_dim)
    var target_h2 = alloc[Float32](TARGET_CONFIG.intermediate_dim)
    var target_ffn_out_gate = alloc[Float32](TARGET_CONFIG.intermediate_dim)
    var target_ffn_out = alloc[Float32](TARGET_CONFIG.embed_dim)
    var target_final_norm_x = alloc[Float32](TARGET_CONFIG.embed_dim)

    # Temporary variable structures for projection worker
    var chunk_size = TARGET_CONFIG.vocab_size // num_tasks
    var local_max_logits = alloc[Float32](num_tasks)
    var local_next_tokens = alloc[Int](num_tasks)

    # Boot Interactive Conversation
    print("=========================================================")
    print("   GEMMA 4 12B-SCALE SPECULATIVE CHAT AGENT              ")
    print("=========================================================")
    print("Embedding Size:      3840")
    print("Attention Heads:     16 Query Heads / 8 KV Heads (GQA/MQA)")
    print("Max Sequence Length: 100")
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
        
        var num_prompt_tokens = len(prompt_tokens)
        var max_steps = num_prompt_tokens + 30
        if max_steps > TARGET_CONFIG.max_seq_len:
            max_steps = TARGET_CONFIG.max_seq_len
            
        var scale = sqrt(Float32(TARGET_CONFIG.embed_dim))
        
        # 1. Draft Prefill Phase: process all prompt tokens layer-by-layer (fully in memory, extremely fast)
        var draft_prompt_x = alloc[Float32](num_prompt_tokens * DRAFT_CONFIG.embed_dim)
        var draft_prompt_ptr = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=draft_prompt_x.__int__())
        for pos in range(num_prompt_tokens):
            var px_offset = pos * DRAFT_CONFIG.embed_dim
            svd.reconstruct(prompt_tokens[pos], draft_prompt_ptr + px_offset)
            for i in range(DRAFT_CONFIG.embed_dim):
                draft_prompt_x[px_offset + i] *= scale

        for layer_idx in range(DRAFT_CONFIG.num_layers):
            var weights = DRAFT_CONFIG.get_layer_weights(layer_idx, draft_base_ptr + DRAFT_CONFIG.get_layer_offset(layer_idx))
            var is_global = DRAFT_CONFIG.is_global_layer(layer_idx)
            var layer_heads = 16
            var layer_kv_heads = 1 if is_global else 8
            var layer_head_dim = 512 if is_global else 256
            var kv_stride = layer_kv_heads * layer_head_dim
            var ratio = layer_heads // layer_kv_heads
            
            for pos in range(num_prompt_tokens):
                var px_offset = pos * DRAFT_CONFIG.embed_dim
                var local_x = draft_prompt_x + px_offset
                
                rms_norm(local_x, weights.input_layernorm_weight, draft_norm_x, DRAFT_CONFIG.embed_dim)
                
                q4_matmul_vec(weights.wq_packed, weights.wq_scales, draft_norm_x, draft_q, DRAFT_CONFIG.embed_dim, layer_heads * layer_head_dim, num_tasks)
                q4_matmul_vec(weights.wk_packed, weights.wk_scales, draft_norm_x, draft_k_vec, DRAFT_CONFIG.embed_dim, layer_kv_heads * layer_head_dim, num_tasks)
                
                if is_global:
                    for i in range(kv_stride):
                        draft_v_vec[i] = draft_k_vec[i]
                else:
                    q4_matmul_vec(weights.wv_packed, weights.wv_scales, draft_norm_x, draft_v_vec, DRAFT_CONFIG.embed_dim, layer_kv_heads * layer_head_dim, num_tasks)
                    
                for h in range(layer_heads):
                    var head_ptr = draft_q + h * layer_head_dim
                    rms_norm(head_ptr, weights.q_norm_weight, head_ptr, layer_head_dim)
                for h in range(layer_kv_heads):
                    var head_ptr = draft_k_vec + h * layer_head_dim
                    rms_norm(head_ptr, weights.k_norm_weight, head_ptr, layer_head_dim)
                for h in range(layer_kv_heads):
                    var head_ptr = draft_v_vec + h * layer_head_dim
                    rms_norm_noscale(head_ptr, head_ptr, layer_head_dim)
                    
                apply_rope(draft_q, layer_heads, layer_head_dim, pos, is_global)
                apply_rope(draft_k_vec, layer_kv_heads, layer_head_dim, pos, is_global)
                
                var layer_cache_offset = layer_cache_offsets[layer_idx]
                var step_cache_offset = layer_cache_offset + pos * kv_stride
                for i in range(kv_stride):
                    draft_k_cache[step_cache_offset + i] = draft_k_vec[i]
                    draft_v_cache[step_cache_offset + i] = draft_v_vec[i]
                    
                for h in range(layer_heads):
                    var h_offset = h * layer_head_dim
                    var g = h // ratio
                    var g_offset = g * layer_head_dim
                    for t in range(pos + 1):
                        var t_cache_offset = layer_cache_offset + t * kv_stride + g_offset
                        var score: Float32 = 0.0
                        for d in range(layer_head_dim):
                            score += draft_q[h_offset + d] * draft_k_cache[t_cache_offset + d]
                        draft_attn_scores[t] = score
                        
                    softmax(draft_attn_scores, pos + 1)
                    
                    for d in range(layer_head_dim):
                        var val_sum: Float32 = 0.0
                        for t in range(pos + 1):
                            var t_cache_offset = layer_cache_offset + t * kv_stride + g_offset
                            val_sum += draft_attn_scores[t] * draft_v_cache[t_cache_offset + d]
                        draft_attn_out[h_offset + d] = val_sum
                        
                q4_matmul_vec(weights.wo_packed, weights.wo_scales, draft_attn_out, draft_proj_attn_out, layer_heads * layer_head_dim, DRAFT_CONFIG.embed_dim, num_tasks)
                
                rms_norm(draft_proj_attn_out, weights.post_attention_layernorm_weight, draft_proj_attn_out, DRAFT_CONFIG.embed_dim)
                for i in range(DRAFT_CONFIG.embed_dim):
                    local_x[i] += draft_proj_attn_out[i]
                    
                rms_norm(local_x, weights.pre_feedforward_layernorm_weight, draft_norm_x, DRAFT_CONFIG.embed_dim)
                q4_matmul_vec(weights.w1_packed, weights.w1_scales, draft_norm_x, draft_h1, DRAFT_CONFIG.embed_dim, DRAFT_CONFIG.intermediate_dim, num_tasks)
                q4_matmul_vec(weights.w3_packed, weights.w3_scales, draft_norm_x, draft_h2, DRAFT_CONFIG.embed_dim, DRAFT_CONFIG.intermediate_dim, num_tasks)
                gelu_and_mul(draft_h1, draft_h2, draft_ffn_out_gate, DRAFT_CONFIG.intermediate_dim)
                
                q4_matmul_vec(weights.w2_packed, weights.w2_scales, draft_ffn_out_gate, draft_ffn_out, DRAFT_CONFIG.intermediate_dim, DRAFT_CONFIG.embed_dim, num_tasks)
                
                rms_norm(draft_ffn_out, weights.post_feedforward_layernorm_weight, draft_ffn_out, DRAFT_CONFIG.embed_dim)
                for i in range(DRAFT_CONFIG.embed_dim):
                    local_x[i] += draft_ffn_out[i]
                    
                var scalar = weights.layer_scalar[0]
                for i in range(DRAFT_CONFIG.embed_dim):
                    local_x[i] *= scalar
        draft_prompt_x.free()

        # 2. Target Model Prefill Phase (disk-paged)
        var t_prefill_start = py.time()
        var target_prompt_x = alloc[Float32](num_prompt_tokens * TARGET_CONFIG.embed_dim)
        var target_prompt_ptr = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=target_prompt_x.__int__())
        for pos in range(num_prompt_tokens):
            var px_offset = pos * TARGET_CONFIG.embed_dim
            svd.reconstruct(prompt_tokens[pos], target_prompt_ptr + px_offset)
            for i in range(TARGET_CONFIG.embed_dim):
                target_prompt_x[px_offset + i] *= scale
                
        # Load Layer 0 target weights (stable base)
        var tag_0: UInt64 = UInt64(8000)
        _ = ring.submit_read(weights_fd, w_a_ptr.__int__(), TARGET_CONFIG.get_w2_offset_in_layer(0), 0, tag_0)
        wait_for_tag(ring, tag_0, out_tag_ptr)

        for layer_idx in range(TARGET_CONFIG.num_layers):
            var active_weights_ptr = w_a_ptr if layer_idx % 2 == 0 else w_b_ptr
            var next_weights_ptr = w_b_ptr if layer_idx % 2 == 0 else w_a_ptr
            
            # Prefetch next layer
            if layer_idx + 1 < TARGET_CONFIG.num_layers:
                var tag_next: UInt64 = UInt64(8000 + layer_idx + 1)
                var offset_next = TARGET_CONFIG.get_layer_offset(layer_idx + 1)
                var size_next = TARGET_CONFIG.get_w2_offset_in_layer(layer_idx + 1)
                _ = ring.submit_read(weights_fd, next_weights_ptr.__int__(), size_next, offset_next, tag_next)
                
            var weights = TARGET_CONFIG.get_layer_weights(layer_idx, active_weights_ptr)
            var is_global = TARGET_CONFIG.is_global_layer(layer_idx)
            var layer_heads = 16
            var layer_kv_heads = 1 if is_global else 8
            var layer_head_dim = 512 if is_global else 256
            var kv_stride = layer_kv_heads * layer_head_dim
            var ratio = layer_heads // layer_kv_heads
            
            var next_layer_completed = False
            
            for pos in range(num_prompt_tokens):
                var px_offset = pos * TARGET_CONFIG.embed_dim
                var local_x = target_prompt_x + px_offset
                
                rms_norm(local_x, weights.input_layernorm_weight, target_norm_x, TARGET_CONFIG.embed_dim)
                
                q8_matmul_vec(weights.wq_packed, weights.wq_scales, target_norm_x, target_q, TARGET_CONFIG.embed_dim, layer_heads * layer_head_dim, num_tasks)
                q8_matmul_vec(weights.wk_packed, weights.wk_scales, target_norm_x, target_k_vec, TARGET_CONFIG.embed_dim, layer_kv_heads * layer_head_dim, num_tasks)
                
                if is_global:
                    for i in range(kv_stride):
                        target_v_vec[i] = target_k_vec[i]
                else:
                    q8_matmul_vec(weights.wv_packed, weights.wv_scales, target_norm_x, target_v_vec, TARGET_CONFIG.embed_dim, layer_kv_heads * layer_head_dim, num_tasks)
                    
                for h in range(layer_heads):
                    var head_ptr = target_q + h * layer_head_dim
                    rms_norm(head_ptr, weights.q_norm_weight, head_ptr, layer_head_dim)
                for h in range(layer_kv_heads):
                    var head_ptr = target_k_vec + h * layer_head_dim
                    rms_norm(head_ptr, weights.k_norm_weight, head_ptr, layer_head_dim)
                for h in range(layer_kv_heads):
                    var head_ptr = target_v_vec + h * layer_head_dim
                    rms_norm_noscale(head_ptr, head_ptr, layer_head_dim)
                    
                apply_rope(target_q, layer_heads, layer_head_dim, pos, is_global)
                apply_rope(target_k_vec, layer_kv_heads, layer_head_dim, pos, is_global)
                
                var layer_cache_offset = layer_cache_offsets[layer_idx]
                var step_cache_offset = layer_cache_offset + pos * kv_stride
                for i in range(kv_stride):
                    target_k_cache[step_cache_offset + i] = target_k_vec[i]
                    target_v_cache[step_cache_offset + i] = target_v_vec[i]
                    
                for h in range(layer_heads):
                    var h_offset = h * layer_head_dim
                    var g = h // ratio
                    var g_offset = g * layer_head_dim
                    
                    for t in range(pos + 1):
                        var t_cache_offset = layer_cache_offset + t * kv_stride + g_offset
                        var score: Float32 = 0.0
                        for d in range(layer_head_dim):
                            score += target_q[h_offset + d] * target_k_cache[t_cache_offset + d]
                        target_attn_scores[t] = score
                        
                    softmax(target_attn_scores, pos + 1)
                    
                    for d in range(layer_head_dim):
                        var val_sum: Float32 = 0.0
                        for t in range(pos + 1):
                            var t_cache_offset = layer_cache_offset + t * kv_stride + g_offset
                            val_sum += target_attn_scores[t] * target_v_cache[t_cache_offset + d]
                        target_attn_out[h_offset + d] = val_sum
                        
                q8_matmul_vec(weights.wo_packed, weights.wo_scales, target_attn_out, target_proj_attn_out, layer_heads * layer_head_dim, TARGET_CONFIG.embed_dim, num_tasks)
                
                rms_norm(target_proj_attn_out, weights.post_attention_layernorm_weight, target_proj_attn_out, TARGET_CONFIG.embed_dim)
                for i in range(TARGET_CONFIG.embed_dim):
                    local_x[i] += target_proj_attn_out[i]
                    
                rms_norm(local_x, weights.pre_feedforward_layernorm_weight, target_norm_x, TARGET_CONFIG.embed_dim)
                q8_matmul_vec(weights.w1_packed, weights.w1_scales, target_norm_x, target_h1, TARGET_CONFIG.embed_dim, TARGET_CONFIG.intermediate_dim, num_tasks)
                q8_matmul_vec(weights.w3_packed, weights.w3_scales, target_norm_x, target_h2, TARGET_CONFIG.embed_dim, TARGET_CONFIG.intermediate_dim, num_tasks)
                gelu_and_mul(target_h1, target_h2, target_ffn_out_gate, TARGET_CONFIG.intermediate_dim)
                
                # W2 sparse loading
                var num_blocks = TARGET_CONFIG.intermediate_dim // 32
                var layer_offset = TARGET_CONFIG.get_layer_offset(layer_idx)
                var w2_offset_in_layer = TARGET_CONFIG.get_w2_offset_in_layer(layer_idx)
                var tag_base_for_w2: UInt64 = 100000 + UInt64(layer_idx) * 1000 + UInt64(pos) * 1000000
                
                var max_active_block: Int = -1
                for b in range(num_blocks - 1, -1, -1):
                    var block_offset_gate = b * 32
                    var block_active = False
                    for c in range(32):
                        if abs(target_ffn_out_gate[block_offset_gate + c]) > 1e-4:
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
                         
                q8_matmul_w2_block_column(weights.w2_packed, target_ffn_out_gate, target_ffn_out, TARGET_CONFIG.intermediate_dim, TARGET_CONFIG.embed_dim, num_tasks, Float32(1e-4))
                
                rms_norm(target_ffn_out, weights.post_feedforward_layernorm_weight, target_ffn_out, TARGET_CONFIG.embed_dim)
                for i in range(TARGET_CONFIG.embed_dim):
                    local_x[i] += target_ffn_out[i]
                    
                var scalar = weights.layer_scalar[0]
                for i in range(TARGET_CONFIG.embed_dim):
                    local_x[i] *= scalar

            if layer_idx + 1 < TARGET_CONFIG.num_layers:
                if not next_layer_completed:
                    var tag_next_val: UInt64 = UInt64(8000 + layer_idx + 1)
                    wait_for_tag(ring, tag_next_val, out_tag_ptr)

        # Retrieve final prefill hidden state of prompt
        var last_px_offset = (num_prompt_tokens - 1) * TARGET_CONFIG.embed_dim
        var target_final_x = alloc[Float32](TARGET_CONFIG.embed_dim)
        for i in range(TARGET_CONFIG.embed_dim):
            target_final_x[i] = target_prompt_x[last_px_offset + i]
        target_prompt_x.free()
        
        rms_norm(target_final_x, final_norm_weight, target_final_norm_x, TARGET_CONFIG.embed_dim)
        target_final_x.free()

        # Low-rank SVD projection for target prefill output
        var z = alloc[Float32](svd.rank)
        for j in range(svd.rank):
            var acc = SIMD[DType.float32, 8](0.0)
            for c in range(0, TARGET_CONFIG.embed_dim, 8):
                acc = acc + target_final_norm_x.load[8](c) * svd.B.load[8](j * TARGET_CONFIG.embed_dim + c)
            z[j] = acc.reduce_add()

        # Project output to vocabulary for the first token
        def initial_proj_worker(t: Int) capturing -> None:
            var start_r = t * chunk_size
            var end_r = (t + 1) * chunk_size if t < num_tasks - 1 else TARGET_CONFIG.vocab_size
            var local_max_l: Float32 = -999999.0
            var local_tok = 0
            for r in range(start_r, end_r):
                var r_offset = r * svd.rank
                var acc = SIMD[DType.float32, 8](0.0)
                for j in range(0, svd.rank, 8):
                    acc = acc + z.load[8](j) * svd.A.load[8](r_offset + j)
                var logit = acc.reduce_add()
                logit = Float32(30.0) * fast_tanh(logit / Float32(30.0))
                if logit > local_max_l:
                    local_max_l = logit
                    local_tok = r
            local_max_logits[t] = local_max_l
            local_next_tokens[t] = local_tok
            
        parallelize[initial_proj_worker](num_tasks)
        z.free()
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
        
        var current_token = next_token
        var pos = num_prompt_tokens
        
        var t_prefill_end = py.time()
        var prefill_time = Float64(py=(t_prefill_end - t_prefill_start))
        print("\n[Mojo Bench] Prefill completed in", prefill_time, "sec (" + String(Float64(num_prompt_tokens) / prefill_time) + " t/s)")
        
        # 3. Speculative Generation Phase
        var tokens_generated = 1
        var t_gen_start = py.time()
        
        var current_K = 4
        var total_accepted = 0
        var total_speculative_steps = 0
        
        while pos < max_steps:
            var K_actual = current_K
            if pos + K_actual >= max_steps:
                K_actual = max_steps - pos - 1
            if K_actual <= 0:
                break
                
            # A. Draft Model Speculative Pass: generate K_actual candidate tokens sequential in memory
            var draft_tokens = List[Int]()
            draft_tokens.append(current_token)
            
            for j in range(K_actual):
                var curr = draft_tokens[j]
                svd.reconstruct(curr, draft_x)
                for i in range(DRAFT_CONFIG.embed_dim):
                    draft_x[i] *= scale
                    
                # Run Draft 48 layers in memory
                for layer_idx in range(DRAFT_CONFIG.num_layers):
                    var weights = DRAFT_CONFIG.get_layer_weights(layer_idx, draft_base_ptr + DRAFT_CONFIG.get_layer_offset(layer_idx))
                    var is_global = DRAFT_CONFIG.is_global_layer(layer_idx)
                    var layer_heads = 16
                    var layer_kv_heads = 1 if is_global else 8
                    var layer_head_dim = 512 if is_global else 256
                    var kv_stride = layer_kv_heads * layer_head_dim
                    var ratio = layer_heads // layer_kv_heads
                    
                    rms_norm(draft_x, weights.input_layernorm_weight, draft_norm_x, DRAFT_CONFIG.embed_dim)
                    
                    q4_matmul_vec(weights.wq_packed, weights.wq_scales, draft_norm_x, draft_q, DRAFT_CONFIG.embed_dim, layer_heads * layer_head_dim, num_tasks)
                    q4_matmul_vec(weights.wk_packed, weights.wk_scales, draft_norm_x, draft_k_vec, DRAFT_CONFIG.embed_dim, layer_kv_heads * layer_head_dim, num_tasks)
                    
                    if is_global:
                        for i in range(kv_stride):
                            draft_v_vec[i] = draft_k_vec[i]
                    else:
                        q4_matmul_vec(weights.wv_packed, weights.wv_scales, draft_norm_x, draft_v_vec, DRAFT_CONFIG.embed_dim, layer_kv_heads * layer_head_dim, num_tasks)
                        
                    for h in range(layer_heads):
                        var head_ptr = draft_q + h * layer_head_dim
                        rms_norm(head_ptr, weights.q_norm_weight, head_ptr, layer_head_dim)
                    for h in range(layer_kv_heads):
                        var head_ptr = draft_k_vec + h * layer_head_dim
                        rms_norm(head_ptr, weights.k_norm_weight, head_ptr, layer_head_dim)
                    for h in range(layer_kv_heads):
                        var head_ptr = draft_v_vec + h * layer_head_dim
                        rms_norm_noscale(head_ptr, head_ptr, layer_head_dim)
                        
                    apply_rope(draft_q, layer_heads, layer_head_dim, pos + j, is_global)
                    apply_rope(draft_k_vec, layer_kv_heads, layer_head_dim, pos + j, is_global)
                    
                    var layer_cache_offset = layer_cache_offsets[layer_idx]
                    var step_cache_offset = layer_cache_offset + (pos + j) * kv_stride
                    for i in range(kv_stride):
                        draft_k_cache[step_cache_offset + i] = draft_k_vec[i]
                        draft_v_cache[step_cache_offset + i] = draft_v_vec[i]
                        
                    for h in range(layer_heads):
                        var h_offset = h * layer_head_dim
                        var g = h // ratio
                        var g_offset = g * layer_head_dim
                        for t in range(pos + j + 1):
                            var t_cache_offset = layer_cache_offset + t * kv_stride + g_offset
                            var score: Float32 = 0.0
                            for d in range(layer_head_dim):
                                score += draft_q[h_offset + d] * draft_k_cache[t_cache_offset + d]
                            draft_attn_scores[t] = score
                            
                        softmax(draft_attn_scores, pos + j + 1)
                        
                        for d in range(layer_head_dim):
                            var val_sum: Float32 = 0.0
                            for t in range(pos + j + 1):
                                var t_cache_offset = layer_cache_offset + t * kv_stride + g_offset
                                val_sum += draft_attn_scores[t] * draft_v_cache[t_cache_offset + d]
                            draft_attn_out[h_offset + d] = val_sum
                            
                    q4_matmul_vec(weights.wo_packed, weights.wo_scales, draft_attn_out, draft_proj_attn_out, layer_heads * layer_head_dim, DRAFT_CONFIG.embed_dim, num_tasks)
                    
                    rms_norm(draft_proj_attn_out, weights.post_attention_layernorm_weight, draft_proj_attn_out, DRAFT_CONFIG.embed_dim)
                    for i in range(DRAFT_CONFIG.embed_dim):
                        draft_x[i] += draft_proj_attn_out[i]
                        
                    rms_norm(draft_x, weights.pre_feedforward_layernorm_weight, draft_norm_x, DRAFT_CONFIG.embed_dim)
                    q4_matmul_vec(weights.w1_packed, weights.w1_scales, draft_norm_x, draft_h1, DRAFT_CONFIG.embed_dim, DRAFT_CONFIG.intermediate_dim, num_tasks)
                    q4_matmul_vec(weights.w3_packed, weights.w3_scales, draft_norm_x, draft_h2, DRAFT_CONFIG.embed_dim, DRAFT_CONFIG.intermediate_dim, num_tasks)
                    gelu_and_mul(draft_h1, draft_h2, draft_ffn_out_gate, DRAFT_CONFIG.intermediate_dim)
                    
                    q4_matmul_vec(weights.w2_packed, weights.w2_scales, draft_ffn_out_gate, draft_ffn_out, DRAFT_CONFIG.intermediate_dim, DRAFT_CONFIG.embed_dim, num_tasks)
                    
                    rms_norm(draft_ffn_out, weights.post_feedforward_layernorm_weight, draft_ffn_out, DRAFT_CONFIG.embed_dim)
                    for i in range(DRAFT_CONFIG.embed_dim):
                        draft_x[i] += draft_ffn_out[i]
                        
                    var scalar = weights.layer_scalar[0]
                    for i in range(DRAFT_CONFIG.embed_dim):
                        draft_x[i] *= scalar

                rms_norm(draft_x, final_norm_weight, draft_norm_x, DRAFT_CONFIG.embed_dim)
                
                # Project draft hidden state to find next candidate token using low-rank SVD
                var z_draft = alloc[Float32](svd.rank)
                for j in range(svd.rank):
                    var acc = SIMD[DType.float32, 8](0.0)
                    for c in range(0, DRAFT_CONFIG.embed_dim, 8):
                        acc = acc + draft_norm_x.load[8](c) * svd.B.load[8](j * DRAFT_CONFIG.embed_dim + c)
                    z_draft[j] = acc.reduce_add()

                var draft_chunk_size = DRAFT_CONFIG.vocab_size // num_tasks
                def draft_proj_worker(t: Int) capturing -> None:
                    var start_r = t * draft_chunk_size
                    var end_r = (t + 1) * draft_chunk_size if t < num_tasks - 1 else DRAFT_CONFIG.vocab_size
                    var local_max_l: Float32 = -999999.0
                    var local_tok = 0
                    for r in range(start_r, end_r):
                        var r_offset = r * svd.rank
                        var acc = SIMD[DType.float32, 8](0.0)
                        for j in range(0, svd.rank, 8):
                            acc = acc + z_draft.load[8](j) * svd.A.load[8](r_offset + j)
                        var logit = acc.reduce_add()
                        logit = Float32(30.0) * fast_tanh(logit / Float32(30.0))
                        if logit > local_max_l:
                            local_max_l = logit
                            local_tok = r
                    local_max_logits[t] = local_max_l
                    local_next_tokens[t] = local_tok
                    
                parallelize[draft_proj_worker](num_tasks)
                z_draft.free()
                var d_max_logit: Float32 = -999999.0
                var d_next_token = 0
                for t in range(num_tasks):
                    if local_max_logits[t] > d_max_logit:
                        d_max_logit = local_max_logits[t]
                        d_next_token = local_next_tokens[t]
                draft_tokens.append(d_next_token)

            # B. Target Model Parallel Pass: stream target weights ONCE and validate K_actual tokens
            for s in range(K_actual + 1):
                var t_seq_offset = s * TARGET_CONFIG.embed_dim
                svd.reconstruct(draft_tokens[s], target_x_seq + t_seq_offset)
                for i in range(TARGET_CONFIG.embed_dim):
                    target_x_seq[t_seq_offset + i] *= scale

            # Load Layer 0 of target model before the loop
            var tag_0_val: UInt64 = UInt64(20000000 + pos * 10)
            _ = ring.submit_read(weights_fd, w_a_ptr.__int__(), TARGET_CONFIG.get_w2_offset_in_layer(0), 0, tag_0_val)
            wait_for_tag(ring, tag_0_val, out_tag_ptr)

            for layer_idx in range(TARGET_CONFIG.num_layers):
                var active_weights_ptr = w_a_ptr if layer_idx % 2 == 0 else w_b_ptr
                var next_weights_ptr = w_b_ptr if layer_idx % 2 == 0 else w_a_ptr
                
                # Prefetch next layer's weights
                if layer_idx + 1 < TARGET_CONFIG.num_layers:
                    var tag_next: UInt64 = UInt64(30000000 + pos * 100 + layer_idx + 1)
                    var offset_next = TARGET_CONFIG.get_layer_offset(layer_idx + 1)
                    var size_next = TARGET_CONFIG.get_w2_offset_in_layer(layer_idx + 1)
                    _ = ring.submit_read(weights_fd, next_weights_ptr.__int__(), size_next, offset_next, tag_next)
                    
                var weights = TARGET_CONFIG.get_layer_weights(layer_idx, active_weights_ptr)
                var is_global = TARGET_CONFIG.is_global_layer(layer_idx)
                var layer_heads = 16
                var layer_kv_heads = 1 if is_global else 8
                var layer_head_dim = 512 if is_global else 256
                var kv_stride = layer_kv_heads * layer_head_dim
                var ratio = layer_heads // layer_kv_heads
                
                var next_layer_completed = False
                
                # Execute sequentially across our K_actual+1 speculative batch steps for this layer
                for s in range(K_actual + 1):
                    var local_x = target_x_seq + s * TARGET_CONFIG.embed_dim
                    
                    rms_norm(local_x, weights.input_layernorm_weight, target_norm_x, TARGET_CONFIG.embed_dim)
                    
                    q8_matmul_vec(weights.wq_packed, weights.wq_scales, target_norm_x, target_q, TARGET_CONFIG.embed_dim, layer_heads * layer_head_dim, num_tasks)
                    q8_matmul_vec(weights.wk_packed, weights.wk_scales, target_norm_x, target_k_vec, TARGET_CONFIG.embed_dim, layer_kv_heads * layer_head_dim, num_tasks)
                    
                    if is_global:
                        for i in range(kv_stride):
                            target_v_vec[i] = target_k_vec[i]
                    else:
                        q8_matmul_vec(weights.wv_packed, weights.wv_scales, target_norm_x, target_v_vec, TARGET_CONFIG.embed_dim, layer_kv_heads * layer_head_dim, num_tasks)
                        
                    for h in range(layer_heads):
                        var head_ptr = target_q + h * layer_head_dim
                        rms_norm(head_ptr, weights.q_norm_weight, head_ptr, layer_head_dim)
                    for h in range(layer_kv_heads):
                        var head_ptr = target_k_vec + h * layer_head_dim
                        rms_norm(head_ptr, weights.k_norm_weight, head_ptr, layer_head_dim)
                    for h in range(layer_kv_heads):
                        var head_ptr = target_v_vec + h * layer_head_dim
                        rms_norm_noscale(head_ptr, head_ptr, layer_head_dim)
                        
                    apply_rope(target_q, layer_heads, layer_head_dim, pos + s, is_global)
                    apply_rope(target_k_vec, layer_kv_heads, layer_head_dim, pos + s, is_global)
                    
                    var layer_cache_offset = layer_cache_offsets[layer_idx]
                    var step_cache_offset = layer_cache_offset + (pos + s) * kv_stride
                    for i in range(kv_stride):
                        target_k_cache[step_cache_offset + i] = target_k_vec[i]
                        target_v_cache[step_cache_offset + i] = target_v_vec[i]
                        
                    for h in range(layer_heads):
                        var h_offset = h * layer_head_dim
                        var g = h // ratio
                        var g_offset = g * layer_head_dim
                        
                        for t in range(pos + s + 1):
                            var t_cache_offset = layer_cache_offset + t * kv_stride + g_offset
                            var score: Float32 = 0.0
                            for d in range(layer_head_dim):
                                score += target_q[h_offset + d] * target_k_cache[t_cache_offset + d]
                            target_attn_scores[t] = score
                            
                        softmax(target_attn_scores, pos + s + 1)
                        
                        for d in range(layer_head_dim):
                            var val_sum: Float32 = 0.0
                            for t in range(pos + s + 1):
                                var t_cache_offset = layer_cache_offset + t * kv_stride + g_offset
                                val_sum += target_attn_scores[t] * target_v_cache[t_cache_offset + d]
                            target_attn_out[h_offset + d] = val_sum
                            
                    q8_matmul_vec(weights.wo_packed, weights.wo_scales, target_attn_out, target_proj_attn_out, layer_heads * layer_head_dim, TARGET_CONFIG.embed_dim, num_tasks)
                    
                    rms_norm(target_proj_attn_out, weights.post_attention_layernorm_weight, target_proj_attn_out, TARGET_CONFIG.embed_dim)
                    for i in range(TARGET_CONFIG.embed_dim):
                        local_x[i] += target_proj_attn_out[i]
                        
                    rms_norm(local_x, weights.pre_feedforward_layernorm_weight, target_norm_x, TARGET_CONFIG.embed_dim)
                    q8_matmul_vec(weights.w1_packed, weights.w1_scales, target_norm_x, target_h1, TARGET_CONFIG.embed_dim, TARGET_CONFIG.intermediate_dim, num_tasks)
                    q8_matmul_vec(weights.w3_packed, weights.w3_scales, target_norm_x, target_h2, TARGET_CONFIG.embed_dim, TARGET_CONFIG.intermediate_dim, num_tasks)
                    gelu_and_mul(target_h1, target_h2, target_ffn_out_gate, TARGET_CONFIG.intermediate_dim)
                    
                    # W2 dynamic sparse direct I/O skip loading
                    var num_blocks = TARGET_CONFIG.intermediate_dim // 32
                    var layer_offset = TARGET_CONFIG.get_layer_offset(layer_idx)
                    var w2_offset_in_layer = TARGET_CONFIG.get_w2_offset_in_layer(layer_idx)
                    var tag_base_for_w2: UInt64 = 400000 + UInt64(layer_idx) * 1000 + UInt64(s) * 1000000 + UInt64(pos) * 100000000
                    
                    var max_active_block: Int = -1
                    for b in range(num_blocks - 1, -1, -1):
                        var block_offset_gate = b * 32
                        var block_active = False
                        for c in range(32):
                            if abs(target_ffn_out_gate[block_offset_gate + c]) > 1e-4:
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
                        
                    var completed_active = 0
                    var attempts = 0
                    var tag_next: UInt64 = UInt64(30000000 + pos * 100 + layer_idx + 1)
                    
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
                             raise Error("Validation weights paging I/O timed out waiting for sparse W2 blocks")
                             
                    q8_matmul_w2_block_column(weights.w2_packed, target_ffn_out_gate, target_ffn_out, TARGET_CONFIG.intermediate_dim, TARGET_CONFIG.embed_dim, num_tasks, Float32(1e-4))
                    
                    rms_norm(target_ffn_out, weights.post_feedforward_layernorm_weight, target_ffn_out, TARGET_CONFIG.embed_dim)
                    for i in range(TARGET_CONFIG.embed_dim):
                        local_x[i] += target_ffn_out[i]
                        
                    var scalar = weights.layer_scalar[0]
                    for i in range(TARGET_CONFIG.embed_dim):
                        local_x[i] *= scalar

                if layer_idx + 1 < TARGET_CONFIG.num_layers:
                    if not next_layer_completed:
                        var tag_next_val: UInt64 = UInt64(30000000 + pos * 100 + layer_idx + 1)
                        wait_for_tag(ring, tag_next_val, out_tag_ptr)

            # C. Vocabulary Projections for all target steps in batch
            var target_predictions = List[Int]()
            for s in range(K_actual + 1):
                var local_x = target_x_seq + s * TARGET_CONFIG.embed_dim
                rms_norm(local_x, final_norm_weight, target_final_norm_x, TARGET_CONFIG.embed_dim)
                
                var z_target = alloc[Float32](svd.rank)
                for j in range(svd.rank):
                    var acc = SIMD[DType.float32, 8](0.0)
                    for c in range(0, TARGET_CONFIG.embed_dim, 8):
                        acc = acc + target_final_norm_x.load[8](c) * svd.B.load[8](j * TARGET_CONFIG.embed_dim + c)
                    z_target[j] = acc.reduce_add()

                def target_proj_worker(t: Int) capturing -> None:
                    var start_r = t * chunk_size
                    var end_r = (t + 1) * chunk_size if t < num_tasks - 1 else TARGET_CONFIG.vocab_size
                    var local_max_l: Float32 = -999999.0
                    var local_tok = 0
                    for r in range(start_r, end_r):
                        var r_offset = r * svd.rank
                        var acc = SIMD[DType.float32, 8](0.0)
                        for j in range(0, svd.rank, 8):
                            acc = acc + z_target.load[8](j) * svd.A.load[8](r_offset + j)
                        var logit = acc.reduce_add()
                        logit = Float32(30.0) * fast_tanh(logit / Float32(30.0))
                        if logit > local_max_l:
                            local_max_l = logit
                            local_tok = r
                    local_max_logits[t] = local_max_l
                    local_next_tokens[t] = local_tok
                    
                parallelize[target_proj_worker](num_tasks)
                z_target.free()
                var t_max_logit: Float32 = -999999.0
                var t_next_token = 0
                for t in range(num_tasks):
                    if local_max_logits[t] > t_max_logit:
                        t_max_logit = local_max_logits[t]
                        t_next_token = local_next_tokens[t]
                target_predictions.append(t_next_token)

            # D. Speculative Verification Decision
            var num_accepted = 0
            for s in range(K_actual):
                print("[Debug Spec] s =", s, " Target:", target_predictions[s], " Draft Proposed:", draft_tokens[s + 1])
                if target_predictions[s] == draft_tokens[s + 1]:
                    num_accepted += 1
                else:
                    break
                    
            var next_starting_token = target_predictions[num_accepted]
            
            # Print all accepted tokens from draft
            for i in range(1, num_accepted + 1):
                var next_token_list = list_cls()
                _ = next_token_list.append(draft_tokens[i])
                var decoded = tokenizer.decode(next_token_list)
                py_builtins.print(decoded, end="", flush=True)
                
            # Print the corrected / continuation token from target
            var next_token_list = list_cls()
            _ = next_token_list.append(next_starting_token)
            var decoded = tokenizer.decode(next_token_list)
            py_builtins.print(decoded, end="", flush=True)
            
            # Record metrics
            total_accepted += num_accepted
            total_speculative_steps += 1

            # Dynamic Speculation Depth (K-Tuning)
            if num_accepted == K_actual:
                current_K = min(current_K + 1, K_MAX)
            elif num_accepted <= 1:
                current_K = max(current_K - 1, 1)

            # Commit steps & rollback position pointer
            if num_accepted == K_actual:
                # The draft model never ran with input draft_tokens[K_actual], so its key/value at pos + K_actual is missing.
                # Run a single forward step of the draft model on draft_tokens[K_actual] at position pos + K_actual
                # to populate the draft KV cache correctly.
                var curr = draft_tokens[K_actual]
                svd.reconstruct(curr, draft_x)
                for i in range(DRAFT_CONFIG.embed_dim):
                    draft_x[i] *= scale
                
                for layer_idx in range(DRAFT_CONFIG.num_layers):
                    var weights = DRAFT_CONFIG.get_layer_weights(layer_idx, draft_base_ptr + DRAFT_CONFIG.get_layer_offset(layer_idx))
                    var is_global = DRAFT_CONFIG.is_global_layer(layer_idx)
                    var layer_heads = 16
                    var layer_kv_heads = 1 if is_global else 8
                    var layer_head_dim = 512 if is_global else 256
                    var kv_stride = layer_kv_heads * layer_head_dim
                    var ratio = layer_heads // layer_kv_heads
                    
                    rms_norm(draft_x, weights.input_layernorm_weight, draft_norm_x, DRAFT_CONFIG.embed_dim)
                    
                    q4_matmul_vec(weights.wq_packed, weights.wq_scales, draft_norm_x, draft_q, DRAFT_CONFIG.embed_dim, layer_heads * layer_head_dim, num_tasks)
                    q4_matmul_vec(weights.wk_packed, weights.wk_scales, draft_norm_x, draft_k_vec, DRAFT_CONFIG.embed_dim, layer_kv_heads * layer_head_dim, num_tasks)
                    
                    if is_global:
                        for i in range(kv_stride):
                            draft_v_vec[i] = draft_k_vec[i]
                    else:
                        q4_matmul_vec(weights.wv_packed, weights.wv_scales, draft_norm_x, draft_v_vec, DRAFT_CONFIG.embed_dim, layer_kv_heads * layer_head_dim, num_tasks)
                        
                    for h in range(layer_heads):
                        var head_ptr = draft_q + h * layer_head_dim
                        rms_norm(head_ptr, weights.q_norm_weight, head_ptr, layer_head_dim)
                    for h in range(layer_kv_heads):
                        var head_ptr = draft_k_vec + h * layer_head_dim
                        rms_norm(head_ptr, weights.k_norm_weight, head_ptr, layer_head_dim)
                    for h in range(layer_kv_heads):
                        var head_ptr = draft_v_vec + h * layer_head_dim
                        rms_norm_noscale(head_ptr, head_ptr, layer_head_dim)
                        
                    apply_rope(draft_q, layer_heads, layer_head_dim, pos + K_actual, is_global)
                    apply_rope(draft_k_vec, layer_kv_heads, layer_head_dim, pos + K_actual, is_global)
                    
                    var layer_cache_offset = layer_cache_offsets[layer_idx]
                    var step_cache_offset = layer_cache_offset + (pos + K_actual) * kv_stride
                    for i in range(kv_stride):
                        draft_k_cache[step_cache_offset + i] = draft_k_vec[i]
                        draft_v_cache[step_cache_offset + i] = draft_v_vec[i]
                        
                    for h in range(layer_heads):
                        var h_offset = h * layer_head_dim
                        var g = h // ratio
                        var g_offset = g * layer_head_dim
                        for t in range(pos + K_actual + 1):
                            var t_cache_offset = layer_cache_offset + t * kv_stride + g_offset
                            var score: Float32 = 0.0
                            for d in range(layer_head_dim):
                                score += draft_q[h_offset + d] * draft_k_cache[t_cache_offset + d]
                            draft_attn_scores[t] = score
                            
                        softmax(draft_attn_scores, pos + K_actual + 1)
                        
                        for d in range(layer_head_dim):
                            var val_sum: Float32 = 0.0
                            for t in range(pos + K_actual + 1):
                                var t_cache_offset = layer_cache_offset + t * kv_stride + g_offset
                                val_sum += draft_attn_scores[t] * draft_v_cache[t_cache_offset + d]
                            draft_attn_out[h_offset + d] = val_sum
                            
                    q4_matmul_vec(weights.wo_packed, weights.wo_scales, draft_attn_out, draft_proj_attn_out, layer_heads * layer_head_dim, DRAFT_CONFIG.embed_dim, num_tasks)
                    
                    rms_norm(draft_proj_attn_out, weights.post_attention_layernorm_weight, draft_proj_attn_out, DRAFT_CONFIG.embed_dim)
                    for i in range(DRAFT_CONFIG.embed_dim):
                        draft_x[i] += draft_proj_attn_out[i]
                        
                    rms_norm(draft_x, weights.pre_feedforward_layernorm_weight, draft_norm_x, DRAFT_CONFIG.embed_dim)
                    q4_matmul_vec(weights.w1_packed, weights.w1_scales, draft_norm_x, draft_h1, DRAFT_CONFIG.embed_dim, DRAFT_CONFIG.intermediate_dim, num_tasks)
                    q4_matmul_vec(weights.w3_packed, weights.w3_scales, draft_norm_x, draft_h2, DRAFT_CONFIG.embed_dim, DRAFT_CONFIG.intermediate_dim, num_tasks)
                    gelu_and_mul(draft_h1, draft_h2, draft_ffn_out_gate, DRAFT_CONFIG.intermediate_dim)
                    
                    q4_matmul_vec(weights.w2_packed, weights.w2_scales, draft_ffn_out_gate, draft_ffn_out, DRAFT_CONFIG.intermediate_dim, DRAFT_CONFIG.embed_dim, num_tasks)
                    
                    rms_norm(draft_ffn_out, weights.post_feedforward_layernorm_weight, draft_ffn_out, DRAFT_CONFIG.embed_dim)
                    for i in range(DRAFT_CONFIG.embed_dim):
                        draft_x[i] += draft_ffn_out[i]
                        
                    var scalar = weights.layer_scalar[0]
                    for i in range(DRAFT_CONFIG.embed_dim):
                        draft_x[i] *= scalar

            pos = pos + num_accepted + 1
            tokens_generated += num_accepted + 1
            current_token = next_starting_token

        var t_gen_end = py.time()
        var gen_time = Float64(py=(t_gen_end - t_gen_start))
        print("\n[Mojo Bench] Speculative generation completed in", gen_time, "sec (" + String(Float64(tokens_generated) / gen_time) + " t/s)")
        if total_speculative_steps > 0:
            print("[Mojo Bench] Speculative acceptance rate: ", Float64(total_accepted) / Float64(total_speculative_steps * K_MAX) * 100.0, "% (Avg accepted: ", Float64(total_accepted) / Float64(total_speculative_steps), " tokens per step)")
