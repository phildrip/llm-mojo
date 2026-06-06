from std.ffi import OwnedDLHandle
from std.memory import alloc
from std.math import sqrt, exp, cos, sin
from std.python import Python
import std.os
from uring import IoUring
from std.algorithm import parallelize
from std.sys.info import num_physical_cores

# -----------------------------------------------------------------------------
# 1. Structural Layer Weights Representation
# -----------------------------------------------------------------------------
struct LayerWeights:
    var is_quantized: Bool

    # Packed (ternary) weights & scales
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

    # Unquantized FP32 weights
    var wq: UnsafePointer[Float32, MutExternalOrigin]
    var wk: UnsafePointer[Float32, MutExternalOrigin]
    var wv: UnsafePointer[Float32, MutExternalOrigin]
    var wo: UnsafePointer[Float32, MutExternalOrigin]
    var w1: UnsafePointer[Float32, MutExternalOrigin]
    var w2: UnsafePointer[Float32, MutExternalOrigin]
    var w3: UnsafePointer[Float32, MutExternalOrigin]

    # Shared full-precision variables (always unquantized)
    var input_layernorm_weight: UnsafePointer[Float32, MutExternalOrigin]
    var post_attention_layernorm_weight: UnsafePointer[Float32, MutExternalOrigin]
    var pre_feedforward_layernorm_weight: UnsafePointer[Float32, MutExternalOrigin]
    var post_feedforward_layernorm_weight: UnsafePointer[Float32, MutExternalOrigin]
    var q_norm_weight: UnsafePointer[Float32, MutExternalOrigin]
    var k_norm_weight: UnsafePointer[Float32, MutExternalOrigin]
    var layer_scalar: UnsafePointer[Float32, MutExternalOrigin]

    def __init__(
        out self,
        is_quantized: Bool,
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
        wq: UnsafePointer[Float32, MutExternalOrigin],
        wk: UnsafePointer[Float32, MutExternalOrigin],
        wv: UnsafePointer[Float32, MutExternalOrigin],
        wo: UnsafePointer[Float32, MutExternalOrigin],
        w1: UnsafePointer[Float32, MutExternalOrigin],
        w2: UnsafePointer[Float32, MutExternalOrigin],
        w3: UnsafePointer[Float32, MutExternalOrigin],
        input_layernorm: UnsafePointer[Float32, MutExternalOrigin],
        post_attention_layernorm: UnsafePointer[Float32, MutExternalOrigin],
        pre_feedforward_layernorm: UnsafePointer[Float32, MutExternalOrigin],
        post_feedforward_layernorm: UnsafePointer[Float32, MutExternalOrigin],
        q_norm: UnsafePointer[Float32, MutExternalOrigin],
        k_norm: UnsafePointer[Float32, MutExternalOrigin],
        layer_scalar: UnsafePointer[Float32, MutExternalOrigin],
    ):
        self.is_quantized = is_quantized
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
        self.wq = wq
        self.wk = wk
        self.wv = wv
        self.wo = wo
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3
        self.input_layernorm_weight = input_layernorm
        self.post_attention_layernorm_weight = post_attention_layernorm
        self.pre_feedforward_layernorm_weight = pre_feedforward_layernorm
        self.post_feedforward_layernorm_weight = post_feedforward_layernorm
        self.q_norm_weight = q_norm
        self.k_norm_weight = k_norm
        self.layer_scalar = layer_scalar

# -----------------------------------------------------------------------------
# 2. Model Configuration & Memory Offsets
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
    def is_layer_quantized(self, layer_idx: Int) -> Bool:
        # Keep layers 0, 1, 46, and 47 unquantized
        return layer_idx > 1 and layer_idx < 46

    @always_inline
    def get_unquantized_layer_aligned_size(self, layer_idx: Int) -> Int:
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
        var input_layernorm_size = Self.embed_dim
        var post_attention_layernorm_size = Self.embed_dim
        var pre_feedforward_layernorm_size = Self.embed_dim
        var post_feedforward_layernorm_size = Self.embed_dim
        var q_norm_size = layer_head_dim
        var k_norm_size = layer_head_dim
        var layer_scalar_size = 1
        
        var total_floats = (
            wq_size + wk_size + wv_size + wo_size +
            w1_size + w2_size + w3_size +
            input_layernorm_size + post_attention_layernorm_size +
            pre_feedforward_layernorm_size + post_feedforward_layernorm_size +
            q_norm_size + k_norm_size + layer_scalar_size
        )
        var raw_bytes_size = total_floats * 4
        return ((raw_bytes_size + 4095) // 4096) * 4096

    @always_inline
    def get_quantized_layer_aligned_size(self, layer_idx: Int) -> Int:
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
        
        # Packed ternary weights: 4 elements per byte
        var packed_weights_size = (wq_size + wk_size + wv_size + wo_size + w1_size + w2_size + w3_size) // 4
        
        # Scale factors (1 per row of each dense matrix)
        var c_out_q = layer_heads * layer_head_dim
        var c_out_k = layer_kv_heads * layer_head_dim
        var c_out_v = 0 if is_global else (layer_kv_heads * layer_head_dim)
        var c_out_o = Self.embed_dim
        var c_out_w1 = Self.intermediate_dim
        var c_out_w2 = Self.embed_dim
        var c_out_w3 = Self.intermediate_dim
        
        var scales_count = c_out_q + c_out_k + c_out_v + c_out_o + c_out_w1 + c_out_w2 + c_out_w3
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

    @always_inline
    def get_layer_aligned_size(self, layer_idx: Int) -> Int:
        if self.is_layer_quantized(layer_idx):
            return self.get_quantized_layer_aligned_size(layer_idx)
        else:
            return self.get_unquantized_layer_aligned_size(layer_idx)

    @always_inline
    def get_layer_offset(self, layer_idx: Int) -> Int:
        var offset = 0
        for i in range(layer_idx):
            offset += self.get_layer_aligned_size(i)
        return offset

    @always_inline
    def get_layer_weights(self, layer_idx: Int, base_ptr: UnsafePointer[UInt8, _]) -> LayerWeights:
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
        
        # Sizing / Count of rows for scales (c_out)
        var c_out_q = layer_heads * layer_head_dim
        var c_out_k = layer_kv_heads * layer_head_dim
        var c_out_v = 0 if is_global else (layer_kv_heads * layer_head_dim)
        var c_out_o = Self.embed_dim
        var c_out_w1 = Self.intermediate_dim
        var c_out_w2 = Self.embed_dim
        var c_out_w3 = Self.intermediate_dim

        var is_quantized = self.is_layer_quantized(layer_idx)

        # Allocate null pointers
        var wq_packed = UnsafePointer[UInt8, MutExternalOrigin]()
        var wk_packed = UnsafePointer[UInt8, MutExternalOrigin]()
        var wv_packed = UnsafePointer[UInt8, MutExternalOrigin]()
        var wo_packed = UnsafePointer[UInt8, MutExternalOrigin]()
        var w1_packed = UnsafePointer[UInt8, MutExternalOrigin]()
        var w2_packed = UnsafePointer[UInt8, MutExternalOrigin]()
        var w3_packed = UnsafePointer[UInt8, MutExternalOrigin]()

        var wq_scales = UnsafePointer[Float32, MutExternalOrigin]()
        var wk_scales = UnsafePointer[Float32, MutExternalOrigin]()
        var wv_scales = UnsafePointer[Float32, MutExternalOrigin]()
        var wo_scales = UnsafePointer[Float32, MutExternalOrigin]()
        var w1_scales = UnsafePointer[Float32, MutExternalOrigin]()
        var w2_scales = UnsafePointer[Float32, MutExternalOrigin]()
        var w3_scales = UnsafePointer[Float32, MutExternalOrigin]()

        var wq = UnsafePointer[Float32, MutExternalOrigin]()
        var wk = UnsafePointer[Float32, MutExternalOrigin]()
        var wv = UnsafePointer[Float32, MutExternalOrigin]()
        var wo = UnsafePointer[Float32, MutExternalOrigin]()
        var w1 = UnsafePointer[Float32, MutExternalOrigin]()
        var w2 = UnsafePointer[Float32, MutExternalOrigin]()
        var w3 = UnsafePointer[Float32, MutExternalOrigin]()

        var input_layernorm = UnsafePointer[Float32, MutExternalOrigin]()
        var post_attention_layernorm = UnsafePointer[Float32, MutExternalOrigin]()
        var pre_feedforward_layernorm = UnsafePointer[Float32, MutExternalOrigin]()
        var post_feedforward_layernorm = UnsafePointer[Float32, MutExternalOrigin]()
        var q_norm = UnsafePointer[Float32, MutExternalOrigin]()
        var k_norm = UnsafePointer[Float32, MutExternalOrigin]()
        var layer_scalar = UnsafePointer[Float32, MutExternalOrigin]()

        if is_quantized:
            var wq_p = base_ptr
            var wk_p = wq_p + (wq_size // 4)
            var wv_p = wk_p + (wk_size // 4)
            var wo_p = wk_p + (wk_size // 4)
            if not is_global:
                wv_p = wk_p + (wk_size // 4)
                wo_p = wv_p + (wv_size // 4)
                
            var w1_p = wo_p + (wo_size // 4)
            var w2_p = w1_p + (w1_size // 4)
            var w3_p = w2_p + (w2_size // 4)
            
            var scales_base = (w3_p + (w3_size // 4)).bitcast[Float32]()
            
            var wq_s = scales_base
            var wk_s = wq_s + c_out_q
            var wv_s = wk_s + c_out_k
            var wo_s = wk_s + c_out_k
            if not is_global:
                wv_s = wk_s + c_out_k
                wo_s = wv_s + c_out_v
                
            var w1_s = wo_s + c_out_o
            var w2_s = w1_s + c_out_w1
            var w3_s = w2_s + c_out_w2
            
            var unchanged_base = w3_s + c_out_w3
            
            var iln = unchanged_base
            var paln = iln + Self.embed_dim
            var pfln = paln + Self.embed_dim
            var pofln = pfln + Self.embed_dim
            var qn = pofln + Self.embed_dim
            var kn = qn + layer_head_dim
            var ls = kn + layer_head_dim

            wq_packed = UnsafePointer[UInt8, MutExternalOrigin](unsafe_from_address=wq_p.__int__())
            wk_packed = UnsafePointer[UInt8, MutExternalOrigin](unsafe_from_address=wk_p.__int__())
            wv_packed = UnsafePointer[UInt8, MutExternalOrigin](unsafe_from_address=wv_p.__int__())
            wo_packed = UnsafePointer[UInt8, MutExternalOrigin](unsafe_from_address=wo_p.__int__())
            w1_packed = UnsafePointer[UInt8, MutExternalOrigin](unsafe_from_address=w1_p.__int__())
            w2_packed = UnsafePointer[UInt8, MutExternalOrigin](unsafe_from_address=w2_p.__int__())
            w3_packed = UnsafePointer[UInt8, MutExternalOrigin](unsafe_from_address=w3_p.__int__())

            wq_scales = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=wq_s.__int__())
            wk_scales = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=wk_s.__int__())
            wv_scales = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=wv_s.__int__())
            wo_scales = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=wo_s.__int__())
            w1_scales = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=w1_s.__int__())
            w2_scales = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=w2_s.__int__())
            w3_scales = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=w3_s.__int__())

            input_layernorm = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=iln.__int__())
            post_attention_layernorm = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=paln.__int__())
            pre_feedforward_layernorm = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=pfln.__int__())
            post_feedforward_layernorm = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=pofln.__int__())
            q_norm = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=qn.__int__())
            k_norm = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=kn.__int__())
            layer_scalar = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=ls.__int__())
        else:
            var base_f32 = base_ptr.bitcast[Float32]()
            var wq_f = base_f32
            var wk_f = wq_f + wq_size
            
            var wv_f = wk_f
            var wo_f = wk_f + wk_size
            if not is_global:
                wv_f = wk_f + wk_size
                wo_f = wv_f + wv_size
                
            var w1_f = wo_f + wo_size
            var w2_f = w1_f + w1_size
            var w3_f = w2_f + w2_size
            var iln = w3_f + w3_size
            var paln = iln + Self.embed_dim
            var pfln = paln + Self.embed_dim
            var pofln = pfln + Self.embed_dim
            var qn = pofln + Self.embed_dim
            var kn = qn + layer_head_dim
            var ls = kn + layer_head_dim

            wq = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=wq_f.__int__())
            wk = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=wk_f.__int__())
            wv = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=wv_f.__int__())
            wo = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=wo_f.__int__())
            w1 = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=w1_f.__int__())
            w2 = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=w2_f.__int__())
            w3 = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=w3_f.__int__())

            input_layernorm = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=iln.__int__())
            post_attention_layernorm = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=paln.__int__())
            pre_feedforward_layernorm = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=pfln.__int__())
            post_feedforward_layernorm = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=pofln.__int__())
            q_norm = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=qn.__int__())
            k_norm = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=kn.__int__())
            layer_scalar = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=ls.__int__())

        return LayerWeights(
            is_quantized,
            wq_packed, wk_packed, wv_packed, wo_packed, w1_packed, w2_packed, w3_packed,
            wq_scales, wk_scales, wv_scales, wo_scales, w1_scales, w2_scales, w3_scales,
            wq, wk, wv, wo, w1, w2, w3,
            input_layernorm, post_attention_layernorm, pre_feedforward_layernorm, post_feedforward_layernorm,
            q_norm, k_norm, layer_scalar
        )

    def __init__(out self):
        pass

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
def matmul_vec(W: UnsafePointer[Float32, _], x: UnsafePointer[Float32, _], out_ptr: UnsafePointer[Float32, MutExternalOrigin], c_in: Int, c_out: Int, num_tasks: Int):
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
    var b = SIMD[DType.uint8, 8](b0, b0, b0, b0, b1, b1, b1, b1)
    var shifts = SIMD[DType.uint8, 8](0, 2, 4, 6, 0, 2, 4, 6)
    var shifted = b >> shifts
    var masked = shifted & 3
    return masked.cast[DType.float32]() - 1.0

@always_inline
def ternary_matmul_vec(
    W_packed: UnsafePointer[UInt8, _],
    scales: UnsafePointer[Float32, _],
    x: UnsafePointer[Float32, _],
    out_ptr: UnsafePointer[Float32, MutExternalOrigin],
    c_in: Int,
    c_out: Int,
    num_tasks: Int,
):
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
def fast_tanh(x: Float32) -> Float32:
    if x > 15.0:
        return 1.0
    elif x < -15.0:
        return -1.0
    var e2x = exp(2.0 * x)
    return (e2x - 1.0) / (e2x + 1.0)

@always_inline
def gelu_and_mul(h1: UnsafePointer[Float32, _], h2: UnsafePointer[Float32, _], out_ptr: UnsafePointer[Float32, MutExternalOrigin], size: Int):
    # Vectorized GeLU: x * 0.5 * (1.0 + erf(x / sqrt(2.0))) approximated using fast tanh
    for i in range(0, size, 8):
        var v1 = h1.load[8](i)
        var v2 = h2.load[8](i)
        
        # GeLU approximation: 0.5 * x * (1.0 + tanh(sqrt(2.0 / pi) * (x + 0.044715 * x^3)))
        var acc = SIMD[DType.float32, 8](0.0)
        for lane in range(8):
            var val = v1[lane]
            var fit = 0.79788456 * (val + 0.044715 * val * val * val)
            var tanh_fit = fast_tanh(fit)
            acc[lane] = 0.5 * val * (1.0 + tanh_fit)
            
        out_ptr.store[8](i, acc * v2)

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
        var e = exp(scores[i] - max_val)
        scores[i] = e
        sum_exp += e
        
    var inv_sum = 1.0 / sum_exp
    for i in range(size):
        scores[i] = scores[i] * inv_sum

def wait_for_tag(ring: IoUring, expected_tag: UInt64, out_tag_ptr: UnsafePointer[UInt64, MutExternalOrigin]) raises:
    var attempts = 0
    while True:
        var res = ring.poll_completion(out_tag_ptr.__int__())
        if res >= 0:
            if out_tag_ptr[0] == expected_tag:
                return
        attempts += 1
        if attempts > 5000000000:
            raise Error("Weights paging I/O timed out waiting for layer: " + String(expected_tag))

def print_stats(name: String, ptr: UnsafePointer[Float32, _], size: Int):
    var sum: Float32 = 0.0
    var sum_sq: Float32 = 0.0
    for i in range(size):
        sum += ptr[i]
        sum_sq += ptr[i] * ptr[i]
    var mean = sum / Float32(size)
    var rms = sqrt(sum_sq / Float32(size))
    print(name + " -> size=" + String(size) + " | mean=" + String(mean) + " | rms=" + String(rms))

# -----------------------------------------------------------------------------
# 4. Main Program
# -----------------------------------------------------------------------------
def main() raises:
    var num_tasks = num_physical_cores()
    print("=========================================================")
    print("   GEMMA 4 12B MIXED-PRECISION TERNARY PIPELINE RUNNER   ")
    print("=========================================================")
    print("Embedding Size:      3840")
    print("Attention Heads:     16 Query Heads / 8 KV Heads (GQA/MQA)")
    print("Model Parameters:    12B Scale Causal Layout")
    print("Mixed Precision:     Layer 0 & 47 in FP32, Layers 1-46 in Ternary")
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
    sys.path.append("/home/phil/git/llm-mojo/.venv/lib/python3.12/site-packages")
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
    var w_a_ptr = UnsafePointer[UInt8, MutExternalOrigin](unsafe_from_address=w_a_addr)

    var memptr_w_b = alloc[Int](1)
    _ = posix_memalign(memptr_w_b.__int__(), 4096, max_layer_bytes)
    var w_b_addr = memptr_w_b[0]
    var w_b_ptr = UnsafePointer[UInt8, MutExternalOrigin](unsafe_from_address=w_b_addr)

    # 3. Open converted real weights file on NVMe with O_DIRECT
    var weights_file = "gemma4_weights_mixed.bin"
    print("Opening converted mixed weights file 'gemma4_weights_mixed.bin' with O_DIRECT read-only...")
    
    var path_bytes = weights_file.as_bytes()
    var path_ptr = alloc[UInt8](len(path_bytes) + 1)
    for i in range(len(path_bytes)):
        path_ptr[i] = path_bytes[i]
    path_ptr[len(path_bytes)] = 0

    # Flags: O_RDONLY (0) | O_DIRECT (16384) = 16384
    var weights_fd = c_open(path_ptr.__int__(), 16384, 0)
    if weights_fd < 0:
        print("CRITICAL: Failed to open 'gemma4_weights_mixed.bin' weights file with O_DIRECT. Make sure you converted weights first!")
        return

    print("Weights file opened successfully. Active size: " + String(CONFIG.get_layer_offset(CONFIG.num_layers)) + " bytes")

    # 3b. Open and Load Real Embedding weights (tied to LM Head)
    var embed_file = "gemma4_embed.bin"
    print("Opening embedding weights file 'gemma4_embed.bin'...")
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

    var k_cache = alloc[Float32](current_cache_offset)
    var v_cache = alloc[Float32](current_cache_offset)
    print("Initialized Key/Value caches of size " + String(current_cache_offset * 4 * 2) + " bytes")
    print("---------------------------------------------------------")

    # 5. Interactive Chat loop
    var list_cls = Python.import_module("builtins").list
    var py_input = Python.import_module("builtins").input

    # Scale factor for embeddings: sqrt(embed_dim) = sqrt(3840) ~ 61.9677
    var scale = Float32(61.96773)

    while True:
        var prompt_str = String(py_input("\nUser > ")).strip()
        if prompt_str == "exit" or prompt_str == "quit":
            break
            
        if len(prompt_str) == 0:
            continue
            
        var formatted_prompt = "<bos><|turn>user\n" + prompt_str + "<turn|>\n<|turn>model\n"
        print("[Tokenizer] Encoded input into ", end="")
        var token_ids = tokenizer.encode(formatted_prompt, add_special_tokens=False)
        var num_prompt_tokens = len(token_ids)
        print(num_prompt_tokens, "real sub-word token IDs:")
        print(token_ids)
        
        # Format Gemma prompt template structure
        print("Agent > ", end="")
        sys.stdout.flush()
        
        var t_prefill_start = py.time()
        
        # 1. Prefill Forward Pass
        var max_steps = CONFIG.max_seq_len
        var max_steps_arg = max_steps
        var max_steps_detected = max_steps_arg
        
        # Allocate flat activation array to hold hidden states for all prompt tokens
        var prompt_x = alloc[Float32](num_prompt_tokens * CONFIG.embed_dim)
        
        # Fill first token activations from embedded weights
        for idx in range(num_prompt_tokens):
            var token_id = Int(py=token_ids[idx])
            var token_emb_offset = token_id * CONFIG.embed_dim
            var px_offset = idx * CONFIG.embed_dim
            for i in range(CONFIG.embed_dim):
                prompt_x[px_offset + i] = embed_ptr[token_emb_offset + i] * scale
                
        var current_token = Int(py=token_ids[num_prompt_tokens - 1])
        
        # Layer 0 weight prefetch
        var tag_0: UInt64 = UInt64(7000)
        _ = ring.submit_read(weights_fd, w_a_ptr.__int__(), CONFIG.get_layer_aligned_size(0), 0, tag_0)
        wait_for_tag(ring, tag_0, out_tag_ptr)

        for layer_idx in range(CONFIG.num_layers):
            var active_weights_ptr = w_a_ptr if layer_idx % 2 == 0 else w_b_ptr
            var next_weights_ptr = w_b_ptr if layer_idx % 2 == 0 else w_a_ptr
            
            # Prefetch next layer's weights
            if layer_idx + 1 < CONFIG.num_layers:
                var tag_next: UInt64 = UInt64(8000 + layer_idx + 1)
                var offset_next = CONFIG.get_layer_offset(layer_idx + 1)
                var size_next = CONFIG.get_layer_aligned_size(layer_idx + 1)
                _ = ring.submit_read(weights_fd, next_weights_ptr.__int__(), size_next, offset_next, tag_next)
                
            var weights = CONFIG.get_layer_weights(layer_idx, active_weights_ptr)
            var is_global = CONFIG.is_global_layer(layer_idx)
            var layer_heads = 16
            var layer_kv_heads = 1 if is_global else 8
            var layer_head_dim = 512 if is_global else 256
            var kv_stride = layer_kv_heads * layer_head_dim
            var ratio = layer_heads // layer_kv_heads
            
            # Loop over all prompt positions
            for pos in range(num_prompt_tokens):
                var px_offset = pos * CONFIG.embed_dim
                var local_x = prompt_x + px_offset
                
                # Setup norm_x
                rms_norm(local_x, weights.input_layernorm_weight, norm_x, CONFIG.embed_dim)
                
                # Compute q, k, v
                if weights.is_quantized:
                    ternary_matmul_vec(weights.wq_packed, weights.wq_scales, norm_x, q, CONFIG.embed_dim, layer_heads * layer_head_dim, num_tasks)
                    ternary_matmul_vec(weights.wk_packed, weights.wk_scales, norm_x, k_vec, CONFIG.embed_dim, layer_kv_heads * layer_head_dim, num_tasks)
                    if is_global:
                        for i in range(kv_stride):
                            v_vec[i] = k_vec[i]
                    else:
                        ternary_matmul_vec(weights.wv_packed, weights.wv_scales, norm_x, v_vec, CONFIG.embed_dim, layer_kv_heads * layer_head_dim, num_tasks)
                else:
                    matmul_vec(weights.wq, norm_x, q, CONFIG.embed_dim, layer_heads * layer_head_dim, num_tasks)
                    matmul_vec(weights.wk, norm_x, k_vec, CONFIG.embed_dim, layer_kv_heads * layer_head_dim, num_tasks)
                    if is_global:
                        for i in range(kv_stride):
                            v_vec[i] = k_vec[i]
                    else:
                        matmul_vec(weights.wv, norm_x, v_vec, CONFIG.embed_dim, layer_kv_heads * layer_head_dim, num_tasks)

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
                        
                if weights.is_quantized:
                    ternary_matmul_vec(weights.wo_packed, weights.wo_scales, attn_out, proj_attn_out, layer_heads * layer_head_dim, CONFIG.embed_dim, num_tasks)
                else:
                    matmul_vec(weights.wo, attn_out, proj_attn_out, layer_heads * layer_head_dim, CONFIG.embed_dim, num_tasks)

                # Post-attention normalization on projected attention output
                rms_norm(proj_attn_out, weights.post_attention_layernorm_weight, proj_attn_out, CONFIG.embed_dim)
                for i in range(CONFIG.embed_dim):
                    local_x[i] += proj_attn_out[i]
                    
                # Pre-feedforward normalization
                rms_norm(local_x, weights.pre_feedforward_layernorm_weight, norm_x, CONFIG.embed_dim)
                if weights.is_quantized:
                    ternary_matmul_vec(weights.w1_packed, weights.w1_scales, norm_x, h1, CONFIG.embed_dim, CONFIG.intermediate_dim, num_tasks)
                    ternary_matmul_vec(weights.w3_packed, weights.w3_scales, norm_x, h2, CONFIG.embed_dim, CONFIG.intermediate_dim, num_tasks)
                    gelu_and_mul(h1, h2, ffn_out_gate, CONFIG.intermediate_dim)
                    ternary_matmul_vec(weights.w2_packed, weights.w2_scales, ffn_out_gate, ffn_out, CONFIG.intermediate_dim, CONFIG.embed_dim, num_tasks)
                else:
                    matmul_vec(weights.w1, norm_x, h1, CONFIG.embed_dim, CONFIG.intermediate_dim, num_tasks)
                    matmul_vec(weights.w3, norm_x, h2, CONFIG.embed_dim, CONFIG.intermediate_dim, num_tasks)
                    gelu_and_mul(h1, h2, ffn_out_gate, CONFIG.intermediate_dim)
                    matmul_vec(weights.w2, ffn_out_gate, ffn_out, CONFIG.intermediate_dim, CONFIG.embed_dim, num_tasks)

                # Post-feedforward normalization on MLP output
                rms_norm(ffn_out, weights.post_feedforward_layernorm_weight, ffn_out, CONFIG.embed_dim)
                for i in range(CONFIG.embed_dim):
                    local_x[i] += ffn_out[i]
                    
                # Scale hidden state by layer_scalar
                var scalar = weights.layer_scalar[0]
                for i in range(CONFIG.embed_dim):
                    local_x[i] *= scalar

            if layer_idx + 1 < CONFIG.num_layers:
                var tag_next: UInt64 = UInt64(8000 + layer_idx + 1)
                wait_for_tag(ring, tag_next, out_tag_ptr)

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
        print(decoded, end="")
        sys.stdout.flush()
        
        current_token = next_token
        
        # 3. Subsequent generation steps
        var tokens_generated = 1
        var t_gen_start = py.time()
        for pos in range(num_prompt_tokens, max_steps):
            # Layer 0 weight prefetch
            var tag_0: UInt64 = UInt64(2000)
            _ = ring.submit_read(weights_fd, w_a_ptr.__int__(), CONFIG.get_layer_aligned_size(0), 0, tag_0)
            wait_for_tag(ring, tag_0, out_tag_ptr)
            
            # Setup input activations x with the dense embedding of current_token (scaled)
            var token_emb_offset = current_token * CONFIG.embed_dim
            for i in range(CONFIG.embed_dim):
                x[i] = embed_ptr[token_emb_offset + i] * scale
                
            # Run the 48 layers of Transformer forward pass
            for layer_idx in range(CONFIG.num_layers):
                var active_weights_ptr = w_a_ptr if layer_idx % 2 == 0 else w_b_ptr
                var next_weights_ptr = w_b_ptr if layer_idx % 2 == 0 else w_a_ptr
                
                if layer_idx + 1 < CONFIG.num_layers:
                    var tag_next: UInt64 = UInt64(3000 + pos * 10 + layer_idx + 1)
                    var offset_next = CONFIG.get_layer_offset(layer_idx + 1)
                    var size_next = CONFIG.get_layer_aligned_size(layer_idx + 1)
                    _ = ring.submit_read(weights_fd, next_weights_ptr.__int__(), size_next, offset_next, tag_next)
                    
                var weights = CONFIG.get_layer_weights(layer_idx, active_weights_ptr)
                var is_global = CONFIG.is_global_layer(layer_idx)
                rms_norm(x, weights.input_layernorm_weight, norm_x, CONFIG.embed_dim)
                
                var layer_heads = 16
                var layer_kv_heads = 1 if is_global else 8
                var layer_head_dim = 512 if is_global else 256
                var kv_stride = layer_kv_heads * layer_head_dim
                
                if weights.is_quantized:
                    ternary_matmul_vec(weights.wq_packed, weights.wq_scales, norm_x, q, CONFIG.embed_dim, layer_heads * layer_head_dim, num_tasks)
                    ternary_matmul_vec(weights.wk_packed, weights.wk_scales, norm_x, k_vec, CONFIG.embed_dim, layer_kv_heads * layer_head_dim, num_tasks)
                    if is_global:
                        for i in range(kv_stride):
                            v_vec[i] = k_vec[i]
                    else:
                        ternary_matmul_vec(weights.wv_packed, weights.wv_scales, norm_x, v_vec, CONFIG.embed_dim, layer_kv_heads * layer_head_dim, num_tasks)
                else:
                    matmul_vec(weights.wq, norm_x, q, CONFIG.embed_dim, layer_heads * layer_head_dim, num_tasks)
                    matmul_vec(weights.wk, norm_x, k_vec, CONFIG.embed_dim, layer_kv_heads * layer_head_dim, num_tasks)
                    if is_global:
                        for i in range(kv_stride):
                            v_vec[i] = k_vec[i]
                    else:
                        matmul_vec(weights.wv, norm_x, v_vec, CONFIG.embed_dim, layer_kv_heads * layer_head_dim, num_tasks)

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
                        
                if weights.is_quantized:
                    ternary_matmul_vec(weights.wo_packed, weights.wo_scales, attn_out, proj_attn_out, layer_heads * layer_head_dim, CONFIG.embed_dim, num_tasks)
                else:
                    matmul_vec(weights.wo, attn_out, proj_attn_out, layer_heads * layer_head_dim, CONFIG.embed_dim, num_tasks)
                
                # Post-attention normalization on projected attention output
                rms_norm(proj_attn_out, weights.post_attention_layernorm_weight, proj_attn_out, CONFIG.embed_dim)
                for i in range(CONFIG.embed_dim):
                    x[i] += proj_attn_out[i]
                    
                # Pre-feedforward normalization
                rms_norm(x, weights.pre_feedforward_layernorm_weight, norm_x, CONFIG.embed_dim)
                if weights.is_quantized:
                    ternary_matmul_vec(weights.w1_packed, weights.w1_scales, norm_x, h1, CONFIG.embed_dim, CONFIG.intermediate_dim, num_tasks)
                    ternary_matmul_vec(weights.w3_packed, weights.w3_scales, norm_x, h2, CONFIG.embed_dim, CONFIG.intermediate_dim, num_tasks)
                    gelu_and_mul(h1, h2, ffn_out_gate, CONFIG.intermediate_dim)
                    ternary_matmul_vec(weights.w2_packed, weights.w2_scales, ffn_out_gate, ffn_out, CONFIG.intermediate_dim, CONFIG.embed_dim, num_tasks)
                else:
                    matmul_vec(weights.w1, norm_x, h1, CONFIG.embed_dim, CONFIG.intermediate_dim, num_tasks)
                    matmul_vec(weights.w3, norm_x, h2, CONFIG.embed_dim, CONFIG.intermediate_dim, num_tasks)
                    gelu_and_mul(h1, h2, ffn_out_gate, CONFIG.intermediate_dim)
                    matmul_vec(weights.w2, ffn_out_gate, ffn_out, CONFIG.intermediate_dim, CONFIG.embed_dim, num_tasks)
                
                # Post-feedforward normalization on MLP output
                rms_norm(ffn_out, weights.post_feedforward_layernorm_weight, ffn_out, CONFIG.embed_dim)
                for i in range(CONFIG.embed_dim):
                    x[i] += ffn_out[i]
                    
                # Scale hidden state by layer_scalar
                var scalar = weights.layer_scalar[0]
                for i in range(CONFIG.embed_dim):
                    x[i] *= scalar
                    
                if layer_idx + 1 < CONFIG.num_layers:
                    var tag_next: UInt64 = UInt64(3000 + pos * 10 + layer_idx + 1)
                    wait_for_tag(ring, tag_next, out_tag_ptr)
                    
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
            print(word, end="")
            sys.stdout.flush()
            
            if next_token == 1 or next_token == 106: # <eos> or <turn|>
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
    k_cache.free()
    v_cache.free()
