from std.ffi import OwnedDLHandle
from std.memory import alloc
from std.collections import InlineArray
from std.math import sqrt, exp, cos, sin, tanh, log
from std.python import Python, PythonObject
import std.os
from std.algorithm import parallelize
from std.sys.info import num_physical_cores
from q4_embed import Q4Embedding
from kfd_gpu import KFDContext, GPUMemoryBuffer, HSACOKernel
from diffusion_kernels import (
    sc_forward_batch,
    sample_and_soft_embed_top32_batch,
    q4_gemm_l1_2x4,
    q4_fused_gate_up_geglu_l1,
    run_diffusion_moe_experts_parallel,
    build_active_canvas_map,
    update_locked_mask,
    gather_active,
    scatter_active_sampler_outputs,
    teacache_eval_gate,
    taylorseer_record_full_step,
    taylorseer_forecast_skipped_step,
    compute_moe_lod_k,
    compute_moe_lod_top_p,
    apply_moe_lod_and_set_cover,
    prep_qkv_norm_rope_kvq_batch,
    flash_dllm_bidirectional_attn,
    adacache_filter_after_layer1,
)
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
    vocab_size: Int = 32768,
    max_seq_len: Int = 512,
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
def rms_norm[
    in_type: DType = DType.float32,
    out_type: DType = DType.float32
](
    x: UnsafePointer[Scalar[in_type], _],
    w: UnsafePointer[Float32, _],
    out_ptr: UnsafePointer[Scalar[out_type], MutExternalOrigin],
    size: Int,
):
    var acc = SIMD[DType.float32, 8](0.0)
    for i in range(0, size, 8):
        var v = x.load[8](i).cast[DType.float32]()
        acc = acc + v * v
    var sum_sq = acc.reduce_add()
    var inv_rms = 1.0 / sqrt(sum_sq / Float32(size) + 1e-6)
    var f16_min = SIMD[DType.float32, 8](-65500.0)
    var f16_max = SIMD[DType.float32, 8](65500.0)
    for i in range(0, size, 8):
        var v = x.load[8](i).cast[DType.float32]()
        var weight = w.load[8](i)
        var res = v * inv_rms * weight
        comptime if out_type == DType.float16:
            res = res.clamp(f16_min, f16_max)
        out_ptr.store[8](i, res.cast[out_type]())

@always_inline
def rms_norm_noscale[
    in_type: DType = DType.float32,
    out_type: DType = DType.float32
](
    x: UnsafePointer[Scalar[in_type], _],
    out_ptr: UnsafePointer[Scalar[out_type], MutExternalOrigin],
    size: Int,
):
    var acc = SIMD[DType.float32, 8](0.0)
    for i in range(0, size, 8):
        var v = x.load[8](i).cast[DType.float32]()
        acc = acc + v * v
    var sum_sq = acc.reduce_add()
    var inv_rms = 1.0 / sqrt(sum_sq / Float32(size) + 1e-6)
    var f16_min = SIMD[DType.float32, 8](-65500.0)
    var f16_max = SIMD[DType.float32, 8](65500.0)
    for i in range(0, size, 8):
        var v = x.load[8](i).cast[DType.float32]()
        var res = v * inv_rms
        comptime if out_type == DType.float16:
            res = res.clamp(f16_min, f16_max)
        out_ptr.store[8](i, res.cast[out_type]())

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
def unpack_q4_to_q8(
    W_packed: UnsafePointer[UInt8, _],
    W_q8: UnsafePointer[Scalar[DType.int8], MutUntrackedOrigin],
    size: Int
):
    var steps = size // 32
    for i in range(steps):
        var packed_idx = i * 16
        var unpacked_idx = i * 32
        var bytes = W_packed.load[16](packed_idx)
        
        var repeated_01 = bytes.shuffle[0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7]()
        var shifted_01 = repeated_01 >> SIMD[DType.uint8, 16](0, 4, 0, 4, 0, 4, 0, 4, 0, 4, 0, 4, 0, 4, 0, 4)
        var masked_01 = (shifted_01 & 15).cast[DType.int8]() - 8
        
        var repeated_23 = bytes.shuffle[8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13, 14, 14, 15, 15]()
        var shifted_23 = repeated_23 >> SIMD[DType.uint8, 16](0, 4, 0, 4, 0, 4, 0, 4, 0, 4, 0, 4, 0, 4, 0, 4)
        var masked_23 = (shifted_23 & 15).cast[DType.int8]() - 8
        
        W_q8.store(unpacked_idx, masked_01)
        W_q8.store(unpacked_idx + 16, masked_23)
        
    var rem = size % 32
    if rem > 0:
        var packed_start = steps * 16
        var unpacked_start = steps * 32
        for j in range(rem // 2):
            var b = W_packed[packed_start + j]
            var w_low = (b & 15).cast[DType.int8]() - 8
            var w_high = ((b >> 4) & 15).cast[DType.int8]() - 8
            W_q8.store(unpacked_start + j * 2, w_low)
            W_q8.store(unpacked_start + j * 2 + 1, w_high)


@always_inline
def q4_gemv_cpu(
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
def q4_gemv_batch(
    W_packed: UnsafePointer[UInt8, _],
    scales: UnsafePointer[Float32, _],
    X: UnsafePointer[Float32, _],
    out_ptr: UnsafePointer[Float32, MutExternalOrigin],
    B: Int,
    c_in: Int,
    c_out: Int,
    num_tasks: Int,
    in_stride: Int = -1,
    out_stride: Int = -1,
):
    var tasks = num_tasks
    if tasks > c_out:
        tasks = c_out
    if tasks < 1:
        tasks = 1
    var x_stride = in_stride if in_stride >= 0 else c_in
    var o_stride = out_stride if out_stride >= 0 else c_out

    var chunk_size = c_out // tasks
    var c_in_packed = c_in // 2
    var c_in_blocks = c_in // 32

    def worker(t: Int) capturing -> None:
        var start_r = t * chunk_size
        var end_r = (t + 1) * chunk_size if t < tasks - 1 else c_out
        var acc = alloc[Float32](B)

        for r in range(start_r, end_r):
            var r_offset_packed = r * c_in_packed
            var r_offset_scales = r * c_in_blocks

            for b in range(B):
                acc[b] = 0.0

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

                for b in range(B):
                    var x_base = b * x_stride + x_offset_base
                    var p = (w0 * X.load[8](x_base)).reduce_add() * scale
                    p += (w1 * X.load[8](x_base + 8)).reduce_add() * scale
                    p += (w2 * X.load[8](x_base + 16)).reduce_add() * scale
                    p += (w3 * X.load[8](x_base + 24)).reduce_add() * scale
                    acc[b] += p

            for b in range(B):
                out_ptr[b * o_stride + r] = acc[b]

        acc.free()

    if tasks == 1:
        worker(0)
    else:
        parallelize[worker](tasks)

# Small-batch register-resident q4 GEMV kernel.
# Pre-allocates MAX_B SIMD8 accumulators in CPU registers.
# Loads each 16-byte quantized weight block exactly once, multiply-accumulating
# against all B active token vectors in registers, and does a single
# horizontal reduction per row at the end of the row loop.
@always_inline
@always_inline
def q4_gemv_small_b[
    MAX_B: Int,
    in_type: DType = DType.float32,
    out_type: DType = DType.float32,
](
    W_packed: UnsafePointer[UInt8, _],
    scales: UnsafePointer[Float32, _],
    X: UnsafePointer[Scalar[in_type], _],
    out_ptr: UnsafePointer[Scalar[out_type], MutExternalOrigin],
    B: Int,
    c_in: Int,
    c_out: Int,
    num_tasks: Int,
    in_stride: Int = -1,
    out_stride: Int = -1,
):
    var tasks = num_tasks
    if tasks > c_out:
        tasks = c_out
    if tasks < 1:
        tasks = 1
    var x_stride = in_stride if in_stride >= 0 else c_in
    var o_stride = out_stride if out_stride >= 0 else c_out

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

                # Pre-scale weights once per block to avoid multiplying scale on every token
                w0 = w0 * scale
                w1 = w1 * scale
                w2 = w2 * scale
                w3 = w3 * scale

                comptime for i in range(MAX_B):
                    if i < B:
                        var x_base = i * x_stride + x_offset_base
                        acc[i] = acc[i] + w0 * X.load[8](x_base).cast[DType.float32]()
                        acc[i] = acc[i] + w1 * X.load[8](x_base + 8).cast[DType.float32]()
                        acc[i] = acc[i] + w2 * X.load[8](x_base + 16).cast[DType.float32]()
                        acc[i] = acc[i] + w3 * X.load[8](x_base + 24).cast[DType.float32]()

            comptime for i in range(MAX_B):
                if i < B:
                    out_ptr[i * o_stride + r] = acc[i].reduce_add().cast[out_type]()

    if tasks == 1:
        worker(0)
    else:
        parallelize[worker](tasks)

# Dynamically dispatches to the most narrow specialized small-batch kernel,
# avoiding weight reload over-fetch loops by keeping weight loading to exactly once.
@always_inline
def q4_gemv_moe_dispatch[
    in_type: DType = DType.float32,
    out_type: DType = DType.float32,
](
    W_packed: UnsafePointer[UInt8, _],
    scales: UnsafePointer[Float32, _],
    X: UnsafePointer[Scalar[in_type], _],
    out_ptr: UnsafePointer[Scalar[out_type], MutExternalOrigin],
    B: Int,
    c_in: Int,
    c_out: Int,
    num_tasks: Int,
    in_stride: Int = -1,
    out_stride: Int = -1,
):
    if B <= 1:
        q4_gemv_small_b[1, in_type, out_type](W_packed, scales, X, out_ptr, B, c_in, c_out, num_tasks, in_stride, out_stride)
    elif B <= 2:
        q4_gemv_small_b[2, in_type, out_type](W_packed, scales, X, out_ptr, B, c_in, c_out, num_tasks, in_stride, out_stride)
    elif B <= 4:
        q4_gemv_small_b[4, in_type, out_type](W_packed, scales, X, out_ptr, B, c_in, c_out, num_tasks, in_stride, out_stride)
    elif B <= 8:
        q4_gemv_small_b[8, in_type, out_type](W_packed, scales, X, out_ptr, B, c_in, c_out, num_tasks, in_stride, out_stride)
    elif B <= 12:
        q4_gemv_small_b[12, in_type, out_type](W_packed, scales, X, out_ptr, B, c_in, c_out, num_tasks, in_stride, out_stride)
    elif B <= 16:
        q4_gemv_small_b[16, in_type, out_type](W_packed, scales, X, out_ptr, B, c_in, c_out, num_tasks, in_stride, out_stride)
    elif B <= 24:
        q4_gemv_small_b[24, in_type, out_type](W_packed, scales, X, out_ptr, B, c_in, c_out, num_tasks, in_stride, out_stride)
    elif B <= 32:
        q4_gemv_small_b[32, in_type, out_type](W_packed, scales, X, out_ptr, B, c_in, c_out, num_tasks, in_stride, out_stride)
    else:
        # Fallback to general tiled version
        q4_gemv_tiled[8, in_type, out_type](W_packed, scales, X, out_ptr, B, c_in, c_out, num_tasks, in_stride, out_stride)

# Tiled batched q4 GEMV. Per output row, processes B in tiles of TILE_B,
# keeping TILE_B SIMD8 accumulators in registers. Each weight block is
# loaded once per row (cached in L1 across tiles). Inner loop does
# SIMD multiply-accumulate (no per-b reduce_add); a single reduce_add
# per (row, b) at the end. Contrast with q4_gemv_batch, which does
# B * c_in_blocks * 4 reduce_adds per row — dominant cost at small B.
@always_inline
def q4_gemv_tiled[
    TILE_B: Int,
    in_type: DType = DType.float32,
    out_type: DType = DType.float32,
](
    W_packed: UnsafePointer[UInt8, _],
    scales: UnsafePointer[Float32, _],
    X: UnsafePointer[Scalar[in_type], _],
    out_ptr: UnsafePointer[Scalar[out_type], MutExternalOrigin],
    B: Int,
    c_in: Int,
    c_out: Int,
    num_tasks: Int,
    in_stride: Int = -1,
    out_stride: Int = -1,
):
    var tasks = num_tasks
    if tasks > c_out:
        tasks = c_out
    if tasks < 1:
        tasks = 1
    var x_stride = in_stride if in_stride >= 0 else c_in
    var o_stride = out_stride if out_stride >= 0 else c_out

    var chunk_size = c_out // tasks
    var c_in_packed = c_in // 2
    var c_in_blocks = c_in // 32

    def worker(t: Int) capturing -> None:
        var start_r = t * chunk_size
        var end_r = (t + 1) * chunk_size if t < tasks - 1 else c_out
        var acc = InlineArray[SIMD[DType.float32, 8], TILE_B](uninitialized=True)

        for r in range(start_r, end_r):
            var r_offset_packed = r * c_in_packed
            var r_offset_scales = r * c_in_blocks

            for b_outer in range(0, B, TILE_B):
                var b_end = b_outer + TILE_B if b_outer + TILE_B < B else B
                var b_count = b_end - b_outer

                comptime for i in range(TILE_B):
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

                    # Pre-scale weights once per block to avoid multiplying scale on every token
                    w0 = w0 * scale
                    w1 = w1 * scale
                    w2 = w2 * scale
                    w3 = w3 * scale

                    comptime for i in range(TILE_B):
                        if i < b_count:
                            var x_base = (b_outer + i) * x_stride + x_offset_base
                            acc[i] = acc[i] + w0 * X.load[8](x_base).cast[DType.float32]()
                            acc[i] = acc[i] + w1 * X.load[8](x_base + 8).cast[DType.float32]()
                            acc[i] = acc[i] + w2 * X.load[8](x_base + 16).cast[DType.float32]()
                            acc[i] = acc[i] + w3 * X.load[8](x_base + 24).cast[DType.float32]()

                comptime for i in range(TILE_B):
                    if i < b_count:
                        out_ptr[(b_outer + i) * o_stride + r] = acc[i].reduce_add().cast[out_type]()

    if tasks == 1:
        worker(0)
    else:
        parallelize[worker](tasks)

@always_inline
def gelu_and_mul[
    type: DType = DType.float32
](
    h1: UnsafePointer[Scalar[type], _],
    h2: UnsafePointer[Scalar[type], _],
    out_ptr: UnsafePointer[Scalar[type], MutExternalOrigin],
    size: Int,
):
    comptime width = 16 if type == DType.float16 else 8
    for i in range(0, size, width):
        var x1 = h1.load[width](i).cast[DType.float32]()
        var sig = 1.0 / (1.0 + exp(SIMD[DType.float32, width](-1.702) * x1))
        var gelu = x1 * sig
        var h2_val = h2.load[width](i).cast[DType.float32]()
        out_ptr.store[width](i, (gelu * h2_val).cast[type]())

@always_inline
def gelu_and_mul_batch[
    type: DType = DType.float32
](
    h1: UnsafePointer[Scalar[type], _],
    h2: UnsafePointer[Scalar[type], _],
    out_ptr: UnsafePointer[Scalar[type], MutExternalOrigin],
    B: Int,
    dim: Int,
):
    comptime width = 16 if type == DType.float16 else 8
    for b in range(B):
        var base = b * dim
        for i in range(0, dim, width):
            var x1 = h1.load[width](base + i).cast[DType.float32]()
            var sig = 1.0 / (1.0 + exp(SIMD[DType.float32, width](-1.702) * x1))
            var h2_val = h2.load[width](base + i).cast[DType.float32]()
            out_ptr.store[width](base + i, (x1 * sig * h2_val).cast[type]())

@always_inline
def rms_norm_batch[
    in_type: DType = DType.float32,
    out_type: DType = DType.float32,
](
    x: UnsafePointer[Scalar[in_type], _],
    w: UnsafePointer[Float32, _],
    out_ptr: UnsafePointer[Scalar[out_type], MutExternalOrigin],
    B: Int,
    dim: Int,
):
    for b in range(B):
        rms_norm[in_type, out_type](x + b * dim, w, out_ptr + b * dim, dim)

@always_inline
def rms_norm_noscale_batch[
    in_type: DType = DType.float32,
    out_type: DType = DType.float32,
](
    x: UnsafePointer[Scalar[in_type], _],
    out_ptr: UnsafePointer[Scalar[out_type], MutExternalOrigin],
    B: Int,
    dim: Int,
):
    for b in range(B):
        rms_norm_noscale[in_type, out_type](x + b * dim, out_ptr + b * dim, dim)

@always_inline
def apply_rope[type: DType = DType.float32](vec: UnsafePointer[Scalar[type], MutExternalOrigin], num_heads: Int, head_dim: Int, pos: Int, is_global: Bool):
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
            
            var val1 = Float32(vec[h_offset + i])
            var val2 = Float32(vec[h_offset + half_dim + i])
            
            vec[h_offset + i] = Scalar[type](val1 * cos_theta - val2 * sin_theta)
            vec[h_offset + half_dim + i] = Scalar[type](val2 * cos_theta + val1 * sin_theta)

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
# 3b. Self-Conditioning MLP Forward (DiffusionGemma SC path)
# -----------------------------------------------------------------------------
# Mirrors vLLM DiffusionGemmaSelfConditioning.forward:
#   x = pre_norm(soft_embeds)
#   sc = down_proj(gelu(gate_proj(x)) * up_proj(x))
#   out = post_norm_noscale(inputs_embeds + sc)   (post_norm has_weight=False)
# With zero soft_embeds the SC signal vanishes and out = post_norm_noscale(inputs_embeds).

def sc_forward[
    inputs_embeds_type: DType = DType.float32,
    sc_out_type: DType = DType.float32,
](
    inputs_embeds: UnsafePointer[Scalar[inputs_embeds_type], _],
    soft_embeds: UnsafePointer[Float32, _],
    sc_gate_packed: UnsafePointer[UInt8, _],
    sc_up_packed: UnsafePointer[UInt8, _],
    sc_down_packed: UnsafePointer[UInt8, _],
    sc_gate_scales: UnsafePointer[Float32, _],
    sc_up_scales: UnsafePointer[Float32, _],
    sc_down_scales: UnsafePointer[Float32, _],
    sc_pre_norm: UnsafePointer[Float32, _],
    buf_normed: UnsafePointer[Float32, MutUntrackedOrigin],
    buf_gate: UnsafePointer[Float32, MutUntrackedOrigin],
    buf_up: UnsafePointer[Float32, MutUntrackedOrigin],
    buf_gated: UnsafePointer[Float32, MutUntrackedOrigin],
    buf_down: UnsafePointer[Float32, MutUntrackedOrigin],
    sc_out: UnsafePointer[Scalar[sc_out_type], MutUntrackedOrigin],
    embed_dim: Int,
    sc_size: Int,
    num_tasks: Int,
):
    # 1. pre_norm the soft_embeds (RMSNorm with sc_pre_norm weight)
    rms_norm(soft_embeds, sc_pre_norm, buf_normed, embed_dim)
    # 2. gate_proj and up_proj: [sc_size, embed_dim] x [embed_dim] -> [sc_size]
    q4_gemv_cpu(sc_gate_packed, sc_gate_scales, buf_normed, buf_gate, embed_dim, sc_size, num_tasks)
    q4_gemv_cpu(sc_up_packed, sc_up_scales, buf_normed, buf_up, embed_dim, sc_size, num_tasks)
    # 3. gelu_and_mul: buf_gated = gelu(buf_gate) * buf_up
    gelu_and_mul(buf_gate, buf_up, buf_gated, sc_size)
    # 4. down_proj: [embed_dim, sc_size] x [sc_size] -> [embed_dim]
    q4_gemv_cpu(sc_down_packed, sc_down_scales, buf_gated, buf_down, sc_size, embed_dim, num_tasks)
    # 5. residual + post_norm (no learned weight)
    for i in range(0, embed_dim, 8):
        var val = inputs_embeds.load[8](i).cast[DType.float32]() + buf_down.load[8](i)
        sc_out.store[8](i, val.cast[sc_out_type]())
    rms_norm_noscale[in_type=sc_out_type, out_type=sc_out_type](sc_out, sc_out, embed_dim)


# -----------------------------------------------------------------------------
# 3c. Diffusion Sampler Helpers
# -----------------------------------------------------------------------------
# Computes soft_embeds = (probs @ embed_weight) * normalizer for one canvas position.
# embed_weight is stored as [vocab, embed_dim] in q4. We want out[d] = sum_r probs[r] * W[r,d].
# Threshold: skip tokens with negligible prob (most of the vocab after softmax).
def probs_to_soft_embed(
    probs: UnsafePointer[Float32, _],
    embed_packed: UnsafePointer[UInt8, _],
    embed_scales: UnsafePointer[Float32, _],
    sc_out: UnsafePointer[Float32, MutUntrackedOrigin],
    vocab_size: Int,
    embed_dim: Int,
    num_tasks: Int,
):
    for d in range(0, embed_dim, 8):
        sc_out.store[8](d, SIMD[DType.float32, 8](0.0))
    var num_blocks = embed_dim // 32
    var chunk_size = vocab_size // num_tasks

    def sc_worker(t: Int) capturing -> None:
        var start_r = t * chunk_size
        var end_r = (t + 1) * chunk_size if t < num_tasks - 1 else vocab_size
        for r in range(start_r, end_r):
            var p = probs[r]
            if p < 1e-8:
                continue
            var row_packed_offset = r * (embed_dim // 2)
            var row_scales_offset = r * (embed_dim // 32)
            for b in range(num_blocks):
                var scale = embed_scales[row_scales_offset + b]
                var block_packed_offset = row_packed_offset + b * 16
                var block_out_offset = b * 32

                var bytes = embed_packed.load[16](block_packed_offset)
                var w0 = SIMD[DType.float32, 8]()
                var w1 = SIMD[DType.float32, 8]()
                var w2 = SIMD[DType.float32, 8]()
                var w3 = SIMD[DType.float32, 8]()
                unpack_16bytes(bytes, w0, w1, w2, w3)

                var p_scale_vec = SIMD[DType.float32, 8](p * scale)
                sc_out.store[8](block_out_offset, sc_out.load[8](block_out_offset) + w0 * p_scale_vec)
                sc_out.store[8](block_out_offset + 8, sc_out.load[8](block_out_offset + 8) + w1 * p_scale_vec)
                sc_out.store[8](block_out_offset + 16, sc_out.load[8](block_out_offset + 16) + w2 * p_scale_vec)
                sc_out.store[8](block_out_offset + 24, sc_out.load[8](block_out_offset + 24) + w3 * p_scale_vec)

    if num_tasks == 1:
        sc_worker(0)
    else:
        parallelize[sc_worker](num_tasks)


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
    var use_gpu = True
    var bench_mode = False
    var bench_steps = 2
    var cmd_args = std.sys.argv()
    for idx in range(1, len(cmd_args)):
        if cmd_args[idx] == "--cpu":
            use_gpu = False
        elif cmd_args[idx] == "--bench":
            bench_mode = True
            bench_steps = 2
        elif cmd_args[idx] == "--bench4":
            bench_mode = True
            bench_steps = 4

    var num_tasks = num_physical_cores()
    print("=========================================================")
    print("   DIFFUSIONGEMMA 26B-A4B-IT  CANVAS DIFFUSION ENGINE   ")
    print("=========================================================")
    print("Embedding Size:      2816")
    print("Attention Heads:     16 Query Heads / 8 or 2 KV Heads (GQA)")
    print("Number of Layers:    30 Layers (5 global attention layers)")
    print("Number of Experts:   128 Experts (Top-8 Router routing)")
    print("Canvas Mode:         256-token canvas, 48 max denoise steps")
    print("Self-Conditioning:  2112-dim gated MLP, RAM-resident Q4_0")
    if use_gpu:
        print("Execution Mode:      Hybrid CPU-iGPU (16x16 Q4 LDS on iGPU + L1-2x4 AVX2 on CPU)")
    else:
        print("Execution Mode:      CPU fallbacks (L1-2x4 AVX2 + Dual-Space Deinterleaved MoE)")
    print("---------------------------------------------------------")

    var CONFIG = MoEConfig[]()

    # Initialize AMD direct GPU userspace queue with PASID safety & cached GTT
    if use_gpu:
        print("Initializing direct AMD GPU userspace queue...")
    else:
        print("Initializing CPU-only context memory structures...")
    var gpu_ctx = KFDContext(
        init_queue=use_gpu,
        default_hsaco_path="/home/phil/git/llm-mojo/scratch/q4_matmul_lds.hsaco" if use_gpu else "",
        default_kernel_name="q4_matmul" if use_gpu else "",
    )
    if use_gpu:
        print("Direct AMD GPU userspace queue initialized successfully! (GPU ID: " + String(gpu_ctx.gpu_id) + ")")
    else:
        print("CPU-only context memory structures initialized successfully!")

    # Setup Cached GTT Flags (fast WB L1/L2/L3 CPU access + explicit cache line flushes at GPU boundaries)
    var gtt_cached_flags = KFDContext.DEFAULT_GTT_FLAGS

    # Initialize FFI and libc functions
    var libc = OwnedDLHandle("libc.so.6")
    var c_open = libc.get_function[def(Int, Int32, Int32) thin abi("C") -> Int32]("open")
    var c_close = libc.get_function[def(Int32) thin abi("C") -> Int32]("close")
    var posix_memalign = libc.get_function[def(Int, Int, Int) thin abi("C") -> Int32]("posix_memalign")
    var free_c = libc.get_function[def(Int) thin abi("C") -> None]("free")
    var c_read = libc.get_function[def(Int32, Int, Int) thin abi("C") -> Int]("read")

    # 1. Allocate and Load DiffusionGemma Decoder Weights (RAM-resident in Cached GTT)
    var weights_file = String("diffusiongemma_q4.bin")
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
    var weights_gpu_buf = gpu_ctx.alloc_gpu_buffer(weights_file_size, gtt_cached_flags, cpu_access=True)
    var base_addr = weights_gpu_buf.cpu_addr
    var base_ptr = UnsafePointer[UInt8, MutExternalOrigin](unsafe_from_address=base_addr)

    print("Loading DiffusionGemma weights into RAM (14.26 GB) in chunks...")
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
    print("Successfully loaded all DiffusionGemma weights into RAM: ", total_read, " bytes")
    _ = c_close(weights_fd)

    if use_gpu:
        print("Flushing CPU cache lines of Q4 weights for iGPU visibility...")
        gpu_ctx.flush_cache_range(base_addr, weights_file_size)
        print("Successfully flushed Q4 weights!")

    # 2. Allocate and Load Final Layer Norm Weights
    var final_norm_file = String("diffusiongemma_final_norm.bin")
    print("Opening final norm weights file 'diffusiongemma_final_norm.bin'...")
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

    # 2b. Load Self-Conditioning MLP weights (Q4_0)
    #    Layout: [gate_proj.packed][up_proj.packed][down_proj.packed]
    #            [gate_proj.scales][up_proj.scales][down_proj.scales]
    #            [pre_norm.weight fp32]
    #    post_norm has no learned weight (has_weight=False per vLLM).
    var sc_file = String("diffusiongemma_sc_q4.bin")
    print("Opening self-conditioning weights file '" + sc_file + "'...")
    var sc_path_bytes = sc_file.as_bytes()
    var sc_path_ptr = alloc[UInt8](len(sc_path_bytes) + 1)
    for i in range(len(sc_path_bytes)):
        sc_path_ptr[i] = sc_path_bytes[i]
    sc_path_ptr[len(sc_path_bytes)] = 0
    var sc_fd = c_open(sc_path_ptr.__int__(), 0, 0)
    sc_path_ptr.free()
    if sc_fd < 0:
        print("CRITICAL: Failed to open '" + sc_file + "'! Diffusion will not work.")
        return

    # SC size: gate(2112x2816) + up(2112x2816) + down(2816x2112) — all 5,940,992 elts
    # packed = 2,970,496 B each ; scales = 185,656 * 4 B each ; pre_norm = 2816 * 4 B
    comptime SC_MATRIX_SIZE = 2112 * 2816  # = 5,940,992 elements per matrix (gate/up/down all same)
    comptime SC_PACKED = SC_MATRIX_SIZE // 2       # 2,970,496 bytes per matrix
    comptime SC_SCALES = SC_MATRIX_SIZE // 32      # 185,656 floats per matrix
    comptime SC_SCALES_BYTES = SC_SCALES * 4
    comptime SC_PRE_NORM_BYTES = 2816 * 4
    comptime SC_TOTAL = (SC_PACKED * 3) + (SC_SCALES_BYTES * 3) + SC_PRE_NORM_BYTES
    comptime SC_ALIGNED = ((SC_TOTAL + 4095) // 4096) * 4096

    var memptr_sc = alloc[Int](1)
    var err_sc = posix_memalign(memptr_sc.__int__(), 4096, SC_ALIGNED)
    if err_sc != 0:
        print("CRITICAL: Failed to allocate SC weight memory!")
        _ = c_close(sc_fd)
        memptr_sc.free()
        return
    var sc_base_addr = memptr_sc[0]
    memptr_sc.free()
    var sc_base = UnsafePointer[UInt8, MutExternalOrigin](unsafe_from_address=sc_base_addr)

    var sc_total_read: Int = 0
    while sc_total_read < SC_ALIGNED:
        var bytes_to_read = SC_ALIGNED - sc_total_read
        var read_res = c_read(sc_fd, sc_base_addr + sc_total_read, bytes_to_read)
        if read_res <= 0:
            print("CRITICAL: Failed to read SC weights!")
            _ = c_close(sc_fd)
            return
        sc_total_read += read_res
    _ = c_close(sc_fd)
    print("Self-conditioning weights loaded: ", sc_total_read, " bytes")

    # Lay out the SC pointers
    var sc_gate_packed = sc_base
    var sc_up_packed = sc_gate_packed + SC_PACKED
    var sc_down_packed = sc_up_packed + SC_PACKED
    var sc_scales_base = (sc_down_packed + SC_PACKED).bitcast[Float32]()
    var sc_gate_scales = sc_scales_base
    var sc_up_scales = sc_gate_scales + SC_SCALES
    var sc_down_scales = sc_up_scales + SC_SCALES
    var sc_pre_norm = sc_down_scales + SC_SCALES

    # 3. Load Q4 Token Embeddings
    print("Loading Q4 Token Embeddings...")
    var embed = Q4Embedding("diffusiongemma_embed_q4_32k.bin", CONFIG.vocab_size, CONFIG.embed_dim)

    # 4. Allocate continuous KV Cache for 30 layers
    var kv_cache_offsets = List[Int]()
    var current_cache_offset = 0
    for l in range(30):
        kv_cache_offsets.append(current_cache_offset)
        var is_global = CONFIG.is_global_layer(l)
        var kv_heads = 2 if is_global else 8
        var head_dim = 512 if is_global else 256
        current_cache_offset += CONFIG.max_seq_len * kv_heads * head_dim
    print("Allocating dLLM-KVQ FP16 KV Cache (", (current_cache_offset * 4) // 1024 // 1024, " MB)...")
    var k_cache = alloc[Float16](current_cache_offset)
    var v_cache = alloc[Float16](current_cache_offset)
    for i in range(current_cache_offset):
        k_cache[i] = Float16(0.0)
        v_cache[i] = Float16(0.0)

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

    # 5b. Diffusion canvas constants and buffers
    comptime CANVAS_LEN = 256
    comptime MAX_DENOISE_STEPS = 48
    comptime SC_SIZE = 2112
    comptime MAX_Q_SIZE = 16 * 512  # 8192 (global layer Q size; sliding = 16*256=4096)
    comptime MAX_KV_SIZE = 8 * 512  # 4096 (max KV stride: sliding 8*256=2048, global 2*512=1024)

    var canvas_hidden_buf = gpu_ctx.alloc_gpu_buffer(CANVAS_LEN * CONFIG.embed_dim * 2, gtt_cached_flags, cpu_access=True)
    var canvas_q_buf = gpu_ctx.alloc_gpu_buffer(CANVAS_LEN * MAX_Q_SIZE * 2, gtt_cached_flags, cpu_access=True)
    var canvas_k_buf = gpu_ctx.alloc_gpu_buffer(CANVAS_LEN * MAX_KV_SIZE * 2, gtt_cached_flags, cpu_access=True)
    var canvas_v_buf = gpu_ctx.alloc_gpu_buffer(CANVAS_LEN * MAX_KV_SIZE * 2, gtt_cached_flags, cpu_access=True)
    var canvas_attn_out_buf = gpu_ctx.alloc_gpu_buffer(CANVAS_LEN * MAX_Q_SIZE * 2, gtt_cached_flags, cpu_access=True)
    var canvas_proj_attn_buf = gpu_ctx.alloc_gpu_buffer(CANVAS_LEN * CONFIG.embed_dim * 4, gtt_cached_flags, cpu_access=True)

    # Canvas hidden states [CANVAS_LEN, embed_dim] — flows through layers
    var canvas_hidden = UnsafePointer[Float16, MutUntrackedOrigin](unsafe_from_address=canvas_hidden_buf.cpu_addr)
    # Canvas Q storage [CANVAS_LEN, MAX_Q_SIZE] — stored in phase 1, consumed in phase 2
    var canvas_q = UnsafePointer[Float16, MutUntrackedOrigin](unsafe_from_address=canvas_q_buf.cpu_addr)
    # Canvas K/V storage [CANVAS_LEN, MAX_KV_SIZE] — for batched QKV proj
    var canvas_k = UnsafePointer[Float16, MutUntrackedOrigin](unsafe_from_address=canvas_k_buf.cpu_addr)
    var canvas_v = UnsafePointer[Float16, MutUntrackedOrigin](unsafe_from_address=canvas_v_buf.cpu_addr)
    # Batched attention output [CANVAS_LEN, MAX_Q_SIZE]
    var canvas_attn_out = UnsafePointer[Float16, MutUntrackedOrigin](unsafe_from_address=canvas_attn_out_buf.cpu_addr)
    # Batched Wo output [CANVAS_LEN, embed_dim]
    var canvas_proj_attn = UnsafePointer[Float32, MutUntrackedOrigin](unsafe_from_address=canvas_proj_attn_buf.cpu_addr)
    # Batched norm output [CANVAS_LEN, embed_dim]
    var canvas_norm_buf = gpu_ctx.alloc_gpu_buffer(CANVAS_LEN * CONFIG.embed_dim * 4, gtt_cached_flags, cpu_access=True)
    var canvas_norm = UnsafePointer[Float32, MutUntrackedOrigin](unsafe_from_address=canvas_norm_buf.cpu_addr)

    # Reusable packed flat temp buffers for Coalesced Batched GEMM launches
    var temp_packed_q_buf = gpu_ctx.alloc_gpu_buffer(CANVAS_LEN * MAX_Q_SIZE * 4, gtt_cached_flags, cpu_access=True)
    var temp_packed_k_buf = gpu_ctx.alloc_gpu_buffer(CANVAS_LEN * CONFIG.standard_mlp_dim * 4, gtt_cached_flags, cpu_access=True)
    var temp_packed_v_buf = gpu_ctx.alloc_gpu_buffer(CANVAS_LEN * CONFIG.standard_mlp_dim * 4, gtt_cached_flags, cpu_access=True)

    var temp_packed_q = UnsafePointer[Float32, MutUntrackedOrigin](unsafe_from_address=temp_packed_q_buf.cpu_addr)
    var temp_packed_k = UnsafePointer[Float32, MutUntrackedOrigin](unsafe_from_address=temp_packed_k_buf.cpu_addr)
    var temp_packed_v = UnsafePointer[Float32, MutUntrackedOrigin](unsafe_from_address=temp_packed_v_buf.cpu_addr)
    # Batched FFN intermediates
    var canvas_ffn_gate = alloc[Float16](CANVAS_LEN * CONFIG.standard_mlp_dim)
    var canvas_ffn_up = alloc[Float16](CANVAS_LEN * CONFIG.standard_mlp_dim)
    var canvas_ffn_gated = alloc[Float16](CANVAS_LEN * CONFIG.standard_mlp_dim)
    var canvas_ffn_down = alloc[Float16](CANVAS_LEN * CONFIG.embed_dim)
    var canvas_shared_norm = alloc[Float16](CANVAS_LEN * CONFIG.embed_dim)
    var canvas_moe_in = alloc[Float16](CANVAS_LEN * CONFIG.embed_dim)
    var canvas_moe_out = alloc[Float16](CANVAS_LEN * CONFIG.embed_dim)
    var canvas_moe_norm = alloc[Float16](CANVAS_LEN * CONFIG.embed_dim)
    var canvas_combined = alloc[Float16](CANVAS_LEN * CONFIG.embed_dim)
    var canvas_ffn_out = alloc[Float16](CANVAS_LEN * CONFIG.embed_dim)
    # Per-token attention scores buffer for parallelized attention
    var canvas_attn_scores = alloc[Float32](CANVAS_LEN * CONFIG.max_seq_len)
    # MoE Dual-Space Deinterleaved F32 scratch buffers (llm-mojo-2tc.2)
    comptime MOE_WORK_STRIDE = CANVAS_LEN * CONFIG.embed_dim + CANVAS_LEN * CONFIG.expert_mlp_dim * 2 + CANVAS_LEN * CONFIG.expert_mlp_dim + CONFIG.embed_dim * 2
    var moe_deinterleaved_x = alloc[Float32](CANVAS_LEN * CONFIG.embed_dim)
    var moe_work_buf = alloc[Float32](num_tasks * MOE_WORK_STRIDE)
    var moe_expert_outs = alloc[Float32](CANVAS_LEN * CONFIG.top_k_experts * CONFIG.embed_dim)
    var canvas_router_scores = alloc[Float32](CANVAS_LEN * CONFIG.num_experts)
    var canvas_top_idx = alloc[Int](CANVAS_LEN * CONFIG.top_k_experts)
    var canvas_top_val = alloc[Float32](CANVAS_LEN * CONFIG.top_k_experts)
    var canvas_top_weights = alloc[Float32](CANVAS_LEN * CONFIG.top_k_experts)
    var canvas_scaled_norm = alloc[Float32](CANVAS_LEN * CONFIG.embed_dim)
    var canvas_routed_norm = alloc[Float32](CANVAS_LEN * CONFIG.embed_dim)
    # Canvas token ids [CANVAS_LEN]
    var canvas_tokens = alloc[Int](CANVAS_LEN)
    var argmax_canvas = alloc[Int](CANVAS_LEN)
    # Self-conditioning soft embeds [CANVAS_LEN, embed_dim]
    var sc_embeds = alloc[Float32](CANVAS_LEN * CONFIG.embed_dim)
    # Batched SC MLP scratch buffers
    var sc_buf_normed = alloc[Float32](CANVAS_LEN * CONFIG.embed_dim)
    var sc_buf_gated = alloc[Float32](CANVAS_LEN * SC_SIZE)
    var sc_buf_down = alloc[Float32](CANVAS_LEN * CONFIG.embed_dim)
    # Per-canvas-token final norm + logits storage
    var canvas_final_norm = alloc[Float32](CANVAS_LEN * CONFIG.embed_dim)
    # Per-token logits buffer for batched logit projection + Top-32 sparse sampler
    var canvas_logits = alloc[Float32](CANVAS_LEN * CONFIG.vocab_size)
    var rand_buf = alloc[Float32](CANVAS_LEN)
    var token_entropy = alloc[Float32](CANVAS_LEN)
    var new_tokens = alloc[Int](CANVAS_LEN)
    var argmax_tokens = alloc[Int](CANVAS_LEN)
    var sorted_idx = alloc[Int](CANVAS_LEN)
    var sorted_ent = alloc[Float32](CANVAS_LEN)
    var eb_mask = alloc[Bool](CANVAS_LEN)
    var history = alloc[Int](2 * CANVAS_LEN)  # ST=2 slots
    # Fast-dLLM / dKV-Cache active-canvas compaction buffers (llm-mojo-2tc.3)
    var locked_mask = alloc[Bool](CANVAS_LEN)
    var active_canvas_idx = alloc[Int](CANVAS_LEN)
    var sc_compact = alloc[Float32](CANVAS_LEN * CONFIG.embed_dim)
    var compact_rand_buf = alloc[Float32](CANVAS_LEN)
    var compact_entropy = alloc[Float32](CANVAS_LEN)
    var compact_argmax = alloc[Int](CANVAS_LEN)
    var compact_new_tokens = alloc[Int](CANVAS_LEN)
    # TeaCache & TaylorSeer feature forecasting buffers (llm-mojo-2tc.4 & 2tc.7)
    var teacache_h5_prev = alloc[Float16](CANVAS_LEN * CONFIG.embed_dim)
    var taylor_h5_curr = alloc[Float16](CANVAS_LEN * CONFIG.embed_dim)
    var taylor_D = alloc[Float32](CANVAS_LEN * CONFIG.embed_dim)
    var taylor_d1 = alloc[Float32](CANVAS_LEN * CONFIG.embed_dim)
    var taylor_d2 = alloc[Float32](CANVAS_LEN * CONFIG.embed_dim)
    var teacache_accum = alloc[Float32](1)
    # AdaCache + dLLM-Cache per-token partial canvas update buffers (llm-mojo-2tc.8)
    var adacache_h1_prev = alloc[Float16](CANVAS_LEN * CONFIG.embed_dim)
    var adacache_skip_streak = alloc[Int](CANVAS_LEN)

    print("Canvas buffers allocated: ", CANVAS_LEN, " tokens, ",
          (CANVAS_LEN * CONFIG.embed_dim * 4 + CANVAS_LEN * MAX_Q_SIZE * 4) // 1024 // 1024, " MB hidden+Q")

    # 6. Initialize tokenizer (skipped in --bench mode for zero-Python PASID safety)
    var tokenizer = PythonObject()
    var py_builtins = PythonObject()
    if not bench_mode:
        var transformers = Python.import_module("transformers")
        print("Loading AutoTokenizer...")
        tokenizer = transformers.AutoTokenizer.from_pretrained("google/gemma-4-12B-it")
        py_builtins = Python.import_module("builtins")
        print("Tokenizer loaded successfully!")

    print("---------------------------------------------------------")
    print("   Chat Agent is ready! Type 'exit' or 'quit' to end.")
    print("---------------------------------------------------------")

    var prompt_scale = sqrt(Float32(CONFIG.embed_dim))
    var bench_iter = 0

    while True:
        print()
        var user_input_str: String
        if bench_mode:
            if bench_iter >= 1:
                break
            user_input_str = "Explain quantum computing in simple terms."
            bench_iter += 1
            print("User > [BENCH] " + user_input_str)
        else:
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


        var prompt_tokens = List[Int]()
        if bench_mode:
            var bench_toks = InlineArray[Int, 16](uninitialized=True)
            bench_toks[0] = 2
            bench_toks[1] = 105
            bench_toks[2] = 2364
            bench_toks[3] = 107
            bench_toks[4] = 1841
            bench_toks[5] = 603
            bench_toks[6] = 496
            bench_toks[7] = 1775
            bench_toks[8] = 675
            bench_toks[9] = 496
            bench_toks[10] = 1212
            bench_toks[11] = 3
            bench_toks[12] = 106
            bench_toks[13] = 107
            bench_toks[14] = 105
            bench_toks[15] = 4368
            for i in range(16):
                prompt_tokens.append(bench_toks[i])
        else:
            var prompt_str = String("<bos><|turn>user\n") + user_input_str + String("<turn|>\n<|turn>model\n")
            var encoded = tokenizer.encode(prompt_str, add_special_tokens=False)
            for i in range(len(encoded)):
                var tok = Int(py=encoded[i])
                if tok >= CONFIG.vocab_size:
                    tok = 3 # Map out-of-vocab tokens to <unk> (index 3) to prevent out-of-bounds errors
                prompt_tokens.append(tok)

        var num_prompt_tokens = len(prompt_tokens)
        if num_prompt_tokens + CANVAS_LEN > CONFIG.max_seq_len:
            print("Prompt too long! Max ", CONFIG.max_seq_len - CANVAS_LEN, " tokens, got ", num_prompt_tokens)
            continue

        # Zero-out the KV Cache for the new interaction
        for i in range(0, current_cache_offset, 16):
            k_cache.store[16](i, SIMD[DType.float16, 16](0.0))
            v_cache.store[16](i, SIMD[DType.float16, 16](0.0))

        print("Agent > (diffusion mode)")

        var t_prefill_start = std.time.perf_counter()
        var t_decode_start: Float64 = 0.0

        # Profiling accumulators (kept for compat with prefill per-layer code)
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

        # --- Encoder Prefill (causal, token-first, writes K/V) ---
        var p_scale_vec_prefill = SIMD[DType.float32, 8](prompt_scale)
        for pos in range(num_prompt_tokens):
            embed.reconstruct(prompt_tokens[pos], local_x)
            for i in range(0, CONFIG.embed_dim, 8):
                local_x.store[8](i, local_x.load[8](i) * p_scale_vec_prefill)

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
                q4_gemv_cpu(weights.wq_packed, weights.wq_scales, norm_x, q, CONFIG.embed_dim, layer_heads * layer_head_dim, num_tasks)
                q4_gemv_cpu(weights.wk_packed, weights.wk_scales, norm_x, k_vec, CONFIG.embed_dim, layer_kv_heads * layer_head_dim, num_tasks)
                if is_global:
                    for i in range(kv_stride):
                        v_vec[i] = k_vec[i]
                else:
                    q4_gemv_cpu(weights.wv_packed, weights.wv_scales, norm_x, v_vec, CONFIG.embed_dim, layer_kv_heads * layer_head_dim, num_tasks)
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

                # 5. KV Cache write (FP16 dLLM-KVQ)
                var layer_cache_offset = kv_cache_offsets[l]
                var step_cache_offset = layer_cache_offset + pos * kv_stride
                for i in range(0, kv_stride, 8):
                    k_cache.store[8](step_cache_offset + i, k_vec.load[8](i).cast[DType.float16]())
                    v_cache.store[8](step_cache_offset + i, v_vec.load[8](i).cast[DType.float16]())
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
                        var score_vec = SIMD[DType.float32, 8](0.0)
                        for d in range(0, layer_head_dim, 8):
                            score_vec += q.load[8](h_offset + d) * k_cache.load[8](t_cache_offset + d).cast[DType.float32]()
                        attn_scores[t] = score_vec.reduce_add()
                        
                    softmax(attn_scores, pos + 1)
                    
                    for d in range(0, layer_head_dim, 8):
                        var val_vec = SIMD[DType.float32, 8](0.0)
                        for t in range(pos + 1):
                            var t_cache_offset = layer_cache_offset + t * kv_stride + g_offset
                            val_vec += SIMD[DType.float32, 8](attn_scores[t]) * v_cache.load[8](t_cache_offset + d).cast[DType.float32]()
                        attn_out.store[8](h_offset + d, val_vec)
                ta1 = std.time.perf_counter()
                if pos >= num_prompt_tokens:
                    p_attn_dot_product += ta1 - ta0

                # 7. Out projection and Attention Residual Add
                ta0 = std.time.perf_counter()
                q4_gemv_cpu(weights.wo_packed, weights.wo_scales, attn_out, proj_attn_out, layer_heads * layer_head_dim, CONFIG.embed_dim, num_tasks)
                rms_norm(proj_attn_out, weights.post_attention_layernorm_weight, proj_attn_out, CONFIG.embed_dim)
                for i in range(0, CONFIG.embed_dim, 8):
                    local_x.store[8](i, local_x.load[8](i) + proj_attn_out.load[8](i))
                ta1 = std.time.perf_counter()
                if pos >= num_prompt_tokens:
                    p_wo_proj += ta1 - ta0

                # 8. Parallel Feedforward: Shared MLP + sparse MoE Experts
                # Path A: Shared MLP
                ta0 = std.time.perf_counter()
                rms_norm(local_x, weights.pre_feedforward_layernorm_weight, norm_x, CONFIG.embed_dim)
                q4_gemv_cpu(weights.w_gate_packed, weights.w_gate_scales, norm_x, shared_h1, CONFIG.embed_dim, CONFIG.standard_mlp_dim, num_tasks)
                q4_gemv_cpu(weights.w_up_packed, weights.w_up_scales, norm_x, shared_h2, CONFIG.embed_dim, CONFIG.standard_mlp_dim, num_tasks)
                gelu_and_mul(shared_h1, shared_h2, shared_ffn_out_gate, CONFIG.standard_mlp_dim)
                q4_gemv_cpu(weights.w_down_packed, weights.w_down_scales, shared_ffn_out_gate, shared_mlp_out, CONFIG.standard_mlp_dim, CONFIG.embed_dim, num_tasks)
                rms_norm(shared_mlp_out, weights.post_feedforward_layernorm_1_weight, shared_h1_norm, CONFIG.embed_dim)
                ta1 = std.time.perf_counter()
                if pos >= num_prompt_tokens:
                    p_shared_mlp += ta1 - ta0

                # Path B: Router and Experts
                ta0 = std.time.perf_counter()
                rms_norm_noscale(local_x, norm_x, CONFIG.embed_dim)
                var r_scale_factor = Float32(1.0 / sqrt(2816.0))
                var r_scale_vec = SIMD[DType.float32, 8](r_scale_factor)
                for j in range(0, CONFIG.embed_dim, 8):
                    scaled_norm_x.store[8](j, norm_x.load[8](j) * weights.router_scale.load[8](j) * r_scale_vec)
                q4_gemv_cpu(weights.w_router_packed, weights.w_router_scales, scaled_norm_x, scores, CONFIG.embed_dim, CONFIG.num_experts, num_tasks)
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

                for i in range(0, CONFIG.embed_dim, 8):
                    moe_out.store[8](i, SIMD[DType.float32, 8](0.0))
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
                    q4_gemv_cpu(exp_gate_up_packed, exp_gate_up_scales, moe_in, exp_gate_up_out, CONFIG.embed_dim, CONFIG.expert_mlp_dim * 2, 1)

                    var gate_ptr = exp_gate_up_out
                    var up_ptr = exp_gate_up_out + CONFIG.expert_mlp_dim
                    gelu_and_mul(gate_ptr, up_ptr, exp_act, CONFIG.expert_mlp_dim)

                    var exp_down_packed = weights.experts_down_packed + e * (CONFIG.embed_dim * CONFIG.expert_mlp_dim // 2)
                    var exp_down_scales = weights.experts_down_scales + e * (CONFIG.embed_dim * CONFIG.expert_mlp_dim // 32)
                    q4_gemv_cpu(exp_down_packed, exp_down_scales, exp_act, exp_down_out, CONFIG.expert_mlp_dim, CONFIG.embed_dim, 1)

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
                    var w_e_vec = SIMD[DType.float32, 8](w_e)
                    for j in range(0, CONFIG.embed_dim, 8):
                        moe_out.store[8](j, moe_out.load[8](j) + exp_down_out.load[8](j) * w_e_vec)

                rms_norm(moe_out, weights.post_feedforward_layernorm_2_weight, moe_out_norm, CONFIG.embed_dim)

                # Combine Shared MLP and MoE Path
                for i in range(0, CONFIG.embed_dim, 8):
                    combined_ffn.store[8](i, shared_h1_norm.load[8](i) + moe_out_norm.load[8](i))
                rms_norm(combined_ffn, weights.post_feedforward_layernorm_weight, ffn_out, CONFIG.embed_dim)

                # Add to residual and multiply by layer_scalar
                var scalar = weights.layer_scalar[0]
                var scalar_vec = SIMD[DType.float32, 8](scalar)
                for i in range(0, CONFIG.embed_dim, 8):
                    local_x.store[8](i, (local_x.load[8](i) + ffn_out.load[8](i)) * scalar_vec)
                ta1 = std.time.perf_counter()
                if pos >= num_prompt_tokens:
                    p_moe_combine += ta1 - ta0

        var t_prefill_end = std.time.perf_counter()
        print("\n  Prefill: ", num_prompt_tokens, " tokens in ",
              String((t_prefill_end - t_prefill_start) * 1000.0), " ms")

        # --- Canvas initialization (pure-Mojo xorshift64* PRNG: zero Python heap allocs) ---
        var rng_state: UInt64 = 0x853C49E6748FEA9B
        for t in range(CANVAS_LEN):
            rng_state ^= rng_state >> 12
            rng_state ^= rng_state << 25
            rng_state ^= rng_state >> 27
            var rnd_u64 = rng_state * 0x2545F4914F6CDD1D
            canvas_tokens[t] = Int(rnd_u64 % UInt64(CONFIG.vocab_size))
        for i in range(CANVAS_LEN * CONFIG.embed_dim):
            sc_embeds[i] = 0.0

        # --- Sampler state ---
        var denoise_step: Int = 0
        var is_encoder_phase: Bool = False
        var hist_len: Int = 0
        var prev_mean_ent: Float32 = 0.0
        var steps_executed: Int = 0
        var max_steps_this_turn = bench_steps if bench_mode else MAX_DENOISE_STEPS

        t_decode_start = std.time.perf_counter()

        # --- Denoise loop ---
        var t_max: Float32 = 0.8
        var t_min: Float32 = 0.4
        var entropy_bound: Float32 = 0.1
        var confidence_threshold: Float32 = 0.005
        for t in range(CANVAS_LEN):
            locked_mask[t] = False
            adacache_skip_streak[t] = 0
        var full_steps_count = 0
        var consec_skips = 0
        teacache_accum[0] = 0.0

        while denoise_step < max_steps_this_turn:
            var active_len = build_active_canvas_map(locked_mask, active_canvas_idx, CANVAS_LEN)
            if active_len == 0:
                print("  [denoise ", denoise_step, "] all 256 tokens locked — early exit!")
                break
            var active_top_k = compute_moe_lod_k(denoise_step, prev_mean_ent)
            var moe_top_p = compute_moe_lod_top_p(denoise_step, prev_mean_ent)
            print("  [denoise ", denoise_step, "] active_len=", active_len, "/", CANVAS_LEN, " moe_lod_k=", active_top_k)
            steps_executed += 1

            # Temperature schedule (computed up front for TeaCache polynomial modulation & Sampler)
            var remaining = MAX_DENOISE_STEPS - denoise_step
            if remaining < 1:
                remaining = 1
            var temp = t_min + (t_max - t_min) * Float32(remaining) / Float32(MAX_DENOISE_STEPS)
            if temp < 1e-10:
                temp = 1e-10

            # Compute canvas embeddings + batched self-conditioning on active_len unconverged tokens (llm-mojo-2tc.1 & 2tc.3)
            var t_sc_start = std.time.perf_counter()
            var p_scale_vec = SIMD[DType.float32, 8](prompt_scale)
            for b in range(active_len):
                var orig_t = active_canvas_idx[b]
                var hidden_b = canvas_hidden + b * CONFIG.embed_dim
                embed.reconstruct[type=DType.float16](canvas_tokens[orig_t], hidden_b)
                for i in range(0, CONFIG.embed_dim, 8):
                    var val = hidden_b.load[8](i).cast[DType.float32]() * p_scale_vec
                    hidden_b.store[8](i, val.cast[DType.float16]())
            var sc_in_ptr = sc_embeds
            if active_len < CANVAS_LEN:
                gather_active[DType.float32](sc_embeds, active_canvas_idx, sc_compact, active_len, CONFIG.embed_dim)
                sc_in_ptr = sc_compact
            sc_forward_batch[inputs_embeds_type=DType.float16, sc_out_type=DType.float16](
                canvas_hidden,
                sc_in_ptr,
                sc_gate_packed, sc_up_packed, sc_down_packed,
                sc_gate_scales, sc_up_scales, sc_down_scales,
                sc_pre_norm,
                sc_buf_normed, sc_buf_gated, sc_buf_down,
                canvas_hidden,
                active_len, CONFIG.embed_dim, SC_SIZE, num_tasks,
            )
            var t_sc_end = std.time.perf_counter()

            # Decoder forward: layer-first, compacted active_len projections, full bidirectional attention
            var skip_mid_layers = False
            for l in range(30):
                if skip_mid_layers and l >= 6 and l <= 22:
                    continue
                var weights = CONFIG.get_layer_weights(l, base_ptr + CONFIG.get_layer_offset(l))
                var is_global = CONFIG.is_global_layer(l)
                var layer_heads = 16
                var layer_kv_heads = 2 if is_global else 8
                var layer_head_dim = 512 if is_global else 256
                var kv_stride = layer_kv_heads * layer_head_dim
                var ratio = layer_heads // layer_kv_heads
                var layer_cache_offset = kv_cache_offsets[l]
                var seq_len = num_prompt_tokens + CANVAS_LEN
                var q_size = layer_heads * layer_head_dim

                var tl_qkv = std.time.perf_counter()
                # Phase 1a: Batched attention input norm + QKV projections on active_len tokens (CPU-iGPU overlap)
                rms_norm_batch[in_type=DType.float16, out_type=DType.float32](canvas_hidden, weights.input_layernorm_weight, canvas_norm, active_len, CONFIG.embed_dim)
                
                if use_gpu:
                    # Flush canvas_norm once, enqueue Wk on iGPU (~18 ms) while CPU computes Wq (+ Wv if sliding)
                    gpu_ctx.flush_cache_range(canvas_norm.__int__(), active_len * CONFIG.embed_dim * 4)
                    var slot_k = gpu_ctx.enqueue_matmul(
                        weights.wk_packed.__int__(),
                        weights.wk_scales.__int__(),
                        canvas_norm.__int__(),
                        temp_packed_k.__int__(),
                        CONFIG.embed_dim,
                        kv_stride,
                        batch_size=active_len,
                        flush_cache=False,
                        ring_doorbell_now=True,
                    )
                    # Concurrently compute Wq (and Wv if sliding) on 8 CPU cores using L1-cached 2x4 AVX2 GEMM
                    q4_gemm_l1_2x4[DType.float32, DType.float16](
                        weights.wq_packed, weights.wq_scales, canvas_norm, canvas_q,
                        active_len, CONFIG.embed_dim, q_size, num_tasks,
                        in_stride=CONFIG.embed_dim, out_stride=MAX_Q_SIZE
                    )
                    if not is_global:
                        q4_gemm_l1_2x4[DType.float32, DType.float16](
                            weights.wv_packed, weights.wv_scales, canvas_norm, canvas_v,
                            active_len, CONFIG.embed_dim, kv_stride, num_tasks,
                            in_stride=CONFIG.embed_dim, out_stride=MAX_KV_SIZE
                        )
                    # Wait for iGPU Wk completion and copy into stride-padded canvas_k
                    gpu_ctx.wait_matmul(slot_k, temp_packed_k.__int__(), kv_stride, batch_size=active_len, flush_cache=True)

                    for b in range(active_len):
                        var src_ptr = temp_packed_k + b * kv_stride
                        var dst_ptr = canvas_k + b * MAX_KV_SIZE
                        for i in range(0, kv_stride, 8):
                            dst_ptr.store[8](i, src_ptr.load[8](i).cast[DType.float16]())

                    if is_global:
                        for b in range(active_len):
                            var v_ptr = canvas_v + b * MAX_KV_SIZE
                            var k_ptr = canvas_k + b * MAX_KV_SIZE
                            for i in range(0, kv_stride, 8):
                                v_ptr.store[8](i, k_ptr.load[8](i))
                else:
                    # CPU L1-cached 2x4 AVX2 batched Q4 GEMM
                    q4_gemm_l1_2x4[DType.float32, DType.float16](weights.wq_packed, weights.wq_scales, canvas_norm, canvas_q, active_len, CONFIG.embed_dim, q_size, num_tasks, in_stride=CONFIG.embed_dim, out_stride=MAX_Q_SIZE)
                    q4_gemm_l1_2x4[DType.float32, DType.float16](weights.wk_packed, weights.wk_scales, canvas_norm, canvas_k, active_len, CONFIG.embed_dim, kv_stride, num_tasks, in_stride=CONFIG.embed_dim, out_stride=MAX_KV_SIZE)
                    if is_global:
                        for b in range(active_len):
                            var v_ptr = canvas_v + b * MAX_KV_SIZE
                            var k_ptr = canvas_k + b * MAX_KV_SIZE
                            for i in range(0, kv_stride, 8):
                                v_ptr.store[8](i, k_ptr.load[8](i))
                    else:
                        q4_gemm_l1_2x4[DType.float32, DType.float16](weights.wv_packed, weights.wv_scales, canvas_norm, canvas_v, active_len, CONFIG.embed_dim, kv_stride, num_tasks, in_stride=CONFIG.embed_dim, out_stride=MAX_KV_SIZE)
                var tl_qkv_end = std.time.perf_counter()

                # Phase 1b: Parallelized Head norms + Hoisted SIMD8 RoPE + FP16 dLLM-KVQ cache scatter (llm-mojo-2tc.9)
                prep_qkv_norm_rope_kvq_batch(
                    canvas_q,
                    canvas_k,
                    canvas_v,
                    weights.q_norm_weight,
                    weights.k_norm_weight,
                    active_canvas_idx,
                    k_cache,
                    v_cache,
                    active_len,
                    num_prompt_tokens,
                    layer_heads,
                    layer_kv_heads,
                    layer_head_dim,
                    kv_stride,
                    layer_cache_offset,
                    MAX_Q_SIZE,
                    MAX_KV_SIZE,
                    is_global,
                )

                # Phase 2a: Flash-dLLM Head-Group & Query-Tile Fused Bidirectional Attention over FP16 dLLM-KVQ
                var tl_attn = std.time.perf_counter()
                flash_dllm_bidirectional_attn(
                    canvas_q,
                    k_cache,
                    v_cache,
                    canvas_attn_out,
                    active_len,
                    seq_len,
                    layer_heads,
                    layer_kv_heads,
                    layer_head_dim,
                    kv_stride,
                    layer_cache_offset,
                    MAX_Q_SIZE,
                )

                var tl_attn_end = std.time.perf_counter()

                # Phase 2b: Batched Wo projection using L1-2x4 AVX2 GEMM + post-attn norm + residual
                q4_gemm_l1_2x4[DType.float16, DType.float32](
                    weights.wo_packed, weights.wo_scales, canvas_attn_out, canvas_proj_attn,
                    active_len, q_size, CONFIG.embed_dim, num_tasks,
                    in_stride=MAX_Q_SIZE, out_stride=CONFIG.embed_dim
                )
                rms_norm_batch[in_type=DType.float32, out_type=DType.float32](canvas_proj_attn, weights.post_attention_layernorm_weight, canvas_proj_attn, active_len, CONFIG.embed_dim)
                var f16_min_v = SIMD[DType.float32, 8](-65500.0)
                var f16_max_v = SIMD[DType.float32, 8](65500.0)
                for b in range(active_len):
                    var hidden = canvas_hidden + b * CONFIG.embed_dim
                    var proj = canvas_proj_attn + b * CONFIG.embed_dim
                    for i in range(0, CONFIG.embed_dim, 8):
                        var val = (hidden.load[8](i).cast[DType.float32]() + proj.load[8](i)).clamp(f16_min_v, f16_max_v)
                        hidden.store[8](i, val.cast[DType.float16]())

                # Phase 2c + 2d: Shared MLP + MoE Experts (overlapped across iGPU & CPU when use_gpu=True)
                var tl_ffn = std.time.perf_counter()
                rms_norm_batch[in_type=DType.float16, out_type=DType.float32](canvas_hidden, weights.pre_feedforward_layernorm_weight, canvas_norm, active_len, CONFIG.embed_dim)
                var slot_gate = Int(0)
                if use_gpu:
                    # Enqueue Shared MLP w_gate on iGPU (~28 ms) while CPU executes MoE + w_up!
                    gpu_ctx.flush_cache_range(canvas_norm.__int__(), active_len * CONFIG.embed_dim * 4)
                    slot_gate = gpu_ctx.enqueue_matmul(
                        weights.w_gate_packed.__int__(),
                        weights.w_gate_scales.__int__(),
                        canvas_norm.__int__(),
                        temp_packed_q.__int__(),
                        CONFIG.embed_dim,
                        CONFIG.standard_mlp_dim,
                        batch_size=active_len,
                        flush_cache=False,
                        ring_doorbell_now=True,
                    )
                else:
                    q4_fused_gate_up_geglu_l1[DType.float32, DType.float16](
                        weights.w_gate_packed, weights.w_gate_scales,
                        weights.w_up_packed, weights.w_up_scales,
                        canvas_norm, canvas_ffn_gated,
                        active_len, CONFIG.embed_dim, CONFIG.standard_mlp_dim, num_tasks,
                    )
                    q4_gemm_l1_2x4[DType.float16, DType.float16](
                        weights.w_down_packed, weights.w_down_scales,
                        canvas_ffn_gated, canvas_ffn_down,
                        active_len, CONFIG.standard_mlp_dim, CONFIG.embed_dim, num_tasks,
                        in_stride=CONFIG.standard_mlp_dim, out_stride=CONFIG.embed_dim,
                    )
                    rms_norm_batch[in_type=DType.float16, out_type=DType.float16](canvas_ffn_down, weights.post_feedforward_layernorm_1_weight, canvas_shared_norm, active_len, CONFIG.embed_dim)
                var tl_ffn_end = std.time.perf_counter()

                # Phase 2d: MoE path (Dual-Space Deinterleaved F32 single-barrier parallel expert dispatch)
                var tl_moe = std.time.perf_counter()
                rms_norm_batch[in_type=DType.float16, out_type=DType.float16](canvas_hidden, weights.pre_feedforward_layernorm_2_weight, canvas_moe_in, active_len, CONFIG.embed_dim)
                var r_scale_factor = Float32(1.0 / sqrt(2816.0))

                def router_norm_worker(b: Int) capturing -> None:
                    var hidden_inout = canvas_hidden + b * CONFIG.embed_dim
                    var scaled_norm_t = canvas_scaled_norm + b * CONFIG.embed_dim
                    var routed_norm_t = canvas_routed_norm + b * CONFIG.embed_dim

                    rms_norm_noscale[in_type=DType.float16, out_type=DType.float32](hidden_inout, routed_norm_t, CONFIG.embed_dim)
                    var r_scale_vec = SIMD[DType.float32, 8](r_scale_factor)
                    for j in range(0, CONFIG.embed_dim, 8):
                        scaled_norm_t.store[8](j, routed_norm_t.load[8](j) * weights.router_scale.load[8](j) * r_scale_vec)

                parallelize[router_norm_worker](active_len)

                q4_gemm_l1_2x4[DType.float32, DType.float32](
                    weights.w_router_packed, weights.w_router_scales,
                    canvas_scaled_norm, canvas_router_scores,
                    active_len, CONFIG.embed_dim, CONFIG.num_experts, num_tasks,
                )

                def router_topk_worker(b: Int) capturing -> None:
                    var scores_t = canvas_router_scores + b * CONFIG.num_experts
                    var top_idx_t = canvas_top_idx + b * CONFIG.top_k_experts
                    var top_val_t = canvas_top_val + b * CONFIG.top_k_experts

                    softmax(scores_t, CONFIG.num_experts)
                    get_top_k[8](scores_t, top_idx_t, top_val_t, CONFIG.num_experts)

                parallelize[router_topk_worker](active_len)

                apply_moe_lod_and_set_cover(
                    canvas_top_idx,
                    canvas_top_val,
                    weights.router_per_expert_scale,
                    canvas_top_weights,
                    active_len,
                    CONFIG.top_k_experts,
                    active_top_k,
                    moe_top_p,
                )

                run_diffusion_moe_experts_parallel(
                    weights.experts_gate_up_packed,
                    weights.experts_gate_up_scales,
                    weights.experts_down_packed,
                    weights.experts_down_scales,
                    canvas_moe_in,
                    canvas_top_idx,
                    canvas_top_weights,
                    canvas_moe_out,
                    active_len,
                    CONFIG.embed_dim,
                    CONFIG.expert_mlp_dim,
                    CONFIG.top_k_experts,
                    num_tasks,
                    moe_deinterleaved_x,
                    moe_work_buf,
                    moe_expert_outs,
                )
                var tl_moe_end = std.time.perf_counter()

                if use_gpu:
                    var tl_ffn2 = std.time.perf_counter()
                    q4_gemm_l1_2x4[DType.float32, DType.float16](
                        weights.w_up_packed, weights.w_up_scales,
                        canvas_norm, canvas_ffn_up,
                        active_len, CONFIG.embed_dim, CONFIG.standard_mlp_dim, num_tasks,
                        in_stride=CONFIG.embed_dim, out_stride=CONFIG.standard_mlp_dim,
                    )
                    gpu_ctx.wait_matmul(slot_gate, temp_packed_q.__int__(), CONFIG.standard_mlp_dim, batch_size=active_len, flush_cache=True)
                    for b in range(active_len):
                        var g_ptr = temp_packed_q + b * CONFIG.standard_mlp_dim
                        var u_ptr = canvas_ffn_up + b * CONFIG.standard_mlp_dim
                        var out_ptr = canvas_ffn_gated + b * CONFIG.standard_mlp_dim
                        for i in range(0, CONFIG.standard_mlp_dim, 8):
                            var g_v = g_ptr.load[8](i)
                            var u_v = u_ptr.load[8](i).cast[DType.float32]()
                            var sig = 1.0 / (1.0 + exp(SIMD[DType.float32, 8](-1.702) * g_v))
                            var act = (g_v * sig * u_v).clamp(f16_min_v, f16_max_v)
                            out_ptr.store[8](i, act.cast[DType.float16]())
                    q4_gemm_l1_2x4[DType.float16, DType.float16](
                        weights.w_down_packed, weights.w_down_scales,
                        canvas_ffn_gated, canvas_ffn_down,
                        active_len, CONFIG.standard_mlp_dim, CONFIG.embed_dim, num_tasks,
                        in_stride=CONFIG.standard_mlp_dim, out_stride=CONFIG.embed_dim,
                    )
                    rms_norm_batch[in_type=DType.float16, out_type=DType.float16](canvas_ffn_down, weights.post_feedforward_layernorm_1_weight, canvas_shared_norm, active_len, CONFIG.embed_dim)
                    tl_ffn_end += std.time.perf_counter() - tl_ffn2

                # Phase 2e: Batched combine + final norm + residual + scalar
                rms_norm_batch[in_type=DType.float16, out_type=DType.float16](canvas_moe_out, weights.post_feedforward_layernorm_2_weight, canvas_moe_norm, active_len, CONFIG.embed_dim)
                var scalar = weights.layer_scalar[0]
                for b in range(active_len):
                    var hidden = canvas_hidden + b * CONFIG.embed_dim
                    var shared_n = canvas_shared_norm + b * CONFIG.embed_dim
                    var moe_n = canvas_moe_norm + b * CONFIG.embed_dim
                    var combined = canvas_combined + b * CONFIG.embed_dim
                    var ffn = canvas_ffn_out + b * CONFIG.embed_dim
                    for i in range(0, CONFIG.embed_dim, 8):
                        var comb_f32 = (shared_n.load[8](i).cast[DType.float32]() + moe_n.load[8](i).cast[DType.float32]()).clamp(f16_min_v, f16_max_v)
                        combined.store[8](i, comb_f32.cast[DType.float16]())
                    rms_norm[in_type=DType.float16, out_type=DType.float16](combined, weights.post_feedforward_layernorm_weight, ffn, CONFIG.embed_dim)
                    var scalar_vec = SIMD[DType.float32, 8](scalar)
                    for i in range(0, CONFIG.embed_dim, 8):
                        var h_val = hidden.load[8](i).cast[DType.float32]()
                        var ffn_val = ffn.load[8](i).cast[DType.float32]()
                        var out_h = ((h_val + ffn_val) * scalar_vec).clamp(f16_min_v, f16_max_v)
                        hidden.store[8](i, out_h.cast[DType.float16]())

                if l == 0 or l == 15 or l == 29:
                    print("    layer ", l, " qkv=", String((tl_qkv_end - tl_qkv) * 1000), "ms",
                          " attn=", String((tl_attn_end - tl_attn) * 1000), "ms",
                          " ffn=", String((tl_ffn_end - tl_ffn) * 1000), "ms",
                          " moe=", String((tl_moe_end - tl_moe) * 1000), "ms")

                # AdaCache (ICCV 2025) + dLLM-Cache per-token partial canvas updates after Layer 1 (llm-mojo-2tc.8)
                if l == 1:
                    var prev_active_len = active_len
                    active_len = adacache_filter_after_layer1(
                        canvas_hidden,
                        adacache_h1_prev,
                        active_canvas_idx,
                        adacache_skip_streak,
                        active_len,
                        CONFIG.embed_dim,
                        denoise_step,
                        k_slow=3,
                        tau_ada=0.08,
                    )
                    if active_len < prev_active_len:
                        print("    -> AdaCache + MoReg: compacted active_len=", active_len, "/", prev_active_len, "(layers 2..29 + logit_proj)")

                # TeaCache (CVPR 2025) & TaylorSeer (ICCV 2025) middle-block forecasting (llm-mojo-2tc.4 & 2tc.7)
                if l == 5:
                    skip_mid_layers = teacache_eval_gate(
                        canvas_hidden,
                        teacache_h5_prev,
                        active_canvas_idx,
                        active_len,
                        CONFIG.embed_dim,
                        temp,
                        full_steps_count,
                        consec_skips,
                        teacache_accum,
                        teacache_thresh=0.72,
                        max_consec_skips=2,
                    )
                    if skip_mid_layers:
                        consec_skips += 1
                        taylorseer_forecast_skipped_step(
                            canvas_hidden,
                            teacache_h5_prev,
                            active_canvas_idx,
                            taylor_D,
                            taylor_d1,
                            taylor_d2,
                            active_len,
                            CONFIG.embed_dim,
                            full_steps_count,
                            consec_skips,
                        )
                        print("    -> TeaCache + TaylorSeer: forecasted layers 6..22 (m=", consec_skips, " accum=", String(teacache_accum[0]), ")")
                    else:
                        for b in range(active_len):
                            var s_ptr = canvas_hidden + b * CONFIG.embed_dim
                            var d_ptr = taylor_h5_curr + b * CONFIG.embed_dim
                            for i in range(0, CONFIG.embed_dim, 8):
                                d_ptr.store[8](i, s_ptr.load[8](i))
                elif l == 22 and not skip_mid_layers:
                    taylorseer_record_full_step(
                        taylor_h5_curr,
                        canvas_hidden,
                        active_canvas_idx,
                        taylor_D,
                        taylor_d1,
                        taylor_d2,
                        active_len,
                        CONFIG.embed_dim,
                        full_steps_count,
                    )
                    full_steps_count += 1
                    consec_skips = 0

            # --- Sampler step (Sparse Top-32 reservoir + soft-embed on active_len tokens, llm-mojo-2tc.1 & 2tc.3) ---
            var t_samp_start = std.time.perf_counter()

            for t in range(CANVAS_LEN):
                rng_state ^= rng_state >> 12
                rng_state ^= rng_state << 25
                rng_state ^= rng_state >> 27
                var rnd_u64 = rng_state * 0x2545F4914F6CDD1D
                rand_buf[t] = Float32(rnd_u64 & 0xFFFFFF) / Float32(16777216.0)

            # Batched logit projection: read embed weights ONCE for active_len unconverged tokens
            rms_norm_batch[in_type=DType.float16, out_type=DType.float32](canvas_hidden, final_norm_weight, canvas_final_norm, active_len, CONFIG.embed_dim)
            q4_gemm_l1_2x4[DType.float32, DType.float32](
                embed.packed_weights, embed.scales, canvas_final_norm, canvas_logits,
                active_len, CONFIG.embed_dim, CONFIG.vocab_size, num_tasks,
                CONFIG.embed_dim, CONFIG.vocab_size
            )
            var t_logit_end = std.time.perf_counter()

            # Parallelized per-token Top-32 softcap, cumulant entropy, sample, sparse sc_embed
            if active_len == CANVAS_LEN:
                sample_and_soft_embed_top32_batch[32](
                    canvas_logits,
                    rand_buf,
                    embed.packed_weights,
                    embed.scales,
                    token_entropy,
                    argmax_tokens,
                    new_tokens,
                    sc_embeds,
                    CANVAS_LEN,
                    CONFIG.vocab_size,
                    CONFIG.embed_dim,
                    temp,
                    prompt_scale,
                    num_tasks,
                )
            else:
                for b in range(active_len):
                    compact_rand_buf[b] = rand_buf[active_canvas_idx[b]]
                sample_and_soft_embed_top32_batch[32](
                    canvas_logits,
                    compact_rand_buf,
                    embed.packed_weights,
                    embed.scales,
                    compact_entropy,
                    compact_argmax,
                    compact_new_tokens,
                    sc_compact,
                    active_len,
                    CONFIG.vocab_size,
                    CONFIG.embed_dim,
                    temp,
                    prompt_scale,
                    num_tasks,
                )
                scatter_active_sampler_outputs[DType.float32](
                    active_canvas_idx,
                    compact_entropy,
                    compact_argmax,
                    compact_new_tokens,
                    sc_compact,
                    token_entropy,
                    argmax_tokens,
                    new_tokens,
                    sc_embeds,
                    locked_mask,
                    canvas_tokens,
                    active_len,
                    CANVAS_LEN,
                    CONFIG.embed_dim,
                )
            var t_samp_end = std.time.perf_counter()
            print("    sc_mlp=", String((t_sc_end - t_sc_start) * 1000.0), "ms",
                  " logit_proj=", String((t_logit_end - t_samp_start) * 1000.0), "ms",
                  " top32_sampler=", String((t_samp_end - t_logit_end) * 1000.0), "ms")

            # Mean entropy + confidence
            var mean_ent: Float32 = 0.0
            for t in range(CANVAS_LEN):
                mean_ent += token_entropy[t]
            mean_ent /= Float32(CANVAS_LEN)
            var confident = mean_ent < confidence_threshold
            print("    mean_entropy=", String(mean_ent), " temp=", String(temp),
                  " confident=", confident)

            # Entropy-bound acceptance mask: sort by ascending entropy, cumsum, cummax
            # Selection sort for 256 elements
            for t in range(CANVAS_LEN):
                sorted_idx[t] = t
                sorted_ent[t] = token_entropy[t]
            for i in range(CANVAS_LEN):
                var min_idx = i
                for j in range(i + 1, CANVAS_LEN):
                    if sorted_ent[j] < sorted_ent[min_idx]:
                        min_idx = j
                if min_idx != i:
                    var tmp_e = sorted_ent[i]
                    sorted_ent[i] = sorted_ent[min_idx]
                    sorted_ent[min_idx] = tmp_e
                    var tmp_i = sorted_idx[i]
                    sorted_idx[i] = sorted_idx[min_idx]
                    sorted_idx[min_idx] = tmp_i

            # cumsum and cummax, accept where (cumsum - cummax) <= entropy_bound
            var cumsum_ent: Float32 = 0.0
            var cummax_ent: Float32 = 0.0
            for t in range(CANVAS_LEN):
                eb_mask[t] = locked_mask[t]
            for i in range(CANVAS_LEN):
                cumsum_ent += sorted_ent[i]
                if sorted_ent[i] > cummax_ent:
                    cummax_ent = sorted_ent[i]
                if (cumsum_ent - cummax_ent) <= entropy_bound:
                    var orig_idx = sorted_idx[i]
                    eb_mask[orig_idx] = True

            # Update history: write argmax_tokens at slot hist_len % ST
            var write_pos = hist_len % 2
            for t in range(CANVAS_LEN):
                history[write_pos * CANVAS_LEN + t] = argmax_tokens[t]
            hist_len += 1

            # Update Fast-dLLM / dKV-Cache locked_mask for subsequent steps (llm-mojo-2tc.3)
            var num_locked = update_locked_mask(
                eb_mask, history, token_entropy, locked_mask, CANVAS_LEN, hist_len
            )
            for t in range(CANVAS_LEN):
                if locked_mask[t]:
                    eb_mask[t] = True

            # Stability check: all ST slots agree
            var stable = True
            if hist_len >= 2:
                for t in range(CANVAS_LEN):
                    if history[0 * CANVAS_LEN + t] != history[1 * CANVAS_LEN + t]:
                        stable = False
                        break
            else:
                stable = False

            # Convergence
            var converged = (stable and confident and hist_len >= 2) or (denoise_step + 1 >= MAX_DENOISE_STEPS)
            print("    stable=", stable, " converged=", converged, " locked=", num_locked, "/", CANVAS_LEN)

            # Update canvas: accept → new_tokens, reject → random re-noise
            var n_accept = 0
            for t in range(CANVAS_LEN):
                if eb_mask[t]:
                    n_accept += 1
            var n_reject = CANVAS_LEN - n_accept
            if n_reject > 0:
                for t in range(CANVAS_LEN):
                    if eb_mask[t]:
                        canvas_tokens[t] = new_tokens[t]
                    else:
                        rng_state ^= rng_state >> 12
                        rng_state ^= rng_state << 25
                        rng_state ^= rng_state >> 27
                        var rnd_u64 = rng_state * 0x2545F4914F6CDD1D
                        canvas_tokens[t] = Int(rnd_u64 % UInt64(CONFIG.vocab_size))
            else:
                for t in range(CANVAS_LEN):
                    canvas_tokens[t] = new_tokens[t]
            print("    accepted=", n_accept, "/", CANVAS_LEN)

            # Calculate local change metrics for adaptive step skipping
            var num_flips = 0
            var entropy_decay: Float32 = 0.0
            if steps_executed > 1:
                for t in range(CANVAS_LEN):
                    if argmax_tokens[t] != argmax_canvas[t]:
                        num_flips += 1
                entropy_decay = prev_mean_ent - mean_ent

            var skip_amount = 1
            if steps_executed > 1:
                # If very few tokens changed or entropy decay has flattened out, skip intermediate steps
                if num_flips <= 2 or (entropy_decay >= 0.0 and entropy_decay < 0.003):
                    skip_amount = 3
                    print("    -> Adaptive skip: num_flips =", num_flips, "entropy_decay =", String(entropy_decay), "| Skipping 2 steps")
                elif num_flips <= 8 or (entropy_decay >= 0.0 and entropy_decay < 0.008):
                    skip_amount = 2
                    print("    -> Adaptive skip: num_flips =", num_flips, "entropy_decay =", String(entropy_decay), "| Skipping 1 step")

            # Store metrics for next step
            prev_mean_ent = mean_ent

            # Update argmax_canvas
            for t in range(CANVAS_LEN):
                argmax_canvas[t] = argmax_tokens[t]

            # If converged: snap canvas to argmax and break
            if converged:
                for t in range(CANVAS_LEN):
                    canvas_tokens[t] = argmax_canvas[t]
                break

            # Zero out sc_embeds for positions that will commit (none in denoise-only mode)
            # All positions continue denoising, so sc_embeds stays

            denoise_step += skip_amount

        var t_end = std.time.perf_counter()
        var denoise_duration = t_end - t_decode_start

        print("\n---------------------------------------------------------")
        print("Diffusion generation complete!")
        print("  Prompt tokens:    ", num_prompt_tokens)
        print("  Canvas length:    ", CANVAS_LEN)
        print("  Denoise steps:    ", steps_executed)
        print("  Total denoise time: ", String(denoise_duration * 1000.0), " ms")
        print("  Avg per step:     ", String(denoise_duration * 1000.0 / Float64(steps_executed)), " ms")
        print("  Output:")
        if bench_mode:
            print("    [bench first 8 argmax token IDs]:", argmax_canvas[0], argmax_canvas[1], argmax_canvas[2], argmax_canvas[3], argmax_canvas[4], argmax_canvas[5], argmax_canvas[6], argmax_canvas[7])
        else:
            var output_list = Python.list()
            for t in range(CANVAS_LEN):
                output_list.append(argmax_canvas[t])
            var decoded_output = tokenizer.decode(output_list)
            py_builtins.print("    ", decoded_output)
        print("---------------------------------------------------------")

    # Cleanup memory
    gpu_ctx.free_gpu_buffer(weights_gpu_buf)
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
    gpu_ctx.free_gpu_buffer(canvas_hidden_buf)
    gpu_ctx.free_gpu_buffer(canvas_q_buf)
    gpu_ctx.free_gpu_buffer(canvas_k_buf)
    gpu_ctx.free_gpu_buffer(canvas_v_buf)
    gpu_ctx.free_gpu_buffer(canvas_attn_out_buf)
    gpu_ctx.free_gpu_buffer(canvas_proj_attn_buf)
    gpu_ctx.free_gpu_buffer(canvas_norm_buf)
    gpu_ctx.free_gpu_buffer(temp_packed_q_buf)
    gpu_ctx.free_gpu_buffer(temp_packed_k_buf)
    gpu_ctx.free_gpu_buffer(temp_packed_v_buf)
    canvas_ffn_gate.free()
    canvas_ffn_up.free()
    canvas_ffn_gated.free()
    canvas_ffn_down.free()
    canvas_shared_norm.free()
    canvas_moe_in.free()
    canvas_moe_out.free()
    canvas_moe_norm.free()
    canvas_combined.free()
    canvas_ffn_out.free()
    canvas_attn_scores.free()
    moe_deinterleaved_x.free()
    moe_work_buf.free()
    moe_expert_outs.free()
    canvas_router_scores.free()
    canvas_top_idx.free()
    canvas_top_val.free()
    canvas_top_weights.free()
    canvas_scaled_norm.free()
    canvas_routed_norm.free()
    canvas_tokens.free()
    argmax_canvas.free()
    sc_embeds.free()
    sc_buf_normed.free()
    sc_buf_gated.free()
    sc_buf_down.free()
    canvas_final_norm.free()
    canvas_logits.free()
    rand_buf.free()
    token_entropy.free()
    new_tokens.free()
    argmax_tokens.free()
    sorted_idx.free()
    sorted_ent.free()
    eb_mask.free()
    history.free()
    locked_mask.free()
    active_canvas_idx.free()
    sc_compact.free()
    compact_rand_buf.free()
    compact_entropy.free()
    compact_argmax.free()
    compact_new_tokens.free()
    teacache_h5_prev.free()
    taylor_h5_curr.free()
    taylor_D.free()
    taylor_d1.free()
    taylor_d2.free()
    teacache_accum.free()
    adacache_h1_prev.free()
    adacache_skip_streak.free()
    gpu_ctx.close()
