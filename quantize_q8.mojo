from std.sys.info import num_physical_cores
from std.memory import alloc
from std.testing import assert_equal
from std.python import Python
from std.algorithm import parallelize
from std.math import min, max

# -----------------------------------------------------------------------------
# Transformer Config & Sizing Math
# -----------------------------------------------------------------------------
struct TransformerConfig:
    comptime embed_dim = 3840
    comptime head_dim = 256
    comptime num_heads = 16
    comptime num_kv_heads = 8
    comptime intermediate_dim = 15360
    comptime num_layers = 48
    comptime vocab_size = 262144

    def __init__(out self):
        pass

    @always_inline
    def is_global_layer(self, layer_idx: Int) -> Bool:
        return layer_idx % 6 == 5

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
        
        # Packed Q8 weights: excluding w2
        var packed_weights_size = wq_size + wk_size + wv_size + wo_size + w1_size + w3_size
        
        # Scale factors (1 per block of 32 elements): excluding w2
        var scales_count = packed_weights_size // 32
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
    def get_quantized_layer_aligned_size(self, layer_idx: Int) -> Int:
        var base_offset = self.get_w2_offset_in_layer(layer_idx)
        var w2_blocks = Self.intermediate_dim // 32
        var w2_size = w2_blocks * 139264
        return base_offset + w2_size

# -----------------------------------------------------------------------------
# Quantization and Packing Kernel (Block Size 32)
# -----------------------------------------------------------------------------
@always_inline
def quantize_and_pack_matrix(
    src_ptr: UnsafePointer[Float32, _],
    dest_packed_ptr: UnsafePointer[Int8, MutExternalOrigin],
    dest_scales_ptr: UnsafePointer[Float32, MutExternalOrigin],
    c_in: Int,
    c_out: Int,
    num_tasks: Int,
):
    var chunk_size = c_out // num_tasks
    var c_in_packed = c_in
    var c_in_blocks = c_in // 32
    
    def worker(t: Int) capturing -> None:
        var start_r = t * chunk_size
        var end_r = (t + 1) * chunk_size if t < num_tasks - 1 else c_out
        
        for r in range(start_r, end_r):
            var r_offset_src = r * c_in
            var r_offset_dest = r * c_in_packed
            var r_offset_scales = r * c_in_blocks
            
            for b in range(c_in_blocks):
                var block_start = r_offset_src + b * 32
                var block_dest = r_offset_dest + b * 32
                var block_scale_idx = r_offset_scales + b
                
                # Compute max absolute value of the block
                var max_abs: Float32 = 0.0
                for i in range(32):
                    var v = abs(src_ptr[block_start + i])
                    if v > max_abs:
                        max_abs = v
                
                var scale = max_abs / 127.0
                if scale < 1e-5:
                    scale = 1e-5
                
                dest_scales_ptr[block_scale_idx] = scale
                
                var inv_scale = 1.0 / scale
                for i in range(32):
                    var v = src_ptr[block_start + i]
                    
                    # Quantize to [-127, 127]
                    var q = Int(max(Float32(-127.0), min(Float32(127.0), Float32(Int(v * inv_scale + 0.5 if v * inv_scale >= 0.0 else v * inv_scale - 0.5)))))
                    
                    dest_packed_ptr[block_dest + i] = Int8(q)
                    
    parallelize[worker](num_tasks)

@always_inline
def quantize_and_pack_w2_block_column(
    src_ptr: UnsafePointer[Float32, _],
    dest_ptr: UnsafePointer[Int8, MutExternalOrigin],
    c_in: Int,  # intermediate_dim (15360)
    c_out: Int, # embed_dim (3840)
    num_tasks: Int,
):
    var num_blocks = c_in // 32 # 480
    var chunk_size = num_blocks // num_tasks
    if chunk_size == 0:
        chunk_size = 1
        
    def worker(t: Int) capturing -> None:
        var start_b = t * chunk_size
        var end_b = (t + 1) * chunk_size if t < num_tasks - 1 else num_blocks
        if start_b >= num_blocks:
            return
            
        for b in range(start_b, end_b):
            # Destination block b offset: b * 139264
            var block_dest_ptr = dest_ptr + b * 139264
            var block_scales_ptr = (block_dest_ptr + 122880).bitcast[Float32]()
            
            for r in range(c_out):
                var row_offset_src = r * c_in + b * 32
                
                # Find max absolute value of these 32 elements
                var max_abs: Float32 = 0.0
                for c in range(32):
                    var v = abs(src_ptr[row_offset_src + c])
                    if v > max_abs:
                        max_abs = v
                        
                var scale = max_abs / 127.0
                if scale < 1e-5:
                    scale = 1e-5
                    
                block_scales_ptr[r] = scale
                
                var inv_scale = 1.0 / scale
                var row_dest_offset = r * 32
                for c in range(32):
                    var v = src_ptr[row_offset_src + c]
                    var q = Int(max(Float32(-127.0), min(Float32(127.0), Float32(Int(v * inv_scale + 0.5 if v * inv_scale >= 0.0 else v * inv_scale - 0.5)))))
                    block_dest_ptr[row_dest_offset + c] = Int8(q)
                    
            # Set the remaining 256 dummy Float32 zeros for scale padding
            for p in range(256):
                block_scales_ptr[c_out + p] = 0.0
                
    parallelize[worker](num_tasks)

# -----------------------------------------------------------------------------
# Main Script Entry
# -----------------------------------------------------------------------------
def main() raises:
    var num_tasks = num_physical_cores()
    var CONFIG = TransformerConfig()
    
    print("=========================================================")
    print("      GEMMA 4 12B PARALLEL Q8_0 WEIGHT QUANTIZER        ")
    print("=========================================================")
    print("Target Cores: ", num_tasks)
    print("Input File:   gemma4_weights.bin")
    print("Output File:  gemma4_weights_q8.bin")
    print("---------------------------------------------------------")
    
    var builtins = Python.import_module("builtins")
    var np = Python.import_module("numpy")
    
    var f_in = builtins.open("gemma4_weights.bin", "rb")
    var f_out = builtins.open("gemma4_weights_q8.bin", "wb")
    
    for i in range(CONFIG.num_layers):
        var is_global = CONFIG.is_global_layer(i)
        var layer_heads = 16
        var layer_head_dim = 512 if is_global else 256
        var layer_kv_heads = 1 if is_global else 8
        
        var wq_size = layer_heads * layer_head_dim * CONFIG.embed_dim
        var wk_size = layer_kv_heads * layer_head_dim * CONFIG.embed_dim
        var wv_size = 0 if is_global else (layer_kv_heads * layer_head_dim * CONFIG.embed_dim)
        var wo_size = CONFIG.embed_dim * layer_heads * layer_head_dim
        var w1_size = CONFIG.intermediate_dim * CONFIG.embed_dim
        var w2_size = CONFIG.embed_dim * CONFIG.intermediate_dim
        var w3_size = CONFIG.intermediate_dim * CONFIG.embed_dim
        
        var input_layernorm_size = CONFIG.embed_dim
        var post_attention_layernorm_size = CONFIG.embed_dim
        var pre_feedforward_layernorm_size = CONFIG.embed_dim
        var post_feedforward_layernorm_size = CONFIG.embed_dim
        var q_norm_size = layer_head_dim
        var k_norm_size = layer_head_dim
        var layer_scalar_size = 1
        
        # Calculate raw floats in unquantized layer
        var total_floats = (
            wq_size + wk_size + wv_size + wo_size +
            w1_size + w2_size + w3_size +
            input_layernorm_size + post_attention_layernorm_size +
            pre_feedforward_layernorm_size + post_feedforward_layernorm_size +
            q_norm_size + k_norm_size + layer_scalar_size
        )
        
        var aligned_bytes_size = CONFIG.get_layer_aligned_size(i)
        
        print("Layer ", i, "/", CONFIG.num_layers, " (", "global" if is_global else "sliding", ")...", end="")
        
        # Read the entire aligned layer block as Python bytes
        var raw_bytes = f_in.read(aligned_bytes_size)
        
        # Convert first total_floats of it to NumPy float32 array
        var layer_np = np.frombuffer(raw_bytes, dtype=np.float32, count=total_floats)
        
        # Get actual memory address of this Float32 array
        var addr = Int(py=layer_np.__array_interface__["data"][0])
        var base_ptr = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=addr)
        
        # Get sub-pointers for dense matrices
        var wq_ptr = base_ptr
        var wk_ptr = wq_ptr + wq_size
        var wv_ptr = wk_ptr
        var wo_ptr = wk_ptr + wk_size
        if not is_global:
            wv_ptr = wk_ptr + wk_size
            wo_ptr = wv_ptr + wv_size
            
        var w1_ptr = wo_ptr + wo_size
        var w2_ptr = w1_ptr + w1_size
        var w3_ptr = w2_ptr + w2_size
        
        # Get sub-pointer to full-precision weights (unchanged)
        var unchanged_floats = (
            input_layernorm_size + post_attention_layernorm_size +
            pre_feedforward_layernorm_size + post_feedforward_layernorm_size +
            q_norm_size + k_norm_size + layer_scalar_size
        )
        
        # 1. Allocate quantized payload arrays in Python/NumPy to manage memory
        var total_packed_bytes_stable = wq_size + wk_size + wv_size + wo_size + w1_size + w3_size
        var packed_stable_np = np.zeros(total_packed_bytes_stable, dtype=np.int8)
        
        var total_scales_stable = total_packed_bytes_stable // 32
        var scales_stable_np = np.zeros(total_scales_stable, dtype=np.float32)
        
        var w2_blocks = CONFIG.intermediate_dim // 32
        var w2_size_padded = w2_blocks * 139264
        var w2_padded_np = np.zeros(w2_size_padded, dtype=np.int8)
        
        # Get UnsafePointers to the packed array, scales array, and w2 padded array
        var packed_stable_ptr = UnsafePointer[Int8, MutExternalOrigin](unsafe_from_address=Int(py=packed_stable_np.__array_interface__["data"][0]))
        var scales_stable_ptr = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(py=scales_stable_np.__array_interface__["data"][0]))
        var w2_padded_ptr = UnsafePointer[Int8, MutExternalOrigin](unsafe_from_address=Int(py=w2_padded_np.__array_interface__["data"][0]))
        
        # Set up sub-pointers for target packed arrays
        var wq_packed_ptr = packed_stable_ptr
        var wk_packed_ptr = wq_packed_ptr + wq_size
        var wv_packed_ptr = wk_packed_ptr + wk_size
        var wo_packed_ptr = wk_packed_ptr + wk_size
        if not is_global:
            wv_packed_ptr = wk_packed_ptr + wk_size
            wo_packed_ptr = wv_packed_ptr + wv_size
            
        var w1_packed_ptr = wo_packed_ptr + wo_size
        var w3_packed_ptr = w1_packed_ptr + w1_size
        
        # Set up sub-pointers for target scales arrays
        var wq_scales_ptr = scales_stable_ptr
        var wk_scales_ptr = wq_scales_ptr + (wq_size // 32)
        var wv_scales_ptr = wk_scales_ptr + (wk_size // 32)
        var wo_scales_ptr = wk_scales_ptr + (wk_size // 32)
        if not is_global:
            wv_scales_ptr = wk_scales_ptr + (wk_size // 32)
            wo_scales_ptr = wv_scales_ptr + (wv_size // 32)
            
        var w1_scales_ptr = wo_scales_ptr + (wo_size // 32)
        var w3_scales_ptr = w1_scales_ptr + (w1_size // 32)
        
        # 2. Parallel Quantize & Pack each dense matrix
        quantize_and_pack_matrix(wq_ptr, wq_packed_ptr, wq_scales_ptr, CONFIG.embed_dim, layer_heads * layer_head_dim, num_tasks)
        quantize_and_pack_matrix(wk_ptr, wk_packed_ptr, wk_scales_ptr, CONFIG.embed_dim, layer_kv_heads * layer_head_dim, num_tasks)
        if not is_global:
            quantize_and_pack_matrix(wv_ptr, wv_packed_ptr, wv_scales_ptr, CONFIG.embed_dim, layer_kv_heads * layer_head_dim, num_tasks)
        quantize_and_pack_matrix(wo_ptr, wo_packed_ptr, wo_scales_ptr, layer_heads * layer_head_dim, CONFIG.embed_dim, num_tasks)
        quantize_and_pack_matrix(w1_ptr, w1_packed_ptr, w1_scales_ptr, CONFIG.embed_dim, CONFIG.intermediate_dim, num_tasks)
        quantize_and_pack_matrix(w3_ptr, w3_packed_ptr, w3_scales_ptr, CONFIG.embed_dim, CONFIG.intermediate_dim, num_tasks)
        
        quantize_and_pack_w2_block_column(w2_ptr, w2_padded_ptr, CONFIG.intermediate_dim, CONFIG.embed_dim, num_tasks)
        
        # 3. Create unchanged float32 numpy slice to write out directly
        var unchanged_np = np.frombuffer(raw_bytes, dtype=np.float32, count=unchanged_floats, offset=(total_floats - unchanged_floats) * 4)
        
        # 4. Write stable portion
        _ = f_out.write(packed_stable_np.tobytes())
        _ = f_out.write(scales_stable_np.tobytes())
        _ = f_out.write(unchanged_np.tobytes())
        
        # 5. Handle aligned page padding for stable base portion
        var raw_stable_size = total_packed_bytes_stable + (total_scales_stable * 4) + (unchanged_floats * 4)
        var aligned_stable_size = CONFIG.get_w2_offset_in_layer(i)
        var stable_padding_needed = aligned_stable_size - raw_stable_size
        if stable_padding_needed > 0:
            var padding_bytes = np.zeros(stable_padding_needed, dtype=np.int8)
            _ = f_out.write(padding_bytes.tobytes())
            
        # 6. Write block-column padded w2 portion
        _ = f_out.write(w2_padded_np.tobytes())
        
        var aligned_out_size = CONFIG.get_quantized_layer_aligned_size(i)
        print(" Done. (Aligned Size: ", aligned_out_size, " bytes)")
        
    f_in.close()
    f_out.close()
    print("=========================================================")
    print(" QUANTIZATION COMPLETED SUCCESSFULLY!")
    print(" Quantized weights written to: gemma4_weights_q8.bin")
    print("=========================================================")
