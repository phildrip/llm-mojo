from std.sys.info import num_physical_cores
from std.memory import alloc
from std.testing import assert_equal
from std.python import Python
from std.algorithm import parallelize
from std.math import min, max, sqrt

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

# -----------------------------------------------------------------------------
# Quantization and Packing Kernel
# -----------------------------------------------------------------------------
@always_inline
def quantize_and_pack_matrix(
    src_ptr: UnsafePointer[Float32, _],
    dest_packed_ptr: UnsafePointer[UInt8, MutExternalOrigin],
    dest_scales_ptr: UnsafePointer[Float32, MutExternalOrigin],
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
            var r_offset_src = r * c_in
            var r_offset_dest = r * c_in_packed
            
            # TWN-style thresholding preserves the row's large-weight support
            # better than rounding every value against the all-weight absmean.
            var sum_abs: Float32 = 0.0
            var sum_sq: Float32 = 0.0
            for c in range(c_in):
                var val = src_ptr[r_offset_src + c]
                sum_abs += abs(val)
                sum_sq += val * val
            var mean_abs = sum_abs / Float32(c_in)
            var threshold = mean_abs * 0.7
            if threshold < 1e-5:
                threshold = 1e-5
            
            var selected_count = 0
            for c in range(c_in):
                var v_abs = abs(src_ptr[r_offset_src + c])
                if v_abs > threshold:
                    selected_count += 1
                    
            var scale = mean_abs
            if selected_count > 0:
                scale = sqrt(sum_sq / Float32(selected_count))
            if scale < 1e-5:
                scale = 1e-5
            dest_scales_ptr[r] = scale
            
            for c_packed in range(c_in_packed):
                var c_orig = c_packed * 4
                var v0 = src_ptr[r_offset_src + c_orig]
                var v1 = src_ptr[r_offset_src + c_orig + 1]
                var v2 = src_ptr[r_offset_src + c_orig + 2]
                var v3 = src_ptr[r_offset_src + c_orig + 3]
                
                var q0 = 2 if v0 > threshold else (0 if v0 < -threshold else 1)
                var q1 = 2 if v1 > threshold else (0 if v1 < -threshold else 1)
                var q2 = 2 if v2 > threshold else (0 if v2 < -threshold else 1)
                var q3 = 2 if v3 > threshold else (0 if v3 < -threshold else 1)
                
                dest_packed_ptr[r_offset_dest + c_packed] = UInt8(q0 | (q1 << 2) | (q2 << 4) | (q3 << 6))
                
    parallelize[worker](num_tasks)

# -----------------------------------------------------------------------------
# Main Script Entry
# -----------------------------------------------------------------------------
def main() raises:
    var num_tasks = num_physical_cores()
    var CONFIG = TransformerConfig()
    
    print("=========================================================")
    print("      GEMMA 4 12B PARALLEL TERNARY WEIGHT QUANTIZER       ")
    print("=========================================================")
    print("Target Cores: ", num_tasks)
    print("Input File:   gemma4_weights.bin")
    print("Output File:  gemma4_weights_ternary.bin")
    print("---------------------------------------------------------")
    
    var builtins = Python.import_module("builtins")
    var np = Python.import_module("numpy")
    
    var f_in = builtins.open("gemma4_weights.bin", "rb")
    var f_out = builtins.open("gemma4_weights_ternary.bin", "wb")
    
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
        var total_packed_bytes = (wq_size + wk_size + wv_size + wo_size + w1_size + w2_size + w3_size) // 4
        var packed_np = np.zeros(total_packed_bytes, dtype=np.uint8)
        
        var c_out_q = layer_heads * layer_head_dim
        var c_out_k = layer_kv_heads * layer_head_dim
        var c_out_v = 0 if is_global else (layer_kv_heads * layer_head_dim)
        var c_out_o = CONFIG.embed_dim
        var c_out_w1 = CONFIG.intermediate_dim
        var c_out_w2 = CONFIG.embed_dim
        var c_out_w3 = CONFIG.intermediate_dim
        
        var total_scales = c_out_q + c_out_k + c_out_v + c_out_o + c_out_w1 + c_out_w2 + c_out_w3
        var scales_np = np.zeros(total_scales, dtype=np.float32)
        
        # Get UnsafePointers to the packed array and scales array
        var packed_ptr = UnsafePointer[UInt8, MutExternalOrigin](unsafe_from_address=Int(py=packed_np.__array_interface__["data"][0]))
        var scales_ptr = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(py=scales_np.__array_interface__["data"][0]))
        
        # Set up sub-pointers for target packed arrays
        var wq_packed_ptr = packed_ptr
        var wk_packed_ptr = wq_packed_ptr + (wq_size // 4)
        var wv_packed_ptr = wk_packed_ptr + (wk_size // 4)
        var wo_packed_ptr = wk_packed_ptr + (wk_size // 4)
        if not is_global:
            wo_packed_ptr = wv_packed_ptr + (wv_size // 4)
            
        var w1_packed_ptr = wo_packed_ptr + (wo_size // 4)
        var w2_packed_ptr = w1_packed_ptr + (w1_size // 4)
        var w3_packed_ptr = w2_packed_ptr + (w2_size // 4)
        
        # Set up sub-pointers for target scales arrays
        var wq_scales_ptr = scales_ptr
        var wk_scales_ptr = wq_scales_ptr + c_out_q
        var wv_scales_ptr = wk_scales_ptr + c_out_k
        var wo_scales_ptr = wk_scales_ptr + c_out_k
        if not is_global:
            wo_scales_ptr = wv_scales_ptr + c_out_v
            
        var w1_scales_ptr = wo_scales_ptr + c_out_o
        var w2_scales_ptr = w1_scales_ptr + c_out_w1
        var w3_scales_ptr = w2_scales_ptr + c_out_w2
        
        # 2. Parallel Quantize & Pack each dense matrix
        quantize_and_pack_matrix(wq_ptr, wq_packed_ptr, wq_scales_ptr, CONFIG.embed_dim, c_out_q, num_tasks)
        quantize_and_pack_matrix(wk_ptr, wk_packed_ptr, wk_scales_ptr, CONFIG.embed_dim, c_out_k, num_tasks)
        if not is_global:
            quantize_and_pack_matrix(wv_ptr, wv_packed_ptr, wv_scales_ptr, CONFIG.embed_dim, c_out_v, num_tasks)
        quantize_and_pack_matrix(wo_ptr, wo_packed_ptr, wo_scales_ptr, layer_heads * layer_head_dim, c_out_o, num_tasks)
        quantize_and_pack_matrix(w1_ptr, w1_packed_ptr, w1_scales_ptr, CONFIG.embed_dim, c_out_w1, num_tasks)
        quantize_and_pack_matrix(w2_ptr, w2_packed_ptr, w2_scales_ptr, CONFIG.intermediate_dim, c_out_w2, num_tasks)
        quantize_and_pack_matrix(w3_ptr, w3_packed_ptr, w3_scales_ptr, CONFIG.embed_dim, c_out_w3, num_tasks)
        
        # 3. Create unchanged float32 numpy slice to write out directly
        var unchanged_np = np.frombuffer(raw_bytes, dtype=np.float32, count=unchanged_floats, offset=(total_floats - unchanged_floats) * 4)
        
        # 4. Write everything to the destination file
        f_out.write(packed_np.tobytes())
        f_out.write(scales_np.tobytes())
        f_out.write(unchanged_np.tobytes())
        
        # 5. Handle aligned page padding
        var raw_out_size = total_packed_bytes + (total_scales * 4) + (unchanged_floats * 4)
        var aligned_out_size = CONFIG.get_quantized_layer_aligned_size(i)
        var padding_needed = aligned_out_size - raw_out_size
        if padding_needed > 0:
            var padding_bytes = np.zeros(padding_needed, dtype=np.uint8)
            f_out.write(padding_bytes.tobytes())
            
        print(" Done. (Aligned Size: ", aligned_out_size, " bytes)")
        
    f_in.close()
    f_out.close()
    print("=========================================================")
    print(" QUANTIZATION COMPLETED SUCCESSFULLY!")
    print(" Quantized weights written to: gemma4_weights_ternary.bin")
    print("=========================================================")
