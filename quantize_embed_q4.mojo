from std.sys.info import num_physical_cores
from std.memory import alloc
from std.algorithm import parallelize
from std.math import min, max
from std.python import Python

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
struct Config:
    comptime embed_dim = 3840
    comptime vocab_size = 262144

# -----------------------------------------------------------------------------
# Quantization and Packing Kernel (Block Size 32)
# -----------------------------------------------------------------------------
@always_inline
def quantize_and_pack_block(
    src_ptr: UnsafePointer[Float32, _],
    dest_packed_ptr: UnsafePointer[UInt8, MutExternalOrigin],
    dest_scales_ptr: UnsafePointer[Float32, MutExternalOrigin],
    num_tokens: Int,
    num_tasks: Int,
):
    var chunk_size = num_tokens // num_tasks
    var c_in_packed = Config.embed_dim // 2
    var c_in_blocks = Config.embed_dim // 32
    
    def worker(t: Int) capturing -> None:
        var start_tok = t * chunk_size
        var end_tok = (t + 1) * chunk_size if t < num_tasks - 1 else num_tokens
        
        for tok in range(start_tok, end_tok):
            var tok_offset_src = tok * Config.embed_dim
            var tok_offset_dest = tok * c_in_packed
            var tok_offset_scales = tok * c_in_blocks
            
            for b in range(c_in_blocks):
                var block_start = tok_offset_src + b * 32
                var block_dest = tok_offset_dest + b * 16
                var block_scale_idx = tok_offset_scales + b
                
                # Compute max absolute value of the block
                var max_abs: Float32 = 0.0
                for i in range(32):
                    var v = abs(src_ptr[block_start + i])
                    if v > max_abs:
                        max_abs = v
                
                var scale = max_abs / 7.0
                if scale < 1e-5:
                    scale = 1e-5
                
                dest_scales_ptr[block_scale_idx] = scale
                
                var inv_scale = 1.0 / scale
                for i in range(0, 32, 2):
                    var v0 = src_ptr[block_start + i]
                    var v1 = src_ptr[block_start + i + 1]
                    
                    # Quantize to [-8, 7]
                    var q0 = Int(max(Float32(-8.0), min(Float32(7.0), Float32(Int(v0 * inv_scale + 0.5 if v0 * inv_scale >= 0.0 else v0 * inv_scale - 0.5)))))
                    var q1 = Int(max(Float32(-8.0), min(Float32(7.0), Float32(Int(v1 * inv_scale + 0.5 if v1 * inv_scale >= 0.0 else v1 * inv_scale - 0.5)))))
                    
                    # Offset to [0, 15] for unsigned packing
                    var u0 = q0 + 8
                    var u1 = q1 + 8
                    
                    dest_packed_ptr[block_dest + (i // 2)] = UInt8(u0 | (u1 << 4))
                    
    parallelize[worker](num_tasks)

# -----------------------------------------------------------------------------
# Main Script Entry
# -----------------------------------------------------------------------------
def main() raises:
    var num_tasks = num_physical_cores()
    
    print("=========================================================")
    print("      GEMMA 4 EMBEDDING PARALLEL Q4_0 QUANTIZER          ")
    print("=========================================================")
    print("Target Cores: ", num_tasks)
    print("Input File:   gemma4_embed.bin")
    print("Output File:  gemma4_embed_q4.bin")
    print("---------------------------------------------------------")
    
    var py = Python.import_module("builtins")
    var np = Python.import_module("numpy")
    
    var f_in = py.open("gemma4_embed.bin", "rb")
    var f_out = py.open("gemma4_embed_q4.bin", "wb")
    
    # Process in chunks of 8192 tokens to remain extremely memory efficient
    var chunk_size_tokens = 8192
    var total_tokens = Config.vocab_size
    var num_chunks = total_tokens // chunk_size_tokens
    
    var bytes_per_token_src = Config.embed_dim * 4
    var bytes_per_chunk_src = chunk_size_tokens * bytes_per_token_src
    
    var struct_mod = Python.import_module("struct")
    var header_bytes = struct_mod.pack("ii", Config.vocab_size, Config.embed_dim)
    _ = f_out.write(header_bytes)
    
    for chunk_idx in range(num_chunks):
        try:
            print("Processing chunk ", chunk_idx + 1, "/", num_chunks, "...", end="", flush=True)
            
            # 1. Read source float32 bytes
            var raw_bytes = f_in.read(bytes_per_chunk_src)
            var read_len = py.len(raw_bytes)
            print(" (read ", read_len, " bytes)... ", end="", flush=True)
            
            var float_count = chunk_size_tokens * Config.embed_dim
            var chunk_np = np.frombuffer(raw_bytes, dtype=np.float32, count=float_count)
            
            var addr = Int(py=chunk_np.__array_interface__["data"][0])
            var src_ptr = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=addr)
            
            # 2. Allocate output packed and scales arrays
            var total_packed_bytes = chunk_size_tokens * (Config.embed_dim // 2)
            var packed_np = np.zeros(total_packed_bytes, dtype=np.uint8)
            
            var total_scales = chunk_size_tokens * (Config.embed_dim // 32)
            var scales_np = np.zeros(total_scales, dtype=np.float32)
            
            var packed_ptr = UnsafePointer[UInt8, MutExternalOrigin](unsafe_from_address=Int(py=packed_np.__array_interface__["data"][0]))
            var scales_ptr = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(py=scales_np.__array_interface__["data"][0]))
            
            # 3. Parallel Quantize
            quantize_and_pack_block(src_ptr, packed_ptr, scales_ptr, chunk_size_tokens, num_tasks)
            
            # 4. Write chunk output to file
            _ = f_out.write(packed_np.tobytes())
            _ = f_out.write(scales_np.tobytes())
            
            # Keep Python objects/buffers alive during parallel execution
            var shape = chunk_np.shape
            print(" Done. (shape: ", shape, ")", flush=True)
        except e:
            print("\nCRITICAL ERROR in chunk ", chunk_idx + 1, ": ", e, flush=True)
            break
        
    f_in.close()
    f_out.close()
    print("=========================================================")
    print(" QUANTIZATION COMPLETED SUCCESSFULLY!")
    print(" Quantized embeddings written to: gemma4_embed_q4.bin")
    print("=========================================================")
