from std.sys import argv
from std.sys.info import num_physical_cores
from std.memory import alloc
from std.python import Python, PythonObject
from std.algorithm import parallelize
from std.math import min, max

# =============================================================================
# Vectorized & Parallel Conversion Routines
# =============================================================================

@always_inline
def parallel_convert_bf16_to_f32(
    src_ptr: UnsafePointer[UInt16, _],
    dest_ptr: UnsafePointer[Float32, MutExternalOrigin],
    length: Int,
    num_tasks: Int,
):
    """
    Highly optimized, vectorized conversion of BF16 to Float32.
    BF16 is the upper 16 bits of Float32, so conversion is a fast left-shift.
    """
    var chunk_size = length // num_tasks
    if chunk_size == 0:
        chunk_size = length

    def worker(t: Int) capturing -> None:
        var start = t * chunk_size
        if start >= length:
            return
        var end = (t + 1) * chunk_size if t < num_tasks - 1 else length
        if end > length:
            end = length
        
        comptime simd_width = 16
        var limit = start + ((end - start) // simd_width) * simd_width
        
        var u32_dest_ptr = dest_ptr.bitcast[UInt32]()
        
        for idx in range(start, limit, simd_width):
            var bf16_val = (src_ptr + idx).load[width=simd_width]()
            var u32_val = bf16_val.cast[DType.uint32]()
            var f32_bits = u32_val << 16
            (u32_dest_ptr + idx).store(f32_bits)
            
        for idx in range(limit, end):
            var val_u16 = src_ptr[idx]
            var val_u32 = UInt32(val_u16) << 16
            u32_dest_ptr[idx] = val_u32
            
    parallelize[worker](num_tasks)

@always_inline
def parallel_convert_f16_to_f32(
    src_ptr: UnsafePointer[Float16, _],
    dest_ptr: UnsafePointer[Float32, MutExternalOrigin],
    length: Int,
    num_tasks: Int,
):
    """
    Vectorized and parallelized conversion of Float16 to Float32.
    """
    var chunk_size = length // num_tasks
    if chunk_size == 0:
        chunk_size = length

    def worker(t: Int) capturing -> None:
        var start = t * chunk_size
        if start >= length:
            return
        var end = (t + 1) * chunk_size if t < num_tasks - 1 else length
        if end > length:
            end = length
        
        comptime simd_width = 16
        var limit = start + ((end - start) // simd_width) * simd_width
        
        for idx in range(start, limit, simd_width):
            var f16_val = (src_ptr + idx).load[width=simd_width]()
            var f32_val = f16_val.cast[DType.float32]()
            (dest_ptr + idx).store(f32_val)
            
        for idx in range(limit, end):
            dest_ptr[idx] = Float32(src_ptr[idx])
            
    parallelize[worker](num_tasks)

@always_inline
def parallel_copy_f32(
    src_ptr: UnsafePointer[Float32, _],
    dest_ptr: UnsafePointer[Float32, MutExternalOrigin],
    length: Int,
    num_tasks: Int,
):
    """
    Parallelized copy of Float32 elements.
    """
    var chunk_size = length // num_tasks
    if chunk_size == 0:
        chunk_size = length

    def worker(t: Int) capturing -> None:
        var start = t * chunk_size
        if start >= length:
            return
        var end = (t + 1) * chunk_size if t < num_tasks - 1 else length
        if end > length:
            end = length
        
        comptime simd_width = 16
        var limit = start + ((end - start) // simd_width) * simd_width
        
        for idx in range(start, limit, simd_width):
            var vals = (src_ptr + idx).load[width=simd_width]()
            (dest_ptr + idx).store(vals)
            
        for idx in range(limit, end):
            dest_ptr[idx] = src_ptr[idx]
            
    parallelize[worker](num_tasks)

# =============================================================================
# Helper Sizing and Loading Logic
# =============================================================================

def get_tensor_size(catalog: PythonObject, key: String) raises -> Int:
    """
    Returns the total number of elements in a tensor from the catalog.
    """
    if not Bool(key in catalog):
        return 0
    var info = catalog[key]
    var shape = info["entry"]["shape"]
    var length = 1
    for s_idx in range(len(shape)):
        length *= Int(py=shape[s_idx])
    return length

def load_and_convert_tensor(
    catalog: PythonObject,
    key: String,
    dest_ptr: UnsafePointer[Float32, MutExternalOrigin],
    num_tasks: Int,
    builtins: PythonObject,
    np: PythonObject,
) raises:
    """
    Reads a tensor's raw bytes, retrieves its physical address via NumPy FFI,
    and runs highly parallelized conversions directly into the destination pointer.
    """
    if not Bool(key in catalog):
        raise Error("Tensor '" + key + "' not found in safetensors catalog.")
        
    var info = catalog[key]
    var filepath = info["filepath"]
    var header_size = Int(py=info["header_size"])
    var entry = info["entry"]
    var dtype = String(entry["dtype"])
    var start = Int(py=entry["data_offsets"][0])
    var end = Int(py=entry["data_offsets"][1])
    var size_bytes = end - start
    
    # Read the raw binary bytes of the tensor
    var f = builtins.open(filepath, "rb")
    _ = f.seek(8 + header_size + start)
    var raw_bytes = f.read(size_bytes)
    _ = f.close()
    
    # Calculate total elements dynamically
    var length = 1
    var shape = entry["shape"]
    for s_idx in range(len(shape)):
        length *= Int(py=shape[s_idx])
        
    if dtype == "BF16":
        var np_arr = np.frombuffer(raw_bytes, dtype=np.uint16)
        var addr = Int(py=np_arr.__array_interface__["data"][0])
        var src_ptr = UnsafePointer[UInt16, MutExternalOrigin](unsafe_from_address=addr)
        parallel_convert_bf16_to_f32(src_ptr, dest_ptr, length, num_tasks)
        _ = raw_bytes
        _ = np_arr
    elif dtype == "F16":
        var np_arr = np.frombuffer(raw_bytes, dtype=np.float16)
        var addr = Int(py=np_arr.__array_interface__["data"][0])
        var src_ptr = UnsafePointer[Float16, MutExternalOrigin](unsafe_from_address=addr)
        parallel_convert_f16_to_f32(src_ptr, dest_ptr, length, num_tasks)
        _ = raw_bytes
        _ = np_arr
    elif dtype == "F32":
        var np_arr = np.frombuffer(raw_bytes, dtype=np.float32)
        var addr = Int(py=np_arr.__array_interface__["data"][0])
        var src_ptr = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=addr)
        parallel_copy_f32(src_ptr, dest_ptr, length, num_tasks)
        _ = raw_bytes
        _ = np_arr
    else:
        raise Error("Unsupported tensor dtype: " + dtype)

# =============================================================================
# Main Program Entry Point (CLI)
# =============================================================================

def main() raises:
    var num_tasks = num_physical_cores()
    
    print("=========================================================")
    print("        GEMMA 4 HIGH-SPEED MOJO WEIGHT CONVERTER         ")
    print("=========================================================")
    print("Running on:  ", num_tasks, " physical CPU cores")
    
    # Import necessary Python libraries for IO & structure metadata
    var sys = Python.import_module("sys")
    var os = Python.import_module("os")
    var json = Python.import_module("json")
    var py_struct = Python.import_module("struct")
    var builtins = Python.import_module("builtins")
    var np = Python.import_module("numpy")
    var argparse = Python.import_module("argparse")
    
    # Define CLI Arguments
    var parser = argparse.ArgumentParser(
        description="Convert Hugging Face Gemma 4 / Gemma weights to Mojo O_DIRECT page-aligned binary format."
    )
    _ = parser.add_argument(
        "--model_id", 
        default="google/gemma-4-12B-it", 
        help="Hugging Face Model ID or local directory containing .safetensors files"
    )
    _ = parser.add_argument(
        "--output", 
        default="gemma4_weights.bin", 
        help="Output bin path for Mojo weight paging"
    )
    _ = parser.add_argument(
        "--only_embeddings", 
        action="store_true", 
        help="Only extract and save token embedding weights to gemma4_embed.bin"
    )
    
    # Properly map Mojo argv to Python argparse list
    var args_list = argv()
    var py_argv = Python.list()
    for idx in range(1, len(args_list)):
        _ = py_argv.append(args_list[idx])
        
    var args = parser.parse_args(py_argv)
    var model_path = String(args.model_id)
    var output_file = String(args.output)
    var only_embeddings = Bool(args.only_embeddings)
    
    # 1. Download Model if directory doesn't exist locally
    if not Bool(os.path.isdir(model_path)):
        print("Downloading weights from Hugging Face Hub for: " + model_path + "...")
        try:
            var hf_hub = Python.import_module("huggingface_hub")
            var snapshot_download = hf_hub.snapshot_download
            model_path = String(snapshot_download(
                repo_id=model_path,
                allow_patterns=["*.safetensors", "config.json"],
                ignore_patterns=["*.bin", "*.pth"]
            ))
            print("Model downloaded successfully to local cache: " + model_path)
        except e:
            raise Error("CRITICAL: Error downloading model. If this is a gated model, make sure you ran 'huggingface-cli login' first.\nError: " + String(e))
            
    # 2. Scan directory for safetensors files
    var files = Python.list()
    var dir_files = os.listdir(model_path)
    for i in range(len(dir_files)):
        var f = String(dir_files[i])
        if f.endswith(".safetensors"):
            _ = files.append(os.path.join(model_path, f))
            
    if len(files) == 0:
        raise Error("No safetensors files found in directory: " + model_path)
        
    var sorted_files = builtins.sorted(files)
    print("Found " + String(len(sorted_files)) + " Safetensors files to convert.")
    
    # 3. Parse headers and build tensor catalog
    print("Parsing safetensors headers...")
    var tensor_catalog = Python.dict()
    
    for f_idx in range(len(sorted_files)):
        var filepath = sorted_files[f_idx]
        var f = builtins.open(filepath, "rb")
        var header_size_bytes = f.read(8)
        if len(header_size_bytes) < 8:
            _ = f.close()
            continue
        var header_size = py_struct.unpack("<Q", header_size_bytes)[0]
        var header_bytes = f.read(header_size)
        _ = f.close()
        
        var header = json.loads(header_bytes.decode("utf-8"))
        for k in header.keys():
            if k != "__metadata__":
                var entry = Python.dict()
                entry["filepath"] = filepath
                entry["header_size"] = header_size
                entry["entry"] = header[k]
                tensor_catalog[k] = entry
                
    print("Loaded catalog with " + String(len(tensor_catalog)) + " tensors.")
    
    # 4. Parse config.json to figure out parameters
    var embed_dim = -1
    var head_dim = 256
    var num_heads = -1
    var num_kv_heads = -1
    var intermediate_dim = -1
    var num_layers = -1
    var vocab_size = -1
    
    var config_path = os.path.join(model_path, "config.json")
    if Bool(os.path.exists(config_path)):
        try:
            var f_cfg = builtins.open(config_path, "r")
            var cfg = json.load(f_cfg)
            _ = f_cfg.close()
            
            var text_cfg = cfg
            if "text_config" in cfg:
                text_cfg = cfg["text_config"]
                
            if "hidden_size" in text_cfg:
                embed_dim = Int(py=text_cfg["hidden_size"])
            if "head_dim" in text_cfg:
                head_dim = Int(py=text_cfg["head_dim"])
            if "num_attention_heads" in text_cfg:
                num_heads = Int(py=text_cfg["num_attention_heads"])
            if "num_key_value_heads" in text_cfg:
                num_kv_heads = Int(py=text_cfg["num_key_value_heads"])
            else:
                num_kv_heads = num_heads
            if "intermediate_size" in text_cfg:
                intermediate_dim = Int(py=text_cfg["intermediate_size"])
            if "num_hidden_layers" in text_cfg:
                num_layers = Int(py=text_cfg["num_hidden_layers"])
            if "vocab_size" in text_cfg:
                vocab_size = Int(py=text_cfg["vocab_size"])
                
            print("\n--- Loaded Architecture Parameters from config.json ---")
            print("embed_dim:        ", embed_dim)
            print("head_dim:         ", head_dim)
            print("num_heads:        ", num_heads)
            print("num_kv_heads:     ", num_kv_heads)
            print("intermediate_dim: ", intermediate_dim)
            print("num_layers:       ", num_layers)
            print("vocab_size:       ", vocab_size)
            print("------------------------------------------------------\n")
        except e:
            print("Warning: Could not parse config.json: ", e)
            
    # Auto-detect fallback
    if embed_dim == -1 or num_heads == -1 or num_kv_heads == -1 or intermediate_dim == -1 or num_layers == -1 or vocab_size == -1:
        print("Auto-detecting missing parameters from safetensors catalog...")
        try:
            var sample_q = ""
            for k in tensor_catalog.keys():
                if "self_attn.q_proj.weight" in k:
                    sample_q = String(k)
                    break
            
            var q_shape = tensor_catalog[sample_q]["entry"]["shape"]
            var out_features = Int(py=q_shape[0])
            var embed_dim_detected = Int(py=q_shape[1])
            if embed_dim == -1:
                embed_dim = embed_dim_detected
                
            var sample_k = ""
            for k in tensor_catalog.keys():
                if "self_attn.k_proj.weight" in k:
                    sample_k = String(k)
                    break
            var k_shape = tensor_catalog[sample_k]["entry"]["shape"]
            var kv_features = Int(py=k_shape[0])
            
            if num_heads == -1:
                num_heads = out_features // head_dim
            if num_kv_heads == -1:
                num_kv_heads = kv_features // head_dim
                
            if intermediate_dim == -1:
                var sample_gate = ""
                for k in tensor_catalog.keys():
                    if "mlp.gate_proj.weight" in k:
                        sample_gate = String(k)
                        break
                intermediate_dim = Int(py=tensor_catalog[sample_gate]["entry"]["shape"][0])
                
            if num_layers == -1:
                var layer_set = builtins.set()
                for k in tensor_catalog.keys():
                    if "layers." in k:
                        var parts = String(k).split("layers.")
                        if len(parts) > 1:
                            var l_idx = parts[1].split(".")[0]
                            _ = layer_set.add(builtins.int(l_idx))
                num_layers = len(layer_set)
                
            if vocab_size == -1:
                var sample_embed = ""
                for k in tensor_catalog.keys():
                    if "embed_tokens.weight" in k:
                        sample_embed = String(k)
                        break
                vocab_size = Int(py=tensor_catalog[sample_embed]["entry"]["shape"][0])
                
            print("\n--- Auto-detected Architecture Parameters ---")
            print("embed_dim:        ", embed_dim)
            print("head_dim:         ", head_dim)
            print("num_heads:        ", num_heads)
            print("num_kv_heads:     ", num_kv_heads)
            print("intermediate_dim: ", intermediate_dim)
            print("num_layers:       ", num_layers)
            print("vocab_size:       ", vocab_size)
            print("---------------------------------------------\n")
        except e:
            raise Error("CRITICAL: Could not auto-detect configuration from tensors: " + String(e))
            
    # Detect Prefix
    var prefix: String
    try:
        var sample_q = ""
        for k in tensor_catalog.keys():
            if "self_attn.q_proj.weight" in k:
                sample_q = String(k)
                break
        var parts = sample_q.split("layers.")
        prefix = String(parts[0])
        print("Detected weight key prefix: '" + prefix + "'")
    except e:
        raise Error("CRITICAL: Error detecting tensor name prefix: " + String(e))
        
    # 5. Extract and Write Final Norm Weight
    var output_norm = "gemma4_final_norm.bin"
    print("Extracting final layer norm weights to " + output_norm + "...")
    var norm_key = prefix + "norm.weight"
    if Bool(norm_key in tensor_catalog):
        try:
            var length = get_tensor_size(tensor_catalog, norm_key)
            var norm_np = np.zeros(length, dtype=np.float32)
            var addr = Int(py=norm_np.__array_interface__["data"][0])
            var dest_ptr = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=addr)
            
            load_and_convert_tensor(tensor_catalog, norm_key, dest_ptr, num_tasks, builtins, np)
            
            var f_norm = builtins.open(output_norm, "wb")
            _ = f_norm.write(norm_np.tobytes())
            _ = f_norm.close()
            print("Success! Final layer norm written to " + output_norm)
        except e:
            print("Warning: Error writing final layer norm: " + String(e))
    else:
        print("Warning: Could not find final layer norm '" + norm_key + "'. Skipping.")
        
    # 6. Extract Only Embeddings if Flag Set
    if only_embeddings:
        var output_embed = "gemma4_embed.bin"
        print("Extracting token embeddings to " + output_embed + "...")
        var embed_key = prefix + "embed_tokens.weight"
        if not Bool(embed_key in tensor_catalog):
            raise Error("Missing embed_tokens tensor: " + embed_key)
            
        var length = get_tensor_size(tensor_catalog, embed_key)
        var embed_np = np.zeros(length, dtype=np.float32)
        var addr = Int(py=embed_np.__array_interface__["data"][0])
        var dest_ptr = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=addr)
        
        load_and_convert_tensor(tensor_catalog, embed_key, dest_ptr, num_tasks, builtins, np)
        
        var raw_bytes = embed_np.tobytes()
        var raw_size = len(raw_bytes)
        var aligned_size = ((raw_size + 4095) // 4096) * 4096
        var padding_needed = aligned_size - raw_size
        
        var f_embed = builtins.open(output_embed, "wb")
        _ = f_embed.write(raw_bytes)
        if padding_needed > 0:
            var padding_bytes = np.zeros(padding_needed, dtype=np.uint8)
            _ = f_embed.write(padding_bytes.tobytes())
        _ = f_embed.close()
        print("Success! Packed and page-aligned embeddings written to " + output_embed + " (" + String(aligned_size) + " bytes)")
        return
        
    # 7. Core Layer Packing Loop
    print("\nWriting packed and page-aligned weights to " + output_file + "...")
    var f_out = builtins.open(output_file, "wb")
    
    for i in range(num_layers):
        var is_global = (i % 6 == 5)
        print("Processing Layer " + String(i) + "/" + String(num_layers) + " (" + ("global" if is_global else "sliding") + ")...", end="")
        
        # Build sequence of keys for this layer
        var layer_keys = List[String]()
        _ = layer_keys.append(prefix + "layers." + String(i) + ".self_attn.q_proj.weight")
        _ = layer_keys.append(prefix + "layers." + String(i) + ".self_attn.k_proj.weight")
        
        if not is_global:
            _ = layer_keys.append(prefix + "layers." + String(i) + ".self_attn.v_proj.weight")
            
        _ = layer_keys.append(prefix + "layers." + String(i) + ".self_attn.o_proj.weight")
        _ = layer_keys.append(prefix + "layers." + String(i) + ".mlp.gate_proj.weight")
        _ = layer_keys.append(prefix + "layers." + String(i) + ".mlp.down_proj.weight")
        _ = layer_keys.append(prefix + "layers." + String(i) + ".mlp.up_proj.weight")
        
        _ = layer_keys.append(prefix + "layers." + String(i) + ".input_layernorm.weight")
        _ = layer_keys.append(prefix + "layers." + String(i) + ".post_attention_layernorm.weight")
        _ = layer_keys.append(prefix + "layers." + String(i) + ".pre_feedforward_layernorm.weight")
        _ = layer_keys.append(prefix + "layers." + String(i) + ".post_feedforward_layernorm.weight")
        
        _ = layer_keys.append(prefix + "layers." + String(i) + ".self_attn.q_norm.weight")
        _ = layer_keys.append(prefix + "layers." + String(i) + ".self_attn.k_norm.weight")
        _ = layer_keys.append(prefix + "layers." + String(i) + ".layer_scalar")
        
        # Calculate total float elements for this layer
        var total_floats = 0
        for k_idx in range(len(layer_keys)):
            var key = layer_keys[k_idx]
            total_floats += get_tensor_size(tensor_catalog, key)
            
        # Allocate flat float32 NumPy array for the entire layer
        var layer_np = np.zeros(total_floats, dtype=np.float32)
        var addr = Int(py=layer_np.__array_interface__["data"][0])
        var dest_ptr = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=addr)
        
        # Load and convert each tensor directly to its specific offset in the dest_ptr
        var current_offset = 0
        for k_idx in range(len(layer_keys)):
            var key = layer_keys[k_idx]
            var t_size = get_tensor_size(tensor_catalog, key)
            if t_size > 0:
                load_and_convert_tensor(tensor_catalog, key, dest_ptr + current_offset, num_tasks, builtins, np)
                current_offset += t_size
                
        # Write to output file
        var layer_bytes = layer_np.tobytes()
        var raw_size = len(layer_bytes)
        var aligned_size = ((raw_size + 4095) // 4096) * 4096
        var padding_needed = aligned_size - raw_size
        
        _ = f_out.write(layer_bytes)
        if padding_needed > 0:
            var padding_bytes = np.zeros(padding_needed, dtype=np.uint8)
            _ = f_out.write(padding_bytes.tobytes())
            
        print(" Done. (Aligned Size: " + String(aligned_size) + " bytes)")
        
    _ = f_out.close()
    
    print("\n=========================================================")
    print("CONVERSION COMPLETED SUCCESSFULLY!")
    print("Packed and page-aligned weights written to: " + output_file)
    print("Use the following parameters inside your model.mojo config:")
    print("        embed_dim = " + String(embed_dim) + ",")
    print("        head_dim = " + String(head_dim) + ",")
    print("        num_heads = " + String(num_heads) + ",")
    print("        num_kv_heads = " + String(num_kv_heads) + ",")
    print("        intermediate_dim = " + String(intermediate_dim) + ",")
    print("        num_layers = " + String(num_layers) + ",")
    print("        vocab_size = " + String(vocab_size) + ",")
    print("=========================================================")
