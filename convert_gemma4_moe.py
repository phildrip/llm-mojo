import os
import sys
import json
import time
import gc
import struct
import numpy as np

def quantize_q4_0_vectorized(x):
    """
    Highly optimized, vectorized Q4_0 block-quantization using NumPy.
    Block size = 32.
    """
    assert x.size % 32 == 0, f"Size must be a multiple of 32, got {x.size}"
    x_reshaped = x.reshape(-1, 32)
    
    # Compute max absolute value per block
    max_abs = np.max(np.abs(x_reshaped), axis=1)
    
    # Scale factor
    scale = max_abs / 7.0
    scale = np.maximum(scale, 1e-5)
    
    # Quantize
    inv_scale = 1.0 / scale
    q = np.round(x_reshaped * inv_scale[:, np.newaxis])
    q = np.clip(q, -8, 7).astype(np.int8)
    
    # Pack unsigned [0, 15]
    u = (q + 8).astype(np.uint8)
    u0 = u[:, 0::2]
    u1 = u[:, 1::2]
    packed = u0 | (u1 << 4)
    
    return packed.flatten(), scale.astype(np.float32).flatten()

def main():
    print("========================================================================")
    print("        GEMMA-4-26B-A4B-IT MEMORY-EFFICIENT Q4_0 MOE CONVERTER          ")
    print("========================================================================")
    
    model_dir = "gemma-4-26B-A4B-it-qat-q4_0-unquantized"
    output_weights = "gemma4_moe_weights_q4.bin"
    output_embed = "gemma4_moe_embed_q4.bin"
    output_norm = "gemma4_moe_final_norm.bin"
    
    if not os.path.exists(model_dir) or not os.path.isdir(model_dir):
        print(f"CRITICAL: Model directory '{model_dir}' not found.")
        sys.exit(1)
        
    # 1. Scan safetensors files and build exact tensor catalog
    print("Scanning safetensors files and parsing headers...")
    files = sorted([os.path.join(model_dir, f) for f in os.listdir(model_dir) if f.endswith(".safetensors")])
    if not files:
        print(f"CRITICAL: No .safetensors files found in '{model_dir}'")
        sys.exit(1)
        
    tensor_catalog = {}
    for filepath in files:
        print(f"  Parsing header of {os.path.basename(filepath)}...")
        with open(filepath, "rb") as f:
            header_size_bytes = f.read(8)
            if len(header_size_bytes) < 8:
                continue
            header_size = struct.unpack("<Q", header_size_bytes)[0]
            header_bytes = f.read(header_size)
            header = json.loads(header_bytes.decode("utf-8"))
            for k, entry in header.items():
                if k != "__metadata__":
                    tensor_catalog[k] = {
                        "filepath": filepath,
                        "header_size": header_size,
                        "entry": entry
                    }
                    
    print(f"Successfully loaded catalog with {len(tensor_catalog)} tensors.")
    prefix = "model.language_model."
    
    # Helper to stream and convert a tensor to float32 NumPy array
    def load_tensor(key):
        if key not in tensor_catalog:
            raise KeyError(f"Required tensor '{key}' not found in safetensors catalog.")
            
        info = tensor_catalog[key]
        filepath = info["filepath"]
        header_size = info["header_size"]
        entry = info["entry"]
        dtype = entry["dtype"]
        shape = entry["shape"]
        start, end = entry["data_offsets"]
        
        with open(filepath, "rb") as f:
            f.seek(8 + header_size + start)
            raw_bytes = f.read(end - start)
            
        if dtype == "BF16":
            uint16_arr = np.frombuffer(raw_bytes, dtype=np.uint16)
            uint32_arr = uint16_arr.astype(np.uint32) << 16
            arr = uint32_arr.view(np.float32)
        elif dtype == "F16":
            arr = np.frombuffer(raw_bytes, dtype=np.float16).astype(np.float32)
        elif dtype == "F32":
            arr = np.frombuffer(raw_bytes, dtype=np.float32)
        else:
            raise ValueError(f"Unsupported dtype: {dtype}")
            
        return arr.reshape(shape)
        
    # 2. Extract token embeddings and quantize to Q4_0
    print("\n--- Processing Token Embeddings ---")
    embed_key = f"{prefix}embed_tokens.weight"
    if embed_key in tensor_catalog:
        start_time = time.time()
        print(f"Loading and converting embeddings tensor '{embed_key}'...")
        embed_weight = load_tensor(embed_key)
        print(f"Embedding shape: {embed_weight.shape}, elements: {embed_weight.size}")
        
        print("Quantizing embeddings to Q4_0...")
        packed_embed, scales_embed = quantize_q4_0_vectorized(embed_weight)
        
        # Combine packed bytes and scales
        embed_payload = packed_embed.tobytes() + scales_embed.tobytes()
        aligned_size = ((len(embed_payload) + 4095) // 4096) * 4096
        padding_needed = aligned_size - len(embed_payload)
        
        print(f"Writing embeddings to {output_embed}...")
        with open(output_embed, "wb") as f_out:
            f_out.write(embed_payload)
            if padding_needed > 0:
                f_out.write(b'\x00' * padding_needed)
        print(f"Embedding conversion complete in {time.time() - start_time:.2f}s ({aligned_size} bytes written)")
        
        # Free memory
        del embed_weight, packed_embed, scales_embed, embed_payload
        gc.collect()
    else:
        print(f"Warning: Could not find embed_tokens tensor '{embed_key}'")
        
    # 3. Extract final layer norm weight (Float32 unchanged)
    print("\n--- Processing Final Layer Norm ---")
    norm_key = f"{prefix}norm.weight"
    if norm_key in tensor_catalog:
        norm_weight = load_tensor(norm_key)
        print(f"Final norm shape: {norm_weight.shape}")
        with open(output_norm, "wb") as f_out:
            f_out.write(norm_weight.tobytes())
        print(f"Final norm written to {output_norm} ({norm_weight.size*4} bytes)")
        del norm_weight
        gc.collect()
    else:
        print(f"Warning: Could not find final layer norm tensor '{norm_key}'")
        
    # 4. Process each layer sequentially
    print("\n--- Quantizing Decoder Layers (Layer-by-Layer) ---")
    num_layers = 30
    total_start_time = time.time()
    
    with open(output_weights, "wb") as f_out:
        for i in range(num_layers):
            layer_start_time = time.time()
            is_global = (i % 6 == 5)
            print(f"Processing Layer {i:02d}/30 ({'global' if is_global else 'sliding'})...")
            
            # Helper to load a layer tensor
            def load_layer_tensor(name):
                return load_tensor(f"{prefix}layers.{i}.{name}")
                
            # Load and slice standard attention projections
            wq = load_layer_tensor("self_attn.q_proj.weight")
            wk = load_layer_tensor("self_attn.k_proj.weight")
            
            if not is_global:
                wv = load_layer_tensor("self_attn.v_proj.weight")
            else:
                wv = np.empty((0,), dtype=np.float32)
                
            wo = load_layer_tensor("self_attn.o_proj.weight")
            
            # Load and slice dense MLP (Shared Expert) projections
            w_gate = load_layer_tensor("mlp.gate_proj.weight")
            w_up = load_layer_tensor("mlp.up_proj.weight")
            w_down = load_layer_tensor("mlp.down_proj.weight")
            
            # Load and slice sparse MoE router projection
            w_router = load_layer_tensor("router.proj.weight")
            
            # Load and slice 128 Experts stacked projections
            experts_gate_up = load_layer_tensor("experts.gate_up_proj")
            experts_down = load_layer_tensor("experts.down_proj")
            
            # Quantize all these matrices and collect their packed bytes and scales
            matrices_to_quantize = [
                ("wq", wq),
                ("wk", wk),
                ("wv", wv),
                ("wo", wo),
                ("w_gate", w_gate),
                ("w_up", w_up),
                ("w_down", w_down),
                ("w_router", w_router),
                ("experts_gate_up", experts_gate_up),
                ("experts_down", experts_down)
            ]
            
            packed_list = []
            scales_list = []
            for name, mat in matrices_to_quantize:
                if mat.size > 0:
                    p, s = quantize_q4_0_vectorized(mat)
                    packed_list.append(p.tobytes())
                    scales_list.append(s.tobytes())
                else:
                    packed_list.append(b"")
                    scales_list.append(b"")
                    
            packed_bytes = b"".join(packed_list)
            scales_bytes = b"".join(scales_list)
            
            # Load Float32 unchanged weights & norms
            input_layernorm = load_layer_tensor("input_layernorm.weight")
            post_attention_layernorm = load_layer_tensor("post_attention_layernorm.weight")
            pre_feedforward_layernorm = load_layer_tensor("pre_feedforward_layernorm.weight")
            pre_feedforward_layernorm_2 = load_layer_tensor("pre_feedforward_layernorm_2.weight")
            post_feedforward_layernorm = load_layer_tensor("post_feedforward_layernorm.weight")
            post_feedforward_layernorm_1 = load_layer_tensor("post_feedforward_layernorm_1.weight")
            post_feedforward_layernorm_2 = load_layer_tensor("post_feedforward_layernorm_2.weight")
            
            q_norm = load_layer_tensor("self_attn.q_norm.weight")
            k_norm = load_layer_tensor("self_attn.k_norm.weight")
            
            router_scale = load_layer_tensor("router.scale")
            router_per_expert_scale = load_layer_tensor("router.per_expert_scale")
            layer_scalar = load_layer_tensor("layer_scalar")
            
            unchanged_weights = [
                input_layernorm,
                post_attention_layernorm,
                pre_feedforward_layernorm,
                pre_feedforward_layernorm_2,
                post_feedforward_layernorm,
                post_feedforward_layernorm_1,
                post_feedforward_layernorm_2,
                q_norm,
                k_norm,
                router_scale,
                router_per_expert_scale,
                layer_scalar
            ]
            
            unchanged_bytes = b"".join([w.tobytes() for w in unchanged_weights])
            
            # Total layer payload
            layer_payload = packed_bytes + scales_bytes + unchanged_bytes
            aligned_size = ((len(layer_payload) + 4095) // 4096) * 4096
            padding_needed = aligned_size - len(layer_payload)
            
            f_out.write(layer_payload)
            if padding_needed > 0:
                f_out.write(b'\x00' * padding_needed)
                
            print(f"  Layer {i:02d} complete in {time.time() - layer_start_time:.2f}s ({aligned_size} bytes written)")
            
            # Cleanup layer matrices and trigger garbage collector
            del wq, wk, wv, wo, w_gate, w_up, w_down, w_router, experts_gate_up, experts_down
            del packed_list, scales_list, packed_bytes, scales_bytes, unchanged_bytes, layer_payload
            gc.collect()
            
    print("\n========================================================================")
    print(f"SUCCESS: All 30 layers quantized and packed into {output_weights}")
    print(f"Total processing time: {time.time() - total_start_time:.2f} seconds")
    print("========================================================================")

if __name__ == "__main__":
    main()
