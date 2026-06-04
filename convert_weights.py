import os
import sys
import json
import struct
import argparse
import numpy as np

try:
    from huggingface_hub import snapshot_download
except ImportError:
    print("This converter requires the 'huggingface_hub' and 'numpy' libraries.")
    print("Please install them first: pip install huggingface_hub numpy")
    sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Convert Hugging Face Gemma 4 / Gemma weights to Mojo O_DIRECT page-aligned binary format.")
    parser.add_argument("--model_id", type=str, default="google/gemma-4-12B-it", help="Hugging Face Model ID or local directory containing .safetensors files")
    parser.add_argument("--output", type=str, default="gemma4_weights.bin", help="Output bin path for Mojo weight paging")
    args = parser.parse_args()

    model_path = args.model_id
    if not os.path.isdir(model_path):
        print(f"Downloading weights from Hugging Face Hub for: {args.model_id}...")
        # Note: Gemma models are gated. Make sure you ran 'huggingface-cli login' first.
        try:
            model_path = snapshot_download(
                repo_id=args.model_id,
                allow_patterns=["*.safetensors", "config.json"],
                ignore_patterns=["*.bin", "*.pth"]
            )
            print(f"Model downloaded successfully to local cache: {model_path}")
        except Exception as e:
            print(f"\nCRITICAL: Error downloading model. If this is a gated model, make sure you ran 'huggingface-cli login' first.\nError: {e}")
            sys.exit(1)

    # 1. Scan for safetensors files
    files = [os.path.join(model_path, f) for f in os.listdir(model_path) if f.endswith(".safetensors")]
    if not files:
        print(f"No safetensors files found in {model_path}")
        sys.exit(1)
    
    files = sorted(files)
    print(f"Found {len(files)} Safetensors files to convert.")

    # 2. Extract tensors map catalog (zero-dependency streaming lookup)
    print("Parsing safetensors headers...")
    tensor_catalog = {}
    for filepath in files:
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

    print(f"Loaded catalog with {len(tensor_catalog)} tensors.")

    # Helper function to load and decode a specific tensor
    def load_tensor_flat(k):
        if k not in tensor_catalog:
            raise KeyError(f"Tensor '{k}' not found in safetensors catalog.")
        
        info = tensor_catalog[k]
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
            
        return arr.flatten()

    # 3. Read config and figure out dimensions
    embed_dim = None
    head_dim = 256
    num_heads = None
    num_kv_heads = None
    intermediate_dim = None
    num_layers = None
    vocab_size = None

    config_path = os.path.join(model_path, "config.json")
    if os.path.exists(config_path):
        try:
            with open(config_path, "r") as f:
                cfg = json.load(f)
            # Support nested text_config if present
            text_cfg = cfg.get("text_config", cfg)
            embed_dim = text_cfg.get("hidden_size")
            head_dim = text_cfg.get("head_dim", 256)
            num_heads = text_cfg.get("num_attention_heads")
            num_kv_heads = text_cfg.get("num_key_value_heads", num_heads)
            intermediate_dim = text_cfg.get("intermediate_size")
            num_layers = text_cfg.get("num_hidden_layers")
            vocab_size = text_cfg.get("vocab_size")
            
            print("\n--- Loaded Architecture Parameters from config.json ---")
            print(f"embed_dim:        {embed_dim}")
            print(f"head_dim:         {head_dim}")
            print(f"num_heads:        {num_heads}")
            print(f"num_kv_heads:     {num_kv_heads}  (GQA Ratio: {num_heads // num_kv_heads}:1)")
            print(f"intermediate_dim: {intermediate_dim}")
            print(f"num_layers:       {num_layers}")
            print(f"vocab_size:       {vocab_size}")
            print("------------------------------------------------------\n")
        except Exception as e:
            print(f"Warning: Could not parse config.json: {e}")

    # Fallback to auto-detection from catalog if config.json parse failed or missing fields
    if None in (embed_dim, num_heads, num_kv_heads, intermediate_dim, num_layers, vocab_size):
        print("Auto-detecting missing parameters from safetensors catalog...")
        try:
            sample_q = next(k for k in tensor_catalog.keys() if "self_attn.q_proj.weight" in k)
            q_shape = tensor_catalog[sample_q]["entry"]["shape"]
            out_features, embed_dim_detected = q_shape
            if embed_dim is None: embed_dim = embed_dim_detected
            
            sample_k = next(k for k in tensor_catalog.keys() if "self_attn.k_proj.weight" in k)
            k_shape = tensor_catalog[sample_k]["entry"]["shape"]
            kv_features, _ = k_shape
            
            if num_heads is None: num_heads = out_features // head_dim
            if num_kv_heads is None: num_kv_heads = kv_features // head_dim
            
            if intermediate_dim is None:
                sample_gate = next(k for k in tensor_catalog.keys() if "mlp.gate_proj.weight" in k)
                intermediate_dim = tensor_catalog[sample_gate]["entry"]["shape"][0]
                
            if num_layers is None:
                layer_indices = sorted(list(set(int(k.split("layers.")[1].split(".")[0]) for k in tensor_catalog.keys() if "layers." in k)))
                num_layers = len(layer_indices)
                
            if vocab_size is None:
                sample_embed = next(k for k in tensor_catalog.keys() if "embed_tokens.weight" in k)
                vocab_size = tensor_catalog[sample_embed]["entry"]["shape"][0]

            print("\n--- Auto-detected Architecture Parameters ---")
            print(f"embed_dim:        {embed_dim}")
            print(f"head_dim:         {head_dim}")
            print(f"num_heads:        {num_heads}")
            print(f"num_kv_heads:     {num_kv_heads}")
            print(f"intermediate_dim: {intermediate_dim}")
            print(f"num_layers:       {num_layers}")
            print(f"vocab_size:       {vocab_size}")
            print("---------------------------------------------\n")
        except Exception as e:
            print(f"CRITICAL: Could not auto-detect configuration from tensors: {e}")
            sys.exit(1)

    # Detect prefix by checking some keys
    try:
        sample_q = next(k for k in tensor_catalog.keys() if "self_attn.q_proj.weight" in k)
        prefix = sample_q.split("layers.")[0] # e.g. "model.language_model." or "model." or ""
        print(f"Detected weight key prefix: '{prefix}'")
    except Exception as e:
        print(f"CRITICAL: Error detecting tensor name prefix: {e}")
        sys.exit(1)

    # 4. Pack layer weights sequentially
    print(f"Writing packed and page-aligned weights to {args.output}...")
    
    with open(args.output, "wb") as f_out:
        for i in range(num_layers):
            print(f"Processing Layer {i}/{num_layers}...", end="", flush=True)
            # Extract weights for Layer i
            try:
                wq = load_tensor_flat(f"{prefix}layers.{i}.self_attn.q_proj.weight")
                wk = load_tensor_flat(f"{prefix}layers.{i}.self_attn.k_proj.weight")
                
                v_key = f"{prefix}layers.{i}.self_attn.v_proj.weight"
                if v_key in tensor_catalog:
                    wv = load_tensor_flat(v_key)
                else:
                    print(f" (attention_k_eq_v fallback: copying k_proj to v_proj for layer {i})", end="", flush=True)
                    wv = wk.copy()
                    
                wo = load_tensor_flat(f"{prefix}layers.{i}.self_attn.o_proj.weight")
                
                w1 = load_tensor_flat(f"{prefix}layers.{i}.mlp.gate_proj.weight")
                w2 = load_tensor_flat(f"{prefix}layers.{i}.mlp.down_proj.weight")
                w3 = load_tensor_flat(f"{prefix}layers.{i}.mlp.up_proj.weight")
                
                rms_attn = load_tensor_flat(f"{prefix}layers.{i}.input_layernorm.weight")
                rms_ffn = load_tensor_flat(f"{prefix}layers.{i}.post_attention_layernorm.weight")
            except KeyError as e:
                print(f"\nCRITICAL: Missing expected tensor in layer {i}: {e}")
                sys.exit(1)
            
            # Combine all weights in our exact binary struct sequence
            layer_payload = np.concatenate([wq, wk, wv, wo, w1, w2, w3, rms_attn, rms_ffn])
            layer_bytes = layer_payload.tobytes()
            
            # Pad to multiple of 4096 bytes (O_DIRECT block size alignment)
            raw_size = len(layer_bytes)
            aligned_size = ((raw_size + 4095) // 4096) * 4096
            padding_needed = aligned_size - raw_size
            
            f_out.write(layer_bytes)
            if padding_needed > 0:
                f_out.write(b'\x00' * padding_needed)
            print(" Done.", flush=True)

    print(f"\nSuccess! Packed and page-aligned weights written to {args.output}")
    print("------------------------------------------------------------------------")
    print("Use the following parameters inside your model.mojo config:")
    print(f"        embed_dim = {embed_dim},")
    print(f"        head_dim = {head_dim},")
    print(f"        num_heads = {num_heads},")
    print(f"        num_kv_heads = {num_kv_heads},")
    print(f"        intermediate_dim = {intermediate_dim},")
    print(f"        num_layers = {num_layers},")
    print(f"        vocab_size = {vocab_size},")
    print("------------------------------------------------------------------------")

if __name__ == "__main__":
    main()
