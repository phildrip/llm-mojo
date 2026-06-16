#!/usr/bin/env python3
import os
import struct
import json
import numpy as np

def main():
    model_dir = "gemma-4-26B-A4B-it-qat-q4_0-unquantized-assistant"
    safetensors_path = os.path.join(model_dir, "model.safetensors")
    output_bin_path = "assistant_weights.bin"

    print(f"Opening {safetensors_path}...")
    if not os.path.exists(safetensors_path):
        print(f"CRITICAL: {safetensors_path} not found!")
        return

    # 1. Parse safetensors JSON header
    with open(safetensors_path, "rb") as f:
        header_size_bytes = f.read(8)
        header_size = struct.unpack("<Q", header_size_bytes)[0]
        header_bytes = f.read(header_size)
        header = json.loads(header_bytes.decode("utf-8"))

    tensor_catalog = {}
    for k, entry in header.items():
        if k != "__metadata__":
            tensor_catalog[k] = entry

    print(f"Loaded safetensors catalog with {len(tensor_catalog)} tensors.")

    # Zero-dependency BF16 decoder helper
    def load_tensor_flat(k):
        if k not in tensor_catalog:
            raise KeyError(f"Required tensor '{k}' not found.")
        entry = tensor_catalog[k]
        dtype = entry["dtype"]
        shape = entry["shape"]
        start, end = entry["data_offsets"]

        with open(safetensors_path, "rb") as f:
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

    print(f"Writing FP32 tensors to {output_bin_path}...")

    # Order of non-layer weights:
    # 1. model.embed_tokens.weight
    # 2. pre_projection.weight
    # 3. post_projection.weight
    # 4. model.norm.weight
    non_layer_keys = [
        "model.embed_tokens.weight",
        "pre_projection.weight",
        "post_projection.weight",
        "model.norm.weight"
    ]

    total_bytes_written = 0

    with open(output_bin_path, "wb") as f_out:
        # Write non-layer weights
        for key in non_layer_keys:
            print(f"  Writing {key}...")
            arr = load_tensor_flat(key)
            bytes_arr = arr.tobytes()
            f_out.write(bytes_arr)
            total_bytes_written += len(bytes_arr)

        # Write layer weights for 4 layers (0, 1, 2, 3)
        for i in range(4):
            print(f"  Writing weights for Layer {i}...")
            layer_keys = [
                f"model.layers.{i}.input_layernorm.weight",
                f"model.layers.{i}.self_attn.q_proj.weight",
                f"model.layers.{i}.self_attn.o_proj.weight",
                f"model.layers.{i}.self_attn.q_norm.weight",
                f"model.layers.{i}.post_attention_layernorm.weight",
                f"model.layers.{i}.layer_scalar",
                f"model.layers.{i}.pre_feedforward_layernorm.weight",
                f"model.layers.{i}.mlp.gate_proj.weight",
                f"model.layers.{i}.mlp.up_proj.weight",
                f"model.layers.{i}.mlp.down_proj.weight",
                f"model.layers.{i}.post_feedforward_layernorm.weight"
            ]
            for key in layer_keys:
                arr = load_tensor_flat(key)
                bytes_arr = arr.tobytes()
                f_out.write(bytes_arr)
                total_bytes_written += len(bytes_arr)

    print(f"Success! Done writing {output_bin_path}. Total bytes: {total_bytes_written} bytes ({total_bytes_written / (1024*1024):.2f} MB).")

if __name__ == "__main__":
    main()
