"""
Convert DiffusionGemma-26B-A4B-it safetensors to Mojo Q4_0 flat binary format.

Outputs:
  diffusiongemma_q4.bin          - per-layer decoder weights (same layout as
                                   gemma4_moe_weights_q4.bin)
  diffusiongemma_sc_q4.bin       - self-conditioning MLP (gate/up/down + pre_norm)
  diffusiongemma_embed_q4.bin    - token embeddings (q4)
  diffusiongemma_final_norm.bin  - final RMSNorm weight (fp32)

Vision tower + encoder.language_model weights are filtered out — text-only.
Layer weights come from model.decoder.layers.* (the encoder.language_model
has no MoE experts, so we must use the decoder path).
"""
import os
import sys
import json
import time
import gc
import struct
import numpy as np
from huggingface_hub import snapshot_download


def quantize_q4_0_vectorized(x):
    """Q4_0 block-quantization, block size = 32."""
    assert x.size % 32 == 0, f"Size must be a multiple of 32, got {x.size}"
    x_reshaped = x.reshape(-1, 32)
    max_abs = np.max(np.abs(x_reshaped), axis=1)
    scale = max_abs / 7.0
    scale = np.maximum(scale, 1e-5)
    inv_scale = 1.0 / scale
    q = np.round(x_reshaped * inv_scale[:, np.newaxis])
    q = np.clip(q, -8, 7).astype(np.int8)
    u = (q + 8).astype(np.uint8)
    u0 = u[:, 0::2]
    u1 = u[:, 1::2]
    packed = u0 | (u1 << 4)
    return packed.flatten(), scale.astype(np.float32).flatten()


# DiffusionGemma text-config constants (from config.json).
EMBED_DIM = 2816
STANDARD_MLP_DIM = 2112
EXPERT_MLP_DIM = 704
NUM_EXPERTS = 128
TOP_K = 8
NUM_LAYERS = 30
SELF_CONDITIONING_SIZE = 2112  # from safetensors header

PREFIX = "model.decoder."


def is_global_layer(i):
    return i % 6 == 5


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Convert DiffusionGemma-26B-A4B-it to Mojo Q4_0 flat binary."
    )
    parser.add_argument("--model_id", default="google/diffusiongemma-26B-A4B-it",
                        help="Hugging Face Model ID or local dir with .safetensors")
    parser.add_argument("--output_prefix", default="diffusiongemma",
                        help="Output prefix (suffixes: _q4.bin, _sc_q4.bin, _embed_q4.bin, _final_norm.bin)")
    parser.add_argument("--no_download", action="store_true",
                        help="Skip HF download even if model_id is not a local dir")
    args = parser.parse_args()

    output_weights = f"{args.output_prefix}_q4.bin"
    output_sc = f"{args.output_prefix}_sc_q4.bin"
    output_embed = f"{args.output_prefix}_embed_q4.bin"
    output_norm = f"{args.output_prefix}_final_norm.bin"

    model_path = args.model_id
    if not os.path.isdir(model_path):
        if args.no_download:
            print(f"CRITICAL: local dir not found and --no_download set: {model_path}")
            sys.exit(1)
        print(f"Downloading weights from Hugging Face Hub for: {model_path}...")
        model_path = snapshot_download(
            repo_id=model_path,
            allow_patterns=["*.safetensors", "config.json"],
            ignore_patterns=["*.bin", "*.pth"],
        )
        print(f"Model downloaded to: {model_path}")

    print("=" * 72)
    print("       DIFFUSIONGEMMA-26B-A4B-IT  Q4_0 CONVERTER")
    print("=" * 72)

    # 1. Scan safetensors and build tensor catalog
    print("Scanning safetensors files and parsing headers...")
    files = sorted([os.path.join(model_path, f)
                    for f in os.listdir(model_path)
                    if f.endswith(".safetensors")])
    if not files:
        print(f"CRITICAL: No .safetensors files in '{model_path}'")
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
                        "entry": entry,
                    }
    print(f"Loaded catalog with {len(tensor_catalog)} tensors.")
    print(f"Filtering to prefix '{PREFIX}' (text-only; skipping vision encoder).")

    def load_tensor(key):
        if key not in tensor_catalog:
            raise KeyError(f"Required tensor '{key}' not in safetensors catalog.")
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
            arr = np.frombuffer(raw_bytes, dtype=np.uint16).astype(np.uint32)
            arr = (arr << 16).view(np.float32)
        elif dtype == "F16":
            arr = np.frombuffer(raw_bytes, dtype=np.float16).astype(np.float32)
        elif dtype == "F32":
            arr = np.frombuffer(raw_bytes, dtype=np.float32)
        else:
            raise ValueError(f"Unsupported dtype: {dtype}")
        return arr.reshape(shape)

    nonexistent_count = 0
    for k in list(tensor_catalog.keys()):
        if not k.startswith(PREFIX):
            del tensor_catalog[k]
            nonexistent_count += 1
    print(f"Filtered out {nonexistent_count} non-decoder tensors "
          f"(vision tower, encoder.language_model, etc.).")
    print(f"Remaining: {len(tensor_catalog)} tensors to convert.")

    # 2. Token embeddings -> Q4
    print("\n--- Token Embeddings (Q4_0) ---")
    embed_key = PREFIX + "embed_tokens.weight"
    if embed_key not in tensor_catalog:
        print(f"CRITICAL: Missing token embeddings: '{embed_key}'")
        sys.exit(1)
    print(f"Loading '{embed_key}'...")
    embed_weight = load_tensor(embed_key)
    print(f"  shape={embed_weight.shape}, elements={embed_weight.size}")
    print("Quantizing to Q4_0...")
    packed, scales = quantize_q4_0_vectorized(embed_weight)
    payload = packed.tobytes() + scales.tobytes()
    aligned = ((len(payload) + 4095) // 4096) * 4096
    pad = aligned - len(payload)
    print(f"Writing to {output_embed}...")
    with open(output_embed, "wb") as f_out:
        f_out.write(payload)
        if pad > 0:
            f_out.write(b"\x00" * pad)
    print(f"Done ({aligned} bytes)")
    del embed_weight, packed, scales, payload
    gc.collect()

    # 3. Final RMSNorm (fp32)
    print("\n--- Final RMSNorm ---")
    norm_key = PREFIX + "norm.weight"
    if norm_key in tensor_catalog:
        norm_weight = load_tensor(norm_key)
        print(f"  shape={norm_weight.shape}")
        with open(output_norm, "wb") as f_out:
            f_out.write(norm_weight.tobytes())
        print(f"Written to {output_norm} ({norm_weight.size * 4} bytes)")
        del norm_weight
        gc.collect()
    else:
        print(f"Warning: missing '{norm_key}' — skipping final norm.")

    # 4. Self-conditioning MLP -> Q4
    #    Layout: [gate_proj.packed][up_proj.packed][down_proj.packed]
    #            [gate_proj.scales][up_proj.scales][down_proj.scales]
    #            [pre_norm.weight fp32]
    #    post_norm has no learned weight (has_weight=False per vLLM), so skip it.
    print("\n--- Self-Conditioning MLP (Q4_0) ---")
    sc_prefix = PREFIX + "self_conditioning."
    sc_keys = [sc_prefix + n for n in
               ("gate_proj.weight", "up_proj.weight",
                "down_proj.weight", "pre_norm.weight")]
    for k in sc_keys:
        if k not in tensor_catalog:
            print(f"CRITICAL: Missing '{k}'")
            sys.exit(1)

    print("Loading SC tensors...")
    sc_gate = load_tensor(sc_prefix + "gate_proj.weight")
    sc_up = load_tensor(sc_prefix + "up_proj.weight")
    sc_down = load_tensor(sc_prefix + "down_proj.weight")
    sc_norm = load_tensor(sc_prefix + "pre_norm.weight")
    print(f"  gate_proj={sc_gate.shape} up_proj={sc_up.shape} "
          f"down_proj={sc_down.shape} pre_norm={sc_norm.shape}")

    print("Quantizing SC projections...")
    g_p, g_s = quantize_q4_0_vectorized(sc_gate)
    u_p, u_s = quantize_q4_0_vectorized(sc_up)
    d_p, d_s = quantize_q4_0_vectorized(sc_down)

    sc_payload = (
        g_p.tobytes() + u_p.tobytes() + d_p.tobytes()
        + g_s.tobytes() + u_s.tobytes() + d_s.tobytes()
        + sc_norm.tobytes()
    )
    aligned = ((len(sc_payload) + 4095) // 4096) * 4096
    pad = aligned - len(sc_payload)
    print(f"Writing to {output_sc}...")
    with open(output_sc, "wb") as f_out:
        f_out.write(sc_payload)
        if pad > 0:
            f_out.write(b"\x00" * pad)
    print(f"Done ({aligned} bytes)")
    del sc_gate, sc_up, sc_down, sc_norm
    del g_p, g_s, u_p, u_s, d_p, d_s, sc_payload
    gc.collect()

    # 5. Per-layer decoder weights (Q4)
    #    Same layout per layer as gemma4_moe_weights_q4.bin — model.mojo's
    #    get_layer_weights() handles the existing wq/wk/wv/wo/... order.
    print("\n--- Decoder Layers (Q4_0) ---")
    print(f"Writing to {output_weights}...")
    total_bytes = 0
    with open(output_weights, "wb") as f_out:
        for i in range(NUM_LAYERS):
            is_global = is_global_layer(i)
            t0 = time.time()
            print(f"Layer {i:02d}/{NUM_LAYERS} "
                  f"({'global' if is_global else 'sliding'})...", end=" ")

            lp = f"{PREFIX}layers.{i}."

            # --- Q4-quantized weights, in the order expected by get_layer_weights:
            #     wq, wk, [wv if not global], wo, w_gate, w_up, w_down,
            #     w_router, experts_gate_up, experts_down
            q4_names = [
                "self_attn.q_proj.weight",
                "self_attn.k_proj.weight",
            ]
            if not is_global:
                q4_names.append("self_attn.v_proj.weight")
            q4_names.extend([
                "self_attn.o_proj.weight",
                "mlp.gate_proj.weight",
                "mlp.up_proj.weight",
                "mlp.down_proj.weight",
                "router.proj.weight",
                "experts.gate_up_proj",
                "experts.down_proj",
            ])
            q4_keys = [lp + n for n in q4_names]
            for k in q4_keys:
                if k not in tensor_catalog:
                    print(f"\nCRITICAL: Missing '{k}'")
                    sys.exit(1)

            packed_list = []
            scales_list = []
            for k in q4_keys:
                w = load_tensor(k)
                p, s = quantize_q4_0_vectorized(w)
                packed_list.append(p.tobytes())
                scales_list.append(s.tobytes())
                del w
            packed_bytes = b"".join(packed_list)
            scales_bytes = b"".join(scales_list)

            # --- fp32-unchanged norms, in get_layer_weights order:
            #     input_layernorm, post_attention_layernorm, pre_feedforward_layernorm,
            #     pre_feedforward_layernorm_2, post_feedforward_layernorm,
            #     post_feedforward_layernorm_1, post_feedforward_layernorm_2,
            #     q_norm, k_norm, router.scale, router.per_expert_scale, layer_scalar
            fp32_names = [
                "input_layernorm.weight",
                "post_attention_layernorm.weight",
                "pre_feedforward_layernorm.weight",
                "pre_feedforward_layernorm_2.weight",
                "post_feedforward_layernorm.weight",
                "post_feedforward_layernorm_1.weight",
                "post_feedforward_layernorm_2.weight",
                "self_attn.q_norm.weight",
                "self_attn.k_norm.weight",
                "router.scale",
                "router.per_expert_scale",
                "layer_scalar",
            ]
            fp32_keys = [lp + n for n in fp32_names]
            for k in fp32_keys:
                if k not in tensor_catalog:
                    print(f"\nCRITICAL: Missing '{k}'")
                    sys.exit(1)
            unchanged_bytes = b"".join(load_tensor(k).tobytes() for k in fp32_keys)

            layer_payload = packed_bytes + scales_bytes + unchanged_bytes
            aligned = ((len(layer_payload) + 4095) // 4096) * 4096
            pad = aligned - len(layer_payload)
            f_out.write(layer_payload)
            if pad > 0:
                f_out.write(b"\x00" * pad)
            total_bytes += aligned
            print(f"Done ({aligned} bytes, {time.time() - t0:.2f}s)")

            del packed_list, scales_list, packed_bytes, scales_bytes
            del unchanged_bytes, layer_payload
            gc.collect()

    print("\n" + "=" * 72)
    print("CONVERSION COMPLETED!")
    print(f"  {output_weights}  ({total_bytes} bytes)")
    print(f"  {output_sc}  (self-conditioning MLP)")
    print(f"  {output_embed}  (token embeddings)")
    print(f"  {output_norm}  (final RMSNorm)")
    print("-" * 72)
    print("Diffusion sampler constants (from generation_config.json):")
    print("  canvas_length        = 256")
    print("  max_denoising_steps  = 48")
    print("  t_min, t_max          = 0.4, 0.8")
    print("  entropy_bound         = 0.1")
    print("  confidence_threshold  = 0.005")
    print("  stability_threshold    = 1")
    print("  logit_softcap         = 30.0")
    print("=" * 72)


if __name__ == "__main__":
    main()