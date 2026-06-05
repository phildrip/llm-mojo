# Gemma-4-12B Quantization Project State

## Overview
We are building a highly optimized, custom inference engine in Mojo for Gemma-4-12B, focusing on memory optimization and ultra-fast hardware integration. The codebase leverages Linux `io_uring` with `O_DIRECT` for asynchronous, zero-copy NVMe weight paging.

Initially, we explored 2-bit ternary weights and 4-bit block-wise quantization. However, after detailed verification, we pivoted to **8-bit block-wise quantization (Q8_0 layout)** to guarantee maximum numerical accuracy and coherence while keeping the memory footprint down to a lean **12GB** (compared to the 41GB FP32 baseline).

Our Q8 inference pipeline is now **fully functional, verified, and generating highly coherent English text** in real-time.

---

## Architecture Details

- **Model:** Gemma-4-12B
- **Layers:** 48 total layers.
- **Attention:** Alternates between "sliding window" and "global" attention. Every 6th layer (i.e. layer 5, 11, 17...) is a "global" layer.
  - **Global Layers:** `kv_heads = 1`, `head_dim = 512`
  - **Sliding Layers:** `kv_heads = 8`, `head_dim = 256`
  - **Query Heads:** 16 across all layers.
- **Activations:** Run in full `Float32`.

---

## 8-Bit Block-Wise Quantization Scheme (Q8_0)

To achieve absolute mathematical coherence matching the FP32 model:
- **Block Size:** 32 elements.
- **Scaling:** One row-wise/block-wise `Float32` scale factor per block of 32 weights (`scale = max_abs_val / 127.0`).
- **Storage:** No complex packing or bit-packing overhead—weights are stored as standard `Int8` values, with one `Float32` scale factor per 32-element block.
- **Memory Footprint:** Drastically reduced from 41GB to **12GB** (`gemma4_weights_q8.bin`), allowing direct, zero-copy paging from NVMe SSDs in real-time.

---

## Core Breakthroughs & Fixes

During implementation, two critical bugs were successfully resolved:
1. **The Attention Scaling Discrepancy:**
   - *Issue:* The Q8 pipeline originally scaled dot-product attention scores by `1.0 / sqrt(head_dim)`. However, auditing the known-good FP32 baseline reference (`model.mojo` and `model_debug.mojo`) revealed that **attention scaling was omitted** (i.e., `attn_scores[t] = score` was used). Scaling by `1.0 / sqrt(head_dim)` flattened the logits, smoothing the softmax distribution and causing the model to output empty newlines or generic whitespace.
   - *Resolution:* Removed the redundant `* scale` from attention scores, matching the FP32 baseline mathematics exactly. Softmax immediately regained its sharpness, restoring high coherence.
2. **Mojo `print()` Buffer Buffering:**
   - *Issue:* Native Mojo `print(..., end="")` does not flush C-level standard output in real-time, masking the tokens being generated.
   - *Resolution:* Switched to Python's fully flushed `py_builtins.print(word, end="", flush=True)` to stream generated tokens to logs and terminal outputs instantly.

---

## Current Status & Verification

### 1. `quantize_q8.mojo` (Completed)
- Successfully compiled and ran on the 41GB FP32 file to produce the 12GB `gemma4_weights_q8.bin` in just over a minute.

### 2. `model_q8.mojo` (Active & Functional)
- Successfully compiled and running as a background task.
- Prefill completes in **13.2 seconds** for 15 tokens (~1.14 tokens/second) over direct NVMe `io_uring` paging.
- **Coherence Verified:** The model is currently generating flawless, highly coherent English. For the prompt *"Once upon a time in a"*, it streams:
  > *...hidden valley where the clouds sat low enough to touch the treetops, there lived a village of...*

---

## Next Steps

1. **Performance Tuning:**
   - Explore vectorizing and parallelizing key hot-paths in our custom Q8 matmul kernels (`q8_matmul_vec`) to push token-generation speeds further.
2. **Clean Up Diagnostic Logs:**
   - Once generation is verified and the user is fully satisfied, clean up temporary debugging scripts and intermediate logs in the `scratch/` directory.
