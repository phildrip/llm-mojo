# Sparsity-Aware NVMe Streaming: Architecture Plan & Roadmap

This document outlines the blueprint and roadmap for implementing a **Sparsity-Aware NVMe Paging Engine** in Mojo. By exploiting structural weight sparsity and dynamic activation sparsity, we can drastically reduce the amount of weight data streamed from the NVMe drive per token, breaking past the physical hardware bandwidth limit.

---

## 1. Phase 1: Activation Sparsity Profiling (Completed)

We implemented an activation sparsity profiling kernel in `model_q8_debug.mojo` and ran a real-time prompt generation pass using **Gemma-4-12B (Q8_0)** weights to measure the sparsity of `ffn_out_gate` (the output of the GeGLU activation layer `gelu(h1) * h2` inside the Feed-Forward Network).

### Profiled Sparsity Results

The table below shows the percentage of elements in the 14,336-dimensional FFN intermediate vector that fall below various thresholds during the prefill phase:

| Layer | Exact Zeros | Abs < $10^{-5}$ | Abs < $10^{-4}$ | Abs < $10^{-3}$ | Abs < $10^{-2}$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Layer 0** | 4.88% | 7.14% | 9.15% | 12.51% | 20.96% |
| **Layer 12** | 3.27% | 13.11% | **24.53%** | **43.13%** | **68.73%** |
| **Layer 24** | 0.00% | 0.03% | 0.45% | 2.90% | 19.11% |
| **Layer 36** | 0.00% | 0.10% | 0.64% | 3.54% | 19.02% |
| **Layer 47** | 1.84% | 4.40% | 7.36% | 13.27% | 25.95% |

### Key Findings
1. **High Layer 12 Sparsity:** In the middle layers of the network, sparsity is extraordinarily high. At a conservative threshold of $< 10^{-4}$, **24.53%** of the activation values are near zero. At a threshold of $< 10^{-3}$, this surges to **43.13%**, and at $< 10^{-2}$, it reaches **68.73%**.
2. **Thresholding Potential:** Since FFN activations go up to `222.0` in max amplitude, thresholding values below $10^{-4}$ or $10^{-3}$ to a hard `0.0` will have virtually zero impact on the model's output coherence, while instantly unlocking up to **43% sparsity** in the middle layers.
3. **Layer Asymmetry:** Sparsity is highly asymmetric across layers. Early layers (Layer 0) and deep layers (Layer 47) have lower activation sparsity (around 10% to 26% below $10^{-2}$), while middle layers (Layer 12) exhibit extremely high sparsity. Our paging engine must handle this dynamically.

---

## 2. Phase 2: Static Weight Sparsity (Zero-Block Skip)

The most straightforward win is to identify static blocks of weights that quantize to hard zeros across the board and bypass paging them.

### Roadmap & Tasks:
- [ ] **Quantizer Modification (`quantize_q8.mojo`):**
  - Group weights into blocks of 32 (our current Q8_0 block size).
  - For each block, check if all 32 quantized integers are exactly `0`.
  - Compile a **Sparsity Bitmask** for each layer (1 bit per block of 32 weights).
  - Write this compact bitmask array as a metadata header at the beginning of the `gemma4_weights_q8.bin` file.
- [ ] **Engine Modification (`model_q8.mojo`):**
  - Read the sparsity bitmask header when booting the engine.
  - Modify the `io_uring` double-buffer loader to check the bitmask before submitting reads.
  - Use Vectored I/O (`IORING_OP_READV`) or multiple parallel non-blocking read queue submissions to page only active block offsets, bypassing zero-blocks on disk entirely.

---

## 3. Phase 3: Dynamic FFN Down-Projection Row-Skipping ($W_2$ Skip)

Leveraging the 25% - 68% activation sparsity we profiled, we can conditionally skip loading parts of the Down-Projection matrix $W_2$ (the heaviest part of each FFN layer).

### Roadmap & Tasks:
- [ ] **FFN Activation Thresholding:**
  - Introduce a configurable threshold (e.g., `1e-4` or `1e-3`) in the FFN activation loop.
  - After computing `gelu_and_mul`, scan `ffn_out_gate` in blocks of 32. If a block's values are all below the threshold, set them to hard `0.0` and mark that block as inactive.
- [ ] **Dynamic io_uring Paging for $W_2$:**
  - Since $W_2$ is paged, split the prefetch of each layer's weights:
    * Prefetch $W_1$, $W_3$, Attention weights, and layer normalization factors in parallel with the previous layer (Pattern B).
    * Halt briefly after computing `ffn_out_gate` to check the active block indices.
    * Submit precise, non-contiguous `io_uring` reads to NVMe to load only the 30% to 75% of $W_2$ blocks that are actually required.
  - Profile the speedup over full monolithic prefetching on slow and fast NVMe SSDs.

---

## 4. Phase 4: Predictive Lookahead Sparsity

To avoid halting during the FFN pass, we will predict the active blocks of Layer $l$ while Layer $l-1$ is still running.

### Roadmap & Tasks:
- [ ] **Low-Rank Classifier Training:**
  - Create a tiny low-rank predictor matrix $W_{pred}$ for each layer (e.g., of size `embed_dim` $\times$ `reduced_dim`).
  - Train this predictor on activation traces of a calibration dataset to map the layer input vector $X_{l-1}$ to the predicted active channels of $X_l$.
- [ ] **Predictive Prefetching integration:**
  - Run the predictor as a micro-kernel (taking $<50$ microseconds) during the attention pass of the previous layer.
  - Pass the predicted bitmask directly to `io_uring` to load only the required sparse blocks of the *entire* next layer in the background.
