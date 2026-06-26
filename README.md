# 🌌 Mojo Gemma-4 26B Speculative MoE Engine

> [!WARNING]  
> **EXPERIMENTAL & WORK IN PROGRESS**  
> This repository is a research-oriented codebase created for learning, performance exploration, and system engineering research. It is very much a work-in-progress and is not intended for production use.

This project implements a highly optimized, CPU-resident **Gemma-4 26B Mixture-of-Experts (MoE)** inference engine written in **Mojo**, utilizing **Speculative Decoding** with co-trained Multi-Token Prediction (MTP) assistant layers and **Continuous Batching** via virtual Paged Attention blocks.

---

## 🚀 Key Architectural Features

1. **Gemma-4 26B MoE Support**: Efficient routing across 128 experts (using Top-8 routing tables) fully resident in system RAM.
2. **Hand-Vectorized SIMD Math Kernels**: Highly parallel, block-unrolled 4-bit (`Q4_0`) matrix multiplication and GELU math kernels leveraging Mojo's direct vector lane memory access.
3. **Continuous Batching Scheduler**: A core-aware dynamic scheduling loop with dynamic sequence loading, prefill/decode slotting, and explicit resource reclamation (**100% Zero-Leak** memory management).
4. **Experimental MTP Speculator**: Integrated support for `Gemma4AssistantForCausalLM` drafting heads which share the target model's upper KV Cache layers (Layers 27 & 29) to guess future sequences.

---

## 🛠️ How to Compile & Run

### 1. Prerequisites
Ensure you have [Pixi](https://pixi.sh/) installed. The workspace relies on a Pixi environment to provide cross-platform library access and Python environment links.

### 2. Build the Binary
To compile the Mojo source into a high-performance native executable (linking with necessary standard math libraries):
```bash
pixi run mojo build model_speculative_moe_batched.mojo -o test_bin -Xlinker -lm
```

### 3. Run the Engine

* **Standard Interactive Chat (Baseline / Recommended for speed)**:
  Runs the optimized main model directly. Highly recommended for daily testing as it bypasses speculative rollbacks, operating at a blazing fast **~4.7 tokens/sec** on CPU:
  ```bash
  pixi run ./test_bin --chat --no-mtp
  ```

* **Experimental Speculative Decoding Mode (MTP)**:
  Runs the co-trained 4-layer speculative draft assistant sequentially ahead of the main model:
  ```bash
  pixi run ./test_bin --chat
  ```

---

## 🎮 Hardware Acceleration & AMD APU (Cezanne) Setup

For environments utilizing integrated graphics, such as the **AMD Cezanne APU (Ryzen 7 5700G, Vega iGPU)**, direct GPU acceleration is supported via `model_q8_gpu.mojo`. Because AMD APUs are non-discrete "integrated" devices, the ROCm / HSA (Heterogeneous System Architecture) runtime and KFD (Kernel Fusion Driver) require specific configurations to properly expose the compute device:

### 1. Device Permissions
Your user account must have permissions to read and write to the ROCm compute device (`/dev/kfd`) and the direct rendering interfaces (`/dev/dri/`). Add your user to the `render` and `video` groups:
```bash
sudo usermod -aG render,video $USER
```
> [!IMPORTANT]
> You must log out and log back in (or reboot) for these group changes to apply. You can verify device access with `ls -l /dev/kfd`.

### 2. HSA GFX Version Override
Since Cezanne's native GFX version (typically `gfx90c` / Vega 8) is unsupported or ignored by the default ROCm/HSA runtime, you must force compatibility by overriding the target GFX version to a supported discrete Vega architecture (`gfx900`):
```bash
export HSA_OVERRIDE_GFX_VERSION=9.0.0
```

### 3. Compiling & Running with GPU Acceleration
To compile and run the GPU-accelerated version of the model:
```bash
# Compile the GPU-targeted model
pixi run mojo build model_q8_gpu.mojo -o model_q8_gpu -Xlinker -lm

# Run with the GFX override
HSA_OVERRIDE_GFX_VERSION=9.0.0 ./model_q8_gpu
```

---

## 🧪 Current Research & WIP Focus Areas

* **Quantization Activation Shift**: Because the main model is quantized to `Q4_0` to fit in RAM while the speculator remains in unquantized `FP32`, the resulting activation shift can cause the speculator's guesses to drift out-of-distribution (generating incoherent word-pieces). We are actively investigating calibration/scaling correction factors to realign the pre-projection layers.
* **Thread Scheduling Tuning**: Optimizing Mojo's parallel scheduler overhead during low-dimension single-token decode steps to maximize multi-core CPU utilization.
