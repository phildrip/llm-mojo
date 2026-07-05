# DiffusionGemma Optimization Roadmap (CPU-First Execution Plan)

This document maps out the concrete, step-by-step engineering roadmap for accelerating **DiffusionGemma (26B-A4B-it)** on the **AMD Ryzen 7 5700G APU** using a **CPU-First** optimization strategy. 

We prioritize resolving the major CPU bottleneck (MoE expert execution) before introducing vocabulary trimming or GPU offloading.

---

## The Bottleneck: MoE Expert Cache Thrashing
Profiling shows a total step time of **~23.5 seconds**. The largest single bottleneck is **MoE expert execution**, consuming **9.0 seconds per step (38% of runtime)**.

* **Why?** Each of the 256 canvas tokens dynamically selects 8 experts. With 128 total experts, each CPU thread sequentially processes random tokens assigned to random experts. This pulls different 4.3 GB/layer expert weights in rapid succession, completely thrashing the **16MB L3 cache** and causing random memory bus reads.
* **The Math Overhead**: `q4_gemv_batch` is unsuited for small batches of tokens (e.g., $B \le 16$ per expert) because it performs $B \times 4 \times \text{blocks}$ horizontal `reduce_add()` operations per row.

```
                   Current MoE Path: ~9.0s (38% of Step Time)
                                        │
                                        ▼
             Phase 1: Custom CPU `q4_gemv_small_b` Kernel (Target: ~2s)
                                        │
                                        ▼
             Phase 2: Vocabulary Trimming to 32K (Target: 8x Sampler Speedup)
                                        │
                                        ▼
             Phase 3: Asymmetric APU Co-Execution (CPU + GPU Split)
```

---

## Phase 1: CPU-Side Kernel Optimization (The Custom `q4_gemv_small_b` Kernel)

Before implementing complex GPU shaders or vocabulary retraining, we will implement a custom Mojo matrix-vector kernel optimized specifically for small batches of tokens ($B \approx 16$) that route to the same expert.

### Step 1.1: Design the `q4_gemv_small_b` CPU Kernel
Instead of executing a horizontal SIMD reduction (`reduce_add()`) for every single token-weight multiplication in the batch (which dominates overhead when $B$ is small), we accumulate in registers across the batch:

1. **Keep $B$ SIMD8 Accumulators in CPU Registers**: Pre-allocate $B$ SIMD8 floating-point accumulators in host CPU registers.
2. **Load Weight Block Once**: Read a single 16-byte block of Q4 weights (unpacked into $4 \times \text{SIMD8}$ weight registers) from DDR4 once.
3. **Batch Vector Multiplications**: Multiply the single unpacked weight block by the $B$ different input vector blocks from the $B$ active tokens.
4. **Register Accumulation**: Accumulate results directly in the $B$ register-resident SIMD8 accumulators.
5. **Single Horizontal Reduction**: Perform a single `reduce_add()` per accumulator only at the very end of the row loop.

```
       Load Weight Block (Once)
               │
               ├─► Multiply with Token_0 Vector Block ─► Accumulate in Reg_0
               ├─► Multiply with Token_1 Vector Block ─► Accumulate in Reg_1
               │   ...
               └─► Multiply with Token_B Vector Block ─► Accumulate in Reg_B
                                   │
                                   ▼
                Single Horizontal Reduction at End of Row Loop
```

### Step 1.2: Restructure the MoE Dispatch Loop
To maximize L3 cache reuse of expert weights, we group token dispatches sequentially:
1. **Token Clustering**: For each layer, run the router to get the selected experts. Build an inverted index mapping each of the 128 experts to the list of tokens that selected it.
2. **Expert-Wise Sequential Execution**: Iterate through active experts sequentially. For each expert, gather its assigned tokens (average batch size $B \approx 16$) and dispatch them to the `q4_gemv_small_b` kernel.
3. **Weight Reuse**: This guarantees that the specific expert's weights are loaded into cache **once** and reused across all $B$ tokens before being evicted, restoring L3 cache efficiency.
4. **Closure Workaround**: Avoid nested closures inside the expert loops to prevent runtime compiler crashes; inline the worker execution directly or use a flat index array.

---

## Phase 2: Vocabulary Trimming to 32K (Reasoning Specialization)

Once the foundational CPU execution is highly optimized and near its theoretical floor, we implement vocabulary slicing to trigger a massive cascading optimization.

```
  Standard Gemma 4:  262,144 Vocab Size (Multilingual generalist)
  Specialized Code:   32,768 Vocab Size (88% Slashed Weight Overhead)
```

### Step 2.1: Slicing Embeddings and LM Head Layers
* **Storage Win**: Slicing the vocabulary from 262,144 down to 32,768 physically deletes unused multilingual rows, slashing the memory footprint of the Embedding and LM Head projection matrices from **~460 MB down to ~58 MB**.
* **Compute Win**: During self-conditioning, the engine calculates a continuous $256 \times V_{\text{vocab}}$ Softmax representation:
  * **262K Vocab**: Requires calculating, storing, and reducing a tensor of size $256 \times 262,144 = \mathbf{67,108,864}$ floating-point values (~268 MB). This streams heavy data across DDR4.
  * **32K Vocab**: Reduces the tensor to $256 \times 32,768 = \mathbf{8,388,608}$ values (~33.5 MB). The self-conditioning loop scales down by **$8\times$** and fits entirely inside the CPU's local cache levels.

### Step 2.2: Slicing and Fine-Tuning Execution
1. **Token Frequency Profiling**: Run logical reasoning and coding datasets (e.g. *The Stack v2*) through the tokenizer to identify unused multilingual tokens.
2. **GGUF Slicing**: Write a script to physically delete GGUF rows corresponding to the discarded tokens in the `embed_tokens.weight` and `lm_head.weight` matrices.
3. **Targeted Diffusion Recovery Fine-Tuning**: Run a brief, low-rank adaptation (LoRA) or head-only fine-tuning pass under a Discrete Uniform Noise Diffusion objective to teach the untouched reasoning core (experts and attention) to map logical representations to the trimmed 32K vocabulary indexes.

---

## Phase 3: Asymmetric APU Co-Execution (CPU + GPU Split)

With CPU execution optimized and the vocab footprint minimized, we activate the hybrid co-execution graph:

### Step 3.1: Zero-Copy SVM Mapping
Map weight buffers into Heterogeneous System Architecture (HSA) unified memory using `/dev/kfd` mapping so both processors access identical pointers with zero copying overhead.

### Step 3.2: Staggered Scheduling
1. **Denoise Steps 1 to 12 (Memory Clearance)**: Route 100% of layer execution to the Vega 8 iGPU. Place CPU host threads into an explicit wait state. This gives the iGPU exclusive, uncontested control of the physical 57 GB/s DDR4 bandwidth to execute heavy sparse MoE routing.
2. **Denoise Steps 12+ (L3 Cache Lock-In)**: Once the canvas converges and expert routing stabilizes, wake up the CPU threads. Let the iGPU handle Multi-Head Attention and dense layers while the CPU executes MoE experts entirely out of the local 16MB L3 cache, achieving concurrent dual-engine execution without memory bus thrashing.
