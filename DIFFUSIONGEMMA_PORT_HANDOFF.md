# DiffusionGemma Port — Handoff Spec

> **Purpose:** Fresh-context handoff for continuing the DiffusionGemma inference
> port in Mojo. Everything needed to resume work without re-reading this
> project's history. Paste this whole file into a new chat as the opening
> message along with "let's continue the DiffusionGemma port".

---

## Project

`/home/phil/git/llm-mojo` — Mojo inference engine for Gemma 4 26B MoE on AMD
Ryzen 7 5700G APU (8-core Zen 3 + Vega 8 iGPU, 64GB DDR4-3200, ~57 GB/s shared
bandwidth). Mojo 1.0.0b2 via pixi. Sandbox/experimental code, not production.

## What's already done (committed, working)

**Commit `a6ccc94`** ("feat: DiffusionGemma converter + scaffolded
model_diffusion.mojo") contains:

### 1. Checkpoint converter (`convert_diffusiongemma.py`)

- Downloads `google/diffusiongemma-26B-A4B-it` from HuggingFace (gated repo,
  needs `huggingface-cli login` — already authenticated on this machine).
- Filters safetensors to `model.decoder.*` only (skips vision tower under
  `model.encoder.vision_tower.*` and `model.encoder.language_model.*` — that's
  a non-MoE encoder pathway we don't need for text-only diffusion).
- Quantizes BF16 → Q4_0 (block size 32, scales fp32, block-max / 7.0).
- Writes 4 flat binary files in the format expected by the existing
  `model_speculative_moe.mojo` weight loader:
  - `diffusiongemma_q4.bin` — 15,311,790,080 bytes (byte-identical size to
    `gemma4_moe_weights_q4.bin`, same per-layer layout)
  - `diffusiongemma_embed_q4.bin` — 440 MB (vocab=262144 × embed_dim=2816, q4)
  - `diffusiongemma_sc_q4.bin` — 11,166,696 bytes (self-conditioning MLP)
  - `diffusiongemma_final_norm.bin` — 11,264 bytes (fp32 RMSNorm weight)
- All 4 files are on disk and verified loadable.

### 2. `model_diffusion.mojo` — working scaffold

Forked from `model_speculative_moe.mojo`. Compiles on Mojo 1.0.0b2. Currently:

- Loads all 4 DiffusionGemma weight files into RAM.
- Allocates and lays out self-conditioning MLP pointers:
  - `sc_gate_packed`, `sc_up_packed`, `sc_down_packed` — q4 packed bytes
  - `sc_gate_scales`, `sc_up_scales`, `sc_down_scales` — fp32 scales
  - `sc_pre_norm` — fp32 (2816 elements)
  - Note: `post_norm` has no learned weight (vLLM: `RMSNorm(has_weight=False)`)
- Still runs the **AR prefill chat loop** (single-token generation per step).
  Produces coherent output: "Cats are graceful and independent companions." at
  ~4.2 tok/s — proves all layer weights load correctly and the decoder backbone
  works. This is the scaffold to replace.

### 3. Mojo 1.0.0b2 migration done

- `pixi.toml`: `mojo = "1.0.0b2.*"`
- FFI uses `thin abi("C")` syntax:
  `libc.get_function[def(Int, Int32, Int32) thin abi("C") -> Int32]("open")`
- Closure captures need explicit convention:
  `def print_row(name: String, x: Float64) {read total, read count}:`
- `model_speculative_moe.mojo` migrated fully. Other model files
  (`model.mojo`, `model_q4.mojo`, etc.) **NOT migrated** — they still use 0.26.2
  FFI syntax and won't build on 1.0.0b2. Don't touch them unless needed.

---

## Reference: vLLM's DiffusionGemma implementation

Primary source of truth, cloned at `~/git/vllm`:

- **`vllm/model_executor/models/diffusion_gemma.py`** (1382 lines) — full
  implementation. Key sections:
  - `class DiffusionGemmaSelfConditioning` (lines 69-95) — SC MLP forward:
    `x = pre_norm(soft_embeds); sc = down_proj(gelu(gate_proj(x)) * up_proj(x));
    return post_norm_no_weight(inputs_embeds + sc)`. structurally identical to
    Gemma4MLP but `post_norm` has no learned scale.
  - `class DiffusionGemmaForConditionalGeneration` (line 140+) — main model.
    Notable: `text_config.attention_k_eq_v = True` (full-attention layers have
    NO v_proj weight; V is aliased from K).
  - `_compiled_sample_step(...)` (lines 503-660) — **the entire diffusion
    sampler as one compiled function**. This is what we need to port. Phases:
    1. Temperature schedule (line 526-527):
       `temp = t_min + (t_max - t_min) * (remaining / max_steps)` where
       `remaining = (max_steps - steps).clamp(min=1)`. With `t_min=0.4`,
       `t_max=0.8`: temp decreases from 0.8 → 0.4 across denoise steps. **This
       is mathematically identical to the spec's
       `τ_t = 0.8 - 0.4 * (t/T)`**, just written against remaining not elapsed.
    2. Gumbel-max sampling (lines 533-542): `argmax(logits/T + Gumbel)` when
       T>0, plain argmax when T==0. We can simplify to plain categorical
       sampling if Gumbel is awkward in Mojo — output quality is similar.
    3. Probs + entropy (lines 545-553): `log_probs = scaled.log_softmax(-1);
       probs = log_probs.exp(); token_entropy = -(probs * log_probs).sum(-1);
       mean_entropy = token_entropy.mean(-1)`. Confidence flag:
       `mean_entropy < confidence_threshold` (0.005).
    4. Entropy-bound acceptance mask (lines 556-560): sort positions by
       ascending entropy, cumsum, take positions where
       `(cumsum - cummax) <= entropy_bound` (0.1). These get the sampled
       token (lock-in); the rest get re-noised to random vocab ids.
    5. Phase transition (lines 568-605): is_encoder_phase flips to True when
       stable AND confident AND history_len >= ST; or when step >= max_steps.
       Reset step to 0 on commit.
    6. History buffer (lines 590-605): ST slots (ST=2 since
       `stability_threshold=1`), tracks argmax across recent steps. Stability
       = all ST slots agree (mismatch == 0).
    7. Self-conditioning soft embed (lines 634-651):
       `soft_embeds = probs @ embed_tokens.weight`. Store to `sc_embeds` buffer
       for the next step's SC MLP. **Skip for slots that will commit next**.
    8. Canvas update (lines 582-586, 654-656): commit → random reinit of
       canvas; denoise → accept/renoise; newly converged → overwrite canvas
       with argmax (clean snap).
- **`vllm/config/diffusion.py`** (26 lines) — `DiffusionConfig` dataclass
  with `canvas_length` and `max_denoising_steps` fields.
- **`vllm/transformers_utils/configs/diffusion_gemma.py`** (44 lines) —
  Python config class; key insight: `_init_text_config` sets
  `self.attention_k_eq_v = True` and `self.enable_moe_block = True` if
  num_experts is present.

---

## Reference: DiffusionGemma config constants (from HF)

From `config.json` (text_config section):

```
hidden_size              = 2816  (embed_dim)
head_dim                 = 256   (sliding) / 512 (global)
num_attention_heads      = 16
num_key_value_heads      = 8   (sliding) / 2 (global)
intermediate_size        = 2112  (shared MLP)
moe_intermediate_size    = 704   (per-expert)
num_experts              = 128
top_k_experts            = 8
num_hidden_layers        = 30
layer_types              = [sliding × 5, global × 1] × 5  → 25 sliding + 5 global
                         (is_global_layer(i) = i % 6 == 5)
sliding_window           = 1024
vocab_size               = 262144
tie_word_embeddings       = true  (LM head shares embed_tokens.weight)
final_logit_softcapping   = 30.0
rope: full_attention rope_theta=1e6, partial_rotary_factor=0.25
rope: sliding_attention rope_theta=1e4, default type
```

From `generation_config.json` (sampler constants — **your spec's numbers
match exactly**):

```
canvas_length             = 256
max_denoising_steps       = 48
t_max                     = 0.8
t_min                     = 0.4
entropy_bound             = 0.1
confidence_threshold      = 0.005
stability_threshold        = 1   (history buffer depth ST = stability_threshold + 1 = 2)
logit_softcap             = 30.0  (apply tanh(logits/30) * 30 before sampling)
```

`self_conditioning_size = 2112` (from safetensors header — same as
`standard_mlp_dim`, coincidence).

---

## Reference: weight tensor layout

Tensors extracted (filtered to `model.decoder.*`, text-only):

```
decoder.embed_tokens.weight                              [262144, 2816] BF16
decoder.layers.{0..29}.self_attn.{q,k}_proj.weight       [.., 2816] BF16
decoder.layers.{0..29}.self_attn.v_proj.weight           MISSING on global layers (i % 6 == 5)
decoder.layers.{0..29}.self_attn.o_proj.weight           [2816, ..] BF16
decoder.layers.{0..29}.self_attn.{q,k}_norm.weight       [head_dim] BF16
decoder.layers.{0..29}.mlp.{gate,up,down}_proj.weight    standard_mlp_dim sizes
decoder.layers.{0..29}.experts.gate_up_proj              [128, 1408, 2816] BF16 (packed gate+up)
decoder.layers.{0..29}.experts.down_proj                 [128, 2816, 704]  BF16
decoder.layers.{0..29}.router.{proj.weight, scale, per_expert_scale}
decoder.layers.{0..29}.layer_scalar                       [1] BF16
decoder.layers.{0..29}.{input, post_attention, pre_feedforward,
                              pre_feedforward_2, post_feedforward,
                              post_feedforward_1, post_feedforward_2}_layernorm.weight
decoder.norm.weight                                       [2816] BF16 (final RMSNorm)
decoder.self_conditioning.{gate, up, down}_proj.weight   [2112 or 2816] BF16
decoder.self_conditioning.pre_norm.weight                 [2816] BF16
```

Binary layout per layer in `diffusiongemma_q4.bin` (matches
`get_layer_weights` in `model_diffusion.mojo`):

```
[packed weights: wq | wk | (wv if not global) | wo | w_gate | w_up | w_down |
                 w_router | experts_gate_up | experts_down]      (q4, 4 bits/elt)
[scales for each, in same order]                                   (fp32, 1 per 32 elts)
[fp32 unchanged: input_ln | post_attn_ln | pre_ff_ln | pre_ff_ln_2 |
                  post_ff_ln | post_ff_ln_1 | post_ff_ln_2 |
                  q_norm | k_norm | router_scale | router_per_expert_scale |
                  layer_scalar]                                    (all fp32)
pad to 4096-byte boundary
```

SC weight file layout (`diffusiongemma_sc_q4.bin`):

```
[sc_gate_packed][sc_up_packed][sc_down_packed]    (q4, SC_PACKED = 2,970,496 B each)
[sc_gate_scales][sc_up_scales][sc_down_scales]    (fp32, SC_SCALES = 185,656 floats each)
[sc_pre_norm.weight]                              (fp32, 2816 elements)
pad to 4096-byte boundary
```

Constants already defined in `model_diffusion.mojo`:

```mojo
comptime SC_MATRIX_SIZE = 2112 * 2816  # 5,940,992 elements per matrix
comptime SC_PACKED = SC_MATRIX_SIZE // 2       # 2,970,496 bytes per matrix
comptime SC_SCALES = SC_MATRIX_SIZE // 32      # 185,656 floats per matrix
comptime SC_SCALES_BYTES = SC_SCALES * 4
comptime SC_PRE_NORM_BYTES = 2816 * 4
comptime SC_TOTAL = (SC_PACKED * 3) + (SC_SCALES_BYTES * 3) + SC_PRE_NORM_BYTES
comptime SC_ALIGNED = ((SC_TOTAL + 4095) // 4096) * 4096
```

---

## Remaining work (ordered, smallest first)

### Step 1: Self-conditioning MLP forward (~50 lines, easy)

Add a `def sc_forward(...)` to `model_diffusion.mojo`. Uses
`q4_gemv_cpu` for the three projections and `gelu_and_mul` from the
existing math kernel section. Pseudocode:

```
def sc_forward(
    inputs_embeds: UnsafePointer[Float32],  # [embed_dim] — current canvas token embedding
    soft_embeds:   UnsafePointer[Float32],  # [embed_dim] — from previous step's sampler
    sc_gate_packed, sc_up_packed, sc_down_packed,
    sc_gate_scales, sc_up_scales, sc_down_scales,
    sc_pre_norm,
    buf_normed:   UnsafePointer[Float32],  # scratch [embed_dim]
    buf_gate:      UnsafePointer[Float32],  # scratch [sc_size=2112]
    buf_up:        UnsafePointer[Float32],  # scratch [sc_size=2112]
    buf_gated:     UnsafePointer[Float32],  # scratch [sc_size=2112]
    buf_down:      UnsafePointer[Float32],  # scratch [embed_dim]
    out:           UnsafePointer[Float32],  # [embed_dim] — write result here
    embed_dim: Int, sc_size: Int, num_tasks: Int,
):
    # 1. pre_norm the soft_embeds (RMSNorm with sc_pre_norm weight)
    rms_norm(soft_embeds, sc_pre_norm, buf_normed, embed_dim)
    # 2. gate_proj and up_proj (q4 matmuls)
    q4_gemv_cpu(sc_gate_packed, sc_gate_scales, buf_normed, buf_gate,
                embed_dim, sc_size, num_tasks)
    q4_gemv_cpu(sc_up_packed,   sc_up_scales,   buf_normed, buf_up,
                embed_dim, sc_size, num_tasks)
    # 3. gelu_and_mul (in-place: buf_gate = gelu(buf_gate) * buf_up)
    gelu_and_mul(buf_gate, buf_up, buf_gated, sc_size)
    # 4. down_proj
    q4_gemv_cpu(sc_down_packed, sc_down_scales, buf_gated, buf_down,
                sc_size, embed_dim, num_tasks)
    # 5. residual + post_norm (no learned weight — rms_norm_noscale)
    for i in range(embed_dim):
        out[i] = inputs_embeds[i] + buf_down[i]
    rms_norm_noscale(out, out, embed_dim)
```

Test by running a canvas forward (Step 3) once with all-zero soft_embeds and
zero SC output — should match no-SC baseline. Then with soft_embeds =
`probs @ embed_weight` from a prior step — verify output changes sensibly.

### Step 2: Bidirectional attention path (modify existing layer forward)

The existing per-layer attention loop in `main()` (around lines 760-790 of
`model_diffusion.mojo`) does causal:

```mojo
for t in range(pos + 1):
    var t_cache_offset = layer_cache_offset + t * kv_stride + g_offset
    var score: Float32 = 0.0
    for d in range(layer_head_dim):
        score += q[h_offset + d] * k_cache[t_cache_offset + d]
    attn_scores[t] = score
softmax(attn_scores, pos + 1)
```

For decoder mode, replace `pos + 1` with `canvas_len` (256). The query slot
attends across all canvas positions symmetrically. KV cache is **read-only**
in decoder mode — do NOT write `k_cache[step_cache_offset + i] = k_vec[i]` for
canvas tokens during denoise. The encoder's KV from prefill is what we read.

Implementation: add an `is_decoder: Bool` parameter to the per-token forward
function (currently inlined into `main()` — refactor to a `def forward_token`
function or use a closure with captured flag). When `is_decoder`:

- Loop range becomes `range(canvas_len)` not `range(pos + 1)`
- Softmax size becomes `canvas_len`
- KV writes skipped
- RoPE positions still use `pos` (the canvas slot index, 0..canvas_len-1)

### Step 3: Canvas-mode batched forward (~500 lines or refactor)

Run a full 256-token forward pass through all 30 layers, producing logits
`[256, 262144]`. Two implementation strategies:

- **(A) Sequential per-token, no batching**: call the existing per-token
  forward 256 times. Wastes weight-bandwidth (each weight read once per
  token, not amortized across 256). Easy to wire up — same code, looped.
- **(B) Batched matmul**: requires `q4_gemv_batch` variant taking
  `[B=256, c_in]` x-matrix instead of `[c_in]` vector. Weight reads happen
  ONCE per layer regardless of B. **This is where diffusion's bandwidth
  amplification actually comes from**.

Recommend **(A) for correctness first**, then optimize to (B) once the
sampler works end-to-end. The audit showed AR per-token is ~195 ms, so a
canvas pass via (A) is ~50 seconds. Painful but workable for development.
Switching to (B) drops that to ~2-3 seconds (theoretical 100× amortization
on weight reads, less in practice).

### Step 4: Diffusion sampler (~300 lines, intricate)

Port `_compiled_sample_step` from
`vllm/model_executor/models/diffusion_gemma.py:503-660`. See reference above
for the phase-by-phase breakdown. Mojo implementation will use
`UnsafePointer[Float32]` buffers in place of torch tensors, with explicit
scratch allocations:

State buffers (allocate once at startup, size per active request):

```
canvas         [canvas_len=256]    Int32   — current canvas token ids
argmax_canvas  [canvas_len=256]    Int32   — last step's argmax (for commit output)
step           scalar              Int     — denoise step counter (0..max_steps)
is_encoder     scalar              Bool    — phase flag (True = causal prefill next)
confident      scalar              Bool    — mean-entropy < 0.005
history        [ST=2][canvas_len]  Int32   — last ST argmax canvases for stability
hist_len       scalar              Int     — count of valid history slots (0..ST)
sc_embeds      [canvas_len][embed_dim]  Float32  — soft embed for next step's SC MLP
```

Sampler per-step algorithm (given model logits `[canvas_len, vocab]`):

1. Compute `temp = t_min + (t_max - t_min) * max(max_steps - step, 1) / max_steps`
2. Scale logits / temp (clamp temp ≥ 1e-10)
3. Gumbel-max sample: `argmax(scaled + gumbel_noise)` if temp > 0 else
   `argmax(scaled)` (simplest Mojo path: use `np.random` via FFI for the
   Gumbel noise — adds one Python call per step, acceptable for now)
4. Also compute clean argmax (no noise) — that's `argmax_tokens`
5. Log-softmax of scaled, exp to probs
6. `token_entropy[t] = -sum_i probs[t,i] * log_probs[t,i]` for each canvas position
7. `mean_entropy = mean(token_entropy)`
8. `confident = mean_entropy < 0.005`
9. Entropy-bound mask:
   - sort positions by ascending `token_entropy` → sorted_idx, sorted_ent
   - cumsum the sorted entropies
   - also cummax the sorted entropies
   - accept positions where `(cumsum - cummax) <= 0.1`
   - scatter back to original indexing → `eb_mask[canvas_len]`
10. For positions where `eb_mask=true`: lock in `new_tokens[t]`
    For positions where `eb_mask=false`: re-noise to random vocab id
11. Update `canvas = new_tokens` (or random for commit branch)
12. Write `argmax_tokens` to history slot `hist_len % ST`; increment hist_len
    (cap at ST)
13. Stability check: history[0] == history[1] (if hist_len >= ST)
14. Convergence = `(stable AND confident AND hist_len >= ST) OR (step >= max_steps)`
15. If converged and currently denoising:
    - Set `is_encoder = True` (next forward is commit / causal)
    - Sampled output = `argmax_canvas` (the locked-in cleaned canvas)
16. If currently committing:
    - Reset `step = 0`, `hist_len = 0`, `is_encoder = False`
    - Re-initialize `canvas` with random vocab ids
17. Update `sc_embeds`:
    - `sc_embeds[t] = probs[t] @ embed_tokens.weight` (one row of LM-head
      embedding matrix multiplied by the [vocab] probs vector — gives a
      [embed_dim] column)
    - Only store for slots that just denoised AND are not about to commit
      (mask: `is_denoise && !is_encoder_phase`)
    - This is the SC input for step t+1's SC MLP forward.
18. `step += 1` if denoising; reset to 0 if just committed.

**Subtle but critical**: step 17 does NOT compute `probs @ embed_weight` for
the whole vocab × embed matrix — that's
`[canvas_len, vocab] @ [vocab, embed_dim]` =
`256 × 262144 × 2816` flops per step which is 1.9 GFLOP transient. Fine
computationally, but the buffer is **NOT** `[canvas_len, vocab]` — store
the `[canvas_len, embed_dim]` result directly, computed from probs as
`[canvas_len, vocab] @ embed_tokens.weight` (lookup via the existing
`Q4Embedding` in `[vocab, embed_dim]` form). Use `q4_gemv_cpu` with the
probs vector acting as the "x" and embed_tokens.weight acting as the
weight matrix.

### Step 5: Phase state machine + chat loop (~100 lines)

Outer driver:

```
1. Tokenize user prompt → prompt_tokens (length P)
2. Encoder prefill: causal forward over prompt_tokens[0..P-1].
   KV cache written for positions 0..P-1.
3. Initialize canvas: np.random.randint(0, vocab, canvas_len) → 256 random tokens
   Place canvas at positions P..P+canvas_len-1 in the cache layout.
   sc_embeds = zeros
   step = 0, is_encoder = False, hist_len = 0
4. While not commit_complete:
     - Compute canvas embeddings (token id → embed) for positions P..P+255
     - Add SC signal: inputs_embeds = sc_forward(canvas_embeds, sc_embeds)
     - Run decoder forward (bidirectional attention, KV read-only)
       → logits[256, vocab]
     - Apply logit softcap: tanh(logits / 30) * 30
     - Call sampler step (above) → updates canvas, sc_embeds, step,
       is_encoder, possibly sets commit_complete
     - If sampler says converged: break inner loop
5. Commit: emit argmax_canvas (256 tokens) as model output
   Decode via tokenizer and print
6. Optional: update KV cache for positions P..P+255 with causal forward
   (encoder mode on the argmax_canvas) so the next canvas block extends
   context. vLLM does this; can skip for first version.
7. Loop back to step 2 for next user turn.
```

Self-conditioning buffer init on first denoise step (no previous probs):
use zeros — equivalent to no SC signal. vLLM does this implicitly via
`sc_keep` mask being 0 on first denoise.

---

## Hardware budget (don't forget)

- AMD Ryzen 7 5700G APU, 8 cores / 16 threads, 64GB DDR4-3200, ~57 GB/s
  shared bandwidth (40 GB/s realistic when contended).
- Vega 8 iGPU (gfx90c) available via direct `/dev/kfd`
  (`test_kernels.mojo:1056-1889` has a working KFD FFI binding — could be
  reused for GPU dispatch but not required for first CPU-only port).
- AR decode is ~4.5 tok/s on `model_speculative_moe.mojo`. Diffusion with
  un-batched canvas forward (strategy A above) ≈ 0.4 canvas/sec which
  becomes ~100 tok/s if the canvas converges in <10 denoise steps. With
  real batched matmul (strategy B) that should jump 20-50×.

## Gotchas already learned

- **Nested `parallelize` is broken** in current Mojo — oversubscribes the
  shared pool. Keep all `parallelize` calls single-level. The repo's
  existing pattern (8 parallel experts, 1 thread each, 1 outer
  `parallelize`) is correct.
- **`std.sys.intrinsics.prefetch` exists on 1.0.0b2** — we never used it
  for AR but could help diffusion's batched matmul. Address as a perf
  optimization later, not at port time.
- **`MutExternalOrigin` deprecated in 1.0.0b2** — warnings only, use
  `MutUntrackedOrigin` for new code (existing code still compiles with
  warnings).
- **Mojo's user-load-friendly iteration convention**: avoid `b""` literal
  (not Mojo); use `builtins.bytes()` via FFI or pre-allocate a small
  empty Python bytes object. For Python tuple returns, use Python lists
  and `[0]`/`[1]` indexing — Mojo's `__getitem__` doesn't bind cleanly to
  Python tuples.
- **AR profiler categories** (from `model_speculative_moe.mojo:1001-1033`):
  embed reconstruct, attn norm, qkv proj, head norm + rope + cache,
  attn dot-prod/softmax, wo proj, shared mlp, moe router, moe experts,
  moe combine, final norm, logit proj, misc. Worth porting to the new
  file as we go for diagnostic value.

## Files to read first when resuming

1. `model_diffusion.mojo` **lines 1-515** — config structs, math kernels,
   `q4_gemv_cpu` (all reusable as-is).
2. `model_diffusion.mojo` **lines 516-end** — `main()` with weight loading,
   SC allocation (lines ~615-670), KV cache layout, AR chat loop (to be
   replaced).
3. `~/git/vllm/vllm/model_executor/models/diffusion_gemma.py` **lines
   503-660** — `_compiled_sample_step`. The single most important
   reference for the sampler port.
4. `~/git/vllm/vllm/model_executor/models/diffusion_gemma.py` **lines
   69-95** — `DiffusionGemmaSelfConditioning.forward` — exact SC MLP
   formula.
5. `convert_diffusiongemma.py` (this repo) — confirms binary layout writes
   match what the loader expects.

## Suggested first concrete step on resume

Write the self-conditioning MLP forward as `def sc_forward(...)` in
`model_diffusion.mojo` (Step 1 above). Test it in isolation: zero
soft_embeds → output should equal `inputs_embeds` (no signal added). Then
add the bidirectional attention path (Step 2) as an `is_decoder` flag
threaded through the per-layer loop. Verify by running a forward on a
256-token canvas and printing per-token norms — decoder mode norms should
match encoder mode for prefill positions 0..P-1 (the encoder KV is read
identically in both modes; only canvas positions P..P+255 differ in
attention range).

Then move to Step 3 (canvas batched forward, strategy A first) and Step 4
(sampler). Save Step 3→B optimization (batched `q4_gemv_batch`) until the
sampler works end-to-end. The 50 sec/canvas via Strategy A is fine for
correctness debugging.