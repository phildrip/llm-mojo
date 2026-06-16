from std.memory import alloc
from std.math import sqrt, cos, sin, exp, tanh
import std.time
from std.algorithm import parallelize
from std.sys.info import num_physical_cores

# -----------------------------------------------------------------------------
# 1. Configuration & Data Structures
# -----------------------------------------------------------------------------

# Configuration for MTP (Multi-Token Prediction / Assistant)
struct MTPConfig:
    var embed_dim: Int          # d_assistant (assistant layer dimension, e.g., 1024)
    var backbone_dim: Int       # d_backbone (target backbone dimension, e.g., 2816)
    var num_layers: Int         # L_nextn (number of assistant layers, e.g., 4)
    var vocab_size: Int         # Vocab head size

    def __init__(out self, embed_dim: Int = 1024, backbone_dim: Int = 2816, num_layers: Int = 4, vocab_size: Int = 10000):
        self.embed_dim = embed_dim
        self.backbone_dim = backbone_dim
        self.num_layers = num_layers
        self.vocab_size = vocab_size

    @always_inline
    def get_pre_proj_size(self) -> Int:
        # Shape: [embed_dim, 2 * backbone_dim]
        return 2 * self.backbone_dim * self.embed_dim

    @always_inline
    def get_post_proj_size(self) -> Int:
        # Shape: [backbone_dim, embed_dim]
        return self.embed_dim * self.backbone_dim

    @always_inline
    def get_lm_head_size(self) -> Int:
        # Shape: [vocab_size, embed_dim]
        return self.vocab_size * self.embed_dim

# Weights for the entire MTP Assistant Model
struct MTPWeights(Copyable, Movable):
    var tok_embd: UnsafePointer[Float32, MutExternalOrigin]
    var pre_proj: UnsafePointer[Float32, MutExternalOrigin]
    var post_proj: UnsafePointer[Float32, MutExternalOrigin]
    var output_norm_weight: UnsafePointer[Float32, MutExternalOrigin]
    var lm_head: UnsafePointer[Float32, MutExternalOrigin]

    def __init__(
        out self,
        tok_embd: UnsafePointer[Float32, MutExternalOrigin],
        pre_proj: UnsafePointer[Float32, MutExternalOrigin],
        post_proj: UnsafePointer[Float32, MutExternalOrigin],
        output_norm_weight: UnsafePointer[Float32, MutExternalOrigin],
        lm_head: UnsafePointer[Float32, MutExternalOrigin]
    ):
        self.tok_embd = tok_embd
        self.pre_proj = pre_proj
        self.post_proj = post_proj
        self.output_norm_weight = output_norm_weight
        self.lm_head = lm_head

# Weights for a single Assistant Decoder Layer
struct AssistantLayerWeights(Copyable, Movable):
    var attn_norm_weight: UnsafePointer[Float32, MutExternalOrigin]
    var wq: UnsafePointer[Float32, MutExternalOrigin]
    var wo: UnsafePointer[Float32, MutExternalOrigin]
    var q_norm_weight: UnsafePointer[Float32, MutExternalOrigin]
    var attn_post_norm_weight: UnsafePointer[Float32, MutExternalOrigin]
    var out_scale: Float32
    var ffn_norm_weight: UnsafePointer[Float32, MutExternalOrigin]
    var ffn_gate: UnsafePointer[Float32, MutExternalOrigin]
    var ffn_up: UnsafePointer[Float32, MutExternalOrigin]
    var ffn_down: UnsafePointer[Float32, MutExternalOrigin]
    var ffn_post_norm_weight: UnsafePointer[Float32, MutExternalOrigin]

    def __init__(
        out self,
        attn_norm_weight: UnsafePointer[Float32, MutExternalOrigin],
        wq: UnsafePointer[Float32, MutExternalOrigin],
        wo: UnsafePointer[Float32, MutExternalOrigin],
        q_norm_weight: UnsafePointer[Float32, MutExternalOrigin],
        attn_post_norm_weight: UnsafePointer[Float32, MutExternalOrigin],
        out_scale: Float32,
        ffn_norm_weight: UnsafePointer[Float32, MutExternalOrigin],
        ffn_gate: UnsafePointer[Float32, MutExternalOrigin],
        ffn_up: UnsafePointer[Float32, MutExternalOrigin],
        ffn_down: UnsafePointer[Float32, MutExternalOrigin],
        ffn_post_norm_weight: UnsafePointer[Float32, MutExternalOrigin]
    ):
        self.attn_norm_weight = attn_norm_weight
        self.wq = wq
        self.wo = wo
        self.q_norm_weight = q_norm_weight
        self.attn_post_norm_weight = attn_post_norm_weight
        self.out_scale = out_scale
        self.ffn_norm_weight = ffn_norm_weight
        self.ffn_gate = ffn_gate
        self.ffn_up = ffn_up
        self.ffn_down = ffn_down
        self.ffn_post_norm_weight = ffn_post_norm_weight


# -----------------------------------------------------------------------------
# 2. Optimized MTP Math Kernels in Mojo (SIMD-16 Vectorized & Thread-Safe)
# -----------------------------------------------------------------------------

# Vectorized Matrix-Vector Multiplication (GEMV)
@always_inline
def gemv(
    mat: UnsafePointer[Float32, _],
    vec: UnsafePointer[Float32, _],
    out_ptr: UnsafePointer[Float32, MutExternalOrigin],
    rows: Int,
    cols: Int,
    num_tasks: Int,
):
    var tasks = num_tasks
    if tasks > rows:
        tasks = rows
    if tasks < 1:
        tasks = 1
    var chunk_size = rows // tasks

    def worker(t: Int) capturing -> None:
        var start_r = t * chunk_size
        var end_r = (t + 1) * chunk_size if t < tasks - 1 else rows
        for r in range(start_r, end_r):
            var acc = SIMD[DType.float32, 16](0.0)
            var row_offset = r * cols
            for c in range(0, cols, 16):
                acc = acc + mat.load[16](row_offset + c) * vec.load[16](c)
            out_ptr[r] = acc.reduce_add()

    if tasks == 1:
        worker(0)
    else:
        parallelize[worker](tasks)

# Vectorized RMSNorm
@always_inline
def rms_norm(
    x: UnsafePointer[Float32, _],
    w: UnsafePointer[Float32, _],
    out_ptr: UnsafePointer[Float32, MutExternalOrigin],
    size: Int,
):
    var acc = SIMD[DType.float32, 16](0.0)
    for i in range(0, size, 16):
        var x_v = x.load[16](i)
        acc = acc + x_v * x_v
    var sum = acc.reduce_add()
    var scale = 1.0 / sqrt(sum / Float32(size) + 1e-6)

    for i in range(0, size, 16):
        out_ptr.store[16](i, x.load[16](i) * scale * w.load[16](i))

# Vectorized RMSNorm without element-wise scaling
@always_inline
def rms_norm_noscale(
    x: UnsafePointer[Float32, _],
    out_ptr: UnsafePointer[Float32, MutExternalOrigin],
    size: Int,
):
    var acc = SIMD[DType.float32, 16](0.0)
    for i in range(0, size, 16):
        var x_v = x.load[16](i)
        acc = acc + x_v * x_v
    var sum = acc.reduce_add()
    var scale = 1.0 / sqrt(sum / Float32(size) + 1e-6)

    for i in range(0, size, 16):
        out_ptr.store[16](i, x.load[16](i) * scale)

# Standard Softmax
@always_inline
def softmax(scores: UnsafePointer[Float32, MutExternalOrigin], size: Int):
    var max_val = scores[0]
    for i in range(1, size):
        if scores[i] > max_val:
            max_val = scores[i]

    var sum: Float32 = 0.0
    for i in range(size):
        scores[i] = exp(scores[i] - max_val)
        sum += scores[i]

    for i in range(size):
        scores[i] /= sum

# Vectorized Rotary Position Embedding (RoPE)
@always_inline
def apply_rope(vec: UnsafePointer[Float32, MutExternalOrigin], num_heads: Int, head_dim: Int, pos: Int, is_global: Bool):
    var half_dim = head_dim // 2
    var base: Float32 = Float32(1000000.0) if is_global else Float32(10000.0)
    var rope_dim = 512 if is_global else 256

    for h in range(num_heads):
        var h_offset = h * head_dim
        for i in range(half_dim):
            var theta: Float32 = 0.0
            if (not is_global) or (i < 64):
                theta = Float32(pos) / (base ** (Float32(2 * i) / Float32(rope_dim)))

            var cos_theta = cos(theta)
            var sin_theta = sin(theta)

            var val1 = vec[h_offset + i]
            var val2 = vec[h_offset + half_dim + i]

            vec[h_offset + i] = val1 * cos_theta - val2 * sin_theta
            vec[h_offset + half_dim + i] = val2 * cos_theta + val1 * sin_theta

# Vectorized Gated-GELU activation
@always_inline
def gelu_and_mul(
    h1: UnsafePointer[Float32, _],
    h2: UnsafePointer[Float32, _],
    out_ptr: UnsafePointer[Float32, MutExternalOrigin],
    size: Int,
):
    for i in range(size):
        var x1 = h1[i]
        var x2 = h2[i]
        var inner = Float32(0.79788456) * (x1 + Float32(0.044715) * x1 * x1 * x1)
        var g = Float32(0.5) * x1 * (Float32(1.0) + tanh(inner))
        out_ptr[i] = g * x2

# Token embedding, scale, and concat
@always_inline
def mtp_embed_and_concat(
    token_id: Int,
    recurrent_h: UnsafePointer[Float32, _],
    embd_matrix: UnsafePointer[Float32, _],
    out_concat: UnsafePointer[Float32, MutExternalOrigin],
    backbone_dim: Int,
):
    var scale = sqrt(Float32(backbone_dim))
    var embd_offset = token_id * backbone_dim

    for i in range(0, backbone_dim, 16):
        var emb_v = embd_matrix.load[16](embd_offset + i) * scale
        out_concat.store[16](i, emb_v)

        var rec_v = recurrent_h.load[16](i)
        out_concat.store[16](backbone_dim + i, rec_v)

# Paged attention sharing directly with the target model's KV caches
@always_inline
def mtp_paged_attention(
    q: UnsafePointer[Float32, _],
    k_pool: UnsafePointer[Float32, _],
    v_pool: UnsafePointer[Float32, _],
    block_table: List[Int],
    block_size: Int,
    pos: Int,
    is_swa: Bool,
    attn_scores: UnsafePointer[Float32, MutExternalOrigin],
    out_attn: UnsafePointer[Float32, MutExternalOrigin],
):
    var num_heads = 16
    var num_kv_heads = 8
    var head_dim = 256
    var kv_stride = num_kv_heads * head_dim
    var ratio = num_heads // num_kv_heads

    var start_t = 0
    if is_swa:
        var window_limit = pos - 1023
        if window_limit > 0:
            start_t = window_limit

    for h in range(num_heads):
        var h_offset = h * head_dim
        var g = h // ratio
        var g_offset = g * head_dim

        # 1. Compute dot product attention scores
        for t in range(start_t, pos + 1):
            var t_logical_blk = t // block_size
            var t_blk_offset = t % block_size
            var t_phys_blk_id = block_table[t_logical_blk]
            var t_cache_offset = (t_phys_blk_id * block_size + t_blk_offset) * kv_stride + g_offset

            var score: Float32 = 0.0
            for d in range(head_dim):
                score += q[h_offset + d] * k_pool[t_cache_offset + d]
            attn_scores[t - start_t] = score

        # 2. Softmax
        var num_scores = pos + 1 - start_t
        softmax(attn_scores, num_scores)

        # 3. Compute weighted values
        for d in range(head_dim):
            var val_sum: Float32 = 0.0
            for t in range(start_t, pos + 1):
                var t_logical_blk = t // block_size
                var t_blk_offset = t % block_size
                var t_phys_blk_id = block_table[t_logical_blk]
                var t_cache_offset = (t_phys_blk_id * block_size + t_blk_offset) * kv_stride + g_offset
                val_sum += attn_scores[t - start_t] * v_pool[t_cache_offset + d]
            out_attn[h_offset + d] = val_sum

# -----------------------------------------------------------------------------
# 3. Assistant Decoder Layer Forward Pass
# -----------------------------------------------------------------------------

@always_inline
def run_assistant_layer(
    layer_idx: Int,
    x: UnsafePointer[Float32, MutExternalOrigin], # input/output layer state (embed_dim)
    weights: AssistantLayerWeights,
    block_table: List[Int],
    block_size: Int,
    k_pool: UnsafePointer[Float32, _],
    v_pool: UnsafePointer[Float32, _],
    pos: Int,
    num_tasks: Int,
    embed_dim: Int,
    intermediate_dim: Int,
    # Temp pre-allocated buffers
    norm_x: UnsafePointer[Float32, MutExternalOrigin],
    q: UnsafePointer[Float32, MutExternalOrigin],
    attn_scores: UnsafePointer[Float32, MutExternalOrigin],
    attn_out: UnsafePointer[Float32, MutExternalOrigin],
    proj_attn_out: UnsafePointer[Float32, MutExternalOrigin],
    ffn_h1: UnsafePointer[Float32, MutExternalOrigin],
    ffn_h2: UnsafePointer[Float32, MutExternalOrigin],
    ffn_out_gate: UnsafePointer[Float32, MutExternalOrigin],
    ffn_out: UnsafePointer[Float32, MutExternalOrigin],
):
    var is_global = (layer_idx == 3) # Layer 3 is full/global attention, layers 0,1,2 are sliding
    var num_heads = 16
    var head_dim = 256

    # 1. Attention Input Norm
    rms_norm(x, weights.attn_norm_weight, norm_x, embed_dim)

    # 2. Query Projection: gemv maps [embed_dim] to [num_heads * head_dim]
    gemv(weights.wq, norm_x, q, num_heads * head_dim, embed_dim, num_tasks)

    # 3. Head Norm on Q
    for h in range(num_heads):
        var head_ptr = q + h * head_dim
        rms_norm(head_ptr, weights.q_norm_weight, head_ptr, head_dim)

    # 4. Apply RoPE on Q
    apply_rope(q, num_heads, head_dim, pos, is_global)

    # 5. Paged Attention against shared target model's KV pools
    mtp_paged_attention(q, k_pool, v_pool, block_table, block_size, pos, not is_global, attn_scores, attn_out)

    # 6. Out Projection: gemv maps [num_heads * head_dim] back to [embed_dim]
    gemv(weights.wo, attn_out, proj_attn_out, embed_dim, num_heads * head_dim, num_tasks)

    # 7. Post attention layernorm
    rms_norm(proj_attn_out, weights.attn_post_norm_weight, proj_attn_out, embed_dim)

    # Residual Addition
    for i in range(embed_dim):
        x[i] += proj_attn_out[i]

    # 8. Feedforward
    rms_norm(x, weights.ffn_norm_weight, norm_x, embed_dim)

    # ffn_gate (W1) maps [embed_dim] to [intermediate_dim]
    gemv(weights.ffn_gate, norm_x, ffn_h1, intermediate_dim, embed_dim, num_tasks)

    # ffn_up (W3) maps [embed_dim] to [intermediate_dim]
    gemv(weights.ffn_up, norm_x, ffn_h2, intermediate_dim, embed_dim, num_tasks)

    # Gated-GELU activation
    gelu_and_mul(ffn_h1, ffn_h2, ffn_out_gate, intermediate_dim)

    # ffn_down (W2) maps [intermediate_dim] to [embed_dim]
    gemv(weights.ffn_down, ffn_out_gate, ffn_out, embed_dim, intermediate_dim, num_tasks)

    # ffn_post_norm
    rms_norm(ffn_out, weights.ffn_post_norm_weight, ffn_out, embed_dim)

    # Residual Add & Output scale
    for i in range(embed_dim):
        x[i] = (x[i] + ffn_out[i]) * weights.out_scale

# -----------------------------------------------------------------------------
# 4. MTP Generate Step (Single Autoregressive Speculative Step)
# -----------------------------------------------------------------------------

def mtp_generate_step(
    token_id: Int,
    recurrent_h: UnsafePointer[Float32, MutExternalOrigin],
    weights: MTPWeights,
    layers: List[AssistantLayerWeights],
    block_table: List[Int],
    block_size: Int,
    k_block_pools: List[UnsafePointer[Float32, MutExternalOrigin]],
    v_block_pools: List[UnsafePointer[Float32, MutExternalOrigin]],
    pos: Int,
    out_h_next: UnsafePointer[Float32, MutExternalOrigin],
    out_logits: UnsafePointer[Float32, MutExternalOrigin],
    CONFIG: MTPConfig,
    num_tasks: Int,
) raises:
    # 1. Allocate temporary workspace memory
    var concat_dim = 2 * CONFIG.backbone_dim
    var x_concat = alloc[Float32](concat_dim)
    var x_assistant = alloc[Float32](CONFIG.embed_dim)

    var norm_x = alloc[Float32](CONFIG.embed_dim)
    var q = alloc[Float32](16 * 256) # num_heads * head_dim
    var attn_scores = alloc[Float32](pos + 1)
    var attn_out = alloc[Float32](16 * 256)
    var proj_attn_out = alloc[Float32](CONFIG.embed_dim)
    
    var intermediate_dim = 8192
    var ffn_h1 = alloc[Float32](intermediate_dim)
    var ffn_h2 = alloc[Float32](intermediate_dim)
    var ffn_out_gate = alloc[Float32](intermediate_dim)
    var ffn_out = alloc[Float32](CONFIG.embed_dim)

    # 2. Embed and Concatenate
    mtp_embed_and_concat(token_id, recurrent_h, weights.tok_embd, x_concat, CONFIG.backbone_dim)

    # 3. Pre-Project to assistant dimension
    gemv(weights.pre_proj, x_concat, x_assistant, CONFIG.embed_dim, concat_dim, num_tasks)

    # 4. Run Assistant Layers
    var cur = x_assistant
    for il in range(CONFIG.num_layers):
        # Layer 0, 1, 2 map to target model layer 28 (SWA). Layer 3 maps to target layer 29 (Full).
        var pool_idx = 0 if il < 3 else 1
        var k_pool = k_block_pools[pool_idx]
        var v_pool = v_block_pools[pool_idx]

        run_assistant_layer(
            il, cur, layers[il], block_table, block_size, k_pool, v_pool, pos, num_tasks,
            CONFIG.embed_dim, intermediate_dim,
            norm_x, q, attn_scores, attn_out, proj_attn_out,
            ffn_h1, ffn_h2, ffn_out_gate, ffn_out
        )

    # 5. Output RMSNorm
    rms_norm(cur, weights.output_norm_weight, cur, CONFIG.embed_dim)

    # 6. Post-Project back to Target Backbone dimension for next recurrent state
    gemv(weights.post_proj, cur, out_h_next, CONFIG.backbone_dim, CONFIG.embed_dim, num_tasks)

    # 7. Project to vocab logits for speculative sampling
    gemv(weights.lm_head, cur, out_logits, CONFIG.vocab_size, CONFIG.embed_dim, num_tasks)

    # 8. Clean up workspace allocations
    x_concat.free()
    x_assistant.free()
    norm_x.free()
    q.free()
    attn_scores.free()
    attn_out.free()
    proj_attn_out.free()
    ffn_h1.free()
    ffn_h2.free()
    ffn_out_gate.free()
    ffn_out.free()

# -----------------------------------------------------------------------------
# 5. Verification & Benchmark Suite
# -----------------------------------------------------------------------------

def main() raises:
    var num_tasks = num_physical_cores()
    print("=========================================================")
    print("   GEMMA 4 MTP SPECEVAL DRAFT LOOP INFERENCE VERIFIER    ")
    print("=========================================================")
    print("Physical Cores Active: ", num_tasks)

    var CONFIG = MTPConfig()
    print("Assistant Layer Dim:   ", CONFIG.embed_dim)
    print("Target Backbone Dim:   ", CONFIG.backbone_dim)
    print("Target Vocabulary Size:", CONFIG.vocab_size)
    print("Assistant Layers:      ", CONFIG.num_layers)
    print("---------------------------------------------------------")

    # 1. Allocate Assistant Layer Parameters
    print("Allocating MTP Assistant model and layer weights...")
    var tok_embd = alloc[Float32](CONFIG.vocab_size * CONFIG.backbone_dim)
    var pre_proj = alloc[Float32](CONFIG.get_pre_proj_size())
    var post_proj = alloc[Float32](CONFIG.get_post_proj_size())
    var output_norm_weight = alloc[Float32](CONFIG.embed_dim)
    var lm_head = alloc[Float32](CONFIG.get_lm_head_size())

    # Build Layers weights lists
    var layer_attn_norm = alloc[Float32](CONFIG.num_layers * CONFIG.embed_dim)
    var layer_wq = alloc[Float32](CONFIG.num_layers * 4096 * CONFIG.embed_dim)
    var layer_wo = alloc[Float32](CONFIG.num_layers * CONFIG.embed_dim * 4096)
    var layer_q_norm = alloc[Float32](CONFIG.num_layers * 256)
    var layer_attn_post_norm = alloc[Float32](CONFIG.num_layers * CONFIG.embed_dim)
    var layer_ffn_norm = alloc[Float32](CONFIG.num_layers * CONFIG.embed_dim)
    var layer_ffn_gate = alloc[Float32](CONFIG.num_layers * 8192 * CONFIG.embed_dim)
    var layer_ffn_up = alloc[Float32](CONFIG.num_layers * 8192 * CONFIG.embed_dim)
    var layer_ffn_down = alloc[Float32](CONFIG.num_layers * CONFIG.embed_dim * 8192)
    var layer_ffn_post_norm = alloc[Float32](CONFIG.num_layers * CONFIG.embed_dim)

    # Initialize model structures
    var weights = MTPWeights(tok_embd, pre_proj, post_proj, output_norm_weight, lm_head)

    var layers = List[AssistantLayerWeights]()
    for il in range(CONFIG.num_layers):
        layers.append(
            AssistantLayerWeights(
                layer_attn_norm + il * CONFIG.embed_dim,
                layer_wq + il * 4096 * CONFIG.embed_dim,
                layer_wo + il * CONFIG.embed_dim * 4096,
                layer_q_norm + il * 256,
                layer_attn_post_norm + il * CONFIG.embed_dim,
                Float32(0.95),
                layer_ffn_norm + il * CONFIG.embed_dim,
                layer_ffn_gate + il * 8192 * CONFIG.embed_dim,
                layer_ffn_up + il * 8192 * CONFIG.embed_dim,
                layer_ffn_down + il * CONFIG.embed_dim * 8192,
                layer_ffn_post_norm + il * CONFIG.embed_dim
            )
        )

    # Allocate Mock shared Target Model KV Pools
    # Gemma 4 uses head_dim=256, num_kv_heads=8. kv_stride = 8 * 256 = 2048.
    var block_size = 16
    var max_blocks = 10
    var kv_stride = 8 * 256
    var pool_size = max_blocks * block_size * kv_stride

    var k_pool_28 = alloc[Float32](pool_size)
    var v_pool_28 = alloc[Float32](pool_size)
    var k_pool_29 = alloc[Float32](pool_size)
    var v_pool_29 = alloc[Float32](pool_size)

    var k_block_pools = List[UnsafePointer[Float32, MutExternalOrigin]]()
    k_block_pools.append(k_pool_28)
    k_block_pools.append(k_pool_29)

    var v_block_pools = List[UnsafePointer[Float32, MutExternalOrigin]]()
    v_block_pools.append(v_pool_28)
    v_block_pools.append(v_pool_29)

    # Setup Block Table mapping
    var block_table = List[Int]()
    block_table.append(2) # Map logical block 0 to physical block 2
    block_table.append(5) # Map logical block 1 to physical block 5

    # 2. Populate Weights and Pools Deterministically
    print("Populating parameters with deterministic non-zero weights...")
    for i in range(CONFIG.vocab_size * CONFIG.backbone_dim):
        tok_embd[i] = Float32(0.005) * Float32((i + 13) % 47)
    for i in range(CONFIG.get_pre_proj_size()):
        pre_proj[i] = Float32(0.0001) * Float32((i + 5) % 113)
    for i in range(CONFIG.get_post_proj_size()):
        post_proj[i] = Float32(0.0001) * Float32((i + 7) % 79)
    for i in range(CONFIG.embed_dim):
        output_norm_weight[i] = Float32(1.0)
    for i in range(CONFIG.get_lm_head_size()):
        lm_head[i] = Float32(0.001) * Float32((i + 3) % 61)

    # Populate layers
    for i in range(CONFIG.num_layers * CONFIG.embed_dim):
        layer_attn_norm[i] = Float32(1.0)
        layer_attn_post_norm[i] = Float32(1.0)
        layer_ffn_norm[i] = Float32(1.0)
        layer_ffn_post_norm[i] = Float32(1.0)

    for i in range(CONFIG.num_layers * 4096 * CONFIG.embed_dim):
        layer_wq[i] = Float32(0.0002) * Float32(i % 103)
        layer_wo[i] = Float32(0.0002) * Float32(i % 97)

    for i in range(CONFIG.num_layers * 256):
        layer_q_norm[i] = Float32(1.0)

    for i in range(CONFIG.num_layers * 8192 * CONFIG.embed_dim):
        layer_ffn_gate[i] = Float32(0.0001) * Float32(i % 83)
        layer_ffn_up[i] = Float32(0.0001) * Float32(i % 89)
        layer_ffn_down[i] = Float32(0.0001) * Float32(i % 73)

    # Populate KV Pools representing target model history context (e.g., prompt context of length 20)
    # This represents 20 tokens already decoded in the target model.
    var prompt_len = 20
    for t in range(prompt_len):
        var t_logical_blk = t // block_size
        var t_blk_offset = t % block_size
        var t_phys_blk_id = block_table[t_logical_blk]
        var t_cache_offset = (t_phys_blk_id * block_size + t_blk_offset) * kv_stride

        for d in range(kv_stride):
            k_pool_28[t_cache_offset + d] = Float32(0.01) * Float32((t * d) % 31)
            v_pool_28[t_cache_offset + d] = Float32(0.01) * Float32((t + d) % 23)
            k_pool_29[t_cache_offset + d] = Float32(0.01) * Float32((t * d + 3) % 29)
            v_pool_29[t_cache_offset + d] = Float32(0.01) * Float32((t + d + 5) % 19)

    print("Populated all arrays successfully.")
    print("---------------------------------------------------------")

    # 3. Execute Autoregressive 4-step MTP speculative drafting sequence
    print("[MTP Draft Phase] Starting autoregressive generation loop...")
    
    # Active inputs
    var active_token = 1423 # Target seed token
    var recurrent_h = alloc[Float32](CONFIG.backbone_dim)
    for i in range(CONFIG.backbone_dim):
        recurrent_h[i] = Float32(0.01) * Float32(i % 100) # Target seed recurrent state (post final RMSNorm)

    # Output buffers
    var out_h_next = alloc[Float32](CONFIG.backbone_dim)
    var out_logits = alloc[Float32](CONFIG.vocab_size)

    var draft_tokens = List[Int]()
    draft_tokens.append(active_token)

    var total_time: Float64 = 0.0
    var num_steps = 4

    for k in range(num_steps):
        # In MTP, the step generating speculative token predicted at position p + k + 1 
        # utilizes the shared KV cache at prompt context position pos = prompt_len - 1
        var pos = prompt_len - 1

        print("  Step ", k + 1, " | Draft Token ID: ", active_token, " | Position: ", pos)
        
        var t0 = std.time.perf_counter()
        
        mtp_generate_step(
            active_token, recurrent_h, weights, layers, block_table, block_size,
            k_block_pools, v_block_pools, pos, out_h_next, out_logits, CONFIG, num_tasks
        )
        
        var t1 = std.time.perf_counter()
        var elapsed = (t1 - t0) * 1000.0 * 1000.0
        total_time += elapsed
        
        # Sample next draft token using greedy argmax on out_logits
        var sampled_tok = 0
        var max_logit = out_logits[0]
        for i in range(1, CONFIG.vocab_size):
            if out_logits[i] > max_logit:
                max_logit = out_logits[i]
                sampled_tok = i

        print("    Generated Candidate Token ID: ", sampled_tok, " (Logit: ", max_logit, ") in ", elapsed, " microseconds.")
        print("    First 3 features of recurrent h_next: [", out_h_next[0], ", ", out_h_next[1], ", ", out_h_next[2], "]")

        # Propagate autoregressively
        active_token = sampled_tok
        draft_tokens.append(active_token)
        for i in range(CONFIG.backbone_dim):
            recurrent_h[i] = out_h_next[i]

    print("---------------------------------------------------------")
    print("  Autoregressive drafting sequence finished successfully!")
    print("  Draft tokens produced: [", draft_tokens[0], ", ", draft_tokens[1], ", ", draft_tokens[2], ", ", draft_tokens[3], ", ", draft_tokens[4], "]")
    print("  Average step generation latency: ", total_time / Float64(num_steps), " microseconds.")
    print("=========================================================")

    # Clean Up
    tok_embd.free()
    pre_proj.free()
    post_proj.free()
    output_norm_weight.free()
    lm_head.free()
    layer_attn_norm.free()
    layer_wq.free()
    layer_wo.free()
    layer_q_norm.free()
    layer_attn_post_norm.free()
    layer_ffn_norm.free()
    layer_ffn_gate.free()
    layer_ffn_up.free()
    layer_ffn_down.free()
    layer_ffn_post_norm.free()
    k_pool_28.free()
    v_pool_28.free()
    k_pool_29.free()
    v_pool_29.free()
    recurrent_h.free()
    out_h_next.free()
    out_logits.free()
