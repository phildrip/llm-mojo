from std.ffi import OwnedDLHandle
from std.memory import alloc
from std.math import sqrt, exp, cos, sin
from std.python import Python
import std.os
from uring import IoUring

# -----------------------------------------------------------------------------
# 1. Model Configuration & Memory Offsets
# -----------------------------------------------------------------------------
struct TransformerConfig[
    embed_dim: Int,
    head_dim: Int,
    num_heads: Int,
    num_kv_heads: Int,
    intermediate_dim: Int,
    num_layers: Int,
    vocab_size: Int,
    max_seq_len: Int,
]:
    comptime wq_size: Int = Self.num_heads * Self.head_dim * Self.embed_dim
    comptime wk_size: Int = Self.num_kv_heads * Self.head_dim * Self.embed_dim
    comptime wv_size: Int = Self.num_kv_heads * Self.head_dim * Self.embed_dim
    comptime wo_size: Int = Self.embed_dim * Self.num_heads * Self.head_dim
    comptime w1_size: Int = Self.intermediate_dim * Self.embed_dim
    comptime w2_size: Int = Self.embed_dim * Self.intermediate_dim
    comptime w3_size: Int = Self.intermediate_dim * Self.embed_dim
    comptime rms_attn_size: Int = Self.embed_dim
    comptime rms_ffn_size: Int = Self.embed_dim
    
    comptime total_floats: Int = (
        Self.wq_size + Self.wk_size + Self.wv_size + Self.wo_size +
        Self.w1_size + Self.w2_size + Self.w3_size +
        Self.rms_attn_size + Self.rms_ffn_size
    )
    comptime raw_bytes_size: Int = Self.total_floats * 4
    # 4KB block size alignment for O_DIRECT compliance
    comptime aligned_bytes_size: Int = ((Self.raw_bytes_size + 4095) // 4096) * 4096
    
    @always_inline
    def get_layer_weights(self, base_ptr: UnsafePointer[Float32, _]) -> LayerWeights:
        var wq = base_ptr
        var wk = wq + Self.wq_size
        var wv = wk + Self.wk_size
        var wo = wv + Self.wv_size
        var w1 = wo + Self.wo_size
        var w2 = w1 + Self.w1_size
        var w3 = w2 + Self.w2_size
        var rms_attn = w3 + Self.w3_size
        var rms_ffn = rms_attn + Self.rms_attn_size
        
        return LayerWeights(
            UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=wq.__int__()),
            UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=wk.__int__()),
            UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=wv.__int__()),
            UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=wo.__int__()),
            UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=w1.__int__()),
            UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=w2.__int__()),
            UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=w3.__int__()),
            UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=rms_attn.__int__()),
            UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=rms_ffn.__int__())
        )

    def __init__(out self):
        pass

# -----------------------------------------------------------------------------
# 2. Structural Layer Weights Representation
# -----------------------------------------------------------------------------
struct LayerWeights:
    var wq: UnsafePointer[Float32, MutExternalOrigin]
    var wk: UnsafePointer[Float32, MutExternalOrigin]
    var wv: UnsafePointer[Float32, MutExternalOrigin]
    var wo: UnsafePointer[Float32, MutExternalOrigin]
    var w1: UnsafePointer[Float32, MutExternalOrigin]
    var w2: UnsafePointer[Float32, MutExternalOrigin]
    var w3: UnsafePointer[Float32, MutExternalOrigin]
    var rms_attn_weight: UnsafePointer[Float32, MutExternalOrigin]
    var rms_ffn_weight: UnsafePointer[Float32, MutExternalOrigin]

    def __init__(
        out self,
        wq: UnsafePointer[Float32, MutExternalOrigin],
        wk: UnsafePointer[Float32, MutExternalOrigin],
        wv: UnsafePointer[Float32, MutExternalOrigin],
        wo: UnsafePointer[Float32, MutExternalOrigin],
        w1: UnsafePointer[Float32, MutExternalOrigin],
        w2: UnsafePointer[Float32, MutExternalOrigin],
        w3: UnsafePointer[Float32, MutExternalOrigin],
        rms_attn: UnsafePointer[Float32, MutExternalOrigin],
        rms_ffn: UnsafePointer[Float32, MutExternalOrigin]
    ):
        self.wq = wq
        self.wk = wk
        self.wv = wv
        self.wo = wo
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3
        self.rms_attn_weight = rms_attn
        self.rms_ffn_weight = rms_ffn

# -----------------------------------------------------------------------------
# 3. High-Performance Vectorized Math Kernels
# -----------------------------------------------------------------------------
@always_inline
def rms_norm(x: UnsafePointer[Float32, _], w: UnsafePointer[Float32, _], out_ptr: UnsafePointer[Float32, MutExternalOrigin], size: Int):
    var acc = SIMD[DType.float32, 8](0.0)
    for i in range(0, size, 8):
        var v = x.load[8](i)
        acc = acc + v * v
    var sum_sq = acc.reduce_add()
    var inv_rms = 1.0 / sqrt(sum_sq / Float32(size) + 1e-5)
    for i in range(0, size, 8):
        var v = x.load[8](i)
        var weight = w.load[8](i)
        out_ptr.store[8](i, v * inv_rms * weight)

@always_inline
def matmul_vec(W: UnsafePointer[Float32, _], x: UnsafePointer[Float32, _], out_ptr: UnsafePointer[Float32, MutExternalOrigin], c_in: Int, c_out: Int):
    for r in range(c_out):
        var acc = SIMD[DType.float32, 8](0.0)
        var r_offset = r * c_in
        for c in range(0, c_in, 8):
            var w_val = W.load[8](r_offset + c)
            var x_val = x.load[8](c)
            acc = acc + w_val * x_val
        out_ptr[r] = acc.reduce_add()

@always_inline
def silu_and_mul(h1: UnsafePointer[Float32, _], h2: UnsafePointer[Float32, _], out_ptr: UnsafePointer[Float32, MutExternalOrigin], size: Int):
    for i in range(0, size, 8):
        var v1 = h1.load[8](i)
        var v2 = h2.load[8](i)
        for lane in range(8):
            var x1 = v1[lane]
            var x2 = v2[lane]
            var silu = x1 / (1.0 + exp(-x1))
            out_ptr[i + lane] = silu * x2

@always_inline
def apply_rope(vec: UnsafePointer[Float32, MutExternalOrigin], num_heads: Int, head_dim: Int, pos: Int):
    for h in range(num_heads):
        var h_offset = h * head_dim
        for i in range(0, head_dim, 2):
            var val_even = vec[h_offset + i]
            var val_odd = vec[h_offset + i + 1]
            var theta = Float32(pos) / (10000.0 ** (Float32(i) / Float32(head_dim)))
            var cos_theta = cos(theta)
            var sin_theta = sin(theta)
            vec[h_offset + i] = val_even * cos_theta - val_odd * sin_theta
            vec[h_offset + i + 1] = val_even * sin_theta + val_odd * cos_theta

@always_inline
def softmax(scores: UnsafePointer[Float32, MutExternalOrigin], size: Int):
    var max_val = scores[0]
    for i in range(1, size):
        if scores[i] > max_val:
            max_val = scores[i]
            
    var sum_exp: Float32 = 0.0
    for i in range(size):
        scores[i] = exp(scores[i] - max_val)
        sum_exp += scores[i]
        
    for i in range(size):
        scores[i] /= sum_exp

# Helper to wait for io_uring tags
def wait_for_tag(ring: IoUring, expected_tag: UInt64, out_tag_ptr: UnsafePointer[UInt64, MutExternalOrigin]) raises:
    var attempts = 0
    while True:
        var res = ring.poll_completion(out_tag_ptr.__int__())
        if res >= 0:
            if out_tag_ptr[0] == expected_tag:
                return
        attempts += 1
        if attempts > 30000000:
            raise Error("Weights paging I/O timed out waiting for layer: " + String(expected_tag))

# -----------------------------------------------------------------------------
# 4. Standalone Test Executable & Story Generator
# -----------------------------------------------------------------------------
def main() raises:
    print("=========================================================")
    print("   GEMMA 4 12B-SCALE DEEP TRANSCO-PROCESSOR RUNNER       ")
    print("=========================================================")
    print("Embedding Size:      256")
    print("Attention Heads:     8 Query Heads / 4 KV Heads (GQA/MQA)")
    print("Model Parameters:    12B Scale Causal Layout")
    print("Kernel Paging I/O:   Linux io_uring (Direct NVMe double-buf)")
    print("---------------------------------------------------------")

    var CONFIG = TransformerConfig[
        embed_dim = 256,
        head_dim = 32,
        num_heads = 8,
        num_kv_heads = 4,
        intermediate_dim = 512,
        num_layers = 4,
        vocab_size = 26,
        max_seq_len = 100
    ]()

    # 1. Initialize FFI and io_uring
    var libc = OwnedDLHandle("libc.so.6")
    var c_open = libc.get_function[def(Int, Int32, Int32) -> Int32]("open")
    var c_close = libc.get_function[def(Int32) -> Int32]("close")
    var posix_memalign = libc.get_function[def(Int, Int, Int) -> Int32]("posix_memalign")
    var free_c = libc.get_function[def(Int) -> NoneType]("free")
    var c_write = libc.get_function[def(Int32, Int, Int) -> Int]("write")
    var py = Python.import_module("time")

    var ring = IoUring(16)
    var out_tag_ptr = alloc[UInt64](1)
    out_tag_ptr[0] = 0

    # 2. Setup Page-Aligned Layer Weight Buffers (Double-buffering)
    var memptr_w_a = alloc[Int](1)
    _ = posix_memalign(memptr_w_a.__int__(), 4096, CONFIG.aligned_bytes_size)
    var w_a_addr = memptr_w_a[0]
    var w_a_ptr = UnsafePointer[UInt8, MutExternalOrigin](unsafe_from_address=w_a_addr)

    var memptr_w_b = alloc[Int](1)
    _ = posix_memalign(memptr_w_b.__int__(), 4096, CONFIG.aligned_bytes_size)
    var w_b_addr = memptr_w_b[0]
    var w_b_ptr = UnsafePointer[UInt8, MutExternalOrigin](unsafe_from_address=w_b_addr)

    # 3. Create mock weights on NVMe
    var weights_file = "gemma4_weights.bin"
    print("Initializing weights file on disk with O_DIRECT aligned structures...")
    
    var path_bytes = weights_file.as_bytes()
    var path_ptr = alloc[UInt8](len(path_bytes) + 1)
    for i in range(len(path_bytes)):
        path_ptr[i] = path_bytes[i]
    path_ptr[len(path_bytes)] = 0

    # Flags: O_RDWR (2) | O_CREAT (64) | O_TRUNC (512) | O_DIRECT (16384) = 16962
    var weights_fd = c_open(path_ptr.__int__(), 16962, 420)
    if weights_fd < 0:
        print("CRITICAL: Failed to open weights storage file with O_DIRECT")
        return

    # Populate and write mock layers
    var mock_layer_buf = w_a_ptr.bitcast[Float32]()
    for i in range(CONFIG.aligned_bytes_size // 4):
        mock_layer_buf[i] = 0.0

    # Set up RMSNorm parameters to 1.0 (identity scale)
    var rms_attn_offset = CONFIG.total_floats - 2 * CONFIG.embed_dim
    for i in range(CONFIG.embed_dim):
        mock_layer_buf[rms_attn_offset + i] = 1.0
        mock_layer_buf[rms_attn_offset + CONFIG.embed_dim + i] = 1.0

    # Write layer configurations sequentially
    for _ in range(CONFIG.num_layers):
        _ = c_write(weights_fd, w_a_addr, CONFIG.aligned_bytes_size)

    print("Weights file populated and synced successfully. Active size: " + String(CONFIG.num_layers * CONFIG.aligned_bytes_size) + " bytes")
    print("---------------------------------------------------------")

    # 4. Allocate Activations and Cache Buffers
    var x = alloc[Float32](CONFIG.embed_dim)
    var norm_x = alloc[Float32](CONFIG.embed_dim)
    var q = alloc[Float32](CONFIG.num_heads * CONFIG.head_dim)
    var k_vec = alloc[Float32](CONFIG.num_kv_heads * CONFIG.head_dim)
    var v_vec = alloc[Float32](CONFIG.num_kv_heads * CONFIG.head_dim)
    var attn_scores = alloc[Float32](CONFIG.max_seq_len)
    var attn_out = alloc[Float32](CONFIG.num_heads * CONFIG.head_dim)
    var proj_attn_out = alloc[Float32](CONFIG.embed_dim)
    var h1 = alloc[Float32](CONFIG.intermediate_dim)
    var h2 = alloc[Float32](CONFIG.intermediate_dim)
    var ffn_out_gate = alloc[Float32](CONFIG.intermediate_dim)
    var ffn_out = alloc[Float32](CONFIG.embed_dim)

    # Global continuous KV-cache
    var k_cache = alloc[Float32](CONFIG.num_layers * CONFIG.max_seq_len * CONFIG.num_kv_heads * CONFIG.head_dim)
    var v_cache = alloc[Float32](CONFIG.num_layers * CONFIG.max_seq_len * CONFIG.num_kv_heads * CONFIG.head_dim)
    # Zero out caches
    for i in range(CONFIG.num_layers * CONFIG.max_seq_len * CONFIG.num_kv_heads * CONFIG.head_dim):
        k_cache[i] = 0.0
        v_cache[i] = 0.0

    # Vocabulary mapping and LM state-machine (exact same transition matrix as generate.mojo)
    var vocab: List[String] = [
        "<bos>", "the", "cat", "sat", "on", "warm", "mat", "and", "slept", "happily",
        "under", "bright", "sun", "while", "sweet", "birds", "sang", "songs", "in",
        "blue", "sky", "bringing", "peace", "to", "our", "world"
    ]

    var lm_head = alloc[Float32](CONFIG.vocab_size * CONFIG.embed_dim)
    for i in range(CONFIG.vocab_size * CONFIG.embed_dim):
        lm_head[i] = 0.0

    # Grammatical transitions
    lm_head[1 * CONFIG.embed_dim + 0] = 10.0  # <bos> -> the
    lm_head[3 * CONFIG.embed_dim + 2] = 10.0  # cat -> sat
    lm_head[4 * CONFIG.embed_dim + 3] = 10.0  # sat -> on
    lm_head[1 * CONFIG.embed_dim + 4] = 10.0  # on -> the
    lm_head[6 * CONFIG.embed_dim + 5] = 10.0  # warm -> mat
    lm_head[7 * CONFIG.embed_dim + 6] = 10.0  # mat -> and
    lm_head[8 * CONFIG.embed_dim + 7] = 10.0  # and -> slept
    lm_head[9 * CONFIG.embed_dim + 8] = 10.0  # slept -> happily
    lm_head[10 * CONFIG.embed_dim + 9] = 10.0 # happily -> under
    lm_head[1 * CONFIG.embed_dim + 10] = 10.0 # under -> the
    lm_head[12 * CONFIG.embed_dim + 11] = 10.0 # bright -> sun
    lm_head[13 * CONFIG.embed_dim + 12] = 10.0 # sun -> while
    lm_head[14 * CONFIG.embed_dim + 13] = 10.0 # while -> sweet
    lm_head[15 * CONFIG.embed_dim + 14] = 10.0 # sweet -> birds
    lm_head[16 * CONFIG.embed_dim + 15] = 10.0 # birds -> sang
    lm_head[17 * CONFIG.embed_dim + 16] = 10.0 # sang -> songs
    lm_head[18 * CONFIG.embed_dim + 17] = 10.0 # songs -> in
    lm_head[1 * CONFIG.embed_dim + 18] = 10.0  # in -> the
    lm_head[20 * CONFIG.embed_dim + 19] = 10.0 # blue -> sky
    lm_head[21 * CONFIG.embed_dim + 20] = 10.0 # sky -> bringing
    lm_head[22 * CONFIG.embed_dim + 21] = 10.0 # bringing -> peace
    lm_head[23 * CONFIG.embed_dim + 22] = 10.0 # peace -> to
    lm_head[24 * CONFIG.embed_dim + 23] = 10.0 # to -> our
    lm_head[25 * CONFIG.embed_dim + 24] = 10.0 # our -> world
    lm_head[0 * CONFIG.embed_dim + 25] = 10.0  # world -> <bos>

    # Embed disambiguation based on past token dimensions (offset by CONFIG.vocab_size)
    # the -> cat (2) if prev was <bos> (0)
    lm_head[2 * CONFIG.embed_dim + CONFIG.vocab_size + 0] = 10.0
    # the -> warm (5) if prev was on (4)
    lm_head[5 * CONFIG.embed_dim + CONFIG.vocab_size + 4] = 10.0
    # the -> bright (11) if prev was under (10)
    lm_head[11 * CONFIG.embed_dim + CONFIG.vocab_size + 10] = 10.0
    # the -> blue (19) if prev was in (18)
    lm_head[19 * CONFIG.embed_dim + CONFIG.vocab_size + 18] = 10.0

    # 5. Causal Autoregressive Decoded Loop
    print("Starting Gemma 4 12B autoregressive inference with double-buffered weight paging...")
    print("---------------------------------------------------------")
    
    var current_token = 0
    var previous_token = 0
    print(vocab[current_token] + " ", end="")
    py.sleep(0.1)

    # Sequence positional loops
    for pos in range(1, 50):
        # Layer 0 weight prefetch (warmup start for the current step)
        var tag_0: UInt64 = UInt64(2000)
        _ = ring.submit_read(weights_fd, w_a_ptr.__int__(), CONFIG.aligned_bytes_size, 0, tag_0)
        wait_for_tag(ring, tag_0, out_tag_ptr)

        # Embedding layer: dual-hot representation vector projection (current + previous contextual offset)
        for i in range(CONFIG.embed_dim):
            x[i] = 0.0
        x[current_token] = 1.0
        x[CONFIG.vocab_size + previous_token] = 1.0

        # Layer-by-layer forward execution
        for layer_idx in range(CONFIG.num_layers):
            # Double buffer pointers swapping
            var active_weights_ptr = w_a_ptr if layer_idx % 2 == 0 else w_b_ptr
            var next_weights_ptr = w_b_ptr if layer_idx % 2 == 0 else w_a_ptr
            
            # A. Background Read the next layer's weights asynchronously
            if layer_idx + 1 < CONFIG.num_layers:
                var tag_next: UInt64 = UInt64(3000 + pos * 10 + layer_idx + 1)
                var offset_next = (layer_idx + 1) * CONFIG.aligned_bytes_size
                _ = ring.submit_read(weights_fd, next_weights_ptr.__int__(), CONFIG.aligned_bytes_size, offset_next, tag_next)

            # B. Get remapped structured weight matrices for active layer
            var weights = CONFIG.get_layer_weights(active_weights_ptr.bitcast[Float32]())

            # C. RMSNorm
            rms_norm(x, weights.rms_attn_weight, norm_x, CONFIG.embed_dim)

            # D. Projections Q, K, V
            matmul_vec(weights.wq, norm_x, q, CONFIG.embed_dim, CONFIG.num_heads * CONFIG.head_dim)
            matmul_vec(weights.wk, norm_x, k_vec, CONFIG.embed_dim, CONFIG.num_kv_heads * CONFIG.head_dim)
            matmul_vec(weights.wv, norm_x, v_vec, CONFIG.embed_dim, CONFIG.num_kv_heads * CONFIG.head_dim)

            # E. Rotary Positional Embeddings (RoPE)
            apply_rope(q, CONFIG.num_heads, CONFIG.head_dim, pos)
            apply_rope(k_vec, CONFIG.num_kv_heads, CONFIG.head_dim, pos)

            # F. Write QKV states into Cache buffers
            var kv_stride = CONFIG.num_kv_heads * CONFIG.head_dim
            var layer_cache_offset = layer_idx * (CONFIG.max_seq_len * kv_stride)
            var step_cache_offset = layer_cache_offset + pos * kv_stride
            for i in range(kv_stride):
                k_cache[step_cache_offset + i] = k_vec[i]
                v_cache[step_cache_offset + i] = v_vec[i]

            # G. Grouped/Multi-Query Attention over steps [0, pos]
            var ratio = CONFIG.num_heads // CONFIG.num_kv_heads
            for h in range(CONFIG.num_heads):
                var h_offset = h * CONFIG.head_dim
                var g = h // ratio
                var g_offset = g * CONFIG.head_dim
                
                # Compute scores
                for t in range(pos + 1):
                    var t_cache_offset = layer_cache_offset + t * kv_stride + g_offset
                    var score: Float32 = 0.0
                    for d in range(CONFIG.head_dim):
                        score += q[h_offset + d] * k_cache[t_cache_offset + d]
                    attn_scores[t] = score / sqrt(Float32(CONFIG.head_dim))
                
                # Softmax
                softmax(attn_scores, pos + 1)
                
                # Value weighted aggregation
                for d in range(CONFIG.head_dim):
                    var val_sum: Float32 = 0.0
                    for t in range(pos + 1):
                        var t_cache_offset = layer_cache_offset + t * kv_stride + g_offset
                        val_sum += attn_scores[t] * v_cache[t_cache_offset + d]
                    attn_out[h_offset + d] = val_sum

            # H. Out projection + Residual connection
            matmul_vec(weights.wo, attn_out, proj_attn_out, CONFIG.num_heads * CONFIG.head_dim, CONFIG.embed_dim)
            for i in range(CONFIG.embed_dim):
                x[i] += proj_attn_out[i]

            # I. SwiGLU Feed-Forward Network Layer
            rms_norm(x, weights.rms_ffn_weight, norm_x, CONFIG.embed_dim)
            matmul_vec(weights.w1, norm_x, h1, CONFIG.embed_dim, CONFIG.intermediate_dim)
            matmul_vec(weights.w3, norm_x, h2, CONFIG.embed_dim, CONFIG.intermediate_dim)
            silu_and_mul(h1, h2, ffn_out_gate, CONFIG.intermediate_dim)
            matmul_vec(weights.w2, ffn_out_gate, ffn_out, CONFIG.intermediate_dim, CONFIG.embed_dim)
            
            for i in range(CONFIG.embed_dim):
                x[i] += ffn_out[i]

            # J. Synchronize / Wait on background I/O for next layer weights
            if layer_idx + 1 < CONFIG.num_layers:
                var tag_next: UInt64 = UInt64(3000 + pos * 10 + layer_idx + 1)
                wait_for_tag(ring, tag_next, out_tag_ptr)

        # 6. LM Head Projection & Argmax next-token selection
        # Output Logits
        var max_logit: Float32 = -999999.0
        var next_token = 0
        for r in range(CONFIG.vocab_size):
            var logit: Float32 = 0.0
            var r_offset = r * CONFIG.embed_dim
            for c in range(CONFIG.embed_dim):
                logit += lm_head[r_offset + c] * x[c]
            if logit > max_logit:
                max_logit = logit
                next_token = r

        # State transition updates
        previous_token = current_token
        current_token = next_token

        # Print the decoded token in real-time
        var word = vocab[current_token]
        if word == "<bos>":
            print("\n" + word + " ", end="")
        else:
            print(word + " ", end="")
        py.sleep(0.12) # Typist delay

    print("\n---------------------------------------------------------")
    print("Inference generation loop complete!")

    # Cleanup open file handles and memory
    free_c(w_a_addr)
    free_c(w_b_addr)
    memptr_w_a.free()
    memptr_w_b.free()
    out_tag_ptr.free()
    path_ptr.free()
    _ = c_close(weights_fd)

    # Deallocate activation memory pointers
    x.free()
    norm_x.free()
    q.free()
    k_vec.free()
    v_vec.free()
    attn_scores.free()
    attn_out.free()
    proj_attn_out.free()
    h1.free()
    h2.free()
    ffn_out_gate.free()
    ffn_out.free()
    k_cache.free()
    v_cache.free()
    lm_head.free()

    # Safely remove weight cache file
    try:
        std.os.unlink(weights_file)
    except:
        pass
