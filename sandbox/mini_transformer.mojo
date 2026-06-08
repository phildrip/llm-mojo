from std.ffi import OwnedDLHandle
from std.memory import alloc
from std.math import sqrt, exp, log, cos, sin
from std.random import rand
from std.algorithm import parallelize

# -----------------------------------------------------------------------------
# 1. High-Performance Vectorized & Parallelized Math Kernels
# -----------------------------------------------------------------------------

@always_inline
def matmul(
    C: UnsafePointer[Float32, MutExternalOrigin],
    A: UnsafePointer[Float32, _],
    B: UnsafePointer[Float32, _],
    M: Int,
    N: Int,
    K: Int,
):
    # C = A * B. C is M x N, A is M x K, B is K x N
    # Parallelize over rows of C (M)
    def worker(i: Int) capturing -> None:
        # Initialize row i of C
        for j in range(N):
            C[i * N + j] = 0.0
            
        for k in range(K):
            var a_val = A[i * K + k]
            var a_simd = SIMD[DType.float32, 8](a_val)
            
            var j = 0
            while j < N - 7:
                var b_simd = B.load[8](k * N + j)
                var c_simd = C.load[8](i * N + j)
                C.store[8](i * N + j, c_simd + a_simd * b_simd)
                j += 8
                
            while j < N:
                C[i * N + j] += a_val * B[k * N + j]
                j += 1

    parallelize[worker](M)

@always_inline
def matmul_AtB(
    C: UnsafePointer[Float32, MutExternalOrigin],
    A: UnsafePointer[Float32, _],
    B: UnsafePointer[Float32, _],
    M: Int,
    N: Int,
    K: Int,
):
    # C = A^T * B. C is M x N, A is K x M, B is K x N
    # Parallelize over rows of C (M)
    def worker(i: Int) capturing -> None:
        for j in range(N):
            C[i * N + j] = 0.0
            
        for k in range(K):
            var a_val = A[k * M + i]
            var a_simd = SIMD[DType.float32, 8](a_val)
            
            var j = 0
            while j < N - 7:
                var b_simd = B.load[8](k * N + j)
                var c_simd = C.load[8](i * N + j)
                C.store[8](i * N + j, c_simd + a_simd * b_simd)
                j += 8
                
            while j < N:
                C[i * N + j] += a_val * B[k * N + j]
                j += 1

    parallelize[worker](M)

@always_inline
def matmul_ABt(
    C: UnsafePointer[Float32, MutExternalOrigin],
    A: UnsafePointer[Float32, _],
    B: UnsafePointer[Float32, _],
    M: Int,
    N: Int,
    K: Int,
):
    # C = A * B^T. C is M x N, A is M x K, B is N x K
    # Parallelize over rows of C (M)
    def worker(i: Int) capturing -> None:
        for j in range(N):
            var acc = SIMD[DType.float32, 8](0.0)
            var k = 0
            while k < K - 7:
                var a_simd = A.load[8](i * K + k)
                var b_simd = B.load[8](j * K + k)
                acc += a_simd * b_simd
                k += 8
                
            var sum = acc.reduce_add()
            while k < K:
                sum += A[i * K + k] * B[j * K + k]
                k += 1
                
            C[i * N + j] = sum

    parallelize[worker](M)

@always_inline
def rms_norm_fwd(
    x: UnsafePointer[Float32, _],
    w: UnsafePointer[Float32, _],
    out_ptr: UnsafePointer[Float32, MutExternalOrigin],
    inv_rms: UnsafePointer[Float32, MutExternalOrigin],
    N: Int,
    D: Int,
):
    for i in range(N):
        var sum_sq = Float32(0.0)
        var k = 0
        var acc = SIMD[DType.float32, 8](0.0)
        while k < D - 7:
            var val = x.load[8](i * D + k)
            acc += val * val
            k += 8
        sum_sq += acc.reduce_add()
        while k < D:
            var val = x[i * D + k]
            sum_sq += val * val
            k += 1
            
        var rms = 1.0 / sqrt(sum_sq / Float32(D) + 1e-5)
        inv_rms[i] = rms
        
        k = 0
        while k < D - 7:
            var val = x.load[8](i * D + k)
            var wt = w.load[8](k)
            out_ptr.store[8](i * D + k, val * rms * wt)
            k += 8
        while k < D:
            out_ptr[i * D + k] = x[i * D + k] * rms * w[k]
            k += 1

@always_inline
def rms_norm_bwd(
    x: UnsafePointer[Float32, _],
    w: UnsafePointer[Float32, _],
    dy: UnsafePointer[Float32, _],
    inv_rms: UnsafePointer[Float32, _],
    dx: UnsafePointer[Float32, MutExternalOrigin],
    dw: UnsafePointer[Float32, MutExternalOrigin],
    N: Int,
    D: Int,
):
    # Accumulate dw
    for k in range(D):
        var sum_dw = Float32(0.0)
        for i in range(N):
            var x_norm = x[i * D + k] * inv_rms[i]
            sum_dw += dy[i * D + k] * x_norm
        dw[k] += sum_dw
            
    # Calculate dx
    for i in range(N):
        var rms = inv_rms[i]
        var dot_prod = Float32(0.0)
        var k = 0
        var acc = SIMD[DType.float32, 8](0.0)
        while k < D - 7:
            var dy_simd = dy.load[8](i * D + k)
            var w_simd = w.load[8](k)
            var x_simd = x.load[8](i * D + k)
            acc += dy_simd * w_simd * x_simd
            k += 8
        dot_prod += acc.reduce_add()
        while k < D:
            dot_prod += dy[i * D + k] * w[k] * x[i * D + k]
            k += 1
            
        var scale = dot_prod * rms * rms / Float32(D)
        
        k = 0
        while k < D - 7:
            var dy_simd = dy.load[8](i * D + k)
            var w_simd = w.load[8](k)
            var x_simd = x.load[8](i * D + k)
            var dx_norm = dy_simd * w_simd
            var res = rms * (dx_norm - x_simd * scale)
            dx.store[8](i * D + k, res)
            k += 8
        while k < D:
            var dx_norm = dy[i * D + k] * w[k]
            dx[i * D + k] = rms * (dx_norm - x[i * D + k] * scale)
            k += 1

@always_inline
def softmax_fwd(
    scores: UnsafePointer[Float32, MutExternalOrigin],
    probs: UnsafePointer[Float32, MutExternalOrigin],
    B: Int,
    H: Int,
    S: Int,
):
    for b in range(B):
        for h in range(H):
            for i in range(S):
                var offset = ((b * H + h) * S + i) * S
                
                var max_val = scores[offset]
                for j in range(1, i + 1):
                    if scores[offset + j] > max_val:
                        max_val = scores[offset + j]
                        
                var sum_exp = Float32(0.0)
                for j in range(i + 1):
                    var val = exp(scores[offset + j] - max_val)
                    probs[offset + j] = val
                    sum_exp += val
                    
                for j in range(i + 1):
                    probs[offset + j] /= sum_exp
                for j in range(i + 1, S):
                    probs[offset + j] = 0.0

@always_inline
def softmax_bwd(
    probs: UnsafePointer[Float32, _],
    d_probs: UnsafePointer[Float32, _],
    d_scores: UnsafePointer[Float32, MutExternalOrigin],
    B: Int,
    H: Int,
    S: Int,
):
    for b in range(B):
        for h in range(H):
            for i in range(S):
                var offset = ((b * H + h) * S + i) * S
                
                var sum_da_a = Float32(0.0)
                for j in range(i + 1):
                    sum_da_a += d_probs[offset + j] * probs[offset + j]
                    
                for j in range(i + 1):
                    d_scores[offset + j] = probs[offset + j] * (d_probs[offset + j] - sum_da_a)
                for j in range(i + 1, S):
                    d_scores[offset + j] = 0.0

# -----------------------------------------------------------------------------
# 1.5. Hourglass Pooling/Unpooling Helpers (In-Place, Zero-Allocation)
# -----------------------------------------------------------------------------

@always_inline
def pool_activations(x: UnsafePointer[Float32, MutExternalOrigin], out_x: UnsafePointer[Float32, MutExternalOrigin], B: Int, S: Int, D: Int):
    for b in range(B):
        for i in range(S // 2):
            var target_offset = (b * (S // 2) + i) * D
            var src1_offset = (b * S + 2 * i) * D
            var src2_offset = (b * S + 2 * i + 1) * D
            for d in range(D):
                out_x[target_offset + d] = 0.5 * (x[src1_offset + d] + x[src2_offset + d])

@always_inline
def unpool_activations(x: UnsafePointer[Float32, MutExternalOrigin], out_x: UnsafePointer[Float32, MutExternalOrigin], B: Int, S: Int, D: Int):
    for b in range(B - 1, -1, -1):
        for i in range((S // 2) - 1, -1, -1):
            var target1_offset = (b * S + 2 * i) * D
            var target2_offset = (b * S + 2 * i + 1) * D
            var src_offset = (b * (S // 2) + i) * D
            for d in range(D - 1, -1, -1):
                var val = x[src_offset + d]
                out_x[target1_offset + d] = val
                out_x[target2_offset + d] = val

@always_inline
def pool_gradients(dx: UnsafePointer[Float32, MutExternalOrigin], out_dx: UnsafePointer[Float32, MutExternalOrigin], B: Int, S: Int, D: Int):
    for b in range(B):
        for i in range(S // 2):
            var target_offset = (b * (S // 2) + i) * D
            var src1_offset = (b * S + 2 * i) * D
            var src2_offset = (b * S + 2 * i + 1) * D
            for d in range(D):
                out_dx[target_offset + d] = dx[src1_offset + d] + dx[src2_offset + d]

@always_inline
def unpool_gradients(dx: UnsafePointer[Float32, MutExternalOrigin], out_dx: UnsafePointer[Float32, MutExternalOrigin], B: Int, S: Int, D: Int):
    for b in range(B - 1, -1, -1):
        for i in range((S // 2) - 1, -1, -1):
            var target1_offset = (b * S + 2 * i) * D
            var target2_offset = (b * S + 2 * i + 1) * D
            var src_offset = (b * (S // 2) + i) * D
            for d in range(D - 1, -1, -1):
                var val = 0.5 * dx[src_offset + d]
                out_dx[target1_offset + d] = val
                out_dx[target2_offset + d] = val

# -----------------------------------------------------------------------------
# 2. Pre-allocated Activations Struct (Zero-Allocation Loop)
# -----------------------------------------------------------------------------

struct Activations:
    var B: Int
    var S: Int
    var N: Int
    var D: Int
    var M: Int
    var L: Int
    var H: Int
    var V: Int
    
    # Raw memory blocks
    var mem_size_floats: Int
    var mem_ptr: UnsafePointer[Float32, MutExternalOrigin]
    
    # Input tokens
    var x_tokens: UnsafePointer[Int32, MutExternalOrigin]
    var y_tokens: UnsafePointer[Int32, MutExternalOrigin]
    
    # Activation Pointers
    var x_embed: UnsafePointer[Float32, MutExternalOrigin]
    var norm1: UnsafePointer[Float32, MutExternalOrigin]
    var norm1_inv_rms: UnsafePointer[Float32, MutExternalOrigin]
    var q: UnsafePointer[Float32, MutExternalOrigin]
    var k: UnsafePointer[Float32, MutExternalOrigin]
    var v: UnsafePointer[Float32, MutExternalOrigin]
    var scores: UnsafePointer[Float32, MutExternalOrigin]
    var attn_probs: UnsafePointer[Float32, MutExternalOrigin]
    var context: UnsafePointer[Float32, MutExternalOrigin]
    var attn_out: UnsafePointer[Float32, MutExternalOrigin]
    var norm2: UnsafePointer[Float32, MutExternalOrigin]
    var norm2_inv_rms: UnsafePointer[Float32, MutExternalOrigin]
    var mlp_in: UnsafePointer[Float32, MutExternalOrigin]
    var mlp_act: UnsafePointer[Float32, MutExternalOrigin]
    var mlp_out: UnsafePointer[Float32, MutExternalOrigin]
    var layer_out: UnsafePointer[Float32, MutExternalOrigin]
    var final_norm_out: UnsafePointer[Float32, MutExternalOrigin]
    var final_norm_inv_rms: UnsafePointer[Float32, MutExternalOrigin]
    var logits: UnsafePointer[Float32, MutExternalOrigin]
    
    # Gradient Pointers (matching size)
    var d_logits: UnsafePointer[Float32, MutExternalOrigin]
    var d_final_norm_out: UnsafePointer[Float32, MutExternalOrigin]
    var d_layer_out: UnsafePointer[Float32, MutExternalOrigin]
    var d_mlp_out: UnsafePointer[Float32, MutExternalOrigin]
    var d_mlp_act: UnsafePointer[Float32, MutExternalOrigin]
    var d_mlp_in: UnsafePointer[Float32, MutExternalOrigin]
    var d_norm2: UnsafePointer[Float32, MutExternalOrigin]
    var d_attn_out: UnsafePointer[Float32, MutExternalOrigin]
    var d_context: UnsafePointer[Float32, MutExternalOrigin]
    var d_attn_probs: UnsafePointer[Float32, MutExternalOrigin]
    var d_scores: UnsafePointer[Float32, MutExternalOrigin]
    var d_q: UnsafePointer[Float32, MutExternalOrigin]
    var d_k: UnsafePointer[Float32, MutExternalOrigin]
    var d_v: UnsafePointer[Float32, MutExternalOrigin]
    var d_norm1: UnsafePointer[Float32, MutExternalOrigin]
    
    def __init__(out self, batch_size: Int, seq_len: Int, embed_dim: Int, intermediate_dim: Int, num_heads: Int, num_layers: Int, vocab_size: Int):
        self.B = batch_size
        self.S = seq_len
        self.N = batch_size * seq_len
        self.D = embed_dim
        self.M = intermediate_dim
        self.L = num_layers
        self.H = num_heads
        self.V = vocab_size
        
        self.x_tokens = alloc[Int32](self.N)
        self.y_tokens = alloc[Int32](self.N)
        
        # Calculate sizing of activations
        var sz_nd = self.N * self.D
        var sz_lnd = self.L * self.N * self.D
        var sz_lhss = self.L * self.B * self.H * self.S * self.S
        var sz_lnm = self.L * self.N * self.M
        var sz_nv = self.N * self.V
        var sz_ln = self.L * self.N
        
        var total_fwd_sz = sz_nd + sz_lnd + sz_lnd + sz_lnd + sz_lnd + sz_lhss + sz_lhss + sz_lnd + sz_lnd + sz_lnd + sz_lnm + sz_lnm + sz_lnd + (self.L + 1) * sz_nd + sz_nd + sz_nv + sz_ln + sz_ln + self.N
        var total_bwd_sz = sz_nv + sz_nd + (self.L + 1) * sz_nd + sz_lnd + sz_lnm + sz_lnm + sz_lnd + sz_lnd + sz_lnd + sz_lhss + sz_lhss + sz_lnd + sz_lnd + sz_lnd + sz_lnd
        
        self.mem_size_floats = total_fwd_sz + total_bwd_sz
        self.mem_ptr = alloc[Float32](self.mem_size_floats)
        
        # Set offsets for forward activations
        var p = self.mem_ptr
        self.x_embed = p; p += sz_nd
        self.norm1 = p; p += sz_lnd
        self.norm1_inv_rms = p; p += sz_ln
        self.q = p; p += sz_lnd
        self.k = p; p += sz_lnd
        self.v = p; p += sz_lnd
        self.scores = p; p += sz_lhss
        self.attn_probs = p; p += sz_lhss
        self.context = p; p += sz_lnd
        self.attn_out = p; p += sz_lnd
        self.norm2 = p; p += sz_lnd
        self.norm2_inv_rms = p; p += sz_ln
        self.mlp_in = p; p += sz_lnm
        self.mlp_act = p; p += sz_lnm
        self.mlp_out = p; p += sz_lnd
        self.layer_out = p; p += (self.L + 1) * sz_nd
        self.final_norm_out = p; p += sz_nd
        self.final_norm_inv_rms = p; p += self.N
        self.logits = p; p += sz_nv
        
        # Set offsets for backward gradients
        self.d_logits = p; p += sz_nv
        self.d_final_norm_out = p; p += sz_nd
        self.d_layer_out = p; p += (self.L + 1) * sz_nd
        self.d_mlp_out = p; p += sz_lnd
        self.d_mlp_act = p; p += sz_lnm
        self.d_mlp_in = p; p += sz_lnm
        self.d_norm2 = p; p += sz_lnd
        self.d_attn_out = p; p += sz_lnd
        self.d_context = p; p += sz_lnd
        self.d_attn_probs = p; p += sz_lhss
        self.d_scores = p; p += sz_lhss
        self.d_q = p; p += sz_lnd
        self.d_k = p; p += sz_lnd
        self.d_v = p; p += sz_lnd
        self.d_norm1 = p; p += sz_lnd
        
    def __del__(deinit self):
        self.x_tokens.free()
        self.y_tokens.free()
        self.mem_ptr.free()

# -----------------------------------------------------------------------------
# 3. MiniTransformer Parameterized Struct
# -----------------------------------------------------------------------------

struct MiniTransformer[
    embed_dim: Int,
    intermediate_dim: Int,
    num_heads: Int,
    num_layers: Int,
    vocab_size: Int,
    max_seq_len: Int,
    reuse_attn: Bool = False,
    hourglass: Bool = False,
]:
    var total_size: Int
    var weights: UnsafePointer[Float32, MutExternalOrigin]
    var grads: UnsafePointer[Float32, MutExternalOrigin]
    var m: UnsafePointer[Float32, MutExternalOrigin]
    var v: UnsafePointer[Float32, MutExternalOrigin]
    
    # Specific parameter offsets
    var w_embed_offset: Int
    var layer_size: Int
    var final_norm_offset: Int
    
    def __init__(out self) raises:
        var D = Self.embed_dim
        var M = Self.intermediate_dim
        var L = Self.num_layers
        var V = Self.vocab_size
        
        self.w_embed_offset = 0
        var w_embed_sz = V * D
        
        self.layer_size = 4 * D * D + D * M + M * D + 2 * D
        var layers_sz = L * self.layer_size
        
        self.final_norm_offset = w_embed_sz + layers_sz
        var final_norm_sz = D
        
        self.total_size = w_embed_sz + layers_sz + final_norm_sz
        
        self.weights = alloc[Float32](self.total_size)
        self.grads = alloc[Float32](self.total_size)
        self.m = alloc[Float32](self.total_size)
        self.v = alloc[Float32](self.total_size)
        
        # Zero gradients and optimizer states
        for i in range(self.total_size):
            self.grads[i] = 0.0
            self.m[i] = 0.0
            self.v[i] = 0.0
            
        # Xavier-style random normal weight initialization
        var r_ptr = alloc[Float64](1)
        for i in range(self.total_size):
            rand(r_ptr, 1)
            var r1 = r_ptr[0]
            rand(r_ptr, 1)
            var r2 = r_ptr[0]
            var norm_val = sqrt(-2.0 * log(r1 + 1e-15)) * cos(2.0 * 3.1415926535 * r2)
            self.weights[i] = Float32(norm_val * 0.02)
        r_ptr.free()
        
        # Reset RMSNorm weights to 1.0 (identity scale at start)
        for l in range(L):
            var attn_norm_ptr = self.get_attn_norm_w(l)
            var mlp_norm_ptr = self.get_mlp_norm_w(l)
            for k in range(D):
                attn_norm_ptr[k] = 1.0
                mlp_norm_ptr[k] = 1.0
        var final_norm_ptr = self.get_final_norm_w()
        for k in range(D):
            final_norm_ptr[k] = 1.0
            
        print("MiniTransformer initialized successfully with", self.total_size, "parameters!")

    def __del__(deinit self):
        self.weights.free()
        self.grads.free()
        self.m.free()
        self.v.free()
        
    # Helper accessors for weights (using Self.embed_dim etc. to self-qualify parameters)
    @always_inline
    def get_w_embed(self) -> UnsafePointer[Float32, MutExternalOrigin]:
        return UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=(self.weights + self.w_embed_offset).__int__())
        
    @always_inline
    def get_w_embed_grad(self) -> UnsafePointer[Float32, MutExternalOrigin]:
        return UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=(self.grads + self.w_embed_offset).__int__())

    @always_inline
    def get_wq_w(self, layer_idx: Int) -> UnsafePointer[Float32, MutExternalOrigin]:
        var D = Self.embed_dim
        var offset = self.w_embed_offset + Self.vocab_size * D + layer_idx * self.layer_size
        return UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=(self.weights + offset).__int__())
        
    @always_inline
    def get_wq_g(self, layer_idx: Int) -> UnsafePointer[Float32, MutExternalOrigin]:
        var D = Self.embed_dim
        var offset = self.w_embed_offset + Self.vocab_size * D + layer_idx * self.layer_size
        return UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=(self.grads + offset).__int__())

    @always_inline
    def get_wk_w(self, layer_idx: Int) -> UnsafePointer[Float32, MutExternalOrigin]:
        var D = Self.embed_dim
        var offset = self.w_embed_offset + Self.vocab_size * D + layer_idx * self.layer_size + D * D
        return UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=(self.weights + offset).__int__())
        
    @always_inline
    def get_wk_g(self, layer_idx: Int) -> UnsafePointer[Float32, MutExternalOrigin]:
        var D = Self.embed_dim
        var offset = self.w_embed_offset + Self.vocab_size * D + layer_idx * self.layer_size + D * D
        return UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=(self.grads + offset).__int__())

    @always_inline
    def get_wv_w(self, layer_idx: Int) -> UnsafePointer[Float32, MutExternalOrigin]:
        var D = Self.embed_dim
        var offset = self.w_embed_offset + Self.vocab_size * D + layer_idx * self.layer_size + 2 * D * D
        return UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=(self.weights + offset).__int__())
        
    @always_inline
    def get_wv_g(self, layer_idx: Int) -> UnsafePointer[Float32, MutExternalOrigin]:
        var D = Self.embed_dim
        var offset = self.w_embed_offset + Self.vocab_size * D + layer_idx * self.layer_size + 2 * D * D
        return UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=(self.grads + offset).__int__())

    @always_inline
    def get_wo_w(self, layer_idx: Int) -> UnsafePointer[Float32, MutExternalOrigin]:
        var D = Self.embed_dim
        var offset = self.w_embed_offset + Self.vocab_size * D + layer_idx * self.layer_size + 3 * D * D
        return UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=(self.weights + offset).__int__())
        
    @always_inline
    def get_wo_g(self, layer_idx: Int) -> UnsafePointer[Float32, MutExternalOrigin]:
        var D = Self.embed_dim
        var offset = self.w_embed_offset + Self.vocab_size * D + layer_idx * self.layer_size + 3 * D * D
        return UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=(self.grads + offset).__int__())

    @always_inline
    def get_win_w(self, layer_idx: Int) -> UnsafePointer[Float32, MutExternalOrigin]:
        var D = Self.embed_dim
        var offset = self.w_embed_offset + Self.vocab_size * D + layer_idx * self.layer_size + 4 * D * D
        return UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=(self.weights + offset).__int__())
        
    @always_inline
    def get_win_g(self, layer_idx: Int) -> UnsafePointer[Float32, MutExternalOrigin]:
        var D = Self.embed_dim
        var offset = self.w_embed_offset + Self.vocab_size * D + layer_idx * self.layer_size + 4 * D * D
        return UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=(self.grads + offset).__int__())

    @always_inline
    def get_wout_w(self, layer_idx: Int) -> UnsafePointer[Float32, MutExternalOrigin]:
        var D = Self.embed_dim
        var M = Self.intermediate_dim
        var offset = self.w_embed_offset + Self.vocab_size * D + layer_idx * self.layer_size + 4 * D * D + D * M
        return UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=(self.weights + offset).__int__())
        
    @always_inline
    def get_wout_g(self, layer_idx: Int) -> UnsafePointer[Float32, MutExternalOrigin]:
        var D = Self.embed_dim
        var M = Self.intermediate_dim
        var offset = self.w_embed_offset + Self.vocab_size * D + layer_idx * self.layer_size + 4 * D * D + D * M
        return UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=(self.grads + offset).__int__())

    @always_inline
    def get_attn_norm_w(self, layer_idx: Int) -> UnsafePointer[Float32, MutExternalOrigin]:
        var D = Self.embed_dim
        var M = Self.intermediate_dim
        var offset = self.w_embed_offset + Self.vocab_size * D + layer_idx * self.layer_size + 4 * D * D + 2 * D * M
        return UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=(self.weights + offset).__int__())
        
    @always_inline
    def get_attn_norm_g(self, layer_idx: Int) -> UnsafePointer[Float32, MutExternalOrigin]:
        var D = Self.embed_dim
        var M = Self.intermediate_dim
        var offset = self.w_embed_offset + Self.vocab_size * D + layer_idx * self.layer_size + 4 * D * D + 2 * D * M
        return UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=(self.grads + offset).__int__())

    @always_inline
    def get_mlp_norm_w(self, layer_idx: Int) -> UnsafePointer[Float32, MutExternalOrigin]:
        var D = Self.embed_dim
        var M = Self.intermediate_dim
        var offset = self.w_embed_offset + Self.vocab_size * D + layer_idx * self.layer_size + 4 * D * D + 2 * D * M + D
        return UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=(self.weights + offset).__int__())
        
    @always_inline
    def get_mlp_norm_g(self, layer_idx: Int) -> UnsafePointer[Float32, MutExternalOrigin]:
        var D = Self.embed_dim
        var M = Self.intermediate_dim
        var offset = self.w_embed_offset + Self.vocab_size * D + layer_idx * self.layer_size + 4 * D * D + 2 * D * M + D
        return UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=(self.grads + offset).__int__())

    @always_inline
    def get_final_norm_w(self) -> UnsafePointer[Float32, MutExternalOrigin]:
        return UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=(self.weights + self.final_norm_offset).__int__())
        
    @always_inline
    def get_final_norm_g(self) -> UnsafePointer[Float32, MutExternalOrigin]:
        return UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=(self.grads + self.final_norm_offset).__int__())

    # -----------------------------------------------------------------------------
    # 4. Forward Pass
    # -----------------------------------------------------------------------------
    def forward(self, act: Activations):
        var B = act.B
        var S = act.S
        var N = act.N
        var D = Self.embed_dim
        var M = Self.intermediate_dim
        var L = Self.num_layers
        var H = Self.num_heads
        var V = Self.vocab_size
        var d = D // H
        
        var w_embed = self.get_w_embed()
        
        # 1. Embedding lookup
        for i in range(N):
            var token = Int(act.x_tokens[i])
            for k in range(D):
                act.x_embed[i * D + k] = w_embed[token * D + k]
                
        # Copy input embeddings into layer_out[0]
        for i in range(N * D):
            act.layer_out[i] = act.x_embed[i]
            
        # 2. Iterate layers
        for l in range(L):
            comptime if Self.hourglass and Self.num_layers >= 3:
                if l == 1:
                    pool_activations(act.layer_out + 1 * N * D, act.layer_out + 1 * N * D, B, S, D)
                elif l == L - 1:
                    unpool_activations(act.layer_out + (L - 1) * N * D, act.layer_out + (L - 1) * N * D, B, S, D)
                    
            var S_layer = S
            var N_layer = N
            comptime if Self.hourglass and Self.num_layers >= 3:
                if l > 0 and l < L - 1:
                    S_layer = S // 2
                    N_layer = B * S_layer
                    
            var layer_in = act.layer_out + l * N * D
            var norm1_ptr = act.norm1 + l * N * D
            var norm1_inv_rms_ptr = act.norm1_inv_rms + l * N
            
            # RMSNorm 1
            rms_norm_fwd(layer_in, self.get_attn_norm_w(l), norm1_ptr, norm1_inv_rms_ptr, N_layer, D)
            
            # Q, K, V Projections
            var q_ptr = act.q + l * N * D
            var k_ptr = act.k + l * N * D
            var v_ptr = act.v + l * N * D
            
            comptime if Self.reuse_attn and not Self.hourglass:
                if l == 0:
                    matmul(q_ptr, norm1_ptr, self.get_wq_w(l), N_layer, D, D)
                    matmul(k_ptr, norm1_ptr, self.get_wk_w(l), N_layer, D, D)
            else:
                matmul(q_ptr, norm1_ptr, self.get_wq_w(l), N_layer, D, D)
                matmul(k_ptr, norm1_ptr, self.get_wk_w(l), N_layer, D, D)
            matmul(v_ptr, norm1_ptr, self.get_wv_w(l), N_layer, D, D)
            
            # Multi-head attention scores and probabilities
            var scores_ptr = act.scores + l * B * H * S * S
            var probs_ptr = act.attn_probs + l * B * H * S * S
            var context_ptr = act.context + l * N * D
            
            comptime if Self.reuse_attn and not Self.hourglass:
                if l == 0:
                    # Compute QK^T / sqrt(d) causally
                    var inv_sqrt_d = 1.0 / sqrt(Float32(d))
                    for b in range(B):
                        for h in range(H):
                            for i in range(S_layer):
                                for j in range(i + 1):
                                    var q_idx = (b * S_layer + i) * D + h * d
                                    var k_idx = (b * S_layer + j) * D + h * d
                                    
                                    var score = Float32(0.0)
                                    for c in range(d):
                                        score += q_ptr[q_idx + c] * k_ptr[k_idx + c]
                                    
                                    scores_ptr[((b * H + h) * S_layer + i) * S_layer + j] = score * inv_sqrt_d
                                for j in range(i + 1, S_layer):
                                    scores_ptr[((b * H + h) * S_layer + i) * S_layer + j] = -1e9 # Mask out future
                                    
                    # Softmax
                    softmax_fwd(scores_ptr, probs_ptr, B, H, S_layer)
                else:
                    # Reuse probabilities from Layer 0
                    probs_ptr = act.attn_probs + 0 * B * H * S * S
            else:
                # Compute QK^T / sqrt(d) causally
                var inv_sqrt_d = 1.0 / sqrt(Float32(d))
                for b in range(B):
                    for h in range(H):
                        for i in range(S_layer):
                            for j in range(i + 1):
                                var q_idx = (b * S_layer + i) * D + h * d
                                var k_idx = (b * S_layer + j) * D + h * d
                                
                                var score = Float32(0.0)
                                for c in range(d):
                                    score += q_ptr[q_idx + c] * k_ptr[k_idx + c]
                                
                                scores_ptr[((b * H + h) * S_layer + i) * S_layer + j] = score * inv_sqrt_d
                            for j in range(i + 1, S_layer):
                                scores_ptr[((b * H + h) * S_layer + i) * S_layer + j] = -1e9 # Mask out future
                                
                # Softmax
                softmax_fwd(scores_ptr, probs_ptr, B, H, S_layer)
            
            # Context = probs * V
            for b in range(B):
                for h in range(H):
                    for i in range(S_layer):
                        var prob_offset = ((b * H + h) * S_layer + i) * S_layer
                        var out_offset = (b * S_layer + i) * D + h * d
                        
                        for c in range(d):
                            var weighted_val = Float32(0.0)
                            for j in range(i + 1):
                                var v_idx = (b * S_layer + j) * D + h * d + c
                                weighted_val += probs_ptr[prob_offset + j] * v_ptr[v_idx]
                            context_ptr[out_offset + c] = weighted_val
                            
            # Attention Output projection
            var attn_out_ptr = act.attn_out + l * N * D
            matmul(attn_out_ptr, context_ptr, self.get_wo_w(l), N_layer, D, D)
            
            # Residual 1
            var layer_mid = act.norm2 + l * N * D # Temporarily write here, or mlp_in
            for i in range(N_layer * D):
                layer_mid[i] = layer_in[i] + attn_out_ptr[i]
                
            # RMSNorm 2
            var norm2_ptr = act.norm2 + l * N * D
            var norm2_inv_rms_ptr = act.norm2_inv_rms + l * N
            rms_norm_fwd(layer_mid, self.get_mlp_norm_w(l), norm2_ptr, norm2_inv_rms_ptr, N_layer, D)
            
            # MLP Linear 1
            var mlp_in_ptr = act.mlp_in + l * N * M
            matmul(mlp_in_ptr, norm2_ptr, self.get_win_w(l), N_layer, M, D)
            
            # ReLU activation
            var mlp_act_ptr = act.mlp_act + l * N * M
            for i in range(N_layer * M):
                var val = mlp_in_ptr[i]
                mlp_act_ptr[i] = val if val > 0.0 else 0.0
                
            # MLP Linear 2
            var mlp_out_ptr = act.mlp_out + l * N * D
            matmul(mlp_out_ptr, mlp_act_ptr, self.get_wout_w(l), N_layer, D, M)
            
            # Residual 2
            var next_layer_in = act.layer_out + (l + 1) * N * D
            for i in range(N_layer * D):
                next_layer_in[i] = layer_mid[i] + mlp_out_ptr[i]
                
        # 3. Final layer RMSNorm
        rms_norm_fwd(act.layer_out + L * N * D, self.get_final_norm_w(), act.final_norm_out, act.final_norm_inv_rms, N, D)
        
        # 4. LM Head (Logits) via transpose of Embedding weights (weight tying!)
        matmul_ABt(act.logits, act.final_norm_out, w_embed, N, V, D)

    # -----------------------------------------------------------------------------
    # 5. Backward Pass
    # -----------------------------------------------------------------------------
    def backward(self, act: Activations) -> Float32:
        var B = act.B
        var S = act.S
        var N = act.N
        var D = Self.embed_dim
        var M = Self.intermediate_dim
        var L = Self.num_layers
        var H = Self.num_heads
        var V = Self.vocab_size
        var d = D // H
        
        var w_embed = self.get_w_embed()
        var w_embed_g = self.get_w_embed_grad()
        
        # Zero all gradients first
        for i in range(self.total_size):
            self.grads[i] = 0.0
            
        for i in range(act.L * act.B * act.H * act.S * act.S):
            act.d_attn_probs[i] = 0.0
            
        # 1. Compute loss and softmax of logits at the logits output layer
        var loss = Float32(0.0)
        for i in range(N):
            var offset = i * V
            var target = Int(act.y_tokens[i])
            
            # Max logit for numerical stability
            var max_logit = act.logits[offset]
            for k in range(1, V):
                if act.logits[offset + k] > max_logit:
                    max_logit = act.logits[offset + k]
                    
            var sum_exp = Float32(0.0)
            for k in range(V):
                var val = exp(act.logits[offset + k] - max_logit)
                act.d_logits[offset + k] = val
                sum_exp += val
                
            for k in range(V):
                act.d_logits[offset + k] /= sum_exp
                
            loss += -log(act.d_logits[offset + target] + 1e-15)
            
            # Backprop gradient at logit: p_k - y_k
            act.d_logits[offset + target] -= 1.0
            
        # Divide loss by batch dimensions
        loss = loss / Float32(N)
        
        # Scale grads of logits
        for i in range(N * V):
            act.d_logits[i] /= Float32(N)
            
        # 2. LM Head Backward (Weight tying!)
        # logits = final_norm_out * w_embed^T
        # dw_embed += d_logits^T * final_norm_out
        matmul_AtB(w_embed_g, act.d_logits, act.final_norm_out, V, D, N)
        
        # d_final_norm_out = d_logits * w_embed
        matmul(act.d_final_norm_out, act.d_logits, w_embed, N, D, V)
        
        # 3. Final layer RMSNorm Backward
        var final_layer_in_grad = act.d_layer_out + L * N * D
        rms_norm_bwd(act.layer_out + L * N * D, self.get_final_norm_w(), act.d_final_norm_out, act.final_norm_inv_rms, final_layer_in_grad, self.get_final_norm_g(), N, D)
        
        # 4. Iterate backward through layers
        var temp_nd = alloc[Float32](N * D)
        
        for l in range(L - 1, -1, -1):
            var S_layer = S
            var N_layer = N
            comptime if Self.hourglass and Self.num_layers >= 3:
                if l > 0 and l < L - 1:
                    S_layer = S // 2
                    N_layer = B * S_layer

            var layer_in = act.layer_out + l * N * D
            var next_layer_in_grad = act.d_layer_out + (l + 1) * N * D
            var layer_in_grad = act.d_layer_out + l * N * D
            
            var norm1_ptr = act.norm1 + l * N * D
            var norm2_ptr = act.norm2 + l * N * D
            var mlp_in_ptr = act.mlp_in + l * N * M
            var mlp_act_ptr = act.mlp_act + l * N * M
            var context_ptr = act.context + l * N * D
            var attn_out_ptr = act.attn_out + l * N * D
            
            var d_mlp_out_ptr = act.d_mlp_out + l * N * D
            var d_mlp_act_ptr = act.d_mlp_act + l * N * M
            var d_mlp_in_ptr = act.d_mlp_in + l * N * M
            var d_norm2_ptr = act.d_norm2 + l * N * D
            var d_attn_out_ptr = act.d_attn_out + l * N * D
            var d_context_ptr = act.d_context + l * N * D
            var d_q_ptr = act.d_q + l * N * D
            var d_k_ptr = act.d_k + l * N * D
            var d_v_ptr = act.d_v + l * N * D
            var d_norm1_ptr = act.d_norm1 + l * N * D
            
            # Residual 2 backward: d_layer_mid = d_layer_out_next, d_mlp_out = d_layer_out_next
            for i in range(N_layer * D):
                d_mlp_out_ptr[i] = next_layer_in_grad[i]
                
            # MLP Out projection backward
            matmul_AtB(self.get_wout_g(l), mlp_act_ptr, d_mlp_out_ptr, M, D, N_layer)
            matmul_ABt(d_mlp_act_ptr, d_mlp_out_ptr, self.get_wout_w(l), N_layer, M, D)
            
            # ReLU backward
            for i in range(N_layer * M):
                d_mlp_in_ptr[i] = d_mlp_act_ptr[i] if mlp_in_ptr[i] > 0.0 else 0.0
                
            # MLP In projection backward
            matmul_AtB(self.get_win_g(l), norm2_ptr, d_mlp_in_ptr, D, M, N_layer)
            matmul_ABt(d_norm2_ptr, d_mlp_in_ptr, self.get_win_w(l), N_layer, D, M)
            
            # RMSNorm 2 backward: input is layer_mid = layer_in + attn_out
            for i in range(N_layer * D):
                temp_nd[i] = layer_in[i] + attn_out_ptr[i]
                
            rms_norm_bwd(temp_nd, self.get_mlp_norm_w(l), d_norm2_ptr, act.norm2_inv_rms + l * N, layer_in_grad, self.get_mlp_norm_g(l), N_layer, D)
            
            # Add gradient from subsequent layer (Residual 2)
            for i in range(N_layer * D):
                layer_in_grad[i] += next_layer_in_grad[i]
                
            # Residual 1 backward: d_attn_out = d_layer_in_grad (accumulated so far)
            for i in range(N_layer * D):
                d_attn_out_ptr[i] = layer_in_grad[i]
                
            # Attention Output projection backward
            matmul_AtB(self.get_wo_g(l), context_ptr, d_attn_out_ptr, D, D, N_layer)
            matmul_ABt(d_context_ptr, d_attn_out_ptr, self.get_wo_w(l), N_layer, D, D)
            
            # Multi-head attention backward
            var layer_offset: Int
            comptime if Self.reuse_attn and not Self.hourglass:
                layer_offset = 0
            else:
                layer_offset = l
                
            var probs_ptr = act.attn_probs + layer_offset * B * H * S * S
            var d_scores_ptr = act.d_scores + l * B * H * S * S
            var d_probs_ptr = act.d_attn_probs + layer_offset * B * H * S * S
            
            var q_ptr = act.q + l * N * D
            var k_ptr = act.k + l * N * D
            var v_ptr = act.v + l * N * D
            
            # 1. d_probs and d_v
            # Clear d_v for this layer
            for i in range(N_layer * D):
                d_v_ptr[i] = 0.0
                
            comptime if not (Self.reuse_attn and not Self.hourglass):
                for i in range(B * H * S_layer * S_layer):
                    d_probs_ptr[i] = 0.0
                
            for b in range(B):
                for h in range(H):
                    for i in range(S_layer):
                        var prob_offset = ((b * H + h) * S_layer + i) * S_layer
                        var context_grad_offset = (b * S_layer + i) * D + h * d
                        
                        for j in range(i + 1):
                            var v_idx = (b * S_layer + j) * D + h * d
                            var prob_val = probs_ptr[prob_offset + j]
                            
                            # Accumulate d_probs
                            var dot_ctx_v = Float32(0.0)
                            for c in range(d):
                                dot_ctx_v += d_context_ptr[context_grad_offset + c] * v_ptr[v_idx + c]
                                # Accumulate d_v
                                d_v_ptr[v_idx + c] += prob_val * d_context_ptr[context_grad_offset + c]
                                
                            d_probs_ptr[prob_offset + j] += dot_ctx_v
                            
            comptime if Self.reuse_attn and not Self.hourglass:
                if l == 0:
                    # 2. Softmax backward to get d_scores
                    softmax_bwd(probs_ptr, d_probs_ptr, d_scores_ptr, B, H, S_layer)
                    
                    # 3. Projections backward to get dQ and dK
                    var inv_sqrt_d = 1.0 / sqrt(Float32(d))
                    for i in range(N_layer * D):
                        d_q_ptr[i] = 0.0
                        d_k_ptr[i] = 0.0
                        
                    for b in range(B):
                        for h in range(H):
                            for i in range(S_layer):
                                for j in range(i + 1):
                                    var d_score = d_scores_ptr[((b * H + h) * S_layer + i) * S_layer + j] * inv_sqrt_d
                                    var q_idx = (b * S_layer + i) * D + h * d
                                    var k_idx = (b * S_layer + j) * D + h * d
                                    
                                    for c in range(d):
                                        d_q_ptr[q_idx + c] += d_score * k_ptr[k_idx + c]
                                        d_k_ptr[k_idx + c] += d_score * q_ptr[q_idx + c]
                                        
                    # Projections Weight Backward (Wq and Wk)
                    matmul_AtB(self.get_wq_g(l), norm1_ptr, d_q_ptr, D, D, N_layer)
                    matmul_AtB(self.get_wk_g(l), norm1_ptr, d_k_ptr, D, D, N_layer)
            else:
                # 2. Softmax backward to get d_scores
                softmax_bwd(probs_ptr, d_probs_ptr, d_scores_ptr, B, H, S_layer)
                
                # 3. Projections backward to get dQ and dK
                var inv_sqrt_d = 1.0 / sqrt(Float32(d))
                for i in range(N_layer * D):
                    d_q_ptr[i] = 0.0
                    d_k_ptr[i] = 0.0
                    
                for b in range(B):
                    for h in range(H):
                        for i in range(S_layer):
                            for j in range(i + 1):
                                var d_score = d_scores_ptr[((b * H + h) * S_layer + i) * S_layer + j] * inv_sqrt_d
                                var q_idx = (b * S_layer + i) * D + h * d
                                var k_idx = (b * S_layer + j) * D + h * d
                                
                                for c in range(d):
                                    d_q_ptr[q_idx + c] += d_score * k_ptr[k_idx + c]
                                    d_k_ptr[k_idx + c] += d_score * q_ptr[q_idx + c]
                                    
                # Projections Weight Backward (Wq and Wk)
                matmul_AtB(self.get_wq_g(l), norm1_ptr, d_q_ptr, D, D, N_layer)
                matmul_AtB(self.get_wk_g(l), norm1_ptr, d_k_ptr, D, D, N_layer)
                
            # Wv is always backpropagated
            matmul_AtB(self.get_wv_g(l), norm1_ptr, d_v_ptr, D, D, N_layer)
            
            # Projections Input Backward (accumulate to d_norm1)
            comptime if Self.reuse_attn and not Self.hourglass:
                if l == 0:
                    matmul_ABt(d_norm1_ptr, d_q_ptr, self.get_wq_w(l), N_layer, D, D)
                    matmul_ABt(temp_nd, d_k_ptr, self.get_wk_w(l), N_layer, D, D)
                    for i in range(N_layer * D):
                        d_norm1_ptr[i] += temp_nd[i]
                    matmul_ABt(temp_nd, d_v_ptr, self.get_wv_w(l), N_layer, D, D)
                    for i in range(N_layer * D):
                        d_norm1_ptr[i] += temp_nd[i]
                else:
                    # Only backpropagate through V
                    matmul_ABt(d_norm1_ptr, d_v_ptr, self.get_wv_w(l), N_layer, D, D)
            else:
                matmul_ABt(d_norm1_ptr, d_q_ptr, self.get_wq_w(l), N_layer, D, D)
                matmul_ABt(temp_nd, d_k_ptr, self.get_wk_w(l), N_layer, D, D)
                for i in range(N_layer * D):
                    d_norm1_ptr[i] += temp_nd[i]
                matmul_ABt(temp_nd, d_v_ptr, self.get_wv_w(l), N_layer, D, D)
                for i in range(N_layer * D):
                    d_norm1_ptr[i] += temp_nd[i]
                
            # RMSNorm 1 backward
            rms_norm_bwd(layer_in, self.get_attn_norm_w(l), d_norm1_ptr, act.norm1_inv_rms + l * N, temp_nd, self.get_attn_norm_g(l), N_layer, D)
            
            # Accumulate layer_in_grad
            for i in range(N_layer * D):
                layer_in_grad[i] += temp_nd[i]
                
            comptime if Self.hourglass and Self.num_layers >= 3:
                if l == L - 1:
                    pool_gradients(act.d_layer_out + l * N * D, act.d_layer_out + l * N * D, B, S, D)
                elif l == 1:
                    unpool_gradients(act.d_layer_out + 1 * N * D, act.d_layer_out + 1 * N * D, B, S, D)
                
        temp_nd.free()
        
        # Gradient for Embeddings is already in w_embed_g
        # We accumulate it on top of SGD/AdamW later!
        return loss

    # -----------------------------------------------------------------------------
    # 6. AdamW Optimizer Step
    # -----------------------------------------------------------------------------
    def adamw_step(self, t: Int, lr: Float32, weight_decay: Float32, beta1: Float32, beta2: Float32, eps: Float32):
        var m_corr = 1.0 - beta1**t
        var v_corr = 1.0 - beta2**t
        
        var i = 0
        while i < self.total_size - 7:
            var w_val = self.weights.load[8](i)
            var g_val = self.grads.load[8](i)
            # Weight decay
            g_val = g_val + weight_decay * w_val
            
            var m_val = self.m.load[8](i)
            var v_val = self.v.load[8](i)
            
            m_val = beta1 * m_val + (1.0 - beta1) * g_val
            v_val = beta2 * v_val + (1.0 - beta2) * g_val * g_val
            
            self.m.store[8](i, m_val)
            self.v.store[8](i, v_val)
            
            var m_hat = m_val / m_corr
            var v_hat = v_val / v_corr
            
            var step_val = lr * m_hat / (sqrt(v_hat) + eps)
            self.weights.store[8](i, w_val - step_val)
            i += 8
            
        while i < self.total_size:
            var w_val = self.weights[i]
            var g_val = self.grads[i] + weight_decay * w_val
            
            self.m[i] = beta1 * self.m[i] + (1.0 - beta1) * g_val
            self.v[i] = beta2 * self.v[i] + (1.0 - beta2) * g_val * g_val
            
            var m_hat = self.m[i] / m_corr
            var v_hat = self.v[i] / v_corr
            
            self.weights[i] -= lr * m_hat / (sqrt(v_hat) + eps)
            i += 1
