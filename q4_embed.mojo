from std.ffi import OwnedDLHandle
from std.memory import alloc
from std.algorithm import parallelize
from std.sys.info import num_physical_cores
from std.math import tanh, sqrt

@always_inline
def unpack_16bytes(
    bytes: SIMD[DType.uint8, 16], 
    mut w0: SIMD[DType.float32, 8], 
    mut w1: SIMD[DType.float32, 8], 
    mut w2: SIMD[DType.float32, 8], 
    mut w3: SIMD[DType.float32, 8]
):
    var repeated_01 = bytes.shuffle[0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7]()
    var shifted_01 = repeated_01 >> SIMD[DType.uint8, 16](0, 4, 0, 4, 0, 4, 0, 4, 0, 4, 0, 4, 0, 4, 0, 4)
    var masked_01 = (shifted_01 & 15).cast[DType.int8]() - 8
    
    var repeated_23 = bytes.shuffle[8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13, 14, 14, 15, 15]()
    var shifted_23 = repeated_23 >> SIMD[DType.uint8, 16](0, 4, 0, 4, 0, 4, 0, 4, 0, 4, 0, 4, 0, 4, 0, 4)
    var masked_23 = (shifted_23 & 15).cast[DType.int8]() - 8
    
    w0 = masked_01.slice[8]().cast[DType.float32]()
    w1 = masked_01.shift_left[8]().slice[8]().cast[DType.float32]()
    w2 = masked_23.slice[8]().cast[DType.float32]()
    w3 = masked_23.shift_left[8]().slice[8]().cast[DType.float32]()


struct Q4Embedding:
    var vocab_size: Int
    var embed_dim: Int
    var packed_weights: UnsafePointer[UInt8, MutExternalOrigin]
    var scales: UnsafePointer[Float32, MutExternalOrigin]
    var row_tail_bound_11: UnsafePointer[Float32, MutExternalOrigin]
    var last_winner_tok: UnsafePointer[Int, MutExternalOrigin]
    var scratch_x_even: UnsafePointer[Float32, MutExternalOrigin]
    var scratch_x_odd: UnsafePointer[Float32, MutExternalOrigin]
    var scratch_shared_best: UnsafePointer[Float32, MutExternalOrigin]
    var _libc: OwnedDLHandle

    def __init__(out self, path: String, vocab_size: Int = 0, embed_dim: Int = 0) raises:
        # Load libc FFI
        var libc = OwnedDLHandle("libc.so.6")
        var c_open = libc.get_function[def(Int, Int32, Int32) thin abi("C") -> Int32]("open")
        var c_close = libc.get_function[def(Int32) thin abi("C") -> Int32]("close")
        var posix_memalign = libc.get_function[def(Int, Int, Int) thin abi("C") -> Int32]("posix_memalign")
        var c_read = libc.get_function[def(Int32, Int, Int) thin abi("C") -> Int]("read")

        # Convert path to null-terminated C string
        var path_bytes = path.as_bytes()
        var path_ptr = alloc[UInt8](len(path_bytes) + 1)
        for i in range(len(path_bytes)):
            path_ptr[i] = path_bytes[i]
        path_ptr[len(path_bytes)] = 0

        # Open file (O_RDONLY = 0)
        var fd = c_open(path_ptr.__int__(), 0, 0)
        path_ptr.free()
        if fd < 0:
            raise Error("CRITICAL: Failed to open Q4 embedding file: " + path)

        if vocab_size > 0 and embed_dim > 0:
            self.vocab_size = vocab_size
            self.embed_dim = embed_dim
        else:
            # Read 8-byte header: [vocab_size (Int32), embed_dim (Int32)]
            var header_ptr = alloc[Int32](2)
            var header_read = c_read(fd, header_ptr.__int__(), 8)
            if header_read != 8:
                _ = c_close(fd)
                header_ptr.free()
                raise Error("CRITICAL: Failed to read header from Q4 embedding file")

            self.vocab_size = Int(header_ptr[0])
            self.embed_dim = Int(header_ptr[1])
            header_ptr.free()

        # Size of packed weights and scales
        var packed_size_bytes = self.vocab_size * (self.embed_dim // 2)
        var scales_size_bytes = self.vocab_size * (self.embed_dim // 32) * 4

        # Allocate page-aligned memory for packed weights
        var memptr_W = alloc[Int](1)
        var err_W = posix_memalign(memptr_W.__int__(), 4096, packed_size_bytes)
        if err_W != 0:
            _ = c_close(fd)
            memptr_W.free()
            raise Error("CRITICAL: Failed to allocate aligned memory for Q4 packed weights")
        var W_addr = memptr_W[0]
        memptr_W.free()
        self.packed_weights = UnsafePointer[UInt8, MutExternalOrigin](unsafe_from_address=W_addr)

        # Allocate page-aligned memory for scales
        var memptr_S = alloc[Int](1)
        var err_S = posix_memalign(memptr_S.__int__(), 4096, scales_size_bytes)
        if err_S != 0:
            _ = c_close(fd)
            memptr_S.free()
            raise Error("CRITICAL: Failed to allocate aligned memory for Q4 scales")
        var S_addr = memptr_S[0]
        memptr_S.free()
        self.scales = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=S_addr)

        var is_flat = (vocab_size > 0 and embed_dim > 0)

        if is_flat:
            # Read packed weights (flat)
            var read_p = 0
            while read_p < packed_size_bytes:
                var r = c_read(fd, W_addr + read_p, packed_size_bytes - read_p)
                if r <= 0:
                    _ = c_close(fd)
                    raise Error("CRITICAL: Error reading flat packed weights")
                read_p += r

            # Read scales (flat)
            var read_s = 0
            while read_s < scales_size_bytes:
                var r = c_read(fd, S_addr + read_s, scales_size_bytes - read_s)
                if r <= 0:
                    _ = c_close(fd)
                    raise Error("CRITICAL: Error reading flat scales")
                read_s += r
        else:
            # Read packed weights and scales chunk-by-chunk to prevent high memory pressures during loading
            # In our quantized file, we interleaved them chunk-by-chunk (8192 tokens per chunk).
            # Let's read them back using the exact same interleaved chunk structure!
            var chunk_size_tokens = 8192
            var num_chunks = self.vocab_size // chunk_size_tokens
            
            var bytes_per_chunk_packed = chunk_size_tokens * (self.embed_dim // 2)
            var bytes_per_chunk_scales = chunk_size_tokens * (self.embed_dim // 32) * 4

            var curr_packed_addr = W_addr
            var curr_scales_addr = S_addr

            for chunk_idx in range(num_chunks):
                # Read packed weights for chunk
                var read_p = 0
                while read_p < bytes_per_chunk_packed:
                    var r = c_read(fd, curr_packed_addr + read_p, bytes_per_chunk_packed - read_p)
                    if r <= 0:
                        _ = c_close(fd)
                        raise Error("CRITICAL: Error reading packed weights in chunk " + String(chunk_idx + 1))
                    read_p += r
                
                # Read scales for chunk
                var read_s = 0
                while read_s < bytes_per_chunk_scales:
                    var r = c_read(fd, curr_scales_addr + read_s, bytes_per_chunk_scales - read_s)
                    if r <= 0:
                        _ = c_close(fd)
                        raise Error("CRITICAL: Error reading scales in chunk " + String(chunk_idx + 1))
                    read_s += r
                
                curr_packed_addr += bytes_per_chunk_packed
                curr_scales_addr += bytes_per_chunk_scales

        _ = c_close(fd)
        self._libc = libc^

        # Precompute exact Euclidean tail L2 norm ||W_{r, tail}||_2 after prefix_blocks (25% of row)
        self.row_tail_bound_11 = alloc[Float32](self.vocab_size)
        self.last_winner_tok = alloc[Int](1)
        self.last_winner_tok[0] = 0
        self.scratch_x_even = alloc[Float32](self.embed_dim // 2)
        self.scratch_x_odd = alloc[Float32](self.embed_dim // 2)
        self.scratch_shared_best = alloc[Float32](1)
        var num_blocks_init = self.embed_dim // 32
        var b_pref = num_blocks_init // 2
        var packed_init_ptr = self.packed_weights
        var scales_init_ptr = self.scales
        var bound_ptr = self.row_tail_bound_11
        var half_dim_init = self.embed_dim // 2
        var vs_init = self.vocab_size
        var init_tasks = max(1, num_physical_cores())

        def init_tail_bounds_worker(t: Int) capturing -> None:
            var chunk = vs_init // init_tasks
            var s_r = t * chunk
            var e_r = (t + 1) * chunk if t < init_tasks - 1 else vs_init
            for r in range(s_r, e_r):
                var r_pack = r * half_dim_init
                var r_off = r * num_blocks_init
                var tail_norm_sq: Float32 = 0.0
                for b in range(b_pref, num_blocks_init):
                    var sc = scales_init_ptr[r_off + b]
                    var bytes = packed_init_ptr.load[16](r_pack + b * 16)
                    var we = ((bytes & 15).cast[DType.int8]() - 8).cast[DType.float32]()
                    var wo = ((bytes >> 4).cast[DType.int8]() - 8).cast[DType.float32]()
                    var blk_sq = (we * we + wo * wo).reduce_add()
                    tail_norm_sq += (sc * sc) * blk_sq
                bound_ptr[r] = sqrt(tail_norm_sq)

        parallelize[init_tail_bounds_worker](init_tasks)

        print("Q4 Embedding Loaded successfully:")
        print("  Vocab:     ", self.vocab_size)
        print("  Dimension: ", self.embed_dim)
        print("  RAM:       ", (packed_size_bytes + scales_size_bytes) // 1024 // 1024, "MB")

    def free(mut self):
        var c_free = self._libc.get_function[def(Int) thin abi("C") -> None]("free")
        c_free(Int(self.packed_weights))
        c_free(Int(self.scales))
        self.row_tail_bound_11.free()
        self.last_winner_tok.free()
        self.scratch_x_even.free()
        self.scratch_x_odd.free()
        self.scratch_shared_best.free()

    @always_inline
    def reconstruct[type: DType = DType.float32](self, token: Int, out_ptr: UnsafePointer[Scalar[type], MutExternalOrigin]) -> None:
        var row_packed_offset = token * (self.embed_dim // 2)
        var row_scales_offset = token * (self.embed_dim // 32)
        var num_blocks = self.embed_dim // 32
        
        for b in range(num_blocks):
            var scale = self.scales[row_scales_offset + b]
            var block_packed_offset = row_packed_offset + b * 16
            var block_out_offset = b * 32
            
            for j in range(4):
                var b0 = self.packed_weights[block_packed_offset + j * 4]
                var b1 = self.packed_weights[block_packed_offset + j * 4 + 1]
                var b2 = self.packed_weights[block_packed_offset + j * 4 + 2]
                var b3 = self.packed_weights[block_packed_offset + j * 4 + 3]
                
                var out_idx = block_out_offset + j * 8
                
                out_ptr[out_idx]     = Scalar[type](Float32(Int(b0 & 15) - 8) * scale)
                out_ptr[out_idx + 1] = Scalar[type](Float32(Int(b0 >> 4) - 8) * scale)
                out_ptr[out_idx + 2] = Scalar[type](Float32(Int(b1 & 15) - 8) * scale)
                out_ptr[out_idx + 3] = Scalar[type](Float32(Int(b1 >> 4) - 8) * scale)
                out_ptr[out_idx + 4] = Scalar[type](Float32(Int(b2 & 15) - 8) * scale)
                out_ptr[out_idx + 5] = Scalar[type](Float32(Int(b2 >> 4) - 8) * scale)
                out_ptr[out_idx + 6] = Scalar[type](Float32(Int(b3 & 15) - 8) * scale)
                out_ptr[out_idx + 7] = Scalar[type](Float32(Int(b3 >> 4) - 8) * scale)

    @always_inline
    def project_logits_parallel_reference(
        self,
        x: UnsafePointer[Float32, _],
        local_max_logits: UnsafePointer[Float32, MutExternalOrigin],
        local_next_tokens: UnsafePointer[Int, MutExternalOrigin],
        num_tasks: Int,
    ) -> None:
        var chunk_size = self.vocab_size // num_tasks
        var num_blocks = self.embed_dim // 32
        
        def ref_proj_worker(t: Int) capturing -> None:
            var start_r = t * chunk_size
            var end_r = (t + 1) * chunk_size if t < num_tasks - 1 else self.vocab_size
            var local_max_l: Float32 = -999999.0
            var local_tok = 0
            
            for r in range(start_r, end_r):
                var row_packed_offset = r * (self.embed_dim // 2)
                var row_scales_offset = r * (self.embed_dim // 32)
                var acc = SIMD[DType.float32, 8](0.0)
                
                for b in range(num_blocks):
                    var scale = self.scales[row_scales_offset + b]
                    var block_packed_offset = row_packed_offset + b * 16
                    var block_x_offset = b * 32
                    
                    var bytes = self.packed_weights.load[16](block_packed_offset)
                    
                    var w0 = SIMD[DType.float32, 8]()
                    var w1 = SIMD[DType.float32, 8]()
                    var w2 = SIMD[DType.float32, 8]()
                    var w3 = SIMD[DType.float32, 8]()
                    unpack_16bytes(bytes, w0, w1, w2, w3)
                    
                    acc = acc + (w0 * x.load[8](block_x_offset)) * scale
                    acc = acc + (w1 * x.load[8](block_x_offset + 8)) * scale
                    acc = acc + (w2 * x.load[8](block_x_offset + 16)) * scale
                    acc = acc + (w3 * x.load[8](block_x_offset + 24)) * scale
                
                var sum = acc.reduce_add()
                var logit = Float32(30.0) * tanh(sum / Float32(30.0))
                if logit > local_max_l:
                    local_max_l = logit
                    local_tok = r
            local_max_logits[t] = local_max_l
            local_next_tokens[t] = local_tok
            
        parallelize[ref_proj_worker](num_tasks)

    @always_inline
    def _eval_row_full(
        self,
        r: Int,
        x_even: UnsafePointer[Float32, _],
        x_odd: UnsafePointer[Float32, _],
        num_blocks: Int,
    ) -> Float32:
        var row_packed_offset = r * (self.embed_dim // 2)
        var row_scales_offset = r * num_blocks
        var acc = SIMD[DType.float32, 16](0.0)
        for b in range(num_blocks):
            var scale = self.scales[row_scales_offset + b]
            var bytes = self.packed_weights.load[16](row_packed_offset + b * 16)
            var w_even = ((bytes & 15).cast[DType.int8]() - 8).cast[DType.float32]()
            var w_odd = ((bytes >> 4).cast[DType.int8]() - 8).cast[DType.float32]()
            var block_dot = w_even * x_even.load[16](b * 16) + w_odd * x_odd.load[16](b * 16)
            acc = acc + block_dot * scale
        return acc.reduce_add()

    @always_inline
    def project_logits_parallel(
        self,
        x: UnsafePointer[Float32, _],
        local_max_logits: UnsafePointer[Float32, MutExternalOrigin],
        local_next_tokens: UnsafePointer[Int, MutExternalOrigin],
        num_tasks: Int,
    ) -> None:
        """Fast Dual-Space Deinterleaved + Early-Z PVS logit projection with monotonic tanh hoisting."""
        var chunk_size = self.vocab_size // num_tasks
        var num_blocks = self.embed_dim // 32
        var prefix_blocks = num_blocks // 2

        # 1. Dual-Space Deinterleaving (1970s Loop-Invariant Permutation):
        # Permute x ONCE into pre-allocated even (low-nibble) and odd (high-nibble) streams
        # so the 262,144-row inner loop executes ZERO byte shuffles, shifts, or heap allocations!
        var half_dim = self.embed_dim // 2
        var x_even = self.scratch_x_even
        var x_odd = self.scratch_x_odd
        var x_tail_sq = SIMD[DType.float32, 16](0.0)
        for b in range(num_blocks):
            var v0 = x.load[16](b * 32)
            var v1 = x.load[16](b * 32 + 16)
            var ev = v0.deinterleave()[0].join(v1.deinterleave()[0])
            var od = v0.deinterleave()[1].join(v1.deinterleave()[1])
            x_even.store[16](b * 16, ev)
            x_odd.store[16](b * 16, od)
            if b >= prefix_blocks:
                x_tail_sq = x_tail_sq + ev * ev + od * od
        var x_tail_rms = sqrt(x_tail_sq.reduce_add())

        # 2. Seed initial lower bound from hot working-set tokens [0..256) + last winning token
        var seed_count = 256 if self.vocab_size >= 256 else self.vocab_size
        var seed_best_sum: Float32 = -1e30
        var seed_best_tok: Int = 0
        var prev_win = self.last_winner_tok[0]
        if prev_win >= seed_count and prev_win < self.vocab_size:
            seed_best_sum = self._eval_row_full(prev_win, x_even, x_odd, num_blocks)
            seed_best_tok = prev_win

        for r in range(seed_count):
            var s = self._eval_row_full(r, x_even, x_odd, num_blocks)
            if s > seed_best_sum or (s == seed_best_sum and r < seed_best_tok):
                seed_best_sum = s
                seed_best_tok = r

        var shared_best = self.scratch_shared_best
        shared_best[0] = seed_best_sum

        def initial_proj_worker(t: Int) capturing -> None:
            var start_r = t * chunk_size
            if start_r < seed_count:
                start_r = seed_count
            var end_r = (t + 1) * chunk_size if t < num_tasks - 1 else self.vocab_size
            var local_max_sum: Float32 = seed_best_sum if t == 0 else -1e30
            var local_tok = seed_best_tok if t == 0 else start_r
            var prune_thresh: Float32 = shared_best[0]

            var r = start_r
            while r + 1 < end_r:
                if (r & 63) == 0:
                    var g_best = shared_best[0]
                    if g_best > prune_thresh:
                        prune_thresh = g_best

                var r0_pack = r * half_dim
                var r1_pack = (r + 1) * half_dim
                var r0_sc = r * num_blocks
                var r1_sc = (r + 1) * num_blocks
                var acc0 = SIMD[DType.float32, 16](0.0)
                var acc1 = SIMD[DType.float32, 16](0.0)

                # Stage 1: Evaluate first 25% of blocks for 2 rows simultaneously (reusing x_even/x_odd in registers)
                for b in range(prefix_blocks):
                    var xe = x_even.load[16](b * 16)
                    var xo = x_odd.load[16](b * 16)
                    var sc0 = self.scales[r0_sc + b]
                    var b0 = self.packed_weights.load[16](r0_pack + b * 16)
                    var we0 = ((b0 & 15).cast[DType.int8]() - 8).cast[DType.float32]()
                    var wo0 = ((b0 >> 4).cast[DType.int8]() - 8).cast[DType.float32]()
                    acc0 = acc0 + (we0 * xe + wo0 * xo) * sc0

                    var sc1 = self.scales[r1_sc + b]
                    var b1 = self.packed_weights.load[16](r1_pack + b * 16)
                    var we1 = ((b1 & 15).cast[DType.int8]() - 8).cast[DType.float32]()
                    var wo1 = ((b1 >> 4).cast[DType.int8]() - 8).cast[DType.float32]()
                    acc1 = acc1 + (we1 * xe + wo1 * xo) * sc1

                var p0 = acc0.reduce_add()
                var p1 = acc1.reduce_add()
                var keep0 = (p0 + self.row_tail_bound_11[r] * x_tail_rms >= prune_thresh)
                var keep1 = (p1 + self.row_tail_bound_11[r + 1] * x_tail_rms >= prune_thresh)

                if keep0 and keep1:
                    for b in range(prefix_blocks, num_blocks):
                        var xe = x_even.load[16](b * 16)
                        var xo = x_odd.load[16](b * 16)
                        var sc0 = self.scales[r0_sc + b]
                        var b0 = self.packed_weights.load[16](r0_pack + b * 16)
                        var we0 = ((b0 & 15).cast[DType.int8]() - 8).cast[DType.float32]()
                        var wo0 = ((b0 >> 4).cast[DType.int8]() - 8).cast[DType.float32]()
                        acc0 = acc0 + (we0 * xe + wo0 * xo) * sc0

                        var sc1 = self.scales[r1_sc + b]
                        var b1 = self.packed_weights.load[16](r1_pack + b * 16)
                        var we1 = ((b1 & 15).cast[DType.int8]() - 8).cast[DType.float32]()
                        var wo1 = ((b1 >> 4).cast[DType.int8]() - 8).cast[DType.float32]()
                        acc1 = acc1 + (we1 * xe + wo1 * xo) * sc1

                    var sum0 = acc0.reduce_add()
                    if sum0 > local_max_sum:
                        local_max_sum = sum0
                        local_tok = r
                        if sum0 > prune_thresh:
                            prune_thresh = sum0
                            shared_best[0] = sum0
                    var sum1 = acc1.reduce_add()
                    if sum1 > local_max_sum:
                        local_max_sum = sum1
                        local_tok = r + 1
                        if sum1 > prune_thresh:
                            prune_thresh = sum1
                            shared_best[0] = sum1
                elif keep0:
                    for b in range(prefix_blocks, num_blocks):
                        var sc0 = self.scales[r0_sc + b]
                        var b0 = self.packed_weights.load[16](r0_pack + b * 16)
                        var we0 = ((b0 & 15).cast[DType.int8]() - 8).cast[DType.float32]()
                        var wo0 = ((b0 >> 4).cast[DType.int8]() - 8).cast[DType.float32]()
                        acc0 = acc0 + (we0 * x_even.load[16](b * 16) + wo0 * x_odd.load[16](b * 16)) * sc0
                    var sum0 = acc0.reduce_add()
                    if sum0 > local_max_sum:
                        local_max_sum = sum0
                        local_tok = r
                        if sum0 > prune_thresh:
                            prune_thresh = sum0
                            shared_best[0] = sum0
                elif keep1:
                    for b in range(prefix_blocks, num_blocks):
                        var sc1 = self.scales[r1_sc + b]
                        var b1 = self.packed_weights.load[16](r1_pack + b * 16)
                        var we1 = ((b1 & 15).cast[DType.int8]() - 8).cast[DType.float32]()
                        var wo1 = ((b1 >> 4).cast[DType.int8]() - 8).cast[DType.float32]()
                        acc1 = acc1 + (we1 * x_even.load[16](b * 16) + wo1 * x_odd.load[16](b * 16)) * sc1
                    var sum1 = acc1.reduce_add()
                    if sum1 > local_max_sum:
                        local_max_sum = sum1
                        local_tok = r + 1
                        if sum1 > prune_thresh:
                            prune_thresh = sum1
                            shared_best[0] = sum1

                r += 2

            while r < end_r:
                var sum = self._eval_row_full(r, x_even, x_odd, num_blocks)
                if sum > local_max_sum:
                    local_max_sum = sum
                    local_tok = r
                    if sum > prune_thresh:
                        prune_thresh = sum
                        shared_best[0] = sum
                r += 1

            # Monotonic tanh soft-cap applied ONCE per worker thread instead of 262,144 times!
            local_max_logits[t] = Float32(30.0) * tanh(local_max_sum / Float32(30.0))
            local_next_tokens[t] = local_tok

        parallelize[initial_proj_worker](num_tasks)

        var overall_best_l: Float32 = -1e30
        var overall_best_tok: Int = 0
        for t in range(num_tasks):
            if local_max_logits[t] > overall_best_l or (
                local_max_logits[t] == overall_best_l and local_next_tokens[t] < overall_best_tok
            ):
                overall_best_l = local_max_logits[t]
                overall_best_tok = local_next_tokens[t]
        self.last_winner_tok[0] = overall_best_tok

    @always_inline
    def project_full_logits(
        self,
        x: UnsafePointer[Float32, _],
        out_logits: UnsafePointer[Float32, MutExternalOrigin],
        num_tasks: Int,
    ) -> None:
        var chunk_size = self.vocab_size // num_tasks
        var num_blocks = self.embed_dim // 32

        def full_proj_worker(t: Int) capturing -> None:
            var start_r = t * chunk_size
            var end_r = (t + 1) * chunk_size if t < num_tasks - 1 else self.vocab_size
            for r in range(start_r, end_r):
                var row_packed_offset = r * (self.embed_dim // 2)
                var row_scales_offset = r * (self.embed_dim // 32)
                var acc = SIMD[DType.float32, 8](0.0)
                for b in range(num_blocks):
                    var scale = self.scales[row_scales_offset + b]
                    var block_packed_offset = row_packed_offset + b * 16
                    var block_x_offset = b * 32
                    var bytes = self.packed_weights.load[16](block_packed_offset)
                    var w0 = SIMD[DType.float32, 8]()
                    var w1 = SIMD[DType.float32, 8]()
                    var w2 = SIMD[DType.float32, 8]()
                    var w3 = SIMD[DType.float32, 8]()
                    unpack_16bytes(bytes, w0, w1, w2, w3)
                    acc = acc + (w0 * x.load[8](block_x_offset)) * scale
                    acc = acc + (w1 * x.load[8](block_x_offset + 8)) * scale
                    acc = acc + (w2 * x.load[8](block_x_offset + 16)) * scale
                    acc = acc + (w3 * x.load[8](block_x_offset + 24)) * scale
                var sum = acc.reduce_add()
                out_logits[r] = Float32(30.0) * tanh(sum / Float32(30.0))

        if num_tasks == 1:
            full_proj_worker(0)
        else:
            parallelize[full_proj_worker](num_tasks)
