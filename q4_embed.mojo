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
    var s1_packed: UnsafePointer[UInt8, MutExternalOrigin]
    var s1_scales: UnsafePointer[Float32, MutExternalOrigin]
    var row_tail_bound_25: UnsafePointer[Float32, MutExternalOrigin]
    var row_tail_bound_11: UnsafePointer[Float32, MutExternalOrigin]
    var row_tail_bound_75: UnsafePointer[Float32, MutExternalOrigin]
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

        # Precompute exact 3-tier Euclidean tail L2 norms (25%, 50%, 75%) and contiguous Stage-1 prefix table
        var s1_blocks = max(1, (self.embed_dim // 32) // 4)
        self.s1_packed = alloc[UInt8](max(16, self.vocab_size * s1_blocks * 16))
        self.s1_scales = alloc[Float32](max(1, self.vocab_size * s1_blocks))
        self.row_tail_bound_25 = alloc[Float32](max(1, self.vocab_size))
        self.row_tail_bound_11 = alloc[Float32](max(1, self.vocab_size))
        self.row_tail_bound_75 = alloc[Float32](max(1, self.vocab_size))
        self.last_winner_tok = alloc[Int](1)
        self.last_winner_tok[0] = 0
        self.scratch_x_even = alloc[Float32](max(16, self.embed_dim // 2))
        self.scratch_x_odd = alloc[Float32](max(16, self.embed_dim // 2))
        self.scratch_shared_best = alloc[Float32](1)
        self.finalize_tail_bounds()

        print("Q4 Embedding Loaded successfully:")
        print("  Vocab:     ", self.vocab_size)
        print("  Dimension: ", self.embed_dim)
        print("  RAM:       ", (packed_size_bytes + scales_size_bytes) // 1024 // 1024, "MB")

    def __init__(out self, vocab_size: Int, embed_dim: Int) raises:
        """Allocate an in-memory Q4Embedding table to be populated via quantize_fp32_to_q4_0 + finalize_tail_bounds()."""
        var libc = OwnedDLHandle("libc.so.6")
        var posix_memalign = libc.get_function[def(Int, Int, Int) thin abi("C") -> Int32]("posix_memalign")
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        var packed_size_bytes = max(4096, self.vocab_size * (self.embed_dim // 2))
        var scales_size_bytes = max(4096, self.vocab_size * (self.embed_dim // 32) * 4)

        var memptr_W = alloc[Int](1)
        var err_W = posix_memalign(memptr_W.__int__(), 4096, packed_size_bytes)
        if err_W != 0:
            memptr_W.free()
            raise Error("CRITICAL: Failed to allocate aligned memory for Q4 packed weights")
        var W_addr = memptr_W[0]
        memptr_W.free()
        self.packed_weights = UnsafePointer[UInt8, MutExternalOrigin](unsafe_from_address=W_addr)

        var memptr_S = alloc[Int](1)
        var err_S = posix_memalign(memptr_S.__int__(), 4096, scales_size_bytes)
        if err_S != 0:
            memptr_S.free()
            raise Error("CRITICAL: Failed to allocate aligned memory for Q4 scales")
        var S_addr = memptr_S[0]
        memptr_S.free()
        self.scales = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=S_addr)

        self._libc = libc^
        var s1_blocks = max(1, (self.embed_dim // 32) // 4)
        self.s1_packed = alloc[UInt8](max(16, self.vocab_size * s1_blocks * 16))
        self.s1_scales = alloc[Float32](max(1, self.vocab_size * s1_blocks))
        self.row_tail_bound_25 = alloc[Float32](max(1, self.vocab_size))
        self.row_tail_bound_11 = alloc[Float32](max(1, self.vocab_size))
        self.row_tail_bound_75 = alloc[Float32](max(1, self.vocab_size))
        self.last_winner_tok = alloc[Int](1)
        self.last_winner_tok[0] = 0
        self.scratch_x_even = alloc[Float32](max(16, self.embed_dim // 2))
        self.scratch_x_odd = alloc[Float32](max(16, self.embed_dim // 2))
        self.scratch_shared_best = alloc[Float32](1)

    def finalize_tail_bounds(mut self):
        var num_blocks_init = self.embed_dim // 32
        var b_25 = max(1, num_blocks_init // 4)
        var b_50 = num_blocks_init // 2
        var b_75 = (num_blocks_init * 3) // 4
        var packed_init_ptr = self.packed_weights
        var scales_init_ptr = self.scales
        var s1_pack_ptr = self.s1_packed
        var s1_sc_ptr = self.s1_scales
        var bound25_ptr = self.row_tail_bound_25
        var bound50_ptr = self.row_tail_bound_11
        var bound75_ptr = self.row_tail_bound_75
        var half_dim_init = self.embed_dim // 2
        var vs_init = self.vocab_size
        var init_tasks = max(1, num_physical_cores())
        if vs_init < init_tasks:
            init_tasks = max(1, vs_init)

        def init_tail_bounds_worker(t: Int) capturing -> None:
            var chunk = vs_init // init_tasks
            var s_r = t * chunk
            var e_r = (t + 1) * chunk if t < init_tasks - 1 else vs_init
            for r in range(s_r, e_r):
                var r_pack = r * half_dim_init
                var r_off = r * num_blocks_init
                var s1_p_off = r * b_25 * 16
                var s1_s_off = r * b_25
                for b in range(b_25):
                    s1_sc_ptr[s1_s_off + b] = scales_init_ptr[r_off + b]
                    s1_pack_ptr.store[16](s1_p_off + b * 16, packed_init_ptr.load[16](r_pack + b * 16))

                var sq_75: Float32 = 0.0
                for b in range(b_75, num_blocks_init):
                    var sc = scales_init_ptr[r_off + b]
                    var bytes = packed_init_ptr.load[16](r_pack + b * 16)
                    var we = ((bytes & 15).cast[DType.int8]() - 8).cast[DType.float32]()
                    var wo = ((bytes >> 4).cast[DType.int8]() - 8).cast[DType.float32]()
                    sq_75 += (sc * sc) * (we * we + wo * wo).reduce_add()
                bound75_ptr[r] = sqrt(sq_75)

                var sq_50 = sq_75
                for b in range(b_50, b_75):
                    var sc = scales_init_ptr[r_off + b]
                    var bytes = packed_init_ptr.load[16](r_pack + b * 16)
                    var we = ((bytes & 15).cast[DType.int8]() - 8).cast[DType.float32]()
                    var wo = ((bytes >> 4).cast[DType.int8]() - 8).cast[DType.float32]()
                    sq_50 += (sc * sc) * (we * we + wo * wo).reduce_add()
                bound50_ptr[r] = sqrt(sq_50)

                var sq_25 = sq_50
                for b in range(b_25, b_50):
                    var sc = scales_init_ptr[r_off + b]
                    var bytes = packed_init_ptr.load[16](r_pack + b * 16)
                    var we = ((bytes & 15).cast[DType.int8]() - 8).cast[DType.float32]()
                    var wo = ((bytes >> 4).cast[DType.int8]() - 8).cast[DType.float32]()
                    sq_25 += (sc * sc) * (we * we + wo * wo).reduce_add()
                bound25_ptr[r] = sqrt(sq_25)

        if init_tasks == 1:
            init_tail_bounds_worker(0)
        else:
            parallelize[init_tail_bounds_worker](init_tasks)

    def free(mut self):
        var c_free = self._libc.get_function[def(Int) thin abi("C") -> None]("free")
        c_free(Int(self.packed_weights))
        c_free(Int(self.scales))
        self.s1_packed.free()
        self.s1_scales.free()
        self.row_tail_bound_25.free()
        self.row_tail_bound_11.free()
        self.row_tail_bound_75.free()
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
    def _eval_row_sieve_tail(
        self,
        r: Int,
        var acc: SIMD[DType.float32, 16],
        b_25: Int,
        b_50: Int,
        b_75: Int,
        num_blocks: Int,
        half_dim: Int,
        x_even: UnsafePointer[Float32, _],
        x_odd: UnsafePointer[Float32, _],
        x_rms_50: Float32,
        x_rms_75: Float32,
        mut prune_thresh: Float32,
        mut local_max_sum: Float32,
        mut local_tok: Int,
        shared_best: UnsafePointer[Float32, MutExternalOrigin],
    ) -> None:
        var r_pack = r * half_dim
        var r_sc = r * num_blocks

        # Stage 2: blocks [b_25 .. b_50)
        for b in range(b_25, b_50):
            var sc = self.scales[r_sc + b]
            var bytes = self.packed_weights.load[16](r_pack + b * 16)
            var we = ((bytes & 15).cast[DType.int8]() - 8).cast[DType.float32]()
            var wo = ((bytes >> 4).cast[DType.int8]() - 8).cast[DType.float32]()
            acc = acc + (we * x_even.load[16](b * 16) + wo * x_odd.load[16](b * 16)) * sc

        var p50 = acc.reduce_add()
        if p50 + self.row_tail_bound_11[r] * x_rms_50 < prune_thresh:
            return

        # Stage 3: blocks [b_50 .. b_75)
        for b in range(b_50, b_75):
            var sc = self.scales[r_sc + b]
            var bytes = self.packed_weights.load[16](r_pack + b * 16)
            var we = ((bytes & 15).cast[DType.int8]() - 8).cast[DType.float32]()
            var wo = ((bytes >> 4).cast[DType.int8]() - 8).cast[DType.float32]()
            acc = acc + (we * x_even.load[16](b * 16) + wo * x_odd.load[16](b * 16)) * sc

        var p75 = acc.reduce_add()
        if p75 + self.row_tail_bound_75[r] * x_rms_75 < prune_thresh:
            return

        # Stage 4: blocks [b_75 .. num_blocks)
        for b in range(b_75, num_blocks):
            var sc = self.scales[r_sc + b]
            var bytes = self.packed_weights.load[16](r_pack + b * 16)
            var we = ((bytes & 15).cast[DType.int8]() - 8).cast[DType.float32]()
            var wo = ((bytes >> 4).cast[DType.int8]() - 8).cast[DType.float32]()
            acc = acc + (we * x_even.load[16](b * 16) + wo * x_odd.load[16](b * 16)) * sc

        var sum = acc.reduce_add()
        if sum > local_max_sum:
            local_max_sum = sum
            local_tok = r
            if sum > prune_thresh:
                prune_thresh = sum
                shared_best[0] = sum

    @always_inline
    def project_logits_parallel(
        self,
        x: UnsafePointer[Float32, _],
        local_max_logits: UnsafePointer[Float32, MutExternalOrigin],
        local_next_tokens: UnsafePointer[Int, MutExternalOrigin],
        num_tasks: Int,
        softcap: Bool = True,
    ) -> None:
        """Fast Dual-Space Deinterleaved + 3-Tier L3-Resident Subspace Sieve logit projection."""
        var chunk_size = self.vocab_size // num_tasks
        var num_blocks = self.embed_dim // 32
        var b_25 = max(1, num_blocks // 4)
        var b_50 = max(b_25, num_blocks // 2)
        var b_75 = max(b_50, (num_blocks * 3) // 4)

        # 1. Dual-Space Deinterleaving + 3-tier tail RMS norms:
        var half_dim = self.embed_dim // 2
        var x_even = self.scratch_x_even
        var x_odd = self.scratch_x_odd
        var sq_25_50 = SIMD[DType.float32, 16](0.0)
        var sq_50_75 = SIMD[DType.float32, 16](0.0)
        var sq_75_100 = SIMD[DType.float32, 16](0.0)
        for b in range(num_blocks):
            var v0 = x.load[16](b * 32)
            var v1 = x.load[16](b * 32 + 16)
            var ev = v0.deinterleave()[0].join(v1.deinterleave()[0])
            var od = v0.deinterleave()[1].join(v1.deinterleave()[1])
            x_even.store[16](b * 16, ev)
            x_odd.store[16](b * 16, od)
            if b >= b_75:
                sq_75_100 = sq_75_100 + ev * ev + od * od
            elif b >= b_50:
                sq_50_75 = sq_50_75 + ev * ev + od * od
            elif b >= b_25:
                sq_25_50 = sq_25_50 + ev * ev + od * od
        var sum_75 = sq_75_100.reduce_add()
        var sum_50 = sum_75 + sq_50_75.reduce_add()
        var sum_25 = sum_50 + sq_25_50.reduce_add()
        var x_rms_75 = sqrt(sum_75)
        var x_rms_50 = sqrt(sum_50)
        var x_rms_25 = sqrt(sum_25)

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
        var s1_row_bytes = b_25 * 16

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

                var r0_s1_p = r * s1_row_bytes
                var r1_s1_p = (r + 1) * s1_row_bytes
                var r0_s1_sc = r * b_25
                var r1_s1_sc = (r + 1) * b_25
                var acc0 = SIMD[DType.float32, 16](0.0)
                var acc1 = SIMD[DType.float32, 16](0.0)

                # Stage 1: Evaluate first 25% of blocks from contiguous L3-resident s1_packed / s1_scales
                for b in range(b_25):
                    var xe = x_even.load[16](b * 16)
                    var xo = x_odd.load[16](b * 16)
                    var sc0 = self.s1_scales[r0_s1_sc + b]
                    var b0 = self.s1_packed.load[16](r0_s1_p + b * 16)
                    var we0 = ((b0 & 15).cast[DType.int8]() - 8).cast[DType.float32]()
                    var wo0 = ((b0 >> 4).cast[DType.int8]() - 8).cast[DType.float32]()
                    acc0 = acc0 + (we0 * xe + wo0 * xo) * sc0

                    var sc1 = self.s1_scales[r1_s1_sc + b]
                    var b1 = self.s1_packed.load[16](r1_s1_p + b * 16)
                    var we1 = ((b1 & 15).cast[DType.int8]() - 8).cast[DType.float32]()
                    var wo1 = ((b1 >> 4).cast[DType.int8]() - 8).cast[DType.float32]()
                    acc1 = acc1 + (we1 * xe + wo1 * xo) * sc1

                var p0 = acc0.reduce_add()
                if p0 + self.row_tail_bound_25[r] * x_rms_25 >= prune_thresh:
                    self._eval_row_sieve_tail(
                        r, acc0, b_25, b_50, b_75, num_blocks, half_dim,
                        x_even, x_odd, x_rms_50, x_rms_75,
                        prune_thresh, local_max_sum, local_tok, shared_best,
                    )

                var p1 = acc1.reduce_add()
                if p1 + self.row_tail_bound_25[r + 1] * x_rms_25 >= prune_thresh:
                    self._eval_row_sieve_tail(
                        r + 1, acc1, b_25, b_50, b_75, num_blocks, half_dim,
                        x_even, x_odd, x_rms_50, x_rms_75,
                        prune_thresh, local_max_sum, local_tok, shared_best,
                    )

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
            if softcap:
                local_max_logits[t] = Float32(30.0) * tanh(local_max_sum / Float32(30.0))
            else:
                local_max_logits[t] = local_max_sum
            local_next_tokens[t] = local_tok

        if num_tasks == 1:
            initial_proj_worker(0)
        else:
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
    def project_logits_batch5_parallel(
        self,
        x_batch: UnsafePointer[Float32, _],
        seed_tokens: UnsafePointer[Int, _],
        out_next_tokens: UnsafePointer[Int, MutUntrackedOrigin],
        num_tasks: Int,
        softcap: Bool = True,
    ) -> None:
        """5-token coalesced Dual-Space Deinterleaved + 3-Tier Subspace Sieve logit projection.
        Streams the 25% Stage-1 prefix from contiguous L3-friendly memory across all 5 tokens,
        and selectively evaluates Stages 2/3/4 ONLY for surviving (row, token) pairs.
        """
        var chunk_size = self.vocab_size // num_tasks
        var num_blocks = self.embed_dim // 32
        var b_25 = max(1, num_blocks // 4)
        var b_50 = max(b_25, num_blocks // 2)
        var b_75 = max(b_50, (num_blocks * 3) // 4)
        var half_dim = self.embed_dim // 2

        var stack_deint = InlineArray[Float32, 5 * 2816](uninitialized=True)
        var x_deint: UnsafePointer[Float32, MutUntrackedOrigin]
        if self.embed_dim <= 2816:
            x_deint = UnsafePointer[Float32, MutUntrackedOrigin](
                unsafe_from_address=Int(UnsafePointer(to=stack_deint))
            )
        else:
            x_deint = alloc[Float32](5 * self.embed_dim)

        var stack_rms25 = InlineArray[Float32, 5](uninitialized=True)
        var stack_rms50 = InlineArray[Float32, 5](uninitialized=True)
        var stack_rms75 = InlineArray[Float32, 5](uninitialized=True)
        for tok in range(5):
            var x_src = x_batch + tok * self.embed_dim
            var xe_dst = x_deint + tok * self.embed_dim
            var xo_dst = xe_dst + half_dim
            var sq_25_50 = SIMD[DType.float32, 16](0.0)
            var sq_50_75 = SIMD[DType.float32, 16](0.0)
            var sq_75_100 = SIMD[DType.float32, 16](0.0)
            for b in range(num_blocks):
                var v0 = x_src.load[16](b * 32)
                var v1 = x_src.load[16](b * 32 + 16)
                var ev = v0.deinterleave()[0].join(v1.deinterleave()[0])
                var od = v0.deinterleave()[1].join(v1.deinterleave()[1])
                xe_dst.store[16](b * 16, ev)
                xo_dst.store[16](b * 16, od)
                if b >= b_75:
                    sq_75_100 = sq_75_100 + ev * ev + od * od
                elif b >= b_50:
                    sq_50_75 = sq_50_75 + ev * ev + od * od
                elif b >= b_25:
                    sq_25_50 = sq_25_50 + ev * ev + od * od
            var s75 = sq_75_100.reduce_add()
            var s50 = s75 + sq_50_75.reduce_add()
            var s25 = s50 + sq_25_50.reduce_add()
            stack_rms75[tok] = sqrt(s75)
            stack_rms50[tok] = sqrt(s50)
            stack_rms25[tok] = sqrt(s25)

        var xe0 = x_deint + 0 * self.embed_dim
        var xo0 = xe0 + half_dim
        var xe1 = x_deint + 1 * self.embed_dim
        var xo1 = xe1 + half_dim
        var xe2 = x_deint + 2 * self.embed_dim
        var xo2 = xe2 + half_dim
        var xe3 = x_deint + 3 * self.embed_dim
        var xo3 = xe3 + half_dim
        var xe4 = x_deint + 4 * self.embed_dim
        var xo4 = xe4 + half_dim

        var rms25_0 = stack_rms25[0]
        var rms25_1 = stack_rms25[1]
        var rms25_2 = stack_rms25[2]
        var rms25_3 = stack_rms25[3]
        var rms25_4 = stack_rms25[4]

        var rms50_0 = stack_rms50[0]
        var rms50_1 = stack_rms50[1]
        var rms50_2 = stack_rms50[2]
        var rms50_3 = stack_rms50[3]
        var rms50_4 = stack_rms50[4]

        var rms75_0 = stack_rms75[0]
        var rms75_1 = stack_rms75[1]
        var rms75_2 = stack_rms75[2]
        var rms75_3 = stack_rms75[3]
        var rms75_4 = stack_rms75[4]

        var seed_count = 256 if self.vocab_size >= 256 else self.vocab_size
        var stack_seed_sum = InlineArray[Float32, 5](uninitialized=True)
        var stack_seed_tok = InlineArray[Int, 5](uninitialized=True)
        var stack_shared_best = InlineArray[Float32, 5](uninitialized=True)
        var shared_best_ptr = UnsafePointer[Float32, MutUntrackedOrigin](
            unsafe_from_address=Int(UnsafePointer(to=stack_shared_best))
        )

        for tok in range(5):
            var xe_t = x_deint + tok * self.embed_dim
            var xo_t = xe_t + half_dim
            var best_s: Float32 = -1e30
            var best_i: Int = 0
            var cand = seed_tokens[tok]
            if cand >= seed_count and cand < self.vocab_size:
                best_s = self._eval_row_full(cand, xe_t, xo_t, num_blocks)
                best_i = cand
            for r in range(seed_count):
                var s = self._eval_row_full(r, xe_t, xo_t, num_blocks)
                if s > best_s or (s == best_s and r < best_i):
                    best_s = s
                    best_i = r
            stack_seed_sum[tok] = best_s
            stack_seed_tok[tok] = best_i
            shared_best_ptr[tok] = best_s

        var stack_thread_sum = InlineArray[Float32, 64 * 5](uninitialized=True)
        var stack_thread_tok = InlineArray[Int, 64 * 5](uninitialized=True)
        var thread_sum_ptr = UnsafePointer[Float32, MutUntrackedOrigin](
            unsafe_from_address=Int(UnsafePointer(to=stack_thread_sum))
        )
        var thread_tok_ptr = UnsafePointer[Int, MutUntrackedOrigin](
            unsafe_from_address=Int(UnsafePointer(to=stack_thread_tok))
        )

        var s_sum0 = stack_seed_sum[0]
        var s_sum1 = stack_seed_sum[1]
        var s_sum2 = stack_seed_sum[2]
        var s_sum3 = stack_seed_sum[3]
        var s_sum4 = stack_seed_sum[4]
        var s_tok0 = stack_seed_tok[0]
        var s_tok1 = stack_seed_tok[1]
        var s_tok2 = stack_seed_tok[2]
        var s_tok3 = stack_seed_tok[3]
        var s_tok4 = stack_seed_tok[4]
        var s1_row_bytes = b_25 * 16

        def batch5_proj_worker(t: Int) capturing -> None:
            var start_r = t * chunk_size
            if start_r < seed_count:
                start_r = seed_count
            var end_r = (t + 1) * chunk_size if t < num_tasks - 1 else self.vocab_size

            var lmax0: Float32 = s_sum0 if t == 0 else -1e30
            var lmax1: Float32 = s_sum1 if t == 0 else -1e30
            var lmax2: Float32 = s_sum2 if t == 0 else -1e30
            var lmax3: Float32 = s_sum3 if t == 0 else -1e30
            var lmax4: Float32 = s_sum4 if t == 0 else -1e30

            var ltok0: Int = s_tok0 if t == 0 else start_r
            var ltok1: Int = s_tok1 if t == 0 else start_r
            var ltok2: Int = s_tok2 if t == 0 else start_r
            var ltok3: Int = s_tok3 if t == 0 else start_r
            var ltok4: Int = s_tok4 if t == 0 else start_r

            var pth0: Float32 = shared_best_ptr[0]
            var pth1: Float32 = shared_best_ptr[1]
            var pth2: Float32 = shared_best_ptr[2]
            var pth3: Float32 = shared_best_ptr[3]
            var pth4: Float32 = shared_best_ptr[4]

            for r in range(start_r, end_r):
                if (r & 63) == 0:
                    if shared_best_ptr[0] > pth0:
                        pth0 = shared_best_ptr[0]
                    if shared_best_ptr[1] > pth1:
                        pth1 = shared_best_ptr[1]
                    if shared_best_ptr[2] > pth2:
                        pth2 = shared_best_ptr[2]
                    if shared_best_ptr[3] > pth3:
                        pth3 = shared_best_ptr[3]
                    if shared_best_ptr[4] > pth4:
                        pth4 = shared_best_ptr[4]

                var r_s1_p = r * s1_row_bytes
                var r_s1_sc = r * b_25
                var acc0 = SIMD[DType.float32, 16](0.0)
                var acc1 = SIMD[DType.float32, 16](0.0)
                var acc2 = SIMD[DType.float32, 16](0.0)
                var acc3 = SIMD[DType.float32, 16](0.0)
                var acc4 = SIMD[DType.float32, 16](0.0)

                # Stage 1: first 25% of blocks from contiguous s1_packed / s1_scales
                for b in range(b_25):
                    var sc = self.s1_scales[r_s1_sc + b]
                    var bytes = self.s1_packed.load[16](r_s1_p + b * 16)
                    var we = ((bytes & 15).cast[DType.int8]() - 8).cast[DType.float32]() * sc
                    var wo = ((bytes >> 4).cast[DType.int8]() - 8).cast[DType.float32]() * sc
                    acc0 = acc0 + (we * xe0.load[16](b * 16) + wo * xo0.load[16](b * 16))
                    acc1 = acc1 + (we * xe1.load[16](b * 16) + wo * xo1.load[16](b * 16))
                    acc2 = acc2 + (we * xe2.load[16](b * 16) + wo * xo2.load[16](b * 16))
                    acc3 = acc3 + (we * xe3.load[16](b * 16) + wo * xo3.load[16](b * 16))
                    acc4 = acc4 + (we * xe4.load[16](b * 16) + wo * xo4.load[16](b * 16))

                var tb25 = self.row_tail_bound_25[r]
                var p0 = acc0.reduce_add()
                var p1 = acc1.reduce_add()
                var p2 = acc2.reduce_add()
                var p3 = acc3.reduce_add()
                var p4 = acc4.reduce_add()

                var k0 = (p0 + tb25 * rms25_0 >= pth0)
                var k1 = (p1 + tb25 * rms25_1 >= pth1)
                var k2 = (p2 + tb25 * rms25_2 >= pth2)
                var k3 = (p3 + tb25 * rms25_3 >= pth3)
                var k4 = (p4 + tb25 * rms25_4 >= pth4)

                if k0 or k1 or k2 or k3 or k4:
                    var r_pack = r * half_dim
                    var r_sc = r * num_blocks

                    # Stage 2: blocks [b_25 .. b_50) with zero inner-loop branches
                    for b in range(b_25, b_50):
                        var sc = self.scales[r_sc + b]
                        var bytes = self.packed_weights.load[16](r_pack + b * 16)
                        var we = ((bytes & 15).cast[DType.int8]() - 8).cast[DType.float32]() * sc
                        var wo = ((bytes >> 4).cast[DType.int8]() - 8).cast[DType.float32]() * sc
                        acc0 = acc0 + (we * xe0.load[16](b * 16) + wo * xo0.load[16](b * 16))
                        acc1 = acc1 + (we * xe1.load[16](b * 16) + wo * xo1.load[16](b * 16))
                        acc2 = acc2 + (we * xe2.load[16](b * 16) + wo * xo2.load[16](b * 16))
                        acc3 = acc3 + (we * xe3.load[16](b * 16) + wo * xo3.load[16](b * 16))
                        acc4 = acc4 + (we * xe4.load[16](b * 16) + wo * xo4.load[16](b * 16))

                    var tb50 = self.row_tail_bound_11[r]
                    k0 = (acc0.reduce_add() + tb50 * rms50_0 >= pth0)
                    k1 = (acc1.reduce_add() + tb50 * rms50_1 >= pth1)
                    k2 = (acc2.reduce_add() + tb50 * rms50_2 >= pth2)
                    k3 = (acc3.reduce_add() + tb50 * rms50_3 >= pth3)
                    k4 = (acc4.reduce_add() + tb50 * rms50_4 >= pth4)

                    if k0 or k1 or k2 or k3 or k4:
                        # Stage 3: blocks [b_50 .. b_75)
                        for b in range(b_50, b_75):
                            var sc = self.scales[r_sc + b]
                            var bytes = self.packed_weights.load[16](r_pack + b * 16)
                            var we = ((bytes & 15).cast[DType.int8]() - 8).cast[DType.float32]() * sc
                            var wo = ((bytes >> 4).cast[DType.int8]() - 8).cast[DType.float32]() * sc
                            acc0 = acc0 + (we * xe0.load[16](b * 16) + wo * xo0.load[16](b * 16))
                            acc1 = acc1 + (we * xe1.load[16](b * 16) + wo * xo1.load[16](b * 16))
                            acc2 = acc2 + (we * xe2.load[16](b * 16) + wo * xo2.load[16](b * 16))
                            acc3 = acc3 + (we * xe3.load[16](b * 16) + wo * xo3.load[16](b * 16))
                            acc4 = acc4 + (we * xe4.load[16](b * 16) + wo * xo4.load[16](b * 16))

                        var tb75 = self.row_tail_bound_75[r]
                        k0 = (acc0.reduce_add() + tb75 * rms75_0 >= pth0)
                        k1 = (acc1.reduce_add() + tb75 * rms75_1 >= pth1)
                        k2 = (acc2.reduce_add() + tb75 * rms75_2 >= pth2)
                        k3 = (acc3.reduce_add() + tb75 * rms75_3 >= pth3)
                        k4 = (acc4.reduce_add() + tb75 * rms75_4 >= pth4)

                        if k0 or k1 or k2 or k3 or k4:
                            # Stage 4: blocks [b_75 .. num_blocks)
                            for b in range(b_75, num_blocks):
                                var sc = self.scales[r_sc + b]
                                var bytes = self.packed_weights.load[16](r_pack + b * 16)
                                var we = ((bytes & 15).cast[DType.int8]() - 8).cast[DType.float32]() * sc
                                var wo = ((bytes >> 4).cast[DType.int8]() - 8).cast[DType.float32]() * sc
                                acc0 = acc0 + (we * xe0.load[16](b * 16) + wo * xo0.load[16](b * 16))
                                acc1 = acc1 + (we * xe1.load[16](b * 16) + wo * xo1.load[16](b * 16))
                                acc2 = acc2 + (we * xe2.load[16](b * 16) + wo * xo2.load[16](b * 16))
                                acc3 = acc3 + (we * xe3.load[16](b * 16) + wo * xo3.load[16](b * 16))
                                acc4 = acc4 + (we * xe4.load[16](b * 16) + wo * xo4.load[16](b * 16))

                            var sum0 = acc0.reduce_add()
                            if sum0 > lmax0:
                                lmax0 = sum0
                                ltok0 = r
                                if sum0 > pth0:
                                    pth0 = sum0
                                    shared_best_ptr[0] = sum0
                            var sum1 = acc1.reduce_add()
                            if sum1 > lmax1:
                                lmax1 = sum1
                                ltok1 = r
                                if sum1 > pth1:
                                    pth1 = sum1
                                    shared_best_ptr[1] = sum1
                            var sum2 = acc2.reduce_add()
                            if sum2 > lmax2:
                                lmax2 = sum2
                                ltok2 = r
                                if sum2 > pth2:
                                    pth2 = sum2
                                    shared_best_ptr[2] = sum2
                            var sum3 = acc3.reduce_add()
                            if sum3 > lmax3:
                                lmax3 = sum3
                                ltok3 = r
                                if sum3 > pth3:
                                    pth3 = sum3
                                    shared_best_ptr[3] = sum3
                            var sum4 = acc4.reduce_add()
                            if sum4 > lmax4:
                                lmax4 = sum4
                                ltok4 = r
                                if sum4 > pth4:
                                    pth4 = sum4
                                    shared_best_ptr[4] = sum4

            var base = t * 5
            if softcap:
                thread_sum_ptr[base + 0] = Float32(30.0) * tanh(lmax0 / Float32(30.0))
                thread_sum_ptr[base + 1] = Float32(30.0) * tanh(lmax1 / Float32(30.0))
                thread_sum_ptr[base + 2] = Float32(30.0) * tanh(lmax2 / Float32(30.0))
                thread_sum_ptr[base + 3] = Float32(30.0) * tanh(lmax3 / Float32(30.0))
                thread_sum_ptr[base + 4] = Float32(30.0) * tanh(lmax4 / Float32(30.0))
            else:
                thread_sum_ptr[base + 0] = lmax0
                thread_sum_ptr[base + 1] = lmax1
                thread_sum_ptr[base + 2] = lmax2
                thread_sum_ptr[base + 3] = lmax3
                thread_sum_ptr[base + 4] = lmax4
            thread_tok_ptr[base + 0] = ltok0
            thread_tok_ptr[base + 1] = ltok1
            thread_tok_ptr[base + 2] = ltok2
            thread_tok_ptr[base + 3] = ltok3
            thread_tok_ptr[base + 4] = ltok4

        if num_tasks == 1:
            batch5_proj_worker(0)
        else:
            parallelize[batch5_proj_worker](num_tasks)

        for tok in range(5):
            var best_l: Float32 = -1e30
            var best_i: Int = 0
            for t in range(num_tasks):
                var tl = thread_sum_ptr[t * 5 + tok]
                var ti = thread_tok_ptr[t * 5 + tok]
                if tl > best_l or (tl == best_l and ti < best_i):
                    best_l = tl
                    best_i = ti
            out_next_tokens[tok] = best_i

        self.last_winner_tok[0] = out_next_tokens[4]
        var _keep_alive = stack_deint[0] + stack_shared_best[0] + stack_thread_sum[0] + Float32(stack_thread_tok[0])
        if _keep_alive > 1e38:
            out_next_tokens[0] += 0
        if self.embed_dim > 2816:
            x_deint.free()

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
