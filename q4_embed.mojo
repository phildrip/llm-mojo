from std.ffi import OwnedDLHandle
from std.memory import alloc
from std.algorithm import parallelize
from std.sys.info import num_physical_cores
from std.math import tanh

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
        print("Q4 Embedding Loaded successfully:")
        print("  Vocab:     ", self.vocab_size)
        print("  Dimension: ", self.embed_dim)
        print("  RAM:       ", (packed_size_bytes + scales_size_bytes) // 1024 // 1024, "MB")

    @always_inline
    def reconstruct(self, token: Int, out_ptr: UnsafePointer[Float32, MutExternalOrigin]) -> None:
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
                
                out_ptr[out_idx]     = Float32(Int(b0 & 15) - 8) * scale
                out_ptr[out_idx + 1] = Float32(Int(b0 >> 4) - 8) * scale
                out_ptr[out_idx + 2] = Float32(Int(b1 & 15) - 8) * scale
                out_ptr[out_idx + 3] = Float32(Int(b1 >> 4) - 8) * scale
                out_ptr[out_idx + 4] = Float32(Int(b2 & 15) - 8) * scale
                out_ptr[out_idx + 5] = Float32(Int(b2 >> 4) - 8) * scale
                out_ptr[out_idx + 6] = Float32(Int(b3 & 15) - 8) * scale
                out_ptr[out_idx + 7] = Float32(Int(b3 >> 4) - 8) * scale

    @always_inline
    def project_logits_parallel(
        self,
        x: UnsafePointer[Float32, _],
        local_max_logits: UnsafePointer[Float32, MutExternalOrigin],
        local_next_tokens: UnsafePointer[Int, MutExternalOrigin],
        num_tasks: Int,
    ) -> None:
        var chunk_size = self.vocab_size // num_tasks
        var num_blocks = self.embed_dim // 32
        
        def initial_proj_worker(t: Int) capturing -> None:
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
                # Apply fast tanh and scaling as per standard gemma4 scaling
                var logit = Float32(30.0) * tanh(sum / Float32(30.0))
                if logit > local_max_l:
                    local_max_l = logit
                    local_tok = r
            local_max_logits[t] = local_max_l
            local_next_tokens[t] = local_tok
            
        parallelize[initial_proj_worker](num_tasks)

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

        parallelize[full_proj_worker](num_tasks)
