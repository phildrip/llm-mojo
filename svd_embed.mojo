from std.ffi import OwnedDLHandle
from std.memory import alloc

struct SvdEmbedding:
    var rank: Int
    var vocab_size: Int
    var embed_dim: Int
    var A: UnsafePointer[Float32, MutExternalOrigin]
    var B: UnsafePointer[Float32, MutExternalOrigin]
    var _libc: OwnedDLHandle

    def __init__(out self, path: String) raises:
        # Load libc FFI
        var libc = OwnedDLHandle("libc.so.6")
        var c_open = libc.get_function[def(Int, Int32, Int32) -> Int32]("open")
        var c_close = libc.get_function[def(Int32) -> Int32]("close")
        var posix_memalign = libc.get_function[def(Int, Int, Int) -> Int32]("posix_memalign")
        var c_read = libc.get_function[def(Int32, Int, Int) -> Int]("read")

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
            raise Error("CRITICAL: Failed to open SVD embedding file: " + path)

        # Read 12-byte header
        var header_ptr = alloc[Int32](3)
        var header_read = c_read(fd, header_ptr.__int__(), 12)
        if header_read != 12:
            _ = c_close(fd)
            header_ptr.free()
            raise Error("CRITICAL: Failed to read header from SVD embedding file")

        self.rank = Int(header_ptr[0])
        self.vocab_size = Int(header_ptr[1])
        self.embed_dim = Int(header_ptr[2])
        header_ptr.free()

        # Allocate page-aligned memory for Factor A
        var a_size_bytes = self.vocab_size * self.rank * 4
        var memptr_A = alloc[Int](1)
        var err_A = posix_memalign(memptr_A.__int__(), 4096, a_size_bytes)
        if err_A != 0:
            _ = c_close(fd)
            memptr_A.free()
            raise Error("CRITICAL: Failed to allocate aligned memory for Factor A")
        var A_addr = memptr_A[0]
        memptr_A.free()
        self.A = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=A_addr)

        # Allocate page-aligned memory for Factor B
        var b_size_bytes = self.rank * self.embed_dim * 4
        var memptr_B = alloc[Int](1)
        var err_B = posix_memalign(memptr_B.__int__(), 4096, b_size_bytes)
        if err_B != 0:
            _ = c_close(fd)
            memptr_B.free()
            # Free previously allocated A memory
            # (Note: we don't have a direct free FFI call here, but for simplicity we rely on system-level cleanup on error)
            raise Error("CRITICAL: Failed to allocate aligned memory for Factor B")
        var B_addr = memptr_B[0]
        memptr_B.free()
        self.B = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=B_addr)

        # Read Factor A
        var total_read_A = 0
        while total_read_A < a_size_bytes:
            var bytes_to_read = a_size_bytes - total_read_A
            var read_res = c_read(fd, A_addr + total_read_A, bytes_to_read)
            if read_res <= 0:
                _ = c_close(fd)
                raise Error("CRITICAL: Error or EOF reading Factor A")
            total_read_A += read_res

        # Read Factor B
        var total_read_B = 0
        while total_read_B < b_size_bytes:
            var bytes_to_read = b_size_bytes - total_read_B
            var read_res = c_read(fd, B_addr + total_read_B, bytes_to_read)
            if read_res <= 0:
                _ = c_close(fd)
                raise Error("CRITICAL: Error or EOF reading Factor B")
            total_read_B += read_res

        _ = c_close(fd)
        self._libc = libc^ # Transfer ownership to preserve FFI function lifespans
        print("SVD Embedding Loaded successfully:")
        print("  Rank:      ", self.rank)
        print("  Vocab:     ", self.vocab_size)
        print("  Dimension: ", self.embed_dim)

    def reconstruct(self, token: Int, out_ptr: UnsafePointer[Float32, MutExternalOrigin]) -> None:
        # Zero-heap on-the-fly reconstruction using AVX/SIMD vector instructions
        # Output is computed as the matrix-vector product: out_ptr = A[token, :] * B
        var t_offset = token * self.rank
        
        # Loop over the embedding dimension in steps of 8 for SIMD vectorization
        for c in range(0, self.embed_dim, 8):
            var acc = SIMD[DType.float32, 8](0.0)
            for r in range(self.rank):
                var a_val = self.A[t_offset + r]
                var b_vec = self.B.load[8](r * self.embed_dim + c)
                acc = acc + a_val * b_vec
            out_ptr.store[8](c, acc)
