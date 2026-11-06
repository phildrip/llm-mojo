from std.ffi import OwnedDLHandle
from std.memory import alloc

# =============================================================================
# Q/K/V Binary Dump format for Gemma-4 26B MoE Global Layers.
#
# Schema:
#   Fixed 64-byte Header:
#     [0]:  UInt32 Magic ('QKVD' = 0x44564B51)
#     [1]:  UInt32 Version = 1
#     [2]:  UInt32 Num Global Layers = 5
#     [3]:  UInt32 Layer 0 = 5
#     [4]:  UInt32 Layer 1 = 11
#     [5]:  UInt32 Layer 2 = 17
#     [6]:  UInt32 Layer 3 = 23
#     [7]:  UInt32 Layer 4 = 29
#     [8]:  UInt32 Num Q Heads = 16
#     [9]:  UInt32 Num KV Heads = 2
#     [10]: UInt32 Head Dim = 512
#     [11]: UInt32 Num Tokens (updated on close)
#     [12..15]: Reserved (0)
#
#   Records:
#     Stride per token: 5 layers * 40,960 bytes = 204,800 bytes.
#     Stride per layer-token:
#       q: 16 * 512 * 4 bytes = 32,768 bytes (offset + 0)
#       k:  2 * 512 * 4 bytes =  4,096 bytes (offset + 32,768)
#       v:  2 * 512 * 4 bytes =  4,096 bytes (offset + 36,864)
#       Total per record: 40,960 bytes (10 OS pages).
#
#     Byte offset of (pos, global_layer_idx):
#       64 + (pos * 5 + global_layer_idx) * 40960
# =============================================================================

comptime QK_MAGIC: UInt32 = 0x44564B51
comptime QK_VERSION: UInt32 = 1
comptime NUM_GLOBAL_LAYERS: Int = 5
comptime NUM_Q_HEADS: Int = 16
comptime NUM_KV_HEADS: Int = 2
comptime HEAD_DIM: Int = 512

comptime Q_FLOATS: Int = NUM_Q_HEADS * HEAD_DIM        # 8192
comptime K_FLOATS: Int = NUM_KV_HEADS * HEAD_DIM       # 1024
comptime V_FLOATS: Int = NUM_KV_HEADS * HEAD_DIM       # 1024
comptime RECORD_FLOATS: Int = Q_FLOATS + K_FLOATS + V_FLOATS  # 10240
comptime RECORD_BYTES: Int = RECORD_FLOATS * 4         # 40960
comptime TOKEN_RECORD_BYTES: Int = NUM_GLOBAL_LAYERS * RECORD_BYTES # 204800
comptime HEADER_BYTES: Int = 64

@always_inline
def global_layer_to_index(layer_num: Int) -> Int:
    if layer_num == 5:
        return 0
    if layer_num == 11:
        return 1
    if layer_num == 17:
        return 2
    if layer_num == 23:
        return 3
    if layer_num == 29:
        return 4
    return -1

@always_inline
def index_to_global_layer(idx: Int) -> Int:
    return idx * 6 + 5


struct QKDumpWriter:
    var fd: Int32
    var path: String
    var max_pos_written: Int
    var libc: OwnedDLHandle
    var c_pwrite: def(Int32, Int, Int, Int) thin abi("C") -> Int
    var c_close: def(Int32) thin abi("C") -> Int32
    var record_buf: UnsafePointer[Float32, MutUntrackedOrigin]

    def __init__(out self) raises:
        self.fd = -1
        self.path = String("")
        self.max_pos_written = -1
        self.libc = OwnedDLHandle("libc.so.6")
        self.c_pwrite = self.libc.get_function[def(Int32, Int, Int, Int) thin abi("C") -> Int]("pwrite64")
        self.c_close = self.libc.get_function[def(Int32) thin abi("C") -> Int32]("close")
        self.record_buf = alloc[Float32](RECORD_FLOATS)

    def __init__(out self, path: String) raises:
        self.path = path
        self.max_pos_written = -1
        self.libc = OwnedDLHandle("libc.so.6")
        var c_open = self.libc.get_function[def(Int, Int32, Int32) thin abi("C") -> Int32]("open")
        self.c_pwrite = self.libc.get_function[def(Int32, Int, Int, Int) thin abi("C") -> Int]("pwrite64")
        self.c_close = self.libc.get_function[def(Int32) thin abi("C") -> Int32]("close")
        self.record_buf = alloc[Float32](RECORD_FLOATS)

        var path_bytes = path.as_bytes()
        var path_ptr = alloc[UInt8](len(path_bytes) + 1)
        for i in range(len(path_bytes)):
            path_ptr[i] = path_bytes[i]
        path_ptr[len(path_bytes)] = 0
        # O_WRONLY | O_CREAT | O_TRUNC = 577, mode 0644 = 420
        self.fd = c_open(path_ptr.__int__(), 577, 420)
        path_ptr.free()
        if self.fd < 0:
            raise Error("Failed to open file for Q/K dump: " + path)

        # Write initial header
        var header = alloc[UInt32](16)
        for i in range(16):
            header[i] = 0
        header[0] = QK_MAGIC
        header[1] = QK_VERSION
        header[2] = UInt32(NUM_GLOBAL_LAYERS)
        header[3] = 5
        header[4] = 11
        header[5] = 17
        header[6] = 23
        header[7] = 29
        header[8] = UInt32(NUM_Q_HEADS)
        header[9] = UInt32(NUM_KV_HEADS)
        header[10] = UInt32(HEAD_DIM)
        header[11] = 0 # tokens count, updated on close

        var written = self.c_pwrite(self.fd, header.__int__(), HEADER_BYTES, 0)
        header.free()
        if written != HEADER_BYTES:
            raise Error("Failed to write header to Q/K dump: " + path)

    @always_inline
    def is_active(self) -> Bool:
        return self.fd >= 0

    def write_record(
        mut self,
        pos: Int,
        layer_idx: Int,
        q: UnsafePointer[Float32, _],
        k: UnsafePointer[Float32, _],
        v: UnsafePointer[Float32, _],
    ) raises:
        var g_idx = global_layer_to_index(layer_idx)
        if g_idx < 0:
            return # not a global layer

        if pos > self.max_pos_written:
            self.max_pos_written = pos

        # Assemble contiguous record: q [8192], k [1024], v [1024]
        for i in range(Q_FLOATS):
            self.record_buf[i] = q[i]
        for i in range(K_FLOATS):
            self.record_buf[Q_FLOATS + i] = k[i]
        for i in range(V_FLOATS):
            self.record_buf[Q_FLOATS + K_FLOATS + i] = v[i]

        var offset = HEADER_BYTES + (pos * NUM_GLOBAL_LAYERS + g_idx) * RECORD_BYTES
        var written = self.c_pwrite(self.fd, self.record_buf.__int__(), RECORD_BYTES, offset)
        if written != RECORD_BYTES:
            raise Error("Failed to write record for token " + String(pos) + " layer " + String(layer_idx))

    def close(mut self):
        if self.fd >= 0:
            # Update token count in header if tokens were written
            if self.max_pos_written >= 0:
                var num_tokens = UInt32(self.max_pos_written + 1)
                var count_ptr = alloc[UInt32](1)
                count_ptr[0] = num_tokens
                _ = self.c_pwrite(self.fd, count_ptr.__int__(), 4, 44)
                count_ptr.free()
            _ = self.c_close(self.fd)
            self.fd = -1
            self.record_buf.free()


struct QKDumpReader:
    var fd: Int32
    var path: String
    var num_tokens: Int
    var libc: OwnedDLHandle
    var c_pread: def(Int32, Int, Int, Int) thin abi("C") -> Int
    var c_close: def(Int32) thin abi("C") -> Int32

    def __init__(out self, path: String) raises:
        self.path = path
        self.num_tokens = 0
        self.libc = OwnedDLHandle("libc.so.6")
        var c_open = self.libc.get_function[def(Int, Int32, Int32) thin abi("C") -> Int32]("open")
        self.c_pread = self.libc.get_function[def(Int32, Int, Int, Int) thin abi("C") -> Int]("pread64")
        self.c_close = self.libc.get_function[def(Int32) thin abi("C") -> Int32]("close")
        var c_lseek = self.libc.get_function[def(Int32, Int, Int32) thin abi("C") -> Int]("lseek64")

        var path_bytes = path.as_bytes()
        var path_ptr = alloc[UInt8](len(path_bytes) + 1)
        for i in range(len(path_bytes)):
            path_ptr[i] = path_bytes[i]
        path_ptr[len(path_bytes)] = 0
        self.fd = c_open(path_ptr.__int__(), 0, 0) # O_RDONLY = 0
        path_ptr.free()
        if self.fd < 0:
            raise Error("Failed to open Q/K dump file: " + path)

        # Read and validate header
        var header = alloc[UInt32](16)
        var bytes_read = self.c_pread(self.fd, header.__int__(), HEADER_BYTES, 0)
        if bytes_read != HEADER_BYTES:
            header.free()
            _ = self.c_close(self.fd)
            raise Error("Failed to read header from Q/K dump: " + path)

        if header[0] != QK_MAGIC:
            header.free()
            _ = self.c_close(self.fd)
            raise Error("Invalid magic in Q/K dump: " + path)

        if header[1] != QK_VERSION:
            header.free()
            _ = self.c_close(self.fd)
            raise Error("Unsupported version in Q/K dump: " + path)

        var header_tokens = Int(header[11])
        header.free()

        # Compute tokens from file size as well
        var file_size = c_lseek(self.fd, 0, 2) # SEEK_END = 2
        var payload_bytes = file_size - HEADER_BYTES
        var tokens_from_file = payload_bytes // TOKEN_RECORD_BYTES
        if header_tokens > 0:
            self.num_tokens = header_tokens
        else:
            self.num_tokens = tokens_from_file

    def read_layer(
        self,
        pos: Int,
        global_layer_idx: Int,
        q_out: UnsafePointer[Float32, _],
        k_out: UnsafePointer[Float32, _],
        v_out: UnsafePointer[Float32, _],
    ) raises:
        if pos < 0 or pos >= self.num_tokens:
            raise Error("pos out of range: " + String(pos) + " vs " + String(self.num_tokens))
        if global_layer_idx < 0 or global_layer_idx >= NUM_GLOBAL_LAYERS:
            raise Error("global_layer_idx out of range: " + String(global_layer_idx))

        var offset = HEADER_BYTES + (pos * NUM_GLOBAL_LAYERS + global_layer_idx) * RECORD_BYTES
        # Read q
        var read_q = self.c_pread(self.fd, q_out.__int__(), Q_FLOATS * 4, offset)
        if read_q != Q_FLOATS * 4:
            raise Error("Failed to read Q at pos " + String(pos))
        # Read k
        var read_k = self.c_pread(self.fd, k_out.__int__(), K_FLOATS * 4, offset + Q_FLOATS * 4)
        if read_k != K_FLOATS * 4:
            raise Error("Failed to read K at pos " + String(pos))
        # Read v
        var read_v = self.c_pread(self.fd, v_out.__int__(), V_FLOATS * 4, offset + (Q_FLOATS + K_FLOATS) * 4)
        if read_v != V_FLOATS * 4:
            raise Error("Failed to read V at pos " + String(pos))

    def read_k_only(
        self,
        pos: Int,
        global_layer_idx: Int,
        k_out: UnsafePointer[Float32, _],
    ) raises:
        if pos < 0 or pos >= self.num_tokens:
            raise Error("pos out of range: " + String(pos) + " vs " + String(self.num_tokens))
        var offset = HEADER_BYTES + (pos * NUM_GLOBAL_LAYERS + global_layer_idx) * RECORD_BYTES + Q_FLOATS * 4
        var read_k = self.c_pread(self.fd, k_out.__int__(), K_FLOATS * 4, offset)
        if read_k != K_FLOATS * 4:
            raise Error("Failed to read K at pos " + String(pos))

    def read_q_only(
        self,
        pos: Int,
        global_layer_idx: Int,
        q_out: UnsafePointer[Float32, _],
    ) raises:
        if pos < 0 or pos >= self.num_tokens:
            raise Error("pos out of range: " + String(pos) + " vs " + String(self.num_tokens))
        var offset = HEADER_BYTES + (pos * NUM_GLOBAL_LAYERS + global_layer_idx) * RECORD_BYTES
        var read_q = self.c_pread(self.fd, q_out.__int__(), Q_FLOATS * 4, offset)
        if read_q != Q_FLOATS * 4:
            raise Error("Failed to read Q at pos " + String(pos))

    def close(mut self):
        if self.fd >= 0:
            _ = self.c_close(self.fd)
            self.fd = -1
