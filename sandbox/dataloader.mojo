from std.ffi import OwnedDLHandle
from std.memory import alloc
from std.random import rand

struct DataLoader:
    var data: UnsafePointer[UInt16, MutExternalOrigin]
    var total_tokens: Int

    def __init__(out self, filepath: String) raises:
        # Open file and load it fully into memory
        var libc = OwnedDLHandle("libc.so.6")
        var c_open = libc.get_function[def(Int, Int32, Int32) -> Int32]("open")
        var c_close = libc.get_function[def(Int32) -> Int32]("close")
        var c_read = libc.get_function[def(Int32, Int, Int) -> Int]("read")
        var c_lseek = libc.get_function[def(Int32, Int, Int32) -> Int]("lseek")
        var posix_memalign = libc.get_function[def(Int, Int, Int) -> Int32]("posix_memalign")
        
        # Open file
        var path_bytes = filepath.as_bytes()
        var path_ptr = alloc[UInt8](len(path_bytes) + 1)
        for i in range(len(path_bytes)):
            path_ptr[i] = path_bytes[i]
        path_ptr[len(path_bytes)] = 0
        
        var fd = c_open(path_ptr.__int__(), 0, 0)
        path_ptr.free()
        if fd < 0:
            raise Error("Failed to open file: " + filepath)
            
        # Get file size
        var file_size = c_lseek(fd, 0, 2) # SEEK_END = 2
        _ = c_lseek(fd, 0, 0) # SEEK_SET = 0
        
        self.total_tokens = file_size // 2 # uint16 is 2 bytes
        
        # Allocate page-aligned memory for data
        var memptr = alloc[Int](1)
        _ = posix_memalign(memptr.__int__(), 4096, file_size)
        var addr = memptr[0]
        memptr.free()
        
        self.data = UnsafePointer[UInt16, MutExternalOrigin](unsafe_from_address=addr)
        
        # Read the file
        var total_read = 0
        while total_read < file_size:
            var bytes_to_read = file_size - total_read
            var read_res = c_read(fd, addr + total_read, bytes_to_read)
            if read_res <= 0:
                break
            total_read += read_res
            
        _ = c_close(fd)
        print("DataLoader successfully loaded", self.total_tokens, "tokens from", filepath)

    def get_batch(self, batch_size: Int, seq_len: Int, x: UnsafePointer[Int32, MutExternalOrigin], y: UnsafePointer[Int32, MutExternalOrigin]):
        var r_ptr = alloc[Float64](1)
        # Draw random start positions and slice
        for b in range(batch_size):
            rand(r_ptr, 1)
            var r = r_ptr[0]
            var max_pos = self.total_tokens - seq_len - 1
            var start_pos = Int(r * Float64(max_pos))
            
            for t in range(seq_len):
                var x_val = Int32(self.data[start_pos + t])
                var y_val = Int32(self.data[start_pos + t + 1])
                x[b * seq_len + t] = x_val
                y[b * seq_len + t] = y_val
        r_ptr.free()

