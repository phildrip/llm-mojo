from std.ffi import OwnedDLHandle

struct IoUring:
    var lib: OwnedDLHandle
    var ring_addr: Int
    var _init_uring: def(Int32) thin abi("C") -> Int
    var _submit_io: def(Int, Int32, Int, Int, Int, UInt64, Int32) thin abi("C") -> Int32
    var _poll_completion: def(Int, Int) thin abi("C") -> Int32
    var _close_uring: def(Int) thin abi("C") -> NoneType

    def __init__(out self, entries: Int32) raises:
        self.lib = OwnedDLHandle("./liburing_pager.so")
        self._init_uring = self.lib.get_function[def(Int32) thin abi("C") -> Int]("init_uring")
        self._submit_io = self.lib.get_function[def(Int, Int32, Int, Int, Int, UInt64, Int32) thin abi("C") -> Int32]("submit_io")
        self._poll_completion = self.lib.get_function[def(Int, Int) thin abi("C") -> Int32]("poll_completion")
        self._close_uring = self.lib.get_function[def(Int) thin abi("C") -> NoneType]("close_uring")
        
        self.ring_addr = self._init_uring(entries)
        if self.ring_addr == 0:
            raise Error("Failed to initialize io_uring queue")

    def submit_read(self, file_fd: Int32, buf_addr: Int, len: Int, offset: Int, user_data: UInt64) -> Int32:
        return self._submit_io(self.ring_addr, file_fd, buf_addr, len, offset, user_data, 0)

    def submit_write(self, file_fd: Int32, buf_addr: Int, len: Int, offset: Int, user_data: UInt64) -> Int32:
        return self._submit_io(self.ring_addr, file_fd, buf_addr, len, offset, user_data, 1)

    def poll_completion(self, out_user_data_addr: Int) -> Int32:
        return self._poll_completion(self.ring_addr, out_user_data_addr)

    def __del__(deinit self):
        if self.ring_addr != 0:
            self._close_uring(self.ring_addr)
        # Force the compiler's lifetime tracker to keep self.lib alive
        # until the very end of the destructor, preventing eager dlclose.
        _ = self.lib
