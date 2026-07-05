from std.ffi import OwnedDLHandle
from std.memory import alloc
from uring import IoUring
import std.os

def main() raises:
    # 1. Create a temporary test file with some content
    var test_file_path = "io_test.txt"
    var content = "Mojo + io_uring is a high-performance combination for Gemma 4 lossless paging!"
    
    try:
        with open(test_file_path, "w") as f:
            f.write(content)
        print("Created test file:", test_file_path)
    except e:
        print("Failed to write test file:", e)
        return

    # 2. Load libc to get raw file descriptor open and close functions
    var libc = OwnedDLHandle("libc.so.6")
    # int open(const char* pathname, int flags)
    var c_open = libc.get_function[def(Int, Int32) thin abi("C") -> Int32]("open")
    # int close(int fd)
    var c_close = libc.get_function[def(Int32) thin abi("C") -> Int32]("close")

    # Open our file using O_RDONLY (0)
    var path_bytes = test_file_path.as_bytes()
    # Allocate a buffer to hold the null-terminated path
    var path_ptr = alloc[UInt8](len(path_bytes) + 1)
    for i in range(len(path_bytes)):
        path_ptr[i] = path_bytes[i]
    path_ptr[len(path_bytes)] = 0 # null terminator
    
    var fd = c_open(path_ptr.__int__(), 0) # O_RDONLY = 0
    if fd < 0:
        print("Failed to open file via C open, fd =", fd)
        path_ptr.free()
        return
    print("Opened file with descriptor fd =", fd)

    # 3. Initialize our io_uring queue
    var ring = IoUring(8)
    print("Initialized io_uring queue successfully!")

    # 4. Allocate buffer for read data (128 bytes)
    var buffer_size = 128
    var read_buf = alloc[UInt8](buffer_size)
    # Zero out the buffer
    for i in range(buffer_size):
        read_buf[i] = 0

    # 5. Submit an asynchronous read request
    var user_data: UInt64 = 42 # Arbitrary tag to identify this read
    var submit_res = ring.submit_read(fd, read_buf.__int__(), buffer_size, 0, user_data)
    if submit_res < 0:
        print("Failed to submit read to io_uring, error =", submit_res)
        read_buf.free()
        path_ptr.free()
        _ = c_close(fd)
        return
    print("Submitted async read request to kernel...")

    # 6. Poll completion in a non-blocking loop
    # Allocate out_tag on the heap
    var out_tag_ptr = alloc[UInt64](1)
    out_tag_ptr[0] = 0
    
    var bytes_transferred = -1
    var attempts = 0
    while attempts < 100000:
        var res = ring.poll_completion(out_tag_ptr.__int__())
        if res >= 0:
            bytes_transferred = Int(res)
            print("Completed I/O! Tag received:", out_tag_ptr[0], "Bytes read:", bytes_transferred)
            break
        # Sleep a microscopic amount or loop
        attempts += 1

    # 7. Print results
    if bytes_transferred >= 0:
        # Convert buffer bytes back to a String and print it
        var result_str = String("")
        for i in range(bytes_transferred):
            result_str += String(chr(Int(read_buf[i])))
        print("Read from disk asynchronously: '" + result_str + "'")
    else:
        print("I/O timed out or failed to complete after attempts:", attempts)

    # Cleanup memory and files
    read_buf.free()
    path_ptr.free()
    out_tag_ptr.free()
    _ = c_close(fd)
    
    # Delete temp file
    try:
        std.os.unlink(test_file_path)
    except:
        pass
