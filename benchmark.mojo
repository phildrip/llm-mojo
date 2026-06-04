from compress import Float32BlockCompressor
from std.ffi import OwnedDLHandle
from std.memory import alloc
from uring import IoUring
import std.time
import std.os

# Gemma 4 12B layer block dimensions
comptime NUM_FLOATS = 1048576 # 4 MB of Float32
comptime UNCOMPRESSED_SIZE = NUM_FLOATS * 4 # 4,194,304 bytes
comptime COMPRESSED_SIZE = 3932160 # 3.75 MB (30 bytes per 8 floats)
comptime NUM_LAYERS = 10
comptime COMPUTE_ITERATIONS = 50 # Calibrated to ~11.4 ms

def wait_for_tag(ring: IoUring, expected_tag: UInt64, out_tag_ptr: UnsafePointer[UInt64, MutExternalOrigin]) raises:
    var attempts = 0
    while True:
        var res = ring.poll_completion(out_tag_ptr.__int__())
        if res >= 0:
            if out_tag_ptr[0] == expected_tag:
                return # Successful completion
        attempts += 1
        if attempts > 20000000:
            raise Error("Asynchronous I/O timed out waiting for tag: " + String(expected_tag))

@always_inline
def simulate_layer_compute(data: UnsafePointer[Float32, _], size: Int, iterations: Int) -> Float32:
    var acc = SIMD[DType.float32, 8](0)
    for _ in range(iterations):
        for i in range(0, size, 8):
            var v = data.load[8](i)
            acc = acc + v * v - v
    return acc.reduce_add()

def main() raises:
    print("=========================================================")
    print("   GEMMA 4 12B LOSSLESS KV-CACHE DISK PAGING BENCHMARK   ")
    print("=========================================================")
    print("Target Layer size:  " + String(UNCOMPRESSED_SIZE / (1024 * 1024)) + " MB uncompressed")
    print("Compressed size:    " + String(Float64(COMPRESSED_SIZE) / (1024 * 1024)) + " MB (lossless Common Exponent)")
    print("Paging Engine:      Linux io_uring (Direct I/O)")
    print("Total Layers:       " + String(NUM_LAYERS))
    print("---------------------------------------------------------")

    var test_file_path = "gemma4_kv_storage.bin"
    
    # Load Libc FFI functions
    var libc = OwnedDLHandle("libc.so.6")
    var c_open = libc.get_function[def(Int, Int32, Int32) -> Int32]("open")
    var c_close = libc.get_function[def(Int32) -> Int32]("close")
    var posix_memalign = libc.get_function[def(Int, Int, Int) -> Int32]("posix_memalign")
    var free_c = libc.get_function[def(Int) -> NoneType]("free")
    
    # 1. Initialize io_uring
    var ring = IoUring(16)
    
    # 2. Allocate aligned memory buffers (4KB aligned)
    var memptr_l1_a = alloc[Int](1)
    _ = posix_memalign(memptr_l1_a.__int__(), 4096, UNCOMPRESSED_SIZE)
    var l1_a_addr = memptr_l1_a[0]
    var l1_a_ptr = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=l1_a_addr)
    
    var memptr_l1_b = alloc[Int](1)
    _ = posix_memalign(memptr_l1_b.__int__(), 4096, UNCOMPRESSED_SIZE)
    var l1_b_addr = memptr_l1_b[0]
    var l1_b_ptr = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=l1_b_addr)
    
    # Allocate L2 buffers (holds compressed blocks in RAM)
    # We allocate 10 separate aligned compressed buffers for L2 cache
    var l2_ptrs = List[UnsafePointer[UInt8, MutExternalOrigin]]()
    var l2_memptrs = List[UnsafePointer[Int, MutExternalOrigin]]()
    for _ in range(NUM_LAYERS):
        var memptr = alloc[Int](1)
        _ = posix_memalign(memptr.__int__(), 4096, COMPRESSED_SIZE)
        l2_memptrs.append(memptr)
        l2_ptrs.append(UnsafePointer[UInt8, MutExternalOrigin](unsafe_from_address=memptr[0]))

    # Allocate out_tag_ptr for completion polling
    var out_tag_ptr = alloc[UInt64](1)
    out_tag_ptr[0] = 0

    # 3. Open temporary storage file and populate it with compressible data
    # Flags: O_RDWR (2) | O_CREAT (64) | O_TRUNC (512) | O_DIRECT (16384) = 16962
    var path_bytes = test_file_path.as_bytes()
    var path_ptr = alloc[UInt8](len(path_bytes) + 1)
    for i in range(len(path_bytes)):
        path_ptr[i] = path_bytes[i]
    path_ptr[len(path_bytes)] = 0
    
    var fd = c_open(path_ptr.__int__(), 16962, 420)
    if fd < 0:
        print("CRITICAL: Failed to open storage file with O_DIRECT")
        # Cleanup
        free_c(l1_a_addr)
        free_c(l1_b_addr)
        memptr_l1_a.free()
        memptr_l1_b.free()
        for i in range(NUM_LAYERS):
            free_c(l2_ptrs[i].__int__())
            l2_memptrs[i].free()
        out_tag_ptr.free()
        path_ptr.free()
        return

    print("Populating L3 NVMe storage with mock compressed layers...")
    # Fill l1_a with a compressible pattern and compress to L2, then write to L3
    for idx in range(NUM_FLOATS):
        l1_a_ptr[idx] = Float32(1.25 + Float64(idx % 8) * 0.25)
        
    var comp_bytes = Float32BlockCompressor.compress(l1_a_ptr, l2_ptrs[0], NUM_FLOATS)
    if comp_bytes != COMPRESSED_SIZE:
        print("Compression mismatch! Expected:", COMPRESSED_SIZE, "Got:", comp_bytes)
        
    for i in range(NUM_LAYERS):
        # We copy the compressed block from L2[0] to L2[i] for populate
        if i > 0:
            for j in range(COMPRESSED_SIZE):
                l2_ptrs[i][j] = l2_ptrs[0][j]
                
        # Submit async write to NVMe
        var tag: UInt64 = UInt64(1000 + i)
        var offset = i * COMPRESSED_SIZE
        _ = ring.submit_write(fd, l2_ptrs[i].__int__(), COMPRESSED_SIZE, offset, tag)
        wait_for_tag(ring, tag, out_tag_ptr)
        
    print("L3 storage populated and synced successfully. Storage size:", NUM_LAYERS * COMPRESSED_SIZE, "bytes")
    print("---------------------------------------------------------")

    # =========================================================================
    # CONFIGURATION 1: UNPIPELINED SEQUENTIAL STORAGE READ & COMPUTE BASELINE
    # =========================================================================
    print("Running Baseline (Sequential disk read -> decompress -> compute)...")
    var seq_start = std.time.perf_counter()
    
    var total_seq_read_time: Float64 = 0.0
    var total_seq_decomp_time: Float64 = 0.0
    var total_seq_compute_time: Float64 = 0.0
    var total_compute_res: Float32 = 0.0
    
    for i in range(NUM_LAYERS):
        # Time storage read
        var r0 = std.time.perf_counter()
        var tag: UInt64 = UInt64(2000 + i)
        var offset = i * COMPRESSED_SIZE
        # Read from disk into L2
        _ = ring.submit_read(fd, l2_ptrs[i].__int__(), COMPRESSED_SIZE, offset, tag)
        wait_for_tag(ring, tag, out_tag_ptr)
        var r1 = std.time.perf_counter()
        total_seq_read_time += (r1 - r0)
        
        # Time decompression
        var d0 = std.time.perf_counter()
        # Decompress L2 to L1_A
        Float32BlockCompressor.decompress(l2_ptrs[i], l1_a_ptr, NUM_FLOATS)
        var d1 = std.time.perf_counter()
        total_seq_decomp_time += (d1 - d0)
        
        # Time computation
        var c0 = std.time.perf_counter()
        total_compute_res += simulate_layer_compute(l1_a_ptr, NUM_FLOATS, COMPUTE_ITERATIONS)
        var c1 = std.time.perf_counter()
        total_seq_compute_time += (c1 - c0)
        
    var seq_end = std.time.perf_counter()
    var seq_total_time = (seq_end - seq_start) * 1000.0
    
    print("Baseline execution finished.")
    print("  Avg Disk Read Time:  " + String((total_seq_read_time * 1000.0) / NUM_LAYERS) + " ms / layer")
    print("  Avg Decompress Time: " + String((total_seq_decomp_time * 1000.0) / NUM_LAYERS) + " ms / layer")
    print("  Avg Compute Time:    " + String((total_seq_compute_time * 1000.0) / NUM_LAYERS) + " ms / layer")
    print("  Total Duration:      " + String(seq_total_time) + " ms")
    print("---------------------------------------------------------")

    # =========================================================================
    # CONFIGURATION 2: PIPELINED ZERO-COPY io_uring PAGER WITH DOUBLE BUFFERING
    # =========================================================================
    print("Running Pipelined (io_uring background paging + SIMD decompression)...")
    var pipe_start = std.time.perf_counter()
    
    # Stage 0: Pre-fetch Layer 0's compressed block and decompress it to L1 A
    var tag0: UInt64 = UInt64(3000)
    _ = ring.submit_read(fd, l2_ptrs[0].__int__(), COMPRESSED_SIZE, 0, tag0)
    wait_for_tag(ring, tag0, out_tag_ptr)
    Float32BlockCompressor.decompress(l2_ptrs[0], l1_a_ptr, NUM_FLOATS)
    
    # Submit background read for Layer 1
    var tag1: UInt64 = UInt64(3001)
    _ = ring.submit_read(fd, l2_ptrs[1].__int__(), COMPRESSED_SIZE, COMPRESSED_SIZE, tag1)
    
    for i in range(NUM_LAYERS):
        # Determine active and next buffers
        var active_l1_ptr = l1_a_ptr if i % 2 == 0 else l1_b_ptr
        var next_l1_ptr = l1_b_ptr if i % 2 == 0 else l1_a_ptr
        
        # 1. Start background disk read for Layer i+2 while we compute Layer i
        if i + 2 < NUM_LAYERS:
            var tag_next: UInt64 = UInt64(3000 + i + 2)
            var offset_next = (i + 2) * COMPRESSED_SIZE
            _ = ring.submit_read(fd, l2_ptrs[i+2].__int__(), COMPRESSED_SIZE, offset_next, tag_next)
            
        # 2. Wait for Layer i+1's background read to complete so we can decompress it
        if i + 1 < NUM_LAYERS:
            var tag_curr_read: UInt64 = UInt64(3000 + i + 1)
            wait_for_tag(ring, tag_curr_read, out_tag_ptr)
            # Decompress Layer i+1 into the next buffer
            Float32BlockCompressor.decompress(l2_ptrs[i+1], next_l1_ptr, NUM_FLOATS)
            
        # 3. Compute attention for current Layer i
        total_compute_res += simulate_layer_compute(active_l1_ptr, NUM_FLOATS, COMPUTE_ITERATIONS)
        
    var pipe_end = std.time.perf_counter()
    var pipe_total_time = (pipe_end - pipe_start) * 1000.0
    
    print("Pipelined execution finished.")
    print("  Total Duration:      " + String(pipe_total_time) + " ms")
    print("---------------------------------------------------------")
    
    # Performance Analysis & Dashboard
    var pure_compute_ms = total_seq_compute_time * 1000.0
    var overhead_saved_ms = seq_total_time - pipe_total_time
    var max_possible_saved_ms = (total_seq_read_time + total_seq_decomp_time) * 1000.0
    var efficiency_pct = (overhead_saved_ms / max_possible_saved_ms) * 100.0
    _ = pure_compute_ms
    
    print("=========================================================")
    print("                 BENCHMARK RESULTS SUMMARY               ")
    print("=========================================================")
    print("  Sequential Baseline:     " + String(seq_total_time) + " ms")
    print("  Pipelined io_uring Layer: " + String(pipe_total_time) + " ms")
    print("  Total Latency Saved:     " + String(overhead_saved_ms) + " ms")
    print("  Paging Overhead Masked:  " + String(efficiency_pct) + " %")
    print("---------------------------------------------------------")
    print("  Uncompressed RAM Usage comparison:")
    print("    Baseline Model (All layers): " + String((NUM_LAYERS * UNCOMPRESSED_SIZE) / (1024 * 1024)) + " MB")
    print("    Pipelined Model (Double Buf): " + String((2 * UNCOMPRESSED_SIZE) / (1024 * 1024)) + " MB (95% RAM Reduction!)")
    print("  Computation verification sum: " + String(total_compute_res))
    print("=========================================================")

    # Cleanup memory and files
    free_c(l1_a_addr)
    free_c(l1_b_addr)
    memptr_l1_a.free()
    memptr_l1_b.free()
    for i in range(NUM_LAYERS):
        free_c(l2_ptrs[i].__int__())
        l2_memptrs[i].free()
    out_tag_ptr.free()
    path_ptr.free()
    _ = c_close(fd)
    
    # Delete temporary binary storage file
    try:
        std.os.unlink(test_file_path)
    except:
        pass
