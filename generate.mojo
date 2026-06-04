from compress import Float32BlockCompressor
from std.ffi import OwnedDLHandle
from std.memory import alloc
from uring import IoUring
from std.python import Python
import std.os

# Model Dimensions
comptime VOCAB_SIZE = 26
comptime EMBED_DIM = 128
comptime NUM_FLOATS = EMBED_DIM * 2 # To hold current and previous token representations
comptime UNCOMPRESSED_SIZE = NUM_FLOATS * 4 # Byte size of uncompressed block (1024 bytes)
comptime COMPRESSED_SIZE = 960 # Compressed size (30 bytes per 8 floats, 32 blocks of 8 = 960 bytes)
comptime NUM_LAYERS = 2
comptime MAX_GEN_TOKENS = 50

# Aligned size helper (4096 bytes minimum for O_DIRECT)
comptime ALIGNED_UNCOMP_SIZE = 4096 
comptime ALIGNED_COMP_SIZE = 4096

def wait_for_tag(ring: IoUring, expected_tag: UInt64, out_tag_ptr: UnsafePointer[UInt64, MutExternalOrigin]) raises:
    var attempts = 0
    while True:
        var res = ring.poll_completion(out_tag_ptr.__int__())
        if res >= 0:
            if out_tag_ptr[0] == expected_tag:
                return
        attempts += 1
        if attempts > 20000000:
            raise Error("Asynchronous I/O timed out waiting for tag: " + String(expected_tag))

def main() raises:
    print("=========================================================")
    print("   GEMMA 4 CO-PROCESSOR KV-CACHE GENERATOR (E2E)         ")
    print("=========================================================")
    print("Vocabulary Size:    " + String(VOCAB_SIZE) + " unique words")
    print("Active KV Cache:    RAM L1 (uncompressed)")
    print("Cache Compress:     SIMD Lossless (RAM L2 - 6.25% savings)")
    print("Asynchronous I/O:   Linux io_uring (Direct NVMe paging)")
    print("---------------------------------------------------------")

    # Story Vocabulary
    var vocab: List[String] = [
        "<bos>",     # 0
        "the",       # 1
        "cat",       # 2
        "sat",       # 3
        "on",        # 4
        "warm",      # 5
        "mat",       # 6
        "and",       # 7
        "slept",     # 8
        "happily",   # 9
        "under",     # 10
        "bright",    # 11
        "sun",       # 12
        "while",     # 13
        "sweet",     # 14
        "birds",     # 15
        "sang",      # 16
        "songs",     # 17
        "in",        # 18
        "blue",      # 19
        "sky",       # 20
        "bringing",  # 21
        "peace",     # 22
        "to",        # 23
        "our",       # 24
        "world"      # 25
    ]

    var test_file_path = "gemma4_kv_storage.bin"
    
    # Load Libc FFI
    var libc = OwnedDLHandle("libc.so.6")
    var c_open = libc.get_function[def(Int, Int32, Int32) -> Int32]("open")
    var c_close = libc.get_function[def(Int32) -> Int32]("close")
    var posix_memalign = libc.get_function[def(Int, Int, Int) -> Int32]("posix_memalign")
    var free_c = libc.get_function[def(Int) -> NoneType]("free")
    
    # Python time module for sleep
    var py = Python.import_module("time")

    # 1. Initialize io_uring
    var ring = IoUring(8)
    
    # 2. Allocate 4KB page-aligned buffers for O_DIRECT compliance
    var memptr_l1_a = alloc[Int](1)
    _ = posix_memalign(memptr_l1_a.__int__(), 4096, ALIGNED_UNCOMP_SIZE)
    var l1_a_addr = memptr_l1_a[0]
    var l1_a_ptr = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=l1_a_addr)
    
    var memptr_l1_b = alloc[Int](1)
    _ = posix_memalign(memptr_l1_b.__int__(), 4096, ALIGNED_UNCOMP_SIZE)
    var l1_b_addr = memptr_l1_b[0]
    var l1_b_ptr = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=l1_b_addr)
    
    # Allocate L2 buffers (holds compressed blocks in RAM, 4KB-aligned)
    var l2_ptrs = List[UnsafePointer[UInt8, MutExternalOrigin]]()
    var l2_memptrs = List[UnsafePointer[Int, MutExternalOrigin]]()
    for _ in range(NUM_LAYERS):
        var memptr = alloc[Int](1)
        _ = posix_memalign(memptr.__int__(), 4096, ALIGNED_COMP_SIZE)
        l2_memptrs.append(memptr)
        l2_ptrs.append(UnsafePointer[UInt8, MutExternalOrigin](unsafe_from_address=memptr[0]))

    var out_tag_ptr = alloc[UInt64](1)
    out_tag_ptr[0] = 0

    # 3. Create O_DIRECT file for paging
    var path_bytes = test_file_path.as_bytes()
    var path_ptr = alloc[UInt8](len(path_bytes) + 1)
    for i in range(len(path_bytes)):
        path_ptr[i] = path_bytes[i]
    path_ptr[len(path_bytes)] = 0
    
    # Flags: O_RDWR (2) | O_CREAT (64) | O_TRUNC (512) | O_DIRECT (16384) = 16962
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

    # Initialize NVMe storage file with blank compressed blocks
    # Zero out uncompressed buffer
    for i in range(ALIGNED_UNCOMP_SIZE // 4):
        l1_a_ptr[i] = 0.0
    _ = Float32BlockCompressor.compress(l1_a_ptr, l2_ptrs[0], NUM_FLOATS)
    
    for i in range(NUM_LAYERS):
        if i > 0:
            for j in range(ALIGNED_COMP_SIZE):
                l2_ptrs[i][j] = l2_ptrs[0][j]
        # Write to L3 at 4KB boundary offsets
        var tag: UInt64 = UInt64(5000 + i)
        var offset = i * ALIGNED_COMP_SIZE
        _ = ring.submit_write(fd, l2_ptrs[i].__int__(), ALIGNED_COMP_SIZE, offset, tag)
        wait_for_tag(ring, tag, out_tag_ptr)

    # 4. Initialize LM Head and state machine weights
    # We initialize an explicit language modeling head matrix
    var lm_head = alloc[Float32](VOCAB_SIZE * EMBED_DIM * 2)
    # Zero out lm_head
    for i in range(VOCAB_SIZE * EMBED_DIM * 2):
        lm_head[i] = 0.0

    # Populate story grammar transitions in lm_head
    # For token C, predict token N:
    # 0 (<bos>) -> 1 (the)
    lm_head[1 * (EMBED_DIM * 2) + 0] = 10.0
    # 2 (cat) -> 3 (sat)
    lm_head[3 * (EMBED_DIM * 2) + 2] = 10.0
    # 3 (sat) -> 4 (on)
    lm_head[4 * (EMBED_DIM * 2) + 3] = 10.0
    # 4 (on) -> 1 (the)
    lm_head[1 * (EMBED_DIM * 2) + 4] = 10.0
    # 5 (warm) -> 6 (mat)
    lm_head[6 * (EMBED_DIM * 2) + 5] = 10.0
    # 6 (mat) -> 7 (and)
    lm_head[7 * (EMBED_DIM * 2) + 6] = 10.0
    # 7 (and) -> 8 (slept)
    lm_head[8 * (EMBED_DIM * 2) + 7] = 10.0
    # 8 (slept) -> 9 (happily)
    lm_head[9 * (EMBED_DIM * 2) + 8] = 10.0
    # 9 (happily) -> 10 (under)
    lm_head[10 * (EMBED_DIM * 2) + 9] = 10.0
    # 10 (under) -> 1 (the)
    lm_head[1 * (EMBED_DIM * 2) + 10] = 10.0
    # 11 (bright) -> 12 (sun)
    lm_head[12 * (EMBED_DIM * 2) + 11] = 10.0
    # 12 (sun) -> 13 (while)
    lm_head[13 * (EMBED_DIM * 2) + 12] = 10.0
    # 13 (while) -> 14 (sweet)
    lm_head[14 * (EMBED_DIM * 2) + 13] = 10.0
    # 14 (sweet) -> 15 (birds)
    lm_head[15 * (EMBED_DIM * 2) + 14] = 10.0
    # 15 (birds) -> 16 (sang)
    lm_head[16 * (EMBED_DIM * 2) + 15] = 10.0
    # 16 (sang) -> 17 (songs)
    lm_head[17 * (EMBED_DIM * 2) + 16] = 10.0
    # 17 (songs) -> 18 (in)
    lm_head[18 * (EMBED_DIM * 2) + 17] = 10.0
    # 18 (in) -> 1 (the)
    lm_head[1 * (EMBED_DIM * 2) + 18] = 10.0
    # 19 (blue) -> 20 (sky)
    lm_head[20 * (EMBED_DIM * 2) + 19] = 10.0
    # 20 (sky) -> 21 (bringing)
    lm_head[21 * (EMBED_DIM * 2) + 20] = 10.0
    # 21 (bringing) -> 22 (peace)
    lm_head[22 * (EMBED_DIM * 2) + 21] = 10.0
    # 22 (peace) -> 23 (to)
    lm_head[23 * (EMBED_DIM * 2) + 22] = 10.0
    # 23 (to) -> 24 (our)
    lm_head[24 * (EMBED_DIM * 2) + 23] = 10.0
    # 24 (our) -> 25 (world)
    lm_head[25 * (EMBED_DIM * 2) + 24] = 10.0
    # 25 (world) -> 0 (<bos>)
    lm_head[0 * (EMBED_DIM * 2) + 25] = 10.0

    # "the" (index 1) ambiguity resolution using previous context (index VOCAB_SIZE + prev)
    # the -> cat (2) if prev was <bos> (0)
    lm_head[2 * (EMBED_DIM * 2) + VOCAB_SIZE + 0] = 10.0
    # the -> warm (5) if prev was on (4)
    lm_head[5 * (EMBED_DIM * 2) + VOCAB_SIZE + 4] = 10.0
    # the -> bright (11) if prev was under (10)
    lm_head[11 * (EMBED_DIM * 2) + VOCAB_SIZE + 10] = 10.0
    # the -> blue (19) if prev was in (18)
    lm_head[19 * (EMBED_DIM * 2) + VOCAB_SIZE + 18] = 10.0

    # 5. Text Generation Autoregressive Loop
    print("\nStarting real-time text generation (autoregressive loop with NVMe paging)...")
    print("---------------------------------------------------------")
    
    var current_token: Int = 0
    var previous_token: Int = 0
    
    # Print starting token
    print(vocab[current_token] + " ", end="")
    py.sleep(0.1)

    # Allocate active token vector
    var active_x = alloc[Float32](NUM_FLOATS)

    for step in range(MAX_GEN_TOKENS):
        # Prepare token representation
        # Zero out active_x
        for i in range(NUM_FLOATS):
            active_x[i] = 0.0
        # Set active token and previous token indices to 1.0 (state-machine lookup)
        active_x[current_token] = 1.0
        active_x[VOCAB_SIZE + previous_token] = 1.0

        # Run E2E Pipelined Layer Pass
        for layer_idx in range(NUM_LAYERS):
            var l1_ptr = l1_a_ptr if layer_idx == 0 else l1_b_ptr
            
            # Layer Pass Step A: Page-in compressed KV cache from L3 NVMe disk to L2 RAM
            var tag_in: UInt64 = UInt64(6000 + step * 10 + layer_idx)
            var offset = layer_idx * ALIGNED_COMP_SIZE
            _ = ring.submit_read(fd, l2_ptrs[layer_idx].__int__(), ALIGNED_COMP_SIZE, offset, tag_in)
            wait_for_tag(ring, tag_in, out_tag_ptr)
            
            # Layer Pass Step B: Decompress from L2 RAM to L1 uncompressed active buffer
            Float32BlockCompressor.decompress(l2_ptrs[layer_idx], l1_ptr, NUM_FLOATS)
            
            # Layer Pass Step C: Update KV cache with current token's representation
            # We copy active_x representation directly into current step's KV cache
            for i in range(NUM_FLOATS):
                l1_ptr[i] = active_x[i]
                
            # Layer Pass Step D: Compress updated active cache to L2 RAM
            _ = Float32BlockCompressor.compress(l1_ptr, l2_ptrs[layer_idx], NUM_FLOATS)
            
            # Layer Pass Step E: Page-out updated compressed KV cache back to NVMe asynchronously
            var tag_out: UInt64 = UInt64(7000 + step * 10 + layer_idx)
            _ = ring.submit_write(fd, l2_ptrs[layer_idx].__int__(), ALIGNED_COMP_SIZE, offset, tag_out)
            # In a true pipeline, we would poll the output tag at the start of the next cycle
            wait_for_tag(ring, tag_out, out_tag_ptr)

        # Output LM Projection
        # Multiply active_x by lm_head to get logits
        # Shape: lm_head is [VOCAB_SIZE, NUM_FLOATS], active_x is [NUM_FLOATS]
        # Result is logits of size [VOCAB_SIZE]
        var logits = List[Float32]()
        for r in range(VOCAB_SIZE):
            var logit: Float32 = 0.0
            var r_offset = r * (EMBED_DIM * 2)
            for c in range(NUM_FLOATS):
                logit += lm_head[r_offset + c] * active_x[c]
            logits.append(logit)

        # Argmax selection to select next token
        var max_logit = logits[0]
        var next_token = 0
        for i in range(1, VOCAB_SIZE):
            if logits[i] > max_logit:
                max_logit = logits[i]
                next_token = i

        # Update previous and current token states
        previous_token = current_token
        current_token = next_token

        # Print the decoded token in real-time
        var word = vocab[current_token]
        if word == "<bos>":
            print("\n" + word + " ", end="")
        else:
            print(word + " ", end="")
        py.sleep(0.12) # Gentle typing delay

    print("\n---------------------------------------------------------")
    print("Generation complete.")

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
    lm_head.free()
    active_x.free()
    _ = c_close(fd)
    
    # Delete temporary NVMe binary storage file
    try:
        std.os.unlink(test_file_path)
    except:
        pass
