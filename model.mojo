from std.ffi import OwnedDLHandle
from std.memory import alloc
from std.math import sqrt, exp, cos, sin
from std.python import Python, PythonObject
import std.os
from uring import IoUring
from std.algorithm import parallelize
from std.sys.info import num_physical_cores
from q4_embed import Q4Embedding
from std.sys import argv

# ==============================================================================
# KFD IOCTL Arguments & Structures
# ==============================================================================

struct kfd_ioctl_acquire_vm_args:
    var drm_fd: UInt32
    var gpu_id: UInt32

    fn __init__(out self, drm_fd: UInt32, gpu_id: UInt32):
        self.drm_fd = drm_fd
        self.gpu_id = gpu_id

struct kfd_ioctl_runtime_enable_args:
    var r_debug: UInt64
    var mode_mask: UInt32
    var capabilities_mask: UInt32

    fn __init__(out self):
        self.r_debug = 0
        self.mode_mask = 0
        self.capabilities_mask = 0

struct kfd_ioctl_alloc_memory_of_gpu_args:
    var va_addr: UInt64
    var size: UInt64
    var handle: UInt64
    var mmap_offset: UInt64
    var gpu_id: UInt32
    var flags: UInt32

    fn __init__(out self, va_addr: UInt64, size: UInt64, gpu_id: UInt32, flags: UInt32):
        self.va_addr = va_addr
        self.size = size
        self.handle = 0
        self.mmap_offset = 0
        self.gpu_id = gpu_id
        self.flags = flags

struct kfd_ioctl_map_memory_to_gpu_args:
    var handle: UInt64
    var device_ids_array_ptr: UInt64
    var n_devices: UInt32
    var pad: UInt32

    fn __init__(out self, handle: UInt64, device_ids_array_ptr: UInt64, n_devices: UInt32):
        self.handle = handle
        self.device_ids_array_ptr = device_ids_array_ptr
        self.n_devices = n_devices
        self.pad = 0

struct kfd_ioctl_unmap_memory_from_gpu_args:
    var handle: UInt64
    var device_ids_array_ptr: UInt64
    var n_devices: UInt32
    var pad: UInt32

    fn __init__(out self, handle: UInt64, device_ids_array_ptr: UInt64, n_devices: UInt32):
        self.handle = handle
        self.device_ids_array_ptr = device_ids_array_ptr
        self.n_devices = n_devices
        self.pad = 0

struct kfd_ioctl_free_memory_of_gpu_args:
    var handle: UInt64

    fn __init__(out self, handle: UInt64):
        self.handle = handle

struct kfd_ioctl_create_queue_args:
    var ring_base_address: UInt64
    var write_pointer_address: UInt64
    var read_pointer_address: UInt64
    var doorbell_offset: UInt64
    var ring_size: UInt32
    var gpu_id: UInt32
    var queue_type: UInt32
    var queue_percentage: UInt32
    var queue_priority: UInt32
    var queue_id: UInt32
    var eop_buffer_address: UInt64
    var eop_buffer_size: UInt64
    var ctx_save_restore_address: UInt64
    var ctx_save_restore_size: UInt32
    var ctl_stack_size: UInt32
    var sdma_engine_id: UInt32
    var pad: UInt32

    fn __init__(
        out self,
        ring_base_address: UInt64,
        write_pointer_address: UInt64,
        read_pointer_address: UInt64,
        ring_size: UInt32,
        gpu_id: UInt32,
        queue_type: UInt32,
        queue_percentage: UInt32,
        queue_priority: UInt32,
        eop_buffer_address: UInt64 = 0,
        eop_buffer_size: UInt64 = 0,
        ctx_save_restore_address: UInt64 = 0,
        ctx_save_restore_size: UInt32 = 0,
        ctl_stack_size: UInt32 = 0,
        sdma_engine_id: UInt32 = 0
    ):
        self.ring_base_address = ring_base_address
        self.write_pointer_address = write_pointer_address
        self.read_pointer_address = read_pointer_address
        self.doorbell_offset = 0
        self.ring_size = ring_size
        self.gpu_id = gpu_id
        self.queue_type = queue_type
        self.queue_percentage = queue_percentage
        self.queue_priority = queue_priority
        self.queue_id = 0
        self.eop_buffer_address = eop_buffer_address
        self.eop_buffer_size = eop_buffer_size
        self.ctx_save_restore_address = ctx_save_restore_address
        self.ctx_save_restore_size = ctx_save_restore_size
        self.ctl_stack_size = ctl_stack_size
        self.sdma_engine_id = sdma_engine_id
        self.pad = 0

struct GPUMemoryBuffer(Copyable, Movable):
    var va_addr: Int
    var size: Int
    var handle: UInt64
    var cpu_addr: Int

    fn __init__(out self):
        self.va_addr = 0
        self.size = 0
        self.handle = 0
        self.cpu_addr = 0

    fn __init__(out self, va_addr: Int, size: Int, handle: UInt64, cpu_addr: Int):
        self.va_addr = va_addr
        self.size = size
        self.handle = handle
        self.cpu_addr = cpu_addr

    fn __init__(out self, *, copy: Self):
        self.va_addr = copy.va_addr
        self.size = copy.size
        self.handle = copy.handle
        self.cpu_addr = copy.cpu_addr

    fn __init__(out self, *, deinit take: Self):
        self.va_addr = take.va_addr
        self.size = take.size
        self.handle = take.handle
        self.cpu_addr = take.cpu_addr

# ==============================================================================
# HSA AQL Dispatch Packet layout (64 bytes)
# ==============================================================================

struct hsa_kernel_dispatch_packet_t(Copyable, Movable):
    var header: UInt16
    var setup: UInt16
    var workgroup_size_x: UInt16
    var workgroup_size_y: UInt16
    var workgroup_size_z: UInt16
    var reserved0: UInt16
    var grid_size_x: UInt32
    var grid_size_y: UInt32
    var grid_size_z: UInt32
    var private_segment_size: UInt32
    var group_segment_size: UInt32
    var kernel_object: UInt64
    var kernarg_address: UInt64
    var reserved2: UInt64
    var completion_signal: UInt64

    fn __init__(
        out self,
        header: UInt16,
        setup: UInt16,
        workgroup_size_x: UInt16,
        workgroup_size_y: UInt16,
        workgroup_size_z: UInt16,
        grid_size_x: UInt32,
        grid_size_y: UInt32,
        grid_size_z: UInt32,
        private_segment_size: UInt32,
        group_segment_size: UInt32,
        kernel_object: UInt64,
        kernarg_address: UInt64,
        completion_signal: UInt64
    ):
        self.header = header
        self.setup = setup
        self.workgroup_size_x = workgroup_size_x
        self.workgroup_size_y = workgroup_size_y
        self.workgroup_size_z = workgroup_size_z
        self.reserved0 = 0
        self.grid_size_x = grid_size_x
        self.grid_size_y = grid_size_y
        self.grid_size_z = grid_size_z
        self.private_segment_size = private_segment_size
        self.group_segment_size = group_segment_size
        self.kernel_object = kernel_object
        self.kernarg_address = kernarg_address
        self.reserved2 = 0
        self.completion_signal = completion_signal

# ==============================================================================
# GPUContext Definition
# ==============================================================================

struct GPUContext(Movable):
    var kfd_fd: Int32
    var drm_fd: Int32
    var gpu_id: UInt32
    var libc: OwnedDLHandle
    var cpu_fence_lib: OwnedDLHandle
    var ring_buf: GPUMemoryBuffer
    var gart_buf: GPUMemoryBuffer
    var cwsr_buf: GPUMemoryBuffer
    var eop_buf: GPUMemoryBuffer
    var doorbell_reg_addr: Int
    var hsaco_buf: GPUMemoryBuffer
    var kernarg_buf: GPUMemoryBuffer
    var completion_buf: GPUMemoryBuffer
    var packet_idx: UInt64
    var weights_gpu_buf: GPUMemoryBuffer
    var x_buf: GPUMemoryBuffer
    var norm_x_buf: GPUMemoryBuffer
    var norm_x_buf_1: GPUMemoryBuffer
    var final_norm_x_buf: GPUMemoryBuffer
    var q_buf: GPUMemoryBuffer
    var q_buf_1: GPUMemoryBuffer
    var k_vec_buf: GPUMemoryBuffer
    var k_vec_buf_1: GPUMemoryBuffer
    var v_vec_buf: GPUMemoryBuffer
    var v_vec_buf_1: GPUMemoryBuffer
    var attn_out_buf: GPUMemoryBuffer
    var attn_out_buf_1: GPUMemoryBuffer
    var proj_attn_out_buf: GPUMemoryBuffer
    var h1_buf: GPUMemoryBuffer
    var h2_buf: GPUMemoryBuffer
    var cpu_sfence_fn: fn(Int32) -> Int32
    var cpu_clflush_fn: fn(Int) -> Int32
    var usleep_fn: fn(UInt32) -> Int32

    fn __init__(
        out self,
        kfd_fd: Int32,
        drm_fd: Int32,
        gpu_id: UInt32,
    ) raises:
        self.kfd_fd = kfd_fd
        self.drm_fd = drm_fd
        self.gpu_id = gpu_id
        self.libc = OwnedDLHandle("libc.so.6")
        self.cpu_fence_lib = OwnedDLHandle("/home/phil/git/llm-mojo/scratch/libcpu_fence.so")
        self.cpu_sfence_fn = self.cpu_fence_lib.get_function[fn(Int32) -> Int32]("cpu_sfence")
        self.cpu_clflush_fn = self.cpu_fence_lib.get_function[fn(Int) -> Int32]("cpu_clflush")
        self.usleep_fn = self.libc.get_function[fn(UInt32) -> Int32]("usleep")
        self.packet_idx = 0
        self.ring_buf = GPUMemoryBuffer()
        self.gart_buf = GPUMemoryBuffer()
        self.cwsr_buf = GPUMemoryBuffer()
        self.eop_buf = GPUMemoryBuffer()
        self.completion_buf = GPUMemoryBuffer()
        self.kernarg_buf = GPUMemoryBuffer()
        self.hsaco_buf = GPUMemoryBuffer()
        self.doorbell_reg_addr = 0
        self.weights_gpu_buf = GPUMemoryBuffer()
        self.x_buf = GPUMemoryBuffer()
        self.norm_x_buf = GPUMemoryBuffer()
        self.norm_x_buf_1 = GPUMemoryBuffer()
        self.final_norm_x_buf = GPUMemoryBuffer()
        self.q_buf = GPUMemoryBuffer()
        self.q_buf_1 = GPUMemoryBuffer()
        self.k_vec_buf = GPUMemoryBuffer()
        self.k_vec_buf_1 = GPUMemoryBuffer()
        self.v_vec_buf = GPUMemoryBuffer()
        self.v_vec_buf_1 = GPUMemoryBuffer()
        self.attn_out_buf = GPUMemoryBuffer()
        self.attn_out_buf_1 = GPUMemoryBuffer()
        self.proj_attn_out_buf = GPUMemoryBuffer()
        self.h1_buf = GPUMemoryBuffer()
        self.h2_buf = GPUMemoryBuffer()
        
        # Configure flags (aligned with verify_gpu_matmul.mojo)
        var gtt_uncached_flags: UInt32 = 0xF6000002 # KFD_IOC_ALLOC_MEM_FLAGS_COHERENT | UNCACHED | PUBLIC | GTT | WRITABLE | EXECUTABLE | NO_SUBSTITUTE
        var vram_private_flags: UInt32 = 0xD0000001 # KFD_IOC_ALLOC_MEM_FLAGS_VRAM | WRITABLE | EXECUTABLE | NO_SUBSTITUTE
        
        var ring_size = 64 * 1024
        self.ring_buf = self.alloc_gpu_buffer(ring_size, gtt_uncached_flags, cpu_access=True)
        self.gart_buf = self.alloc_gpu_buffer(4096, gtt_uncached_flags, cpu_access=True)
        self.cwsr_buf = self.alloc_gpu_buffer(2801664, vram_private_flags, cpu_access=False)
        self.eop_buf = self.alloc_gpu_buffer(4096, vram_private_flags, cpu_access=False)
        self.completion_buf = self.alloc_gpu_buffer(1024 * 64, gtt_uncached_flags, cpu_access=True)
        self.kernarg_buf = self.alloc_gpu_buffer(1024 * 256, gtt_uncached_flags, cpu_access=True)

        # Zero ring and gart descriptor
        var ring_ptr = UnsafePointer[UInt32, MutExternalOrigin](unsafe_from_address=self.ring_buf.cpu_addr)
        for i in range(ring_size // 4):
            ring_ptr[i] = 0

        var gart_u32_ptr = UnsafePointer[UInt32, MutExternalOrigin](unsafe_from_address=self.gart_buf.cpu_addr)
        for i in range(4096 // 4):
            gart_u32_ptr[i] = 0

        # Load CPU fence functions
        var cpu_sfence = self.cpu_fence_lib.get_function[fn(Int32) -> Int32]("cpu_sfence")
        var cpu_clflush = self.cpu_fence_lib.get_function[fn(Int) -> Int32]("cpu_clflush")

        # Structure a proper 64-byte aligned amd_signal_t block at self.gart_buf.cpu_addr + 512
        var sig_cpu_addr = self.gart_buf.cpu_addr + 512
        var sig_va_addr = self.gart_buf.va_addr + 512

        # Zero out the 64-byte signal area first
        var sig_u64_ptr = UnsafePointer[UInt64, MutExternalOrigin](unsafe_from_address=sig_cpu_addr)
        for i in range(8):
            sig_u64_ptr[i] = 0

        # Offset 0: kind (Int64) = -1 (AMD_SIGNAL_KIND_DOORBELL)
        UnsafePointer[Int64, MutExternalOrigin](unsafe_from_address=sig_cpu_addr + 0)[0] = -1
        # Offset 8: hardware_doorbell_ptr (UInt64) = 0 initially (mapped and filled after queue creation)
        UnsafePointer[UInt64, MutExternalOrigin](unsafe_from_address=sig_cpu_addr + 8)[0] = 0
        # Offset 48: queue_ptr (UInt64) = base_address (gart_buf.va_addr)
        UnsafePointer[UInt64, MutExternalOrigin](unsafe_from_address=sig_cpu_addr + 48)[0] = UInt64(self.gart_buf.va_addr)

        # Flush the signal descriptor cache line before queue creation
        _ = cpu_clflush(sig_cpu_addr)
        _ = cpu_sfence(0)

        # Configure amd_queue_t / hsa_queue_t
        var base = self.gart_buf.cpu_addr
        UnsafePointer[UInt32, MutExternalOrigin](unsafe_from_address=base + 0)[0] = 2       # queue_type = AQL
        UnsafePointer[UInt32, MutExternalOrigin](unsafe_from_address=base + 4)[0] = 1       # features (HSA_QUEUE_FEATURE_KERNEL_DISPATCH = 1)
        UnsafePointer[UInt64, MutExternalOrigin](unsafe_from_address=base + 8)[0] = UInt64(self.ring_buf.va_addr) # base_address
        UnsafePointer[UInt64, MutExternalOrigin](unsafe_from_address=base + 16)[0] = UInt64(sig_va_addr) # doorbell_signal mapped BEFORE queue creation!
        UnsafePointer[UInt32, MutExternalOrigin](unsafe_from_address=base + 24)[0] = 1024   # size (64KB / 64 bytes/packet = 1024)

        UnsafePointer[UInt32, MutExternalOrigin](unsafe_from_address=base + 72)[0] = 7
        UnsafePointer[UInt32, MutExternalOrigin](unsafe_from_address=base + 76)[0] = 39
        UnsafePointer[UInt32, MutExternalOrigin](unsafe_from_address=base + 136)[0] = 128   # read_dispatch_id_field_base_byte_offset
        UnsafePointer[UInt32, MutExternalOrigin](unsafe_from_address=base + 180)[0] = 10

        # Flush the entire queue descriptor cache lines
        _ = cpu_clflush(base)
        _ = cpu_clflush(base + 64)
        _ = cpu_clflush(base + 128)
        _ = cpu_clflush(base + 180)
        _ = cpu_sfence(0)

        # Create hardware queue
        var c_ioctl = self.libc.get_function[fn(Int32, Int, Int) -> Int32]("ioctl")
        var create_args_ptr = alloc[kfd_ioctl_create_queue_args](1)
        create_args_ptr[0] = kfd_ioctl_create_queue_args(
            ring_base_address=UInt64(self.ring_buf.va_addr),
            write_pointer_address=UInt64(self.gart_buf.va_addr + 56),
            read_pointer_address=UInt64(self.gart_buf.va_addr + 128),
            ring_size=UInt32(ring_size),
            gpu_id=self.gpu_id,
            queue_type=2, # AQL
            queue_percentage=100,
            queue_priority=7,
            eop_buffer_address=UInt64(self.eop_buf.va_addr),
            eop_buffer_size=4096,
            ctx_save_restore_address=UInt64(self.cwsr_buf.va_addr),
            ctx_save_restore_size=2789376,
            ctl_stack_size=4096,
            sdma_engine_id=0
        )
        var res = c_ioctl(self.kfd_fd, 0xC0604B02, create_args_ptr.__int__()) # AMDKFD_IOC_CREATE_QUEUE
        if res < 0:
            create_args_ptr.free()
            raise Error("Queue creation failed!")

        var doorbell_offset = create_args_ptr[0].doorbell_offset
        create_args_ptr.free()

        # Map doorbell page
        var doorbells_base = doorbell_offset & (~0x1fff)
        var doorbell_page_size = 0x2000
        var c_mmap = self.libc.get_function[fn(Int, Int, Int32, Int32, Int32, Int) -> Int]("mmap")
        var doorbell_mmap_addr = c_mmap(0, doorbell_page_size, 1 | 2, 0x01, self.kfd_fd, Int(doorbells_base)) # PROT_READ | PROT_WRITE, MAP_SHARED
        if doorbell_mmap_addr == -1:
            raise Error("Failed to mmap doorbell page!")
        self.doorbell_reg_addr = doorbell_mmap_addr + (Int(doorbell_offset) - Int(doorbells_base))

        # Update signal doorbell pointer
        UnsafePointer[UInt64, MutExternalOrigin](unsafe_from_address=sig_cpu_addr + 8)[0] = UInt64(self.doorbell_reg_addr)
        _ = cpu_clflush(sig_cpu_addr)
        _ = cpu_sfence(0)

        # Load Compiled GCN Shader into GPU unified memory
        self.hsaco_buf = self.alloc_gpu_buffer(16384, gtt_uncached_flags, cpu_access=True)
        var hsaco_u32_ptr = UnsafePointer[UInt32, MutExternalOrigin](unsafe_from_address=self.hsaco_buf.cpu_addr)
        for i in range(16384 // 4):
            hsaco_u32_ptr[i] = 0

        # Read file into temporary flat buffer
        var c_open = self.libc.get_function[fn(Int, Int32) -> Int32]("open")
        var c_read = self.libc.get_function[fn(Int32, Int, Int) -> Int]("read")
        var c_close = self.libc.get_function[fn(Int32) -> Int32]("close")

        var hsaco_file = "/home/phil/git/llm-mojo/scratch/q8_matmul.hsaco"
        var hsaco_file_bytes = hsaco_file.as_bytes()
        var hsaco_file_ptr = alloc[UInt8](len(hsaco_file_bytes) + 1)
        for i in range(len(hsaco_file_bytes)):
            hsaco_file_ptr[i] = hsaco_file_bytes[i]
        hsaco_file_ptr[len(hsaco_file_bytes)] = 0
        var hsaco_fd = c_open(hsaco_file_ptr.__int__(), 0) # O_RDONLY
        hsaco_file_ptr.free()
        if hsaco_fd < 0:
            raise Error("Failed to open q8_matmul.hsaco!")

        var temp_file_buf = alloc[UInt8](8192)
        var bytes_read = c_read(hsaco_fd, temp_file_buf.__int__(), 8192)
        _ = c_close(hsaco_fd)

        # Map ELF segments to virtual memory layout (do not copy flat):
        # Segment 1 (RO): offset 0x0, virt_addr 0x0, size 0xa80 (covering KD descriptor)
        var segment1_src = temp_file_buf
        var segment1_dst = UnsafePointer[UInt8, MutExternalOrigin](unsafe_from_address=self.hsaco_buf.cpu_addr + 0)
        for i in range(0xa80):
            segment1_dst[i] = segment1_src[i]

        # Segment 2 (RX): offset 0xb00, virt_addr 0x1b00, size 0x414 (full instructions payload)
        var segment2_src = temp_file_buf + 0xb00
        var segment2_dst = UnsafePointer[UInt8, MutExternalOrigin](unsafe_from_address=self.hsaco_buf.cpu_addr + 0x1b00)
        for i in range(0x414):
            segment2_dst[i] = segment2_src[i]

        # Segment 3 (RW): offset 0xf18, virt_addr 0x2f18, size 0x70
        var segment3_src = temp_file_buf + 0xf18
        var segment3_dst = UnsafePointer[UInt8, MutExternalOrigin](unsafe_from_address=self.hsaco_buf.cpu_addr + 0x2f18)
        for i in range(0x70):
            segment3_dst[i] = segment3_src[i]
        temp_file_buf.free()

    fn __init__(out self, *, deinit take: Self):
        self.kfd_fd = take.kfd_fd
        self.drm_fd = take.drm_fd
        self.gpu_id = take.gpu_id
        self.libc = take.libc^
        self.cpu_fence_lib = take.cpu_fence_lib^
        self.ring_buf = take.ring_buf^
        self.gart_buf = take.gart_buf^
        self.cwsr_buf = take.cwsr_buf^
        self.eop_buf = take.eop_buf^
        self.doorbell_reg_addr = take.doorbell_reg_addr
        self.hsaco_buf = take.hsaco_buf^
        self.kernarg_buf = take.kernarg_buf^
        self.completion_buf = take.completion_buf^
        self.packet_idx = take.packet_idx
        self.weights_gpu_buf = take.weights_gpu_buf^
        self.x_buf = take.x_buf^
        self.norm_x_buf = take.norm_x_buf^
        self.norm_x_buf_1 = take.norm_x_buf_1^
        self.final_norm_x_buf = take.final_norm_x_buf^
        self.q_buf = take.q_buf^
        self.q_buf_1 = take.q_buf_1^
        self.k_vec_buf = take.k_vec_buf^
        self.k_vec_buf_1 = take.k_vec_buf_1^
        self.v_vec_buf = take.v_vec_buf^
        self.v_vec_buf_1 = take.v_vec_buf_1^
        self.attn_out_buf = take.attn_out_buf^
        self.attn_out_buf_1 = take.attn_out_buf_1^
        self.proj_attn_out_buf = take.proj_attn_out_buf^
        self.h1_buf = take.h1_buf^
        self.h2_buf = take.h2_buf^
        self.cpu_sfence_fn = take.cpu_sfence_fn
        self.cpu_clflush_fn = take.cpu_clflush_fn
        self.usleep_fn = take.usleep_fn

    fn alloc_gpu_buffer(self, size: Int, flags: UInt32, cpu_access: Bool) raises -> GPUMemoryBuffer:
        var c_mmap = self.libc.get_function[fn(Int, Int, Int32, Int32, Int32, Int) -> Int]("mmap")
        var c_ioctl = self.libc.get_function[fn(Int32, Int, Int) -> Int32]("ioctl")
        var c_munmap = self.libc.get_function[fn(Int, Int) -> Int32]("munmap")
        
        var va_addr = c_mmap(0, size, 0, 0x02 | 0x20, -1, 0) # PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS
        if va_addr == -1 or va_addr == 0:
            raise Error("mmap failed")
            
        var alloc_args_ptr = alloc[kfd_ioctl_alloc_memory_of_gpu_args](1)
        alloc_args_ptr[0] = kfd_ioctl_alloc_memory_of_gpu_args(va_addr=UInt64(va_addr), size=UInt64(size), gpu_id=self.gpu_id, flags=flags)
        var r = c_ioctl(self.kfd_fd, 0xC0284B16, alloc_args_ptr.__int__()) # AMDKFD_IOC_ALLOC_MEMORY_OF_GPU
        if r < 0:
            alloc_args_ptr.free()
            _ = c_munmap(va_addr, size)
            raise Error("ALLOC_MEMORY failed")
        var handle = alloc_args_ptr[0].handle
        var mmap_offset = alloc_args_ptr[0].mmap_offset
        alloc_args_ptr.free()

        var cpu_addr = 0
        if cpu_access:
            cpu_addr = c_mmap(va_addr, size, 1 | 2, 0x01 | 0x10, self.drm_fd, Int(mmap_offset)) # PROT_READ | PROT_WRITE, MAP_SHARED | MAP_FIXED
            if cpu_addr == -1 or cpu_addr != va_addr:
                raise Error("CPU mmap failed")

        var device_ids_ptr = alloc[UInt32](1)
        device_ids_ptr[0] = self.gpu_id
        var map_args_ptr = alloc[kfd_ioctl_map_memory_to_gpu_args](1)
        map_args_ptr[0] = kfd_ioctl_map_memory_to_gpu_args(handle=handle, device_ids_array_ptr=UInt64(device_ids_ptr.__int__()), n_devices=1)
        r = c_ioctl(self.kfd_fd, 0xC0184B18, map_args_ptr.__int__()) # AMDKFD_IOC_MAP_MEMORY_TO_GPU
        device_ids_ptr.free()
        map_args_ptr.free()
        if r < 0:
            raise Error("MAP failed")

        print("  [GPU Alloc] size =", size, "va =", va_addr, "cpu =", cpu_addr, "handle =", handle)
        return GPUMemoryBuffer(va_addr, size, handle, cpu_addr)

    fn free_gpu_buffer(self, buf: GPUMemoryBuffer) raises:
        var c_ioctl = self.libc.get_function[fn(Int32, Int, Int) -> Int32]("ioctl")
        var c_munmap = self.libc.get_function[fn(Int, Int) -> Int32]("munmap")
        
        var unmap_device_ids_ptr = alloc[UInt32](1)
        unmap_device_ids_ptr[0] = self.gpu_id
        var unmap_args_ptr = alloc[kfd_ioctl_unmap_memory_from_gpu_args](1)
        unmap_args_ptr[0] = kfd_ioctl_unmap_memory_from_gpu_args(handle=buf.handle, device_ids_array_ptr=UInt64(unmap_device_ids_ptr.__int__()), n_devices=1)
        _ = c_ioctl(self.kfd_fd, 0xC0184B19, unmap_args_ptr.__int__()) # AMDKFD_IOC_UNMAP_MEMORY_FROM_GPU
        unmap_device_ids_ptr.free()
        unmap_args_ptr.free()
        
        if buf.cpu_addr != 0:
            _ = c_munmap(buf.cpu_addr, buf.size)
        else:
            _ = c_munmap(buf.va_addr, buf.size)

        var free_args_ptr = alloc[kfd_ioctl_free_memory_of_gpu_args](1)
        free_args_ptr[0] = kfd_ioctl_free_memory_of_gpu_args(handle=buf.handle)
        _ = c_ioctl(self.kfd_fd, 0x40084B17, free_args_ptr.__int__()) # AMDKFD_IOC_FREE_MEMORY_OF_GPU
        free_args_ptr.free()

    fn get_gpu_va(self, cpu_addr: Int) -> UInt64:
        if cpu_addr == 0:
            return 0
        if self.weights_gpu_buf.size > 0 and cpu_addr >= self.weights_gpu_buf.cpu_addr and cpu_addr < self.weights_gpu_buf.cpu_addr + self.weights_gpu_buf.size:
            return UInt64(self.weights_gpu_buf.va_addr + (cpu_addr - self.weights_gpu_buf.cpu_addr))
        if self.x_buf.size > 0 and cpu_addr >= self.x_buf.cpu_addr and cpu_addr < self.x_buf.cpu_addr + self.x_buf.size:
            return UInt64(self.x_buf.va_addr + (cpu_addr - self.x_buf.cpu_addr))
        if self.norm_x_buf.size > 0 and cpu_addr >= self.norm_x_buf.cpu_addr and cpu_addr < self.norm_x_buf.cpu_addr + self.norm_x_buf.size:
            return UInt64(self.norm_x_buf.va_addr + (cpu_addr - self.norm_x_buf.cpu_addr))
        if self.norm_x_buf_1.size > 0 and cpu_addr >= self.norm_x_buf_1.cpu_addr and cpu_addr < self.norm_x_buf_1.cpu_addr + self.norm_x_buf_1.size:
            return UInt64(self.norm_x_buf_1.va_addr + (cpu_addr - self.norm_x_buf_1.cpu_addr))
        if self.final_norm_x_buf.size > 0 and cpu_addr >= self.final_norm_x_buf.cpu_addr and cpu_addr < self.final_norm_x_buf.cpu_addr + self.final_norm_x_buf.size:
            return UInt64(self.final_norm_x_buf.va_addr + (cpu_addr - self.final_norm_x_buf.cpu_addr))
        if self.q_buf.size > 0 and cpu_addr >= self.q_buf.cpu_addr and cpu_addr < self.q_buf.cpu_addr + self.q_buf.size:
            return UInt64(self.q_buf.va_addr + (cpu_addr - self.q_buf.cpu_addr))
        if self.q_buf_1.size > 0 and cpu_addr >= self.q_buf_1.cpu_addr and cpu_addr < self.q_buf_1.cpu_addr + self.q_buf_1.size:
            return UInt64(self.q_buf_1.va_addr + (cpu_addr - self.q_buf_1.cpu_addr))
        if self.k_vec_buf.size > 0 and cpu_addr >= self.k_vec_buf.cpu_addr and cpu_addr < self.k_vec_buf.cpu_addr + self.k_vec_buf.size:
            return UInt64(self.k_vec_buf.va_addr + (cpu_addr - self.k_vec_buf.cpu_addr))
        if self.k_vec_buf_1.size > 0 and cpu_addr >= self.k_vec_buf_1.cpu_addr and cpu_addr < self.k_vec_buf_1.cpu_addr + self.k_vec_buf_1.size:
            return UInt64(self.k_vec_buf_1.va_addr + (cpu_addr - self.k_vec_buf_1.cpu_addr))
        if self.v_vec_buf.size > 0 and cpu_addr >= self.v_vec_buf.cpu_addr and cpu_addr < self.v_vec_buf.cpu_addr + self.v_vec_buf.size:
            return UInt64(self.v_vec_buf.va_addr + (cpu_addr - self.v_vec_buf.cpu_addr))
        if self.v_vec_buf_1.size > 0 and cpu_addr >= self.v_vec_buf_1.cpu_addr and cpu_addr < self.v_vec_buf_1.cpu_addr + self.v_vec_buf_1.size:
            return UInt64(self.v_vec_buf_1.va_addr + (cpu_addr - self.v_vec_buf_1.cpu_addr))
        if self.attn_out_buf.size > 0 and cpu_addr >= self.attn_out_buf.cpu_addr and cpu_addr < self.attn_out_buf.cpu_addr + self.attn_out_buf.size:
            return UInt64(self.attn_out_buf.va_addr + (cpu_addr - self.attn_out_buf.cpu_addr))
        if self.attn_out_buf_1.size > 0 and cpu_addr >= self.attn_out_buf_1.cpu_addr and cpu_addr < self.attn_out_buf_1.cpu_addr + self.attn_out_buf_1.size:
            return UInt64(self.attn_out_buf_1.va_addr + (cpu_addr - self.attn_out_buf_1.cpu_addr))
        if self.proj_attn_out_buf.size > 0 and cpu_addr >= self.proj_attn_out_buf.cpu_addr and cpu_addr < self.proj_attn_out_buf.cpu_addr + self.proj_attn_out_buf.size:
            return UInt64(self.proj_attn_out_buf.va_addr + (cpu_addr - self.proj_attn_out_buf.cpu_addr))
        if self.h1_buf.size > 0 and cpu_addr >= self.h1_buf.cpu_addr and cpu_addr < self.h1_buf.cpu_addr + self.h1_buf.size:
            return UInt64(self.h1_buf.va_addr + (cpu_addr - self.h1_buf.cpu_addr))
        if self.h2_buf.size > 0 and cpu_addr >= self.h2_buf.cpu_addr and cpu_addr < self.h2_buf.cpu_addr + self.h2_buf.size:
            return UInt64(self.h2_buf.va_addr + (cpu_addr - self.h2_buf.cpu_addr))
        print("WARNING: get_gpu_va did not match address", cpu_addr, "falling back to raw va_addr!")
        return UInt64(cpu_addr)

    fn enqueue_matmul(
        mut self,
        W_packed_addr: Int,
        scales_addr: Int,
        x_addr: Int,
        out_ptr_addr: Int,
        c_in: Int,
        c_out: Int,
    ) raises -> Int:
        var cpu_sfence = self.cpu_sfence_fn
        var cpu_clflush = self.cpu_clflush_fn

        # Flush CPU cache lines for input vector x to ensure GPU reads current activations
        var input_size_bytes = c_in * 4
        var base_in = (x_addr // 64) * 64
        var end_in = x_addr + input_size_bytes
        while base_in < end_in:
            _ = cpu_clflush(base_in)
            base_in += 64
        _ = cpu_sfence(0)

        var slot = Int(self.packet_idx % 1024)

        # 1. Reset completion flag to 0 and atomic counter to 0 for this slot
        var completion_slot_cpu_addr = self.completion_buf.cpu_addr + slot * 64
        var completion_slot_va_addr = self.completion_buf.va_addr + slot * 64
        var completion_ptr = UnsafePointer[Int32, MutExternalOrigin](unsafe_from_address=completion_slot_cpu_addr)
        completion_ptr[0] = 0
        completion_ptr[1] = 0 # reset atomic counter to 0!
        _ = cpu_sfence(0)
        _ = cpu_clflush(completion_slot_cpu_addr)

        # Translate CPU pointers to GPU Virtual Addresses using range scanner
        var gpu_W_packed_va = self.get_gpu_va(W_packed_addr)
        var gpu_scales_va = self.get_gpu_va(scales_addr)
        var gpu_x_va = self.get_gpu_va(x_addr)
        var gpu_out_ptr_va = self.get_gpu_va(out_ptr_addr)

        # 2. Pack kernel arguments into this slot's kernarg memory
        var kernarg_slot_cpu_addr = self.kernarg_buf.cpu_addr + slot * 256
        var kernarg_slot_va_addr = self.kernarg_buf.va_addr + slot * 256

        var kernarg_p = UnsafePointer[UInt64, MutExternalOrigin](unsafe_from_address=kernarg_slot_cpu_addr)
        kernarg_p[0] = gpu_W_packed_va
        kernarg_p[1] = gpu_scales_va
        kernarg_p[2] = gpu_x_va
        kernarg_p[3] = gpu_out_ptr_va
        kernarg_p[4] = UInt64(completion_slot_va_addr)

        var num_workgroups = 32
        var workgroup_size = 256
        var grid_stride = num_workgroups * workgroup_size

        var kernarg_u32_p = UnsafePointer[UInt32, MutExternalOrigin](unsafe_from_address=kernarg_slot_cpu_addr + 40)
        kernarg_u32_p[0] = UInt32(c_in)
        kernarg_u32_p[1] = UInt32(c_out)
        kernarg_u32_p[2] = UInt32(grid_stride)
        kernarg_u32_p[3] = UInt32(num_workgroups)
        
        _ = cpu_sfence(0)
        _ = cpu_clflush(kernarg_slot_cpu_addr)

        # 3. Format HSA AQL Dispatch Packet (64 bytes)
        var packet = hsa_kernel_dispatch_packet_t(
            header=5378,              # Type=2 (Kernel Dispatch), Barrier=1, Acquire=2 (System), Release=2 (System)
            setup=1,                 # 1D dimensions
            workgroup_size_x=256,    # 256 threads in workgroup
            workgroup_size_y=1,
            workgroup_size_z=1,
            grid_size_x=UInt32(grid_stride), # Grid size = num_workgroups * 256
            grid_size_y=1,
            grid_size_z=1,
            private_segment_size=0,
            group_segment_size=0,
            kernel_object=UInt64(self.hsaco_buf.va_addr + 0xa40), # KD address!
            kernarg_address=UInt64(kernarg_slot_va_addr),
            completion_signal=0
        )

        # 4. Write dispatch packet to circular ring buffer
        var ring_packet_ptr = UnsafePointer[hsa_kernel_dispatch_packet_t, MutExternalOrigin](unsafe_from_address=self.ring_buf.cpu_addr)
        ring_packet_ptr[slot] = packet^

        # Update write pointer in GART memory
        var wptr_ptr = UnsafePointer[UInt64, MutExternalOrigin](unsafe_from_address=self.gart_buf.cpu_addr + 56)
        self.packet_idx += 1
        wptr_ptr[0] = self.packet_idx

        # Memory fences & Flush cache lines for written data
        _ = cpu_sfence(0)
        _ = cpu_clflush(self.ring_buf.cpu_addr + slot * 64)
        _ = cpu_clflush(self.gart_buf.cpu_addr + 56)
        _ = cpu_sfence(0) # Ensure GART writes settle before ringing doorbell

        # Ring doorbell! Write 0-based packet_idx value to mapped doorbell register
        var doorbell_ptr = UnsafePointer[UInt64, MutExternalOrigin](unsafe_from_address=self.doorbell_reg_addr)
        doorbell_ptr[0] = self.packet_idx - 1
        _ = cpu_sfence(0) # Flush doorbell write through PCIe WC buffers

        return slot

    fn wait_matmul(
        self,
        slot: Int,
        out_ptr_addr: Int,
        c_out: Int,
    ) raises:
        var cpu_sfence = self.cpu_sfence_fn
        var cpu_clflush = self.cpu_clflush_fn
        var completion_slot_cpu_addr = self.completion_buf.cpu_addr + slot * 64
        var completion_ptr = UnsafePointer[Int32, MutExternalOrigin](unsafe_from_address=completion_slot_cpu_addr)

        # Wait for GPU completion flag with adaptive back-off
        var completed = False
        var attempts = 0
        var c_usleep = self.usleep_fn
        while not completed:
            if completion_ptr[0] == 1:
                completed = True
                break
            attempts += 1
            if attempts > 500000: # Wait longer (approx 2-3ms) before sleeping to avoid OS scheduling latency
                _ = cpu_clflush(completion_slot_cpu_addr)
                _ = cpu_sfence(0)
                _ = c_usleep(5) # 5 microseconds sleep to prevent memory controller / bus starvation
            elif attempts % 5000 == 0:
                _ = cpu_clflush(completion_slot_cpu_addr)
                _ = cpu_sfence(0)
            if attempts > 10000000: # Failsafe timeout
                var current_wptr = UnsafePointer[UInt64, MutExternalOrigin](unsafe_from_address=self.gart_buf.cpu_addr + 56)[0]
                var current_rptr = UnsafePointer[UInt64, MutExternalOrigin](unsafe_from_address=self.gart_buf.cpu_addr + 128)[0]
                print("GPU Execution Timeout details:")
                print("  Attempts reached:", attempts)
                print("  GART wptr:", current_wptr, "| GART rptr:", current_rptr)
                print("  Completion flag in memory:", completion_ptr[0])
                print("  Atomic counter in memory:", completion_ptr[1])
                raise Error("GPU execution timed out in wait_matmul!")

        # Flush/invalidate CPU cache lines for output vector out to ensure CPU reads the GPU's results
        var output_size_bytes = c_out * 4
        var base_out = (out_ptr_addr // 64) * 64
        var end_out = out_ptr_addr + output_size_bytes
        while base_out < end_out:
            _ = cpu_clflush(base_out)
            base_out += 64
        _ = cpu_sfence(0)

    fn launch_matmul(
        mut self,
        W_packed_addr: Int,
        scales_addr: Int,
        x_addr: Int,
        out_ptr_addr: Int,
        c_in: Int,
        c_out: Int,
    ) raises:
        var slot = self.enqueue_matmul(W_packed_addr, scales_addr, x_addr, out_ptr_addr, c_in, c_out)
        self.wait_matmul(slot, out_ptr_addr, c_out)

# -----------------------------------------------------------------------------
# 1. Model Configurations & Memory Offsets
# -----------------------------------------------------------------------------

# Target Model Configuration (Q8_0 layout)
struct TargetConfig[
    embed_dim: Int,
    head_dim: Int,
    num_heads: Int,
    num_kv_heads: Int,
    intermediate_dim: Int,
    num_layers: Int,
    vocab_size: Int,
    max_seq_len: Int,
]:
    @always_inline
    def is_global_layer(self, layer_idx: Int) -> Bool:
        return layer_idx % 6 == 5

    @always_inline
    def get_w2_offset_in_layer(self, layer_idx: Int) -> Int:
        var is_global = self.is_global_layer(layer_idx)
        var layer_heads = 16
        var layer_head_dim = 512 if is_global else 256
        var layer_kv_heads = 1 if is_global else 8
        
        var wq_size = layer_heads * layer_head_dim * Self.embed_dim
        var wk_size = layer_kv_heads * layer_head_dim * Self.embed_dim
        var wv_size = 0 if is_global else (layer_kv_heads * layer_head_dim * Self.embed_dim)
        var wo_size = Self.embed_dim * layer_heads * layer_head_dim
        var w1_size = Self.intermediate_dim * Self.embed_dim
        var w3_size = Self.intermediate_dim * Self.embed_dim
        
        var packed_weights_size = wq_size + wk_size + wv_size + wo_size + w1_size + w3_size
        var scales_count = packed_weights_size // 32
        var scales_size_bytes = scales_count * 4
        
        var input_layernorm_size = Self.embed_dim
        var post_attention_layernorm_size = Self.embed_dim
        var pre_feedforward_layernorm_size = Self.embed_dim
        var post_feedforward_layernorm_size = Self.embed_dim
        var q_norm_size = layer_head_dim
        var k_norm_size = layer_head_dim
        var layer_scalar_size = 1
        
        var unchanged_size_bytes = (
            input_layernorm_size + post_attention_layernorm_size +
            pre_feedforward_layernorm_size + post_feedforward_layernorm_size +
            q_norm_size + k_norm_size + layer_scalar_size
        ) * 4
        
        var raw_bytes_size = packed_weights_size + scales_size_bytes + unchanged_size_bytes
        return ((raw_bytes_size + 4095) // 4096) * 4096

    @always_inline
    def get_layer_aligned_size(self, layer_idx: Int) -> Int:
        var base_offset = self.get_w2_offset_in_layer(layer_idx)
        var w2_blocks = Self.intermediate_dim // 32
        var w2_size = w2_blocks * 139264
        return base_offset + w2_size

    def get_layer_offset(self, layer_idx: Int) -> Int:
        var offset = 0
        for i in range(layer_idx):
            offset += self.get_layer_aligned_size(i)
        return offset

    @always_inline
    def get_layer_weights(self, layer_idx: Int, base_ptr: UnsafePointer[Int8, _]) -> Q8LayerWeights:
        var is_global = self.is_global_layer(layer_idx)
        var layer_heads = 16
        var layer_head_dim = 512 if is_global else 256
        var layer_kv_heads = 1 if is_global else 8
        
        var wq_size = layer_heads * layer_head_dim * Self.embed_dim
        var wk_size = layer_kv_heads * layer_head_dim * Self.embed_dim
        var wv_size = 0 if is_global else (layer_kv_heads * layer_head_dim * Self.embed_dim)
        var wo_size = Self.embed_dim * layer_heads * layer_head_dim
        var w1_size = Self.intermediate_dim * Self.embed_dim
        var w3_size = Self.intermediate_dim * Self.embed_dim
        
        var wq_size_bytes = wq_size
        var wk_size_bytes = wk_size
        var wv_size_bytes = wv_size
        var wo_size_bytes = wo_size
        var w1_size_bytes = w1_size
        var w3_size_bytes = w3_size
        
        var c_blocks_q = wq_size // 32
        var c_blocks_k = wk_size // 32
        var c_blocks_v = wv_size // 32
        var c_blocks_o = wo_size // 32
        var c_blocks_w1 = w1_size // 32
        var c_blocks_w3 = w3_size // 32

        var wq_packed = base_ptr
        var wk_packed = wq_packed + wq_size_bytes
        var wv_packed = wk_packed + wk_size_bytes
        var wo_packed = wk_packed + wk_size_bytes
        if not is_global:
            wv_packed = wk_packed + wk_size_bytes
            wo_packed = wv_packed + wv_size_bytes
            
        var w1_packed = wo_packed + wo_size_bytes
        var w3_packed = w1_packed + w1_size_bytes
        
        var scales_base = (w3_packed + w3_size_bytes).bitcast[Float32]()
        
        var wq_scales = scales_base
        var wk_scales = wq_scales + c_blocks_q
        var wv_scales = wk_scales + c_blocks_k
        var wo_scales = wk_scales + c_blocks_k
        if not is_global:
            wv_scales = wk_scales + c_blocks_k
            wo_scales = wv_scales + c_blocks_v
            
        var w1_scales = wo_scales + c_blocks_o
        var w3_scales = w1_scales + c_blocks_w1
        
        var unchanged_base = w3_scales + c_blocks_w3
        
        var input_layernorm = unchanged_base
        var post_attention_layernorm = input_layernorm + Self.embed_dim
        var pre_feedforward_layernorm = post_attention_layernorm + Self.embed_dim
        var post_feedforward_layernorm = pre_feedforward_layernorm + Self.embed_dim
        var q_norm = post_feedforward_layernorm + Self.embed_dim
        var k_norm = q_norm + layer_head_dim
        var layer_scalar = k_norm + layer_head_dim
        
        var w2_packed = base_ptr + self.get_w2_offset_in_layer(layer_idx)
        var w2_scales = (w2_packed + 122880).bitcast[Float32]()
        
        return Q8LayerWeights(
            UnsafePointer[Int8, MutExternalOrigin](unsafe_from_address=wq_packed.__int__()),
            UnsafePointer[Int8, MutExternalOrigin](unsafe_from_address=wk_packed.__int__()),
            UnsafePointer[Int8, MutExternalOrigin](unsafe_from_address=wv_packed.__int__()),
            UnsafePointer[Int8, MutExternalOrigin](unsafe_from_address=wo_packed.__int__()),
            UnsafePointer[Int8, MutExternalOrigin](unsafe_from_address=w1_packed.__int__()),
            UnsafePointer[Int8, MutExternalOrigin](unsafe_from_address=w2_packed.__int__()),
            UnsafePointer[Int8, MutExternalOrigin](unsafe_from_address=w3_packed.__int__()),
            UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=wq_scales.__int__()),
            UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=wk_scales.__int__()),
            UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=wv_scales.__int__()),
            UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=wo_scales.__int__()),
            UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=w1_scales.__int__()),
            UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=w2_scales.__int__()),
            UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=w3_scales.__int__()),
            UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=input_layernorm.__int__()),
            UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=post_attention_layernorm.__int__()),
            UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=pre_feedforward_layernorm.__int__()),
            UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=post_feedforward_layernorm.__int__()),
            UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=q_norm.__int__()),
            UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=k_norm.__int__()),
            UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=layer_scalar.__int__()),
        )

    def __init__(out self):
        pass


# Draft Model Configuration (4-bit Q4_0, fully RAM-resident)
struct DraftConfig[
    embed_dim: Int,
    head_dim: Int,
    num_heads: Int,
    num_kv_heads: Int,
    intermediate_dim: Int,
    num_layers: Int,
    vocab_size: Int,
    max_seq_len: Int,
]:
    @always_inline
    def is_global_layer(self, layer_idx: Int) -> Bool:
        return layer_idx % 6 == 5

    @always_inline
    def get_layer_weights(self, layer_idx: Int, base_ptr: UnsafePointer[UInt8, _]) -> Q4LayerWeights:
        var is_global = self.is_global_layer(layer_idx)
        var layer_heads = 16
        var layer_head_dim = 512 if is_global else 256
        var layer_kv_heads = 1 if is_global else 8
        
        var wq_size = layer_heads * layer_head_dim * Self.embed_dim
        var wk_size = layer_kv_heads * layer_head_dim * Self.embed_dim
        var wv_size = 0 if is_global else (layer_kv_heads * layer_head_dim * Self.embed_dim)
        var wo_size = Self.embed_dim * layer_heads * layer_head_dim
        var w1_size = Self.intermediate_dim * Self.embed_dim
        var w2_size = Self.embed_dim * Self.intermediate_dim
        var w3_size = Self.intermediate_dim * Self.embed_dim
        
        var wq_size_bytes = wq_size // 2
        var wk_size_bytes = wk_size // 2
        var wv_size_bytes = wv_size // 2
        var wo_size_bytes = wo_size // 2
        var w1_size_bytes = w1_size // 2
        var w2_size_bytes = w2_size // 2
        var w3_size_bytes = w3_size // 2
        
        var c_blocks_q = wq_size // 32
        var c_blocks_k = wk_size // 32
        var c_blocks_v = wv_size // 32
        var c_blocks_o = wo_size // 32
        var c_blocks_w1 = w1_size // 32
        var c_blocks_w2 = w2_size // 32
        var c_blocks_w3 = w3_size // 32

        # Pointers to packed weights (UInt8)
        var wq_packed = base_ptr
        var wk_packed = wq_packed + wq_size_bytes
        var wv_packed = wk_packed + wk_size_bytes # fallback/initialized
        var wo_packed = wk_packed + wk_size_bytes
        if not is_global:
            wv_packed = wk_packed + wk_size_bytes
            wo_packed = wv_packed + wv_size_bytes
            
        var w1_packed = wo_packed + wo_size_bytes
        var w2_packed = w1_packed + w1_size_bytes
        var w3_packed = w2_packed + w2_size_bytes
        
        # Scales base address: immediately after the end of w3_packed
        var scales_base = (w3_packed + w3_size_bytes).bitcast[Float32]()
        
        var wq_scales = scales_base
        var wk_scales = wq_scales + c_blocks_q
        var wv_scales = wk_scales + c_blocks_k # fallback/initialized
        var wo_scales = wk_scales + c_blocks_k
        if not is_global:
            wv_scales = wk_scales + c_blocks_k
            wo_scales = wv_scales + c_blocks_v
            
        var w1_scales = wo_scales + c_blocks_o
        var w2_scales = w1_scales + c_blocks_w1
        var w3_scales = w2_scales + c_blocks_w2
        
        # Unchanged Float32 parameters base address: immediately after the end of w3_scales
        var unchanged_base = w3_scales + c_blocks_w3
        
        var input_layernorm = unchanged_base
        var post_attention_layernorm = input_layernorm + Self.embed_dim
        var pre_feedforward_layernorm = post_attention_layernorm + Self.embed_dim
        var post_feedforward_layernorm = pre_feedforward_layernorm + Self.embed_dim
        var q_norm = post_feedforward_layernorm + Self.embed_dim
        var k_norm = q_norm + layer_head_dim
        var layer_scalar = k_norm + layer_head_dim
        
        return Q4LayerWeights(
            UnsafePointer[UInt8, MutExternalOrigin](unsafe_from_address=wq_packed.__int__()),
            UnsafePointer[UInt8, MutExternalOrigin](unsafe_from_address=wk_packed.__int__()),
            UnsafePointer[UInt8, MutExternalOrigin](unsafe_from_address=wv_packed.__int__()),
            UnsafePointer[UInt8, MutExternalOrigin](unsafe_from_address=wo_packed.__int__()),
            UnsafePointer[UInt8, MutExternalOrigin](unsafe_from_address=w1_packed.__int__()),
            UnsafePointer[UInt8, MutExternalOrigin](unsafe_from_address=w2_packed.__int__()),
            UnsafePointer[UInt8, MutExternalOrigin](unsafe_from_address=w3_packed.__int__()),
            UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=wq_scales.__int__()),
            UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=wk_scales.__int__()),
            UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=wv_scales.__int__()),
            UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=wo_scales.__int__()),
            UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=w1_scales.__int__()),
            UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=w2_scales.__int__()),
            UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=w3_scales.__int__()),
            UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=input_layernorm.__int__()),
            UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=post_attention_layernorm.__int__()),
            UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=pre_feedforward_layernorm.__int__()),
            UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=post_feedforward_layernorm.__int__()),
            UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=q_norm.__int__()),
            UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=k_norm.__int__()),
            UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=layer_scalar.__int__()),
        )

    @always_inline
    def get_layer_aligned_size(self, layer_idx: Int) -> Int:
        var is_global = self.is_global_layer(layer_idx)
        var layer_heads = 16
        var layer_head_dim = 512 if is_global else 256
        var layer_kv_heads = 1 if is_global else 8
        
        var wq_size = layer_heads * layer_head_dim * Self.embed_dim
        var wk_size = layer_kv_heads * layer_head_dim * Self.embed_dim
        var wv_size = 0 if is_global else (layer_kv_heads * layer_head_dim * Self.embed_dim)
        var wo_size = Self.embed_dim * layer_heads * layer_head_dim
        var w1_size = Self.intermediate_dim * Self.embed_dim
        var w2_size = Self.embed_dim * Self.intermediate_dim
        var w3_size = Self.intermediate_dim * Self.embed_dim
        
        # Packed Q4 weights: 2 elements per byte
        var packed_weights_size = (wq_size + wk_size + wv_size + wo_size + w1_size + w2_size + w3_size) // 2
        
        # Scale factors (1 per block of 32 elements)
        var scales_count = (wq_size + wk_size + wv_size + wo_size + w1_size + w2_size + w3_size) // 32
        var scales_size_bytes = scales_count * 4
        
        # Float32 unchanged weights
        var input_layernorm_size = Self.embed_dim
        var post_attention_layernorm_size = Self.embed_dim
        var pre_feedforward_layernorm_size = Self.embed_dim
        var post_feedforward_layernorm_size = Self.embed_dim
        var q_norm_size = layer_head_dim
        var k_norm_size = layer_head_dim
        var layer_scalar_size = 1
        
        var unchanged_size_bytes = (
            input_layernorm_size + post_attention_layernorm_size +
            pre_feedforward_layernorm_size + post_feedforward_layernorm_size +
            q_norm_size + k_norm_size + layer_scalar_size
        ) * 4
        
        var raw_bytes_size = packed_weights_size + scales_size_bytes + unchanged_size_bytes
        return ((raw_bytes_size + 4095) // 4096) * 4096

    def get_layer_offset(self, layer_idx: Int) -> Int:
        var offset = 0
        for i in range(layer_idx):
            offset += self.get_layer_aligned_size(i)
        return offset

    def __init__(out self):
        pass


# -----------------------------------------------------------------------------
# 2. Structural Layer Weights Representations
# -----------------------------------------------------------------------------
struct Q8LayerWeights:
    var wq_packed: UnsafePointer[Int8, MutExternalOrigin]
    var wk_packed: UnsafePointer[Int8, MutExternalOrigin]
    var wv_packed: UnsafePointer[Int8, MutExternalOrigin]
    var wo_packed: UnsafePointer[Int8, MutExternalOrigin]
    var w1_packed: UnsafePointer[Int8, MutExternalOrigin]
    var w2_packed: UnsafePointer[Int8, MutExternalOrigin]
    var w3_packed: UnsafePointer[Int8, MutExternalOrigin]

    var wq_scales: UnsafePointer[Float32, MutExternalOrigin]
    var wk_scales: UnsafePointer[Float32, MutExternalOrigin]
    var wv_scales: UnsafePointer[Float32, MutExternalOrigin]
    var wo_scales: UnsafePointer[Float32, MutExternalOrigin]
    var w1_scales: UnsafePointer[Float32, MutExternalOrigin]
    var w2_scales: UnsafePointer[Float32, MutExternalOrigin]
    var w3_scales: UnsafePointer[Float32, MutExternalOrigin]

    var input_layernorm_weight: UnsafePointer[Float32, MutExternalOrigin]
    var post_attention_layernorm_weight: UnsafePointer[Float32, MutExternalOrigin]
    var pre_feedforward_layernorm_weight: UnsafePointer[Float32, MutExternalOrigin]
    var post_feedforward_layernorm_weight: UnsafePointer[Float32, MutExternalOrigin]
    var q_norm_weight: UnsafePointer[Float32, MutExternalOrigin]
    var k_norm_weight: UnsafePointer[Float32, MutExternalOrigin]
    var layer_scalar: UnsafePointer[Float32, MutExternalOrigin]

    def __init__(
        out self,
        wq_packed: UnsafePointer[Int8, MutExternalOrigin],
        wk_packed: UnsafePointer[Int8, MutExternalOrigin],
        wv_packed: UnsafePointer[Int8, MutExternalOrigin],
        wo_packed: UnsafePointer[Int8, MutExternalOrigin],
        w1_packed: UnsafePointer[Int8, MutExternalOrigin],
        w2_packed: UnsafePointer[Int8, MutExternalOrigin],
        w3_packed: UnsafePointer[Int8, MutExternalOrigin],
        wq_scales: UnsafePointer[Float32, MutExternalOrigin],
        wk_scales: UnsafePointer[Float32, MutExternalOrigin],
        wv_scales: UnsafePointer[Float32, MutExternalOrigin],
        wo_scales: UnsafePointer[Float32, MutExternalOrigin],
        w1_scales: UnsafePointer[Float32, MutExternalOrigin],
        w2_scales: UnsafePointer[Float32, MutExternalOrigin],
        w3_scales: UnsafePointer[Float32, MutExternalOrigin],
        input_layernorm: UnsafePointer[Float32, MutExternalOrigin],
        post_attention_layernorm: UnsafePointer[Float32, MutExternalOrigin],
        pre_feedforward_layernorm: UnsafePointer[Float32, MutExternalOrigin],
        post_feedforward_layernorm: UnsafePointer[Float32, MutExternalOrigin],
        q_norm: UnsafePointer[Float32, MutExternalOrigin],
        k_norm: UnsafePointer[Float32, MutExternalOrigin],
        layer_scalar: UnsafePointer[Float32, MutExternalOrigin],
    ):
        self.wq_packed = wq_packed
        self.wk_packed = wk_packed
        self.wv_packed = wv_packed
        self.wo_packed = wo_packed
        self.w1_packed = w1_packed
        self.w2_packed = w2_packed
        self.w3_packed = w3_packed
        self.wq_scales = wq_scales
        self.wk_scales = wk_scales
        self.wv_scales = wv_scales
        self.wo_scales = wo_scales
        self.w1_scales = w1_scales
        self.w2_scales = w2_scales
        self.w3_scales = w3_scales
        self.input_layernorm_weight = input_layernorm
        self.post_attention_layernorm_weight = post_attention_layernorm
        self.pre_feedforward_layernorm_weight = pre_feedforward_layernorm
        self.post_feedforward_layernorm_weight = post_feedforward_layernorm
        self.q_norm_weight = q_norm
        self.k_norm_weight = k_norm
        self.layer_scalar = layer_scalar


struct Q4LayerWeights:
    var wq_packed: UnsafePointer[UInt8, MutExternalOrigin]
    var wk_packed: UnsafePointer[UInt8, MutExternalOrigin]
    var wv_packed: UnsafePointer[UInt8, MutExternalOrigin]
    var wo_packed: UnsafePointer[UInt8, MutExternalOrigin]
    var w1_packed: UnsafePointer[UInt8, MutExternalOrigin]
    var w2_packed: UnsafePointer[UInt8, MutExternalOrigin]
    var w3_packed: UnsafePointer[UInt8, MutExternalOrigin]

    var wq_scales: UnsafePointer[Float32, MutExternalOrigin]
    var wk_scales: UnsafePointer[Float32, MutExternalOrigin]
    var wv_scales: UnsafePointer[Float32, MutExternalOrigin]
    var wo_scales: UnsafePointer[Float32, MutExternalOrigin]
    var w1_scales: UnsafePointer[Float32, MutExternalOrigin]
    var w2_scales: UnsafePointer[Float32, MutExternalOrigin]
    var w3_scales: UnsafePointer[Float32, MutExternalOrigin]

    var input_layernorm_weight: UnsafePointer[Float32, MutExternalOrigin]
    var post_attention_layernorm_weight: UnsafePointer[Float32, MutExternalOrigin]
    var pre_feedforward_layernorm_weight: UnsafePointer[Float32, MutExternalOrigin]
    var post_feedforward_layernorm_weight: UnsafePointer[Float32, MutExternalOrigin]
    var q_norm_weight: UnsafePointer[Float32, MutExternalOrigin]
    var k_norm_weight: UnsafePointer[Float32, MutExternalOrigin]
    var layer_scalar: UnsafePointer[Float32, MutExternalOrigin]

    def __init__(
        out self,
        wq_packed: UnsafePointer[UInt8, MutExternalOrigin],
        wk_packed: UnsafePointer[UInt8, MutExternalOrigin],
        wv_packed: UnsafePointer[UInt8, MutExternalOrigin],
        wo_packed: UnsafePointer[UInt8, MutExternalOrigin],
        w1_packed: UnsafePointer[UInt8, MutExternalOrigin],
        w2_packed: UnsafePointer[UInt8, MutExternalOrigin],
        w3_packed: UnsafePointer[UInt8, MutExternalOrigin],
        wq_scales: UnsafePointer[Float32, MutExternalOrigin],
        wk_scales: UnsafePointer[Float32, MutExternalOrigin],
        wv_scales: UnsafePointer[Float32, MutExternalOrigin],
        wo_scales: UnsafePointer[Float32, MutExternalOrigin],
        w1_scales: UnsafePointer[Float32, MutExternalOrigin],
        w2_scales: UnsafePointer[Float32, MutExternalOrigin],
        w3_scales: UnsafePointer[Float32, MutExternalOrigin],
        input_layernorm: UnsafePointer[Float32, MutExternalOrigin],
        post_attention_layernorm: UnsafePointer[Float32, MutExternalOrigin],
        pre_feedforward_layernorm: UnsafePointer[Float32, MutExternalOrigin],
        post_feedforward_layernorm: UnsafePointer[Float32, MutExternalOrigin],
        q_norm: UnsafePointer[Float32, MutExternalOrigin],
        k_norm: UnsafePointer[Float32, MutExternalOrigin],
        layer_scalar: UnsafePointer[Float32, MutExternalOrigin],
    ):
        self.wq_packed = wq_packed
        self.wk_packed = wk_packed
        self.wv_packed = wv_packed
        self.wo_packed = wo_packed
        self.w1_packed = w1_packed
        self.w2_packed = w2_packed
        self.w3_packed = w3_packed
        self.wq_scales = wq_scales
        self.wk_scales = wk_scales
        self.wv_scales = wv_scales
        self.wo_scales = wo_scales
        self.w1_scales = w1_scales
        self.w2_scales = w2_scales
        self.w3_scales = w3_scales
        self.input_layernorm_weight = input_layernorm
        self.post_attention_layernorm_weight = post_attention_layernorm
        self.pre_feedforward_layernorm_weight = pre_feedforward_layernorm
        self.post_feedforward_layernorm_weight = post_feedforward_layernorm
        self.q_norm_weight = q_norm
        self.k_norm_weight = k_norm
        self.layer_scalar = layer_scalar


# -----------------------------------------------------------------------------
# 3. High-Performance Vectorized Math Kernels
# -----------------------------------------------------------------------------
@always_inline
def rms_norm(x: UnsafePointer[Float32, _], w: UnsafePointer[Float32, _], out_ptr: UnsafePointer[Float32, MutExternalOrigin], size: Int):
    var acc = SIMD[DType.float32, 8](0.0)
    for i in range(0, size, 8):
        var v = x.load[8](i)
        acc = acc + v * v
    var sum_sq = acc.reduce_add()
    var inv_rms = 1.0 / sqrt(sum_sq / Float32(size) + 1e-6)
    for i in range(0, size, 8):
        var v = x.load[8](i)
        var weight = w.load[8](i)
        out_ptr.store[8](i, v * inv_rms * weight)

@always_inline
def rms_norm_noscale(x: UnsafePointer[Float32, _], out_ptr: UnsafePointer[Float32, MutExternalOrigin], size: Int):
    var acc = SIMD[DType.float32, 8](0.0)
    for i in range(0, size, 8):
        var v = x.load[8](i)
        acc = acc + v * v
    var sum_sq = acc.reduce_add()
    var inv_rms = 1.0 / sqrt(sum_sq / Float32(size) + 1e-6)
    for i in range(0, size, 8):
        var v = x.load[8](i)
        out_ptr.store[8](i, v * inv_rms)

@always_inline
def q8_matmul_vec(
    W_packed: UnsafePointer[Int8, _],
    scales: UnsafePointer[Float32, _],
    x: UnsafePointer[Float32, _],
    out_ptr: UnsafePointer[Float32, MutExternalOrigin],
    c_in: Int,
    c_out: Int,
    num_tasks: Int,
    threshold: Float32 = 0.0,
):
    var c_in_blocks = c_in // 32
    
    if threshold > 0.0:
        var active_blocks = List[Int]()
        for b in range(c_in_blocks):
            var active = False
            var x_offset = b * 32
            for i in range(32):
                if abs(x[x_offset + i]) > threshold:
                    active = True
                    break
            if active:
                active_blocks.append(b)
                
        var num_active = len(active_blocks)
        if num_active == 0:
            for r in range(c_out):
                out_ptr[r] = 0.0
            return
            
        var chunk_size = c_out // num_tasks
        
        def sparse_worker(t: Int) capturing -> None:
            var start_r = t * chunk_size
            var end_r = (t + 1) * chunk_size if t < num_tasks - 1 else c_out
            
            for r in range(start_r, end_r):
                var acc = SIMD[DType.float32, 8](0.0)
                var r_offset_packed = r * c_in
                var r_offset_scales = r * c_in_blocks
                
                for idx in range(num_active):
                    var b = active_blocks[idx]
                    var scale = scales[r_offset_scales + b]
                    var block_p = r_offset_packed + b * 32
                    var x_offset = b * 32
                    
                    var w0 = W_packed.load[8](block_p).cast[DType.float32]()
                    var w1 = W_packed.load[8](block_p + 8).cast[DType.float32]()
                    var w2 = W_packed.load[8](block_p + 16).cast[DType.float32]()
                    var w3 = W_packed.load[8](block_p + 24).cast[DType.float32]()
                    
                    acc = acc + (w0 * x.load[8](x_offset)) * scale
                    acc = acc + (w1 * x.load[8](x_offset + 8)) * scale
                    acc = acc + (w2 * x.load[8](x_offset + 16)) * scale
                    acc = acc + (w3 * x.load[8](x_offset + 24)) * scale
                    
                out_ptr[r] = acc.reduce_add()
                
        parallelize[sparse_worker](num_tasks)
    else:
        var chunk_size = c_out // num_tasks
        
        def dense_worker(t: Int) capturing -> None:
            var start_r = t * chunk_size
            var end_r = (t + 1) * chunk_size if t < num_tasks - 1 else c_out
            
            for r in range(start_r, end_r):
                var acc = SIMD[DType.float32, 8](0.0)
                var r_offset_packed = r * c_in
                var r_offset_scales = r * c_in_blocks
                
                for b in range(c_in_blocks):
                    var scale = scales[r_offset_scales + b]
                    var block_p = r_offset_packed + b * 32
                    var x_offset = b * 32
                    
                    var w0 = W_packed.load[8](block_p).cast[DType.float32]()
                    var w1 = W_packed.load[8](block_p + 8).cast[DType.float32]()
                    var w2 = W_packed.load[8](block_p + 16).cast[DType.float32]()
                    var w3 = W_packed.load[8](block_p + 24).cast[DType.float32]()
                    
                    acc = acc + (w0 * x.load[8](x_offset)) * scale
                    acc = acc + (w1 * x.load[8](x_offset + 8)) * scale
                    acc = acc + (w2 * x.load[8](x_offset + 16)) * scale
                    acc = acc + (w3 * x.load[8](x_offset + 24)) * scale
                    
                out_ptr[r] = acc.reduce_add()
                
        parallelize[dense_worker](num_tasks)

@always_inline
def q8_matmul_w2_block_column(
    W_packed: UnsafePointer[Int8, _],
    x: UnsafePointer[Float32, _],
    out_ptr: UnsafePointer[Float32, MutExternalOrigin],
    c_in: Int,
    c_out: Int,
    num_tasks: Int,
    threshold: Float32 = 1e-4,
    use_scratch: Bool = False,
    w2_scratch: UnsafePointer[Int8, MutExternalOrigin] = UnsafePointer[Int8, MutExternalOrigin](),
):
    var c_in_blocks = c_in // 32
    
    var active_blocks = List[Int]()
    var max_active_block = -1
    for b in range(c_in_blocks):
        var active = False
        var x_offset = b * 32
        for i in range(32):
            if abs(x[x_offset + i]) > threshold:
                active = True
                break
        if active:
            active_blocks.append(b)
            max_active_block = b
            
    var num_active = len(active_blocks)
    if num_active == 0:
        for r in range(c_out):
            out_ptr[r] = 0.0
        return
        
    var chunk_size = c_out // num_tasks
    
    if use_scratch:
        try:
            var libc = OwnedDLHandle("libc.so.6")
            var c_memcpy = libc.get_function[fn(Int, Int, Int) -> Int]("memcpy")
            var copy_size = (max_active_block + 1) * 139264
            _ = c_memcpy(w2_scratch.__int__(), W_packed.__int__(), copy_size)
        except e:
            print("Error loading libc for copy:", e)
    
    def sparse_worker(t: Int) capturing -> None:
        var start_r = t * chunk_size
        var end_r = (t + 1) * chunk_size if t < num_tasks - 1 else c_out
        
        for r in range(start_r, end_r):
            var acc = SIMD[DType.float32, 8](0.0)
            
            for idx in range(num_active):
                var b = active_blocks[idx]
                var block_addr = w2_scratch.__int__() + b * 139264 if use_scratch else W_packed.__int__() + b * 139264
                var block_base = UnsafePointer[Int8, MutExternalOrigin](unsafe_from_address=block_addr)
                var block_scales = (block_base + 122880).bitcast[Float32]()
                
                var scale = block_scales[r]
                var block_p = block_base + r * 32
                var x_offset = b * 32
                
                var w0 = block_p.load[8]().cast[DType.float32]()
                var w1 = block_p.load[8](8).cast[DType.float32]()
                var w2 = block_p.load[8](16).cast[DType.float32]()
                var w3 = block_p.load[8](24).cast[DType.float32]()
                
                var dot = w0 * x.load[8](x_offset) + w1 * x.load[8](x_offset + 8) + w2 * x.load[8](x_offset + 16) + w3 * x.load[8](x_offset + 24)
                acc = acc + dot * scale
                
            out_ptr[r] = acc.reduce_add()
            
    parallelize[sparse_worker](num_tasks)


# --- 4-bit Q4_0 Unpacking & Matmul ---
@always_inline
def unpack_4bytes(b0: UInt8, b1: UInt8, b2: UInt8, b3: UInt8) -> SIMD[DType.float32, 8]:
    var chunk = SIMD[DType.uint8, 8](b0, b1, b2, b3, 0, 0, 0, 0)
    var repeated = chunk.shuffle[0, 0, 1, 1, 2, 2, 3, 3]()
    var shifted = repeated >> SIMD[DType.uint8, 8](0, 4, 0, 4, 0, 4, 0, 4)
    var masked = shifted & 15
    return (masked.cast[DType.int8]() - 8).cast[DType.float32]()


@always_inline
def q4_matmul_vec(
    W_packed: UnsafePointer[UInt8, _],
    scales: UnsafePointer[Float32, _],
    x: UnsafePointer[Float32, _],
    out_ptr: UnsafePointer[Float32, MutExternalOrigin],
    c_in: Int,
    c_out: Int,
    num_tasks: Int,
):
    var chunk_size = c_out // num_tasks
    var c_in_packed = c_in // 2
    var c_in_blocks = c_in // 32
    
    def worker(t: Int) capturing -> None:
        var start_r = t * chunk_size
        var end_r = (t + 1) * chunk_size if t < num_tasks - 1 else c_out
        
        for r in range(start_r, end_r):
            var acc = SIMD[DType.float32, 8](0.0)
            
            var r_offset_packed = r * c_in_packed
            var r_offset_scales = r * c_in_blocks
            
            for b in range(c_in_blocks):
                var scale = scales[r_offset_scales + b]
                var block_p = r_offset_packed + b * 16
                var x_offset = b * 32
                
                # Load 16 bytes of packed weights at once
                var bytes = W_packed.load[16](block_p)
                
                var w0 = unpack_4bytes(bytes[0], bytes[1], bytes[2], bytes[3])
                var w1 = unpack_4bytes(bytes[4], bytes[5], bytes[6], bytes[7])
                var w2 = unpack_4bytes(bytes[8], bytes[9], bytes[10], bytes[11])
                var w3 = unpack_4bytes(bytes[12], bytes[13], bytes[14], bytes[15])
                
                acc = acc + (w0 * x.load[8](x_offset)) * scale
                acc = acc + (w1 * x.load[8](x_offset + 8)) * scale
                acc = acc + (w2 * x.load[8](x_offset + 16)) * scale
                acc = acc + (w3 * x.load[8](x_offset + 24)) * scale
                
            out_ptr[r] = acc.reduce_add()
            
    parallelize[worker](num_tasks)


@always_inline
def fast_tanh(x: Float32) -> Float32:
    if x > 15.0:
        return 1.0
    elif x < -15.0:
        return -1.0
    var e2x = exp(2.0 * x)
    return (e2x - 1.0) / (e2x + 1.0)

@always_inline
def gelu_and_mul(h1: UnsafePointer[Float32, _], h2: UnsafePointer[Float32, _], out_ptr: UnsafePointer[Float32, MutExternalOrigin], size: Int):
    for i in range(size):
        var x1 = h1[i]
        var sig = 1.0 / (1.0 + exp(-1.702 * x1))
        var gelu = x1 * sig
        out_ptr[i] = gelu * h2[i]

@always_inline
def apply_rope(vec: UnsafePointer[Float32, MutExternalOrigin], num_heads: Int, head_dim: Int, pos: Int, is_global: Bool):
    var half_dim = head_dim // 2
    var base: Float32 = Float32(1000000.0) if is_global else Float32(10000.0)
    var rope_dim = 512 if is_global else 256
    
    for h in range(num_heads):
        var h_offset = h * head_dim
        for i in range(half_dim):
            var theta: Float32 = 0.0
            if (not is_global) or (i < 64):
                theta = Float32(pos) / (base ** (Float32(2 * i) / Float32(rope_dim)))
                
            var cos_theta = cos(theta)
            var sin_theta = sin(theta)
            
            var val1 = vec[h_offset + i]
            var val2 = vec[h_offset + half_dim + i]
            
            vec[h_offset + i] = val1 * cos_theta - val2 * sin_theta
            vec[h_offset + half_dim + i] = val2 * cos_theta + val1 * sin_theta

@always_inline
def softmax(scores: UnsafePointer[Float32, MutExternalOrigin], size: Int):
    var max_val = scores[0]
    for i in range(1, size):
        if scores[i] > max_val:
            max_val = scores[i]
            
    var sum_exp: Float32 = 0.0
    for i in range(size):
        scores[i] = exp(scores[i] - max_val)
        sum_exp += scores[i]
        
    for i in range(size):
        scores[i] /= sum_exp

# Helper to wait for io_uring tags
def wait_for_tag(ring: IoUring, expected_tag: UInt64, out_tag_ptr: UnsafePointer[UInt64, MutExternalOrigin]) raises:
    var attempts = 0
    while True:
        var res = ring.poll_completion(out_tag_ptr.__int__())
        if res >= 0:
            if out_tag_ptr[0] == expected_tag:
                return
        attempts += 1
        if attempts > 30000000:
            raise Error("Weights paging I/O timed out waiting for layer tag: " + String(expected_tag))

fn string_to_int(s: String) -> Int:
    var bytes = s.as_bytes()
    var val = 0
    var sign = 1
    var start = 0
    if len(bytes) > 0 and bytes[0] == 45: # "-" is 45
        sign = -1
        start = 1
    for i in range(start, len(bytes)):
        var c = bytes[i]
        if c >= 48 and c <= 57:
            val = val * 10 + (Int(c) - 48)
    return val * sign

fn string_to_float(s: String) -> Float32:
    var bytes = s.as_bytes()
    var val: Float32 = 0.0
    var sign: Float32 = 1.0
    var start = 0
    if len(bytes) > 0 and bytes[0] == 45: # "-" is 45
        sign = -1.0
        start = 1
    var i = start
    while i < len(bytes) and bytes[i] != 46 and bytes[i] != 101 and bytes[i] != 69: # "." is 46, "e" is 101, "E" is 69
        var c = bytes[i]
        if c >= 48 and c <= 57:
            val = val * 10.0 + Float32(c - 48)
        i += 1
    if i < len(bytes) and bytes[i] == 46:
        i += 1
        var dec: Float32 = 0.1
        while i < len(bytes) and bytes[i] != 101 and bytes[i] != 69:
            var c = bytes[i]
            if c >= 48 and c <= 57:
                val = val + Float32(c - 48) * dec
                dec *= 0.1
            i += 1
    if i < len(bytes) and (bytes[i] == 101 or bytes[i] == 69):
        i += 1
        var exp_sign = 1
        if i < len(bytes) and bytes[i] == 45: # "-" is 45
            exp_sign = -1
            i += 1
        elif i < len(bytes) and bytes[i] == 43: # "+" is 43
            i += 1
        var exp_val = 0
        while i < len(bytes):
            var c = bytes[i]
            if c >= 48 and c <= 57:
                exp_val = exp_val * 10 + (Int(c) - 48)
            i += 1
        if exp_sign == -1:
            for _ in range(exp_val):
                val /= 10.0
        else:
            for _ in range(exp_val):
                val *= 10.0
    return val * sign

# -----------------------------------------------------------------------------
# 4. Main Executive Engine
# -----------------------------------------------------------------------------
def main() raises:
    var num_tasks = num_physical_cores()

    # Initialize FFI and libc functions
    var libc = OwnedDLHandle("libc.so.6")
    var c_open = libc.get_function[def(Int, Int32, Int32) -> Int32]("open")
    var c_close = libc.get_function[def(Int32) -> Int32]("close")
    var posix_memalign = libc.get_function[def(Int, Int, Int) -> Int32]("posix_memalign")
    var free_c = libc.get_function[def(Int) -> NoneType]("free")
    var c_read = libc.get_function[def(Int32, Int, Int) -> Int]("read")

    var py: PythonObject

    # Pure Mojo Command Line Arguments Parser
    var mode = String("speculative")
    var resident = String("ram")
    var w2_threshold: Float32 = 0.0001
    var target_weights_file = String("gemma4_weights_q8_sorted.bin")
    var draft_weights_file = String("gemma4_weights_q4.bin")
    var prompt_str = String("What is the capital of France?")
    var max_steps = 50

    var args_list = argv()
    var arg_idx = 1
    while arg_idx < len(args_list):
        var arg = args_list[arg_idx]
        if arg == "--mode" and arg_idx + 1 < len(args_list):
            mode = args_list[arg_idx + 1]
            arg_idx += 2
        elif arg == "--resident" and arg_idx + 1 < len(args_list):
            resident = args_list[arg_idx + 1]
            arg_idx += 2
        elif arg == "--w2-threshold" and arg_idx + 1 < len(args_list):
            w2_threshold = string_to_float(args_list[arg_idx + 1])
            arg_idx += 2
        elif arg == "--target-weights" and arg_idx + 1 < len(args_list):
            target_weights_file = args_list[arg_idx + 1]
            arg_idx += 2
        elif arg == "--draft-weights" and arg_idx + 1 < len(args_list):
            draft_weights_file = args_list[arg_idx + 1]
            arg_idx += 2
        elif arg == "--prompt" and arg_idx + 1 < len(args_list):
            prompt_str = args_list[arg_idx + 1]
            arg_idx += 2
        elif arg == "--steps" and arg_idx + 1 < len(args_list):
            max_steps = string_to_int(args_list[arg_idx + 1])
            arg_idx += 2
        else:
            arg_idx += 1

    print("=========================================================")
    print("   GEMMA 4 12B UNIFIED INFERENCE ENGINE (Mojo)           ")
    print("=========================================================")
    print("Embedding Size:      3840")
    print("Attention Heads:     16 Query Heads / 8 KV Heads (GQA/MQA)")
    print("Model Parameters:    12B Scale Causal Speculative Layout")
    print("Execution Mode:     ", mode.upper())
    print("Weight Residency:   ", resident.upper())
    print("W2 Skip Threshold:  ", w2_threshold)
    print("Target Weights:     ", target_weights_file)
    if mode == "speculative":
        print("Draft Weights:      ", draft_weights_file)
    print("---------------------------------------------------------")

    var TARGET_CONFIG = TargetConfig[
        embed_dim = 3840,
        head_dim = 256,
        num_heads = 16,
        num_kv_heads = 8,
        intermediate_dim = 15360,
        num_layers = 48,
        vocab_size = 262144,
        max_seq_len = 100
    ]()

    var DRAFT_CONFIG = DraftConfig[
        embed_dim = 3840,
        head_dim = 256,
        num_heads = 16,
        num_kv_heads = 8,
        intermediate_dim = 15360,
        num_layers = 48,
        vocab_size = 262144,
        max_seq_len = 100
    ]()

    var tokenizer: PythonObject


    # Setup target weights memory
    var target_file_size = TARGET_CONFIG.get_layer_offset(TARGET_CONFIG.num_layers)
    var target_base_ptr = UnsafePointer[Int8, MutExternalOrigin]()
    var w_a_ptr = UnsafePointer[Int8, MutExternalOrigin]()
    var w_b_ptr = UnsafePointer[Int8, MutExternalOrigin]()
    var weights_fd: Int32 = -1
    var gpu_ctx_ptr = UnsafePointer[GPUContext, MutExternalOrigin]()
    var w2_scratch = UnsafePointer[Int8, MutExternalOrigin]()

    var ring = IoUring(16)
    var out_tag_ptr = alloc[UInt64](1)
    out_tag_ptr[0] = 0

    if resident == "ram":
        # Load entire target weights file synchronously into RAM
        print("Opening target weights file '" + target_weights_file + "'...")
        var path_bytes = target_weights_file.as_bytes()
        var path_ptr = alloc[UInt8](len(path_bytes) + 1)
        for i in range(len(path_bytes)):
            path_ptr[i] = path_bytes[i]
        path_ptr[len(path_bytes)] = 0

        weights_fd = c_open(path_ptr.__int__(), 0, 0)
        if weights_fd < 0:
            print("CRITICAL: Failed to open target weights file.")
            return

        var memptr_target = alloc[Int](1)
        _ = posix_memalign(memptr_target.__int__(), 4096, target_file_size)
        var target_base_addr = memptr_target[0]
        target_base_ptr = UnsafePointer[Int8, MutExternalOrigin](unsafe_from_address=target_base_addr)

        print("Loading entire target weights fully into RAM (" + String(target_file_size / (1024 * 1024 * 1024)) + " GB)...")
        var target_total_read = 0
        while target_total_read < target_file_size:
            var bytes_to_read = target_file_size - target_total_read
            var read_res = c_read(weights_fd, target_base_ptr.__int__() + target_total_read, bytes_to_read)
            if read_res <= 0:
                print("CRITICAL: Failed to read from target weights file!")
                return
            target_total_read += read_res
        print("Loaded entire target weights fully into RAM: ", target_total_read, " bytes")
        _ = c_close(weights_fd)
        path_ptr.free()
    elif resident == "gpu":
        print("Scanning topology for AMD GPU...")
        var gpu_id: UInt32 = 0
        var c_ioctl = libc.get_function[def(Int32, Int, Int) -> Int32]("ioctl")
        var c_memcpy = libc.get_function[def(Int, Int, Int) -> Int]("memcpy")

        for node_idx in range(1, 16):
            var path = "/sys/class/kfd/kfd/topology/nodes/" + String(node_idx) + "/gpu_id"
            var path_bytes = path.as_bytes()
            var path_ptr = alloc[UInt8](len(path_bytes) + 1)
            for i in range(len(path_bytes)):
                path_ptr[i] = path_bytes[i]
            path_ptr[len(path_bytes)] = 0
            var fd = c_open(path_ptr.__int__(), 0, 0)
            path_ptr.free()
            
            if fd >= 0:
                var buf = alloc[UInt8](64)
                var bytes_read = c_read(fd, buf.__int__(), 63)
                _ = c_close(fd)
                if bytes_read > 0:
                    buf[bytes_read] = 0
                    var val: Int = 0
                    for i in range(bytes_read):
                        var char_code = buf[i]
                        if char_code >= 48 and char_code <= 57:
                            val = val * 10 + (Int(char_code) - 48)
                    buf.free()
                    if val > 0:
                        gpu_id = UInt32(val)
                        break
                else:
                    buf.free()

        if gpu_id == 0:
            gpu_id = 21148 # default fallback

        print(" -> Resolved GpuID =", gpu_id)

        # Open Dev nodes
        var kfd_path = "/dev/kfd"
        var kfd_path_bytes = kfd_path.as_bytes()
        var kfd_path_ptr = alloc[UInt8](len(kfd_path_bytes) + 1)
        for i in range(len(kfd_path_bytes)):
            kfd_path_ptr[i] = kfd_path_bytes[i]
        kfd_path_ptr[len(kfd_path_bytes)] = 0
        var kfd_fd = c_open(kfd_path_ptr.__int__(), 2, 0)
        kfd_path_ptr.free()

        var drm_path = "/dev/dri/renderD128"
        var drm_path_bytes = drm_path.as_bytes()
        var drm_path_ptr = alloc[UInt8](len(drm_path_bytes) + 1)
        for i in range(len(drm_path_bytes)):
            drm_path_ptr[i] = drm_path_bytes[i]
        drm_path_ptr[len(drm_path_bytes)] = 0
        var drm_fd = c_open(drm_path_ptr.__int__(), 2, 0)
        drm_path_ptr.free()

        if kfd_fd < 0 or drm_fd < 0:
            print("CRITICAL: Failed to open KFD/DRM device nodes!")
            return

        # Acquire VM
        var acq_args_ptr = alloc[kfd_ioctl_acquire_vm_args](1)
        acq_args_ptr[0] = kfd_ioctl_acquire_vm_args(drm_fd=UInt32(drm_fd), gpu_id=gpu_id)
        var res = c_ioctl(kfd_fd, 0x40084B15, acq_args_ptr.__int__())
        acq_args_ptr.free()
        if res < 0:
            print("CRITICAL: VM Acquisition failed!")
            return

        # Enable Runtime
        var runtime_enable_ptr = alloc[kfd_ioctl_runtime_enable_args](1)
        runtime_enable_ptr[0] = kfd_ioctl_runtime_enable_args()
        res = c_ioctl(kfd_fd, 0xC0104B25, runtime_enable_ptr.__int__())
        runtime_enable_ptr.free()
        if res < 0:
            print("CRITICAL: Runtime enable failed!")
            return

        print(" -> Initializing GPU userspace queue and loading ELF compiled GCN shader...")
        gpu_ctx_ptr = alloc[GPUContext](1)
        gpu_ctx_ptr[].__init__(kfd_fd, drm_fd, gpu_id)
        print(" -> GPU Context initialized successfully!")

        var gtt_cached_flags: UInt32 = 0xF6000002
        print(" -> Allocating GTT weight buffer for the entire 12.29 GB of weights...")
        var weights_gpu_buf = gpu_ctx_ptr[].alloc_gpu_buffer(target_file_size, gtt_cached_flags, cpu_access=True)
        gpu_ctx_ptr[].weights_gpu_buf = weights_gpu_buf.copy()
        target_base_ptr = UnsafePointer[Int8, MutExternalOrigin](unsafe_from_address=weights_gpu_buf.cpu_addr)

        # Initialize Python time module for load timing
        py = Python.import_module("time")

        print("Opening target weights file '" + target_weights_file + "'...")
        var t_path_bytes = target_weights_file.as_bytes()
        var t_path_ptr = alloc[UInt8](len(t_path_bytes) + 1)
        for i in range(len(t_path_bytes)):
            t_path_ptr[i] = t_path_bytes[i]
        t_path_ptr[len(t_path_bytes)] = 0
        weights_fd = c_open(t_path_ptr.__int__(), 0, 0)
        t_path_ptr.free()
        if weights_fd < 0:
            print("CRITICAL: Failed to open target weights file.")
            return

        var total_read_weights = 0
        print("Loading weights into GPU GTT memory...")
        var t_load_start = py.time()
        while total_read_weights < target_file_size:
            var bytes_to_read = target_file_size - total_read_weights
            if bytes_to_read > 1024 * 1024 * 1024:
                bytes_to_read = 1024 * 1024 * 1024
            var read_res = c_read(weights_fd, weights_gpu_buf.cpu_addr + total_read_weights, bytes_to_read)
            if read_res <= 0:
                print("CRITICAL: Failed to read from weights file!")
                return
            total_read_weights += read_res
            print(" -> Loaded ", total_read_weights // (1024*1024), " MB / ", target_file_size // (1024*1024), " MB")
            
        var t_load_end = py.time()
        print("Weights loaded into GTT memory successfully in " + String(Float64(py=(t_load_end - t_load_start))) + " sec!")
        _ = c_close(weights_fd)

        # Allocate GTT activation buffers
        var x_buf = gpu_ctx_ptr[].alloc_gpu_buffer(((TARGET_CONFIG.embed_dim * 4 + 4095) // 4096) * 4096 + 4096, gtt_cached_flags, cpu_access=True)
        var norm_x_buf = gpu_ctx_ptr[].alloc_gpu_buffer(((TARGET_CONFIG.embed_dim * 4 + 4095) // 4096) * 4096 + 4096, gtt_cached_flags, cpu_access=True)
        var norm_x_buf_1 = gpu_ctx_ptr[].alloc_gpu_buffer(((TARGET_CONFIG.embed_dim * 4 + 4095) // 4096) * 4096 + 4096, gtt_cached_flags, cpu_access=True)
        var final_norm_x_buf = gpu_ctx_ptr[].alloc_gpu_buffer(((TARGET_CONFIG.embed_dim * 4 + 4095) // 4096) * 4096 + 4096, gtt_cached_flags, cpu_access=True)
        
        var q_buf = gpu_ctx_ptr[].alloc_gpu_buffer(((8192 * 4 + 4095) // 4096) * 4096 + 4096, gtt_cached_flags, cpu_access=True)
        var q_buf_1 = gpu_ctx_ptr[].alloc_gpu_buffer(((8192 * 4 + 4095) // 4096) * 4096 + 4096, gtt_cached_flags, cpu_access=True)
        var k_vec_buf = gpu_ctx_ptr[].alloc_gpu_buffer(((2048 * 4 + 4095) // 4096) * 4096 + 4096, gtt_cached_flags, cpu_access=True)
        var k_vec_buf_1 = gpu_ctx_ptr[].alloc_gpu_buffer(((2048 * 4 + 4095) // 4096) * 4096 + 4096, gtt_cached_flags, cpu_access=True)
        var v_vec_buf = gpu_ctx_ptr[].alloc_gpu_buffer(((2048 * 4 + 4095) // 4096) * 4096 + 4096, gtt_cached_flags, cpu_access=True)
        var v_vec_buf_1 = gpu_ctx_ptr[].alloc_gpu_buffer(((2048 * 4 + 4095) // 4096) * 4096 + 4096, gtt_cached_flags, cpu_access=True)
        
        var attn_out_buf = gpu_ctx_ptr[].alloc_gpu_buffer(((8192 * 4 + 4095) // 4096) * 4096 + 4096, gtt_cached_flags, cpu_access=True)
        var attn_out_buf_1 = gpu_ctx_ptr[].alloc_gpu_buffer(((8192 * 4 + 4095) // 4096) * 4096 + 4096, gtt_cached_flags, cpu_access=True)
        var proj_attn_out_buf = gpu_ctx_ptr[].alloc_gpu_buffer(((TARGET_CONFIG.embed_dim * 4 + 4095) // 4096) * 4096 + 4096, gtt_cached_flags, cpu_access=True)
        var h1_buf = gpu_ctx_ptr[].alloc_gpu_buffer(((TARGET_CONFIG.intermediate_dim * 4 + 4095) // 4096) * 4096 + 4096, gtt_cached_flags, cpu_access=True)
        var h2_buf = gpu_ctx_ptr[].alloc_gpu_buffer(((TARGET_CONFIG.intermediate_dim * 4 + 4095) // 4096) * 4096 + 4096, gtt_cached_flags, cpu_access=True)

        gpu_ctx_ptr[].x_buf = x_buf.copy()
        gpu_ctx_ptr[].norm_x_buf = norm_x_buf.copy()
        gpu_ctx_ptr[].norm_x_buf_1 = norm_x_buf_1.copy()
        gpu_ctx_ptr[].final_norm_x_buf = final_norm_x_buf.copy()
        gpu_ctx_ptr[].q_buf = q_buf.copy()
        gpu_ctx_ptr[].q_buf_1 = q_buf_1.copy()
        gpu_ctx_ptr[].k_vec_buf = k_vec_buf.copy()
        gpu_ctx_ptr[].k_vec_buf_1 = k_vec_buf_1.copy()
        gpu_ctx_ptr[].v_vec_buf = v_vec_buf.copy()
        gpu_ctx_ptr[].v_vec_buf_1 = v_vec_buf_1.copy()
        gpu_ctx_ptr[].attn_out_buf = attn_out_buf.copy()
        gpu_ctx_ptr[].attn_out_buf_1 = attn_out_buf_1.copy()
        gpu_ctx_ptr[].proj_attn_out_buf = proj_attn_out_buf.copy()
        gpu_ctx_ptr[].h1_buf = h1_buf.copy()
        gpu_ctx_ptr[].h2_buf = h2_buf.copy()

        w2_scratch = alloc[Int8](480 * 139264)
    else:
        # Paged mode: setup double buffers of size max_layer_bytes
        var max_layer_bytes = 0
        for i in range(TARGET_CONFIG.num_layers):
            var sz = TARGET_CONFIG.get_layer_aligned_size(i)
            if sz > max_layer_bytes:
                max_layer_bytes = sz

        var memptr_w_a = alloc[Int](1)
        _ = posix_memalign(memptr_w_a.__int__(), 4096, max_layer_bytes)
        var w_a_addr = memptr_w_a[0]
        w_a_ptr = UnsafePointer[Int8, MutExternalOrigin](unsafe_from_address=w_a_addr)

        var memptr_w_b = alloc[Int](1)
        _ = posix_memalign(memptr_w_b.__int__(), 4096, max_layer_bytes)
        var w_b_addr = memptr_w_b[0]
        w_b_ptr = UnsafePointer[Int8, MutExternalOrigin](unsafe_from_address=w_b_addr)

        print("Opening target weights file '" + target_weights_file + "' with O_DIRECT read-only...")
        var path_bytes = target_weights_file.as_bytes()
        var path_ptr = alloc[UInt8](len(path_bytes) + 1)
        for i in range(len(path_bytes)):
            path_ptr[i] = path_bytes[i]
        path_ptr[len(path_bytes)] = 0

        # Flags: O_RDONLY (0) | O_DIRECT (16384) = 16384
        weights_fd = c_open(path_ptr.__int__(), 16384, 0)
        if weights_fd < 0:
            print("CRITICAL: Failed to open target weights file with O_DIRECT.")
            return
        else:
            print("Target weights file opened successfully. Active size: ", target_file_size, " bytes")
        path_ptr.free()

    # Load Embeddings and final norm
    print("Loading Q4 embeddings from gemma4_embed_q4.bin...")
    var embed = Q4Embedding("gemma4_embed_q4.bin")

    var final_norm_file = "gemma4_final_norm.bin"
    var final_norm_path_bytes = final_norm_file.as_bytes()
    var final_norm_path_ptr = alloc[UInt8](len(final_norm_path_bytes) + 1)
    for i in range(len(final_norm_path_bytes)):
        final_norm_path_ptr[i] = final_norm_path_bytes[i]
    final_norm_path_ptr[len(final_norm_path_bytes)] = 0
    var final_norm_fd = c_open(final_norm_path_ptr.__int__(), 0, 0)
    if final_norm_fd < 0:
        print("CRITICAL: Failed to open 'gemma4_final_norm.bin'!")
        return

    var final_norm_weight = alloc[Float32](TARGET_CONFIG.embed_dim)
    var final_norm_size_bytes = TARGET_CONFIG.embed_dim * 4
    var total_read = 0
    while total_read < final_norm_size_bytes:
        var bytes_to_read = final_norm_size_bytes - total_read
        var read_res = c_read(final_norm_fd, final_norm_weight.__int__() + total_read, bytes_to_read)
        if read_res <= 0:
            print("CRITICAL: Failed to read from gemma4_final_norm.bin!")
            return
        total_read += read_res
    print("Loaded final norm weights into memory: ", total_read, " bytes")
    print("---------------------------------------------------------")
    _ = c_close(final_norm_fd)
    final_norm_path_ptr.free()

    # Initialize Draft Model weights (only if in speculative mode)
    var draft_base_ptr = UnsafePointer[UInt8, MutExternalOrigin]()
    if mode == "speculative":
        print("Opening Q4_0 draft weights file '" + draft_weights_file + "'...")
        var draft_path_bytes = draft_weights_file.as_bytes()
        var draft_path_ptr = alloc[UInt8](len(draft_path_bytes) + 1)
        for i in range(len(draft_path_bytes)):
            draft_path_ptr[i] = draft_path_bytes[i]
        draft_path_ptr[len(draft_path_bytes)] = 0
        var draft_fd = c_open(draft_path_ptr.__int__(), 0, 0)
        if draft_fd < 0:
            print("CRITICAL: Failed to open '" + draft_weights_file + "'!")
            return

        var draft_file_size = DRAFT_CONFIG.get_layer_offset(DRAFT_CONFIG.num_layers)
        var memptr_draft = alloc[Int](1)
        _ = posix_memalign(memptr_draft.__int__(), 4096, draft_file_size)
        var draft_base_addr = memptr_draft[0]
        draft_base_ptr = UnsafePointer[UInt8, MutExternalOrigin](unsafe_from_address=draft_base_addr)

        print("Loading entire Q4_0 draft weights into memory (6.82 GB)...")
        total_read = 0
        while total_read < draft_file_size:
            var bytes_to_read = draft_file_size - total_read
            var read_res = c_read(draft_fd, draft_base_addr + total_read, bytes_to_read)
            if read_res <= 0:
                print("CRITICAL: Failed to read from " + draft_weights_file + "!")
                return
            total_read += read_res
        print("Loaded Q4 draft weights fully into RAM: ", total_read, " bytes")
        print("---------------------------------------------------------")
        _ = c_close(draft_fd)
        draft_path_ptr.free()


    # 5. Allocate continuous KV caches and activation buffers
    # Precompute layer cache offsets and total cache size
    var layer_cache_offsets = List[Int]()
    var current_cache_offset = 0
    for i in range(TARGET_CONFIG.num_layers):
        layer_cache_offsets.append(current_cache_offset)
        var is_global = TARGET_CONFIG.is_global_layer(i)
        var layer_kv_heads = 1 if is_global else 8
        var layer_head_dim = 512 if is_global else 256
        current_cache_offset += TARGET_CONFIG.max_seq_len * layer_kv_heads * layer_head_dim
    var total_cache_size = current_cache_offset

    # Draft (Ternary) KV Cache
    var draft_k_cache = alloc[Float32](total_cache_size)
    var draft_v_cache = alloc[Float32](total_cache_size)
    
    # Target (Q8_0) KV Cache
    var target_k_cache = alloc[Float32](total_cache_size)
    var target_v_cache = alloc[Float32](total_cache_size)

    for i in range(total_cache_size):
        draft_k_cache[i] = 0.0
        draft_v_cache[i] = 0.0
        target_k_cache[i] = 0.0
        target_v_cache[i] = 0.0

    # Common shared draft activations
    var draft_x = alloc[Float32](DRAFT_CONFIG.embed_dim)
    var draft_norm_x = alloc[Float32](DRAFT_CONFIG.embed_dim)
    var draft_q = alloc[Float32](8192)
    var draft_k_vec = alloc[Float32](2048)
    var draft_v_vec = alloc[Float32](2048)
    var draft_attn_scores = alloc[Float32](DRAFT_CONFIG.max_seq_len)
    var draft_attn_out = alloc[Float32](8192)
    var draft_proj_attn_out = alloc[Float32](DRAFT_CONFIG.embed_dim)
    var draft_h1 = alloc[Float32](DRAFT_CONFIG.intermediate_dim)
    var draft_h2 = alloc[Float32](DRAFT_CONFIG.intermediate_dim)
    var draft_ffn_out_gate = alloc[Float32](DRAFT_CONFIG.intermediate_dim)
    var draft_ffn_out = alloc[Float32](DRAFT_CONFIG.embed_dim)

    # Common target activations (can validation-batch up to K+1 tokens sequentially inside layer loops)
    comptime K_MAX = 5
    var target_x_seq = alloc[Float32]((K_MAX + 1) * TARGET_CONFIG.embed_dim)
    var target_norm_x = alloc[Float32](TARGET_CONFIG.embed_dim)
    var target_q = alloc[Float32](8192)
    var target_k_vec = alloc[Float32](2048)
    var target_v_vec = alloc[Float32](2048)
    var target_attn_scores = alloc[Float32](TARGET_CONFIG.max_seq_len)
    var target_attn_out = alloc[Float32](8192)
    var target_proj_attn_out = alloc[Float32](TARGET_CONFIG.embed_dim)
    var target_h1 = alloc[Float32](TARGET_CONFIG.intermediate_dim)
    var target_h2 = alloc[Float32](TARGET_CONFIG.intermediate_dim)
    var target_ffn_out_gate = alloc[Float32](TARGET_CONFIG.intermediate_dim)
    var target_ffn_out = alloc[Float32](TARGET_CONFIG.embed_dim)
    var target_final_norm_x = alloc[Float32](TARGET_CONFIG.embed_dim)

    # Temporary variable structures for projection worker
    var chunk_size = TARGET_CONFIG.vocab_size // num_tasks
    var local_max_logits = alloc[Float32](num_tasks)
    var local_next_tokens = alloc[Int](num_tasks)

    # Initialize py for generation timing and load Gemma 4 tokenizer after all GTT memory and weights are allocated and loaded
    py = Python.import_module("time")
    var transformers = Python.import_module("transformers")
    print("Loading Gemma 4 tokenizer...")
    tokenizer = transformers.AutoTokenizer.from_pretrained("google/gemma-4-12B-it")
    print("Tokenizer loaded successfully!")

    # Boot Interactive Conversation
    print("=========================================================")
    print("   GEMMA 4 12B-SCALE SPECULATIVE CHAT AGENT              ")
    print("=========================================================")
    print("Embedding Size:      3840")
    print("Attention Heads:     16 Query Heads / 8 KV Heads (GQA/MQA)")
    print("Max Sequence Length: 100")
    print("---------------------------------------------------------")
    print("(Type 'exit' or 'quit' to end the conversation)")

    var py_builtins = Python.import_module("builtins")
    var list_cls = py_builtins.list

    while True:
        print()
        var user_input = py_builtins.input("User > ")
        var user_input_str = String(user_input).strip()
        if user_input_str == "exit" or user_input_str == "quit":
            break
            
        var formatted_prompt = "<bos><|turn>user\n" + user_input_str + "<turn|>\n<|turn>model\n"
        var encoded = tokenizer.encode(formatted_prompt, add_special_tokens=False)
        var num_encoded = len(encoded)
        if num_encoded == 0:
            print("Agent > (Please type something!)")
            continue
            
        print("[Tokenizer] Encoded input into", num_encoded, "real sub-word token IDs:")
        var token_ids_str = String("")
        var prompt_tokens = List[Int]()
        for i in range(num_encoded):
            var tid = Int(py=encoded[i])
            token_ids_str += String(tid) + " "
            prompt_tokens.append(tid)
        print(token_ids_str)
            
        print("Agent > ", end="")
        
        var num_prompt_tokens = len(prompt_tokens)
        var max_steps = num_prompt_tokens + 30
        if max_steps > TARGET_CONFIG.max_seq_len:
            max_steps = TARGET_CONFIG.max_seq_len
            
        var scale = sqrt(Float32(TARGET_CONFIG.embed_dim))
        
        # 1. Draft Prefill Phase: process all prompt tokens layer-by-layer (fully in memory, extremely fast)
        if mode == "speculative":
            var draft_prompt_x = alloc[Float32](num_prompt_tokens * DRAFT_CONFIG.embed_dim)
            var draft_prompt_ptr = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=draft_prompt_x.__int__())
            for pos in range(num_prompt_tokens):
                var px_offset = pos * DRAFT_CONFIG.embed_dim
                embed.reconstruct(prompt_tokens[pos], draft_prompt_ptr + px_offset)
                for i in range(DRAFT_CONFIG.embed_dim):
                    draft_prompt_x[px_offset + i] *= scale

            for layer_idx in range(DRAFT_CONFIG.num_layers):
                var weights = DRAFT_CONFIG.get_layer_weights(layer_idx, draft_base_ptr + DRAFT_CONFIG.get_layer_offset(layer_idx))
                var is_global = DRAFT_CONFIG.is_global_layer(layer_idx)
                var layer_heads = 16
                var layer_kv_heads = 1 if is_global else 8
                var layer_head_dim = 512 if is_global else 256
                var kv_stride = layer_kv_heads * layer_head_dim
                var ratio = layer_heads // layer_kv_heads
                
                for pos in range(num_prompt_tokens):
                    var px_offset = pos * DRAFT_CONFIG.embed_dim
                    var local_x = draft_prompt_x + px_offset
                    
                    rms_norm(local_x, weights.input_layernorm_weight, draft_norm_x, DRAFT_CONFIG.embed_dim)
                    
                    q4_matmul_vec(weights.wq_packed, weights.wq_scales, draft_norm_x, draft_q, DRAFT_CONFIG.embed_dim, layer_heads * layer_head_dim, num_tasks)
                    q4_matmul_vec(weights.wk_packed, weights.wk_scales, draft_norm_x, draft_k_vec, DRAFT_CONFIG.embed_dim, layer_kv_heads * layer_head_dim, num_tasks)
                    
                    if is_global:
                        for i in range(kv_stride):
                            draft_v_vec[i] = draft_k_vec[i]
                    else:
                        q4_matmul_vec(weights.wv_packed, weights.wv_scales, draft_norm_x, draft_v_vec, DRAFT_CONFIG.embed_dim, layer_kv_heads * layer_head_dim, num_tasks)
                        
                    for h in range(layer_heads):
                        var head_ptr = draft_q + h * layer_head_dim
                        rms_norm(head_ptr, weights.q_norm_weight, head_ptr, layer_head_dim)
                    for h in range(layer_kv_heads):
                        var head_ptr = draft_k_vec + h * layer_head_dim
                        rms_norm(head_ptr, weights.k_norm_weight, head_ptr, layer_head_dim)
                    for h in range(layer_kv_heads):
                        var head_ptr = draft_v_vec + h * layer_head_dim
                        rms_norm_noscale(head_ptr, head_ptr, layer_head_dim)
                        
                    apply_rope(draft_q, layer_heads, layer_head_dim, pos, is_global)
                    apply_rope(draft_k_vec, layer_kv_heads, layer_head_dim, pos, is_global)
                    
                    var layer_cache_offset = layer_cache_offsets[layer_idx]
                    var step_cache_offset = layer_cache_offset + pos * kv_stride
                    for i in range(kv_stride):
                        draft_k_cache[step_cache_offset + i] = draft_k_vec[i]
                        draft_v_cache[step_cache_offset + i] = draft_v_vec[i]
                        
                    for h in range(layer_heads):
                        var h_offset = h * layer_head_dim
                        var g = h // ratio
                        var g_offset = g * layer_head_dim
                        for t in range(pos + 1):
                            var t_cache_offset = layer_cache_offset + t * kv_stride + g_offset
                            var score: Float32 = 0.0
                            for d in range(layer_head_dim):
                                score += draft_q[h_offset + d] * draft_k_cache[t_cache_offset + d]
                            draft_attn_scores[t] = score
                            
                        softmax(draft_attn_scores, pos + 1)
                        
                        for d in range(layer_head_dim):
                            var val_sum: Float32 = 0.0
                            for t in range(pos + 1):
                                var t_cache_offset = layer_cache_offset + t * kv_stride + g_offset
                                val_sum += draft_attn_scores[t] * draft_v_cache[t_cache_offset + d]
                            draft_attn_out[h_offset + d] = val_sum
                            
                    q4_matmul_vec(weights.wo_packed, weights.wo_scales, draft_attn_out, draft_proj_attn_out, layer_heads * layer_head_dim, DRAFT_CONFIG.embed_dim, num_tasks)
                    
                    rms_norm(draft_proj_attn_out, weights.post_attention_layernorm_weight, draft_proj_attn_out, DRAFT_CONFIG.embed_dim)
                    for i in range(DRAFT_CONFIG.embed_dim):
                        local_x[i] += draft_proj_attn_out[i]
                        
                    rms_norm(local_x, weights.pre_feedforward_layernorm_weight, draft_norm_x, DRAFT_CONFIG.embed_dim)
                    q4_matmul_vec(weights.w1_packed, weights.w1_scales, draft_norm_x, draft_h1, DRAFT_CONFIG.embed_dim, DRAFT_CONFIG.intermediate_dim, num_tasks)
                    q4_matmul_vec(weights.w3_packed, weights.w3_scales, draft_norm_x, draft_h2, DRAFT_CONFIG.embed_dim, DRAFT_CONFIG.intermediate_dim, num_tasks)
                    gelu_and_mul(draft_h1, draft_h2, draft_ffn_out_gate, DRAFT_CONFIG.intermediate_dim)
                    
                    q4_matmul_vec(weights.w2_packed, weights.w2_scales, draft_ffn_out_gate, draft_ffn_out, DRAFT_CONFIG.intermediate_dim, DRAFT_CONFIG.embed_dim, num_tasks)
                    
                    rms_norm(draft_ffn_out, weights.post_feedforward_layernorm_weight, draft_ffn_out, DRAFT_CONFIG.embed_dim)
                    for i in range(DRAFT_CONFIG.embed_dim):
                        local_x[i] += draft_ffn_out[i]
                        
                    var scalar = weights.layer_scalar[0]
                    for i in range(DRAFT_CONFIG.embed_dim):
                        local_x[i] *= scalar
            draft_prompt_x.free()

        # 2. Target Model Prefill Phase
        var t_prefill_start = py.time()
        var current_token = 0
        var pos = 0

        if resident == "gpu":
            print(" -> Running GPU-accelerated Target Model Prefill Phase...")
            var prompt_x = alloc[Float32](num_prompt_tokens * TARGET_CONFIG.embed_dim)
            var target_prompt_ptr = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=prompt_x.__int__())
            for pos_idx in range(num_prompt_tokens):
                var px_offset = pos_idx * TARGET_CONFIG.embed_dim
                embed.reconstruct(prompt_tokens[pos_idx], target_prompt_ptr + px_offset)
                for i in range(TARGET_CONFIG.embed_dim):
                    prompt_x[px_offset + i] *= scale

            # Setup pointers to GPU memory
            var norm_x = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=gpu_ctx_ptr[].norm_x_buf.cpu_addr)
            var norm_x_1 = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=gpu_ctx_ptr[].norm_x_buf_1.cpu_addr)
            var q = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=gpu_ctx_ptr[].q_buf.cpu_addr)
            var q_1 = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=gpu_ctx_ptr[].q_buf_1.cpu_addr)
            var k_vec = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=gpu_ctx_ptr[].k_vec_buf.cpu_addr)
            var k_vec_1 = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=gpu_ctx_ptr[].k_vec_buf_1.cpu_addr)
            var v_vec = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=gpu_ctx_ptr[].v_vec_buf.cpu_addr)
            var v_vec_1 = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=gpu_ctx_ptr[].v_vec_buf_1.cpu_addr)
            var attn_out = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=gpu_ctx_ptr[].attn_out_buf.cpu_addr)
            var attn_out_1 = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=gpu_ctx_ptr[].attn_out_buf_1.cpu_addr)
            var proj_attn_out = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=gpu_ctx_ptr[].proj_attn_out_buf.cpu_addr)
            var h1 = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=gpu_ctx_ptr[].h1_buf.cpu_addr)
            var h2 = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=gpu_ctx_ptr[].h2_buf.cpu_addr)

            var attn_scores_prefill = alloc[Float32](TARGET_CONFIG.num_heads * TARGET_CONFIG.max_seq_len)
            var ffn_out_gate = alloc[Float32](TARGET_CONFIG.intermediate_dim)
            var ffn_out = alloc[Float32](TARGET_CONFIG.embed_dim)

            for layer_idx in range(TARGET_CONFIG.num_layers):
                var active_weights_ptr = target_base_ptr + TARGET_CONFIG.get_layer_offset(layer_idx)
                var weights = TARGET_CONFIG.get_layer_weights(layer_idx, active_weights_ptr)
                var is_global = TARGET_CONFIG.is_global_layer(layer_idx)
                var layer_heads = 16
                var layer_kv_heads = 1 if is_global else 8
                var layer_head_dim = 512 if is_global else 256
                var kv_stride = layer_kv_heads * layer_head_dim
                var ratio = layer_heads // layer_kv_heads

                # Enqueue first position
                var active_slot_q = -1
                var active_slot_k = -1
                var active_slot_v = -1
                if num_prompt_tokens > 0:
                    var first_local_x = prompt_x + 0
                    rms_norm(first_local_x, weights.input_layernorm_weight, norm_x, TARGET_CONFIG.embed_dim)
                    active_slot_q = gpu_ctx_ptr[].enqueue_matmul(weights.wq_packed.__int__(), weights.wq_scales.__int__(), norm_x.__int__(), q.__int__(), TARGET_CONFIG.embed_dim, layer_heads * layer_head_dim)
                    active_slot_k = gpu_ctx_ptr[].enqueue_matmul(weights.wk_packed.__int__(), weights.wk_scales.__int__(), norm_x.__int__(), k_vec.__int__(), TARGET_CONFIG.embed_dim, layer_kv_heads * layer_head_dim)
                    if not is_global:
                        active_slot_v = gpu_ctx_ptr[].enqueue_matmul(weights.wv_packed.__int__(), weights.wv_scales.__int__(), norm_x.__int__(), v_vec.__int__(), TARGET_CONFIG.embed_dim, layer_kv_heads * layer_head_dim)

                for pos_idx in range(num_prompt_tokens):
                    var px_offset = pos_idx * TARGET_CONFIG.embed_dim
                    var local_x = prompt_x + px_offset
                    
                    var idx = pos_idx % 2
                    var cur_norm_x = norm_x if idx == 0 else norm_x_1
                    var cur_q = q if idx == 0 else q_1
                    var cur_k_vec = k_vec if idx == 0 else k_vec_1
                    var cur_v_vec = v_vec if idx == 0 else v_vec_1
                    var cur_attn_out = attn_out if idx == 0 else attn_out_1
                    
                    gpu_ctx_ptr[].wait_matmul(active_slot_q, cur_q.__int__(), layer_heads * layer_head_dim)
                    gpu_ctx_ptr[].wait_matmul(active_slot_k, cur_k_vec.__int__(), layer_kv_heads * layer_head_dim)
                    if is_global:
                        for i in range(kv_stride):
                            cur_v_vec[i] = cur_k_vec[i]
                    else:
                        gpu_ctx_ptr[].wait_matmul(active_slot_v, cur_v_vec.__int__(), layer_kv_heads * layer_head_dim)
                        
                    # Asynchronously pre-enqueue next position
                    var next_slot_q = -1
                    var next_slot_k = -1
                    var next_slot_v = -1
                    if pos_idx + 1 < num_prompt_tokens:
                        var next_pos = pos_idx + 1
                        var next_idx = next_pos % 2
                        var next_norm_x = norm_x if next_idx == 0 else norm_x_1
                        var next_q = q if next_idx == 0 else q_1
                        var next_k_vec = k_vec if next_idx == 0 else k_vec_1
                        var next_v_vec = v_vec if next_idx == 0 else v_vec_1
                        
                        var next_local_x = prompt_x + next_pos * TARGET_CONFIG.embed_dim
                        rms_norm(next_local_x, weights.input_layernorm_weight, next_norm_x, TARGET_CONFIG.embed_dim)
                        
                        next_slot_q = gpu_ctx_ptr[].enqueue_matmul(weights.wq_packed.__int__(), weights.wq_scales.__int__(), next_norm_x.__int__(), next_q.__int__(), TARGET_CONFIG.embed_dim, layer_heads * layer_head_dim)
                        next_slot_k = gpu_ctx_ptr[].enqueue_matmul(weights.wk_packed.__int__(), weights.wk_scales.__int__(), next_norm_x.__int__(), next_k_vec.__int__(), TARGET_CONFIG.embed_dim, layer_kv_heads * layer_head_dim)
                        if not is_global:
                            next_slot_v = gpu_ctx_ptr[].enqueue_matmul(weights.wv_packed.__int__(), weights.wv_scales.__int__(), next_norm_x.__int__(), next_v_vec.__int__(), TARGET_CONFIG.embed_dim, layer_kv_heads * layer_head_dim)

                    for h in range(layer_heads):
                        var head_ptr = cur_q + h * layer_head_dim
                        rms_norm(head_ptr, weights.q_norm_weight, head_ptr, layer_head_dim)
                    for h in range(layer_kv_heads):
                        var head_ptr = cur_k_vec + h * layer_head_dim
                        rms_norm(head_ptr, weights.k_norm_weight, head_ptr, layer_head_dim)
                    for h in range(layer_kv_heads):
                        var head_ptr = cur_v_vec + h * layer_head_dim
                        rms_norm_noscale(head_ptr, head_ptr, layer_head_dim)
                        
                    apply_rope(cur_q, layer_heads, layer_head_dim, pos_idx, is_global)
                    apply_rope(cur_k_vec, layer_kv_heads, layer_head_dim, pos_idx, is_global)
                    
                    var layer_cache_offset = layer_cache_offsets[layer_idx]
                    var step_cache_offset = layer_cache_offset + pos_idx * kv_stride
                    for i in range(kv_stride):
                        target_k_cache[step_cache_offset + i] = cur_k_vec[i]
                        target_v_cache[step_cache_offset + i] = cur_v_vec[i]
                        
                    for h in range(layer_heads):
                        var h_offset = h * layer_head_dim
                        var g = h // ratio
                        var g_offset = g * layer_head_dim
                        var head_scores = attn_scores_prefill + h * TARGET_CONFIG.max_seq_len
                        
                        for t in range(pos_idx + 1):
                            var t_cache_offset = layer_cache_offset + t * kv_stride + g_offset
                            var score_vec = SIMD[DType.float32, 16](0.0)
                            for d in range(0, layer_head_dim, 16):
                                score_vec += cur_q.load[16](h_offset + d) * target_k_cache.load[16](t_cache_offset + d)
                            head_scores[t] = score_vec.reduce_add()
                            
                        softmax(head_scores, pos_idx + 1)
                        
                        for d in range(0, layer_head_dim, 16):
                            cur_attn_out.store[16](h_offset + d, SIMD[DType.float32, 16](0.0))
                            
                        for t in range(pos_idx + 1):
                            var t_cache_offset = layer_cache_offset + t * kv_stride + g_offset
                            var s_vec = SIMD[DType.float32, 16](head_scores[t])
                            for d in range(0, layer_head_dim, 16):
                                var v = target_v_cache.load[16](t_cache_offset + d)
                                var acc = cur_attn_out.load[16](h_offset + d)
                                cur_attn_out.store[16](h_offset + d, acc + s_vec * v)
                            
                    gpu_ctx_ptr[].launch_matmul(weights.wo_packed.__int__(), weights.wo_scales.__int__(), cur_attn_out.__int__(), proj_attn_out.__int__(), layer_heads * layer_head_dim, TARGET_CONFIG.embed_dim)
                    
                    rms_norm(proj_attn_out, weights.post_attention_layernorm_weight, proj_attn_out, TARGET_CONFIG.embed_dim)
                    for i in range(TARGET_CONFIG.embed_dim):
                        local_x[i] += proj_attn_out[i]
                        
                    rms_norm(local_x, weights.pre_feedforward_layernorm_weight, cur_norm_x, TARGET_CONFIG.embed_dim)
                    
                    var slot_w1 = gpu_ctx_ptr[].enqueue_matmul(weights.w1_packed.__int__(), weights.w1_scales.__int__(), cur_norm_x.__int__(), h1.__int__(), TARGET_CONFIG.embed_dim, TARGET_CONFIG.intermediate_dim)
                    var slot_w3 = gpu_ctx_ptr[].enqueue_matmul(weights.w3_packed.__int__(), weights.w3_scales.__int__(), cur_norm_x.__int__(), h2.__int__(), TARGET_CONFIG.embed_dim, TARGET_CONFIG.intermediate_dim)
                    
                    gpu_ctx_ptr[].wait_matmul(slot_w1, h1.__int__(), TARGET_CONFIG.intermediate_dim)
                    gpu_ctx_ptr[].wait_matmul(slot_w3, h2.__int__(), TARGET_CONFIG.intermediate_dim)
                    
                    gelu_and_mul(h1, h2, ffn_out_gate, TARGET_CONFIG.intermediate_dim)
                    
                    q8_matmul_w2_block_column(weights.w2_packed, ffn_out_gate, ffn_out, TARGET_CONFIG.intermediate_dim, TARGET_CONFIG.embed_dim, num_tasks, w2_threshold, use_scratch=True, w2_scratch=w2_scratch)
                    
                    rms_norm(ffn_out, weights.post_feedforward_layernorm_weight, ffn_out, TARGET_CONFIG.embed_dim)
                    for i in range(TARGET_CONFIG.embed_dim):
                        local_x[i] += ffn_out[i]
                        
                    var scalar = weights.layer_scalar[0]
                    for i in range(TARGET_CONFIG.embed_dim):
                        local_x[i] *= scalar
                        
                    active_slot_q = next_slot_q
                    active_slot_k = next_slot_k
                    active_slot_v = next_slot_v

            # Retrieve final prefill hidden state of prompt
            var last_px_offset = (num_prompt_tokens - 1) * TARGET_CONFIG.embed_dim
            var target_final_x = alloc[Float32](TARGET_CONFIG.embed_dim)
            for i in range(TARGET_CONFIG.embed_dim):
                target_final_x[i] = prompt_x[last_px_offset + i]
            prompt_x.free()
            attn_scores_prefill.free()
            ffn_out_gate.free()
            ffn_out.free()

            rms_norm(target_final_x, final_norm_weight, target_final_norm_x, TARGET_CONFIG.embed_dim)
            target_final_x.free()

            # Q4 high-dimensional parallel output projection for target prefill output
            embed.project_logits_parallel(target_final_norm_x, local_max_logits, local_next_tokens, num_tasks)
            var max_logit: Float32 = -999999.0
            var next_token = 0
            for t in range(num_tasks):
                if local_max_logits[t] > max_logit:
                    max_logit = local_max_logits[t]
                    next_token = local_next_tokens[t]

            # Print the first generated token
            var next_token_list = list_cls()
            _ = next_token_list.append(next_token)
            var decoded = tokenizer.decode(next_token_list)
            py_builtins.print(decoded, end="", flush=True)
            
            current_token = next_token
            pos = num_prompt_tokens
        else:
            var target_prompt_x = alloc[Float32](num_prompt_tokens * TARGET_CONFIG.embed_dim)
            var target_prompt_ptr = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=target_prompt_x.__int__())
            for pos_idx in range(num_prompt_tokens):
                var px_offset = pos_idx * TARGET_CONFIG.embed_dim
                embed.reconstruct(prompt_tokens[pos_idx], target_prompt_ptr + px_offset)
                for i in range(TARGET_CONFIG.embed_dim):
                    target_prompt_x[px_offset + i] *= scale
                    
            # Load Layer 0 target weights (stable base) if paged
            if resident == "paged":
                var tag_0: UInt64 = UInt64(8000)
                _ = ring.submit_read(weights_fd, w_a_ptr.__int__(), TARGET_CONFIG.get_w2_offset_in_layer(0), 0, tag_0)
                wait_for_tag(ring, tag_0, out_tag_ptr)

            for layer_idx in range(TARGET_CONFIG.num_layers):
                var active_weights_ptr: UnsafePointer[Int8, MutExternalOrigin]
                var next_layer_completed = False
                
                if resident == "ram":
                    active_weights_ptr = target_base_ptr + TARGET_CONFIG.get_layer_offset(layer_idx)
                else: # paged
                    active_weights_ptr = w_a_ptr if layer_idx % 2 == 0 else w_b_ptr
                    var next_weights_ptr = w_b_ptr if layer_idx % 2 == 0 else w_a_ptr
                    
                    # Prefetch next layer
                    if layer_idx + 1 < TARGET_CONFIG.num_layers:
                        var tag_next: UInt64 = UInt64(8000 + layer_idx + 1)
                        var offset_next = TARGET_CONFIG.get_layer_offset(layer_idx + 1)
                        var size_next = TARGET_CONFIG.get_w2_offset_in_layer(layer_idx + 1)
                        _ = ring.submit_read(weights_fd, next_weights_ptr.__int__(), size_next, offset_next, tag_next)
                    
                var weights = TARGET_CONFIG.get_layer_weights(layer_idx, active_weights_ptr)
                var is_global = TARGET_CONFIG.is_global_layer(layer_idx)
                var layer_heads = 16
                var layer_kv_heads = 1 if is_global else 8
                var layer_head_dim = 512 if is_global else 256
                var kv_stride = layer_kv_heads * layer_head_dim
                var ratio = layer_heads // layer_kv_heads
                
                for pos_idx in range(num_prompt_tokens):
                    var px_offset = pos_idx * TARGET_CONFIG.embed_dim
                    var local_x = target_prompt_x + px_offset
                    
                    rms_norm(local_x, weights.input_layernorm_weight, target_norm_x, TARGET_CONFIG.embed_dim)
                    
                    q8_matmul_vec(weights.wq_packed, weights.wq_scales, target_norm_x, target_q, TARGET_CONFIG.embed_dim, layer_heads * layer_head_dim, num_tasks)
                    q8_matmul_vec(weights.wk_packed, weights.wk_scales, target_norm_x, target_k_vec, TARGET_CONFIG.embed_dim, layer_kv_heads * layer_head_dim, num_tasks)
                    
                    if is_global:
                        for i in range(kv_stride):
                            target_v_vec[i] = target_k_vec[i]
                    else:
                        q8_matmul_vec(weights.wv_packed, weights.wv_scales, target_norm_x, target_v_vec, TARGET_CONFIG.embed_dim, layer_kv_heads * layer_head_dim, num_tasks)
                        
                    for h in range(layer_heads):
                        var head_ptr = target_q + h * layer_head_dim
                        rms_norm(head_ptr, weights.q_norm_weight, head_ptr, layer_head_dim)
                    for h in range(layer_kv_heads):
                        var head_ptr = target_k_vec + h * layer_head_dim
                        rms_norm(head_ptr, weights.k_norm_weight, head_ptr, layer_head_dim)
                    for h in range(layer_kv_heads):
                        var head_ptr = target_v_vec + h * layer_head_dim
                        rms_norm_noscale(head_ptr, head_ptr, layer_head_dim)
                        
                    apply_rope(target_q, layer_heads, layer_head_dim, pos_idx, is_global)
                    apply_rope(target_k_vec, layer_kv_heads, layer_head_dim, pos_idx, is_global)
                    
                    var layer_cache_offset = layer_cache_offsets[layer_idx]
                    var step_cache_offset = layer_cache_offset + pos_idx * kv_stride
                    for i in range(kv_stride):
                        target_k_cache[step_cache_offset + i] = target_k_vec[i]
                        target_v_cache[step_cache_offset + i] = target_v_vec[i]
                        
                    for h in range(layer_heads):
                        var h_offset = h * layer_head_dim
                        var g = h // ratio
                        var g_offset = g * layer_head_dim
                        
                        for t in range(pos_idx + 1):
                            var t_cache_offset = layer_cache_offset + t * kv_stride + g_offset
                            var score: Float32 = 0.0
                            for d in range(layer_head_dim):
                                score += target_q[h_offset + d] * target_k_cache[t_cache_offset + d]
                            target_attn_scores[t] = score
                            
                        softmax(target_attn_scores, pos_idx + 1)
                        
                        for d in range(layer_head_dim):
                            var val_sum: Float32 = 0.0
                            for t in range(pos_idx + 1):
                                var t_cache_offset = layer_cache_offset + t * kv_stride + g_offset
                                val_sum += target_attn_scores[t] * target_v_cache[t_cache_offset + d]
                            target_attn_out[h_offset + d] = val_sum
                            
                    q8_matmul_vec(weights.wo_packed, weights.wo_scales, target_attn_out, target_proj_attn_out, layer_heads * layer_head_dim, TARGET_CONFIG.embed_dim, num_tasks)
                    
                    rms_norm(target_proj_attn_out, weights.post_attention_layernorm_weight, target_proj_attn_out, TARGET_CONFIG.embed_dim)
                    for i in range(TARGET_CONFIG.embed_dim):
                        local_x[i] += target_proj_attn_out[i]
                        
                    rms_norm(local_x, weights.pre_feedforward_layernorm_weight, target_norm_x, TARGET_CONFIG.embed_dim)
                    q8_matmul_vec(weights.w1_packed, weights.w1_scales, target_norm_x, target_h1, TARGET_CONFIG.embed_dim, TARGET_CONFIG.intermediate_dim, num_tasks)
                    q8_matmul_vec(weights.w3_packed, weights.w3_scales, target_norm_x, target_h2, TARGET_CONFIG.embed_dim, TARGET_CONFIG.intermediate_dim, num_tasks)
                    gelu_and_mul(target_h1, target_h2, target_ffn_out_gate, TARGET_CONFIG.intermediate_dim)
                    
                    # W2 sparse loading
                    var num_blocks = TARGET_CONFIG.intermediate_dim // 32
                    var layer_offset = TARGET_CONFIG.get_layer_offset(layer_idx)
                    var w2_offset_in_layer = TARGET_CONFIG.get_w2_offset_in_layer(layer_idx)
                    var tag_base_for_w2: UInt64 = 100000 + UInt64(layer_idx) * 1000 + UInt64(pos_idx) * 1000000
                    
                    var max_active_block: Int = -1
                    if w2_threshold == 0.0:
                        max_active_block = num_blocks - 1
                    else:
                        for b in range(num_blocks - 1, -1, -1):
                            var block_offset_gate = b * 32
                            var block_active = False
                            for c in range(32):
                                if abs(target_ffn_out_gate[block_offset_gate + c]) > w2_threshold:
                                    block_active = True
                                    break
                            if block_active:
                                max_active_block = b
                                break
                            
                    var num_active = 0
                    if resident == "paged":
                        if max_active_block >= 0:
                            num_active = 1
                            var size_to_read = (max_active_block + 1) * 139264
                            var block_offset_in_file = layer_offset + w2_offset_in_layer
                            var dest_mem_addr = weights.w2_packed.__int__()
                            _ = ring.submit_read(weights_fd, dest_mem_addr, size_to_read, block_offset_in_file, tag_base_for_w2)
                            
                        var completed_active = 0
                        var attempts = 0
                        var tag_next_gen: UInt64 = UInt64(8000 + layer_idx + 1)
                        
                        while completed_active < num_active:
                             var res_ring = ring.poll_completion(out_tag_ptr.__int__())
                             if res_ring >= 0:
                                 var t = out_tag_ptr[0]
                                 if t == tag_base_for_w2:
                                     completed_active += 1
                                 elif t == tag_next_gen:
                                     next_layer_completed = True
                             attempts += 1
                             if attempts > 1000000000:
                                 raise Error("Prefill weights paging I/O timed out waiting for sparse W2 blocks")
                                 
                    q8_matmul_w2_block_column(weights.w2_packed, target_ffn_out_gate, target_ffn_out, TARGET_CONFIG.intermediate_dim, TARGET_CONFIG.embed_dim, num_tasks, w2_threshold)
                    
                    rms_norm(target_ffn_out, weights.post_feedforward_layernorm_weight, target_ffn_out, TARGET_CONFIG.embed_dim)
                    for i in range(TARGET_CONFIG.embed_dim):
                        local_x[i] += target_ffn_out[i]
                        
                    var scalar = weights.layer_scalar[0]
                    for i in range(TARGET_CONFIG.embed_dim):
                        local_x[i] *= scalar

                if layer_idx + 1 < TARGET_CONFIG.num_layers:
                    if resident == "paged" and not next_layer_completed:
                        var tag_next_val: UInt64 = UInt64(8000 + layer_idx + 1)
                        wait_for_tag(ring, tag_next_val, out_tag_ptr)

            # Retrieve final prefill hidden state of prompt
            var last_px_offset = (num_prompt_tokens - 1) * TARGET_CONFIG.embed_dim
            var target_final_x = alloc[Float32](TARGET_CONFIG.embed_dim)
            for i in range(TARGET_CONFIG.embed_dim):
                target_final_x[i] = target_prompt_x[last_px_offset + i]
            target_prompt_x.free()
            
            rms_norm(target_final_x, final_norm_weight, target_final_norm_x, TARGET_CONFIG.embed_dim)
            target_final_x.free()

            # Q4 high-dimensional parallel output projection for target prefill output
            embed.project_logits_parallel(target_final_norm_x, local_max_logits, local_next_tokens, num_tasks)
            var max_logit: Float32 = -999999.0
            var next_token = 0
            for t in range(num_tasks):
                if local_max_logits[t] > max_logit:
                    max_logit = local_max_logits[t]
                    next_token = local_next_tokens[t]

            # Print the first generated token
            var next_token_list = list_cls()
            _ = next_token_list.append(next_token)
            var decoded = tokenizer.decode(next_token_list)
            py_builtins.print(decoded, end="", flush=True)
            
            current_token = next_token
            pos = num_prompt_tokens
        
        var t_prefill_end = py.time()
        var prefill_time = Float64(py=(t_prefill_end - t_prefill_start))
        print("\n[Mojo Bench] Prefill completed in", prefill_time, "sec (" + String(Float64(num_prompt_tokens) / prefill_time) + " t/s)")
        
        # 3. Speculative Generation Phase
        var tokens_generated = 1
        var t_gen_start = py.time()
        
        var current_K = 4
        var total_accepted = 0
        var total_speculative_steps = 0
        
        while pos < max_steps:
            var K_actual = current_K
            if mode == "single":
                K_actual = 0
            else:
                if pos + K_actual >= max_steps:
                    K_actual = max_steps - pos - 1
                if K_actual <= 0:
                    break
                
            # A. Draft Model Speculative Pass: generate K_actual candidate tokens sequential in memory
            var draft_tokens = List[Int]()
            draft_tokens.append(current_token)
            
            if mode == "speculative":
                for j in range(K_actual):
                    var curr = draft_tokens[j]
                    embed.reconstruct(curr, draft_x)
                    for i in range(DRAFT_CONFIG.embed_dim):
                        draft_x[i] *= scale
                        
                    # Run Draft 48 layers in memory
                    for layer_idx in range(DRAFT_CONFIG.num_layers):
                        var weights = DRAFT_CONFIG.get_layer_weights(layer_idx, draft_base_ptr + DRAFT_CONFIG.get_layer_offset(layer_idx))
                        var is_global = DRAFT_CONFIG.is_global_layer(layer_idx)
                        var layer_heads = 16
                        var layer_kv_heads = 1 if is_global else 8
                        var layer_head_dim = 512 if is_global else 256
                        var kv_stride = layer_kv_heads * layer_head_dim
                        var ratio = layer_heads // layer_kv_heads
                        
                        rms_norm(draft_x, weights.input_layernorm_weight, draft_norm_x, DRAFT_CONFIG.embed_dim)
                        
                        q4_matmul_vec(weights.wq_packed, weights.wq_scales, draft_norm_x, draft_q, DRAFT_CONFIG.embed_dim, layer_heads * layer_head_dim, num_tasks)
                        q4_matmul_vec(weights.wk_packed, weights.wk_scales, draft_norm_x, draft_k_vec, DRAFT_CONFIG.embed_dim, layer_kv_heads * layer_head_dim, num_tasks)
                        
                        if is_global:
                            for i in range(kv_stride):
                                draft_v_vec[i] = draft_k_vec[i]
                        else:
                            q4_matmul_vec(weights.wv_packed, weights.wv_scales, draft_norm_x, draft_v_vec, DRAFT_CONFIG.embed_dim, layer_kv_heads * layer_head_dim, num_tasks)
                            
                        for h in range(layer_heads):
                            var head_ptr = draft_q + h * layer_head_dim
                            rms_norm(head_ptr, weights.q_norm_weight, head_ptr, layer_head_dim)
                        for h in range(layer_kv_heads):
                            var head_ptr = draft_k_vec + h * layer_head_dim
                            rms_norm(head_ptr, weights.k_norm_weight, head_ptr, layer_head_dim)
                        for h in range(layer_kv_heads):
                            var head_ptr = draft_v_vec + h * layer_head_dim
                            rms_norm_noscale(head_ptr, head_ptr, layer_head_dim)
                            
                        apply_rope(draft_q, layer_heads, layer_head_dim, pos + j, is_global)
                        apply_rope(draft_k_vec, layer_kv_heads, layer_head_dim, pos + j, is_global)
                        
                        var layer_cache_offset = layer_cache_offsets[layer_idx]
                        var step_cache_offset = layer_cache_offset + (pos + j) * kv_stride
                        for i in range(kv_stride):
                            draft_k_cache[step_cache_offset + i] = draft_k_vec[i]
                            draft_v_cache[step_cache_offset + i] = draft_v_vec[i]
                            
                        for h in range(layer_heads):
                            var h_offset = h * layer_head_dim
                            var g = h // ratio
                            var g_offset = g * layer_head_dim
                            for t in range(pos + j + 1):
                                var t_cache_offset = layer_cache_offset + t * kv_stride + g_offset
                                var score: Float32 = 0.0
                                for d in range(layer_head_dim):
                                    score += draft_q[h_offset + d] * draft_k_cache[t_cache_offset + d]
                                draft_attn_scores[t] = score
                                
                            softmax(draft_attn_scores, pos + j + 1)
                            
                            for d in range(layer_head_dim):
                                var val_sum: Float32 = 0.0
                                for t in range(pos + j + 1):
                                    var t_cache_offset = layer_cache_offset + t * kv_stride + g_offset
                                    val_sum += draft_attn_scores[t] * draft_v_cache[t_cache_offset + d]
                                draft_attn_out[h_offset + d] = val_sum
                                
                        q4_matmul_vec(weights.wo_packed, weights.wo_scales, draft_attn_out, draft_proj_attn_out, layer_heads * layer_head_dim, DRAFT_CONFIG.embed_dim, num_tasks)
                        
                        rms_norm(draft_proj_attn_out, weights.post_attention_layernorm_weight, draft_proj_attn_out, DRAFT_CONFIG.embed_dim)
                        for i in range(DRAFT_CONFIG.embed_dim):
                            draft_x[i] += draft_proj_attn_out[i]
                            
                        rms_norm(draft_x, weights.pre_feedforward_layernorm_weight, draft_norm_x, DRAFT_CONFIG.embed_dim)
                        q4_matmul_vec(weights.w1_packed, weights.w1_scales, draft_norm_x, draft_h1, DRAFT_CONFIG.embed_dim, DRAFT_CONFIG.intermediate_dim, num_tasks)
                        q4_matmul_vec(weights.w3_packed, weights.w3_scales, draft_norm_x, draft_h2, DRAFT_CONFIG.embed_dim, DRAFT_CONFIG.intermediate_dim, num_tasks)
                        gelu_and_mul(draft_h1, draft_h2, draft_ffn_out_gate, DRAFT_CONFIG.intermediate_dim)
                        
                        q4_matmul_vec(weights.w2_packed, weights.w2_scales, draft_ffn_out_gate, draft_ffn_out, DRAFT_CONFIG.intermediate_dim, DRAFT_CONFIG.embed_dim, num_tasks)
                        
                        rms_norm(draft_ffn_out, weights.post_feedforward_layernorm_weight, draft_ffn_out, DRAFT_CONFIG.embed_dim)
                        for i in range(DRAFT_CONFIG.embed_dim):
                            draft_x[i] += draft_ffn_out[i]
                            
                        var scalar = weights.layer_scalar[0]
                        for i in range(DRAFT_CONFIG.embed_dim):
                            draft_x[i] *= scalar

                    rms_norm(draft_x, final_norm_weight, draft_norm_x, DRAFT_CONFIG.embed_dim)
                    
                    # Q4 parallel logits projection for draft step
                    embed.project_logits_parallel(draft_norm_x, local_max_logits, local_next_tokens, num_tasks)
                    var d_max_logit: Float32 = -999999.0
                    var d_next_token = 0
                    for t in range(num_tasks):
                        if local_max_logits[t] > d_max_logit:
                            d_max_logit = local_max_logits[t]
                            d_next_token = local_next_tokens[t]
                    draft_tokens.append(d_next_token)

            # B. Target Model Parallel Pass: stream target weights ONCE and validate K_actual tokens
            for s in range(K_actual + 1):
                var t_seq_offset = s * TARGET_CONFIG.embed_dim
                embed.reconstruct(draft_tokens[s], target_x_seq + t_seq_offset)
                for i in range(TARGET_CONFIG.embed_dim):
                    target_x_seq[t_seq_offset + i] *= scale

            if resident == "gpu":
                # Setup pointers to GPU memory
                var norm_x = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=gpu_ctx_ptr[].norm_x_buf.cpu_addr)
                var norm_x_1 = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=gpu_ctx_ptr[].norm_x_buf_1.cpu_addr)
                var q = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=gpu_ctx_ptr[].q_buf.cpu_addr)
                var q_1 = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=gpu_ctx_ptr[].q_buf_1.cpu_addr)
                var k_vec = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=gpu_ctx_ptr[].k_vec_buf.cpu_addr)
                var k_vec_1 = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=gpu_ctx_ptr[].k_vec_buf_1.cpu_addr)
                var v_vec = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=gpu_ctx_ptr[].v_vec_buf.cpu_addr)
                var v_vec_1 = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=gpu_ctx_ptr[].v_vec_buf_1.cpu_addr)
                var attn_out = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=gpu_ctx_ptr[].attn_out_buf.cpu_addr)
                var attn_out_1 = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=gpu_ctx_ptr[].attn_out_buf_1.cpu_addr)
                var proj_attn_out = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=gpu_ctx_ptr[].proj_attn_out_buf.cpu_addr)
                var h1 = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=gpu_ctx_ptr[].h1_buf.cpu_addr)
                var h2 = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=gpu_ctx_ptr[].h2_buf.cpu_addr)

                var attn_scores_val = alloc[Float32](TARGET_CONFIG.num_heads * TARGET_CONFIG.max_seq_len)
                var ffn_out_gate = alloc[Float32](TARGET_CONFIG.intermediate_dim)
                var ffn_out = alloc[Float32](TARGET_CONFIG.embed_dim)

                for layer_idx in range(TARGET_CONFIG.num_layers):
                    var active_weights_ptr = target_base_ptr + TARGET_CONFIG.get_layer_offset(layer_idx)
                    var weights = TARGET_CONFIG.get_layer_weights(layer_idx, active_weights_ptr)
                    var is_global = TARGET_CONFIG.is_global_layer(layer_idx)
                    var layer_heads = 16
                    var layer_kv_heads = 1 if is_global else 8
                    var layer_head_dim = 512 if is_global else 256
                    var kv_stride = layer_kv_heads * layer_head_dim
                    var ratio = layer_heads // layer_kv_heads

                    # Enqueue first step
                    var active_slot_q = -1
                    var active_slot_k = -1
                    var active_slot_v = -1
                    
                    var first_local_x = target_x_seq + 0
                    rms_norm(first_local_x, weights.input_layernorm_weight, norm_x, TARGET_CONFIG.embed_dim)
                    active_slot_q = gpu_ctx_ptr[].enqueue_matmul(weights.wq_packed.__int__(), weights.wq_scales.__int__(), norm_x.__int__(), q.__int__(), TARGET_CONFIG.embed_dim, layer_heads * layer_head_dim)
                    active_slot_k = gpu_ctx_ptr[].enqueue_matmul(weights.wk_packed.__int__(), weights.wk_scales.__int__(), norm_x.__int__(), k_vec.__int__(), TARGET_CONFIG.embed_dim, layer_kv_heads * layer_head_dim)
                    if not is_global:
                        active_slot_v = gpu_ctx_ptr[].enqueue_matmul(weights.wv_packed.__int__(), weights.wv_scales.__int__(), norm_x.__int__(), v_vec.__int__(), TARGET_CONFIG.embed_dim, layer_kv_heads * layer_head_dim)

                    for s in range(K_actual + 1):
                        var local_x = target_x_seq + s * TARGET_CONFIG.embed_dim
                        
                        var idx = s % 2
                        var cur_norm_x = norm_x if idx == 0 else norm_x_1
                        var cur_q = q if idx == 0 else q_1
                        var cur_k_vec = k_vec if idx == 0 else k_vec_1
                        var cur_v_vec = v_vec if idx == 0 else v_vec_1
                        var cur_attn_out = attn_out if idx == 0 else attn_out_1
                        
                        gpu_ctx_ptr[].wait_matmul(active_slot_q, cur_q.__int__(), layer_heads * layer_head_dim)
                        gpu_ctx_ptr[].wait_matmul(active_slot_k, cur_k_vec.__int__(), layer_kv_heads * layer_head_dim)
                        if is_global:
                            for i in range(kv_stride):
                                cur_v_vec[i] = cur_k_vec[i]
                        else:
                            gpu_ctx_ptr[].wait_matmul(active_slot_v, cur_v_vec.__int__(), layer_kv_heads * layer_head_dim)
                            
                        # Asynchronously pre-enqueue next step
                        var next_slot_q = -1
                        var next_slot_k = -1
                        var next_slot_v = -1
                        if s + 1 < K_actual + 1:
                            var next_s = s + 1
                            var next_idx = next_s % 2
                            var next_norm_x = norm_x if next_idx == 0 else norm_x_1
                            var next_q = q if next_idx == 0 else q_1
                            var next_k_vec = k_vec if next_idx == 0 else k_vec_1
                            var next_v_vec = v_vec if next_idx == 0 else v_vec_1
                            
                            var next_local_x = target_x_seq + next_s * TARGET_CONFIG.embed_dim
                            rms_norm(next_local_x, weights.input_layernorm_weight, next_norm_x, TARGET_CONFIG.embed_dim)
                            
                            next_slot_q = gpu_ctx_ptr[].enqueue_matmul(weights.wq_packed.__int__(), weights.wq_scales.__int__(), next_norm_x.__int__(), next_q.__int__(), TARGET_CONFIG.embed_dim, layer_heads * layer_head_dim)
                            next_slot_k = gpu_ctx_ptr[].enqueue_matmul(weights.wk_packed.__int__(), weights.wk_scales.__int__(), next_norm_x.__int__(), next_k_vec.__int__(), TARGET_CONFIG.embed_dim, layer_kv_heads * layer_head_dim)
                            if not is_global:
                                next_slot_v = gpu_ctx_ptr[].enqueue_matmul(weights.wv_packed.__int__(), weights.wv_scales.__int__(), next_norm_x.__int__(), next_v_vec.__int__(), TARGET_CONFIG.embed_dim, layer_kv_heads * layer_head_dim)

                        for h in range(layer_heads):
                            var head_ptr = cur_q + h * layer_head_dim
                            rms_norm(head_ptr, weights.q_norm_weight, head_ptr, layer_head_dim)
                        for h in range(layer_kv_heads):
                            var head_ptr = cur_k_vec + h * layer_head_dim
                            rms_norm(head_ptr, weights.k_norm_weight, head_ptr, layer_head_dim)
                        for h in range(layer_kv_heads):
                            var head_ptr = cur_v_vec + h * layer_head_dim
                            rms_norm_noscale(head_ptr, head_ptr, layer_head_dim)
                            
                        apply_rope(cur_q, layer_heads, layer_head_dim, pos + s, is_global)
                        apply_rope(cur_k_vec, layer_kv_heads, layer_head_dim, pos + s, is_global)
                        
                        var layer_cache_offset = layer_cache_offsets[layer_idx]
                        var step_cache_offset = layer_cache_offset + (pos + s) * kv_stride
                        for i in range(kv_stride):
                            target_k_cache[step_cache_offset + i] = cur_k_vec[i]
                            target_v_cache[step_cache_offset + i] = cur_v_vec[i]
                            
                        for h in range(layer_heads):
                            var h_offset = h * layer_head_dim
                            var g = h // ratio
                            var g_offset = g * layer_head_dim
                            var head_scores = attn_scores_val + h * TARGET_CONFIG.max_seq_len
                            
                            for t in range(pos + s + 1):
                                var t_cache_offset = layer_cache_offset + t * kv_stride + g_offset
                                var score_vec = SIMD[DType.float32, 16](0.0)
                                for d in range(0, layer_head_dim, 16):
                                    score_vec += cur_q.load[16](h_offset + d) * target_k_cache.load[16](t_cache_offset + d)
                                head_scores[t] = score_vec.reduce_add()
                                
                            softmax(head_scores, pos + s + 1)
                            
                            for d in range(0, layer_head_dim, 16):
                                cur_attn_out.store[16](h_offset + d, SIMD[DType.float32, 16](0.0))
                                
                            for t in range(pos + s + 1):
                                var t_cache_offset = layer_cache_offset + t * kv_stride + g_offset
                                var s_vec = SIMD[DType.float32, 16](head_scores[t])
                                for d in range(0, layer_head_dim, 16):
                                    var v = target_v_cache.load[16](t_cache_offset + d)
                                    var acc = cur_attn_out.load[16](h_offset + d)
                                    cur_attn_out.store[16](h_offset + d, acc + s_vec * v)
                                
                        gpu_ctx_ptr[].launch_matmul(weights.wo_packed.__int__(), weights.wo_scales.__int__(), cur_attn_out.__int__(), proj_attn_out.__int__(), layer_heads * layer_head_dim, TARGET_CONFIG.embed_dim)
                        
                        rms_norm(proj_attn_out, weights.post_attention_layernorm_weight, proj_attn_out, TARGET_CONFIG.embed_dim)
                        for i in range(TARGET_CONFIG.embed_dim):
                            local_x[i] += proj_attn_out[i]
                            
                        rms_norm(local_x, weights.pre_feedforward_layernorm_weight, cur_norm_x, TARGET_CONFIG.embed_dim)
                        
                        var slot_w1 = gpu_ctx_ptr[].enqueue_matmul(weights.w1_packed.__int__(), weights.w1_scales.__int__(), cur_norm_x.__int__(), h1.__int__(), TARGET_CONFIG.embed_dim, TARGET_CONFIG.intermediate_dim)
                        var slot_w3 = gpu_ctx_ptr[].enqueue_matmul(weights.w3_packed.__int__(), weights.w3_scales.__int__(), cur_norm_x.__int__(), h2.__int__(), TARGET_CONFIG.embed_dim, TARGET_CONFIG.intermediate_dim)
                        
                        gpu_ctx_ptr[].wait_matmul(slot_w1, h1.__int__(), TARGET_CONFIG.intermediate_dim)
                        gpu_ctx_ptr[].wait_matmul(slot_w3, h2.__int__(), TARGET_CONFIG.intermediate_dim)
                        
                        gelu_and_mul(h1, h2, ffn_out_gate, TARGET_CONFIG.intermediate_dim)
                        
                        q8_matmul_w2_block_column(weights.w2_packed, ffn_out_gate, ffn_out, TARGET_CONFIG.intermediate_dim, TARGET_CONFIG.embed_dim, num_tasks, w2_threshold, use_scratch=True, w2_scratch=w2_scratch)
                        
                        rms_norm(ffn_out, weights.post_feedforward_layernorm_weight, ffn_out, TARGET_CONFIG.embed_dim)
                        for i in range(TARGET_CONFIG.embed_dim):
                            local_x[i] += ffn_out[i]
                            
                        var scalar = weights.layer_scalar[0]
                        for i in range(TARGET_CONFIG.embed_dim):
                            local_x[i] *= scalar
                            
                        active_slot_q = next_slot_q
                        active_slot_k = next_slot_k
                        active_slot_v = next_slot_v

                attn_scores_val.free()
                ffn_out_gate.free()
                ffn_out.free()
            else:
                if resident == "paged":
                    # Load Layer 0 of target model before the loop
                    var tag_0_val: UInt64 = UInt64(20000000 + pos * 10)
                    _ = ring.submit_read(weights_fd, w_a_ptr.__int__(), TARGET_CONFIG.get_w2_offset_in_layer(0), 0, tag_0_val)
                    wait_for_tag(ring, tag_0_val, out_tag_ptr)

                for layer_idx in range(TARGET_CONFIG.num_layers):
                    var active_weights_ptr: UnsafePointer[Int8, MutExternalOrigin]
                    if resident == "ram":
                        active_weights_ptr = target_base_ptr + TARGET_CONFIG.get_layer_offset(layer_idx)
                    else: # paged
                        active_weights_ptr = w_a_ptr if layer_idx % 2 == 0 else w_b_ptr
                        var next_weights_ptr = w_b_ptr if layer_idx % 2 == 0 else w_a_ptr
                        
                        # Prefetch next layer's weights
                        if layer_idx + 1 < TARGET_CONFIG.num_layers:
                            var tag_next: UInt64 = UInt64(30000000 + pos * 100 + layer_idx + 1)
                            var offset_next = TARGET_CONFIG.get_layer_offset(layer_idx + 1)
                            var size_next = TARGET_CONFIG.get_w2_offset_in_layer(layer_idx + 1)
                            _ = ring.submit_read(weights_fd, next_weights_ptr.__int__(), size_next, offset_next, tag_next)
                        
                    var weights = TARGET_CONFIG.get_layer_weights(layer_idx, active_weights_ptr)
                    var is_global = TARGET_CONFIG.is_global_layer(layer_idx)
                    var layer_heads = 16
                    var layer_kv_heads = 1 if is_global else 8
                    var layer_head_dim = 512 if is_global else 256
                    var kv_stride = layer_kv_heads * layer_head_dim
                    var ratio = layer_heads // layer_kv_heads
                    
                    var next_layer_completed = False
                    
                    # Execute sequentially across our K_actual+1 speculative batch steps for this layer
                    for s in range(K_actual + 1):
                        var local_x = target_x_seq + s * TARGET_CONFIG.embed_dim
                        
                        rms_norm(local_x, weights.input_layernorm_weight, target_norm_x, TARGET_CONFIG.embed_dim)
                        
                        q8_matmul_vec(weights.wq_packed, weights.wq_scales, target_norm_x, target_q, TARGET_CONFIG.embed_dim, layer_heads * layer_head_dim, num_tasks)
                        q8_matmul_vec(weights.wk_packed, weights.wk_scales, target_norm_x, target_k_vec, TARGET_CONFIG.embed_dim, layer_kv_heads * layer_head_dim, num_tasks)
                        
                        if is_global:
                            for i in range(kv_stride):
                                target_v_vec[i] = target_k_vec[i]
                        else:
                            q8_matmul_vec(weights.wv_packed, weights.wv_scales, target_norm_x, target_v_vec, TARGET_CONFIG.embed_dim, layer_kv_heads * layer_head_dim, num_tasks)
                            
                        for h in range(layer_heads):
                            var head_ptr = target_q + h * layer_head_dim
                            rms_norm(head_ptr, weights.q_norm_weight, head_ptr, layer_head_dim)
                        for h in range(layer_kv_heads):
                            var head_ptr = target_k_vec + h * layer_head_dim
                            rms_norm(head_ptr, weights.k_norm_weight, head_ptr, layer_head_dim)
                        for h in range(layer_kv_heads):
                            var head_ptr = target_v_vec + h * layer_head_dim
                            rms_norm_noscale(head_ptr, head_ptr, layer_head_dim)
                            
                        apply_rope(target_q, layer_heads, layer_head_dim, pos + s, is_global)
                        apply_rope(target_k_vec, layer_kv_heads, layer_head_dim, pos + s, is_global)
                        
                        var layer_cache_offset = layer_cache_offsets[layer_idx]
                        var step_cache_offset = layer_cache_offset + (pos + s) * kv_stride
                        for i in range(kv_stride):
                            target_k_cache[step_cache_offset + i] = target_k_vec[i]
                            target_v_cache[step_cache_offset + i] = target_v_vec[i]
                            
                        for h in range(layer_heads):
                            var h_offset = h * layer_head_dim
                            var g = h // ratio
                            var g_offset = g * layer_head_dim
                            
                            for t in range(pos + s + 1):
                                var t_cache_offset = layer_cache_offset + t * kv_stride + g_offset
                                var score: Float32 = 0.0
                                for d in range(layer_head_dim):
                                    score += target_q[h_offset + d] * target_k_cache[t_cache_offset + d]
                                target_attn_scores[t] = score
                                
                            softmax(target_attn_scores, pos + s + 1)
                            
                            for d in range(layer_head_dim):
                                var val_sum: Float32 = 0.0
                                for t in range(pos + s + 1):
                                    var t_cache_offset = layer_cache_offset + t * kv_stride + g_offset
                                    val_sum += target_attn_scores[t] * target_v_cache[t_cache_offset + d]
                                target_attn_out[h_offset + d] = val_sum
                                
                        q8_matmul_vec(weights.wo_packed, weights.wo_scales, target_attn_out, target_proj_attn_out, layer_heads * layer_head_dim, TARGET_CONFIG.embed_dim, num_tasks)
                        
                        rms_norm(target_proj_attn_out, weights.post_attention_layernorm_weight, target_proj_attn_out, TARGET_CONFIG.embed_dim)
                        for i in range(TARGET_CONFIG.embed_dim):
                            local_x[i] += target_proj_attn_out[i]
                            
                        rms_norm(local_x, weights.pre_feedforward_layernorm_weight, target_norm_x, TARGET_CONFIG.embed_dim)
                        q8_matmul_vec(weights.w1_packed, weights.w1_scales, target_norm_x, target_h1, TARGET_CONFIG.embed_dim, TARGET_CONFIG.intermediate_dim, num_tasks)
                        q8_matmul_vec(weights.w3_packed, weights.w3_scales, target_norm_x, target_h2, TARGET_CONFIG.embed_dim, TARGET_CONFIG.intermediate_dim, num_tasks)
                        gelu_and_mul(target_h1, target_h2, target_ffn_out_gate, TARGET_CONFIG.intermediate_dim)
                        
                        # W2 dynamic sparse direct I/O skip loading
                        var num_blocks = TARGET_CONFIG.intermediate_dim // 32
                        var layer_offset = TARGET_CONFIG.get_layer_offset(layer_idx)
                        var w2_offset_in_layer = TARGET_CONFIG.get_w2_offset_in_layer(layer_idx)
                        var tag_base_for_w2: UInt64 = 400000 + UInt64(layer_idx) * 1000 + UInt64(s) * 1000000 + UInt64(pos) * 100000000
                        
                        var max_active_block: Int = -1
                        if w2_threshold == 0.0:
                            max_active_block = num_blocks - 1
                        else:
                            for b in range(num_blocks - 1, -1, -1):
                                var block_offset_gate = b * 32
                                var block_active = False
                                for c in range(32):
                                    if abs(target_ffn_out_gate[block_offset_gate + c]) > w2_threshold:
                                        block_active = True
                                        break
                                if block_active:
                                    max_active_block = b
                                    break
                                    
                        var num_active = 0
                        if resident == "paged":
                            if max_active_block >= 0:
                                num_active = 1
                                var size_to_read = (max_active_block + 1) * 139264
                                var block_offset_in_file = layer_offset + w2_offset_in_layer
                                var dest_mem_addr = weights.w2_packed.__int__()
                                _ = ring.submit_read(weights_fd, dest_mem_addr, size_to_read, block_offset_in_file, tag_base_for_w2)
                                
                            var completed_active = 0
                            var attempts = 0
                            var tag_next_gen: UInt64 = UInt64(30000000 + pos * 100 + layer_idx + 1)
                            
                            while completed_active < num_active:
                                 var res = ring.poll_completion(out_tag_ptr.__int__())
                                 if res >= 0:
                                     var t = out_tag_ptr[0]
                                     if t == tag_base_for_w2:
                                         completed_active += 1
                                     elif t == tag_next_gen:
                                         next_layer_completed = True
                                 attempts += 1
                                 if attempts > 1000000000:
                                     raise Error("Validation weights paging I/O timed out waiting for sparse W2 blocks")
                                     
                        q8_matmul_w2_block_column(weights.w2_packed, target_ffn_out_gate, target_ffn_out, TARGET_CONFIG.intermediate_dim, TARGET_CONFIG.embed_dim, num_tasks, w2_threshold)
                        
                        rms_norm(target_ffn_out, weights.post_feedforward_layernorm_weight, target_ffn_out, TARGET_CONFIG.embed_dim)
                        for i in range(TARGET_CONFIG.embed_dim):
                            local_x[i] += target_ffn_out[i]
                            
                        var scalar = weights.layer_scalar[0]
                        for i in range(TARGET_CONFIG.embed_dim):
                            local_x[i] *= scalar

                    if resident == "paged" and layer_idx + 1 < TARGET_CONFIG.num_layers:
                        if not next_layer_completed:
                            var tag_next_val: UInt64 = UInt64(30000000 + pos * 100 + layer_idx + 1)
                            wait_for_tag(ring, tag_next_val, out_tag_ptr)

            # C. Vocabulary Projections for all target steps in batch
            var target_predictions = List[Int]()
            for s in range(K_actual + 1):
                var local_x = target_x_seq + s * TARGET_CONFIG.embed_dim
                rms_norm(local_x, final_norm_weight, target_final_norm_x, TARGET_CONFIG.embed_dim)
                
                embed.project_logits_parallel(target_final_norm_x, local_max_logits, local_next_tokens, num_tasks)
                var t_max_logit: Float32 = -999999.0
                var t_next_token = 0
                for t in range(num_tasks):
                    if local_max_logits[t] > t_max_logit:
                        t_max_logit = local_max_logits[t]
                        t_next_token = local_next_tokens[t]
                target_predictions.append(t_next_token)

            # D. Speculative Verification Decision
            var num_accepted = 0
            if mode == "speculative":
                for s in range(K_actual):
                    print("[Debug Spec] s =", s, " Target:", target_predictions[s], " Draft Proposed:", draft_tokens[s + 1])
                    if target_predictions[s] == draft_tokens[s + 1]:
                        num_accepted += 1
                    else:
                        break
                    
            var next_starting_token = target_predictions[num_accepted]
            
            # Print all accepted tokens from draft
            for i in range(1, num_accepted + 1):
                var next_token_list = list_cls()
                _ = next_token_list.append(draft_tokens[i])
                var decoded = tokenizer.decode(next_token_list)
                py_builtins.print(decoded, end="", flush=True)
                
            # Print the corrected / continuation token from target
            var next_token_list = list_cls()
            _ = next_token_list.append(next_starting_token)
            var decoded = tokenizer.decode(next_token_list)
            py_builtins.print(decoded, end="", flush=True)
            
            # Record metrics
            total_accepted += num_accepted
            total_speculative_steps += 1

            # Dynamic Speculation Depth (K-Tuning)
            if mode == "speculative":
                if num_accepted == K_actual:
                    current_K = min(current_K + 1, K_MAX)
                elif num_accepted <= 1:
                    current_K = max(current_K - 1, 1)

            # Commit steps & rollback position pointer
            if mode == "speculative" and num_accepted == K_actual:
                # The draft model never ran with input draft_tokens[K_actual], so its key/value at pos + K_actual is missing.
                # Run a single forward step of the draft model on draft_tokens[K_actual] at position pos + K_actual
                # to populate the draft KV cache correctly.
                var curr = draft_tokens[K_actual]
                embed.reconstruct(curr, draft_x)
                for i in range(DRAFT_CONFIG.embed_dim):
                    draft_x[i] *= scale
                
                for layer_idx in range(DRAFT_CONFIG.num_layers):
                    var weights = DRAFT_CONFIG.get_layer_weights(layer_idx, draft_base_ptr + DRAFT_CONFIG.get_layer_offset(layer_idx))
                    var is_global = DRAFT_CONFIG.is_global_layer(layer_idx)
                    var layer_heads = 16
                    var layer_kv_heads = 1 if is_global else 8
                    var layer_head_dim = 512 if is_global else 256
                    var kv_stride = layer_kv_heads * layer_head_dim
                    var ratio = layer_heads // layer_kv_heads
                    
                    rms_norm(draft_x, weights.input_layernorm_weight, draft_norm_x, DRAFT_CONFIG.embed_dim)
                    
                    q4_matmul_vec(weights.wq_packed, weights.wq_scales, draft_norm_x, draft_q, DRAFT_CONFIG.embed_dim, layer_heads * layer_head_dim, num_tasks)
                    q4_matmul_vec(weights.wk_packed, weights.wk_scales, draft_norm_x, draft_k_vec, DRAFT_CONFIG.embed_dim, layer_kv_heads * layer_head_dim, num_tasks)
                    
                    if is_global:
                        for i in range(kv_stride):
                            draft_v_vec[i] = draft_k_vec[i]
                    else:
                        q4_matmul_vec(weights.wv_packed, weights.wv_scales, draft_norm_x, draft_v_vec, DRAFT_CONFIG.embed_dim, layer_kv_heads * layer_head_dim, num_tasks)
                        
                    for h in range(layer_heads):
                        var head_ptr = draft_q + h * layer_head_dim
                        rms_norm(head_ptr, weights.q_norm_weight, head_ptr, layer_head_dim)
                    for h in range(layer_kv_heads):
                        var head_ptr = draft_k_vec + h * layer_head_dim
                        rms_norm(head_ptr, weights.k_norm_weight, head_ptr, layer_head_dim)
                    for h in range(layer_kv_heads):
                        var head_ptr = draft_v_vec + h * layer_head_dim
                        rms_norm_noscale(head_ptr, head_ptr, layer_head_dim)
                        
                    apply_rope(draft_q, layer_heads, layer_head_dim, pos + K_actual, is_global)
                    apply_rope(draft_k_vec, layer_kv_heads, layer_head_dim, pos + K_actual, is_global)
                    
                    var layer_cache_offset = layer_cache_offsets[layer_idx]
                    var step_cache_offset = layer_cache_offset + (pos + K_actual) * kv_stride
                    for i in range(kv_stride):
                        draft_k_cache[step_cache_offset + i] = draft_k_vec[i]
                        draft_v_cache[step_cache_offset + i] = draft_v_vec[i]
                        
                    for h in range(layer_heads):
                        var h_offset = h * layer_head_dim
                        var g = h // ratio
                        var g_offset = g * layer_head_dim
                        for t in range(pos + K_actual + 1):
                            var t_cache_offset = layer_cache_offset + t * kv_stride + g_offset
                            var score: Float32 = 0.0
                            for d in range(layer_head_dim):
                                score += draft_q[h_offset + d] * draft_k_cache[t_cache_offset + d]
                            draft_attn_scores[t] = score
                            
                        softmax(draft_attn_scores, pos + K_actual + 1)
                        
                        for d in range(layer_head_dim):
                            var val_sum: Float32 = 0.0
                            for t in range(pos + K_actual + 1):
                                var t_cache_offset = layer_cache_offset + t * kv_stride + g_offset
                                val_sum += draft_attn_scores[t] * draft_v_cache[t_cache_offset + d]
                            draft_attn_out[h_offset + d] = val_sum
                            
                    q4_matmul_vec(weights.wo_packed, weights.wo_scales, draft_attn_out, draft_proj_attn_out, layer_heads * layer_head_dim, DRAFT_CONFIG.embed_dim, num_tasks)
                    
                    rms_norm(draft_proj_attn_out, weights.post_attention_layernorm_weight, draft_proj_attn_out, DRAFT_CONFIG.embed_dim)
                    for i in range(DRAFT_CONFIG.embed_dim):
                        draft_x[i] += draft_proj_attn_out[i]
                        
                    rms_norm(draft_x, weights.pre_feedforward_layernorm_weight, draft_norm_x, DRAFT_CONFIG.embed_dim)
                    q4_matmul_vec(weights.w1_packed, weights.w1_scales, draft_norm_x, draft_h1, DRAFT_CONFIG.embed_dim, DRAFT_CONFIG.intermediate_dim, num_tasks)
                    q4_matmul_vec(weights.w3_packed, weights.w3_scales, draft_norm_x, draft_h2, DRAFT_CONFIG.embed_dim, DRAFT_CONFIG.intermediate_dim, num_tasks)
                    gelu_and_mul(draft_h1, draft_h2, draft_ffn_out_gate, DRAFT_CONFIG.intermediate_dim)
                    
                    q4_matmul_vec(weights.w2_packed, weights.w2_scales, draft_ffn_out_gate, draft_ffn_out, DRAFT_CONFIG.intermediate_dim, DRAFT_CONFIG.embed_dim, num_tasks)
                    
                    rms_norm(draft_ffn_out, weights.post_feedforward_layernorm_weight, draft_ffn_out, DRAFT_CONFIG.embed_dim)
                    for i in range(DRAFT_CONFIG.embed_dim):
                        draft_x[i] += draft_ffn_out[i]
                        
                    var scalar = weights.layer_scalar[0]
                    for i in range(DRAFT_CONFIG.embed_dim):
                        draft_x[i] *= scalar

            pos = pos + num_accepted + 1
            tokens_generated += num_accepted + 1
            current_token = next_starting_token

        var t_gen_end = py.time()
        var gen_time = Float64(py=(t_gen_end - t_gen_start))
        print("\n[Mojo Bench] Speculative generation completed in", gen_time, "sec (" + String(Float64(tokens_generated) / gen_time) + " t/s)")
        if total_speculative_steps > 0:
            print("[Mojo Bench] Speculative acceptance rate: ", Float64(total_accepted) / Float64(total_speculative_steps * K_MAX) * 100.0, "% (Avg accepted: ", Float64(total_accepted) / Float64(total_speculative_steps), " tokens per step)")

    _ = ring
    _ = out_tag_ptr
