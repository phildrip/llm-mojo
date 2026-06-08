from std.ffi import OwnedDLHandle
from std.memory import alloc
from std.math import sqrt, exp, cos, sin
from std.python import Python
import std.os
from uring import IoUring
from std.algorithm import parallelize
from std.sys.info import num_physical_cores

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

struct GPUContext:
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
# 1. Model Configuration & Memory Offsets
# -----------------------------------------------------------------------------
struct TransformerConfig[
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

        # Pointers to packed weights (Int8)
        var wq_packed = base_ptr
        var wk_packed = wq_packed + wq_size_bytes
        var wv_packed = wk_packed + wk_size_bytes # fallback/initialized
        var wo_packed = wk_packed + wk_size_bytes
        if not is_global:
            wv_packed = wk_packed + wk_size_bytes
            wo_packed = wv_packed + wv_size_bytes
            
        var w1_packed = wo_packed + wo_size_bytes
        var w3_packed = w1_packed + w1_size_bytes
        
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
        var w3_scales = w1_scales + c_blocks_w1
        
        # Unchanged Float32 parameters base address: immediately after the end of w3_scales
        var unchanged_base = w3_scales + c_blocks_w3
        
        var input_layernorm = unchanged_base
        var post_attention_layernorm = input_layernorm + Self.embed_dim
        var pre_feedforward_layernorm = post_attention_layernorm + Self.embed_dim
        var post_feedforward_layernorm = pre_feedforward_layernorm + Self.embed_dim
        var q_norm = post_feedforward_layernorm + Self.embed_dim
        var k_norm = q_norm + layer_head_dim
        var layer_scalar = k_norm + layer_head_dim
        
        # Resolve w2 pointers using new layout at end of layer
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

# -----------------------------------------------------------------------------
# 2. Structural Layer Weights Representation
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
def q8_matmul_w2_block_column(
    W_packed: UnsafePointer[Int8, _],
    x: UnsafePointer[Float32, _],
    out_ptr: UnsafePointer[Float32, MutExternalOrigin],
    c_in: Int,
    c_out: Int,
    num_tasks: Int,
    c_memcpy: fn(Int, Int, Int) -> Int,
    w2_scratch: UnsafePointer[Int8, MutExternalOrigin],
    threshold: Float32 = 1e-4,
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
    
    # Copy all blocks up to max_active_block contiguously from GTT to cached memory
    var copy_size = (max_active_block + 1) * 139264
    _ = c_memcpy(w2_scratch.__int__(), W_packed.__int__(), copy_size)
    
    def sparse_worker(t: Int) capturing -> None:
        var start_r = t * chunk_size
        var end_r = (t + 1) * chunk_size if t < num_tasks - 1 else c_out
        
        for r in range(start_r, end_r):
            var acc = SIMD[DType.float32, 8](0.0)
            
            for idx in range(num_active):
                var b = active_blocks[idx]
                var block_base = w2_scratch + b * 139264
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
    for i in range(0, size, 8):
        var v1 = h1.load[8](i)
        var v2 = h2.load[8](i)
        for lane in range(8):
            var x1 = v1[lane]
            var x2 = v2[lane]
            var tanh_arg = 0.79788456 * x1 * (1.0 + 0.044715 * x1 * x1)
            var gelu = 0.5 * x1 * (1.0 + fast_tanh(tanh_arg))
            out_ptr[i + lane] = gelu * x2

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
    print(" [io_uring] wait_for_tag: waiting for expected_tag =", expected_tag)
    var libc = OwnedDLHandle("libc.so.6")
    var c_usleep = libc.get_function[fn(UInt32) -> Int32]("usleep")
    var attempts = 0
    while True:
        var res = ring.poll_completion(out_tag_ptr.__int__())
        if res >= 0:
            if out_tag_ptr[0] == expected_tag:
                print(" [io_uring] wait_for_tag: found expected_tag =", expected_tag, "transferred =", res, "bytes")
                return
            else:
                print(" [io_uring] wait_for_tag: completed tag =", out_tag_ptr[0], "res =", res, "(ignoring, waiting for", expected_tag, ")")
        elif res != -11:
            raise Error("Weights paging io_uring error: " + String(res) + " waiting for layer: " + String(expected_tag))
        attempts += 1
        if attempts > 1000:
            _ = c_usleep(20) # 20 us sleep to prevent CPU starvation
        if attempts > 100000: # with usleep, 100k is around 2-3 seconds, plenty of time for disk I/O!
            raise Error("Weights paging I/O timed out waiting for layer: " + String(expected_tag))

# -----------------------------------------------------------------------------
# 4. Standalone Test Executable & Story Generator
# -----------------------------------------------------------------------------
def main() raises:
    var num_tasks = num_physical_cores()
    print("=========================================================")
    print("   GEMMA 4 12B-SCALE DEEP TRANSCO-PROCESSOR RUNNER       ")
    print("=========================================================")
    print("Embedding Size:      3840")
    print("Attention Heads:     16 Query Heads / 8 KV Heads (GQA/MQA)")
    print("Model Parameters:    12B Scale Causal Layout")
    print("Kernel Paging I/O:   Fully Synchronous DRAM-to-GTT Cache")
    print("Hardware Accelerator: AMD Cezanne Vega iGPU (Direct Userspace)")
    print("---------------------------------------------------------")

    var CONFIG = TransformerConfig[
        embed_dim = 3840,
        head_dim = 256,
        num_heads = 16,
        num_kv_heads = 8,
        intermediate_dim = 15360,
        num_layers = 48,
        vocab_size = 262144,
        max_seq_len = 100
    ]()

    # Initialize GPU context
    print("Scanning topology for AMD GPU...")
    var gpu_id: UInt32 = 0
    var libc = OwnedDLHandle("libc.so.6")
    var c_open = libc.get_function[fn(Int, Int32, Int32) -> Int32]("open")
    var c_close = libc.get_function[fn(Int32) -> Int32]("close")
    var c_read = libc.get_function[fn(Int32, Int, Int) -> Int]("read")
    var c_ioctl = libc.get_function[fn(Int32, Int, Int) -> Int32]("ioctl")
    var posix_memalign = libc.get_function[fn(Int, Int, Int) -> Int32]("posix_memalign")
    var free_c = libc.get_function[fn(Int) -> NoneType]("free")
    var c_memcpy = libc.get_function[fn(Int, Int, Int) -> Int]("memcpy")

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
    var gpu_ctx = GPUContext(kfd_fd, drm_fd, gpu_id)
    print(" -> GPU Context initialized successfully!")

    var py = Python.import_module("time")
    var sys = Python.import_module("sys")
    # sys.path.append("/home/phil/git/llm-mojo/.venv/lib/python3.12/site-packages")
    var transformers = Python.import_module("transformers")
    print("Loading Gemma 4 tokenizer...")
    var tokenizer = transformers.AutoTokenizer.from_pretrained("google/gemma-4-12B-it")
    print("Tokenizer loaded successfully!")

    var ring = IoUring(16)
    var out_tag_ptr = alloc[UInt64](1)
    out_tag_ptr[0] = 0

    var gtt_cached_flags: UInt32 = 0xF6000002 # KFD_IOC_ALLOC_MEM_FLAGS_COHERENT | UNCACHED | PUBLIC | GTT | WRITABLE | EXECUTABLE | NO_SUBSTITUTE (UNCACHED!)
    var total_weights_size = CONFIG.get_layer_offset(CONFIG.num_layers)
    print(" -> Allocating GTT weight buffer for the entire 12.29 GB of weights...")
    var weights_gpu_buf = gpu_ctx.alloc_gpu_buffer(total_weights_size, gtt_cached_flags, cpu_access=True)
    gpu_ctx.weights_gpu_buf = weights_gpu_buf.copy()
    var weights_gpu_ptr = UnsafePointer[Int8, MutExternalOrigin](unsafe_from_address=weights_gpu_buf.cpu_addr)

    var weights_file = "gemma4_weights_q8_sorted.bin"
    print("Opening weights file 'gemma4_weights_q8_sorted.bin'...")
    var path_bytes = weights_file.as_bytes()
    var path_ptr = alloc[UInt8](len(path_bytes) + 1)
    for i in range(len(path_bytes)):
        path_ptr[i] = path_bytes[i]
    path_ptr[len(path_bytes)] = 0
    var weights_fd = c_open(path_ptr.__int__(), 0, 0) # 0 is O_RDONLY
    if weights_fd < 0:
        print("CRITICAL: Failed to open weights file. Verify conversion path!")
        return

    var total_read_weights = 0
    print("Loading weights into GTT memory...")
    var t_load_start = py.time()
    while total_read_weights < total_weights_size:
        var bytes_to_read = total_weights_size - total_read_weights
        if bytes_to_read > 1024 * 1024 * 1024:
            bytes_to_read = 1024 * 1024 * 1024 # 1GB chunk
        var read_res = c_read(weights_fd, weights_gpu_buf.cpu_addr + total_read_weights, bytes_to_read)
        if read_res <= 0:
            print("CRITICAL: Failed to read from weights file!")
            return
        total_read_weights += read_res
        print(" -> Loaded ", total_read_weights // (1024*1024), " MB / ", total_weights_size // (1024*1024), " MB")
        
    var t_load_end = py.time()
    print("Weights loaded into GTT memory successfully in " + String(Float64(py=(t_load_end - t_load_start))) + " sec!")
    _ = c_close(weights_fd)
    path_ptr.free()

    # Embeddings
    var embed_file = "gemma4_embed.bin"
    var embed_path_bytes = embed_file.as_bytes()
    var embed_path_ptr = alloc[UInt8](len(embed_path_bytes) + 1)
    for i in range(len(embed_path_bytes)):
        embed_path_ptr[i] = embed_path_bytes[i]
    embed_path_ptr[len(embed_path_bytes)] = 0
    var embed_fd = c_open(embed_path_ptr.__int__(), 0, 0)
    if embed_fd < 0:
        print("CRITICAL: Failed to open embeddings file!")
        return

    var embed_size_bytes = CONFIG.vocab_size * CONFIG.embed_dim * 4
    var memptr_embed = alloc[Int](1)
    _ = posix_memalign(memptr_embed.__int__(), 4096, embed_size_bytes)
    var embed_addr = memptr_embed[0]
    var embed_ptr = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=embed_addr)

    var total_read = 0
    while total_read < embed_size_bytes:
        var bytes_to_read = embed_size_bytes - total_read
        var read_res = c_read(embed_fd, embed_addr + total_read, bytes_to_read)
        if read_res <= 0:
            print("CRITICAL: Failed to read embeddings!")
            return
        total_read += read_res
    print("Loaded embeddings into memory: " + String(total_read) + " bytes")
    _ = c_close(embed_fd)
    embed_path_ptr.free()

    # Final Layer Norm weights
    var final_norm_file = "gemma4_final_norm.bin"
    var final_norm_path_bytes = final_norm_file.as_bytes()
    var final_norm_path_ptr = alloc[UInt8](len(final_norm_path_bytes) + 1)
    for i in range(len(final_norm_path_bytes)):
        final_norm_path_ptr[i] = final_norm_path_bytes[i]
    final_norm_path_ptr[len(final_norm_path_bytes)] = 0
    var final_norm_fd = c_open(final_norm_path_ptr.__int__(), 0, 0)
    if final_norm_fd < 0:
        print("CRITICAL: Failed to open final norm file!")
        return

    var final_norm_weight = alloc[Float32](CONFIG.embed_dim)
    var final_norm_size_bytes = CONFIG.embed_dim * 4
    total_read = 0
    while total_read < final_norm_size_bytes:
        var bytes_to_read = final_norm_size_bytes - total_read
        var read_res = c_read(final_norm_fd, final_norm_weight.__int__() + total_read, bytes_to_read)
        if read_res <= 0:
            print("CRITICAL: Failed to read final norm weights!")
            return
        total_read += read_res
    print("Loaded final norm weights: " + String(total_read) + " bytes")
    _ = c_close(final_norm_fd)
    final_norm_path_ptr.free()

    var x_buf = gpu_ctx.alloc_gpu_buffer(((CONFIG.embed_dim * 4 + 4095) // 4096) * 4096 + 4096, gtt_cached_flags, cpu_access=True)
    var norm_x_buf = gpu_ctx.alloc_gpu_buffer(((CONFIG.embed_dim * 4 + 4095) // 4096) * 4096 + 4096, gtt_cached_flags, cpu_access=True)
    var norm_x_buf_1 = gpu_ctx.alloc_gpu_buffer(((CONFIG.embed_dim * 4 + 4095) // 4096) * 4096 + 4096, gtt_cached_flags, cpu_access=True)
    var final_norm_x_buf = gpu_ctx.alloc_gpu_buffer(((CONFIG.embed_dim * 4 + 4095) // 4096) * 4096 + 4096, gtt_cached_flags, cpu_access=True)
    
    var q_buf = gpu_ctx.alloc_gpu_buffer(((8192 * 4 + 4095) // 4096) * 4096 + 4096, gtt_cached_flags, cpu_access=True)
    var q_buf_1 = gpu_ctx.alloc_gpu_buffer(((8192 * 4 + 4095) // 4096) * 4096 + 4096, gtt_cached_flags, cpu_access=True)
    var k_vec_buf = gpu_ctx.alloc_gpu_buffer(((2048 * 4 + 4095) // 4096) * 4096 + 4096, gtt_cached_flags, cpu_access=True)
    var k_vec_buf_1 = gpu_ctx.alloc_gpu_buffer(((2048 * 4 + 4095) // 4096) * 4096 + 4096, gtt_cached_flags, cpu_access=True)
    var v_vec_buf = gpu_ctx.alloc_gpu_buffer(((2048 * 4 + 4095) // 4096) * 4096 + 4096, gtt_cached_flags, cpu_access=True)
    var v_vec_buf_1 = gpu_ctx.alloc_gpu_buffer(((2048 * 4 + 4095) // 4096) * 4096 + 4096, gtt_cached_flags, cpu_access=True)
    
    var attn_out_buf = gpu_ctx.alloc_gpu_buffer(((8192 * 4 + 4095) // 4096) * 4096 + 4096, gtt_cached_flags, cpu_access=True)
    var attn_out_buf_1 = gpu_ctx.alloc_gpu_buffer(((8192 * 4 + 4095) // 4096) * 4096 + 4096, gtt_cached_flags, cpu_access=True)
    var proj_attn_out_buf = gpu_ctx.alloc_gpu_buffer(((CONFIG.embed_dim * 4 + 4095) // 4096) * 4096 + 4096, gtt_cached_flags, cpu_access=True)
    var h1_buf = gpu_ctx.alloc_gpu_buffer(((CONFIG.intermediate_dim * 4 + 4095) // 4096) * 4096 + 4096, gtt_cached_flags, cpu_access=True)
    var h2_buf = gpu_ctx.alloc_gpu_buffer(((CONFIG.intermediate_dim * 4 + 4095) // 4096) * 4096 + 4096, gtt_cached_flags, cpu_access=True)

    gpu_ctx.x_buf = x_buf.copy()
    gpu_ctx.norm_x_buf = norm_x_buf.copy()
    gpu_ctx.norm_x_buf_1 = norm_x_buf_1.copy()
    gpu_ctx.final_norm_x_buf = final_norm_x_buf.copy()
    gpu_ctx.q_buf = q_buf.copy()
    gpu_ctx.q_buf_1 = q_buf_1.copy()
    gpu_ctx.k_vec_buf = k_vec_buf.copy()
    gpu_ctx.k_vec_buf_1 = k_vec_buf_1.copy()
    gpu_ctx.v_vec_buf = v_vec_buf.copy()
    gpu_ctx.v_vec_buf_1 = v_vec_buf_1.copy()
    gpu_ctx.attn_out_buf = attn_out_buf.copy()
    gpu_ctx.attn_out_buf_1 = attn_out_buf_1.copy()
    gpu_ctx.proj_attn_out_buf = proj_attn_out_buf.copy()
    gpu_ctx.h1_buf = h1_buf.copy()
    gpu_ctx.h2_buf = h2_buf.copy()

    var x = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=x_buf.cpu_addr)
    var norm_x = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=norm_x_buf.cpu_addr)
    var norm_x_1 = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=norm_x_buf_1.cpu_addr)
    var final_norm_x = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=final_norm_x_buf.cpu_addr)
    var q = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=q_buf.cpu_addr)
    var q_1 = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=q_buf_1.cpu_addr)
    var k_vec = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=k_vec_buf.cpu_addr)
    var k_vec_1 = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=k_vec_buf_1.cpu_addr)
    var v_vec = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=v_vec_buf.cpu_addr)
    var v_vec_1 = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=v_vec_buf_1.cpu_addr)
    
    var attn_scores = alloc[Float32](CONFIG.num_heads * CONFIG.max_seq_len)
    var attn_out = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=attn_out_buf.cpu_addr)
    var attn_out_1 = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=attn_out_buf_1.cpu_addr)
    var proj_attn_out = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=proj_attn_out_buf.cpu_addr)
    var h1 = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=h1_buf.cpu_addr)
    var h2 = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=h2_buf.cpu_addr)
    
    var ffn_out_gate = alloc[Float32](CONFIG.intermediate_dim)
    var ffn_out = alloc[Float32](CONFIG.embed_dim)
    var w2_scratch = alloc[Int8](480 * 139264)

    # KV-cache setup
    var layer_cache_offsets = List[Int]()
    var current_cache_offset = 0
    for i in range(CONFIG.num_layers):
        layer_cache_offsets.append(current_cache_offset)
        var is_global = CONFIG.is_global_layer(i)
        var layer_kv_heads = 1 if is_global else 8
        var layer_head_dim = 512 if is_global else 256
        current_cache_offset += CONFIG.max_seq_len * layer_kv_heads * layer_head_dim
    var total_cache_size = current_cache_offset

    var k_cache = alloc[Float32](total_cache_size)
    var v_cache = alloc[Float32](total_cache_size)
    for i in range(total_cache_size):
        k_cache[i] = 0.0
        v_cache[i] = 0.0

    print("=========================================================")
    print("   GEMMA 4 12B iGPU HYBRID ACCELERATED CHAT AGENT        ")
    print("=========================================================")
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
            
        print("[Tokenizer] Encoded input into", num_encoded, "tokens:")
        var token_ids_str = String("")
        var prompt_tokens = List[Int]()
        for i in range(num_encoded):
            var tid = Int(py=encoded[i])
            token_ids_str += String(tid) + " "
            prompt_tokens.append(tid)
        print(token_ids_str)
            
        print("Agent > ", end="")
        
        var current_token: Int
        var num_prompt_tokens = len(prompt_tokens)
        var max_steps = num_prompt_tokens + 30
        if max_steps > CONFIG.max_seq_len:
            max_steps = CONFIG.max_seq_len
            
        var scale = sqrt(Float32(CONFIG.embed_dim))
        
        # 1. Prefill phase: process all prompt tokens layer-by-layer
        var t_prefill_start = py.time()
        var prompt_x = alloc[Float32](num_prompt_tokens * CONFIG.embed_dim)
        for pos in range(num_prompt_tokens):
            var token_emb_offset = prompt_tokens[pos] * CONFIG.embed_dim
            var px_offset = pos * CONFIG.embed_dim
            for i in range(CONFIG.embed_dim):
                prompt_x[px_offset + i] = embed_ptr[token_emb_offset + i] * scale
                
        for layer_idx in range(CONFIG.num_layers):
            print(" [DEBUG] Entering layer", layer_idx)
            var active_weights_ptr = weights_gpu_ptr + CONFIG.get_layer_offset(layer_idx)
            print(" [DEBUG] active_weights_ptr offset calculated")
            var weights = CONFIG.get_layer_weights(layer_idx, active_weights_ptr)
            print(" [DEBUG] get_layer_weights completed")
            var is_global = CONFIG.is_global_layer(layer_idx)
            var layer_heads = 16
            var layer_kv_heads = 1 if is_global else 8
            var layer_head_dim = 512 if is_global else 256
            var kv_stride = layer_kv_heads * layer_head_dim
            var ratio = layer_heads // layer_kv_heads
            
            # Enqueue the first position (pos = 0)
            var active_slot_q = -1
            var active_slot_k = -1
            var active_slot_v = -1
            if num_prompt_tokens > 0:
                var first_local_x = prompt_x + 0
                rms_norm(first_local_x, weights.input_layernorm_weight, norm_x, CONFIG.embed_dim)
                active_slot_q = gpu_ctx.enqueue_matmul(weights.wq_packed.__int__(), weights.wq_scales.__int__(), norm_x.__int__(), q.__int__(), CONFIG.embed_dim, layer_heads * layer_head_dim)
                active_slot_k = gpu_ctx.enqueue_matmul(weights.wk_packed.__int__(), weights.wk_scales.__int__(), norm_x.__int__(), k_vec.__int__(), CONFIG.embed_dim, layer_kv_heads * layer_head_dim)
                if not is_global:
                    active_slot_v = gpu_ctx.enqueue_matmul(weights.wv_packed.__int__(), weights.wv_scales.__int__(), norm_x.__int__(), v_vec.__int__(), CONFIG.embed_dim, layer_kv_heads * layer_head_dim)

            # Loop over all prompt positions
            for pos in range(num_prompt_tokens):
                var px_offset = pos * CONFIG.embed_dim
                var local_x = prompt_x + px_offset
                
                var idx = pos % 2
                var cur_norm_x = norm_x if idx == 0 else norm_x_1
                var cur_q = q if idx == 0 else q_1
                var cur_k_vec = k_vec if idx == 0 else k_vec_1
                var cur_v_vec = v_vec if idx == 0 else v_vec_1
                var cur_attn_out = attn_out if idx == 0 else attn_out_1
                
                # Wait for the GPU to finish Q, K, and V projections for the current position
                gpu_ctx.wait_matmul(active_slot_q, cur_q.__int__(), layer_heads * layer_head_dim)
                gpu_ctx.wait_matmul(active_slot_k, cur_k_vec.__int__(), layer_kv_heads * layer_head_dim)
                if is_global:
                    for i in range(kv_stride):
                        cur_v_vec[i] = cur_k_vec[i]
                else:
                    gpu_ctx.wait_matmul(active_slot_v, cur_v_vec.__int__(), layer_kv_heads * layer_head_dim)
                
                # Asynchronously pre-enqueue projections for the next position (pos + 1)
                var next_slot_q = -1
                var next_slot_k = -1
                var next_slot_v = -1
                if pos + 1 < num_prompt_tokens:
                    var next_pos = pos + 1
                    var next_idx = next_pos % 2
                    var next_norm_x = norm_x if next_idx == 0 else norm_x_1
                    var next_q = q if next_idx == 0 else q_1
                    var next_k_vec = k_vec if next_idx == 0 else k_vec_1
                    var next_v_vec = v_vec if next_idx == 0 else v_vec_1
                    
                    var next_local_x = prompt_x + next_pos * CONFIG.embed_dim
                    rms_norm(next_local_x, weights.input_layernorm_weight, next_norm_x, CONFIG.embed_dim)
                    
                    next_slot_q = gpu_ctx.enqueue_matmul(weights.wq_packed.__int__(), weights.wq_scales.__int__(), next_norm_x.__int__(), next_q.__int__(), CONFIG.embed_dim, layer_heads * layer_head_dim)
                    next_slot_k = gpu_ctx.enqueue_matmul(weights.wk_packed.__int__(), weights.wk_scales.__int__(), next_norm_x.__int__(), next_k_vec.__int__(), CONFIG.embed_dim, layer_kv_heads * layer_head_dim)
                    if not is_global:
                        next_slot_v = gpu_ctx.enqueue_matmul(weights.wv_packed.__int__(), weights.wv_scales.__int__(), next_norm_x.__int__(), next_v_vec.__int__(), CONFIG.embed_dim, layer_kv_heads * layer_head_dim)
                
                # While the next token's GPU work is running, CPU processes the current position
                # Apply head-wise RMS normalization
                for h in range(layer_heads):
                    var head_ptr = cur_q + h * layer_head_dim
                    rms_norm(head_ptr, weights.q_norm_weight, head_ptr, layer_head_dim)
                for h in range(layer_kv_heads):
                    var head_ptr = cur_k_vec + h * layer_head_dim
                    rms_norm(head_ptr, weights.k_norm_weight, head_ptr, layer_head_dim)
                for h in range(layer_kv_heads):
                    var head_ptr = cur_v_vec + h * layer_head_dim
                    rms_norm_noscale(head_ptr, head_ptr, layer_head_dim)
                    
                apply_rope(cur_q, layer_heads, layer_head_dim, pos, is_global)
                apply_rope(cur_k_vec, layer_kv_heads, layer_head_dim, pos, is_global)
                
                var layer_cache_offset = layer_cache_offsets[layer_idx]
                var step_cache_offset = layer_cache_offset + pos * kv_stride
                for i in range(kv_stride):
                    k_cache[step_cache_offset + i] = cur_k_vec[i]
                    v_cache[step_cache_offset + i] = cur_v_vec[i]
                    
                def prefill_head_worker(h: Int) capturing -> None:
                    var local_idx = pos % 2
                    var local_q = q if local_idx == 0 else q_1
                    var local_attn_out = attn_out if local_idx == 0 else attn_out_1
                    
                    var h_offset = h * layer_head_dim
                    var g = h // ratio
                    var g_offset = g * layer_head_dim
                    var head_scores = attn_scores + h * CONFIG.max_seq_len
                    
                    for t in range(pos + 1):
                        var t_cache_offset = layer_cache_offset + t * kv_stride + g_offset
                        var score_vec = SIMD[DType.float32, 16](0.0)
                        for d in range(0, layer_head_dim, 16):
                            score_vec += local_q.load[16](h_offset + d) * k_cache.load[16](t_cache_offset + d)
                        head_scores[t] = score_vec.reduce_add()
                        
                    softmax(head_scores, pos + 1)
                    
                    for d in range(0, layer_head_dim, 16):
                        local_attn_out.store[16](h_offset + d, SIMD[DType.float32, 16](0.0))
                        
                    for t in range(pos + 1):
                        var t_cache_offset = layer_cache_offset + t * kv_stride + g_offset
                        var s_vec = SIMD[DType.float32, 16](head_scores[t])
                        for d in range(0, layer_head_dim, 16):
                            var v = v_cache.load[16](t_cache_offset + d)
                            var acc = local_attn_out.load[16](h_offset + d)
                            local_attn_out.store[16](h_offset + d, acc + s_vec * v)
                        
                parallelize[prefill_head_worker](layer_heads)
                        
                # Compute wo on GPU!
                gpu_ctx.launch_matmul(weights.wo_packed.__int__(), weights.wo_scales.__int__(), cur_attn_out.__int__(), proj_attn_out.__int__(), layer_heads * layer_head_dim, CONFIG.embed_dim)
                
                # Post-attention normalization on projected attention output
                rms_norm(proj_attn_out, weights.post_attention_layernorm_weight, proj_attn_out, CONFIG.embed_dim)
                for i in range(CONFIG.embed_dim):
                    local_x[i] += proj_attn_out[i]
                    
                # Pre-feedforward normalization
                rms_norm(local_x, weights.pre_feedforward_layernorm_weight, cur_norm_x, CONFIG.embed_dim)
                
                # Compute w1, w3 on GPU asynchronously!
                var slot_w1 = gpu_ctx.enqueue_matmul(weights.w1_packed.__int__(), weights.w1_scales.__int__(), cur_norm_x.__int__(), h1.__int__(), CONFIG.embed_dim, CONFIG.intermediate_dim)
                var slot_w3 = gpu_ctx.enqueue_matmul(weights.w3_packed.__int__(), weights.w3_scales.__int__(), cur_norm_x.__int__(), h2.__int__(), CONFIG.embed_dim, CONFIG.intermediate_dim)
                
                gpu_ctx.wait_matmul(slot_w1, h1.__int__(), CONFIG.intermediate_dim)
                gpu_ctx.wait_matmul(slot_w3, h2.__int__(), CONFIG.intermediate_dim)
                
                gelu_and_mul(h1, h2, ffn_out_gate, CONFIG.intermediate_dim)
                
                q8_matmul_w2_block_column(weights.w2_packed, ffn_out_gate, ffn_out, CONFIG.intermediate_dim, CONFIG.embed_dim, num_tasks, c_memcpy, w2_scratch, Float32(1e-4))
                
                # Post-feedforward normalization on MLP output
                rms_norm(ffn_out, weights.post_feedforward_layernorm_weight, ffn_out, CONFIG.embed_dim)
                for i in range(CONFIG.embed_dim):
                    local_x[i] += ffn_out[i]
                    
                # Scale hidden state by layer_scalar
                var scalar = weights.layer_scalar[0]
                for i in range(CONFIG.embed_dim):
                    local_x[i] *= scalar

                # Advance pipeline state
                active_slot_q = next_slot_q
                active_slot_k = next_slot_k
                active_slot_v = next_slot_v

        # Copy the last prompt token's hidden state to x
        var last_px_offset = (num_prompt_tokens - 1) * CONFIG.embed_dim
        for i in range(CONFIG.embed_dim):
            x[i] = prompt_x[last_px_offset + i]
        prompt_x.free()
        
        var t_prefill_end = py.time()
        var prefill_time = Float64(py=(t_prefill_end - t_prefill_start))
        print("\n[Mojo Bench] Prefill completed in", prefill_time, "sec (" + String(Float64(num_prompt_tokens) / prefill_time) + " t/s)")
        
        # 2. Setup parallel projection worker and select first generated token
        rms_norm(x, final_norm_weight, final_norm_x, CONFIG.embed_dim)

        var chunk_size = CONFIG.vocab_size // num_tasks
        var local_max_logits = alloc[Float32](num_tasks)
        var local_next_tokens = alloc[Int](num_tasks)
        
        def proj_worker(t: Int) capturing -> None:
            var start_r = t * chunk_size
            var end_r = (t + 1) * chunk_size if t < num_tasks - 1 else CONFIG.vocab_size
            
            var local_max_l: Float32 = -999999.0
            var local_tok = 0
            
            for r in range(start_r, end_r):
                var r_offset = r * CONFIG.embed_dim
                var acc = SIMD[DType.float32, 8](0.0)
                for c in range(0, CONFIG.embed_dim, 8):
                    acc = acc + final_norm_x.load[8](c) * embed_ptr.load[8](r_offset + c)
                var logit = acc.reduce_add()
                logit = Float32(30.0) * fast_tanh(logit / Float32(30.0))
                
                if logit > local_max_l:
                    local_max_l = logit
                    local_tok = r
                    
            local_max_logits[t] = local_max_l
            local_next_tokens[t] = local_tok
            
        parallelize[proj_worker](num_tasks)
        
        # Reduce to find global max
        var max_logit: Float32 = -999999.0
        var next_token = 0
        for t in range(num_tasks):
            if local_max_logits[t] > max_logit:
                max_logit = local_max_logits[t]
                next_token = local_next_tokens[t]
                
        # Print first token
        var next_token_list = list_cls()
        _ = next_token_list.append(next_token)
        var decoded = tokenizer.decode(next_token_list)
        py_builtins.print(decoded, end="", flush=True)
        
        current_token = next_token
        
        # 3. Subsequent generation steps
        var tokens_generated = 1
        var t_gen_start = py.time()
        for pos in range(num_prompt_tokens, max_steps):
            # Setup input activations x with scaled current_token embedding
            var token_emb_offset = current_token * CONFIG.embed_dim
            for i in range(CONFIG.embed_dim):
                x[i] = embed_ptr[token_emb_offset + i] * scale
                
            # Run the Transformer forward pass
            for layer_idx in range(CONFIG.num_layers):
                var active_weights_ptr = weights_gpu_ptr + CONFIG.get_layer_offset(layer_idx)
                var weights = CONFIG.get_layer_weights(layer_idx, active_weights_ptr)
                var is_global = CONFIG.is_global_layer(layer_idx)
                rms_norm(x, weights.input_layernorm_weight, norm_x, CONFIG.embed_dim)
                
                var layer_heads = 16
                var layer_kv_heads = 1 if is_global else 8
                var layer_head_dim = 512 if is_global else 256
                var kv_stride = layer_kv_heads * layer_head_dim
                
                # Compute q, k on GPU asynchronously!
                var slot_q = gpu_ctx.enqueue_matmul(weights.wq_packed.__int__(), weights.wq_scales.__int__(), norm_x.__int__(), q.__int__(), CONFIG.embed_dim, layer_heads * layer_head_dim)
                var slot_k = gpu_ctx.enqueue_matmul(weights.wk_packed.__int__(), weights.wk_scales.__int__(), norm_x.__int__(), k_vec.__int__(), CONFIG.embed_dim, layer_kv_heads * layer_head_dim)
                var slot_v: Int = -1
                
                if not is_global:
                    slot_v = gpu_ctx.enqueue_matmul(weights.wv_packed.__int__(), weights.wv_scales.__int__(), norm_x.__int__(), v_vec.__int__(), CONFIG.embed_dim, layer_kv_heads * layer_head_dim)
                
                # Now wait for them to complete
                gpu_ctx.wait_matmul(slot_q, q.__int__(), layer_heads * layer_head_dim)
                gpu_ctx.wait_matmul(slot_k, k_vec.__int__(), layer_kv_heads * layer_head_dim)
                
                if is_global:
                    for i in range(kv_stride):
                        v_vec[i] = k_vec[i]
                else:
                    gpu_ctx.wait_matmul(slot_v, v_vec.__int__(), layer_kv_heads * layer_head_dim)
                    
                # Apply head-wise RMS normalization
                for h in range(layer_heads):
                    var head_ptr = q + h * layer_head_dim
                    rms_norm(head_ptr, weights.q_norm_weight, head_ptr, layer_head_dim)
                for h in range(layer_kv_heads):
                    var head_ptr = k_vec + h * layer_head_dim
                    rms_norm(head_ptr, weights.k_norm_weight, head_ptr, layer_head_dim)
                for h in range(layer_kv_heads):
                    var head_ptr = v_vec + h * layer_head_dim
                    rms_norm_noscale(head_ptr, head_ptr, layer_head_dim)
                    
                apply_rope(q, layer_heads, layer_head_dim, pos, is_global)
                apply_rope(k_vec, layer_kv_heads, layer_head_dim, pos, is_global)
                
                var layer_cache_offset = layer_cache_offsets[layer_idx]
                var step_cache_offset = layer_cache_offset + pos * kv_stride
                for i in range(kv_stride):
                    k_cache[step_cache_offset + i] = k_vec[i]
                    v_cache[step_cache_offset + i] = v_vec[i]
                    
                var ratio = layer_heads // layer_kv_heads
                def gen_head_worker(h: Int) capturing -> None:
                    var h_offset = h * layer_head_dim
                    var g = h // ratio
                    var g_offset = g * layer_head_dim
                    var head_scores = attn_scores + h * CONFIG.max_seq_len
                    
                    for t in range(pos + 1):
                        var t_cache_offset = layer_cache_offset + t * kv_stride + g_offset
                        var score_vec = SIMD[DType.float32, 16](0.0)
                        for d in range(0, layer_head_dim, 16):
                            score_vec += q.load[16](h_offset + d) * k_cache.load[16](t_cache_offset + d)
                        head_scores[t] = score_vec.reduce_add()
                        
                    softmax(head_scores, pos + 1)
                    
                    for d in range(0, layer_head_dim, 16):
                        attn_out.store[16](h_offset + d, SIMD[DType.float32, 16](0.0))
                        
                    for t in range(pos + 1):
                        var t_cache_offset = layer_cache_offset + t * kv_stride + g_offset
                        var s_vec = SIMD[DType.float32, 16](head_scores[t])
                        for d in range(0, layer_head_dim, 16):
                            var v = v_cache.load[16](t_cache_offset + d)
                            var acc = attn_out.load[16](h_offset + d)
                            attn_out.store[16](h_offset + d, acc + s_vec * v)
                        
                for h in range(layer_heads):
                    gen_head_worker(h)
                        
                # Compute wo on GPU!
                gpu_ctx.launch_matmul(weights.wo_packed.__int__(), weights.wo_scales.__int__(), attn_out.__int__(), proj_attn_out.__int__(), layer_heads * layer_head_dim, CONFIG.embed_dim)
                
                # Post-attention normalization on projected attention output
                rms_norm(proj_attn_out, weights.post_attention_layernorm_weight, proj_attn_out, CONFIG.embed_dim)
                for i in range(CONFIG.embed_dim):
                    x[i] += proj_attn_out[i]
                    
                # Pre-feedforward normalization
                rms_norm(x, weights.pre_feedforward_layernorm_weight, norm_x, CONFIG.embed_dim)
                
                # Compute w1, w3 on GPU asynchronously!
                var slot_w1 = gpu_ctx.enqueue_matmul(weights.w1_packed.__int__(), weights.w1_scales.__int__(), norm_x.__int__(), h1.__int__(), CONFIG.embed_dim, CONFIG.intermediate_dim)
                var slot_w3 = gpu_ctx.enqueue_matmul(weights.w3_packed.__int__(), weights.w3_scales.__int__(), norm_x.__int__(), h2.__int__(), CONFIG.embed_dim, CONFIG.intermediate_dim)
                
                gpu_ctx.wait_matmul(slot_w1, h1.__int__(), CONFIG.intermediate_dim)
                gpu_ctx.wait_matmul(slot_w3, h2.__int__(), CONFIG.intermediate_dim)
                
                gelu_and_mul(h1, h2, ffn_out_gate, CONFIG.intermediate_dim)
                
                # Evaluate the W2 matmul directly on the resident weights
                q8_matmul_w2_block_column(weights.w2_packed, ffn_out_gate, ffn_out, CONFIG.intermediate_dim, CONFIG.embed_dim, num_tasks, c_memcpy, w2_scratch, Float32(1e-4))
                
                # Post-feedforward normalization on MLP output
                rms_norm(ffn_out, weights.post_feedforward_layernorm_weight, ffn_out, CONFIG.embed_dim)
                for i in range(CONFIG.embed_dim):
                    x[i] += ffn_out[i]
                    
                # Scale hidden state by layer_scalar
                var scalar = weights.layer_scalar[0]
                for i in range(CONFIG.embed_dim):
                    x[i] *= scalar
                    
            # Predict next token
            rms_norm(x, final_norm_weight, final_norm_x, CONFIG.embed_dim)
            parallelize[proj_worker](num_tasks)
            
            # Reduce to find global max
            max_logit = -999999.0
            next_token = 0
            for t in range(num_tasks):
                if local_max_logits[t] > max_logit:
                    max_logit = local_max_logits[t]
                    next_token = local_next_tokens[t]
                    
            # Print word
            var next_token_list_inner = list_cls()
            _ = next_token_list_inner.append(next_token)
            var word = String(tokenizer.decode(next_token_list_inner))
            py_builtins.print(word, end="", flush=True)
            
            if next_token == 1 or next_token == 106: # <eos> or <turn|>
                print()
                break
                
            current_token = next_token
            tokens_generated += 1
            
        var t_gen_end = py.time()
        var gen_time = Float64(py=(t_gen_end - t_gen_start))
        print("\n[Mojo Bench] Generation completed in", gen_time, "sec (" + String(Float64(tokens_generated) / gen_time) + " t/s)")
        
        local_max_logits.free()
        local_next_tokens.free()

    print("\n---------------------------------------------------------")
    print("Inference generation loop complete!")

    # Cleanup resources
    gpu_ctx.free_gpu_buffer(weights_gpu_buf)
    gpu_ctx.free_gpu_buffer(x_buf)
    gpu_ctx.free_gpu_buffer(norm_x_buf)
    gpu_ctx.free_gpu_buffer(norm_x_buf_1)
    gpu_ctx.free_gpu_buffer(final_norm_x_buf)
    gpu_ctx.free_gpu_buffer(q_buf)
    gpu_ctx.free_gpu_buffer(q_buf_1)
    gpu_ctx.free_gpu_buffer(k_vec_buf)
    gpu_ctx.free_gpu_buffer(k_vec_buf_1)
    gpu_ctx.free_gpu_buffer(v_vec_buf)
    gpu_ctx.free_gpu_buffer(v_vec_buf_1)
    gpu_ctx.free_gpu_buffer(attn_out_buf)
    gpu_ctx.free_gpu_buffer(attn_out_buf_1)
    gpu_ctx.free_gpu_buffer(proj_attn_out_buf)
    gpu_ctx.free_gpu_buffer(h1_buf)
    gpu_ctx.free_gpu_buffer(h2_buf)

    gpu_ctx.free_gpu_buffer(gpu_ctx.ring_buf)
    gpu_ctx.free_gpu_buffer(gpu_ctx.gart_buf)
    gpu_ctx.free_gpu_buffer(gpu_ctx.cwsr_buf)
    gpu_ctx.free_gpu_buffer(gpu_ctx.eop_buf)
    gpu_ctx.free_gpu_buffer(gpu_ctx.completion_buf)
    gpu_ctx.free_gpu_buffer(gpu_ctx.kernarg_buf)
    gpu_ctx.free_gpu_buffer(gpu_ctx.hsaco_buf)

    free_c(embed_addr)
    memptr_embed.free()
    out_tag_ptr.free()

    final_norm_weight.free()
    attn_scores.free()
    ffn_out_gate.free()
    ffn_out.free()
    k_cache.free()
    v_cache.free()
    w2_scratch.free()

    _ = c_close(kfd_fd)
    _ = c_close(drm_fd)
