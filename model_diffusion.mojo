from std.ffi import OwnedDLHandle
from std.memory import alloc
from std.collections import InlineArray
from std.math import sqrt, exp, cos, sin, tanh, log
from std.python import Python, PythonObject
import std.os
from std.algorithm import parallelize
from std.sys.info import num_physical_cores
from q4_embed import Q4Embedding
import std.time

# ==============================================================================
# KFD IOCTL Arguments, structures, and GPU Memory helper structures
# ==============================================================================

struct kfd_ioctl_acquire_vm_args:
    var drm_fd: UInt32
    var gpu_id: UInt32

    def __init__(out self, drm_fd: UInt32, gpu_id: UInt32):
        self.drm_fd = drm_fd
        self.gpu_id = gpu_id

struct kfd_ioctl_runtime_enable_args:
    var r_debug: UInt64
    var mode_mask: UInt32
    var capabilities_mask: UInt32

    def __init__(out self):
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

    def __init__(out self, va_addr: UInt64, size: UInt64, gpu_id: UInt32, flags: UInt32):
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

    def __init__(out self, handle: UInt64, device_ids_array_ptr: UInt64, n_devices: UInt32):
        self.handle = handle
        self.device_ids_array_ptr = device_ids_array_ptr
        self.n_devices = n_devices
        self.pad = 0

struct kfd_ioctl_unmap_memory_from_gpu_args:
    var handle: UInt64
    var device_ids_array_ptr: UInt64
    var n_devices: UInt32
    var pad: UInt32

    def __init__(out self, handle: UInt64, device_ids_array_ptr: UInt64, n_devices: UInt32):
        self.handle = handle
        self.device_ids_array_ptr = device_ids_array_ptr
        self.n_devices = n_devices
        self.pad = 0

struct kfd_ioctl_free_memory_of_gpu_args:
    var handle: UInt64

    def __init__(out self, handle: UInt64):
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

    def __init__(
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

struct kfd_ioctl_destroy_queue_args:
    var queue_id: UInt32
    var pad: UInt32

    def __init__(out self, queue_id: UInt32):
        self.queue_id = queue_id
        self.pad = 0

struct GPUMemoryBuffer(Copyable, Movable):
    var va_addr: Int
    var size: Int
    var handle: UInt64
    var cpu_addr: Int

    def __init__(out self):
        self.va_addr = 0
        self.size = 0
        self.handle = 0
        self.cpu_addr = 0

    def __init__(out self, va_addr: Int, size: Int, handle: UInt64, cpu_addr: Int):
        self.va_addr = va_addr
        self.size = size
        self.handle = handle
        self.cpu_addr = cpu_addr

    def __init__(out self, *, copy: Self):
        self.va_addr = copy.va_addr
        self.size = copy.size
        self.handle = copy.handle
        self.cpu_addr = copy.cpu_addr

    def __init__(out self, *, deinit take: Self):
        self.va_addr = take.va_addr
        self.size = take.size
        self.handle = take.handle
        self.cpu_addr = take.cpu_addr

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

    def __init__(
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

struct GPUContext(Movable):
    var has_queue: Bool
    var libc: OwnedDLHandle
    var cpu_fence_lib: OwnedDLHandle
    var kfd_fd: Int32
    var drm_fd: Int32
    var gpu_id: UInt32
    
    # libc function pointers
    var c_open: def(Int, Int32) thin abi("C") -> Int32
    var c_close: def(Int32) thin abi("C") -> Int32
    var c_ioctl: def(Int32, Int, Int) thin abi("C") -> Int32
    var c_mmap: def(Int, Int, Int32, Int32, Int32, Int) thin abi("C") -> Int
    var c_munmap: def(Int, Int) thin abi("C") -> Int32
    var c_read: def(Int32, Int, Int) thin abi("C") -> Int
    var c_errno_loc: def() thin abi("C") -> Int
    var cpu_sfence: def(Int32) thin abi("C") -> Int32
    var cpu_clflush: def(Int) thin abi("C") -> Int32
    var usleep_fn: def(UInt32) thin abi("C") -> Int32

    # Queue buffers and registers
    var ring_buf: GPUMemoryBuffer
    var gart_buf: GPUMemoryBuffer
    var cwsr_buf: GPUMemoryBuffer
    var eop_buf: GPUMemoryBuffer
    var hsaco_buf: GPUMemoryBuffer
    var kernarg_buf: GPUMemoryBuffer
    var completion_buf: GPUMemoryBuffer
    var packet_idx: UInt64
    
    var queue_id: UInt32
    var doorbell_mmap_addr: Int
    var doorbell_reg_addr: Int

    # Constants
    comptime AMDKFD_IOC_ACQUIRE_VM = 0x40084B15
    comptime AMDKFD_IOC_RUNTIME_ENABLE = 0xC0104B25
    comptime AMDKFD_IOC_ALLOC_MEMORY_OF_GPU = 0xC0284B16
    comptime AMDKFD_IOC_FREE_MEMORY_OF_GPU = 0x40084B17
    comptime AMDKFD_IOC_MAP_MEMORY_TO_GPU = 0xC0184B18
    comptime AMDKFD_IOC_UNMAP_MEMORY_FROM_GPU = 0xC0184B19
    comptime AMDKFD_IOC_CREATE_QUEUE = 0xC0604B02
    comptime AMDKFD_IOC_DESTROY_QUEUE = 0xC0084B03

    comptime PROT_NONE = 0
    comptime PROT_READ = 1
    comptime PROT_WRITE = 2
    comptime MAP_SHARED = 0x01
    comptime MAP_PRIVATE = 0x02
    comptime MAP_FIXED = 0x10
    comptime MAP_ANONYMOUS = 0x20

    comptime KFD_IOC_ALLOC_MEM_FLAGS_VRAM = 1 << 0
    comptime KFD_IOC_ALLOC_MEM_FLAGS_GTT = 1 << 1
    comptime KFD_IOC_ALLOC_MEM_FLAGS_WRITABLE = 1 << 31
    comptime KFD_IOC_ALLOC_MEM_FLAGS_EXECUTABLE = 1 << 30
    comptime KFD_IOC_ALLOC_MEM_FLAGS_PUBLIC = 1 << 29
    comptime KFD_IOC_ALLOC_MEM_FLAGS_NO_SUBSTITUTE = 1 << 28
    comptime KFD_IOC_ALLOC_MEM_FLAGS_COHERENT = 1 << 26
    comptime KFD_IOC_ALLOC_MEM_FLAGS_UNCACHED = 1 << 25

    def __init__(out self, init_queue: Bool = True) raises:
        self.has_queue = init_queue
        self.libc = OwnedDLHandle("libc.so.6")
        self.cpu_fence_lib = OwnedDLHandle("/home/phil/git/llm-mojo/scratch/libcpu_fence.so")
        
        self.c_open = self.libc.get_function[def(Int, Int32) thin abi("C") -> Int32]("open")
        self.c_close = self.libc.get_function[def(Int32) thin abi("C") -> Int32]("close")
        self.c_ioctl = self.libc.get_function[def(Int32, Int, Int) thin abi("C") -> Int32]("ioctl")
        self.c_mmap = self.libc.get_function[def(Int, Int, Int32, Int32, Int32, Int) thin abi("C") -> Int]("mmap")
        self.c_munmap = self.libc.get_function[def(Int, Int) thin abi("C") -> Int32]("munmap")
        self.c_read = self.libc.get_function[def(Int32, Int, Int) thin abi("C") -> Int]("read")
        self.c_errno_loc = self.libc.get_function[def() thin abi("C") -> Int]("__errno_location")
        self.cpu_sfence = self.cpu_fence_lib.get_function[def(Int32) thin abi("C") -> Int32]("cpu_sfence")
        self.cpu_clflush = self.cpu_fence_lib.get_function[def(Int) thin abi("C") -> Int32]("cpu_clflush")

        self.kfd_fd = -1
        self.drm_fd = -1
        self.gpu_id = 0
        self.queue_id = 0
        self.doorbell_mmap_addr = 0
        self.doorbell_reg_addr = 0
        self.packet_idx = 0
        self.ring_buf = GPUMemoryBuffer()
        self.gart_buf = GPUMemoryBuffer()
        self.cwsr_buf = GPUMemoryBuffer()
        self.eop_buf = GPUMemoryBuffer()
        self.hsaco_buf = GPUMemoryBuffer()
        self.kernarg_buf = GPUMemoryBuffer()
        self.completion_buf = GPUMemoryBuffer()
        self.usleep_fn = self.libc.get_function[def(UInt32) thin abi("C") -> Int32]("usleep")

        if not init_queue:
            return

        var kfd_path = "/dev/kfd"
        var kfd_path_bytes = kfd_path.as_bytes()
        var kfd_path_ptr = alloc[UInt8](len(kfd_path_bytes) + 1)
        for i in range(len(kfd_path_bytes)):
            kfd_path_ptr[i] = kfd_path_bytes[i]
        kfd_path_ptr[len(kfd_path_bytes)] = 0
        self.kfd_fd = self.c_open(kfd_path_ptr.__int__(), 2) # O_RDWR
        kfd_path_ptr.free()
        
        if self.kfd_fd < 0:
            raise Error("Failed to open /dev/kfd")

        var drm_path = "/dev/dri/renderD128"
        var drm_path_bytes = drm_path.as_bytes()
        var drm_path_ptr = alloc[UInt8](len(drm_path_bytes) + 1)
        for i in range(len(drm_path_bytes)):
            drm_path_ptr[i] = drm_path_bytes[i]
        drm_path_ptr[len(drm_path_bytes)] = 0
        self.drm_fd = self.c_open(drm_path_ptr.__int__(), 2) # O_RDWR
        drm_path_ptr.free()

        if self.drm_fd < 0:
            _ = self.c_close(self.kfd_fd)
            raise Error("Failed to open /dev/dri/renderD128")

        # Discover GPU ID from topology
        self.gpu_id = 0
        for node_idx in range(1, 16):
            var path = "/sys/class/kfd/kfd/topology/nodes/" + String(node_idx) + "/gpu_id"
            var path_bytes = path.as_bytes()
            var path_ptr = alloc[UInt8](len(path_bytes) + 1)
            for i in range(len(path_bytes)):
                path_ptr[i] = path_bytes[i]
            path_ptr[len(path_bytes)] = 0
            var fd = self.c_open(path_ptr.__int__(), 0)
            path_ptr.free()
            
            if fd >= 0:
                var buf = alloc[UInt8](64)
                var bytes_read = self.c_read(fd, buf.__int__(), 63)
                _ = self.c_close(fd)
                if bytes_read > 0:
                    buf[bytes_read] = 0
                    var val: Int = 0
                    for i in range(bytes_read):
                        var char_code = buf[i]
                        if char_code >= 48 and char_code <= 57:
                            val = val * 10 + (Int(char_code) - 48)
                    buf.free()
                    if val > 0:
                        self.gpu_id = UInt32(val)
                        break
                else:
                    buf.free()

        if self.gpu_id == 0:
            self.gpu_id = 21148 # default fallback

        # VM Acquisition
        var acq_args_ptr = alloc[kfd_ioctl_acquire_vm_args](1)
        acq_args_ptr[0] = kfd_ioctl_acquire_vm_args(drm_fd=UInt32(self.drm_fd), gpu_id=self.gpu_id)
        var r = self.c_ioctl(self.kfd_fd, self.AMDKFD_IOC_ACQUIRE_VM, acq_args_ptr.__int__())
        acq_args_ptr.free()
        if r < 0:
            _ = self.c_close(self.kfd_fd)
            _ = self.c_close(self.drm_fd)
            raise Error("GPU VM Acquisition failed!")

        # Enable Runtime
        var runtime_enable_ptr = alloc[kfd_ioctl_runtime_enable_args](1)
        runtime_enable_ptr[0] = kfd_ioctl_runtime_enable_args()
        r = self.c_ioctl(self.kfd_fd, self.AMDKFD_IOC_RUNTIME_ENABLE, runtime_enable_ptr.__int__())
        runtime_enable_ptr.free()
        if r < 0:
            _ = self.c_close(self.kfd_fd)
            _ = self.c_close(self.drm_fd)
            raise Error("GPU Runtime enable failed!")

        # Setup base flags
        var base_flags = (Self.KFD_IOC_ALLOC_MEM_FLAGS_WRITABLE |
                          Self.KFD_IOC_ALLOC_MEM_FLAGS_EXECUTABLE |
                          Self.KFD_IOC_ALLOC_MEM_FLAGS_NO_SUBSTITUTE)

        var gtt_uncached_flags = (base_flags |
                                  Self.KFD_IOC_ALLOC_MEM_FLAGS_GTT |
                                  Self.KFD_IOC_ALLOC_MEM_FLAGS_COHERENT |
                                  Self.KFD_IOC_ALLOC_MEM_FLAGS_UNCACHED |
                                  Self.KFD_IOC_ALLOC_MEM_FLAGS_PUBLIC)

        var vram_private_flags = (base_flags |
                                  Self.KFD_IOC_ALLOC_MEM_FLAGS_VRAM)

        # Setup local references for nested function capture
        var c_mmap = self.c_mmap
        var c_ioctl = self.c_ioctl
        var c_munmap = self.c_munmap
        var kfd_fd = self.kfd_fd
        var drm_fd = self.drm_fd
        var gpu_id = self.gpu_id
        
        var PROT_NONE = Self.PROT_NONE
        var PROT_READ = Self.PROT_READ
        var PROT_WRITE = Self.PROT_WRITE
        var MAP_SHARED = Self.MAP_SHARED
        var MAP_PRIVATE = Self.MAP_PRIVATE
        var MAP_FIXED = Self.MAP_FIXED
        var MAP_ANONYMOUS = Self.MAP_ANONYMOUS
        
        var AMDKFD_IOC_ALLOC_MEMORY_OF_GPU = Self.AMDKFD_IOC_ALLOC_MEMORY_OF_GPU
        var AMDKFD_IOC_FREE_MEMORY_OF_GPU = Self.AMDKFD_IOC_FREE_MEMORY_OF_GPU
        var AMDKFD_IOC_MAP_MEMORY_TO_GPU = Self.AMDKFD_IOC_MAP_MEMORY_TO_GPU
        var AMDKFD_IOC_UNMAP_MEMORY_FROM_GPU = Self.AMDKFD_IOC_UNMAP_MEMORY_FROM_GPU

        def alloc_gpu_buf(size: Int, flags: UInt32, cpu_access: Bool) capturing raises -> GPUMemoryBuffer:
            var va_addr = c_mmap(0, size, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0)
            if va_addr == -1 or va_addr == 0:
                raise Error("mmap failed")
            
            var alloc_args_ptr = alloc[kfd_ioctl_alloc_memory_of_gpu_args](1)
            alloc_args_ptr[0] = kfd_ioctl_alloc_memory_of_gpu_args(va_addr=UInt64(va_addr), size=UInt64(size), gpu_id=gpu_id, flags=flags)
            var io_res = c_ioctl(kfd_fd, AMDKFD_IOC_ALLOC_MEMORY_OF_GPU, alloc_args_ptr.__int__())
            if io_res < 0:
                alloc_args_ptr.free()
                _ = c_munmap(va_addr, size)
                raise Error("ALLOC_MEMORY failed")
            var handle = alloc_args_ptr[0].handle
            var mmap_offset = alloc_args_ptr[0].mmap_offset
            alloc_args_ptr.free()

            var cpu_addr = 0
            if cpu_access:
                cpu_addr = c_mmap(va_addr, size, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_FIXED, drm_fd, Int(mmap_offset))
                if cpu_addr == -1 or cpu_addr != va_addr:
                    raise Error("CPU mmap failed")

            var device_ids_ptr = alloc[UInt32](1)
            device_ids_ptr[0] = gpu_id
            var map_args_ptr = alloc[kfd_ioctl_map_memory_to_gpu_args](1)
            map_args_ptr[0] = kfd_ioctl_map_memory_to_gpu_args(handle=handle, device_ids_array_ptr=UInt64(device_ids_ptr.__int__()), n_devices=1)
            io_res = c_ioctl(kfd_fd, AMDKFD_IOC_MAP_MEMORY_TO_GPU, map_args_ptr.__int__())
            device_ids_ptr.free()
            map_args_ptr.free()
            if io_res < 0:
                raise Error("MAP failed")

            return GPUMemoryBuffer(va_addr, size, handle, cpu_addr)

        def free_gpu_buf(buf: GPUMemoryBuffer) capturing raises:
            var unmap_device_ids_ptr = alloc[UInt32](1)
            unmap_device_ids_ptr[0] = gpu_id
            var unmap_args_ptr = alloc[kfd_ioctl_unmap_memory_from_gpu_args](1)
            unmap_args_ptr[0] = kfd_ioctl_unmap_memory_from_gpu_args(handle=buf.handle, device_ids_array_ptr=UInt64(unmap_device_ids_ptr.__int__()), n_devices=1)
            _ = c_ioctl(kfd_fd, AMDKFD_IOC_UNMAP_MEMORY_FROM_GPU, unmap_args_ptr.__int__())
            unmap_device_ids_ptr.free()
            unmap_args_ptr.free()
            
            if buf.cpu_addr != 0:
                _ = c_munmap(buf.cpu_addr, buf.size)
            else:
                _ = c_munmap(buf.va_addr, buf.size)

            var free_args_ptr = alloc[kfd_ioctl_free_memory_of_gpu_args](1)
            free_args_ptr[0] = kfd_ioctl_free_memory_of_gpu_args(handle=buf.handle)
            _ = c_ioctl(kfd_fd, AMDKFD_IOC_FREE_MEMORY_OF_GPU, free_args_ptr.__int__())
            free_args_ptr.free()

        # Initialize ring, gart, cwsr, eop buffers
        var ring_buf = alloc_gpu_buf(64 * 1024, UInt32(gtt_uncached_flags), cpu_access=True)
        var gart_buf = alloc_gpu_buf(4096, UInt32(gtt_uncached_flags), cpu_access=True)
        var cwsr_buf = alloc_gpu_buf(2801664, UInt32(vram_private_flags), cpu_access=False)
        var eop_buf = alloc_gpu_buf(4096, UInt32(vram_private_flags), cpu_access=False)

        # Initialize ring buffer
        var ring_ptr = UnsafePointer[UInt32, MutUntrackedOrigin](unsafe_from_address=ring_buf.cpu_addr)
        for i in range((64 * 1024) // 4):
            ring_ptr[i] = 0

        # Initialize GART buffer
        var gart_u32_ptr = UnsafePointer[UInt32, MutUntrackedOrigin](unsafe_from_address=gart_buf.cpu_addr)
        for i in range(4096 // 4):
            gart_u32_ptr[i] = 0

        var sig_cpu_addr = gart_buf.cpu_addr + 512
        var sig_va_addr = gart_buf.va_addr + 512

        var sig_u64_ptr = UnsafePointer[UInt64, MutUntrackedOrigin](unsafe_from_address=sig_cpu_addr)
        for i in range(8):
            sig_u64_ptr[i] = 0

        UnsafePointer[Int64, MutUntrackedOrigin](unsafe_from_address=sig_cpu_addr + 0)[0] = -1
        UnsafePointer[UInt64, MutUntrackedOrigin](unsafe_from_address=sig_cpu_addr + 8)[0] = 0
        UnsafePointer[UInt64, MutUntrackedOrigin](unsafe_from_address=sig_cpu_addr + 48)[0] = UInt64(gart_buf.va_addr)

        _ = self.cpu_clflush(sig_cpu_addr)
        _ = self.cpu_sfence(0)

        var q_base = gart_buf.cpu_addr
        UnsafePointer[UInt32, MutUntrackedOrigin](unsafe_from_address=q_base + 0)[0] = 2       # queue_type = AQL
        UnsafePointer[UInt32, MutUntrackedOrigin](unsafe_from_address=q_base + 4)[0] = 1       # features
        UnsafePointer[UInt64, MutUntrackedOrigin](unsafe_from_address=q_base + 8)[0] = UInt64(ring_buf.va_addr)
        UnsafePointer[UInt64, MutUntrackedOrigin](unsafe_from_address=q_base + 16)[0] = UInt64(sig_va_addr)
        UnsafePointer[UInt32, MutUntrackedOrigin](unsafe_from_address=q_base + 24)[0] = 1024   # size

        UnsafePointer[UInt32, MutUntrackedOrigin](unsafe_from_address=q_base + 72)[0] = 7
        UnsafePointer[UInt32, MutUntrackedOrigin](unsafe_from_address=q_base + 76)[0] = 39
        UnsafePointer[UInt32, MutUntrackedOrigin](unsafe_from_address=q_base + 136)[0] = 128
        UnsafePointer[UInt32, MutUntrackedOrigin](unsafe_from_address=q_base + 180)[0] = 10

        _ = self.cpu_clflush(q_base)
        _ = self.cpu_clflush(q_base + 64)
        _ = self.cpu_clflush(q_base + 128)
        _ = self.cpu_clflush(q_base + 180)
        _ = self.cpu_sfence(0)

        # Create AQL queue in KFD
        var create_args_ptr = alloc[kfd_ioctl_create_queue_args](1)
        create_args_ptr[0] = kfd_ioctl_create_queue_args(
            ring_base_address=UInt64(ring_buf.va_addr),
            write_pointer_address=UInt64(gart_buf.va_addr + 56),
            read_pointer_address=UInt64(gart_buf.va_addr + 128),
            ring_size=UInt32(64 * 1024),
            gpu_id=self.gpu_id,
            queue_type=2,
            queue_percentage=100,
            queue_priority=7,
            eop_buffer_address=UInt64(eop_buf.va_addr),
            eop_buffer_size=4096,
            ctx_save_restore_address=UInt64(cwsr_buf.va_addr),
            ctx_save_restore_size=2789376,
            ctl_stack_size=4096,
            sdma_engine_id=0
        )
        r = self.c_ioctl(self.kfd_fd, Self.AMDKFD_IOC_CREATE_QUEUE, create_args_ptr.__int__())
        if r < 0:
            var err = UnsafePointer[Int32, MutUntrackedOrigin](unsafe_from_address=self.c_errno_loc())[0]
            free_gpu_buf(eop_buf)
            free_gpu_buf(cwsr_buf)
            free_gpu_buf(gart_buf)
            free_gpu_buf(ring_buf)
            _ = self.c_close(self.kfd_fd)
            _ = self.c_close(self.drm_fd)
            create_args_ptr.free()
            raise Error("KFD Queue creation failed with errno: " + String(err))

        self.queue_id = create_args_ptr[0].queue_id
        var doorbell_offset = create_args_ptr[0].doorbell_offset
        create_args_ptr.free()

        var doorbells_base = doorbell_offset & (~0x1fff)
        var doorbell_page_size = 0x2000
        self.doorbell_mmap_addr = self.c_mmap(0, doorbell_page_size, PROT_READ | PROT_WRITE, MAP_SHARED, self.kfd_fd, Int(doorbells_base))
        if self.doorbell_mmap_addr == -1:
            raise Error("Doorbell page mmap failed")
        self.doorbell_reg_addr = self.doorbell_mmap_addr + (Int(doorbell_offset) - Int(doorbells_base))

        UnsafePointer[UInt64, MutUntrackedOrigin](unsafe_from_address=sig_cpu_addr + 8)[0] = UInt64(self.doorbell_reg_addr)
        _ = self.cpu_clflush(sig_cpu_addr)
        _ = self.cpu_sfence(0)

        # Successfully created queue, now transfer buffer ownership to self fields
        self.ring_buf = ring_buf^
        self.gart_buf = gart_buf^
        self.cwsr_buf = cwsr_buf^
        self.eop_buf = eop_buf^
        self.completion_buf = alloc_gpu_buf(1024 * 64, UInt32(gtt_uncached_flags), cpu_access=True)
        self.kernarg_buf = alloc_gpu_buf(1024 * 256, UInt32(gtt_uncached_flags), cpu_access=True)
        self.packet_idx = 0
        self.usleep_fn = self.libc.get_function[def(UInt32) thin abi("C") -> Int32]("usleep")

        # Load GCN compiled ELF shader
        var hsaco_buf = alloc_gpu_buf(16384, UInt32(gtt_uncached_flags), cpu_access=True)
        self.hsaco_buf = hsaco_buf^
        var hsaco_u32_ptr = UnsafePointer[UInt32, MutUntrackedOrigin](unsafe_from_address=self.hsaco_buf.cpu_addr)
        for i in range(16384 // 4):
            hsaco_u32_ptr[i] = 0

        var hsaco_file = "/home/phil/git/llm-mojo/scratch/q8_matmul_lds.hsaco"
        var hsaco_file_bytes = hsaco_file.as_bytes()
        var hsaco_file_ptr = alloc[UInt8](len(hsaco_file_bytes) + 1)
        for i in range(len(hsaco_file_bytes)):
            hsaco_file_ptr[i] = hsaco_file_bytes[i]
        hsaco_file_ptr[len(hsaco_file_bytes)] = 0
        var hsaco_fd = self.c_open(hsaco_file_ptr.__int__(), 0)
        hsaco_file_ptr.free()
        if hsaco_fd < 0:
            raise Error("Failed to open q8_matmul_lds.hsaco")

        var temp_file_buf = alloc[UInt8](8192)
        var bytes_read = self.c_read(hsaco_fd, temp_file_buf.__int__(), 8192)
        _ = self.c_close(hsaco_fd)
        if bytes_read <= 0:
            temp_file_buf.free()
            raise Error("Failed to read any bytes from q8_matmul_lds.hsaco")

        # Map ELF segments to virtual memory layout (do not copy flat):
        # Segment 1 (RO): offset 0x0, virt_addr 0x0, size 0xac0 (covering KD descriptor)
        var segment1_src = temp_file_buf
        var segment1_dst = UnsafePointer[UInt8, MutUntrackedOrigin](unsafe_from_address=self.hsaco_buf.cpu_addr + 0)
        for i in range(0xac0):
            segment1_dst[i] = segment1_src[i]

        var segment2_src = temp_file_buf + 0xb00
        var segment2_dst = UnsafePointer[UInt8, MutUntrackedOrigin](unsafe_from_address=self.hsaco_buf.cpu_addr + 0x1b00)
        for i in range(0x700):
            segment2_dst[i] = segment2_src[i]

        var segment3_src = temp_file_buf + 0x1200
        var segment3_dst = UnsafePointer[UInt8, MutUntrackedOrigin](unsafe_from_address=self.hsaco_buf.cpu_addr + 0x3200)
        for i in range(0x70):
            segment3_dst[i] = segment3_src[i]

        temp_file_buf.free()

    def __init__(out self, *, deinit take: Self):
        self.has_queue = take.has_queue
        self.libc = take.libc^
        self.cpu_fence_lib = take.cpu_fence_lib^
        self.kfd_fd = take.kfd_fd
        self.drm_fd = take.drm_fd
        self.gpu_id = take.gpu_id
        self.c_open = take.c_open
        self.c_close = take.c_close
        self.c_ioctl = take.c_ioctl
        self.c_mmap = take.c_mmap
        self.c_munmap = take.c_munmap
        self.c_read = take.c_read
        self.c_errno_loc = take.c_errno_loc
        self.cpu_sfence = take.cpu_sfence
        self.cpu_clflush = take.cpu_clflush
        self.usleep_fn = take.usleep_fn
        self.ring_buf = take.ring_buf^
        self.gart_buf = take.gart_buf^
        self.cwsr_buf = take.cwsr_buf^
        self.eop_buf = take.eop_buf^
        self.hsaco_buf = take.hsaco_buf^
        self.kernarg_buf = take.kernarg_buf^
        self.completion_buf = take.completion_buf^
        self.packet_idx = take.packet_idx
        self.queue_id = take.queue_id
        self.doorbell_mmap_addr = take.doorbell_mmap_addr
        self.doorbell_reg_addr = take.doorbell_reg_addr

    def __del__(deinit self):
        if self.has_queue:
            # Destroy queue first
            var destroy_args_ptr = alloc[kfd_ioctl_destroy_queue_args](1)
            destroy_args_ptr[0] = kfd_ioctl_destroy_queue_args(queue_id=self.queue_id)
            _ = self.c_ioctl(self.kfd_fd, self.AMDKFD_IOC_DESTROY_QUEUE, destroy_args_ptr.__int__())
            destroy_args_ptr.free()

            _ = self.c_munmap(self.doorbell_mmap_addr, 0x2000)

            # Free all queue buffers
            try:
                self.free_gpu_buffer(self.kernarg_buf)
                self.free_gpu_buffer(self.completion_buf)
                self.free_gpu_buffer(self.hsaco_buf)
                self.free_gpu_buffer(self.eop_buf)
                self.free_gpu_buffer(self.cwsr_buf)
                self.free_gpu_buffer(self.gart_buf)
                self.free_gpu_buffer(self.ring_buf)
            except:
                pass

            _ = self.c_close(self.kfd_fd)
            _ = self.c_close(self.drm_fd)

    def alloc_gpu_buffer(self, size: Int, flags: UInt32, cpu_access: Bool) raises -> GPUMemoryBuffer:
        if not self.has_queue:
            var cpu_addr = self.c_mmap(0, size, self.PROT_READ | self.PROT_WRITE, self.MAP_PRIVATE | self.MAP_ANONYMOUS, -1, 0)
            if cpu_addr == -1 or cpu_addr == 0:
                raise Error("CPU-only mmap failed")
            return GPUMemoryBuffer(cpu_addr, size, 0, cpu_addr)

        var va_addr = self.c_mmap(0, size, self.PROT_NONE, self.MAP_PRIVATE | self.MAP_ANONYMOUS, -1, 0)
        if va_addr == -1 or va_addr == 0:
            raise Error("mmap failed")
        
        var alloc_args_ptr = alloc[kfd_ioctl_alloc_memory_of_gpu_args](1)
        alloc_args_ptr[0] = kfd_ioctl_alloc_memory_of_gpu_args(va_addr=UInt64(va_addr), size=UInt64(size), gpu_id=self.gpu_id, flags=flags)
        var io_res = self.c_ioctl(self.kfd_fd, self.AMDKFD_IOC_ALLOC_MEMORY_OF_GPU, alloc_args_ptr.__int__())
        if io_res < 0:
            alloc_args_ptr.free()
            _ = self.c_munmap(va_addr, size)
            raise Error("ALLOC_MEMORY failed")
        var handle = alloc_args_ptr[0].handle
        var mmap_offset = alloc_args_ptr[0].mmap_offset
        alloc_args_ptr.free()

        var cpu_addr = 0
        if cpu_access:
            cpu_addr = self.c_mmap(va_addr, size, self.PROT_READ | self.PROT_WRITE, self.MAP_SHARED | self.MAP_FIXED, self.drm_fd, Int(mmap_offset))
            if cpu_addr == -1 or cpu_addr != va_addr:
                raise Error("CPU mmap failed")

        var device_ids_ptr = alloc[UInt32](1)
        device_ids_ptr[0] = self.gpu_id
        var map_args_ptr = alloc[kfd_ioctl_map_memory_to_gpu_args](1)
        map_args_ptr[0] = kfd_ioctl_map_memory_to_gpu_args(handle=handle, device_ids_array_ptr=UInt64(device_ids_ptr.__int__()), n_devices=1)
        io_res = self.c_ioctl(self.kfd_fd, self.AMDKFD_IOC_MAP_MEMORY_TO_GPU, map_args_ptr.__int__())
        device_ids_ptr.free()
        map_args_ptr.free()
        if io_res < 0:
            raise Error("MAP failed")

        return GPUMemoryBuffer(va_addr, size, handle, cpu_addr)

    def free_gpu_buffer(self, buf: GPUMemoryBuffer) raises:
        if not self.has_queue:
            _ = self.c_munmap(buf.cpu_addr, buf.size)
            return

        var unmap_device_ids_ptr = alloc[UInt32](1)
        unmap_device_ids_ptr[0] = self.gpu_id
        var unmap_args_ptr = alloc[kfd_ioctl_unmap_memory_from_gpu_args](1)
        unmap_args_ptr[0] = kfd_ioctl_unmap_memory_from_gpu_args(handle=buf.handle, device_ids_array_ptr=UInt64(unmap_device_ids_ptr.__int__()), n_devices=1)
        _ = self.c_ioctl(self.kfd_fd, self.AMDKFD_IOC_UNMAP_MEMORY_FROM_GPU, unmap_args_ptr.__int__())
        unmap_device_ids_ptr.free()
        unmap_args_ptr.free()
        
        if buf.cpu_addr != 0:
            _ = self.c_munmap(buf.cpu_addr, buf.size)
        else:
            _ = self.c_munmap(buf.va_addr, buf.size)

        var free_args_ptr = alloc[kfd_ioctl_free_memory_of_gpu_args](1)
        free_args_ptr[0] = kfd_ioctl_free_memory_of_gpu_args(handle=buf.handle)
        _ = self.c_ioctl(self.kfd_fd, self.AMDKFD_IOC_FREE_MEMORY_OF_GPU, free_args_ptr.__int__())
        free_args_ptr.free()

    def get_gpu_va(self, cpu_addr: Int) -> UInt64:
        return UInt64(cpu_addr)

    def enqueue_matmul(
        mut self,
        W_packed_addr: Int,
        scales_addr: Int,
        x_addr: Int,
        out_ptr_addr: Int,
        c_in: Int,
        c_out: Int,
        batch_size: Int = 1,
        flush_cache: Bool = True,
    ) raises -> Int:
        var cpu_sfence = self.cpu_sfence
        var cpu_clflush = self.cpu_clflush

        # Flush CPU cache lines for input vector x to ensure GPU reads current activations
        if flush_cache:
            var input_size_bytes = c_in * batch_size * 4
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
        var completion_ptr = UnsafePointer[Int32, MutUntrackedOrigin](unsafe_from_address=completion_slot_cpu_addr)
        completion_ptr[0] = 0
        completion_ptr[1] = 0 # reset atomic counter to 0!
        _ = cpu_sfence(0)
        _ = cpu_clflush(completion_slot_cpu_addr)

        # Translate CPU pointers to GPU Virtual Addresses
        var gpu_W_packed_va = self.get_gpu_va(W_packed_addr)
        var gpu_scales_va = self.get_gpu_va(scales_addr)
        var gpu_x_va = self.get_gpu_va(x_addr)
        var gpu_out_ptr_va = self.get_gpu_va(out_ptr_addr)

        # 2. Pack kernel arguments into this slot's kernarg memory
        var kernarg_slot_cpu_addr = self.kernarg_buf.cpu_addr + slot * 256
        var kernarg_slot_va_addr = self.kernarg_buf.va_addr + slot * 256

        var kernarg_p = UnsafePointer[UInt64, MutUntrackedOrigin](unsafe_from_address=kernarg_slot_cpu_addr)
        kernarg_p[0] = gpu_W_packed_va
        kernarg_p[1] = gpu_scales_va
        kernarg_p[2] = gpu_x_va
        kernarg_p[3] = gpu_out_ptr_va
        kernarg_p[4] = UInt64(completion_slot_va_addr)

        var num_row_tiles = (c_out + 15) // 16
        var num_token_tiles = (batch_size + 15) // 16
        var num_workgroups = num_row_tiles * num_token_tiles
        if num_workgroups < 1:
            num_workgroups = 1
        var grid_stride = num_workgroups

        var kernarg_u32_p = UnsafePointer[UInt32, MutUntrackedOrigin](unsafe_from_address=kernarg_slot_cpu_addr + 40)
        kernarg_u32_p[0] = UInt32(c_in)
        kernarg_u32_p[1] = UInt32(c_out)
        kernarg_u32_p[2] = UInt32(grid_stride) # grid_stride in workgroups
        kernarg_u32_p[3] = UInt32(num_workgroups) # num_workgroups
        kernarg_u32_p[4] = UInt32(batch_size)
        
        _ = cpu_sfence(0)
        _ = cpu_clflush(kernarg_slot_cpu_addr)

        # 3. Format HSA AQL Dispatch Packet (64 bytes)
        var packet = hsa_kernel_dispatch_packet_t(
            header=5378,              # Type=2 (Kernel Dispatch), Barrier=1, Acquire=2 (System), Release=2 (System)
            setup=1,                 # 1D dimensions
            workgroup_size_x=256,    # 256 threads in workgroup
            workgroup_size_y=1,
            workgroup_size_z=1,
            grid_size_x=UInt32(num_workgroups * 256), # Grid size = num_workgroups * 256
            grid_size_y=1,
            grid_size_z=1,
            private_segment_size=0,
            group_segment_size=2560,
            kernel_object=UInt64(self.hsaco_buf.va_addr + 0xa80), # KD address at 0xa80!
            kernarg_address=UInt64(kernarg_slot_va_addr),
            completion_signal=0
        )

        # 4. Write dispatch packet to circular ring buffer
        var ring_packet_ptr = UnsafePointer[hsa_kernel_dispatch_packet_t, MutUntrackedOrigin](unsafe_from_address=self.ring_buf.cpu_addr)
        ring_packet_ptr[slot] = packet^

        # Update write pointer in GART memory
        var wptr_ptr = UnsafePointer[UInt64, MutUntrackedOrigin](unsafe_from_address=self.gart_buf.cpu_addr + 56)
        self.packet_idx += 1
        wptr_ptr[0] = self.packet_idx

        # Memory fences & Flush cache lines for written data
        _ = cpu_sfence(0)
        _ = cpu_clflush(self.ring_buf.cpu_addr + slot * 64)
        _ = cpu_clflush(self.gart_buf.cpu_addr + 56)
        _ = cpu_sfence(0) # Ensure GART writes settle before ringing doorbell

        # Ring doorbell! Write 0-based packet_idx value to mapped doorbell register
        var doorbell_ptr = UnsafePointer[UInt64, MutUntrackedOrigin](unsafe_from_address=self.doorbell_reg_addr)
        doorbell_ptr[0] = self.packet_idx - 1
        _ = cpu_sfence(0) # Flush doorbell write through PCIe WC buffers

        return slot

    def wait_matmul(
        self,
        slot: Int,
        out_ptr_addr: Int,
        c_out: Int,
        batch_size: Int = 1,
        flush_cache: Bool = True,
    ) raises:
        var cpu_sfence = self.cpu_sfence
        var cpu_clflush = self.cpu_clflush
        var completion_slot_cpu_addr = self.completion_buf.cpu_addr + slot * 64
        var completion_ptr = UnsafePointer[Int32, MutUntrackedOrigin](unsafe_from_address=completion_slot_cpu_addr)

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
                var current_wptr = UnsafePointer[UInt64, MutUntrackedOrigin](unsafe_from_address=self.gart_buf.cpu_addr + 56)[0]
                var current_rptr = UnsafePointer[UInt64, MutUntrackedOrigin](unsafe_from_address=self.gart_buf.cpu_addr + 128)[0]
                print("GPU Execution Timeout details:")
                print("  Attempts reached:", attempts)
                print("  GART wptr:", current_wptr, "| GART rptr:", current_rptr)
                print("  Completion flag in memory:", completion_ptr[0])
                print("  Atomic counter in memory:", completion_ptr[1])
                raise Error("GPU execution timed out in wait_matmul!")

        # Flush/invalidate CPU cache lines for output vector out to ensure CPU reads the GPU's results
        if flush_cache:
            var output_size_bytes = c_out * batch_size * 4
            var base_out = (out_ptr_addr // 64) * 64
            var end_out = out_ptr_addr + output_size_bytes
            while base_out < end_out:
                _ = cpu_clflush(base_out)
                base_out += 64

# -----------------------------------------------------------------------------
# 1. Model Configuration & Memory Offsets
# -----------------------------------------------------------------------------

struct MoEConfig[
    embed_dim: Int = 2816,
    standard_mlp_dim: Int = 2112,
    expert_mlp_dim: Int = 704,
    num_experts: Int = 128,
    top_k_experts: Int = 8,
    num_layers: Int = 30,
    vocab_size: Int = 32768,
    max_seq_len: Int = 512,
]:
    @always_inline
    def is_global_layer(self, layer_idx: Int) -> Bool:
        return layer_idx % 6 == 5

    @always_inline
    def get_layer_aligned_size(self, layer_idx: Int) -> Int:
        var is_global = self.is_global_layer(layer_idx)
        var layer_heads = 16
        var layer_head_dim = 512 if is_global else 256
        var layer_kv_heads = 2 if is_global else 8
        
        var wq_size = layer_heads * layer_head_dim * Self.embed_dim
        var wk_size = layer_kv_heads * layer_head_dim * Self.embed_dim
        var wv_size = 0 if is_global else (layer_kv_heads * layer_head_dim * Self.embed_dim)
        var wo_size = Self.embed_dim * layer_heads * layer_head_dim
        var w_gate_size = Self.standard_mlp_dim * Self.embed_dim
        var w_up_size = Self.standard_mlp_dim * Self.embed_dim
        var w_down_size = Self.embed_dim * Self.standard_mlp_dim
        var w_router_size = Self.num_experts * Self.embed_dim
        var experts_gate_up_size = Self.num_experts * (Self.expert_mlp_dim * 2) * Self.embed_dim
        var experts_down_size = Self.num_experts * Self.embed_dim * Self.expert_mlp_dim
        
        var packed_weights_size = (
            wq_size + wk_size + wv_size + wo_size +
            w_gate_size + w_up_size + w_down_size +
            w_router_size + experts_gate_up_size + experts_down_size
        ) // 2
        
        var scales_count = (
            wq_size + wk_size + wv_size + wo_size +
            w_gate_size + w_up_size + w_down_size +
            w_router_size + experts_gate_up_size + experts_down_size
        ) // 32
        var scales_size_bytes = scales_count * 4
        
        var input_layernorm_size = Self.embed_dim
        var post_attention_layernorm_size = Self.embed_dim
        var pre_feedforward_layernorm_size = Self.embed_dim
        var pre_feedforward_layernorm_2_size = Self.embed_dim
        var post_feedforward_layernorm_size = Self.embed_dim
        var post_feedforward_layernorm_1_size = Self.embed_dim
        var post_feedforward_layernorm_2_size = Self.embed_dim
        
        var q_norm_size = layer_head_dim
        var k_norm_size = layer_head_dim
        var router_scale_size = Self.embed_dim
        var router_per_expert_scale_size = 128
        var layer_scalar_size = 1
        
        var unchanged_size_bytes = (
            input_layernorm_size + post_attention_layernorm_size +
            pre_feedforward_layernorm_size + pre_feedforward_layernorm_2_size +
            post_feedforward_layernorm_size + post_feedforward_layernorm_1_size +
            post_feedforward_layernorm_2_size +
            q_norm_size + k_norm_size +
            router_scale_size + router_per_expert_scale_size + layer_scalar_size
        ) * 4
        
        var raw_bytes_size = packed_weights_size + scales_size_bytes + unchanged_size_bytes
        return ((raw_bytes_size + 4095) // 4096) * 4096

    def get_layer_offset(self, layer_idx: Int) -> Int:
        var offset = 0
        for i in range(layer_idx):
            offset += self.get_layer_aligned_size(i)
        return offset

    def get_q8_attn_size(self, layer_idx: Int) -> Int:
        var is_global = self.is_global_layer(layer_idx)
        var layer_heads = 16
        var layer_head_dim = 512 if is_global else 256
        var layer_kv_heads = 2 if is_global else 8
        
        var wq_size = layer_heads * layer_head_dim * Self.embed_dim
        var wk_size = layer_kv_heads * layer_head_dim * Self.embed_dim
        var wv_size = 0 if is_global else (layer_kv_heads * layer_head_dim * Self.embed_dim)
        var wo_size = Self.embed_dim * layer_heads * layer_head_dim
        
        var raw_size = wq_size + wk_size + wv_size + wo_size
        return ((raw_size + 4095) // 4096) * 4096

    def get_q8_attn_offset(self, layer_idx: Int) -> Int:
        var offset = 0
        for i in range(layer_idx):
            offset += self.get_q8_attn_size(i)
        return offset

    @always_inline
    def get_layer_weights(self, layer_idx: Int, base_ptr: UnsafePointer[UInt8, _], q8_attn_base: Int = 0) -> MoELayerWeights:
        var is_global = self.is_global_layer(layer_idx)
        var layer_heads = 16
        var layer_head_dim = 512 if is_global else 256
        var layer_kv_heads = 2 if is_global else 8
        
        var wq_size = layer_heads * layer_head_dim * Self.embed_dim
        var wk_size = layer_kv_heads * layer_head_dim * Self.embed_dim
        var wv_size = 0 if is_global else (layer_kv_heads * layer_head_dim * Self.embed_dim)
        var wo_size = Self.embed_dim * layer_heads * layer_head_dim
        var w_gate_size = Self.standard_mlp_dim * Self.embed_dim
        var w_up_size = Self.standard_mlp_dim * Self.embed_dim
        var w_down_size = Self.embed_dim * Self.standard_mlp_dim
        var w_router_size = Self.num_experts * Self.embed_dim
        var experts_gate_up_size = Self.num_experts * (Self.expert_mlp_dim * 2) * Self.embed_dim
        var experts_down_size = Self.num_experts * Self.embed_dim * Self.expert_mlp_dim
        
        var wq_size_bytes = wq_size // 2
        var wk_size_bytes = wk_size // 2
        var wv_size_bytes = wv_size // 2
        var wo_size_bytes = wo_size // 2
        var w_gate_size_bytes = w_gate_size // 2
        var w_up_size_bytes = w_up_size // 2
        var w_down_size_bytes = w_down_size // 2
        var w_router_size_bytes = w_router_size // 2
        var experts_gate_up_size_bytes = experts_gate_up_size // 2
        var experts_down_size_bytes = experts_down_size // 2
        
        var wq_packed = base_ptr
        var wk_packed = wq_packed + wq_size_bytes
        var wv_packed = wk_packed + wk_size_bytes
        var wo_packed = wv_packed + wv_size_bytes
        var w_gate_packed = wo_packed + wo_size_bytes
        var w_up_packed = w_gate_packed + w_gate_size_bytes
        var w_down_packed = w_up_packed + w_up_size_bytes
        var w_router_packed = w_down_packed + w_down_size_bytes
        var experts_gate_up_packed = w_router_packed + w_router_size_bytes
        var experts_down_packed = experts_gate_up_packed + experts_gate_up_size_bytes
        
        var scales_base = (experts_down_packed + experts_down_size_bytes).bitcast[Float32]()
        
        var c_blocks_q = wq_size // 32
        var c_blocks_k = wk_size // 32
        var c_blocks_v = wv_size // 32
        var c_blocks_o = wo_size // 32
        var c_blocks_w_gate = w_gate_size // 32
        var c_blocks_w_up = w_up_size // 32
        var c_blocks_w_down = w_down_size // 32
        var c_blocks_w_router = w_router_size // 32
        var c_blocks_experts_gate_up = experts_gate_up_size // 32
        var c_blocks_experts_down = experts_down_size // 32
        
        var wq_scales = scales_base
        var wk_scales = wq_scales + c_blocks_q
        var wv_scales = wk_scales + c_blocks_k
        var wo_scales = wv_scales + c_blocks_v
        var w_gate_scales = wo_scales + c_blocks_o
        var w_up_scales = w_gate_scales + c_blocks_w_gate
        var w_down_scales = w_up_scales + c_blocks_w_up
        var w_router_scales = w_down_scales + c_blocks_w_down
        var experts_gate_up_scales = w_router_scales + c_blocks_w_router
        var experts_down_scales = experts_gate_up_scales + c_blocks_experts_gate_up
        
        var unchanged_base = experts_down_scales + c_blocks_experts_down
        
        var input_layernorm = unchanged_base
        var post_attention_layernorm = input_layernorm + Self.embed_dim
        var pre_feedforward_layernorm = post_attention_layernorm + Self.embed_dim
        var pre_feedforward_layernorm_2 = pre_feedforward_layernorm + Self.embed_dim
        var post_feedforward_layernorm = pre_feedforward_layernorm_2 + Self.embed_dim
        var post_feedforward_layernorm_1 = post_feedforward_layernorm + Self.embed_dim
        var post_feedforward_layernorm_2 = post_feedforward_layernorm_1 + Self.embed_dim
        var q_norm = post_feedforward_layernorm_2 + Self.embed_dim
        var k_norm = q_norm + layer_head_dim
        var router_scale = k_norm + layer_head_dim
        var router_per_expert_scale = router_scale + Self.embed_dim
        var layer_scalar = router_per_expert_scale + 128

        var wq_q8_addr = 0
        var wk_q8_addr = 0
        var wv_q8_addr = 0
        var wo_q8_addr = 0
        
        if q8_attn_base != 0:
            var layer_q8_offset = self.get_q8_attn_offset(layer_idx)
            wq_q8_addr = q8_attn_base + layer_q8_offset
            wk_q8_addr = wq_q8_addr + wq_size
            wv_q8_addr = wk_q8_addr + wk_size
            wo_q8_addr = wv_q8_addr + wv_size
        
        return MoELayerWeights(
            UnsafePointer[UInt8, MutExternalOrigin](unsafe_from_address=wq_packed.__int__()),
            UnsafePointer[UInt8, MutExternalOrigin](unsafe_from_address=wk_packed.__int__()),
            UnsafePointer[UInt8, MutExternalOrigin](unsafe_from_address=wv_packed.__int__()),
            UnsafePointer[UInt8, MutExternalOrigin](unsafe_from_address=wo_packed.__int__()),
            UnsafePointer[Scalar[DType.int8], MutExternalOrigin](unsafe_from_address=wq_q8_addr),
            UnsafePointer[Scalar[DType.int8], MutExternalOrigin](unsafe_from_address=wk_q8_addr),
            UnsafePointer[Scalar[DType.int8], MutExternalOrigin](unsafe_from_address=wv_q8_addr),
            UnsafePointer[Scalar[DType.int8], MutExternalOrigin](unsafe_from_address=wo_q8_addr),
            UnsafePointer[UInt8, MutExternalOrigin](unsafe_from_address=w_gate_packed.__int__()),
            UnsafePointer[UInt8, MutExternalOrigin](unsafe_from_address=w_up_packed.__int__()),
            UnsafePointer[UInt8, MutExternalOrigin](unsafe_from_address=w_down_packed.__int__()),
            UnsafePointer[UInt8, MutExternalOrigin](unsafe_from_address=w_router_packed.__int__()),
            UnsafePointer[UInt8, MutExternalOrigin](unsafe_from_address=experts_gate_up_packed.__int__()),
            UnsafePointer[UInt8, MutExternalOrigin](unsafe_from_address=experts_down_packed.__int__()),
            UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=wq_scales.__int__()),
            UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=wk_scales.__int__()),
            UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=wv_scales.__int__()),
            UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=wo_scales.__int__()),
            UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=w_gate_scales.__int__()),
            UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=w_up_scales.__int__()),
            UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=w_down_scales.__int__()),
            UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=w_router_scales.__int__()),
            UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=experts_gate_up_scales.__int__()),
            UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=experts_down_scales.__int__()),
            UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=input_layernorm.__int__()),
            UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=post_attention_layernorm.__int__()),
            UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=pre_feedforward_layernorm.__int__()),
            UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=pre_feedforward_layernorm_2.__int__()),
            UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=post_feedforward_layernorm.__int__()),
            UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=post_feedforward_layernorm_1.__int__()),
            UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=post_feedforward_layernorm_2.__int__()),
            UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=q_norm.__int__()),
            UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=k_norm.__int__()),
            UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=router_scale.__int__()),
            UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=router_per_expert_scale.__int__()),
            UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=layer_scalar.__int__())
        )

    def __init__(out self):
        pass


# -----------------------------------------------------------------------------
# 2. Structural Layer Weights Representation
# -----------------------------------------------------------------------------

struct MoELayerWeights:
    var wq_packed: UnsafePointer[UInt8, MutExternalOrigin]
    var wk_packed: UnsafePointer[UInt8, MutExternalOrigin]
    var wv_packed: UnsafePointer[UInt8, MutExternalOrigin]
    var wo_packed: UnsafePointer[UInt8, MutExternalOrigin]
    var wq_q8: UnsafePointer[Scalar[DType.int8], MutExternalOrigin]
    var wk_q8: UnsafePointer[Scalar[DType.int8], MutExternalOrigin]
    var wv_q8: UnsafePointer[Scalar[DType.int8], MutExternalOrigin]
    var wo_q8: UnsafePointer[Scalar[DType.int8], MutExternalOrigin]
    var w_gate_packed: UnsafePointer[UInt8, MutExternalOrigin]
    var w_up_packed: UnsafePointer[UInt8, MutExternalOrigin]
    var w_down_packed: UnsafePointer[UInt8, MutExternalOrigin]
    var w_router_packed: UnsafePointer[UInt8, MutExternalOrigin]
    var experts_gate_up_packed: UnsafePointer[UInt8, MutExternalOrigin]
    var experts_down_packed: UnsafePointer[UInt8, MutExternalOrigin]

    var wq_scales: UnsafePointer[Float32, MutExternalOrigin]
    var wk_scales: UnsafePointer[Float32, MutExternalOrigin]
    var wv_scales: UnsafePointer[Float32, MutExternalOrigin]
    var wo_scales: UnsafePointer[Float32, MutExternalOrigin]
    var w_gate_scales: UnsafePointer[Float32, MutExternalOrigin]
    var w_up_scales: UnsafePointer[Float32, MutExternalOrigin]
    var w_down_scales: UnsafePointer[Float32, MutExternalOrigin]
    var w_router_scales: UnsafePointer[Float32, MutExternalOrigin]
    var experts_gate_up_scales: UnsafePointer[Float32, MutExternalOrigin]
    var experts_down_scales: UnsafePointer[Float32, MutExternalOrigin]

    var input_layernorm_weight: UnsafePointer[Float32, MutExternalOrigin]
    var post_attention_layernorm_weight: UnsafePointer[Float32, MutExternalOrigin]
    var pre_feedforward_layernorm_weight: UnsafePointer[Float32, MutExternalOrigin]
    var pre_feedforward_layernorm_2_weight: UnsafePointer[Float32, MutExternalOrigin]
    var post_feedforward_layernorm_weight: UnsafePointer[Float32, MutExternalOrigin]
    var post_feedforward_layernorm_1_weight: UnsafePointer[Float32, MutExternalOrigin]
    var post_feedforward_layernorm_2_weight: UnsafePointer[Float32, MutExternalOrigin]
    var q_norm_weight: UnsafePointer[Float32, MutExternalOrigin]
    var k_norm_weight: UnsafePointer[Float32, MutExternalOrigin]
    var router_scale: UnsafePointer[Float32, MutExternalOrigin]
    var router_per_expert_scale: UnsafePointer[Float32, MutExternalOrigin]
    var layer_scalar: UnsafePointer[Float32, MutExternalOrigin]

    def __init__(
        out self,
        wq_packed: UnsafePointer[UInt8, MutExternalOrigin],
        wk_packed: UnsafePointer[UInt8, MutExternalOrigin],
        wv_packed: UnsafePointer[UInt8, MutExternalOrigin],
        wo_packed: UnsafePointer[UInt8, MutExternalOrigin],
        wq_q8: UnsafePointer[Scalar[DType.int8], MutExternalOrigin],
        wk_q8: UnsafePointer[Scalar[DType.int8], MutExternalOrigin],
        wv_q8: UnsafePointer[Scalar[DType.int8], MutExternalOrigin],
        wo_q8: UnsafePointer[Scalar[DType.int8], MutExternalOrigin],
        w_gate_packed: UnsafePointer[UInt8, MutExternalOrigin],
        w_up_packed: UnsafePointer[UInt8, MutExternalOrigin],
        w_down_packed: UnsafePointer[UInt8, MutExternalOrigin],
        w_router_packed: UnsafePointer[UInt8, MutExternalOrigin],
        experts_gate_up_packed: UnsafePointer[UInt8, MutExternalOrigin],
        experts_down_packed: UnsafePointer[UInt8, MutExternalOrigin],
        wq_scales: UnsafePointer[Float32, MutExternalOrigin],
        wk_scales: UnsafePointer[Float32, MutExternalOrigin],
        wv_scales: UnsafePointer[Float32, MutExternalOrigin],
        wo_scales: UnsafePointer[Float32, MutExternalOrigin],
        w_gate_scales: UnsafePointer[Float32, MutExternalOrigin],
        w_up_scales: UnsafePointer[Float32, MutExternalOrigin],
        w_down_scales: UnsafePointer[Float32, MutExternalOrigin],
        w_router_scales: UnsafePointer[Float32, MutExternalOrigin],
        experts_gate_up_scales: UnsafePointer[Float32, MutExternalOrigin],
        experts_down_scales: UnsafePointer[Float32, MutExternalOrigin],
        input_layernorm_weight: UnsafePointer[Float32, MutExternalOrigin],
        post_attention_layernorm_weight: UnsafePointer[Float32, MutExternalOrigin],
        pre_feedforward_layernorm_weight: UnsafePointer[Float32, MutExternalOrigin],
        pre_feedforward_layernorm_2_weight: UnsafePointer[Float32, MutExternalOrigin],
        post_feedforward_layernorm_weight: UnsafePointer[Float32, MutExternalOrigin],
        post_feedforward_layernorm_1_weight: UnsafePointer[Float32, MutExternalOrigin],
        post_feedforward_layernorm_2_weight: UnsafePointer[Float32, MutExternalOrigin],
        q_norm_weight: UnsafePointer[Float32, MutExternalOrigin],
        k_norm_weight: UnsafePointer[Float32, MutExternalOrigin],
        router_scale: UnsafePointer[Float32, MutExternalOrigin],
        router_per_expert_scale: UnsafePointer[Float32, MutExternalOrigin],
        layer_scalar: UnsafePointer[Float32, MutExternalOrigin],
    ):
        self.wq_packed = wq_packed
        self.wk_packed = wk_packed
        self.wv_packed = wv_packed
        self.wo_packed = wo_packed
        self.wq_q8 = wq_q8
        self.wk_q8 = wk_q8
        self.wv_q8 = wv_q8
        self.wo_q8 = wo_q8
        self.w_gate_packed = w_gate_packed
        self.w_up_packed = w_up_packed
        self.w_down_packed = w_down_packed
        self.w_router_packed = w_router_packed
        self.experts_gate_up_packed = experts_gate_up_packed
        self.experts_down_packed = experts_down_packed
        self.wq_scales = wq_scales
        self.wk_scales = wk_scales
        self.wv_scales = wv_scales
        self.wo_scales = wo_scales
        self.w_gate_scales = w_gate_scales
        self.w_up_scales = w_up_scales
        self.w_down_scales = w_down_scales
        self.w_router_scales = w_router_scales
        self.experts_gate_up_scales = experts_gate_up_scales
        self.experts_down_scales = experts_down_scales
        self.input_layernorm_weight = input_layernorm_weight
        self.post_attention_layernorm_weight = post_attention_layernorm_weight
        self.pre_feedforward_layernorm_weight = pre_feedforward_layernorm_weight
        self.pre_feedforward_layernorm_2_weight = pre_feedforward_layernorm_2_weight
        self.post_feedforward_layernorm_weight = post_feedforward_layernorm_weight
        self.post_feedforward_layernorm_1_weight = post_feedforward_layernorm_1_weight
        self.post_feedforward_layernorm_2_weight = post_feedforward_layernorm_2_weight
        self.q_norm_weight = q_norm_weight
        self.k_norm_weight = k_norm_weight
        self.router_scale = router_scale
        self.router_per_expert_scale = router_per_expert_scale
        self.layer_scalar = layer_scalar


# -----------------------------------------------------------------------------
# 3. High-Performance Vectorized Math Kernels
# -----------------------------------------------------------------------------

@always_inline
def rms_norm[
    in_type: DType = DType.float32,
    out_type: DType = DType.float32
](
    x: UnsafePointer[Scalar[in_type], _],
    w: UnsafePointer[Float32, _],
    out_ptr: UnsafePointer[Scalar[out_type], MutExternalOrigin],
    size: Int,
):
    var acc = SIMD[DType.float32, 8](0.0)
    for i in range(0, size, 8):
        var v = x.load[8](i).cast[DType.float32]()
        acc = acc + v * v
    var sum_sq = acc.reduce_add()
    var inv_rms = 1.0 / sqrt(sum_sq / Float32(size) + 1e-6)
    for i in range(0, size, 8):
        var v = x.load[8](i).cast[DType.float32]()
        var weight = w.load[8](i)
        out_ptr.store[8](i, (v * inv_rms * weight).cast[out_type]())

@always_inline
def rms_norm_noscale[
    in_type: DType = DType.float32,
    out_type: DType = DType.float32
](
    x: UnsafePointer[Scalar[in_type], _],
    out_ptr: UnsafePointer[Scalar[out_type], MutExternalOrigin],
    size: Int,
):
    var acc = SIMD[DType.float32, 8](0.0)
    for i in range(0, size, 8):
        var v = x.load[8](i).cast[DType.float32]()
        acc = acc + v * v
    var sum_sq = acc.reduce_add()
    var inv_rms = 1.0 / sqrt(sum_sq / Float32(size) + 1e-6)
    for i in range(0, size, 8):
        var v = x.load[8](i).cast[DType.float32]()
        out_ptr.store[8](i, (v * inv_rms).cast[out_type]())

@always_inline
def unpack_4bytes(b0: UInt8, b1: UInt8, b2: UInt8, b3: UInt8) -> SIMD[DType.float32, 8]:
    var chunk = SIMD[DType.uint8, 8](b0, b1, b2, b3, 0, 0, 0, 0)
    var repeated = chunk.shuffle[0, 0, 1, 1, 2, 2, 3, 3]()
    var shifted = repeated >> SIMD[DType.uint8, 8](0, 4, 0, 4, 0, 4, 0, 4)
    var masked = shifted & 15
    return (masked.cast[DType.int8]() - 8).cast[DType.float32]()


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
@always_inline
def unpack_q4_to_q8(
    W_packed: UnsafePointer[UInt8, _],
    W_q8: UnsafePointer[Scalar[DType.int8], MutUntrackedOrigin],
    size: Int
):
    var steps = size // 32
    for i in range(steps):
        var packed_idx = i * 16
        var unpacked_idx = i * 32
        var bytes = W_packed.load[16](packed_idx)
        
        var repeated_01 = bytes.shuffle[0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7]()
        var shifted_01 = repeated_01 >> SIMD[DType.uint8, 16](0, 4, 0, 4, 0, 4, 0, 4, 0, 4, 0, 4, 0, 4, 0, 4)
        var masked_01 = (shifted_01 & 15).cast[DType.int8]() - 8
        
        var repeated_23 = bytes.shuffle[8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13, 14, 14, 15, 15]()
        var shifted_23 = repeated_23 >> SIMD[DType.uint8, 16](0, 4, 0, 4, 0, 4, 0, 4, 0, 4, 0, 4, 0, 4, 0, 4)
        var masked_23 = (shifted_23 & 15).cast[DType.int8]() - 8
        
        W_q8.store(unpacked_idx, masked_01)
        W_q8.store(unpacked_idx + 16, masked_23)
        
    var rem = size % 32
    if rem > 0:
        var packed_start = steps * 16
        var unpacked_start = steps * 32
        for j in range(rem // 2):
            var b = W_packed[packed_start + j]
            var w_low = (b & 15).cast[DType.int8]() - 8
            var w_high = ((b >> 4) & 15).cast[DType.int8]() - 8
            W_q8.store(unpacked_start + j * 2, w_low)
            W_q8.store(unpacked_start + j * 2 + 1, w_high)


@always_inline
def q4_gemv_cpu(
    W_packed: UnsafePointer[UInt8, _],
    scales: UnsafePointer[Float32, _],
    x: UnsafePointer[Float32, _],
    out_ptr: UnsafePointer[Float32, MutExternalOrigin],
    c_in: Int,
    c_out: Int,
    num_tasks: Int,
):
    var tasks = num_tasks
    if tasks > c_out:
        tasks = c_out
    if tasks < 1:
        tasks = 1
        
    var chunk_size = c_out // tasks
    var c_in_packed = c_in // 2
    var c_in_blocks = c_in // 32
    
    def worker(t: Int) capturing -> None:
        var start_r = t * chunk_size
        var end_r = (t + 1) * chunk_size if t < tasks - 1 else c_out
        
        for r in range(start_r, end_r):
            var acc = SIMD[DType.float32, 8](0.0)
            
            var r_offset_packed = r * c_in_packed
            var r_offset_scales = r * c_in_blocks
            
            for b in range(c_in_blocks):
                var scale = scales[r_offset_scales + b]
                var block_p = r_offset_packed + b * 16
                var x_offset = b * 32

                var bytes = W_packed.load[16](block_p)

                var w0 = SIMD[DType.float32, 8]()
                var w1 = SIMD[DType.float32, 8]()
                var w2 = SIMD[DType.float32, 8]()
                var w3 = SIMD[DType.float32, 8]()
                unpack_16bytes(bytes, w0, w1, w2, w3)

                acc = acc + (w0 * x.load[8](x_offset)) * scale
                acc = acc + (w1 * x.load[8](x_offset + 8)) * scale
                acc = acc + (w2 * x.load[8](x_offset + 16)) * scale
                acc = acc + (w3 * x.load[8](x_offset + 24)) * scale
                
            out_ptr[r] = acc.reduce_add()
            
    if tasks == 1:
        worker(0)
    else:
        parallelize[worker](tasks)

@always_inline
def q4_gemv_batch(
    W_packed: UnsafePointer[UInt8, _],
    scales: UnsafePointer[Float32, _],
    X: UnsafePointer[Float32, _],
    out_ptr: UnsafePointer[Float32, MutExternalOrigin],
    B: Int,
    c_in: Int,
    c_out: Int,
    num_tasks: Int,
    in_stride: Int = -1,
    out_stride: Int = -1,
):
    var tasks = num_tasks
    if tasks > c_out:
        tasks = c_out
    if tasks < 1:
        tasks = 1
    var x_stride = in_stride if in_stride >= 0 else c_in
    var o_stride = out_stride if out_stride >= 0 else c_out

    var chunk_size = c_out // tasks
    var c_in_packed = c_in // 2
    var c_in_blocks = c_in // 32

    def worker(t: Int) capturing -> None:
        var start_r = t * chunk_size
        var end_r = (t + 1) * chunk_size if t < tasks - 1 else c_out
        var acc = alloc[Float32](B)

        for r in range(start_r, end_r):
            var r_offset_packed = r * c_in_packed
            var r_offset_scales = r * c_in_blocks

            for b in range(B):
                acc[b] = 0.0

            for blk in range(c_in_blocks):
                var scale = scales[r_offset_scales + blk]
                var block_p = r_offset_packed + blk * 16
                var x_offset_base = blk * 32

                var bytes = W_packed.load[16](block_p)
                var w0 = SIMD[DType.float32, 8]()
                var w1 = SIMD[DType.float32, 8]()
                var w2 = SIMD[DType.float32, 8]()
                var w3 = SIMD[DType.float32, 8]()
                unpack_16bytes(bytes, w0, w1, w2, w3)

                for b in range(B):
                    var x_base = b * x_stride + x_offset_base
                    var p = (w0 * X.load[8](x_base)).reduce_add() * scale
                    p += (w1 * X.load[8](x_base + 8)).reduce_add() * scale
                    p += (w2 * X.load[8](x_base + 16)).reduce_add() * scale
                    p += (w3 * X.load[8](x_base + 24)).reduce_add() * scale
                    acc[b] += p

            for b in range(B):
                out_ptr[b * o_stride + r] = acc[b]

        acc.free()

    if tasks == 1:
        worker(0)
    else:
        parallelize[worker](tasks)

# Small-batch register-resident q4 GEMV kernel.
# Pre-allocates MAX_B SIMD8 accumulators in CPU registers.
# Loads each 16-byte quantized weight block exactly once, multiply-accumulating
# against all B active token vectors in registers, and does a single
# horizontal reduction per row at the end of the row loop.
@always_inline
@always_inline
def q4_gemv_small_b[
    MAX_B: Int,
    in_type: DType = DType.float32,
    out_type: DType = DType.float32,
](
    W_packed: UnsafePointer[UInt8, _],
    scales: UnsafePointer[Float32, _],
    X: UnsafePointer[Scalar[in_type], _],
    out_ptr: UnsafePointer[Scalar[out_type], MutExternalOrigin],
    B: Int,
    c_in: Int,
    c_out: Int,
    num_tasks: Int,
    in_stride: Int = -1,
    out_stride: Int = -1,
):
    var tasks = num_tasks
    if tasks > c_out:
        tasks = c_out
    if tasks < 1:
        tasks = 1
    var x_stride = in_stride if in_stride >= 0 else c_in
    var o_stride = out_stride if out_stride >= 0 else c_out

    var chunk_size = c_out // tasks
    var c_in_packed = c_in // 2
    var c_in_blocks = c_in // 32

    def worker(t: Int) capturing -> None:
        var start_r = t * chunk_size
        var end_r = (t + 1) * chunk_size if t < tasks - 1 else c_out
        var acc = InlineArray[SIMD[DType.float32, 8], MAX_B](uninitialized=True)

        for r in range(start_r, end_r):
            var r_offset_packed = r * c_in_packed
            var r_offset_scales = r * c_in_blocks

            comptime for i in range(MAX_B):
                acc[i] = SIMD[DType.float32, 8](0.0)

            for blk in range(c_in_blocks):
                var scale = scales[r_offset_scales + blk]
                var block_p = r_offset_packed + blk * 16
                var x_offset_base = blk * 32

                var bytes = W_packed.load[16](block_p)
                var w0 = SIMD[DType.float32, 8]()
                var w1 = SIMD[DType.float32, 8]()
                var w2 = SIMD[DType.float32, 8]()
                var w3 = SIMD[DType.float32, 8]()
                unpack_16bytes(bytes, w0, w1, w2, w3)

                # Pre-scale weights once per block to avoid multiplying scale on every token
                w0 = w0 * scale
                w1 = w1 * scale
                w2 = w2 * scale
                w3 = w3 * scale

                comptime for i in range(MAX_B):
                    if i < B:
                        var x_base = i * x_stride + x_offset_base
                        acc[i] = acc[i] + w0 * X.load[8](x_base).cast[DType.float32]()
                        acc[i] = acc[i] + w1 * X.load[8](x_base + 8).cast[DType.float32]()
                        acc[i] = acc[i] + w2 * X.load[8](x_base + 16).cast[DType.float32]()
                        acc[i] = acc[i] + w3 * X.load[8](x_base + 24).cast[DType.float32]()

            comptime for i in range(MAX_B):
                if i < B:
                    out_ptr[i * o_stride + r] = acc[i].reduce_add().cast[out_type]()

    if tasks == 1:
        worker(0)
    else:
        parallelize[worker](tasks)

# Dynamically dispatches to the most narrow specialized small-batch kernel,
# avoiding weight reload over-fetch loops by keeping weight loading to exactly once.
@always_inline
def q4_gemv_moe_dispatch[
    in_type: DType = DType.float32,
    out_type: DType = DType.float32,
](
    W_packed: UnsafePointer[UInt8, _],
    scales: UnsafePointer[Float32, _],
    X: UnsafePointer[Scalar[in_type], _],
    out_ptr: UnsafePointer[Scalar[out_type], MutExternalOrigin],
    B: Int,
    c_in: Int,
    c_out: Int,
    num_tasks: Int,
    in_stride: Int = -1,
    out_stride: Int = -1,
):
    if B <= 1:
        q4_gemv_small_b[1, in_type, out_type](W_packed, scales, X, out_ptr, B, c_in, c_out, num_tasks, in_stride, out_stride)
    elif B <= 2:
        q4_gemv_small_b[2, in_type, out_type](W_packed, scales, X, out_ptr, B, c_in, c_out, num_tasks, in_stride, out_stride)
    elif B <= 4:
        q4_gemv_small_b[4, in_type, out_type](W_packed, scales, X, out_ptr, B, c_in, c_out, num_tasks, in_stride, out_stride)
    elif B <= 8:
        q4_gemv_small_b[8, in_type, out_type](W_packed, scales, X, out_ptr, B, c_in, c_out, num_tasks, in_stride, out_stride)
    elif B <= 12:
        q4_gemv_small_b[12, in_type, out_type](W_packed, scales, X, out_ptr, B, c_in, c_out, num_tasks, in_stride, out_stride)
    elif B <= 16:
        q4_gemv_small_b[16, in_type, out_type](W_packed, scales, X, out_ptr, B, c_in, c_out, num_tasks, in_stride, out_stride)
    elif B <= 24:
        q4_gemv_small_b[24, in_type, out_type](W_packed, scales, X, out_ptr, B, c_in, c_out, num_tasks, in_stride, out_stride)
    elif B <= 32:
        q4_gemv_small_b[32, in_type, out_type](W_packed, scales, X, out_ptr, B, c_in, c_out, num_tasks, in_stride, out_stride)
    else:
        # Fallback to general tiled version
        q4_gemv_tiled[8, in_type, out_type](W_packed, scales, X, out_ptr, B, c_in, c_out, num_tasks, in_stride, out_stride)

# Tiled batched q4 GEMV. Per output row, processes B in tiles of TILE_B,
# keeping TILE_B SIMD8 accumulators in registers. Each weight block is
# loaded once per row (cached in L1 across tiles). Inner loop does
# SIMD multiply-accumulate (no per-b reduce_add); a single reduce_add
# per (row, b) at the end. Contrast with q4_gemv_batch, which does
# B * c_in_blocks * 4 reduce_adds per row — dominant cost at small B.
@always_inline
def q4_gemv_tiled[
    TILE_B: Int,
    in_type: DType = DType.float32,
    out_type: DType = DType.float32,
](
    W_packed: UnsafePointer[UInt8, _],
    scales: UnsafePointer[Float32, _],
    X: UnsafePointer[Scalar[in_type], _],
    out_ptr: UnsafePointer[Scalar[out_type], MutExternalOrigin],
    B: Int,
    c_in: Int,
    c_out: Int,
    num_tasks: Int,
    in_stride: Int = -1,
    out_stride: Int = -1,
):
    var tasks = num_tasks
    if tasks > c_out:
        tasks = c_out
    if tasks < 1:
        tasks = 1
    var x_stride = in_stride if in_stride >= 0 else c_in
    var o_stride = out_stride if out_stride >= 0 else c_out

    var chunk_size = c_out // tasks
    var c_in_packed = c_in // 2
    var c_in_blocks = c_in // 32

    def worker(t: Int) capturing -> None:
        var start_r = t * chunk_size
        var end_r = (t + 1) * chunk_size if t < tasks - 1 else c_out
        var acc = InlineArray[SIMD[DType.float32, 8], TILE_B](uninitialized=True)

        for r in range(start_r, end_r):
            var r_offset_packed = r * c_in_packed
            var r_offset_scales = r * c_in_blocks

            for b_outer in range(0, B, TILE_B):
                var b_end = b_outer + TILE_B if b_outer + TILE_B < B else B
                var b_count = b_end - b_outer

                comptime for i in range(TILE_B):
                    acc[i] = SIMD[DType.float32, 8](0.0)

                for blk in range(c_in_blocks):
                    var scale = scales[r_offset_scales + blk]
                    var block_p = r_offset_packed + blk * 16
                    var x_offset_base = blk * 32

                    var bytes = W_packed.load[16](block_p)
                    var w0 = SIMD[DType.float32, 8]()
                    var w1 = SIMD[DType.float32, 8]()
                    var w2 = SIMD[DType.float32, 8]()
                    var w3 = SIMD[DType.float32, 8]()
                    unpack_16bytes(bytes, w0, w1, w2, w3)

                    # Pre-scale weights once per block to avoid multiplying scale on every token
                    w0 = w0 * scale
                    w1 = w1 * scale
                    w2 = w2 * scale
                    w3 = w3 * scale

                    comptime for i in range(TILE_B):
                        if i < b_count:
                            var x_base = (b_outer + i) * x_stride + x_offset_base
                            acc[i] = acc[i] + w0 * X.load[8](x_base).cast[DType.float32]()
                            acc[i] = acc[i] + w1 * X.load[8](x_base + 8).cast[DType.float32]()
                            acc[i] = acc[i] + w2 * X.load[8](x_base + 16).cast[DType.float32]()
                            acc[i] = acc[i] + w3 * X.load[8](x_base + 24).cast[DType.float32]()

                comptime for i in range(TILE_B):
                    if i < b_count:
                        out_ptr[(b_outer + i) * o_stride + r] = acc[i].reduce_add().cast[out_type]()

    if tasks == 1:
        worker(0)
    else:
        parallelize[worker](tasks)

@always_inline
def gelu_and_mul[
    type: DType = DType.float32
](
    h1: UnsafePointer[Scalar[type], _],
    h2: UnsafePointer[Scalar[type], _],
    out_ptr: UnsafePointer[Scalar[type], MutExternalOrigin],
    size: Int,
):
    comptime width = 16 if type == DType.float16 else 8
    for i in range(0, size, width):
        var x1 = h1.load[width](i).cast[DType.float32]()
        var sig = 1.0 / (1.0 + exp(SIMD[DType.float32, width](-1.702) * x1))
        var gelu = x1 * sig
        var h2_val = h2.load[width](i).cast[DType.float32]()
        out_ptr.store[width](i, (gelu * h2_val).cast[type]())

@always_inline
def gelu_and_mul_batch[
    type: DType = DType.float32
](
    h1: UnsafePointer[Scalar[type], _],
    h2: UnsafePointer[Scalar[type], _],
    out_ptr: UnsafePointer[Scalar[type], MutExternalOrigin],
    B: Int,
    dim: Int,
):
    comptime width = 16 if type == DType.float16 else 8
    for b in range(B):
        var base = b * dim
        for i in range(0, dim, width):
            var x1 = h1.load[width](base + i).cast[DType.float32]()
            var sig = 1.0 / (1.0 + exp(SIMD[DType.float32, width](-1.702) * x1))
            var h2_val = h2.load[width](base + i).cast[DType.float32]()
            out_ptr.store[width](base + i, (x1 * sig * h2_val).cast[type]())

@always_inline
def rms_norm_batch[
    in_type: DType = DType.float32,
    out_type: DType = DType.float32,
](
    x: UnsafePointer[Scalar[in_type], _],
    w: UnsafePointer[Float32, _],
    out_ptr: UnsafePointer[Scalar[out_type], MutExternalOrigin],
    B: Int,
    dim: Int,
):
    for b in range(B):
        rms_norm[in_type, out_type](x + b * dim, w, out_ptr + b * dim, dim)

@always_inline
def rms_norm_noscale_batch[
    in_type: DType = DType.float32,
    out_type: DType = DType.float32,
](
    x: UnsafePointer[Scalar[in_type], _],
    out_ptr: UnsafePointer[Scalar[out_type], MutExternalOrigin],
    B: Int,
    dim: Int,
):
    for b in range(B):
        rms_norm_noscale[in_type, out_type](x + b * dim, out_ptr + b * dim, dim)

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
        scores[i] /= (sum_exp + 1e-10)

@always_inline
def get_top_k[k: Int](probs: UnsafePointer[Float32, _], top_idx: UnsafePointer[Int, MutExternalOrigin], top_val: UnsafePointer[Float32, MutExternalOrigin], size: Int):
    for i in range(k):
        top_idx[i] = -1
        top_val[i] = -999999.0
        
    for i in range(size):
        var val = probs[i]
        var curr_val = val
        var curr_idx = i
        
        for j in range(k):
            if curr_val > top_val[j]:
                var temp_val = top_val[j]
                var temp_idx = top_idx[j]
                top_val[j] = curr_val
                top_idx[j] = curr_idx
                curr_val = temp_val
                curr_idx = temp_idx


# -----------------------------------------------------------------------------
# 3b. Self-Conditioning MLP Forward (DiffusionGemma SC path)
# -----------------------------------------------------------------------------
# Mirrors vLLM DiffusionGemmaSelfConditioning.forward:
#   x = pre_norm(soft_embeds)
#   sc = down_proj(gelu(gate_proj(x)) * up_proj(x))
#   out = post_norm_noscale(inputs_embeds + sc)   (post_norm has_weight=False)
# With zero soft_embeds the SC signal vanishes and out = post_norm_noscale(inputs_embeds).

def sc_forward(
    inputs_embeds: UnsafePointer[Float32, _],
    soft_embeds: UnsafePointer[Float32, _],
    sc_gate_packed: UnsafePointer[UInt8, _],
    sc_up_packed: UnsafePointer[UInt8, _],
    sc_down_packed: UnsafePointer[UInt8, _],
    sc_gate_scales: UnsafePointer[Float32, _],
    sc_up_scales: UnsafePointer[Float32, _],
    sc_down_scales: UnsafePointer[Float32, _],
    sc_pre_norm: UnsafePointer[Float32, _],
    buf_normed: UnsafePointer[Float32, MutUntrackedOrigin],
    buf_gate: UnsafePointer[Float32, MutUntrackedOrigin],
    buf_up: UnsafePointer[Float32, MutUntrackedOrigin],
    buf_gated: UnsafePointer[Float32, MutUntrackedOrigin],
    buf_down: UnsafePointer[Float32, MutUntrackedOrigin],
    sc_out: UnsafePointer[Float32, MutUntrackedOrigin],
    embed_dim: Int,
    sc_size: Int,
    num_tasks: Int,
):
    # 1. pre_norm the soft_embeds (RMSNorm with sc_pre_norm weight)
    rms_norm(soft_embeds, sc_pre_norm, buf_normed, embed_dim)
    # 2. gate_proj and up_proj: [sc_size, embed_dim] x [embed_dim] -> [sc_size]
    q4_gemv_cpu(sc_gate_packed, sc_gate_scales, buf_normed, buf_gate, embed_dim, sc_size, num_tasks)
    q4_gemv_cpu(sc_up_packed, sc_up_scales, buf_normed, buf_up, embed_dim, sc_size, num_tasks)
    # 3. gelu_and_mul: buf_gated = gelu(buf_gate) * buf_up
    gelu_and_mul(buf_gate, buf_up, buf_gated, sc_size)
    # 4. down_proj: [embed_dim, sc_size] x [sc_size] -> [embed_dim]
    q4_gemv_cpu(sc_down_packed, sc_down_scales, buf_gated, buf_down, sc_size, embed_dim, num_tasks)
    # 5. residual + post_norm (no learned weight)
    for i in range(0, embed_dim, 8):
        sc_out.store[8](i, inputs_embeds.load[8](i) + buf_down.load[8](i))
    rms_norm_noscale(sc_out, sc_out, embed_dim)


# -----------------------------------------------------------------------------
# 3c. Diffusion Sampler Helpers
# -----------------------------------------------------------------------------
# Computes soft_embeds = (probs @ embed_weight) * normalizer for one canvas position.
# embed_weight is stored as [vocab, embed_dim] in q4. We want out[d] = sum_r probs[r] * W[r,d].
# Threshold: skip tokens with negligible prob (most of the vocab after softmax).
def probs_to_soft_embed(
    probs: UnsafePointer[Float32, _],
    embed_packed: UnsafePointer[UInt8, _],
    embed_scales: UnsafePointer[Float32, _],
    sc_out: UnsafePointer[Float32, MutUntrackedOrigin],
    vocab_size: Int,
    embed_dim: Int,
    num_tasks: Int,
):
    for d in range(0, embed_dim, 8):
        sc_out.store[8](d, SIMD[DType.float32, 8](0.0))
    var num_blocks = embed_dim // 32
    var chunk_size = vocab_size // num_tasks

    def sc_worker(t: Int) capturing -> None:
        var start_r = t * chunk_size
        var end_r = (t + 1) * chunk_size if t < num_tasks - 1 else vocab_size
        for r in range(start_r, end_r):
            var p = probs[r]
            if p < 1e-8:
                continue
            var row_packed_offset = r * (embed_dim // 2)
            var row_scales_offset = r * (embed_dim // 32)
            for b in range(num_blocks):
                var scale = embed_scales[row_scales_offset + b]
                var block_packed_offset = row_packed_offset + b * 16
                var block_out_offset = b * 32

                var bytes = embed_packed.load[16](block_packed_offset)
                var w0 = SIMD[DType.float32, 8]()
                var w1 = SIMD[DType.float32, 8]()
                var w2 = SIMD[DType.float32, 8]()
                var w3 = SIMD[DType.float32, 8]()
                unpack_16bytes(bytes, w0, w1, w2, w3)

                var p_scale_vec = SIMD[DType.float32, 8](p * scale)
                sc_out.store[8](block_out_offset, sc_out.load[8](block_out_offset) + w0 * p_scale_vec)
                sc_out.store[8](block_out_offset + 8, sc_out.load[8](block_out_offset + 8) + w1 * p_scale_vec)
                sc_out.store[8](block_out_offset + 16, sc_out.load[8](block_out_offset + 16) + w2 * p_scale_vec)
                sc_out.store[8](block_out_offset + 24, sc_out.load[8](block_out_offset + 24) + w3 * p_scale_vec)

    if num_tasks == 1:
        sc_worker(0)
    else:
        parallelize[sc_worker](num_tasks)


# -----------------------------------------------------------------------------
# 4. Custom Padding Helpers & Main Executive Engine
# -----------------------------------------------------------------------------

def pad_right(s: String, width: Int) -> String:
    var res = s
    while res.byte_length() < width:
        res += " "
    return res

def pad_left(s: String, width: Int) -> String:
    var res = s
    while res.byte_length() < width:
        res = " " + res
    return res

def main() raises:
    var use_gpu = True
    var cmd_args = std.sys.argv()
    for idx in range(1, len(cmd_args)):
        if cmd_args[idx] == "--cpu":
            use_gpu = False

    var num_tasks = num_physical_cores()
    print("=========================================================")
    print("   DIFFUSIONGEMMA 26B-A4B-IT  CANVAS DIFFUSION ENGINE   ")
    print("=========================================================")
    print("Embedding Size:      2816")
    print("Attention Heads:     16 Query Heads / 8 or 2 KV Heads (GQA)")
    print("Number of Layers:    30 Layers (5 global attention layers)")
    print("Number of Experts:   128 Experts (Top-8 Router routing)")
    print("Canvas Mode:         256-token canvas, 48 max denoise steps")
    print("Self-Conditioning:  2112-dim gated MLP, RAM-resident Q4_0")
    if use_gpu:
        print("Execution Mode:      Hybrid CPU-iGPU (Attention on iGPU via AMD KFD)")
    else:
        print("Execution Mode:      CPU fallbacks (Multi-threaded tiled kernels)")
    print("---------------------------------------------------------")

    var CONFIG = MoEConfig[]()

    # Initialize AMD direct GPU userspace queue
    if use_gpu:
        print("Initializing direct AMD GPU userspace queue...")
    else:
        print("Initializing CPU-only context memory structures...")
    var gpu_ctx = GPUContext(use_gpu)
    if use_gpu:
        print("Direct AMD GPU userspace queue initialized successfully! (GPU ID: " + String(gpu_ctx.gpu_id) + ")")
    else:
        print("CPU-only context memory structures initialized successfully!")

    # Setup SVM Flags
    var base_flags = (GPUContext.KFD_IOC_ALLOC_MEM_FLAGS_WRITABLE |
                      GPUContext.KFD_IOC_ALLOC_MEM_FLAGS_EXECUTABLE |
                      GPUContext.KFD_IOC_ALLOC_MEM_FLAGS_NO_SUBSTITUTE)

    var gtt_uncached_flags = (base_flags |
                              GPUContext.KFD_IOC_ALLOC_MEM_FLAGS_GTT |
                              GPUContext.KFD_IOC_ALLOC_MEM_FLAGS_COHERENT |
                              GPUContext.KFD_IOC_ALLOC_MEM_FLAGS_UNCACHED |
                              GPUContext.KFD_IOC_ALLOC_MEM_FLAGS_PUBLIC)

    # Initialize FFI and libc functions
    var libc = OwnedDLHandle("libc.so.6")
    var c_open = libc.get_function[def(Int, Int32, Int32) thin abi("C") -> Int32]("open")
    var c_close = libc.get_function[def(Int32) thin abi("C") -> Int32]("close")
    var posix_memalign = libc.get_function[def(Int, Int, Int) thin abi("C") -> Int32]("posix_memalign")
    var free_c = libc.get_function[def(Int) thin abi("C") -> None]("free")
    var c_read = libc.get_function[def(Int32, Int, Int) thin abi("C") -> Int]("read")

    var q8_attn_base: Int = 0
    var q8_attn_size: Int = 0
    var q8_attn_gpu_buf = GPUMemoryBuffer()
    if use_gpu:
        q8_attn_size = CONFIG.get_q8_attn_offset(CONFIG.num_layers)
        q8_attn_gpu_buf = gpu_ctx.alloc_gpu_buffer(q8_attn_size, UInt32(gtt_uncached_flags), cpu_access=True)
        q8_attn_base = q8_attn_gpu_buf.cpu_addr

    # 1. Allocate and Load DiffusionGemma Decoder Weights (RAM-resident)
    var weights_file = String("diffusiongemma_q4.bin")
    print("Opening MoE weights file '" + weights_file + "'...")
    var path_bytes = weights_file.as_bytes()
    var path_ptr = alloc[UInt8](len(path_bytes) + 1)
    for i in range(len(path_bytes)):
        path_ptr[i] = path_bytes[i]
    path_ptr[len(path_bytes)] = 0
    var weights_fd = c_open(path_ptr.__int__(), 0, 0)
    path_ptr.free()
    if weights_fd < 0:
        print("CRITICAL: Failed to open weights file '" + weights_file + "'!")
        return

    var weights_file_size: Int = 15311790080
    var weights_gpu_buf = gpu_ctx.alloc_gpu_buffer(weights_file_size, UInt32(gtt_uncached_flags), cpu_access=True)
    var base_addr = weights_gpu_buf.cpu_addr
    var base_ptr = UnsafePointer[UInt8, MutExternalOrigin](unsafe_from_address=base_addr)

    print("Loading DiffusionGemma weights into RAM (14.26 GB) in chunks...")
    var total_read: Int = 0
    var chunk_bytes: Int = 268435456 # 256 MB chunks
    while total_read < weights_file_size:
        var bytes_to_read = weights_file_size - total_read
        if bytes_to_read > chunk_bytes:
            bytes_to_read = chunk_bytes
        var read_res = c_read(weights_fd, base_addr + total_read, bytes_to_read)
        if read_res <= 0:
            print("CRITICAL: Error reading weights from file!")
            _ = c_close(weights_fd)
            return
        total_read += read_res
        if total_read % (2 * 1024 * 1024 * 1024) < chunk_bytes:
            print("  Loaded: ", total_read // 1024 // 1024 // 1024, " GB / 14 GB...")
    print("Successfully loaded all DiffusionGemma weights into RAM: ", total_read, " bytes")
    _ = c_close(weights_fd)

    if use_gpu:
        print("Unpacking Q4 attention weights to Q8 in host SVM memory...")
        var clflush = gpu_ctx.cpu_clflush
        var sfence = gpu_ctx.cpu_sfence
        for l in range(30):
            var weights = CONFIG.get_layer_weights(l, base_ptr + CONFIG.get_layer_offset(l))
            var is_global = CONFIG.is_global_layer(l)
            var layer_heads = 16
            var layer_head_dim = 512 if is_global else 256
            var layer_kv_heads = 2 if is_global else 8
            
            var wq_size = layer_heads * layer_head_dim * CONFIG.embed_dim
            var wk_size = layer_kv_heads * layer_head_dim * CONFIG.embed_dim
            var wv_size = 0 if is_global else (layer_kv_heads * layer_head_dim * CONFIG.embed_dim)
            var wo_size = CONFIG.embed_dim * layer_heads * layer_head_dim
            
            var layer_q8_offset = CONFIG.get_q8_attn_offset(l)
            var wq_q8_addr = q8_attn_base + layer_q8_offset
            var wk_q8_addr = wq_q8_addr + wq_size
            var wv_q8_addr = wk_q8_addr + wk_size
            var wo_q8_addr = wv_q8_addr + wv_size
            
            unpack_q4_to_q8(weights.wq_packed, UnsafePointer[Scalar[DType.int8], MutExternalOrigin](unsafe_from_address=wq_q8_addr), wq_size)
            unpack_q4_to_q8(weights.wk_packed, UnsafePointer[Scalar[DType.int8], MutExternalOrigin](unsafe_from_address=wk_q8_addr), wk_size)
            if not is_global:
                unpack_q4_to_q8(weights.wv_packed, UnsafePointer[Scalar[DType.int8], MutExternalOrigin](unsafe_from_address=wv_q8_addr), wv_size)
            unpack_q4_to_q8(weights.wo_packed, UnsafePointer[Scalar[DType.int8], MutExternalOrigin](unsafe_from_address=wo_q8_addr), wo_size)
            
        print("Flushing CPU cache lines of unpacked Q8 attention weights...")
        var base_flush = (q8_attn_base // 64) * 64
        var end_flush = q8_attn_base + q8_attn_size
        while base_flush < end_flush:
            _ = clflush(base_flush)
            base_flush += 64
        _ = sfence(0)
        print("Successfully unpacked and flushed Q8 attention weights!")

    # 2. Allocate and Load Final Layer Norm Weights
    var final_norm_file = String("diffusiongemma_final_norm.bin")
    print("Opening final norm weights file 'diffusiongemma_final_norm.bin'...")
    var final_norm_path_bytes = final_norm_file.as_bytes()
    var final_norm_path_ptr = alloc[UInt8](len(final_norm_path_bytes) + 1)
    for i in range(len(final_norm_path_bytes)):
        final_norm_path_ptr[i] = final_norm_path_bytes[i]
    final_norm_path_ptr[len(final_norm_path_bytes)] = 0
    var final_norm_fd = c_open(final_norm_path_ptr.__int__(), 0, 0)
    final_norm_path_ptr.free()
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
            print("CRITICAL: Failed to read from final norm weights file!")
            _ = c_close(final_norm_fd)
            final_norm_weight.free()
            return
        total_read += read_res
    _ = c_close(final_norm_fd)
    print("Final norm weights loaded: ", total_read, " bytes")

    # 2b. Load Self-Conditioning MLP weights (Q4_0)
    #    Layout: [gate_proj.packed][up_proj.packed][down_proj.packed]
    #            [gate_proj.scales][up_proj.scales][down_proj.scales]
    #            [pre_norm.weight fp32]
    #    post_norm has no learned weight (has_weight=False per vLLM).
    var sc_file = String("diffusiongemma_sc_q4.bin")
    print("Opening self-conditioning weights file '" + sc_file + "'...")
    var sc_path_bytes = sc_file.as_bytes()
    var sc_path_ptr = alloc[UInt8](len(sc_path_bytes) + 1)
    for i in range(len(sc_path_bytes)):
        sc_path_ptr[i] = sc_path_bytes[i]
    sc_path_ptr[len(sc_path_bytes)] = 0
    var sc_fd = c_open(sc_path_ptr.__int__(), 0, 0)
    sc_path_ptr.free()
    if sc_fd < 0:
        print("CRITICAL: Failed to open '" + sc_file + "'! Diffusion will not work.")
        return

    # SC size: gate(2112x2816) + up(2112x2816) + down(2816x2112) — all 5,940,992 elts
    # packed = 2,970,496 B each ; scales = 185,656 * 4 B each ; pre_norm = 2816 * 4 B
    comptime SC_MATRIX_SIZE = 2112 * 2816  # = 5,940,992 elements per matrix (gate/up/down all same)
    comptime SC_PACKED = SC_MATRIX_SIZE // 2       # 2,970,496 bytes per matrix
    comptime SC_SCALES = SC_MATRIX_SIZE // 32      # 185,656 floats per matrix
    comptime SC_SCALES_BYTES = SC_SCALES * 4
    comptime SC_PRE_NORM_BYTES = 2816 * 4
    comptime SC_TOTAL = (SC_PACKED * 3) + (SC_SCALES_BYTES * 3) + SC_PRE_NORM_BYTES
    comptime SC_ALIGNED = ((SC_TOTAL + 4095) // 4096) * 4096

    var memptr_sc = alloc[Int](1)
    var err_sc = posix_memalign(memptr_sc.__int__(), 4096, SC_ALIGNED)
    if err_sc != 0:
        print("CRITICAL: Failed to allocate SC weight memory!")
        _ = c_close(sc_fd)
        memptr_sc.free()
        return
    var sc_base_addr = memptr_sc[0]
    memptr_sc.free()
    var sc_base = UnsafePointer[UInt8, MutExternalOrigin](unsafe_from_address=sc_base_addr)

    var sc_total_read: Int = 0
    while sc_total_read < SC_ALIGNED:
        var bytes_to_read = SC_ALIGNED - sc_total_read
        var read_res = c_read(sc_fd, sc_base_addr + sc_total_read, bytes_to_read)
        if read_res <= 0:
            print("CRITICAL: Failed to read SC weights!")
            _ = c_close(sc_fd)
            return
        sc_total_read += read_res
    _ = c_close(sc_fd)
    print("Self-conditioning weights loaded: ", sc_total_read, " bytes")

    # Lay out the SC pointers
    var sc_gate_packed = sc_base
    var sc_up_packed = sc_gate_packed + SC_PACKED
    var sc_down_packed = sc_up_packed + SC_PACKED
    var sc_scales_base = (sc_down_packed + SC_PACKED).bitcast[Float32]()
    var sc_gate_scales = sc_scales_base
    var sc_up_scales = sc_gate_scales + SC_SCALES
    var sc_down_scales = sc_up_scales + SC_SCALES
    var sc_pre_norm = sc_down_scales + SC_SCALES

    # 3. Load Q4 Token Embeddings
    print("Loading Q4 Token Embeddings...")
    var embed = Q4Embedding("diffusiongemma_embed_q4_32k.bin", CONFIG.vocab_size, CONFIG.embed_dim)

    # 4. Allocate continuous KV Cache for 30 layers
    var kv_cache_offsets = List[Int]()
    var current_cache_offset = 0
    for l in range(30):
        kv_cache_offsets.append(current_cache_offset)
        var is_global = CONFIG.is_global_layer(l)
        var kv_heads = 2 if is_global else 8
        var head_dim = 512 if is_global else 256
        current_cache_offset += CONFIG.max_seq_len * kv_heads * head_dim
    print("Allocating KV Cache (", (current_cache_offset * 8) // 1024 // 1024, " MB)...")
    var k_cache = alloc[Float32](current_cache_offset)
    var v_cache = alloc[Float32](current_cache_offset)
    for i in range(current_cache_offset):
        k_cache[i] = 0.0
        v_cache[i] = 0.0

    # 5. Allocate thread-safe reusable activation buffers
    var local_x = alloc[Float32](CONFIG.embed_dim)
    var norm_x = alloc[Float32](CONFIG.embed_dim)
    var scaled_norm_x = alloc[Float32](CONFIG.embed_dim)
    var q = alloc[Float32](8192)
    var k_vec = alloc[Float32](2048)
    var v_vec = alloc[Float32](2048)
    var attn_scores = alloc[Float32](CONFIG.max_seq_len)
    var attn_out = alloc[Float32](8192)
    var proj_attn_out = alloc[Float32](CONFIG.embed_dim)
    
    var shared_h1 = alloc[Float32](CONFIG.standard_mlp_dim)
    var shared_h2 = alloc[Float32](CONFIG.standard_mlp_dim)
    var shared_ffn_out_gate = alloc[Float32](CONFIG.standard_mlp_dim)
    var shared_mlp_out = alloc[Float32](CONFIG.embed_dim)
    var shared_h1_norm = alloc[Float32](CONFIG.embed_dim)
    
    var moe_in = alloc[Float32](CONFIG.embed_dim)
    var moe_out = alloc[Float32](CONFIG.embed_dim)
    var gate_up_out_all = alloc[Float32](CONFIG.top_k_experts * (CONFIG.expert_mlp_dim * 2))
    var act_all = alloc[Float32](CONFIG.top_k_experts * CONFIG.expert_mlp_dim)
    var down_out_all = alloc[Float32](CONFIG.top_k_experts * CONFIG.embed_dim)
    var moe_out_norm = alloc[Float32](CONFIG.embed_dim)
    var combined_ffn = alloc[Float32](CONFIG.embed_dim)
    var ffn_out = alloc[Float32](CONFIG.embed_dim)
    
    var scores = alloc[Float32](128)
    var top_idx = alloc[Int](8)
    var top_val = alloc[Float32](8)
    var top_weights = alloc[Float32](8)
    
    var target_final_norm_x = alloc[Float32](CONFIG.embed_dim)
    var local_max_logits = alloc[Float32](num_tasks)
    var local_next_tokens = alloc[Int](num_tasks)

    var total_experts_evaluated: Int = 0
    var total_experts_skipped: Int = 0

    # 5b. Diffusion canvas constants and buffers
    comptime CANVAS_LEN = 256
    comptime MAX_DENOISE_STEPS = 48
    comptime SC_SIZE = 2112
    comptime MAX_Q_SIZE = 16 * 512  # 8192 (global layer Q size; sliding = 16*256=4096)
    comptime MAX_KV_SIZE = 8 * 512  # 4096 (max KV stride: sliding 8*256=2048, global 2*512=1024)



    var canvas_hidden_buf = gpu_ctx.alloc_gpu_buffer(CANVAS_LEN * CONFIG.embed_dim * 4, UInt32(gtt_uncached_flags), cpu_access=True)
    var canvas_q_buf = gpu_ctx.alloc_gpu_buffer(CANVAS_LEN * MAX_Q_SIZE * 4, UInt32(gtt_uncached_flags), cpu_access=True)
    var canvas_k_buf = gpu_ctx.alloc_gpu_buffer(CANVAS_LEN * MAX_KV_SIZE * 4, UInt32(gtt_uncached_flags), cpu_access=True)
    var canvas_v_buf = gpu_ctx.alloc_gpu_buffer(CANVAS_LEN * MAX_KV_SIZE * 4, UInt32(gtt_uncached_flags), cpu_access=True)
    var canvas_attn_out_buf = gpu_ctx.alloc_gpu_buffer(CANVAS_LEN * MAX_Q_SIZE * 4, UInt32(gtt_uncached_flags), cpu_access=True)
    var canvas_proj_attn_buf = gpu_ctx.alloc_gpu_buffer(CANVAS_LEN * CONFIG.embed_dim * 4, UInt32(gtt_uncached_flags), cpu_access=True)

    # Canvas hidden states [CANVAS_LEN, embed_dim] — flows through layers
    var canvas_hidden = UnsafePointer[Float32, MutUntrackedOrigin](unsafe_from_address=canvas_hidden_buf.cpu_addr)
    # Canvas Q storage [CANVAS_LEN, MAX_Q_SIZE] — stored in phase 1, consumed in phase 2
    var canvas_q = UnsafePointer[Float32, MutUntrackedOrigin](unsafe_from_address=canvas_q_buf.cpu_addr)
    # Canvas K/V storage [CANVAS_LEN, MAX_KV_SIZE] — for batched QKV proj
    var canvas_k = UnsafePointer[Float32, MutUntrackedOrigin](unsafe_from_address=canvas_k_buf.cpu_addr)
    var canvas_v = UnsafePointer[Float32, MutUntrackedOrigin](unsafe_from_address=canvas_v_buf.cpu_addr)
    # Batched attention output [CANVAS_LEN, MAX_Q_SIZE]
    var canvas_attn_out = UnsafePointer[Float32, MutUntrackedOrigin](unsafe_from_address=canvas_attn_out_buf.cpu_addr)
    # Batched Wo output [CANVAS_LEN, embed_dim]
    var canvas_proj_attn = UnsafePointer[Float32, MutUntrackedOrigin](unsafe_from_address=canvas_proj_attn_buf.cpu_addr)
    # Batched norm output [CANVAS_LEN, embed_dim]
    var canvas_norm_buf = gpu_ctx.alloc_gpu_buffer(CANVAS_LEN * CONFIG.embed_dim * 4, UInt32(gtt_uncached_flags), cpu_access=True)
    var canvas_norm = UnsafePointer[Float32, MutUntrackedOrigin](unsafe_from_address=canvas_norm_buf.cpu_addr)

    # Reusable packed flat temp buffers for Coalesced Batched GEMM launches
    var temp_packed_q_buf = gpu_ctx.alloc_gpu_buffer(CANVAS_LEN * MAX_Q_SIZE * 4, UInt32(gtt_uncached_flags), cpu_access=True)
    var temp_packed_k_buf = gpu_ctx.alloc_gpu_buffer(CANVAS_LEN * MAX_KV_SIZE * 4, UInt32(gtt_uncached_flags), cpu_access=True)
    var temp_packed_v_buf = gpu_ctx.alloc_gpu_buffer(CANVAS_LEN * MAX_KV_SIZE * 4, UInt32(gtt_uncached_flags), cpu_access=True)

    var temp_packed_q = UnsafePointer[Float32, MutUntrackedOrigin](unsafe_from_address=temp_packed_q_buf.cpu_addr)
    var temp_packed_k = UnsafePointer[Float32, MutUntrackedOrigin](unsafe_from_address=temp_packed_k_buf.cpu_addr)
    var temp_packed_v = UnsafePointer[Float32, MutUntrackedOrigin](unsafe_from_address=temp_packed_v_buf.cpu_addr)
    # Batched FFN intermediates
    var canvas_ffn_gate = alloc[Float16](CANVAS_LEN * CONFIG.standard_mlp_dim)
    var canvas_ffn_up = alloc[Float16](CANVAS_LEN * CONFIG.standard_mlp_dim)
    var canvas_ffn_gated = alloc[Float16](CANVAS_LEN * CONFIG.standard_mlp_dim)
    var canvas_ffn_down = alloc[Float16](CANVAS_LEN * CONFIG.embed_dim)
    var canvas_shared_norm = alloc[Float16](CANVAS_LEN * CONFIG.embed_dim)
    var canvas_moe_in = alloc[Float16](CANVAS_LEN * CONFIG.embed_dim)
    var canvas_moe_out = alloc[Float16](CANVAS_LEN * CONFIG.embed_dim)
    var canvas_moe_norm = alloc[Float16](CANVAS_LEN * CONFIG.embed_dim)
    var canvas_combined = alloc[Float16](CANVAS_LEN * CONFIG.embed_dim)
    var canvas_ffn_out = alloc[Float16](CANVAS_LEN * CONFIG.embed_dim)
    # Per-token attention scores buffer for parallelized attention
    var canvas_attn_scores = alloc[Float32](CANVAS_LEN * CONFIG.max_seq_len)
    # MoE Batched Expert scratch buffers
    var temp_x = alloc[Float16](CANVAS_LEN * CONFIG.embed_dim)
    var temp_gate_up = alloc[Float16](CANVAS_LEN * CONFIG.expert_mlp_dim * 2)
    var temp_act = alloc[Float16](CANVAS_LEN * CONFIG.expert_mlp_dim)
    var temp_down = alloc[Float16](CANVAS_LEN * CONFIG.embed_dim)
    var canvas_router_scores = alloc[Float32](CANVAS_LEN * CONFIG.num_experts)
    var canvas_top_idx = alloc[Int](CANVAS_LEN * CONFIG.top_k_experts)
    var canvas_top_val = alloc[Float32](CANVAS_LEN * CONFIG.top_k_experts)
    var canvas_top_weights = alloc[Float32](CANVAS_LEN * CONFIG.top_k_experts)
    var canvas_scaled_norm = alloc[Float32](CANVAS_LEN * CONFIG.embed_dim)
    var canvas_routed_norm = alloc[Float32](CANVAS_LEN * CONFIG.embed_dim)
    # Canvas token ids [CANVAS_LEN]
    var canvas_tokens = alloc[Int](CANVAS_LEN)
    var argmax_canvas = alloc[Int](CANVAS_LEN)
    # Self-conditioning soft embeds [CANVAS_LEN, embed_dim]
    var sc_embeds = alloc[Float32](CANVAS_LEN * CONFIG.embed_dim)
    # SC MLP scratch buffers
    var sc_buf_normed = alloc[Float32](CONFIG.embed_dim)
    var sc_buf_gate = alloc[Float32](SC_SIZE)
    var sc_buf_up = alloc[Float32](SC_SIZE)
    var sc_buf_gated = alloc[Float32](SC_SIZE)
    var sc_buf_down = alloc[Float32](CONFIG.embed_dim)
    var sc_out_buf = alloc[Float32](CONFIG.embed_dim)
    # Per-canvas-token final norm + logits storage
    var canvas_final_norm = alloc[Float32](CANVAS_LEN * CONFIG.embed_dim)
    # Sampler buffers
    var sampler_logits = alloc[Float32](CONFIG.vocab_size)
    var sampler_probs = alloc[Float32](CONFIG.vocab_size)
    # Per-token logits/probs buffers for batched logit projection
    var canvas_logits = alloc[Float32](CANVAS_LEN * CONFIG.vocab_size)
    var canvas_probs = alloc[Float32](CANVAS_LEN * CONFIG.vocab_size)
    var token_entropy = alloc[Float32](CANVAS_LEN)
    var new_tokens = alloc[Int](CANVAS_LEN)
    var argmax_tokens = alloc[Int](CANVAS_LEN)
    var sorted_idx = alloc[Int](CANVAS_LEN)
    var sorted_ent = alloc[Float32](CANVAS_LEN)
    var eb_mask = alloc[Bool](CANVAS_LEN)
    var history = alloc[Int](2 * CANVAS_LEN)  # ST=2 slots

    print("Canvas buffers allocated: ", CANVAS_LEN, " tokens, ",
          (CANVAS_LEN * CONFIG.embed_dim * 4 + CANVAS_LEN * MAX_Q_SIZE * 4) // 1024 // 1024, " MB hidden+Q")

    # 6. Initialize tokenizer
    var py = Python.import_module("time")
    var transformers = Python.import_module("transformers")
    print("Loading AutoTokenizer...")
    var tokenizer = transformers.AutoTokenizer.from_pretrained("google/gemma-4-12B-it")
    var py_builtins = Python.import_module("builtins")
    print("Tokenizer loaded successfully!")

    print("---------------------------------------------------------")
    print("   Chat Agent is ready! Type 'exit' or 'quit' to end.")
    print("---------------------------------------------------------")

    var prompt_scale = sqrt(Float32(CONFIG.embed_dim))

    while True:
        print()
        var user_input_str: String
        try:
            var user_input = py_builtins.input("User > ")
            user_input_str = String(String(user_input).strip())
        except err:
            print("\nError reading input:", err)
            break


        if user_input_str == "exit" or user_input_str == "quit":
            break
        if len(user_input_str) == 0:
            continue


        var prompt_str = String("<bos><|turn>user\n") + user_input_str + String("<turn|>\n<|turn>model\n")
        var encoded = tokenizer.encode(prompt_str, add_special_tokens=False)
        var prompt_tokens = List[Int]()
        for i in range(len(encoded)):
            var tok = Int(py=encoded[i])
            if tok >= CONFIG.vocab_size:
                tok = 3 # Map out-of-vocab tokens to <unk> (index 3) to prevent out-of-bounds errors
            prompt_tokens.append(tok)

        var num_prompt_tokens = len(prompt_tokens)
        if num_prompt_tokens + CANVAS_LEN > CONFIG.max_seq_len:
            print("Prompt too long! Max ", CONFIG.max_seq_len - CANVAS_LEN, " tokens, got ", num_prompt_tokens)
            continue

        # Zero-out the KV Cache for the new interaction
        for i in range(current_cache_offset):
            k_cache[i] = 0.0
            v_cache[i] = 0.0

        print("Agent > (diffusion mode)")

        var t_prefill_start = std.time.perf_counter()
        var t_decode_start: Float64 = 0.0

        # Profiling accumulators (kept for compat with prefill per-layer code)
        var p_embed_reconstruct: Float64 = 0.0
        var p_attn_norm: Float64 = 0.0
        var p_qkv_proj: Float64 = 0.0
        var p_head_norm_rope_cache: Float64 = 0.0
        var p_attn_dot_product: Float64 = 0.0
        var p_wo_proj: Float64 = 0.0
        var p_shared_mlp: Float64 = 0.0
        var p_moe_router: Float64 = 0.0
        var p_moe_experts: Float64 = 0.0
        var p_moe_combine: Float64 = 0.0
        var p_final_norm: Float64 = 0.0
        var p_logit_proj: Float64 = 0.0
        var p_misc_decode: Float64 = 0.0

        # --- Encoder Prefill (causal, token-first, writes K/V) ---
        var p_scale_vec_prefill = SIMD[DType.float32, 8](prompt_scale)
        for pos in range(num_prompt_tokens):
            embed.reconstruct(prompt_tokens[pos], local_x)
            for i in range(0, CONFIG.embed_dim, 8):
                local_x.store[8](i, local_x.load[8](i) * p_scale_vec_prefill)

            # Run layers
            for l in range(30):
                var weights = CONFIG.get_layer_weights(l, base_ptr + CONFIG.get_layer_offset(l), q8_attn_base)
                var is_global = CONFIG.is_global_layer(l)
                var layer_heads = 16
                var layer_kv_heads = 2 if is_global else 8
                var layer_head_dim = 512 if is_global else 256
                var kv_stride = layer_kv_heads * layer_head_dim
                var ratio = layer_heads // layer_kv_heads

                # 1. Attention Input Norm
                var ta0 = std.time.perf_counter()
                rms_norm(local_x, weights.input_layernorm_weight, norm_x, CONFIG.embed_dim)
                var ta1 = std.time.perf_counter()
                if pos >= num_prompt_tokens:
                    p_attn_norm += ta1 - ta0

                # 2. Attention projections
                ta0 = std.time.perf_counter()
                q4_gemv_cpu(weights.wq_packed, weights.wq_scales, norm_x, q, CONFIG.embed_dim, layer_heads * layer_head_dim, num_tasks)
                q4_gemv_cpu(weights.wk_packed, weights.wk_scales, norm_x, k_vec, CONFIG.embed_dim, layer_kv_heads * layer_head_dim, num_tasks)
                if is_global:
                    for i in range(kv_stride):
                        v_vec[i] = k_vec[i]
                else:
                    q4_gemv_cpu(weights.wv_packed, weights.wv_scales, norm_x, v_vec, CONFIG.embed_dim, layer_kv_heads * layer_head_dim, num_tasks)
                ta1 = std.time.perf_counter()
                if pos >= num_prompt_tokens:
                    p_qkv_proj += ta1 - ta0

                # 3. Head-Normalizations & RoPE & KV cache write
                ta0 = std.time.perf_counter()
                for h in range(layer_heads):
                    var head_ptr = q + h * layer_head_dim
                    rms_norm(head_ptr, weights.q_norm_weight, head_ptr, layer_head_dim)
                for h in range(layer_kv_heads):
                    var head_ptr = k_vec + h * layer_head_dim
                    rms_norm(head_ptr, weights.k_norm_weight, head_ptr, layer_head_dim)
                for h in range(layer_kv_heads):
                    var head_ptr = v_vec + h * layer_head_dim
                    rms_norm_noscale(head_ptr, head_ptr, layer_head_dim)

                # 4. RoPE
                apply_rope(q, layer_heads, layer_head_dim, pos, is_global)
                apply_rope(k_vec, layer_kv_heads, layer_head_dim, pos, is_global)

                # 5. KV Cache write
                var layer_cache_offset = kv_cache_offsets[l]
                var step_cache_offset = layer_cache_offset + pos * kv_stride
                for i in range(kv_stride):
                    k_cache[step_cache_offset + i] = k_vec[i]
                    v_cache[step_cache_offset + i] = v_vec[i]
                ta1 = std.time.perf_counter()
                if pos >= num_prompt_tokens:
                    p_head_norm_rope_cache += ta1 - ta0

                # 6. Dot-product Attention
                ta0 = std.time.perf_counter()
                for h in range(layer_heads):
                    var h_offset = h * layer_head_dim
                    var g = h // ratio
                    var g_offset = g * layer_head_dim
                    for t in range(pos + 1):
                        var t_cache_offset = layer_cache_offset + t * kv_stride + g_offset
                        var score: Float32 = 0.0
                        for d in range(layer_head_dim):
                            score += q[h_offset + d] * k_cache[t_cache_offset + d]
                        attn_scores[t] = score
                        
                    softmax(attn_scores, pos + 1)
                    
                    for d in range(layer_head_dim):
                        var val_sum: Float32 = 0.0
                        for t in range(pos + 1):
                            var t_cache_offset = layer_cache_offset + t * kv_stride + g_offset
                            val_sum += attn_scores[t] * v_cache[t_cache_offset + d]
                        attn_out[h_offset + d] = val_sum
                ta1 = std.time.perf_counter()
                if pos >= num_prompt_tokens:
                    p_attn_dot_product += ta1 - ta0

                # 7. Out projection and Attention Residual Add
                ta0 = std.time.perf_counter()
                q4_gemv_cpu(weights.wo_packed, weights.wo_scales, attn_out, proj_attn_out, layer_heads * layer_head_dim, CONFIG.embed_dim, num_tasks)
                rms_norm(proj_attn_out, weights.post_attention_layernorm_weight, proj_attn_out, CONFIG.embed_dim)
                for i in range(0, CONFIG.embed_dim, 8):
                    local_x.store[8](i, local_x.load[8](i) + proj_attn_out.load[8](i))
                ta1 = std.time.perf_counter()
                if pos >= num_prompt_tokens:
                    p_wo_proj += ta1 - ta0

                # 8. Parallel Feedforward: Shared MLP + sparse MoE Experts
                # Path A: Shared MLP
                ta0 = std.time.perf_counter()
                rms_norm(local_x, weights.pre_feedforward_layernorm_weight, norm_x, CONFIG.embed_dim)
                q4_gemv_cpu(weights.w_gate_packed, weights.w_gate_scales, norm_x, shared_h1, CONFIG.embed_dim, CONFIG.standard_mlp_dim, num_tasks)
                q4_gemv_cpu(weights.w_up_packed, weights.w_up_scales, norm_x, shared_h2, CONFIG.embed_dim, CONFIG.standard_mlp_dim, num_tasks)
                gelu_and_mul(shared_h1, shared_h2, shared_ffn_out_gate, CONFIG.standard_mlp_dim)
                q4_gemv_cpu(weights.w_down_packed, weights.w_down_scales, shared_ffn_out_gate, shared_mlp_out, CONFIG.standard_mlp_dim, CONFIG.embed_dim, num_tasks)
                rms_norm(shared_mlp_out, weights.post_feedforward_layernorm_1_weight, shared_h1_norm, CONFIG.embed_dim)
                ta1 = std.time.perf_counter()
                if pos >= num_prompt_tokens:
                    p_shared_mlp += ta1 - ta0

                # Path B: Router and Experts
                ta0 = std.time.perf_counter()
                rms_norm_noscale(local_x, norm_x, CONFIG.embed_dim)
                var r_scale_factor = Float32(1.0 / sqrt(2816.0))
                var r_scale_vec = SIMD[DType.float32, 8](r_scale_factor)
                for j in range(0, CONFIG.embed_dim, 8):
                    scaled_norm_x.store[8](j, norm_x.load[8](j) * weights.router_scale.load[8](j) * r_scale_vec)
                q4_gemv_cpu(weights.w_router_packed, weights.w_router_scales, scaled_norm_x, scores, CONFIG.embed_dim, CONFIG.num_experts, num_tasks)
                softmax(scores, CONFIG.num_experts)
                get_top_k[8](scores, top_idx, top_val, CONFIG.num_experts)
                
                var sum_top_val: Float32 = 0.0
                for i in range(CONFIG.top_k_experts):
                    sum_top_val += top_val[i]
                for i in range(CONFIG.top_k_experts):
                    var idx = top_idx[i]
                    var norm_prob = top_val[i] / (sum_top_val + 1e-10)
                    if norm_prob < 0.05:
                        top_weights[i] = 0.0
                        if pos >= num_prompt_tokens:
                            total_experts_skipped += 1
                    else:
                        top_weights[i] = norm_prob * weights.router_per_expert_scale[idx]
                    if pos >= num_prompt_tokens:
                        total_experts_evaluated += 1

                for i in range(0, CONFIG.embed_dim, 8):
                    moe_out.store[8](i, SIMD[DType.float32, 8](0.0))
                rms_norm(local_x, weights.pre_feedforward_layernorm_2_weight, moe_in, CONFIG.embed_dim)
                ta1 = std.time.perf_counter()
                if pos >= num_prompt_tokens:
                    p_moe_router += ta1 - ta0

                ta0 = std.time.perf_counter()
                def expert_worker(exp_idx: Int) capturing -> None:
                    var e = top_idx[exp_idx]
                    var w_e = top_weights[exp_idx]
                    if w_e == 0.0:
                        return

                    var exp_gate_up_out = gate_up_out_all + exp_idx * (CONFIG.expert_mlp_dim * 2)
                    var exp_act = act_all + exp_idx * CONFIG.expert_mlp_dim
                    var exp_down_out = down_out_all + exp_idx * CONFIG.embed_dim

                    var exp_gate_up_packed = weights.experts_gate_up_packed + e * (CONFIG.expert_mlp_dim * 2 * CONFIG.embed_dim // 2)
                    var exp_gate_up_scales = weights.experts_gate_up_scales + e * (CONFIG.expert_mlp_dim * 2 * CONFIG.embed_dim // 32)
                    q4_gemv_cpu(exp_gate_up_packed, exp_gate_up_scales, moe_in, exp_gate_up_out, CONFIG.embed_dim, CONFIG.expert_mlp_dim * 2, 1)

                    var gate_ptr = exp_gate_up_out
                    var up_ptr = exp_gate_up_out + CONFIG.expert_mlp_dim
                    gelu_and_mul(gate_ptr, up_ptr, exp_act, CONFIG.expert_mlp_dim)

                    var exp_down_packed = weights.experts_down_packed + e * (CONFIG.embed_dim * CONFIG.expert_mlp_dim // 2)
                    var exp_down_scales = weights.experts_down_scales + e * (CONFIG.embed_dim * CONFIG.expert_mlp_dim // 32)
                    q4_gemv_cpu(exp_down_packed, exp_down_scales, exp_act, exp_down_out, CONFIG.expert_mlp_dim, CONFIG.embed_dim, 1)

                parallelize[expert_worker](CONFIG.top_k_experts)
                ta1 = std.time.perf_counter()
                if pos >= num_prompt_tokens:
                    p_moe_experts += ta1 - ta0

                ta0 = std.time.perf_counter()
                for exp_idx in range(CONFIG.top_k_experts):
                    var w_e = top_weights[exp_idx]
                    if w_e == 0.0:
                        continue
                    var exp_down_out = down_out_all + exp_idx * CONFIG.embed_dim
                    var w_e_vec = SIMD[DType.float32, 8](w_e)
                    for j in range(0, CONFIG.embed_dim, 8):
                        moe_out.store[8](j, moe_out.load[8](j) + exp_down_out.load[8](j) * w_e_vec)

                rms_norm(moe_out, weights.post_feedforward_layernorm_2_weight, moe_out_norm, CONFIG.embed_dim)

                # Combine Shared MLP and MoE Path
                for i in range(0, CONFIG.embed_dim, 8):
                    combined_ffn.store[8](i, shared_h1_norm.load[8](i) + moe_out_norm.load[8](i))
                rms_norm(combined_ffn, weights.post_feedforward_layernorm_weight, ffn_out, CONFIG.embed_dim)

                # Add to residual and multiply by layer_scalar
                var scalar = weights.layer_scalar[0]
                var scalar_vec = SIMD[DType.float32, 8](scalar)
                for i in range(0, CONFIG.embed_dim, 8):
                    local_x.store[8](i, (local_x.load[8](i) + ffn_out.load[8](i)) * scalar_vec)
                ta1 = std.time.perf_counter()
                if pos >= num_prompt_tokens:
                    p_moe_combine += ta1 - ta0

        var t_prefill_end = std.time.perf_counter()
        print("\n  Prefill: ", num_prompt_tokens, " tokens in ",
              String((t_prefill_end - t_prefill_start) * 1000.0), " ms")

        # --- Canvas initialization ---
        var np_random = Python.import_module("numpy.random")
        var canvas_ids_py = np_random.randint(0, CONFIG.vocab_size, CANVAS_LEN)
        for t in range(CANVAS_LEN):
            canvas_tokens[t] = Int(py=canvas_ids_py[t])
        for i in range(CANVAS_LEN * CONFIG.embed_dim):
            sc_embeds[i] = 0.0

        # --- Sampler state ---
        var denoise_step: Int = 0
        var is_encoder_phase: Bool = False
        var hist_len: Int = 0
        var prev_mean_ent: Float32 = 0.0
        var steps_executed: Int = 0

        t_decode_start = std.time.perf_counter()

        # --- Denoise loop ---
        var t_max: Float32 = 0.8
        var t_min: Float32 = 0.4
        var entropy_bound: Float32 = 0.1
        var confidence_threshold: Float32 = 0.005

        while denoise_step < MAX_DENOISE_STEPS:
            print("  [denoise ", denoise_step, "]")
            steps_executed += 1

            # Compute canvas embeddings + self-conditioning
            for t in range(CANVAS_LEN):
                var hidden_t = canvas_hidden + t * CONFIG.embed_dim
                embed.reconstruct(canvas_tokens[t], hidden_t)
                var p_scale_vec = SIMD[DType.float32, 8](prompt_scale)
                for i in range(0, CONFIG.embed_dim, 8):
                    hidden_t.store[8](i, hidden_t.load[8](i) * p_scale_vec)
                sc_forward(
                    hidden_t,
                    sc_embeds + t * CONFIG.embed_dim,
                    sc_gate_packed, sc_up_packed, sc_down_packed,
                    sc_gate_scales, sc_up_scales, sc_down_scales,
                    sc_pre_norm,
                    sc_buf_normed, sc_buf_gate, sc_buf_up, sc_buf_gated, sc_buf_down,
                    sc_out_buf,
                    CONFIG.embed_dim, SC_SIZE, num_tasks,
                )
                for i in range(0, CONFIG.embed_dim, 8):
                    hidden_t.store[8](i, sc_out_buf.load[8](i))

            # Decoder forward: layer-first, batched projections, bidirectional attention
            for l in range(30):
                var weights = CONFIG.get_layer_weights(l, base_ptr + CONFIG.get_layer_offset(l), q8_attn_base)
                var is_global = CONFIG.is_global_layer(l)
                var layer_heads = 16
                var layer_kv_heads = 2 if is_global else 8
                var layer_head_dim = 512 if is_global else 256
                var kv_stride = layer_kv_heads * layer_head_dim
                var ratio = layer_heads // layer_kv_heads
                var layer_cache_offset = kv_cache_offsets[l]
                var seq_len = num_prompt_tokens + CANVAS_LEN
                var q_size = layer_heads * layer_head_dim
                var clflush = gpu_ctx.cpu_clflush
                var sfence = gpu_ctx.cpu_sfence

                var tl_qkv = std.time.perf_counter()
                # Phase 1a: Batched attention input norm + QKV projections
                rms_norm_batch(canvas_hidden, weights.input_layernorm_weight, canvas_norm, CANVAS_LEN, CONFIG.embed_dim)
                
                if use_gpu:
                    # Contiguous CPU cache flush on canvas_norm before enqueuing matmuls
                    var base_norm = (canvas_norm.__int__() // 64) * 64
                    var end_norm = canvas_norm.__int__() + (CANVAS_LEN * CONFIG.embed_dim * 4)
                    while base_norm < end_norm:
                        _ = clflush(base_norm)
                        base_norm += 64
                    _ = sfence(0)

                    # Asynchronously enqueue Q, K, and V projections for all tokens on GPU using Coalesced Batched GEMM
                    var slot_q = gpu_ctx.enqueue_matmul(
                        weights.wq_q8.__int__(),
                        weights.wq_scales.__int__(),
                        canvas_norm.__int__(),
                        temp_packed_q.__int__(),
                        CONFIG.embed_dim,
                        q_size,
                        batch_size=CANVAS_LEN,
                        flush_cache=False
                    )
                    var slot_k = gpu_ctx.enqueue_matmul(
                        weights.wk_q8.__int__(),
                        weights.wk_scales.__int__(),
                        canvas_norm.__int__(),
                        temp_packed_k.__int__(),
                        CONFIG.embed_dim,
                        kv_stride,
                        batch_size=CANVAS_LEN,
                        flush_cache=False
                    )
                    var slot_v = Int(0)
                    if not is_global:
                        slot_v = gpu_ctx.enqueue_matmul(
                            weights.wv_q8.__int__(),
                            weights.wv_scales.__int__(),
                            canvas_norm.__int__(),
                            temp_packed_v.__int__(),
                            CONFIG.embed_dim,
                            kv_stride,
                            batch_size=CANVAS_LEN,
                            flush_cache=False
                        )
                    
                    # Wait for GPU completions of the coalesced batched runs
                    gpu_ctx.wait_matmul(slot_q, temp_packed_q.__int__(), q_size, batch_size=CANVAS_LEN, flush_cache=False)
                    gpu_ctx.wait_matmul(slot_k, temp_packed_k.__int__(), kv_stride, batch_size=CANVAS_LEN, flush_cache=False)
                    if not is_global:
                        gpu_ctx.wait_matmul(slot_v, temp_packed_v.__int__(), kv_stride, batch_size=CANVAS_LEN, flush_cache=False)

                    # Un-stride from packed temp buffers to stride-padded canvas buffers
                    for t in range(CANVAS_LEN):
                        var src_ptr = temp_packed_q + t * q_size
                        var dst_ptr = canvas_q + t * MAX_Q_SIZE
                        for i in range(0, q_size, 8):
                            dst_ptr.store[8](i, src_ptr.load[8](i))

                    for t in range(CANVAS_LEN):
                        var src_ptr = temp_packed_k + t * kv_stride
                        var dst_ptr = canvas_k + t * MAX_KV_SIZE
                        for i in range(0, kv_stride, 8):
                            dst_ptr.store[8](i, src_ptr.load[8](i))

                    if is_global:
                        for t in range(CANVAS_LEN):
                            var v_ptr = canvas_v + t * MAX_KV_SIZE
                            var k_ptr = canvas_k + t * MAX_KV_SIZE
                            for i in range(0, kv_stride, 8):
                                v_ptr.store[8](i, k_ptr.load[8](i))
                    else:
                        for t in range(CANVAS_LEN):
                            var src_ptr = temp_packed_v + t * kv_stride
                            var dst_ptr = canvas_v + t * MAX_KV_SIZE
                            for i in range(0, kv_stride, 8):
                                dst_ptr.store[8](i, src_ptr.load[8](i))

                    # Contiguous CPU cache invalidate on canvas_q, canvas_k, and canvas_v
                    var base_q = (canvas_q.__int__() // 64) * 64
                    var end_q = canvas_q.__int__() + (CANVAS_LEN * MAX_Q_SIZE * 4)
                    while base_q < end_q:
                        _ = clflush(base_q)
                        base_q += 64

                    var base_k = (canvas_k.__int__() // 64) * 64
                    var end_k = canvas_k.__int__() + (CANVAS_LEN * MAX_KV_SIZE * 4)
                    while base_k < end_k:
                        _ = clflush(base_k)
                        base_k += 64

                    if not is_global:
                        var base_v = (canvas_v.__int__() // 64) * 64
                        var end_v = canvas_v.__int__() + (CANVAS_LEN * MAX_KV_SIZE * 4)
                        while base_v < end_v:
                            _ = clflush(base_v)
                            base_v += 64
                    _ = sfence(0)
                else:
                    # CPU Fallback using General Tiled batched q4 GEMV
                    q4_gemv_tiled[8](weights.wq_packed, weights.wq_scales, canvas_norm, canvas_q, CANVAS_LEN, CONFIG.embed_dim, q_size, num_tasks, in_stride=CONFIG.embed_dim, out_stride=MAX_Q_SIZE)
                    q4_gemv_tiled[8](weights.wk_packed, weights.wk_scales, canvas_norm, canvas_k, CANVAS_LEN, CONFIG.embed_dim, kv_stride, num_tasks, in_stride=CONFIG.embed_dim, out_stride=MAX_KV_SIZE)
                    if is_global:
                        for t in range(CANVAS_LEN):
                            var v_ptr = canvas_v + t * MAX_KV_SIZE
                            var k_ptr = canvas_k + t * MAX_KV_SIZE
                            for i in range(0, kv_stride, 8):
                                v_ptr.store[8](i, k_ptr.load[8](i))
                    else:
                        q4_gemv_tiled[8](weights.wv_packed, weights.wv_scales, canvas_norm, canvas_v, CANVAS_LEN, CONFIG.embed_dim, kv_stride, num_tasks, in_stride=CONFIG.embed_dim, out_stride=MAX_KV_SIZE)
                var tl_qkv_end = std.time.perf_counter()

                # Phase 1b: Head norms + RoPE + KV cache write (per-token, element-wise)
                for t in range(CANVAS_LEN):
                    var pos = num_prompt_tokens + t
                    var q_store = canvas_q + t * MAX_Q_SIZE
                    var k_store = canvas_k + t * MAX_KV_SIZE
                    var v_store = canvas_v + t * MAX_KV_SIZE

                    for h in range(layer_heads):
                        rms_norm(q_store + h * layer_head_dim, weights.q_norm_weight, q_store + h * layer_head_dim, layer_head_dim)
                    for h in range(layer_kv_heads):
                        rms_norm(k_store + h * layer_head_dim, weights.k_norm_weight, k_store + h * layer_head_dim, layer_head_dim)
                    for h in range(layer_kv_heads):
                        rms_norm_noscale(v_store + h * layer_head_dim, v_store + h * layer_head_dim, layer_head_dim)

                    apply_rope(q_store, layer_heads, layer_head_dim, pos, is_global)
                    apply_rope(k_store, layer_kv_heads, layer_head_dim, pos, is_global)

                    var step_cache_offset = layer_cache_offset + pos * kv_stride
                    for i in range(0, kv_stride, 8):
                        k_cache.store[8](step_cache_offset + i, k_store.load[8](i))
                        v_cache.store[8](step_cache_offset + i, v_store.load[8](i))

                # Phase 2a: Bidirectional attention (parallelized over canvas tokens)
                var tl_attn = std.time.perf_counter()
                def attn_worker(t: Int) capturing -> None:
                    var q_load = canvas_q + t * MAX_Q_SIZE
                    var attn_out_ptr = canvas_attn_out + t * MAX_Q_SIZE
                    var scores_t = canvas_attn_scores + t * CONFIG.max_seq_len

                    var q_regs = InlineArray[SIMD[DType.float32, 8], 64](uninitialized=True)
                    var acc_regs = InlineArray[SIMD[DType.float32, 8], 64](uninitialized=True)

                    for h in range(layer_heads):
                        var h_offset = h * layer_head_dim
                        var g = h // ratio
                        var g_offset = g * layer_head_dim
                        var num_simd_lanes = layer_head_dim // 8

                        # Cache query head into thread-local stack registers
                        for d in range(num_simd_lanes):
                            q_regs[d] = q_load.load[8](h_offset + d * 8)

                        for tt in range(seq_len):
                            var t_cache_offset = layer_cache_offset + tt * kv_stride + g_offset
                            var score_vec = SIMD[DType.float32, 8](0.0)
                            for d in range(num_simd_lanes):
                                score_vec += q_regs[d] * k_cache.load[8](t_cache_offset + d * 8)
                            scores_t[tt] = score_vec.reduce_add()

                        softmax(scores_t, seq_len)

                        # Accumulate value vectors directly in registers
                        for d in range(num_simd_lanes):
                            acc_regs[d] = SIMD[DType.float32, 8](0.0)

                        for tt in range(seq_len):
                            var t_cache_offset = layer_cache_offset + tt * kv_stride + g_offset
                            var s_vec = SIMD[DType.float32, 8](scores_t[tt])
                            for d in range(num_simd_lanes):
                                var v_vec = v_cache.load[8](t_cache_offset + d * 8)
                                acc_regs[d] += s_vec * v_vec

                        # Write back the final compiled attention heads once
                        for d in range(num_simd_lanes):
                            attn_out_ptr.store[8](h_offset + d * 8, acc_regs[d])

                parallelize[attn_worker](CANVAS_LEN)

                var tl_attn_end = std.time.perf_counter()

                if use_gpu:
                    # Contiguous CPU cache flush on canvas_attn_out before Wo projection
                    var base_attn_out = (canvas_attn_out.__int__() // 64) * 64
                    var end_attn_out = canvas_attn_out.__int__() + (CANVAS_LEN * MAX_Q_SIZE * 4)
                    while base_attn_out < end_attn_out:
                        _ = clflush(base_attn_out)
                        base_attn_out += 64
                    _ = sfence(0)

                    # Pack/copy strided canvas_attn_out into temp_packed_q before enqueuing matmul
                    for t in range(CANVAS_LEN):
                        var src_ptr = canvas_attn_out + t * MAX_Q_SIZE
                        var dst_ptr = temp_packed_q + t * q_size
                        for i in range(0, q_size, 8):
                            dst_ptr.store[8](i, src_ptr.load[8](i))

                    # Contiguous CPU cache flush on temp_packed_q before Wo projection
                    var base_temp_q = (temp_packed_q.__int__() // 64) * 64
                    var end_temp_q = temp_packed_q.__int__() + (CANVAS_LEN * q_size * 4)
                    while base_temp_q < end_temp_q:
                        _ = clflush(base_temp_q)
                        base_temp_q += 64
                    _ = sfence(0)

                    # Phase 2b: Coalesced Batched Wo projection + post-attn norm + residual
                    var slot_o = gpu_ctx.enqueue_matmul(
                        weights.wo_q8.__int__(),
                        weights.wo_scales.__int__(),
                        temp_packed_q.__int__(),
                        canvas_proj_attn.__int__(),
                        q_size,
                        CONFIG.embed_dim,
                        batch_size=CANVAS_LEN,
                        flush_cache=False
                    )
                    gpu_ctx.wait_matmul(slot_o, canvas_proj_attn.__int__(), CONFIG.embed_dim, batch_size=CANVAS_LEN, flush_cache=False)

                    # Contiguous CPU cache invalidate on canvas_proj_attn
                    var base_proj = (canvas_proj_attn.__int__() // 64) * 64
                    var end_proj = canvas_proj_attn.__int__() + (CANVAS_LEN * CONFIG.embed_dim * 4)
                    while base_proj < end_proj:
                        _ = clflush(base_proj)
                        base_proj += 64
                    _ = sfence(0)
                else:
                    # Phase 2b: CPU Fallback using General Tiled batched q4 GEMV
                    q4_gemv_tiled[8](weights.wo_packed, weights.wo_scales, canvas_attn_out, canvas_proj_attn, CANVAS_LEN, q_size, CONFIG.embed_dim, num_tasks, in_stride=MAX_Q_SIZE, out_stride=CONFIG.embed_dim)
                rms_norm_batch(canvas_proj_attn, weights.post_attention_layernorm_weight, canvas_proj_attn, CANVAS_LEN, CONFIG.embed_dim)
                for t in range(CANVAS_LEN):
                    var hidden = canvas_hidden + t * CONFIG.embed_dim
                    var proj = canvas_proj_attn + t * CONFIG.embed_dim
                    for i in range(0, CONFIG.embed_dim, 8):
                        hidden.store[8](i, hidden.load[8](i) + proj.load[8](i))

                # Phase 2c: Batched FFN (shared MLP path)
                var tl_ffn = std.time.perf_counter()
                rms_norm_batch(canvas_hidden, weights.pre_feedforward_layernorm_weight, canvas_norm, CANVAS_LEN, CONFIG.embed_dim)
                q4_gemv_tiled[8](weights.w_gate_packed, weights.w_gate_scales, canvas_norm, canvas_ffn_gate, CANVAS_LEN, CONFIG.embed_dim, CONFIG.standard_mlp_dim, num_tasks)
                q4_gemv_tiled[8](weights.w_up_packed, weights.w_up_scales, canvas_norm, canvas_ffn_up, CANVAS_LEN, CONFIG.embed_dim, CONFIG.standard_mlp_dim, num_tasks)
                gelu_and_mul_batch(canvas_ffn_gate, canvas_ffn_up, canvas_ffn_gated, CANVAS_LEN, CONFIG.standard_mlp_dim)
                q4_gemv_tiled[8](weights.w_down_packed, weights.w_down_scales, canvas_ffn_gated, canvas_ffn_down, CANVAS_LEN, CONFIG.standard_mlp_dim, CONFIG.embed_dim, num_tasks)
                rms_norm_batch(canvas_ffn_down, weights.post_feedforward_layernorm_1_weight, canvas_shared_norm, CANVAS_LEN, CONFIG.embed_dim)
                var tl_ffn_end = std.time.perf_counter()

                # Phase 2d: MoE path (parallelized over tokens, sequential experts per token)
                var tl_moe = std.time.perf_counter()
                rms_norm_batch(canvas_hidden, weights.pre_feedforward_layernorm_2_weight, canvas_moe_in, CANVAS_LEN, CONFIG.embed_dim)
                var r_scale_factor = Float32(1.0 / sqrt(2816.0))

                def router_worker(t: Int) capturing -> None:
                    var hidden_inout = canvas_hidden + t * CONFIG.embed_dim
                    var scores_t = canvas_router_scores + t * CONFIG.num_experts
                    var top_idx_t = canvas_top_idx + t * CONFIG.top_k_experts
                    var top_val_t = canvas_top_val + t * CONFIG.top_k_experts
                    var top_weights_t = canvas_top_weights + t * CONFIG.top_k_experts
                    var scaled_norm_t = canvas_scaled_norm + t * CONFIG.embed_dim
                    var routed_norm_t = canvas_routed_norm + t * CONFIG.embed_dim

                    rms_norm_noscale(hidden_inout, routed_norm_t, CONFIG.embed_dim)
                    var r_scale_vec = SIMD[DType.float32, 8](r_scale_factor)
                    for j in range(0, CONFIG.embed_dim, 8):
                        scaled_norm_t.store[8](j, routed_norm_t.load[8](j) * weights.router_scale.load[8](j) * r_scale_vec)
                    q4_gemv_cpu(weights.w_router_packed, weights.w_router_scales, scaled_norm_t, scores_t, CONFIG.embed_dim, CONFIG.num_experts, 1)
                    softmax(scores_t, CONFIG.num_experts)
                    get_top_k[8](scores_t, top_idx_t, top_val_t, CONFIG.num_experts)

                    var sum_top_val: Float32 = 0.0
                    for i in range(CONFIG.top_k_experts):
                        sum_top_val += top_val_t[i]
                    for i in range(CONFIG.top_k_experts):
                        var idx = top_idx_t[i]
                        var norm_prob = top_val_t[i] / (sum_top_val + 1e-10)
                        if norm_prob < 0.05:
                            top_weights_t[i] = 0.0
                        else:
                            top_weights_t[i] = norm_prob * weights.router_per_expert_scale[idx]

                parallelize[router_worker](CANVAS_LEN)

                # 1. Zero out expert counts
                var expert_counts = InlineArray[Int, 128](uninitialized=True)
                for i in range(128):
                    expert_counts[i] = 0

                # 2. Count active assignments per expert
                for t in range(CANVAS_LEN):
                    var top_weights_t = canvas_top_weights + t * CONFIG.top_k_experts
                    var top_idx_t = canvas_top_idx + t * CONFIG.top_k_experts
                    for s in range(CONFIG.top_k_experts):
                        var w_e = top_weights_t[s]
                        if w_e > 0.0:
                            var e = top_idx_t[s]
                            expert_counts[e] += 1

                # 3. Compute prefix sum (offsets)
                var expert_offsets = InlineArray[Int, 128](uninitialized=True)
                var current_offset = 0
                for e in range(128):
                    expert_offsets[e] = current_offset
                    current_offset += expert_counts[e]

                # 4. Fill assignment arrays
                var write_offsets = InlineArray[Int, 128](uninitialized=True)
                for e in range(128):
                    write_offsets[e] = expert_offsets[e]

                var assignments_t = InlineArray[Int, 2048](uninitialized=True)
                var assignments_s = InlineArray[Int, 2048](uninitialized=True)

                for t in range(CANVAS_LEN):
                    var top_weights_t = canvas_top_weights + t * CONFIG.top_k_experts
                    var top_idx_t = canvas_top_idx + t * CONFIG.top_k_experts
                    for s in range(CONFIG.top_k_experts):
                        var w_e = top_weights_t[s]
                        if w_e > 0.0:
                            var e = top_idx_t[s]
                            var write_pos = write_offsets[e]
                            assignments_t[write_pos] = t
                            assignments_s[write_pos] = s
                            write_offsets[e] += 1

                # 5. Zero out canvas_moe_out before the expert loop
                for t in range(CANVAS_LEN):
                    var dst = canvas_moe_out + t * CONFIG.embed_dim
                    for j in range(0, CONFIG.embed_dim, 8):
                        dst.store[8](j, SIMD[DType.float16, 8](0.0))

                # 6. Sequential expert loop (no closures here!)
                for e in range(128):
                    var cnt = expert_counts[e]
                    if cnt == 0:
                        continue
                    var offset = expert_offsets[e]

                    # Gather inputs for expert e
                    for i in range(cnt):
                        var t = assignments_t[offset + i]
                        var src = canvas_moe_in + t * CONFIG.embed_dim
                        var dst = temp_x + i * CONFIG.embed_dim
                        for j in range(0, CONFIG.embed_dim, 8):
                            dst.store[8](j, src.load[8](j))

                    # Gate + Up projection
                    var exp_gate_up_packed = weights.experts_gate_up_packed + e * (CONFIG.expert_mlp_dim * 2 * CONFIG.embed_dim // 2)
                    var exp_gate_up_scales = weights.experts_gate_up_scales + e * (CONFIG.expert_mlp_dim * 2 * CONFIG.embed_dim // 32)
                    q4_gemv_moe_dispatch(
                        exp_gate_up_packed,
                        exp_gate_up_scales,
                        temp_x,
                        temp_gate_up,
                        cnt,
                        CONFIG.embed_dim,
                        CONFIG.expert_mlp_dim * 2,
                        num_tasks,
                        CONFIG.embed_dim,
                        CONFIG.expert_mlp_dim * 2
                    )

                    # Activation
                    for i in range(cnt):
                        var gate_ptr = temp_gate_up + i * (CONFIG.expert_mlp_dim * 2)
                        var up_ptr = gate_ptr + CONFIG.expert_mlp_dim
                        var act_ptr = temp_act + i * CONFIG.expert_mlp_dim
                        gelu_and_mul(gate_ptr, up_ptr, act_ptr, CONFIG.expert_mlp_dim)

                    # Down projection
                    var exp_down_packed = weights.experts_down_packed + e * (CONFIG.embed_dim * CONFIG.expert_mlp_dim // 2)
                    var exp_down_scales = weights.experts_down_scales + e * (CONFIG.embed_dim * CONFIG.expert_mlp_dim // 32)
                    q4_gemv_moe_dispatch(
                        exp_down_packed,
                        exp_down_scales,
                        temp_act,
                        temp_down,
                        cnt,
                        CONFIG.expert_mlp_dim,
                        CONFIG.embed_dim,
                        num_tasks,
                        CONFIG.expert_mlp_dim,
                        CONFIG.embed_dim
                    )

                    # Scatter and accumulate back to canvas_moe_out
                    for i in range(cnt):
                        var t = assignments_t[offset + i]
                        var s = assignments_s[offset + i]
                        var top_weights_t = canvas_top_weights + t * CONFIG.top_k_experts
                        var w_e = top_weights_t[s]
                        
                        var src = temp_down + i * CONFIG.embed_dim
                        var dst = canvas_moe_out + t * CONFIG.embed_dim
                        for j in range(0, CONFIG.embed_dim, 8):
                            var val = dst.load[8](j).cast[DType.float32]() + src.load[8](j).cast[DType.float32]() * w_e
                            dst.store[8](j, val.cast[DType.float16]())

                # Phase 2e: Batched combine + final norm + residual + scalar
                var tl_moe_end = std.time.perf_counter()
                rms_norm_batch(canvas_moe_out, weights.post_feedforward_layernorm_2_weight, canvas_moe_norm, CANVAS_LEN, CONFIG.embed_dim)
                var scalar = weights.layer_scalar[0]
                for t in range(CANVAS_LEN):
                    var hidden = canvas_hidden + t * CONFIG.embed_dim
                    var shared_n = canvas_shared_norm + t * CONFIG.embed_dim
                    var moe_n = canvas_moe_norm + t * CONFIG.embed_dim
                    var combined = canvas_combined + t * CONFIG.embed_dim
                    var ffn = canvas_ffn_out + t * CONFIG.embed_dim
                    for i in range(0, CONFIG.embed_dim, 8):
                        combined.store[8](i, shared_n.load[8](i) + moe_n.load[8](i))
                    rms_norm(combined, weights.post_feedforward_layernorm_weight, ffn, CONFIG.embed_dim)
                    var scalar_vec = SIMD[DType.float32, 8](scalar)
                    for i in range(0, CONFIG.embed_dim, 8):
                        hidden.store[8](i, (hidden.load[8](i) + ffn.load[8](i).cast[DType.float32]()) * scalar_vec)

                if l == 0 or l == 15 or l == 29:
                    print("    layer ", l, " qkv=", String((tl_qkv_end - tl_qkv) * 1000), "ms",
                          " attn=", String((tl_attn_end - tl_attn) * 1000), "ms",
                          " ffn=", String((tl_ffn_end - tl_ffn) * 1000), "ms",
                          " moe=", String((tl_moe_end - tl_moe) * 1000), "ms")

            # --- Sampler step ---
            # Temperature schedule
            var remaining = MAX_DENOISE_STEPS - denoise_step
            if remaining < 1:
                remaining = 1
            var temp = t_min + (t_max - t_min) * Float32(remaining) / Float32(MAX_DENOISE_STEPS)
            if temp < 1e-10:
                temp = 1e-10

            # Per-canvas-position: full logits, log_softmax, entropy, sample, argmax, SC soft embed
            var rand_u = np_random.rand(CANVAS_LEN)
            var rand_buf = alloc[Float32](CANVAS_LEN)
            for t in range(CANVAS_LEN):
                rand_buf[t] = Float32(Float64(py=rand_u[t]))

            # Batched logit projection: read embed weights ONCE for all 256 tokens
            rms_norm_batch(canvas_hidden, final_norm_weight, canvas_final_norm, CANVAS_LEN, CONFIG.embed_dim)
            q4_gemv_tiled[8](embed.packed_weights, embed.scales, canvas_final_norm, canvas_logits,
                          CANVAS_LEN, CONFIG.embed_dim, CONFIG.vocab_size, num_tasks,
                          CONFIG.embed_dim, CONFIG.vocab_size)

            # Parallelized per-token: scale+softcap, softmax, entropy, argmax, sample, sc_embed
            def sampler_worker(t: Int) capturing -> None:
                var logits_t = canvas_logits + t * CONFIG.vocab_size
                var probs_t = canvas_probs + t * CONFIG.vocab_size

                for v in range(CONFIG.vocab_size):
                    logits_t[v] = (Float32(30.0) * tanh(logits_t[v] / Float32(30.0))) / temp

                var max_logit: Float32 = logits_t[0]
                for v in range(1, CONFIG.vocab_size):
                    if logits_t[v] > max_logit:
                        max_logit = logits_t[v]
                var sum_exp: Float32 = 0.0
                for v in range(CONFIG.vocab_size):
                    var lp = exp(logits_t[v] - max_logit)
                    probs_t[v] = lp
                    sum_exp += lp
                var log_sum = log(sum_exp + 1e-30)

                var ent: Float32 = 0.0
                for v in range(CONFIG.vocab_size):
                    var p = probs_t[v] / (sum_exp + 1e-30)
                    probs_t[v] = p
                    if p > 1e-12:
                        ent -= p * (logits_t[v] - max_logit - log_sum)
                token_entropy[t] = ent

                var t_argmax = 0
                var t_max_logit = logits_t[0]
                for v in range(1, CONFIG.vocab_size):
                    if logits_t[v] > t_max_logit:
                        t_max_logit = logits_t[v]
                        t_argmax = v
                argmax_tokens[t] = t_argmax

                var u = rand_buf[t]
                var cumsum: Float32 = 0.0
                var sampled = t_argmax
                for v in range(CONFIG.vocab_size):
                    cumsum += probs_t[v]
                    if cumsum >= u:
                        sampled = v
                        break
                new_tokens[t] = sampled

                probs_to_soft_embed(
                    probs_t,
                    embed.packed_weights, embed.scales,
                    sc_embeds + t * CONFIG.embed_dim,
                    CONFIG.vocab_size, CONFIG.embed_dim, 1,
                )
                var sc_embeds_t = sc_embeds + t * CONFIG.embed_dim
                var p_scale_vec = SIMD[DType.float32, 8](prompt_scale)
                for i in range(0, CONFIG.embed_dim, 8):
                    sc_embeds_t.store[8](i, sc_embeds_t.load[8](i) * p_scale_vec)

            parallelize[sampler_worker](CANVAS_LEN)
            rand_buf.free()

            # Mean entropy + confidence
            var mean_ent: Float32 = 0.0
            for t in range(CANVAS_LEN):
                mean_ent += token_entropy[t]
            mean_ent /= Float32(CANVAS_LEN)
            var confident = mean_ent < confidence_threshold
            print("    mean_entropy=", String(mean_ent), " temp=", String(temp),
                  " confident=", confident)

            # Entropy-bound acceptance mask: sort by ascending entropy, cumsum, cummax
            # Selection sort for 256 elements
            for t in range(CANVAS_LEN):
                sorted_idx[t] = t
                sorted_ent[t] = token_entropy[t]
            for i in range(CANVAS_LEN):
                var min_idx = i
                for j in range(i + 1, CANVAS_LEN):
                    if sorted_ent[j] < sorted_ent[min_idx]:
                        min_idx = j
                if min_idx != i:
                    var tmp_e = sorted_ent[i]
                    sorted_ent[i] = sorted_ent[min_idx]
                    sorted_ent[min_idx] = tmp_e
                    var tmp_i = sorted_idx[i]
                    sorted_idx[i] = sorted_idx[min_idx]
                    sorted_idx[min_idx] = tmp_i

            # cumsum and cummax, accept where (cumsum - cummax) <= entropy_bound
            var cumsum_ent: Float32 = 0.0
            var cummax_ent: Float32 = 0.0
            for t in range(CANVAS_LEN):
                eb_mask[t] = False
            for i in range(CANVAS_LEN):
                cumsum_ent += sorted_ent[i]
                if sorted_ent[i] > cummax_ent:
                    cummax_ent = sorted_ent[i]
                if (cumsum_ent - cummax_ent) <= entropy_bound:
                    var orig_idx = sorted_idx[i]
                    eb_mask[orig_idx] = True

            # Update history: write argmax_tokens at slot hist_len % ST
            var write_pos = hist_len % 2
            for t in range(CANVAS_LEN):
                history[write_pos * CANVAS_LEN + t] = argmax_tokens[t]
            hist_len += 1

            # Stability check: all ST slots agree
            var stable = True
            if hist_len >= 2:
                for t in range(CANVAS_LEN):
                    if history[0 * CANVAS_LEN + t] != history[1 * CANVAS_LEN + t]:
                        stable = False
                        break
            else:
                stable = False

            # Convergence
            var converged = (stable and confident and hist_len >= 2) or (denoise_step + 1 >= MAX_DENOISE_STEPS)
            print("    stable=", stable, " converged=", converged)

            # Update canvas: accept → new_tokens, reject → random re-noise
            var n_accept = 0
            for t in range(CANVAS_LEN):
                if eb_mask[t]:
                    n_accept += 1
            var n_reject = CANVAS_LEN - n_accept
            if n_reject > 0:
                var reject_ids_py = np_random.randint(0, CONFIG.vocab_size, n_reject)
                var reject_idx = 0
                for t in range(CANVAS_LEN):
                    if eb_mask[t]:
                        canvas_tokens[t] = new_tokens[t]
                    else:
                        canvas_tokens[t] = Int(py=reject_ids_py[reject_idx])
                        reject_idx += 1
            else:
                for t in range(CANVAS_LEN):
                    canvas_tokens[t] = new_tokens[t]
            print("    accepted=", n_accept, "/", CANVAS_LEN)

            # Calculate local change metrics for adaptive step skipping
            var num_flips = 0
            var entropy_decay: Float32 = 0.0
            if steps_executed > 1:
                for t in range(CANVAS_LEN):
                    if argmax_tokens[t] != argmax_canvas[t]:
                        num_flips += 1
                entropy_decay = prev_mean_ent - mean_ent

            var skip_amount = 1
            if steps_executed > 1:
                # If very few tokens changed or entropy decay has flattened out, skip intermediate steps
                if num_flips <= 2 or (entropy_decay >= 0.0 and entropy_decay < 0.003):
                    skip_amount = 3
                    print("    -> Adaptive skip: num_flips =", num_flips, "entropy_decay =", String(entropy_decay), "| Skipping 2 steps")
                elif num_flips <= 8 or (entropy_decay >= 0.0 and entropy_decay < 0.008):
                    skip_amount = 2
                    print("    -> Adaptive skip: num_flips =", num_flips, "entropy_decay =", String(entropy_decay), "| Skipping 1 step")

            # Store metrics for next step
            prev_mean_ent = mean_ent

            # Update argmax_canvas
            for t in range(CANVAS_LEN):
                argmax_canvas[t] = argmax_tokens[t]

            # If converged: snap canvas to argmax and break
            if converged:
                for t in range(CANVAS_LEN):
                    canvas_tokens[t] = argmax_canvas[t]
                break

            # Zero out sc_embeds for positions that will commit (none in denoise-only mode)
            # All positions continue denoising, so sc_embeds stays

            denoise_step += skip_amount

        var t_end = std.time.perf_counter()
        var denoise_duration = t_end - t_decode_start

        print("\n---------------------------------------------------------")
        print("Diffusion generation complete!")
        print("  Prompt tokens:    ", num_prompt_tokens)
        print("  Canvas length:    ", CANVAS_LEN)
        print("  Denoise steps:    ", steps_executed)
        print("  Total denoise time: ", String(denoise_duration * 1000.0), " ms")
        print("  Avg per step:     ", String(denoise_duration * 1000.0 / Float64(steps_executed)), " ms")
        print("  Output:")
        var output_list = Python.list()
        for t in range(CANVAS_LEN):
            output_list.append(argmax_canvas[t])
        var decoded_output = tokenizer.decode(output_list)
        py_builtins.print("    ", decoded_output)
        print("---------------------------------------------------------")

    # Cleanup memory
    gpu_ctx.free_gpu_buffer(weights_gpu_buf)
    final_norm_weight.free()
    k_cache.free()
    v_cache.free()
    local_x.free()
    norm_x.free()
    scaled_norm_x.free()
    q.free()
    k_vec.free()
    v_vec.free()
    attn_scores.free()
    attn_out.free()
    proj_attn_out.free()
    shared_h1.free()
    shared_h2.free()
    shared_ffn_out_gate.free()
    shared_mlp_out.free()
    shared_h1_norm.free()
    moe_in.free()
    moe_out.free()
    gate_up_out_all.free()
    act_all.free()
    down_out_all.free()
    moe_out_norm.free()
    combined_ffn.free()
    ffn_out.free()
    scores.free()
    top_idx.free()
    top_val.free()
    top_weights.free()
    target_final_norm_x.free()
    local_max_logits.free()
    local_next_tokens.free()
    gpu_ctx.free_gpu_buffer(canvas_hidden_buf)
    gpu_ctx.free_gpu_buffer(canvas_q_buf)
    gpu_ctx.free_gpu_buffer(canvas_k_buf)
    gpu_ctx.free_gpu_buffer(canvas_v_buf)
    gpu_ctx.free_gpu_buffer(canvas_attn_out_buf)
    gpu_ctx.free_gpu_buffer(canvas_proj_attn_buf)
    gpu_ctx.free_gpu_buffer(canvas_norm_buf)
    gpu_ctx.free_gpu_buffer(temp_packed_q_buf)
    gpu_ctx.free_gpu_buffer(temp_packed_k_buf)
    gpu_ctx.free_gpu_buffer(temp_packed_v_buf)
    canvas_ffn_gate.free()
    canvas_ffn_up.free()
    canvas_ffn_gated.free()
    canvas_ffn_down.free()
    canvas_shared_norm.free()
    canvas_moe_in.free()
    canvas_moe_out.free()
    canvas_moe_norm.free()
    canvas_combined.free()
    canvas_ffn_out.free()
    canvas_attn_scores.free()
    temp_x.free()
    temp_gate_up.free()
    temp_act.free()
    temp_down.free()
    canvas_router_scores.free()
    canvas_top_idx.free()
    canvas_top_val.free()
    canvas_top_weights.free()
    canvas_scaled_norm.free()
    canvas_routed_norm.free()
    canvas_tokens.free()
    argmax_canvas.free()
    sc_embeds.free()
    sc_buf_normed.free()
    sc_buf_gate.free()
    sc_buf_up.free()
    sc_buf_gated.free()
    sc_buf_down.free()
    sc_out_buf.free()
    canvas_final_norm.free()
    sampler_logits.free()
    sampler_probs.free()
    canvas_logits.free()
    canvas_probs.free()
    token_entropy.free()
    new_tokens.free()
    argmax_tokens.free()
    sorted_idx.free()
    sorted_ent.free()
    eb_mask.free()
    history.free()
