from std.ffi import OwnedDLHandle
from std.memory import alloc
from std.collections import InlineArray, List
import std.os
import std.time

# ==============================================================================
# AMD KFD Bare-Metal Userspace Runtime & Dynamic ELF64 (.hsaco) Loader
# ==============================================================================
# Zero-ROCm / Zero-HIP userspace compute runtime for AMD APUs/GPUs via /dev/kfd
# and /dev/dri/renderD128. Provides:
#   1. SVM / GTT zero-copy page-aligned memory allocation & cache-line coherency
#   2. Dynamic ELF64 (.hsaco) PT_LOAD segment mapper and symbol resolver (.kd)
#   3. Multi-packet AQL ring buffer scheduling with optional barrier & doorbell
#      batching for concurrent multi-kernel execution across Vega CUs.
# ==============================================================================


struct kfd_ioctl_acquire_vm_args(Copyable, Movable):
    var drm_fd: UInt32
    var gpu_id: UInt32

    def __init__(out self, drm_fd: UInt32, gpu_id: UInt32):
        self.drm_fd = drm_fd
        self.gpu_id = gpu_id


struct kfd_ioctl_runtime_enable_args(Copyable, Movable):
    var r_debug: UInt64
    var mode_mask: UInt32
    var capabilities_mask: UInt32

    def __init__(out self):
        self.r_debug = 0
        self.mode_mask = 0
        self.capabilities_mask = 0


struct kfd_ioctl_alloc_memory_of_gpu_args(Copyable, Movable):
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


struct kfd_ioctl_map_memory_to_gpu_args(Copyable, Movable):
    var handle: UInt64
    var device_ids_array_ptr: UInt64
    var n_devices: UInt32
    var pad: UInt32

    def __init__(out self, handle: UInt64, device_ids_array_ptr: UInt64, n_devices: UInt32):
        self.handle = handle
        self.device_ids_array_ptr = device_ids_array_ptr
        self.n_devices = n_devices
        self.pad = 0


struct kfd_ioctl_unmap_memory_from_gpu_args(Copyable, Movable):
    var handle: UInt64
    var device_ids_array_ptr: UInt64
    var n_devices: UInt32
    var pad: UInt32

    def __init__(out self, handle: UInt64, device_ids_array_ptr: UInt64, n_devices: UInt32):
        self.handle = handle
        self.device_ids_array_ptr = device_ids_array_ptr
        self.n_devices = n_devices
        self.pad = 0


struct kfd_ioctl_free_memory_of_gpu_args(Copyable, Movable):
    var handle: UInt64

    def __init__(out self, handle: UInt64):
        self.handle = handle


struct kfd_ioctl_create_queue_args(Copyable, Movable):
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
        sdma_engine_id: UInt32 = 0,
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


struct kfd_ioctl_destroy_queue_args(Copyable, Movable):
    var queue_id: UInt32
    var pad: UInt32

    def __init__(out self, queue_id: UInt32):
        self.queue_id = queue_id
        self.pad = 0


struct GPUMemoryBuffer(ImplicitlyCopyable, Copyable, Movable):
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


struct hsa_kernel_dispatch_packet_t(ImplicitlyCopyable, Copyable, Movable):
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
        completion_signal: UInt64,
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


struct HSACOKernel(ImplicitlyCopyable, Copyable, Movable):
    """Resolved AMDGPU kernel descriptor metadata from an ELF64 .hsaco object."""
    var kd_va_addr: UInt64
    var kd_offset: Int
    var group_segment_size: UInt32
    var private_segment_size: UInt32

    def __init__(out self):
        self.kd_va_addr = 0
        self.kd_offset = 0
        self.group_segment_size = 0
        self.private_segment_size = 0

    def __init__(
        out self,
        kd_va_addr: UInt64,
        kd_offset: Int,
        group_segment_size: UInt32,
        private_segment_size: UInt32,
    ):
        self.kd_va_addr = kd_va_addr
        self.kd_offset = kd_offset
        self.group_segment_size = group_segment_size
        self.private_segment_size = private_segment_size


struct HSACOModule(Copyable, Movable):
    """Loaded ELF64 .hsaco code object mapped into GPU GTT memory."""
    var gpu_buf: GPUMemoryBuffer
    var raw_bytes: UnsafePointer[UInt8, MutUntrackedOrigin]
    var raw_size: Int

    def __init__(out self):
        self.gpu_buf = GPUMemoryBuffer()
        self.raw_bytes = UnsafePointer[UInt8, MutUntrackedOrigin](unsafe_from_address=1)
        self.raw_size = 0

    def __init__(
        out self,
        gpu_buf: GPUMemoryBuffer,
        raw_bytes: UnsafePointer[UInt8, MutUntrackedOrigin],
        raw_size: Int,
    ):
        self.gpu_buf = gpu_buf
        self.raw_bytes = raw_bytes
        self.raw_size = raw_size

    def __init__(out self, *, copy: Self):
        self.gpu_buf = copy.gpu_buf
        self.raw_bytes = copy.raw_bytes
        self.raw_size = copy.raw_size

    def __init__(out self, *, deinit take: Self):
        self.gpu_buf = take.gpu_buf^
        self.raw_bytes = take.raw_bytes
        self.raw_size = take.raw_size

    def get_kernel(self, kernel_name: String) raises -> HSACOKernel:
        """Dynamically resolves `<kernel_name>.kd` from the ELF64 symbol tables."""
        if self.raw_size < 64:
            raise Error("HSACOModule is not initialized or truncated")

        var target_sym = kernel_name + ".kd"
        var target_bytes = target_sym.as_bytes()
        var target_len = len(target_bytes)

        var base = self.raw_bytes.__int__()
        var e_shoff = Int(UnsafePointer[UInt64, MutUntrackedOrigin](unsafe_from_address=base + 40)[0])
        var e_shentsize = Int(UnsafePointer[UInt16, MutUntrackedOrigin](unsafe_from_address=base + 58)[0])
        var e_shnum = Int(UnsafePointer[UInt16, MutUntrackedOrigin](unsafe_from_address=base + 60)[0])

        if e_shoff <= 0 or e_shoff + e_shnum * e_shentsize > self.raw_size:
            raise Error("Invalid ELF64 section header table in .hsaco")

        # Scan SHT_SYMTAB (2) and SHT_DYNSYM (11)
        for s_idx in range(e_shnum):
            var sh_addr = base + e_shoff + s_idx * e_shentsize
            var sh_type = UnsafePointer[UInt32, MutUntrackedOrigin](unsafe_from_address=sh_addr + 4)[0]
            if sh_type != 2 and sh_type != 11:
                continue

            var sym_off = Int(UnsafePointer[UInt64, MutUntrackedOrigin](unsafe_from_address=sh_addr + 24)[0])
            var sym_size = Int(UnsafePointer[UInt64, MutUntrackedOrigin](unsafe_from_address=sh_addr + 32)[0])
            var str_sh_idx = Int(UnsafePointer[UInt32, MutUntrackedOrigin](unsafe_from_address=sh_addr + 40)[0])
            var sym_entsize = Int(UnsafePointer[UInt64, MutUntrackedOrigin](unsafe_from_address=sh_addr + 56)[0])
            if sym_entsize == 0:
                sym_entsize = 24

            if str_sh_idx < 0 or str_sh_idx >= e_shnum:
                continue
            var str_sh_addr = base + e_shoff + str_sh_idx * e_shentsize
            var str_off = Int(UnsafePointer[UInt64, MutUntrackedOrigin](unsafe_from_address=str_sh_addr + 24)[0])
            var str_size = Int(UnsafePointer[UInt64, MutUntrackedOrigin](unsafe_from_address=str_sh_addr + 32)[0])

            var num_syms = sym_size // sym_entsize
            for i in range(num_syms):
                var sym_addr = base + sym_off + i * sym_entsize
                var st_name = Int(UnsafePointer[UInt32, MutUntrackedOrigin](unsafe_from_address=sym_addr + 0)[0])
                var st_value = Int(UnsafePointer[UInt64, MutUntrackedOrigin](unsafe_from_address=sym_addr + 8)[0])

                if st_name <= 0 or st_name + target_len >= str_size:
                    continue

                var name_ptr = UnsafePointer[UInt8, MutUntrackedOrigin](unsafe_from_address=base + str_off + st_name)
                var matched = True
                for c_idx in range(target_len):
                    if name_ptr[c_idx] != target_bytes[c_idx]:
                        matched = False
                        break
                if matched and name_ptr[target_len] == 0:
                    if st_value < 0 or st_value + 64 > self.gpu_buf.size:
                        raise Error("Kernel descriptor offset out of bounds in .hsaco")
                    var kd_cpu_addr = self.gpu_buf.cpu_addr + st_value
                    var group_seg = UnsafePointer[UInt32, MutUntrackedOrigin](unsafe_from_address=kd_cpu_addr + 0)[0]
                    var priv_seg = UnsafePointer[UInt32, MutUntrackedOrigin](unsafe_from_address=kd_cpu_addr + 4)[0]
                    return HSACOKernel(
                        kd_va_addr=UInt64(self.gpu_buf.va_addr + st_value),
                        kd_offset=st_value,
                        group_segment_size=group_seg,
                        private_segment_size=priv_seg,
                    )

        raise Error("Kernel descriptor symbol '" + target_sym + "' not found in .hsaco")


@always_inline
def align_up_4096(size: Int) -> Int:
    return ((size + 4095) // 4096) * 4096


struct KFDContext(Movable):
    var has_queue: Bool
    var libc: OwnedDLHandle
    var cpu_fence_lib: OwnedDLHandle
    var kfd_fd: Int32
    var drm_fd: Int32
    var gpu_id: UInt32

    var c_open: def(Int, Int32) thin abi("C") -> Int32
    var c_close: def(Int32) thin abi("C") -> Int32
    var c_ioctl: def(Int32, Int, Int) thin abi("C") -> Int32
    var c_mmap: def(Int, Int, Int32, Int32, Int32, Int) thin abi("C") -> Int
    var c_munmap: def(Int, Int) thin abi("C") -> Int32
    var c_read: def(Int32, Int, Int) thin abi("C") -> Int
    var c_errno_loc: def() thin abi("C") -> Int
    var cpu_sfence: def(Int32) thin abi("C") -> Int32
    var cpu_clflush: def(Int) thin abi("C") -> Int32
    var cpu_clflush_range: def(Int, Int) thin abi("C") -> Int32
    var cpu_wait_completion: def(Int) thin abi("C") -> Int32
    var cpu_disable_glibc_mmap: def(Int32) thin abi("C") -> Int32
    var usleep_fn: def(UInt32) thin abi("C") -> Int32

    var ring_buf: GPUMemoryBuffer
    var gart_buf: GPUMemoryBuffer
    var cwsr_buf: GPUMemoryBuffer
    var eop_buf: GPUMemoryBuffer
    var kernarg_buf: GPUMemoryBuffer
    var completion_buf: GPUMemoryBuffer
    var modules: List[HSACOModule]
    var default_kernel: HSACOKernel
    var packet_idx: UInt64

    var queue_id: UInt32
    var doorbell_mmap_addr: Int
    var doorbell_reg_addr: Int

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

    comptime DEFAULT_GTT_FLAGS = UInt32(
        Self.KFD_IOC_ALLOC_MEM_FLAGS_WRITABLE
        | Self.KFD_IOC_ALLOC_MEM_FLAGS_EXECUTABLE
        | Self.KFD_IOC_ALLOC_MEM_FLAGS_NO_SUBSTITUTE
        | Self.KFD_IOC_ALLOC_MEM_FLAGS_GTT
        | Self.KFD_IOC_ALLOC_MEM_FLAGS_PUBLIC
    )

    comptime CONTROL_GTT_FLAGS = UInt32(
        Self.KFD_IOC_ALLOC_MEM_FLAGS_WRITABLE
        | Self.KFD_IOC_ALLOC_MEM_FLAGS_EXECUTABLE
        | Self.KFD_IOC_ALLOC_MEM_FLAGS_NO_SUBSTITUTE
        | Self.KFD_IOC_ALLOC_MEM_FLAGS_GTT
        | Self.KFD_IOC_ALLOC_MEM_FLAGS_COHERENT
        | Self.KFD_IOC_ALLOC_MEM_FLAGS_UNCACHED
        | Self.KFD_IOC_ALLOC_MEM_FLAGS_PUBLIC
    )

    comptime DEFAULT_VRAM_FLAGS = UInt32(
        Self.KFD_IOC_ALLOC_MEM_FLAGS_WRITABLE
        | Self.KFD_IOC_ALLOC_MEM_FLAGS_EXECUTABLE
        | Self.KFD_IOC_ALLOC_MEM_FLAGS_NO_SUBSTITUTE
        | Self.KFD_IOC_ALLOC_MEM_FLAGS_VRAM
    )

    # AQL Packet Header Constants:
    # Type = 2 (HSA_PACKET_TYPE_KERNEL_DISPATCH), Acquire = 2 (SYSTEM), Release = 2 (SYSTEM)
    # Barrier bit is bit 8 (1 << 8 = 256).
    # With barrier=1: 2 | (1 << 8) | (2 << 9) | (2 << 11) = 2 + 256 + 1024 + 4096 = 5378
    # With barrier=0: 2 | (0 << 8) | (2 << 9) | (2 << 11) = 2 + 1024 + 4096 = 5122
    comptime AQL_HEADER_BARRIER = UInt16(5378)
    comptime AQL_HEADER_NO_BARRIER = UInt16(5122)

    def __init__(
        out self,
        init_queue: Bool = True,
        default_hsaco_path: String = "",
        default_kernel_name: String = "",
    ) raises:
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
        self.cpu_clflush_range = self.cpu_fence_lib.get_function[def(Int, Int) thin abi("C") -> Int32]("cpu_clflush_range")
        self.cpu_wait_completion = self.cpu_fence_lib.get_function[def(Int) thin abi("C") -> Int32]("cpu_wait_completion")
        self.cpu_disable_glibc_mmap = self.cpu_fence_lib.get_function[def(Int32) thin abi("C") -> Int32]("cpu_disable_glibc_mmap")
        self.usleep_fn = self.libc.get_function[def(UInt32) thin abi("C") -> Int32]("usleep")

        # Prevent glibc free() from ever calling munmap() or shrinking the brk heap while KFD PASID is active.
        # On amdgpu/gfx90c, munmap() on the process mm_struct fires mmu_notifier_invalidate_range_start,
        # triggering CWSR wave preemption and UTCL2 invalidation on the shared PASID.
        _ = self.cpu_disable_glibc_mmap(0)

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
        self.kernarg_buf = GPUMemoryBuffer()
        self.completion_buf = GPUMemoryBuffer()
        self.modules = List[HSACOModule]()
        self.default_kernel = HSACOKernel()

        if not init_queue:
            return

        self.kfd_fd = self._open_cstr("/dev/kfd", 2)  # O_RDWR
        if self.kfd_fd < 0:
            raise Error("Failed to open /dev/kfd")

        self.drm_fd = self._open_cstr("/dev/dri/renderD128", 2)  # O_RDWR
        if self.drm_fd < 0:
            _ = self.c_close(self.kfd_fd)
            raise Error("Failed to open /dev/dri/renderD128")

        # Discover GPU ID from sysfs topology
        self.gpu_id = 0
        for node_idx in range(1, 16):
            var path = "/sys/class/kfd/kfd/topology/nodes/" + String(node_idx) + "/gpu_id"
            var fd = self._open_cstr(path, 0)
            if fd >= 0:
                var buf = alloc[UInt8](64)
                var bytes_read = self.c_read(fd, buf.__int__(), 63)
                _ = self.c_close(fd)
                if bytes_read > 0:
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
            self.gpu_id = 21148

        # 1. Acquire VM
        var acq_args_ptr = alloc[kfd_ioctl_acquire_vm_args](1)
        acq_args_ptr[0] = kfd_ioctl_acquire_vm_args(drm_fd=UInt32(self.drm_fd), gpu_id=self.gpu_id)
        var r = self.c_ioctl(self.kfd_fd, Self.AMDKFD_IOC_ACQUIRE_VM, acq_args_ptr.__int__())
        acq_args_ptr.free()
        if r < 0:
            _ = self.c_close(self.kfd_fd)
            _ = self.c_close(self.drm_fd)
            raise Error("GPU VM Acquisition failed!")

        # 2. Enable Runtime
        var runtime_enable_ptr = alloc[kfd_ioctl_runtime_enable_args](1)
        runtime_enable_ptr[0] = kfd_ioctl_runtime_enable_args()
        r = self.c_ioctl(self.kfd_fd, Self.AMDKFD_IOC_RUNTIME_ENABLE, runtime_enable_ptr.__int__())
        runtime_enable_ptr.free()
        if r < 0:
            _ = self.c_close(self.kfd_fd)
            _ = self.c_close(self.drm_fd)
            raise Error("GPU Runtime enable failed!")

        # 3. Allocate Queue Control & Ring Buffers
        var ring_buf = self.alloc_gpu_buffer(64 * 1024, Self.CONTROL_GTT_FLAGS, cpu_access=True)
        var gart_buf = self.alloc_gpu_buffer(4096, Self.CONTROL_GTT_FLAGS, cpu_access=True)
        var cwsr_buf = self.alloc_gpu_buffer(2801664, Self.DEFAULT_VRAM_FLAGS, cpu_access=False)
        var eop_buf = self.alloc_gpu_buffer(4096, Self.DEFAULT_VRAM_FLAGS, cpu_access=False)

        var ring_ptr = UnsafePointer[UInt32, MutUntrackedOrigin](unsafe_from_address=ring_buf.cpu_addr)
        for i in range((64 * 1024) // 4):
            ring_ptr[i] = 0

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
        UnsafePointer[UInt32, MutUntrackedOrigin](unsafe_from_address=q_base + 0)[0] = 2     # queue_type = AQL
        UnsafePointer[UInt32, MutUntrackedOrigin](unsafe_from_address=q_base + 4)[0] = 1     # features
        UnsafePointer[UInt64, MutUntrackedOrigin](unsafe_from_address=q_base + 8)[0] = UInt64(ring_buf.va_addr)
        UnsafePointer[UInt64, MutUntrackedOrigin](unsafe_from_address=q_base + 16)[0] = UInt64(sig_va_addr)
        UnsafePointer[UInt32, MutUntrackedOrigin](unsafe_from_address=q_base + 24)[0] = 1024  # size in packets

        UnsafePointer[UInt32, MutUntrackedOrigin](unsafe_from_address=q_base + 72)[0] = 7
        UnsafePointer[UInt32, MutUntrackedOrigin](unsafe_from_address=q_base + 76)[0] = 39
        UnsafePointer[UInt32, MutUntrackedOrigin](unsafe_from_address=q_base + 136)[0] = 128
        UnsafePointer[UInt32, MutUntrackedOrigin](unsafe_from_address=q_base + 180)[0] = 10

        _ = self.cpu_clflush(q_base)
        _ = self.cpu_clflush(q_base + 64)
        _ = self.cpu_clflush(q_base + 128)
        _ = self.cpu_clflush(q_base + 180)
        _ = self.cpu_sfence(0)

        # 4. Create AQL Queue in KFD
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
            sdma_engine_id=0,
        )
        r = self.c_ioctl(self.kfd_fd, Self.AMDKFD_IOC_CREATE_QUEUE, create_args_ptr.__int__())
        if r < 0:
            var err = UnsafePointer[Int32, MutUntrackedOrigin](unsafe_from_address=self.c_errno_loc())[0]
            self.free_gpu_buffer(eop_buf)
            self.free_gpu_buffer(cwsr_buf)
            self.free_gpu_buffer(gart_buf)
            self.free_gpu_buffer(ring_buf)
            _ = self.c_close(self.kfd_fd)
            _ = self.c_close(self.drm_fd)
            create_args_ptr.free()
            raise Error("KFD Queue creation failed with errno: " + String(err))

        self.queue_id = create_args_ptr[0].queue_id
        var doorbell_offset = create_args_ptr[0].doorbell_offset
        create_args_ptr.free()

        var doorbells_base = doorbell_offset & (~0x1FFF)
        var doorbell_page_size = 0x2000
        self.doorbell_mmap_addr = self.c_mmap(
            0,
            doorbell_page_size,
            Self.PROT_READ | Self.PROT_WRITE,
            Self.MAP_SHARED,
            self.kfd_fd,
            Int(doorbells_base),
        )
        if self.doorbell_mmap_addr == -1:
            raise Error("Doorbell page mmap failed")
        self.doorbell_reg_addr = self.doorbell_mmap_addr + (Int(doorbell_offset) - Int(doorbells_base))

        UnsafePointer[UInt64, MutUntrackedOrigin](unsafe_from_address=sig_cpu_addr + 8)[0] = UInt64(self.doorbell_reg_addr)
        _ = self.cpu_clflush(sig_cpu_addr)
        _ = self.cpu_sfence(0)

        self.ring_buf = ring_buf^
        self.gart_buf = gart_buf^
        self.cwsr_buf = cwsr_buf^
        self.eop_buf = eop_buf^
        self.completion_buf = self.alloc_gpu_buffer(1024 * 64, Self.CONTROL_GTT_FLAGS, cpu_access=True)
        self.kernarg_buf = self.alloc_gpu_buffer(1024 * 512, Self.CONTROL_GTT_FLAGS, cpu_access=True)
        var ka_u32 = UnsafePointer[UInt32, MutUntrackedOrigin](unsafe_from_address=self.kernarg_buf.cpu_addr)
        for i in range((1024 * 512) // 4):
            ka_u32[i] = 0
        self.packet_idx = 0

        if default_hsaco_path.byte_length() > 0:
            var mod_idx = self.load_hsaco(default_hsaco_path)
            if default_kernel_name.byte_length() > 0:
                self.default_kernel = self.modules[mod_idx].get_kernel(default_kernel_name)

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
        self.cpu_clflush_range = take.cpu_clflush_range
        self.cpu_wait_completion = take.cpu_wait_completion
        self.cpu_disable_glibc_mmap = take.cpu_disable_glibc_mmap
        self.usleep_fn = take.usleep_fn
        self.ring_buf = take.ring_buf^
        self.gart_buf = take.gart_buf^
        self.cwsr_buf = take.cwsr_buf^
        self.eop_buf = take.eop_buf^
        self.kernarg_buf = take.kernarg_buf^
        self.completion_buf = take.completion_buf^
        self.modules = take.modules^
        self.default_kernel = take.default_kernel
        self.packet_idx = take.packet_idx
        self.queue_id = take.queue_id
        self.doorbell_mmap_addr = take.doorbell_mmap_addr
        self.doorbell_reg_addr = take.doorbell_reg_addr

    def __del__(deinit self):
        if self.has_queue:
            var destroy_args_ptr = alloc[kfd_ioctl_destroy_queue_args](1)
            destroy_args_ptr[0] = kfd_ioctl_destroy_queue_args(queue_id=self.queue_id)
            _ = self.c_ioctl(self.kfd_fd, Self.AMDKFD_IOC_DESTROY_QUEUE, destroy_args_ptr.__int__())
            destroy_args_ptr.free()

            _ = self.c_munmap(self.doorbell_mmap_addr, 0x2000)

            try:
                for i in range(len(self.modules)):
                    if self.modules[i].raw_size > 0:
                        self.modules[i].raw_bytes.free()
                    if self.modules[i].gpu_buf.size > 0:
                        self.free_gpu_buffer(self.modules[i].gpu_buf)
                self.free_gpu_buffer(self.kernarg_buf)
                self.free_gpu_buffer(self.completion_buf)
                self.free_gpu_buffer(self.eop_buf)
                self.free_gpu_buffer(self.cwsr_buf)
                self.free_gpu_buffer(self.gart_buf)
                self.free_gpu_buffer(self.ring_buf)
            except:
                pass

            _ = self.c_close(self.kfd_fd)
            _ = self.c_close(self.drm_fd)
        else:
            for i in range(len(self.modules)):
                if self.modules[i].raw_size > 0:
                    self.modules[i].raw_bytes.free()
                if self.modules[i].gpu_buf.size > 0:
                    try:
                        self.free_gpu_buffer(self.modules[i].gpu_buf)
                    except:
                        pass

    def _open_cstr(self, path: String, flags: Int32) -> Int32:
        var path_bytes = path.as_bytes()
        var n = len(path_bytes)
        var ptr = alloc[UInt8](n + 1)
        for i in range(n):
            ptr[i] = path_bytes[i]
        ptr[n] = 0
        var fd = self.c_open(ptr.__int__(), flags)
        ptr.free()
        return fd

    @always_inline
    def flush_cache_range(self, addr: Int, size_bytes: Int):
        """Flushes/invalidates CPU cache lines covering [addr, addr + size_bytes) and issues sfence."""
        if size_bytes <= 0:
            return
        _ = self.cpu_clflush_range(addr, size_bytes)

    def alloc_gpu_buffer(
        self,
        size: Int,
        flags: UInt32 = Self.DEFAULT_GTT_FLAGS,
        cpu_access: Bool = True,
    ) raises -> GPUMemoryBuffer:
        var aligned_size = align_up_4096(size)
        if aligned_size <= 0:
            aligned_size = 4096

        if not self.has_queue:
            var cpu_addr = self.c_mmap(
                0,
                aligned_size,
                Self.PROT_READ | Self.PROT_WRITE,
                Self.MAP_PRIVATE | Self.MAP_ANONYMOUS,
                -1,
                0,
            )
            if cpu_addr == -1 or cpu_addr == 0:
                raise Error("CPU-only mmap failed")
            return GPUMemoryBuffer(cpu_addr, aligned_size, 0, cpu_addr)

        var va_addr = self.c_mmap(
            0,
            aligned_size,
            Self.PROT_NONE,
            Self.MAP_PRIVATE | Self.MAP_ANONYMOUS,
            -1,
            0,
        )
        if va_addr == -1 or va_addr == 0:
            raise Error("mmap VA reservation failed")

        var alloc_args_ptr = alloc[kfd_ioctl_alloc_memory_of_gpu_args](1)
        alloc_args_ptr[0] = kfd_ioctl_alloc_memory_of_gpu_args(
            va_addr=UInt64(va_addr),
            size=UInt64(aligned_size),
            gpu_id=self.gpu_id,
            flags=flags,
        )
        var io_res = self.c_ioctl(self.kfd_fd, Self.AMDKFD_IOC_ALLOC_MEMORY_OF_GPU, alloc_args_ptr.__int__())
        if io_res < 0:
            alloc_args_ptr.free()
            _ = self.c_munmap(va_addr, aligned_size)
            raise Error("AMDKFD_IOC_ALLOC_MEMORY_OF_GPU failed")
        var handle = alloc_args_ptr[0].handle
        var mmap_offset = alloc_args_ptr[0].mmap_offset
        alloc_args_ptr.free()

        var cpu_addr = 0
        if cpu_access:
            cpu_addr = self.c_mmap(
                va_addr,
                aligned_size,
                Self.PROT_READ | Self.PROT_WRITE,
                Self.MAP_SHARED | Self.MAP_FIXED,
                self.drm_fd,
                Int(mmap_offset),
            )
            if cpu_addr == -1 or cpu_addr != va_addr:
                var free_args_ptr = alloc[kfd_ioctl_free_memory_of_gpu_args](1)
                free_args_ptr[0] = kfd_ioctl_free_memory_of_gpu_args(handle=handle)
                _ = self.c_ioctl(self.kfd_fd, Self.AMDKFD_IOC_FREE_MEMORY_OF_GPU, free_args_ptr.__int__())
                free_args_ptr.free()
                _ = self.c_munmap(va_addr, aligned_size)
                raise Error("CPU DRM mmap failed")

        var device_ids_ptr = alloc[UInt32](1)
        device_ids_ptr[0] = self.gpu_id
        var map_args_ptr = alloc[kfd_ioctl_map_memory_to_gpu_args](1)
        map_args_ptr[0] = kfd_ioctl_map_memory_to_gpu_args(
            handle=handle,
            device_ids_array_ptr=UInt64(device_ids_ptr.__int__()),
            n_devices=1,
        )
        io_res = self.c_ioctl(self.kfd_fd, Self.AMDKFD_IOC_MAP_MEMORY_TO_GPU, map_args_ptr.__int__())
        device_ids_ptr.free()
        map_args_ptr.free()
        if io_res < 0:
            _ = self.c_munmap(va_addr, aligned_size)
            var free_args_ptr = alloc[kfd_ioctl_free_memory_of_gpu_args](1)
            free_args_ptr[0] = kfd_ioctl_free_memory_of_gpu_args(handle=handle)
            _ = self.c_ioctl(self.kfd_fd, Self.AMDKFD_IOC_FREE_MEMORY_OF_GPU, free_args_ptr.__int__())
            free_args_ptr.free()
            raise Error("AMDKFD_IOC_MAP_MEMORY_TO_GPU failed")

        return GPUMemoryBuffer(va_addr, aligned_size, handle, cpu_addr)

    def free_gpu_buffer(self, buf: GPUMemoryBuffer) raises:
        if buf.size <= 0:
            return
        if not self.has_queue:
            _ = self.c_munmap(buf.cpu_addr, buf.size)
            return

        var unmap_device_ids_ptr = alloc[UInt32](1)
        unmap_device_ids_ptr[0] = self.gpu_id
        var unmap_args_ptr = alloc[kfd_ioctl_unmap_memory_from_gpu_args](1)
        unmap_args_ptr[0] = kfd_ioctl_unmap_memory_from_gpu_args(
            handle=buf.handle,
            device_ids_array_ptr=UInt64(unmap_device_ids_ptr.__int__()),
            n_devices=1,
        )
        _ = self.c_ioctl(self.kfd_fd, Self.AMDKFD_IOC_UNMAP_MEMORY_FROM_GPU, unmap_args_ptr.__int__())
        unmap_device_ids_ptr.free()
        unmap_args_ptr.free()

        if buf.cpu_addr != 0:
            _ = self.c_munmap(buf.cpu_addr, buf.size)
        else:
            _ = self.c_munmap(buf.va_addr, buf.size)

        var free_args_ptr = alloc[kfd_ioctl_free_memory_of_gpu_args](1)
        free_args_ptr[0] = kfd_ioctl_free_memory_of_gpu_args(handle=buf.handle)
        _ = self.c_ioctl(self.kfd_fd, Self.AMDKFD_IOC_FREE_MEMORY_OF_GPU, free_args_ptr.__int__())
        free_args_ptr.free()

    def load_hsaco(mut self, hsaco_path: String) raises -> Int:
        """Dynamically parses an ELF64 .hsaco file and maps all PT_LOAD segments into GTT memory."""
        var fd = self._open_cstr(hsaco_path, 0)
        if fd < 0:
            raise Error("Failed to open .hsaco file: " + hsaco_path)

        var max_file_bytes = 4 * 1024 * 1024
        var raw_buf = alloc[UInt8](max_file_bytes)
        var total_read = 0
        while total_read < max_file_bytes:
            var r = self.c_read(fd, raw_buf.__int__() + total_read, max_file_bytes - total_read)
            if r <= 0:
                break
            total_read += r
        _ = self.c_close(fd)

        if total_read >= max_file_bytes:
            raw_buf.free()
            raise Error(".hsaco file exceeds 4 MB buffer limit: " + hsaco_path)

        if total_read < 64:
            raw_buf.free()
            raise Error("Failed to read valid ELF64 header from: " + hsaco_path)

        # Verify ELF64 magic: 0x7f 'E' 'L' 'F', class=2 (64-bit), data=1 (LSB)
        if (
            raw_buf[0] != 0x7F
            or raw_buf[1] != 0x45
            or raw_buf[2] != 0x4C
            or raw_buf[3] != 0x46
            or raw_buf[4] != 2
            or raw_buf[5] != 1
        ):
            raw_buf.free()
            raise Error("Invalid ELF64 header magic in: " + hsaco_path)

        var base = raw_buf.__int__()
        var e_phoff = Int(UnsafePointer[UInt64, MutUntrackedOrigin](unsafe_from_address=base + 32)[0])
        var e_phentsize = Int(UnsafePointer[UInt16, MutUntrackedOrigin](unsafe_from_address=base + 54)[0])
        var e_phnum = Int(UnsafePointer[UInt16, MutUntrackedOrigin](unsafe_from_address=base + 56)[0])

        if e_phoff <= 0 or e_phoff + e_phnum * e_phentsize > total_read:
            raw_buf.free()
            raise Error("Invalid ELF64 program header table in: " + hsaco_path)

        # 1. Compute total virtual memory span across all PT_LOAD (p_type == 1) segments
        var max_vaddr_end = 0
        for p_idx in range(e_phnum):
            var ph_addr = base + e_phoff + p_idx * e_phentsize
            var p_type = UnsafePointer[UInt32, MutUntrackedOrigin](unsafe_from_address=ph_addr + 0)[0]
            if p_type != 1:  # PT_LOAD == 1
                continue
            var p_vaddr = Int(UnsafePointer[UInt64, MutUntrackedOrigin](unsafe_from_address=ph_addr + 16)[0])
            var p_memsz = Int(UnsafePointer[UInt64, MutUntrackedOrigin](unsafe_from_address=ph_addr + 40)[0])
            var seg_end = p_vaddr + p_memsz
            if seg_end > max_vaddr_end:
                max_vaddr_end = seg_end

        var map_size = align_up_4096(max_vaddr_end)
        if map_size < 16384:
            map_size = 16384

        var hsaco_gpu_buf = self.alloc_gpu_buffer(map_size, Self.DEFAULT_GTT_FLAGS, cpu_access=True)
        var dst_u32 = UnsafePointer[UInt32, MutUntrackedOrigin](unsafe_from_address=hsaco_gpu_buf.cpu_addr)
        for i in range(map_size // 4):
            dst_u32[i] = 0

        # 2. Map every PT_LOAD segment from (p_offset, p_filesz) -> (p_vaddr)
        for p_idx in range(e_phnum):
            var ph_addr = base + e_phoff + p_idx * e_phentsize
            var p_type = UnsafePointer[UInt32, MutUntrackedOrigin](unsafe_from_address=ph_addr + 0)[0]
            if p_type != 1:
                continue
            var p_offset = Int(UnsafePointer[UInt64, MutUntrackedOrigin](unsafe_from_address=ph_addr + 8)[0])
            var p_vaddr = Int(UnsafePointer[UInt64, MutUntrackedOrigin](unsafe_from_address=ph_addr + 16)[0])
            var p_filesz = Int(UnsafePointer[UInt64, MutUntrackedOrigin](unsafe_from_address=ph_addr + 32)[0])

            if p_offset + p_filesz > total_read or p_vaddr + p_filesz > map_size:
                self.free_gpu_buffer(hsaco_gpu_buf)
                raw_buf.free()
                raise Error("PT_LOAD segment out of bounds in: " + hsaco_path)

            var src_ptr = raw_buf + p_offset
            var dst_ptr = UnsafePointer[UInt8, MutUntrackedOrigin](unsafe_from_address=hsaco_gpu_buf.cpu_addr + p_vaddr)
            for b in range(p_filesz):
                dst_ptr[b] = src_ptr[b]

        if self.has_queue:
            self.flush_cache_range(hsaco_gpu_buf.cpu_addr, map_size)

        var mod_idx = len(self.modules)
        self.modules.append(HSACOModule(hsaco_gpu_buf, raw_buf, total_read))
        return mod_idx

    def get_kernel(self, kernel_name: String, module_idx: Int = 0) raises -> HSACOKernel:
        return self.modules[module_idx].get_kernel(kernel_name)

    @always_inline
    def ring_doorbell(mut self):
        """Flushes GART write pointer and rings the mapped hardware doorbell once."""
        var cpu_sfence = self.cpu_sfence
        var cpu_clflush = self.cpu_clflush
        var wptr_ptr = UnsafePointer[UInt64, MutUntrackedOrigin](unsafe_from_address=self.gart_buf.cpu_addr + 56)
        wptr_ptr[0] = self.packet_idx

        _ = cpu_sfence(0)
        _ = cpu_clflush(self.gart_buf.cpu_addr + 56)
        _ = cpu_sfence(0)

        var doorbell_ptr = UnsafePointer[UInt64, MutUntrackedOrigin](unsafe_from_address=self.doorbell_reg_addr)
        doorbell_ptr[0] = self.packet_idx - 1
        _ = cpu_sfence(0)

    def enqueue_kernel_matmul(
        mut self,
        kernel: HSACOKernel,
        W_packed_addr: Int,
        scales_addr: Int,
        x_addr: Int,
        out_ptr_addr: Int,
        c_in: Int,
        c_out: Int,
        batch_size: Int = 1,
        out_stride: Int = 0,
        flush_input: Bool = True,
        barrier: Bool = False,
        ring_doorbell_now: Bool = True,
        flush_output_before: Bool = True,
    ) raises -> Int:
        if c_in <= 0 or (c_in & 31) != 0:
            raise Error("enqueue_kernel_matmul requires c_in > 0 and divisible by 32")
        if c_out <= 0 or batch_size <= 0:
            raise Error("enqueue_kernel_matmul requires c_out > 0 and batch_size > 0")

        var cpu_sfence = self.cpu_sfence
        var cpu_clflush = self.cpu_clflush

        if flush_input:
            self.flush_cache_range(x_addr, c_in * batch_size * 4)

        var effective_out_stride = out_stride if out_stride > 0 else c_out
        if flush_output_before and out_ptr_addr != 0 and effective_out_stride == c_out:
            self.flush_cache_range(out_ptr_addr, c_out * batch_size * 4)

        var slot = Int(self.packet_idx % 1024)

        # 1. Reset completion flag & atomic workgroup counter for this slot
        var completion_slot_cpu_addr = self.completion_buf.cpu_addr + slot * 64
        var completion_slot_va_addr = self.completion_buf.va_addr + slot * 64
        var completion_ptr = UnsafePointer[Int32, MutUntrackedOrigin](unsafe_from_address=completion_slot_cpu_addr)
        completion_ptr[0] = 0
        completion_ptr[1] = 0
        _ = cpu_sfence(0)
        _ = cpu_clflush(completion_slot_cpu_addr)

        # 2. Pack kernel arguments into slot's kernarg memory
        var kernarg_slot_cpu_addr = self.kernarg_buf.cpu_addr + slot * 512
        var kernarg_slot_va_addr = self.kernarg_buf.va_addr + slot * 512

        var kernarg_p = UnsafePointer[UInt64, MutUntrackedOrigin](unsafe_from_address=kernarg_slot_cpu_addr)
        kernarg_p[0] = UInt64(W_packed_addr)
        kernarg_p[1] = UInt64(scales_addr)
        kernarg_p[2] = UInt64(x_addr)
        kernarg_p[3] = UInt64(out_ptr_addr)
        kernarg_p[4] = UInt64(completion_slot_va_addr)

        var num_row_tiles = (c_out + 15) // 16
        var num_token_tiles = (batch_size + 15) // 16
        var num_workgroups = num_row_tiles * num_token_tiles

        var kernarg_u32_p = UnsafePointer[UInt32, MutUntrackedOrigin](unsafe_from_address=kernarg_slot_cpu_addr + 40)
        kernarg_u32_p[0] = UInt32(c_in)
        kernarg_u32_p[1] = UInt32(c_out)
        kernarg_u32_p[2] = UInt32(num_workgroups)       # grid_stride / out_stride compat
        kernarg_u32_p[3] = UInt32(num_workgroups)       # num_workgroups
        kernarg_u32_p[4] = UInt32(batch_size)           # batch_size
        kernarg_u32_p[5] = UInt32(effective_out_stride) # out_stride (elements per token row)

        _ = cpu_sfence(0)
        _ = cpu_clflush(kernarg_slot_cpu_addr)

        # 3. Format HSA AQL Dispatch Packet (64 bytes), writing header=1 (INVALID) first, then header last
        var pkt_header = Self.AQL_HEADER_BARRIER if barrier else Self.AQL_HEADER_NO_BARRIER
        var packet = hsa_kernel_dispatch_packet_t(
            header=1,
            setup=1,
            workgroup_size_x=256,
            workgroup_size_y=1,
            workgroup_size_z=1,
            grid_size_x=UInt32(num_workgroups * 256),
            grid_size_y=1,
            grid_size_z=1,
            private_segment_size=kernel.private_segment_size,
            group_segment_size=kernel.group_segment_size,
            kernel_object=kernel.kd_va_addr,
            kernarg_address=UInt64(kernarg_slot_va_addr),
            completion_signal=0,
        )

        # 4. Write dispatch packet body to circular ring buffer, fence, then publish header last
        var ring_slot_addr = self.ring_buf.cpu_addr + slot * 64
        var ring_packet_ptr = UnsafePointer[hsa_kernel_dispatch_packet_t, MutUntrackedOrigin](
            unsafe_from_address=ring_slot_addr
        )
        ring_packet_ptr[0] = packet^
        _ = cpu_sfence(0)
        UnsafePointer[UInt16, MutUntrackedOrigin](unsafe_from_address=ring_slot_addr)[0] = pkt_header
        self.packet_idx += 1

        _ = cpu_sfence(0)
        _ = cpu_clflush(ring_slot_addr)

        if ring_doorbell_now:
            self.ring_doorbell()

        return slot

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
        barrier: Bool = False,
        ring_doorbell_now: Bool = True,
    ) raises -> Int:
        var k = self.default_kernel
        return self.enqueue_kernel_matmul(
            k,
            W_packed_addr,
            scales_addr,
            x_addr,
            out_ptr_addr,
            c_in,
            c_out,
            batch_size=batch_size,
            out_stride=c_out,
            flush_input=flush_cache,
            barrier=barrier,
            ring_doorbell_now=ring_doorbell_now,
        )

    def wait_matmul(
        self,
        slot: Int,
        out_ptr_addr: Int,
        c_out: Int,
        batch_size: Int = 1,
        flush_cache: Bool = True,
        out_elem_bytes: Int = 4,
    ) raises:
        var completion_slot_cpu_addr = self.completion_buf.cpu_addr + slot * 64
        var wait_rc = self.cpu_wait_completion(completion_slot_cpu_addr)
        if wait_rc != 0:
            var completion_ptr = UnsafePointer[Int32, MutUntrackedOrigin](unsafe_from_address=completion_slot_cpu_addr)
            var current_wptr = UnsafePointer[UInt64, MutUntrackedOrigin](
                unsafe_from_address=self.gart_buf.cpu_addr + 56
            )[0]
            var current_rptr = UnsafePointer[UInt64, MutUntrackedOrigin](
                unsafe_from_address=self.gart_buf.cpu_addr + 128
            )[0]
            print("GPU Execution Timeout details:")
            print("  GART wptr:", current_wptr, "| GART rptr:", current_rptr)
            print("  Completion flag in memory:", completion_ptr[0])
            print("  Atomic counter in memory:", completion_ptr[1])
            raise Error("GPU execution timed out in wait_matmul!")

        if flush_cache and out_ptr_addr != 0:
            self.flush_cache_range(out_ptr_addr, c_out * batch_size * out_elem_bytes)

    def launch_matmul(
        mut self,
        W_packed_addr: Int,
        scales_addr: Int,
        x_addr: Int,
        out_ptr_addr: Int,
        c_in: Int,
        c_out: Int,
        batch_size: Int = 1,
    ) raises:
        var slot = self.enqueue_matmul(
            W_packed_addr,
            scales_addr,
            x_addr,
            out_ptr_addr,
            c_in,
            c_out,
            batch_size=batch_size,
        )
        self.wait_matmul(slot, out_ptr_addr, c_out, batch_size=batch_size)
