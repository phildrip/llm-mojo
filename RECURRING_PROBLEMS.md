# LLM-Mojo: Low-Level GPU Debugging and Recurring Problems

This document serves as a persistent, up-to-date reference for recurring issues, layouts, and solutions encountered while building direct userspace GPU compute queues on AMD Vega (GFX9) hardware using Mojo.

---

## 1. The GFX9 Doorbell & MMIO Mechanism

### Hardware Mechanics
The GPU Command Processor (CP) or Microcode Scheduler (MEC) monitors a physical MMIO memory-mapped area called **doorbells**.
- For GFX9 (Vega), the Fusion Driver (KFD) returns a physical `doorbell_offset` (e.g. `0xd498c00000000000`).
- Doorbells must be page-aligned (typically `0x2000` or `0x1000` bytes) before mapping via `/dev/kfd` and standard `mmap`.
- Writing the new Write Pointer (`wptr` / `write_dispatch_id`) to the MMIO register wakes up the GPU.

### Memory & Cache Coherency (CRITICAL)
- **Write-Combining (WC) Buffers:** GART memory used for queue descriptors is mapped on the host with Write-Combining policies. CPU writes to WC memory are stored in internal write-fill buffers and **are not immediately visible to the GPU**.
- **Flushing CPU Cache Lines:** Before ringing the doorbell, you **must** flush the write pointer and ring buffer cache lines:
  1. Store Fence (`sfence`) to serialize stores.
  2. Cache line flush (`clflush`) on the `write_pointer_address` (`base + 56`) and the `ring_buffer` packets.
  3. Another Store Fence (`sfence`).
- **MMIO Doorbell Write Size:** GFX9 doorbells typically expect a **64-bit MMIO write** of the new write pointer value (`wptr`) to the doorbell register, though some kernels accept a 32-bit write.

---

## 2. Queue Struct Geometry: `hsa_queue_t` and `amd_queue_t`

AMD's Microcode Engine Scheduler (MEC) reads queue descriptor structs aligned to **64-byte** cache lines.

### `hsa_queue_t` Layout (HSA Standard)
| Offset (Bytes) | Size (Bytes) | Field Name | Description |
| :--- | :--- | :--- | :--- |
| **0** | 4 | `queue_type` | Queue protocol (e.g., `2` for AQL queue) |
| **4** | 4 | `features` | Features (e.g. `HSA_QUEUE_FEATURE_KERNEL_DISPATCH` = `2`) |
| **8** | 8 | `base_address` | Pointer to the ring buffer holding AQL packets |
| **16** | 8 | `doorbell_signal` | Doorbell offset resolved during queue creation |
| **24** | 4 | `size` | Maximum number of packets (power of 2) |

### `amd_queue_t` Extension (AMD Specific)
| Offset (Bytes) | Size (Bytes) | Field Name | Description |
| :--- | :--- | :--- | :--- |
| **0** | 40 | `hsa_queue` | Nested standard `hsa_queue_t` |
| **56** | 8 | `write_dispatch_id` | **64-bit Write Pointer (`wptr`)** tracked by userspace |
| **128** | 8 | `read_dispatch_id` | **64-bit Read Pointer (`rptr`)** updated by GPU CP |
| **136** | 4 | `read_dispatch_id_field_base_byte_offset` | Offset of `read_dispatch_id` relative to base (initialized to `128`) |
| **180** | 4 | `queue_properties` | Queue priority/features (typically set to `10`) |

### Struct Offset Discrepancies
- In some documentation, `read_dispatch_id` is documented at offset **1024**.
- On GFX9, the actual hardware structure (`amd_queue_t`) places `read_dispatch_id` at offset **128** (with `read_dispatch_id_field_base_byte_offset` = `128`).
- For KFD queue creation, the ioctl expects pointers to where these dispatch IDs reside in GART:
  - `write_pointer_address = gart_buf.va_addr + 56`
  - `read_pointer_address = gart_buf.va_addr + 128`

---

## 3. Mojo Eager Destruction & FFI Lifetimes

### The Problem
Mojo uses **eager variable destruction**—variables are deallocated as soon as they are no longer referenced in the code block.
- If you load a dynamic helper library (e.g., `libcpu_fence.so`) via `OwnedDLHandle`, and use it to get FFI functions:
  ```mojo
  var cpu_fence_lib = OwnedDLHandle("libcpu_fence.so")
  var cpu_sfence = cpu_fence_lib.get_function[...]("cpu_sfence")
  ```
- The compiler detects that `cpu_fence_lib` itself is never used again after `get_function` and calls `__del__(cpu_fence_lib)`, which calls `dlclose`.
- When `cpu_sfence()` is later executed, the library is unloaded, resulting in a **Segmentation Fault (Exit Code 139)**.

### The Solution
Always anchor FFI handle lifetimes explicitly at the end of the enclosing block or function using a dummy reference:
```mojo
# ... execute code calling FFI functions ...

_ = cpu_fence_lib # Keep library handle loaded until this point!
```

---

## 4. Querying Type Sizes (`sizeof` workaround)

Mojo does not have a built-in `sizeof` or `size_of` keyword. Standard memory allocations with `UnsafePointer` or `alloc[T]` manage sizes automatically under the hood, but raw size checks in bytes are often required for debugging.

### Safe Systems-level Workaround
We can query structural size in bytes at runtime/compile-time using `alloc[T]` pointer arithmetic:
```mojo
from std.memory import alloc

@always_inline
def sizeof[T: AnyType]() -> Int:
    var p = alloc[T](2)
    var p0 = Int(p)
    var p1 = Int(p + 1)
    p.free()
    return p1 - p0
```
This avoids MLIR/LLVM layout macros and is fully compatible with Mojo 2026 syntax.

---

## 5. Diagnostic and Log Inspection

- **Do NOT use `dmesg` commands** on this workspace.
- **Use `/var/log/kern.log`:** All kernel driver messages, GPU page faults, VM faults, and queue evictions are logged to `/var/log/kern.log`.
- To inspect driver events, run:
  ```bash
  grep -a "amdgpu" /var/log/kern.log | tail -n 50
  grep -a "kfd" /var/log/kern.log | tail -n 50
  ```

---

## 6. Hybrid CPU+iGPU Shared PASID & `munmap()` MMU-Notifier Trap (`gfxhub0` UTCL2 Fault)

### The Problem
When CPU worker threads run in the same process (`mm_struct` / `PASID`) as a bare-metal KFD AQL queue (`AMDKFD_IOC_ACQUIRE_VM`), any large temporary heap allocation that exceeds glibc's `M_MMAP_THRESHOLD` (`128 KB`) causes `free()` to invoke `munmap()` while GPU waves may be in flight.
- On Linux `amdgpu` (`gfx90c`), `munmap()` on the process's `mm_struct` fires `mmu_notifier_invalidate_range_start`, which triggers **CWSR wave preemption and UTCL2 TLB invalidation on the shared PASID** (`[gfxhub0] retry page fault ... from IH client 0x1b (UTCL2)`).
- Because a bare-metal user-mode AQL queue has no CWSR trap-handler shader loaded, CWSR preemption faults and hard-locks the iGPU.

### The Solution
1. **Disable glibc `mmap()` and heap trimming at KFD startup (`scratch/cpu_fence.c` $\to$ `cpu_disable_glibc_mmap`):**
   ```c
   mallopt(M_MMAP_MAX, 0);
   mallopt(M_TRIM_THRESHOLD, -1);
   ```
   `KFDContext.__init__` calls `cpu_disable_glibc_mmap(0)` immediately so `free()` never calls `munmap()` or shrinks the `brk` heap while KFD is active.
2. **Zero heap allocations (`alloc`/`free`) and `gc.disable()` while GPU waves are in flight:**
   Use pre-allocated scratch buffers or stack `InlineArray` buffers inside hybrid CPU+iGPU regions, and disable Python's cyclic GC (`gc.disable()`) after loading `AutoTokenizer`.
3. **Keep VRAM carveout allocations minimal (`DEFAULT_GTT_FLAGS` for data buffers):**
   Only `cwsr_buf` (`2.8 MB`) and `eop_buf` (`4 KB`) use `DEFAULT_VRAM_FLAGS`; all weight and activation buffers use `DEFAULT_GTT_FLAGS` so the desktop compositor's VRAM carveout is never starved.
