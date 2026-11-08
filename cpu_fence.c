#include <stdint.h>
#include <unistd.h>
#include <malloc.h>

int32_t add_one(int32_t x) {
    return x + 1;
}

int32_t cpu_disable_glibc_mmap(int32_t dummy) {
    mallopt(M_MMAP_MAX, 0);
    mallopt(M_TRIM_THRESHOLD, -1);
    return dummy;
}

int32_t cpu_sfence(int32_t dummy) {
    __builtin_ia32_sfence();
    return dummy;
}

int32_t cpu_mfence(int32_t dummy) {
    __builtin_ia32_mfence();
    return dummy;
}

int32_t cpu_clflush(const void *ptr) {
    __builtin_ia32_clflush(ptr);
    return 0;
}

int32_t cpu_clflush_range(uintptr_t addr, int64_t size_bytes) {
    if (size_bytes <= 0) return 0;
    uintptr_t base = addr & ~(uintptr_t)63;
    uintptr_t end = addr + (uintptr_t)size_bytes;
    for (; base < end; base += 64) {
        __builtin_ia32_clflush((const void *)base);
    }
    __builtin_ia32_sfence();
    return 0;
}

int32_t cpu_copy_u8_to_f32_and_flush(uintptr_t src_u8_addr, uintptr_t dst_f32_addr, int64_t n_elems, float scale) {
    if (n_elems <= 0) return 0;
    const uint8_t *src = (const uint8_t *)src_u8_addr;
    float *dst = (float *)dst_f32_addr;
    for (int64_t i = 0; i < n_elems; i++) {
        dst[i] = (float)src[i] * scale;
    }
    return cpu_clflush_range(dst_f32_addr, n_elems * 4);
}

int32_t cpu_wait_completion(uintptr_t slot_addr) {
    volatile int32_t *flag_ptr = (volatile int32_t *)slot_addr;
    int32_t attempts = 0;
    while (*flag_ptr != 1) {
        __builtin_ia32_pause();
        attempts++;
        if ((attempts & 255) == 0) {
            __builtin_ia32_clflush((const void *)slot_addr);
            __builtin_ia32_sfence();
        }
        if (attempts > 500000) {
            usleep(5);
            if (attempts > 10000000) {
                return -1;
            }
        }
    }
    return 0;
}
