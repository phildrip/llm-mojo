#include "liburing/src/include/liburing.h"
#include <fcntl.h>
#include <unistd.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include <errno.h>

// Initialize the io_uring instance and return an opaque pointer to the ring
void* init_uring(int entries) {
    struct io_uring* ring = malloc(sizeof(struct io_uring));
    if (!ring) return NULL;
    
    // Initialize the queue with no special flags (blocking/non-blocking are handled at syscall level)
    int ret = io_uring_queue_init(entries, ring, 0);
    if (ret < 0) {
        free(ring);
        return NULL;
    }
    return (void*)ring;
}

// Submit a read or write operation asynchronously
int submit_io(void* ring_ptr, int file_fd, void* buf, size_t len, off_t offset, uint64_t user_data, int is_write) {
    struct io_uring* ring = (struct io_uring*)ring_ptr;
    if (!ring) return -1;

    struct io_uring_sqe* sqe = io_uring_get_sqe(ring);
    if (!sqe) {
        return -2; // Submission Queue is full
    }

    if (is_write) {
        io_uring_prep_write(sqe, file_fd, buf, len, offset);
    } else {
        io_uring_prep_read(sqe, file_fd, buf, len, offset);
    }

    // Set user_data so Mojo can identify this completion
    io_uring_sqe_set_data64(sqe, user_data);

    // Submit SQE to the kernel
    int ret = io_uring_submit(ring);
    if (ret < 0) {
        return ret;
    }
    return 0; // Success
}

// Non-blocking check for completed I/O operations
// Returns:
//   >= 0 : number of bytes transferred (success)
//   < 0  : error code (e.g., -EAGAIN if no completions, or typical I/O error)
int poll_completion(void* ring_ptr, uint64_t* out_user_data) {
    struct io_uring* ring = (struct io_uring*)ring_ptr;
    if (!ring) return -1;

    struct io_uring_cqe* cqe = NULL;
    
    // Peek at the completion queue (non-blocking)
    int ret = io_uring_peek_cqe(ring, &cqe);
    if (ret < 0) {
        return -EAGAIN; // No completion available
    }

    if (cqe) {
        *out_user_data = io_uring_cqe_get_data64(cqe);
        int res = cqe->res; // Result of the read/write (bytes transferred or negative error)
        io_uring_cqe_seen(ring, cqe); // Mark CQE as consumed
        return res;
    }

    return -EAGAIN;
}

// Clean up and free the io_uring queue
void close_uring(void* ring_ptr) {
    struct io_uring* ring = (struct io_uring*)ring_ptr;
    if (ring) {
        io_uring_queue_exit(ring);
        free(ring);
    }
}
