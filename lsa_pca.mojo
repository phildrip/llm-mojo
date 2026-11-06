from std.math import sqrt
from std.memory import alloc
from std.ffi import OwnedDLHandle
from qk_dump import QKDumpReader, Q_FLOATS, K_FLOATS, V_FLOATS, NUM_GLOBAL_LAYERS
from lsa_reference import (
    extract_position_free_and_regroup,
    GEMMA4_NUM_Q_HEADS,
    GEMMA4_NUM_KV_HEADS,
    GEMMA4_HEAD_DIM,
    GEMMA4_INDEX_DIM,
)

comptime LSA_PROJ_MAGIC: UInt32 = 0x3141534C  # "LSA1"
comptime DEFAULT_PCA_DIM: Int = 64

# -----------------------------------------------------------------------------
# 1. Covariance Matrix Computation
# -----------------------------------------------------------------------------

def compute_covariance(
    keys: UnsafePointer[Float32, _],
    num_keys: Int,
    dim: Int,
    cov_out: UnsafePointer[Float32, MutUntrackedOrigin],
):
    """
    Computes the sample covariance matrix C = (1/N) * sum_t (k_t - mu)(k_t - mu)^T.
    cov_out has size dim x dim.
    """
    # 1. Compute mean
    var mean = alloc[Float32](dim)
    for d in range(dim):
        mean[d] = 0.0

    for t in range(num_keys):
        var k_ptr = keys + t * dim
        for d in range(dim):
            mean[d] += k_ptr[d]

    var inv_n = 1.0 / Float32(num_keys)
    for d in range(dim):
        mean[d] *= inv_n

    # 2. Accumulate upper triangle of centered outer products
    for i in range(dim * dim):
        cov_out[i] = 0.0

    for t in range(num_keys):
        var k_ptr = keys + t * dim
        for i in range(dim):
            var centered_i = k_ptr[i] - mean[i]
            var row_off = i * dim
            for j in range(i, dim):
                var centered_j = k_ptr[j] - mean[j]
                cov_out[row_off + j] += centered_i * centered_j

    # 3. Scale by 1/N and mirror to lower triangle
    for i in range(dim):
        var row_off = i * dim
        for j in range(i, dim):
            var val = cov_out[row_off + j] * inv_n
            cov_out[row_off + j] = val
            cov_out[j * dim + i] = val

    mean.free()


# -----------------------------------------------------------------------------
# 2. Classical Jacobi Algorithm for Small Symmetric Matrices (e.g. 64x64)
# -----------------------------------------------------------------------------

def jacobi_diagonalize(
    matrix: UnsafePointer[Float32, _],
    dim: Int,
    eigenvectors_out: UnsafePointer[Float32, MutUntrackedOrigin],
    eigenvalues_out: UnsafePointer[Float32, MutUntrackedOrigin],
    max_sweeps: Int = 30,
    tol: Float32 = 1e-7,
):
    """
    Computes exact eigenvalues and eigenvectors of a small symmetric matrix (e.g. 64x64)
    using classical cyclic Jacobi rotations.
    """
    var a = alloc[Float32](dim * dim)
    for i in range(dim * dim):
        a[i] = matrix[i]

    # Initialize eigenvectors to identity
    for i in range(dim * dim):
        eigenvectors_out[i] = 0.0
    for i in range(dim):
        eigenvectors_out[i * dim + i] = 1.0

    for _ in range(max_sweeps):
        var max_offdiag: Float32 = 0.0
        for p in range(dim):
            var p_off = p * dim
            for q in range(p + 1, dim):
                var val = abs(a[p_off + q])
                if val > max_offdiag:
                    max_offdiag = val

        if max_offdiag < tol:
            break

        for p in range(dim):
            var p_off = p * dim
            for q in range(p + 1, dim):
                var q_off = q * dim
                var apq = a[p_off + q]
                if abs(apq) < 1e-12:
                    continue

                var app = a[p_off + p]
                var aqq = a[q_off + q]
                var tau = (aqq - app) / (2.0 * apq)
                var t: Float32
                if tau >= 0.0:
                    t = 1.0 / (tau + sqrt(1.0 + tau * tau))
                else:
                    t = -1.0 / (-tau + sqrt(1.0 + tau * tau))

                var c = 1.0 / sqrt(1.0 + t * t)
                var s = t * c

                # Update matrix A
                a[p_off + p] = app - t * apq
                a[q_off + q] = aqq + t * apq
                a[p_off + q] = 0.0
                a[q_off + p] = 0.0

                for r in range(dim):
                    if r != p and r != q:
                        var arp = a[r * dim + p]
                        var arq = a[r * dim + q]
                        a[r * dim + p] = c * arp - s * arq
                        a[p * dim + r] = a[r * dim + p]
                        a[r * dim + q] = s * arp + c * arq
                        a[q * dim + r] = a[r * dim + q]

                # Update eigenvectors matrix V
                for r in range(dim):
                    var vrp = eigenvectors_out[r * dim + p]
                    var vrq = eigenvectors_out[r * dim + q]
                    eigenvectors_out[r * dim + p] = c * vrp - s * vrq
                    eigenvectors_out[r * dim + q] = s * vrp + c * vrq

    # Extract eigenvalues from diagonal
    for i in range(dim):
        eigenvalues_out[i] = a[i * dim + i]

    # Sort eigenvalues and eigenvectors in descending order
    for i in range(dim):
        var max_idx = i
        var max_val = eigenvalues_out[i]
        for j in range(i + 1, dim):
            if eigenvalues_out[j] > max_val:
                max_val = eigenvalues_out[j]
                max_idx = j

        if max_idx != i:
            eigenvalues_out[max_idx] = eigenvalues_out[i]
            eigenvalues_out[i] = max_val
            # Swap eigenvector columns i and max_idx
            for r in range(dim):
                var tmp = eigenvectors_out[r * dim + i]
                eigenvectors_out[r * dim + i] = eigenvectors_out[r * dim + max_idx]
                eigenvectors_out[r * dim + max_idx] = tmp

    a.free()


# -----------------------------------------------------------------------------
# 3. Block Subspace Iteration
# -----------------------------------------------------------------------------

def block_subspace_iteration(
    cov: UnsafePointer[Float32, _],
    dim: Int,
    k: Int,
    basis_out: UnsafePointer[Float32, MutUntrackedOrigin],
    eigenvalues_out: UnsafePointer[Float32, MutUntrackedOrigin],
    max_iters: Int = 60,
    tol: Float32 = 1e-4,
):
    """
    Computes top-k eigenvectors of symmetric cov (dim x dim) via subspace iteration
    with Modified Gram-Schmidt and Rayleigh-Ritz Jacobi refinement.
    basis_out will hold [k, dim] (row-major: each row is an eigenvector of length dim).
    """
    # V is [dim, k] (columns are eigenvectors)
    var v = alloc[Float32](dim * k)
    var y = alloc[Float32](dim * k)

    # Initialize V with deterministic pseudo-random orthogonal basis
    var seed: UInt64 = 123456789
    for i in range(dim * k):
        seed = seed * 6364136223846793005 + 1442695040888963407
        var bit_val = (seed >> 40) & 0xFFFFFF
        v[i] = Float32(bit_val) / Float32(0x800000) - 1.0

    # Orthonormalize initial V via Modified Gram-Schmidt
    for j in range(k):
        # Orthogonalize against earlier columns
        for prev in range(j):
            var dot: Float32 = 0.0
            for r in range(dim):
                dot += v[r * k + prev] * v[r * k + j]
            for r in range(dim):
                v[r * k + j] -= dot * v[r * k + prev]

        # Normalize
        var col_sq: Float32 = 0.0
        for r in range(dim):
            col_sq += v[r * k + j] * v[r * k + j]
        var inv_col_norm = 1.0 / sqrt(col_sq if col_sq > 1e-12 else 1e-12)
        for r in range(dim):
            v[r * k + j] *= inv_col_norm

    # Subspace iteration loop
    for _ in range(max_iters):
        # Step A: Y = Cov * V (dim x dim * dim x k -> dim x k)
        for r in range(dim):
            var cov_r_off = r * dim
            for c in range(k):
                var sum_val: Float32 = 0.0
                for d in range(dim):
                    sum_val += cov[cov_r_off + d] * v[d * k + c]
                y[r * k + c] = sum_val

        # Step B: Modified Gram-Schmidt orthonormalization of Y -> V
        for j in range(k):
            for r in range(dim):
                v[r * k + j] = y[r * k + j]

            for prev in range(j):
                var dot: Float32 = 0.0
                for r in range(dim):
                    dot += v[r * k + prev] * v[r * k + j]
                for r in range(dim):
                    v[r * k + j] -= dot * v[r * k + prev]

            var col_sq: Float32 = 0.0
            for r in range(dim):
                col_sq += v[r * k + j] * v[r * k + j]
            var inv_norm = 1.0 / sqrt(col_sq if col_sq > 1e-12 else 1e-12)
            for r in range(dim):
                v[r * k + j] *= inv_norm

    # Step C: Rayleigh-Ritz Projection: M = V^T * Cov * V (k x k)
    var cv = alloc[Float32](dim * k)
    for r in range(dim):
        var cov_r_off = r * dim
        for c in range(k):
            var sum_val: Float32 = 0.0
            for d in range(dim):
                sum_val += cov[cov_r_off + d] * v[d * k + c]
            cv[r * k + c] = sum_val

    var m = alloc[Float32](k * k)
    for i in range(k):
        for j in range(k):
            var dot: Float32 = 0.0
            for r in range(dim):
                dot += v[r * k + i] * cv[r * k + j]
            m[i * k + j] = dot

    # Diagonalize small k x k matrix M using Jacobi
    var eig_vecs = alloc[Float32](k * k)
    jacobi_diagonalize(m, k, eig_vecs, eigenvalues_out)

    # Final eigenvectors: U = V * eig_vecs (dim x k)
    # Write into basis_out as [k, dim] (row-major: basis_out[row, col])
    for j in range(k):
        var row_off = j * dim
        for r in range(dim):
            var sum_val: Float32 = 0.0
            for c in range(k):
                sum_val += v[r * k + c] * eig_vecs[c * k + j]
            basis_out[row_off + r] = sum_val

    v.free()
    y.free()
    cv.free()
    m.free()
    eig_vecs.free()


# -----------------------------------------------------------------------------
# 4. Save and Load Projections File
# -----------------------------------------------------------------------------

def save_projections_file(
    path: String,
    num_layers: Int,
    in_dim: Int,
    idx_dim: Int,
    projections: UnsafePointer[Float32, _],
) raises:
    """
    Saves projections for all global layers into binary file.
    projections buffer holds [num_layers, idx_dim, in_dim] floats.
    """
    var libc = OwnedDLHandle("libc.so.6")
    var c_open = libc.get_function[def(Int, Int32, Int32) thin abi("C") -> Int32]("open")
    var c_pwrite = libc.get_function[def(Int32, Int, Int, Int) thin abi("C") -> Int]("pwrite64")
    var c_close = libc.get_function[def(Int32) thin abi("C") -> Int32]("close")

    var path_bytes = path.as_bytes()
    var path_ptr = alloc[UInt8](len(path_bytes) + 1)
    for i in range(len(path_bytes)):
        path_ptr[i] = path_bytes[i]
    path_ptr[len(path_bytes)] = 0
    var fd = c_open(path_ptr.__int__(), 577, 420)
    path_ptr.free()
    if fd < 0:
        raise Error("Failed to open projections file for write: " + path)

    # 16-byte header: magic, num_layers, in_dim, idx_dim
    var header = alloc[UInt32](4)
    header[0] = LSA_PROJ_MAGIC
    header[1] = UInt32(num_layers)
    header[2] = UInt32(in_dim)
    header[3] = UInt32(idx_dim)
    _ = c_pwrite(fd, header.__int__(), 16, 0)
    header.free()

    var layer_weights_bytes = idx_dim * in_dim * 4
    for l in range(num_layers):
        var src_ptr = projections + l * (idx_dim * in_dim)
        # Write W_Q
        var offset_q = 16 + (l * 2) * layer_weights_bytes
        _ = c_pwrite(fd, src_ptr.__int__(), layer_weights_bytes, offset_q)
        # Write W_K (identical to W_Q in PCA training-free baseline)
        var offset_k = 16 + (l * 2 + 1) * layer_weights_bytes
        _ = c_pwrite(fd, src_ptr.__int__(), layer_weights_bytes, offset_k)

    _ = c_close(fd)


def load_projections_file(
    path: String,
    projections_out: UnsafePointer[Float32, MutUntrackedOrigin],
) raises:
    """
    Loads projections for all global layers from binary file.
    projections_out will receive [num_layers, idx_dim, in_dim] floats.
    """
    var libc = OwnedDLHandle("libc.so.6")
    var c_open = libc.get_function[def(Int, Int32, Int32) thin abi("C") -> Int32]("open")
    var c_pread = libc.get_function[def(Int32, Int, Int, Int) thin abi("C") -> Int]("pread64")
    var c_close = libc.get_function[def(Int32) thin abi("C") -> Int32]("close")

    var path_bytes = path.as_bytes()
    var path_ptr = alloc[UInt8](len(path_bytes) + 1)
    for i in range(len(path_bytes)):
        path_ptr[i] = path_bytes[i]
    path_ptr[len(path_bytes)] = 0
    var fd = c_open(path_ptr.__int__(), 0, 0)
    path_ptr.free()
    if fd < 0:
        raise Error("Failed to open projections file for read: " + path)

    var header = alloc[UInt32](4)
    _ = c_pread(fd, header.__int__(), 16, 0)
    if header[0] != LSA_PROJ_MAGIC:
        header.free()
        _ = c_close(fd)
        raise Error("Invalid magic in projections file: " + path)

    var num_layers = Int(header[1])
    var in_dim = Int(header[2])
    var idx_dim = Int(header[3])
    header.free()

    var layer_weights_bytes = idx_dim * in_dim * 4
    for l in range(num_layers):
        var dest_ptr = projections_out + l * (idx_dim * in_dim)
        var offset_q = 16 + (l * 2) * layer_weights_bytes
        _ = c_pread(fd, dest_ptr.__int__(), layer_weights_bytes, offset_q)

    _ = c_close(fd)
