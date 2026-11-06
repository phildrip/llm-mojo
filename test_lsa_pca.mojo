from std.memory import alloc
from std.math import sqrt
from std.testing import assert_equal, assert_true
from lsa_pca import (
    compute_covariance,
    jacobi_diagonalize,
    block_subspace_iteration,
    save_projections_file,
    load_projections_file,
)

def test_covariance() raises:
    var num_keys = 4
    var dim = 2
    var keys = alloc[Float32](num_keys * dim)
    # Points: (1, 2), (3, 4), (5, 6), (7, 8)
    # Mean: (4, 5)
    # Centered: (-3, -3), (-1, -1), (1, 1), (3, 3)
    # Covariance: (9 + 1 + 1 + 9) / 4 = 20 / 4 = 5.0 for all elements!
    keys[0] = 1.0; keys[1] = 2.0
    keys[2] = 3.0; keys[3] = 4.0
    keys[4] = 5.0; keys[5] = 6.0
    keys[6] = 7.0; keys[7] = 8.0

    var cov = alloc[Float32](dim * dim)
    compute_covariance(keys, num_keys, dim, cov)

    assert_true(abs(cov[0] - 5.0) < 1e-5)
    assert_true(abs(cov[1] - 5.0) < 1e-5)
    assert_true(abs(cov[2] - 5.0) < 1e-5)
    assert_true(abs(cov[3] - 5.0) < 1e-5)

    keys.free()
    cov.free()
    print("test_covariance passed")

def test_jacobi_diagonalize() raises:
    # 3x3 symmetric matrix
    var dim = 3
    var a = alloc[Float32](dim * dim)
    a[0] = 2.0; a[1] = -1.0; a[2] = 0.0
    a[3] = -1.0; a[4] = 2.0; a[5] = -1.0
    a[6] = 0.0; a[7] = -1.0; a[8] = 2.0

    var eig_vecs = alloc[Float32](dim * dim)
    var eig_vals = alloc[Float32](dim)

    jacobi_diagonalize(a, dim, eig_vecs, eig_vals)

    # Known analytical eigenvalues for this tridiagonal matrix:
    # 2 + sqrt(2) ≈ 3.4142, 2.0, 2 - sqrt(2) ≈ 0.58578
    var sqrt2 = Float32(sqrt(2.0))
    assert_true(abs(eig_vals[0] - (Float32(2.0) + sqrt2)) < 1e-4)
    assert_true(abs(eig_vals[1] - Float32(2.0)) < 1e-4)
    assert_true(abs(eig_vals[2] - (Float32(2.0) - sqrt2)) < 1e-4)

    # Verify orthonormality: V^T * V = I
    for i in range(dim):
        for j in range(dim):
            var dot: Float32 = 0.0
            for r in range(dim):
                dot += eig_vecs[r * dim + i] * eig_vecs[r * dim + j]
            var expected: Float32 = 1.0 if i == j else 0.0
            assert_true(abs(dot - expected) < 1e-4)

    a.free()
    eig_vecs.free()
    eig_vals.free()
    print("test_jacobi_diagonalize passed")

def test_block_subspace_iteration() raises:
    # Create a 16x16 symmetric positive definite matrix with known spectra
    var dim = 16
    var k = 4
    var cov = alloc[Float32](dim * dim)
    for i in range(dim * dim):
        cov[i] = 0.0

    # Diagonal with dominant eigenvalues: 100.0, 50.0, 25.0, 10.0, 1.0, 1.0...
    cov[0] = 100.0
    cov[1 * dim + 1] = 50.0
    cov[2 * dim + 2] = 25.0
    cov[3 * dim + 3] = 10.0
    for i in range(4, dim):
        cov[i * dim + i] = 1.0

    var basis = alloc[Float32](k * dim)
    var eig_vals = alloc[Float32](k)

    block_subspace_iteration(cov, dim, k, basis, eig_vals)

    # Verify discovered top-4 eigenvalues match 100, 50, 25, 10
    assert_true(abs(eig_vals[0] - 100.0) < 1e-2)
    assert_true(abs(eig_vals[1] - 50.0) < 1e-2)
    assert_true(abs(eig_vals[2] - 25.0) < 1e-2)
    assert_true(abs(eig_vals[3] - 10.0) < 1e-2)

    # Verify residual: ||Cov * v - lambda * v|| < 1e-3
    for j in range(k):
        var lambda_val = eig_vals[j]
        var res_sq: Float32 = 0.0
        for r in range(dim):
            var cv_r: Float32 = 0.0
            for c in range(dim):
                cv_r += cov[r * dim + c] * basis[j * dim + c]
            var diff = cv_r - lambda_val * basis[j * dim + r]
            res_sq += diff * diff
        assert_true(sqrt(res_sq) < 1e-3)

    cov.free()
    basis.free()
    eig_vals.free()
    print("test_block_subspace_iteration passed")

def test_save_load_projections() raises:
    var test_path = String("/tmp/test_projections.bin")
    var num_layers = 5
    var in_dim = 768
    var idx_dim = 64
    var total_floats = num_layers * idx_dim * in_dim

    var proj_orig = alloc[Float32](total_floats)
    for i in range(total_floats):
        proj_orig[i] = Float32(Float64(i % 100) * 0.01)

    save_projections_file(test_path, num_layers, in_dim, idx_dim, proj_orig)

    var proj_loaded = alloc[Float32](total_floats)
    load_projections_file(test_path, proj_loaded)

    for i in range(total_floats):
        assert_equal(proj_orig[i], proj_loaded[i])

    proj_orig.free()
    proj_loaded.free()
    print("test_save_load_projections passed")

def main() raises:
    test_covariance()
    test_jacobi_diagonalize()
    test_block_subspace_iteration()
    test_save_load_projections()
    print("ALL LSA PCA TESTS PASSED")
