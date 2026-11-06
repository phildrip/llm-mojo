from std.memory import alloc
from std.math import sqrt
from std.testing import assert_equal, assert_true
from qk_dump import QKDumpReader, Q_FLOATS, K_FLOATS, V_FLOATS
from lsa_reference import (
    extract_position_free_and_regroup,
    pool_keys,
    ward_merge,
    build_buckets,
    expand_buckets,
    lsa_select_keys,
    GEMMA4_NUM_Q_HEADS,
    GEMMA4_NUM_KV_HEADS,
    GEMMA4_HEAD_DIM,
    GEMMA4_INDEX_DIM,
    DEFAULT_POOL_SIZE,
    DEFAULT_CLUSTER_BLOCK_SIZE,
    DEFAULT_BUCKET_SIZE,
    DEFAULT_BUCKET_CAP,
)

def test_extract_and_regroup() raises:
    var q = alloc[Float32](GEMMA4_NUM_Q_HEADS * GEMMA4_HEAD_DIM)
    var k = alloc[Float32](GEMMA4_NUM_KV_HEADS * GEMMA4_HEAD_DIM)
    var q_idx = alloc[Float32](GEMMA4_INDEX_DIM)
    var k_idx = alloc[Float32](GEMMA4_INDEX_DIM)

    # Fill q and k with non-zero values
    for i in range(GEMMA4_NUM_Q_HEADS * GEMMA4_HEAD_DIM):
        q[i] = Float32(0.1 + Float64(i % 10) * 0.05)
    for i in range(GEMMA4_NUM_KV_HEADS * GEMMA4_HEAD_DIM):
        k[i] = Float32(0.2 + Float64(i % 7) * 0.03)

    extract_position_free_and_regroup(q, k, q_idx, k_idx)

    # 1. Verify unit norm
    var sq_q: Float32 = 0.0
    var sq_k: Float32 = 0.0
    for i in range(GEMMA4_INDEX_DIM):
        sq_q += q_idx[i] * q_idx[i]
        sq_k += k_idx[i] * k_idx[i]

    var norm_q = sqrt(sq_q)
    var norm_k = sqrt(sq_k)
    assert_true(abs(norm_q - 1.0) < 1e-5)
    assert_true(abs(norm_k - 1.0) < 1e-5)

    # 2. Check regroup math:
    # First 192 dims of Group 0 in q_idx should be proportional to sum of heads 0..7 at dims 64..255
    var expected_sum_0: Float32 = 0.0
    var expected_sum_1: Float32 = 0.0
    for h in range(8):
        expected_sum_0 += q[h * GEMMA4_HEAD_DIM + 64]
        expected_sum_1 += q[h * GEMMA4_HEAD_DIM + 65]
    var expected_ratio_q = expected_sum_0 / expected_sum_1
    var actual_ratio_q = q_idx[0] / q_idx[1]
    assert_true(abs(expected_ratio_q - actual_ratio_q) < 1e-5)

    q.free()
    k.free()
    q_idx.free()
    k_idx.free()
    print("test_extract_and_regroup passed")

def test_pool_keys() raises:
    var num_keys = 8
    var dim = 4
    var pool_size = 4
    var keys = alloc[Float32](num_keys * dim)
    var pools = alloc[Float32]((num_keys // pool_size) * dim)

    for i in range(num_keys * dim):
        keys[i] = 1.0

    pool_keys(keys, num_keys, dim, pool_size, pools)

    # Each pool should have sum = 4.0
    for i in range(2 * dim):
        assert_equal(pools[i], 4.0)

    keys.free()
    pools.free()
    print("test_pool_keys passed")

def test_ward_clustering_synthetic() raises:
    # 4 distinct synthetic clusters in 16 dims, 4 pools each (total 16 pools)
    # Merging 16 pools down to 4 buckets should cleanly separate them
    var num_pools = 16
    var dim = 16
    var pool_size = 4
    var bucket_count = 4
    var bucket_cap = 32

    var pools = alloc[Float32](num_pools * dim)
    for i in range(num_pools * dim):
        pools[i] = 0.0

    # Cluster 0: pools 0..3 have 1.0 at dim 0
    # Cluster 1: pools 4..7 have 1.0 at dim 4
    # Cluster 2: pools 8..11 have 1.0 at dim 8
    # Cluster 3: pools 12..15 have 1.0 at dim 12
    for c in range(4):
        for p in range(4):
            var pool_id = c * 4 + p
            pools[pool_id * dim + c * 4] = 1.0 + Float32(p) * 0.01

    var labels = alloc[Int](num_pools)
    ward_merge(pools, num_pools, dim, pool_size, bucket_count, bucket_cap, labels)

    # All pools in the same cluster must share the same label
    for c in range(4):
        var expected_label = labels[c * 4]
        for p in range(1, 4):
            assert_equal(labels[c * 4 + p], expected_label)

    # The 4 clusters must have 4 distinct labels (0, 1, 2, 3)
    var label_seen = List[Bool]()
    for _ in range(4):
        label_seen.append(False)
    for c in range(4):
        var l = labels[c * 4]
        assert_true(l >= 0 and l < 4)
        assert_true(not label_seen[l])
        label_seen[l] = True

    pools.free()
    labels.free()
    print("test_ward_clustering_synthetic passed")

def test_build_and_expand_buckets() raises:
    # Test building buckets and expanding
    var num_keys = 64
    var dim = 8
    var pool_size = 4
    var bucket_size = 16 # 4 buckets of 16
    var bucket_cap = 32
    var cluster_block_size = 64

    var keys = alloc[Float32](num_keys * dim)
    for i in range(num_keys * dim):
        keys[i] = 0.1

    var buckets = build_buckets(
        keys, num_keys, dim, pool_size, bucket_size, bucket_cap, cluster_block_size, "position"
    )

    assert_equal(buckets.total_buckets, 4)
    assert_equal(buckets.block_bucket_count, 4)

    # Test expand_buckets
    var picked = List[Int]()
    picked.append(0) # bucket 0 (tokens 0..15)
    picked.append(2) # bucket 2 (tokens 32..47)

    var position = 63
    var top_k = 10
    var sink_size = 2
    var window_size = 4

    var selected = expand_buckets(picked, position, buckets, top_k, sink_size, window_size)

    # Sinks: 0, 1
    # Window: 60, 61, 62, 63
    # Top-k members (from buckets 0 and 2, strictly between sink_size=2 and window_start=60):
    # bucket 0: 2, 3, 4, 5, 6, 7, 8, 9, 10, 11
    # Total selected should be 10 members + 4 window + 2 sinks = 16
    assert_equal(len(selected), 16)

    # First top_k elements are the picked members
    for i in range(top_k):
        assert_equal(selected[i], 2 + i)

    # Next 4 are the window: 60, 61, 62, 63
    assert_equal(selected[10], 60)
    assert_equal(selected[11], 61)
    assert_equal(selected[12], 62)
    assert_equal(selected[13], 63)

    # Last 2 are the sinks: 0, 1
    assert_equal(selected[14], 0)
    assert_equal(selected[15], 1)

    keys.free()
    print("test_build_and_expand_buckets passed")

def test_captured_tokens_roundtrip() raises:
    var reader = QKDumpReader("/tmp/test_captured_qk.bin")
    assert_true(reader.num_tokens >= 10)

    var q = alloc[Float32](Q_FLOATS)
    var k = alloc[Float32](K_FLOATS)
    var v = alloc[Float32](V_FLOATS)
    var q_idx = alloc[Float32](GEMMA4_INDEX_DIM)
    var k_idx = alloc[Float32](GEMMA4_INDEX_DIM)

    # Read token 0, layer 0 (global layer 5)
    reader.read_layer(0, 0, q, k, v)
    extract_position_free_and_regroup(q, k, q_idx, k_idx)

    # Verify unit norm of extracted vectors
    var sq_q: Float32 = 0.0
    var sq_k: Float32 = 0.0
    for i in range(GEMMA4_INDEX_DIM):
        sq_q += q_idx[i] * q_idx[i]
        sq_k += k_idx[i] * k_idx[i]

    var norm_q = sqrt(sq_q)
    var norm_k = sqrt(sq_k)
    assert_true(abs(norm_q - 1.0) < 1e-4)
    assert_true(abs(norm_k - 1.0) < 1e-4)

    reader.close()
    q.free()
    k.free()
    v.free()
    q_idx.free()
    k_idx.free()
    print("test_captured_tokens_roundtrip passed")

def main() raises:
    test_extract_and_regroup()
    test_pool_keys()
    test_ward_clustering_synthetic()
    test_build_and_expand_buckets()
    test_captured_tokens_roundtrip()
    print("ALL LSA REFERENCE TESTS PASSED")
