from std.memory import alloc
from std.math import sqrt
from std.testing import assert_equal, assert_true
from qk_dump import QKDumpReader, Q_FLOATS, K_FLOATS, V_FLOATS
from lsa_pca import load_projections_file, DEFAULT_PCA_DIM
from lsa_reference import (
    extract_position_free_and_regroup,
    GEMMA4_NUM_Q_HEADS,
    GEMMA4_NUM_KV_HEADS,
    GEMMA4_HEAD_DIM,
    GEMMA4_INDEX_DIM,
)
from attention import paged_attention_layer
from lsa_core import (
    lsa_index_query,
    lsa_index_key,
    LSALayerIndex,
    lsa_sparse_paged_attention,
    LSA_CLUSTER_BLOCK_SIZE,
    LSA_BLOCK_BUCKET_COUNT,
    LSA_DIM,
)

def test_simd_index_vectors() raises:
    var proj = alloc[Float32](5 * DEFAULT_PCA_DIM * GEMMA4_INDEX_DIM)
    load_projections_file("gemma4_lsa_projections.bin", proj)

    var reader = QKDumpReader("scratch/lsa/captured_qk_1500.bin")
    var q = alloc[Float32](Q_FLOATS)
    var k = alloc[Float32](K_FLOATS)
    var v = alloc[Float32](V_FLOATS)

    reader.read_layer(0, 0, q, k, v)
    reader.close()

    # 1. Scalar Reference computation
    var q_768_ref = alloc[Float32](GEMMA4_INDEX_DIM)
    var k_768_ref = alloc[Float32](GEMMA4_INDEX_DIM)
    extract_position_free_and_regroup(q, k, q_768_ref, k_768_ref)

    var q_64_ref = alloc[Float32](DEFAULT_PCA_DIM)
    var k_64_ref = alloc[Float32](DEFAULT_PCA_DIM)

    # Reference projection
    var sum_sq_q_ref: Float32 = 0.0
    var sum_sq_k_ref: Float32 = 0.0
    for r in range(DEFAULT_PCA_DIM):
        var r_off = r * GEMMA4_INDEX_DIM
        var dot_q: Float32 = 0.0
        var dot_k: Float32 = 0.0
        for d in range(GEMMA4_INDEX_DIM):
            dot_q += proj[r_off + d] * q_768_ref[d]
            dot_k += proj[r_off + d] * k_768_ref[d]
        q_64_ref[r] = dot_q
        k_64_ref[r] = dot_k
        sum_sq_q_ref += dot_q * dot_q
        sum_sq_k_ref += dot_k * dot_k

    var inv_q_ref = 1.0 / sqrt(sum_sq_q_ref)
    var inv_k_ref = 1.0 / sqrt(sum_sq_k_ref)
    for r in range(DEFAULT_PCA_DIM):
        q_64_ref[r] *= inv_q_ref
        k_64_ref[r] *= inv_k_ref

    # 2. Fast SIMD computation
    var q_64_simd = alloc[Float32](DEFAULT_PCA_DIM)
    var k_64_simd = alloc[Float32](DEFAULT_PCA_DIM)

    lsa_index_query(q, proj, q_64_simd)
    lsa_index_key(k, proj, k_64_simd)

    # 3. Assert max absolute error < 1e-5
    for r in range(DEFAULT_PCA_DIM):
        var diff_q = abs(q_64_simd[r] - q_64_ref[r])
        var diff_k = abs(k_64_simd[r] - k_64_ref[r])
        assert_true(diff_q < 1e-5)
        assert_true(diff_k < 1e-5)

    proj.free()
    q.free()
    k.free()
    v.free()
    q_768_ref.free()
    k_768_ref.free()
    q_64_ref.free()
    k_64_ref.free()
    q_64_simd.free()
    k_64_simd.free()
    print("test_simd_index_vectors passed")

def test_lsa_layer_index_lifecycle() raises:
    var index = LSALayerIndex(max_capacity=2048)
    assert_equal(index.closed_blocks, 0)

    # Append 512 synthetic keys (1 full cluster block)
    var k_unit = alloc[Float32](LSA_DIM)
    for i in range(LSA_DIM):
        k_unit[i] = 0.0

    for pos in range(512):
        var cluster_id = pos // 128
        k_unit[cluster_id * 16] = 1.0
        index.append_key(pos, k_unit)

    # Block 0 should now be closed!
    assert_equal(index.closed_blocks, 1)

    # Centroids should be normalized
    for b in range(LSA_BLOCK_BUCKET_COUNT):
        var c_off = b * LSA_DIM
        var sq: Float32 = 0.0
        for d in range(LSA_DIM):
            sq += index.centroids[c_off + d] * index.centroids[c_off + d]
        if index.sizes[b] > 0:
            assert_true(abs(sqrt(sq) - 1.0) < 1e-4)

    # Test top-p selection
    var q_probe = alloc[Float32](LSA_DIM)
    for i in range(LSA_DIM):
        q_probe[i] = 0.0
    q_probe[0] = 1.0 # matches cluster 0

    var picked = index.select_top_p(q_probe, pos=512, top_p=4)
    assert_equal(len(picked), 4)

    # Test expand_buckets
    var selected = index.expand_buckets(picked, pos=512, top_k=64, sink_size=4, window_size=16)
    assert_true(len(selected) > 0)

    # Clean up
    k_unit.free()
    q_probe.free()
    index.free()
    print("test_lsa_layer_index_lifecycle passed")

def test_sparse_attention_dense_equivalence() raises:
    var block_size = 16
    var num_tokens = 32
    var num_heads = 16
    var num_kv_heads = 2
    var head_dim = 512
    var kv_stride = num_kv_heads * head_dim

    var q = alloc[Float32](num_heads * head_dim)
    var k_pool = alloc[Float32](num_tokens * kv_stride)
    var v_pool = alloc[Float32](num_tokens * kv_stride)

    for i in range(num_heads * head_dim):
        q[i] = Float32(Float64(i % 13) * 0.01)
    for i in range(num_tokens * kv_stride):
        k_pool[i] = Float32(Float64(i % 17) * 0.01)
        v_pool[i] = Float32(Float64(i % 19) * 0.01)

    var block_table = List[Int]()
    block_table.append(0)
    block_table.append(1)

    var attn_scores_dense = alloc[Float32](num_heads * num_tokens)
    var attn_rows_dense = alloc[Int](num_tokens)
    var out_dense = alloc[Float32](num_heads * head_dim)

    # 1. Run standard dense attention
    paged_attention_layer(
        q, k_pool, v_pool, block_table, block_size,
        pos=num_tokens - 1,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        window=0,
        scores=attn_scores_dense,
        rows=attn_rows_dense,
        attn_out=out_dense,
    )

    # 2. Run sparse attention with ALL tokens selected [0..31]
    var all_keys = List[Int]()
    for t in range(num_tokens):
        all_keys.append(t)

    var attn_scores_sparse = alloc[Float32](num_heads * num_tokens)
    var attn_rows_sparse = alloc[Int](num_tokens)
    var out_sparse = alloc[Float32](num_heads * head_dim)

    lsa_sparse_paged_attention(
        q, k_pool, v_pool, block_table, block_size,
        all_keys,
        num_heads, num_kv_heads, head_dim,
        attn_scores_sparse, attn_rows_sparse, out_sparse,
    )

    # 3. Assert outputs are bit-exact with dense paged_attention_layer
    for i in range(num_heads * head_dim):
        assert_equal(out_sparse[i], out_dense[i])

    q.free()
    k_pool.free()
    v_pool.free()
    attn_scores_dense.free()
    attn_rows_dense.free()
    out_dense.free()
    attn_scores_sparse.free()
    attn_rows_sparse.free()
    out_sparse.free()
    print("test_sparse_attention_dense_equivalence passed")

def main() raises:
    test_simd_index_vectors()
    test_lsa_layer_index_lifecycle()
    test_sparse_attention_dense_equivalence()
    print("ALL LSA CORE TESTS PASSED")
