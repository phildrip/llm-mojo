# ===----------------------------------------------------------------------=== #
# Benchmark Suite: Latent Sparse Attention (LSA) for Gemma-4 26B MoE
#
# Measures:
#   1. SIMD Index Projection throughput (768-D -> 64-D + L2 norm)
#   2. SIMD Ward hierarchical clustering latency per 512-token block
#   3. Top-p centroid search & expansion latency across context sizes
#      (T = 1k, 4k, 16k, 32k, 64k, 128k)
#   4. Paged Attention latency: Dense O(T) vs LSA Sparse Gather O(k)
#   5. Memory footprint audit across context lengths
# ===----------------------------------------------------------------------=== #

from std.memory import alloc
import std.time
from std.math import sqrt

from lsa_pca import load_projections_file, DEFAULT_PCA_DIM
from lsa_core import (
    lsa_index_query,
    lsa_index_key,
    LSALayerIndex,
    lsa_sparse_paged_attention,
    LSA_CLUSTER_BLOCK_SIZE,
    LSA_BLOCK_BUCKET_COUNT,
    LSA_DIM,
    GEMMA4_INDEX_DIM,
)
from attention import paged_attention_layer
from qk_dump import QKDumpReader, Q_FLOATS, K_FLOATS, V_FLOATS

def pad_left(s: String, width: Int) -> String:
    var res = String("")
    var pad = width - s.byte_length()
    for _ in range(pad):
        res += " "
    res += s
    return res

def pad_right(s: String, width: Int) -> String:
    var res = s
    var pad = width - s.byte_length()
    for _ in range(pad):
        res += " "
    return res

def format_f(val: Float64, decimals: Int = 2) -> String:
    var sign = "-" if val < 0.0 else ""
    var v = -val if val < 0.0 else val
    var mul: Float64 = 100.0
    if decimals == 1:
        mul = 10.0
    elif decimals == 3:
        mul = 1000.0
    var rounded = Float64(Int(v * mul + 0.5)) / mul
    return sign + String(rounded)

def main() raises:
    print("==========================================================================")
    print("   LSA (LATENT SPARSE ATTENTION) PERFORMANCE & SCALING BENCHMARK          ")
    print("   Target: Gemma-4 26B MoE Global Layers (Hq=16, Hkv=2, d=512)            ")
    print("==========================================================================")
    print()

    # Load real projection weights
    var proj = alloc[Float32](5 * DEFAULT_PCA_DIM * GEMMA4_INDEX_DIM)
    load_projections_file("gemma4_lsa_projections.bin", proj)

    # Load real activations from scratch dump
    var reader = QKDumpReader("scratch/lsa/captured_qk_1500.bin")
    var q_buf = alloc[Float32](Q_FLOATS)
    var k_buf = alloc[Float32](K_FLOATS)
    var v_buf = alloc[Float32](V_FLOATS)
    reader.read_layer(0, 0, q_buf, k_buf, v_buf)
    reader.close()

    var q_lsa = alloc[Float32](LSA_DIM)
    var k_lsa = alloc[Float32](LSA_DIM)

    # -------------------------------------------------------------------------
    # 1. Microbenchmark: SIMD Projection & Normalization (768-D -> 64-D)
    # -------------------------------------------------------------------------
    print("--- 1. SIMD PROJECTION & L2 NORMALIZATION (768-D -> 64-D) ---")
    var iters = 20000

    # Warmup
    for _ in range(500):
        lsa_index_key(k_buf, proj, k_lsa)
        lsa_index_query(q_buf, proj, q_lsa)

    var t0 = std.time.perf_counter()
    for _ in range(iters):
        lsa_index_key(k_buf, proj, k_lsa)
    var t1 = std.time.perf_counter()
    var key_proj_ns = ((t1 - t0) / Float64(iters)) * 1e9

    t0 = std.time.perf_counter()
    for _ in range(iters):
        lsa_index_query(q_buf, proj, q_lsa)
    t1 = std.time.perf_counter()
    var query_proj_ns = ((t1 - t0) / Float64(iters)) * 1e9

    print("Key Projection + L2 Norm   : " + format_f(key_proj_ns, 1) + " ns / token (" + format_f(1e9 / key_proj_ns / 1e6, 2) + " M tok/s)")
    print("Query Projection + L2 Norm : " + format_f(query_proj_ns, 1) + " ns / token (" + format_f(1e9 / query_proj_ns / 1e6, 2) + " M tok/s)")
    print()

    # -------------------------------------------------------------------------
    # 2. Microbenchmark: Mutual-Cheapest Ward Clustering per 512-Token Block
    # -------------------------------------------------------------------------
    print("--- 2. MUTUAL-CHEAPEST WARD CLUSTERING (512 keys -> 32 clusters) ---")
    var ward_idx = LSALayerIndex(max_capacity=2048)

    # Fill 512 staging keys
    for i in range(LSA_CLUSTER_BLOCK_SIZE):
        for d in range(LSA_DIM):
            k_lsa[d] = Float32((i * 17 + d * 31) % 100) / 100.0
        var sq: Float32 = 0.0
        for d in range(LSA_DIM):
            sq += k_lsa[d] * k_lsa[d]
        var inv_n = 1.0 / sqrt(sq)
        for d in range(LSA_DIM):
            k_lsa[d] *= inv_n
        ward_idx.append_key(i, k_lsa)

    # ward_cluster_block already ran once at pos=511, reset closed_blocks to bench it repeatedly
    var ward_iters = 200
    t0 = std.time.perf_counter()
    for _ in range(ward_iters):
        ward_idx.closed_blocks = 0
        ward_idx.ward_cluster_block()
    t1 = std.time.perf_counter()
    var ward_us = ((t1 - t0) / Float64(ward_iters)) * 1e6

    print("Ward 512->32 Clustering Latency : " + format_f(ward_us, 1) + " µs / 512-token block")
    print("Amortized Key Ingestion Rate    : " + format_f(ward_us / 512.0 * 1000.0, 1) + " ns / token overhead")
    print()
    ward_idx.free()

    # -------------------------------------------------------------------------
    # 3. Microbenchmark: Top-p Centroid Scan & Expansion Across Context Sizes
    # -------------------------------------------------------------------------
    print("--- 3. TOP-p CENTROID SCAN & EXPANSION LATENCY SCALING ---")
    print("  Columns: Context (T) | Blocks | Centroid Scan (µs) | Expand (µs) | Total (µs)")
    print("  ----------------------------------------------------------------------------")

    var test_contexts = List[Int]()
    test_contexts.append(1024)
    test_contexts.append(4096)
    test_contexts.append(16384)
    test_contexts.append(32768)
    test_contexts.append(65536)
    test_contexts.append(131072)

    var max_T = 131072
    var lsa_idx = LSALayerIndex(max_capacity=max_T)

    # Ingest synthetic keys up to max_T
    for t in range(max_T):
        var d_base = (t * 7) % 64
        for d in range(LSA_DIM):
            k_lsa[d] = 0.05
        k_lsa[d_base] = 1.0
        var sq: Float32 = 0.0
        for d in range(LSA_DIM):
            sq += k_lsa[d] * k_lsa[d]
        var inv_n = 1.0 / sqrt(sq)
        for d in range(LSA_DIM):
            k_lsa[d] *= inv_n
        lsa_idx.append_key(t, k_lsa)

    # Set query
    for d in range(LSA_DIM):
        q_lsa[d] = 0.1
    q_lsa[0] = 1.0

    var scan_iters = 500
    for ci in range(len(test_contexts)):
        var cur_T = test_contexts[ci]
        var num_blocks = cur_T // LSA_CLUSTER_BLOCK_SIZE

        # Benchmark select_top_p
        t0 = std.time.perf_counter()
        for _ in range(scan_iters):
            var picked = lsa_idx.select_top_p(q_lsa, cur_T, 20)
        t1 = std.time.perf_counter()
        var scan_us = ((t1 - t0) / Float64(scan_iters)) * 1e6

        # Benchmark expand_buckets
        var picked = lsa_idx.select_top_p(q_lsa, cur_T, 20)
        t0 = std.time.perf_counter()
        for _ in range(scan_iters):
            var selected = lsa_idx.expand_buckets(picked, cur_T, 256, 4, 128)
        t1 = std.time.perf_counter()
        var expand_us = ((t1 - t0) / Float64(scan_iters)) * 1e6

        var total_us = scan_us + expand_us
        print("  " + pad_right(String(cur_T), 12) + " | " + pad_right(String(num_blocks), 6) + " | " + pad_left(format_f(scan_us, 1), 18) + " | " + pad_left(format_f(expand_us, 1), 11) + " | " + pad_left(format_f(total_us, 1), 10))

    print()

    # -------------------------------------------------------------------------
    # 4. Kernel Benchmark: Sparse Paged Attention Gather vs Dense Attention
    # -------------------------------------------------------------------------
    print("--- 4. ATTENTION EXECUTION LATENCY: DENSE O(T) vs LSA SPARSE O(k) ---")
    print("  Setup: Gemma-4 Global Layer (Hq=16, Hkv=2, d=512), BlockSize=16, k=256")
    print("  Columns: Context (T) | Dense Attention (µs) | LSA Sparse Attn (µs) | Gather Speedup")
    print("  -----------------------------------------------------------------------------------")

    var block_size = 16
    var max_blocks = max_T // block_size
    var block_table = List[Int]()
    for b in range(max_blocks):
        block_table.append(b)

    var k_pool = alloc[Float32](max_blocks * block_size * 2 * 512)
    var v_pool = alloc[Float32](max_blocks * block_size * 2 * 512)
    for i in range(1000):
        k_pool[i] = 0.01
        v_pool[i] = 0.01

    var attn_scores = alloc[Float32](16 * max_T)
    var attn_rows = alloc[Int](max_T)
    var attn_out_dense = alloc[Float32](16 * 512)
    var attn_out_sparse = alloc[Float32](16 * 512)

    var q_global = alloc[Float32](16 * 512)
    for i in range(16 * 512):
        q_global[i] = 0.02

    var attn_iters = 30

    for ci in range(len(test_contexts)):
        var cur_T = test_contexts[ci]

        # Warmup
        paged_attention_layer(
            q_global, k_pool, v_pool, block_table, block_size, cur_T - 1,
            16, 2, 512, 0, attn_scores, attn_rows, attn_out_dense, 0, 0
        )

        t0 = std.time.perf_counter()
        for _ in range(attn_iters):
            paged_attention_layer(
                q_global, k_pool, v_pool, block_table, block_size, cur_T - 1,
                16, 2, 512, 0, attn_scores, attn_rows, attn_out_dense, 0, 0
            )
        t1 = std.time.perf_counter()
        var dense_us = ((t1 - t0) / Float64(attn_iters)) * 1e6

        # LSA Sparse Attention (k=256)
        var picked = lsa_idx.select_top_p(q_lsa, cur_T - 1, 20)
        var selected_keys = lsa_idx.expand_buckets(picked, cur_T - 1, 256, 4, 128)

        # Warmup
        lsa_sparse_paged_attention(
            q_global, k_pool, v_pool, block_table, block_size, selected_keys,
            16, 2, 512, attn_scores, attn_rows, attn_out_sparse
        )

        t0 = std.time.perf_counter()
        for _ in range(attn_iters):
            lsa_sparse_paged_attention(
                q_global, k_pool, v_pool, block_table, block_size, selected_keys,
                16, 2, 512, attn_scores, attn_rows, attn_out_sparse
            )
        t1 = std.time.perf_counter()
        var sparse_us = ((t1 - t0) / Float64(attn_iters)) * 1e6

        var speedup = dense_us / sparse_us
        print("  " + pad_right(String(cur_T), 12) + " | " + pad_left(format_f(dense_us, 1), 20) + " | " + pad_left(format_f(sparse_us, 1), 20) + " | " + pad_left(format_f(speedup, 1) + "x", 14))

    print()

    # -------------------------------------------------------------------------
    # 5. Memory Footprint Audit: Dense vs Sliding Window vs LSA
    # -------------------------------------------------------------------------
    print("--- 5. MEMORY FOOTPRINT AUDIT ACROSS CONTEXT LENGTHS ---")
    print("  Gemma-4 26B Architecture:")
    print("    - 25 Sliding-Window Layers: Hq=16, Hkv=8, d=256 (32 KB / tok dense, ring-buffer locked to 1024)")
    print("    - 5 Global Attention Layers: Hq=16, Hkv=2, d=512 (8 KB / tok dense)")
    print("    - Total Dense Model KV: 25*32 KB + 5*8 KB = 840 KB / token (FP32)")
    print()
    print("  Columns: Context (T) | Dense Full (MB) | SWA 1024 Only (MB) | LSA + SWA (MB) | KV Savings")
    print("  ------------------------------------------------------------------------------------------")

    var swa_fixed_mb: Float64 = (25.0 * 1024.0 * 8.0 * 256.0 * 4.0 * 2.0) / (1024.0 * 1024.0)

    for ci in range(len(test_contexts)):
        var cur_T = Float64(test_contexts[ci])
        # Dense full across all 30 layers
        var dense_full_mb = (cur_T * 840.0 * 1024.0) / (1024.0 * 1024.0)

        # SWA 1024 on 25 layers + Dense global on 5 layers
        var swa_dense_global_mb = swa_fixed_mb + (5.0 * cur_T * 2.0 * 512.0 * 4.0 * 2.0) / (1024.0 * 1024.0)

        # LSA Index overhead: 64 floats * 4B / 32 tok (centroids) + 32 * 4B / 32 tok (members) = 12 B / tok per layer
        var lsa_idx_overhead_mb = (5.0 * cur_T * 12.0) / (1024.0 * 1024.0)
        var lsa_total_mb = swa_dense_global_mb + lsa_idx_overhead_mb

        var ratio = dense_full_mb / lsa_total_mb
        print("  " + pad_right(String(Int(cur_T)), 12) + " | " + pad_left(format_f(dense_full_mb, 1), 15) + " | " + pad_left(format_f(swa_dense_global_mb, 1), 18) + " | " + pad_left(format_f(lsa_total_mb, 1), 14) + " | " + pad_left(format_f(ratio, 1) + "x", 10))

    print()
    print("==========================================================================")
    print("   LSA BENCHMARK COMPLETED SUCCESSFULLY                                   ")
    print("==========================================================================")

    lsa_idx.free()
    k_pool.free()
    v_pool.free()
    attn_scores.free()
    attn_rows.free()
    attn_out_dense.free()
    attn_out_sparse.free()
    q_global.free()
    q_lsa.free()
    k_lsa.free()
    q_buf.free()
    k_buf.free()
    v_buf.free()
    proj.free()
