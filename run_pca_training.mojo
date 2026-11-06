import std.time
from std.memory import alloc
from qk_dump import QKDumpReader, Q_FLOATS, K_FLOATS, V_FLOATS, NUM_GLOBAL_LAYERS, index_to_global_layer
from lsa_reference import (
    extract_position_free_and_regroup,
    GEMMA4_NUM_Q_HEADS,
    GEMMA4_NUM_KV_HEADS,
    GEMMA4_HEAD_DIM,
    GEMMA4_INDEX_DIM,
)
from lsa_pca import (
    compute_covariance,
    block_subspace_iteration,
    save_projections_file,
    DEFAULT_PCA_DIM,
)

def main() raises:
    var captured_path = String("scratch/lsa/captured_qk_1500.bin")
    var output_projections_path = String("gemma4_lsa_projections.bin")

    var cmd_args = std.sys.argv()
    if len(cmd_args) > 1:
        captured_path = String(cmd_args[1])
    if len(cmd_args) > 2:
        output_projections_path = String(cmd_args[2])

    print("=========================================================")
    print("      LSA GEMMA-4 26B MoE PCA PROJECTION GENERATOR        ")
    print("=========================================================")
    print("Source Q/K Dump :", captured_path)
    print("Output File     :", output_projections_path)
    print("---------------------------------------------------------")

    var reader = QKDumpReader(captured_path)
    var num_tokens = reader.num_tokens
    print("Captured tokens :", num_tokens)
    if num_tokens < 64:
        raise Error("Not enough tokens to compute stable PCA (minimum 64 tokens)")

    var q_buf = alloc[Float32](Q_FLOATS)
    var k_buf = alloc[Float32](K_FLOATS)
    var v_buf = alloc[Float32](V_FLOATS)
    var q_idx = alloc[Float32](GEMMA4_INDEX_DIM)

    # Buffer for all position-free keys for one layer: [num_tokens, 768]
    var layer_keys = alloc[Float32](num_tokens * GEMMA4_INDEX_DIM)
    var cov = alloc[Float32](GEMMA4_INDEX_DIM * GEMMA4_INDEX_DIM)

    # Output projections: 5 layers * 64 * 768 floats
    var projections = alloc[Float32](NUM_GLOBAL_LAYERS * DEFAULT_PCA_DIM * GEMMA4_INDEX_DIM)

    for l_idx in range(NUM_GLOBAL_LAYERS):
        var global_layer_num = index_to_global_layer(l_idx)
        print("\nProcessing Global Layer " + String(global_layer_num) + " (layer index " + String(l_idx) + " of 5)...")

        var t0 = std.time.perf_counter()
        # 1. Extract position-free keys for all tokens
        for t in range(num_tokens):
            reader.read_layer(t, l_idx, q_buf, k_buf, v_buf)
            var k_idx = layer_keys + t * GEMMA4_INDEX_DIM
            extract_position_free_and_regroup(q_buf, k_buf, q_idx, k_idx)

        var t_extract = std.time.perf_counter()
        print("  Keys extracted in: " + String((t_extract - t0) * 1000.0) + " ms")

        # 2. Compute 768x768 covariance
        compute_covariance(layer_keys, num_tokens, GEMMA4_INDEX_DIM, cov)
        var t_cov = std.time.perf_counter()
        print("  Covariance computed in: " + String((t_cov - t_extract) * 1000.0) + " ms")

        # 3. Block Subspace Iteration to extract top-64 principal components
        var basis_out = projections + l_idx * (DEFAULT_PCA_DIM * GEMMA4_INDEX_DIM)
        var eig_vals = alloc[Float32](DEFAULT_PCA_DIM)

        block_subspace_iteration(cov, GEMMA4_INDEX_DIM, DEFAULT_PCA_DIM, basis_out, eig_vals)
        var t_pca = std.time.perf_counter()
        print("  Subspace iteration (top 64) in: " + String((t_pca - t_cov) * 1000.0) + " ms")

        # Check variance explained
        var top_sum: Float32 = 0.0
        for i in range(DEFAULT_PCA_DIM):
            top_sum += eig_vals[i]

        var trace_cov: Float32 = 0.0
        for i in range(GEMMA4_INDEX_DIM):
            trace_cov += cov[i * GEMMA4_INDEX_DIM + i]

        var ratio = top_sum / (trace_cov if trace_cov > 1e-12 else 1e-12)
        print("  Top-1 eigenvalue   : " + String(eig_vals[0]))
        print("  Top-64 eigenvalue  : " + String(eig_vals[DEFAULT_PCA_DIM - 1]))
        print("  Variance explained : " + String(ratio * 100.0) + " %")
        eig_vals.free()

    reader.close()

    # Save to binary file
    save_projections_file(
        output_projections_path,
        NUM_GLOBAL_LAYERS,
        GEMMA4_INDEX_DIM,
        DEFAULT_PCA_DIM,
        projections,
    )
    print("\nSaved projections to: " + output_projections_path)

    # Clean up
    q_buf.free()
    k_buf.free()
    v_buf.free()
    q_idx.free()
    layer_keys.free()
    cov.free()
    projections.free()
    print("Done!")
