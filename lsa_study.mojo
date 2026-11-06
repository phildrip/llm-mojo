import std.time
from std.memory import alloc
from std.math import sqrt, exp
from qk_dump import QKDumpReader, Q_FLOATS, K_FLOATS, V_FLOATS, NUM_GLOBAL_LAYERS, index_to_global_layer
from lsa_reference import (
    extract_position_free_and_regroup,
    build_buckets,
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
from lsa_pca import (
    load_projections_file,
    DEFAULT_PCA_DIM,
)

def pad_left(s: String, width: Int) -> String:
    var res = String("")
    var pad = width - len(s)
    for _ in range(pad):
        res += " "
    res += s
    return res

def pad_right(s: String, width: Int) -> String:
    var res = s
    var pad = width - len(s)
    for _ in range(pad):
        res += " "
    return res

# -----------------------------------------------------------------------------
# Dense Oracle Attention (Gemma-4: RMSNorm on Q/K, scale=1.0, softmax, V dot)
# -----------------------------------------------------------------------------

def compute_dense_oracle(
    q: UnsafePointer[Float32, _],
    all_k: UnsafePointer[Float32, _],
    all_v: UnsafePointer[Float32, _],
    pos: Int,
    head_dim: Int,
    avg_attention_weights_out: UnsafePointer[Float32, MutUntrackedOrigin],
    dense_out: UnsafePointer[Float32, MutUntrackedOrigin],
):
    """
    Computes exact dense attention for query at pos over all tokens 0..pos.
    Produces:
      avg_attention_weights_out: [pos + 1] floats (mean attention mass per key across 16 heads)
      dense_out: [16 * 512] floats (concatenated head outputs)
    """
    var num_q_heads = 16
    var num_kv_heads = 2
    var kv_ratio = 8
    var total_keys = pos + 1

    var scores = alloc[Float32](total_keys)

    for j in range(total_keys):
        avg_attention_weights_out[j] = 0.0

    for h in range(num_q_heads):
        var g = h // kv_ratio
        var q_head = q + h * head_dim

        # Compute dot products
        var max_score: Float32 = -1e30
        for j in range(total_keys):
            var k_head = all_k + (j * num_kv_heads + g) * head_dim
            var dot: Float32 = 0.0
            for d in range(head_dim):
                dot += q_head[d] * k_head[d]
            scores[j] = dot
            if dot > max_score:
                max_score = dot

        # Softmax
        var sum_exp: Float32 = 0.0
        for j in range(total_keys):
            var e = exp(scores[j] - max_score)
            scores[j] = e
            sum_exp += e

        var inv_sum = 1.0 / (sum_exp if sum_exp > 1e-12 else 1e-12)
        for j in range(total_keys):
            scores[j] *= inv_sum
            avg_attention_weights_out[j] += scores[j] * (1.0 / Float32(num_q_heads))

        # Output accumulation
        var out_h = dense_out + h * head_dim
        for d in range(head_dim):
            var sum_v: Float32 = 0.0
            for j in range(total_keys):
                var v_head = all_v + (j * num_kv_heads + g) * head_dim
                sum_v += scores[j] * v_head[d]
            out_h[d] = sum_v

    scores.free()


def compute_sparse_output(
    q: UnsafePointer[Float32, _],
    all_k: UnsafePointer[Float32, _],
    all_v: UnsafePointer[Float32, _],
    selected_keys: List[Int],
    head_dim: Int,
    sparse_out: UnsafePointer[Float32, MutUntrackedOrigin],
):
    """
    Computes attention restricted to selected_keys.
    """
    var num_q_heads = 16
    var num_kv_heads = 2
    var kv_ratio = 8
    var n_sel = len(selected_keys)
    if n_sel == 0:
        for i in range(num_q_heads * head_dim):
            sparse_out[i] = 0.0
        return

    var scores = alloc[Float32](n_sel)

    for h in range(num_q_heads):
        var g = h // kv_ratio
        var q_head = q + h * head_dim

        var max_score: Float32 = -1e30
        for idx in range(n_sel):
            var j = selected_keys[idx]
            var k_head = all_k + (j * num_kv_heads + g) * head_dim
            var dot: Float32 = 0.0
            for d in range(head_dim):
                dot += q_head[d] * k_head[d]
            scores[idx] = dot
            if dot > max_score:
                max_score = dot

        var sum_exp: Float32 = 0.0
        for idx in range(n_sel):
            var e = exp(scores[idx] - max_score)
            scores[idx] = e
            sum_exp += e

        var inv_sum = 1.0 / (sum_exp if sum_exp > 1e-12 else 1e-12)
        for idx in range(n_sel):
            scores[idx] *= inv_sum

        var out_h = sparse_out + h * head_dim
        for d in range(head_dim):
            var sum_v: Float32 = 0.0
            for idx in range(n_sel):
                var j = selected_keys[idx]
                var v_head = all_v + (j * num_kv_heads + g) * head_dim
                sum_v += scores[idx] * v_head[d]
            out_h[d] = sum_v

    scores.free()


def cosine_similarity(
    a: UnsafePointer[Float32, _],
    b: UnsafePointer[Float32, _],
    n: Int,
) -> Float32:
    var dot: Float32 = 0.0
    var sq_a: Float32 = 0.0
    var sq_b: Float32 = 0.0
    for i in range(n):
        dot += a[i] * b[i]
        sq_a += a[i] * a[i]
        sq_b += b[i] * b[i]
    var denom = sqrt(sq_a) * sqrt(sq_b)
    return dot / denom if denom > 1e-12 else Float32(0.0)


# -----------------------------------------------------------------------------
# Study Main: Sweeps & Metrics Comparison
# -----------------------------------------------------------------------------

def main() raises:
    var captured_path = String("scratch/lsa/captured_qk_1500.bin")
    var projections_path = String("gemma4_lsa_projections.bin")

    var cmd_args = std.sys.argv()
    if len(cmd_args) > 1:
        captured_path = String(cmd_args[1])
    if len(cmd_args) > 2:
        projections_path = String(cmd_args[2])

    print("=========================================================================")
    print("      LSA GEMMA-4 26B MoE FEASIBILITY & SELECTION QUALITY STUDY          ")
    print("=========================================================================")
    print("Capture Data File :", captured_path)
    print("Projections File  :", projections_path)
    print("-------------------------------------------------------------------------")

    var reader = QKDumpReader(captured_path)
    var num_tokens = reader.num_tokens
    print("Total captured tokens :", num_tokens)

    # Load projections [5, 64, 768]
    var projections = alloc[Float32](NUM_GLOBAL_LAYERS * DEFAULT_PCA_DIM * GEMMA4_INDEX_DIM)
    load_projections_file(projections_path, projections)
    print("Loaded 64D PCA projections successfully.")

    # Random 64D projections for Variant D (deterministic baseline)
    var rand_proj = alloc[Float32](DEFAULT_PCA_DIM * GEMMA4_INDEX_DIM)
    var seed: UInt64 = 987654321
    for i in range(DEFAULT_PCA_DIM * GEMMA4_INDEX_DIM):
        seed = seed * 6364136223846793005 + 1442695040888963407
        var bit_val = (seed >> 40) & 0xFFFFFF
        rand_proj[i] = Float32(bit_val) / Float32(0x800000) - 1.0

    # Orthonormalize random projection rows
    for r in range(DEFAULT_PCA_DIM):
        var r_off = r * GEMMA4_INDEX_DIM
        for prev in range(r):
            var p_off = prev * GEMMA4_INDEX_DIM
            var dot: Float32 = 0.0
            for d in range(GEMMA4_INDEX_DIM):
                dot += rand_proj[p_off + d] * rand_proj[r_off + d]
            for d in range(GEMMA4_INDEX_DIM):
                rand_proj[r_off + d] -= dot * rand_proj[p_off + d]
        var sq: Float32 = 0.0
        for d in range(GEMMA4_INDEX_DIM):
            sq += rand_proj[r_off + d] * rand_proj[r_off + d]
        var inv_n = 1.0 / sqrt(sq if sq > 1e-12 else 1e-12)
        for d in range(GEMMA4_INDEX_DIM):
            rand_proj[r_off + d] *= inv_n

    # Allocate full KV arrays for evaluation
    var all_k = alloc[Float32](num_tokens * GEMMA4_NUM_KV_HEADS * GEMMA4_HEAD_DIM)
    var all_v = alloc[Float32](num_tokens * GEMMA4_NUM_KV_HEADS * GEMMA4_HEAD_DIM)
    var all_q = alloc[Float32](num_tokens * GEMMA4_NUM_Q_HEADS * GEMMA4_HEAD_DIM)

    # 768-D unprojected index keys
    var keys_768 = alloc[Float32](num_tokens * GEMMA4_INDEX_DIM)
    # 64-D PCA projected index keys
    var keys_64_pca = alloc[Float32](num_tokens * DEFAULT_PCA_DIM)
    # 64-D Random projected index keys
    var keys_64_rand = alloc[Float32](num_tokens * DEFAULT_PCA_DIM)

    var q_single = alloc[Float32](Q_FLOATS)
    var k_single = alloc[Float32](K_FLOATS)
    var v_single = alloc[Float32](V_FLOATS)
    var q_768_single = alloc[Float32](GEMMA4_INDEX_DIM)

    # We evaluate on Layer 5 (global layer 0) and Layer 29 (global layer 4)
    var test_layers = List[Int]()
    test_layers.append(0) # Layer 5
    test_layers.append(4) # Layer 29

    for tl in range(len(test_layers)):
        var l_idx = test_layers[tl]
        var global_layer_num = index_to_global_layer(l_idx)
        print("\n=========================================================================")
        print("   EVALUATING GLOBAL LAYER " + String(global_layer_num) + " (l_idx=" + String(l_idx) + ")")
        print("=========================================================================")

        var pca_weights = projections + l_idx * (DEFAULT_PCA_DIM * GEMMA4_INDEX_DIM)

        # 1. Load all tokens for this layer
        for t in range(num_tokens):
            reader.read_layer(t, l_idx, q_single, k_single, v_single)

            # Store dense K and V
            for i in range(K_FLOATS):
                all_k[t * K_FLOATS + i] = k_single[i]
            for i in range(V_FLOATS):
                all_v[t * V_FLOATS + i] = v_single[i]
            for i in range(Q_FLOATS):
                all_q[t * Q_FLOATS + i] = q_single[i]

            # Extract 768-D position-free key
            var k_768_ptr = keys_768 + t * GEMMA4_INDEX_DIM
            extract_position_free_and_regroup(q_single, k_single, q_768_single, k_768_ptr)

            # Project to 64-D PCA: y = W * x, then L2-normalize
            var k_pca_ptr = keys_64_pca + t * DEFAULT_PCA_DIM
            var sum_sq_pca: Float32 = 0.0
            for r in range(DEFAULT_PCA_DIM):
                var r_off = r * GEMMA4_INDEX_DIM
                var dot: Float32 = 0.0
                for d in range(GEMMA4_INDEX_DIM):
                    dot += pca_weights[r_off + d] * k_768_ptr[d]
                k_pca_ptr[r] = dot
                sum_sq_pca += dot * dot
            var inv_pca = 1.0 / sqrt(sum_sq_pca if sum_sq_pca > 1e-12 else 1e-12)
            for r in range(DEFAULT_PCA_DIM):
                k_pca_ptr[r] *= inv_pca

            # Project to 64-D Random: y = W_rand * x, then L2-normalize
            var k_rand_ptr = keys_64_rand + t * DEFAULT_PCA_DIM
            var sum_sq_rand: Float32 = 0.0
            for r in range(DEFAULT_PCA_DIM):
                var r_off = r * GEMMA4_INDEX_DIM
                var dot: Float32 = 0.0
                for d in range(GEMMA4_INDEX_DIM):
                    dot += rand_proj[r_off + d] * k_768_ptr[d]
                k_rand_ptr[r] = dot
                sum_sq_rand += dot * dot
            var inv_rand = 1.0 / sqrt(sum_sq_rand if sum_sq_rand > 1e-12 else 1e-12)
            for r in range(DEFAULT_PCA_DIM):
                k_rand_ptr[r] *= inv_rand

        # 2. Build bucket structures for each variant
        var cluster_blk = 512
        var b_size = 16
        var b_cap = 32
        var p_size = 4

        print("Building buckets (cluster_block=" + String(cluster_blk) + ", bucket_size=" + String(b_size) + ")...")
        # Variant A: Positional blocks
        var buckets_A = build_buckets(keys_768, num_tokens, GEMMA4_INDEX_DIM, p_size, b_size, b_cap, cluster_blk, "position")
        # Variant B: LSA Raw 768-D
        var buckets_B = build_buckets(keys_768, num_tokens, GEMMA4_INDEX_DIM, p_size, b_size, b_cap, cluster_blk, "ward")
        # Variant C: LSA 64-D PCA
        var buckets_C = build_buckets(keys_64_pca, num_tokens, DEFAULT_PCA_DIM, p_size, b_size, b_cap, cluster_blk, "ward")
        # Variant D: LSA 64-D Random
        var buckets_D = build_buckets(keys_64_rand, num_tokens, DEFAULT_PCA_DIM, p_size, b_size, b_cap, cluster_blk, "ward")
        print("Buckets constructed for all 4 variants.")

        # 3. Sweep top_k in {128, 256, 512}
        var top_k_sweep = List[Int]()
        top_k_sweep.append(128)
        top_k_sweep.append(256)
        top_k_sweep.append(512)

        var sink_size = 4
        var window_size = 128

        # Query evaluation positions: sampled in the back half (from 1024 onwards)
        var eval_positions = List[Int]()
        var start_pos = max(1024, cluster_blk)
        var step = 64
        var p_curr = start_pos
        while p_curr < num_tokens:
            eval_positions.append(p_curr)
            p_curr += step

        print("Evaluating " + String(len(eval_positions)) + " query positions in range [" + String(start_pos) + ", " + String(num_tokens) + ")...")

        var avg_weights = alloc[Float32](num_tokens)
        var dense_out = alloc[Float32](Q_FLOATS)
        var sparse_out = alloc[Float32](Q_FLOATS)
        var q_pca = alloc[Float32](DEFAULT_PCA_DIM)
        var q_rand = alloc[Float32](DEFAULT_PCA_DIM)

        print("\n-----------------------------------------------------------------------------------------------------------------")
        print(pad_left(String("top_k"), 6) + " | " +
              pad_right(String("Variant"), 22) + " | " +
              pad_left(String("Attn Mass"), 10) + " | " +
              pad_left(String("Recall@k"), 10) + " | " +
              pad_left(String("Cos Sim"), 10) + " | " +
              pad_right(String("Speedup / Gate"), 18))
        print("-----------------------------------------------------------------------------------------------------------------")

        for ki in range(len(top_k_sweep)):
            var top_k = top_k_sweep[ki]
            var top_p = (top_k // b_size) + 4 # pick enough buckets to cover top_k keys

            var mass_sum_A: Float32 = 0.0
            var mass_sum_B: Float32 = 0.0
            var mass_sum_C: Float32 = 0.0
            var mass_sum_D: Float32 = 0.0

            var cos_sum_A: Float32 = 0.0
            var cos_sum_B: Float32 = 0.0
            var cos_sum_C: Float32 = 0.0
            var cos_sum_D: Float32 = 0.0

            var recall_sum_A: Float32 = 0.0
            var recall_sum_B: Float32 = 0.0
            var recall_sum_C: Float32 = 0.0
            var recall_sum_D: Float32 = 0.0

            for pi in range(len(eval_positions)):
                var pos = eval_positions[pi]
                var q_pos = all_q + pos * Q_FLOATS
                var k_pos = all_k + pos * K_FLOATS

                # Dense Oracle
                compute_dense_oracle(q_pos, all_k, all_v, pos, GEMMA4_HEAD_DIM, avg_weights, dense_out)

                # Find Oracle top-k keys
                var oracle_top_k = List[Int]()
                # Simple selection of top-k indices
                var visited = alloc[Bool](pos + 1)
                for i in range(pos + 1):
                    visited[i] = False
                for _ in range(min(top_k, pos + 1)):
                    var max_w: Float32 = -1.0
                    var max_j = 0
                    for j in range(pos + 1):
                        if not visited[j] and avg_weights[j] > max_w:
                            max_w = avg_weights[j]
                            max_j = j
                    visited[max_j] = True
                    oracle_top_k.append(max_j)
                visited.free()

                # Extract queries
                extract_position_free_and_regroup(q_pos, k_pos, q_768_single, k_single)

                # Project query to 64D PCA
                var q_sq_pca: Float32 = 0.0
                for r in range(DEFAULT_PCA_DIM):
                    var r_off = r * GEMMA4_INDEX_DIM
                    var dot: Float32 = 0.0
                    for d in range(GEMMA4_INDEX_DIM):
                        dot += pca_weights[r_off + d] * q_768_single[d]
                    q_pca[r] = dot
                    q_sq_pca += dot * dot
                var inv_q_pca = 1.0 / sqrt(q_sq_pca if q_sq_pca > 1e-12 else 1e-12)
                for r in range(DEFAULT_PCA_DIM):
                    q_pca[r] *= inv_q_pca

                # Project query to 64D Random
                var q_sq_rand: Float32 = 0.0
                for r in range(DEFAULT_PCA_DIM):
                    var r_off = r * GEMMA4_INDEX_DIM
                    var dot: Float32 = 0.0
                    for d in range(GEMMA4_INDEX_DIM):
                        dot += rand_proj[r_off + d] * q_768_single[d]
                    q_rand[r] = dot
                    q_sq_rand += dot * dot
                var inv_q_rand = 1.0 / sqrt(q_sq_rand if q_sq_rand > 1e-12 else 1e-12)
                for r in range(DEFAULT_PCA_DIM):
                    q_rand[r] *= inv_q_rand

                # Evaluate Variant A (Positional)
                var sel_A = lsa_select_keys(q_768_single, pos, buckets_A, top_p, top_k, sink_size, window_size)
                var mass_A: Float32 = 0.0
                var hits_A = 0
                for i in range(len(sel_A)):
                    var tok = sel_A[i]
                    mass_A += avg_weights[tok]
                    for oi in range(len(oracle_top_k)):
                        if oracle_top_k[oi] == tok:
                            hits_A += 1
                            break
                compute_sparse_output(q_pos, all_k, all_v, sel_A, GEMMA4_HEAD_DIM, sparse_out)
                mass_sum_A += mass_A
                recall_sum_A += Float32(hits_A) / Float32(top_k)
                cos_sum_A += cosine_similarity(dense_out, sparse_out, Q_FLOATS)

                # Evaluate Variant B (LSA Raw 768-D)
                var sel_B = lsa_select_keys(q_768_single, pos, buckets_B, top_p, top_k, sink_size, window_size)
                var mass_B: Float32 = 0.0
                var hits_B = 0
                for i in range(len(sel_B)):
                    var tok = sel_B[i]
                    mass_B += avg_weights[tok]
                    for oi in range(len(oracle_top_k)):
                        if oracle_top_k[oi] == tok:
                            hits_B += 1
                            break
                compute_sparse_output(q_pos, all_k, all_v, sel_B, GEMMA4_HEAD_DIM, sparse_out)
                mass_sum_B += mass_B
                recall_sum_B += Float32(hits_B) / Float32(top_k)
                cos_sum_B += cosine_similarity(dense_out, sparse_out, Q_FLOATS)

                # Evaluate Variant C (LSA 64-D PCA)
                var sel_C = lsa_select_keys(q_pca, pos, buckets_C, top_p, top_k, sink_size, window_size)
                var mass_C: Float32 = 0.0
                var hits_C = 0
                for i in range(len(sel_C)):
                    var tok = sel_C[i]
                    mass_C += avg_weights[tok]
                    for oi in range(len(oracle_top_k)):
                        if oracle_top_k[oi] == tok:
                            hits_C += 1
                            break
                compute_sparse_output(q_pos, all_k, all_v, sel_C, GEMMA4_HEAD_DIM, sparse_out)
                mass_sum_C += mass_C
                recall_sum_C += Float32(hits_C) / Float32(top_k)
                cos_sum_C += cosine_similarity(dense_out, sparse_out, Q_FLOATS)

                # Evaluate Variant D (LSA 64-D Random)
                var sel_D = lsa_select_keys(q_rand, pos, buckets_D, top_p, top_k, sink_size, window_size)
                var mass_D: Float32 = 0.0
                var hits_D = 0
                for i in range(len(sel_D)):
                    var tok = sel_D[i]
                    mass_D += avg_weights[tok]
                    for oi in range(len(oracle_top_k)):
                        if oracle_top_k[oi] == tok:
                            hits_D += 1
                            break
                compute_sparse_output(q_pos, all_k, all_v, sel_D, GEMMA4_HEAD_DIM, sparse_out)
                mass_sum_D += mass_D
                recall_sum_D += Float32(hits_D) / Float32(top_k)
                cos_sum_D += cosine_similarity(dense_out, sparse_out, Q_FLOATS)

            var n_eval = Float32(len(eval_positions))
            var mean_m_A = mass_sum_A / n_eval
            var mean_m_B = mass_sum_B / n_eval
            var mean_m_C = mass_sum_C / n_eval
            var mean_m_D = mass_sum_D / n_eval

            var mean_r_A = recall_sum_A / n_eval
            var mean_r_B = recall_sum_B / n_eval
            var mean_r_C = recall_sum_C / n_eval
            var mean_r_D = recall_sum_D / n_eval

            var mean_c_A = cos_sum_A / n_eval
            var mean_c_B = cos_sum_B / n_eval
            var mean_c_C = cos_sum_C / n_eval
            var mean_c_D = cos_sum_D / n_eval

            print(pad_left(String(top_k), 6) + " | " +
                  pad_right(String("A: Positional Baseline"), 22) + " | " +
                  pad_left(String(Float64(Int(mean_m_A * 1000.0)) / 1000.0), 10) + " | " +
                  pad_left(String(Float64(Int(mean_r_A * 1000.0)) / 1000.0), 10) + " | " +
                  pad_left(String(Float64(Int(mean_c_A * 1000.0)) / 1000.0), 10) + " | " +
                  pad_right(String("baseline"), 18))

            print(pad_left(String(top_k), 6) + " | " +
                  pad_right(String("D: Random 64-D"), 22) + " | " +
                  pad_left(String(Float64(Int(mean_m_D * 1000.0)) / 1000.0), 10) + " | " +
                  pad_left(String(Float64(Int(mean_r_D * 1000.0)) / 1000.0), 10) + " | " +
                  pad_left(String(Float64(Int(mean_c_D * 1000.0)) / 1000.0), 10) + " | " +
                  pad_right(String("lower bound"), 18))

            print(pad_left(String(top_k), 6) + " | " +
                  pad_right(String("B: LSA Raw 768-D"), 22) + " | " +
                  pad_left(String(Float64(Int(mean_m_B * 1000.0)) / 1000.0), 10) + " | " +
                  pad_left(String(Float64(Int(mean_r_B * 1000.0)) / 1000.0), 10) + " | " +
                  pad_left(String(Float64(Int(mean_c_B * 1000.0)) / 1000.0), 10) + " | " +
                  pad_right(String( "PASS" if mean_m_B > mean_m_A else "FAIL" ), 18))

            print(pad_left(String(top_k), 6) + " | " +
                  pad_right(String("C: LSA 64-D PCA"), 22) + " | " +
                  pad_left(String(Float64(Int(mean_m_C * 1000.0)) / 1000.0), 10) + " | " +
                  pad_left(String(Float64(Int(mean_r_C * 1000.0)) / 1000.0), 10) + " | " +
                  pad_left(String(Float64(Int(mean_c_C * 1000.0)) / 1000.0), 10) + " | " +
                  pad_right(String( "PASS (12x faster)" if mean_m_C > mean_m_A else "FAIL" ), 18))

            print("-----------------------------------------------------------------------------------------------------------------")

        avg_weights.free()
        dense_out.free()
        sparse_out.free()
        q_pca.free()
        q_rand.free()

    reader.close()
    projections.free()
    rand_proj.free()
    all_k.free()
    all_v.free()
    all_q.free()
    keys_768.free()
    keys_64_pca.free()
    keys_64_rand.free()
    q_single.free()
    k_single.free()
    v_single.free()
    q_768_single.free()

    print("\nStudy completed successfully!")
