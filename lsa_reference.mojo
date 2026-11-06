# =============================================================================
# Latent Sparse Attention (LSA) Reference Implementation in Mojo
#
# Ported from Prefrontal AI's reference implementation (Apache-2.0 License).
# Original implementation:
#   https://github.com/Prefrontal-AI/latent-sparse-attention
#   src/lsa/core/{index_space.py, ward.py, reference.py}
#
# Copyright (c) 2026 Prefrontal AI
# Copyright (c) 2026 llm-mojo contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

from std.math import sqrt
from std.memory import alloc

# -----------------------------------------------------------------------------
# Gemma-4 Architecture Constants for LSA Index Space
# -----------------------------------------------------------------------------
# Global layers (5, 11, 17, 23, 29):
#   16 Query Heads, 2 KV Heads (GQA group ratio = 8).
#   Head dimension = 512.
#   RoPE rotates dims [0, 64) paired with [256, 320) (128 rotary dims).
#   Position-free dims are [64, 256) (192 dims) and [320, 512) (192 dims).
#   Position-free per head = 192 + 192 = 384 dims.
#   Regrouped index vector = 2 KV groups * 384 = 768 dims.
# -----------------------------------------------------------------------------

comptime GEMMA4_NUM_Q_HEADS: Int = 16
comptime GEMMA4_NUM_KV_HEADS: Int = 2
comptime GEMMA4_HEAD_DIM: Int = 512
comptime GEMMA4_POS_FREE_DIMS: Int = 384
comptime GEMMA4_INDEX_DIM: Int = GEMMA4_NUM_KV_HEADS * GEMMA4_POS_FREE_DIMS # 768

comptime DEFAULT_POOL_SIZE: Int = 4
comptime DEFAULT_CLUSTER_BLOCK_SIZE: Int = 512
comptime DEFAULT_BUCKET_SIZE: Int = 16
comptime DEFAULT_BUCKET_CAP: Int = 32


# -----------------------------------------------------------------------------
# 1. Position-Free Projection & Regrouping
# -----------------------------------------------------------------------------

def extract_position_free_and_regroup(
    q: UnsafePointer[Float32, _],
    k: UnsafePointer[Float32, _],
    q_index: UnsafePointer[Float32, MutUntrackedOrigin],
    k_index: UnsafePointer[Float32, MutUntrackedOrigin],
):
    """
    Extracts the unrotated, position-free sub-term of Gemma-4 global layer attention
    and collapses query heads into GQA KV-groups.

    Query regrouping:
      Group 0: Query heads 0..7 (summed across 8 heads)
      Group 1: Query heads 8..15 (summed across 8 heads)
      Concatenated -> 768 dims, then L2-normalized.

    Key regrouping:
      KV head 0 (384 position-free dims)
      KV head 1 (384 position-free dims)
      Concatenated -> 768 dims, then L2-normalized.
    """
    # 1. Process queries
    for g in range(GEMMA4_NUM_KV_HEADS):
        var g_out_offset = g * GEMMA4_POS_FREE_DIMS
        var head_start = g * 8
        var head_end = head_start + 8

        # First unrotated chunk: dims [64, 256) (192 floats)
        for d in range(192):
            var head_dim_idx = 64 + d
            var sum_val: Float32 = 0.0
            for h in range(head_start, head_end):
                sum_val += q[h * GEMMA4_HEAD_DIM + head_dim_idx]
            q_index[g_out_offset + d] = sum_val

        # Second unrotated chunk: dims [320, 512) (192 floats)
        for d in range(192):
            var head_dim_idx = 320 + d
            var sum_val: Float32 = 0.0
            for h in range(head_start, head_end):
                sum_val += q[h * GEMMA4_HEAD_DIM + head_dim_idx]
            q_index[g_out_offset + 192 + d] = sum_val

    # 2. Process keys
    for g in range(GEMMA4_NUM_KV_HEADS):
        var g_out_offset = g * GEMMA4_POS_FREE_DIMS
        var h_base = g * GEMMA4_HEAD_DIM

        # First unrotated chunk: dims [64, 256)
        for d in range(192):
            k_index[g_out_offset + d] = k[h_base + 64 + d]

        # Second unrotated chunk: dims [320, 512)
        for d in range(192):
            k_index[g_out_offset + 192 + d] = k[h_base + 320 + d]

    # 3. L2 normalize rows (zero vectors remain zero)
    var sum_sq_q: Float32 = 0.0
    var sum_sq_k: Float32 = 0.0
    for i in range(GEMMA4_INDEX_DIM):
        sum_sq_q += q_index[i] * q_index[i]
        sum_sq_k += k_index[i] * k_index[i]

    if sum_sq_q > 1e-12:
        var inv_norm_q = 1.0 / sqrt(sum_sq_q)
        for i in range(GEMMA4_INDEX_DIM):
            q_index[i] *= inv_norm_q

    if sum_sq_k > 1e-12:
        var inv_norm_k = 1.0 / sqrt(sum_sq_k)
        for i in range(GEMMA4_INDEX_DIM):
            k_index[i] *= inv_norm_k


# -----------------------------------------------------------------------------
# 2. Key Pooling
# -----------------------------------------------------------------------------

def pool_keys(
    keys: UnsafePointer[Float32, _],
    num_keys: Int,
    index_dim: Int,
    pool_size: Int,
    pools_out: UnsafePointer[Float32, MutUntrackedOrigin],
):
    """
    Sum each run of `pool_size` consecutive index keys into a pool key.
    Output holds (num_keys // pool_size) pool vectors of dimension `index_dim`.
    """
    var num_pools = num_keys // pool_size
    for p in range(num_pools):
        var p_offset = p * index_dim
        for d in range(index_dim):
            var s: Float32 = 0.0
            for j in range(pool_size):
                s += keys[(p * pool_size + j) * index_dim + d]
            pools_out[p_offset + d] = s


# -----------------------------------------------------------------------------
# 3. Ward Clustering
# -----------------------------------------------------------------------------

struct MergeCandidate(ImplicitlyCopyable, Copyable, Movable):
    var cost: Float32
    var i: Int
    var j: Int

    def __init__(out self, cost: Float32, i: Int, j: Int):
        self.cost = cost
        self.i = i
        self.j = j

    def __lt__(self, other: MergeCandidate) -> Bool:
        if self.cost != other.cost:
            return self.cost < other.cost
        if self.i != other.i:
            return self.i < other.i
        return self.j < other.j


def sort_merge_candidates(mut list: List[MergeCandidate]):
    for i in range(1, len(list)):
        var key = list[i]
        var j = i - 1
        while j >= 0 and key < list[j]:
            list[j + 1] = list[j]
            j -= 1
        list[j + 1] = key


def ward_merge(
    pools: UnsafePointer[Float32, _],
    num_pools: Int,
    index_dim: Int,
    pool_size: Int,
    bucket_count: Int,
    bucket_cap: Int,
    labels_out: UnsafePointer[Int, MutUntrackedOrigin],
    max_rounds: Int = 256,
) raises:
    """
    Merges `num_pools` down to `bucket_count` buckets using mutual-cheapest Ward merges.
    Writes the resulting bucket index [0, bucket_count) for each pool into `labels_out`.
    """
    # Working state
    var key_sums = alloc[Float32](num_pools * index_dim)
    for i in range(num_pools * index_dim):
        key_sums[i] = pools[i]

    var bucket_sizes = alloc[Int](num_pools)
    var alive = alloc[Bool](num_pools)
    var parent = alloc[Int](num_pools)

    for i in range(num_pools):
        bucket_sizes[i] = pool_size
        alive[i] = True
        parent[i] = i

    var active_count = num_pools
    var gram = alloc[Float32](num_pools * num_pools)
    var alignments = alloc[Float32](num_pools)
    var best_partner = alloc[Int](num_pools)
    var best_cost = alloc[Float32](num_pools)

    for _ in range(max_rounds):
        var merges_left = active_count - bucket_count
        if merges_left <= 0:
            break

        # 1. Compute Gram matrix G[i, j] and alignments
        for i in range(num_pools):
            if not alive[i]:
                continue
            for j in range(i, num_pools):
                if not alive[j]:
                    continue
                var dot: Float32 = 0.0
                var i_off = i * index_dim
                var j_off = j * index_dim
                for d in range(index_dim):
                    dot += key_sums[i_off + d] * key_sums[j_off + d]
                gram[i * num_pools + j] = dot
                gram[j * num_pools + i] = dot

        for i in range(num_pools):
            if alive[i]:
                var g_ii = gram[i * num_pools + i]
                alignments[i] = sqrt(g_ii if g_ii > 0.0 else 0.0)

        # 2. For each alive pool, find its cheapest allowed merge partner
        for i in range(num_pools):
            best_partner[i] = -1
            best_cost[i] = Float32(1e30)
            if not alive[i]:
                continue

            var min_c: Float32 = Float32(1e30)
            var min_j = -1
            for j in range(num_pools):
                if j == i or not alive[j]:
                    continue
                if bucket_sizes[i] + bucket_sizes[j] > bucket_cap:
                    continue

                var g_ii = gram[i * num_pools + i]
                var g_jj = gram[j * num_pools + j]
                var g_ij = gram[i * num_pools + j]
                var merged_sq = g_ii + g_jj + 2.0 * g_ij
                var merged_align = sqrt(merged_sq if merged_sq > 0.0 else 0.0)
                var cost = alignments[i] + alignments[j] - merged_align

                if cost < min_c:
                    min_c = cost
                    min_j = j
                elif cost == min_c and (min_j < 0 or j < min_j):
                    min_j = j

            best_partner[i] = min_j
            best_cost[i] = min_c

        # 3. Find mutual-cheapest pairs
        var candidates = List[MergeCandidate]()
        for i in range(num_pools):
            if not alive[i] or best_partner[i] < 0:
                continue
            var j = best_partner[i]
            if j > i and alive[j] and best_partner[j] == i:
                candidates.append(MergeCandidate(best_cost[i], i, j))

        if len(candidates) == 0:
            break

        # 4. Sort mutual candidates by cost ascending (lower index on tie)
        sort_merge_candidates(candidates)

        # 5. Merge pairs up to merges_left
        var merges_to_do = min(len(candidates), merges_left)
        var merges_done = 0
        for idx in range(len(candidates)):
            if merges_done >= merges_to_do:
                break
            var pair = candidates[idx]
            var u = pair.i
            var v = pair.j
            # Check both are still alive (in case of earlier merges)
            if alive[u] and alive[v] and (bucket_sizes[u] + bucket_sizes[v] <= bucket_cap):
                # Merge v into u
                var u_off = u * index_dim
                var v_off = v * index_dim
                for d in range(index_dim):
                    key_sums[u_off + d] += key_sums[v_off + d]
                bucket_sizes[u] += bucket_sizes[v]
                alive[v] = False
                parent[v] = u
                active_count -= 1
                merges_done += 1

        if merges_done == 0:
            break

    # Resolve parent pointers with path compression
    for i in range(num_pools):
        var curr = i
        while not alive[curr] and parent[curr] != curr:
            curr = parent[curr]
        parent[i] = curr

    # Number surviving roots densely from 0 to bucket_count - 1 in pool index order
    var root_to_label = alloc[Int](num_pools)
    for i in range(num_pools):
        root_to_label[i] = -1

    var current_label = 0
    for i in range(num_pools):
        if alive[i]:
            root_to_label[i] = current_label
            current_label += 1

    for i in range(num_pools):
        var root = parent[i]
        labels_out[i] = root_to_label[root]

    # Clean up
    key_sums.free()
    bucket_sizes.free()
    alive.free()
    parent.free()
    gram.free()
    alignments.free()
    best_partner.free()
    best_cost.free()
    root_to_label.free()


# -----------------------------------------------------------------------------
# 4. Bucket Construction (Ward or Positional)
# -----------------------------------------------------------------------------

struct Buckets:
    var bucket_centroids: List[Float32] # [total_buckets, index_dim]
    var bucket_members: List[Int]       # [total_buckets, bucket_cap], -1 padded
    var bucket_sizes: List[Int]         # [total_buckets]
    var block_bucket_count: Int
    var cluster_block_size: Int
    var index_dim: Int
    var bucket_cap: Int
    var total_buckets: Int

    def __init__(
        out self,
        var centroids: List[Float32],
        var members: List[Int],
        var sizes: List[Int],
        block_bucket_count: Int,
        cluster_block_size: Int,
        index_dim: Int,
        bucket_cap: Int,
        total_buckets: Int,
    ):
        self.bucket_centroids = centroids^
        self.bucket_members = members^
        self.bucket_sizes = sizes^
        self.block_bucket_count = block_bucket_count
        self.cluster_block_size = cluster_block_size
        self.index_dim = index_dim
        self.bucket_cap = bucket_cap
        self.total_buckets = total_buckets


def build_buckets(
    index_keys: UnsafePointer[Float32, _],
    num_keys: Int,
    index_dim: Int,
    pool_size: Int = DEFAULT_POOL_SIZE,
    bucket_size: Int = DEFAULT_BUCKET_SIZE,
    bucket_cap: Int = DEFAULT_BUCKET_CAP,
    cluster_block_size: Int = DEFAULT_CLUSTER_BLOCK_SIZE,
    cluster_algorithm: String = "ward",
) raises -> Buckets:
    """
    Cluster closed cluster blocks of index keys into buckets.
    """
    var block_count = num_keys // cluster_block_size
    var block_bucket_count = cluster_block_size // bucket_size
    var total_buckets = block_count * block_bucket_count
    var pools_per_block = cluster_block_size // pool_size

    var all_centroids = List[Float32]()
    all_centroids.reserve(total_buckets * index_dim)
    for _ in range(total_buckets * index_dim):
        all_centroids.append(0.0)

    var all_members = List[Int]()
    all_members.reserve(total_buckets * bucket_cap)
    for _ in range(total_buckets * bucket_cap):
        all_members.append(-1)

    var all_sizes = List[Int]()
    all_sizes.reserve(total_buckets)
    for _ in range(total_buckets):
        all_sizes.append(0)

    var pool_buf = alloc[Float32](pools_per_block * index_dim)
    var pool_labels = alloc[Int](pools_per_block)

    for b in range(block_count):
        var block_keys = index_keys + (b * cluster_block_size) * index_dim
        pool_keys(block_keys, cluster_block_size, index_dim, pool_size, pool_buf)

        if cluster_algorithm == "position":
            var pools_per_bucket = bucket_size // pool_size
            for p in range(pools_per_block):
                pool_labels[p] = p // pools_per_bucket
        elif cluster_algorithm == "ward":
            ward_merge(
                pool_buf,
                pools_per_block,
                index_dim,
                pool_size,
                block_bucket_count,
                bucket_cap,
                pool_labels,
            )
        else:
            raise Error("Unknown cluster algorithm: " + cluster_algorithm)

        var b_bucket_base = b * block_bucket_count

        # Assign keys to buckets and compute key sums
        for p in range(pools_per_block):
            var bucket_id = b_bucket_base + pool_labels[p]
            for j in range(pool_size):
                var local_pos = p * pool_size + j
                var global_pos = b * cluster_block_size + local_pos
                var k_ptr = block_keys + local_pos * index_dim

                # Add to centroid sum
                var c_off = bucket_id * index_dim
                for d in range(index_dim):
                    all_centroids[c_off + d] += k_ptr[d]

                # Append to members
                var cur_size = all_sizes[bucket_id]
                if cur_size < bucket_cap:
                    all_members[bucket_id * bucket_cap + cur_size] = global_pos
                    all_sizes[bucket_id] = cur_size + 1

    pool_buf.free()
    pool_labels.free()

    # L2 normalize centroids
    for bucket in range(total_buckets):
        var c_off = bucket * index_dim
        var sum_sq: Float32 = 0.0
        for d in range(index_dim):
            sum_sq += all_centroids[c_off + d] * all_centroids[c_off + d]
        if sum_sq > 1e-12:
            var inv_norm = 1.0 / sqrt(sum_sq)
            for d in range(index_dim):
                all_centroids[c_off + d] *= inv_norm

    return Buckets(
        all_centroids^,
        all_members^,
        all_sizes^,
        block_bucket_count,
        cluster_block_size,
        index_dim,
        bucket_cap,
        total_buckets,
    )


# -----------------------------------------------------------------------------
# 5. Bucket Scoring & Expansion
# -----------------------------------------------------------------------------

struct ScoredBucket(ImplicitlyCopyable, Copyable, Movable):
    var score: Float32
    var bucket_id: Int

    def __init__(out self, score: Float32, bucket_id: Int):
        self.score = score
        self.bucket_id = bucket_id

    def __lt__(self, other: ScoredBucket) -> Bool:
        if self.score != other.score:
            # Descending order by score: higher score is smaller in sort order
            return self.score > other.score
        # On tie: lower bucket ID first
        return self.bucket_id < other.bucket_id


def sort_scored_buckets(mut list: List[ScoredBucket]):
    for i in range(1, len(list)):
        var key = list[i]
        var j = i - 1
        while j >= 0 and key < list[j]:
            list[j + 1] = list[j]
            j -= 1
        list[j + 1] = key


def expand_buckets(
    picked_buckets: List[Int],
    position: Int,
    buckets: Buckets,
    top_k: Int,
    sink_size: Int,
    window_size: Int,
) -> List[Int]:
    """
    Expands picked buckets to token indices, followed by window and sinks.
    Exact spec match to reference.py::expand_buckets.
    """
    var window_start = max(0, position + 1 - window_size)
    var selected = List[Int]()

    # 1. Add members from picked buckets (best bucket first)
    for b_idx in range(len(picked_buckets)):
        var b = picked_buckets[b_idx]
        var b_size = buckets.bucket_sizes[b]
        for m_idx in range(b_size):
            var token_id = buckets.bucket_members[b * buckets.bucket_cap + m_idx]
            if token_id >= sink_size and token_id < window_start:
                selected.append(token_id)
                if len(selected) >= top_k:
                    break
        if len(selected) >= top_k:
            break

    # 2. Append window: [window_start, position] inclusive
    for t in range(window_start, position + 1):
        selected.append(t)

    # 3. Append sinks: [0, min(sink_size, window_start, position + 1))
    var sink_limit = min(sink_size, window_start)
    sink_limit = min(sink_limit, position + 1)
    for t in range(sink_limit):
        selected.append(t)

    return selected^


def lsa_select_keys(
    query_index: UnsafePointer[Float32, _],
    position: Int,
    buckets: Buckets,
    top_p: Int,
    top_k: Int,
    sink_size: Int,
    window_size: Int,
) -> List[Int]:
    """
    Scores visible buckets and expands to selected token indices for a single query.
    Exact spec match to reference.py::_read_row.
    """
    # Only buckets from closed cluster blocks strictly prior to query's current block
    var visible_buckets = min(
        (position // buckets.cluster_block_size) * buckets.block_bucket_count,
        buckets.total_buckets,
    )

    if visible_buckets == 0:
        # Fallback: only window and sinks
        var empty_picked = List[Int]()
        return expand_buckets(empty_picked, position, buckets, top_k, sink_size, window_size)

    # Score visible buckets via dot product
    var scored = List[ScoredBucket]()
    scored.reserve(visible_buckets)
    for b in range(visible_buckets):
        var c_off = b * buckets.index_dim
        var dot: Float32 = 0.0
        for d in range(buckets.index_dim):
            dot += query_index[d] * buckets.bucket_centroids[c_off + d]
        scored.append(ScoredBucket(dot, b))

    # Stable sort by score descending, tie to lower bucket index
    sort_scored_buckets(scored)

    var p = min(top_p, visible_buckets)
    var picked = List[Int]()
    picked.reserve(p)
    for i in range(p):
        picked.append(scored[i].bucket_id)

    return expand_buckets(picked, position, buckets, top_k, sink_size, window_size)
