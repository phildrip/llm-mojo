# =============================================================================
# Latent Sparse Attention (LSA) Fast SIMD Engine Kernel
#
# High-performance SIMD implementation for Gemma-4 26B MoE global layers:
#   - SIMD position-free extraction, GQA regrouping, and 64D projection
#   - Pre-allocated zero-heap LSALayerIndex with centroid and member cache
#   - SIMD Gram matrix and mutual-cheapest Ward clustering on block close
#   - 64-bit packed integer compare and bounded min-heap for top-p selection
#   - Sparse paged attention gather over selected tokens J_i
# =============================================================================

from std.math import sqrt, exp
from std.memory import alloc
from std.algorithm import sort
from attention import ring_row

comptime GEMMA4_NUM_Q_HEADS: Int = 16
comptime GEMMA4_NUM_KV_HEADS: Int = 2
comptime GEMMA4_HEAD_DIM: Int = 512
comptime GEMMA4_POS_FREE_DIMS: Int = 384
comptime GEMMA4_INDEX_DIM: Int = 768
comptime LSA_DIM: Int = 64

comptime LSA_POOL_SIZE: Int = 4
comptime LSA_CLUSTER_BLOCK_SIZE: Int = 512
comptime LSA_BUCKET_SIZE: Int = 16
comptime LSA_BUCKET_CAP: Int = 32
comptime LSA_BLOCK_BUCKET_COUNT: Int = LSA_CLUSTER_BLOCK_SIZE // LSA_BUCKET_SIZE # 32
comptime LSA_POOLS_PER_BLOCK: Int = LSA_CLUSTER_BLOCK_SIZE // LSA_POOL_SIZE       # 128

# -----------------------------------------------------------------------------
# 1. Fast SIMD Position-Free Projection & Normalization
# -----------------------------------------------------------------------------

@always_inline
def lsa_index_query(
    q: UnsafePointer[Float32, _],
    proj_weights: UnsafePointer[Float32, _],
    q_idx_out: UnsafePointer[Float32, MutUntrackedOrigin],
):
    """
    Extracts Gemma-4 position-free query dimensions ([64:256) and [320:512)),
    sums across 8 query heads in each of the 2 KV groups into a 768-d vector,
    projects through [64, 768] projection weights using SIMD[Float32, 8],
    and L2-normalizes to a 64-d unit vector.
    """
    var x_buf = alloc[Float32](GEMMA4_INDEX_DIM)

    # 1. Vectorized regrouping: Group 0 (heads 0..7) and Group 1 (heads 8..15)
    for g in range(GEMMA4_NUM_KV_HEADS):
        var g_out_offset = g * GEMMA4_POS_FREE_DIMS
        var head_start = g * 8

        # First chunk: dims [64, 256) (192 floats = 24 SIMD vectors of width 8)
        for d in range(0, 192, 8):
            var s = SIMD[DType.float32, 8](0.0)
            for h in range(head_start, head_start + 8):
                s += q.load[8](h * GEMMA4_HEAD_DIM + 64 + d)
            x_buf.store[8](g_out_offset + d, s)

        # Second chunk: dims [320, 512) (192 floats = 24 SIMD vectors of width 8)
        for d in range(0, 192, 8):
            var s = SIMD[DType.float32, 8](0.0)
            for h in range(head_start, head_start + 8):
                s += q.load[8](h * GEMMA4_HEAD_DIM + 320 + d)
            x_buf.store[8](g_out_offset + 192 + d, s)

    # 2. Vectorized projection: y = W * x ([64, 768] * [768] -> [64])
    var sum_sq: Float32 = 0.0
    for r in range(LSA_DIM):
        var w_row = proj_weights + r * GEMMA4_INDEX_DIM
        var acc = SIMD[DType.float32, 8](0.0)
        for c in range(0, GEMMA4_INDEX_DIM, 8):
            acc += w_row.load[8](c) * x_buf.load[8](c)
        var dot = acc.reduce_add()
        q_idx_out[r] = dot
        sum_sq += dot * dot

    # 3. L2 Normalization
    var inv_norm = 1.0 / sqrt(sum_sq if sum_sq > 1e-12 else 1e-12)
    var inv_norm_vec = SIMD[DType.float32, 8](inv_norm)
    for r in range(0, LSA_DIM, 8):
        q_idx_out.store[8](r, q_idx_out.load[8](r) * inv_norm_vec)

    x_buf.free()


@always_inline
def lsa_index_key(
    k: UnsafePointer[Float32, _],
    proj_weights: UnsafePointer[Float32, _],
    k_idx_out: UnsafePointer[Float32, MutUntrackedOrigin],
):
    """
    Extracts Gemma-4 position-free key dimensions ([64:256) and [320:512)),
    concatenates the 2 KV heads into a 768-d vector,
    projects through [64, 768] projection weights using SIMD[Float32, 8],
    and L2-normalizes to a 64-d unit vector.
    """
    var x_buf = alloc[Float32](GEMMA4_INDEX_DIM)

    # 1. Vectorized key copy
    for g in range(GEMMA4_NUM_KV_HEADS):
        var g_out_offset = g * GEMMA4_POS_FREE_DIMS
        var h_base = g * GEMMA4_HEAD_DIM

        for d in range(0, 192, 8):
            x_buf.store[8](g_out_offset + d, k.load[8](h_base + 64 + d))

        for d in range(0, 192, 8):
            x_buf.store[8](g_out_offset + 192 + d, k.load[8](h_base + 320 + d))

    # 2. Vectorized projection: y = W * x
    var sum_sq: Float32 = 0.0
    for r in range(LSA_DIM):
        var w_row = proj_weights + r * GEMMA4_INDEX_DIM
        var acc = SIMD[DType.float32, 8](0.0)
        for c in range(0, GEMMA4_INDEX_DIM, 8):
            acc += w_row.load[8](c) * x_buf.load[8](c)
        var dot = acc.reduce_add()
        k_idx_out[r] = dot
        sum_sq += dot * dot

    # 3. L2 Normalization
    var inv_norm = 1.0 / sqrt(sum_sq if sum_sq > 1e-12 else 1e-12)
    var inv_norm_vec = SIMD[DType.float32, 8](inv_norm)
    for r in range(0, LSA_DIM, 8):
        k_idx_out.store[8](r, k_idx_out.load[8](r) * inv_norm_vec)

    x_buf.free()


# -----------------------------------------------------------------------------
# 2. Packed 64-bit Score Comparator for Branchless Top-P Ordering
# -----------------------------------------------------------------------------

struct BucketHeapItem(ImplicitlyCopyable, Copyable, Movable):
    var score: Float32
    var bucket_id: Int

    def __init__(out self, score: Float32, bucket_id: Int):
        self.score = score
        self.bucket_id = bucket_id

    def __lt__(self, other: BucketHeapItem) -> Bool:
        # Min-heap ordering: smallest score is at the root
        if self.score != other.score:
            return self.score < other.score
        # Tie-breaker for min-heap: higher bucket_id is evicted first
        return self.bucket_id > other.bucket_id


def heap_push(mut heap: List[BucketHeapItem], item: BucketHeapItem):
    heap.append(item)
    var idx = len(heap) - 1
    while idx > 0:
        var parent = (idx - 1) // 2
        if heap[idx] < heap[parent]:
            var tmp = heap[idx]
            heap[idx] = heap[parent]
            heap[parent] = tmp
            idx = parent
        else:
            break


def heap_pop(mut heap: List[BucketHeapItem]) -> BucketHeapItem:
    var root = heap[0]
    var last = heap.pop()
    if len(heap) > 0:
        heap[0] = last
        var idx = 0
        var n = len(heap)
        while True:
            var left = 2 * idx + 1
            var right = 2 * idx + 2
            var smallest = idx
            if left < n and heap[left] < heap[smallest]:
                smallest = left
            if right < n and heap[right] < heap[smallest]:
                smallest = right
            if smallest != idx:
                var tmp = heap[idx]
                heap[idx] = heap[smallest]
                heap[smallest] = tmp
                idx = smallest
            else:
                break
    return root


# -----------------------------------------------------------------------------
# 3. LSALayerIndex Data Structure (Zero-Allocation Hot Path)
# -----------------------------------------------------------------------------

struct LSAMergePair(ImplicitlyCopyable, Copyable, Movable):
    var cost: Float32
    var i: Int
    var j: Int

    def __init__(out self, cost: Float32, i: Int, j: Int):
        self.cost = cost
        self.i = i
        self.j = j

    def __lt__(self, other: LSAMergePair) -> Bool:
        if self.cost != other.cost:
            return self.cost < other.cost
        if self.i != other.i:
            return self.i < other.i
        return self.j < other.j


def sort_merge_pairs(mut list: List[LSAMergePair]):
    for i in range(1, len(list)):
        var key = list[i]
        var j = i - 1
        while j >= 0 and key < list[j]:
            list[j + 1] = list[j]
            j -= 1
        list[j + 1] = key


struct LSALayerIndex:
    var max_capacity: Int
    var max_buckets: Int
    var closed_blocks: Int
    var total_tokens_appended: Int

    # Bucket cache
    var centroids: UnsafePointer[Float32, MutUntrackedOrigin]
    var members: UnsafePointer[Int, MutUntrackedOrigin]
    var sizes: UnsafePointer[Int, MutUntrackedOrigin]

    # Staging buffers for the currently open cluster block
    var staging_keys: UnsafePointer[Float32, MutUntrackedOrigin]
    var staging_pools: UnsafePointer[Float32, MutUntrackedOrigin]
    var staging_labels: UnsafePointer[Int, MutUntrackedOrigin]

    # Pre-allocated scratch for Ward clustering (128 pools)
    var ward_key_sums: UnsafePointer[Float32, MutUntrackedOrigin]
    var ward_bucket_sizes: UnsafePointer[Int, MutUntrackedOrigin]
    var ward_alive: UnsafePointer[Bool, MutUntrackedOrigin]
    var ward_parent: UnsafePointer[Int, MutUntrackedOrigin]
    var ward_gram: UnsafePointer[Float32, MutUntrackedOrigin]
    var ward_alignments: UnsafePointer[Float32, MutUntrackedOrigin]
    var ward_best_partner: UnsafePointer[Int, MutUntrackedOrigin]
    var ward_best_cost: UnsafePointer[Float32, MutUntrackedOrigin]
    var ward_root_to_label: UnsafePointer[Int, MutUntrackedOrigin]

    def __init__(out self, max_capacity: Int = 131072):
        self.max_capacity = max_capacity
        var max_blocks = (max_capacity + LSA_CLUSTER_BLOCK_SIZE - 1) // LSA_CLUSTER_BLOCK_SIZE
        self.max_buckets = max_blocks * LSA_BLOCK_BUCKET_COUNT
        self.closed_blocks = 0
        self.total_tokens_appended = 0

        self.centroids = alloc[Float32](self.max_buckets * LSA_DIM)
        self.members = alloc[Int](self.max_buckets * LSA_BUCKET_CAP)
        self.sizes = alloc[Int](self.max_buckets)

        for i in range(self.max_buckets * LSA_BUCKET_CAP):
            self.members[i] = -1
        for i in range(self.max_buckets):
            self.sizes[i] = 0

        self.staging_keys = alloc[Float32](LSA_CLUSTER_BLOCK_SIZE * LSA_DIM)
        self.staging_pools = alloc[Float32](LSA_POOLS_PER_BLOCK * LSA_DIM)
        self.staging_labels = alloc[Int](LSA_POOLS_PER_BLOCK)

        # Pre-allocate Ward scratch
        self.ward_key_sums = alloc[Float32](LSA_POOLS_PER_BLOCK * LSA_DIM)
        self.ward_bucket_sizes = alloc[Int](LSA_POOLS_PER_BLOCK)
        self.ward_alive = alloc[Bool](LSA_POOLS_PER_BLOCK)
        self.ward_parent = alloc[Int](LSA_POOLS_PER_BLOCK)
        self.ward_gram = alloc[Float32](LSA_POOLS_PER_BLOCK * LSA_POOLS_PER_BLOCK)
        self.ward_alignments = alloc[Float32](LSA_POOLS_PER_BLOCK)
        self.ward_best_partner = alloc[Int](LSA_POOLS_PER_BLOCK)
        self.ward_best_cost = alloc[Float32](LSA_POOLS_PER_BLOCK)
        self.ward_root_to_label = alloc[Int](LSA_POOLS_PER_BLOCK)

    def reset(mut self):
        self.closed_blocks = 0
        self.total_tokens_appended = 0
        for i in range(self.max_buckets):
            self.sizes[i] = 0

    def append_key(mut self, pos: Int, k_idx: UnsafePointer[Float32, _]):
        """
        Appends a 64-d unit index key at position pos.
        When pos closes a 512-token cluster block, runs Ward clustering and populates bucket cache.
        """
        var local_pos = pos % LSA_CLUSTER_BLOCK_SIZE
        var dest = self.staging_keys + local_pos * LSA_DIM
        for i in range(0, LSA_DIM, 8):
            dest.store[8](i, k_idx.load[8](i))

        self.total_tokens_appended += 1

        if local_pos == LSA_CLUSTER_BLOCK_SIZE - 1:
            # Cluster block has closed: run Ward clustering!
            self.ward_cluster_block()

    def ward_cluster_block(mut self):
        var b = self.closed_blocks
        var block_start_token = b * LSA_CLUSTER_BLOCK_SIZE
        var bucket_base = b * LSA_BLOCK_BUCKET_COUNT

        # 1. Pool 4 consecutive keys -> 128 pools
        for p in range(LSA_POOLS_PER_BLOCK):
            var p_off = p * LSA_DIM
            for d in range(0, LSA_DIM, 8):
                var s = SIMD[DType.float32, 8](0.0)
                for j in range(LSA_POOL_SIZE):
                    var k_ptr = self.staging_keys + (p * LSA_POOL_SIZE + j) * LSA_DIM
                    s += k_ptr.load[8](d)
                self.staging_pools.store[8](p_off + d, s)

        # 2. Initialize Ward state
        for i in range(LSA_POOLS_PER_BLOCK * LSA_DIM):
            self.ward_key_sums[i] = self.staging_pools[i]

        for i in range(LSA_POOLS_PER_BLOCK):
            self.ward_bucket_sizes[i] = LSA_POOL_SIZE
            self.ward_alive[i] = True
            self.ward_parent[i] = i

        var active_count = LSA_POOLS_PER_BLOCK

        # 3. Ward merge rounds
        for _ in range(256):
            var merges_left = active_count - LSA_BLOCK_BUCKET_COUNT
            if merges_left <= 0:
                break

            # Symmetric Gram matrix using SIMD dot products
            for i in range(LSA_POOLS_PER_BLOCK):
                if not self.ward_alive[i]:
                    continue
                var i_off = i * LSA_DIM
                for j in range(i, LSA_POOLS_PER_BLOCK):
                    if not self.ward_alive[j]:
                        continue
                    var j_off = j * LSA_DIM
                    var acc = SIMD[DType.float32, 8](0.0)
                    for d in range(0, LSA_DIM, 8):
                        acc += self.ward_key_sums.load[8](i_off + d) * self.ward_key_sums.load[8](j_off + d)
                    var dot = acc.reduce_add()
                    self.ward_gram[i * LSA_POOLS_PER_BLOCK + j] = dot
                    self.ward_gram[j * LSA_POOLS_PER_BLOCK + i] = dot

            for i in range(LSA_POOLS_PER_BLOCK):
                if self.ward_alive[i]:
                    var g_ii = self.ward_gram[i * LSA_POOLS_PER_BLOCK + i]
                    self.ward_alignments[i] = sqrt(g_ii if g_ii > 0.0 else 0.0)

            # Find cheapest allowed partner
            for i in range(LSA_POOLS_PER_BLOCK):
                self.ward_best_partner[i] = -1
                self.ward_best_cost[i] = Float32(1e30)
                if not self.ward_alive[i]:
                    continue

                var min_c: Float32 = Float32(1e30)
                var min_j = -1
                for j in range(LSA_POOLS_PER_BLOCK):
                    if j == i or not self.ward_alive[j]:
                        continue
                    if self.ward_bucket_sizes[i] + self.ward_bucket_sizes[j] > LSA_BUCKET_CAP:
                        continue

                    var g_ii = self.ward_gram[i * LSA_POOLS_PER_BLOCK + i]
                    var g_jj = self.ward_gram[j * LSA_POOLS_PER_BLOCK + j]
                    var g_ij = self.ward_gram[i * LSA_POOLS_PER_BLOCK + j]
                    var merged_sq = g_ii + g_jj + 2.0 * g_ij
                    var merged_align = sqrt(merged_sq if merged_sq > 0.0 else 0.0)
                    var cost = self.ward_alignments[i] + self.ward_alignments[j] - merged_align

                    if cost < min_c:
                        min_c = cost
                        min_j = j
                    elif cost == min_c and (min_j < 0 or j < min_j):
                        min_j = j

                self.ward_best_partner[i] = min_j
                self.ward_best_cost[i] = min_c

            # Collect mutual pairs
            var candidates = List[LSAMergePair]()
            for i in range(LSA_POOLS_PER_BLOCK):
                if not self.ward_alive[i] or self.ward_best_partner[i] < 0:
                    continue
                var j = self.ward_best_partner[i]
                if j > i and self.ward_alive[j] and self.ward_best_partner[j] == i:
                    candidates.append(LSAMergePair(self.ward_best_cost[i], i, j))

            if len(candidates) == 0:
                break

            sort_merge_pairs(candidates)

            var merges_to_do = min(len(candidates), merges_left)
            var merges_done = 0
            for idx in range(len(candidates)):
                if merges_done >= merges_to_do:
                    break
                var pair = candidates[idx]
                var u = pair.i
                var v = pair.j
                if self.ward_alive[u] and self.ward_alive[v] and (self.ward_bucket_sizes[u] + self.ward_bucket_sizes[v] <= LSA_BUCKET_CAP):
                    var u_off = u * LSA_DIM
                    var v_off = v * LSA_DIM
                    for d in range(0, LSA_DIM, 8):
                        self.ward_key_sums.store[8](u_off + d, self.ward_key_sums.load[8](u_off + d) + self.ward_key_sums.load[8](v_off + d))
                    self.ward_bucket_sizes[u] += self.ward_bucket_sizes[v]
                    self.ward_alive[v] = False
                    self.ward_parent[v] = u
                    active_count -= 1
                    merges_done += 1

            if merges_done == 0:
                break

        # Path compression
        for i in range(LSA_POOLS_PER_BLOCK):
            var curr = i
            while not self.ward_alive[curr] and self.ward_parent[curr] != curr:
                curr = self.ward_parent[curr]
            self.ward_parent[i] = curr

        # Renumber surviving roots 0..31 in pool index order
        for i in range(LSA_POOLS_PER_BLOCK):
            self.ward_root_to_label[i] = -1

        var label = 0
        for i in range(LSA_POOLS_PER_BLOCK):
            if self.ward_alive[i]:
                self.ward_root_to_label[i] = label
                label += 1

        for i in range(LSA_POOLS_PER_BLOCK):
            var root = self.ward_parent[i]
            self.staging_labels[i] = self.ward_root_to_label[root]

        # Reset bucket storage for this block
        for k in range(LSA_BLOCK_BUCKET_COUNT):
            var b_id = bucket_base + k
            self.sizes[b_id] = 0
            var c_off = b_id * LSA_DIM
            for d in range(0, LSA_DIM, 8):
                self.centroids.store[8](c_off + d, SIMD[DType.float32, 8](0.0))

        # Assign members and sum centroid vectors
        for p in range(LSA_POOLS_PER_BLOCK):
            var bucket_id = bucket_base + self.staging_labels[p]
            var c_off = bucket_id * LSA_DIM
            for j in range(LSA_POOL_SIZE):
                var local_tok = p * LSA_POOL_SIZE + j
                var global_tok = block_start_token + local_tok
                var k_ptr = self.staging_keys + local_tok * LSA_DIM

                for d in range(0, LSA_DIM, 8):
                    self.centroids.store[8](c_off + d, self.centroids.load[8](c_off + d) + k_ptr.load[8](d))

                var cur_size = self.sizes[bucket_id]
                if cur_size < LSA_BUCKET_CAP:
                    self.members[bucket_id * LSA_BUCKET_CAP + cur_size] = global_tok
                    self.sizes[bucket_id] = cur_size + 1

        # L2-normalize centroids
        for k in range(LSA_BLOCK_BUCKET_COUNT):
            var b_id = bucket_base + k
            var c_off = b_id * LSA_DIM
            var acc = SIMD[DType.float32, 8](0.0)
            for d in range(0, LSA_DIM, 8):
                var val = self.centroids.load[8](c_off + d)
                acc += val * val
            var sum_sq = acc.reduce_add()
            var inv_norm = 1.0 / sqrt(sum_sq if sum_sq > 1e-12 else 1e-12)
            var inv_norm_vec = SIMD[DType.float32, 8](inv_norm)
            for d in range(0, LSA_DIM, 8):
                self.centroids.store[8](c_off + d, self.centroids.load[8](c_off + d) * inv_norm_vec)

        self.closed_blocks += 1

    def select_top_p(
        self,
        q_idx: UnsafePointer[Float32, _],
        pos: Int,
        top_p: Int,
    ) -> List[Int]:
        """
        Scores visible centroids and returns the top_p bucket indices in descending score order.
        """
        var visible_blocks = pos // LSA_CLUSTER_BLOCK_SIZE
        var visible_buckets = min(visible_blocks * LSA_BLOCK_BUCKET_COUNT, self.closed_blocks * LSA_BLOCK_BUCKET_COUNT)

        if visible_buckets == 0:
            return List[Int]()

        var p = min(top_p, visible_buckets)
        var heap = List[BucketHeapItem]()
        heap.reserve(p + 1)

        for b in range(visible_buckets):
            var c_off = b * LSA_DIM
            var acc = SIMD[DType.float32, 8](0.0)
            for d in range(0, LSA_DIM, 8):
                acc += q_idx.load[8](d) * self.centroids.load[8](c_off + d)
            var score = acc.reduce_add()

            var item = BucketHeapItem(score, b)
            if len(heap) < p:
                heap_push(heap, item)
            else:
                # If item is greater than the current minimum in the heap, replace root
                if heap[0] < item:
                    _ = heap_pop(heap)
                    heap_push(heap, item)

        # Extract items from min-heap and reverse to get descending order
        var sorted_items = List[BucketHeapItem]()
        sorted_items.reserve(p)
        while len(heap) > 0:
            sorted_items.append(heap_pop(heap))

        var picked = List[Int]()
        picked.reserve(p)
        for i in range(len(sorted_items) - 1, -1, -1):
            picked.append(sorted_items[i].bucket_id)

        return picked^

    def expand_buckets(
        self,
        picked_buckets: List[Int],
        pos: Int,
        top_k: Int,
        sink_size: Int,
        window_size: Int,
    ) -> List[Int]:
        """
        Expands picked buckets to token indices, followed by window and sinks.
        """
        var window_start = max(0, pos + 1 - window_size)
        var selected = List[Int]()

        # 1. Bucket members
        for b_idx in range(len(picked_buckets)):
            var b = picked_buckets[b_idx]
            var b_size = self.sizes[b]
            for m in range(b_size):
                var tok = self.members[b * LSA_BUCKET_CAP + m]
                if tok >= sink_size and tok < window_start:
                    selected.append(tok)
                    if len(selected) >= top_k:
                        break
            if len(selected) >= top_k:
                break

        # 2. Window: [window_start, pos]
        for t in range(window_start, pos + 1):
            selected.append(t)

        # 3. Sinks: [0, min(sink_size, window_start, pos + 1))
        var sink_limit = min(sink_size, window_start)
        sink_limit = min(sink_limit, pos + 1)
        for t in range(sink_limit):
            selected.append(t)

        return selected^

    def free(mut self):
        self.centroids.free()
        self.members.free()
        self.sizes.free()
        self.staging_keys.free()
        self.staging_pools.free()
        self.staging_labels.free()
        self.ward_key_sums.free()
        self.ward_bucket_sizes.free()
        self.ward_alive.free()
        self.ward_parent.free()
        self.ward_gram.free()
        self.ward_alignments.free()
        self.ward_best_partner.free()
        self.ward_best_cost.free()
        self.ward_root_to_label.free()


# -----------------------------------------------------------------------------
# 4. Fast Sparse Paged Attention Gather (Zero Scaling, KV-Head-Major)
# -----------------------------------------------------------------------------

@always_inline
def lsa_sparse_paged_attention(
    q: UnsafePointer[Float32, _],
    k_pool: UnsafePointer[Float32, _],
    v_pool: UnsafePointer[Float32, _],
    block_table: List[Int],
    block_size: Int,
    selected_keys: List[Int],
    num_heads: Int,
    num_kv_heads: Int,
    head_dim: Int,
    attn_scores: UnsafePointer[Float32, MutUntrackedOrigin],
    attn_rows: UnsafePointer[Int, MutUntrackedOrigin],
    out_attn: UnsafePointer[Float32, MutUntrackedOrigin],
):
    """
    Computes sparse attention over the gathered token set J_i.
    Reuses Phase 0 per-KV-head caching and precomputed physical rows.
    """
    var n_sel = len(selected_keys)
    if n_sel == 0:
        for i in range(num_heads * head_dim):
            out_attn[i] = 0.0
        return

    var kv_stride = num_kv_heads * head_dim
    var ratio = num_heads // num_kv_heads

    # 1. Precompute physical row offsets for all selected tokens
    for idx in range(n_sel):
        var t = selected_keys[idx]
        var logical_blk = t // block_size
        var blk_offset = t % block_size
        var phys_blk_id = block_table[logical_blk]
        attn_rows[idx] = phys_blk_id * block_size + blk_offset

    # 2. KV-Group-Major Attention
    for g in range(num_kv_heads):
        var g_offset = g * head_dim
        var h_base = g * ratio

        for h_rel in range(ratio):
            var h = h_base + h_rel
            var h_offset = h * head_dim
            var q_ptr = q + h_offset

            # Compute dot product scores
            for idx in range(n_sel):
                var row_offset = attn_rows[idx] * kv_stride + g_offset
                var k_ptr = k_pool + row_offset
                var acc = SIMD[DType.float32, 8](0.0)
                for d in range(0, head_dim, 8):
                    acc += q_ptr.load[8](d) * k_ptr.load[8](d)
                attn_scores[idx] = acc.reduce_add()

            # Softmax
            var max_val = attn_scores[0]
            for idx in range(1, n_sel):
                if attn_scores[idx] > max_val:
                    max_val = attn_scores[idx]

            var sum_exp: Float32 = 0.0
            for idx in range(n_sel):
                var e = exp(attn_scores[idx] - max_val)
                attn_scores[idx] = e
                sum_exp += e

            var inv_sum = 1.0 / (sum_exp if sum_exp > 1e-12 else 1e-12)
            for idx in range(n_sel):
                attn_scores[idx] *= inv_sum

            # Weighted V gather
            var out_h = out_attn + h_offset
            for d in range(0, head_dim, 8):
                var val_vec = SIMD[DType.float32, 8](0.0)
                for idx in range(n_sel):
                    var row_offset = attn_rows[idx] * kv_stride + g_offset
                    var v_ptr = v_pool + row_offset
                    val_vec += v_ptr.load[8](d) * SIMD[DType.float32, 8](attn_scores[idx])
                out_h.store[8](d, val_vec)
