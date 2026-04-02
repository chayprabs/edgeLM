# Section 11: KV Cache Design & Optimization -- Deep Research

## Overview

This document covers deep research findings on KV cache design for the EdgeLM inference
engine. Focus areas: ring buffer implementation patterns, memory layout (SoA vs AoS),
and production source code study from llama.cpp, Mistral, and vLLM. All findings are
evaluated for relevance to our target: i7-12700H, DDR4-3200 (~40 GB/s), AVX2/AVX-VNNI,
BitNet 3B with GQA (8 KV heads, head_dim=128), targeting 100+ tok/s.

---

## Topic A: Ring Buffer / Circular Buffer Implementation

### Finding A1: llama.cpp Uses Linear Search with Wraparound, Not a True Ring Buffer

**Source:** `llama-kv-cache.cpp` (ggerganov/llama.cpp, current master)

**Key idea:** llama.cpp's KV cache is a flat array with a "head pointer" per stream
that linearly scans for free cells, wrapping to position 0 when it reaches the end.
This is conceptually simpler than a true circular buffer.

**Technical details:**

The core find_slot algorithm:
```c
while (true) {
    if (head_cur + n_test > cells.size()) {
        n_tested += cells.size() - head_cur;
        head_cur = 0;  // wraparound to beginning
        continue;
    }
    for (uint32_t i = 0; i < n_test; i++) {
        bool can_use = cells.is_empty(idx);
        if (!can_use && cells.seq_count(idx) == 1) {
            // Check SWA mask - can reuse if token is outside sliding window
            if (is_masked_swa(n_swa, swa_type, pos_cell, seq_pos_max + 1))
                can_use = true;
        }
        if (can_use) res.idxs[s].push_back(idx);
    }
}
```

After apply_ubatch completes: `head = sinfo.idxs[s].back() + 1`, advancing past
allocated cells. Optimization: if head_cur > cells.get_used() + 2*n_tokens, reset
to 0 to prioritize filling gaps.

**Relevance to EdgeLM:** For single-user inference, this linear-scan-with-wraparound
is simple and effective. For our 2048-context case with ~2048 cells, the scan is
trivially fast. We can simplify further since we have exactly one sequence -- no need
for multi-sequence cell tracking. A simple write pointer with `pos % capacity`
modular indexing suffices.

---

### Finding A2: Mistral's Rolling Buffer Cache Uses `pos % W` Modular Indexing

**Source:** `mistral-inference/cache.py` (mistralai/mistral-inference, GitHub)

**Key idea:** Mistral's official inference code uses true modular arithmetic
(`position % window_size`) to map sequence positions to physical buffer slots,
creating a genuine ring buffer for sliding window attention.

**Technical details:**

Cache position mapping formula:
```python
cache_positions = positions % cache_size + batch_idx * cache_size
```

The cache tensor shape is `(max_batch_size, cache_size, n_kv_heads, head_dim)`.

The `unrotate()` function recovers chronological order from the ring buffer:
```python
def unrotate(cache, seqlen):
    position = seqlen % cache.shape[0]
    if seqlen < cache.shape[0]:
        return cache[:seqlen]
    elif position == 0:
        return cache
    else:
        return torch.cat([cache[position:], cache[:position]], dim=0)
```

Three pre-fill/decode modes:
- First prefill (seqpos[0] == 0): BlockDiagonalCausalMask
- Subsequent prefill: BlockDiagonalMask with variable KV lengths
- Decode (single token): BlockDiagonalCausalWithOffsetPaddedKeysMask

**Relevance to EdgeLM:** The `pos % W` pattern is exactly what we should use. For
our case with power-of-2 cache sizes, this becomes `pos & (W-1)` -- a single AND
instruction instead of expensive integer division. The unrotate pattern is needed
when we need to read the cache in chronological order (e.g., for beam search or
cache serialization), but for normal attention computation with RoPE, physical
order doesn't matter since positions are encoded in the keys themselves.

---

### Finding A3: Mistral Sliding Window Attention -- 4096-Token Window with Ring Buffer

**Source:** https://mistral.ai/news/announcing-mistral-7b

**Key idea:** Mistral 7B uses a 4096-token sliding window where each layer attends
only to the previous W=4096 tokens, but due to layer stacking, information from
tokens up to W*n_layers positions back can propagate through the network.

**Technical details:**

- Window size W = 4096 tokens
- Each layer attends to [i-W, i] hidden states at layer k-1
- Memory savings: ~50% at 8192 tokens vs full attention
- Combined with FlashAttention: 2x speed improvement for 16k-length sequences
- The ring buffer approach means cache memory is FIXED at W entries regardless
  of total sequence length

**Relevance to EdgeLM:** Our target models (BitNet-b1.58-2B-4T and bitnet_b1_58-3B)
use full attention, not sliding window, so we need a full context ring buffer. However,
the ring buffer pattern is still valuable: when context exceeds our allocated cache
size (e.g., 2048), we can evict oldest tokens. The key insight is that ring buffer
memory is FIXED -- no dynamic allocation, no fragmentation, completely predictable
memory usage. This is ideal for our 6-7 GB RAM budget.

---

### Finding A4: Power-of-2 Sizing with Bitmask Eliminates Division

**Source:** Fabian Giesen, "Ring Buffers and Queues" (fgiesen.wordpress.com)

**Key idea:** When ring buffer size is a power of 2, the modulo operation `pos % size`
becomes `pos & (size - 1)`, which is a single-cycle bitwise AND instruction. This
eliminates expensive integer division (20-90 cycles on x86) from the critical path.

**Technical details:**

The "virtual stream" model:
- Maintain absolute write/read positions (never wrap them, let them overflow naturally)
- Only reduce modulo when indexing into the array: `buffer[write_pos & (SIZE-1)]`
- Queue size = write_pos - read_pos (works even with unsigned overflow)
- No ambiguity between full and empty states
- On PowerPC, address generation can be done with a single `rlwinm` instruction
  when both SIZE and sizeof(ElemType) are powers of 2

Performance comparison:
- General modulo: requires idiv instruction (20-90+ cycles on x86)
- Power-of-2 bitmask: single AND instruction (1 cycle)
- This is ~20-90x faster for index computation

**Relevance to EdgeLM:** CRITICAL for our design. We should use power-of-2 cache
sizes (1024, 2048, 4096). For our target 2048-context: cache_size = 2048,
index = pos & 2047 (0x7FF). This single AND instruction replaces modular division
on every KV cache access. For 30 layers x 8 KV heads x 2 (K+V) = 480 cache
accesses per token, saving ~20 cycles each = ~9600 cycles saved per token.

---

### Finding A5: llama.cpp Context Shifting Re-applies RoPE When Cache Fills

**Source:** llama.cpp PR #3228 (ggerganov/llama.cpp GitHub)

**Key idea:** When the KV cache fills up, llama.cpp shifts existing entries and
re-applies RoPE with adjusted position values, rather than discarding old tokens.
This is called "context shifting" and avoids full re-evaluation of the prompt.

**Technical details:**

The `llama_kv_cache_shift_seq()` mechanism:
1. Shift existing K-cache data in the buffer
2. Re-apply RoPE with adjusted positions: rotate back, apply new RoPE, rotate forward
3. Maintain sequence continuity without full re-evaluation

The shift uses Hadamard matrices for quantization-aware rotation:
```c
// For quantized K values:
tmp = ggml_cast(ctx, cur, GGML_TYPE_F32);
tmp = ggml_mul_mat_aux(ctx, tmp, rot);      // rotate back (inverse Hadamard)
tmp = ggml_rope_ext(ctx, tmp, shift, ...);   // apply new RoPE
tmp = ggml_mul_mat_aux(ctx, tmp, rot);      // rotate forward (Hadamard)
```

Hadamard matrices are precomputed for power-of-2 dimensions (64, 128, 256).

**Relevance to EdgeLM:** Context shifting is an alternative to ring buffer eviction.
With a ring buffer, old tokens simply get overwritten. With context shifting, all
tokens get new position IDs. For single-user inference, ring buffer eviction is
simpler and cheaper (no RoPE recomputation). But context shifting preserves more
semantic information from the full conversation history.

---

### Finding A6: Critical Design Choice -- Pre-RoPE vs Post-RoPE Key Storage

**Source:** llama.cpp `llama-kv-cache.cpp` + `llama-graph.cpp` (current master),
Mistral `transformer_layers.py`, KVQuant paper (arXiv:2401.18079)

**Key idea:** llama.cpp stores keys PRE-RoPE (without positional encoding applied),
then applies RoPE dynamically at attention time. Mistral stores keys POST-RoPE
(with positions baked in). This has major implications for ring buffer design.

**Technical details:**

**llama.cpp (Pre-RoPE storage):**
```c
// Store K into cache WITHOUT RoPE
ggml_build_forward_expand(gf, mctx_cur->cpy_k(ctx0, k_cur, k_idxs, il));

// RoPE applied later at attention time via rotation matrices
// inp->self_k_rot applied within build_attn_mha()
```

Advantages of pre-RoPE:
- Ring buffer wraparound is trivial -- no position mismatch
- Context shifting just updates position metadata, no key recomputation
- Easier KV cache quantization (KVQuant paper: "Pre-RoPE Key Quantization"
  reduces distortion from RoPE's rotational artifacts)
- Position-agnostic cache entries can be reused across different positions

**Mistral (Post-RoPE storage):**
```python
xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)
# Then store to cache AFTER RoPE applied
cache.update(xk, xv)
```

Advantages of post-RoPE:
- RoPE computed once per token (not on every attention query)
- Simpler attention kernel (no RoPE step during attention)
- For sliding window where positions are always close to current, works fine

**Relevance to EdgeLM:** For our ring buffer design, PRE-RoPE storage is strongly
preferred because:
1. No position mismatch when ring buffer wraps (physical slot 0 can hold any position)
2. Enables context shifting without recomputing keys
3. Better quantization quality (KVQuant findings)
4. Trade-off: requires applying RoPE during attention, but RoPE is cheap (~5.3ms
   per EleutherAI benchmark, and our sequence lengths are small)

However, there is a compute cost: with 2048 context and 30 layers, we apply RoPE
to 2048 keys per layer per token during decode. At 8 KV heads x 128 dim = 1024
elements per key, this is 2048 x 1024 = 2M elements per layer. With AVX2 (8 floats
per cycle), this is ~256K cycles per layer, or ~7.7M cycles total for 30 layers.
At 4.7 GHz, that's ~1.6ms per token -- acceptable for our 100 tok/s target (10ms budget).

**DECISION: Use pre-RoPE key storage.** Apply RoPE dynamically during attention.

---

## Topic B: Memory Layout -- SoA vs AoS, Head Organization

### Finding B1: llama.cpp Uses SoA Layout -- Separate K and V Tensors Per Layer

**Source:** `llama-kv-cache.cpp`, `llama-kv-cache.h` (ggerganov/llama.cpp)

**Key idea:** llama.cpp stores K and V in completely separate 3D tensors per layer,
with shape `[n_embd_k_gqa, kv_size, n_stream]` for K and similarly for V. This is
a Structure-of-Arrays (SoA) approach.

**Technical details:**

Tensor creation:
```c
ggml_tensor * k = ggml_new_tensor_3d(ctx, type_k,
                     n_embd_k_gqa,  // embedding dim (e.g., 1024 for 8 heads x 128 dim)
                     kv_size,       // number of cache slots (e.g., 2048)
                     n_stream);     // number of parallel streams (1 for single-user)

ggml_tensor * v = ggml_new_tensor_3d(ctx, type_v,
                     n_embd_v_gqa,  // embedding dim
                     kv_size,       // cache slots
                     n_stream);     // streams
```

Per-stream views: `ggml_view_2d(ctx, k, n_embd_k_gqa, kv_size, k->nb[1], s*k->nb[2])`

The V cache has a v_trans flag (default true) that transposes the layout when
Flash Attention is disabled.

GQA dimension calculation:
```c
n_embd_k_gqa = n_embd_head_k * n_head_kv  // e.g., 128 * 8 = 1024
n_embd_v_gqa = n_embd_head_v * n_head_kv  // e.g., 128 * 8 = 1024
```

**Memory layout in practice (for our model: 8 KV heads, head_dim=128):**

For one layer with kv_size=2048, n_stream=1:
- K tensor: [1024, 2048, 1] = 1024 * 2048 elements per layer
- V tensor: [1024, 2048, 1] = 1024 * 2048 elements per layer
- Total per layer (FP16): 2 * 1024 * 2048 * 2 bytes = 8 MB
- Total for 30 layers: 240 MB at FP16

**Relevance to EdgeLM:** SoA is the right choice for our case. During decode-phase
attention, we need to compute Q*K^T for all cached keys, then apply softmax, then
multiply by V. Separate K and V arrays mean:
- K access pattern: read all 2048 keys contiguously for dot product with Q
- V access pattern: read all 2048 values for weighted sum
- No wasted bandwidth reading V when computing Q*K^T and vice versa

---

### Finding B2: HuggingFace Transformers Uses [batch, num_heads, seq_len, head_dim] Layout

**Source:** https://huggingface.co/blog/optimize-llm

**Key idea:** The standard PyTorch/HuggingFace layout for KV cache is
`(batch_size, num_heads, sequence_length, embed_size_per_head)`, with keys
and values stored separately in a tuple per layer.

**Technical details:**

Cache shape: `(batch_size, num_heads, sequence_length, embed_size_per_head)`

For our model (batch=1, 8 KV heads, 2048 context, head_dim=128):
- K shape: (1, 8, 2048, 128)
- V shape: (1, 8, 2048, 128)

The cache is organized as tuple of (K, V) pairs, one per layer:
`past_key_values = [(K_layer0, V_layer0), (K_layer1, V_layer1), ...]`

Memory per token across all layers: `2 * num_layers * num_kv_heads * head_dim * bytes_per_elem`

For our model at FP16: 2 * 30 * 8 * 128 * 2 = 122,880 bytes (~120 KB per token)
For 2048 tokens: ~240 MB total

With MQA (1 KV head): reduces to ~30 MB. With GQA (8 KV heads out of 32 Q heads):
the 4:1 ratio gives 4x savings vs standard MHA, matching our model's design.

**Relevance to EdgeLM:** The `[num_heads, seq_len, head_dim]` layout (ignoring
batch dim for single-user) is the standard. But we should consider whether
`[seq_len, num_heads, head_dim]` might be better for our access patterns during
decode. See Finding B3 for analysis.

---

### Finding B3: Optimal Layout Analysis for GQA Decode on CPU

**Source:** Synthesis of llama.cpp code, HuggingFace layout, and Mistral implementation

**Key idea:** For CPU inference with GQA, the optimal memory layout depends on the
operation: Q*K^T benefits from contiguous keys per head across sequence positions,
while the softmax(QK^T)*V operation benefits from contiguous values per position.

**Technical details:**

**Option 1: [head][seq][dim] -- llama.cpp's effective layout**
- K[h][s][d] where h=0..7, s=0..2047, d=0..127
- Pro: For Q*K^T, iterating over seq positions for a single head is contiguous
- Pro: Each head's data is contiguous -- good for per-head processing
- Con: Reading all heads for a single position requires stride jumps
- Memory per head: 2048 * 128 * 2 = 512 KB -- fits in L2 cache (1.25 MB/P-core)

**Option 2: [seq][head][dim] -- interleaved layout**
- K[s][h][d] where s=0..2047, h=0..7, d=0..127
- Pro: All heads for a single position are contiguous (good for GQA broadcast)
- Con: Per-head sequential access has stride = 8*128*2 = 2048 bytes between
  consecutive positions for same head
- Cache-line analysis: at 64 bytes/line, 128*2=256 bytes per head-dim, so
  4 cache lines per head entry. With 8 heads, each position = 32 cache lines.

**Option 3: [seq][head*dim] -- flattened layout**
- K[s][h*d] where s=0..2047, combined dim = 8*128 = 1024
- Pro: Simple 2D array, easy ring buffer indexing
- Con: Same stride issues as Option 2 for per-head access

**Analysis for decode-phase attention (the bottleneck):**

During decode, Q is a single vector [8_q_heads * 128]. For GQA with 4:1 ratio,
each KV head serves 4 Q heads. The critical operation is:

For each KV head h (0..7):
  score[s] = dot(Q_group[h], K[h][s])  for s = 0..2047
  This is a matrix-vector multiply: [4, 128] * [128, 2048] -> [4, 2048]

With Option 1 [head][seq][dim]: K[h] is a contiguous [2048, 128] block = 512 KB.
This FITS in L2 cache. Streaming through it for dot products is highly efficient.

With Option 2 [seq][head][dim]: K entries for head h are scattered with 2KB stride.
For 2048 positions, we touch 2048 * 4 cache lines = 512 KB worth of data, but
we also pull in the other 7 heads' data, wasting ~75% of bandwidth.

**Conclusion: Option 1 [head][seq][dim] is clearly optimal for CPU decode.**

**Relevance to EdgeLM:** Use [head][seq][dim] layout for both K and V caches.
This means per-layer, the K cache is 8 contiguous blocks of [2048][128], where
each block is 512 KB and fits entirely in L2 cache of a P-core. During decode,
we process one KV head at a time, keeping the working set in L2.

For our INT8 quantized cache: each block = 2048 * 128 * 1 = 256 KB, which
fits even more comfortably in L2 (1.25 MB per P-core). We can fit ~4 heads
simultaneously in L2.

---

### Finding B4: V-Cache Transposition for Efficient Weighted Sum

**Source:** llama.cpp `llama-kv-cache.h`, `llama-kv-cache.cpp`

**Key idea:** llama.cpp supports a `v_trans` flag (default true when Flash Attention
is disabled) that transposes the V cache layout. This optimizes the softmax(QK^T) * V
operation by making the dimension being summed over contiguous in memory.

**Technical details:**

Normal V layout: V[head][seq][dim] -- shape [n_kv_heads, kv_size, head_dim]
Transposed V layout: V[head][dim][seq] -- shape [n_kv_heads, head_dim, kv_size]

For the weighted sum: output[d] = sum_s(weights[s] * V[s][d])

With normal layout V[seq][dim]: we need V values at each position but across
dimensions -- this is a column access pattern with poor locality.

With transposed layout V[dim][seq]: for each dimension d, V[d][0..2047] is
contiguous. We multiply element-wise with weights[0..2047] and sum. This is
a streaming SIMD operation -- ideal for AVX2.

In llama.cpp, when v_trans is true:
```c
// V indices are expanded: [n_tokens * n_embd_v_gqa] instead of [n_tokens]
// This enables scattered writes to the transposed layout
```

**Relevance to EdgeLM:** V transposition is a significant optimization for CPU
decode. The weighted sum `output = weights^T * V` with transposed V becomes a
series of SIMD dot products along contiguous memory. With AVX2:

For one KV head, one dimension:
  output[d] = _mm256_fmadd_ps(weights_vec, V_row[d], accumulator)
  Processing 8 positions per cycle, 2048 positions = 256 iterations

This is 256 * 128 = 32,768 FMA operations per head, or 262,144 for 8 heads.
At ~1 FMA/cycle with AVX2 FP32, this takes ~262K cycles = ~56 microseconds
at 4.7 GHz.

**DECISION: Use transposed V layout [head][dim][seq] for the value cache.**

---

## Topic H: Production Source Code Study

### Finding H1: llama.cpp KV Cache Complete Architecture

**Source:** `llama-kv-cache.h`, `llama-kv-cache.cpp`, `llama-kv-cells.h`,
`llama-memory.h`, `llama-hparams.cpp` (ggerganov/llama.cpp)

**Key idea:** llama.cpp has a sophisticated multi-level KV cache abstraction with
abstract memory interface, cell tracking via bitsets, per-layer tensors, multi-stream
support, and sequence management. Much of this complexity is unnecessary for
single-user inference.

**Technical details:**

**Class hierarchy:**
```
llama_memory_i (abstract base -- "general concept of LLM memory")
├── llama_kv_cache (standard KV cache)
├── llama_kv_cache_iswa (interleaved Sliding Window Attention -- dual cache)
└── llama_memory_context_i (context for batch processing)
    └── llama_kv_cache_context
```

**Cell tracking (llama_kv_cells):**
```c
struct llama_kv_cells {
    std::set<uint32_t> used;                    // active cell indices
    std::vector<llama_pos> pos;                 // position per cell (-1 = empty)
    std::vector<llama_kv_cell_ext> ext;         // 2D spatial coords (M-RoPE)
    std::vector<llama_pos> shift;               // accumulated position adjustments
    std::vector<seq_set_t> seq;                 // bitset per cell (which sequences)
    std::map<llama_pos, int> seq_pos[MAX_SEQ];  // position->count per sequence
};
```

Where `seq_set_t = std::bitset<LLAMA_MAX_SEQ>` tracks which sequences own each cell.

**Context parameters:**
```c
struct llama_context_params {
    uint32_t n_ctx;         // text context size (padded to 256)
    uint32_t n_batch;       // logical max batch size
    uint32_t n_ubatch;      // physical max batch size
    uint32_t n_seq_max;     // max sequences
    ggml_type type_k;       // K cache data type [EXPERIMENTAL]
    ggml_type type_v;       // V cache data type [EXPERIMENTAL]
    float defrag_thold;     // [DEPRECATED] defragmentation threshold
    bool offload_kqv;       // offload to GPU
    bool kv_unified;        // unified across sequences
};
```

**Public API (sequence-based):**
```c
void llama_memory_clear(mem, data);
bool llama_memory_seq_rm(mem, seq_id, p0, p1);   // remove range [p0, p1)
void llama_memory_seq_cp(mem, src, dst, p0, p1);  // copy sequence
void llama_memory_seq_keep(mem, seq_id);           // keep only one seq
void llama_memory_seq_add(mem, seq_id, p0, p1, delta);  // shift positions
void llama_memory_seq_div(mem, seq_id, p0, p1, d);      // divide positions
llama_pos llama_memory_seq_pos_min(mem, seq_id);
llama_pos llama_memory_seq_pos_max(mem, seq_id);
bool llama_memory_can_shift(mem);
```

**Relevance to EdgeLM:** We can dramatically simplify this. For single-user inference
with one sequence, we need:
- No multi-sequence bitset tracking
- No seq_cp/seq_rm/seq_keep operations
- No multi-stream support
- Just: a flat ring buffer with a write pointer, per-layer K/V tensors,
  and a simple used-count

Our struct can be approximately:
```c
typedef struct {
    int32_t  n_layers;
    int32_t  n_kv_heads;
    int32_t  head_dim;
    int32_t  capacity;       // power of 2, e.g., 2048
    int32_t  mask;           // capacity - 1, for bitmask indexing
    int32_t  n_used;         // number of valid entries
    int32_t  write_pos;      // next write position (absolute, never wraps)
    int32_t  type_k;         // GGML_TYPE_Q8_0, GGML_TYPE_F16, etc.
    int32_t  type_v;
    // Per-layer K and V buffers, 64-byte aligned
    void*    k[MAX_LAYERS];  // [n_kv_heads][capacity][head_dim] for K
    void*    v[MAX_LAYERS];  // [n_kv_heads][head_dim][capacity] for V (transposed)
    int32_t* positions;      // [capacity] absolute position of each slot
} edgelm_kv_cache;
```

Indexing: `slot = write_pos & mask;` (single AND instruction)

---

### Finding H2: llama.cpp iSWA -- Dual Cache for Mixed Attention Layers

**Source:** `llama-kv-cache-iswa.h`, `llama-kv-cache-iswa.cpp` (ggerganov/llama.cpp)

**Key idea:** llama.cpp implements an "interleaved Sliding Window Attention" (iSWA)
system with TWO separate KV caches: one for full-attention layers and one for
sliding-window layers. Layers are routed to the appropriate cache based on
`hparams.is_swa(il)`.

**Technical details:**

```c
// Layer filtering determines cache assignment
filter_base = [](int32_t il) { return !model.hparams.is_swa(il); };
filter_swa  = [](int32_t il) { return model.hparams.is_swa(il); };

// SWA cache sizing
uint32_t size_swa = GGML_PAD(
    min(size_base, hparams.n_swa * (unified ? n_seq_max : 1) + n_ubatch),
    256  // pad to 256 for performance
);
```

Operations like seq_rm, seq_cp, seq_keep are delegated to BOTH caches simultaneously.

**Relevance to EdgeLM:** Our target BitNet models use full attention on all layers
(no SWA layers), so we don't need dual caches. However, the concept of different
cache sizes per layer is worth noting -- if we ever support Mistral-style models,
SWA layers need only W entries instead of full context.

---

### Finding H3: vLLM PagedAttention -- Block Table for KV Cache

**Source:** https://vllm.ai/blog/vllm, arXiv:2309.06180 (vLLM paper)

**Key idea:** vLLM partitions KV cache into fixed-size blocks (like OS virtual memory
pages), maps logical sequence positions to non-contiguous physical blocks via a
block table, and allocates blocks on demand. This eliminates the 60-80% memory
waste from pre-allocation in traditional systems.

**Technical details:**

- Block table: maps logical blocks -> physical blocks (like page table)
- Each block stores KV pairs for a fixed number of tokens (typically 16)
- Physical blocks allocated on demand as new tokens generated
- Memory waste only in last block of each sequence (< 4% waste)
- Copy-on-Write: shared prefix blocks have reference counts, copied only when
  one sequence diverges
- LLaMA-13B KV cache: up to 1.7 GB per sequence
- Performance: 24x over HuggingFace, 3.5x over TGI
- Memory sharing for parallel sampling: up to 55% memory reduction, 2.2x throughput

**Relevance to EdgeLM:** PagedAttention is designed for multi-user GPU serving and
is OVERKILL for single-user CPU inference. The block table indirection adds overhead
without benefit when there's only one sequence. However, one idea is useful:

**Prompt caching / prefix sharing:** If we want to reuse KV cache across multiple
prompts with shared prefixes (e.g., system prompt), we could use a simple two-tier
approach: fixed prefix cache + ring buffer for new tokens. No need for full paging.

---

### Finding H4: vAttention -- Virtual Memory for Contiguous KV Cache

**Source:** arXiv:2405.04437 (vAttention paper)

**Key idea:** vAttention decouples virtual and physical memory for KV cache using
CUDA virtual memory APIs, maintaining contiguous virtual addresses while allowing
fragmented physical allocation. This achieves 1.23x throughput over PagedAttention
with simpler code and better kernel compatibility.

**Technical details:**

- PagedAttention's non-contiguous layout causes "non-trivial programming and
  performance overheads" in attention kernels
- vAttention keeps virtual memory contiguous (standard attention kernels work
  unchanged) while physical allocation is on-demand
- Up to 1.23x speedup vs PagedAttention kernels in FlashAttention/FlashInfer
- LLM-specific optimizations for CUDA virtual memory limitations

**Relevance to EdgeLM:** Reinforces that contiguous memory layout is important for
kernel performance. Our ring buffer with power-of-2 sizing provides a contiguous
buffer that works naturally with SIMD kernels. No virtual memory tricks needed for
single-user CPU inference -- just pre-allocate the full cache at startup.

---

### Finding H5: BitNet.cpp Has No Custom KV Cache -- Uses llama.cpp's Cache

**Source:** github.com/microsoft/BitNet, `src/` directory listing

**Key idea:** The BitNet repository (Microsoft's official bitnet.cpp) only contains
custom kernel files (`ggml-bitnet-lut.cpp`, `ggml-bitnet-mad.cpp`) for weight
multiplication. It relies entirely on llama.cpp's infrastructure for KV cache,
tokenization, and the rest of the inference pipeline.

**Technical details:**

BitNet src/ contains only:
- `ggml-bitnet-lut.cpp` -- lookup table operations for ternary weights
- `ggml-bitnet-mad.cpp` -- multiply-accumulate kernels with configurable tiling

The KV cache, attention mechanism, and model loading all use unmodified llama.cpp
code. There are no ternary-model-specific KV cache optimizations.

**Relevance to EdgeLM:** This confirms that KV cache design is orthogonal to weight
quantization format. Our ternary weight kernels produce standard FP32/FP16 activations
that feed into a standard KV cache. The K and V projections output regular floating-point
values regardless of whether weights are ternary. Our KV cache design can be
completely independent of our ternary matmul kernel design.

---

## Additional Findings: KV Cache Quantization

### Finding Q1: KIVI -- Asymmetric Quantization: Keys Per-Channel, Values Per-Token

**Source:** arXiv:2402.02750 (KIVI, ICML 2024)

**Key idea:** Keys and values have fundamentally different distribution patterns and
should be quantized differently. Keys have outliers in specific channels (dimensions)
and should be quantized per-channel. Values have more uniform distributions and
should be quantized per-token.

**Technical details:**

- Keys: quantized per-channel (group along channel/dim dimension)
- Values: quantized per-token (group along sequence dimension)
- Achieves 2-bit quantization with near-zero quality degradation
- 2.6x peak memory reduction (including model weights)
- 4x larger batch sizes possible
- 2.35x to 3.47x throughput improvement
- Tuning-free (no calibration needed)
- Models tested: LLaMA, Falcon, Mistral

**Relevance to EdgeLM:** For our INT8/FP8 KV cache quantization:
- K cache: quantize per-channel (per-dimension across positions)
  Scale factors: one per head_dim = 128 scale factors per head per layer
- V cache: quantize per-token (per-position across dimensions)
  Scale factors: one per position = 2048 scale factors per head per layer

This is easy to implement with our [head][seq][dim] K layout and [head][dim][seq]
V layout, since the quantization grouping aligns with contiguous memory access.

---

### Finding Q2: KVQuant -- Pre-RoPE Quantization Reduces Distortion

**Source:** arXiv:2401.18079 (KVQuant, NeurIPS 2024)

**Key idea:** Quantizing keys BEFORE RoPE application significantly reduces
quantization error because RoPE's rotation introduces high-frequency patterns
that are hard to quantize. Pre-RoPE keys have smoother distributions.

**Technical details:**

Four innovations:
1. Per-channel key quantization (same as KIVI finding)
2. Pre-RoPE key quantization -- apply RoPE AFTER dequantization at attention time
3. Non-uniform per-layer sensitivity-weighted datatypes
4. Per-vector dense-and-sparse quantization (isolate outliers)

Results:
- < 0.1 perplexity degradation at 3-bit quantization
- 1 million context on single A100-80GB
- 10 million context on 8-GPU system
- ~1.7x speedup with custom CUDA kernels

**Relevance to EdgeLM:** This STRONGLY reinforces our Finding A6 decision to store
keys pre-RoPE. Since we're also planning INT8 quantization of the KV cache, pre-RoPE
storage gives us a double win:
1. Simpler ring buffer design (no position-dependent data)
2. Better quantization quality (smoother distributions)

---

### Finding Q3: HuggingFace KV Cache Quantization -- INT4 is the Sweet Spot

**Source:** https://huggingface.co/blog/kv-cache-quantization

**Key idea:** INT4 KV cache quantization maintains FP16 quality with ~2.5x memory
savings. INT2 causes significant quality degradation. Group size of 64 with residual
length of 128 tokens (kept at full precision) is the recommended configuration.

**Technical details:**

Quantization formula: `X_Q = round(X / S) - Z`
where S = (max(X) - min(X)) / (max_val - min_val), Z = round(-min(X) / S)

Grouping: input `(batch, heads, tokens, head_dim)` grouped into `(n_groups, group_size=64)`

Residual cache: last 128 tokens kept at FP16 to avoid repeated quantize/dequantize

Perplexity results (LLaMA-2-7B on PG-19):
- INT4: nearly identical to FP16
- INT2: significant degradation (TREC: 63.0 -> 55.0, SAMSum: 41.12 -> 14.04)

Memory impact:
- FP16 + Flash Attention: 40K tokens on A100-80GB
- INT4 + Flash Attention: 128K tokens (3.2x more)

Speed trade-off: INT4 quantization adds latency (2.5x slower generation with
combined weight + KV quantization)

**Relevance to EdgeLM:** For our DDR4 bandwidth-limited system, INT8 is the likely
sweet spot (not INT4). INT8 gives 2x memory reduction with minimal quality impact
and simpler SIMD implementation. INT4 would give 4x reduction but:
- Our cache is already small (~60 MB at INT8 for 2048 context)
- INT4 dequantization is more complex with AVX2
- Quality risk is higher

The group_size=64 finding is useful: quantize in groups of 64 elements with
separate scale/zero-point per group. For head_dim=128, that's 2 groups per head
per position.

---

### Finding Q4: llama.cpp Supports Q8_0, Q4_0, Q4_1 for K Cache

**Source:** llama.cpp PR #4309, Feature matrix wiki

**Key idea:** llama.cpp's K cache supports quantized types Q8_0, Q4_0, Q4_1 on CPU
(AVX/AVX2), Metal, CUDA, ROCm, and Vulkan backends. V cache quantization was added
later. Performance with quantized K cache was noted as "a bit disappointing" on CUDA
due to per-head kernel overhead.

**Technical details:**

Supported types via `type_k` and `type_v` parameters:
- F16 (default)
- Q8_0 (8-bit with block-level scaling)
- Q4_0 (4-bit symmetric)
- Q4_1 (4-bit asymmetric)

CPU backend (AVX/AVX2): fully supported for K cache quantization.

Performance note: quantized K cache requires copy kernels (F32 -> Q8_0, F32 -> Q4_0/Q4_1).
The main overhead is quantization during the write path, not during the read path.

Context shifting doesn't work with quantized K cache (requires dequantize -> re-RoPE
-> requantize, which is expensive).

**Relevance to EdgeLM:** Q8_0 (8-bit with per-block scale) is well-tested in llama.cpp
on CPU with AVX2. We should follow this format:
- Block size: 32 elements (standard ggml Q8_0)
- Each block: 32 INT8 values + 1 FP16 scale = 34 bytes per block
- For head_dim=128: 4 blocks = 136 bytes per head per position
- vs FP16: 256 bytes per head per position
- Savings: 47% (not quite 2x due to scale overhead)

Alternative: use raw INT8 with per-head-per-layer scale factors (fewer scale values,
closer to true 2x savings). Trade-off: slightly more quantization error.

---

## Summary of Recommendations for EdgeLM KV Cache Design

### Architecture Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Buffer type | Ring buffer with power-of-2 capacity | `pos & (cap-1)` bitmask indexing |
| Capacity | 2048 (or configurable power-of-2) | Matches target context length |
| K storage | Pre-RoPE | Simpler ring buffer, better quantization |
| K layout | [layer][head][seq][dim] | Contiguous per-head access in L2 |
| V layout | [layer][head][dim][seq] (transposed) | Efficient weighted sum with SIMD |
| K/V separation | Separate arrays (SoA) | Independent access patterns |
| Quantization | INT8 per-channel for K, per-token for V | KIVI/KVQuant findings |
| Alignment | 64-byte aligned (AVX2 requirement) | All buffer starts on cache-line boundary |
| Cell tracking | Simple write pointer + position array | No multi-sequence overhead |

### Memory Budget

For BitNet-b1.58-2B-4T (30 layers, 8 KV heads, head_dim=128, capacity=2048):

| Format | Per-layer K | Per-layer V | Total (30 layers) |
|--------|-------------|-------------|-------------------|
| FP16 | 4.0 MB | 4.0 MB | 240 MB |
| INT8 | 2.0 MB | 2.0 MB | 120 MB |
| Q8_0 | ~2.1 MB | ~2.1 MB | ~128 MB |
| INT4 | 1.0 MB | 1.0 MB | 60 MB |

With INT8: 120 MB total -- well within our 6-7 GB budget. Even FP16 at 240 MB
is acceptable given our ~400-600 MB model size.

### Proposed Struct

```c
#define KV_CACHE_MAX_LAYERS 32

typedef struct {
    // Configuration
    int32_t  n_layers;
    int32_t  n_kv_heads;      // 8 for our model
    int32_t  head_dim;        // 128
    int32_t  capacity;        // power of 2, e.g., 2048
    int32_t  mask;            // capacity - 1
    int32_t  type_k;          // e.g., EDGELM_TYPE_INT8
    int32_t  type_v;          // e.g., EDGELM_TYPE_INT8

    // Ring buffer state
    int32_t  write_pos;       // absolute position (never wraps, use & mask)
    int32_t  n_used;          // min(write_pos, capacity)
    int32_t  positions[/*capacity*/];  // absolute position per slot

    // Per-layer K cache: [n_kv_heads][capacity][head_dim]
    // Pre-RoPE, row-major per head for contiguous access
    void*    k_data[KV_CACHE_MAX_LAYERS];
    float*   k_scales[KV_CACHE_MAX_LAYERS];  // quantization scales

    // Per-layer V cache: [n_kv_heads][head_dim][capacity] (TRANSPOSED)
    // Optimized for weighted sum
    void*    v_data[KV_CACHE_MAX_LAYERS];
    float*   v_scales[KV_CACHE_MAX_LAYERS];  // quantization scales
} edgelm_kv_cache_t;

// Indexing (critical hot path):
static inline int32_t kv_slot(const edgelm_kv_cache_t* cache, int32_t pos) {
    return pos & cache->mask;  // single AND instruction
}
```

### Critical Implementation Notes

1. **All buffers must be 64-byte aligned** -- use `_aligned_malloc(size, 64)` on Windows
   or `VirtualAlloc` with `MEM_LARGE_PAGES` for 2MB huge pages to reduce TLB pressure.

2. **Pre-allocate everything at startup** -- no dynamic allocation during inference.
   Total allocation: ~120 MB for INT8, allocated once.

3. **Position tracking is absolute** -- the `positions[]` array maps each physical
   slot to its absolute sequence position. RoPE frequencies are derived from these
   absolute positions at attention time.

4. **Write path (per token, per layer):**
   - Compute K, V projections (FP32 output from ternary matmul)
   - Quantize K to INT8 (per-channel scales)
   - Quantize V to INT8 (per-token scales)
   - Write to `k_data[layer][head][slot][dim]`
   - Write to `v_data[layer][head][dim][slot]` (note: transposed write pattern)
   - Update `positions[slot] = write_pos`
   - Increment `write_pos`

5. **Read path (decode attention, per layer):**
   - For each KV head h:
     - Dequantize K[h][0..n_used][0..127] to FP32
     - Apply RoPE using positions[] array
     - Compute Q[group]*K^T (4 Q heads per KV head for GQA)
     - Apply causal mask
     - Softmax
     - Dequantize V[h][0..127][0..n_used] to FP32
     - Compute weighted sum: output = softmax_weights * V

6. **Prefetching:** Issue `_mm_prefetch` for the next head's K/V data while
   processing the current head. Since each head's data is ~256 KB (INT8),
   prefetching 2-3 cache lines ahead in the inner loop helps hide memory latency.

---

## Topic F: Speculative Decoding KV Cache Management

### Finding F1: Cache Rollback is Pointer Truncation, Not Zeroing

**Source:** llama.cpp `examples/speculative/speculative.cpp` source code;
HuggingFace assisted generation blog (huggingface.co/blog/assisted-generation)

**Key idea:** When N draft tokens are speculatively decoded and the verifier
rejects after token K, entries K+1...N are invalidated by **truncating the
cache via sequence removal** (`llama_memory_seq_rm`), NOT by zeroing data.
Stale data remains in memory but is invisible because the sequence length
counter is decremented.

**Technical details:**

In llama.cpp's speculative decoding:
```c
// After verification determines n_accepted tokens out of n_draft:
// Remove entries beyond acceptance point from TARGET model cache
llama_memory_seq_rm(mem_tgt, s_keep, n_past_tgt, -1);  // -1 = to end
llama_memory_seq_cp(mem_tgt, s_keep, 0, -1, -1);       // consolidate

// Remove entries beyond acceptance point from DRAFT model cache
llama_memory_seq_rm(mem_dft, 0, n_past_dft, -1);
llama_memory_seq_keep(mem_dft, 0);                      // keep only seq 0
```

The `seq_rm` function marks cells as unused by clearing sequence membership
bits in the cell metadata -- it does NOT zero the actual K/V tensor data.
The linear allocator's head position determines valid data boundaries.

The `prepare()` function also supports a rollback mechanism: it pre-validates
slot placement without modifying state, stores old cell state in a recovery
stack, tests ubatch placement sequentially, and restores original state if any
ubatch fails placement. Returns all validated positions atomically.

HuggingFace's assisted generation validation loop:
1. Assistant generates N candidate tokens using greedy decoding
2. Main model performs a single forward pass on the full candidate sequence
3. Output logits are compared token-by-token with candidates
4. First mismatch invalidates all subsequent candidates
5. Keep matching tokens plus the first divergent token from main model
6. Adaptive heuristic: start with 5 candidates, +2 on full acceptance, -1 on rejection

**Relevance to EdgeLM:** For our C implementation, rollback is trivially cheap:
just decrement the write position counter. No memset needed. The KV cache write
pointer acts like a stack pointer -- push on speculate, pop on reject. With our
ring buffer design, this is O(1). For single-sequence single-user inference, we
don't even need sequence ID tracking -- just a single `n_past` counter per layer.

```c
// Speculative rollback in EdgeLM -- trivially cheap
void kv_cache_rollback(edgelm_kv_cache_t* cache, int32_t n_accepted) {
    // Just move the write pointer back. Stale data is harmless.
    cache->write_pos -= (n_drafted - n_accepted);
    cache->n_used = (cache->write_pos < cache->capacity)
                  ? cache->write_pos : cache->capacity;
}
```

**Estimated impact:** Zero overhead for rollback (pointer arithmetic only).
**Implementation complexity:** Very low.

---

### Finding F2: Draft and Target Models Use Separate KV Caches

**Source:** llama.cpp speculative implementation (`ctx_dft`, `ctx_tgt` are
independent contexts); HuggingFace `assistant_model` API; EAGLE paper
(arXiv:2401.15077)

**Key idea:** Draft and target models maintain **completely separate** KV caches.
They never share KV entries. After verification, the accepted tokens' KV entries
exist in BOTH caches (computed independently by each model during their respective
forward passes).

**Technical details:**

- **llama.cpp:** Creates two independent `llama_context` objects, each with its own
  KV cache. The draft model runs speculatively filling its cache, then the target
  model runs verification filling its own cache. On rejection, both caches are
  truncated independently.
- **EAGLE approach:** The draft "model" is a lightweight head attached to the TARGET
  model's penultimate layer features (second-to-top layer). It reuses the target's
  hidden states rather than maintaining a separate full KV cache. EAGLE achieves
  2.7x-3.5x speedup on LLaMA2-Chat 70B by operating at the feature level rather
  than token level. The key insight: "autoregression at the feature level is more
  straightforward than at the token level."
- **EAGLE-2:** Extends EAGLE with context-aware dynamic draft trees where tree shape
  adapts based on calibrated confidence scores. Achieves 3.05x-4.26x speedup,
  20-40% faster than EAGLE-1.
- **HuggingFace:** The `assistant_model` creates a fully independent model with its
  own cache. Adaptive heuristic: 5 initial candidates, +2 on full accept, -1 on reject.
  INT8 quantization of the assistant model yields up to 3x speedup (vs 2x at FP16).

**Relevance to EdgeLM:** Since our draft model (running on E-cores) will be much
smaller than the target, its KV cache is negligible. For a hypothetical draft with
6 layers, 4 KV-heads, head_dim=64 at INT8:

```
Draft cache per token: 2 * 6 * 4 * 64 * 1 = 3,072 bytes = 3 KB
Draft cache at 2048 context: ~6 MB
```

Both caches fit easily in our memory budget. The EAGLE approach (using target model
features) is interesting for future optimization but requires architectural coupling
between draft and target that complicates the implementation.

**Estimated impact:** Minor memory overhead (~5% of total KV budget for draft cache).
**Implementation complexity:** Low -- two independent ring buffers.

---

### Finding F3: Tree-Structured Speculation Uses Attention Masks and Index-Based Pruning

**Source:** Medusa source code (`FasterDecoding/Medusa/medusa/model/utils.py`);
SpecInfer paper (arXiv:2305.09781); EAGLE-2 (arXiv:2406.16858)

**Key idea:** Tree-structured speculation generates multiple candidate continuations
as a tree, runs them ALL through the target model in one forward pass using a custom
tree attention mask, then prunes the KV cache to keep only the accepted path.

**Technical details:**

**Tree attention mask (Medusa):**
```python
# Initialize: each position attends to itself + root token
medusa_attn_mask = torch.eye(medusa_len, medusa_len)
medusa_attn_mask[:, 0] = 1   # all positions attend to root

# For each node, enable attention to ancestors:
for node in sorted_tree:
    for ancestor in node.ancestors:
        medusa_attn_mask[node.pos, ancestor.pos] = 1
```

This creates a causal mask that respects the tree topology: siblings cannot
attend to each other, only to shared ancestors.

**KV cache after tree verification (Medusa):**
After the target model processes all tree candidates in one pass, generating KV
entries for ALL branches:
```python
# Select only the accepted path's KV entries
select_indices = get_accepted_path_indices(tree_results)
tgt = past_key_values_data[..., select_indices, :]
dst = past_key_values_data[..., :len(select_indices), :]
dst.copy_(tgt, non_blocking=True)
current_length_data.fill_(prev_input_len + len(select_indices))
```

The cache is compacted by copying accepted entries to contiguous positions and
updating the length tracker.

**SpecInfer tree branching:**
Uses sequence ID cloning: `llama_memory_seq_cp(mem_dft, s, n_seq_cur, -1, -1)`
creates parallel cache branches. After verification, all but the accepted branch
are removed. Achieves 1.5-2.8x for distributed, 2.6-3.5x for offloading.

**Relevance to EdgeLM:** For CPU inference, tree speculation is likely **not worth
the complexity**:
- The GPU benefit comes from batching all tree candidates in ONE forward pass
  (parallel matmuls). On CPU, verification of K tree candidates costs ~K serial
  forward passes -- no batching advantage.
- Simple serial speculation (5-8 draft tokens) is recommended for Phase 3.
- Tree speculation could be reconsidered if we implement batched verification on
  the iGPU (attention over all candidates simultaneously).

**Estimated impact:** Tree adds ~20-30% complexity for marginal CPU benefit.
**Implementation complexity:** High for tree, Low for serial.

---

## Topic G: Multi-threaded KV Cache Access and Write Patterns

### Finding G1: False Sharing is Naturally Avoided by KV Entry Size

**Source:** C++ `hardware_destructive_interference_size` = 64 bytes on x86-64;
Intel Alder Lake L1 cache line = 64 bytes; rigtorp.se/ringbuffer benchmarks

**Key idea:** During decode, thread 0 writes the new K/V entry at position `n_past`
while threads 1-5 read older entries for attention computation. With head_dim=128
at INT8 precision, each KV entry per head is 128 bytes = exactly 2 cache lines.
As long as entries are 64-byte aligned, false sharing between writer and readers
is **impossible** because they occupy completely different cache lines.

**Technical details:**

Per-head KV entry sizes:
| Dtype | Entry Size | Cache Lines | Min Separation |
|-------|-----------|-------------|----------------|
| FP32  | 512 bytes | 8           | 512 bytes      |
| FP16  | 256 bytes | 4           | 256 bytes      |
| INT8  | 128 bytes | 2           | 128 bytes      |
| FP8   | 128 bytes | 2           | 128 bytes      |

Even at INT8 (the smallest we'd use), the writer at position `n_past` writes to
cache lines `[n_past*128, n_past*128+127]`. The nearest concurrent reader at
position `n_past-1` reads from `[(n_past-1)*128, (n_past-1)*128+127]`. These are
128 bytes apart = 2 full cache lines of separation. No MESI coherency traffic.

**Cache line alignment guarantee:** With 64-byte-aligned base buffer and 128-byte
entries, every entry starts at a cache-line boundary automatically.

**The REAL false sharing risk -- metadata:**
```c
// BAD: write_pos on same cache line as read-accessed metadata
struct bad_cache {
    uint32_t write_pos;    // written by producer thread
    uint32_t n_used;       // read by all consumer threads
    // ... both on same 64-byte cache line!
};

// GOOD: each on its own cache line
struct good_cache {
    alignas(64) uint32_t write_pos;  // own cache line
    alignas(64) uint32_t n_used;     // own cache line
};
```

Benchmark data from rigtorp.se: false sharing causes ~6x slowdown (550ms vs 89ms)
for concurrent counter updates. Separating to different cache lines eliminates this.

**Relevance to EdgeLM:** False sharing is a non-issue for KV data buffers due to
natural entry size. The only concern is metadata (write pointer, length counter).
Solution: `alignas(64)` on all shared metadata fields.

**Estimated impact:** Prevents 6x potential slowdown on metadata updates.
**Implementation complexity:** Trivial -- just alignment annotations.

---

### Finding G2: Quantize-on-Write is Optimal for Single-User Decode

**Source:** HuggingFace KV cache quantization blog; KIVI paper (arXiv:2402.02750);
KVQuant paper (arXiv:2401.18079)

**Key idea:** Two strategies exist: (1) quantize-on-write (convert new KV entry to
INT8 immediately) or (2) residual cache (keep recent entries at FP16, batch-quantize
later). For single-user decode at 100+ tok/s, quantize-on-write is clearly better.

**Technical details:**

**Strategy 1 -- Quantize-on-write:**
- Each new KV pair (2 * n_kv_heads * head_dim = 2048 elements) is quantized inline
- Cost: ~2048 elements / 32 per AVX2 vector = 64 SIMD operations per K/V
- Total: 128 SIMD operations per layer * 30 layers = 3840 SIMD ops per token
- At ~1 cycle per op, ~4K cycles = <1 microsecond at 4.7 GHz
- Benefit: entire cache is always INT8, simplifying the attention kernel

**Strategy 2 -- Residual cache (KIVI style):**
- Keep last 128 tokens in FP16, batch-quantize when residual fills up
- Advantage: most-attended recent tokens at full precision
- Disadvantage: mixed-precision attention kernel, more complex code path
- Batch quantization: 128 * 2048 = 262K elements per batch, still fast

**KIVI asymmetric quantization:**
- Keys per-channel: group along dim axis, 128 elements per group for head_dim=128
  → 1 scale + 1 zero-point per head per layer (very few calibration parameters)
- Values per-token: group along seq axis, 2048 elements per group for full context
  → 1 scale + 1 zero-point per position per head per layer
- At INT8: 2-bit overhead per group for scale/zero is negligible

**Affine quantization formula:**
```c
// Per-channel key quantization (applied along dim axis)
for (int d = 0; d < head_dim; d++) {
    float min_val = channel_min[d];
    float max_val = channel_max[d];
    float scale = (max_val - min_val) / 255.0f;
    float zero = roundf(-min_val / scale);
    k_scale[d] = scale;
    k_zero[d] = (int8_t)zero;
    for (int s = 0; s < n_seq; s++) {
        k_quant[s][d] = (int8_t)(roundf(k_float[s][d] / scale) - zero);
    }
}
```

**Relevance to EdgeLM:** Use **quantize-on-write with INT8** for these reasons:
1. <1 us quantization cost per token (negligible vs 10ms decode budget)
2. Simpler attention kernel (single data type throughout)
3. Consistent bandwidth reduction from the first token
4. Per-token for values, per-channel for keys (KIVI recommendation)
5. Pre-RoPE key storage further improves INT8 quantization quality (KVQuant finding)

If quality testing reveals issues, add a small FP16 residual buffer (last 16-32 tokens).

**Estimated impact:** INT8 quantize-on-write saves 50% bandwidth vs FP16, <0.1% latency.
**Implementation complexity:** Low for per-token INT8. Medium for asymmetric per-channel.

---

### Finding G3: SPSC Ring Buffer Techniques -- Cached Indices and Memory Ordering

**Source:** Erik Rigtorp's ring buffer (rigtorp.se/ringbuffer); Linux Journal
lock-free MPMC queue article

**Key idea:** High-performance SPSC ring buffers use cache-line-padded indices,
cached remote index copies, and relaxed memory ordering. These patterns partially
apply to KV cache metadata management.

**Technical details:**

**Rigtorp's optimized SPSC ring buffer:**
```cpp
struct ringbuffer {
    std::vector<int> data_;
    alignas(64) std::atomic<size_t> readIdx_{0};
    alignas(64) size_t writeIdxCached_{0};     // reader's local copy of writeIdx
    alignas(64) std::atomic<size_t> writeIdx_{0};
    alignas(64) size_t readIdxCached_{0};      // writer's local copy of readIdx
};
```

The key optimization: instead of atomically loading the remote index on every
operation, maintain a local cached copy. Only reload the atomic when the cache
indicates the buffer might be full/empty.

**Benchmark results (Rigtorp, cross-chiplet on AMD):**
- Baseline: 5.5M items/sec, ~300M cache misses (3 per operation)
- Optimized: 112M items/sec, ~15M cache misses (0.15 per operation)
- **20x throughput improvement** from cached indices
- Cache miss ratio: 91.2% -> 32.5%

**Memory ordering on x86:**
- x86 has Total Store Order (TSO) -- stores are never reordered with other stores
- Only need `memory_order_release` for writes, `memory_order_acquire` for reads
- A compiler barrier (`asm volatile("" ::: "memory")`) suffices to prevent the
  compiler from caching values in registers
- No full memory fences (`mfence`) needed on x86 for SPSC patterns

**Linux Journal lock-free queue (non-CAS approach):**
- Uses `__sync_fetch_and_add` for atomic position reservation
- Per-thread position tracking avoids global contention
- Deliberately avoids CAS for updating cached min values -- accepts occasional
  stale values rather than paying CAS cost
- 3.7x speedup vs mutex-based queues

**Relevance to EdgeLM:** The KV cache access pattern is simpler than a traditional
SPSC queue:
- Producer writes one entry per decode step
- Consumers read ALL valid entries every step (no consumer pointer advancement)
- Ring wraps only when exceeding max context (oldest entry eviction)

So the cached-index optimization doesn't directly apply (consumers don't advance
a read pointer). But the critical takeaways are:

1. **Pad metadata to cache lines** -- `alignas(64)` on write_pos prevents coherency
   thrashing with reader threads
2. **Use relaxed memory ordering on x86** -- no fences needed between write_pos
   update and data write (TSO guarantees store ordering)
3. **Compiler barriers suffice** -- `asm volatile("" ::: "memory")` prevents the
   compiler from reordering or caching the write_pos update
4. **Power-of-2 sizing** -- enables `pos & mask` instead of modulo division

**Estimated impact:** Proper metadata padding prevents potential 6x slowdown.
**Implementation complexity:** Trivial.

---

## Topic I: Attention Sinks / Two-Region Buffer Design

### Finding I1: StreamingLLM Two-Region Cache -- 4 Sinks + Circular Window

**Source:** StreamingLLM paper (arXiv:2309.17453, ICLR 2024); MIT-HAN-Lab
implementation (`streaming-llm/streaming_llm/kv_cache.py`)

**Key idea:** LLMs exhibit "attention sinks" -- the first few tokens receive
disproportionately high attention scores regardless of semantic content.
StreamingLLM retains a fixed set of initial "sink" tokens plus a sliding
window of recent tokens, enabling infinite-length generation with fixed memory.

**Technical details:**

**Default configuration:** 4 sink tokens, 512 recent window = 516 total entries.

**Python implementation:**
```python
class StartRecentKVCache:
    def __init__(self, start_size=4, recent_size=512):
        self.start_size = start_size
        self.recent_size = recent_size
        self.cache_size = start_size + recent_size

    def __call__(self, past_key_values):
        seq_len = past_key_values[0][0].size(self.k_seq_dim)
        if seq_len <= self.cache_size:
            return past_key_values  # no eviction needed
        return [
            [
                torch.cat([
                    k[:, :, :self.start_size, ...],             # sink region
                    k[:, :, seq_len - self.recent_size:, ...]   # recent window
                ], dim=self.k_seq_dim),
                torch.cat([
                    v[:, :, :self.start_size, ...],
                    v[:, :, seq_len - self.recent_size:, ...]
                ], dim=self.v_seq_dim),
            ]
            for k, v in past_key_values
        ]

    def evict_for_space(self, past_key_values, num_coming):
        seq_len = past_key_values[0][0].size(self.k_seq_dim)
        if seq_len + num_coming <= self.cache_size:
            return past_key_values
        return [
            [
                torch.cat([
                    k[:, :, :self.start_size, ...],
                    k[:, :, seq_len - self.recent_size + num_coming:, ...]
                ], dim=self.k_seq_dim),
                # ...
            ]
            for k, v in past_key_values
        ]
```

**Performance:** 22.2x speedup over sliding-window-with-recomputation on 4M+ tokens.
Successfully deployed across LLaMA-2, MPT, Falcon, and Pythia models.

**Enhancement:** Adding a dedicated attention sink token (placeholder) during
pre-training further improves quality, but requires training changes.

**Relevance to EdgeLM:** In C, the two-region design maps to a contiguous array
with the sink region at indices [0, n_sink) and the circular window at indices
[n_sink, n_sink + window_size). The write pointer for the window region wraps
using modular arithmetic:

```c
typedef struct {
    // Region 1: Sink tokens (permanent, never evicted)
    uint32_t n_sink;          // typically 4

    // Region 2: Recent window (circular)
    uint32_t window_size;     // power of 2, e.g., 2044
    uint32_t window_mask;     // window_size - 1
    uint32_t window_write;    // write position within window (wraps)

    // Combined
    uint32_t capacity;        // n_sink + window_size
} kv_cache_regions_t;

// Write new entry:
static inline uint32_t kv_cache_write(kv_cache_regions_t* r) {
    if (r->total_written < r->n_sink) {
        // Still filling sink region
        return r->total_written++;
    }
    // Write to circular window region
    uint32_t slot = r->n_sink + (r->window_write & r->window_mask);
    r->window_write++;
    return slot;
}
```

**Overhead vs simple ring buffer:** One extra branch per write (sink vs window).
Attention must iterate over two contiguous ranges instead of one, but both ranges
are memory-contiguous so the overhead is just one extra loop setup (~10 cycles).

**Estimated impact:** Enables infinite context with fixed memory; ~5% overhead.
**Implementation complexity:** Low.

---

### Finding I2: Sink Token Count -- 4 is Robust, 1 Recovers 90%

**Source:** StreamingLLM paper experiments across 8 model families; SCBench
KV cache analysis (arXiv:2412.10319, ICLR 2025)

**Key idea:** The StreamingLLM paper tested sink counts of 1, 2, 4, and 8
across LLaMA-2 (7B/13B/70B), Falcon (7B/40B), MPT (7B/30B), and Pythia
(1.4B-12B). 4 sink tokens is robust, with diminishing returns beyond 4.

**Technical details:**

| Sink Count | Performance Recovery | Notes |
|-----------|---------------------|-------|
| 0         | Catastrophic failure | Perplexity explodes |
| 1         | ~90% of full cache   | First token is critical |
| 4         | ~98-99%              | Recommended default |
| 8         | ~99%                 | Negligible improvement over 4 |

**Why attention sinks exist:** The first token acts as a "garbage collector" for
attention mass. Softmax requires attention weights to sum to 1.0. When the model
doesn't know where to attend, unused probability mass concentrates on the first
token. This is a property of **softmax normalization**, not of the specific model
weights -- so it should be present in BitNet models too.

**BitNet-specific consideration:** Attention sinks have been validated only on
standard FP16/BF16 models. BitNet's ternary weights produce different attention
distributions (potentially more uniform due to constrained weight values). The
phenomenon should still exist (softmax property), but the optimal sink count may
differ. **Empirical testing required.**

**SCBench findings (ICLR 2025):** Sub-O(n) memory methods (including StreamingLLM-style
approaches) suffer in multi-turn scenarios. Dynamic sparsity outperforms static
patterns. Layer-level sparsity in hybrid architectures effectively reduces memory.

**Relevance to EdgeLM:** Use 4 sink tokens as default, make it configurable. Memory
cost: 4 * 60 KB (INT8) = 240 KB (negligible). Test with BitNet models to validate.

**Estimated impact:** Correct sink count prevents perplexity degradation.
**Implementation complexity:** Trivial -- one config parameter.

---

### Finding I3: Heterogeneous Precision -- FP16 Sinks + INT8 Window

**Source:** HuggingFace KV cache quantization blog; KIVI residual cache design

**Key idea:** Sink tokens are read EVERY decode step (permanent), so quantization
error accumulates over the entire sequence. Recent window tokens are read until
evicted. This asymmetry suggests keeping sinks at higher precision.

**Technical details:**

**Heterogeneous precision strategy:**
- Sink tokens (4 entries): FP16 = 256 bytes/token/layer
- Window entries: INT8 = 128 bytes/token/layer
- Extra memory for FP16 sinks: 4 * (256-128) * 30 layers = 15,360 bytes = 15 KB
- This is negligible (<0.02% of total cache)

**KIVI residual cache pattern:**
- Keep last 128 tokens in full precision, quantize older tokens
- Naturally gives the most-attended tokens higher precision
- For EdgeLM: could combine with two-region design:
  - Sink tokens: FP16 (permanent, always attended)
  - Last 128 of window: FP16 (most-attended recent)
  - Older window: INT8 (less critical)

**Alternative:** Uniform INT8 everywhere. KIVI shows INT2 works for many tasks,
so INT8 sinks should be perfectly fine. Start with uniform INT8, add heterogeneous
precision only if quality testing demands it.

**Relevance to EdgeLM:** Start with uniform INT8. The quality impact of INT8 sinks
is likely negligible for our models. If needed, promoting sinks to FP16 costs only
15 KB extra memory.

**Estimated impact:** FP16 sinks add negligible memory, may improve long-context quality.
**Implementation complexity:** Low.

---

## Topic J: Adjacent Domain and Novel Techniques

### Finding J1: PagedAttention -- OS-Inspired Block Management (Reference)

**Source:** vLLM paper (arXiv:2309.06180, SOSP 2023); vllm.ai blog

**Key idea:** PagedAttention partitions KV cache into fixed-size blocks (like OS
memory pages), maps logical positions to physical blocks via a page table, and
allocates on demand. Reduces memory waste from 60-80% to <4%.

**Technical details:**
- Each block stores KV for a fixed number of tokens (typically 16)
- Block table: maps (seq_id, logical_block) -> physical_block_ptr
- On-demand allocation from a free list
- Copy-on-Write for shared prefixes (reference counting)
- For parallel sampling (beam search): up to 55% memory savings via sharing
- vLLM: 24x over HuggingFace, 3.5x over TGI

**vAttention improvement (ASPLOS 2025):** Uses CUDA virtual memory APIs to maintain
contiguous virtual addresses mapped to non-contiguous physical pages. Achieves
1.23x over PagedAttention without modifying attention kernels.

**Relevance to EdgeLM:** PagedAttention is designed for multi-tenant GPU serving
with hundreds of concurrent sequences. For single-user CPU inference, it is overkill.
One sequence = one contiguous buffer, zero fragmentation. However, the block concept
is useful for potential NVMe offloading (swap blocks to/from NVMe in fixed-size units).

**Estimated impact:** N/A for single-sequence. Useful for NVMe offloading design.
**Implementation complexity:** N/A for initial design.

---

### Finding J2: RadixAttention -- Prefix Caching via Radix Trees

**Source:** SGLang system (lmsys.org blog, 2024-01-17)

**Key idea:** RadixAttention uses a radix tree (compressed trie) to map token
sequences to cached KV tensors. When a new request shares a prefix with a previous
one, matching KV entries are reused without recomputation.

**Technical details:**
- **Data structure:** Radix tree with edges labeled by token sequences (not single
  tokens). Each node points to KV cache blocks on GPU. Tree stored on CPU (low overhead).
- **Page-level granularity:** One page per token.
- **Automatic matching:** Runtime receives full prompts, automatically finds prefix match.
- **LRU eviction:** Leaf nodes evicted in LRU order when memory is full.
- **Compatible** with continuous batching and PagedAttention.

**Relevance to EdgeLM:** For single-user, the full radix tree is unnecessary. But
**prefix caching across conversation turns** is directly valuable:

```c
// Simple prefix reuse for multi-turn conversation
typedef struct {
    uint32_t* cached_token_ids;   // token IDs of cached prefix
    uint32_t  cached_length;      // number of cached tokens
    // KV cache already contains these entries at positions [0, cached_length)
} prefix_cache_t;

// On new turn: compare new prompt tokens with cached prefix
uint32_t prefix_match = 0;
while (prefix_match < cached_length &&
       prefix_match < new_prompt_length &&
       cached_token_ids[prefix_match] == new_prompt_tokens[prefix_match]) {
    prefix_match++;
}

if (prefix_match > 0) {
    // Skip KV computation for the first prefix_match tokens
    // Start processing from position prefix_match
    kv_cache->write_pos = prefix_match;
}
```

This avoids re-processing the system prompt and conversation history on each turn.
For a 500-token system prompt at 2ms/token prefill, that saves ~1 second per turn.

**Estimated impact:** Saves seconds of prefill latency in multi-turn conversations.
**Implementation complexity:** Low for simple prefix matching.

---

### Finding J3: Novel KV Compression Techniques (2024-2025)

**Source:** MiniCache (arXiv:2405.14366); PyramidKV (arXiv:2406.02069);
PyramidInfer (ACL 2024); Cross-Layer Attention (arXiv:2405.12981);
ThinK (arXiv:2407.21018, ICLR 2025 Spotlight); DeepSeek-V2 MLA (arXiv:2405.04434)

**Key idea:** Multiple 2024-2025 papers demonstrate KV cache compression far beyond
simple quantization, exploiting structural redundancies at the layer, channel, and
token level.

**Technical details:**

**MiniCache -- Adjacent-Layer KV Merging:**
- KV states in middle-to-deep layers are highly similar between adjacent layers
- Decompose into magnitude + direction; interpolate directions between layers
- Token retention strategy keeps distinct pairs unmerged
- Result: 5.02x compression, ~5x throughput, 41% memory reduction (LLaMA-2-7B)

**PyramidKV -- Layer-Adaptive Cache Allocation:**
- "Pyramidal Information Funneling": attention scatters widely in lower layers,
  consolidates in upper layers
- Allocate more KV entries to lower layers, fewer to upper layers
- Full-cache performance with only 12% of KV entries on LongBench
- LLaMA-3-70B: 100% needle retrieval with just 128 KV entries

**PyramidInfer -- Layer-Wise Importance Decay:**
- Crucial KV pairs decrease layer by layer
- Selective retention per layer: 2.2x throughput, 54% memory reduction

**Cross-Layer Attention (CLA):**
- Shares K/V heads between adjacent layers (extending GQA across depth)
- 2x additional cache reduction over MQA at comparable accuracy
- Tested on 1B-3B models (our scale!) trained from scratch

**ThinK -- Channel-Level Pruning:**
- ~90% of attention weight focuses on a subset of key channels
- Query-driven pruning of less-important channels: 20% cache reduction
- Combined with KIVI INT2: 2.8x total reduction, up to 5x batch size

**DeepSeek-V2 MLA (Multi-head Latent Attention):**
- Compresses KV cache into a low-rank latent vector
- 93.3% reduction in KV cache requirements
- Requires architectural support (not applicable to existing models)

**SparQ Attention -- Selective KV Loading:**
- Predict which KV entries will have high attention weight
- Only load those entries from memory: up to 8x bandwidth savings
- No model modification needed; works on LLaMA-2/3, Mistral, Gemma

**Relevance to EdgeLM:** For our bandwidth-constrained DDR4 system, these are
Phase 5 optimizations:

1. **SparQ selective loading** -- Most immediately applicable. During attention,
   do a quick approximate dot product with a subset of key dimensions to predict
   which KV entries are important, then load only those. Could reduce attention
   bandwidth by 4-8x. Works without model changes.

2. **PyramidKV layer-adaptive allocation** -- Give more slots to lower layers,
   fewer to upper layers. Easy to implement, reduces total cache size.

3. **MiniCache layer merging** -- Worth testing if BitNet shows adjacent-layer
   similarity. Could halve the cache by sharing between layer pairs.

4. **CLA/MLA** -- Require architectural changes, not applicable to current models.

**Estimated impact:** 2-8x additional compression beyond INT8 (Phase 5+).
**Implementation complexity:** Medium-High per technique.

---

### Finding J4: H2O -- Heavy Hitter Oracle for Eviction

**Source:** H2O paper (arXiv:2306.14048)

**Key idea:** Instead of FIFO eviction (ring buffer), keep "Heavy Hitter" tokens
that contribute most to attention scores. Formulated as dynamic submodular
optimization with theoretical guarantees.

**Technical details:**
- Heavy Hitters (H2): ~20% of tokens receive the majority of attention weight
- Eviction: maintain cumulative attention scores, evict lowest-scoring token
- Dynamic balance between recent tokens and historically important tokens
- Results: comparable quality to full cache with 20% heavy hitters
- Throughput improvement: up to 29x vs DeepSpeed, 1.9x latency reduction
- **Attention sinks are naturally Heavy Hitters** -- first tokens always score high

**Algorithm sketch:**
```c
// Per layer, per head:
float cumulative_score[MAX_SEQ];   // running attention score accumulator

// After each attention computation:
for (int s = 0; s < n_used; s++) {
    cumulative_score[s] += attention_weights[s];
}

// When eviction needed (cache full):
int victim = find_min_score(cumulative_score, n_used, keep_last_N=32);
evict_token(cache, victim);
```

**Relevance to EdgeLM:** H2O is a superior eviction policy to our simple two-region
design (sinks + FIFO window). However, it adds per-step overhead:
- Memory: 4 bytes/token/layer for cumulative scores = ~240 KB for 2048 context
- Compute: score accumulation after attention (cheap), min-finding for eviction
  (occasional, amortizable)
- Complexity: Need to compact the cache when evicting non-tail entries

For Phase 3, the two-region design is simpler and good enough. H2O is a Phase 5
enhancement if long-context quality is insufficient.

**Estimated impact:** Better quality than FIFO for long contexts.
**Implementation complexity:** Medium.

---

### Finding J5: TOVA and Keyformer -- Attention-Guided Cache Compression

**Source:** TOVA (arXiv:2401.06104); Keyformer (MLSys 2024)

**Key idea:** Both TOVA and Keyformer use attention scores to identify and retain
only the most important tokens in the KV cache. TOVA achieves 8x compression
(1/8 cache) with near-lossless quality. Keyformer: 2.1x latency reduction.

**Technical details:**

**TOVA (Token Omission Via Attention):**
- Views transformers as "unbounded multi-state RNNs"
- Converts to bounded variants by constraining KV cache size
- Token selection based on cumulative attention across layers and heads
- 8x compression with near-full performance
- 4.8x throughput improvement
- Training-free, works on any pre-trained model

**Keyformer:**
- "~90% of attention weight focuses on a specific subset of 'key' tokens"
- Novel scoring function to identify these key tokens
- Selectively preserves only key tokens in cache
- 2.1x latency reduction, 2.4x token generation throughput
- Works on GPT-J, Cerebras-GPT, MPT (MLSys 2024)

**Relevance to EdgeLM:** For our 2048-context target with INT8 KV cache (120 MB),
memory is not the constraint -- bandwidth is. These techniques reduce the amount
of data read during attention, directly reducing bandwidth consumption. At 8x
compression, we'd read only ~15 MB per decode step instead of 120 MB -- bringing
attention bandwidth from 3ms to <0.4ms. However, implementing this requires tracking
attention scores and dynamically managing a variable-size cache, which adds
significant complexity.

**Estimated impact:** Up to 4.8x attention bandwidth reduction (Phase 5+).
**Implementation complexity:** Medium-High.

---

### Finding J6: Cake -- Hybrid Compute+I/O for Prefix KV Loading

**Source:** Cake paper (arXiv:2410.03065)

**Key idea:** For long-context prefill, Cake parallelizes KV cache computation
and I/O loading simultaneously. A bidirectional scheduler dynamically balances
compute vs storage loading, achieving 2.6x reduction in Time-to-First-Token.

**Technical details:**
- When prefix KV cache exists on NVMe/SSD, loading it takes time (I/O bound)
- Computing it from scratch takes time (compute bound)
- Cake does BOTH in parallel: compute some layers while loading others
- Adaptive scheduling based on current I/O bandwidth and compute availability
- 2.6x average TTFT reduction over compute-only or I/O-only approaches

**Relevance to EdgeLM:** Directly relevant for our NVMe SSD (PCIe 4.0, ~5-6 GB/s).
For multi-turn conversations, we could:
- Save the previous turn's KV cache to NVMe between turns
- On new turn: load matching prefix KV from NVMe (I/O) while computing
  non-matching portion (compute)
- With 5 GB/s NVMe and 120 MB INT8 cache: load time = 24ms

For our 100+ tok/s scenario, the KV cache is small enough that NVMe loading is
already fast. But for longer contexts (4096+), the Cake approach could help.

**Estimated impact:** Faster multi-turn conversation startup (Phase 5).
**Implementation complexity:** Medium.

---

## Updated Summary of Recommendations

### Architecture Decisions (Complete)

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Buffer type | Two-region ring buffer | 4 sinks + circular window (StreamingLLM) |
| Capacity | 2048 (power-of-2 window) | `pos & mask` bitmask indexing |
| K storage | Pre-RoPE | Better quantization, simpler ring buffer |
| K layout | [layer][head][seq][dim] | Contiguous per-head L2 access |
| V layout | [layer][head][dim][seq] | Transposed for efficient weighted sum |
| K/V separation | Separate arrays (SoA) | Independent access patterns |
| Quantization | INT8 quantize-on-write | Per-channel K, per-token V (KIVI) |
| Alignment | 64-byte aligned | AVX2 + cache line false sharing avoidance |
| Metadata padding | alignas(64) per field | Prevent false sharing on write_pos |
| Sink tokens | 4 (configurable) | StreamingLLM default, robust across models |
| Spec decoding | Separate draft/target caches | Independent ring buffers |
| Spec rollback | Decrement write_pos | O(1), no zeroing needed |
| Prefix caching | Simple prefix match | Reuse KV across conversation turns |
| Eviction | FIFO (ring) for Phase 1-3 | H2O/TOVA for Phase 5 if needed |

### Memory Budget (Final)

For BitNet-b1.58-2B-4T (30 layers, 8 KV heads, head_dim=128, capacity=2048):

| Component | Size | Notes |
|-----------|------|-------|
| Target KV cache (INT8) | 120 MB | Main inference cache |
| Target sink region (FP16) | 15 KB | 4 sink tokens (optional upgrade) |
| Draft KV cache (INT8) | ~6 MB | For speculative decoding |
| Position array | 8 KB | 2048 * sizeof(int32_t) |
| Quantization scales | ~480 KB | Per-channel/per-token |
| Attention score tracking | ~240 KB | For H2O eviction (Phase 5) |
| **Total** | **~127 MB** | Well within 6-7 GB budget |

### Phase-by-Phase Implementation Plan

**Phase 1 (Naive):**
- Simple contiguous array, FP32, no eviction, max 512 context
- Focus on correctness, not performance

**Phase 2 (Optimized):**
- Two-region ring buffer (4 sinks + circular INT8 window)
- Quantize-on-write with KIVI-style asymmetric quantization
- Pre-RoPE key storage, transposed V layout
- 64-byte aligned everything, alignas(64) on metadata

**Phase 3 (Speculative Decoding):**
- Separate target + draft KV caches
- Rollback via write_pos decrement (O(1))
- Serial speculation (5-8 tokens)
- Simple prefix caching for multi-turn conversations

**Phase 5 (Advanced):**
- SparQ-style selective KV loading (bandwidth reduction)
- H2O/TOVA attention-guided eviction (quality improvement)
- PyramidKV layer-adaptive allocation (memory reduction)
- NVMe KV cache persistence (Cake-style parallel load/compute)

---

## Topic C: Allocation Strategy -- Large Pages, Pre-allocation, Arena

### Finding C1: Windows Large Pages -- VirtualAlloc + MEM_LARGE_PAGES for KV Cache

**Source:** Microsoft Learn "Large-Page Support" (learn.microsoft.com/en-us/windows/win32/memory/large-page-support)

**Key idea:** Windows supports 2 MB large pages via `VirtualAlloc` with `MEM_LARGE_PAGES`.
Each large page uses a single TLB entry, covering 512x more address space than a 4 KB
page. For a 120-150 MB KV cache, standard 4 KB pages require ~30,000-38,000 TLB entries;
2 MB large pages require only 60-75 entries, eliminating TLB misses entirely.

**Technical details:**

Requirements and constraints:
- Must obtain `SeLockMemoryPrivilege` via `AdjustTokenPrivileges()`. User account must
  be in the "Lock pages in memory" group (configured via Local Security Policy / secpol.msc).
- Allocation size and alignment must be multiples of `GetLargePageMinimum()` (2 MB on x86-64).
- Memory is always non-pageable (always resident in physical RAM) -- no soft page faults
  during generation.
- Memory is NOT part of the working set (it's part of process private bytes but
  non-pageable, so the OS never pages it out).
- Cannot reserve then commit separately -- must do both in single call
  (`MEM_COMMIT | MEM_RESERVE | MEM_LARGE_PAGES`).
- **Fragmentation risk:** After long system uptime, 2 MB contiguous physical pages may
  be scarce. Microsoft recommends: "applications should avoid making repeated large-page
  allocations and instead allocate all large pages one time, at startup."
- Large-page allocations are not subject to job limits.

TLB analysis for Golden Cove i7-12700H:
- L1 DTLB: ~64 entries for 4 KB pages, ~32 entries for 2 MB pages
- L2 TLB: 2048 entries for 4 KB pages, 1024 entries for 2 MB pages
- 120 MB KV cache at 4 KB pages: 30,720 entries needed -- massive TLB pressure, frequent
  L2 TLB misses (only 2048 entries available), requires page table walks (~30-50 cycles each)
- 120 MB KV cache at 2 MB pages: 60 entries needed -- fits entirely in L2 TLB (1024 entries
  available), zero TLB misses

Expected performance impact:
- **Sequential access pattern** (scanning KV within a head): Hardware prefetcher
  triggers TLB prefetching, partially mitigating TLB misses with 4 KB pages. Large pages
  still help by ~3-5% due to eliminating page-boundary stalls.
- **Stride/cross-head access** (jumping between heads): With [head][seq][dim] layout
  and 256 KB per head (INT8), each head starts at a different page range. Large pages
  prevent TLB thrashing during head transitions.
- **Overall estimate:** 3-8% attention throughput improvement, more significant for
  longer context lengths.

```c
// KV cache allocation with large page support
static void* alloc_kv_buffer(size_t size) {
    // Enable SeLockMemoryPrivilege
    HANDLE token;
    OpenProcessToken(GetCurrentProcess(), TOKEN_ADJUST_PRIVILEGES | TOKEN_QUERY, &token);
    TOKEN_PRIVILEGES tp;
    LookupPrivilegeValue(NULL, SE_LOCK_MEMORY_NAME, &tp.Privileges[0].Luid);
    tp.PrivilegeCount = 1;
    tp.Privileges[0].Attributes = SE_PRIVILEGE_ENABLED;
    AdjustTokenPrivileges(token, FALSE, &tp, 0, NULL, NULL);
    CloseHandle(token);

    // Round up to large page multiple
    SIZE_T lp_min = GetLargePageMinimum();  // 2 MB
    SIZE_T alloc_size = (size + lp_min - 1) & ~(lp_min - 1);

    // Try large pages first
    void *ptr = VirtualAlloc(NULL, alloc_size,
        MEM_COMMIT | MEM_RESERVE | MEM_LARGE_PAGES, PAGE_READWRITE);
    if (ptr) return ptr;  // Success: physically contiguous, TLB-friendly

    // Fallback to standard pages (still page-aligned > 64-byte AVX2 requirement)
    return VirtualAlloc(NULL, size, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
}
```

**Relevance to EdgeLM:** Implement as a low-complexity optimization with fallback.
The primary benefit is guaranteed physical residency (zero page faults) and TLB
elimination. Since we already need VirtualAlloc for alignment, adding MEM_LARGE_PAGES
is minimal extra complexity.

---

### Finding C2: Contiguous Arena Pattern with Computed Offsets

**Source:** Synthesis of llama.cpp allocation patterns and arena allocator design

**Key idea:** Allocate the entire KV cache as two contiguous blocks (K arena + V arena),
then use computed offsets for per-layer per-head access. This guarantees no fragmentation,
perfect alignment, and enables large pages with a single allocation call each.

**Technical details:**

For BitNet-2B-4T at INT8, 4096 context:
```
K arena: [30 layers][5 KV-heads][4096 tokens][128 dim] = 78,643,200 bytes = 75.0 MB
V arena: [30 layers][5 KV-heads][128 dim][4096 tokens] = 78,643,200 bytes = 75.0 MB (transposed)
Total: 150.0 MB in 2 allocations (or 1 combined allocation)
With MEM_LARGE_PAGES: rounds to 76 MB + 76 MB = 152 MB (76 large pages)
```

Offset calculation (zero overhead, compile-time strides):
```c
typedef struct {
    void    *k_base;     // K arena: [layer][head][token][dim]
    void    *v_base;     // V arena: [layer][head][dim][token]
    int32_t  n_layers;
    int32_t  n_kv_heads;
    int32_t  head_dim;
    int32_t  max_ctx;    // power of 2
    int32_t  ctx_mask;   // max_ctx - 1
    size_t   head_stride_k;  // max_ctx * head_dim
    size_t   layer_stride_k; // n_kv_heads * head_stride_k
    size_t   head_stride_v;  // head_dim * max_ctx
    size_t   layer_stride_v; // n_kv_heads * head_stride_v
} kv_arena_t;

// K cache access: K[layer][head][token][dim]
static inline void* k_ptr(kv_arena_t *a, int layer, int head, int token) {
    return (uint8_t*)a->k_base
        + layer * a->layer_stride_k
        + head  * a->head_stride_k
        + token * a->head_dim;
}

// V cache access: V[layer][head][dim][token] (transposed)
static inline void* v_ptr(kv_arena_t *a, int layer, int head, int dim_offset) {
    return (uint8_t*)a->v_base
        + layer * a->layer_stride_v
        + head  * a->head_stride_v
        + dim_offset * a->max_ctx;
}
```

Alignment analysis:
- VirtualAlloc returns page-aligned (4096-byte) -- exceeds 64-byte AVX2 requirement
- With MEM_LARGE_PAGES: 2 MB aligned
- Per-head K block: 4096 * 128 = 524,288 bytes = 512 KB (naturally aligned to 512 KB
  within the arena since it's at a multiple of 512 KB from base)
- Ring buffer indexing: `slot = write_pos & ctx_mask` (single AND instruction)

**Relevance to EdgeLM:** This is the recommended allocation pattern. Two VirtualAlloc
calls at startup, computed offsets for all access. Zero dynamic allocation during
inference. Integrates naturally with ring buffer (bitmask indexing) and large pages.

---

## Topic D: NVMe Offloading for Extended Context

### Finding D1: LLM in a Flash -- Flash Storage Read Characteristics

**Source:** "LLM in a Flash" (Apple, arXiv:2312.11514)

**Key idea:** Apple's paper maps flash storage characteristics to inference workloads.
Flash read performance is highly dependent on read size and pattern: sequential reads
achieve 6+ GB/s but sparse random reads drop to ~1.25 GB/s. The minimum efficient read
size is 32 KB -- below this, latency-to-first-byte dominates throughput. Row-column
bundling doubles chunk sizes to improve throughput.

**Technical details:**

Flash read performance on Apple M1 Max SSD:
- Sequential: 6+ GB/s
- Sparse random: ~1.25 GB/s
- 32 parallel threads amortize per-request latency
- Direct I/O (F_NOCACHE / O_DIRECT) essential to bypass filesystem cache

Latency breakdown for OPT 6.7B on M1 Max CPU (FP16):
- I/O (flash -> DRAM): 105 ms per token
- Memory management (neuron add/delete): 57-58 ms
- Compute (forward pass): 506 ms
- Total: 669 ms/token (vs 3182 ms naive = 4.75x speedup)

Windowing technique (reduces DRAM requirements):
- Sliding window of last k=4-5 tokens predicts needed neurons
- FFN DRAM requirement drops from 10% to 2.4% of model with k=4
- Total DRAM: ~52% of model size (vs 100% for full model)

Row-column bundling:
- Co-locate up-projection column i and down-projection row i on flash
- Doubles read chunk from `d_model * bytes` to `2 * d_model * bytes`
- "Sometimes beneficial to read more than needed in larger chunks, then discard"

Pre-allocation strategy:
- Pre-allocate per-layer DRAM buffers: `Req_i * 2 * d_model` elements
- Neuron deletion: O(c) ID + O(c * d_model) rewrite (swap with last, no realloc)
- Neuron addition: O(1) append to end of pre-allocated buffer

**Relevance to EdgeLM:** These techniques target weight loading (FFN neurons), not KV
cache directly. But transferable principles:
1. **32 KB minimum read size** for flash applies to KV page-in as well
2. **Pre-allocation** of known-size DRAM buffers matches our arena pattern
3. **Direct I/O** on Windows: `FILE_FLAG_NO_BUFFERING` with CreateFile
4. For EdgeLM's primary use case, KV cache fits in DRAM; flash offloading is stretch goal

---

### Finding D2: FlexGen -- KV Cache Compression and Three-Tier Scheduling

**Source:** FlexGen (arXiv:2303.06865, Sheng et al.)

**Key idea:** FlexGen demonstrates that KV cache can be compressed to 4 bits using
group-wise asymmetric quantization (g=64) with negligible accuracy loss. Its three-tier
scheduling (GPU/CPU/disk) with zig-zag block schedule achieves 100x throughput via I/O
overlap. Key insight for EdgeLM: 4-bit KV cache is validated at scale.

**Technical details:**

4-bit KV cache compression:
- Group-wise asymmetric: `x_quant = round((x - min)/(max - min) * 15)` per g=64 elements
- Key cache grouped along channel dimension, value cache along hidden dimension
- OPT-175B accuracy: Lambada 0.758->0.756, WikiText PPL 10.82->10.94
- Memory overhead: 4 bytes (min/max FP16) per 64 elements = 6.25%

KV cache dominance at scale:
- OPT-175B batch=512, in=512, out=32: KV cache peak = **1.2 TB** (3.8x model weights!)
- This extreme ratio doesn't apply to EdgeLM (single user, short context), but validates
  that KV cache quantization is essential for scaling

CPU attention delegation:
- When KV cache resides on CPU, computing attention on CPU saves more I/O than
  transferring KV to GPU
- Beneficial when sequence length >= 512
- For EdgeLM: KV cache IS on CPU, validating our all-CPU approach

Zig-zag block schedule overlaps 6 concurrent operations per compute step, achieving
~2x throughput from I/O hiding.

**Relevance to EdgeLM:** FlexGen validates: (1) 4-bit KV cache with g=64 grouping
works with negligible loss, (2) CPU attention computation is viable when KV is in
CPU memory, (3) overlap of I/O and compute matters for throughput. For EdgeLM's primary
case, all KV is in DRAM so disk I/O overlap is irrelevant. But 4-bit KV quantization
(INT4 with g=64) is a validated option for extending context to 8K-16K if needed.

---

### Finding D3: NVMe Latency Analysis -- Offloading Not Viable at 100+ tok/s

**Source:** DirectStorage documentation (optimal 32-64 KB reads); LLM in a Flash
(1.25 GB/s random reads); NVMe specification analysis

**Key idea:** PCIe 4.0 NVMe random read latency of 10-25 us per request means that
paging KV cache from NVMe during active decode is too slow for the 100+ tok/s target
(10 ms budget per token). All actively-accessed KV data must be in DRAM during decode.
NVMe is only viable for cold storage (session persistence, evicted context).

**Technical details:**

NVMe random read performance (typical PCIe 4.0):
- 4 KB random: ~700K-1M IOPS at QD32, ~10-15 us latency at QD1
- 64 KB random: ~100-150K IOPS, ~15-25 us at QD1
- DirectStorage finding: "performance has a large jump starting with 32 KB reads and
  starts to top out with 64 KB" -- NVMe reads in 64 KB-aligned internal blocks
- Reading 4 KB wastes 60 KB of internal bandwidth (drive reads full 64 KB block)

Page-in analysis for one decode step (BitNet-2B-4T, 30 layers, INT8):
- Per layer, 5 KV-heads, each head: 128 bytes/dim * ctx tokens
- If paging 64 KB blocks (512 tokens per block at INT8 head_dim=128):
  - Worst case (all layers, all heads cold): 30 * 5 * 2 (K+V) = 300 pages
  - At 64 KB per page, 1.25 GB/s random: 300 * 64 KB = 19.2 MB, ~15 ms
  - **Exceeds 10 ms token budget** -- not viable for active decode

- Best case (one layer paged at a time): 5 * 2 = 10 pages = 640 KB, ~0.5 ms
  - Viable if only one layer's KV is cold, but requires perfect prefetching

Practical NVMe use cases for KV cache:
1. **Session persistence:** Save/load full KV cache between sessions (~150 MB,
   takes ~25 ms at 6 GB/s sequential -- fast enough for startup)
2. **Extended context cold storage:** Evict old tokens (beyond DRAM window) to NVMe,
   page-in during prompt processing (not during decode)
3. **Model weight streaming:** This is where NVMe bandwidth matters most (Section 05)

**Relevance to EdgeLM:** NVMe offloading during active decode is NOT viable at 100+ tok/s.
All KV data must be in DRAM. NVMe is useful only for:
- Startup: loading persisted KV cache for conversation continuation
- Prompt processing: pre-loading extended context tokens into DRAM before decode begins
- Cold storage: saving evicted context for potential future retrieval

---

### Finding D4: Eviction Policies -- StreamingLLM Sinks + Sliding Window

**Source:** StreamingLLM (arXiv:2309.17453); SnapKV (arXiv:2404.14469)

**Key idea:** For context exceeding DRAM capacity, the optimal eviction strategy keeps
"attention sink" tokens (first 4-8 in sequence, which receive disproportionate attention)
plus a sliding window of recent tokens. This simple policy from StreamingLLM achieves
near-full-cache quality with 22.2x speedup for streaming inference.

**Technical details:**

StreamingLLM attention sinks:
- Initial tokens (first 4 typical) receive high attention across all heads/layers
- Removing initial tokens causes catastrophic quality degradation
- Architecture-agnostic: works on LLaMA-2, MPT, Falcon, Pythia without fine-tuning
- Supports up to 4 million tokens in streaming mode with fixed memory
- Formula: keep first S sink tokens + last W window tokens (S=4, W=2048 typical)

SnapKV clustering-based compression:
- Uses observation window at prompt end to predict per-head important positions
- Clusters important KV positions (beyond just recency and sinks)
- 3.6x faster decoding, 8.2x memory improvement
- More complex to implement than StreamingLLM but higher quality retention

Eviction tier recommendation:
```
Tier 1 (DRAM, never evict): attention sinks (first 4-8 tokens)
Tier 2 (DRAM, ring buffer): recent window (last N tokens, configurable)
Tier 3 (NVMe cold storage): everything between sinks and window
```

**Relevance to EdgeLM:** The StreamingLLM "sinks + window" strategy integrates naturally
with our ring buffer design. Implementation: keep a separate small fixed buffer for 4-8
sink tokens (already in Finding A3), plus the ring buffer for recent tokens. This is
already planned in our two-region cache design. For the primary use case (<=4096 ctx),
no eviction is needed. For extended context, this strategy is simple and effective.

---

### Finding D5: Windows Async I/O vs DirectStorage for NVMe KV Paging

**Source:** Microsoft DirectStorage documentation (learn.microsoft.com); Windows
OVERLAPPED I/O API documentation

**Key idea:** DirectStorage on Windows requires D3D12 infrastructure and is designed for
GPU asset streaming. For CPU inference KV cache paging (low IOPS, simple read patterns),
standard Windows `ReadFile` with `OVERLAPPED` and `FILE_FLAG_NO_BUFFERING` is simpler and
sufficient. DirectStorage's benefits (50K IOPS at 5-10% CPU, hardware decompression) are
irrelevant for our use case.

**Technical details:**

DirectStorage analysis (rejected for EdgeLM):
- Requires D3D12 device, command queue, fence infrastructure
- Queue-based batch submission (good for 50K+ IOPS game asset streaming)
- Hardware decompression support (irrelevant for uncompressed KV data)
- CPU target: 50K IOPS at 5-10% of one core
- Notification via ID3D12Fence or StatusArray -- requires D3D12 setup
- Optimal read size 32-64 KB (same as standard NVMe reads)
- **Massive overengineering** for ~10-50 reads per token generation step

Recommended approach -- standard Windows overlapped I/O:
```c
// Open for direct NVMe access
HANDLE hFile = CreateFileA("kv_swap.bin",
    GENERIC_READ | GENERIC_WRITE, 0, NULL, CREATE_ALWAYS,
    FILE_FLAG_OVERLAPPED | FILE_FLAG_NO_BUFFERING, NULL);

// Async read (64 KB pages, sector-aligned)
OVERLAPPED ov = {0};
ov.Offset = page_offset_lo;
ov.OffsetHigh = page_offset_hi;
ov.hEvent = CreateEvent(NULL, TRUE, FALSE, NULL);

ReadFile(hFile, aligned_buf, 65536, NULL, &ov);
// ... compute current work while I/O completes ...
GetOverlappedResult(hFile, &ov, &bytes_read, TRUE);
```

For multiple concurrent reads, use IO Completion Ports:
```c
HANDLE iocp = CreateIoCompletionPort(hFile, NULL, (ULONG_PTR)ctx, 0);
// Multiple concurrent ReadFile calls with different OVERLAPPED structs
// GetQueuedCompletionStatus() to process completions
```

**Relevance to EdgeLM:** Use standard OVERLAPPED I/O with FILE_FLAG_NO_BUFFERING if
NVMe paging is implemented. This provides direct NVMe access, async completion, and
64 KB page alignment -- everything needed for KV cache paging. DirectStorage's gaming-
oriented infrastructure adds no value for our low-IOPS CPU inference workload.

---

## Topic E: Prefetching Strategies for KV Cache Access

### Finding E1: Golden Cove Hardware Prefetcher Handles Sequential KV Scans Well

**Source:** Chips and Cheese "Popping the Hood on Golden Cove" (chipsandcheese.com)

**Key idea:** Golden Cove P-cores have aggressive hardware prefetchers with a 64 bytes/cycle
path to L2 -- one full cache line per cycle. For sequential streaming within a single KV
head, the hardware prefetcher is highly effective after a ~4-8 access warm-up period.
Software prefetching adds value primarily at access pattern transitions (head boundaries,
layer boundaries) and on E-cores which have conservative prefetchers.

**Technical details:**

Golden Cove P-core cache and prefetch:
- L1D: 48 KB, 12-way, 5-cycle latency, 3x 256-bit loads/cycle (96 bytes/cycle)
- L2: 1.25 MB per P-core, 64 bytes/cycle path, ~12-14 cycle latency
- L3: 24 MB shared, 32 bytes/cycle per slice (12 slices), ~40-50 cycle latency
- Hardware prefetcher: detects sequential and stride patterns after ~4-8 accesses
- 512-entry ROB (45% larger than Sunny Cove) hides memory latency during prefetch
- Trades higher latency at ALL cache levels for massive bandwidth

KV cache access pattern analysis:
- Within a head (K scan): sequential access through [token][dim] -- hardware prefetcher
  excels here, easily saturating L2->L1 bandwidth
- Head transition: stride jump of `max_ctx * head_dim` bytes. Hardware prefetcher must
  detect new sequential pattern -- cold-start penalty of ~4-8 cache lines
- Layer transition: jump to different arena region. Full cold-start on new stream.

**Relevance to EdgeLM:** The hardware prefetcher handles ~90-95% of KV cache access well.
Software prefetch ROI is highest at transition points. Quantified: head transition cold-
start = 8 lines * 14 cycles L2 latency = 112 cycles. With 5-32 heads and 26-30 layers,
total transition penalty = ~5000-15000 cycles/token. Software prefetch can reclaim ~70%
of this = ~3500-10500 cycles = ~0.7-2.2 us saved per token.

---

### Finding E2: Prefetch Hint Selection and Distance for KV Cache

**Source:** Intel intrinsics documentation; Golden Cove cache hierarchy analysis

**Key idea:** The four `_mm_prefetch` hints (T0/T1/T2/NTA) target different cache levels.
For KV cache access, use T0 within-head (next few lines), T1 for next-head warm-up,
T2 for next-layer into L3, and avoid NTA (may bypass L2, causing worse latency on actual
loads). Optimal prefetch distance is 7-8 cache lines on P-cores, 14-16 on E-cores.

**Technical details:**

Prefetch hint behavior on Golden Cove:
- `_MM_HINT_T0`: Fetch into L1, L2, L3. Data available in ~5 cycles (L1 hit).
  Use for data needed within ~100 cycles.
- `_MM_HINT_T1`: Fetch into L2 and L3 (skip L1). Data available in ~14 cycles.
  Use for data needed in ~200-500 cycles (next head's KV start).
- `_MM_HINT_T2`: Fetch into L3 only. Data available in ~45 cycles.
  Use for data needed in ~500+ cycles (next layer's KV data).
- `_MM_HINT_NTA`: Non-temporal, streaming buffer or low-priority L1 slot.
  **WARNING:** On some architectures, NTA bypasses L2. Subsequent load instructions
  hit L3 (~45 cycles) instead of L1 (~5 cycles) -- 9x worse latency. Avoid for data
  that will be loaded by regular instructions immediately after.

Prefetch distance calculation:
```
P-core (Golden Cove, 4.7 GHz):
  L2 latency: ~14 cycles
  Processing rate: ~2 cycles per cache line (SIMD dot product)
  Optimal distance: 14 / 2 = 7 cache lines = 448 bytes ahead

E-core (Gracemont):
  L2 latency: ~17 cycles (shared, may be higher under contention)
  Processing rate: ~4 cycles per cache line (128-bit SIMD)
  Optimal distance: 17 / 4 = ~4-5 lines, but conservative prefetcher means
  we should go further: 14-16 lines = ~1 KB ahead
```

Recommended implementation:
```c
// Three-tier prefetch strategy for attention decode
void attention_decode(kv_cache_t *cache, int layer, ...) {
    // Prefetch NEXT LAYER's first head into L3 (layer transition)
    if (layer + 1 < cache->n_layers) {
        for (int i = 0; i < 16; i++)
            _mm_prefetch((char*)k_ptr(cache, layer+1, 0, 0) + i*64, _MM_HINT_T2);
    }

    for (int h = 0; h < cache->n_kv_heads; h++) {
        // Prefetch NEXT HEAD's start into L2 (head transition)
        if (h + 1 < cache->n_kv_heads) {
            for (int i = 0; i < 8; i++) {
                _mm_prefetch((char*)k_ptr(cache, layer, h+1, 0) + i*64, _MM_HINT_T1);
                _mm_prefetch((char*)v_ptr(cache, layer, h+1, 0) + i*64, _MM_HINT_T1);
            }
        }

        // Within-head sequential scan with T0 prefetch
        int dist = is_ecore ? 16 : 8;
        for (int t = 0; t < n_tokens; t++) {
            if (t + dist < n_tokens)
                _mm_prefetch((char*)k_ptr(cache, layer, h, t+dist), _MM_HINT_T0);
            // ... compute dot product for token t ...
        }
    }
}
```

**Relevance to EdgeLM:** This three-tier strategy covers all transition points with
minimal overhead (~50 prefetch instructions per head transition, ~16 per layer
transition). Total overhead per layer: ~50 * n_kv_heads + 16 instructions = ~270
instructions for 2B-4T (5 heads). At ~1 cycle each = negligible cost, ~2-5% benefit
from eliminating transition cold-starts.

---

### Finding E3: E-Core Prefetching is Critical for Hybrid Workloads

**Source:** Chips and Cheese "Gracemont: Revenge of the Atom Cores" (chipsandcheese.com)

**Key idea:** Gracemont E-cores have "conservative prefetchers that aim to save power by
not transferring data unless absolutely necessary, at the expense of bandwidth." If ANY
KV cache work runs on E-cores, software prefetching is essential -- not optional. Without
it, E-core bandwidth utilization drops to ~40-60% of peak.

**Technical details:**

E-core hardware prefetcher characteristics:
- Power-optimized: minimal speculative fetching
- L1D: 32 KB, 3-cycle latency (FASTER than Golden Cove's 5 cycles)
- L2: 2 MB shared per 4-core cluster, 17-cycle latency
- Under 4-core load: effective L2 per core drops to ~512 KB
- Measured DDR4 bandwidth: "paradoxically higher on DDR4 than DDR5 due to conservative
  prefetching" -- the prefetcher doesn't generate enough requests to benefit from DDR5's
  higher bandwidth, making DDR4 sufficient
- 128-bit SIMD (half of P-core 256-bit) -- half vector width per instruction
- Power: 5.72 W per core vs 21.05 W for Golden Cove during encoding workloads

Impact on KV cache access without software prefetch:
- Sequential scan: hardware prefetcher eventually catches up, but with significant
  warm-up penalty per head/layer
- Multi-head iteration: prefetcher resets at each stride change, losing ~20-30 cycles
  per transition vs ~10 on P-core
- Estimated bandwidth loss without SW prefetch: 20-40% vs P-core's ~5-10%

**Relevance to EdgeLM:** If KV cache work is distributed to E-cores (e.g., during Phase 3
multi-threading with P-core/E-core workload split), E-cores MUST have aggressive software
prefetching. Use 16 lines ahead (vs 8 on P-cores) and prefetch both K and V at head
transitions. Alternatively, restrict KV cache attention computation to P-cores only and
use E-cores for less bandwidth-sensitive work (tokenization, sampling, FFN partial
computation).

---

## Topic K: KV Cache Sizing and Budget Analysis

### Finding K1: Corrected Model Configurations from HuggingFace

**Source:** `huggingface.co/microsoft/bitnet-b1.58-2B-4T/config.json`;
`huggingface.co/1bitLLM/bitnet_b1_58-3B/config.json`

**Key idea:** The two target models have significantly different KV cache characteristics.
The 2B-4T model uses GQA (5 KV-heads for 20 Q-heads, 4:1 ratio, head_dim=128), making
its KV cache 4.33x smaller per token than the 3B model which uses full MHA (32 KV-heads,
head_dim=100). Additionally, their max context lengths differ: 4096 for 2B-4T vs 2048
for 3B.

**Technical details:**

| Parameter | BitNet-2B-4T | bitnet_b1_58-3B |
|-----------|-------------|-----------------|
| Layers | 30 | 26 |
| Q-heads | 20 | 32 |
| KV-heads | 5 (GQA 4:1) | 32 (MHA) |
| head_dim | 128 | 100 |
| hidden_size | 2560 | 3200 |
| intermediate_size | 6912 | 8640 |
| max_ctx | 4096 | 2048 |
| activation | relu2 (squared ReLU) | silu (SwiGLU) |
| torch_dtype | bfloat16 | float16 |
| vocab_size | 128,256 | 32,002 |
| RoPE theta | 500,000 | 10,000 |

KV elements per token:
- 2B-4T: `2 * 5 * 128 * 30 = 38,400`
- 3B: `2 * 32 * 100 * 26 = 166,400` (4.33x more)

**NOTE:** Previous analysis in Topics A/B/H assumed 8 KV-heads for the 2B-4T model based
on early documentation. The confirmed config shows **5 KV-heads**, making KV cache even
smaller than previously estimated. The budget tables in this topic use the correct values.

---

### Finding K2: Complete KV Cache Sizing Tables

**Source:** Computed from confirmed model configs

**BitNet-2B-4T** (38,400 elements/token):

| Context | FP16 | INT8 | FP8 | INT4 | INT2 |
|---------|------|------|-----|------|------|
| 512 | 37.5 MB | 18.8 MB | 18.8 MB | 9.4 MB | 4.7 MB |
| 1024 | 75.0 MB | 37.5 MB | 37.5 MB | 18.8 MB | 9.4 MB |
| 2048 | 150.0 MB | 75.0 MB | 75.0 MB | 37.5 MB | 18.8 MB |
| 4096 | 300.0 MB | 150.0 MB | 150.0 MB | 75.0 MB | 37.5 MB |
| 8192 | 600.0 MB | 300.0 MB | 300.0 MB | 150.0 MB | 75.0 MB |
| 16384 | 1200.0 MB | 600.0 MB | 600.0 MB | 300.0 MB | 150.0 MB |

**bitnet_b1_58-3B** (166,400 elements/token):

| Context | FP16 | INT8 | FP8 | INT4 | INT2 |
|---------|------|------|-----|------|------|
| 512 | 162.5 MB | 81.3 MB | 81.3 MB | 40.6 MB | 20.3 MB |
| 1024 | 325.0 MB | 162.5 MB | 162.5 MB | 81.3 MB | 40.6 MB |
| 2048 | 650.0 MB | 325.0 MB | 325.0 MB | 162.5 MB | 81.3 MB |
| 4096 | 1300.0 MB | 650.0 MB | 650.0 MB | 325.0 MB | 162.5 MB |

---

### Finding K3: Full System Memory Budget Allocation

**Source:** Model configs, ternary packing analysis, activation size estimation

**BitNet-2B-4T at INT8, 4096 context (primary target):**

| Component | Size | Notes |
|-----------|------|-------|
| Ternary model weights | ~400 MB | 2.4B params * ~2 bits + scales/overhead |
| Embedding table | ~65 MB | 128,256 * 2560 * 2B (BF16) |
| KV cache (K arena) | 75 MB | 30 * 5 * 4096 * 128 * 1B |
| KV cache (V arena) | 75 MB | 30 * 5 * 128 * 4096 * 1B (transposed) |
| KV quantization scales | ~2 MB | Per-channel K + per-token V |
| Activation buffer | ~50 MB | Peak: hidden * intermediate * 4B (FP32) |
| FFN intermediate | ~27 MB | 6912 elements * 4B * 2 (gate+up) |
| Tokenizer (BPE) | ~5 MB | 128K vocab + merge rules |
| Position tracking | 16 KB | 4096 * 4B |
| Thread stacks | ~20 MB | 14 threads * ~1.5 MB |
| **Total** | **~720 MB** | **10.0% of 7 GB** |
| **Headroom** | **~6.3 GB** | Available for OS, growth, experiments |

**bitnet_b1_58-3B at INT8, 2048 context:**

| Component | Size | Notes |
|-----------|------|-------|
| Ternary model weights | ~600 MB | 3.3B params * ~2 bits + scales/overhead |
| Embedding table | ~200 MB | 32,002 * 3200 * 2B (FP16) |
| KV cache (K + V) | 325 MB | Full MHA, 32 KV-heads |
| Activation + scratch | ~70 MB | Larger hidden/intermediate |
| Overhead | ~30 MB | Tokenizer, threads, etc. |
| **Total** | **~1,225 MB** | **17.0% of 7 GB** |
| **Headroom** | **~5.8 GB** | Still substantial |

**Key takeaway:** Neither model comes close to stressing the 6-7 GB budget. Even the
larger 3B model at full 2048 context uses only 17% of available memory. This means:
1. No need for NVMe offloading in primary use case
2. Extended context (8K-16K) is feasible for 2B-4T model
3. Multiple concurrent KV caches could be supported
4. FP16 KV cache is even affordable (300 MB for 2B-4T @ 4096) if INT8 quality is a concern

---

### Finding K4: Ternary Weights Are Orthogonal to KV Cache Precision

**Source:** BitNet b1.58 paper (arXiv:2402.17764); model configs; BitNet.cpp source

**Key idea:** Despite ternary weights {-1, 0, +1}, activations (K/V projections) remain
at full precision (FP16/BF16). The ternary matmul produces full-precision output because
the input is full-precision. KV cache quantization requirements are identical to standard
models -- the ternary nature neither helps nor hurts KV precision needs.

**Technical details:**

Activation flow in BitNet:
```
Input x (FP16/BF16) -> Quantize to 8-bit -> Ternary matmul (add/subtract of x elements)
    -> Accumulate in FP32 -> Cast to FP16/BF16 -> Output activation
```

K/V projection: `K = x @ W_k` where W_k is ternary but x is FP16/BF16.
Output K has same dynamic range as x (it's a sum of input elements with +1/0/-1 weights).

Confirmed configurations:
- 3B model: `weight_bits=1, input_bits=8, torch_dtype=float16`
- 2B-4T: `torch_dtype=bfloat16, quant_method=bitnet`

BitNet.cpp uses unmodified llama.cpp KV cache (FP16 default, supports Q8_0/Q4_0).
No ternary-specific KV cache optimizations exist in any known implementation.

Standard KV quantization techniques apply unchanged:
- KIVI 2-bit asymmetric (per-channel K, per-token V): validated on LLaMA/Falcon/Mistral
- KVQuant 3-bit with pre-RoPE: <0.1 PPL degradation
- FlexGen 4-bit g=64: Lambada 0.758->0.756
- llama.cpp Q8_0: well-tested on CPU with AVX2

**Relevance to EdgeLM:** INT8 KV cache is safe and recommended. The ternary weight
optimization is completely independent of KV cache design. Our KV cache module (arena
allocation, ring buffer, quantization, prefetching) can be designed and tested without
any dependency on the ternary matmul kernel -- they are orthogonal components.

---

## Final Updated Implementation Priority (All Topics)

1. **Critical:** Pre-allocated contiguous arena via VirtualAlloc (Topics C, B)
2. **Critical:** Ring buffer with power-of-2 bitmask indexing (Topic A)
3. **Critical:** [head][seq][dim] K + [head][dim][seq] V (transposed) layout (Topic B)
4. **Critical:** Pre-RoPE key storage (Topics A, Q)
5. **High:** INT8 quantization (per-channel K, per-token V via KIVI) (Topic Q)
6. **Medium:** Three-tier software prefetch (within-head T0, cross-head T1, cross-layer T2) (Topic E)
7. **Medium:** 2 MB large pages via MEM_LARGE_PAGES with fallback (Topic C)
8. **Low:** INT4 KV quantization for extended context (g=64 grouping) (Topic D)
9. **Low:** E-core aggressive prefetching (16 lines vs 8 on P-core) (Topic E)
10. **Stretch:** NVMe KV persistence via OVERLAPPED I/O (Topic D)
11. **Stretch:** StreamingLLM sinks + window for >max_ctx streaming (Topic D)
12. **Not recommended:** DirectStorage, PagedAttention (Topics D, H)

## Audit Addendum (2026-04-02)

- **Session semantics are a missing systems angle.** Even if v1 stays single
  request, the KV cache design should leave room for:
  - prompt reuse,
  - resumed sessions,
  - and serialized cache metadata

  without forcing a redesign of indices and RoPE position handling.
- **Sliding-window policy needs explicit metadata, not hidden arithmetic.** If
  windowed or sink-token modes are later enabled, the cache must record what the
  logical position means after wrap/eviction.
- **Cache debugging tools will save a lot of time.** A lightweight dump/inspect
  mode for layer/head/token slices would make correctness failures much easier to
  localize.
