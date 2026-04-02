# Section 08: Attention Mechanism (GQA / MHA) -- Extended Research

## Overview

The attention mechanism is the second most compute-intensive component of the transformer forward pass (after FFN), accounting for ~30-50% of per-token latency depending on sequence length. For EdgeLM targeting 100+ tok/s on i7-12700H, attention optimization matters primarily for two reasons: (1) the QKV and output projections are ternary matmuls that consume ~28% of attention time and are bandwidth-bound, and (2) the score computation (QK^T, softmax, score*V) becomes increasingly significant at longer sequence lengths and is compute-bound in FP32.

This section covers Multi-Head Attention (MHA), Grouped-Query Attention (GQA), and all CPU-specific optimizations for the attention pipeline on consumer Intel hardware.

## What the Deep Dive Already Covers

- **MHA vs GQA vs MQA tradeoffs:** KV cache sizes (MHA: 716MB, GQA: 179MB, MQA: 22.4MB for 3B@2048 context FP16)
- **FlashAttention CPU adaptation:** Block-wise processing with online softmax, primarily a prefill optimization
- **Linear attention alternatives:** RWKV, Mamba, RetNet, GLA -- quality gaps narrowing but limited 3B model availability
- **Sliding window attention:** Ring buffer KV, fixed memory, used by Mistral (W=4096)
- **iGPU offload decision framework:** CPU faster for seq_len<512, iGPU may help for seq_len>1024
- **Attention timing breakdown:** QKV projection ~1.4ms, attention compute ~1.4ms, output projection ~0.84ms per 28 layers at seq_len=512
- **Complete forward pass timing:** Target 10ms/token = 100 tok/s, attention + projections = ~3.64ms of ~9-10ms total

## New Findings

### 1. CPU FlashAttention -- State of the Art

#### 1.1 No Production CPU FlashAttention Exists
- **Source:** https://github.com/Dao-AILab/flash-attention
- **Key finding:** As of March 2026, there is NO official CPU implementation of FlashAttention. The Dao-AILab repository targets NVIDIA GPUs exclusively (CUTLASS backend for H100/A100/Ada/Ampere). Triton language has experimental CPU support but is not production-ready.
- **Relevance to EdgeLM:** This is a research gap -- a well-optimized CPU FlashAttention using AVX2 could be a paper contribution.
- **Implementation complexity:** High
- **Why it's hard on CPU:** GPU FlashAttention exploits SRAM (128KB/SM) for tiling; CPU L1 (48KB on Golden Cove) is smaller and shared differently. The algorithm principles (online softmax, block-wise tiling) transfer, but tile sizes and memory access patterns need complete redesign.

#### 1.2 llama.cpp Flash Attention (PR #5021)
- **Source:** https://github.com/ggml-org/llama.cpp/pull/5021
- **Key idea:** llama.cpp added a "Flash Attention" flag that implements tiled attention with online softmax to avoid materializing the full N×N attention matrix.
- **Relevance to EdgeLM:** Reference implementation to study, though it relies on auto-vectorization rather than hand-tuned AVX2 intrinsics.
- **Performance:** Benefits primarily during prefill for long prompts (>2K tokens). Minimal impact on single-token decode.
- **Implementation complexity:** Medium -- the algorithm is straightforward, the SIMD optimization is the hard part.

### 2. GQA Implementation Strategies for CPU

#### 2.1 Zero-Copy KV Broadcasting via Register Reuse
- **Source:** Derived from bitnet.cpp analysis and Section 06/07 research
- **Key idea:** For GQA with 32 Q heads and 8 KV heads (4:1 ratio), load each KV head into AVX2 registers once and reuse across all 4 Q heads in the group. This eliminates 3/4 of KV memory loads.
- **Relevance to EdgeLM:** Directly applicable. BitNet-b1.58-2B-4T uses 32 Q heads / 8 KV heads.
- **Estimated impact:** 4x reduction in KV cache bandwidth during attention computation
- **Implementation complexity:** Low-Medium
- **Details:**
```c
// GQA register broadcasting -- load KV once, reuse 4x
for (int kv_group = 0; kv_group < 8; kv_group++) {
    // Load KV head to registers ONCE (128 dim = 16 YMM registers)
    __m256 K_reg[16], V_reg[16];
    for (int i = 0; i < 16; i++) {
        K_reg[i] = _mm256_load_ps(&K_cache[kv_group * head_dim + i * 8]);
        V_reg[i] = _mm256_load_ps(&V_cache[kv_group * head_dim + i * 8]);
    }
    // Compute attention for 4 Q-heads sharing this KV head
    for (int q_off = 0; q_off < 4; q_off++) {
        int q_head = kv_group * 4 + q_off;
        compute_attention_from_registers(Q[q_head], K_reg, V_reg, output[q_head]);
    }
}
```

#### 2.2 Interleaved Memory Layout for GQA
- **Source:** Derived from cache-aware weight packing research (Section 06)
- **Key idea:** Instead of storing heads separately `[Head0_all][Head1_all]...`, interleave by GQA group so that Q heads and their shared KV head are contiguous in memory, maximizing hardware prefetcher effectiveness.
- **Relevance to EdgeLM:** Medium-high. Requires weight repacking at model load time but improves cache utilization during attention.
- **Estimated impact:** 1.3-1.5x cache hit rate improvement for attention
- **Implementation complexity:** Medium (requires custom memory layout at load time)

#### 2.3 GQA Head Dimension SIMD Alignment
- **Source:** Architecture analysis
- **Key idea:** Head dimension of 128 (used by BitNet-b1.58-2B-4T and Llama-3.2-3B) divides perfectly into AVX2 width: 128 FP32 values = 16 YMM registers. This means Q@K^T dot products require exactly 16 FMA iterations with no tail handling.
- **Relevance to EdgeLM:** Perfect alignment -- no wasted SIMD lanes.
- **Estimated impact:** Clean codegen, no masking overhead
- **Implementation complexity:** Low (natural fit)

### 3. BitNet Attention Specifics

#### 3.1 BitNet Attention Architecture
- **Source:** arXiv:2504.12285 (BitNet b1.58 2B4T Technical Report), arXiv:2402.17764
- **Key finding:** BitNet attention has a critical asymmetry -- QKV projections and output projection use ternary {-1,0,+1} weights (bandwidth-bound), but the attention score computation (QK^T, softmax, score*V) runs in **FP32** for numerical precision.
- **Relevance to EdgeLM:** Cannot optimize attention scores the same way as FFN matmuls. Attention scores are compute-bound, not bandwidth-bound.
- **Estimated impact:** Attention scores account for ~14% of total per-token time
- **Details:**
  - Q/K/V projections: ternary matmul via I2_S kernel (VPMADDUBSW)
  - Activations quantized to INT8 per-token absmax before projections
  - RoPE applied after projection in FP32
  - Softmax: FP32 (critical for quality -- cannot be ternary-quantized)
  - Output projection: ternary matmul back to hidden dimension

#### 3.2 BitNet Uses Squared ReLU, Not SiLU
- **Source:** arXiv:2402.17764
- **Key idea:** BitNet b1.58 uses ReLU²(x) = max(0,x)² instead of SiLU. This is relevant because the FFN after attention uses this cheaper activation: just 2 SIMD instructions (`_mm256_max_ps` + `_mm256_mul_ps`) vs SiLU's expensive sigmoid.
- **Relevance to EdgeLM:** Simplifies FFN kernel; attention output feeds into ReLU² not SiLU.
- **Implementation complexity:** Low (simpler than SiLU)

#### 3.3 SubLN (Sub-Layer Normalization) in BitNet
- **Source:** arXiv:2504.12285
- **Key idea:** BitNet applies RMSNorm INSIDE the residual branch before each linear layer (SubLN), not just at the start of each block. This means attention has additional normalization points.
- **Relevance to EdgeLM:** Must implement SubLN correctly; it's not standard pre-norm or post-norm.
- **Implementation complexity:** Low (just additional RMSNorm calls, which are negligible in latency)

### 4. Softmax Optimization

#### 4.1 Online Softmax (Single-Pass Algorithm)
- **Source:** FlashAttention paper (Dao et al., 2022), Welford's algorithm
- **Key idea:** Compute softmax in a single pass maintaining running max and running sum, avoiding the standard two-pass approach (pass 1: find max, pass 2: compute exp and sum).
- **Relevance to EdgeLM:** Halves memory traffic for softmax computation. Critical for prefill, minor for decode (softmax over single query position).
- **Estimated impact:** 15-20% speedup for prefill softmax
- **Implementation complexity:** Low
- **Details:**
```c
// Online softmax -- single pass
float running_max = -FLT_MAX;
float running_sum = 0.0f;
for (int j = 0; j < seq_len; j++) {
    float score = dot_product(Q_head, &K_cache[j * head_dim], head_dim);
    float old_max = running_max;
    running_max = fmaxf(running_max, score);
    // Rescale previous sum for new max
    running_sum = running_sum * expf(old_max - running_max) + expf(score - running_max);
    // Accumulate weighted V
    float weight = expf(score - running_max);
    for (int d = 0; d < head_dim; d++)
        output[d] = output[d] * expf(old_max - running_max) + weight * V_cache[j * head_dim + d];
}
// Final normalization
for (int d = 0; d < head_dim; d++)
    output[d] /= running_sum;
```

#### 4.2 Integer Softmax via Lookup Table
- **Source:** Integer-only attention papers (2024-2025), T-MAC methodology
- **Key idea:** Replace expensive `exp()` calls with a 256-entry lookup table. Quantize attention scores to 8-bit indices, look up pre-computed exp values.
- **Relevance to EdgeLM:** `exp()` has no native AVX2 instruction. Polynomial approximation takes 5-10 instructions; LUT takes 1-2 (VPSHUFB or scalar table lookup).
- **Estimated impact:** 2-3.7x faster softmax computation
- **Implementation complexity:** Low
- **Accuracy:** <1% loss with 256-entry LUT; sufficient for inference
- **Details:** Quantize score range [min_score, max_score] into 256 bins, pre-compute `exp(bin_center)` for each bin. During attention: `weight = LUT[(int)((score - min_score) * 255 / range)]`.

#### 4.3 AVX2 Fast Exp Approximation
- **Source:** Cephes library, Intel SVML, Agner Fog's optimization resources
- **Key idea:** Minimax polynomial approximation of exp(x) using AVX2 FMA instructions. Achieves ~5 ULP accuracy in 6-8 instructions.
- **Relevance to EdgeLM:** If LUT softmax quality is insufficient, polynomial exp is the fallback.
- **Estimated impact:** 2x faster than scalar `expf()` loop
- **Implementation complexity:** Medium
- **Details:**
```c
// Fast vectorized exp using AVX2 (Cephes-style)
static inline __m256 fast_exp_avx2(__m256 x) {
    __m256 t = _mm256_fmadd_ps(x, _mm256_set1_ps(1.4426950408f), _mm256_set1_ps(0.5f));
    __m256i ti = _mm256_cvttps_epi32(t);
    __m256 tf = _mm256_cvtepi32_ps(ti);
    __m256 r = _mm256_sub_ps(x, _mm256_mul_ps(tf, _mm256_set1_ps(0.693147180f)));
    // Polynomial: 1 + r + r²/2 + r³/6 + r⁴/24
    __m256 p = _mm256_fmadd_ps(r, _mm256_set1_ps(0.0416666f), _mm256_set1_ps(0.1666666f));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(0.5f));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.0f));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.0f));
    // Scale by 2^ti (bit manipulation on float exponent)
    __m256i exp_bits = _mm256_slli_epi32(_mm256_add_epi32(ti, _mm256_set1_epi32(127)), 23);
    return _mm256_mul_ps(p, _mm256_castsi256_ps(exp_bits));
}
```

### 5. Memory-Efficient Attention Techniques

#### 5.1 Streaming Attention (No N×N Materialization)
- **Source:** Standard CPU inference practice (llama.cpp, whisper.cpp)
- **Key idea:** During single-token decode, compute Q@K^T row-by-row, accumulate to output, never allocate the full seq_len×seq_len matrix. Memory is O(seq_len) not O(seq_len²).
- **Relevance to EdgeLM:** Must-have baseline. The N×N matrix for 4096 context = 64MB -- would blow L3 cache.
- **Estimated impact:** Baseline approach, not optional
- **Implementation complexity:** Low

#### 5.2 Tiled Attention for Cache Efficiency
- **Source:** llamafile/tinyBLAS methodology, CPU cache optimization literature
- **Key idea:** Load Q vector once into registers, stream K vectors through L1/L2 cache sequentially. The hardware prefetcher on Golden Cove is aggressive and will prefetch sequential K entries. Unroll the outer loop to reuse Q across 256+ K computations before moving to next Q head.
- **Relevance to EdgeLM:** Golden Cove's 3-load-per-cycle bandwidth from L1 (96 bytes/cycle) makes this highly effective.
- **Estimated impact:** 20-40% improvement over naive attention loop
- **Implementation complexity:** Medium
- **Details:** llamafile/tinyBLAS achieves 2-5x over llama.cpp using similar tiling patterns for matmul. The same principle applies to attention score computation.

#### 5.3 Sliding Window Attention with Ring Buffer
- **Source:** Mistral (W=4096), arXiv:2309.17453
- **Key idea:** Limit attention to last W tokens. KV cache becomes a fixed-size circular buffer. Memory is constant regardless of total conversation length.
- **Relevance to EdgeLM:** Perfect for bounded memory (6-7GB budget). With W=4096 and FP8 KV: ~32MB per layer × 28 layers = ~896MB -- fits easily.
- **Estimated impact:** Constant memory, 4-8x reduction for long contexts
- **Implementation complexity:** Low
- **Tradeoff:** Cannot attend beyond window (long-range recall limited)

#### 5.4 Attention Sinks (StreamingLLM)
- **Source:** arXiv:2309.16512 (Xiao et al., 2023)
- **Key idea:** LLMs inherently attend heavily to the first few tokens ("attention sinks") regardless of their content. Keep [first 4 tokens] + [recent W tokens] in KV cache, discard the middle.
- **Relevance to EdgeLM:** Enables "infinite" context with bounded KV cache. The sink tokens always stay in L1 cache (tiny), recent tokens in ring buffer.
- **Estimated impact:** Enables unlimited conversation length with fixed memory
- **Implementation complexity:** Low (two KV buffers: sink + recent ring)
- **Quality:** Good for recent context, no long-range recall beyond window

#### 5.5 INT8 KV Cache with Inline Dequantization
- **Source:** vLLM, BitNet papers, T-MAC (EuroSys 2025)
- **Key idea:** Store KV cache in INT8 (per-vector scale factor). Dequantize inline during attention score computation: one `vpmovsx` + `vcvtdq2ps` + `vmulps` per KV load.
- **Relevance to EdgeLM:** 2x memory reduction over FP16, 4x over FP32. Critical for staying within 6-7GB budget at longer contexts.
- **Estimated impact:** 2x KV cache memory reduction, <0.5% accuracy loss
- **Implementation complexity:** Low-Medium
- **Details:** Per-vector quantization: `k_int8[i] = round(k_fp32[i] / scale)` where `scale = max(abs(k_fp32)) / 127`. Dequant: `k_fp32[i] = k_int8[i] * scale`.

#### 5.6 Asymmetric KV Quantization (Keys ≠ Values)
- **Source:** arXiv:2502.15075 "Quantize What Counts" (2025)
- **Key idea:** Keys need higher precision than values because they participate in the softmax (exponential amplifies errors). Use 8-bit keys, 4-bit values (or FP8 keys, INT4 values).
- **Relevance to EdgeLM:** Further reduces KV cache with minimal quality impact. Keys at INT8 + Values at INT4 = 1.5 bytes/element average vs 4 bytes FP32.
- **Estimated impact:** Additional 1.3-2x KV memory savings beyond uniform quantization
- **Implementation complexity:** Medium

### 6. Sparse and Approximate Attention

#### 6.1 QUOKA: Sparse Attention via Query Similarity
- **Source:** 2024 research on CPU sparse attention
- **Key idea:** For each query, attend only to keys with high cosine similarity (skip low-relevance KV pairs). Computes mean query, thresholds by similarity, masks low-score entries.
- **Relevance to EdgeLM:** 7x speedup reported on Intel Xeon CPUs with 88% KV pair reduction. Most beneficial during prefill; limited value for single-token decode.
- **Estimated impact:** Up to 7x for prefill, ~1.2x for decode
- **Implementation complexity:** Medium-High

#### 6.2 Attention Skipping (MobileLLM-Flash)
- **Source:** arXiv:2603.15954 (March 2026)
- **Key idea:** Skip attention computation entirely for certain layers/positions when learned patterns indicate it's unnecessary. Uses hardware-in-the-loop architecture search.
- **Relevance to EdgeLM:** 1.8x faster prefill, 1.6x faster decode on mobile CPUs. Works with standard runtime (no custom kernels). Supports 8K context.
- **Estimated impact:** 1.6-1.8x speedup
- **Implementation complexity:** High (requires architecture search or predefined skip patterns)
- **Models:** 350M, 650M, 1.4B variants tested; could apply to 3B

#### 6.3 SnapKV: Dynamic KV Cache Compression
- **Source:** arXiv:2404.14469 (Zhang et al., 2024)
- **Key idea:** Cluster attention head features to identify which tokens are "important" and keep only those in KV cache (10-20% of full size).
- **Relevance to EdgeLM:** Massive bandwidth savings for long contexts, but requires per-head importance scoring overhead.
- **Estimated impact:** 5-10x KV cache reduction for long contexts
- **Implementation complexity:** High

### 7. Linear Attention Alternatives

#### 7.1 RWKV: Linear-Time Recurrent Attention
- **Source:** arXiv:2305.13048 (Bo Peng et al., 2023)
- **Key idea:** Replace softmax attention with linear recurrence: state = state * decay + K*V^T, output = Q @ state. O(1) memory and compute per token during decode.
- **Relevance to EdgeLM:** Ideal for CPU inference -- no growing KV cache, constant compute. But requires RWKV-specific model (not compatible with Llama/BitNet).
- **Estimated impact:** O(1) vs O(seq_len) per-token decode
- **Implementation complexity:** Low (simpler kernel than softmax attention)
- **Availability:** 3B and 14B models exist; not ternary-quantized

#### 7.2 RACE Attention: Linear-Time via LSH
- **Source:** arXiv:2510.04008 (ICLR 2026)
- **Key idea:** Use Locality-Sensitive Hashing (LSH) with angular similarity instead of softmax exponentials. Gaussian random projections for approximation.
- **Relevance to EdgeLM:** Demonstrated 75M token sequences on Intel Xeon Gold 5220R (16-core, AVX-512). LSH-based indexing works naturally with CPU cache hierarchy.
- **Estimated impact:** Linear time scaling, enables very long contexts
- **Implementation complexity:** High (research-level, not production-ready)

#### 7.3 Hedgehog: Learnable Linear Attention
- **Source:** arXiv:2402.04347 (2024)
- **Key idea:** Train a linear attention approximation that recovers 99% of full softmax attention quality using learned feature maps.
- **Relevance to EdgeLM:** If linear attention quality is a concern, this approach closes the gap.
- **Estimated impact:** ~99% quality recovery with O(n) complexity
- **Implementation complexity:** High (requires model retraining)

### 8. RoPE (Rotary Position Embeddings) Optimization

#### 8.1 SIMD RoPE with Complex Multiply
- **Source:** llama.cpp implementation, architecture analysis
- **Key idea:** RoPE rotates pairs of dimensions using sin/cos. Encode as complex multiply: `[cos(θ), -sin(θ), sin(θ), cos(θ)]` in YMM register, use VFMADD for 2-3 cycle latency per query position.
- **Relevance to EdgeLM:** Low-hanging fruit. Pre-compute sin/cos tables at startup. Per-token: just multiply and add.
- **Estimated impact:** RoPE is ~0.14ms/28 layers (negligible), but clean SIMD implementation prevents it from becoming a bottleneck.
- **Implementation complexity:** Low
- **Details:** Pre-compute RoPE tables on E-cores during model loading. Store as interleaved [cos, -sin, sin, cos] for direct AVX2 FMA.

#### 8.2 RoPE on E-Cores
- **Source:** EdgeLM architecture design
- **Key idea:** Offload RoPE computation to E-cores while P-cores handle the heavier ternary matmul for QKV projections. RoPE is element-wise and lightweight.
- **Relevance to EdgeLM:** Frees P-core cycles for bandwidth-bound ternary operations.
- **Estimated impact:** ~1-2% overall improvement (RoPE is already fast)
- **Implementation complexity:** Low

### 9. Intel-Specific Attention Optimizations

#### 9.1 Intel IPEX (Extension for Transformers) Attention Kernels
- **Source:** https://intel.github.io/intel-extension-for-transformers/
- **Key idea:** Intel's IPEX 2.0+ includes AVX2/AVX-VNNI optimized attention kernels with fused QK^T + softmax.
- **Relevance to EdgeLM:** Reference implementation to study. However, IPEX is Python/PyTorch-based and cannot be used directly in EdgeLM's pure C engine.
- **Implementation complexity:** N/A (study only)
- **Key techniques to extract:** Fused kernel patterns, tiling strategies, VNNI utilization for INT8 attention.

#### 9.2 Intel Efficient LLM on GPU (Segment KV Cache)
- **Source:** Intel research (2024), from research-papers-data.json
- **Key idea:** Fuse decoder layer operations and use "Segment KV Cache" policy with customized Scaled-Dot-Product-Attention kernel.
- **Relevance to EdgeLM:** The segment KV cache concept (grouping KV entries into segments for batch processing) could reduce cache thrashing on CPU.
- **Estimated impact:** 10-15% for long contexts
- **Implementation complexity:** Medium

#### 9.3 Intel Ultra-Low-Bit Microkernels
- **Source:** arXiv:2508.06753 (August 2025)
- **Key idea:** 2.2x speedup over BitNet using VNNI-optimized 2-bit kernels for weight computation.
- **Relevance to EdgeLM:** Directly applicable to QKV and output projection (ternary matmul portion of attention), not to the FP32 score computation.
- **Estimated impact:** 2.2x for ternary projection portion (~28% of attention time)
- **Implementation complexity:** Medium-High

### 10. Prefill vs Decode Optimization Split

#### 10.1 Separate Kernels for Each Phase
- **Source:** Industry best practice (all major inference engines)
- **Key idea:** Prefill processes the entire prompt at once (batch GEMM, compute-bound). Decode generates one token at a time (GEMV, memory-bound). These have fundamentally different optimization strategies.
- **Relevance to EdgeLM:** Must implement two attention code paths. Decode is the bottleneck for tok/s metric.
- **Estimated impact:** 20-50% improvement from specialization
- **Implementation complexity:** Medium
- **Details:**
  - **Prefill:** Batch multiple query positions, use tiled GEMM for QK^T, amortize weight reads. Online softmax critical here.
  - **Decode:** Single query vector, stream through entire KV cache. Prefetch next cache line. Fuse softmax into score loop.

#### 10.2 Chunked Prefill (UPipe)
- **Source:** "Untied Ulysses" (2026)
- **Key idea:** During prefill, process heads in groups that fit intermediates in L1/L2. Results in 87.5% reduction in intermediate tensor memory.
- **Relevance to EdgeLM:** Important for long prompts where full prefill would exceed cache.
- **Estimated impact:** 2-4x prefill speedup for prompts >1024 tokens
- **Implementation complexity:** Medium

### 11. Attention Compute is NOT the Primary Bottleneck

#### 11.1 Bandwidth Analysis
- **Source:** EdgeLM deep dive Section 13, BitNet benchmarks
- **Key finding:** For single-token decode at seq_len<2048, the FFN (3 ternary matmuls per layer, ~67% of weights) dominates latency. Attention score computation (FP32, compute-bound) is secondary. The bottleneck shifts to attention only at very long sequences (>4K tokens).
- **Relevance to EdgeLM:** Attention optimization provides 5-15% additional speedup as a multiplicative factor, not the 2-3x gains from ternary kernel optimization.
- **Implication:** Implement attention correctly first, optimize later. Focus engineering effort on ternary FFN kernels (Section 07) for the biggest gains toward 100 tok/s.

#### 11.2 Timing Breakdown (BitNet b1.58 on i7-13700H)
- **Source:** BitNet benchmarks, Section 04 research
- Total per-token: ~29ms (= ~34 tok/s)
- QKV + output projection (ternary, bandwidth-bound): ~8ms (~28%)
- Attention scores (FP32, compute-bound): ~3ms (~10%)
- Softmax: ~1ms (~3.5%)
- Attention apply (score*V): ~1.5ms (~5%)
- FFN (ternary, bandwidth-bound): ~12ms (~41%)
- Other (RMSNorm, RoPE, embedding, sampling): ~3.5ms (~12%)

## Techniques Comparison Matrix

| Technique | Source | Impact | Complexity | Already in Deep Dive? |
|-----------|--------|--------|------------|----------------------|
| GQA with register broadcasting | Architecture analysis | 4x KV bandwidth reduction | Low-Medium | Partially (GQA yes, broadcasting no) |
| Online softmax (single-pass) | FlashAttention/Welford | 15-20% prefill softmax | Low | Yes (mentioned) |
| Integer softmax via LUT | INT-attention papers 2024 | 2-3.7x softmax speed | Low | No |
| AVX2 fast exp approximation | Cephes/Agner Fog | 2x over scalar expf() | Medium | No |
| Tiled attention (cache-aware) | llamafile methodology | 20-40% attention | Medium | No |
| Streaming attention (no N×N) | Standard practice | Baseline requirement | Low | Yes |
| Sliding window + ring buffer | Mistral | Constant memory | Low | Yes |
| Attention sinks (StreamingLLM) | arXiv:2309.16512 | Infinite context | Low | Yes (briefly) |
| INT8 KV cache quantization | vLLM, T-MAC | 2x KV memory | Low-Medium | Yes (mentioned) |
| Asymmetric KV quantization | arXiv:2502.15075 | 1.3-2x additional KV savings | Medium | No |
| Sparse attention (QUOKA) | 2024 CPU research | 7x prefill on Xeon | Medium-High | No |
| Attention skipping (MobileLLM) | arXiv:2603.15954 | 1.6-1.8x decode | High | No |
| SnapKV dynamic compression | arXiv:2404.14469 | 5-10x KV reduction | High | No |
| Separate prefill/decode kernels | Industry practice | 20-50% specialization | Medium | Partially |
| Chunked prefill (UPipe) | 2026 research | 2-4x long prefill | Medium | No |
| RACE linear attention (LSH) | arXiv:2510.04008 | Linear time scaling | High | No |
| Hedgehog learnable linear attn | arXiv:2402.04347 | 99% quality at O(n) | High | No |
| Intel Ultra-Low-Bit microkernels | arXiv:2508.06753 | 2.2x ternary projections | Medium-High | No |
| BitNet SubLN architecture | arXiv:2504.12285 | Correctness requirement | Low | No |
| BitNet Squared ReLU | arXiv:2402.17764 | Simpler activation | Low | No |
| Interleaved GQA memory layout | Cache optimization | 1.3-1.5x cache hits | Medium | No |
| RoPE SIMD via complex multiply | llama.cpp | Clean implementation | Low | Partially |
| RWKV linear recurrence | arXiv:2305.13048 | O(1) decode | Low kernel | No (model change) |

## Recommendations for EdgeLM

Ordered by impact-to-effort ratio for the i7-12700H target:

### Must-Have (Phase 1-2)

1. **Streaming attention (no N×N materialization)** -- Baseline requirement. Never allocate seq_len² matrix. O(seq_len) memory.

2. **GQA with register broadcasting** -- Load each KV head once, reuse for all Q heads in group. 4x KV bandwidth reduction. Low implementation effort, high impact.

3. **Separate prefill and decode kernels** -- Prefill is batch GEMM (compute-bound), decode is GEMV (memory-bound). Different optimization strategies required.

4. **INT8 KV cache with inline dequant** -- 2x memory reduction. Critical for staying within 6-7GB budget at seq_len>1024. Per-vector scale factor, single AVX2 instruction for dequant.

5. **Pre-computed RoPE tables** -- Compute sin/cos at startup, store as interleaved format for direct AVX2 FMA. Offload to E-cores.

### Should-Have (Phase 2-3)

6. **Integer softmax via LUT** -- Replace expensive exp() with 256-entry lookup table. 2-3.7x faster softmax. <1% accuracy loss. Very low implementation cost.

7. **AVX2 fast exp fallback** -- If LUT quality is insufficient, use Cephes-style polynomial approximation (6-8 FMA instructions). 2x faster than scalar expf().

8. **Tiled attention for L1/L2 locality** -- Load Q once, stream K through cache. Exploit Golden Cove's 3-load-per-cycle L1 bandwidth. 20-40% improvement.

9. **Sliding window + attention sinks** -- Ring buffer KV for recent W tokens + first 4 tokens (sinks). Enables "infinite" conversation with bounded memory.

10. **Asymmetric KV quantization** -- Keys at INT8, values at INT4. Keys need higher precision for softmax stability. Additional 1.3-2x KV savings.

### Nice-to-Have (Phase 4-5)

11. **Attention skipping** -- Skip attention for certain layers (MobileLLM-Flash approach). 1.6x decode speedup but requires predefined skip patterns.

12. **QUOKA sparse attention** -- For prefill only. Skip low-similarity KV pairs. Up to 7x prefill speedup.

13. **SnapKV dynamic compression** -- Keep only important tokens in KV cache (10-20% of full). High complexity, high reward for long contexts.

14. **Chunked prefill** -- Process long prompts in L1/L2-sized chunks. 2-4x prefill speedup for >1024 token prompts.

### Research/Paper Contributions

15. **CPU FlashAttention with AVX2** -- No production implementation exists. A well-optimized CPU FlashAttention using AVX2 intrinsics and online softmax would be a novel contribution for the paper.

16. **RACE-style linear attention on consumer CPU** -- Demonstrate linear-time attention on i7-12700H. Paper contribution: first consumer hardware benchmarks.

17. **Quantify attention vs FFN bottleneck crossover** -- At what sequence length does attention become the primary bottleneck on DDR4-3200? This analysis would be valuable for the paper.

## References

1. Dao, T. "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning." arXiv:2305.10434, 2023.
2. Ainslie, J. et al. "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints." arXiv:2305.13245, EMNLP 2023.
3. Ma, S. et al. "The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits." arXiv:2402.17764, 2024.
4. Microsoft Research. "BitNet b1.58 2B4T Technical Report." arXiv:2504.12285, 2025.
5. Xiao, G. et al. "Efficient Streaming Language Models with Attention Sinks." arXiv:2309.16512, 2023.
6. Zhang, Y. et al. "SnapKV: LLM Knows What You are Looking for Before Generation." arXiv:2404.14469, 2024.
7. Peng, B. et al. "RWKV: Reinventing RNNs for the Transformer Era." arXiv:2305.13048, 2023.
8. Anonymous. "Quantize What Counts: More for Keys, Less for Values." arXiv:2502.15075, 2025.
9. Wei, T. et al. "T-MAC: Table-based MAC for Low-Bit LLM Deployment on Edge." arXiv:2407.00088, EuroSys 2025.
10. MobileLLM-Flash. "Attention Skipping for Long-Context Mobile LLM Inference." arXiv:2603.15954, 2026.
11. RACE Attention. "Linear Time Scaling via LSH." arXiv:2510.04008, ICLR 2026.
12. Zhang, S. et al. "Hedgehog: Learnable Linear Attention." arXiv:2402.04347, 2024.
13. Intel. "Ultra-Low-Bit Microkernels for Efficient LLM Inference." arXiv:2508.06753, 2025.
14. Kwon, W. et al. "Efficient Memory Management for Large Language Model Serving with PagedAttention." SOSP 2023.
15. llama.cpp Flash Attention PR. https://github.com/ggml-org/llama.cpp/pull/5021
16. Intel IPEX for Transformers. https://intel.github.io/intel-extension-for-transformers/
17. Bai, Y. et al. "A Survey of Linear Attention." https://github.com/btzyd/Awesome-Linear-Attention-Survey, 2024.
18. Sun, Y. et al. "Retentive Network: A Successor to Transformer for Large Language Models." arXiv:2307.08621, 2024.
19. Gu, A. & Dao, T. "Mamba: Linear-Time Sequence Modeling with Selective State Spaces." arXiv:2312.00752, 2023.
20. vLLM. "Quantized KV Cache." https://docs.vllm.ai/en/latest/features/quantization/quantized_kvcache
21. Tunstall, J. "llamafile: Turning LLMs into Runnable Files." https://justine.lol/matmul/

## Audit Addendum (2026-04-02)

- **Causal-mask specialization for decode deserves its own path.** Batch-1 decode
  with a single live query token should not pay for generic mask logic.
- **GQA KV layout and attention kernel design should be co-tuned.** The optimal
  head/group layout is inseparable from the dot-product and weighted-sum
  kernels that consume it.
- **Long-context policy should stay visible in the attention section.** Sliding
  windows, sinks, or eviction rules are not just cache choices; they change the
  effective attention contract.
