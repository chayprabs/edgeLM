# Section 04: BitNet / Ternary Quantization Theory -- Extended Research

## Overview

This section covers the theoretical and practical foundations of BitNet b1.58 ternary quantization:
the {-1, 0, +1} weight representation, absmean quantization, activation scaling, GGUF encoding formats
(TQ1_0, TQ2_0, I2_S), the BitNet v2 Hadamard extension, and concrete performance numbers for our
target hardware class. This is the single most important architectural decision in EdgeLM -- choosing
ternary weights is what makes 100+ tok/s possible at all.

## What the Deep Dive Already Covers

- Ternary weights {-1, 0, +1} reduce model to ~0.4-0.6 GB for 3B params
- Bandwidth math: 40 GB/s / 0.6 GB = ~67 tok/s baseline
- Matmul becomes conditional add/subtract (no multiplies)
- Target models: BitNet-b1.58-2B-4T (primary) and bitnet_b1_58-3B (secondary)
- bitnet.cpp as reference implementation (do not fork)

---

## New Findings

### 1. Absmean Quantization -- Exact Formula and Properties

#### 1.1 The Quantization Function

- **Source:** arXiv:2402.17764 (BitNet b1.58, Ma et al., Microsoft Research, Feb 2024)
- **Key idea:** Weights are quantized using the average absolute value as the scale factor, then rounded and clipped to {-1, 0, +1}.
- **Formula:**
  ```
  γ = (1 / n*m) * Σ|W_ij|        (average absolute value, i.e. absmean)
  W̃ = RoundClip(W / (γ + ε), -1, 1)

  where RoundClip(x, a, b) = max(a, min(b, round(x)))
  ```
- **Relevance to EdgeLM:** This defines our weight storage format. Each weight matrix W has exactly ONE scale factor γ (per-tensor quantization). During inference we store W̃ as 2-bit ternary and γ as FP16 or FP32. The dequantization is: W_approx = W̃ * γ. But in the ternary kernel we never actually dequantize -- instead we absorb γ into the output scaling.
- **Estimated impact:** Understanding this formula is critical for implementing the GEMV kernel correctly (scale application order).
- **Implementation complexity:** Low -- simple math, one scale per matrix.
- **Details:**
  - ε is a small constant to avoid division by zero (typically 1e-8)
  - The scale γ is computed during training and stored alongside W̃
  - Per-TENSOR (not per-row, not per-group): one γ for the entire weight matrix
  - This is simpler than GPTQ/AWQ which use per-group scales
  - Zero weights are genuinely stored as 0 (no implicit zero-point offset)

#### 1.2 Sparsity Statistics

- **Source:** arXiv:2402.17764 section 3.2; community analysis of released model weights
- **Key idea:** A typical BitNet b1.58 weight matrix has ~40-50% zeros (sparsity varies by layer type and depth).
- **Relevance to EdgeLM:** We already know from Section 02 research that dense SIMD (using all ternary values) outperforms sparse skipping at this sparsity level. This confirms: do NOT branch on zero weights; process all elements with SIMD.
- **Details:**
  - Attention Q/K/V projection layers: ~40% zeros typical
  - FFN up/gate projections: ~45-55% zeros (more sparse)
  - Deeper layers tend to have slightly higher sparsity
  - Total across all weights: ~45% average zero fraction
  - The XOR+AND trick in Section 02 handles zeros efficiently without branching

#### 1.3 Activation Quantization: Per-Token Absmax to INT8

- **Source:** arXiv:2402.17764; arXiv:2504.12285 (2B4T technical report)
- **Key idea:** Activations (inputs to each BitLinear layer) are quantized to INT8 using per-token absolute maximum scaling.
- **Formula:**
  ```
  Q_b = 2^(b-1) - 1 = 127  (for b=8 bits)
  η = absmax(X) = max(|X_i|)  per token
  X̃ = RoundClip(Q_b / (η + ε) * X, -Q_b, Q_b)
  ```
- **Designation:** W1.58A8 -- 1.58-bit weights, 8-bit activations
- **Relevance to EdgeLM:** The GEMV kernel computes INT8_activations × INT2_weights → INT32_accumulator → scale by (η * γ) / Q_b → FP32 output. The VPDPBUSD instruction is designed exactly for INT8×INT8→INT32 accumulation. For ternary weights encoded as {0, 1, 2} mapped to {-1, 0, +1}, we use signed INT8 activations with VPDPBUSD.
- **Implementation complexity:** Low -- per-token means one max-reduction per row of activations, done once before the GEMV.

---

### 2. Model Architecture -- What Makes BitNet Different

#### 2.1 BitNet b1.58 2B4T Architecture Specs

- **Source:** arXiv:2504.12285, HuggingFace model card (microsoft/bitnet-b1.58-2B-4T)
- **Key specs:**
  | Parameter | Value |
  |-----------|-------|
  | Total params | ~2.4B |
  | Non-embedding params | ~0.4B equivalent memory |
  | Context length | 4,096 tokens |
  | Vocabulary | 128,256 (LLaMA 3 tokenizer) |
  | Positional encoding | RoPE |
  | Activation function | Squared ReLU (ReLU²) -- NOT SiLU |
  | Normalization | SubLN (sublayer normalization) |
  | Biases | None in linear/norm layers |
  | Training tokens | 4 trillion |

- **Critical difference from LLaMA:** Uses **Squared ReLU** (ReLU²) instead of SiLU/GELU in FFN layers. This is simpler to implement and SIMD-friendly (just a ReLU then a multiply).
- **SubLN:** Normalization is applied INSIDE the residual branch (before the linear layer) rather than pre-norm. This is the original BitNet architecture choice.
- **Relevance to EdgeLM:** Our FFN implementation needs ReLU² not SiLU. SubLN means every linear layer is preceded by a RMSNorm (or LayerNorm). This doubles the number of normalization operations vs LLaMA, but each is cheap.

#### 2.2 bitnet_b1_58-3B Architecture

- **Source:** HuggingFace community, bitnet.cpp supported models list
- **Key info:** The 3B model is a community model (not official Microsoft release). It uses the same BitNet b1.58 quantization but may differ in training quality/tokens. Less well-characterized than 2B4T.
- **Relevance to EdgeLM:** The 2B4T model is the primary target due to: (a) official Microsoft support, (b) 4T training tokens → better quality, (c) 0.4 GB non-embedding weight = ~100 tok/s theoretical on our hardware.

#### 2.3 Memory Layout: Embeddings Are NOT Ternary

- **Source:** HuggingFace GGUF repo (ggml-model-i2_s.gguf = 1.19 GB for 2B4T)
- **Key finding:** The GGUF file is 1.19 GB despite non-embedding weights being only ~0.4 GB. The difference (~0.8 GB) is the embedding table stored at higher precision.
- **Embedding table size:** 128,256 vocab × hidden_dim × 2 bytes (FP16) ≈ 128K × 2048 × 2 = ~512 MB if hidden_dim=2048.
- **Inference implication:** During token generation, the embedding lookup is a single sparse row read (negligible bandwidth). The output projection (unembedding) IS a large matmul but happens once per token.
- **Relevance to EdgeLM:** For bandwidth calculation, use 0.4 GB (non-embedding ternary weights), NOT 1.19 GB (full GGUF file). This gives: 40 GB/s / 0.4 GB = **100 tok/s theoretical** for the 2B4T model -- right at our target even before any optimization.

---

### 3. GGUF Ternary Encoding Formats

#### 3.1 TQ2_0 -- The Fastest Format on AVX2

- **Source:** llama.cpp PR #8151 (github.com/ggerganov/llama.cpp/pull/8151)
- **Key idea:** 2 bits per ternary value, 4 values per byte, 64-element blocks with one FP16 scale.
- **Format layout:**
  ```
  Block size: 64 ternary elements
  Data: 16 bytes (4 values/byte × 16 bytes = 64 values)
  Scale: 2 bytes (FP16)
  Total: 18 bytes per 64 elements = 2.0625 bits per weight

  Encoding: {-1→0, 0→1, +1→2} or {-1→01, 0→00, +1→10} (2 bits per value)
  ```
- **Benchmark on Intel Core m3-8100Y (AVX2):** **141.83 GB/s** throughput
  - Q4_K baseline: 64.17 GB/s
  - **TQ2_0 is 2.21x faster than Q4_K on AVX2**
  - TQ1_0: 70.31 GB/s
- **Relevance to EdgeLM:** TQ2_0 is the format to target for our AVX2 kernels. At 141 GB/s on a weaker CPU (m3-8100Y, ~2-core), our i7-12700H with 6 P-cores should achieve even higher effective bandwidth utilization.
- **Implementation complexity:** Low -- straightforward 2-bit extraction with AND+shift.

#### 3.2 TQ1_0 -- Maximum Compression

- **Source:** llama.cpp PR #8151
- **Key idea:** Packs 5 ternary values per byte using base-3 encoding (3^5 = 243 < 256).
- **Format layout:**
  ```
  Block size: 256 elements
  First 240 elements: 48 bytes (5 values/byte, base-3)
  Last 16 elements: 4 bytes (4 values/byte, 2-bit)
  Scale: 2 bytes (FP16)
  Total: 54 bytes per 256 elements = 1.6875 bits per weight

  Decode: byte b encodes (b0, b1, b2, b3, b4) where b = b0 + 3*b1 + 9*b2 + 27*b3 + 81*b4
  ```
- **Compression ratio vs F16:** 7616 MB → 948 MB for a 3.9B model (8x compression)
- **Benchmark:** 70.31 GB/s on AVX2 (half the speed of TQ2_0, due to base-3 decode overhead)
- **Relevance to EdgeLM:** TQ1_0 saves ~15% memory vs TQ2_0 but is 2x slower to decode. For a bandwidth-limited workload, the decode overhead likely cancels any bandwidth savings. **TQ2_0 is likely better for our use case.** The exception: if RAM is critically short, TQ1_0 keeps the 3B model at ~0.52 GB vs ~0.6 GB for TQ2_0.
- **Implementation complexity:** Medium -- base-3 decode requires multiply/modulo or lookup.

#### 3.3 I2_S -- bitnet.cpp's Native Format

- **Source:** microsoft/BitNet GitHub, HuggingFace model card
- **Key idea:** 2-bit signed integers, the format used in bitnet.cpp's official kernels. The released GGUF (ggml-model-i2_s.gguf = 1.19 GB) uses this format.
- **Encoding:** {-1→-1 as signed 2-bit, 0→0, +1→+1} packed 4 values per byte
- **GGUF file size (2B4T):** 1.19 GB total (includes FP16 embeddings)
- **Relevance to EdgeLM:** I2_S is functionally equivalent to TQ2_0 in bit layout but the block structure and scale storage may differ. Since bitnet.cpp uses I2_S and TQ2_0 is the llama.cpp format, we can study both for our custom kernel. TQ2_0 has better benchmark data; I2_S has more battle-tested code to study.

---

### 4. Inference Performance on Target Hardware

#### 4.1 Benchmark: i7-13800H (Closest to Our i7-12700H)

- **Source:** arXiv:2504.12285 (BitNet b1.58 2B4T Technical Report), Table in paper
- **Hardware:** Surface Laptop Studio 2, Intel i7-13800H, 8 threads
- **Result:** **29ms per output token (TPOT)** = **34.5 tok/s** at 8 threads

**Comparison context:**
| Model | Memory (non-emb) | CPU Latency | Energy/seq |
|-------|-----------------|-------------|------------|
| BitNet b1.58 2B4T | 0.4 GB | **29ms** | **0.028J** |
| Comparable FP models | 1.4-4.8 GB | 41-124ms | 0.186-0.649J |

- **Relevance to EdgeLM:** The i7-13800H has 6 P-cores + 8 E-cores (same topology as our i7-12700H) but slightly higher clocks. At 8 threads with bitnet.cpp (NOT our optimized engine), we get 34.5 tok/s. Our target is 100+ tok/s -- that's **2.9x improvement** needed over bitnet.cpp's baseline.
- **The gap to close:** bitnet.cpp achieves 2.37-6.17x vs FP baseline; our goal is ~3x over bitnet.cpp itself via:
  1. AVX-VNNI vs bitnet.cpp's generic AVX2 (2.2x gain from Section 02 research)
  2. Large pages + prefetching (20-30% gain from Section 03 research)
  3. 6 P-core pinning + optimal thread count
  4. This math works: 34.5 × 2.2 × 1.3 = ~98.6 tok/s → plausible 100+ tok/s

#### 4.2 Theoretical Bandwidth Ceiling

- **Our hardware:** 40 GB/s DDR4 real bandwidth
- **2B4T non-embedding weights:** ~0.4 GB
- **Theoretical maximum:** 40 / 0.4 = **100 tok/s** (one full weight pass per token)
- **3B model non-embedding weights:** ~0.6 GB
- **Theoretical maximum (3B):** 40 / 0.6 = **~67 tok/s**

This means:
- For 2B4T: we need to achieve ~100% bandwidth utilization to hit 100 tok/s -- tight but possible with prefetching
- For 3B: we need to exceed 100% via speculative decoding or iGPU offload to reach 100+ tok/s
- **Conclusion: 2B4T is the right primary target.** The 3B model requires advanced optimizations just to break 100 tok/s.

#### 4.3 bitnet.cpp Speed Gap Analysis

- **Source:** Microsoft BitNet README; arXiv:2410.16144 (bitnet.cpp paper)
- **bitnet.cpp x86 speedups reported:** 2.37x-6.17x (range across model sizes)
- **For 2B model specifically:** likely ~3-4x over llama.cpp Q4_K baseline
- **Our i7-12700H vs i7-13800H:** ~5-10% slower (lower base clock, similar architecture)
- **Expected bitnet.cpp performance on our hardware:** ~31-33 tok/s
- **Relevance to EdgeLM:** Our custom engine needs to be ~3x faster than bitnet.cpp for 100 tok/s. This is achievable based on Section 02 findings (VNNI-optimized kernels get 2.2x over bitnet.cpp's MAD approach).

---

### 5. BitNet v2 -- 4-Bit Activations

#### 5.1 Hadamard Transformation for Outlier Reduction

- **Source:** arXiv:2504.18415 (BitNet v2, Ma et al., Apr 2025)
- **Key idea:** Apply an online Walsh-Hadamard Transform (WHT) before quantizing activations. This redistributes outlier values across all dimensions, making the distribution Gaussian-like and suitable for 4-bit quantization.
- **Formula:**
  ```
  H_0 = (1)
  H_m = (1/√2) * [[H_{m-1}, H_{m-1}], [H_{m-1}, -H_{m-1}]]

  Applied: X' = H_m × X  (before quantization)
  ```
- **Complexity:** O(n log n) using fast Hadamard transform -- not O(n²)
- **Why it works:** Transformer activations have channel-wise outliers (a few channels with values 10-100x larger than average). These outliers force large quantization ranges. WHT spreads the energy across ALL channels, eliminating outliers.
- **Relevance to EdgeLM:** BitNet v2 is not yet available as a released model (trained from scratch at 400M-7B scale in the paper). For our current targets (2B4T, 3B), we use INT8 activations (W1.58A8). However, if we extend to BitNet v2 models, we get 4-bit activations → halved activation bandwidth → 10-20% further improvement on large batch sizes.

#### 5.2 INT4 Activation Quantization Formula

- **Source:** arXiv:2504.18415
- **Formula:**
  ```
  β = mean(|X|)  (per-token absmean)
  QINT4(X) = (β / √7) × RoundClip(√7 / (β + ε) × X, -8, 7)
  ```
- **Range:** Symmetric 4-bit: [-8, 7] (standard INT4)
- **Training:** Two-stage: Stage 1 with INT8 activations (95B tokens), Stage 2 continue-train with INT4 (5B tokens)
- **Performance:** "Minimal degradation" vs b1.58 at all tested scales (400M-7B)
- **Relevance to EdgeLM:** For future model versions. The INT4 format would enable `VPDPBUSD` with 4-bit activations repacked to INT8 (two INT4 values per INT8 byte) -- still usable with our VNNI kernels.

---

### 6. Quality and Model Selection Analysis

#### 6.1 Perplexity vs Other Quantization Methods

- **Source:** arXiv:2402.17764 (BitNet b1.58 paper, Table 1)
- **Key results at 3B scale:**
  | Model | Perplexity (WikiText-2) | Memory |
  |-------|------------------------|--------|
  | BitNet b1.58 3B | 9.91 | 3.55x less than FP16 |
  | BitNet b1.58 3.9B | 9.62 | 3.32x less |
  | FP16 baseline (3B) | ~9.9 | baseline |
  - BitNet b1.58 at 3B **matches FP16 perplexity** despite ternary weights
  - End-task benchmarks (PIQA, ARC, HellaSwag): also matches FP16

- **2B4T downstream benchmarks:**
  - MMLU: 53.17 (competitive with Phi-2, Gemma-2B class models)
  - GSM8K: 58.38 (math reasoning -- solid for 2B)
  - HumanEval+: 38.40
  - Average across all tasks: 54.19

#### 6.2 BitNet vs Post-Training Quantization (PTQ)

- **Source:** arXiv:2402.17764 discussion section; community experiments
- **Key finding:** BitNet b1.58 is trained FROM SCRATCH with quantization-aware training (QAT), not post-training quantized. Applying post-training ternary quantization to a pre-trained Llama model gives significantly worse quality.
- **Why QAT is necessary:** Straight-through estimator (STE) during training allows gradients to flow through the non-differentiable RoundClip function. The model learns weight distributions that quantize well. PTQ on Llama-3 to ternary: perplexity degrades significantly (estimated 2-5 points).
- **Relevance to EdgeLM:** We cannot simply load Llama-3.2-3B and apply ternary quantization. We must use models that were natively trained as BitNet. The stretch goal (custom ternary quantization of Llama-3.2-3B) is HIGH RISK for quality.

#### 6.3 Per-Tensor vs Per-Group Scale -- Quality Trade-Off

- **Source:** arXiv:2402.17764; GPTQ/AWQ papers for comparison
- **BitNet b1.58:** Per-tensor scale (one γ per weight matrix)
- **GPTQ/AWQ Q4:** Per-group scale (typically one scale per 32-128 weights)
- **Quality impact:** Per-tensor is slightly worse in theory than per-group because the single scale must accommodate outlier weights. BitNet compensates by training with per-tensor quantization from the start.
- **Performance impact:** Per-tensor means only 1 FP16 value stored per matrix (negligible storage) vs per-group which stores `n_weights / group_size` FP16 values (adds ~3% overhead for group_size=128).
- **Relevance to EdgeLM:** Our scale storage is trivial (one FP16 per weight matrix). Scale application is one multiply of the final accumulator -- happens outside the inner loop. This is simpler and faster than per-group scale application.

---

### 7. Training Infrastructure and Straight-Through Estimator

#### 7.1 STE for Ternary Quantization

- **Source:** arXiv:2402.17764; Bengio et al. 2013 (original STE paper)
- **Key idea:** The RoundClip function has zero gradient almost everywhere. The Straight-Through Estimator passes gradients through as if RoundClip were the identity function.
- **Implementation:**
  ```python
  # Forward: quantize weights
  W_quantized = RoundClip(W / (gamma + eps), -1, 1)

  # Backward: STE -- pretend quantization didn't happen
  # gradient w.r.t W ≈ gradient w.r.t W_quantized * (1/gamma)
  # (where |W/gamma| <= 1, otherwise gradient is 0)
  ```
- **Relevance to EdgeLM:** We are NOT training -- only doing inference. But understanding STE explains why BitNet weights are "well-behaved" for quantization: the training process explicitly shapes the weight distribution to quantize cleanly. This is why ternary quality matches FP16.

---

### 8. Practical Encoding: How to Pack Ternary Weights

#### 8.1 Optimal Packing for AVX2 GEMV

- **Source:** Section 02 research findings; llama.cpp TQ2_0 implementation
- **Recommended encoding for EdgeLM:**
  ```
  Option A: TQ2_0-style (2 bits/value, 64-element blocks)
  - Bits: {-1 → 0b01, 0 → 0b00, +1 → 0b10}  (or {0,1,2})
  - Packed: 4 values per byte, 16 bytes per 64 weights
  - Scale: 1× FP16 per block
  - Total: 18 bytes per 64 weights = 2.0625 bpw
  - Decode: 2x AND + 2x shift per 4 values

  Option B: VNNI-interleaved (for optimal VPDPBUSD throughput)
  - Interleave {pos_mask, neg_mask} for groups of 32 weights
  - pos_mask[i] = (W[i] == +1) as 0x01 byte
  - neg_mask[i] = (W[i] == -1) as 0x01 byte
  - Stored as int8, usable directly with VPDPBUSD
  - ~2 bytes/weight (less compressed but zero decode overhead)
  ```
- **Recommendation:** Store weights in TQ2_0 format for minimum memory footprint. Unpack to VNNI-interleaved format at load time (one-time cost at startup). Cache the repacked format to disk for subsequent runs.
- **Relevance to EdgeLM:** The two-format approach (compressed on disk, VNNI-ready in RAM) gives us both minimum storage AND maximum compute throughput.

#### 8.2 Scale Factor Storage and Application

- **Source:** arXiv:2402.17764; TQ2_0 format spec
- **Storage:** One FP16 scale γ per block (64 or 256 weights). For the full 2B4T model:
  - At block_size=64: ~12M blocks × 2 bytes = ~24 MB of scales (negligible vs 0.4 GB weights)
  - At block_size=256: ~3M blocks × 2 bytes = ~6 MB of scales
- **Application in GEMV:**
  ```c
  // Inner loop: INT8 × INT2 → INT32 accumulator
  for (block) {
      int32_t sum = vnni_dot_product(activations_int8, weights_int2);
      // After inner loop: dequantize output
      float output = sum * (activation_scale * weight_scale / Q_b);
  }
  ```
- **Relevance to EdgeLM:** Scale multiplication happens ONCE PER OUTPUT ELEMENT (outside the tight inner loop), not per weight. This is a minor cost. Store scales as FP16 for memory efficiency; convert to FP32 at application time.

---

### 9. Summary: EdgeLM-Specific Insights

#### 9.1 Why 2B4T is Better Than 3B for Our Hardware

| Factor | 2B4T | 3B |
|--------|------|----|
| Non-embedding weight memory | ~0.4 GB | ~0.6 GB |
| Theoretical tok/s ceiling | 100 tok/s | 67 tok/s |
| Training quality | 4T tokens, official | Community, fewer tokens |
| Benchmark scores | 54.19 avg | Unknown |
| Available optimized kernels | bitnet.cpp I2_S | bitnet.cpp I2_S |
| Path to 100+ tok/s | Achievable at ~95% bandwidth | Requires speculative decoding or iGPU |

**Decision: Use 2B4T as primary target.** The bandwidth math is the deciding factor.

#### 9.2 Kernel Design Summary

The optimal ternary GEMV for our hardware:
1. **Weights:** Stored as TQ2_0 (2-bit packed, 0.4 GB for 2B4T) on disk
2. **Runtime format:** Unpacked to VNNI-interleaved INT8 masks in RAM (uses more RAM but zero decode overhead in hot path)
3. **Activation:** Per-token absmax → INT8 (SIMD reduction, done once per layer input)
4. **Compute:** VPDPBUSD accumulation of INT8×INT8 → INT32
5. **Dequantize:** One FP32 multiply per output element (outside inner loop)
6. **Scale storage:** FP16 per 64-weight block (negligible overhead)

---

## Techniques Comparison Matrix

| Technique | Source | Impact | Complexity | Already in Deep Dive? |
|-----------|--------|--------|------------|----------------------|
| Absmean quantization formula (exact) | arXiv:2402.17764 | Critical | Low | No (only named, not detailed) |
| Per-tensor scale (one γ/matrix) | arXiv:2402.17764 | High | Low | No |
| W1.58A8 designation | arXiv:2504.12285 | Medium | Low | No |
| 2B4T: 0.4 GB non-embedding = 100 tok/s ceiling | HuggingFace | Critical | N/A | No |
| 2B4T benchmark: 29ms/token on i7-13800H | arXiv:2504.12285 | High | N/A | No |
| TQ2_0: 141 GB/s, 2.2x faster than Q4_K | llama.cpp PR #8151 | Very High | Low | No |
| TQ1_0: base-3 packing, 1.6875 bpw | llama.cpp PR #8151 | High | Medium | No |
| I2_S: bitnet.cpp native format, 1.19 GB GGUF | bitnet.cpp | Medium | Low | No |
| Embeddings NOT ternary (512 MB FP16) | HuggingFace | High (sizing) | N/A | No |
| Squared ReLU instead of SiLU | arXiv:2504.12285 | Medium | Low | No |
| SubLN normalization (pre-linear RMSNorm) | arXiv:2504.12285 | Medium | Low | No |
| Hadamard transform for 4-bit activations | arXiv:2504.18415 | Medium (future) | Medium | No |
| QAT required; PTQ ternary quality is poor | arXiv:2402.17764 | High (design) | N/A | No |
| bitnet.cpp 34.5 tok/s on i7-13800H | arXiv:2504.12285 | Critical (baseline) | N/A | No |
| 3x over bitnet.cpp needed for 100 tok/s | Derived | Critical | N/A | No |

---

## Recommendations for EdgeLM

1. **Lock in 2B4T as primary target** -- 0.4 GB non-embedding weights gives exactly 100 tok/s theoretical at 40 GB/s. The 3B model requires speculative decoding just to reach 100 tok/s.

2. **Use TQ2_0 as the on-disk storage format** -- 141 GB/s throughput on AVX2 vs 64 GB/s for Q4_K. Implement the 2-bit packer/unpacker as part of the weight loader. Store repacked VNNI-interleaved weights in RAM for the hot path.

3. **Implement per-token INT8 absmax activation quantization as SIMD reduction** -- one horizontal max over 256/512 FP32 values using AVX2, then scale all values. This is a one-time cost per layer input, amortized across all output neurons.

4. **Apply scale factor OUTSIDE the inner loop** -- per-tensor scale means one FP32 multiply per output element after the VPDPBUSD accumulation. Do not mix scale application into the inner loop.

5. **Use Squared ReLU (not SiLU) in FFN** -- the 2B4T architecture uses ReLU², which is `max(0, x)²`. This is 2 SIMD instructions (vpmax + vmul) vs SiLU's erf/exp approximation.

6. **Expect 34.5 tok/s baseline with bitnet.cpp on our hardware** -- our custom engine at 100+ tok/s represents a ~3x speedup over bitnet.cpp, achievable via VNNI optimization (2.2x) + prefetching + core pinning.

7. **Do NOT attempt PTQ ternary quantization of Llama-3** -- quality will be unacceptable. Stick to natively trained BitNet models.

8. **Profile zero-weight sparsity per layer** -- even though we don't branch on zeros, knowing ~45% sparsity helps verify our kernels are correctly processing all three ternary states.

---

## References

1. arXiv:2402.17764 -- "The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits" (Ma et al., Microsoft Research, Feb 2024)
2. arXiv:2504.12285 -- "BitNet b1.58 2B4T Technical Report" (Microsoft Research, Apr 2025)
3. arXiv:2504.18415 -- "BitNet v2: Native 4-bit Activations for 1-bit LLMs" (Ma et al., Apr 2025)
4. arXiv:2410.16144 -- "bitnet.cpp: Efficient Edge Inference for Ternary LLMs on CPUs" (Microsoft Research, Oct 2024)
5. llama.cpp PR #8151 -- TQ1_0/TQ2_0 quantization formats (github.com/ggerganov/llama.cpp/pull/8151)
6. HuggingFace -- microsoft/bitnet-b1.58-2B-4T model card (huggingface.co/microsoft/bitnet-b1.58-2B-4T)
7. HuggingFace -- microsoft/bitnet-b1.58-2B-4T-gguf file listing (1.19 GB i2_s format)
8. GitHub -- microsoft/BitNet README (github.com/microsoft/BitNet)
9. Bengio et al. 2013 -- "Estimating or Propagating Gradients Through Stochastic Neurons" (STE original)

## Audit Addendum (2026-04-02)

- **Checkpoint availability is part of the quantization story.** For EdgeLM, the
  best ternary path is the one whose checkpoints, tokenizer contract, and
  runtime conversion path are actually manageable on Windows.
- **Layerwise heterogeneity is likely worth preserving in metadata.** Even in a
  ternary-first engine, embeddings, attention projections, or activations may
  still justify different handling flags.
- **The paper should keep training-native ternary models and post-training
  ternarization sharply separated.** That distinction is easy to blur and would
  otherwise overstate how general BitNet-like results really are.
