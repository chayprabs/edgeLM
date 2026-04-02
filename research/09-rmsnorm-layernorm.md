# Section 09: RMSNorm & Layer Normalization -- Extended Research

## Overview
RMSNorm is applied before every attention and FFN block in modern LLMs (LLaMA, BitNet b1.58). It normalizes activations by dividing by the root-mean-square, then scaling by learnable weights gamma. For EdgeLM's 3B ternary model on i7-12700H, RMSNorm is **not** the primary bottleneck (matmul dominates), but it executes ~50-60 times per token across all layers, and poor implementation creates unnecessary memory traffic on our bandwidth-constrained DDR4 bus.

**Formula:** `RMSNorm(x) = (x / sqrt(mean(x^2) + eps)) * gamma`

For hidden_dim=3072-3200, each RMSNorm touches ~12-13 KB of data -- fits entirely in L1 cache (48 KB on Golden Cove P-cores).

## What the Deep Dive Already Covers
- RMSNorm as a component of the transformer pipeline
- Naive C implementation planned for Phase 1
- AVX2 optimization with rsqrt + FMA planned for Phase 2
- Listed as part of `rmsnorm_avx2.c` kernel file

## New Findings

### 1. FlashNorm -- Eliminate Gamma Multiply Entirely
#### Weight Absorption into Subsequent Linear Layer
- **Source:** Graef, Wasielewski, Clapp, "FlashNorm: Fast Normalization for LLMs," arXiv:2407.09577
- **Key idea:** Absorb RMSNorm gamma weights into the next linear layer's weight matrix at load time. Since `z = W @ (gamma * x/RMS(x))` = `(W * diag(gamma)) @ (x/RMS(x))`, precompute `W' = W * diag(gamma)` offline.
- **Relevance to EdgeLM:** Extremely high. Every RMSNorm in BitNet b1.58 is followed by a linear layer (QKV projection or FFN). Absorbing gamma during weight repacking at load time eliminates the per-element multiply from every normalization call. For ternary weights, the fused W' can be re-ternarized after absorption since gamma just scales columns.
- **Estimated impact:** ~10% speedup on normalization-heavy paths. On OpenELM-270M with 4-bit quantization: baseline 204 tok/s -> 225 tok/s without gamma multiply.
- **Implementation complexity:** Low -- one-time matrix column scaling during model load.
- **Details:** The remaining runtime normalization becomes just: (1) sum-of-squares reduction, (2) rsqrt, (3) scalar multiply of matmul output by 1/RMS. The scalar division can be deferred and fused into the matmul epilogue -- multiply each output accumulator by the scalar at the end. This means **zero additional memory traffic** for normalization.
- **Savings formula:** For GLU-based FFN (SwiGLU as in BitNet/LLaMA): saves `2f - n` multiplications per layer, where f=FFN dim, n=hidden dim.

### 2. llama.cpp's Unoptimized RMSNorm -- Opportunity
#### Current State of the Art is Actually Naive
- **Source:** `ggml/src/ggml-cpu/ops.cpp` lines 3711-3780, llama.cpp repository
- **Key idea:** llama.cpp's RMSNorm has a `// TODO: optimize` comment. The sum-of-squares loop is **completely scalar** -- no SIMD at all. Only the final scaling pass uses SIMD via `ggml_vec_scale_f32`. Furthermore, the weight multiplication is a separate graph operation (`ggml_mul`), causing an extra memory round-trip.
- **Relevance to EdgeLM:** Major opportunity to beat llama.cpp on this operation. A fused AVX2 implementation that does sum-of-squares + normalize + weight-multiply in minimal passes would be significantly faster.
- **Estimated impact:** 3-5x speedup on the RMSNorm operation itself vs llama.cpp's scalar reduction.
- **Implementation complexity:** Low
- **Details:** llama.cpp code pattern:
  ```c
  // SCALAR sum-of-squares (no SIMD!)
  ggml_float sum = 0.0;  // double precision accumulator
  for (int64_t i00 = 0; i00 < ne00; i00++) {
      sum += (ggml_float)(x[i00] * x[i00]);
  }
  const float mean = sum/ne00;
  float * y = (float *) ((char *) dst->data + ...);
  memcpy(y, x, ne00 * sizeof(float));  // copy input to output
  const float scale = 1.0f/sqrtf(mean + eps);
  ggml_vec_scale_f32(ne00, y, scale);  // SIMD scaling
  // weight multiply happens as separate graph node
  ```

### 3. Optimal AVX2 RMSNorm Kernel Design
#### Fused Three-Phase Implementation
- **Source:** Synthesis from llama.cpp simd-mappings.h, karpathy/llama2.c, Intel intrinsics guide
- **Key idea:** Combine sum-of-squares, normalize, and weight-multiply into a two-pass kernel (pass 1: reduction, pass 2: fused scale+weight), using 4x unrolled FMA accumulators.
- **Relevance to EdgeLM:** Direct implementation target for `rmsnorm_avx2.c`.
- **Estimated impact:** ~200-250 cycles per row for hidden_dim=3072, or ~0.05 us at 4.7 GHz.
- **Implementation complexity:** Medium
- **Details:**
  ```c
  void rmsnorm_avx2(float* __restrict out,
                    const float* __restrict x,
                    const float* __restrict weight,  // NULL if FlashNorm
                    int n, float eps) {
      // Phase 1: Vectorized sum-of-squares, 4x unrolled
      __m256 sum0 = _mm256_setzero_ps();
      __m256 sum1 = _mm256_setzero_ps();
      __m256 sum2 = _mm256_setzero_ps();
      __m256 sum3 = _mm256_setzero_ps();
      int i = 0;
      for (; i + 31 < n; i += 32) {
          __m256 v0 = _mm256_load_ps(x + i);
          __m256 v1 = _mm256_load_ps(x + i + 8);
          __m256 v2 = _mm256_load_ps(x + i + 16);
          __m256 v3 = _mm256_load_ps(x + i + 24);
          sum0 = _mm256_fmadd_ps(v0, v0, sum0);
          sum1 = _mm256_fmadd_ps(v1, v1, sum1);
          sum2 = _mm256_fmadd_ps(v2, v2, sum2);
          sum3 = _mm256_fmadd_ps(v3, v3, sum3);
      }
      // Horizontal reduction
      sum0 = _mm256_add_ps(sum0, sum1);
      sum2 = _mm256_add_ps(sum2, sum3);
      sum0 = _mm256_add_ps(sum0, sum2);
      __m128 hi = _mm256_extractf128_ps(sum0, 1);
      __m128 lo = _mm256_castps256_ps128(sum0);
      __m128 t = _mm_add_ps(hi, lo);
      t = _mm_hadd_ps(t, t);
      t = _mm_hadd_ps(t, t);
      float ss = _mm_cvtss_f32(t);

      // Phase 2: scalar rsqrt
      float scale = 1.0f / sqrtf(ss / n + eps);

      // Phase 3: Fused normalize + weight multiply
      __m256 vscale = _mm256_set1_ps(scale);
      i = 0;
      if (weight) {
          for (; i + 31 < n; i += 32) {
              __m256 vx0 = _mm256_load_ps(x + i);
              __m256 vw0 = _mm256_load_ps(weight + i);
              _mm256_store_ps(out + i, _mm256_mul_ps(vw0, _mm256_mul_ps(vx0, vscale)));
              // ... repeat for +8, +16, +24
          }
      } else {  // FlashNorm: no weight multiply
          for (; i + 31 < n; i += 32) {
              _mm256_store_ps(out + i, _mm256_mul_ps(_mm256_load_ps(x + i), vscale));
              // ... repeat for +8, +16, +24
          }
      }
  }
  ```
  Key design: 4x unrolled FMA hides latency, aligned loads assume 64-byte aligned buffers, fused weight multiply avoids extra memory round-trip.

### 4. Reciprocal Square Root: rsqrt + Newton-Raphson
#### Hardware Approximation with Refinement
- **Source:** Intel Intrinsics Guide (felixcloutier.com/x86/rsqrtps), geometrian.com fast sqrt tutorial
- **Key idea:** `_mm256_rsqrt_ps` gives ~11.5 bits of accuracy in ~4 cycles. One Newton-Raphson step brings it to full FP32 precision (~23 bits) in ~8-10 total cycles. Exact `sqrtf + div` takes ~23-30 cycles.
- **Relevance to EdgeLM:** Since the rsqrt is computed once per row (scalar), even `1.0f/sqrtf()` is fine (~12 cycles for one scalar). For batch processing multiple rows, the vectorized rsqrt + NR saves significant time.
- **Estimated impact:** Negligible for single-row; meaningful for batch normalization of multiple rows.
- **Implementation complexity:** Low
- **Details:** Newton-Raphson refinement pattern:
  ```c
  __m256 y0 = _mm256_rsqrt_ps(x);
  __m256 half_x = _mm256_mul_ps(_mm256_set1_ps(0.5f), x);
  __m256 y0_sq = _mm256_mul_ps(y0, y0);
  __m256 correction = _mm256_fnmadd_ps(half_x, y0_sq, _mm256_set1_ps(1.5f));
  __m256 y1 = _mm256_mul_ps(y0, correction);  // full FP32 precision
  ```

### 5. ggml Horizontal Reduction Pattern
#### Standard AVX2 Tree Reduction
- **Source:** `ggml/src/ggml-cpu/simd-mappings.h` lines 568-620, llama.cpp
- **Key idea:** Tree-reduce 4 x __m256 accumulators to a single scalar: pairwise add registers, extract high/low 128-bit halves, two horizontal adds.
- **Relevance to EdgeLM:** Standard pattern to adopt for the sum-of-squares reduction.
- **Implementation complexity:** Low
- **Details:**
  ```c
  // Tree reduce 4 registers to 1
  x[0] = _mm256_add_ps(x[0], x[2]);
  x[1] = _mm256_add_ps(x[1], x[3]);
  x[0] = _mm256_add_ps(x[0], x[1]);
  // Extract and horizontal reduce
  __m128 t0 = _mm_add_ps(_mm256_castps256_ps128(x[0]),
                          _mm256_extractf128_ps(x[0], 1));
  __m128 t1 = _mm_hadd_ps(t0, t0);
  float result = _mm_cvtss_f32(_mm_hadd_ps(t1, t1));
  ```

### 6. Dynamic Tanh (DyT) -- Normalization-Free Alternative
#### Replace RMSNorm with Element-wise tanh
- **Source:** Zhu, Chen, He, LeCun, Liu, "Transformers without Normalization," CVPR 2025. arXiv:2503.10622
- **Key idea:** Replace `RMSNorm(x)` with `DyT(x) = gamma * tanh(alpha * x) + beta`, where alpha is a learned scalar. Eliminates the reduction entirely -- becomes purely element-wise.
- **Relevance to EdgeLM:** Eliminates horizontal reduction and rsqrt. Maps perfectly to SIMD: `_mm256_mul_ps(alpha, x)` + vectorized tanh. However, **requires retraining from scratch** -- existing BitNet b1.58 models use RMSNorm.
- **Estimated impact:** Layer-only: 52% faster than RMSNorm. Full model: ~8% faster. But requires model retraining.
- **Implementation complexity:** Low (implementation), Very High (requires new model)
- **Details:**
  - LLaMA pretraining results: identical accuracy at 7B/13B/34B/70B (zero-shot 0.513/0.529/0.536/0.549 matching RMSNorm exactly)
  - Alpha initialization: 0.8 for attention, 0.2 for other layers (7B); smaller alpha for larger models
  - tanh can be approximated with Pade approximant in ~10 AVX2 instructions per 8 floats
  - Critical: removing tanh entirely (identity) causes divergence -- the nonlinear squashing is essential

### 7. Partial RMSNorm (pRMSNorm)
#### Compute RMS from Subset of Elements
- **Source:** Zhang & Sennrich, "Root Mean Square Layer Normalization," NeurIPS 2019. arXiv:1910.07467
- **Key idea:** Estimate RMS from only p% of input elements. At p=25%, compute sum-of-squares over 800 elements instead of 3200, 4x reduction in the reduction loop.
- **Relevance to EdgeLM:** The scaling pass still touches all elements, so savings are only in the reduction phase. For hidden_dim=3200, the reduction is already ~100 cycles -- saving 75 cycles is negligible vs the ~10ms token budget.
- **Estimated impact:** Minimal (<1% of total inference time)
- **Implementation complexity:** Low
- **Details:** The original paper showed comparable accuracy. However, for a production engine targeting a specific model, the model must have been trained with pRMSNorm to use it at inference.

### 8. Normalization is Memory-Bound, Not Compute-Bound
#### Fusion is the Primary Optimization Strategy
- **Source:** Triton fused LayerNorm tutorial (triton-lang.org), oneDNN documentation
- **Key idea:** Normalization throughput is limited by memory bandwidth, not arithmetic. Triton's fused kernel achieves 988 GB/s vs PyTorch's 560 GB/s at N=8192 -- the improvement comes from reducing memory round-trips, not reducing FLOPs.
- **Relevance to EdgeLM:** On DDR4-3200 (~40 GB/s), every unnecessary write-then-read costs ~0.3 us per 12 KB vector. The ideal implementation: read x once, compute RMS in registers, and feed normalized values directly into the matmul without writing intermediate results to DRAM.
- **Estimated impact:** Up to 2x improvement on normalization throughput via fusion.
- **Implementation complexity:** Medium-High (requires tight coupling with matmul kernel)
- **Details:** Three fusion strategies ranked by feasibility:
  1. **FlashNorm + scalar epilogue** (easiest): Absorb gamma into weights, apply 1/RMS as scalar multiply on matmul output. Zero extra memory traffic.
  2. **Inline normalization in matmul prologue** (medium): The matmul kernel reads x anyway -- compute sum-of-squares during the first pass over x, then normalize on-the-fly during accumulation.
  3. **Full pipeline fusion** (hardest): Fuse residual add + RMSNorm + matmul into a single kernel. Maximum bandwidth savings but complex implementation.

### 9. Numerical Stability: FP32 is the Natural Path on x86
#### No Precision Concerns for CPU Implementation
- **Source:** PyTorch nn.RMSNorm docs, HuggingFace Transformers patterns, Candle implementation
- **Key idea:** GPU implementations upcast FP16/BF16 to FP32 for normalization. On x86 CPUs, FP32 is the native SIMD width -- AVX2 operates on 8xFP32 natively. No upcasting needed.
- **Relevance to EdgeLM:** This is a CPU advantage. For hidden_dim=3200, sum of 3200 squared FP32 values stays well within FP32 range (max ~3.4e38). No double-precision accumulator needed unless activations have extreme magnitude.
- **Estimated impact:** Simplifies implementation -- no mixed-precision concerns.
- **Implementation complexity:** N/A (simplification)
- **Details:** Standard epsilon values: 1e-5 (LLaMA) or 1e-6 (BitNet). Add epsilon *before* the sqrt: `1/sqrt(mean + eps)`. llama.cpp uses double-precision accumulator as extra caution, but FP32 with 4 independent accumulators provides sufficient precision through compensating summation.

### 10. karpathy/llama2.c -- Fused Reference Implementation
#### Weight Multiply Inside Normalization Loop
- **Source:** `run.c` in karpathy/llama2.c repository
- **Key idea:** Unlike llama.cpp which separates normalization and weight multiply into two graph operations, llama2.c fuses them in a single function with two loops: (1) sum-of-squares, (2) scale + weight multiply.
- **Relevance to EdgeLM:** This is the right pattern for EdgeLM. One function, two passes over data, both in L1 cache. No intermediate write to DRAM.
- **Implementation complexity:** Low
- **Details:**
  ```c
  void rmsnorm(float* o, float* x, float* weight, int size) {
      float ss = 0.0f;
      for (int j = 0; j < size; j++) {
          ss += x[j] * x[j];
      }
      ss /= size;
      ss += 1e-5f;
      ss = 1.0f / sqrtf(ss);
      for (int j = 0; j < size; j++) {
          o[j] = weight[j] * (ss * x[j]);
      }
  }
  ```

### 11. BitNet b1.58 Normalization Architecture
#### RMSNorm Before Quantization in BitLinear
- **Source:** Wang et al., "BitNet: Scaling 1-Bit Transformers," arXiv:2310.11453; Ma et al., "The Era of 1-bit LLMs," arXiv:2402.17764
- **Key idea:** BitNet applies Sub-Layer Normalization (SubLN) before quantization. BitNet b1.58 uses standard RMSNorm in LLaMA-compatible architecture. The normalization controls activation magnitude before the hard ternary quantization step, preventing catastrophic quantization error.
- **Relevance to EdgeLM:** Confirms the target model uses standard RMSNorm, making FlashNorm directly applicable. The RMSNorm on activations is the primary floating-point operation per layer (besides activation functions), since ternary matmul is just conditional add/subtract.
- **Implementation complexity:** N/A (architecture confirmation)
- **Details:** HuggingFace BitNetQuantConfig uses `rms_norm_eps=1e-6` and optional RMSNorm before activation quantization.

### 12. SmoothQuant Interaction with Normalization
#### Per-Channel Scaling Absorbed into Gamma
- **Source:** Xiao et al., "SmoothQuant," ICML 2023. arXiv:2211.10438
- **Key idea:** Activation outliers concentrate in specific channels. A per-channel scaling factor `s` migrates quantization difficulty from activations to weights: `Y = (X * diag(s)^-1) @ (diag(s) * W)`. This `s` can be fused with RMSNorm gamma: `gamma_fused = gamma / s`.
- **Relevance to EdgeLM:** Creates a three-way fusion opportunity: RMSNorm + SmoothQuant + Linear = `modified_W @ (x / RMS(x))`. For ternary models, activation outlier handling is critical since quantization to {-1, 0, +1} is extremely lossy.
- **Estimated impact:** Depends on activation distribution; could significantly reduce quantization error.
- **Implementation complexity:** Medium (requires calibration data to compute `s`)

### 13. oneDNN RMSNorm Implementation Choices
#### Intel's Own Optimization Decisions
- **Source:** oneDNN documentation (uxlfoundation.github.io)
- **Key idea:** Intel's oneDNN includes `dnnl_rms_norm` mode within layer normalization primitive. Key choices: statistics always in FP32 regardless of input type, optimal when last logical axis is last in physical memory, supports in-place operation.
- **Relevance to EdgeLM:** Validates our design choices (FP32 compute, contiguous memory layout). For zero-dependency EdgeLM, studying oneDNN's choices is more useful than depending on it.
- **Implementation complexity:** N/A (reference)

### 14. AVX2 Frequency Throttling -- Non-Issue for Inference
#### Warm SIMD Units During Continuous Generation
- **Source:** Travis Downs (travisdowns.github.io)
- **Key idea:** AVX2 256-bit instructions can trigger dispatch throttling after ~680 us of idle time, causing ~25% throughput loss for 8-20 us during voltage ramp-up. During continuous inference, SIMD units stay warm and this penalty does not apply.
- **Relevance to EdgeLM:** Non-issue during token generation. Only matters for cold-start of the first token. Can be mitigated by running a dummy SIMD warmup before first inference.
- **Implementation complexity:** Trivial (warmup loop)

### 15. In-Place Normalization
#### Avoid Separate Output Buffer
- **Source:** oneDNN documentation, general optimization practice
- **Key idea:** Compute RMSNorm in-place (src == dst) to halve memory traffic in the scaling pass. Read x, compute sum-of-squares, then overwrite x with normalized values.
- **Relevance to EdgeLM:** Saves one write of hidden_dim floats (~12.8 KB) per normalization. Small absolute savings but good practice.
- **Estimated impact:** ~25% less memory traffic for the normalization operation.
- **Implementation complexity:** Low

## Techniques Comparison Matrix

| Technique | Source | Impact | Complexity | Already in Deep Dive? |
|-----------|--------|--------|------------|----------------------|
| FlashNorm (gamma absorption) | arXiv:2407.09577 | High | Low | No |
| AVX2 vectorized sum-of-squares | Intel intrinsics | High | Low | Partially (mentioned, not detailed) |
| 4x unrolled FMA accumulators | llamafile patterns | Medium | Low | No |
| Fused normalize + weight multiply | karpathy/llama2.c | Medium | Low | No |
| rsqrt + Newton-Raphson | Intel ISA | Low | Low | Partially |
| Matmul epilogue scalar fusion | FlashNorm paper | High | Medium | No |
| Dynamic Tanh (DyT) | arXiv:2503.10622 | High | Very High (retraining) | No |
| Partial RMSNorm (pRMSNorm) | arXiv:1910.07467 | Minimal | Low | No |
| SmoothQuant + gamma fusion | arXiv:2211.10438 | Medium | Medium | No |
| In-place operation | oneDNN | Low | Low | No |
| Double-precision accumulator | llama.cpp | Low (safety) | Low | No |
| Inline norm in matmul prologue | Novel synthesis | High | High | No |
| SIMD warmup for cold start | Travis Downs | Trivial | Trivial | No |
| ggml tree reduction pattern | llama.cpp | Low | Low | No |
| Memory-bound analysis (fusion focus) | Triton tutorial | Strategic | N/A | No |

## Recommendations for EdgeLM

Ranked by impact-to-effort ratio:

1. **FlashNorm gamma absorption (MUST DO).** At model load time, multiply each column of the subsequent linear layer's weight matrix by the corresponding gamma value. This is a one-time cost during weight repacking and eliminates the per-element gamma multiply from every RMSNorm call at inference. The 1/RMS scalar can then be applied as a matmul epilogue -- zero extra memory traffic. This is the single highest-impact optimization.

2. **Fused AVX2 RMSNorm kernel.** Implement the two-pass kernel: (a) 4x-unrolled FMA sum-of-squares with tree reduction, (b) fused scale+weight multiply (if not using FlashNorm). Use `_mm256_load_ps` (aligned) since all EdgeLM buffers are 64-byte aligned. This replaces llama.cpp's scalar reduction with a 3-5x faster vectorized version.

3. **Matmul epilogue fusion.** With FlashNorm, the only remaining normalization work is computing the 1/RMS scalar. Apply this as a single multiply on each matmul output element in the accumulation epilogue. This completely eliminates RMSNorm as a separate operation in the pipeline.

4. **In-place normalization.** When FlashNorm is not used (e.g., for the final RMSNorm before the output projection), normalize in-place to avoid allocating a separate output buffer. Saves ~12.8 KB write per call.

5. **Scalar sqrtf for rsqrt.** Since the rsqrt is computed once per row (not per element), `1.0f/sqrtf(mean + eps)` is perfectly adequate. Only use vectorized rsqrt + Newton-Raphson if batch-normalizing multiple rows simultaneously.

6. **Consider DyT for future custom models.** If EdgeLM ever trains its own ternary model, Dynamic Tanh normalization eliminates the horizontal reduction entirely and maps perfectly to element-wise SIMD. Keep this in mind for the research paper as a "future work" direction.

7. **Epsilon = 1e-6.** Match BitNet b1.58's epsilon value. Add epsilon before the sqrt.

## Performance Budget

For a 3B model with ~30 layers, each token requires ~60 RMSNorm calls (2 per layer: pre-attention + pre-FFN):
- **Without optimization:** ~60 * 1 us = 60 us per token (scalar reduction + separate weight multiply)
- **With AVX2 kernel:** ~60 * 0.2 us = 12 us per token
- **With FlashNorm + matmul fusion:** ~60 * 0.05 us = 3 us per token (just the reduction, scale folded into matmul)
- **Token budget at 100 tok/s:** 10,000 us per token
- **RMSNorm as % of budget:** 0.03% (with FlashNorm) to 0.6% (without optimization)

**Conclusion:** RMSNorm is NOT a bottleneck for EdgeLM. Even a naive implementation would be acceptable. However, FlashNorm gamma absorption is still worth implementing because it reduces memory traffic (fewer bytes read/written per layer), and the matmul epilogue fusion simplifies the pipeline.

## References

1. Zhang, B. & Sennrich, R. "Root Mean Square Layer Normalization." NeurIPS 2019. arXiv:1910.07467
2. Graef, Wasielewski, Clapp. "FlashNorm: Fast Normalization for LLMs." arXiv:2407.09577
3. Zhu, Chen, He, LeCun, Liu. "Transformers without Normalization." CVPR 2025. arXiv:2503.10622
4. Wang et al. "BitNet: Scaling 1-Bit Transformers for Large Language Models." arXiv:2310.11453
5. Ma et al. "The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits." arXiv:2402.17764
6. Xiao et al. "SmoothQuant: Accurate and Efficient Post-Training Quantization for LLMs." ICML 2023. arXiv:2211.10438
7. Touvron et al. "LLaMA: Open and Efficient Foundation Language Models." arXiv:2302.13971
8. llama.cpp GGML source: `ggml/src/ggml-cpu/ops.cpp` (RMSNorm), `simd-mappings.h` (AVX2 reduction)
9. karpathy/llama2.c: `run.c` (fused RMSNorm reference)
10. Intel Intrinsics Guide: `_mm256_rsqrt_ps`, `_mm256_fmadd_ps`, `_mm256_extractf128_ps`
11. oneDNN documentation: Layer Normalization / RMSNorm primitive (uxlfoundation.github.io)
12. Triton fused LayerNorm tutorial (triton-lang.org)
13. Tunney, J. "LLM Inference on CPUs" (justine.lol/matmul)
14. Travis Downs. AVX2 frequency throttling analysis (travisdowns.github.io)
15. Loshchilov et al. "nGPT: Normalized Transformer with Representation Learning on the Hypersphere." arXiv:2410.01131
16. HuggingFace Candle: `candle-nn/src/layer_norm.rs` (RMSNorm implementation)
17. HuggingFace Transformers: BitNetQuantConfig documentation
18. Zhu et al. "Scalable MatMul-free Language Modeling." arXiv:2406.02528

## Audit Addendum (2026-04-02)

- **Residual-add + RMSNorm fusion deserves dedicated measurement.** The highest
  value optimization here may be reducing memory traffic across the residual
  boundary rather than making the normalization math itself faster.
- **Per-core specialization may be justified.** A slightly smaller E-core
  normalization kernel could outperform a P-core-tuned variant when the
  orchestration side owns these ops.
- **Normalization telemetry should include bytes touched.** This keeps the file's
  main conclusion honest: the bottleneck is usually memory traffic, not flops.
