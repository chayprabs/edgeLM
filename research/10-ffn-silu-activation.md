# Section 10: FFN / SiLU Activation -- Extended Research

## Overview
The Feed-Forward Network (FFN) block is the **largest compute consumer** in each transformer layer, accounting for approximately 2/3 of all FLOPs. Modern LLMs (LLaMA, BitNet) use the SwiGLU variant:

```
FFN(x) = SiLU(x @ W_gate) * (x @ W_up) @ W_down
```

where `SiLU(x) = x * sigmoid(x)` and `*` denotes element-wise multiplication. This requires **three** matrix multiplications per layer (gate, up, down) plus an element-wise activation and multiply.

For EdgeLM's target models:
- **bitnet_b1_58-3B**: hidden=3200, intermediate=8640, 26 layers, activation=**silu** (SwiGLU)
- **BitNet-b1.58-2B-4T**: hidden=2560, intermediate=6912, 30 layers, activation=**relu2** (squared ReLU)

The critical insight is that these two target models use **different** activation functions -- SiLU vs ReLU squared -- which has major implications for optimization strategy, particularly around activation sparsity.

**Per-layer FFN data movement** (bitnet_b1_58-3B, ternary weights):
- W_gate: 3200 x 8640 = 27.6M weights = ~3.5 MB packed ternary
- W_up: 3200 x 8640 = 27.6M weights = ~3.5 MB packed ternary
- W_down: 8640 x 3200 = 27.6M weights = ~3.5 MB packed ternary
- Total FFN per layer: ~10.5 MB weight data (exceeds L3 slice per core, fits in shared 24MB L3)

## What the Deep Dive Already Covers
- FFN/MLP listed as a key component using AVX-VNNI ternary matmul
- FFN layers identified as weight-bound (best on CPU with AVX2)
- Naive C implementation planned for Phase 1, SIMD optimization in Phase 2
- Hybrid pipeline: CPU handles FFN while iGPU handles attention
- File structure: `ffn.c/h` with ternary kernel files
- Basic SwiGLU formula and computation flow

## New Findings

### 1. Microsoft's BitNet-2B-4T Uses ReLU Squared, Not SiLU
#### Critical Architecture Divergence Between Target Models
- **Source:** HuggingFace config: `microsoft/bitnet-b1.58-2B-4T/config.json` -- `"hidden_act": "relu2"`; vs `1bitLLM/bitnet_b1_58-3B/config.json` -- `"hidden_act": "silu"`
- **Key idea:** Microsoft's latest and best-performing BitNet model (2B-4T, trained on 4T tokens) uses `relu2` (squared ReLU: `max(0,x)^2`) instead of SiLU. This is a deliberate architectural choice from the Primer paper (So et al., arXiv:2109.08668) that produces **naturally sparse activations** -- any input <= 0 produces exactly zero output.
- **Relevance to EdgeLM:** Enormous. ReLU^2 is both cheaper to compute (no sigmoid/exp) AND produces sparse intermediate activations. For ternary models where the matmul is already cheap (add/subtract), the activation function cost becomes proportionally more significant. We must support both activation functions but should optimize differently for each.
- **Estimated impact:** ReLU^2 itself is ~5-10x faster than SiLU per element (no exp). Sparsity exploitation on the down_proj can save 30-60% of that matmul.
- **Implementation complexity:** Low for basic ReLU^2; Medium to exploit sparsity in down_proj.
- **Details:**
  ```c
  // ReLU^2 -- trivially SIMD-friendly
  __m256 relu2_avx2(__m256 x) {
      __m256 zero = _mm256_setzero_ps();
      __m256 relu = _mm256_max_ps(x, zero);
      return _mm256_mul_ps(relu, relu);  // square it
  }

  // SiLU requires sigmoid approximation (see Finding #3)
  // SiLU(x) = x * sigmoid(x)
  ```

### 2. Fused Gate+Up Projection -- Single Memory Pass Over Input
#### Halving Input Memory Traffic for SwiGLU
- **Source:** llamafile matmul optimization (justine.lol/matmul), llama.cpp `build_ffn` architecture
- **Key idea:** The gate and up projections both read the same input vector `x`. Instead of doing two separate matmuls (`x @ W_gate` and `x @ W_up`), interleave them: process one tile of W_gate, then the corresponding tile of W_up, while `x` is still in L1/L2 cache. This halves the input read bandwidth.
- **Relevance to EdgeLM:** On our DDR4-3200 bandwidth-constrained system, every byte of saved memory traffic matters. The input vector (3200 floats = 12.8 KB for FP32, 6.4 KB for FP16) fits in L1 cache. By interleaving gate and up tiles, we read the input once instead of twice.
- **Estimated impact:** 5-15% FFN throughput improvement from reduced memory traffic.
- **Implementation complexity:** Medium -- requires reorganizing the matmul scheduling.
- **Details:** Two implementation strategies:
  1. **Interleaved tiling:** Process a tile of W_gate rows, then the same-indexed tile of W_up rows, keeping input cached.
  2. **Concatenated weights:** Pack W_gate and W_up into a single `[2*intermediate, hidden]` matrix, do one matmul, split the result. llama.cpp does this conceptually via its graph builder. For ternary weights, the packed format naturally supports this since we control the memory layout.

### 3. Fast SiLU via Polynomial Sigmoid Approximation
#### Avoiding exp() in the Hot Path
- **Source:** Intel intrinsics patterns, fast sigmoid algorithms (Schraudolph 1999, various SIMD implementations)
- **Key idea:** SiLU(x) = x * sigmoid(x), and sigmoid is the bottleneck because `expf()` is extremely expensive (~20-50 cycles per element on Golden Cove). Replace sigmoid with a polynomial or rational approximation accurate to ~1e-4 (more than sufficient for inference).
- **Relevance to EdgeLM:** With 8640 intermediate elements per layer, naive SiLU costs ~8640 * 30 cycles = ~260K cycles per layer. A 3rd-degree polynomial approximation costs ~5 cycles per element = ~43K cycles. Savings: ~200K cycles/layer * 26 layers = 5.2M cycles/token.
- **Estimated impact:** Negligible for overall tok/s (matmul dominates) but removes a latency spike in the pipeline.
- **Implementation complexity:** Low
- **Details:** Best approaches for AVX2:
  ```c
  // Method 1: Piecewise rational approximation (fastest, ~4 cycles/element)
  // sigmoid(x) ≈ clamp(0.5 + x * (0.25 - 0.0078125 * x^2), 0, 1) for |x| < 4
  // For |x| >= 4, sigmoid saturates to 0 or 1

  // Method 2: Schraudolph's fast exp via float bit manipulation
  // exp(x) ≈ reinterpret(int(x * 12102203.0 + 1065353216))
  // Then sigmoid(x) = 1 / (1 + exp(-x))

  // Method 3: 5th-order minimax polynomial (most accurate, ~6 cycles)
  // sigmoid(x) ≈ 0.5 + c1*x + c3*x^3 + c5*x^5 for |x| < 6
  // Coefficients from Remez algorithm optimization

  // AVX2 implementation of Method 1:
  __m256 fast_sigmoid_avx2(__m256 x) {
      __m256 half = _mm256_set1_ps(0.5f);
      __m256 quarter = _mm256_set1_ps(0.25f);
      __m256 c2 = _mm256_set1_ps(-0.0078125f);
      __m256 one = _mm256_set1_ps(1.0f);
      __m256 zero = _mm256_setzero_ps();

      __m256 x2 = _mm256_mul_ps(x, x);
      __m256 inner = _mm256_fmadd_ps(c2, x2, quarter);  // 0.25 - 0.0078125*x^2
      __m256 result = _mm256_fmadd_ps(x, inner, half);    // 0.5 + x*(...)
      result = _mm256_min_ps(result, one);                 // clamp to [0, 1]
      result = _mm256_max_ps(result, zero);
      return result;
  }

  __m256 fast_silu_avx2(__m256 x) {
      return _mm256_mul_ps(x, fast_sigmoid_avx2(x));
  }
  ```

### 4. Activation Sparsity Exploitation for Down Projection
#### Skipping Zero Activations in ReLU^2 Models
- **Source:** ProSparse (Song et al., arXiv:2402.13516), PowerInfer (arXiv:2312.12456), DejaVu (arXiv:2310.17157)
- **Key idea:** With ReLU^2 activation (BitNet-2B-4T), a large fraction (60-90%) of the intermediate activations are exactly zero. The down_proj matmul `result = activated @ W_down` can skip columns of W_down corresponding to zero activations, potentially saving 60-90% of that matmul's compute.
- **Relevance to EdgeLM:** The down_proj is the single largest matmul in the FFN (8640 x 3200). If 70% of activations are zero, we only need to process 2592 non-zero rows instead of 8640 -- a 3.3x speedup on the most expensive operation per layer.
- **Estimated impact:** 20-40% total FFN speedup for ReLU^2 models, near-zero benefit for SiLU models (SiLU outputs are never exactly zero).
- **Implementation complexity:** Medium
- **Details:**
  ```c
  // Sparse down_proj for ReLU^2 models
  void ffn_down_proj_sparse(
      float* output,           // [hidden_dim]
      const float* activated,  // [intermediate_dim] -- many zeros with ReLU^2
      const int8_t* W_down,    // [intermediate_dim, hidden_dim] ternary packed
      int intermediate_dim, int hidden_dim)
  {
      memset(output, 0, hidden_dim * sizeof(float));
      for (int i = 0; i < intermediate_dim; i++) {
          if (activated[i] == 0.0f) continue;  // skip zero rows
          // accumulate: output += activated[i] * W_down[i, :]
          // For ternary: this is just scaled add/subtract of W_down row
          ternary_axpy_avx2(output, activated[i], &W_down[i * packed_stride], hidden_dim);
      }
  }
  ```
  For SIMD efficiency, batch non-zero indices and process in groups of 8+.

### 5. T-MAC Lookup Table Approach for FFN
#### Replace Multiply-Accumulate with Table Lookup
- **Source:** T-MAC (github.com/microsoft/T-MAC), BitNet.cpp inference framework
- **Key idea:** T-MAC groups 4 ternary weight bits together, precomputes all 2^4=16 possible partial sums with the corresponding activation values, stores them in a lookup table, then uses `pshufb` (SSSE3/AVX2) for fast table lookup. This replaces multiply-add with table lookup + add.
- **Relevance to EdgeLM:** BitNet.cpp (Microsoft's official framework) is built on T-MAC. On x86, they report 2.37x to 6.17x speedup over llama.cpp. The technique applies equally to all three FFN matmuls. The `pshufb` instruction is available on our i7-12700H.
- **Estimated impact:** 2-4x speedup on FFN matmuls vs naive dequantize-then-multiply.
- **Implementation complexity:** High -- requires careful weight repacking and LUT management.
- **Details:** The key insight is that with ternary weights, each group of 4 weights can only produce 3^4=81 distinct partial sums (but packed as 4 bits = 16 entries per nibble). The LUT fits in a single AVX2 register (32 bytes = 16 x 2-byte entries or 32 x 1-byte entries). `vpshufb` performs 32 parallel lookups in one cycle.

  For the FFN specifically:
  - Gate/Up projections: LUT indexed by 4-bit weight groups, accumulated with activation values
  - Down projection: Same approach, but can combine with sparsity (only build LUTs for non-zero activation groups)

### 6. Fused SwiGLU Kernel -- Eliminate Intermediate Buffer
#### Single-Pass Activation + Element-Wise Multiply
- **Source:** llama.cpp `ggml_swiglu_split` operation, general kernel fusion principles
- **Key idea:** Instead of computing gate_output = SiLU(gate_proj), then doing element-wise multiply gate_output * up_proj, fuse these into a single pass: read both gate and up results, apply SiLU to gate, multiply, write result. This halves the memory traffic for the intermediate buffer (8640 floats = 34 KB).
- **Relevance to EdgeLM:** The 34 KB intermediate buffer just barely fits in L1 (48 KB Golden Cove). A fused kernel keeps it entirely register-resident during the operation, saving one full read+write of 34 KB.
- **Estimated impact:** 5-8% FFN speedup from eliminated memory round-trip.
- **Implementation complexity:** Low
- **Details:**
  ```c
  // Fused SwiGLU: reads gate[] and up[], writes out[]
  void fused_swiglu_avx2(float* out, const float* gate, const float* up, int n) {
      for (int i = 0; i < n; i += 8) {
          __m256 g = _mm256_load_ps(&gate[i]);
          __m256 u = _mm256_load_ps(&up[i]);
          __m256 silu_g = _mm256_mul_ps(g, fast_sigmoid_avx2(g));  // SiLU(gate)
          __m256 result = _mm256_mul_ps(silu_g, u);                // * up
          _mm256_store_ps(&out[i], result);
      }
  }

  // For ReLU^2 models, even simpler and can output sparsity mask:
  void fused_relu2_glu_avx2(float* out, uint32_t* mask, const float* gate,
                             const float* up, int n) {
      __m256 zero = _mm256_setzero_ps();
      for (int i = 0; i < n; i += 8) {
          __m256 g = _mm256_load_ps(&gate[i]);
          __m256 u = _mm256_load_ps(&up[i]);
          __m256 relu = _mm256_max_ps(g, zero);
          __m256 relu2 = _mm256_mul_ps(relu, relu);
          __m256 result = _mm256_mul_ps(relu2, u);
          _mm256_store_ps(&out[i], result);
          // Extract sparsity mask for down_proj optimization
          int nonzero = _mm256_movemask_ps(_mm256_cmp_ps(relu, zero, _CMP_GT_OQ));
          mask[i/8] = nonzero;  // 8-bit mask per group of 8
      }
  }
  ```

### 7. llamafile's Matmul Tiling Strategy for FFN
#### 3x4 Outer-Loop Unrolling with Register Reuse
- **Source:** justine.lol/matmul/ -- llamafile's custom BLAS implementation
- **Key idea:** For FFN matmuls, use a 3x4 tile strategy: load one A-matrix vector, multiply against 4 B-matrix vectors, maintaining 12 accumulator registers. This achieves 8 FMA operations per cycle with shared operands, giving 2x speedup over Intel MKL for L2-fitting matrices.
- **Relevance to EdgeLM:** During autoregressive decoding (batch=1), the FFN matmul is a matrix-vector product (gemv). The 3x4 tiling translates to processing 4 output elements simultaneously while streaming through the weight matrix. For ternary weights, the "FMA" becomes conditional add/subtract, but the register reuse and cache behavior principles still apply.
- **Estimated impact:** 1.5-2x over naive matmul implementation for FFN-sized matrices.
- **Implementation complexity:** Medium
- **Details:** Key architectural observations from llamafile:
  - Alder Lake specifically: 50x speedup on FP16 by properly avoiding E-cores for SIMD
  - For batch=1 gemv: the bottleneck is weight loading, not compute. Tile for maximum L2 cache reuse.
  - Cooperative threading model (duty-based work distribution) avoids futex overhead vs OpenMP
  - For ternary: 4 output accumulators can be maintained in 4 AVX2 registers, with weights decoded from packed ternary in 1-2 registers, leaving ~10 registers for prefetching and temporary values

### 8. BitNet a4.8: Hybrid Activation Quantization for FFN
#### 4-bit Activations + 8-bit Intermediates with Sparsification
- **Source:** BitNet a4.8 (arXiv:2411.04965)
- **Key idea:** Use 4-bit quantized activations for FFN inputs (gate/up projection inputs), then sparsify intermediate states and quantize to 8-bit. This enables INT4/FP4 kernel operations and activates only 55% of parameters. Supports 3-bit KV cache.
- **Relevance to EdgeLM:** If we quantize FFN activations to INT8 or even INT4, the gate/up matmuls can use VNNI integer dot-product instructions (`vpdpbusd`) instead of float operations. VNNI processes 4x INT8 multiplies per cycle per lane vs 1 FP32 FMA. Combined with ternary weights, this means the matmul becomes: lookup ternary action (+1/-1/0) applied to INT8 activation values.
- **Estimated impact:** 2-4x speedup on FFN matmuls if activations are quantized to INT8 and VNNI is used.
- **Implementation complexity:** High -- requires calibration for activation quantization ranges and careful handling of accumulator overflow.
- **Details:** The VNNI path for ternary FFN:
  ```
  For each ternary weight group:
    If weight = +1: accumulator += int8_activation
    If weight = -1: accumulator -= int8_activation
    If weight = 0:  skip

  With VNNI (vpdpbusd): pack 4 activations as uint8, multiply with
  weight bytes (+1/-1/0 encoded), accumulate into int32.
  Processes 32 activations per cycle per YMM register.
  ```

### 9. Activation Recomputation vs Storage Trade-off
#### Eliminating the Intermediate Buffer Entirely
- **Source:** FlashAttention-style memory optimization principles, gradient checkpointing techniques adapted for inference
- **Key idea:** Instead of storing the full gate_proj and up_proj results (2 * 8640 * 4 bytes = 69 KB), compute them tile-by-tile. Process a tile of gate and up, apply activation, multiply, and immediately use the result as input to the corresponding tile of down_proj. This eliminates the need to store any intermediate FFN buffer.
- **Relevance to EdgeLM:** The intermediate buffer (69 KB) exceeds L1 cache (48 KB on Golden Cove). By tiling the FFN computation so each intermediate tile stays in registers/L1, we avoid L2 round-trips. For autoregressive decoding (batch=1), the "tile" is just a group of intermediate dimensions.
- **Estimated impact:** 10-20% FFN speedup from improved cache utilization on bandwidth-constrained DDR4.
- **Implementation complexity:** High -- requires restructuring the FFN computation order.
- **Details:** The tiled FFN pipeline:
  ```
  For each tile of intermediate_dim (e.g., 256 elements):
    1. Compute gate_tile = x @ W_gate[tile_start:tile_end, :]  // small matmul
    2. Compute up_tile = x @ W_up[tile_start:tile_end, :]      // small matmul
    3. activated_tile = SiLU(gate_tile) * up_tile               // in registers
    4. output += activated_tile @ W_down[:, tile_start:tile_end] // accumulate
  ```
  This restructures FFN from 3 large matmuls to many small fused tiles. The tile size should be chosen to keep intermediate values in L1 (~256-512 elements = 1-2 KB).

### 10. E-core Utilization for Element-Wise FFN Operations
#### Offloading Activation Functions to Gracemont Cores
- **Source:** Intel Alder Lake heterogeneous scheduling principles, llamafile Alder Lake observations
- **Key idea:** P-cores (Golden Cove) are optimal for the heavy ternary matmuls (AVX2 256-bit). The element-wise operations (SiLU, ReLU^2, element-wise multiply) are simple scalar-friendly work that E-cores (Gracemont) can handle. While P-cores compute the next matmul, E-cores apply activations to the previous result.
- **Relevance to EdgeLM:** Our 8 E-cores support AVX2 (128-bit SIMD on Gracemont) and can compute SiLU/ReLU^2 on 8640 elements in ~2-5 microseconds. This overlaps with P-core matmul work, effectively hiding activation latency entirely.
- **Estimated impact:** 3-5% overall throughput from latency hiding (activation compute is small relative to matmul).
- **Implementation complexity:** Medium -- requires pipeline scheduling between P-cores and E-cores.
- **Details:** Pipeline schedule per layer:
  ```
  P-cores:  [gate matmul L] [up matmul L]   [down matmul L]   [gate matmul L+1]...
  E-cores:            [idle] [SiLU+mul L]    [idle]            [SiLU+mul L+1]...
  ```
  The key constraint: E-cores must finish activation before P-cores need the result for down_proj. At 8640 elements / 4 lanes / 8 cores = ~270 elements per E-core, this completes in <1 us -- well within the ~50-100 us matmul time.

### 11. Weight Layout Optimization: Column-Major vs Row-Major for FFN
#### Matching Memory Layout to Access Pattern
- **Source:** llamafile matmul optimization (justine.lol/matmul), BLAS conventions
- **Key idea:** For autoregressive decoding (batch=1), FFN matmuls are matrix-vector products. Gate/up projections compute `y = W @ x` where W is [intermediate x hidden]. If W is stored row-major, each output element requires reading one contiguous row -- good for spatial locality. But for ternary packed weights, the packing dimension matters: packing along the hidden dimension (input) allows processing 16-32 weights per packed word against the cached input vector.
- **Relevance to EdgeLM:** The optimal ternary weight layout for FFN depends on the operation:
  - **Gate/Up (hidden -> intermediate):** Pack weights along hidden_dim (input dimension). Each packed group processes against the same input vector, which stays in L1.
  - **Down (intermediate -> hidden):** Pack weights along intermediate_dim (input dimension). For sparse ReLU^2 models, row-skip is cheap in this layout.
- **Estimated impact:** 10-30% depending on current layout inefficiencies.
- **Implementation complexity:** Medium -- weight repacking at model load time.

### 12. ProSparse: Training-Time Sparsity Enhancement
#### Achieving 89% Activation Sparsity in FFN
- **Source:** ProSparse (Song et al., arXiv:2402.13516)
- **Key idea:** Replace SiLU/GELU with ReLU in LLMs, then apply progressive sparsity regularization to push activation sparsity from ~67% (naive ReLU) to 89%+. Achieves up to 4.52x inference speedup with minimal quality loss.
- **Relevance to EdgeLM:** While we can't retrain the target models, this research confirms that: (a) ReLU-family activations produce naturally sparse FFN outputs, (b) the BitNet-2B-4T model's use of ReLU^2 likely has 70-85% sparsity even without special training, and (c) sparsity exploitation in the down_proj is a proven technique worth implementing.
- **Estimated impact:** Validates our sparsity exploitation strategy (Finding #4) with published numbers.
- **Implementation complexity:** N/A (training technique, but informs our inference optimization).

### 13. DejaVu / PowerInfer: Neuron Prediction for FFN
#### Predicting Which FFN Neurons Activate Before Computing Them
- **Source:** DejaVu (arXiv:2310.17157), PowerInfer (arXiv:2312.12456, SOSP 2024)
- **Key idea:** A small predictor network (trained offline) can predict with >90% accuracy which FFN neurons will be activated, BEFORE running the full gate/up projection. This allows skipping inactive columns of W_gate and W_up entirely, not just W_down.
- **Relevance to EdgeLM:** For ReLU^2 models, a tiny predictor (e.g., 256-dim hidden layer, ~50 KB) could predict the ~30% of active neurons, reducing ALL THREE FFN matmuls by 70%. PowerInfer achieves 11.69x speedup on OPT-175B; at our scale (3B), the overhead of prediction is proportionally higher but still worthwhile if sparsity is >70%.
- **Estimated impact:** 1.5-2.5x FFN speedup for ReLU^2 models (conservative estimate at 3B scale).
- **Implementation complexity:** High -- requires training/calibrating a predictor, storing it, and fast prediction path.
- **Details:** Simple predictor design for EdgeLM:
  ```
  predictor_input = RMSNorm output (hidden_dim=2560 or 3200)
  predictor_hidden = predictor_input @ W_pred  (hidden_dim x 256, FP16)
  predictor_output = ReLU(predictor_hidden) @ W_pred2  (256 x intermediate_dim)
  active_mask = predictor_output > threshold

  // Then only compute gate/up/down for active neuron indices
  ```
  The predictor itself is ~2-3 MB total and runs in <5 us.

### 14. Shared Input Quantization for Gate and Up Projections
#### Quantize Once, Use Twice
- **Source:** BitNet architecture design (Wang et al., arXiv:2310.11453), BitNet a4.8 (arXiv:2411.04965)
- **Key idea:** In BitNet, activations are quantized to INT8 before each linear layer. Since gate_proj and up_proj share the same input (the RMSNorm output), the input quantization (absmax scaling + rounding) only needs to happen once. The quantized INT8 input vector is reused for both projections.
- **Relevance to EdgeLM:** Saves one quantization pass per FFN layer. More importantly, the INT8 input vector is half the size of FP32 (3200 bytes vs 12800 bytes for hidden_dim=3200), improving L1 cache utilization when the input is read for both projections.
- **Estimated impact:** 2-3% FFN speedup from eliminated redundant quantization + smaller input footprint.
- **Implementation complexity:** Low
- **Details:** The quantization is simple absmax:
  ```c
  // Quantize once before both gate and up projections
  float scale = 127.0f / max_abs(input, hidden_dim);
  int8_t quantized_input[hidden_dim];
  for (int i = 0; i < hidden_dim; i++)
      quantized_input[i] = (int8_t)roundf(input[i] * scale);

  // Reuse for both: gate_out = W_gate @ quantized_input
  //                  up_out  = W_up  @ quantized_input
  ```

### 15. FlashNorm Gamma Absorption into FFN Weights
#### Eliminating Pre-FFN Normalization Overhead
- **Source:** FlashNorm (Graef et al., arXiv:2407.09577), detailed in Section 09 research
- **Key idea:** The RMSNorm gamma weights before the FFN can be absorbed into W_gate and W_up at model load time: `W_gate' = W_gate * diag(gamma)`. This eliminates the per-element multiply from the normalization preceding every FFN block.
- **Relevance to EdgeLM:** Particularly impactful for ternary models because the absorbed weights can be re-ternarized (gamma just scales columns, and re-ternarization rounds back to {-1,0,+1}). This means zero runtime cost for the pre-FFN normalization multiply.
- **Estimated impact:** ~5% speedup on the normalization+FFN combined path.
- **Implementation complexity:** Low -- one-time operation during weight loading/repacking.

### 16. SwiGLU Parameter Count Adjustment
#### The 8/3 Intermediate Size Ratio
- **Source:** Shazeer (arXiv:2002.05202), LLaMA architecture design (Touvron et al.)
- **Key idea:** SwiGLU/GLU variants use 3 weight matrices instead of 2 (gate + up + down vs just up + down in standard FFN). To maintain the same parameter count, the intermediate dimension is reduced by factor 2/3. The standard ratio is `intermediate_size = (8/3) * hidden_size` rounded to a multiple of 256. LLaMA-3B uses 8640 (= 3200 * 2.7), BitNet-2B-4T uses 6912 (= 2560 * 2.7).
- **Relevance to EdgeLM:** The intermediate dimension directly determines FFN memory traffic. Understanding this ratio is important for estimating bandwidth requirements and choosing tile sizes. The 2.7x ratio means each FFN layer has `3 * hidden * intermediate = 3 * 3200 * 8640 = 82.9M` ternary weights = ~10.4 MB packed.
- **Estimated impact:** Design knowledge, not a new optimization.
- **Implementation complexity:** N/A

## Techniques Comparison Matrix

| Technique | Source | Impact | Complexity | Already in Deep Dive? |
|-----------|--------|--------|------------|----------------------|
| ReLU^2 support (BitNet-2B-4T) | HuggingFace config | High | Low | No |
| Fused gate+up projection | llamafile, llama.cpp | Medium (5-15%) | Medium | No |
| Fast SiLU polynomial approx | SIMD optimization literature | Low-Medium | Low | No |
| Sparse down_proj (ReLU^2) | ProSparse, PowerInfer | High (20-40%) | Medium | No |
| T-MAC LUT for FFN matmuls | Microsoft T-MAC, BitNet.cpp | High (2-4x) | High | Partially |
| Fused SwiGLU kernel | llama.cpp swiglu_split | Medium (5-8%) | Low | No |
| 3x4 matmul tiling | llamafile | Medium (1.5-2x) | Medium | Partially |
| BitNet a4.8 activation quant | arXiv:2411.04965 | High (2-4x) | High | No |
| Tiled FFN (no intermediate buf) | FlashAttention-style | Medium (10-20%) | High | No |
| E-core activation offload | Alder Lake scheduling | Low (3-5%) | Medium | No |
| Weight layout optimization | BLAS conventions | Medium (10-30%) | Medium | Partially |
| ProSparse (89% sparsity) | arXiv:2402.13516 | Validates sparsity | N/A | No |
| Neuron prediction (DejaVu) | arXiv:2310.17157 | High (1.5-2.5x) | High | No |
| Shared input quantization | BitNet architecture | Low (2-3%) | Low | No |
| FlashNorm gamma absorption | arXiv:2407.09577 | Low (5%) | Low | No (in Sec 09) |
| SwiGLU 8/3 ratio | Shazeer 2020 | Design knowledge | N/A | No |

## Recommendations for EdgeLM

Ranked by impact-to-effort ratio:

1. **Support both SiLU and ReLU^2 activation functions** (Low effort, High impact)
   - BitNet-2B-4T uses ReLU^2, bitnet_b1_58-3B uses SiLU. Both must work.
   - ReLU^2 is trivially SIMD and enables sparsity optimizations.

2. **Implement fused SwiGLU/ReLU^2-GLU kernel** (Low effort, Medium impact)
   - Combine activation + element-wise multiply into single pass.
   - For ReLU^2, simultaneously generate sparsity bitmask.

3. **Implement fast polynomial SiLU approximation** (Low effort, Low-Medium impact)
   - Avoid `expf()` in the hot path with a 3-5 term polynomial.
   - Sufficient accuracy for inference (error < 1e-4).

4. **Exploit ReLU^2 sparsity in down_proj** (Medium effort, High impact)
   - Skip zero-activation rows in the down projection.
   - Expected 60-85% of rows skippable, giving 2-4x speedup on down_proj.

5. **Fuse gate+up projections** (Medium effort, Medium impact)
   - Interleave or concatenate to read input vector once.
   - Shared input quantization (quantize to INT8 once for both).

6. **T-MAC LUT-based ternary matmul for FFN** (High effort, High impact)
   - Use `vpshufb` for 32 parallel lookups per cycle.
   - Applies to all three FFN projections.

7. **INT8 activation quantization + VNNI** (High effort, High impact)
   - Quantize FFN inputs to INT8, use `vpdpbusd` for dot products.
   - 4x theoretical throughput over FP32 path.

8. **Tiled FFN computation** (High effort, Medium impact)
   - Fuse gate+up+activation+down into tiled pipeline.
   - Keeps intermediate values in L1/registers, eliminates buffer.

9. **Neuron prediction for ReLU^2 models** (High effort, High impact)
   - Predict active neurons to skip all three projections.
   - Only worthwhile if sparsity > 70% (likely for ReLU^2).

10. **E-core activation offload** (Medium effort, Low impact)
    - Pipeline activation computation on E-cores while P-cores do matmul.
    - Small benefit since activation is already fast relative to matmul.

## References

1. Shazeer, N. (2020). "GLU Variants Improve Transformer." arXiv:2002.05202.
2. So, D. et al. (2021). "Primer: Searching for Efficient Transformers for Language Modeling." arXiv:2109.08668.
3. Wang, H. et al. (2023). "BitNet: Scaling 1-bit Transformers for Large Language Models." arXiv:2310.11453.
4. Ma, S. et al. (2024). "The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits." arXiv:2402.17764.
5. Microsoft (2025). BitNet-b1.58-2B-4T model config. HuggingFace: `microsoft/bitnet-b1.58-2B-4T`.
6. 1bitLLM (2024). bitnet_b1_58-3B model config. HuggingFace: `1bitLLM/bitnet_b1_58-3B`.
7. Song, C. et al. (2024). "ProSparse: Introducing and Enhancing Intrinsic Activation Sparsity within Large Language Models." arXiv:2402.13516.
8. Liu, Z. et al. (2023). "DejaVu: Contextual Sparsity for Efficient LLMs at Inference Time." arXiv:2310.17157.
9. Song, Y. et al. (2023). "PowerInfer: Fast Large Language Model Serving with a Consumer-grade GPU." arXiv:2312.12456.
10. Wang, H. et al. (2024). "BitNet a4.8: 4-bit Activations for 1-bit LLMs." arXiv:2411.04965.
11. Microsoft (2024). T-MAC: Table-Based Low-Bit LLM Inference. GitHub: `microsoft/T-MAC`.
12. Microsoft (2024). BitNet.cpp: 1-Bit LLM Inference Framework. GitHub: `microsoft/BitNet`.
13. Tunstall, J. (2024). llamafile matmul optimization. https://justine.lol/matmul/
14. Karpathy, A. (2023). llama2.c -- Inference Llama 2 in one file of pure C. GitHub: `karpathy/llama2.c`.
15. ggml-org (2024). llama.cpp -- LLM inference in C/C++. GitHub: `ggml-org/llama.cpp`.
16. Graef, R. et al. (2024). "FlashNorm: Fast Normalization for LLMs." arXiv:2407.09577.
17. Schraudolph, N. (1999). "A Fast, Compact Approximation of the Exponential Function." Neural Computation.

## Audit Addendum (2026-04-02)

- **FFN analysis should stay model-specific.** BitNet-style `ReLU^2` and
  Llama-style SwiGLU/SiLU are close enough architecturally to share runtime
  structure, but not close enough to justify one undifferentiated optimization
  story.
- **The next big FFN question is fusion order.** The runtime should benchmark
  whether the best boundary is:
  - gate/up fusion only,
  - full fused intermediate handling,
  - or more conservative staging for register-pressure reasons.
- **Neuron ordering and panel layout deserve explicit FFN benchmarking.** FFN is
  so bandwidth-heavy that small layout changes can matter more than clever
  element-wise math.
