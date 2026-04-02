# Section 12: FP8 / INT8 Quantization for KV Cache -- Deep Research

## Overview

This document covers deep research findings on INT8/FP8 quantization and dequantization
for KV cache, with focus on AVX2 SIMD vectorization for the EdgeLM inference engine.
Key areas: quantization block formats, AVX2/AVX-VNNI vectorized kernels from llama.cpp,
per-channel vs per-token quantization strategies (KIVI paper), scale factor management,
and fused quantize-on-write / dequantize-on-read patterns.

All findings evaluated for: i7-12700H, DDR4-3200 (~40 GB/s), AVX2 + AVX-VNNI (VEX-encoded
VPDPBUSD, NOT AVX-512), BitNet 3B with GQA (8 KV heads, head_dim=128), 2048-token context,
targeting 100+ tok/s.

---

## Topic A: Quantization Block Formats and Data Structures

### Finding A1: llama.cpp Q8_0 Block Structure -- The Gold Standard for INT8 KV Cache

**Source:** `ggml/src/ggml-common.h` (ggml-org/llama.cpp, GitHub)
https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-common.h

**Key idea:** The Q8_0 format uses a simple symmetric quantization scheme with a single
FP16 scale factor per 32-element block. This is the most natural choice for KV cache
INT8 quantization due to its simplicity, low overhead, and excellent SIMD compatibility.

**Block structure:**

```c
#define QK8_0 32

typedef struct {
    ggml_half d;       // scale factor (FP16, 2 bytes)
    int8_t qs[QK8_0];  // 32 quantized values (32 bytes)
} block_q8_0;
// Total size: 34 bytes per block of 32 elements
// Effective bits per element: 34/32 * 8 = 8.5 bits
```

**Quantization formula (symmetric, absmax):**
```
scale = absmax(block) / 127.0
quantized[i] = round(value[i] / scale)
```

**Dequantization formula:**
```
value[i] = quantized[i] * scale
```

**Relevance to EdgeLM:** Q8_0 is the optimal format for our KV cache. With head_dim=128,
each attention head's K or V vector spans exactly 4 Q8_0 blocks (128/32 = 4). The 34-byte
block aligns naturally with cache lines. Memory savings: FP32 KV cache for 2048 tokens,
8 KV heads, head_dim 128 = 2048 * 8 * 128 * 4 bytes = 8 MB per K or V. With Q8_0:
2048 * 8 * (128 * 8.5/8) bytes = ~2.2 MB per K or V. **3.6x compression.**

---

### Finding A2: Q8_1 Block Structure -- Includes Precomputed Sum for Dot Products

**Source:** `ggml/src/ggml-common.h` (ggml-org/llama.cpp, GitHub)

**Block structure:**

```c
#define QK8_1 32

typedef struct {
    union {
        struct {
            ggml_half d;   // scale factor (FP16)
            ggml_half s;   // d * sum(qs[i]) (FP16)
        };
        ggml_half2 ds;
    };
    int8_t qs[QK8_1];     // 32 quantized values
} block_q8_1;
// Total size: 36 bytes per block of 32 elements
// Effective bits per element: 36/32 * 8 = 9.0 bits
```

**Key difference from Q8_0:** The `s` field stores `d * sum(qs[i])`, a precomputed weighted
sum. This enables efficient dot products with asymmetric quantized formats (like Q4_1) where
a zero-point offset needs to be subtracted. The sum term cancels out the cross-product with
the zero-point.

**Relevance to EdgeLM:** Q8_1 is useful if we ever need to compute dot products between
quantized KV cache values and 4-bit quantized attention weights. For our use case (ternary
weights, FP32 queries), Q8_0 is sufficient and saves 2 bytes per block.

---

### Finding A3: Q4_0 and Q4_1 for Aggressive KV Cache Compression

**Source:** `ggml/src/ggml-common.h` (ggml-org/llama.cpp, GitHub)

**Q4_0 block (symmetric, 4-bit):**
```c
#define QK4_0 32
typedef struct {
    ggml_half d;            // scale factor (FP16, 2 bytes)
    uint8_t qs[QK4_0 / 2]; // 16 bytes storing 32 nibbles
} block_q4_0;
// Total: 18 bytes per 32 elements = 4.5 bits/element
```

**Q4_1 block (asymmetric, 4-bit with min):**
```c
#define QK4_1 32
typedef struct {
    union {
        struct {
            ggml_half d;   // scale (FP16)
            ggml_half m;   // minimum (FP16)
        };
        ggml_half2 dm;
    };
    uint8_t qs[QK4_1 / 2]; // 16 bytes storing 32 nibbles
} block_q4_1;
// Total: 20 bytes per 32 elements = 5.0 bits/element
```

**Relevance to EdgeLM:** Q4_0/Q4_1 provide ~8x compression but with measurable quality loss.
llama.cpp PR #4312 confirmed minimal perplexity impact for K-cache Q4_1 on CPU. For EdgeLM,
Q8_0 is the safe default; Q4_0 is a stretch goal if memory pressure requires it (e.g.,
longer contexts or larger models).

---

## Topic B: AVX2 Vectorized Quantization Kernels

### Finding B1: AVX2 Q8_0 Quantization -- Complete Vectorized Implementation

**Source:** `ggml/src/ggml-cpu/arch/x86/quants.c` (ggml-org/llama.cpp, GitHub)
https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cpu/arch/x86/quants.c

**Key idea:** The AVX2 quantize_row_q8_0 processes one 32-element block per iteration using
4 YMM registers. The algorithm: (1) load 32 floats, (2) compute absmax via SIMD horizontal
reduction, (3) scale all values, (4) round and convert to int32, (5) pack int32 -> int16 ->
int8 with lane-crossing permutation fix.

**Complete AVX2 implementation:**

```c
void quantize_row_q8_0(const float * GGML_RESTRICT x, void * GGML_RESTRICT vy,
                       int64_t k) {
    assert(QK8_0 == 32);
    assert(k % QK8_0 == 0);
    const int nb = k / QK8_0;
    block_q8_0 * GGML_RESTRICT y = vy;

    for (int i = 0; i < nb; i++) {
        // Step 1: Load 32 floats into 4 AVX registers (8 floats each)
        __m256 v0 = _mm256_loadu_ps(x);
        __m256 v1 = _mm256_loadu_ps(x + 8);
        __m256 v2 = _mm256_loadu_ps(x + 16);
        __m256 v3 = _mm256_loadu_ps(x + 24);
        x += 32;

        // Step 2: Compute absolute max across all 32 floats
        // Clear sign bits to get absolute values
        const __m256 signBit = _mm256_set1_ps(-0.0f);
        __m256 maxAbs = _mm256_andnot_ps(signBit, v0);
        maxAbs = _mm256_max_ps(maxAbs, _mm256_andnot_ps(signBit, v1));
        maxAbs = _mm256_max_ps(maxAbs, _mm256_andnot_ps(signBit, v2));
        maxAbs = _mm256_max_ps(maxAbs, _mm256_andnot_ps(signBit, v3));

        // Horizontal max reduction: 256-bit -> 128-bit -> scalar
        __m128 max4 = _mm_max_ps(_mm256_extractf128_ps(maxAbs, 1),
                                 _mm256_castps256_ps128(maxAbs));
        max4 = _mm_max_ps(max4, _mm_movehl_ps(max4, max4));
        max4 = _mm_max_ss(max4, _mm_movehdup_ps(max4));
        const float maxScalar = _mm_cvtss_f32(max4);

        // Step 3: Compute scale and inverse scale
        const float d = maxScalar / 127.f;
        y[i].d = GGML_CPU_FP32_TO_FP16(d);
        const float id = (maxScalar != 0.0f) ? 127.f / maxScalar : 0.0f;
        const __m256 mul = _mm256_set1_ps(id);

        // Step 4: Scale, round, convert float32 -> int32
        v0 = _mm256_mul_ps(v0, mul);
        v1 = _mm256_mul_ps(v1, mul);
        v2 = _mm256_mul_ps(v2, mul);
        v3 = _mm256_mul_ps(v3, mul);

        v0 = _mm256_round_ps(v0, _MM_ROUND_NEAREST);
        v1 = _mm256_round_ps(v1, _MM_ROUND_NEAREST);
        v2 = _mm256_round_ps(v2, _MM_ROUND_NEAREST);
        v3 = _mm256_round_ps(v3, _MM_ROUND_NEAREST);

        __m256i i0 = _mm256_cvtps_epi32(v0);
        __m256i i1 = _mm256_cvtps_epi32(v1);
        __m256i i2 = _mm256_cvtps_epi32(v2);
        __m256i i3 = _mm256_cvtps_epi32(v3);

        // Step 5: Pack int32 -> int16 -> int8 with permutation fix
        i0 = _mm256_packs_epi32(i0, i1);  // 8xi32,8xi32 -> 16xi16
        i2 = _mm256_packs_epi32(i2, i3);  // 8xi32,8xi32 -> 16xi16
        i0 = _mm256_packs_epi16(i0, i2);  // 16xi16,16xi16 -> 32xi8

        // Fix lane-crossing ordering from packs instructions
        const __m256i perm = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);
        i0 = _mm256_permutevar8x32_epi32(i0, perm);

        // Store 32 bytes of quantized values
        _mm256_storeu_si256((__m256i *)y[i].qs, i0);
    }
}
```

**Why the permutation is needed:** AVX2 `_mm256_packs_epi32` operates on 128-bit lanes
independently. After packing int32 -> int16 -> int8, the bytes from the low 128-bit lane
and high 128-bit lane are interleaved incorrectly:
- Without permute: [A0 A1 B0 B1 A2 A3 B2 B3] (elements from lanes mixed)
- After permute(0,4,1,5,2,6,3,7): [A0 A1 A2 A3 B0 B1 B2 B3] (sequential order)

**Performance characteristics:**
- Processes 32 floats (128 bytes input) -> 34 bytes output per iteration
- 4 loads + 4 muls + 4 rounds + 4 cvts + 2 packs + 1 pack16 + 1 permute + 1 store
- ~20 SIMD instructions per block of 32 elements
- At ~1 instruction/cycle throughput: ~20 cycles per 32 elements = 0.625 cycles/element
- For head_dim=128: 4 blocks = ~80 cycles = ~33 ns at 2.4 GHz

**Relevance to EdgeLM:** This is the exact kernel we need for quantize-on-write to the
KV cache. When a new K or V vector arrives (head_dim=128 FP32 values), we call this to
produce 4 block_q8_0 structs (136 bytes total) before writing to the cache. The kernel
is fast enough (~33 ns) to be negligible compared to the matmul cost of producing K/V.

---

### Finding B2: Horizontal Max Reduction Pattern for Absmax Computation

**Source:** `ggml/src/ggml-cpu/arch/x86/quants.c` (ggml-org/llama.cpp, GitHub)

**Key idea:** Computing the absolute maximum across 32 floats requires a multi-step
horizontal reduction. The pattern used in llama.cpp is optimal for AVX2.

**Detailed breakdown:**

```c
// Phase 1: Element-wise max across 4 vectors (all absolute values)
const __m256 signBit = _mm256_set1_ps(-0.0f);  // 0x80000000
__m256 maxAbs = _mm256_andnot_ps(signBit, v0);  // clear sign bit = abs()
maxAbs = _mm256_max_ps(maxAbs, _mm256_andnot_ps(signBit, v1));
maxAbs = _mm256_max_ps(maxAbs, _mm256_andnot_ps(signBit, v2));
maxAbs = _mm256_max_ps(maxAbs, _mm256_andnot_ps(signBit, v3));
// Now maxAbs has element-wise max of abs values: 8 floats

// Phase 2: Reduce 8 floats -> 4 floats (cross-lane)
__m128 max4 = _mm_max_ps(
    _mm256_extractf128_ps(maxAbs, 1),    // high 128 bits
    _mm256_castps256_ps128(maxAbs)        // low 128 bits (free, no instruction)
);
// max4 = [max(0,4), max(1,5), max(2,6), max(3,7)]

// Phase 3: Reduce 4 floats -> 2 floats
max4 = _mm_max_ps(max4, _mm_movehl_ps(max4, max4));
// max4 = [max(0,4,2,6), max(1,5,3,7), ...]

// Phase 4: Reduce 2 floats -> 1 float
max4 = _mm_max_ss(max4, _mm_movehdup_ps(max4));
// max4[0] = absolute maximum of all 32 elements

const float maxScalar = _mm_cvtss_f32(max4);
```

**Instruction count:** ~7 instructions for the reduction (after the initial element-wise max).
The `_mm256_castps256_ps128` is a compiler hint (zero instructions). Total: ~10 instructions
for absmax of 32 floats.

**Relevance to EdgeLM:** This exact pattern is needed for our quantize-on-write kernel.
An alternative approach for KV cache specifically: since we know the distribution of
attention K/V values is roughly Gaussian, we could use a running exponential moving average
of the scale factor to avoid recomputing absmax each time. But the overhead is so small
(~10 instructions) that the exact computation is preferable for accuracy.

---

### Finding B3: INT8 Dot Product with AVX2 maddubs + madd Pattern

**Source:** `ggml/src/ggml-cpu/arch/x86/quants.c` (ggml-org/llama.cpp, GitHub)

**Key idea:** The `mul_sum_i8_pairs_float` function computes the dot product of two
32-element INT8 vectors, returning the result as 8 packed FP32 values. This is the
critical inner-loop operation for dequantized attention score computation.

**The sign-trick for signed*signed multiplication:**

AVX2's `_mm256_maddubs_epi16` computes unsigned*signed byte products. To multiply
two signed vectors, llama.cpp uses `_mm256_sign_epi8` to convert:
- `ax = sign(x, x)` -- makes x unsigned (absolute value)
- `sy = sign(y, x)` -- flips y's sign where x was negative

This transforms `signed * signed` into `unsigned * signed`, compatible with maddubs.

```c
static inline __m256 mul_sum_i8_pairs_float(const __m256i x, const __m256i y) {
#if __AVXVNNIINT8__
    // VNNI signed*signed path (Alder Lake does NOT have this)
    const __m256i zero = _mm256_setzero_si256();
    const __m256i summed_pairs = _mm256_dpbssd_epi32(zero, x, y);
    return _mm256_cvtepi32_ps(summed_pairs);
#else
    // Standard AVX2 path
    const __m256i ax = _mm256_sign_epi8(x, x);  // abs(x)
    const __m256i sy = _mm256_sign_epi8(y, x);  // y with x's signs applied
    return mul_sum_us8_pairs_float(ax, sy);
#endif
}

static inline __m256 mul_sum_us8_pairs_float(const __m256i ax, const __m256i sy) {
#if defined(__AVXVNNI__) || (defined(__AVX512VNNI__) && defined(__AVX512VL__))
    // AVX-VNNI path (available on Alder Lake i7-12700H!)
    const __m256i zero = _mm256_setzero_si256();
    const __m256i summed_pairs = _mm256_dpbusd_epi32(zero, ax, sy);
    return _mm256_cvtepi32_ps(summed_pairs);
#else
    // Fallback: maddubs + madd
    const __m256i dot = _mm256_maddubs_epi16(ax, sy);
    return sum_i16_pairs_float(dot);
#endif
}

static inline __m256 sum_i16_pairs_float(const __m256i x) {
    const __m256i ones = _mm256_set1_epi16(1);
    const __m256i summed_pairs = _mm256_madd_epi16(ones, x);
    return _mm256_cvtepi32_ps(summed_pairs);
}
```

**The three-tier instruction chain (without VNNI):**
1. `_mm256_maddubs_epi16(ax, sy)`: 32 unsigned*signed byte multiplies -> 16 int16 sums
2. `_mm256_madd_epi16(ones, dot)`: 16 int16 pair sums -> 8 int32 sums
3. `_mm256_cvtepi32_ps(summed_pairs)`: 8 int32 -> 8 float32

**With AVX-VNNI (i7-12700H P-cores have this!):**
1. `_mm256_dpbusd_epi32(zero, ax, sy)`: 32 unsigned*signed byte multiplies -> 8 int32 sums
2. `_mm256_cvtepi32_ps(summed_pairs)`: 8 int32 -> 8 float32

**AVX-VNNI saves 1-2 instructions per block and has higher throughput on Golden Cove.**

**Relevance to EdgeLM:** This is the critical kernel for attention score computation with
quantized KV cache. During Q*K^T, the query vector Q is FP32 and K is Q8_0. We need to:
1. Load the Q8_0 block (scale + 32 int8 values)
2. Quantize the corresponding Q segment to INT8 (or compute in mixed precision)
3. Use this dot product kernel
4. Multiply by scale factors

For head_dim=128 with block_size=32: 4 dot products per attention score. With AVX-VNNI,
each dot product is ~3-4 instructions, so ~16 instructions per attention score. This is
extremely efficient.

---

### Finding B4: Horizontal Float Sum (hsum_float_8) and Int Sum (hsum_i32_8)

**Source:** `ggml/src/ggml-cpu/arch/x86/quants.c` (ggml-org/llama.cpp, GitHub)

**Float horizontal sum (8 floats -> 1 float):**
```c
static inline float hsum_float_8(const __m256 x) {
    __m128 res = _mm256_extractf128_ps(x, 1);
    res = _mm_add_ps(res, _mm256_castps256_ps128(x));
    res = _mm_add_ps(res, _mm_movehl_ps(res, res));
    res = _mm_add_ss(res, _mm_movehdup_ps(res));
    return _mm_cvtss_f32(res);
}
```

**Integer horizontal sum (8 int32s -> 1 int32):**
```c
static inline int hsum_i32_8(const __m256i a) {
    const __m128i sum128 = _mm_add_epi32(
        _mm256_castsi256_si128(a),
        _mm256_extractf128_si256(a, 1));
    const __m128i hi64 = _mm_unpackhi_epi64(sum128, sum128);
    const __m128i sum64 = _mm_add_epi32(hi64, sum128);
    const __m128i hi32  = _mm_shuffle_epi32(sum64, _MM_SHUFFLE(2, 3, 0, 1));
    return _mm_cvtsi128_si32(_mm_add_epi32(sum64, hi32));
}
```

**Relevance to EdgeLM:** These are needed at the end of every dot product (one per
attention score computation). The float version is used after accumulating Q*K dot
products; the integer version is used in Q8_1 quantization to compute the sum field.

---

### Finding B5: Complete Q8_0 Dot Product -- The Full Attention Score Kernel

**Source:** `ggml/src/ggml-cpu/arch/x86/quants.c` (ggml-org/llama.cpp, GitHub)

**Key idea:** `ggml_vec_dot_q8_0_q8_0` computes the dot product of two Q8_0-quantized
vectors. This is directly applicable when both Q and K are quantized (or when K and V
are both Q8_0 in different contexts).

```c
void ggml_vec_dot_q8_0_q8_0(int n, float * GGML_RESTRICT s, size_t bs,
                            const void * GGML_RESTRICT vx, size_t bx,
                            const void * GGML_RESTRICT vy, size_t by,
                            int nrc) {
    const int qk = QK8_0;
    const int nb = n / qk;
    const block_q8_0 * GGML_RESTRICT x = vx;
    const block_q8_0 * GGML_RESTRICT y = vy;

    float sumf = 0;
    int ib = 0;

#if defined(__AVX2__)
    __m256 acc = _mm256_setzero_ps();

    for (; ib < nb; ++ib) {
        // Load and multiply scale factors
        const __m256 d = _mm256_set1_ps(
            GGML_CPU_FP16_TO_FP32(x[ib].d) *
            GGML_CPU_FP16_TO_FP32(y[ib].d));

        // Load quantized values
        __m256i qx = _mm256_loadu_si256((const __m256i *)x[ib].qs);
        __m256i qy = _mm256_loadu_si256((const __m256i *)y[ib].qs);

        // Dot product of 32 int8 pairs -> 8 float sums
        const __m256 q = mul_sum_i8_pairs_float(qx, qy);

        // Accumulate: acc += scale * dotproduct
        acc = _mm256_fmadd_ps(d, q, acc);
    }

    sumf = hsum_float_8(acc);
#endif

    // Scalar fallback for remainder
    for (; ib < nb; ++ib) {
        int sumi = 0;
        for (int j = 0; j < qk; j++) {
            sumi += x[ib].qs[j] * y[ib].qs[j];
        }
        sumf += sumi * (GGML_CPU_FP16_TO_FP32(x[ib].d) *
                        GGML_CPU_FP16_TO_FP32(y[ib].d));
    }

    *s = sumf;
}
```

**Per-block operation count (AVX2 inner loop):**
- 2 FP16->FP32 conversions + 1 multiply + 1 broadcast: ~4 ops
- 2 loads (32 bytes each): 2 ops
- mul_sum_i8_pairs_float: ~4 ops (with VNNI) or ~6 ops (without)
- 1 FMA: 1 op
- **Total: ~9-11 ops per block of 32 elements**

**For head_dim=128 (4 blocks):** ~40-44 ops + 1 hsum = ~45 ops per dot product.
At ~2 GHz effective throughput: ~22 ns per attention score. For 2048-token context:
2048 * 22 ns = ~45 us per head, * 8 heads = ~360 us for all K dot products.

**Relevance to EdgeLM:** This kernel template is exactly what we need for Q*K^T
computation when K cache is Q8_0. However, our Q vector comes from the attention
projection in FP32, so we need a mixed-precision variant: quantize Q to Q8_0 on the
fly, then use this kernel. Alternatively, we can dequantize K to FP32 and use FP32 dot
product -- but the INT8 path is ~4x less memory traffic.

---

## Topic C: AVX-VNNI on Alder Lake (i7-12700H)

### Finding C1: VPDPBUSD Instruction -- The Key Accelerator

**Source:** Wikipedia (AVX-512#VNNI), llama.cpp source code analysis

**Key idea:** The i7-12700H's Golden Cove P-cores support AVX-VNNI (VEX-encoded, NOT
AVX-512 VNNI). This provides `_mm256_dpbusd_epi32` (VPDPBUSD) which computes:
```
For each 32-bit lane i:
  dst[i] += sum(unsigned_byte[i*4+j] * signed_byte[i*4+j]) for j=0..3
```

This does 32 byte multiplications and 8 accumulations in a single instruction, replacing
the 3-instruction chain of maddubs + madd + cvt.

**Critical detail about operand types:** VPDPBUSD requires the FIRST operand to be
**unsigned** bytes and the SECOND to be **signed** bytes. This is why the llama.cpp code
uses the `_mm256_sign_epi8` trick to convert signed*signed into unsigned*signed:

```c
const __m256i ax = _mm256_sign_epi8(x, x);  // abs(x) -> unsigned
const __m256i sy = _mm256_sign_epi8(y, x);  // y * sign(x) -> adjusted signed
_mm256_dpbusd_epi32(zero, ax, sy);           // unsigned * signed dot product
```

**VEX vs EVEX encoding:**
- AVX-512 VNNI (EVEX): `_mm256_dpbusd_epi32` -- requires AVX-512VL + AVX-512VNNI
- AVX-VNNI (VEX): `_mm256_dpbusd_avx_epi32` -- requires only AVX-VNNI flag
- Alder Lake has VEX-encoded AVX-VNNI, NOT EVEX-encoded AVX-512 VNNI
- In llama.cpp: gated by `__AVXVNNI__` preprocessor define

**Compiler flag needed:** `-mavxvnni` (GCC/Clang) or `/arch:AVX2` with VNNI detection (MSVC)

**Performance on Golden Cove:**
- VPDPBUSD throughput: 1 instruction per cycle on port 0 or port 5
- Latency: 5 cycles
- Compared to maddubs (1/cycle, 4-cycle latency) + madd (1/cycle, 5-cycle latency)
- Net speedup: ~1.5x for INT8 dot products when VNNI is available

**Relevance to EdgeLM:** We MUST compile with AVX-VNNI support for optimal INT8 KV cache
dot products. The P-cores can execute VPDPBUSD at 1/cycle throughput, making quantized
attention scores significantly faster than FP32 alternatives. Note: E-cores (Gracemont)
also support AVX-VNNI but with potentially different throughput characteristics.

---

### Finding C2: AVXVNNIINT8 -- Even Newer Signed*Signed Extension (NOT on Alder Lake)

**Source:** llama.cpp `mul_sum_i8_pairs_float` source code

**Key detail:** llama.cpp also checks for `__AVXVNNIINT8__` which enables
`_mm256_dpbssd_epi32` -- a signed*signed variant that eliminates the need for the
sign-flip trick entirely. This instruction is **NOT available on Alder Lake**; it requires
Sierra Forest / Lunar Lake or later.

```c
#if __AVXVNNIINT8__
    // NOT available on our i7-12700H
    const __m256i zero = _mm256_setzero_si256();
    const __m256i summed_pairs = _mm256_dpbssd_epi32(zero, x, y);
    return _mm256_cvtepi32_ps(summed_pairs);
```

**Relevance to EdgeLM:** Do NOT target this. Use the `__AVXVNNI__` path with the
sign-flip trick. The sign-flip adds only 2 instructions (two `_mm256_sign_epi8`) and
is well worth it for the VNNI acceleration.

---

## Topic D: Per-Channel vs Per-Token Quantization Strategy (KIVI)

### Finding D1: KIVI Paper -- Asymmetric Quantization is Critical for Quality

**Source:** "KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache"
(Yuan et al., ICML 2024)
https://arxiv.org/abs/2402.02750
https://github.com/jy-yuan/KIVI

**Key finding:** Keys and Values have fundamentally different distributions and MUST be
quantized along different dimensions:

- **Key cache: quantize per-CHANNEL** (group elements along the channel/head_dim dimension)
  - Reason: Key channels have persistent outliers (certain channels always have large magnitudes)
  - Per-token quantization of keys causes 5x larger attention score error than per-channel
  - The outlier channels come from the input activation distribution (consistent with SmoothQuant findings)

- **Value cache: quantize per-TOKEN** (group elements along the token dimension)
  - Reason: Value cache has no obvious outlier pattern across channels
  - But attention output = A * V is a weighted sum of value rows (tokens)
  - Due to attention sparsity (~84% of attention weights are near zero), per-token
    quantization confines error to individual tokens, not affecting important tokens
  - Per-channel quantization of values causes 15x larger attention output error

**Experimental validation (Llama-2-13B, 2-bit quantization):**

| Configuration | CoQA | TruthfulQA | GSM8K |
|---|---|---|---|
| 16-bit (baseline) | 66.37 | 29.53 | 22.67 |
| 2-bit K-perChannel, V-perToken | **63.53** | **28.60** | **12.21** |
| 2-bit K-perToken, V-perToken | 52.93 | 24.98 | 4.55 |
| 2-bit K-perChannel, V-perChannel | 2.88 | 0.74 | 0.00 |
| 2-bit K-perToken, V-perChannel | 2.80 | 0.26 | 0.08 |

**The wrong choice of quantization axis completely destroys model quality at low bit-widths.**

**Relevance to EdgeLM:** At 8-bit (our target), per-token quantization for both K and V
works fine (llama.cpp confirms minimal quality loss). The KIVI findings are critical if we
ever need to push to 4-bit or lower. For INT8 Q8_0 with block_size=32:
- Per-token K quantization: each token's K vector (head_dim=128) gets 4 blocks, each with
  its own scale. Fine for 8-bit.
- Per-token V quantization: same structure. Fine for 8-bit.
- For 4-bit, we should switch K to per-channel quantization.

---

### Finding D2: KIVI Residual Cache -- Keeping Recent Tokens in Full Precision

**Source:** KIVI paper, Section 3.3 and Algorithm 1

**Key idea:** KIVI maintains a "residual cache" of the most recent R tokens in full FP16
precision. Only older tokens are quantized. This is critical for maintaining quality on
hard reasoning tasks (GSM8K improves from 5.76% to 12.74% with R=128).

**Algorithm structure:**
1. During prefill: quantize all but last R tokens, keep R in FP16
2. During decoding: append new token to residual in FP16
3. When residual reaches R tokens:
   - For K: quantize per-channel across the R tokens, append to quantized cache
   - For V: quantize per-token (one token at a time), append to quantized cache
   - Reset residual to empty
4. Attention computation: tiled matmul combining quantized and FP16 parts
   - `A_g = Q * dequant(K_g^T)` (quantized part)
   - `A_r = Q * K_r^T` (residual FP16 part)
   - `A = concat(A_g, A_r)` then softmax

**Ablation results (Llama-2-13B, 2-bit, GSM8K):**
- Residual length 32: 20.62%
- Residual length 64: 19.86%
- Residual length 128: 20.77%
- Group size 32: 20.77% (best)
- Group size 64: 21.00% (similar)
- Group size 128: 17.29% (worse)

**Relevance to EdgeLM:** For our 2048-token context with INT8, the residual cache adds
complexity but may not be necessary at 8-bit precision (quality loss is already minimal).
If we pursue 4-bit KV cache, implementing a residual of 32-128 tokens in FP32 is
straightforward: maintain a small ring buffer in FP32 alongside the main Q8/Q4 cache.
The tiled attention computation (quantized part + residual part) maps naturally to
separate SIMD kernel calls.

---

### Finding D3: KIVI Quantization Formulas -- Asymmetric (min/max) Scheme

**Source:** KIVI paper Section 3.1 and quant/new_pack.py

**Quantization formula (asymmetric, min-max based):**
```
z_X = min(X)                           // zero-point
s_X = (max(X) - min(X)) / (2^B - 1)   // scale factor
Q(X) = round((X - z_X) / s_X)          // quantize to [0, 2^B-1]
X' = Q(X) * s_X + z_X                  // dequantize
```

This is asymmetric quantization (different from Q8_0's symmetric absmax scheme).
Asymmetric is better for distributions not centered at zero.

**Per-channel Key quantization implementation (from KIVI code):**
```python
def quant_and_pack_kcache(k, group_size, bits):
    # k shape: [batch, n_heads, seq_len, head_dim]
    B, nh, T, D = k.shape
    num_groups = T // group_size  # groups along TOKEN dimension
    data = k.view(B, nh, num_groups, group_size, D)
    mn = torch.min(data, dim=-2, keepdim=True)[0]  # min per group along tokens
    mx = torch.max(data, dim=-2, keepdim=True)[0]  # max per group along tokens
    scale = (mx - mn) / (2**bits - 1)
    data = (data - mn) / scale
    data = data.clamp_(0, 2**bits - 1).round_().to(torch.int32)
    code = pack_tensor(data, bits, pack_dim=2)
    return code, scale, mn
```

**Key detail about "per-channel":** In KIVI's implementation, "per-channel" means grouping
along the TOKEN dimension (dim=-2 in the reshaped tensor), so each group spans `group_size`
tokens and has independent scale/zero-point per channel. This keeps outlier channels from
affecting the scale of other channels.

**Relevance to EdgeLM:** For our INT8 implementation, the simpler symmetric Q8_0 scheme
(absmax, no zero-point) is sufficient and much cheaper to compute in SIMD (no subtraction
of zero-point needed during dequantization). Asymmetric quantization adds one extra
FP16 field (zero-point) and one extra subtraction per dequantization. Reserve asymmetric
quantization for a potential 4-bit mode.

---

## Topic E: Quantize-on-Write / Dequantize-on-Read Architecture

### Finding E1: llama.cpp's Quantize-on-Write Pattern via ggml_set_rows

**Source:** `src/llama-kv-cache.cpp` and `ggml/src/ggml.c` (ggml-org/llama.cpp)
https://github.com/ggml-org/llama.cpp/blob/master/src/llama-kv-cache.cpp

**Key idea:** In llama.cpp, the KV cache stores tensors in the specified quantized type.
When new K/V values are computed (in FP32), they are quantized during the copy/set
operation via the type's registered `from_float` callback.

**Write path (quantize-on-write):**
```cpp
// llama-kv-cache.cpp -- cpy_k and cpy_v methods
ggml_tensor * cpy_k(ggml_context * ctx, ggml_tensor * k_cur,
                    ggml_tensor * k_idxs) {
    return ggml_set_rows(ctx, k, k_cur, k_idxs);
    // k is the cache tensor with type_k (e.g., GGML_TYPE_Q8_0)
    // k_cur is the new key in FP32
    // ggml_set_rows internally calls quantize_row_q8_0 for the conversion
}
```

**Read path (dequantize-on-read):**
```cpp
// get_k and get_v return views of the quantized tensor
ggml_tensor * get_k(ggml_context * ctx, int il, int n_kv, int ns) {
    return ggml_view_4d(ctx, k,
        hparams.n_embd_head_k(il),   // head_dim
        hparams.n_head_kv(il),        // n_kv_heads
        n_kv, ns, ...);
}
// The view is used directly in attention computation
// GGML's computation graph handles dequantization transparently
```

**Type registration for Q8_0:**
```c
// ggml.c type traits
[GGML_TYPE_Q8_0] = {
    .type_name     = "q8_0",
    .blck_size     = QK8_0,           // 32
    .type_size     = sizeof(block_q8_0), // 34
    .is_quantized  = true,
    .to_float      = dequantize_row_q8_0,
    .from_float_ref = quantize_row_q8_0_ref,
}
```

**Relevance to EdgeLM:** Our architecture should follow this pattern:
1. **Write path:** After computing K = X * W_K (FP32), immediately quantize to Q8_0
   using the AVX2 kernel and write to the ring buffer cache.
2. **Read path:** During attention, either:
   a. Dequantize K block-by-block into a temporary FP32 buffer, then compute Q*K^T in FP32
   b. Quantize Q to Q8_0, then use the INT8 dot product kernel directly
   Option (b) uses less memory bandwidth but introduces double quantization error.
   Option (a) is simpler and the dequantization cost is trivial.

---

### Finding E2: Dequantize-on-Read -- The Scalar and Vectorized Paths

**Source:** `ggml/src/ggml-quants.c` (ggml-org/llama.cpp, GitHub)

**Reference (scalar) dequantization:**
```c
void dequantize_row_q8_0(const block_q8_0 * GGML_RESTRICT x,
                         float * GGML_RESTRICT y, int64_t k) {
    static const int qk = QK8_0;
    assert(k % qk == 0);
    const int nb = k / qk;

    for (int i = 0; i < nb; i++) {
        const float d = GGML_FP16_TO_FP32(x[i].d);
        for (int j = 0; j < qk; ++j) {
            y[i*qk + j] = x[i].qs[j] * d;
        }
    }
}
```

**AVX2 vectorized dequantization (EdgeLM design):**

While llama.cpp's x86 quants.c does not include an explicit AVX2 dequantize_row_q8_0
(it relies on the generic code path or fused dot products), the vectorized version is
straightforward to write:

```c
// EdgeLM: AVX2 dequantize_row_q8_0
void dequantize_row_q8_0_avx2(const block_q8_0 * GGML_RESTRICT x,
                               float * GGML_RESTRICT y, int64_t k) {
    const int nb = k / QK8_0;
    for (int i = 0; i < nb; i++) {
        const __m256 d = _mm256_set1_ps(GGML_FP16_TO_FP32(x[i].d));

        // Load 32 int8 values
        const __m256i q = _mm256_loadu_si256((const __m256i *)x[i].qs);

        // Unpack int8 -> int32 (4 groups of 8)
        const __m128i q_lo = _mm256_castsi256_si128(q);      // low 16 bytes
        const __m128i q_hi = _mm256_extracti128_si256(q, 1); // high 16 bytes

        // Sign-extend int8 -> int32, convert to float, multiply by scale
        __m256i i0 = _mm256_cvtepi8_epi32(q_lo);
        _mm256_storeu_ps(y + i*32 + 0,  _mm256_mul_ps(_mm256_cvtepi32_ps(i0), d));

        __m256i i1 = _mm256_cvtepi8_epi32(_mm_srli_si128(q_lo, 8));
        _mm256_storeu_ps(y + i*32 + 8,  _mm256_mul_ps(_mm256_cvtepi32_ps(i1), d));

        __m256i i2 = _mm256_cvtepi8_epi32(q_hi);
        _mm256_storeu_ps(y + i*32 + 16, _mm256_mul_ps(_mm256_cvtepi32_ps(i2), d));

        __m256i i3 = _mm256_cvtepi8_epi32(_mm_srli_si128(q_hi, 8));
        _mm256_storeu_ps(y + i*32 + 24, _mm256_mul_ps(_mm256_cvtepi32_ps(i3), d));
    }
}
```

**Operations per block:**
- 1 FP16->FP32 conversion + broadcast
- 1 load (32 bytes)
- 4x: sign-extend int8->int32, convert int32->float, multiply by scale, store
- Total: ~17 instructions per block of 32 elements
- For head_dim=128: ~68 instructions = ~28 ns at 2.4 GHz

**Relevance to EdgeLM:** Dequantization is cheap enough that the "dequantize K to FP32
then compute Q*K^T in FP32" approach has negligible overhead. The dequantization of a
full K vector (head_dim=128) takes ~28 ns. For 2048 tokens * 8 heads: 2048 * 8 * 28 ns
= ~460 us. This is comparable to the actual dot product cost, so the total attention
with dequantize-on-read adds ~2x overhead vs keeping K in FP32 (but saves 4x memory).

---

## Topic F: HuggingFace Transformers KV Cache Quantization

### Finding F1: Quanto and HQQ Backends -- Production KV Cache Quantization

**Source:** HuggingFace Blog: "KV Cache Quantization"
https://huggingface.co/blog/kv-cache-quantization

**Key findings:**
- Two backends supported: Quanto (int2, int4) and HQQ (int2, int4, int8)
- Uses per-token quantization for both K and V by default
- Affine (asymmetric) quantization: `X_Q = round(X / S) - Z`
- Configurable group_size (default: 64)
- Residual cache: keeps last 128 tokens in FP16 by default

**Memory savings:**
- LLaMA-2-7B, 10K context: FP16 ~5GB -> INT4 ~2GB (2.5x reduction)
- With Flash Attention + quantization: 128K tokens on 80GB A100

**Quality impact (INT4, Quanto, LongBench):**
- TREC: 63.0 (FP16) vs 63.0 (INT4) -- identical
- SAMSum: 41.12 (FP16) vs 41.3 (INT4) -- slightly better
- TriviaQA: 84.28 (FP16) vs 84.76 (INT4) -- slightly better
- INT2 shows significant degradation: SAMSum drops from 41.12 to 14.04

**Relevance to EdgeLM:** Confirms that INT4+ KV cache quantization is essentially
lossless on most benchmarks. Our INT8 choice is extremely conservative and safe. The
group_size=64 setting is interesting -- we could use 64 instead of 32, halving the
number of scale factors at the cost of slightly higher quantization error. For
head_dim=128, group_size=64 gives exactly 2 blocks per head (vs 4 with size 32).

---

## Topic G: FP8 Formats and Their Relevance

### Finding G1: E4M3 vs E5M2 FP8 Formats

**Source:** "FP8 Formats for Deep Learning" (Micikevicius et al., 2022)
https://arxiv.org/abs/2209.05433

**Two FP8 encodings:**
- **E4M3** (4-bit exponent, 3-bit mantissa): Extended dynamic range by not representing
  infinities and having only one NaN pattern. Range: [-448, 448]. Precision: ~3.5 significant bits.
- **E5M2** (5-bit exponent, 2-bit mantissa): Follows IEEE 754 conventions for special values.
  Range: [-57344, 57344]. Precision: ~2.5 significant bits.

**Key finding:** FP8 succeeds where INT8 fails -- some models that "resisted fixed point
int8 quantization" performed acceptably with FP8 post-training quantization. FP8 is more
flexible for distributions with large dynamic range.

**Relevance to EdgeLM:** FP8 is NOT natively supported by AVX2 or AVX-VNNI. There are no
hardware FP8 instructions on Alder Lake. Implementing FP8 would require software emulation
(convert FP8 -> FP16/FP32 using lookup tables or bit manipulation), which adds overhead
compared to INT8 with native SIMD support. **Recommendation: Use INT8 (Q8_0) for KV cache,
not FP8.** The only scenario where FP8 would help is if the KV cache distribution has
extreme outliers that INT8's symmetric quantization can't handle -- but at 8-bit precision,
this is rarely an issue.

---

## Topic H: Scale Factor Storage and Memory Overhead

### Finding H1: Scale Factor Overhead Analysis for Different Block Sizes

**Analysis based on block format definitions**

**Q8_0 (block_size=32, symmetric):**
- Scale: 2 bytes (FP16 `d`)
- Data: 32 bytes (32 x int8)
- Total: 34 bytes per 32 elements
- Overhead: 2/34 = 5.9%
- Effective bits per element: 8.5

**Q8_1 (block_size=32, symmetric + sum):**
- Scale + sum: 4 bytes (2x FP16)
- Data: 32 bytes
- Total: 36 bytes per 32 elements
- Overhead: 4/36 = 11.1%
- Effective bits per element: 9.0

**Q4_0 (block_size=32, symmetric):**
- Scale: 2 bytes (FP16)
- Data: 16 bytes (32 x 4-bit nibbles packed)
- Total: 18 bytes per 32 elements
- Overhead: 2/18 = 11.1%
- Effective bits per element: 4.5

**For EdgeLM KV cache (head_dim=128, Q8_0, group_size=32):**
- Per head per token: 128 elements = 4 blocks = 4 * 34 = 136 bytes
- vs FP32: 128 * 4 = 512 bytes
- Compression ratio: 512 / 136 = 3.76x
- Scale factor overhead: 4 * 2 = 8 bytes out of 136 = 5.9%

**Hypothetical group_size=64 for Q8_0:**
- Per block: 2 + 64 = 66 bytes per 64 elements
- Per head per token: 128 elements = 2 blocks = 2 * 66 = 132 bytes
- Compression ratio: 512 / 132 = 3.88x
- Scale factor overhead: 2 * 2 = 4 bytes out of 132 = 3.0%
- Slightly better compression but requires custom block format (not Q8_0)

**Full KV cache memory comparison (2048 tokens, 8 KV heads, head_dim=128):**

| Format | Per-head per-token | Total K+V cache | Compression |
|--------|-------------------|-----------------|-------------|
| FP32 | 512 bytes | 16.0 MB | 1.0x |
| FP16 | 256 bytes | 8.0 MB | 2.0x |
| Q8_0 (bs=32) | 136 bytes | 4.25 MB | 3.76x |
| Q4_0 (bs=32) | 72 bytes | 2.25 MB | 7.11x |
| INT2 (KIVI, bs=32) | ~40 bytes | 1.25 MB | 12.8x |

**Relevance to EdgeLM:** With Q8_0 KV cache, our total K+V memory is ~4.25 MB for 2048
tokens. This is well within our 6-7 GB budget and leaves ample room for model weights
(~0.6 GB for BitNet 3B). Scale factor overhead at 5.9% is negligible. No need for
larger block sizes or more aggressive formats.

---

## Topic I: Practical Implementation Recommendations for EdgeLM

### Finding I1: Recommended KV Cache Quantization Architecture

Based on all research findings, here is the recommended implementation:

**Format:** Q8_0 (symmetric, absmax, block_size=32)

**Data layout (per KV head per layer):**
```
KV cache memory layout (single head, single layer):
  K cache: block_q8_0[max_seq_len * (head_dim / 32)]
  V cache: block_q8_0[max_seq_len * (head_dim / 32)]

For max_seq_len=2048, head_dim=128:
  K: 2048 * 4 = 8192 block_q8_0 structures = 278,528 bytes (~272 KB)
  V: same = 272 KB
  Total per layer: 8 heads * (272 + 272) KB = 4,352 KB = ~4.25 MB
  For 26 layers (3B model): 26 * 4.25 = ~110 MB
```

**Write path (quantize-on-write):**
1. New K vector arrives: float[128] from attention projection
2. Call quantize_row_q8_0_avx2(k_vec, &cache[pos * 4], 128)
3. Produces 4 block_q8_0 structs (136 bytes) written to ring buffer position
4. Cost: ~33 ns on P-core

**Read path option A (dequantize-on-read, RECOMMENDED for simplicity):**
1. For each token position in context:
   - Load 4 block_q8_0 structs (136 bytes)
   - Dequantize to float[128] in scratch buffer
   - Compute dot product Q * K_i in FP32
2. Cost: ~50 ns per attention score (28 ns deq + 22 ns dot)
3. Total for 2048 tokens, 8 heads: ~820 us

**Read path option B (quantized dot product, FASTER for memory-bound scenarios):**
1. Quantize Q vector to Q8_0 once: 33 ns
2. For each token position:
   - Load 4 block_q8_0 structs (136 bytes) from K cache
   - Call ggml_vec_dot_q8_0_q8_0 directly
3. Cost: ~22 ns per attention score
4. Total for 2048 tokens, 8 heads: ~360 us + 33 ns quantize Q = ~360 us
5. **2.3x faster than option A** but with slight quantization error from Q

**Recommendation:** Start with option A (simpler, no Q quantization error), switch to
option B if attention becomes a bottleneck. The AVX-VNNI dot product kernel makes
option B especially attractive on the P-cores.

---

### Finding I2: Alignment and Cache Line Considerations

**Critical requirements:**
1. All block_q8_0 arrays must be 64-byte aligned for optimal AVX2 access
2. block_q8_0 is 34 bytes -- NOT naturally aligned to any power of 2
3. Solution: pad to 64 bytes per block (wastes 30 bytes) OR accept unaligned access
4. Better solution: keep blocks contiguous, align the START of each row

**Recommended alignment strategy:**
```c
// Allocate KV cache with 64-byte alignment at row level
// Each row = head_dim/32 blocks = 4 blocks = 136 bytes
// Pad each row to 192 bytes (next multiple of 64) -- 41% overhead
// OR: keep 136 bytes contiguous, align at head level

// Option: Use _mm256_loadu_si256 (unaligned load) -- only ~1 cycle penalty
// on Golden Cove, and avoids the 41% padding waste.
// This is what llama.cpp does: all loads use _loadu variants.
```

**Practical recommendation:** Use unaligned loads (_mm256_loadu_ps / _mm256_loadu_si256).
Golden Cove has essentially zero penalty for unaligned loads that don't cross cache line
boundaries. block_q8_0 at 34 bytes means some blocks will cross cache lines, but the
penalty is at most 1 extra cycle. The memory savings from not padding far outweigh the
tiny performance cost.

---

### Finding I3: Integration with BitNet Ternary Weights

**Key consideration:** In BitNet inference, the attention projection matrices W_K and W_V
are ternary {-1, 0, +1}. The output of X * W_K is FP32 (reduced via conditional
add/subtract). This FP32 result is what gets quantized to Q8_0 for KV cache storage.

**The pipeline for each new token:**
```
Input embedding (FP32) --[RMSNorm]--> normalized (FP32)
  --[ternary matmul W_Q]--> Q (FP32, head_dim per head)
  --[ternary matmul W_K]--> K (FP32, head_dim per head)
  --[ternary matmul W_V]--> V (FP32, head_dim per head)
  --[quantize_row_q8_0]--> K_q8 (Q8_0, stored in cache)
  --[quantize_row_q8_0]--> V_q8 (Q8_0, stored in cache)

Attention computation:
  Q (FP32) * [dequantize(K_cache)]^T --> scores (FP32)
  softmax(scores) * dequantize(V_cache) --> output (FP32)
```

**End-to-end quantization error:** The ternary matmul already introduces no quantization
error (it's exact conditional add/subtract). The only quantization error is in the
Q8_0 conversion, which at 8-bit is extremely small (< 0.4% relative error typically).

---

## Summary of Key Findings

| Aspect | Recommendation | Rationale |
|--------|---------------|-----------|
| **Format** | Q8_0 (symmetric, bs=32) | Simple, fast SIMD, minimal quality loss |
| **Quantization** | Per-token for both K and V | Sufficient at 8-bit; per-channel K only needed at 4-bit |
| **Write path** | AVX2 quantize_row_q8_0 | ~33 ns per head_dim=128 vector |
| **Read path** | Dequantize-on-read (option A) or INT8 dot product (option B) | A: simpler; B: 2.3x faster |
| **VNNI** | Use _mm256_dpbusd_epi32 on P-cores | ~1.5x speedup for INT8 dot products |
| **Alignment** | Unaligned loads, align at allocation level | Zero effective penalty on Golden Cove |
| **FP8** | Do NOT use | No hardware support on Alder Lake |
| **Compression** | 3.76x vs FP32 (Q8_0) | 4.25 MB total KV cache for 2048 tokens |
| **Fallback** | Q4_0 with per-channel K quantization | If memory pressure requires it |

---

## Sources

1. llama.cpp ggml-common.h -- block_q8_0, block_q8_1, block_q4_0 struct definitions
   https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-common.h

2. llama.cpp x86 quants.c -- AVX2 quantize_row_q8_0, vec_dot_q8_0_q8_0, helpers
   https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cpu/arch/x86/quants.c

3. llama.cpp PR #4309 -- KV cache quantization (K cache Q8_0/Q4_0/Q4_1)
   https://github.com/ggml-org/llama.cpp/pull/4309

4. llama.cpp PR #4312 -- KV cache quantization types and CLI flags
   https://github.com/ggml-org/llama.cpp/issues/4312

5. llama.cpp llama-kv-cache.cpp -- KV cache write/read paths with quantization
   https://github.com/ggml-org/llama.cpp/blob/master/src/llama-kv-cache.cpp

6. KIVI paper (Yuan et al., ICML 2024) -- Asymmetric 2-bit KV cache quantization
   https://arxiv.org/abs/2402.02750

7. KIVI source code -- quant/new_pack.py quantization implementation
   https://github.com/jy-yuan/KIVI

8. KVQuant paper (Hooper et al., NeurIPS 2024) -- Per-channel key quantization
   https://arxiv.org/abs/2401.18079

9. HuggingFace Blog -- KV cache quantization with Quanto/HQQ
   https://huggingface.co/blog/kv-cache-quantization

10. FP8 paper (Micikevicius et al., 2022) -- E4M3 vs E5M2 formats
    https://arxiv.org/abs/2209.05433

11. QServe paper -- W4A8KV4 with SmoothAttention
    https://arxiv.org/abs/2405.04532

12. ONNX Runtime -- CPU quantization operators
    https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/contrib_ops/cpu/quantization

13. llama.cpp x86 repack.cpp -- VNNI-accelerated Q8 packing with _mm256_dpbusd_epi32
    https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cpu/arch/x86/repack.cpp

## Audit Addendum (2026-04-02)

- **Age-based mixed precision deserves explicit benchmarking.** A residual
  full-precision tail for the newest tokens may outperform one static precision
  policy across the whole cache.
- **Quality drift should be tracked at the request level, not only on aggregate
  tasks.** Useful measurements include:
  - attention-score error on held-out traces,
  - greedy token divergence depth,
  - and TTFT/decode overhead from quantize/dequant stages.
- **Dequant fusion should be treated as the next optimization frontier.** The
  best implementation may never materialize large dequantized temporary buffers.
