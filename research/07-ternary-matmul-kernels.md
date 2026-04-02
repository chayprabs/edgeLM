# Section 07: Ternary MatMul Kernel Design -- Extended Research

## Overview

This section covers the design of custom ternary matrix multiplication kernels for
BitNet 1.58-bit weights {-1, 0, +1} on Intel Alder Lake (i7-12700H) using AVX2
and AVX-VNNI. The kernel is the single most performance-critical component of the
inference engine, responsible for >90% of compute time during token generation.

The fundamental insight: ternary matmul eliminates all multiplications. With weights
constrained to {-1, 0, +1}, the operation `y = W * x` becomes conditional addition
and subtraction of activation values. The challenge is implementing this efficiently
using SIMD instructions that were designed for integer multiply-accumulate.

## What the Deep Dive Already Covers

- Ternary matmul as conditional add/subtract: `y = W * x` with `W ∈ {-1, 0, +1}`
- Pack 128 ternary weights into 256 bits (2 bits each) in AVX2 register
- VPSHUFB (byte shuffle) as a 4-bit LUT to map ternary codes to masks
- VPADDB/VPSUBB for conditional add/subtract with 4x unroll
- Software prefetch next cache line while processing current one
- AVX-VNNI VPDPBUSD for inner products (pack ternary as INT8 {-1,0,+1})
- T-MAC LUT-based approach concept: precompute all possible partial sums
- Expected 2-3x improvement over BitNet's I2_S/TL2 kernels
- P-cores for matmul workers, E-cores for orchestration/speculative decode

## New Findings

---

## Finding 1: BitNet.cpp I2_S Kernel -- The MADDUBS Trick

**Source:** https://github.com/microsoft/BitNet/blob/main/src/ggml-bitnet-mad.cpp

### Technique

BitNet.cpp's I2_S (Integer 2-bit Signed) kernel encodes ternary weights {-1, 0, +1}
as 2-bit unsigned values {0, 1, 2} packed 4 per byte. The kernel then uses
`_mm256_maddubs_epi16` (VPMADDUBSW) to compute dot products between the 2-bit
weight codes and 8-bit quantized activations.

### How It Works

**Weight encoding:**
- -1 -> 0 (unsigned byte value)
- 0  -> 1 (unsigned byte value)
- +1 -> 2 (unsigned byte value)

**Packing:** 4 weights per byte using 2-bit fields. For AVX2, `QK_I2_S = 128`
elements per quantization block (128 weights = 32 bytes packed).

**Inner loop (AVX2):**
```c
// Load 32 bytes = 128 packed 2-bit weights
__m256i xq8_3 = _mm256_loadu_si256((const __m256i*)(px));

// Extract 4 groups of 32 weights via shift+mask
__m256i xq8_0 = _mm256_and_si256(_mm256_srli_epi16(xq8_3, 6), mask); // bits [7:6]
__m256i xq8_1 = _mm256_and_si256(_mm256_srli_epi16(xq8_3, 4), mask); // bits [5:4]
__m256i xq8_2 = _mm256_and_si256(_mm256_srli_epi16(xq8_3, 2), mask); // bits [3:2]
xq8_3         = _mm256_and_si256(xq8_3, mask);                        // bits [1:0]
// mask = broadcast(0x03)

// Load 4 x 32 bytes of 8-bit quantized activations
__m256i yq8_0 = _mm256_loadu_si256((const __m256i*)(py));
__m256i yq8_1 = _mm256_loadu_si256((const __m256i*)(py + 32));
__m256i yq8_2 = _mm256_loadu_si256((const __m256i*)(py + 64));
__m256i yq8_3 = _mm256_loadu_si256((const __m256i*)(py + 96));

// MADDUBS: unsigned_weight[i] * signed_activation[i] + unsigned_weight[i+1] * signed_activation[i+1]
// Result: 16-bit signed, pairs of products added horizontally
xq8_0 = _mm256_maddubs_epi16(xq8_0, yq8_0);
xq8_1 = _mm256_maddubs_epi16(xq8_1, yq8_1);
xq8_2 = _mm256_maddubs_epi16(xq8_2, yq8_2);
xq8_3 = _mm256_maddubs_epi16(xq8_3, yq8_3);

// Accumulate into 16-bit register
accu = _mm256_add_epi16(accu, _mm256_add_epi16(xq8_0, xq8_1));
accu = _mm256_add_epi16(accu, _mm256_add_epi16(xq8_2, xq8_3));
```

**Result interpretation:** Since weights are encoded as {0, 1, 2}:
- Weight = 0 (-1 in ternary): result = 0 * activation = 0 (WRONG -- requires bias correction)
- Weight = 1 (0 in ternary):  result = 1 * activation = activation
- Weight = 2 (+1 in ternary): result = 2 * activation = 2 * activation

The actual ternary dot product is recovered by: `result = maddubs_result - sum(activations)`.
This subtracts the activation once from each product, converting {0, 1, 2} * act into
{-1, 0, +1} * act.

### Performance (from BitNet.cpp benchmarks)

- x86 CPUs: 2.37x to 6.17x speedup vs llama.cpp baseline
- Energy reduction: 71.9% to 82.2%
- January 2026 update added parallel kernel with configurable tiling: additional 1.15x-2.1x

### Tiling Configuration (from gemm-config.h)

**x86 with ACT_PARALLEL (recommended for GEMV):**
- ROW_BLOCK_SIZE = 4
- COL_BLOCK_SIZE = 128
- PARALLEL_SIZE = 4

**x86 without ACT_PARALLEL:**
- ROW_BLOCK_SIZE = 128
- COL_BLOCK_SIZE = 32
- PARALLEL_SIZE = 8

### Applicability to EdgeLM

Directly applicable. The I2_S kernel is the baseline we should implement first. Key
observations:
- Uses standard AVX2 (no AVX-512 needed)
- The MADDUBS trick avoids branching entirely
- 128-element quantization blocks align well with AVX2 register width
- No explicit prefetching -- relies on hardware prefetcher
- Room for improvement: no VNNI path, no explicit prefetch, tiling not tuned for Alder Lake

---

## Finding 2: T-MAC LUT-Based Approach -- Eliminating Multiply Entirely

**Source:** https://arxiv.org/abs/2407.00088, https://github.com/microsoft/T-MAC

### Technique

T-MAC replaces multiply-accumulate with table lookup. Instead of computing
`weight * activation` for each pair, it precomputes ALL possible partial sums for
groups of weights and stores them in a lookup table (LUT). During inference, weight
indices directly index the LUT to retrieve precomputed results.

### Algorithm

**Step 1: Bit-plane decomposition**
For n-bit weights: `W = sum(2^i * W_i)` for i=0..n-1, where W_i are 1-bit matrices.

**Step 2: Group activations**
Group every g=4 activation elements together. For g=4, there are 2^4 = 16 possible
combinations of {+1, -1} weights applied to 4 activations.

**Step 3: Precompute LUT**
For each group of 4 activations [a0, a1, a2, a3], compute all 16 sums:
- Index 0000: -a0 - a1 - a2 - a3
- Index 0001: -a0 - a1 - a2 + a3
- ...
- Index 1111: +a0 + a1 + a2 + a3

Store in a 16-entry table that fits in one AVX2 lane (16 bytes for int8 entries).

**Step 4: Lookup and accumulate**
Weight nibbles (4 bits per group of 4 weights) index directly into the LUT.
Use VPSHUFB (`_mm256_shuffle_epi8`) for parallel lookup -- 32 lookups per instruction.

**Step 5: Shift-and-accumulate for multi-bit**
For n-bit weights, repeat for each bit plane and shift-accumulate:
`result = sum(2^i * lookup_result_for_bit_plane_i)`

### Key Innovation: Axis Reordering

Traditional GEMM: loop over spatial (N, M) then temporal (K) dimensions.
T-MAC: loops K first, then spatial. This keeps the LUT small (fits in registers)
instead of requiring a massive [N, K] table.

### VPSHUFB on AVX2

```
VPSHUFB ymm1, ymm2, ymm3
```
- ymm2 = LUT (16 entries duplicated in both 128-bit lanes)
- ymm3 = weight indices (4-bit nibbles as byte values 0-15)
- ymm1 = result (32 parallel lookups)

**Critical AVX2 limitation:** VPSHUFB operates independently on each 128-bit lane.
The LUT must be duplicated in both halves of the 256-bit register.

### Table Size Optimization (Sign Bit Trick)

Instead of 2^g = 16 entries, exploit symmetry: store only 2^(g-1) = 8 entries
(positive sums), then negate on-the-fly using a sign bit. This halves table size
and speeds up precomputation.

### Fast 8-Bit Aggregation

Instead of converting to 16-bit after each lookup (halving throughput), accumulate
in 8-bit using averaging instructions (`_mm256_avg_epu8`). Defer precision conversion.
Introduces small probabilistic bias but negligible impact on model quality.

### Performance Numbers

**Kernel-level speedup vs llama.cpp (single thread):**
| Bit Width | Speedup |
|-----------|---------|
| 1-bit     | 11.2x   |
| 2-bit     | 5.8x    |
| 3-bit     | 4.7x    |
| 4-bit     | 3.1x    |

**End-to-end BitNet-b1.58-3B:**
- M2-Ultra single core: 30 tok/s
- M2-Ultra 8 cores: 71 tok/s
- Raspberry Pi 5: 11 tok/s
- Surface Laptop 7: 4-5x vs llama.cpp

**Key insight:** LUT approach scales linearly with bit width. At 1.58 bits
(ternary = 2-bit representation), the speedup is ~5.8x over dequantize-then-multiply.

### Applicability to EdgeLM

Highly applicable as an alternative to the MADDUBS approach. For ternary specifically:
- Ternary weights decompose into 2 bit planes (since we need 2 bits to encode {-1, 0, +1})
- Each LUT has 16 entries for g=4, fitting perfectly in one AVX2 128-bit lane
- VPSHUFB on Alder Lake P-cores: 1 cycle latency, 0.5 CPI throughput (2 per cycle)
- This may outperform MADDUBS approach because it truly eliminates all multiplies

**Trade-off vs MADDUBS:** LUT requires precomputation of tables from activations
(per-token overhead), but the lookup+accumulate is potentially faster than
multiply+accumulate for very low bit widths.

---

## Finding 3: Instruction Throughput on Alder Lake -- Critical Performance Data

**Source:** https://uops.info

### Key Instructions for Ternary Kernels

**Alder Lake P-cores (Golden Cove):**

| Instruction | Operation | Latency | Throughput (CPI) | uops | Port |
|-------------|-----------|---------|-------------------|------|------|
| VPMADDUBSW ymm | u8*s8 + horizontal pair | 5 cycles | 0.50 | 1 | p04 |
| VPSHUFB ymm | byte shuffle / LUT | 1 cycle | 0.50 | 1 | p15 |
| VPADDD ymm | 32-bit integer add | 1 cycle | 0.33 | 1 | p015 |
| VPADDW ymm | 16-bit integer add | 1 cycle | 0.33 | 1 | p015 |
| VPSRLW ymm | 16-bit shift right | 1 cycle | 0.50 | 1 | p01 |
| VPAND ymm | bitwise AND | 1 cycle | 0.33 | 1 | p015 |
| VMOVDQU ymm | 256-bit load | ~5 cycles | 0.50 | 1 | p23 |

**Alder Lake E-cores (Gracemont):**

| Instruction | Operation | Latency | Throughput (CPI) | uops |
|-------------|-----------|---------|-------------------|------|
| VPMADDUBSW ymm | u8*s8 + horizontal pair | 3 cycles | 2.00 | 2 |
| VPSHUFB ymm | byte shuffle / LUT | 2 cycles | 2.00 | 2 |

### Analysis for Kernel Design

**P-cores are dramatically better for SIMD:**
- VPMADDUBSW: 2 per cycle on P-cores vs 0.5 per cycle on E-cores (4x difference)
- VPSHUFB: 2 per cycle on P-cores vs 0.5 per cycle on E-cores (4x difference)

**MADDUBS vs VPSHUFB on P-cores:**
- Both have identical throughput (0.50 CPI = 2 per cycle)
- VPSHUFB has much lower latency (1 vs 5 cycles)
- They use different ports (p04 vs p15), so they can execute in parallel!

**Critical insight:** A hybrid kernel could interleave MADDUBS and VPSHUFB operations
to use both port groups simultaneously, potentially doubling throughput.

### VPDPBUSD (AVX-VNNI) Analysis

VPDPBUSD performs: `dst[31:0] += (u8[0]*s8[0] + u8[1]*s8[1] + u8[2]*s8[2] + u8[3]*s8[3])`

This is a fused version of the AVX2 chain: VPMADDUBSW + VPMADDWD + VPADDD.
Available on Alder Lake as AVX-VNNI (VEX-encoded, 256-bit, no AVX-512 required).

**Advantages over MADDUBS chain:**
- Single instruction instead of 3
- Accumulates directly into 32-bit (no intermediate 16-bit saturation risk)
- For ternary weights {0, 1, 2}, each value fits in unsigned byte -- perfect fit

**For ternary matmul:** Encode weights as u8 {0, 1, 2}, activations as s8.
VPDPBUSD computes 32 dot products of 4 elements each in one instruction.
Result: 8x int32 accumulator values, each summing 4 weight*activation products.

---

## Finding 4: The Two Kernel Paradigms -- MADDUBS vs LUT

### Paradigm 1: MADDUBS (Multiply-Accumulate)

**Approach:** Encode ternary as 2-bit unsigned, use integer multiply-add instructions.

**Computation chain (AVX2):**
```
VPMADDUBSW: u8_weight * s8_activation -> s16 (pairs added)
VPMADDWD:   s16 * {1,1,...} -> s32 (pairs added, effectively widening sum)
VPADDD:     accumulate s32
```

**With AVX-VNNI (single instruction):**
```
VPDPBUSD: u8_weight * s8_activation -> s32 (4 products summed + accumulated)
```

**Throughput analysis (P-core, per cycle):**
- AVX2 chain: limited by MADDUBS at 0.5 CPI = 2 instr/cycle
  - Each processes 32 byte pairs = 64 elements (with horizontal pairing)
  - Effective: 128 ternary ops per cycle
- AVX-VNNI: VPDPBUSD at ~0.5 CPI (estimated similar to MADDUBS)
  - Each processes 32 bytes = 128 elements (4-way horizontal sum)
  - Effective: 256 ternary ops per cycle (2x over AVX2 chain)

### Paradigm 2: LUT (Table Lookup)

**Approach:** Precompute all possible partial sums, use VPSHUFB for parallel lookup.

**Computation:**
```
[Precompute] Build 16-entry LUT from 4 activations: ~16 adds per table
[Lookup]     VPSHUFB: 32 parallel lookups per instruction at 0.5 CPI
[Accumulate] VPADDB/VPADDW: accumulate results
```

**Throughput analysis (P-core):**
- VPSHUFB: 0.5 CPI = 2 instr/cycle
  - Each processes 32 lookups, each lookup covers g=4 weights
  - Effective: 256 ternary ops per cycle
- But must also count precomputation overhead

### Comparison

| Metric | MADDUBS (AVX2) | VPDPBUSD (VNNI) | LUT (VPSHUFB) |
|--------|----------------|------------------|----------------|
| Ternary ops/cycle | ~128 | ~256 | ~256 (minus precomp) |
| Latency | 5 cycles | ~5 cycles | 1 cycle |
| Port pressure | p04 | p04 (likely) | p15 |
| Precomputation | None | None | Required per token |
| Code complexity | Low | Low | Medium |
| Memory overhead | None | None | LUT tables |

### Recommendation for EdgeLM

1. **Start with MADDUBS (I2_S style):** simplest, proven, good baseline
2. **Add VNNI path:** same approach but use VPDPBUSD for ~2x throughput
3. **Implement LUT as alternative:** may win for pure GEMV (batch=1 decoding)
4. **Explore hybrid:** MADDUBS on p04 + VPSHUFB on p15 simultaneously

---

## Finding 5: Micro-Kernel Register Tiling for Ternary GEMV

**Source:** Algorithmica GEMM tutorial, gemma.cpp matmul kernels

### The GEMV Problem

During autoregressive decoding, each token generation requires a matrix-vector
multiply: `y[M] = W[M,K] * x[K]` where M = output dimension (e.g., 4096) and
K = input dimension (e.g., 4096). This is GEMV, not GEMM.

GEMV is memory-bandwidth bound (each weight read only once per token), unlike GEMM
which can amortize weight loads across batch elements.

### Register Tiling Strategy

For GEMV, tile along M (output rows) to maximize register utilization:

**Tile size: M_tile x K_tile**
- M_tile = number of output rows computed simultaneously
- K_tile = number of input elements processed per inner loop iteration

**AVX2 register budget:** 16 x ymm registers
- Accumulator registers: M_tile registers (one per output row)
- Activation register: 1 (reused across all rows)
- Weight registers: M_tile (one per row, or reuse from memory)
- Temp registers: 2-3 for bit extraction

**Optimal M_tile for ternary GEMV:**
With 16 registers:
- 4 for accumulators (4 output rows)
- 4 for extracted weight lanes
- 4 for activation loads
- 4 for temporaries
=> M_tile = 4, processing 4 output rows per inner loop pass

BitNet.cpp confirms this: PARALLEL_SIZE = 4 for x86 with ACT_PARALLEL.

### gemma.cpp's 4x4 Approach

gemma.cpp uses a 4x4 micro-kernel for BF16:
- 16 accumulator registers (4 rows x 4 columns)
- Exploits FMA dual-issue to hide latency
- Uses "vector-length agnostic transpose" via StoreInterleaved4 for horizontal reduction

For ternary GEMV, we can adapt this to a 4x1 micro-kernel (4 rows, 1 activation vector):
- 4 accumulator ymm registers
- 1 activation ymm register (broadcast along rows)
- 4 weight ymm registers (one per row)
- 7 remaining for bit extraction and temporaries

### Memory Access Pattern

**Weight streaming:** Load 4 rows of packed weights (4 x 32 bytes = 128 bytes per iteration).
Each iteration processes 128 weights per row (32 bytes * 4 weights/byte).

**Activation reuse:** Same activation vector multiplied against all 4 rows.
Load once, reuse 4x. This is the key advantage of tiling along M.

**Bandwidth calculation:**
- Weights: 4 rows * 32 bytes = 128 bytes loaded
- Activations: 128 bytes loaded (4 x 32-byte chunks for 128 int8 values)
- Total: 256 bytes per iteration, computing 4 x 128 = 512 ternary ops
- Arithmetic intensity: 512 ops / 256 bytes = 2 ops/byte
- At 40 GB/s DDR4: theoretical max = 80 Gops/s ternary

---

## Finding 6: AVX-VNNI VPDPBUSD for Ternary -- The Optimal Instruction

**Source:** oneDNN documentation, Intel intrinsics guide, Wikipedia AVX-512 VNNI

### Instruction Semantics

```
VPDPBUSD ymm0, ymm1, ymm2

For each dword lane [0..7]:
  ymm0.dword[i] += ymm1.ubyte[4i+0] * ymm2.sbyte[4i+0]
                  + ymm1.ubyte[4i+1] * ymm2.sbyte[4i+1]
                  + ymm1.ubyte[4i+2] * ymm2.sbyte[4i+2]
                  + ymm1.ubyte[4i+3] * ymm2.sbyte[4i+3]
```

This is exactly: multiply 4 unsigned bytes by 4 signed bytes, sum products,
accumulate into 32-bit integer. All in one instruction.

### Why This Is Perfect for Ternary

- Weights {-1, 0, +1} encoded as unsigned bytes {0, 1, 2}
- Activations quantized to signed int8 [-127, +127]
- VPDPBUSD computes the exact dot product (with bias offset)
- No intermediate 16-bit saturation risk (unlike VPMADDUBSW)
- Accumulates directly into 32-bit -- no VPMADDWD step needed
- Available on Alder Lake as VEX-encoded AVX-VNNI (256-bit, no AVX-512)

### AVX2 Fallback Chain

For comparison, without VNNI the equivalent requires 3 instructions:
```c
// Step 1: u8 * s8 -> s16 (with horizontal pairing)
__m256i tmp16 = _mm256_maddubs_epi16(weights_u8, activations_s8);

// Step 2: s16 -> s32 (horizontal pairing with ones vector)
__m256i ones = _mm256_set1_epi16(1);
__m256i tmp32 = _mm256_madd_epi16(tmp16, ones);

// Step 3: accumulate
acc = _mm256_add_epi32(acc, tmp32);
```

### Saturation Risk with MADDUBS

VPMADDUBSW saturates to int16 range [-32768, +32767].
For ternary weights {0, 1, 2} and int8 activations [-127, +127]:
- Max single product: 2 * 127 = 254
- Max pair sum: 254 + 254 = 508
- Well within int16 range -- NO saturation risk for ternary!

This means the MADDUBS approach is safe for ternary, unlike general int8 where
saturation can occur (255 * 127 + 255 * 127 = 64,770 > 32,767).

---

## Finding 7: BitNet b1.58 Paper -- The Computation Paradigm

**Source:** https://arxiv.org/abs/2402.17764

### Weight Quantization

Weights quantized using absmean function:
```
W_tilde = RoundClip(W / (gamma + epsilon), -1, 1)
gamma = mean(|W_ij|)    // average absolute value
```

Activations quantized to 8-bit per-token: scaled to [-Qb, Qb] range.

### The Fundamental Computation Shift

Matrix multiplication with ternary weights becomes:
```
y[i] = sum_j(W[i,j] * x[j])

Since W[i,j] in {-1, 0, +1}:
y[i] = sum_{j where W=+1}(x[j]) - sum_{j where W=-1}(x[j])
```

This is pure addition and subtraction. No multiplications whatsoever.

### Architecture Details (LLaMA-alike)

- RMSNorm (not LayerNorm)
- SwiGLU activation (not ReLU/GELU)
- Rotary positional embeddings (RoPE)
- No biases in linear layers
- Group Query Attention (GQA)

### Performance Benchmarks (from paper)

| Model | Size | Memory | Latency | PPL |
|-------|------|--------|---------|-----|
| LLaMA FP16 | 3B | 7.89 GB | 5.07ms | 10.04 |
| BitNet b1.58 | 3B | 2.22 GB (3.55x less) | 1.87ms (2.71x faster) | 9.91 |

At 3B parameters, BitNet b1.58 MATCHES full-precision LLaMA in both perplexity
and task accuracy while using 3.55x less memory and running 2.71x faster.

### Energy Savings

BitNet b1.58 saves 71.4x arithmetic energy for matrix multiplication at 7nm.
The dominant cost is INT8 addition (vs FP16 multiply + FP16 add for LLaMA).

---

## Finding 8: BitNet a4.8 -- Hybrid Quantization for Attention

**Source:** https://arxiv.org/abs/2411.04965

### Technique

BitNet a4.8 uses a hybrid approach:
- **Ternary weights** for all linear layers (same as b1.58)
- **4-bit activations** (INT4/FP4) for attention and FFN layers
- **8-bit activations** for sparsified intermediate states
- **3-bit KV cache**

### Key Innovation

Outlier channels in activations cause quantization errors. BitNet a4.8 addresses
this by sparsifying (zeroing out) outlier channels and quantizing them separately
at higher precision.

### Performance

- Comparable quality to BitNet b1.58
- Faster inference via 4-bit kernel utilization
- Only 55% of parameters activated (sparse)
- 3-bit KV cache reduces memory further

### Applicability to EdgeLM

The 4-bit activation idea is relevant: if we can use INT4 activations in some layers,
the VNNI instruction can process even more elements per cycle. However, this requires
model retraining -- not applicable to existing BitNet b1.58 models which use INT8
activations.

---

## Finding 9: Weight Data Layout for Maximum Throughput

**Source:** T-MAC paper, BitNet.cpp gemm-config.h

### The Problem

Naive weight storage (row-major or column-major) causes cache misses when tiling.
The kernel needs weights laid out to match the exact access pattern of the tiled
computation.

### T-MAC's Weight Permutation

Weights are reordered offline (once at model load) to match the tiled access order:
1. Divide weight matrix into tiles of size [M_tm, K_tk]
2. Flatten each tile into a contiguous byte sequence
3. Concatenate tiles in the order they will be accessed during computation
4. Apply byte-level interleaving to match little-endian architecture

This converts random-access tile loads into sequential memory reads.

### BitNet.cpp's Layout

- Packed 2-bit weights: 4 weights per byte, 32 bytes per group
- Groups of 32 iterations process 1024 bytes of weights + 4096 bytes of activations
- Weight pointer advances by 32 bytes per inner loop step (one cache line)
- Activation pointer advances by 128 bytes per step (4 cache lines)

### Optimal Layout for EdgeLM

**Recommendation:** Pack weights in tile-sequential order during model loading:

```
For M_tile = 4, K_tile = 128:
  Memory layout: [row0_k0..127][row1_k0..127][row2_k0..127][row3_k0..127]
                 [row0_k128..255][row1_k128..255]...
```

Each 4-row tile segment = 4 * 32 bytes = 128 bytes = 2 cache lines.
Sequential access maximizes hardware prefetcher effectiveness.

---

## Finding 10: Kernel Variants -- GEMV vs GEMM vs Batched

### GEMV (Batch = 1, Token Generation)

This is the primary target. During autoregressive decoding, each forward pass
computes `y = W * x` where x is a single vector.

**Characteristics:**
- Memory-bandwidth bound (weight matrix read once, no reuse)
- Tile along M only (multiple output rows per iteration)
- Peak throughput limited by DDR4 bandwidth, not compute

**Throughput ceiling:**
```
Model size (ternary 3B) = ~0.6 GB
DDR4-3200 bandwidth = ~40 GB/s
Max tok/s = 40 / 0.6 = ~67 tok/s (without any optimization)
With prefetching (+20-40%): ~80-94 tok/s
With L3 cache hits for small layers: potentially higher for some layers
```

### GEMM (Batch > 1, Prefill/Prompt Processing)

During prompt processing, multiple tokens are processed simultaneously.
`Y[M, B] = W[M, K] * X[K, B]` where B = batch/sequence length.

**Characteristics:**
- Compute-bound for large B (weight matrix reused across batch)
- Tile along both M and N (batch dimension)
- Can approach peak FLOPS

**Micro-kernel for batched ternary:**
Tile: M_tile x N_tile x K_tile
- M_tile = 4 (output rows)
- N_tile = 4 (batch elements)
- K_tile = 128 (reduction dimension)
- Requires 4*4 = 16 accumulator registers (uses all 16 ymm regs)

### BitNet.cpp Kernel Variants

1. `ggml_vec_dot_i2_i8_s_1x1`: Single row, single column (scalar-like)
2. `ggml_vec_dot_i2_i8_s_1x4_32W`: 4 rows, 32-wide (hand-tuned)
3. `ggml_vec_dot_i2_i8_s_1xN`: PARALLEL_SIZE rows, 1 activation vector
4. `ggml_vec_dot_i2_i8_s_Nx1`: 1 weight row, PARALLEL_SIZE activation vectors

The dispatch selects based on `nrc % PARALLEL_SIZE`.

---

## Finding 11: Optimal Inner Loop Design for EdgeLM

### Proposed AVX-VNNI Ternary GEMV Kernel

Based on all findings, here is the optimal kernel design:

```c
// Process 4 output rows simultaneously
// K_tile = 128 elements per iteration
// Weights: 2-bit packed, 32 bytes per row per K_tile
// Activations: int8, 128 bytes per K_tile

void ternary_gemv_4rows_vnni(
    const uint8_t* weights,  // [4 rows * 32 bytes] packed 2-bit
    const int8_t* activations,  // [128] int8 values
    int32_t* accumulators,   // [4] output accumulators
    int K_blocks             // number of 128-element blocks
) {
    __m256i acc0 = _mm256_setzero_si256();
    __m256i acc1 = _mm256_setzero_si256();
    __m256i acc2 = _mm256_setzero_si256();
    __m256i acc3 = _mm256_setzero_si256();

    for (int kb = 0; kb < K_blocks; kb++) {
        // Load 128 activations (4 x 32 bytes)
        __m256i a0 = _mm256_loadu_si256((__m256i*)(activations + kb*128));
        __m256i a1 = _mm256_loadu_si256((__m256i*)(activations + kb*128 + 32));
        __m256i a2 = _mm256_loadu_si256((__m256i*)(activations + kb*128 + 64));
        __m256i a3 = _mm256_loadu_si256((__m256i*)(activations + kb*128 + 96));

        // For each of 4 rows:
        for (int r = 0; r < 4; r++) {
            // Load 32 bytes of packed weights (128 2-bit values)
            __m256i w_packed = _mm256_loadu_si256(
                (__m256i*)(weights + (kb*4 + r) * 32));

            // Extract 4 groups of 32 weight bytes
            __m256i w0 = _mm256_and_si256(
                _mm256_srli_epi16(w_packed, 6), mask03);
            __m256i w1 = _mm256_and_si256(
                _mm256_srli_epi16(w_packed, 4), mask03);
            __m256i w2 = _mm256_and_si256(
                _mm256_srli_epi16(w_packed, 2), mask03);
            __m256i w3 = _mm256_and_si256(w_packed, mask03);

            // VPDPBUSD: acc += u8_weight * s8_activation (4-way horizontal)
            acc[r] = _mm256_dpbusd_epi32(acc[r], w0, a0);
            acc[r] = _mm256_dpbusd_epi32(acc[r], w1, a1);
            acc[r] = _mm256_dpbusd_epi32(acc[r], w2, a2);
            acc[r] = _mm256_dpbusd_epi32(acc[r], w3, a3);
        }
    }

    // Horizontal reduction of each acc[r] to single int32
    // Apply bias correction: subtract sum(activations) per row
    // Apply scale factor: multiply by weight_scale * activation_scale
}
```

### Register Pressure Analysis

Per inner loop iteration:
- 4 activation registers (a0-a3)
- 4 accumulator registers (acc0-acc3)
- 1 packed weight register
- 4 extracted weight registers (w0-w3)
- 1 mask register
- Total: 14 of 16 ymm registers -- tight but feasible

### Expected Performance

**On Alder Lake P-core:**
- Inner loop: 4 rows * 4 VPDPBUSD = 16 VNNI instructions per K_tile
- At 0.5 CPI: 16 * 0.5 = 8 cycles for 4*128 = 512 ternary ops
- Plus 4 loads (weights) + 4 loads (activations) + 4 shifts + 4 masks
- Total estimate: ~20-25 cycles per 512 ops
- At 4.7 GHz: ~96-120 Gops/s per P-core

**But bandwidth-limited for GEMV:**
- 128 + 128 = 256 bytes per iteration
- At 40 GB/s: 156M iterations/s * 512 ops = 80 Gops/s
- Bottleneck is DDR4 bandwidth, not compute

---

## Finding 12: The Bias Correction for {0, 1, 2} Encoding

### The Problem

Encoding ternary {-1, 0, +1} as unsigned {0, 1, 2} introduces a +1 bias:
```
encoded_weight = true_weight + 1
maddubs_result = sum(encoded_weight[i] * activation[i])
               = sum((true_weight[i] + 1) * activation[i])
               = sum(true_weight[i] * activation[i]) + sum(activation[i])
```

### The Correction

```
true_dot_product = maddubs_result - sum(activations)
```

The `sum(activations)` term can be precomputed once per K_tile block and subtracted
from all output rows that use that activation block. This is O(K) work shared across
all M output rows -- negligible overhead.

### Implementation

```c
// Precompute activation sum for this K_tile block
int32_t act_sum = 0;
for (int k = 0; k < K_tile; k++) {
    act_sum += activations[kb * K_tile + k];
}

// After accumulation loop:
for (int r = 0; r < 4; r++) {
    result[r] = horizontal_sum(acc[r]) - act_sum;
}
```

### Alternative: Encode as {-1, 0, +1} Signed

If we encode weights as signed bytes {-1, 0, +1} instead of unsigned {0, 1, 2},
we can use `_mm256_maddubs_epi16` with activations as the unsigned operand.
But activations can be negative, so they cannot be the unsigned operand.

VPDPBUSD also requires unsigned * signed. So the {0, 1, 2} encoding with bias
correction is the correct approach.

---

## Finding 13: Multi-Threading Strategy for GEMV

### Row Partitioning

For GEMV, the M dimension is embarrassingly parallel. Each thread computes a
disjoint subset of output rows:

```
Thread 0: rows [0, M/num_threads)
Thread 1: rows [M/num_threads, 2*M/num_threads)
...
```

No synchronization needed during computation (no shared accumulator).

### P-core vs E-core Assignment

From instruction throughput data:
- P-core VPMADDUBSW: 2/cycle (0.5 CPI)
- E-core VPMADDUBSW: 0.5/cycle (2.0 CPI)
- P-core is 4x faster for SIMD

**Strategy:** Give P-cores 4x more rows than E-cores.
With 6 P-cores + 8 E-cores (14 threads total):
- P-core equivalent work units: 6 * 4 = 24
- E-core equivalent work units: 8 * 1 = 8
- Total: 32 units
- P-core share: 24/32 = 75% of rows
- E-core share: 8/32 = 25% of rows

### Bandwidth Sharing

All threads share the DDR4 bandwidth (40 GB/s). With 14 threads:
- Per-thread bandwidth: ~2.9 GB/s
- But weight data is read from L3 cache (shared 24 MB) for small models
- 0.6 GB model fits 40% in L3 -- significant cache hit rate for repeated layers

---

## Summary: Recommended Kernel Implementation Plan

### Phase 1: Baseline MADDUBS Kernel
- Implement I2_S-style kernel with VPMADDUBSW
- 4-row tiling (PARALLEL_SIZE = 4)
- 128-element K blocks
- Bias correction via precomputed activation sum
- Target: match BitNet.cpp performance (~20-30 tok/s single thread)

### Phase 2: VNNI Upgrade
- Replace MADDUBS chain with VPDPBUSD
- Same tiling, same data layout
- Expected ~1.5-2x improvement from instruction fusion
- Target: ~40-50 tok/s single thread

### Phase 3: LUT Alternative
- Implement T-MAC-style VPSHUFB kernel for comparison
- g=4 grouping, 16-entry LUT per group
- Fast 8-bit aggregation
- May win for pure GEMV due to lower latency

### Phase 4: Hybrid Kernel
- MADDUBS on port p04 + VPSHUFB on port p15 simultaneously
- Requires careful interleaving of instructions
- Potential 2x over single-paradigm kernel

### Phase 5: Multi-threaded + Prefetching
- Row-partitioned across 6P + 8E cores
- Software prefetch (PREFETCHT0/T1) for next tile's weights
- Large page backing for weight tensors (2MB pages)

---

---

## Finding 14: T-SAR In-Register LUT -- Decomposition Insight for AVX2

**Source:** https://arxiv.org/html/2511.13676v1 (Nov 2025)

### The Key Insight We Can Use

T-SAR proposes custom ISA extensions (TLUT, TGEMV) that we cannot use on shipping x86 CPUs.
However, the **ternary decomposition into two binary dot products** is implementable in
standard AVX2:

**Decomposition:**
- Original ternary weights W in {-1, 0, +1}
- Dense weights W_D in {-1, +1}: non-zero weights mapped to +/-1
- Sparse mask W_S in {0, 1}: indicator for zero positions
- Dot product: `W . x = W_D . x - W_S . x` (difference of two binary dot products)

Each binary dot product can use XNOR + popcount (for sign-based binary) or the
VPSHUFB-based popcount from TABv2 (see Finding 15).

### T-SAR Performance Numbers (Custom ISA, Not Directly Achievable)

- GEMM (prefill, batch=128): 5.6-24.5x latency reduction vs baselines
- GEMV (decode, batch=1): 1.1-86.2x throughput improvement
- Llama-8B workstation: **128.96 tok/s** (0.616 J/token)
- Llama-8B laptop: **61.00 tok/s** (0.405 J/token)
- Memory request volume reduced **8.7-13.8x** vs T-MAC

### Critical Finding: T-MAC LUT Memory Traffic Problem

T-SAR quantifies a major problem with T-MAC's approach:
- **TLUTs account for 87.6% of all memory transactions** despite occupying <0.01% of RAM
- The random access pattern of LUT lookups causes severe cache pollution
- T-SAR eliminates this by generating LUTs within SIMD registers at runtime
- This directly impacts our decision: T-MAC's LUT approach may be **worse than I2_S**
  on our bandwidth-limited DDR4-3200 system due to cache pollution

### Applicability to EdgeLM

The decomposition approach (ternary = dense_binary - sparse_binary) is a viable
**alternative kernel** if I2_S + VNNI doesn't reach our targets. Implementation:
1. Pre-decompose weights during model loading (one-time cost)
2. Store W_D and W_S as packed bit vectors (1 bit/weight each = 2 bpw total)
3. For each dot product: XNOR(W_D_bits, activation_sign_bits) + VPSHUFB popcount
4. Subtract sparse correction: popcount(W_S AND activation_magnitude)
5. Net result = true ternary dot product

Trade-off: requires 2 passes over activations but eliminates all integer multiplication.

---

## Finding 15: TABv2/FasterTNN -- VPSHUFB Popcount for AVX2

**Source:** https://github.com/fpgasystems/FasterTNN (ETH Zurich)
**Source:** IEEE TVLSI 2026, ISLPED 2025

### VPSHUFB-Based SIMD Popcount

The key technique replaces scalar `POPCNT` (which operates on 64-bit registers) with
a VPSHUFB-based SIMD popcount that processes 32 bytes per instruction:

```c
// Nibble popcount LUT
__m256i popcount_lut = _mm256_setr_epi8(
    0,1,1,2,1,2,2,3,1,2,2,3,2,3,3,4,  // lane 0
    0,1,1,2,1,2,2,3,1,2,2,3,2,3,3,4   // lane 1
);
__m256i mask_0f = _mm256_set1_epi8(0x0F);

// Count bits in 32 bytes (256 bits) using VPSHUFB
__m256i lo = _mm256_and_si256(data, mask_0f);
__m256i hi = _mm256_and_si256(_mm256_srli_epi16(data, 4), mask_0f);
__m256i popcnt = _mm256_add_epi8(
    _mm256_shuffle_epi8(popcount_lut, lo),
    _mm256_shuffle_epi8(popcount_lut, hi)
);
// popcnt[i] = popcount of byte data[i]
```

This is 5 instructions for 32 bytes vs scalar POPCNT which needs 4 instructions for
32 bytes (load + popcnt for each 64-bit word). The SIMD version is faster due to
better instruction-level parallelism and port utilization.

### TABv2 Results

- 15% reduction in total instruction count
- 2.2x theoretical speedup over naive popcount approach
- Up to **2.1x faster** than original TAB library on AVX2
- Layer-level speedup up to **2.7x on AVX2**, 2.3x on ARM

### Ternary Encoding in TABv2

TABv2 encodes ternary weights using separate sign and magnitude bit planes:
- **Magnitude plane**: 1 if weight != 0, 0 if weight == 0 (sparsity mask)
- **Sign plane**: 1 if weight == +1, 0 if weight == -1 (only meaningful where magnitude=1)
- Fused quantization + bit-packing + im2col in one loop for data locality

### Applicability to EdgeLM

Directly applicable if we implement the T-SAR decomposition approach. The VPSHUFB
popcount technique is also useful for general bit-counting operations in our pipeline
(e.g., counting sparsity, activation statistics).

---

## Finding 16: Spectra 1.1 -- Novel 1.6-bit Packing and TriRun Kernel

**Source:** https://arxiv.org/html/2506.23025 (ICLR 2025, ACL 2025)

### TQ2 (2-bit) Packing -- Confirms Our Approach

Identical to our planned format:
- Map {-1, 0, +1} -> {0, 1, 2} via d' = d + 1
- Pack using bitwise shifts: `b = sum(d'[j] * 2^(2j))` for k values per block
- 256-element blocks with FP16 scale factor (66 bytes per 256 elements = 0.258 bytes/element)
- Unpack: `d'[j] = (b >> 2j) & 0x03`, then `d[j] = d'[j] - 1`

### TQ1 (1.6-bit) Packing -- Novel Technique

Exploits mathematical near-equivalence: 3^5 = 243 ≈ 256 = 2^8

- Encodes k=5 trits into p=8 bits using base-3 representation
- **Encoding:** `b = floor((sum(d'[j] * 3^(4-j)) * 256 + 242) / 243)`
- **Decoding (vectorizable):** "multiplication-based scheme that iteratively extracts
  trits without explicit division or modulo operations"
  - Each trit extracted via: multiply by constant, shift right, mask
  - Avoids expensive integer division on x86
- **Lossless guarantee:** Theorem 1 proves lossless iff 2^p > 3^k (256 > 243, satisfied)
- **Information-theoretic efficiency:** log2(3) = 1.585 bits/trit; TQ1 at 1.6 bpw is
  only 1% over the theoretical minimum

### Bandwidth Savings: TQ1 vs TQ2

| Format | bpw | 2.4B model size | 3.3B model size | DDR4 tok/s ceiling |
|--------|-----|-----------------|-----------------|-------------------|
| TQ2 | 2.0 | 0.60 GB | 0.83 GB | 67 / 48 |
| TQ1 | 1.6 | 0.48 GB | 0.66 GB | 83 / 61 |

TQ1 gives **24% more theoretical tok/s** from bandwidth savings alone.

### TriRun Kernel Details (GPU, Reference Only)

- FP16 x INT2 mixed-precision matmul on tensor cores
- 16 ternary weights packed per 32-bit integer
- Dequantization uses "carefully selected bit masks and lookup-based 3-input logical operation"
- Double-buffered async memory copies with `cp.async` and cache hints to reduce L2 pollution
- Up to 7-8x speedup over FP16 at batch 16-32 on L40S
- Critical batch for memory-bound -> compute-bound transition: ~13 tokens on L40

### CPU-Specific Results

- TQ2 on Apple M4: outperforms 4-bit GGML in both speed and quality
- TQ1: smaller footprint but slower on CPU due to decode overhead
- AMD EPYC 7502 and Apple M4 Max: detailed benchmarks in paper Tables 5-6
- **Key finding:** "TQ1 is slower than TQ2 due to additional fixed-point multiplication
  operations" -- the decode cost outweighs bandwidth savings on high-bandwidth systems

### Decision for EdgeLM

- **TQ2 is our primary format** -- simpler decode, proven performance
- **TQ1 is a stretch goal** for the 3.3B model where bandwidth is tighter
- The 3.3B model at TQ1 (0.66 GB) would have identical bandwidth profile to 2.4B at
  TQ2 (0.60 GB), giving us a 38% larger model at the same performance

---

## Finding 17: catid/bitnet_cpu -- Real AVX2 Benchmarks on Alder Lake

**Source:** https://github.com/catid/bitnet_cpu

### Implementation Details

Community implementation with two SIMD paths:

**AVX-512 kernel:**
- Uses `_mm256_mask_add_epi16()` and `_mm256_mask_sub_epi16()` for masked conditional
  add/subtract -- the purest ternary kernel form (no multiply at all)
- Weights packed at 2 bits/parameter

**AVX2 kernel:**
- Uses `_mm256_sign_epi8()` (VPSIGNB) for sign-based computation
- VPSIGNB: if b > 0, return a; if b == 0, return 0; if b < 0, return -a
- Perfect mapping for ternary weights stored as signed bytes
- Weights stored at 1 byte/parameter (8 bpw, not bandwidth-optimal)
- OpenMP multi-threading with unrolled inner loops

### Benchmark Results (Critical for Our Planning)

| CPU | Architecture | tok/s | Notes |
|-----|-------------|-------|-------|
| Intel Xeon W-2295 | AVX-512 | ~40 | Server CPU |
| AMD Ryzen 9 7950X | AVX-512 | ~50 | Desktop, Zen 4 |
| AMD Ryzen 9 7950X | AVX2 | ~15 | Same CPU, AVX2 only |
| Intel i9-12900K | AVX2 | ~28 | **Alder Lake, our closest reference** |
| AMD Ryzen 5 7535HS | AVX2 | ~15 | Laptop |

### Analysis for EdgeLM

The **i9-12900K result (~28 tok/s)** is our most relevant baseline:
- i9-12900K: 8P + 8E cores, same Golden Cove + Gracemont architecture
- i7-12700H: 6P + 8E cores, slightly lower P-core clocks, laptop TDP
- catid's kernel uses 8 bpw (1 byte/weight), NOT 2 bpw packed
- catid's kernel has no cache tiling, no explicit prefetching, no P/E-core scheduling

**Expected improvement path from 28 tok/s baseline:**
1. Switch to 2-bit packing: 4x less bandwidth -> potential 2-3x improvement
2. Add VNNI (VPDPBUSD): ~1.5-2x over MADDUBS chain
3. Cache tiling for L3 residence: ~1.3-1.5x
4. Software prefetching: ~1.2-1.4x
5. P-core/E-core aware scheduling: ~1.1-1.2x
6. Combined theoretical: 28 * 2.5 * 1.7 * 1.4 * 1.3 * 1.15 ≈ 250 tok/s (upper bound)
7. Realistic with inefficiencies: **80-120 tok/s** (our target range)

---

## Finding 18: BitNet.cpp mpGEMM Paper -- Lossless INT16 Accumulation

**Source:** https://arxiv.org/html/2502.11880 (ACL 2025)

### Lossless Inference Technique

The key innovation ensuring bit-exact results vs FP32 reference:

1. **INT16 accumulation:** Maintain partial sums in INT16 during LUT lookups
2. **Pack-and-unpack:** Split INT16 results via SIMD pack instruction, perform dual
   table lookups on packed results, reconstruct via unpack
3. **No quantization loss:** Unlike T-MAC which introduces small rounding errors during
   8-bit aggregation, bitnet.cpp maintains exact INT16 precision throughout

### Ternary Multiply Formula

An elegant single-bit sign technique:
```
x = sign XOR (sign + x)
```
where sign in {0, 1}. This enables ternary application using only addition and XOR,
with the sign bit determining whether to add or subtract.

### TL2 Mirror Consolidation (1.67 bpw)

- Group size = 3 ternary weights
- 3^3 = 27 possible combinations -> 27 LUT entries
- Split into 1-bit sign weight + 4-bit index weight = 5 bits for 3 weights
- Effective: 5/3 = 1.67 bits per weight
- Mirror consolidation: exploit symmetry to halve effective table size

### Performance Benchmarks (Directly Comparable Hardware)

**Intel i7-13700H (next-gen Raptor Lake, close to our i7-12700H):**
- 100B model Float16: 0.67 tok/s
- 100B model I2_S: 1.65 tok/s (2.46x speedup)
- 100B model TL2_0: 1.69 tok/s (2.52x speedup)

**Scaling to 2.4B model:** (100B/2.4B) * 1.65 = **69 tok/s** I2_S baseline
**Scaling to 3.3B model:** (100B/3.3B) * 1.65 = **50 tok/s** I2_S baseline

These scale estimates assume linear bandwidth scaling (valid for bandwidth-bound GEMV).
Our custom kernel optimizations should add 1.5-2x on top.

### Key Architectural Detail

BitNet.cpp builds on llama.cpp as a submodule:
- Custom kernels in `ggml-bitnet-mad.cpp`
- Generated LUT kernels in `bitnet-lut-kernels.h` (build-time generated)
- Configuration in `include/gemm-config.h`
- Three-phase execution: Setup (CPU detection) -> Model Prep (GGUF conversion) ->
  Inference (compiled kernels)

---

## Finding 19: Justine's llamafile -- Recursive Tiling and 84-Kernel Architecture

**Source:** https://justine.lol/matmul/

### Key Techniques Applicable to EdgeLM

**Recursive `mnpack` tile selection:**
- 3x4 tiles when both dimensions allow
- 4x1 or 1x4 for rectangular cases
- 1x1 for remainders
- Thread distribution: `duty = (tiles + nth - 1) / nth`

**Outer product with vectorized accumulation:**
```c
c0 = _mm256_fmadd_ps(a0, k0, c0);  // Reuse a0 across multiple k columns
c1 = _mm256_fmadd_ps(a0, k1, c1);
```
Load matrix-A column once, multiply against multiple B columns. This maximizes register
reuse and minimizes load bandwidth pressure.

**Critical bandwidth observation:**
"AVX-512 shows minimal improvement over AVX2 in naive implementations because the kernel
is memory-bound" -- confirming bandwidth as the true bottleneck, not instruction width.

### Performance on Alder Lake Family

| CPU | Model | Format | tok/s |
|-----|-------|--------|-------|
| i9-14900K | TinyLlama-1.1B | F16 | 407 |
| i9-9900 (Skylake) | Various | F16 | 23 (2x vs llama.cpp) |

The i9-14900K result (407 tok/s for 1.1B F16) gives us a ceiling: at ~2 GB model size
(F16), that's 407 tok/s from ~40 GB/s bandwidth = full bandwidth utilization.
For our 0.4-0.6 GB ternary model, the bandwidth ceiling allows 67-100 tok/s.

### KTransformers Integration (iqk_mul_mat)

- llamafile's `iqk_mul_mat` kernel integrated into KTransformers
- Achieves **150-400% speedup** over baseline llama.cpp for GGUF-quantized weights
- Supports 30+ GGML quantization types including IQ2_XXS, IQ3_S
- AVX2, AVX512, AVX512-VNNI, AVX512-VBMI, AVX512-BF16, and AMX paths

---

## Finding 20: CPU GEMM Hierarchical Blocking -- Design Rules

**Source:** https://salykova.github.io/matmul-cpu

### 6-Level Blocking Strategy (Applied to Ternary)

For our i7-12700H P-core cache hierarchy (L1d=48KB, L2=1.25MB, L3=25MB shared):

**Register tile (mr x nr):**
- mr = 32 (one YMM register of INT8 values)
- nr = 4 (number of output columns per micro-kernel)
- 4 accumulator registers + 4 weight registers + 4 activation registers + 4 temps = 16 YMM

**L1 tile (panel of packed B):**
- L1 = 48 KB, target 50% occupancy = 24 KB
- kc x nr = 24KB -> kc = 6144 ternary weights at 2 bpw = 1536 bytes per column
- kc = 6144, nr = 4 -> L1 panel = 6 KB (fits easily)

**L2 tile (packed A block):**
- L2 = 1.25 MB, target 75% = 960 KB
- mc x kc at 2 bpw: mc = 960KB / (6144 * 0.25 bytes) ≈ 625 rows

**L3 tile (packed B block):**
- L3 = 25 MB shared across 6 P-cores, ~4 MB per core
- nc x kc: nc = 4MB / (6144 * 0.25 bytes) ≈ 2600 columns

### Set-Associative Cache Conflict Warning

From Michal Pitr's analysis (https://michalpitr.substack.com/p/optimizing-matrix-multiplication):
"tile[0][0], tile[0][32], tile[0][64], tile[0][96] all map to the same cache set"

This means our weight packing MUST ensure contiguous memory layout. Stride-based access
to unpacked weights will thrash L1 cache due to set conflicts. The repacking step at
model load time eliminates this problem.

### Packing for Sequential Access

Column-major per panel for A (activations), row-major per panel for B (weights).
Zero-padding at boundaries avoids branch in inner loop.
Pack once at model load, store in `.edgelm` cache file.

### Multi-Threading

`#pragma omp parallel for collapse(2)` over output tile loops.
For our 6 P-cores: parallelize the jr (column tiles) and ir (row tiles) loops.
E-cores get separate, smaller work chunks due to 4x lower SIMD throughput.

---

## Finding 21: LUT Tensor Core -- Operator Fusion Insight

**Source:** https://arxiv.org/html/2408.06003v1

### The Fusion Insight We Can Use

LUT Tensor Core's key software optimization: **fuse RMSNorm output directly into LUT
precomputation**, eliminating one memory roundtrip.

For our pipeline, this means:
1. RMSNorm produces FP32/FP16 normalized activations
2. Instead of writing to memory and re-reading for quantization:
   - Quantize activations to INT8 in-register
   - If using LUT approach: build LUT directly from just-computed values
   - If using MADDUBS approach: feed INT8 directly to matmul kernel
3. Saves one full activation vector write + read (~32 KB for hidden_dim=4096)

### Weight Reinterpretation for Table Symmetry

LUT Tensor Core maps {0,1} binary weights to {-1,+1} signed weights, which makes
the lookup table symmetric: `LUT[-w] = -LUT[w]`. This halves the table size.

For ternary {-1,0,+1}, a similar trick: store only entries for non-zero weights,
then use the sparsity mask to skip zero-weight lookups entirely.

---

## Finding 22: KVTQ -- Ternary KV Cache Quantization

**Source:** https://openreview.net/forum?id=eZAlb8fX5y

### Technique

Applies ternary quantization to KV cache values (not just model weights):
- Uses "group of ternary digits of different quantization steps"
- Multiple ternary digits approximate each FP16 value
- Different quantization step sizes per group for accuracy

### Bandwidth Impact

If KV cache uses ternary (2 bpw) instead of our planned FP8 (8 bpw):
- 4x reduction in KV cache bandwidth
- For long sequences (2048+ tokens), KV cache dominates attention memory traffic
- Could be critical for staying within 6-7 GB RAM budget at longer contexts

### Applicability to EdgeLM

Worth investigating in Phase 5 (advanced optimizations). Our Phase 1-2 plan uses FP8
KV cache, which is simpler and well-proven. Ternary KV cache is a research contribution
that could feature in the MLSys/EuroSys paper.

---

## Finding 23: Community Experience and Failure Modes

### llama.cpp BitNet Integration (GitHub Issue #5761)

- **Source:** https://github.com/ggml-org/llama.cpp/issues/5761
- Issue filed requesting BitNet b1.58 ternary support
- Key finding: llama.cpp's `IQ1_S` quantization already uses ternary {-1, 0, +1}
- Issue closed as "stale" -- no native BitNet integration happened
- Consensus: wait for official code from Microsoft before implementing
- **Lesson for EdgeLM:** We're building from scratch, not waiting for llama.cpp integration

### ROCm Ternary Failure (HN Discussion)

- **Source:** https://news.ycombinator.com/item?id=47301180
- Community member: "My attempts to try ternary encodings from Unsloth with llama.cpp
  on ROCm failed miserably"
- Neither GGML nor ROCm could execute ternary on gfx1201 hardware
- CPU fallback proved insufficient
- **Lesson:** Existing frameworks lack proper ternary support, confirming the need
  for custom implementation

### Native Training vs Post-Hoc Quantization

- Strong community consensus: ternary models should be **trained natively** (like
  BitNet b1.58 2B4T) rather than post-hoc quantized from FP16
- Post-training quantization to ternary causes significant quality loss
- Our target models (BitNet-b1.58-2B-4T, bitnet_b1_58-3B) are all natively trained
- **No post-hoc ternary quantization path needed for EdgeLM**

---

## Techniques Comparison Matrix

| Technique | Source | Impact | Complexity | Already in Deep Dive? |
|-----------|--------|--------|------------|----------------------|
| I2_S MADDUBS kernel (2-bit unsigned) | BitNet.cpp [1] | Baseline (2.4-6.2x vs llama.cpp) | Low | Partially (mentioned, not detailed) |
| T-MAC LUT (VPSHUFB g=4) | Microsoft [2,3] | 5.8x over llama.cpp (2-bit) | Medium | Partially (concept only) |
| AVX-VNNI VPDPBUSD fusion | Intel ISA [6,7] | ~2x over MADDUBS chain | Low | Yes (mentioned) |
| 4-row register tiling (M_tile=4) | BitNet.cpp, gemma.cpp [8,9] | 1.3-1.5x over M_tile=1 | Medium | No |
| VPSIGNB ternary multiply | catid/bitnet_cpu [14] | Simplest kernel, ~28 tok/s Alder Lake | Low | No |
| T-SAR binary decomposition | arxiv 2511.13676 [11] | Eliminates LUT memory traffic | Medium-High | No |
| TABv2 VPSHUFB popcount | ETH Zurich [12] | 2.1-2.7x for binary accumulation | Medium | No |
| Spectra TQ1 1.6-bit packing | ICLR/ACL 2025 [13] | 24% bandwidth savings (slower decode) | High | No |
| TQ2 2-bit packing (confirmed) | Spectra/llama.cpp [13] | Validated fastest ternary on AVX2 | Low | Partially |
| Hybrid MADDUBS+LUT (dual port) | Analysis (p04+p15) | Theoretical 2x over single paradigm | High | No |
| Hierarchical cache blocking (6 levels) | salykova, Pitr [17,19] | Eliminates cache conflict misses | Medium | No |
| Operator fusion (RMSNorm→quantize→matmul) | LUT Tensor Core [18] | Saves 32KB roundtrip per layer | Medium | No |
| LUT memory traffic problem (T-SAR finding) | arxiv 2511.13676 [11] | Critical anti-pattern: 87.6% traffic | N/A | No |
| Bias correction for {0,1,2} encoding | BitNet.cpp [1] | Required for correctness | Low | No |
| KVTQ ternary KV cache | OpenReview [20] | 4x KV bandwidth reduction | High | No |
| Recursive mnpack tiling (llamafile) | justine.lol [16] | Adaptive tile selection, 5x speedup | Medium-High | No |
| BitNet a4.8 hybrid quantization | arxiv 2411.04965 [5] | 4-bit activations, 55% sparse | High | No |
| INT16 lossless accumulation (mpGEMM) | arxiv 2502.11880 [15] | Bit-exact results vs FP32 | Medium | No |

## Recommendations for EdgeLM

Ordered by impact-to-effort ratio:

1. **I2_S baseline with VNNI upgrade (Phase 1-2):** Implement the proven MADDUBS kernel
   from BitNet.cpp first, then swap VPMADDUBSW chain for single VPDPBUSD. This is the
   highest-confidence path. Expected: 65-80 tok/s single-thread on 2.4B model.

2. **4-row register tiling with sequential weight layout (Phase 2):** Use M_tile=4,
   K_tile=128 micro-kernel with tile-sequential weight packing at model load time.
   14 of 16 YMM registers used. Expected: 1.3-1.5x over untiled.

3. **Multi-threaded row partitioning (Phase 3):** Embarrassingly parallel across M
   dimension. P-cores get 4x more rows than E-cores (75%/25% split for 6P+8E).
   Expected: ~6-8x over single P-core.

4. **Software prefetching + large pages (Phase 3):** PREFETCHT0/T1 for next tile's
   weights 2-3 cache lines ahead. VirtualAlloc with MEM_LARGE_PAGES for 2MB pages.
   Expected: 20-40% bandwidth improvement.

5. **Avoid T-MAC LUT approach (anti-recommendation):** T-SAR finding shows LUTs cause
   87.6% of memory transactions. On our bandwidth-limited DDR4-3200, the MADDUBS/VNNI
   path is almost certainly faster. Test to confirm, but don't prioritize LUT.

6. **Operator fusion: RMSNorm → INT8 quantize → matmul (Phase 3):** Eliminate one
   memory roundtrip per layer (~32KB saved). Quantize activations in-register
   immediately after RMSNorm, feed directly to matmul kernel.

7. **TQ2 as primary format, TQ1 as stretch (Phase 2):** TQ2 is confirmed fastest
   ternary on AVX2 (simpler decode). TQ1 gives 24% bandwidth savings but decode
   overhead negates gains on CPU. Only consider TQ1 for 3.3B model if needed.

8. **Hybrid MADDUBS+LUT (Phase 4 experiment):** MADDUBS uses port p04, VPSHUFB uses
   port p15 -- they can execute simultaneously. Requires careful instruction
   interleaving. Theoretical 2x but high implementation complexity.

9. **T-SAR binary decomposition (Phase 4 alternative):** If VNNI doesn't reach targets,
   decomposing ternary into two binary dot products with VPSHUFB popcount is viable.
   Eliminates all integer multiplication. Requires 2 passes over activations.

10. **KVTQ ternary KV cache (Phase 5/paper):** 4x bandwidth reduction for KV cache.
    Research contribution for MLSys/EuroSys paper. Not needed for initial 100 tok/s
    target but critical for long-context scenarios.

### Revised Performance Expectations

| Configuration | Expected tok/s (2.4B) | Expected tok/s (3.3B) |
|--------------|----------------------|----------------------|
| catid baseline (AVX2, no tiling) | ~28 | ~20 |
| I2_S MADDUBS (tiled) | 45-55 | 33-40 |
| VNNI VPDPBUSD (tiled) | 65-80 | 48-58 |
| + Prefetching | 80-100 | 58-72 |
| + Multi-thread (14 cores) | **100-130** | **75-95** |
| + L3 cache amplification | **110-140** | **80-105** |

The 2.4B model (0.4 GB) is highly likely to exceed 100 tok/s.
The 3.3B model (0.6 GB) will require all optimizations to reach 100 tok/s.

---

## References

1. BitNet.cpp I2_S kernel: https://github.com/microsoft/BitNet/blob/main/src/ggml-bitnet-mad.cpp
2. T-MAC paper: https://arxiv.org/abs/2407.00088
3. T-MAC repo: https://github.com/microsoft/T-MAC
4. BitNet b1.58 paper: https://arxiv.org/abs/2402.17764
5. BitNet a4.8 paper: https://arxiv.org/abs/2411.04965
6. Instruction timing (uops.info): https://uops.info/html-instr/VPMADDUBSW_YMM_YMM_YMM.html
7. oneDNN int8 computation guide: https://uxlfoundation.github.io/oneDNN/dev_guide_int8_computations.html
8. Algorithmica GEMM tutorial: https://en.algorithmica.org/hpc/algorithms/matmul/
9. gemma.cpp matmul kernels: https://github.com/google/gemma.cpp/blob/main/ops/matmul-inl.h
10. BitNet b1.58 2B4T release: https://arxiv.org/abs/2504.12285
11. T-SAR (in-register LUT): https://arxiv.org/abs/2511.13676
12. TABv2/FasterTNN: https://github.com/fpgasystems/FasterTNN
13. Spectra 1.1 (TQ1/TQ2 packing): https://arxiv.org/abs/2506.23025
14. catid/bitnet_cpu (AVX2 benchmarks): https://github.com/catid/bitnet_cpu
15. bitnet.cpp mpGEMM paper: https://arxiv.org/abs/2502.11880
16. justine.lol matmul (llamafile): https://justine.lol/matmul/
17. salykova GEMM tutorial: https://salykova.github.io/matmul-cpu
18. LUT Tensor Core: https://arxiv.org/abs/2408.06003
19. Michal Pitr matmul: https://michalpitr.substack.com/p/optimizing-matrix-multiplication
20. KVTQ (ternary KV cache): https://openreview.net/forum?id=eZAlb8fX5y
21. MiCo-Lib (T-MAC style LUT): https://github.com/HKUSTGZ-MICS-LYU/MiCo-Lib
22. OpenGraviton: https://opengraviton.github.io/paper.html
23. Larq Compute Engine: https://github.com/larq/compute-engine
24. BitNet.cpp DeepWiki architecture: https://deepwiki.com/microsoft/BitNet/4-technical-architecture
25. llama.cpp ternary issue: https://github.com/ggml-org/llama.cpp/issues/5761

## Audit Addendum (2026-04-02)

- **EdgeLM should treat decode and prefill kernels as different products.**
  Batch-1 decode latency and prompt-heavy prefill throughput are different
  optimization problems and deserve separate kernel families if necessary.
- **Post-op fusion needs explicit ranking.** The next high-value kernel work is
  not just more raw matmul throughput, but deciding where:
  - bias,
  - residual add,
  - dequant,
  - or activation preparation

  can be fused without hurting register pressure too badly.
- **Kernel evaluation should include energy and bytes moved, not only tok/s.**
  That will matter later for the paper when two kernels are close on latency but
  differ in bandwidth pressure or thermal sustainability.
