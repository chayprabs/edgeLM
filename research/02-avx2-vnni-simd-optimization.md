# Section 02: SIMD Instruction Sets — Extended Research

## Overview
This section covers the SIMD instruction sets available on the i7-12700H (Alder Lake) and how to optimally exploit them for ternary LLM inference. The CPU provides AVX2 (256-bit), AVX-VNNI (VPDPBUSD), FMA3, and SSE4.2 — but critically lacks AVX-512 and AMX. Every matmul kernel, attention computation, and normalization layer depends on maximizing SIMD throughput. Getting this right is the single biggest determinant of whether we hit 100+ tok/s.

## What the Deep Dive Already Covers
- Complete AVX2 instruction inventory: integer arithmetic, shuffle/permute, bitwise ops, float ops, horizontal reductions
- AVX-VNNI VPDPBUSD instruction semantics and comparison with VPMADDUBSW chain
- What AVX-512/AMX would give us (2x throughput) — not available
- ARM NEON/Apple AMX comparison explaining why Apple Silicon is fast
- Port assignments and throughput numbers for key instructions
- Basic ternary matmul strategy using VPSHUFB LUT or VPDPBUSD

## New Findings

### 1. AVX-VNNI Kernel Optimization

#### 1.1 VNNI Achieves 2.2x Over bitnet.cpp on Alder Lake
- **Source:** https://arxiv.org/abs/2508.06753 (Intel researchers, "Ultra-Low-Bit Microkernels for AI-PC", Aug 2025)
- **Key idea:** Custom 1-bit and 2-bit GEMM kernels using VPDPBUSD with VNNI4-interleaved weight layout achieve 2.2x speedup over bitnet.cpp on Alder Lake CPUs. The unpack sequence for 128 2-bit entries requires just 1 shift + 2 AND + 4 byte-shuffles per 256-bit load.
- **Relevance to EdgeLM:** Directly applicable — our i7-12700H has AVX-VNNI. If bitnet.cpp does ~50 tok/s for 3B on our hardware, 2.2x puts us at ~110 tok/s, exceeding the 100 tok/s target. The VNNI4-interleaved layout is the key enabler.
- **Estimated impact:** 2.2x kernel speedup (VERY HIGH)
- **Implementation complexity:** High — requires custom weight layout and unpack logic
- **Details:** The paper integrates with PyTorch-TPP and uses libxsmm JIT compilation to generate platform-optimal microkernels at runtime. The JIT approach avoids hand-written assembly while achieving near-optimal performance. End-to-end 2-bit inference outperforms state-of-the-art by 2.2x, with up to 7x speedup vs 16-bit models.

#### 1.2 AVX-VNNI Has Identical Throughput to AVX-512 VNNI on Alder Lake
- **Source:** Reddit r/hardware discussion; confirmed by uops.info ADL-P measurements
- **Key idea:** Alder Lake P-cores have two 256-bit vector units. AVX-512 is executed as fused 256-bit micro-ops on these CPUs. Therefore AVX-VNNI (256-bit VPDPBUSD) achieves the SAME per-cycle throughput as AVX-512 VNNI. There is zero throughput penalty for lacking AVX-512.
- **Relevance to EdgeLM:** Eliminates a major design concern — we are NOT leaving performance on the table. Any AVX-512 VNNI kernel design can be straightforwardly adapted to 256-bit with identical per-cycle throughput on our hardware.
- **Estimated impact:** Removes constraint (HIGH — architectural confidence)
- **Implementation complexity:** N/A — this is architectural knowledge
- **Details:** This means benchmarks from AVX-512 VNNI papers can be directly translated to our AVX-VNNI expected performance, adjusted only for the 2x iteration count on 256-bit vectors.

#### 1.3 VPDPBUSD Eliminates INT16 Saturation Hazard
- **Source:** https://uxlfoundation.github.io/oneDNN/dev_guide_int8_computations.html (oneDNN documentation)
- **Key idea:** The standard AVX2 INT8 chain (VPMADDUBSW → VPMADDWD → VPADDD) has a correctness hazard: VPMADDUBSW produces signed INT16 results that can saturate. With ternary weights {-1,0,+1} and INT8 activations, saturation is possible when activations are large. VPDPBUSD computes 4 u8×s8 products directly to INT32 WITHOUT intermediate saturation — mathematically correct by construction.
- **Relevance to EdgeLM:** Critical correctness issue. On our i7-12700H with AVX-VNNI, we should use VPDPBUSD as the primary accumulation instruction, not the VPMADDUBSW chain. This gives us both correctness AND performance.
- **Estimated impact:** Correctness fix + performance improvement (HIGH)
- **Implementation complexity:** Low — single instruction replacement
- **Details:** For s8×s8 operations (when both operands are signed), oneDNN uses an offset technique: convert s8 to u8 by adding 128, then subtract compensation 128×W. For ternary weights encoded as {0,1,2} (unsigned), this is unnecessary — direct u8×s8 works.

#### 1.4 40% Kernel Speedup from Direct VNNI Optimization of bitnet.cpp
- **Source:** https://github.com/microsoft/BitNet/issues/259
- **Key idea:** A community contributor collapsed bitnet.cpp's 85-line ggml_vec_dot_i2_i8_s kernel into a short loop using _mm512_dpbusd_epi32, achieving ~40% kernel speedup. The 256-bit equivalent _mm256_dpbusd_epi32 would require 2x iterations but identical throughput on Alder Lake P-cores.
- **Relevance to EdgeLM:** A 40% speedup from VNNI alone is significant. Combined with optimized weight layout (Finding 1.1), this pushes well past 100 tok/s.
- **Estimated impact:** 40% kernel speedup (HIGH)
- **Implementation complexity:** Medium — adapting 512-bit approach to 256-bit is straightforward

#### 1.5 VNNI 4x1 Blocking Strategy
- **Source:** https://intel.github.io/intel-extension-for-transformers/latest/docs/intel_extension_for_transformers/transformers/runtime/kernels/docs/kernel_desc/kernel_vnni.html
- **Key idea:** Intel's VNNI GEMM kernel uses 4x1 sparse blocking (4 rows in M dimension, 1 in K), with each row broadcast to 4×16 elements matching VPDPBUSD's accumulation requirements. This 4x1 tiling delivers ~2x better performance than 1x4 alternatives. Activations are reordered on-the-fly using "four load and two swizzle instructions."
- **Relevance to EdgeLM:** The 4x1 blocking strategy is directly applicable to our VNNI kernels. For 256-bit (YMM) registers, this would process 4×8 blocks per iteration.
- **Estimated impact:** 2x over naive VNNI tiling (MEDIUM-HIGH)
- **Implementation complexity:** Medium — well-documented blocking pattern

### 2. Ternary Kernel Techniques

#### 2.1 _mm256_sign_epi8 for Zero-Cost Ternary Multiply
- **Source:** https://github.com/catid/bitnet_cpu
- **Key idea:** The AVX2 intrinsic `_mm256_sign_epi8(activation, weight)` implements exact ternary {-1,0,+1} multiplication in a SINGLE instruction: if weight<0 negate activation, if weight==0 zero it, if weight>0 keep it. Latency ~1 cycle. On i9-12900K: ~28 tok/s with 1 byte/weight packing.
- **Relevance to EdgeLM:** The simplest and most elegant ternary multiply — one instruction does the job. The downside is 8 bits per weight (wasteful for bandwidth). But combining with a 2-bit unpack step yields a clean kernel: unpack 2-bit → INT8 {-1,0,+1} → _mm256_sign_epi8.
- **Estimated impact:** Simplifies kernel design (MEDIUM-HIGH)
- **Implementation complexity:** Low — single intrinsic

#### 2.2 256 Ternary Products in 6 AVX2 Instructions
- **Source:** https://news.ycombinator.com/item?id=39535800 (HackerNews discussion)
- **Key idea:** Encode ternary weights using two bit registers (positive indicator and negative indicator). Product via XOR (signs) + AND (magnitude). This achieves 256 ternary products in just 6 AVX2 instructions. Estimated ~0.5-1 TOPS per core.
- **Relevance to EdgeLM:** This XOR+AND encoding is the most instruction-efficient ternary multiply known. Combined with proper accumulation, it forms the basis of the fastest possible ternary GEMV kernel on AVX2.
- **Estimated impact:** Minimal instruction count per product (HIGH)
- **Implementation complexity:** Low — straightforward bit operations
- **Details:** The encoding: weight=+1 → pos=1,neg=0; weight=-1 → pos=0,neg=1; weight=0 → pos=0,neg=0. Multiply: result_pos = activation AND pos_mask; result_neg = activation AND neg_mask; result = result_pos - result_neg. Zero weights naturally produce zero via AND.

#### 2.3 TQ1_0 and TQ2_0 Ternary Packing in llama.cpp
- **Source:** https://github.com/ggerganov/llama.cpp/pull/8151 (merged Sep 2024)
- **Key idea:** TQ1_0 packs 5 ternary values per byte (1.6875 bits/weight, exploiting 3^5=243<256). TQ2_0 uses simple 2 bits/value (2.0625 bits/weight including scale). TQ2_0 achieves **141.83 GB/s on Intel AVX2** vs 64.17 GB/s for Q4_K — roughly 2.2x faster. TQ2_0 is reported as "the fastest quant on compute-bound AVX2 computers."
- **Relevance to EdgeLM:** TQ2_0 being 2.2x faster than Q4_K on AVX2 validates the ternary approach decisively. The simpler 2-bit format (TQ2_0) may be preferable to TQ1_0's complex 5-trits-per-byte decode for our use case.
- **Estimated impact:** 2.2x over Q4_K (HIGH — baseline validation)
- **Implementation complexity:** Low-Medium — TQ2_0 is simple; TQ1_0 decode is complex

#### 2.4 Dense SIMD Beats Sparse at 40% Sparsity (Anti-Pattern)
- **Source:** https://github.com/HyperFoldUK/sparse-ternary-fma
- **Key idea:** At BitNet's typical ~40% zero-weight sparsity, dense SIMD kernels are FASTER than sparse skip-based approaches. The overhead of detecting and skipping zeros exceeds the savings. Sparse only wins at 80%+ sparsity. Branchless XOR-based encoding achieves 2.3x over scalar baseline.
- **Relevance to EdgeLM:** Critical anti-pattern — do NOT try to exploit zero-weight sparsity in our ternary kernels. Dense SIMD processing is the correct approach for BitNet's sparsity level.
- **Estimated impact:** Prevents wasted optimization effort (HIGH — negative result)
- **Implementation complexity:** N/A — don't implement sparse skipping

#### 2.5 bitnet.cpp MAD Kernels Outperform LUT on AVX2
- **Source:** https://arxiv.org/abs/2502.11880 (bitnet.cpp mpGEMM library, Feb 2025)
- **Key idea:** On AVX2, MAD-based kernels (VPMADDUBSW) take ~3.7ns per operation vs LUT-based kernels (VPSHUFB) at ~6.2ns. The paper attributes this to "insufficient hardware support" for LUT on x86 AVX2. The TL2 kernel with LUT is faster on ARM (where tbl instruction is efficient), but MAD wins on x86.
- **Relevance to EdgeLM:** Hardware-dependent finding. On our i7-12700H, prefer MAD/VNNI-based approach over VPSHUFB LUT for the primary kernel path. LUT may still be useful for E-core fallback or specific sub-operations.
- **Estimated impact:** Guides kernel selection (HIGH — avoids wrong approach)
- **Implementation complexity:** N/A — design decision

### 3. Golden Cove / Gracemont Microarchitecture Details

#### 3.1 Golden Cove P-Core SIMD Execution Details
- **Source:** uops.info ADL-P measurements, Wikipedia Golden Cove article
- **Key idea:** Golden Cove has 12 execution ports with TWO full 256-bit FMA units (ports 0 and 1), enabling 2 FMA/cycle = 32 FP32 ops/cycle. Key port assignments: FMA/mul on p01 (0.50 throughput), add on p15, shuffle/permute on p5 only (1.00 throughput — bottleneck), loads on p23A (0.33 throughput, ~3/cycle), stores on p49+p78. 512-entry ROB, 332-entry FP register file, 6-wide decode.
- **Relevance to EdgeLM:** FMA and integer multiply share ports 0/1, while shuffle uses port 5 exclusively. This means FMA and VPSHUFB can execute in parallel without contention. Loads (3/cycle) exceed FMA demand (2 inputs/cycle), so Golden Cove is NOT load-bottlenecked from L1 cache.
- **Estimated impact:** Defines performance ceiling (CRITICAL)
- **Implementation complexity:** Low — understanding guides kernel design
- **Details:**

| Instruction | Latency | Throughput | Ports | uops |
|---|---|---|---|---|
| VFMADD231PS ymm | 4 cycles | 0.50 (2/cyc) | p01 | 1 |
| VMULPS ymm | 4 cycles | 0.50 (2/cyc) | p01 | 1 |
| VADDPS ymm | 3 cycles | 0.50 (2/cyc) | p15 | 1 |
| VPMADDUBSW ymm | 5 cycles | 0.50 (2/cyc) | p01 | 1 |
| VPMADDWD ymm | 5 cycles | 0.50 (2/cyc) | p01 | 1 |
| VPERMPS ymm | 3 cycles | 1.00 (1/cyc) | p5 | 1 |
| VPSHUFB ymm | 1 cycle | 1.00 (1/cyc) | p5 | 1 |
| VMOVAPS ymm load | ~5 cycles | 0.33 (~3/cyc) | p23A | 1 |
| VMOVAPS ymm store | ~4 cycles | 0.50 (2/cyc) | p49+p78 | 2 |

#### 3.2 Gracemont E-Core: 128-bit SIMD Splitting
- **Source:** Chips and Cheese "Gracemont: Revenge of the Atom Cores", uops.info ADL-E measurements
- **Key idea:** Gracemont has 128-bit vector units. ALL 256-bit AVX2 instructions are split into 2×128-bit micro-ops (like AMD Zen 1). FMA throughput is 1/cycle with 6-cycle latency (vs 0.50/2/cycle, 4-cycle on Golden Cove). VPMADDUBSW is 2 uops at 2.00 cycle throughput (4x slower than P-core). 256-entry ROB, 207-entry FP register file.
- **Relevance to EdgeLM:** E-cores deliver roughly 2-4x less SIMD throughput per core than P-cores. For our 6P+8E config: P-cores provide ~60% of total SIMD throughput despite being only 43% of core count. E-cores should NOT be used for matmul hot path.
- **Estimated impact:** Guides thread scheduling (HIGH)
- **Implementation complexity:** Medium — requires heterogeneous-aware scheduling
- **Details:**

| Instruction | P-Core Throughput | E-Core Throughput | Ratio |
|---|---|---|---|
| VFMADD231PS ymm | 0.50 (2/cyc) | 1.00 (1/cyc) | 2x |
| VPMADDUBSW ymm | 0.50 (2/cyc) | 2.00 (0.5/cyc) | 4x |
| VPERMPS ymm | 1.00 (1/cyc) | 2.00 (0.5/cyc) | 2x |
| VMOVAPS ymm load | 0.33 (~3/cyc) | 1.00 (1/cyc) | 3x |

#### 3.3 llamafile Explicitly Avoids E-Cores on Alder Lake
- **Source:** https://justine.lol/matmul/
- **Key idea:** llamafile's tinyBLAS "takes special care to not run on your efficiency cores" on Alder Lake. This design choice achieves 63 tok/s vs 40 tok/s baseline for Mistral 7B q8_0 — a 57% improvement just from proper core pinning.
- **Relevance to EdgeLM:** Validates our planned P-core-only SIMD strategy. The 57% improvement from core pinning alone is a massive win with zero algorithmic complexity.
- **Estimated impact:** 57% improvement from core pinning (HIGH)
- **Implementation complexity:** Medium — CPU topology detection via CPUID leaf 0x1A or GetLogicalProcessorInformationEx on Windows

### 4. AVX2 Frequency and Thermal Behavior

#### 4.1 No AVX2 Frequency Throttling on Alder Lake
- **Source:** Wikipedia AVX CPU frequency scaling section; Travis Downs blog on Ice Lake+
- **Key idea:** Intel eliminated/greatly reduced AVX2 frequency offsets starting with Rocket Lake (11th gen). On Alder Lake, there are no documented AVX2-specific frequency penalties. The old Skylake "license levels" (L0/L1/L2) no longer apply for 256-bit operations. All-core AVX2 achieves 89-94% of peak single-core frequency.
- **Relevance to EdgeLM:** Excellent news — sustained AVX2 FMA workloads are viable without frequency cliffs. No need for complex duty-cycling strategies. Verify empirically with HWiNFO on our specific laptop.
- **Estimated impact:** Removes concern (HIGH positive)
- **Implementation complexity:** Low — no mitigation needed

#### 4.2 Thermal Throttling Is the Real Limiter
- **Source:** Intel i7-12700H specifications
- **Key idea:** The 45W PBP vs 115W MTP gap means sustained all-core AVX2 FMA will thermally throttle to well below peak turbo. Laptop thermal envelope, not AVX frequency offset, is the actual sustained performance limiter. P-core turbo: 4.7 GHz single-core drops to estimated 3.5-4.0 GHz sustained all-core under thermal load.
- **Relevance to EdgeLM:** Thermal monitoring and adaptive thread count are more important than AVX2 duty-cycling. Consider monitoring CPU temperature and reducing active P-core count if throttling detected. Always plugged in helps (no battery power limits), but cooling is the bottleneck.
- **Estimated impact:** Affects sustained performance (MEDIUM)
- **Implementation complexity:** Medium — requires thermal monitoring API

### 5. Register Blocking and Kernel Tiling

#### 5.1 Optimal 3x4 Register Blocking for 16 YMM Registers
- **Source:** https://justine.lol/matmul/ (llamafile), https://salykova.github.io/matmul-cpu
- **Key idea:** With only 16 YMM registers in AVX2, the optimal FP32 micro-kernel is 3x4 (12 accumulators + 4 temporaries). Each 256-bit YMM holds 8 FP32, so actual tile is 3×32 output elements. Pattern: broadcast A element → FMA with 4 B column vectors → accumulate into 12 output registers. Achieved ~810 GFLOP/s on i9-14900K.
- **Relevance to EdgeLM:** Direct template for our SGEMM kernel design. For ternary VNNI kernels, register allocation changes: VPDPBUSD accumulates into INT32, needing similar 12-accumulator layout with remaining registers for weight unpacking.
- **Estimated impact:** Defines kernel efficiency ceiling (HIGH)
- **Implementation complexity:** Medium-High — requires careful register allocation
- **Details:** Why 3x4 specifically: 2x4=8 accumulators wastes registers (insufficient ILP); 3x4=12 is the sweet spot; 4x4=16 leaves NO temp registers causing spills. Use VFMADD231PS ymm,ymm,[mem] to fuse loads with FMA, saving a register.

#### 5.2 tinyBLAS Outer-Product Vectorization with Recursive Tiling
- **Source:** https://justine.lol/matmul/, https://deepwiki.com/mozilla-ai/llamafile
- **Key idea:** tinyBLAS uses outer-product vectorization (unrolling outer loops, not inner) to maximize register reuse. Recursive `mnpack()` adaptively selects kernel tile sizes (1×1 to 5×5). Runtime CPU dispatch selects optimal kernel based on detected ISA (detects AVX-VNNI on Alder Lake). Achieved 5x speedup over llama.cpp for F16 on Alder Lake.
- **Relevance to EdgeLM:** The runtime dispatch detecting AVX-VNNI is directly relevant. For ternary matmul, adapt the same tiling strategy but replace FMA with sign_epi8 or VPDPBUSD. The recursive tiling handles arbitrary matrix dimensions cleanly.
- **Estimated impact:** 5x over baseline (HIGH — architectural pattern)
- **Implementation complexity:** Medium-High

#### 5.3 Block-Interleaved Weight Packing: 61% Speedup
- **Source:** https://github.com/ggerganov/llama.cpp/pull/12332
- **Key idea:** Group 8 quantization blocks into an interleaved structure (Block_Q4_Kx8). Scales/mins stored in 12-byte groups for cache efficiency. Vectorized quantization function. Achieves 45.80 → 73.77 t/s prompt processing on Ryzen 7600X (61% speedup) with zero perplexity loss. Trade-off: increased model loading time due to runtime repacking.
- **Relevance to EdgeLM:** Block interleaving is a proven technique for SIMD GEMM acceleration on AVX2. We should apply the same principle to ternary weight blocks. Runtime repacking at load time is acceptable — cache repacked weights to disk.
- **Estimated impact:** 61% for prompt processing (HIGH)
- **Implementation complexity:** Medium

### 6. GFNI and Novel Instruction Usage

#### 6.1 GFNI Instructions Available on Alder Lake for Bit Manipulation
- **Source:** https://www.corsix.org/content/galois-field-instructions-2021-cpus
- **Key idea:** The gf2p8affineqb instruction (part of GFNI, available on Alder Lake) enables arbitrary bit permutations within bytes, bit reversal, shifts of any magnitude, and arbitrary bitwise XOR patterns — ALL in a single instruction. This replaces multiple shift/mask/OR sequences when unpacking ternary weights from packed bit formats.
- **Relevance to EdgeLM:** Could replace 3-5 AVX2 instructions with a single GFNI instruction in the weight unpacking path. Particularly useful for converting 2-bit packed ternary weights into the format needed by VPDPBUSD.
- **Estimated impact:** Reduces unpack overhead (MEDIUM)
- **Implementation complexity:** Low — single instruction replacement
- **Details:** GFNI is confirmed available on Ice Lake and later, including Alder Lake. Check with CPUID before using. The instruction takes an 8×8 bit matrix as an immediate operand, applying an affine transformation to each byte in a 256-bit register.

#### 6.2 Carry-Save Adder Networks for Bit-Parallel Accumulation
- **Source:** https://arxiv.org/abs/2412.16370
- **Key idea:** Instead of VPSHUFB-based popcount, use Carry-Save Adder (CSA) networks built from basic AND/XOR/OR operations. CSA has a 3-operation critical path and approaches memory-bound performance with inputs as small as 4 KiB on AVX2. The full-adder compresses three inputs to two (sum + carry) without any lookup table.
- **Relevance to EdgeLM:** CSA networks could replace VPSHUFB for accumulating ternary multiply results. If ternary multiply is encoded as bit operations, CSA accumulates partial sums without horizontal reduction until the very end. Avoids the 16-entry LUT size limitation entirely.
- **Estimated impact:** Alternative accumulation path (MEDIUM)
- **Implementation complexity:** Medium-High — requires careful bit-level design

### 7. Compiler and Software Engineering

#### 7.1 Manual SIMD Still Beats Auto-Vectorization by ~1.83x
- **Source:** Daniel Lemire's blog (Feb 2026) — hex conversion benchmarks
- **Key idea:** Table lookup: 3.1 GB/s → autovectorized arithmetic: 23 GB/s → manual SIMD intrinsics: 42 GB/s. Compiler auto-vectorization gets ~55% of manual SIMD performance. For hot loops, hand-written intrinsics remain necessary.
- **Relevance to EdgeLM:** Budget hand-written intrinsics for GEMM/GEMV and attention kernels. Use auto-vectorization (Clang -O3 -march=alderlake) for everything else (norms, sampling, tokenizer).
- **Estimated impact:** Quantifies the tradeoff (MEDIUM)
- **Implementation complexity:** N/A — strategic decision

#### 7.2 Force-Inline Prevents 20x Performance Cliff
- **Source:** Daniel Lemire's blog (Feb 2026)
- **Key idea:** Inlining a simple function enabled compiler vectorization, achieving 20x speedup. A function call boundary completely blocked SIMD auto-vectorization.
- **Relevance to EdgeLM:** ALL inner-loop functions MUST use __forceinline (MSVC) or __attribute__((always_inline)) (GCC/Clang). This is a common pitfall with devastating consequences.
- **Estimated impact:** Prevents 20x regression (HIGH — easy to miss)
- **Implementation complexity:** Low — attribute on function declarations

#### 7.3 Clang -Rpass Diagnostics for Vectorization
- **Source:** https://llvm.org/docs/Vectorizers.html
- **Key idea:** Clang's `-Rpass=loop-vectorize`, `-Rpass-missed=loop-vectorize`, and `-Rpass-analysis=loop-vectorize` flags report exactly which loops were vectorized and why others were missed. Can also force vector width with `-mllvm -force-vector-width=8`.
- **Relevance to EdgeLM:** Essential for finding auto-vectorization failures in non-critical paths. Use during development to ensure all loops outside the hand-tuned hot path are auto-vectorized.
- **Estimated impact:** Catches missed optimizations (MEDIUM)
- **Implementation complexity:** Low — compiler flags

#### 7.4 libxsmm JIT for Custom Ternary GEMM Kernels
- **Source:** https://arxiv.org/abs/2508.06753, https://github.com/libxsmm/libxsmm
- **Key idea:** The 2.2x-over-bitnet.cpp result (Finding 1.1) uses libxsmm's JIT compiler to generate specialized GEMM microkernels at runtime. libxsmm supports AVX2, VNNI, and custom data types. The JIT approach generates platform-optimal code without hand-written assembly.
- **Relevance to EdgeLM:** Rather than hand-writing AVX2/VNNI assembly, we could use libxsmm as a backend. This dramatically reduces development time while achieving near-optimal performance. However, it introduces a dependency (counter to "zero dependencies" philosophy).
- **Estimated impact:** Fastest path to high performance (HIGH)
- **Implementation complexity:** Medium — requires learning libxsmm API

### 8. Data Layout and Packing

#### 8.1 FBGEMM ROW_INTERLEAVE=4 for VNNI Alignment
- **Source:** https://github.com/pytorch/FBGEMM/blob/main/src/PackBMatrix.cc
- **Key idea:** For INT8 AVX2 with 32-bit accumulation: ROW_INTERLEAVE=4 (4 K-dimension elements packed contiguously per output column), NCB=8 (8 output columns per tile). This matches VPDPBUSD's 4-element dot product perfectly. Zero-padding for partial tiles.
- **Relevance to EdgeLM:** Weight packing MUST interleave K elements in groups of 4 to feed VNNI instructions efficiently. This is the production-grade standard from Facebook's inference stack.
- **Estimated impact:** Foundational packing pattern (HIGH)
- **Implementation complexity:** Medium

#### 8.2 MLAS Zero-Point Handling via Bit-Flip
- **Source:** https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/core/mlas/lib/qgemm_kernel_avx2.cpp
- **Key idea:** ONNX Runtime's MLAS kernel converts s8 to u8 cheaply via `ZeroPointB ^ 0x80` (bit-flip of sign bit). For VNNI, uses specialized MlasGemmU8U8KernelAvx2Vnni path with M=4 rows (vs M=6 for non-VNNI). Also uses VPSHUFB for 4×4 byte transpose during data layout conversion.
- **Relevance to EdgeLM:** The bit-flip trick for sign conversion is a single-instruction optimization. The M=4 vs M=6 distinction for VNNI vs non-VNNI is important for our kernel tile size selection.
- **Estimated impact:** Micro-optimization (LOW-MEDIUM)
- **Implementation complexity:** Low

#### 8.3 bitnet.cpp TL2 Packing at 1.67 bits/weight
- **Source:** https://arxiv.org/abs/2502.11880
- **Key idea:** TL2 groups 3 ternary weights per index. Mirror consolidation: exploits symmetry to store 1-bit sign + 4-bit LUT index = 5 bits for 3 trits = 1.67 bits/weight (vs 2.0 for TQ2_0). Uses VPSHUFB for 16-entry table lookups. 2.32x speedup over T-MAC baseline.
- **Relevance to EdgeLM:** 17% smaller model than TQ2_0, reducing memory bandwidth needs. However, the decoding complexity is higher, and MAD-based approaches outperform LUT on our AVX2 hardware (Finding 2.5). Best used if bandwidth is the bottleneck.
- **Estimated impact:** 17% bandwidth reduction (MEDIUM-HIGH)
- **Implementation complexity:** High — complex packing scheme

### 9. Recent Papers and Emerging Techniques

#### 9.1 BitNet v2: Native 4-bit Activations with Ternary Weights
- **Source:** https://arxiv.org/abs/2504.18415 (Apr 2025)
- **Key idea:** Online Hadamard transformation before activation quantization smooths outliers into Gaussian-like distributions, enabling native 4-bit activations with 1-bit weights. The H-BitLinear module achieves this without quality loss.
- **Relevance to EdgeLM:** 4-bit activations + ternary weights = extremely efficient SIMD computation. The Hadamard transform is just additions and subtractions — SIMD-friendly. Reduces activation memory/compute by 2x vs INT8 activations.
- **Estimated impact:** 2x activation compute reduction (HIGH)
- **Implementation complexity:** Medium

#### 9.2 Mixed-Precision GEMM Micro-Kernels Paper (Jun 2025)
- **Source:** https://arxiv.org/abs/2506.11728
- **Key idea:** Advocates shifting from AXPY-centric (rank-1 updates with FMA) to DOT-product-centric operations for quantized inference. AVX-VNNI's VPDPBUSD IS the dot-product paradigm for x86. Data packing must co-design with micro-kernel loop structure.
- **Relevance to EdgeLM:** Confirms our VNNI-first approach is architecturally correct. The co-design principle means weight packing format and kernel tile size must be designed together, not independently.
- **Estimated impact:** Architectural validation (MEDIUM)
- **Implementation complexity:** N/A — design guidance

#### 9.3 FasterTNN/TABv2: SIMD Popcount for Ternary GEMM on AVX2
- **Source:** https://ieeexplore.ieee.org/document/11360778 (ETH Zurich, ISLPED 2025)
- **Key idea:** Replaces scalar popcount with VPSHUFB-based SIMD popcount for binary/ternary neural network bitwise GEMM on AVX2. TABv2 reduces total instruction count by ~15% through improved ternary encoding using separate sign/magnitude bit planes.
- **Relevance to EdgeLM:** Specifically targets our hardware constraints (AVX2 without AVX-512 VPOPCNTDQ). The two-bit sign/magnitude encoding (separate from bitnet.cpp's encoding) may offer advantages for SIMD processing.
- **Estimated impact:** 15% instruction reduction (MEDIUM-HIGH)
- **Implementation complexity:** Medium — academic implementation available

#### 9.4 BMC: Memory-Compute Balanced CPU Inference
- **Source:** https://arxiv.org/abs/2511.12031 (Nov 2025)
- **Key idea:** Periodic KV cache allocation with redundant rows eliminates copy overhead, achieving up to 3.2x throughput over HuggingFace baseline. Redundant rows repurposed for speculative decoding (additional 1.39x). Addresses memory allocation as a real bottleneck for CPU inference.
- **Relevance to EdgeLM:** Memory allocation overhead is non-trivial on CPU. The periodic allocation pattern with redundant rows is a clean system-level optimization. The speculative decoding synergy makes it doubly valuable.
- **Estimated impact:** 3.2x system throughput improvement (MEDIUM-HIGH)
- **Implementation complexity:** Medium

#### 9.5 ELUTQ: Hierarchical Quantization for LUT-Based Inference
- **Source:** https://arxiv.org/abs/2510.19482 (Oct 2025)
- **Key idea:** Hierarchical Linear Quantization (HLQ) eliminates dequantization overhead entirely. Bit-serial LUT-based GEMM achieves performance comparable to QAT without retraining at 2-bit precision.
- **Relevance to EdgeLM:** Alternative quantization approach that better fits non-uniform weight distributions. If BitNet ternary quality is insufficient, HLQ could provide better quality at 2-bit without the dequantization penalty.
- **Estimated impact:** Quality improvement path (MEDIUM)
- **Implementation complexity:** Medium

### 10. Multi-Stream and ILP Techniques

#### 10.1 Multi-Stream Parallel Processing from Compression
- **Source:** https://fgiesen.wordpress.com/2023/10/29/entropy-decoding-in-oodle-data-x86-64-6-stream-huffman-decoders/
- **Key idea:** Process 6 independent data streams in parallel to convert latency-bound work into throughput-bound work. Uses ANDN for non-destructive masking, SHRX for variable-length consumption. Achieves theoretical 1.57 cycles/symbol.
- **Relevance to EdgeLM:** The multi-stream principle directly applies to processing multiple output neurons in parallel during GEMV. If a single dot product is latency-bound (due to accumulator dependency chain), interleaving 4-6 independent dot products saturates execution ports.
- **Estimated impact:** Eliminates latency bottleneck (MEDIUM-HIGH)
- **Implementation complexity:** Medium

#### 10.2 IQK Unpack-Once-Reuse Pattern
- **Source:** https://deepwiki.com/mozilla-ai/llamafile (IQK documentation)
- **Key idea:** Unpack quantized weights once, then reuse the unpacked values across multiple activation vectors. This amortizes the expensive unpack step across batch dimension.
- **Relevance to EdgeLM:** Critical for prompt processing (batch > 1) where the same weights multiply against multiple token activations. Less relevant for autoregressive generation (batch = 1), but still applies to multi-head attention where Q/K/V projections share weights.
- **Estimated impact:** Reduces unpack overhead proportional to batch size (MEDIUM)
- **Implementation complexity:** Medium

## Techniques Comparison Matrix

| Technique | Source | Impact | Complexity | Already in Deep Dive? |
|-----------|--------|--------|------------|----------------------|
| VNNI 2.2x over bitnet.cpp | arXiv 2508.06753 | Very High | High | No |
| AVX-VNNI == AVX-512 VNNI throughput | Reddit/uops.info | High | N/A | No |
| VPDPBUSD eliminates INT16 saturation | oneDNN docs | High | Low | No |
| _mm256_sign_epi8 ternary multiply | catid/bitnet_cpu | Medium-High | Low | No |
| 256 products in 6 instructions (XOR+AND) | HackerNews | High | Low | No |
| TQ2_0: 141 GB/s on AVX2 | llama.cpp #8151 | High | Low | No |
| Dense beats sparse at 40% sparsity | sparse-ternary-fma | High | N/A | No |
| MAD outperforms LUT on AVX2 | bitnet.cpp paper | High | N/A | Partially (LUT mentioned, not comparison) |
| Golden Cove detailed port assignments | uops.info | Critical | Low | Partially (basic numbers given) |
| Gracemont 128-bit splitting | Chips and Cheese | High | Medium | No (only "50-70%" mentioned) |
| llamafile E-core avoidance: 57% gain | justine.lol | High | Medium | No |
| No AVX2 frequency throttling | Wikipedia/Travis Downs | High | Low | No |
| 3x4 register blocking optimal | justine.lol/salykova | High | Medium-High | No |
| Block-interleaved packing: 61% speedup | llama.cpp #12332 | High | Medium | No |
| GFNI for bit manipulation | corsix.org | Medium | Low | No |
| CSA networks for accumulation | arXiv 2412.16370 | Medium | Medium-High | No |
| Force-inline prevents 20x regression | Lemire blog | High | Low | No |
| libxsmm JIT kernels | arXiv 2508.06753 | High | Medium | No |
| BitNet v2: 4-bit activations | arXiv 2504.18415 | High | Medium | No |
| VNNI 4x1 blocking strategy | Intel docs | Medium-High | Medium | No |
| FasterTNN SIMD popcount | ISLPED 2025 | Medium-High | Medium | No |
| BMC periodic KV allocation | arXiv 2511.12031 | Medium-High | Medium | No |
| Manual SIMD beats auto-vec by 1.83x | Lemire blog | Medium | N/A | No |

## Recommendations for EdgeLM

Ordered by impact-to-effort ratio:

1. **Use VPDPBUSD (AVX-VNNI) as primary accumulation instruction** — Eliminates saturation hazard AND provides best throughput. Single instruction replacement. Our i7-12700H has full AVX-VNNI support with identical throughput to AVX-512 VNNI. (Effort: LOW, Impact: VERY HIGH)

2. **Pin all SIMD work to P-cores, avoid E-cores for compute** — llamafile proved 57% improvement just from core pinning. Detect core types via CPUID leaf 0x1A or GetLogicalProcessorInformationEx. E-cores for I/O, tokenization, prefetching only. (Effort: MEDIUM, Impact: HIGH)

3. **Adopt VNNI4-interleaved weight layout** — The Intel AI-PC paper's 2.2x over bitnet.cpp came primarily from optimized weight packing for VPDPBUSD. Repack weights at load time, cache repacked format to disk. (Effort: HIGH, Impact: VERY HIGH)

4. **Use TQ2_0-style 2-bit ternary packing as storage format** — Proven at 141 GB/s on AVX2 in llama.cpp. Simple, fast, well-tested. Convert to VNNI-interleaved format at load time. (Effort: LOW, Impact: HIGH)

5. **Implement 3x4 register-blocked micro-kernel** — Proven optimal for 16 YMM registers. 12 accumulators + 4 temps. Use VFMADD231PS ymm,ymm,[mem] to fuse loads. (Effort: MEDIUM-HIGH, Impact: HIGH)

6. **Use branchless XOR+AND ternary encoding** — 256 products in 6 instructions. Combined with VPDPBUSD accumulation, this is the fastest possible ternary GEMV path. (Effort: LOW, Impact: HIGH)

7. **Do NOT implement zero-weight sparse skipping** — Dense SIMD definitively outperforms sparse approaches at BitNet's 40% sparsity level. Don't waste time on this. (Effort: ZERO, Impact: HIGH — prevents wasted work)

8. **Prefer MAD/VNNI kernels over LUT/VPSHUFB on x86** — bitnet.cpp paper shows MAD at 3.7ns beats LUT at 6.2ns on AVX2. LUT (T-MAC style) is better on ARM but worse on our hardware. (Effort: ZERO, Impact: HIGH — correct design decision)

9. **Force-inline all inner-loop functions** — Prevents catastrophic 20x regression from compiler failing to vectorize across function boundaries. Use __forceinline / __attribute__((always_inline)). (Effort: LOW, Impact: HIGH)

10. **Investigate GFNI for weight unpacking** — Single gf2p8affineqb instruction could replace 3-5 shift/mask/OR instructions in the unpack path. Available on Alder Lake. (Effort: LOW, Impact: MEDIUM)

11. **Consider libxsmm JIT as acceleration path** — The 2.2x result used libxsmm to generate optimal kernels. Trades "zero dependencies" purity for faster development and near-optimal codegen. Decision: use for prototyping, hand-write final kernels. (Effort: MEDIUM, Impact: HIGH)

12. **Implement block-interleaved weight packing** — 61% speedup for prompt processing proven in llama.cpp. Group 8 ternary blocks, interleave scales and quants. (Effort: MEDIUM, Impact: HIGH for prefill)

## References

1. "Ultra-Low-Bit Microkernels for AI-PC" — arXiv:2508.06753 (Intel, Aug 2025)
2. "bitnet.cpp: Efficient Edge Inference for Ternary LLMs" — arXiv:2502.11880 (Microsoft, Feb 2025)
3. "T-MAC: Table Lookup for Low-Bit LLM Deployment on Edge" — arXiv:2407.00088 (Microsoft, Jul 2024, EuroSys 2025)
4. "BitNet b1.58 2B4T Technical Report" — arXiv:2504.12285 (Apr 2025)
5. "BitNet v2: Native 4-bit Activations" — arXiv:2504.18415 (Apr 2025)
6. "Mixed-Precision GEMM Micro-Kernels" — arXiv:2506.11728 (Jun 2025)
7. "ELUTQ: LUT-Based Edge Inference" — arXiv:2510.19482 (Oct 2025)
8. "BMC: Memory-Compute Balanced CPU Inference" — arXiv:2511.12031 (Nov 2025)
9. "Carry-Save Adder Networks for Bit-Parallel Accumulation" — arXiv:2412.16370 (Dec 2024)
10. "FasterTNN/TABv2: Ternary NN Inference on AVX2" — IEEE ISLPED 2025
11. llama.cpp TQ1_0/TQ2_0 PR — https://github.com/ggerganov/llama.cpp/pull/8151
12. llama.cpp Block-Interleaved Q4_K — https://github.com/ggerganov/llama.cpp/pull/12332
13. sparse-ternary-fma — https://github.com/HyperFoldUK/sparse-ternary-fma
14. catid/bitnet_cpu — https://github.com/catid/bitnet_cpu
15. llamafile tinyBLAS — https://justine.lol/matmul/
16. MLAS AVX2 Quantized GEMM — https://github.com/microsoft/onnxruntime (qgemm_kernel_avx2.cpp)
17. FBGEMM PackBMatrix — https://github.com/pytorch/FBGEMM (PackBMatrix.cc)
18. oneDNN INT8 Computations — https://uxlfoundation.github.io/oneDNN/dev_guide_int8_computations.html
19. Intel VNNI Kernel Documentation — https://intel.github.io/intel-extension-for-transformers/
20. GFNI Instructions — https://www.corsix.org/content/galois-field-instructions-2021-cpus
21. Algorithmica SIMD Guide — https://en.algorithmica.org/hpc/simd/
22. uops.info Instruction Tables — https://uops.info (ADL-P and ADL-E measurements)
23. Agner Fog Optimization Resources — https://agner.org/optimize/
24. Chips and Cheese: Gracemont — https://chipsandcheese.com/2021/12/21/gracemont-revenge-of-the-atom-cores/
25. Wikipedia: Golden Cove, Gracemont, Alder Lake, AVX frequency scaling
26. Daniel Lemire's Blog — https://lemire.me/blog/ (auto-vectorization and inlining findings)
27. Fabian Giesen's Blog — https://fgiesen.wordpress.com/ (multi-stream decoding)
28. libxsmm — https://github.com/libxsmm/libxsmm
29. BitNet VNNI Issue — https://github.com/microsoft/BitNet/issues/259
30. Matmul CPU Tutorial — https://salykova.github.io/matmul-cpu
31. SIMD Taxonomy — https://branchfree.org/2024/06/09/a-draft-taxonomy-of-simd-usage/
32. Validark SIMD Classification — https://validark.dev/posts/eine-kleine-vectorized-classification/
33. HackerNews Ternary Discussion — https://news.ycombinator.com/item?id=39535800
