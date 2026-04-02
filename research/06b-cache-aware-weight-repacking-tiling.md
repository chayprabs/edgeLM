# Section 06b: Cache-Aware Weight Repacking & Matrix Tiling Strategies -- Extended Research

## Overview
This supplementary research covers cache-aware matrix layouts, BLAS-style packing strategies,
and Intel Alder Lake-specific cache hierarchy optimization for ternary weight matrices. The
findings here extend Section 06 (Weight Packing & Repacking Strategies) with deeper analysis
of how to organize {-1, 0, +1} weight data in memory to maximize cache utilization during
matrix-vector and matrix-matrix operations on the i7-12700H.

**Target hardware cache hierarchy:**
- P-core (Golden Cove): 48 KB L1d (12-way), 1.25 MB L2 (10-way), 24 MB shared L3
- E-core (Gracemont): 32 KB L1d, 2 MB L2 per 4-core cluster (shared)
- Cache line: 64 bytes everywhere
- DDR4-3200 dual-channel: ~40 GB/s real bandwidth

---

## 1. Cache-Oblivious and Cache-Aware Matrix Layouts

### 1.1 Cache-Oblivious Recursive Algorithms
- **Source:** https://en.algorithmica.org/hpc/external-memory/oblivious/
- **Technique:** Recursive divide-and-conquer that automatically optimizes across all cache
  hierarchy levels without knowing specific cache sizes. Subdivide matrix until data fits
  in cache (N^2 <= M), then perform computation on subblocks.
- **I/O complexity:** O(N^3 / (B * sqrt(M))) for matrix multiply, vs O(N^3/B) for naive --
  improvement factor of sqrt(M)/1 where M = cache size in elements.
- **Base case threshold:** Typically 32x32 matrices when L1 cache size is unknown.
- **Performance impact:** Theoretical optimal for all cache levels simultaneously. In practice,
  constant factors make cache-aware (tuned) approaches 10-30% faster for known hardware.
- **Applicability to ternary weights:** Moderate. The recursive splitting pattern works for
  any matrix format. However, ternary matrices at 2 bits per weight are so compact that the
  overhead of recursive function calls and index computation may dominate. A 3B model's
  largest weight matrix (~4096 x 4096) at 2 bpw is only ~4 MB -- the recursion depth to
  reach L1 is shallow (2-3 levels). Cache-aware tiling with known block sizes is preferred.

### 1.2 Z-Order (Morton) Curve Matrix Layout
- **Source:** https://en.wikipedia.org/wiki/Z-order_curve
- **Source:** https://lemire.me/blog/2018/01/09/how-fast-can-you-bit-interleave-32-bit-integers-simd-edition/
- **Technique:** Map 2D matrix indices to 1D memory addresses by bit-interleaving row and
  column indices. Elements (i,j) are stored at address formed by interleaving bits of i and j.
  This preserves 2D spatial locality in the 1D memory layout.
- **Key property:** Strassen's algorithm uses Z-order arrangement. Subroutine for multiplying
  two blocks does not need to know total matrix size, only block size and memory location.
- **Performance data:** Valsalam & Skjellum (2002) demonstrated effective use for matrix
  multiply. Typical improvement over row-major: 15-40% for matrices exceeding L2 cache,
  negligible for matrices fitting in L2.
- **Morton code computation (SIMD):** On AVX2 with vpshufb lookup tables, Morton codes
  can be computed at 1.6 cycles per pair of coordinates. This is 30% faster than using
  the hardware PDEP instruction (2.1 cycles), and works well on AMD where PDEP is slow.
- **Applicability to ternary weights:** Low-to-moderate for inference. Z-order layout excels
  when both dimensions of a matrix are accessed with similar frequency (true for GEMM, but
  NOT for inference's dominant matrix-vector multiply where we stream through one dimension
  fully and index the other by the input token). For MatVec, row-major or panel-major is
  superior. Z-order could benefit prefill (batch GEMM) but adds index computation overhead.
  **Verdict: Skip for EdgeLM.** Panel-major (BLIS-style) is better for our mixed MatVec/GEMM
  workload.

### 1.3 Hilbert Curve Layout
- **Technique:** Similar to Z-order but with better locality preservation. Hilbert curves
  never jump between distant quadrants, unlike Z-order which has discontinuities.
- **Performance data:** Typically 5-10% better cache hit rate than Z-order for matrix
  traversal, but with higher index computation cost (no simple bit-interleave formula).
- **Applicability to ternary weights:** Very low. The index computation cost (requiring lookup
  tables or state machines) is prohibitive for the inner loop of ternary matmul where we
  process 128 elements every ~8 cycles. The locality benefit over simple tiling is marginal
  for matrices that fit in L3 (our case). **Verdict: Skip entirely.**

### 1.4 Blocked (Tiled) Matrix Multiplication
- **Source:** https://en.algorithmica.org/hpc/algorithms/matmul/
- **Source:** https://coffeebeforearch.github.io/2020/06/23/mmul.html
- **Key findings from Algorithmica's matrix multiply guide:**
  - Naive: ~0.42 GFLOPS -> With transposition: ~0.55 GFLOPS -> Vectorized: ~2.3 GFLOPS
    -> Kernel optimized: ~8 GFLOPS -> Full optimization: ~24 GFLOPS (93% of BLAS, 50x speedup)
  - Three-level hierarchical blocking:
    - **L3 block:** 64 columns of B
    - **L2 block:** 120 rows of A
    - **L1 block:** 240 rows of B
  - The blocking strategy "completely removes the memory bottleneck" by ensuring data reuse
  - 6x16 microkernel uses 12 of 16 YMM registers for accumulation
- **Key findings from CoffeeBeforeArch:**
  - 16-element blocks as fundamental tiling unit (matches 64-byte cache line for FP32)
  - Blocked-column serial (138ms) outperformed naive parallel (152ms) for 1024x1024 matrix
  - Alignment via aligned_alloc improved performance from 386ms to 309ms (20% improvement)
  - Avoiding power-of-two matrix dimensions prevents set-associativity conflicts
  - Best result: blocked-column + parallel = 30ms (35x over baseline 1067ms)
- **Performance impact:** 10-50x over naive, depending on matrix size vs cache size.
- **Applicability to ternary weights:** HIGH. This is the foundation of our approach. For
  ternary 2-bit weights, a 64-byte cache line holds 256 weight values. Our blocking strategy:
  - L1 tile (48 KB P-core): ~192 cache lines = ~49,152 ternary weights per tile
  - L2 tile (1.25 MB P-core): ~5,120 cache lines = ~1.3M ternary weights per tile
  - L3 tile (24 MB shared): entire model layer (~19 MB for 3B/32 layers)

---

## 2. Panel-Major and Tile-Major Layouts from BLAS Libraries

### 2.1 BLIS Framework Architecture
- **Source:** https://github.com/flame/blis/blob/master/docs/KernelsHowTo.md
- **Source:** https://github.com/flame/blis/blob/master/config/haswell/bli_cntx_init_haswell.c
- **Source:** https://github.com/flame/blis/blob/master/config/skx/bli_cntx_init_skx.c
- **Source:** https://github.com/flame/how-to-optimize-gemm/wiki
- **Reference papers:**
  - "BLIS: A Framework for Rapidly Instantiating BLAS Functionality" (Van Zee & van de Geijn, ACM TOMS 2015)
  - "Anatomy of High-Performance Matrix Multiplication" (Goto & van de Geijn, ACM TOMS 2008)
  - "BLISlab: A Sandbox for Optimizing GEMM" (Huang & van de Geijn, arXiv:1609.00076, 2016)

#### The 5-Loop GEMM Algorithm
The BLIS/GotoBLAS approach uses five nested loops with three levels of cache blocking:

```
Loop 5 (NC): partition B columns into panels of width NC  -> fits in L3
  Loop 4 (KC): partition the K dimension into slabs of depth KC  -> fits in L2
    Pack B panel (KC x NC) into row-panel format
    Loop 3 (MC): partition A rows into panels of height MC  -> fits in L1
      Pack A panel (MC x KC) into column-panel format
      Loop 2 (NR): iterate over B micropanels of width NR
        Loop 1 (MR): iterate over A micropanels of height MR
          Microkernel: compute MR x NR output tile via rank-1 updates
```

#### Packing Formats
- **A micropanel:** MC x KC block stored in **column-panel** format with leading dimension
  PACKMR (= MR). Elements within each MR-tall column are contiguous in memory. This means
  the microkernel reads MR contiguous elements per load instruction.
- **B micropanel:** KC x NC block stored in **row-panel** format with leading dimension
  PACKNR (= NR). Elements within each NR-wide row are contiguous. The microkernel broadcasts
  or scatters from these NR-contiguous elements.
- **Why panel format matters:** In standard row-major/column-major, accessing an MR-tall
  column of a row-major matrix requires strided loads (one element per cache line in the
  worst case). Panel-major format guarantees that the microkernel's access pattern is
  purely sequential, maximizing cache line utilization.

#### Concrete Parameters from BLIS Configurations

**Haswell (AVX2, 256-bit, most similar to Golden Cove for AVX2 code):**
- MR = 6, NR = 16 (float); MR = 6, NR = 8 (double)
- MC = 168, NC = 4080, KC = 256 (all types)
- Cache mapping: A panel (168 x 256 x 4B = 168 KB) fits in L2; B panel (256 x 4080 x 4B = 4 MB) in L3

**Skylake-X (AVX-512, 512-bit):**
- MR = 32, NR = 12 (float); MR = 16, NR = 14 (double)
- MC = 480, NC = 3072, KC = 384 (float); MC = 240, NC = 3752, KC = 256 (double)
- Cache mapping: A panel (480 x 384 x 4B = 720 KB) fits in 1 MB L2; B panel in L3

#### Register Blocking (MR x NR)
The microkernel computes an MR x NR output tile using:
- MR x NR / SIMD_WIDTH accumulator registers (e.g., 6 x 16 / 8 = 12 YMM registers for Haswell float)
- 1-2 registers for loading A column elements
- 1 register for broadcasting B row element
- Total: ~15 of 16 available YMM registers (Haswell) or ~30 of 32 ZMM registers (SKX)

**The key formula:** `memory_access_reduction = 2 * MR * NR / (MR + NR)`
For Haswell float (6x16): reduction factor = 2 * 96 / 22 = 8.7x fewer memory accesses than naive.

#### Performance Impact
- Packing overhead: typically 3-8% of total GEMM time for large matrices
- Benefit: 50-100x speedup over naive, 2-4x over cache-unaware vectorized code
- Packing amortization: the packed buffer is reused across the entire inner loop iterations.
  For a panel reused NR times in the microkernel loop, the amortization is NR:1.

#### Applicability to Ternary Weights
**HIGH.** The BLIS packing philosophy maps directly to our ternary kernel design:
- **Pre-pack at load time** (not per-inference): Since weights are constant, we pack ONCE
  into panel-major format during model loading. Zero runtime packing overhead.
- **MR/NR for ternary:** With 2-bit weights, a 256-bit YMM register holds 128 ternary values.
  Our microkernel should process MR=128 (one register of weights) x NR=4-8 (accumulator columns).
  But since ternary matmul is add/subtract (not FMA), we can use more accumulators.
- **Proposed EdgeLM microkernel:** MR=128 weights x NR=8 output columns = 8 YMM accumulator
  registers + 1 weight register + 1 activation register = 10 of 16 YMM registers. This
  leaves 6 registers for prefetch scheduling and temporary values.
- **Panel layout for ternary:** Pack weight columns into panels of 128 elements wide (MR=128
  ternary values = 32 bytes = half a cache line). Two such panels fill exactly one 64-byte
  cache line, enabling perfect cache line utilization.

### 2.2 GotoBLAS Packing Strategy
- **Source:** https://github.com/flame/how-to-optimize-gemm/wiki
- **Reference:** Goto & van de Geijn, "Anatomy of High-Performance Matrix Multiplication," 2008
- **Key insight:** The critical observation by Goto is that the A panel should fit in L2 cache
  (not L1), because the hardware prefetcher can stream from L2 to L1 efficiently during
  microkernel execution. The B panel streams from L3 (or memory), with software prefetch
  used to bring B data into L2 ahead of the microkernel.
- **The "inner kernel" strategy:**
  - A panel (MC x KC) is packed and pinned in L2
  - B panel (KC x NR) is loaded one micropanel at a time into L1
  - Output C tile (MC x NR) is in registers + L1
  - The A panel is reused NC/NR times (once per B micropanel)
- **Packing step yields "a surprisingly large performance boost"** -- the wiki explicitly
  calls this out as a critical optimization, not optional.
- **Performance:** Achieves 90% of processor peak after all optimizations.
- **Applicability to ternary weights:** The Goto insight about L2 residency for A panels is
  directly applicable. For our ternary weights, the weight matrix plays the role of A (fixed,
  reusable). We pre-pack it into column-panel format at load time. During inference:
  - Weight panel (KC x MC at 2 bpw): for KC=256, MC=4096, size = 256 KB -- fits in 1.25 MB L2
  - Activation vector (KC elements at 8 bpw): for KC=256, size = 256 B -- trivially fits in L1
  - Output accumulator (MC elements at 32 bpw): for MC=4096, size = 16 KB -- fits in L1

### 2.3 Salykova's Practical CPU GEMM Implementation
- **Source:** https://salykova.github.io/matmul-cpu
- **Key implementation details:**
  - **Microkernel:** 16x6 (MR=16, NR=6) on Ryzen 9700X with 16 YMM registers
  - **Register allocation:** 12 accumulator registers (6 cols x 2 per 16-float col) + 2 column
    vector + 1 broadcast = 15 of 16 registers
  - **Cache blocking (single-threaded on Ryzen 9700X):**
    - NC = 1535 (L3), MC = 1024 (L2), KC = determined by L1
  - **Multi-threaded (8 cores):**
    - MC = MR x threads x 5 = 16 x 8 x 5 = 640
    - NC = NR x threads x 50 = 6 x 8 x 50 = 2400
  - **Packing layout:** A packed as column-major (MC x KC), B packed as row-major (KC x NC)
  - **Padding strategy:** Undersized blocks copied into zero-padded buffers to avoid masking
    overhead in the kernel -- trades memory for branch elimination.
  - **Key optimization:** Explicit unrolling of accumulator variables (C00-C15) rather than
    arrays to prevent compiler register spilling.
  - **Performance:** Competitive with OpenBLAS; approaches 93% of theoretical peak.
- **Applicability to ternary weights:** HIGH. The zero-padding strategy for boundary blocks
  is directly applicable to our ternary kernels. Since ternary 0 means "skip" (add nothing),
  zero-padding naturally produces correct results. The explicit variable unrolling pattern
  should be used in our kernel to prevent the compiler from spilling accumulators to stack.

### 2.4 oneDNN Memory Formats for Intel Hardware
- **Source:** https://uxlfoundation.github.io/oneDNN/dev_guide_inner_product.html
- **Key findings:**
  - oneDNN recommends using `format_tag::any` to let the library choose optimal layout
  - For inner product (GEMM): weight formats `oi`, `oiw`, `oihw`, `oidhw`, `io`, `wio`, `hwio`
  - For convolution: blocked formats like `nChw8c` (8-channel blocks) and `nChw16c`
    (16-channel blocks) pack the channel dimension into SIMD-width groups
  - The blocking factor matches SIMD register width: 8 for AVX2 float, 16 for AVX-512 float
  - Source tensors with spatial dimensions are "flattened to 2D" for inner product
  - INT8 quantization: `u8`/`s8` sources with `s8` weights, per-channel scales supported
  - **Channels-last (NHWC)** is recommended over NCHW on Intel platforms for better cache
    behavior -- "more friendly to Intel platforms, and thus generally yields better performance"
- **Applicability to ternary weights:** Moderate. The blocked format concept (grouping the
  channel/feature dimension into SIMD-width blocks) applies directly. For our 2-bit ternary
  weights with AVX2 (256-bit = 128 ternary values per register), the natural blocking factor
  is 128 -- pack 128 consecutive weights contiguously (already what I2_S does). The oneDNN
  recommendation of `format_tag::any` reinforces that optimal layout depends on kernel
  implementation and should be co-designed with the microkernel.

### 2.5 Intel Extension for PyTorch Tuning Guidance
- **Source:** https://github.com/intel/intel-extension-for-pytorch/blob/main/docs/tutorials/performance_tuning/tuning_guide.md
- **Key findings for weight layout:**
  - NHWC (channels-last) format recommended over PyTorch default NCHW for Intel CPUs
  - oneDNN primitive caching: 1024 cached primitives by default, expandable to 65536
    for variable-shape workloads
  - Thread affinity: `KMP_AFFINITY=granularity=fine,compact,1,0` binds consecutive threads
    to separate cores, minimizing cache invalidation
  - `KMP_BLOCKTIME=0` or `1` for inference (no idle spinning)
  - Physical cores preferred over logical cores (HT disabled for SIMD)
  - `torch.set_flush_denormal(True)` to avoid denormal number performance penalty
  - Jemalloc/TCMalloc outperform system malloc for inference workloads
- **Applicability to ternary weights:** The thread affinity and denormal flushing advice
  applies directly. For our custom engine, we implement explicit P-core/E-core affinity
  rather than relying on KMP_AFFINITY. The channels-last recommendation reinforces that
  the "output feature" dimension should be the innermost (contiguous) dimension in our
  weight layout -- this matches column-panel format.

---

## 3. Recent Papers and Systems (2024-2026) on Weight Layout Optimization

### 3.1 T-MAC: Table Lookup for Low-Bit LLM Inference (EuroSys 2025)
- **Source:** https://github.com/microsoft/T-MAC, https://arxiv.org/abs/2407.00088
- **Authors:** Jianyu Wei, Shijie Cao, Ting Cao, Lingxiao Ma, Lei Wang, Yanyong Zhang, Mao Yang
- **Technique:** Replaces traditional matrix multiply with lookup table (LUT) approach.
  Groups low-precision weights (1-4 bit) into small units, precomputes all possible partial
  sums, stores in tables. Uses `pshufb` (x86) / `tbl` (ARM) SIMD instructions for fast
  table lookup instead of FMA operations.
- **Weight layout:** Weights are reorganized into groups of 4 for 1-bit, with precomputed
  LUTs. The LUT size is optimized from 2^n to 2^(n-1) by incorporating a sign bit.
  Configurable tiling with parallel kernel implementations.
- **Performance data:**
  - BitNet-3B: 20 tok/s (1 core), 48 tok/s (4 cores) on Snapdragon X
  - BitNet-3B: 30 tok/s (1 core), 71 tok/s (8 cores) on M2-Ultra
  - 4-5x speedup vs llama.cpp; 70% energy reduction
  - T-MAC on 2 CPU cores matches llama.cpp on 8 cores
- **Estimated performance impact:** 4-5x over dequantize-then-multiply approach.
- **Applicability to ternary weights:** DIRECTLY applicable. T-MAC is one of the two main
  approaches (LUT vs direct MAD) for ternary inference. However, on x86 with AVX2, the
  `vpshufb` instruction operates on 32 bytes and can do 32 simultaneous lookups, but
  each lookup only handles 4-bit indices. For ternary (2-bit), the direct add/subtract
  approach may be more efficient because: (1) no LUT construction overhead, (2) the
  conditional add/subtract path uses fewer instructions than LUT+accumulate, (3) AVX2
  `_mm256_maddubs_epi16` can process 32 ternary multiplications in a single instruction.
  **Verdict:** Benchmark both approaches, but expect direct MAD to win on our hardware.

### 3.2 BitNet.cpp Kernel Architecture (Microsoft, 2024)
- **Source:** https://github.com/microsoft/BitNet, https://arxiv.org/abs/2402.17764
- **Technique:** Custom kernels for 1.58-bit ternary inference. Three kernel variants:
  - I2_S: 2-bit packed, direct multiply-accumulate (both x86 and ARM)
  - TL1: ARM-specific LUT kernel
  - TL2: x86-specific LUT kernel (built on T-MAC methodology)
- **Weight tiling configuration (x86):**
  - With ACT_PARALLEL: ROW_BLOCK=4, COL_BLOCK=128, PARALLEL=4
  - Without ACT_PARALLEL: ROW_BLOCK=128, COL_BLOCK=32, PARALLEL=8
  - Latest parallel kernels with configurable tiling: additional 1.15-2.1x speedup
- **Performance:** 2.37-6.17x speedup on x86 CPUs vs llama.cpp FP16 baseline
- **Applicability to ternary weights:** This IS our reference implementation. The key insight
  from their tiling: ROW_BLOCK=4 with ACT_PARALLEL means processing 4 output rows
  simultaneously, each accumulating 128 weight values per iteration. This maps to 4 YMM
  accumulators x 128 ternary values = 4 x 32 bytes = 128 bytes of weight data per
  microkernel iteration. At 2 cycles per iteration, this is 64 bytes/cycle throughput.

### 3.3 BitNet b1.58 (Microsoft Research, 2024)
- **Source:** https://arxiv.org/abs/2402.17764
- **Authors:** Shuming Ma, Hongyu Wang, Lingxiao Ma, Lei Wang, et al.
- **Key finding:** Every parameter quantized to {-1, 0, 1}. Matches full-precision
  Transformer LLM with same model size and training tokens. Enables specialized hardware
  design for 1-bit LLMs.
- **Computation paradigm:** Matrix multiply becomes conditional add/subtract. No actual
  multiplication hardware needed. Weight value determines operation: -1 -> subtract,
  0 -> skip, +1 -> add.
- **Applicability:** Defines the computation model for our engine. The weight layout must
  be optimized for the add/subtract decision logic, not for multiply-accumulate.

### 3.4 BitNet a4.8 (Microsoft Research, 2024)
- **Source:** https://arxiv.org/abs/2411.04965
- **Authors:** Hongyu Wang, Shuming Ma, Furu Wei
- **Technique:** 4-bit activations with 1-bit weights. Hybrid quantization and sparsification
  strategy. Applies 4-bit activations to attention and FFN layers. Activates only 55% of
  parameters. Supports 3-bit KV cache.
- **Performance impact:** Enables INT4/FP4 kernels for faster inference.
- **Applicability to ternary weights:** Relevant for our activation quantization strategy.
  If we quantize activations to INT8 (our current plan), we process 8-bit x 2-bit operations.
  BitNet a4.8 suggests INT4 activations could work, which would double our effective
  throughput (4-bit activations = 2x more activations per cache line). Worth exploring
  as a Phase 5 optimization.

### 3.5 llamafile/tinyBLAS Custom GEMM
- **Source:** https://justine.lol/matmul/
- **Key weight layout insights:**
  - Outer loop unrolling (3x4 tiles) instead of inner loop -- each weight vector multiplied
    with k0, k1, k2, k3 before loading next weight
  - Register reuse pattern: load weight once, multiply by 4 activation elements
  - Tile-based work stealing thread model (no OpenMP overhead)
  - 2x faster than MKL for matrices fitting in L2 cache
- **Performance:** 790 GigaFLOPS vs MKL's 384 on same hardware; 1.6x over llama.cpp on i9-14900K
- **Applicability to ternary weights:** HIGH. The outer-loop unrolling strategy translates
  directly: load 128 ternary weights (1 YMM register), then apply to 4-8 different activation
  elements before moving to next weight chunk. This maximizes weight register reuse.

### 3.6 llama.cpp Matrix Multiplication Architecture
- **Source:** https://github.com/ggerganov/llama.cpp (ggml-cpu.c analysis)
- **Weight packing at inference time:** Converts source weights to optimized vec_dot format
  if needed. Conversion parallelized across threads.
- **Tiling:** 16x16 block size for output matrix to reduce cache misses and false sharing.
- **Thread scheduling:** Dynamic chunk allocation with atomic fetch. Strategy:
  - Distribute work across inner or outer loop based on which dimension is larger
  - If total chunks < threads x 4: redistribute for better utilization
  - NUMA-aware: "chunking by thread was measured to perform better on NUMA systems"
- **Quantization dispatch:** `type_traits_cpu` array maps each quantized format to
  specialized vec_dot function, with ARM MATMUL_INT8 enabling dual-row processing.
- **Applicability to ternary weights:** The 16x16 tiling is conservative. For our ternary
  format with 128-element blocks, a 128x8 tile (128 weights x 8 outputs) would better
  match our microkernel. The dynamic chunk allocation with atomics is a good model for
  our work-stealing thread pool.

---

## 4. Intel Alder Lake Cache Hierarchy Optimization

### 4.1 Golden Cove (P-core) Cache Characteristics
- **Sources:** Intel optimization manual, architectural analysis
- **L1 Data Cache:**
  - Size: 48 KB per core (12-way associative)
  - Latency: 5 cycles
  - Bandwidth: 2 x 64 bytes/cycle (2 loads + 1 store per cycle)
  - Load ports: 2 (port 2, port 3); Store port: 1 (port 4)
  - Peak L1 read bandwidth at 4.7 GHz: 2 x 64 x 4.7 = ~601 GB/s per core
- **L2 Cache:**
  - Size: 1.25 MB per core (10-way associative)
  - Latency: ~12-14 cycles (~3 ns at 4.7 GHz)
  - Bandwidth: 64 bytes/cycle per core
  - Peak L2 read bandwidth at 4.7 GHz: 64 x 4.7 = ~300 GB/s per core
  - Inclusive of L1 (maintains coherency)
- **L3 Cache (shared):**
  - Size: 24 MB total (i7-12700H), ~20-way associative
  - Latency: ~40-50 cycles (~10 ns at 4.7 GHz)
  - Bandwidth: ~200 GB/s aggregate across all cores
  - Non-inclusive (victim cache behavior)
- **Prefetchers (Golden Cove):**
  - L1 DCU prefetcher: detects sequential access, prefetches next cache line
  - L1 IP prefetcher: tracks per-instruction access patterns
  - L2 HW prefetcher: detects strides, prefetches into L2
  - L2 adjacent line prefetcher: fetches both cache lines in a 128-byte pair
  - Enhanced in Golden Cove: better stride detection, deeper prefetch depth
- **TLB:**
  - L1 DTLB: 64 entries (4 KB pages), 8 entries (2 MB pages)
  - L2 STLB: 2048 entries (4 KB and 2 MB pages)

### 4.2 Gracemont (E-core) Cache Characteristics
- **L1 Data Cache:** 32 KB per core (8-way)
- **L2 Cache:** 2 MB per 4-core cluster (shared, 16-way)
- **Key differences from Golden Cove:**
  - Smaller L1 (32 KB vs 48 KB) -- tighter tile size constraints
  - Shared L2 (2 MB / 4 cores = 500 KB effective per core) vs private 1.25 MB
  - Weaker prefetchers -- more benefit from software prefetch
  - No AVX-512 (same as P-cores on Alder Lake -- fused off)
  - AVX2 at 128-bit execution (2 cycles per 256-bit op) vs P-core native 256-bit

### 4.3 Cache-Aware Tile Sizes for Ternary MatVec on i7-12700H

**P-core (Golden Cove) tile sizing:**
```
Given:
  L1d = 48 KB, L2 = 1.25 MB
  Weight format: 2 bits per element (4 elements per byte)
  Activation format: INT8 (1 byte per element)
  Accumulator format: INT32 (4 bytes per element)

L1 tile strategy (activation + accumulator must fit in L1):
  - Activation vector slice: KC elements x 1 byte = KC bytes
  - Accumulator tile: MR elements x 4 bytes = 4*MR bytes
  - Weight tile in L1: MC x KC / 4 bytes
  - Constraint: KC + 4*MR + MC*KC/4 <= 48 KB

For MR=128 (one weight register), KC=512:
  - Activation: 512 bytes
  - Accumulator: 512 bytes
  - Weight: 128 * 512 / 4 = 16,384 bytes = 16 KB
  - Total: ~17 KB -- fits easily in 48 KB L1d

For MR=256, KC=1024:
  - Activation: 1 KB
  - Accumulator: 1 KB
  - Weight: 256 * 1024 / 4 = 64 KB -- exceeds L1d!
  - Need MR=256, KC=512: weight = 32 KB, total = 34 KB -- OK

L2 tile strategy (weight panel must fit in L2):
  - Weight panel: MC x KC / 4 bytes <= 1.25 MB
  - For KC=512: MC <= 1.25 MB * 4 / 512 = 10,240
  - For a 4096-dim model: MC=4096, weight panel = 4096*512/4 = 512 KB -- fits in L2
  - For a 11008-dim FFN: MC=5504, weight panel = 5504*512/4 = 704 KB -- fits in L2
```

**E-core (Gracemont) tile sizing:**
```
L1d = 32 KB, effective L2 per core = 500 KB
  - MR=128, KC=256: weight = 128*256/4 = 8 KB + act 256B + acc 512B = ~9 KB -- OK for L1
  - L2: MC=4096, KC=256: weight = 4096*256/4 = 256 KB -- fits in 500 KB effective
```

**Recommended tile parameters for EdgeLM:**
```
P-core microkernel:
  MR = 128 (ternary weights per register load)
  NR = 4 (output columns processed simultaneously)
  KC = 512 (inner dimension slice)
  MC = 4096 (full hidden dimension when possible)
  NC = model_dim (process full output in one pass for MatVec)

E-core microkernel:
  MR = 128 (same weight register width)
  NR = 2 (fewer accumulators due to 128-bit AVX2 execution)
  KC = 256 (smaller to fit in 32 KB L1)
  MC = 4096
  NC = model_dim
```

### 4.4 Intel Cache Optimization Techniques
- **Source:** https://travisdowns.github.io/blog/2020/05/18/icelake-zero-opt.html
- **Key findings (Ice Lake, applicable to Golden Cove):**
  - Store elimination: 96% silent (eliminated) evictions for redundant zero stores
  - 256-bit stores outperform 512-bit stores when writing zeros to L2 region (45% faster)
  - AVX-512 triggers voltage scaling that reduces dispatch throughput during transitions
  - Zero-over-zero writes are heavily optimized by the cache controller
- **Applicability to ternary weights:** The zero-value optimization is relevant because
  ternary weights have high zero sparsity (BitNet b1.58 typically 30-40% zeros). When
  accumulating with ternary weights, zero-valued weights contribute nothing. We can
  potentially skip zero blocks entirely. The store elimination behavior means that writing
  zero-initialized accumulator buffers is nearly free.

### 4.5 Software Prefetch Strategy for Ternary Weight Streaming
Based on the Golden Cove prefetcher characteristics:

**Strategy 1: Sequential weight streaming (MatVec decode)**
- Hardware L2 prefetcher handles sequential access well
- Software prefetch adds minimal value for the main weight stream
- Use `_mm_prefetch(addr + 8*64, _MM_HINT_T0)` only if profiling shows L1 misses
  (8 cache lines = 512 bytes ahead, ~2-4 microkernel iterations)

**Strategy 2: Layer-ahead prefetch**
- At start of layer N: prefetch layer N+1 weights to L3 with _MM_HINT_T2
- Layer weight size: ~19 MB (for 3B/32 layers at 2 bpw)
- This exceeds L3 (24 MB) if done all at once -- prefetch only the first MC x KC tile
  of the next layer instead (~512 KB)

**Strategy 3: KV cache access prefetch**
- KV cache access is non-sequential (token-dependent indices)
- Software prefetch is ESSENTIAL here: `_mm_prefetch(kv_cache + token_idx * stride, _MM_HINT_T0)`
- Prefetch multiple KV entries for upcoming attention heads

**Strategy 4: E-core software prefetch**
- Gracemont prefetchers are weaker than Golden Cove
- Add explicit software prefetch for ALL access patterns on E-cores
- Prefetch distance: 4-8 cache lines ahead (256-512 bytes)

---

## 5. Repacking Overhead Amortization

### 5.1 When to Repack
- **At model load time (once per session):** Convert GGUF TQ2_0 format to panel-major
  compute layout. Cost: ~50ms for 0.6 GB model (sequential memcpy with layout transformation).
- **Cache to disk:** Write repacked layout to `.edgelm` file. Subsequent loads skip repacking
  entirely (~120ms for direct file read into VirtualAlloc buffer).
- **NEVER repack per-inference:** The entire point of pre-packing is to eliminate per-token
  overhead. Weights are constant -- pack once, compute many times.

### 5.2 Repacking Cost Analysis
```
Model size: 0.6 GB (3B ternary) = 614 MB
NVMe read bandwidth: 5 GB/s
Repack throughput (memcpy + index transform): ~10 GB/s (L3-to-L3)

First load:  614 MB / 5 GB/s (read) + 614 MB / 10 GB/s (repack) = 123ms + 61ms = 184ms
Cached load: 614 MB / 5 GB/s (direct read) = 123ms
Savings: 61ms per session after first load (33% of startup time)

Per-token amortization (assume 1000 token session):
  First session: 184ms / 1000 = 0.184ms per token (~0.018ms overhead)
  Cached session: 123ms / 1000 = 0.123ms per token
  Either is negligible vs the ~10ms per token target at 100 tok/s
```

### 5.3 Panel-Major Repacking Algorithm for Ternary Weights

```c
// Repack from row-major TQ2_0 blocks to panel-major layout for microkernel
//
// Input: row-major weight matrix W[M][K] stored as TQ2_0 blocks
//        (each block = 256 ternary values in 64 bytes + 2 byte scale)
//
// Output: panel-major layout where each MC x KC panel has MR-wide columns
//         stored contiguously for sequential microkernel access
//
// Memory layout after repacking:
//   For each KC-width slice of the K dimension:
//     For each MC-height slice of the M dimension:
//       Data is stored as MC/MR consecutive micropanels
//       Each micropanel: MR elements x KC depth, column-by-column
//       MR ternary values = MR/4 bytes per column
//       Total micropanel: MR/4 * KC bytes
//
// Example for MR=128, KC=512:
//   Micropanel = 128/4 * 512 = 16,384 bytes = 16 KB
//   For MC=4096: 4096/128 = 32 micropanels per panel
//   Panel total = 32 * 16 KB = 512 KB (fits in 1.25 MB L2)

void repack_ternary_to_panel_major(
    const block_tq2_0* src,    // GGUF source
    uint8_t* dst,              // 64-byte aligned destination
    int M, int K,              // matrix dimensions
    int MC, int KC, int MR     // blocking parameters
) {
    for (int kc = 0; kc < K; kc += KC) {
        int kc_eff = min(KC, K - kc);
        for (int mc = 0; mc < M; mc += MC) {
            int mc_eff = min(MC, M - mc);
            for (int mr = 0; mr < mc_eff; mr += MR) {
                int mr_eff = min(MR, mc_eff - mr);
                // Pack one micropanel: mr_eff rows x kc_eff columns
                for (int col = 0; col < kc_eff; col++) {
                    // Copy mr_eff ternary values for column 'col'
                    // from rows [mc+mr .. mc+mr+mr_eff) at K-position kc+col
                    pack_ternary_column(src, dst, mc + mr, kc + col,
                                        mr_eff, M, K);
                    dst += (mr_eff + 3) / 4;  // ceil(mr_eff/4) bytes
                }
                // Pad micropanel to 64-byte boundary
                dst = align_up(dst, 64);
            }
        }
    }
}
```

---

## 6. Summary of Actionable Findings

| Finding | Source | Impact | Priority | Applicability to Ternary |
|---------|--------|--------|----------|--------------------------|
| BLIS panel-major packing: column-panel A, row-panel B | BLIS framework, Goto & van de Geijn 2008 | 2-4x over cache-unaware | P0 | HIGH -- pre-pack weights at load time |
| Haswell BLIS params: MR=6, NR=16, MC=168, NC=4080, KC=256 | BLIS config/haswell | Starting point for tuning | P1 | Adapt: MR=128 (ternary), NR=4-8, KC=512 |
| Microkernel register blocking: 12-15 of 16 YMM registers | Salykova, Algorithmica | 50-100x over naive | P0 | 8-10 accumulator + 2 weight + 1 activation |
| Zero-padding boundary blocks (avoid masking in kernel) | Salykova matmul-cpu | 5-10% kernel speedup | P1 | Ternary zero = no-op, natural padding |
| Cache tile: L1 = activation+acc, L2 = weight panel | GotoBLAS algorithm | Eliminates memory bottleneck | P0 | Weight panel 512 KB fits 1.25 MB L2 |
| P-core tiles: MR=128, KC=512, ~17 KB in L1 | Analysis for i7-12700H | Optimal for 48 KB L1d | P0 | Verified by cache size calculation |
| E-core tiles: MR=128, KC=256, ~9 KB in L1 | Analysis for Gracemont | Optimal for 32 KB L1d | P1 | Separate tile config for E-cores |
| Layer-ahead prefetch to L3 (_MM_HINT_T2) | Prefetch analysis | Hide DRAM latency at transitions | P1 | Prefetch first tile of next layer |
| Z-order/Morton layout | Wikipedia, Lemire | 15-40% for GEMM, not MatVec | Skip | Not beneficial for inference MatVec |
| Hilbert curve layout | Academic literature | 5-10% over Z-order | Skip | Index computation too expensive |
| T-MAC LUT approach: 4-5x over llama.cpp | T-MAC (EuroSys 2025) | Benchmark against direct MAD | P2 | Direct MAD likely wins on AVX2 |
| BitNet.cpp tiling: ROW_BLOCK=4, COL_BLOCK=128 | BitNet.cpp gemm-config.h | Reference tiling for x86 | P1 | Tune for our specific L2 size |
| oneDNN channels-last (NHWC) for Intel CPUs | Intel Extension for PyTorch | Better cache behavior | P1 | Output dim = innermost dimension |
| Golden Cove L2: 1.25 MB, 64 B/cycle, ~300 GB/s | Intel architecture docs | Sets L2 tile budget | P0 | Weight panel <= 1.25 MB |
| Gracemont L2: 2 MB shared / 4 cores = 500 KB eff | Intel architecture docs | Sets E-core tile budget | P1 | Smaller tiles for E-core kernels |
| Explicit variable unrolling prevents register spilling | Salykova implementation | 10-20% kernel improvement | P1 | Use for accumulator declarations |
| Outer-loop unrolling (reuse weight, vary activation) | llamafile/tinyBLAS | 1.6-2x over standard BLAS | P0 | Load weight once, apply to N activations |
| Repacking overhead: ~61ms for 0.6 GB, amortized to 0 | Cost analysis | Negligible per-token | P2 | Cache repacked to .edgelm file |
| HW prefetcher handles sequential; SW needed for KV cache | Golden Cove analysis | 10-20% for non-sequential | P1 | Essential for attention, optional for FFN |

## Audit Addendum (2026-04-02)

- **Decode GEMV and prefill GEMM should not necessarily share one packing
  layout.** If the project later supports both strongly, EdgeLM may want a
  decode-optimized default pack and a separately materialized prefill-oriented
  variant only when prompt-heavy workloads justify it.
- **Tile sizes should be auto-tuned against cache conflicts, not only capacity.**
  The right panel dimensions are constrained by set mapping and replacement
  behavior, not just "does it fit in L2."
- **A pack-version database is worth planning early.** Repacking choices should
  be versioned so benchmark results can be tied back to a concrete layout
  generation rule.
