# Section 06: Weight Packing & Repacking Strategies -- Extended Research

## Overview
Weight packing and repacking is the bridge between on-disk model format (GGUF) and the
compute-optimal in-memory layout that SIMD kernels consume. For ternary {-1, 0, +1} models,
the packing strategy directly determines memory bandwidth utilization, cache efficiency,
and kernel throughput. On the i7-12700H with 40 GB/s DDR4-3200, every byte of wasted
bandwidth costs ~2.5 ns -- at 100+ tok/s targets, the packing format IS the performance.

## What the Deep Dive Already Covers
- Ternary 2-bit packing: 4 values per byte, 64-byte aligned for AVX2
- Weight repacking at load time from GGUF to compute layout
- Caching repacked weights to disk (`.edgelm` cache format)
- TQ2_0 as primary format (2.0 bpw), I2_S as bitnet.cpp's native format
- mmap-based loading with VirtualAlloc + MEM_LARGE_PAGES

## New Findings

### 1. Real-World Performance Benchmarks for Ternary Inference

#### 1.1 BitNet.cpp Official Benchmarks (Microsoft)
- **Source:** https://github.com/microsoft/BitNet, https://arxiv.org/abs/2410.16144
- **Key data:**
  - x86 CPUs: 2.37x to 6.17x speedup over llama.cpp FP16 baseline
  - ARM CPUs: 1.37x to 5.07x speedup
  - Energy reduction on x86: 71.9% to 82.2%
  - 100B parameter BitNet b1.58 model runs at 5-7 tok/s on single CPU (human reading speed)
  - Latest parallel kernel update: additional 1.15x to 2.1x speedup over initial release
- **Relevance to EdgeLM:** These speedups are vs FP16 baseline, not vs Q4. For our 2.4B model
  (~0.4 GB ternary) on i7-12700H, we should expect significantly higher tok/s than the 100B
  benchmark. Extrapolating: if 100B = 5-7 tok/s, then 2.4B should be ~200-290 tok/s at
  equivalent efficiency (linear in model size for bandwidth-bound workloads), but real-world
  cache effects and overhead will reduce this. Target 100-120 tok/s is conservative and achievable.
- **Implementation note:** BitNet.cpp builds on llama.cpp framework with custom I2_S/TL1/TL2 kernels.

#### 1.2 T-MAC Lookup Table Performance (Microsoft Research)
- **Source:** https://github.com/microsoft/T-MAC, https://arxiv.org/abs/2407.00088
- **Key data:**
  - BitNet-3B on Surface Laptop 7 (Snapdragon X): 20 tok/s (1 core), 48 tok/s (4 cores)
  - BitNet-3B on M2-Ultra: 30 tok/s (1 core), 71 tok/s (8 cores)
  - BitNet-3B on Raspberry Pi 5: 11 tok/s
  - Llama-2-7B W2 prefill (256 tokens, 4 threads): T-MAC 50.1 tok/s vs llama.cpp 12.0 tok/s (4.2x)
  - Llama-2-7B W2 prefill (256 tokens, 8 threads): T-MAC 94.4 tok/s vs llama.cpp 21.3 tok/s (4.4x)
  - 4x throughput vs llama.cpp, 70% energy reduction
  - T-MAC 2-core = llama.cpp 8-core for equivalent throughput (reaching 40 tok/s)
  - Snapdragon X Elite: T-MAC CPU 12.6 tok/s (2 cores) vs NPU 10.4 tok/s on Llama-2-7B W4
- **Relevance to EdgeLM:** T-MAC's LUT approach achieves 48 tok/s for 3B ternary on 4 ARM cores.
  Our i7-12700H has 6 P-cores (Golden Cove, higher IPC than ARM) + 8 E-cores. With custom
  AVX2 kernels (not LUT), we should match or exceed this. The 71 tok/s on M2-Ultra 8 cores
  is the closest comparable to our 14-thread (6P+8E) configuration. Custom direct-compute
  kernels should beat LUT for ternary since add/subtract is cheaper than table lookup on x86.

#### 1.3 HuggingFace 1.58-bit Quantization Results
- **Source:** https://huggingface.co/blog/1_58_llm_extreme_quantization
- **Key data:**
  - Ternary packing: 4 values per byte using 2-bit encoding {0->-1, 1->0, 2->+1}
  - 8B model: 128 GB FP16 -> ~2.8 GB packed int8 (97% reduction)
  - Weight quantization: `scale = 1/mean(|W|)`, `W_q = clamp[-1,1](round(W * scale))`
  - Activation quantization: 8-bit per-token, `scale = 127/max(|X|)`
  - 71.4x energy reduction for arithmetic operations
  - Computation paradigm: INT8 addition replaces FP16 multiply-add
  - BitBlas kernel outperforms custom Triton and torch.matmul for mixed INT8 ops
  - Triton kernel implements on-the-fly weight unpacking from packed int8
- **Relevance to EdgeLM:** Confirms our approach. The scale factor per tensor/group is stored
  alongside packed weights. Our custom C kernels can skip the Triton overhead entirely.
  The 2-bit encoding matches our planned I2_S-compatible format.

#### 1.4 llamafile/tinyBLAS Performance (Justine Tunney)
- **Source:** https://justine.lol/matmul/
- **Key data:**
  - Intel i9-14900K, Mistral 7B Q8_0: 63 tok/s prompt processing (1.6x vs llama.cpp)
  - Raspberry Pi 5, TinyLLaMA F16: 62 tok/s (2.2x vs llama.cpp)
  - AMD Threadripper 7995WX, Mistral 7B F16: 485 tok/s (2.5x vs llama.cpp)
  - Achieves 790 GigaFLOPS vs MKL's 384 GigaFLOPS on same hardware
  - Key technique: outer loop unrolling (3x4 tiles) instead of inner loop optimization
  - Register reuse: each `a` vector multiplied with k0,k1,k2,k3 before next load
  - Tile-based work stealing thread model (no OpenMP)
  - 2x faster than MKL for matrices fitting in L2 cache
- **Relevance to EdgeLM:** The outer-loop unrolling pattern is directly applicable to our ternary
  kernels. Instead of FMA, we do conditional add/subtract, but the register reuse strategy
  is the same. The 1.6x over llama.cpp on i9-14900K (similar arch to i7-12700H) shows that
  custom kernels consistently beat generic BLAS.

### 2. Open-Source Weight Packing Implementations

#### 2.1 BitNet.cpp I2_S Format (Microsoft)
- **Source:** https://github.com/microsoft/BitNet/blob/main/src/ggml-bitnet-mad.cpp
- **Key data:**
  - **Encoding:** 2-bit per weight: `0 = -1`, `1 = 0`, `2 = +1`
  - **Quantization:** `q8[i] = (src[i] * scale > 0) ? 2 : (fabs(src[i]) < 1e-6) ? 1 : 0`
  - **Block size (x86/AVX2):** QK_I2_S = 128 elements per block
  - **Packing:** 128 ternary values into 32 bytes (4 values per byte, bits 0-1, 2-3, 4-5, 6-7)
  - **Row layout:** 32 groups of 32 elements = 1024 bytes per row block
  - **Scale factor:** single float stored at offset n/4
  - **Alignment:** 32-byte padding added; function returns `nrow * row_size / 4 + 32`
  - **AVX2 bit extraction:**
    ```c
    xq8_0 = _mm256_srli_epi16(xq8, 6);  // extract bits 6-7
    xq8_1 = _mm256_srli_epi16(xq8, 4);  // extract bits 4-5
    xq8_2 = _mm256_srli_epi16(xq8, 2);  // extract bits 2-3
    xq8_3 = xq8;                         // bits 0-1
    // All masked with 0x03
    ```
  - **Multiply-accumulate:** Uses `_mm256_maddubs_epi16` + `_mm256_madd_epi16` for int16->int32 accumulation
  - **Kernel variants:** 1x1 (baseline), 1xN (parallel rows), Nx1 (parallel cols with shared x-data)
- **Relevance to EdgeLM:** This IS our reference implementation. The I2_S format with
  128-element blocks and 32-byte packing is exactly what we should target. The bit extraction
  via shifts is efficient on AVX2 but we can potentially improve by using a different packing
  order that avoids the shift chain (e.g., interleaved layout where adjacent bytes hold
  adjacent groups, pre-shifted for direct masking).

#### 2.2 BitNet.cpp Kernel Configuration
- **Source:** https://github.com/microsoft/BitNet/blob/main/include/gemm-config.h
- **Key data (x86 with ACT_PARALLEL):**
  - ROW_BLOCK_SIZE = 4
  - COL_BLOCK_SIZE = 128
  - PARALLEL_SIZE = 4
  - Without ACT_PARALLEL: ROW_BLOCK_SIZE=128, COL_BLOCK_SIZE=32, PARALLEL_SIZE=8
- **Relevance to EdgeLM:** These parameters are tuned for generic x86. Our i7-12700H has
  1.25 MB L2 per P-core, so we can likely increase COL_BLOCK_SIZE for better data reuse.
  The ROW_BLOCK_SIZE=4 with PARALLEL_SIZE=4 means processing 4 output elements simultaneously
  across 4 accumulators -- this fits well within AVX2's 16 YMM register file.

#### 2.3 BitNet.cpp LUT Format (TL2)
- **Source:** https://github.com/microsoft/BitNet/blob/main/src/ggml-bitnet-lut.cpp
- **Key data:**
  - x86 TL2 workspace: `ne10 * ne11 * 11 * sizeof(int8_t) + 2 * ne11 * 2 * sizeof(float)`
  - ARM TL1 workspace: `ne10 * ne11 * 15 * sizeof(int8_t) + 1 * ne11 * 2 * sizeof(float)`
  - Memory aligned to 64-byte boundaries
  - Built on T-MAC's lookup table methodology
  - Supports fp32-to-fp16 conversion buffers
  - References external `bitnet-lut-kernels.h` for actual kernel implementations
- **Relevance to EdgeLM:** The LUT approach requires 11 bytes per element workspace (x86),
  which is 5.5x more memory than direct 2-bit storage. For our 6-7 GB RAM budget, direct
  compute (MAD) is more memory-efficient. LUT may be worth benchmarking for E-cores where
  the simpler memory access pattern could compensate for the larger working set.

#### 2.4 GGML TQ2_0 Ternary Format (llama.cpp)
- **Source:** https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-common.h
- **Key data:**
  ```c
  typedef struct {
      uint8_t qs[QK_K/4];  // QK_K=256, so 64 bytes of packed 2-bit values
      ggml_half d;          // 2-byte FP16 scale factor
  } block_tq2_0;            // Total: 66 bytes per 256 elements = 2.0625 bpw
  ```
  - 2 bits per element, 4 values per byte
  - Single FP16 scale factor per 256-element block
  - Static assert verifies no padding: exact `sizeof(ggml_half) + QK_K/4` bytes
  - Straightforward: direct 2-bit encoding, no base-3 tricks
- **Relevance to EdgeLM:** TQ2_0 is our PRIMARY input format from GGUF. The 256-element
  superblock with 64 bytes of packed data + 2-byte scale is simple to parse. However,
  for compute we may want to repack into 128-element blocks (matching AVX2 register width
  better) with the scale replicated or pre-broadcast. The 66-byte block size is NOT
  64-byte aligned -- we MUST repack for aligned access.

#### 2.5 GGML TQ1_0 Format (Higher Compression)
- **Source:** https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-common.h
- **Key data:**
  ```c
  typedef struct {
      uint8_t qs[(QK_K - 4*QK_K/64) / 5];  // base-3 packed: 5 values per byte (3^5=243<256)
      uint8_t qh[QK_K/64];                   // high bits
      ggml_half d;                            // scale
  } block_tq1_0;                              // 1.6875 bpw
  ```
  - Uses base-3 encoding: 5 ternary values packed into 1 byte (3^5=243 fits in uint8)
  - Separate high bits array for remaining elements
  - 1.6875 bpw (16% less bandwidth than TQ2_0)
- **Relevance to EdgeLM:** TQ1_0 saves 16% bandwidth but base-3 decoding (division/modulo by 3)
  is expensive on x86 -- previous research showed 2x slower throughput (70 vs 142 GB/s).
  NOT recommended for our engine unless model size forces it. Stick with TQ2_0/I2_S.

#### 2.6 GGML K-Quant Block Structure (Reference)
- **Source:** https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-quants.c
- **Key data:**
  - QK_K = 256 elements per superblock across all K-quants
  - Q2_K: 16-element subgroups, scale+min packed in 4+4 bits per group
  - Q4_K/Q5_K: `get_scale_min_k4()` extracts scales from packed nibbles
  - Q6_K: separates low 4 bits (ql, 64 bytes) and high 2 bits (qh, 32 bytes)
  - All use bitwise operations (shift/mask) in inner loops -- no FP in dequant
  - FP16 scale factors: 2 bytes each (`d`, `m`, `dmin`)
  - Reference implementations use scalar code as portable baseline
- **Relevance to EdgeLM:** Our TQ2_0 repacking should follow the same philosophy:
  bitwise-only inner loop, FP scale applied once per block after integer accumulation.
  The 256-element superblock convention means GGUF models will always have weights
  in multiples of 256, simplifying our block processing.

### 3. Memory Alignment and Page-Level Optimization

#### 3.1 Large Pages on Windows (VirtualAlloc + MEM_LARGE_PAGES)
- **Source:** https://learn.microsoft.com/en-us/windows/win32/memory/large-page-support
- **Key data:**
  - Requires `SeLockMemoryPrivilege` via `AdjustTokenPrivileges()`
  - Get minimum size via `GetLargePageMinimum()` (returns 2 MB on x86-64)
  - Allocation size AND alignment must be multiples of large page minimum
  - Memory is always read/write and NON-PAGEABLE (physically resident)
  - Must reserve+commit as single operation (cannot commit previously reserved range)
  - Part of process private bytes but NOT working set (non-pageable by definition)
  - Not subject to job limits
  - Allocate ALL large pages at startup -- fragmentation makes late allocation unreliable
  - Each large page translation uses single TLB entry (vs 512 entries for same 2 MB range with 4K pages)
- **Relevance to EdgeLM:** CRITICAL optimization. Our ~0.6 GB model needs ~300 large pages.
  With 4K pages that would be 153,600 TLB entries -- far exceeding the ~1500-2000 L2 TLB
  entries on Golden Cove. With 2MB pages, only 300 TLB entries needed. Must allocate at
  startup before memory fragments. Requires admin-granted SeLockMemoryPrivilege.

#### 3.2 Huge Pages Performance Data (rigtorp.se)
- **Source:** https://rigtorp.se/hugepages/
- **Key data (Linux benchmarks, conceptually applicable):**
  - **4K pages baseline:** 19.4M TLB load misses (93.43% miss rate), 0.709s execution
  - **2MB huge pages:** 6,320 TLB load misses (0.07% miss rate), 0.544s execution
  - **Performance improvement: ~23% faster** with 2MB pages
  - **1GB huge pages:** No additional benefit over 2MB for the test workload
  - TLB miss rate dropped from 93.43% to 0.07% (1300x reduction)
  - mimalloc library recommended for easy huge page integration
- **Relevance to EdgeLM:** 23% speedup is significant and essentially free once configured.
  The 93% -> 0.07% TLB miss reduction is dramatic. For our streaming weight access pattern
  (sequential read through model layers), TLB misses would otherwise be a constant overhead
  on every 4K page boundary. 2MB pages are sufficient -- 1GB pages offer no extra benefit.

#### 3.3 64-Byte Alignment for AVX2
- **Key data (from multiple sources):**
  - AVX2 loads/stores operate on 32-byte (256-bit) vectors
  - Cache lines are 64 bytes on all modern Intel (Golden Cove, Gracemont)
  - Misaligned 256-bit loads that cross cache line boundaries incur ~2x latency penalty
  - 64-byte alignment guarantees two full AVX2 loads per cache line, zero splits
  - BitNet.cpp uses 64-byte alignment: `Memory aligned to 64-byte boundaries`
  - GGML default alignment: 16 bytes (insufficient for optimal AVX2)
  - VirtualAlloc with MEM_LARGE_PAGES returns page-aligned memory (2MB boundary) -- automatically 64-byte aligned
- **Relevance to EdgeLM:** All weight buffers MUST be 64-byte aligned. This is non-negotiable
  for peak AVX2 throughput. Our arena allocator should round up all allocations to 64-byte
  boundaries. Using VirtualAlloc for the main weight buffer automatically satisfies this.

### 4. Weight Prefetching Strategies

#### 4.1 Software Prefetching Fundamentals (_mm_prefetch)
- **Key data (from Intel documentation and optimization guides):**
  - `_mm_prefetch(addr, _MM_HINT_T0)`: Prefetch to L1 + L2 + L3
  - `_mm_prefetch(addr, _MM_HINT_T1)`: Prefetch to L2 + L3 (skip L1)
  - `_mm_prefetch(addr, _MM_HINT_T2)`: Prefetch to L3 only
  - `_mm_prefetch(addr, _MM_HINT_NTA)`: Non-temporal, minimize cache pollution
  - Cache line size: 64 bytes on Golden Cove and Gracemont
  - Prefetch granularity: one cache line (64 bytes) per instruction
  - Optimal prefetch distance depends on memory latency and computation time
  - General formula: `distance = memory_latency / compute_time_per_element * element_size`
  - For DDR4-3200: ~80ns memory latency, so prefetch ~80ns worth of computation ahead
  - Golden Cove L2 hardware prefetcher: aggressive, handles sequential patterns well
  - Software prefetch most beneficial for non-sequential or strided access patterns
- **Relevance to EdgeLM:** For sequential weight streaming (our primary pattern), the hardware
  prefetcher on Golden Cove is already aggressive. Software prefetch is most valuable for:
  (a) next-layer prefetching (non-sequential jump to different weight region),
  (b) KV cache access (scattered/random pattern),
  (c) E-core Gracemont which has weaker hardware prefetching.

#### 4.2 Layer-Ahead Prefetching for Transformer Inference
- **Key data (synthesized from implementation analysis):**
  - **Concept:** While computing layer N, prefetch weights for layer N+1 into L3/L2
  - **Why it works:** Transformer layers are sequential; the next layer's weights are at a
    known, deterministic memory offset. Without prefetch, the first access to layer N+1
    weights hits DRAM (~80ns). With L3 prefetch, it hits L3 (~30-40ns on Alder Lake).
  - **Implementation pattern:**
    ```c
    // At start of layer N computation:
    const uint8_t* next_layer_weights = weights + (layer + 1) * layer_stride;
    for (size_t off = 0; off < layer_weight_size; off += 64) {
        _mm_prefetch((const char*)(next_layer_weights + off), _MM_HINT_T2);  // L3 only
    }
    ```
  - **Prefetch bandwidth budget:** Must not saturate memory bus with prefetch requests.
    For 0.6 GB model / 32 layers = ~19 MB per layer. At 40 GB/s bandwidth, prefetching
    19 MB takes ~0.5ms. Layer computation should take longer than this for prefetch to help.
  - **BitNet.cpp approach:** Relies on sequential access enabling hardware prefetcher;
    no explicit software prefetch in the MAD kernels (implicit via sequential access pattern)
  - **llamafile approach:** Compiler-driven prefetching via interleaved loads and FMA
    operations; lets CPU speculation hide latency naturally
- **Relevance to EdgeLM:** Layer-ahead prefetch with _MM_HINT_T2 (L3 only) is our best bet
  for next-layer warming. Use _MM_HINT_T0 for intra-layer block prefetching only when
  processing non-sequential patterns. Budget: no more than 10% of memory bandwidth for
  prefetch (~4 GB/s of prefetch traffic). With 19 MB per layer and ~1ms per layer at
  100 tok/s, we have 40 MB/s per-layer budget -- more than enough for L3 warming.

#### 4.3 Prefetch Distance Tuning
- **Key data:**
  - Golden Cove L1 data cache: 48 KB per P-core
  - Golden Cove L2 cache: 1.25 MB per P-core (unified)
  - Gracemont L1 data cache: 32 KB per E-core
  - Gracemont L2 cache: 2 MB per 4-core cluster (shared)
  - Shared L3: 24 MB (i7-12700H)
  - DDR4-3200 latency: ~80ns from CPU to DRAM
  - L3 latency: ~30-40ns on Alder Lake
  - L2 latency: ~12-14 cycles (~5ns at 4.7 GHz P-core boost)
  - **Optimal prefetch distance for weight streaming:**
    - For L1 prefetch (T0): 8-16 cache lines ahead (~512-1024 bytes)
    - For L2 prefetch (T1): 32-64 cache lines ahead (~2-4 KB)
    - For L3 prefetch (T2): 256+ cache lines ahead (~16 KB+), or next-layer
  - **Rule of thumb:** `prefetch_distance = memory_latency_cycles / cycles_per_cacheline_consumed`
  - For our ternary matmul: ~2-4 cycles to process 64 bytes of weights (pure add/sub),
    so at 80ns (~375 cycles) DRAM latency, prefetch ~94-188 cache lines (6-12 KB) ahead
- **Relevance to EdgeLM:** Start with 8 KB prefetch distance for intra-block weight prefetch
  and full-layer prefetch for next layer. Tune empirically -- the hardware prefetcher may
  already handle sequential access, making software prefetch redundant for the main loop
  but essential for layer transitions.

### 5. Disk Caching of Repacked Weights

#### 5.1 llama.cpp mmap Implementation
- **Source:** https://github.com/ggml-org/llama.cpp/blob/master/src/llama-mmap.cpp
- **Key data:**
  - **POSIX:** `mmap()` with `MAP_SHARED`, optional `MAP_POPULATE` for prefaulting
  - **Windows:** `CreateFileMappingA()` + `MapViewOfFile()`
  - **Prefaulting:** Linux uses `posix_madvise(POSIX_MADV_WILLNEED)` for optional preload;
    Windows uses `PrefetchVirtualMemory()` (dynamically loaded via GetProcAddress for Vista+)
  - **Direct I/O:** Linux supports `O_DIRECT` for bypassing page cache, requires filesystem
    block-size alignment: `off_t aligned_offset = offset & ~(alignment - 1)`
  - **NUMA:** `posix_madvise(POSIX_MADV_RANDOM)` to distribute pages across NUMA nodes
  - **Fragment unmapping:** Selective `munmap()` for fine-grained memory release
  - **Chunked reads:** 64 MB maximum per read call to accommodate system limits
  - **Fallback:** When O_DIRECT fails (EFAULT/EINVAL), auto-reopens with buffered I/O
- **Relevance to EdgeLM:** For Windows, use `CreateFileMappingA` + `MapViewOfFile` for initial
  GGUF loading. For the repacked cache, we should use `VirtualAlloc` + manual file I/O
  (not mmap) because we need large pages for the compute buffer and mmap cannot provide those.
  Strategy: mmap the GGUF for reading, VirtualAlloc the compute buffer with large pages,
  repack, then write the repacked buffer to a `.edgelm` cache file for fast subsequent loads.

#### 5.2 Repacked Weight Cache Design
- **Key data (synthesized from implementation patterns):**
  - **First load path:** GGUF mmap -> parse headers -> VirtualAlloc(MEM_LARGE_PAGES) for
    compute buffer -> repack weights (TQ2_0 -> aligned I2_S-style layout) -> write cache file
  - **Subsequent load path:** Read cache file directly into VirtualAlloc buffer (bypass repack)
  - **Cache file format:**
    ```
    Magic (4 bytes): "ELWC" (EdgeLM Weight Cache)
    Version (4 bytes): format version for invalidation
    Source hash (32 bytes): SHA-256 of original GGUF file (for invalidation)
    Model metadata (variable): layer count, dims, quantization params
    Padding to 2MB boundary (for direct large-page load)
    Weight data (aligned, repacked): direct memcpy into VirtualAlloc buffer
    ```
  - **Cache invalidation:** Compare GGUF file hash; if mismatch, re-repack
  - **Expected timing (0.6 GB model on PCIe 4.0 NVMe at 5 GB/s):**
    - First load (repack): ~120ms read + ~50ms repack = ~170ms total
    - Cached load: ~120ms read (direct copy, no repack) -- 30% faster
    - With `PrefetchVirtualMemory`: overlap DMA with initialization code
  - **llamafile approach:** Uses mmap of GGUF directly (no repacking) with
    compute-time dequantization -- simpler but slower for repeated inference
- **Relevance to EdgeLM:** The cache file eliminates repack overhead on subsequent runs.
  The key insight is aligning the cache file data to match the large-page VirtualAlloc
  buffer layout so we can use a single sequential read. On NVMe at 5 GB/s, 0.6 GB
  loads in 120ms regardless -- the repack saving (~50ms) matters for total startup time
  but not for inference throughput.

#### 5.3 Memory-Mapped File Strategies for Weight Loading
- **Key data:**
  - mmap advantages: zero-copy for read-only access, OS handles paging, lazy loading
  - mmap disadvantages: cannot use large pages, page faults on first access are ~4us each,
    for 0.6 GB = 153,600 page faults = ~614ms of fault handling (worse than sequential read!)
  - Sequential read + VirtualAlloc: predictable timing, supports large pages, no fault overhead
  - Hybrid approach: mmap for GGUF parsing (only touch header/metadata), VirtualAlloc for weights
  - `PrefetchVirtualMemory()` on Windows: asynchronous prefault, can overlap with other init
  - For cached weights: sequential `ReadFile()` into pre-allocated large-page buffer is optimal
  - llama.cpp uses `MAP_POPULATE` (Linux) to prefault all pages at mmap time, converting
    lazy faults into a single sequential read -- but still cannot use large pages
- **Relevance to EdgeLM:** DO NOT use mmap for the main weight buffer. Use VirtualAlloc with
  MEM_LARGE_PAGES and sequential ReadFile(). The 2MB large pages reduce TLB entries from
  153,600 to 300, and sequential read avoids the page-fault overhead entirely. mmap is
  only appropriate for the initial GGUF parsing phase.

### 6. Optimal Repacking Layout for AVX2 Ternary Kernels

#### 6.1 Proposed EdgeLM Repacked Format
Based on analysis of BitNet.cpp I2_S, GGML TQ2_0, and llamafile tiling:

```
EdgeLM Weight Block (compute-optimal layout):
- Block size: 128 elements (matches AVX2 processing width in BitNet.cpp)
- Packing: 4 ternary values per byte, 32 bytes per block
- Encoding: 0b00 = -1, 0b01 = 0, 0b10 = +1 (same as I2_S)
- Scale: FP16 (2 bytes), broadcast to __m256 at block processing time
- Block struct:
    uint8_t data[32];     // 128 packed ternary values
    uint16_t scale;       // FP16 scale factor
    uint8_t pad[30];      // Pad to 64 bytes for cache line alignment
  Total: 64 bytes per 128 elements = 4.0 bpw (with padding)

Layer layout:
- All weight blocks for one layer stored contiguously
- Layer header: 64 bytes (dims, element count, checksum)
- Blocks laid out in column-major order for matmul access pattern
- Layer padding to 2 MB boundary (for large-page alignment)
```

**Trade-off analysis:**
- 4.0 bpw vs 2.0625 bpw (TQ2_0): 2x more bandwidth for weights BUT eliminates:
  - Scale factor lookup (scale is co-located with data in same cache line)
  - Alignment fixup (every block starts on cache line boundary)
  - TLB misses (2MB layer alignment + large pages)
- At 40 GB/s DDR4: 0.6 GB model becomes ~1.2 GB with padding -> 30ms per full-model read
  vs 15ms for TQ2_0. At 100 tok/s (10ms/token), the extra 15ms per full model scan is
  amortized across the full inference -- actual overhead is ~0.5ms per token for the padding.

**Alternative: Compact layout (2.0625 bpw) with separate scale array:**
```
Per-layer:
  uint8_t packed_weights[n_elements / 4];  // Dense 2-bit packed, 64-byte aligned
  uint16_t scales[n_elements / 256];       // One FP16 scale per 256 elements
```
This is more bandwidth-efficient but requires two memory streams per matmul operation.

**Recommendation:** Start with the compact layout (matches TQ2_0 wire format, minimal
repacking needed) and only switch to padded 64-byte blocks if profiling shows scale
fetch latency or alignment issues are measurable bottlenecks.

#### 6.2 Repacking Algorithm
```c
// Repack TQ2_0 GGUF blocks into aligned compute buffer
void repack_tq2_to_compute(
    const block_tq2_0* src,   // GGUF mmap'd source
    uint8_t* dst_weights,     // 64-byte aligned VirtualAlloc buffer
    uint16_t* dst_scales,     // Separate scale array
    size_t n_blocks)
{
    for (size_t b = 0; b < n_blocks; b++) {
        // Copy 64 bytes of packed data (256 elements / 4 = 64 bytes)
        memcpy(dst_weights + b * 64, src[b].qs, 64);
        // Copy scale
        dst_scales[b] = src[b].d;
    }
    // dst_weights is now dense, 64-byte aligned, sequential
    // dst_scales is a compact array, likely fits in L1 cache
}
```

This is essentially a scatter-to-gather transformation: TQ2_0 interleaves data+scale
in 66-byte blocks (not aligned); our layout separates them into aligned streams.

### 7. Alder Lake Cache Hierarchy (i7-12700H Reference)

- **Source:** https://en.wikipedia.org/wiki/Alder_Lake_(microprocessor)
- **Key data:**
  - **P-cores (Golden Cove, 6 cores):**
    - L1i: 32 KB per core (8-way)
    - L1d: 48 KB per core (12-way)
    - L2: 1.25 MB per core (unified, 10-way)
  - **E-cores (Gracemont, 8 cores in 2 clusters):**
    - L1i: 64 KB per core
    - L1d: 32 KB per core
    - L2: 2 MB per 4-core cluster (shared)
  - **L3 (shared):** 24 MB (i7-12700H, 20-way)
  - **Cache line:** 64 bytes everywhere
  - **Memory:** DDR4-3200 dual-channel, ~40 GB/s theoretical
  - **L2 bandwidth (Golden Cove):** 1 cache line/cycle = 64 bytes/cycle at ~4.7 GHz = ~300 GB/s
  - **L3 bandwidth:** ~200 GB/s aggregate (all cores)

- **Relevance to EdgeLM:**
  - Per-layer weight working set: ~19 MB (0.6 GB / 32 layers) -> exceeds L3 (24 MB) but fits if
    only weight data (without activations/KV cache)
  - Per-block working set should target L2: 1.25 MB P-core, 0.5 MB per E-core
  - Tile sizes for matmul should keep activation tile in L1 (48 KB) while streaming weights from L2
  - Hardware prefetcher on Golden Cove is aggressive for sequential patterns -- supplement
    with software prefetch only for non-sequential accesses

### 8. DRAM Refresh Impact on Bandwidth
- **Source:** https://blog.cloudflare.com/every-7-8us-your-computers-memory-has-a-hiccup
- **Key data:**
  - DRAM refresh every 7812.5 ns (tREFI), recovery time 75-120ns (tRFC)
  - Memory unavailable during refresh: 0.4-5% of total time
  - Latency spikes: typical 140ns access can jump to 360ns during refresh
  - Above 85C: refresh interval halves (3906 ns), doubling bandwidth loss
  - Impact on sustained bandwidth: real-world DDR4-3200 achieves ~38-40 GB/s
    (vs 51.2 GB/s theoretical) partly due to refresh overhead
- **Relevance to EdgeLM:** Budget for ~40 GB/s real bandwidth (not theoretical 51.2 GB/s).
  DRAM refresh is one factor in the ~22% efficiency gap. Laptop thermal management matters --
  keep memory cool to avoid halved refresh intervals. This is already accounted for in our
  bandwidth math (40 GB/s / 0.6 GB = ~67 tok/s baseline).

### 9. T-MAC VPSHUFB Lookup Table Kernel (Critical Technique)

#### 9.1 Core LUT Mechanism
- **Source:** https://arxiv.org/abs/2407.00088, https://github.com/microsoft/T-MAC
- **Key idea:** Replace multiply-accumulate with table lookup. For groups of g=4 weight bits,
  precompute a 16-entry LUT from the activation vector containing all possible partial sums.
  VPSHUFB (`_mm256_shuffle_epi8`) performs 32 parallel lookups per cycle.
- **AVX2 kernel code from T-MAC source (`python/t_mac/intrins/tbl.cc`):**
  ```c
  // Extract 4-bit packed indices from weights
  __m128i vec_a_bot = _mm_and_si128(vec_as, vec_mask);  // low nibble
  __m128i vec_a_top = _mm_and_si128(_mm_srli_epi16(vec_as, 4), vec_mask);  // high nibble
  // Duplicate LUT into both 128-bit lanes
  __m256i vec_lut_ = _mm256_set_m128i(vec_lut[k], vec_lut[k]);
  __m256i vec_a = _mm256_set_m128i(vec_a_top, vec_a_bot);
  // Single-instruction 32-way parallel lookup
  __m256i vec_v = _mm256_shuffle_epi8(vec_lut_, vec_a);
  ```
- **Sign-bit reduction trick:** Table size halved from 2^g to 2^(g-1) because each positive
  LUT entry pairs with its negative counterpart (mirror symmetry).
- **Relevance to EdgeLM:** VPSHUFB has 1-cycle latency on Golden Cove P-cores with 0.5-cycle
  throughput (2 per cycle). Each P-core can perform 64 LUT lookups/cycle. With 6 P-cores at
  4.7 GHz = ~1.8 billion lookups/second. However, for ternary weights, direct MAD via
  `_mm256_maddubs_epi16` may beat LUT since add/subtract is simpler than table construction.
  **Must benchmark both approaches.**

#### 9.2 VPSHUFB as Parallel 16-Entry Lookup Table
- **Source:** https://en.algorithmica.org/hpc/simd/shuffling/
- **Key constraint:** AVX2 VPSHUFB operates within 128-bit lanes -- the table must be duplicated
  in both halves of the 256-bit register. T-MAC handles this with `_mm256_set_m128i(lut, lut)`.
- **AVX2 popcount via VPSHUFB (Harley-Seal algorithm):**
  ```c
  const reg lookup = _mm256_setr_epi8(
      0, 1, 1, 2, 1, 2, 2, 3,
      1, 2, 2, 3, 2, 3, 3, 4,
      // repeated for second lane
  );
  // Split byte into nibbles, lookup each, sum -> ~30 bytes/cycle throughput
  ```
- **Relevance to EdgeLM:** This is an alternative to popcount-based ternary matmul (see 9.5).

### 10. Alternative Ternary Kernel Approaches

#### 10.1 VPSIGNB Branchless Ternary Multiply (Single-Instruction)
- **Source:** https://en.algorithmica.org/hpc/simd/masking/
- **Key idea:** `VPSIGNB(activation, weight_as_int8)` with weights stored as {-1, 0, +1} in
  INT8 format directly computes the ternary multiplication in ONE instruction. It negates
  elements where sign operand is negative, passes through where positive, zeros where zero.
- **Kernel sketch:**
  ```c
  // weights: INT8 {-1, 0, +1}, activations: INT8
  __m256i result = _mm256_sign_epi8(activation, weight);
  // result[i] = activation[i] * sign(weight[i])
  // Then reduce with: _mm256_maddubs_epi16 -> _mm256_madd_epi16 -> hadd
  ```
- **Trade-off:** Processes 32 weights per iteration (YMM register of INT8 weights). The I2_S
  approach packs 128 weights per 32-byte load. VPSIGNB requires 4x more memory bandwidth
  (1 byte per weight vs 2 bits per weight). **NOT recommended for bandwidth-bound scenario.**
- **Use case:** Useful for L1/L2-resident data where compute is bottleneck, not bandwidth.

#### 10.2 XNOR+Popcount Ternary Approach (Dual Bitplane)
- **Source:** https://arxiv.org/abs/1603.05279 (XNOR-Net)
- **Key idea:** Store ternary weights as two bit planes: sign_plane + zero_mask. Dot product:
  `popcount(XNOR(activation_bits, sign_bits) AND mask_bits) * 2 - popcount(mask_bits)`
- **AVX2 popcount:** Use VPSHUFB-based popcount emulation (~30 bytes/cycle throughput)
- **Trade-off:** Requires binarized activations (lose precision). Not suitable for INT8
  activations in our pipeline. **Interesting for paper discussion but not for primary kernel.**

#### 10.3 R3-Engine Zero-Copy Ternary Inference
- **Source:** https://github.com/r3-engine/r3-engine
- **Key data:** 80-117 tok/s single-threaded on Ryzen 9950X3D for BitNet-b1.58-2B.
  Uses VPOPCNTDQ (AVX-512) for branchless ternary matmul. mmap ping-pong buffer with
  zero heap allocations.
- **Critical limitation:** Requires AVX-512 VPOPCNTDQ, fused off on Alder Lake.
- **Applicable ideas:** The mmap ping-pong buffer architecture and zero-allocation design
  pattern are applicable regardless of ISA. The ~9ms per forward pass on 2.4B model
  validates that 100+ tok/s is physically achievable on fast CPUs.

#### 10.4 ik_llama.cpp AVX-VNNI Optimizations
- **Source:** https://github.com/ikawrakow/ik_llama.cpp
- **Key data:** IQ1_KT, IQ2_KT, IQ3_KT, IQ4_KT trellis-based quantization formats.
  AVX-VNNI-specific optimizations (PRs 1446, 1455, 1467). Row-interleaved packing.
  First-class BitNet-b1.58-2B-4T support.
- **Relevance to EdgeLM:** The AVX-VNNI optimizations are directly relevant to our i7-12700H
  which supports AVX-VNNI on both P-cores and E-cores. Worth studying the IQ1_KT
  implementation for insights on weight layout optimization for VPDPBUSD instruction.

#### 10.5 LUT-GEMM Binary-Coding Quantization (Naver)
- **Source:** https://arxiv.org/abs/2206.09557, https://github.com/naver-aics/lut-gemm
- **Key idea:** BCQ represents weights as binary codes with learned scaling factors.
  LUT eliminates dequantization entirely. Targets GPU (CUDA 80+).
- **Applicability:** Principles apply to CPU; the concept of never materializing FP weights
  aligns with our integer-only inner loop design.

### 11. BLIS Panel-Major Packing for Ternary Weights

#### 11.1 The Goto/BLIS 5-Loop Algorithm
- **Source:** https://github.com/flame/blis/blob/master/docs/KernelsHowTo.md
- **Reference:** Goto & van de Geijn, "Anatomy of High-Performance Matrix Multiplication," 2008
- **Key concept:** Five nested loops with three cache blocking levels. A panel (MC x KC)
  packed into column-panel format fits in L2. B panel (KC x NC) in row-panel format fits in L3.
  Microkernel computes MR x NR output tile using rank-1 updates.
- **Why panel format matters:** Standard row-major requires strided loads for column access.
  Panel format guarantees purely sequential access in the microkernel, maximizing cache utilization.
- **Packing overhead:** 3-8% of GEMM time for runtime packing, but **ZERO for our case**
  since we pre-pack at model load time (weights are constant).

#### 11.2 Concrete Tile Parameters for i7-12700H
- **Haswell BLIS (reference, closest AVX2 config):** MR=6, NR=16, MC=168, NC=4080, KC=256
- **Proposed EdgeLM P-core microkernel:**
  ```
  MR = 128 (ternary weights per register load = one 32-byte packed block)
  NR = 4-8 (output columns processed simultaneously)
  KC = 512 (inner dimension slice)
  MC = 4096 (full hidden dimension when possible)

  Register budget: 8-10 accumulators + 2 weight + 1 activation = 11-13 of 16 YMM
  Weight panel: 4096 * 512 / 4 = 512 KB (fits 1.25 MB L2)
  L1 tile: 512B (activation) + 512B (accumulator) + 16KB (weight) = ~17 KB (fits 48 KB L1d)
  ```
- **Proposed EdgeLM E-core microkernel:**
  ```
  MR = 128, NR = 2, KC = 256
  Weight panel: 4096 * 256 / 4 = 256 KB (fits 500 KB effective L2)
  L1 tile: 256B + 512B + 8KB = ~9 KB (fits 32 KB L1d)
  ```
- **Memory access reduction:** `2 * MR * NR / (MR + NR)` = for MR=128, NR=8: 2*1024/136 = 15x
  fewer memory accesses than naive.

#### 11.3 Panel-Major Repacking Algorithm
```c
// Repack from row-major TQ2_0 blocks to panel-major layout
// Each MC x KC panel has MR-wide columns stored contiguously
// MR=128 ternary values = 32 bytes per column
// Micropanel: 32 * KC bytes = 16 KB for KC=512
// Panel: (MC/128) * 16 KB = 512 KB for MC=4096

void repack_ternary_to_panel_major(
    const block_tq2_0* src, uint8_t* dst,
    int M, int K, int MC, int KC, int MR)
{
    for (int kc = 0; kc < K; kc += KC) {
        int kc_eff = min(KC, K - kc);
        for (int mc = 0; mc < M; mc += MC) {
            int mc_eff = min(MC, M - mc);
            for (int mr = 0; mr < mc_eff; mr += MR) {
                int mr_eff = min(MR, mc_eff - mr);
                for (int col = 0; col < kc_eff; col++) {
                    pack_ternary_column(src, dst, mc + mr, kc + col,
                                        mr_eff, M, K);
                    dst += (mr_eff + 3) / 4;
                }
                dst = align_up(dst, 64);  // Pad to cache line
            }
        }
    }
}
```

#### 11.4 Zero-Padding for Boundary Blocks
- **Source:** Salykova's matmul-cpu implementation (https://salykova.github.io/matmul-cpu)
- **Key insight:** Copy undersized boundary blocks into zero-padded buffers rather than adding
  masking logic to the kernel. For ternary weights, zero = "skip" (add nothing), so padding
  with zeros naturally produces correct results. Trades memory for branch elimination.

### 12. Hardware Prefetcher Behavior on Golden Cove vs Gracemont

#### 12.1 Golden Cove (P-core) Prefetchers
- L1 DCU prefetcher: detects sequential access, prefetches next cache line
- L1 IP prefetcher: tracks per-instruction access patterns
- L2 HW prefetcher: detects strides, prefetches into L2
- L2 adjacent line prefetcher: fetches both cache lines in a 128-byte pair
- Enhanced in Golden Cove: better stride detection, deeper prefetch depth
- **Verdict:** Hardware handles sequential weight streaming well. Software prefetch needed
  only for layer transitions and KV cache access.

#### 12.2 Gracemont (E-core) Prefetchers
- Weaker than Golden Cove, fewer tracking entries
- AVX2 executes at 128-bit width (2 cycles per 256-bit op)
- **Verdict:** Add explicit software prefetch for ALL access patterns on E-cores.
  Prefetch distance: 4-8 cache lines ahead (256-512 bytes).

#### 12.3 Store Elimination for Zero-Sparse Ternary Weights
- **Source:** https://travisdowns.github.io/blog/2020/05/18/icelake-zero-opt.html
- 96% of redundant zero stores are silently eliminated by the cache controller.
- Ternary weights have ~30-40% zeros. Zero-initialized accumulator buffers are nearly free.
- 256-bit stores outperform 512-bit stores when writing zeros to L2 region (45% faster).

## Techniques Comparison Matrix

| Technique | Source | Impact | Complexity | Already in Deep Dive? |
|-----------|--------|--------|------------|----------------------|
| I2_S 2-bit packing (128-element blocks) | BitNet.cpp | P0 - defines format | Low | Partially (mentioned, not detailed) |
| TQ2_0 -> aligned repack (66B -> 64B) | GGML/llama.cpp | P0 - required | Low | No |
| 2MB large pages (VirtualAlloc) | MS docs, rigtorp | P0 - 23% speedup | Medium | Yes (mentioned) |
| Separate scale array from weights | Analysis | P1 - cleaner streaming | Low | No |
| BLIS panel-major pre-packing | Goto/BLIS | P0 - cache optimal | Medium | No |
| P-core tiles: MR=128, KC=512 | Analysis | P0 - fits 48KB L1 | Low | No |
| E-core tiles: MR=128, KC=256 | Analysis | P1 - fits 32KB L1 | Low | No |
| T-MAC VPSHUFB LUT kernel | T-MAC/EuroSys 2025 | P1 - benchmark vs MAD | High | Partially (LUT concept) |
| VPSIGNB single-instruction ternary | Algorithmica | P2 - L1-resident only | Low | No |
| XNOR+popcount dual bitplane | XNOR-Net | P3 - needs binary acts | Medium | No |
| R3-engine ping-pong buffers | r3-engine | P2 - architecture pattern | Medium | No |
| ik_llama.cpp AVX-VNNI kernels | ik_llama.cpp | P1 - study for VPDPBUSD | Medium | No |
| Layer-ahead prefetch (_MM_HINT_T2) | Analysis | P1 - hide DRAM latency | Low | Partially |
| Zero-padding boundary blocks | Salykova | P1 - eliminate branches | Low | No |
| Outer-loop unrolling (weight reuse) | llamafile/tinyBLAS | P0 - 1.6-2x speedup | Medium | No |
| .edgelm disk cache for repacked weights | Analysis | P2 - faster startup | Medium | Partially |
| Store elimination for zero-sparse | Travis Downs | P2 - free optimization | None | No |
| DRAM refresh bandwidth budget (40 GB/s) | Cloudflare | P0 - correct baseline | None | Partially |

## Recommendations for EdgeLM

Ordered by impact-to-effort ratio:

1. **Use I2_S-style 2-bit packing (128-element blocks, 32 bytes per block)** with separate
   scale array. Repack from TQ2_0 GGUF at load time. This is the foundation -- everything
   else builds on this format. (P0, Low effort)

2. **Allocate weight buffer with VirtualAlloc + MEM_LARGE_PAGES (2MB pages).** 23% speedup
   from TLB miss elimination. One-time setup cost. Do NOT use mmap for the compute buffer.
   (P0, Low effort)

3. **Implement BLIS-style panel-major layout for weight matrices.** Pre-pack at load time
   (zero runtime overhead). Use MR=128, KC=512 for P-cores, MR=128, KC=256 for E-cores.
   Weight panel fits in L2 (512 KB for P-core). (P0, Medium effort)

4. **Apply outer-loop unrolling pattern from llamafile:** load weight block once, apply to
   4-8 different activation elements before loading next weights. This maximizes register
   reuse and is the single biggest kernel-level optimization. (P0, Medium effort)

5. **Use `_mm256_maddubs_epi16` as the core accumulation instruction.** This processes 32
   ternary-times-INT8 multiplications in a single instruction. Combined with shift-based
   2-bit extraction from I2_S format, this is the reference kernel path. (P0, Low effort)

6. **Benchmark T-MAC VPSHUFB LUT against direct MAD kernel.** T-MAC claims 4-5x over
   llama.cpp, but on x86 with AVX2, direct MAD may win for ternary. Implement both,
   profile, keep the winner. (P1, High effort)

7. **Add layer-ahead prefetch with _MM_HINT_T2.** Warm next layer's weight data into L3
   while computing current layer. Low effort, measurable latency hiding. (P1, Low effort)

8. **Study ik_llama.cpp AVX-VNNI kernels** for VPDPBUSD-based ternary accumulation.
   AVX-VNNI is available on both P-cores and E-cores of our i7-12700H. May offer better
   throughput than pure AVX2 for the accumulation step. (P1, Medium effort)

9. **Implement .edgelm disk cache** for repacked weights. Save ~50ms per session after
   first load. Low priority but easy to implement. (P2, Low effort)

10. **Zero-pad boundary blocks** instead of adding masking logic to the kernel. Ternary
    zero = no-op, so padding is functionally correct. Eliminates branches in hot path.
    (P1, Low effort)

## References

1. BitNet.cpp: https://github.com/microsoft/BitNet
2. BitNet paper: https://arxiv.org/abs/2410.16144
3. BitNet b1.58: https://arxiv.org/abs/2402.17764
4. T-MAC: https://github.com/microsoft/T-MAC
5. T-MAC paper: https://arxiv.org/abs/2407.00088
6. llamafile/tinyBLAS: https://justine.lol/matmul/
7. HuggingFace 1.58-bit: https://huggingface.co/blog/1_58_llm_extreme_quantization
8. llama.cpp TQ formats: https://github.com/ggml-org/llama.cpp/pull/8151
9. GGML common.h: https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-common.h
10. GGML quants.c: https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-quants.c
11. Windows Large Pages: https://learn.microsoft.com/en-us/windows/win32/memory/large-page-support
12. rigtorp Huge Pages: https://rigtorp.se/hugepages/
13. DRAM refresh: https://blog.cloudflare.com/every-7-8us-your-computers-memory-has-a-hiccup
14. llama.cpp mmap: https://github.com/ggml-org/llama.cpp/blob/master/src/llama-mmap.cpp
15. BLIS framework: https://github.com/flame/blis/blob/master/docs/KernelsHowTo.md
16. Goto & van de Geijn, "Anatomy of High-Performance Matrix Multiplication," 2008
17. BLIS: Van Zee & van de Geijn, ACM TOMS 2015
18. Algorithmica SIMD: https://en.algorithmica.org/hpc/simd/shuffling/
19. Algorithmica matmul: https://en.algorithmica.org/hpc/algorithms/matmul/
20. Salykova matmul-cpu: https://salykova.github.io/matmul-cpu
21. Travis Downs zero-opt: https://travisdowns.github.io/blog/2020/05/18/icelake-zero-opt.html
22. R3-engine: https://github.com/r3-engine/r3-engine
23. ik_llama.cpp: https://github.com/ikawrakow/ik_llama.cpp
24. XNOR-Net: https://arxiv.org/abs/1603.05279
25. LUT-GEMM: https://arxiv.org/abs/2206.09557
26. oneDNN inner product: https://uxlfoundation.github.io/oneDNN/dev_guide_inner_product.html
27. Intel Extension for PyTorch: https://github.com/intel/intel-extension-for-pytorch
28. BitNet a4.8: https://arxiv.org/abs/2411.04965
29. Alder Lake: https://en.wikipedia.org/wiki/Alder_Lake_(microprocessor)
30. CoffeeBeforeArch blocked matmul: https://coffeebeforearch.github.io/2020/06/23/mmul.html
31. HackerNews BitNet discussion: https://news.ycombinator.com/item?id=41490196
32. Z-order curves: https://en.wikipedia.org/wiki/Z-order_curve
33. Morton code SIMD: https://lemire.me/blog/2018/01/09/how-fast-can-you-bit-interleave-32-bit-integers-simd-edition/

| Finding | Source | Impact | Priority |
|---------|--------|--------|----------|
| I2_S format: 2-bit, 128-element blocks, AVX2 shift extraction | BitNet.cpp | Defines our packing format | P0 |
| TQ2_0 66-byte blocks need repack to 64-byte aligned | GGML common.h | Required for aligned AVX2 loads | P0 |
| 2MB large pages: 93% -> 0.07% TLB miss rate, 23% speedup | rigtorp.se | Free performance, one-time setup | P0 |
| VirtualAlloc + MEM_LARGE_PAGES, not mmap, for weight buffer | MS docs + analysis | Avoids 153K page faults for 0.6 GB | P0 |
| Separate scale array from packed weights for aligned streaming | Analysis of TQ2_0 layout | Enables pure 64-byte aligned weight reads | P1 |
| Layer-ahead prefetch with _MM_HINT_T2 for next-layer warming | Prefetch analysis | Hides DRAM latency at layer transitions | P1 |
| Cache repacked weights to `.edgelm` file, save ~50ms on reload | llamafile/llama.cpp patterns | Reduces startup from ~170ms to ~120ms | P2 |
| T-MAC 48 tok/s on 4 ARM cores for 3B ternary = our lower bound | T-MAC benchmarks | Validates 100+ tok/s on 14 x86 threads | P1 |
| Outer-loop unrolling (3x4 tiles) beats MKL by 2x | llamafile/tinyBLAS | Apply to ternary kernel tiling strategy | P1 |
| BitNet.cpp ROW_BLOCK=4, COL_BLOCK=128, PARALLEL=4 on x86 | gemm-config.h | Starting point for our tile tuning | P1 |
| _mm256_maddubs_epi16 for ternary accumulation pattern | BitNet.cpp MAD kernel | Core AVX2 instruction for our matmul | P0 |
| Budget 40 GB/s real bandwidth (not 51.2 theoretical) | DRAM refresh analysis | Correct baseline for tok/s calculations | P0 |

## Audit Addendum (2026-04-02)

- **Packing policy should encode intended consumer kernel family.** A "universal"
  packed format sounds attractive, but a decode-optimized pack for one kernel
  family may actively harm another.
- **Cache invalidation and benchmarking must stay linked.** Every repack layout
  change should force:
  - cache version bumps,
  - benchmark metadata updates,
  - and fresh ablation runs.
- **Packing should preserve room for future hybrid execution.** Even if the iGPU
  path starts later, the runtime metadata should avoid assuming CPU-only
  consumers forever.
