# Section 05: GGUF Format & Model Loading

## Overview
This section covers the GGUF binary file format, model loading strategies, and weight preparation for EdgeLM. Model loading is the critical path from disk to first token -- for a ~0.5 GB ternary model on PCIe 4.0 NVMe, the theoretical floor is ~90ms. The goal is to parse GGUF efficiently, mmap or stream weights into SIMD-aligned buffers, and repack into a compute-optimal layout with minimal startup latency.

## What the Deep Dive Already Covers
- GGUF as the target input format (read GGUF, repack to custom runtime format)
- mmap-based loading with weight repacking at load time
- Caching repacked weights to disk for fast subsequent loads
- ~300 lines of C for a minimal GGUF parser
- Phase 1 tasks: GGUF parser, memory allocator, weight loader
- TQ2_0 as primary ternary format (2.0 bpw, 141.83 GB/s on AVX2)
- TQ1_0 as alternative (1.6875 bpw, 70.31 GB/s -- 2x slower due to base-3 decode)
- I2_S as bitnet.cpp's native format (functionally similar to TQ2_0)

## New Findings

### 1. GGUF Binary Format Specification (Complete)

#### 1.1 Header Structure & File Layout
- **Source:** https://github.com/ggerganov/ggml/blob/master/docs/gguf.md, https://huggingface.co/docs/hub/en/gguf
- **Key idea:** GGUF is a sequential binary format -- magic, version, counts, metadata KVs, tensor infos, padding, then raw tensor data.
- **Relevance to EdgeLM:** Parsing is strictly sequential (no random access needed), making a minimal C parser straightforward.
- **Implementation complexity:** Low
- **Details:**

```c
// Fixed header: 24 bytes
struct gguf_header_t {
    uint32_t magic;              // 0x47475546 ("GGUF")
    uint32_t version;            // Currently 3
    uint64_t tensor_count;
    uint64_t metadata_kv_count;
    // Followed by: metadata_kv[metadata_kv_count] (variable length)
};

// File layout (sequential):
// 1. Header (24 bytes + variable metadata)
// 2. Tensor info array (one per tensor)
// 3. Padding (0x00 bytes to alignment boundary)
// 4. Tensor data (raw weight bytes, each offset alignment-aligned)
```

Strings are `uint64_t length` + `char data[length]` (NOT null-terminated). Tensor info:
```c
struct gguf_tensor_info_t {
    gguf_string_t name;           // max 64 bytes content
    uint32_t      n_dimensions;   // max 4
    uint64_t      dimensions[n_dimensions];
    uint32_t      type;           // ggml_type enum
    uint64_t      offset;         // relative to tensor data section start
};
```

#### 1.2 Alignment Rules
- **Source:** https://github.com/ggerganov/ggml/blob/master/docs/gguf.md
- **Key idea:** Default alignment is 32 bytes (from `general.alignment` metadata key). Must be power of 2, minimum multiple of 8. All tensor data offsets are multiples of alignment.
- **Relevance to EdgeLM:** For AVX2, we want 64-byte alignment. When writing our repacked cache format, use `general.alignment = 64`. When reading standard GGUF files, the 32-byte default is sufficient for AVX2 (which requires 32-byte alignment for `_mm256_load_*`), but 64-byte alignment avoids cache line splits.
- **Implementation complexity:** Low

#### 1.3 Metadata Value Types
- **Source:** https://github.com/ggerganov/ggml/blob/master/docs/gguf.md
- **Key idea:** 13 value types (UINT8 through FLOAT64, plus STRING, ARRAY, BOOL). Keys use `lower_snake_case` with `.` separators.
- **Relevance to EdgeLM:** We only need to parse a handful of metadata keys: `general.architecture`, `general.alignment`, tensor dimensions, layer count, head count, vocab size. Can skip unknown keys by reading their type and advancing the correct number of bytes.
- **Implementation complexity:** Low
- **Details:**

| ID | Type | Size |
|----|------|------|
| 0-5 | UINT8 through INT32 | 1-4 bytes |
| 6 | FLOAT32 | 4 bytes |
| 7 | BOOL | 1 byte |
| 8 | STRING | 8-byte length + data |
| 9 | ARRAY | type(4) + count(8) + elements |
| 10-12 | UINT64, INT64, FLOAT64 | 8 bytes each |

#### 1.4 Endianness & Version History
- **Source:** https://github.com/ggerganov/ggml/blob/master/docs/gguf.md
- **Key idea:** Little-endian by default. Version 3 (current since ~2023) added big-endian support but it's rarely used. Format has been stable -- only additive changes (new ggml_type enum values).
- **Relevance to EdgeLM:** Target V3 little-endian only. No need to handle big-endian or older versions for BitNet models.
- **Implementation complexity:** Low

### 2. Ternary Quantization Type Details

#### 2.1 Complete ggml_type Enum (Ternary-Relevant Subset)
- **Source:** https://github.com/ggml-org/llama.cpp/pull/8151, https://github.com/ggml-org/ggml/blob/master/src/ggml-common.h
- **Key idea:** TQ1_0 = type ID 34, TQ2_0 = type ID 35. Total enum count = 41 as of 2026.
- **Relevance to EdgeLM:** Our parser needs to recognize type IDs 34 and 35 for ternary tensors, plus F16 (ID 1) and Q4_K/Q6_K for embeddings and output layers.
- **Estimated impact:** Correctness requirement
- **Implementation complexity:** Low
- **Details:**

| ID | Name | Block Size | Bytes/Block | BPW |
|----|------|-----------|-------------|-----|
| 34 | TQ1_0 | 256 | 54 | 1.6875 |
| 35 | TQ2_0 | 256 | 66 | 2.0625 |

Recent additions (2025-2026): MXFP4 (ID 39), NVFP4 (ID 40) -- not relevant to EdgeLM but confirms format stability.

#### 2.2 TQ2_0 Block Structure (Primary Target)
- **Source:** https://github.com/ggml-org/ggml/blob/master/src/ggml-common.h
- **Key idea:** 256 weights per block. 64 bytes of packed 2-bit values (4 per byte) + 2 bytes FP16 scale = 66 bytes total.
- **Relevance to EdgeLM:** This is our primary input format. Decoding is trivial: `q = (qs[j] >> (l*2)) & 3; value = (q - 1) * d`.
- **Estimated impact:** 2x faster decode than TQ1_0 on AVX2
- **Implementation complexity:** Low
- **Details:**
```c
typedef struct {
    uint8_t qs[QK_K/4];  // 64 bytes (256/4)
    ggml_half d;          // 2 bytes FP16 scale
} block_tq2_0;            // 66 bytes per 256 weights
// Encoding: {-1→0, 0→1, +1→2}, 4 values per byte, 2 bits each
```

#### 2.3 TQ1_0 Block Structure (Compression Alternative)
- **Source:** https://github.com/ggml-org/ggml/blob/master/src/ggml-common.h
- **Key idea:** Base-3 packing: 5 trits per byte (3^5=243 < 256). 48 bytes for 240 weights + 4 bytes for remaining 16 + 2 bytes scale = 54 bytes per 256 weights.
- **Relevance to EdgeLM:** Only useful if RAM is critically constrained. Saves ~18% memory vs TQ2_0 but halves decode throughput. For a 3B model: TQ1_0 ≈ 0.52 GB vs TQ2_0 ≈ 0.63 GB. Both fit easily in budget.
- **Implementation complexity:** Medium (base-3 arithmetic with lookup tables)
- **Details:**
```c
typedef struct {
    uint8_t qs[(QK_K - 4*QK_K/64) / 5];  // 48 bytes (240 values, 5 per byte)
    uint8_t qh[QK_K/64];                   // 4 bytes (16 remaining values)
    ggml_half d;                            // 2 bytes FP16 scale
} block_tq1_0;                              // 54 bytes per 256 weights
// Decode: pow3[6] = {1,3,9,27,81,243}; q = qs[j]*pow3[n]; xi = ((uint16_t)q*3)>>8
```

#### 2.4 Embedding & Non-Ternary Tensor Handling
- **Source:** HuggingFace GGUF repo analysis, research/04-bitnet-ternary-quantization.md
- **Key idea:** BitNet 2B4T GGUF is 1.19 GB total, but only ~0.4 GB is ternary weights. The rest (~0.8 GB) is the FP16 embedding table (128K vocab × hidden_dim × 2 bytes). Token embeddings typically use Q4_K, output projection uses Q6_K.
- **Relevance to EdgeLM:** Bandwidth calculation should use 0.4 GB (ternary only), not 1.19 GB. Gives 40 GB/s / 0.4 GB = **100 tok/s theoretical**. Embedding lookup is sparse (single row per token, negligible bandwidth).
- **Estimated impact:** Correct performance ceiling calculation
- **Implementation complexity:** N/A (design insight)

### 3. GGUF Parsing Implementation Strategy

#### 3.1 Minimal Parser Architecture
- **Source:** https://github.com/ggml-org/llama.cpp/blob/master/src/llama-model-loader.cpp
- **Key idea:** llama.cpp's `gguf_init_from_file()` returns a `gguf_context` with all metadata and tensor info. Core flow: validate magic → read counts → iterate metadata KVs → iterate tensor infos → compute data section offset.
- **Relevance to EdgeLM:** We can implement an even simpler version since we target specific architectures (BitNet/Llama). Skip unknown metadata keys, only extract what we need.
- **Estimated impact:** ~200 lines of C for a complete parser
- **Implementation complexity:** Low
- **Details:**
```c
// Pseudocode for minimal GGUF parser
int edgelm_load_gguf(const char* path, edgelm_model* model) {
    // 1. Open + mmap file
    // 2. Read 24-byte fixed header, validate magic = 0x46554747, version >= 2
    // 3. Walk metadata_kv_count entries:
    //    - Read key string, value type, value
    //    - Extract: architecture, n_layers, n_heads, n_embd, vocab_size, alignment
    //    - Skip all other keys (advance by value size)
    // 4. Walk tensor_count entries:
    //    - Read name, n_dims, dims[], type, offset
    //    - Store in tensor lookup table (hash map by name)
    // 5. Compute data_offset = align(current_position, alignment)
    // 6. Each tensor's data at: mmap_base + data_offset + tensor.offset
    return 0;
}
```

#### 3.2 Tensor Name Conventions
- **Source:** llama.cpp model loader, GGUF spec
- **Key idea:** Tensor names follow patterns like `blk.{N}.attn_q.weight`, `blk.{N}.ffn_gate.weight`, `token_embd.weight`, `output.weight`. Names are used for mapping to model architecture.
- **Relevance to EdgeLM:** Build a simple string-match dispatcher rather than a full hash map. For BitNet 2B4T with ~26 layers, there are ~150-200 tensors total.
- **Implementation complexity:** Low

### 4. Memory-Mapped Loading (mmap)

#### 4.1 Windows mmap via CreateFileMapping + MapViewOfFile
- **Source:** Microsoft Learn documentation
- **Key idea:** `CreateFileMapping` creates a section object backed by a file; `MapViewOfFile` maps it into the process address space. Pages load on-demand via page faults. Multiple processes sharing the same mapping use the same physical pages via the OS page cache.
- **Relevance to EdgeLM:** This is the primary loading mechanism. After first access, pages stay in the OS page cache -- subsequent process starts see ~1-5ms warm start (virtual address setup only, no physical I/O).
- **Estimated impact:** Warm start: 1-5ms; Cold start: ~90ms (page faults on first access)
- **Implementation complexity:** Low
- **Details:**
```c
// Windows mmap pseudocode
HANDLE hFile = CreateFileA(path, GENERIC_READ, FILE_SHARE_READ, NULL,
                           OPEN_EXISTING, FILE_FLAG_SEQUENTIAL_SCAN, NULL);
HANDLE hMapping = CreateFileMappingA(hFile, NULL, PAGE_READONLY, 0, 0, NULL);
void* base = MapViewOfFile(hMapping, FILE_MAP_READ, 0, 0, 0);
// Access tensor data: (uint8_t*)base + data_offset + tensor.offset
```

#### 4.2 mmap vs Direct Read Tradeoffs
- **Source:** https://martinuke0.github.io/posts/2025-12-06-memory-mapped-files-mmap.../, ServerlessLLM (OSDI 2024)
- **Key idea:** mmap has overhead from page faults (each fault = kernel trap + page table update). For initial cold load, direct `ReadFile` into pre-allocated aligned buffers can be faster by avoiding per-page fault overhead. But mmap wins for warm starts (pages already resident).
- **Relevance to EdgeLM:** Hybrid approach: use mmap for the common case (warm start). For known cold start scenarios, optionally use `ReadFile` with `FILE_FLAG_NO_BUFFERING | FILE_FLAG_SEQUENTIAL_SCAN` into a large-page buffer.
- **Estimated impact:** Cold start: direct read ~91ms vs mmap ~100-150ms (page fault overhead); Warm start: mmap ~1-5ms vs direct read ~91ms
- **Implementation complexity:** Medium (two code paths)

#### 4.3 PrefetchVirtualMemory for Proactive Page Loading
- **Source:** Microsoft Learn documentation, community reports
- **Key idea:** Windows API `PrefetchVirtualMemory()` hints the OS to preload specific address ranges into physical memory. Can be called from a background thread immediately after mmap setup.
- **Relevance to EdgeLM:** Call after `MapViewOfFile` to start loading pages while the main thread parses metadata. Reported as unreliable in some cases, but worth using as a hint.
- **Estimated impact:** May reduce cold start by 20-40ms by overlapping page loading with initialization
- **Implementation complexity:** Low

#### 4.4 Large Pages with SEC_LARGE_PAGES
- **Source:** Microsoft Learn documentation
- **Key idea:** `CreateFileMapping` with `SEC_LARGE_PAGES` flag uses 2MB pages, reducing TLB pressure. Requires `SeLockMemoryPrivilege` (must be granted to user account).
- **Relevance to EdgeLM:** For a 0.5 GB model, standard 4KB pages = 131,072 TLB entries vs 2MB pages = 256 entries. During inference, TLB misses on weight access can cost 10-50 cycles each. Large pages eliminate this.
- **Estimated impact:** 5-15% inference throughput improvement from reduced TLB misses
- **Implementation complexity:** Low (but requires privilege setup)

### 5. Fast Cold Start Architecture

#### 5.1 Sub-Second Cold Start is Mathematically Achievable
- **Source:** Analysis based on PCIe 4.0 NVMe specs
- **Key idea:** 0.5 GB model on 5.5 GB/s NVMe = 91ms raw read. With metadata parsing (~1ms), buffer allocation (~1ms), and weight repacking (~10-50ms), total cold start of 100-200ms is achievable.
- **Relevance to EdgeLM:** This is a major selling point for the paper. Most inference engines (llama.cpp, vLLM) report multi-second startup times for much larger models. Our ternary model's small size enables qualitatively different startup behavior.
- **Estimated impact:** 100-200ms cold start, 1-10ms warm start
- **Implementation complexity:** Medium

#### 5.2 ServerlessLLM Loading-Optimized Format
- **Source:** https://arxiv.org/html/2401.14351v2 (OSDI 2024)
- **Key idea:** Custom checkpoint format with tensors grouped by destination, aligned to word sizes, with a separate index file. Uses O_DIRECT + pread() to bypass OS page cache. Achieves 3.6-8.2x faster loading than Safetensors, 90%+ bandwidth saturation.
- **Relevance to EdgeLM:** Directly applicable design pattern. Our custom repacked cache format should follow the same principles: minimal header, 64-byte-aligned tensor data, tensors ordered in inference-execution order (layer 0 first, then layer 1, etc.) for sequential read.
- **Estimated impact:** Near-theoretical bandwidth utilization on cold start
- **Implementation complexity:** Medium

#### 5.3 Overlap I/O with Initialization
- **Source:** General systems engineering, NVIDIA Model Streamer design
- **Key idea:** While model bytes stream from NVMe, concurrently: allocate KV cache buffers, initialize tokenizer, set up thread pool, configure iGPU. These tasks are CPU-bound and don't compete with NVMe bandwidth.
- **Relevance to EdgeLM:** With 91ms of I/O time, there's ample opportunity to complete all initialization work in parallel. Use Windows overlapped I/O or a background mmap prefault thread.
- **Estimated impact:** Eliminates non-I/O startup latency entirely
- **Implementation complexity:** Low-Medium

### 6. Weight Repacking Strategy

#### 6.1 Two-Phase Loading: Parse GGUF → Repack → Cache
- **Source:** implementation-plan.md, bitnet.cpp design patterns
- **Key idea:** First run: parse GGUF, repack TQ2_0 blocks into SIMD-optimal layout, write repacked format to disk cache. Subsequent runs: load repacked format directly (skip GGUF parsing entirely).
- **Relevance to EdgeLM:** The repacked format is custom to our AVX2 kernels. GGUF is the interchange format; our cache format is the runtime format.
- **Estimated impact:** Eliminates repacking cost on subsequent runs (saves ~10-50ms)
- **Implementation complexity:** Medium
- **Details:**
  - Repacked cache file: magic + version + tensor_count + tensor_index + aligned_tensor_data
  - Tensor index: array of {name_hash, offset, size, type, dims}
  - Validate cache: check magic + file size + GGUF file mtime/size hash

#### 6.2 VNNI-Interleaved Packing for AVX-VNNI
- **Source:** research/04-bitnet-ternary-quantization.md (Section 8)
- **Key idea:** For VPDPBUSD throughput, store weights as separate pos_mask and neg_mask byte arrays where pos_mask[i] = (W[i] == +1) as 0x01, neg_mask[i] = (W[i] == -1) as 0x01. ~2 bytes/weight but zero decode overhead.
- **Relevance to EdgeLM:** This is the ideal runtime format. Store on disk in TQ2_0 (compact), unpack to VNNI-interleaved at load time (one-time cost).
- **Estimated impact:** Zero decode overhead in inner loop (decode cost moved to load time)
- **Implementation complexity:** Medium

#### 6.3 Inference-Order Tensor Layout
- **Source:** ServerlessLLM (OSDI 2024)
- **Key idea:** Order tensors in the file by inference execution order: embedding → layer 0 (attn_q, attn_k, attn_v, attn_output, ffn_gate, ffn_up, ffn_down, norm) → layer 1 → ... → output_norm → output. This enables streaming/prefetching.
- **Relevance to EdgeLM:** When layers execute sequentially, the next layer's weights are always the next bytes in the file. The OS prefetcher (or explicit prefetch) can stay ahead of execution.
- **Estimated impact:** Improved spatial locality, better hardware prefetcher behavior
- **Implementation complexity:** Low (just a reordering at repack time)

### 7. Compression Analysis

#### 7.1 LZ4/Zstd Compression is Counterproductive
- **Source:** https://lz4.org, https://facebook.github.io/zstd/, CERN/FNAL benchmarks
- **Key idea:** LZ4 decompresses at ~3.85 GB/s, Zstd at ~1.6 GB/s. On PCIe 4.0 NVMe (5.5 GB/s), reading 0.5 GB uncompressed (91ms) is faster than reading 0.24 GB compressed + decompressing (44ms + 130ms = 174ms).
- **Relevance to EdgeLM:** Do NOT compress the model file. NVMe bandwidth exceeds single-core decompression throughput for this file size. Exception: if distributing models over network (slower bandwidth), compression helps.
- **Estimated impact:** Negative (~2x slower if compressed)
- **Implementation complexity:** N/A (recommendation: don't do it)

#### 7.2 Exception: Ternary Weights May Compress Exceptionally Well
- **Source:** Analysis of ternary weight statistics
- **Key idea:** Ternary weights {-1, 0, +1} with ~60-70% zeros could achieve 4-8x compression ratios with LZ4 (vs typical 2.1x). At 8x: 0.5 GB → 0.0625 GB on disk, read in 11ms, decompress in 16ms = 27ms total. This WOULD beat uncompressed.
- **Relevance to EdgeLM:** Worth benchmarking actual compression ratios on BitNet weights. If >4x is achievable, a compressed format with multi-threaded LZ4 decode could be faster than uncompressed read.
- **Estimated impact:** Potentially 2-3x faster cold start IF compression ratio is high enough
- **Implementation complexity:** Medium (need to benchmark, add LZ4 dependency or implement minimal decoder)

### 8. Model Integrity & Validation

#### 8.1 GGUF Has No Built-in Checksums
- **Source:** https://malcolm-mill.github.io/LLM/gguf-file-structure-guide/
- **Key idea:** GGUF only validates via magic bytes ("GGUF") and version number. No per-tensor hash, no file-level CRC.
- **Relevance to EdgeLM:** We need our own validation for the repacked cache format to detect corruption or version mismatch.
- **Implementation complexity:** Low

#### 8.2 Tiered Validation Strategy
- **Source:** xxhash.com, general systems engineering
- **Key idea:** Tier 1 (microseconds): file size + header magic + stored mtime hash. Tier 2 (50ms): full xxHash-64 of file (xxHash runs at ~10 GB/s). Tier 3 (background): per-tensor hash validation after first token.
- **Relevance to EdgeLM:** For sub-second startup, use Tier 1 only. Run Tier 2 in background after inference starts. Store a sidecar `.hash` file alongside the repacked cache.
- **Estimated impact:** <0.1ms validation overhead on startup path
- **Implementation complexity:** Low

### 9. Process-Persistent Model Caching

#### 9.1 OS Page Cache as Implicit Warm Cache
- **Source:** https://martinuke0.github.io/posts/2025-12-06-memory-mapped-files-mmap.../
- **Key idea:** When using mmap, the OS page cache retains model pages after the process exits. Next process start finds pages already resident -- load time drops to virtual address setup only (~1-5ms).
- **Relevance to EdgeLM:** This is the default behavior with mmap on Windows. No extra code needed. The 0.5 GB model is small enough to remain cached alongside normal system usage (16 GB total RAM, ~6-7 GB budget).
- **Estimated impact:** 1-5ms warm start vs 91ms cold start
- **Implementation complexity:** Zero (automatic with mmap)

#### 9.2 Named File Mapping for Guaranteed Persistence
- **Source:** Microsoft Learn documentation
- **Key idea:** `CreateFileMapping` with a name (e.g., `"Local\\EdgeLM_Model_Cache"`) creates a named section object. A lightweight daemon can hold the handle open, guaranteeing the mapping survives process restarts and pages aren't evicted.
- **Relevance to EdgeLM:** Optional optimization for demo/benchmark scenarios. For normal use, OS page cache is sufficient.
- **Estimated impact:** Guarantees warm start behavior
- **Implementation complexity:** Low

#### 9.3 Shared Memory for Multi-Instance
- **Source:** vLLM shared memory IPC cache, Maru project (https://github.com/xcena-dev/maru)
- **Key idea:** Multiple inference processes can share the same physical model pages via named file mappings. Useful if running multiple EdgeLM instances (e.g., different prompt contexts).
- **Relevance to EdgeLM:** Low priority for initial implementation, but architecturally clean if using named mappings from the start.
- **Implementation complexity:** Low (falls out of named mapping design)

### 10. Recent Papers & Projects (2024-2026)

#### 10.1 ServerlessLLM (OSDI 2024)
- **Source:** https://arxiv.org/html/2401.14351v2
- **Key idea:** Loading-optimized checkpoint format with O_DIRECT, achieving 3.6-8.2x faster loading than Safetensors. Tensors grouped by destination, aligned to word sizes, separate index.
- **Relevance to EdgeLM:** Gold standard for custom format design. Directly applicable principles.
- **Estimated impact:** Near-theoretical bandwidth saturation
- **Implementation complexity:** Medium

#### 10.2 fastsafetensors (IBM, IEEE CLOUD 2025)
- **Source:** https://arxiv.org/html/2505.23072v1
- **Key idea:** Parallel tensor deserialization avoiding mmap. 4.8-7.5x speedup. Key finding: "model loading accounted for an average of 92% of startup latency."
- **Relevance to EdgeLM:** Confirms that loading optimization is the highest-leverage startup improvement. Our ternary model's small size makes this even more dominant.
- **Implementation complexity:** N/A (validates approach)

#### 10.3 FlowLoader (Springer 2025)
- **Source:** https://link.springer.com/article/10.1007/s11227-025-07646-4
- **Key idea:** Multi-tier local caching + parallel communication pipeline for LLM cold-start reduction.
- **Relevance to EdgeLM:** Multi-tier caching concept (NVMe → page cache → mapped memory) aligns with our approach.
- **Implementation complexity:** N/A (architectural pattern)

#### 10.4 DeepNVMe / FastPersist (DeepSpeed 2024)
- **Source:** https://pytorch.org/blog/deepnvme-affordable-i-o-scaling-for-deep-learning-applications/
- **Key idea:** 20x+ checkpoint I/O speedup using Linux AIO, NVMe queue depth optimization, RAID-0. Achieves near-hardware bandwidth on reads.
- **Relevance to EdgeLM:** The NVMe queue depth insight applies: even on Windows, using overlapped I/O with multiple outstanding read requests can better utilize NVMe parallelism.
- **Implementation complexity:** Medium

#### 10.5 ENDOR Sparse Weight Format (2024)
- **Source:** https://arxiv.org/html/2406.11674v1
- **Key idea:** Hardware-friendly sparse format using bitmaps for non-zero element positions. Designed for low decompression overhead with pruned weights.
- **Relevance to EdgeLM:** Ternary weights with ~60-70% zeros are inherently sparse. A bitmap-based format (zero_mask + nonzero_values) could be more efficient than TQ2_0 for storage, though TQ2_0's 2-bit encoding is already near-optimal for ternary.
- **Estimated impact:** Minimal improvement over TQ2_0 for true ternary weights
- **Implementation complexity:** Medium

#### 10.6 Persistent KV Cache for Edge Devices (2026)
- **Source:** https://smallaimodel.substack.com/p/2026-02-17-persistent-kv-cache-multi
- **Key idea:** Disk-persisting KV cache state enables resuming conversations without recomputation. Critical for edge devices with limited memory.
- **Relevance to EdgeLM:** Directly applicable. Save KV cache to NVMe between sessions. For a 2K context with FP8 KV: ~50-100 MB, readable in ~10-20ms from NVMe.
- **Estimated impact:** Eliminates prompt re-processing on session resume
- **Implementation complexity:** Medium

## Techniques Comparison Matrix

| Technique | Source | Impact | Complexity | Already in Deep Dive? |
|-----------|--------|--------|------------|----------------------|
| TQ2_0 as primary ternary format | ggml PR #8151 | Critical | Low | Yes |
| Minimal GGUF parser (~200 LOC) | llama.cpp source | Required | Low | Yes |
| mmap via CreateFileMapping | Microsoft docs | High | Low | Partially |
| PrefetchVirtualMemory hint | Microsoft docs | Low-Medium | Low | No |
| SEC_LARGE_PAGES for mmap | Microsoft docs | Medium (5-15%) | Low | No |
| Custom repacked cache format | ServerlessLLM | High | Medium | Partially |
| VNNI-interleaved runtime packing | Prior research | High | Medium | Yes |
| Inference-order tensor layout | ServerlessLLM | Medium | Low | No |
| FILE_FLAG_NO_BUFFERING cold start | Systems engineering | Medium | Medium | No |
| Tiered validation (xxHash) | xxhash.com | Low | Low | No |
| OS page cache warm start | mmap inherent | High (1-5ms) | Zero | No |
| Named file mapping persistence | Microsoft docs | Low-Medium | Low | No |
| Overlap I/O with init work | General | Medium | Low-Medium | No |
| Compression (LZ4) for ternary | lz4.org | Negative to Medium* | Medium | No |
| Persistent KV cache to NVMe | 2026 paper | Medium | Medium | No |
| O_DIRECT + aligned buffers | ServerlessLLM | Medium | Medium | No |

*Depends on actual compression ratio of ternary weights.

## Recommendations for EdgeLM

Ranked by impact-to-effort ratio:

1. **Implement minimal GGUF parser** (~200 lines C). Parse header, extract metadata for BitNet/Llama architectures, build tensor offset table. Skip unknown metadata. This is the Phase 1 foundation.

2. **Use mmap (CreateFileMapping + MapViewOfFile) as primary loader.** This gives automatic warm-start caching via OS page cache (1-5ms subsequent starts) with zero extra code. Use `FILE_FLAG_SEQUENTIAL_SCAN` hint on the file handle.

3. **Design a custom repacked cache format** following ServerlessLLM principles: minimal header, 64-byte-aligned tensors in inference execution order, VNNI-interleaved packing. First run parses GGUF + repacks + writes cache. Subsequent runs load cache directly.

4. **Add PrefetchVirtualMemory call** immediately after MapViewOfFile to start background page loading while the main thread initializes tokenizer and allocates KV cache buffers.

5. **Use SEC_LARGE_PAGES for model weight mapping** to reduce TLB pressure during inference (5-15% throughput gain). Requires one-time privilege setup (`SeLockMemoryPrivilege`).

6. **Implement tiered validation**: Tier 1 on critical path (magic + file size, <0.1ms), Tier 2 in background (xxHash-64 of full file, ~50ms, after first token).

7. **Benchmark LZ4 compression on actual ternary weights.** If compression ratio exceeds 4x, a compressed cache format with multi-threaded LZ4 decode could beat uncompressed NVMe reads.

8. **Consider persistent KV cache** (later phase): save/restore KV cache to NVMe for session continuity, avoiding prompt re-processing.

## References

1. GGUF Format Specification -- https://github.com/ggerganov/ggml/blob/master/docs/gguf.md
2. HuggingFace GGUF Documentation -- https://huggingface.co/docs/hub/en/gguf
3. llama.cpp Ternary Quantization PR #8151 -- https://github.com/ggml-org/llama.cpp/pull/8151
4. ggml-common.h (block structs) -- https://github.com/ggml-org/ggml/blob/master/src/ggml-common.h
5. ggml-quants.c (decode implementations) -- https://github.com/ggml-org/ggml/blob/master/src/ggml-quants.c
6. llama-model-loader.cpp -- https://github.com/ggml-org/llama.cpp/blob/master/src/llama-model-loader.cpp
7. ServerlessLLM (OSDI 2024) -- https://arxiv.org/html/2401.14351v2
8. ServerlessLLM Deep Wiki -- https://deepwiki.com/ServerlessLLM/ServerlessLLM/4.1-model-storage-and-loading
9. fastsafetensors (IBM, CLOUD 2025) -- https://arxiv.org/html/2505.23072v1
10. FlowLoader (Springer 2025) -- https://link.springer.com/article/10.1007/s11227-025-07646-4
11. DeepNVMe/FastPersist -- https://pytorch.org/blog/deepnvme-affordable-i-o-scaling-for-deep-learning-applications/
12. ENDOR Sparse Format -- https://arxiv.org/html/2406.11674v1
13. Persistent KV Cache for Edge -- https://smallaimodel.substack.com/p/2026-02-17-persistent-kv-cache-multi
14. Maru KV Cache Engine -- https://github.com/xcena-dev/maru
15. Scale AI Cold Start Blog -- https://scale.com/blog/reduce-cold-start-time-llm-inference
16. NVIDIA Model Streamer -- https://developer.nvidia.com/blog/reducing-cold-start-latency-for-llm-inference-with-nvidia-runai-model-streamer/
17. BentoML Cold Start Blog -- https://www.bentoml.com/blog/25x-faster-cold-starts-for-llms-on-kubernetes
18. GGUF File Structure Guide -- https://malcolm-mill.github.io/LLM/gguf-file-structure-guide/
19. Memory-Mapped Files Overview -- https://martinuke0.github.io/posts/2025-12-06-memory-mapped-files-mmap.../
20. llama.cpp mmap Refactor Issue -- https://github.com/ggml-org/llama.cpp/issues/16180
21. mmap vs Direct I/O Discussion -- https://github.com/ggml-org/llama.cpp/discussions/18758
22. LZ4 Compression -- https://lz4.org
23. Zstd Compression -- https://facebook.github.io/zstd/
24. GGUF vs GPTQ vs AWQ Comparison -- https://localaimaster.com/blog/quantization-explained
25. E2E Networks Quantization Guide -- https://www.e2enetworks.com/blog/which-quantization-method-is-best-for-you-gguf-gptq-or-awq
26. Microsoft CreateFileMapping -- https://learn.microsoft.com/en-us/windows/win32/api/memoryapi/nf-memoryapi-createfilemappinga
27. Microsoft PrefetchVirtualMemory -- https://learn.microsoft.com/en-us/windows/win32/api/memoryapi/nf-memoryapi-prefetchvirtualmemory

## Audit Addendum (2026-04-02)

- **Repacked artifact invalidation should be deterministic.** A runtime cache
  keyed only by model filename is too weak; it should include at least source
  metadata, pack-layout version, and target kernel family.
- **Unsupported GGUF variants should fail loudly.** It is better to reject
  unknown tensor types or incompatible tokenizer metadata than to accept them
  and discover semantic drift much later.
- **Progressive validation is worth formalizing.** A robust load order is:
  1. header and metadata validation,
  2. tensor table validation,
  3. tokenizer contract validation,
  4. then repack and runtime allocation.
