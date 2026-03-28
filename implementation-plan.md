# Edge LM: Custom Inference Engine Plan
## 100+ tok/s on 3B LLM on Consumer Intel Hardware

**Author:** Chait
**Date:** March 25, 2026
**Hardware:** Intel i7-12700H, 16GB DDR4-3200, Intel Iris Xe 96 EUs, Windows 11, NVMe SSD
**Goal:** 100-120 tok/s on 3B param LLM | Custom engine (not a fork) | Research paper

---

## Table of Contents
1. [Executive Summary](#1-executive-summary)
2. [Hardware Analysis & Constraints](#2-hardware-analysis--constraints)
3. [The Memory Bandwidth Wall](#3-the-memory-bandwidth-wall)
4. [Existing Baseline Benchmarks](#4-existing-baseline-benchmarks)
5. [Architecture Strategy](#5-architecture-strategy)
6. [Model Selection](#6-model-selection)
7. [Core Optimizations (Ranked by Impact)](#7-core-optimizations-ranked-by-impact)
8. [Implementation Plan (Phase by Phase)](#8-implementation-plan-phase-by-phase)
9. [Experiment Schedule (One at a Time)](#9-experiment-schedule-one-at-a-time)
10. [Benchmarking Methodology](#10-benchmarking-methodology)
11. [Research Paper Structure](#11-research-paper-structure)
12. [What NOT To Do (Lessons Learned)](#12-what-not-to-do-lessons-learned)
13. [Risk Analysis & Fallback Plans](#13-risk-analysis--fallback-plans)
14. [References & Resources](#14-references--resources)

---

## 1. Executive Summary

This plan details how to build a custom inference engine that achieves 100+ tokens/second for 3B parameter LLMs on a consumer Intel laptop (i7-12700H, 16GB RAM, no dGPU). Current tools achieve 5-7 tok/s (llama.cpp) and 40-60 tok/s (BitNet). Our target is 100-120 tok/s.

**The core insight:** The memory bandwidth wall (DDR4-3200 at ~38-42 GB/s real-world) makes 100+ tok/s impossible with standard Q4_0 quantization (~1.7GB model). The only viable path is:

1. **BitNet 1.58-bit ternary weights** (~0.4-0.6GB for 3B) to break through the bandwidth ceiling
2. **Custom AVX2/AVX-VNNI kernels** for maximum CPU throughput on ternary operations
3. **NVMe streaming + OS page cache** for efficient model/KV cache management
4. **Hybrid CPU+iGPU compute** using Intel Iris Xe for attention/softmax offload

**Inspiration:** flash-moe (danveloper/flash-moe) achieved 5.7 tok/s on a 397B MoE model on MacBook using custom Objective-C + Metal shaders, 3-command-buffer pipeline, NVMe streaming via pread(), and 2-bit expert requantization. We adapt their philosophy (not their code) to Intel/Windows.

---

## 2. Hardware Analysis & Constraints

### CPU: Intel i7-12700H (Alder Lake, 12th Gen)
| Feature | Details |
|---------|---------|
| Cores | 6 P-cores (Golden Cove) + 8 E-cores (Gracemont) = 20 threads |
| Clock | P-cores: up to 4.7 GHz, E-cores: up to 3.5 GHz |
| L2 Cache | 1.25 MB per P-core, 2 MB shared per 4 E-cores = 11.5 MB total |
| L3 Cache | 24 MB shared |
| SIMD | SSE4.2, AVX2 (256-bit), AVX-VNNI (VPDPBUSD) |
| NO AVX-512 | Permanently fused off on Alder Lake (both P and E cores) |
| NO AMX | Not available on mobile Alder Lake |
| Optimal threads for SIMD | 14 (6 P-core physical + 8 E-core) — HT adds contention for SIMD workloads |

### Memory: DDR4-3200 Dual Channel
| Feature | Details |
|---------|---------|
| Capacity | 16 GB total |
| Theoretical bandwidth | 51.2 GB/s (dual channel) |
| Real-world bandwidth | ~38-42 GB/s (measured with benchmarks) |
| Budget for inference | 6-7 GB max (leave rest for OS + other apps) |

### iGPU: Intel Iris Xe (96 EUs)
| Feature | Details |
|---------|---------|
| Execution Units | 96 EUs, up to 1.45 GHz |
| FP32 throughput | ~2.2 TFLOPS |
| FP16 throughput | ~4.4 TFLOPS |
| INT8 throughput | ~8.8 TOPS |
| Shared memory | Uses system RAM (configurable 128MB-8GB via BIOS) |
| API | OpenCL 3.0, oneAPI/SYCL (Level Zero), Vulkan compute |
| Key limitation | Shares DDR4 bandwidth with CPU — concurrent use reduces effective bandwidth for both |

### Storage: NVMe SSD
| Feature | Details |
|---------|---------|
| Free space | 200 GB |
| Sequential read | ~5-6.5 GB/s (PCIe 4.0 x4) |
| Random 4K read | ~500K-700K IOPS |
| Windows async I/O | ReadFile + OVERLAPPED + IOCP, or DirectStorage SDK |

### Hard Constraints
- **Max 6-7 GB RAM** for all inference processes combined
- **No AVX-512** — all SIMD must use 256-bit AVX2 paths
- **Shared memory bus** — CPU and iGPU compete for DDR4 bandwidth
- **Windows 11** — must use Win32 API for async I/O, thread affinity, large pages
- **Always plugged in** — can use max power/turbo settings

---

## 3. The Memory Bandwidth Wall

This is THE fundamental constraint. During autoregressive decoding (generating one token at a time), the bottleneck is reading model weights from memory, not compute.

### The Math

```
tokens_per_second = memory_bandwidth / bytes_per_token

bytes_per_token ≈ model_size_in_bytes (must read all weights for each token)
```

| Quantization | Model Size (3B) | Theoretical Max tok/s @ 40 GB/s |
|-------------|-----------------|--------------------------------|
| FP16 (16-bit) | ~6.0 GB | ~6.7 tok/s |
| Q8_0 (8-bit) | ~3.0 GB | ~13.3 tok/s |
| Q4_0 (4-bit) | ~1.7 GB | ~23.5 tok/s |
| Q4_K_M (4.5-bit) | ~1.9 GB | ~21.1 tok/s |
| Q2_K (2-bit) | ~1.1 GB | ~36.4 tok/s |
| **BitNet 1.58-bit** | **~0.4-0.6 GB** | **~67-100 tok/s** |
| BitNet 1.58 + optimizations | ~0.4-0.6 GB | **100-150 tok/s** (with reduced overhead) |

**Key insight:** Even with perfect memory access patterns, Q4_0 tops out around 23 tok/s on this hardware. To reach 100+ tok/s, we MUST use 1.58-bit ternary quantization (BitNet) or find ways to avoid reading all weights every token.

### How to Beat the Wall
1. **Extreme quantization** (BitNet 1.58-bit): Smallest possible model footprint
2. **Model fits in L3 cache**: 24 MB L3 = not enough for 3B even at 1.58-bit, but hot layers can be cached
3. **Prefetching**: Hide memory latency with software prefetch instructions
4. **Ternary arithmetic**: Replace multiply-accumulate with conditional add/subtract (no multiplication needed!)
5. **Batch tokens in prefill**: Amortize weight reads across multiple tokens during prompt processing

---

## 4. Existing Baseline Benchmarks

### On Our Hardware (i7-12700H) or Very Similar

| Engine | Model | Quantization | tok/s | Notes |
|--------|-------|-------------|-------|-------|
| llama.cpp | Llama-3.2-3B-Instruct | Q4_K_M | 5-7 | Our baseline measurement |
| BitNet (bitnet.cpp) | BitNet-b1.58-2B-4T (2.5B) | 1.58-bit | ~46 | i7-13700H benchmark (very similar CPU) |
| BitNet (bitnet.cpp) | bitnet_b1_58-3B (3.3B) | 1.58-bit | ~30 | i7-13700H, I2_S kernel |
| BitNet (bitnet.cpp) | bitnet_b1_58-3B (3.3B) | 1.58-bit | ~35 | i7-13700H, TL2_0 kernel |
| T-MAC | BitNet-3B | LUT-based | ~40-50 | Estimated from M2 Ultra scaling (71 tok/s on 8 perf cores) |

### Key Observations
- **BitNet's official 2B4T**: 46.33 tok/s on i7-13700H, 0.4GB memory, 29ms decode latency
- **BitNet 3B (I2_S kernel)**: ~30 tok/s on similar Intel hardware
- **BitNet 3B (TL2_0 kernel)**: ~35 tok/s — the TL2 kernel is ~17% faster than I2_S
- **Gap to target**: We need ~3x improvement over BitNet's current 30-35 tok/s for 3B models
- **llama.cpp vs BitNet**: BitNet is 4.19x-6.17x faster than llama.cpp on same hardware

### What This Tells Us
BitNet on similar hardware does 30-46 tok/s for 2.5-3B models. To reach 100+, we need:
- Better kernels than BitNet's current implementation (they're not fully optimized for our specific CPU)
- Exploit AVX-VNNI (VPDPBUSD) which BitNet may not fully utilize
- Add iGPU offloading for non-weight-bound operations
- Pipeline parallelism between CPU cores and iGPU
- Speculative decoding for effective throughput boost

---

## 5. Architecture Strategy

### Design Philosophy (Inspired by flash-moe)
1. **Zero dependencies** — Custom C/C++ engine, no Python runtime, no ONNX, no frameworks
2. **Hardware-specific** — Written FOR i7-12700H, not abstracting away the hardware
3. **Ternary-first** — Architecture assumes 1.58-bit weights as the primary format
4. **Memory hierarchy aware** — Explicitly manage L2, L3, RAM, and NVMe tiers
5. **Async everything** — Overlap I/O, compute, and iGPU work

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      Edge LM Engine                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────┐  ┌──────────────┐  ┌────────────────────────────┐ │
│  │ Tokenizer │  │ GGUF/Custom  │  │ Weight Manager              │ │
│  │ (C impl)  │  │ Model Loader │  │ (Ternary packing, prefetch)│ │
│  └────┬─────┘  └──────┬───────┘  └─────────┬──────────────────┘ │
│       │               │                     │                    │
│  ┌────▼───────────────▼─────────────────────▼──────────────────┐ │
│  │              Transformer Pipeline                            │ │
│  │                                                              │ │
│  │  ┌─────────────┐  ┌──────────────┐  ┌───────────────────┐  │ │
│  │  │ RMSNorm     │  │ Attention     │  │ FFN/MLP           │  │ │
│  │  │ (AVX2)      │  │ (CPU+iGPU)   │  │ (AVX-VNNI ternary)│  │ │
│  │  └─────────────┘  └──────────────┘  └───────────────────┘  │ │
│  │                                                              │ │
│  │  ┌─────────────┐  ┌──────────────┐  ┌───────────────────┐  │ │
│  │  │ KV Cache    │  │ Sampler      │  │ Speculative       │  │ │
│  │  │ (Ring buf)  │  │ (Top-p/k)    │  │ Decode (optional) │  │ │
│  │  └─────────────┘  └──────────────┘  └───────────────────┘  │ │
│  └──────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  ┌──────────────────┐  ┌──────────────────┐                     │
│  │ Thread Pool       │  │ iGPU Scheduler   │                     │
│  │ (P-core + E-core  │  │ (OpenCL/L0)      │                     │
│  │  affinity pinned) │  │                  │                     │
│  └──────────────────┘  └──────────────────┘                     │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────────┐ │
│  │ Memory Manager (VirtualAlloc + Large Pages + NVMe streaming) │ │
│  └──────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

### Key Architectural Decisions

1. **Language: C with inline ASM for critical kernels**
   - C for portability and simplicity (not C++ — minimal abstraction overhead)
   - Inline AVX2/AVX-VNNI assembly for matmul kernels
   - No STL, no exceptions, no RTTI, no virtual dispatch

2. **Model Format: GGUF (read) + Custom ternary pack (runtime)**
   - Parse GGUF files directly (~300 lines of C)
   - On load: repack weights into ternary format optimized for our AVX2 kernels
   - Store repacked weights to disk as cache for fast subsequent loads

3. **Threading Model: Explicit core affinity**
   - P-cores (6): Heavy SIMD matmul work
   - E-cores (8): Lightweight tasks (tokenization, sampling, I/O, prefetching)
   - Pin threads to specific cores using SetThreadAffinityMask
   - Custom work-stealing scheduler (no OS thread pool)

4. **Memory Model: Tiered**
   - L3 cache (24MB): Hot weights + current layer activations
   - RAM (6-7GB budget): Full model + KV cache
   - NVMe: Extended KV cache for very long contexts

---

## 6. Model Selection

### Primary Target: BitNet-b1.58 3B Model

| Model | Params | Bits/Weight | Size | Quality | Speed Potential |
|-------|--------|-------------|------|---------|-----------------|
| **BitNet-b1.58-2B-4T** | 2.4B | 1.58 | 0.4 GB | Matches Qwen2.5-1.5B | Highest (smallest) |
| **bitnet_b1_58-3B** | 3.3B | 1.58 | ~0.6 GB | Decent for 3B class | High |
| Llama-3.2-3B (Q4_K_M) | 3.2B | ~4.5 | 1.9 GB | Best quality at 3B | Medium (bandwidth-limited) |
| Llama-3.2-3B (Q2_K) | 3.2B | ~2 | 1.1 GB | Degraded quality | Medium-High |

### Recommendation
- **Start with BitNet-b1.58-2B-4T** (2.4B, 0.4GB) — smallest, fastest, official benchmarks available
- **Then bitnet_b1_58-3B** (3.3B, 0.6GB) — true 3B target
- **Stretch goal: Custom ternary quantization of Llama-3.2-3B** — apply GPTQ/BitNet-style quantization to get best-quality 3B model in ternary format

### Why BitNet 1.58-bit is Perfect for Our Use Case
1. **Weights are {-1, 0, +1}** — matmul becomes conditional add/subtract, no FP multiply needed
2. **0.4-0.6 GB for 3B** — fits entirely in RAM with huge headroom
3. **CPU-friendly** — designed for CPU inference, unlike GPU-first models
4. **AVX-VNNI synergy** — VPDPBUSD can process 64 ternary weights per instruction cycle (packed as INT8)
5. **Already proven** — BitNet official achieves 30-46 tok/s on similar CPU without our optimizations

---

## 7. Core Optimizations (Ranked by Impact)

### Tier 1: Critical (Must Have for 100+ tok/s)

#### 7.1 Custom AVX2/AVX-VNNI Ternary Matmul Kernels
**Expected impact: 2-3x over BitNet's current kernels**

BitNet's kernels (I2_S, TL2) are general-purpose. We write kernels specifically for i7-12700H:

```
Ternary matmul: y = W * x where W ∈ {-1, 0, +1}
For each output element:
  sum = 0
  for each weight:
    if w == +1: sum += x[i]
    if w == -1: sum -= x[i]
    if w ==  0: skip (no-op)
```

**Implementation approach:**
- Pack 128 ternary weights into 256 bits (2 bits each) in AVX2 register
- Use VPSHUFB (byte shuffle) as a 4-bit LUT to map ternary codes to masks
- Use VPADDB/VPSUBB for conditional add/subtract
- Unroll 4x to hide latency
- Software prefetch next cache line while processing current one
- Use AVX-VNNI VPDPBUSD for inner products where applicable (pack ternary as INT8 {-1,0,+1})

**Key insight from T-MAC:** Lookup-table (LUT) based approach eliminates multiplications entirely. T-MAC precomputes all possible partial sums for groups of weights and uses table lookups instead of arithmetic. This is perfect for ternary weights where the LUT is tiny (3^k entries for k-weight groups).

#### 7.2 Explicit Memory Prefetching
**Expected impact: 20-40% improvement in memory throughput utilization**

- Use `_mm_prefetch` with `_MM_HINT_T0` (L1), `_MM_HINT_T1` (L2), `_MM_HINT_T2` (L3)
- Prefetch next weight block 2-3 cache lines ahead of current computation
- Prefetch next layer's weights while finishing current layer
- Align all weight buffers to 64-byte cache line boundaries

#### 7.3 Thread Affinity & Core Scheduling
**Expected impact: 15-25% improvement from eliminating cross-core migration**

```
P-cores (0-5): Matmul workers — each processes a slice of the output dimension
E-cores (6-13):
  - Core 6: Main thread (tokenizer, sampler, orchestration)
  - Core 7: iGPU submission thread
  - Core 8-9: Prefetch/memcpy workers
  - Core 10-13: Available for speculative decode draft model
```

- Use `SetThreadAffinityMask()` and `SetThreadIdealProcessor()`
- Use `VirtualAlloc` with `MEM_LARGE_PAGES` for weight buffers (2MB pages reduce TLB misses)
- NUMA-aware allocation (though single-socket, still matters for page placement)

### Tier 2: Important (Needed to Reach 100+)

#### 7.4 Hybrid CPU+iGPU Pipeline
**Expected impact: 15-30% by offloading attention to iGPU**

The FFN/MLP layers are weight-bound (ternary matmul) — best on CPU with AVX2.
The attention layers are compute-bound (QKV projection, softmax, score computation) — can benefit from iGPU parallelism.

**Pipeline:**
```
Token N:    [CPU: FFN layer L]  [iGPU: Attention layer L+1]
Token N+1:  [CPU: FFN layer L+1]  [iGPU: Attention layer L+2]
```

- Use OpenCL or Level Zero (oneAPI) for iGPU dispatch
- Pre-compile attention kernels as OpenCL .cl files
- Double-buffer: CPU writes input to buffer A while iGPU reads from buffer B
- **WARNING:** iGPU shares DDR4 bandwidth with CPU. Must carefully schedule to avoid bandwidth contention. May need to time-slice: CPU-heavy phase, then iGPU-heavy phase.

#### 7.5 Optimized KV Cache
**Expected impact: 10-20% for longer sequences**

- Ring buffer implementation (fixed-size, overwrites oldest entries)
- FP8 or INT8 KV cache quantization (halves KV memory vs FP16)
- Grouped-Query Attention (GQA) support — Llama 3.2 uses 8 KV heads vs 32 Q heads
- KV cache in contiguous memory with 64-byte alignment

#### 7.6 Fast Tokenizer
**Expected impact: Eliminates startup latency, keeps pipeline fed**

- C implementation of BPE tokenizer (no Python, no sentencepiece library)
- Precompute merge table as perfect hash
- Target: <5ms for 1000-token input (flash-moe achieves 180ms startup vs 3500ms Python)

### Tier 3: Stretch Goals (For Paper / Extra Performance)

#### 7.7 Speculative Decoding
**Expected impact: 1.5-2.5x effective throughput**

- Use a tiny draft model (e.g., 0.5B ternary) on E-cores
- Draft model generates K candidate tokens
- Main model verifies all K in one batch forward pass
- Accept N <= K tokens, reject rest
- Net effect: generate ~2-3 tokens per main-model forward pass

#### 7.8 NVMe Streaming for Extended Context
**Expected impact: Enables 32K+ context without RAM increase**

- KV cache overflow to NVMe when context exceeds RAM budget
- Use OVERLAPPED + IOCP for async page-in
- DirectStorage SDK for lowest-overhead NVMe access
- FILE_FLAG_NO_BUFFERING + VirtualAlloc aligned buffers for direct I/O
- Trust OS page cache for recently-used KV pages (flash-moe approach)

#### 7.9 Weight Repacking for Cache Locality
**Expected impact: 5-10% from reduced cache misses**

- Reorder weight matrices for sequential access pattern during matmul
- Tile weights to match L2 cache size (1.25 MB per P-core)
- Block size: process weights in 64KB blocks that fit in L1

---

## 8. Implementation Plan (Phase by Phase)

### Phase 0: Environment Setup (Week 1)
- [ ] Install Visual Studio 2022 with C++ desktop development workload
- [ ] Install Intel oneAPI Base Toolkit (for OpenCL/SYCL iGPU development)
- [ ] Install CMake, Ninja build system
- [ ] Set up project structure in `fast-llm-inference/`
- [ ] Download BitNet-b1.58-2B-4T model from HuggingFace
- [ ] Verify BitNet official repo compiles and runs on our hardware (establishes ground truth baseline)
- [ ] Document exact baseline numbers: tok/s, memory usage, CPU utilization

### Phase 1: Minimal Viable Engine (Weeks 2-4)
**Goal: Load model, run inference, get ANY output — even if slow**

- [ ] GGUF parser: Read model metadata, tensor shapes, quantization info
- [ ] Memory allocator: VirtualAlloc with large pages for weight buffers
- [ ] Weight loader: Load ternary weights, repack into optimized layout
- [ ] Basic transformer forward pass:
  - RMSNorm (naive C implementation)
  - Attention (naive — single-threaded, no KV cache optimization)
  - FFN/MLP with ternary matmul (naive — use simple loops, no SIMD yet)
  - Rotary Position Embeddings (RoPE)
- [ ] Sampling: Top-p, top-k, temperature
- [ ] BPE tokenizer in C
- [ ] End-to-end: Prompt in, tokens out
- **Expected performance: 5-15 tok/s (naive C, no optimizations)**

### Phase 2: AVX2 Kernel Optimization (Weeks 5-7)
**Goal: Maximize single-core throughput with SIMD**

- [ ] Implement AVX2 ternary matmul kernel (basic version)
- [ ] Benchmark: measure GFLOPS vs theoretical peak
- [ ] Implement AVX-VNNI (VPDPBUSD) variant for INT8 dot products
- [ ] Implement T-MAC style LUT-based ternary kernel
- [ ] A/B test: direct ternary vs LUT approach — pick winner
- [ ] Add software prefetching to winning kernel
- [ ] Optimize RMSNorm with AVX2 (rsqrt + FMA)
- [ ] Optimize attention score computation with AVX2
- [ ] Profile with Intel VTune — find remaining bottlenecks
- **Expected performance: 30-50 tok/s (matching or exceeding BitNet)**

### Phase 3: Multi-threading & Core Affinity (Weeks 8-9)
**Goal: Scale across all 14 useful cores**

- [ ] Implement thread pool with explicit core affinity
- [ ] Partition matmul across P-cores (row-parallel)
- [ ] Assign orchestration to E-cores
- [ ] Implement work-stealing for load balancing
- [ ] Memory: align per-thread buffers to avoid false sharing
- [ ] Benchmark scaling: 1 core, 6 P-cores, 6P+8E
- **Expected performance: 50-80 tok/s**

### Phase 4: iGPU Offloading (Weeks 10-12)
**Goal: Use Iris Xe for attention/softmax, free CPU for matmul**

- [ ] Set up OpenCL or Level Zero for Iris Xe
- [ ] Implement attention kernel on iGPU (Q*K^T, softmax, *V)
- [ ] Implement double-buffered CPU<->iGPU data transfer
- [ ] Profile bandwidth contention — adjust scheduling
- [ ] Benchmark: CPU-only vs CPU+iGPU hybrid
- [ ] If bandwidth contention too high: time-slice instead of overlap
- **Expected performance: 70-100 tok/s**

### Phase 5: Advanced Optimizations (Weeks 13-16)
**Goal: Push past 100 tok/s, add speculative decoding**

- [ ] Implement speculative decoding with draft model
- [ ] Optimize KV cache: FP8 quantization, ring buffer
- [ ] Large page support (2MB pages via SeLockMemoryPrivilege)
- [ ] NVMe streaming for extended context
- [ ] Final profiling and micro-optimization pass
- [ ] Test with bitnet_b1_58-3B (full 3B target)
- **Expected performance: 100-120+ tok/s**

### Phase 6: Paper & Polish (Weeks 17-20)
**Goal: Write research paper, clean up code, publish**

- [ ] Run comprehensive benchmarks across all configurations
- [ ] Compare with llama.cpp, BitNet, T-MAC on same hardware
- [ ] Extrapolate results to 5B models (for paper)
- [ ] Write research paper (see Section 11)
- [ ] Clean up code, add README, open-source

---

## 9. Experiment Schedule (One at a Time)

**CRITICAL RULE: Only ONE experiment runs at a time. Max 6-7 GB RAM. Always ask before running.**

Each experiment follows this protocol:
1. Define hypothesis and expected outcome
2. Write the code change
3. **ASK BEFORE RUNNING** (never auto-run heavy processes)
4. Run single experiment
5. Record results in `results/experiment_NNN.json`
6. Analyze before moving to next experiment

### Experiment List (Sequential)

```
EXP-001: Baseline — Run BitNet official on our hardware, record exact tok/s
EXP-002: Baseline — Run llama.cpp with our GGUF model, record exact tok/s
EXP-003: Minimal engine — GGUF parse + naive forward pass, verify correctness
EXP-004: Naive ternary matmul — C loops, measure tok/s baseline for our engine
EXP-005: AVX2 ternary kernel v1 — basic SIMD, measure improvement over naive
EXP-006: AVX2 ternary kernel v2 — with prefetching, measure improvement
EXP-007: AVX-VNNI kernel — VPDPBUSD approach, compare with v2
EXP-008: LUT-based kernel (T-MAC style) — compare with direct ternary
EXP-009: Winner kernel + multi-thread (6 P-cores only)
EXP-010: Winner kernel + multi-thread (6P + 8E cores)
EXP-011: Thread affinity pinning — measure improvement over OS scheduling
EXP-012: Large pages (2MB) — measure TLB miss reduction
EXP-013: iGPU attention kernel — OpenCL, measure standalone throughput
EXP-014: CPU+iGPU hybrid — combined pipeline, measure end-to-end tok/s
EXP-015: KV cache optimization — FP8 quantization
EXP-016: Speculative decoding — draft model on E-cores
EXP-017: Full pipeline — all optimizations combined
EXP-018: 3B model test (bitnet_b1_58-3B)
EXP-019: Quality evaluation — compare output quality with llama.cpp
EXP-020: Ablation study — disable each optimization, measure individual contribution
```

### Results Recording Format
```json
{
  "experiment_id": "EXP-001",
  "date": "2026-XX-XX",
  "description": "BitNet official baseline on i7-12700H",
  "model": "BitNet-b1.58-2B-4T",
  "config": {
    "threads": 14,
    "kernel": "I2_S",
    "prompt_tokens": 128,
    "gen_tokens": 256
  },
  "results": {
    "prefill_tok_s": null,
    "decode_tok_s": null,
    "peak_ram_mb": null,
    "cpu_utilization_pct": null,
    "first_token_ms": null
  },
  "notes": ""
}
```

---

## 10. Benchmarking Methodology

### Standard Benchmark Suite
For every engine configuration, run these 5 tests:

| Test | Prompt Length | Generation Length | Purpose |
|------|-------------|-------------------|---------|
| Short Q&A | 32 tokens | 128 tokens | Typical chatbot usage |
| Medium | 256 tokens | 256 tokens | Standard benchmark |
| Long prompt | 1024 tokens | 128 tokens | Test prefill speed |
| Long generation | 64 tokens | 512 tokens | Sustained decode speed |
| Stress test | 2048 tokens | 1024 tokens | Max context + generation |

### Metrics to Record
1. **Prefill speed** (tok/s): How fast we process the input prompt
2. **Decode speed** (tok/s): How fast we generate output tokens (THE key metric)
3. **Time to first token** (ms): Latency before first output token
4. **Peak RAM usage** (MB): Via Windows Performance Counters
5. **CPU utilization** (%): Per-core, via QueryProcessCycleTime
6. **iGPU utilization** (%): Via Intel GPU Tools
7. **Power consumption** (W): Via Intel Power Gadget (if available)
8. **Output quality**: Perplexity on WikiText-2, MMLU score, or manual evaluation

### Comparison Baselines
- llama.cpp (latest) with Q4_K_M on same hardware
- BitNet official (bitnet.cpp) on same hardware
- T-MAC (if compilable on Windows/Intel)
- Our engine at each optimization stage

### Statistical Rigor (for paper)
- Run each benchmark 10 times
- Report: mean, median, std dev, min, max
- Warm up: 3 runs before recording
- Control: Close all other applications, set power plan to "Best Performance"
- Report hardware: exact CPU stepping, BIOS version, RAM timings

---

## 11. Research Paper Structure

### Title (Working)
"Edge LM: Achieving 100+ Tokens/Second for 3B LLMs on Consumer Intel Hardware"

### Abstract
- Problem: Local LLM inference on consumer hardware is too slow for practical use
- Approach: Custom inference engine with ternary-first kernels, hybrid CPU+iGPU pipeline, cache-aware scheduling
- Result: Xxtok/s on i7-12700H, Yx faster than llama.cpp, Zx faster than BitNet
- Contribution: Proves 100+ tok/s is achievable on $1000 laptop without dGPU

### Paper Outline

1. **Introduction** (1 page)
   - Motivation: local LLM inference matters for privacy, offline use, cost
   - Challenge: consumer hardware severely bandwidth-limited
   - Contribution: custom engine achieving 100+ tok/s

2. **Background & Related Work** (2 pages)
   - LLM inference fundamentals (autoregressive decoding, memory-bound nature)
   - Quantization: GPTQ, AWQ, BitNet b1.58, SmoothQuant
   - CPU inference: llama.cpp, BitNet, T-MAC, llamafile
   - GPU-less inference: flash-moe, LLM in a Flash
   - Speculative decoding
   - Intel-specific: AVX2, AVX-VNNI, Iris Xe

3. **System Design** (3 pages)
   - Hardware analysis of i7-12700H
   - Memory bandwidth wall analysis
   - Architecture overview
   - Ternary-first design decisions

4. **Key Optimizations** (4 pages)
   - 4.1: AVX2/AVX-VNNI ternary matmul kernels
   - 4.2: T-MAC-inspired LUT approach
   - 4.3: Cache-aware weight layout and prefetching
   - 4.4: Hybrid CPU+iGPU pipeline
   - 4.5: Thread affinity and core scheduling
   - 4.6: Speculative decoding adaptation

5. **Evaluation** (3 pages)
   - Benchmark methodology
   - End-to-end performance comparison
   - Ablation study (each optimization's contribution)
   - Scaling analysis (extrapolation to 5B)
   - Quality evaluation (perplexity, MMLU)

6. **Discussion** (1 page)
   - Lessons learned
   - Limitations (hardware-specific, model quality tradeoffs)
   - When iGPU helps vs hurts (bandwidth contention)

7. **Conclusion & Future Work** (0.5 page)
   - Summary of results
   - Future: extend to other Intel platforms, AMD APUs, ARM

### Target Venues
- arXiv preprint (immediate)
- MLSys 2027 or EuroSys 2027 (conference submission)
- NeurIPS 2026 Workshop on Efficient ML (if timeline works)

---

## 12. What NOT To Do (Lessons Learned)

### From Previous Session Crash
1. **NEVER run multiple experiments simultaneously** — Previous session ran ~108 experiments (3 LLMs x multiple configs) at once, crashing the system
2. **NEVER auto-launch heavy processes** — Always ask the user before running anything that loads a model or uses significant compute/memory
3. **Max 6-7 GB RAM for all processes** — Exceeding this WILL crash the 16 GB system
4. **ONE experiment at a time, sequentially** — Record results, analyze, then proceed

### Technical Pitfalls to Avoid
5. **Don't target AVX-512** — It's permanently fused off on Alder Lake. Code WILL crash with illegal instruction
6. **Don't use HyperThreading for SIMD workloads** — 14 physical threads is optimal, 20 logical threads adds contention
7. **Don't overlap CPU and iGPU memory-intensive work** — They share DDR4 bandwidth. Time-slice instead
8. **Don't use FP16 on CPU** — Intel CPUs have no native FP16 SIMD (unlike ARM NEON). Use INT8 or INT16 for quantized ops
9. **Don't use Python for anything in the hot path** — Pure C/C++ only. Python tokenizer alone adds 3+ seconds startup
10. **Don't fork llama.cpp or BitNet** — Custom engine. Study their code for ideas, but write from scratch
11. **Don't try to optimize everything at once** — Implement naive first, profile, optimize the bottleneck, repeat
12. **Don't ignore correctness** — Verify output matches reference implementation before optimizing. Wrong answers fast are worthless
13. **Don't assume Q4 models can reach 100 tok/s** — The bandwidth math says no. Must use 1.58-bit ternary
14. **Don't allocate iGPU shared memory via BIOS beyond 256MB** — More steals from CPU RAM budget

### Project Management Pitfalls
15. **Save plan and results to disk** — Don't rely on chat history (it gets lost)
16. **Git commit after each working milestone** — Can always roll back
17. **Profile before optimizing** — Intel VTune is free and essential
18. **Test on actual hardware, not theoretical numbers** — Papers lie, benchmarks don't

---

## 13. Risk Analysis & Fallback Plans

### Risk 1: Can't reach 100 tok/s with BitNet 3B
**Probability:** Medium (30-35 tok/s baseline, need ~3x improvement)
**Mitigation:**
- Fall back to 2.4B (BitNet-2B-4T) which is ~40% smaller and should be proportionally faster
- If 80+ tok/s achieved, still publishable as a significant improvement
- Speculative decoding can provide 1.5-2.5x effective throughput boost

### Risk 2: iGPU bandwidth contention makes hybrid pipeline slower
**Probability:** High (shared DDR4 bus is THE bottleneck)
**Mitigation:**
- Time-slice approach: CPU-heavy phase, then iGPU-heavy phase
- Use iGPU only for compute-bound operations (softmax, RoPE) not memory-bound
- May need to limit iGPU to specific layers only
- Paper contribution: document when iGPU helps vs hurts on shared-memory architectures

### Risk 3: Model quality is unacceptable at 1.58-bit
**Probability:** Low (BitNet-2B-4T matches Qwen2.5-1.5B on benchmarks)
**Mitigation:**
- BitNet models are specifically trained for ternary weights (not post-training quantized)
- If quality is an issue, use 2-bit quantization (Q2_K) as compromise
- Document quality-speed tradeoff curves in paper

### Risk 4: AVX-VNNI doesn't provide expected speedup
**Probability:** Medium (VNNI is 256-bit on Alder Lake, not 512-bit)
**Mitigation:**
- Pure AVX2 kernels are the baseline — VNNI is bonus
- LUT-based approach (T-MAC style) may outperform VNNI for ternary
- Benchmark both approaches, use whichever is faster

### Risk 5: Custom engine has bugs that produce wrong output
**Probability:** High (inevitable during development)
**Mitigation:**
- Validate every layer output against reference implementation (BitNet or llama.cpp)
- Use small test model first (e.g., 0.1B random weights) for rapid iteration
- Unit test each kernel with known input/output pairs
- Numerical tolerance testing: compare FP32 reference vs our INT8/ternary output

---

## 14. References & Resources

### Key Papers
1. **BitNet: Scaling 1-bit Transformers** (Ma et al., 2024) — The foundation for 1.58-bit weights
2. **The Era of 1-bit LLMs** (Ma et al., 2024) — BitNet b1.58 architecture
3. **bitnet.cpp** (Microsoft, 2024) — Official inference framework, our primary benchmark
4. **T-MAC: Table-Based MAC** (Wei et al., 2024) — LUT-based approach for low-bit inference
5. **LLM in a Flash** (Alizadeh et al., Apple, 2024) — Flash memory techniques for large models
6. **AWQ: Activation-aware Weight Quantization** (Lin et al., 2024) — Quality-preserving quantization
7. **Flash Attention 2** (Dao, 2023) — Optimized attention computation
8. **Speculative Decoding** (Leviathan et al., 2023) — Draft-verify approach

### Key Repositories
- flash-moe: github.com/danveloper/flash-moe (architecture inspiration)
- bitnet.cpp: github.com/microsoft/BitNet
- T-MAC: github.com/microsoft/T-MAC
- llama.cpp: github.com/ggml-org/llama.cpp
- llamafile: github.com/Mozilla-Ocho/llamafile (custom AVX2 tinyBLAS kernels)
- whisper.cpp: github.com/ggml-org/whisper.cpp (ggml reference for C inference)

### Intel Developer Resources
- Intel Intrinsics Guide: software.intel.com/sites/landingpage/IntrinsicsGuide/
- Intel VTune Profiler: Free download from Intel oneAPI
- Intel OpenCL SDK (for Iris Xe)
- AVX-VNNI programming guide (VPDPBUSD instruction)

### Models to Download
1. `BitNet-b1.58-2B-4T` — From HuggingFace: 1bitLLM/bitnet_b1_58-large
2. `bitnet_b1_58-3B` — From HuggingFace: 1bitLLM/bitnet_b1_58-3B
3. Already have: `Llama-3.2-3B-Instruct-Q4_K_M.gguf` (for comparison baseline)

### Local Research Files
- `C:\Users\chait\OneDrive\Desktop\edgeLm-research\extracted_data.json` — 100 structured technique entries
- `C:\Users\chait\OneDrive\Desktop\edgeLm-research\papers.txt` — 464 paper/resource links
- `C:\Users\chait\Project\Edge-LM research\edge_lm_papers.json` — 100 detailed paper summaries with ratings

---

## Appendix A: Theoretical Performance Model

### Decode Speed Estimation

For BitNet 3B (0.6 GB weights), ternary matmul on 6 P-cores:

```
Weight read bandwidth needed at 100 tok/s:
  0.6 GB × 100 = 60 GB/s → Exceeds DDR4-3200 bandwidth (40 GB/s)

But with ternary packing (2 bits/weight) and AVX2 processing:
  - 3B weights × 2 bits = 0.75 GB packed
  - With overhead (activations, KV cache): ~1.0 GB total memory access per token
  - At 40 GB/s real bandwidth: 40 tok/s theoretical max

With optimizations:
  - L3 cache hits for hot layers: effectively 200+ GB/s for cached data
  - Prefetching hides 30-50% of memory latency
  - iGPU offload reduces CPU memory pressure by 15-20%
  - Speculative decoding: 2x effective throughput

Estimated achievable:
  - Without speculative: 60-80 tok/s
  - With speculative: 90-120 tok/s
```

### For BitNet 2.4B (0.4 GB weights):
```
  - 0.4 GB × 100 = 40 GB/s → Just at DDR4 bandwidth limit
  - With L3 caching + prefetch: 80-100 tok/s feasible without speculative
  - With speculative: 120-150 tok/s
```

### Conclusion
- **100 tok/s on 2.4B model: HIGH confidence** (0.4 GB model, bandwidth math works)
- **100 tok/s on 3B model: MEDIUM confidence** (0.6 GB model, needs speculative decoding)
- **100 tok/s on 3B Q4 model: NOT POSSIBLE** (1.7 GB model, bandwidth ceiling ~23 tok/s)

---

## Appendix B: Project Directory Structure

```
fast-llm-inference/
├── src/
│   ├── main.c              # Entry point, CLI
│   ├── engine.c/h          # Core inference engine
│   ├── gguf.c/h            # GGUF model parser
│   ├── model.c/h           # Model loading, weight management
│   ├── transformer.c/h     # Transformer forward pass
│   ├── attention.c/h       # Attention mechanism
│   ├── ffn.c/h             # Feed-forward network
│   ├── kernels/
│   │   ├── ternary_avx2.c  # AVX2 ternary matmul
│   │   ├── ternary_vnni.c  # AVX-VNNI variant
│   │   ├── ternary_lut.c   # LUT-based (T-MAC style)
│   │   ├── rmsnorm_avx2.c  # RMSNorm with SIMD
│   │   └── rope_avx2.c     # Rotary embeddings with SIMD
│   ├── memory.c/h          # Memory manager (VirtualAlloc, large pages)
│   ├── threading.c/h       # Thread pool with core affinity
│   ├── igpu/
│   │   ├── opencl_init.c   # OpenCL initialization for Iris Xe
│   │   ├── attention.cl    # Attention kernel for iGPU
│   │   └── softmax.cl      # Softmax kernel for iGPU
│   ├── tokenizer.c/h       # BPE tokenizer in C
│   ├── sampler.c/h         # Token sampling (top-p, top-k, temp)
│   ├── kv_cache.c/h        # KV cache with FP8 quantization
│   └── speculative.c/h     # Speculative decoding
├── benchmarks/
│   ├── bench_kernels.c     # Microbenchmark individual kernels
│   ├── bench_e2e.c         # End-to-end benchmark
│   └── compare.py          # Generate comparison charts
├── results/
│   └── experiment_NNN.json # Individual experiment results
├── paper/
│   ├── edge_lm.tex         # LaTeX paper
│   ├── figures/            # Benchmark charts, architecture diagrams
│   └── references.bib      # Bibliography
├── models/
│   └── (model files)
├── tools/
│   ├── BitNet/             # Reference (read-only)
│   ├── T-MAC/              # Reference (read-only)
│   └── llama.cpp/          # Reference (read-only)
├── docs/
│   └── implementation-plan.md  # This plan (copy)
├── CMakeLists.txt
└── README.md
```

---

## Quick Start Checklist

When starting the next session:
1. Read this plan from `implementation-plan.md` in the project root
2. Check which phase/experiment we're on
3. Read latest results from `fast-llm-inference/results/`
4. Continue from where we left off
5. **NEVER run heavy processes without asking first**

---

*Plan created: March 25, 2026*
*Last updated: March 25, 2026*




