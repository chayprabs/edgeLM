<<<<<<< HEAD
# EdgeLM

**A custom inference engine targeting 100+ tokens/second for 3B-parameter LLMs on consumer Intel hardware -- no discrete GPU required.**

EdgeLM is a from-scratch C inference engine designed to push the limits of local LLM inference on a standard Intel laptop. By combining BitNet 1.58-bit ternary quantization, hand-tuned AVX2/AVX-VNNI SIMD kernels, hybrid CPU+iGPU compute, and cache-aware scheduling, we aim to achieve 100-120 tok/s on hardware that currently manages 5-7 tok/s with standard tools.

This project doubles as the foundation for a research paper targeting MLSys/EuroSys 2027.

---

## The Problem

Running LLMs locally on consumer hardware is painfully slow. On a typical Intel laptop (i7-12700H, 16GB DDR4, no dGPU):

| Engine | Model | Quantization | Speed |
|--------|-------|-------------|-------|
| llama.cpp | Llama-3.2-3B | Q4_K_M | **5-7 tok/s** |
| bitnet.cpp | BitNet 3B | 1.58-bit | **30-35 tok/s** |
| **EdgeLM (target)** | **BitNet 3B** | **1.58-bit** | **100-120 tok/s** |

The fundamental bottleneck is memory bandwidth: DDR4-3200 delivers ~40 GB/s, and a Q4 model at 1.7GB requires reading all weights per token, capping throughput at ~23 tok/s. No amount of compute optimization can fix this.

## The Solution

EdgeLM breaks through the bandwidth wall with a multi-layered approach:

1. **BitNet 1.58-bit ternary weights** {-1, 0, +1} shrink a 3B model to ~0.6 GB, and turn matrix multiplication into conditional add/subtract (no floating-point multiply needed)
2. **Custom AVX2/AVX-VNNI kernels** exploit the i7-12700H's VPDPBUSD instruction for 2.2x over bitnet.cpp's kernels
3. **P-core/E-core aware scheduling** pins SIMD-heavy work to Golden Cove P-cores (2 FMA/cycle) and offloads I/O to Gracemont E-cores
4. **Hybrid CPU+iGPU pipeline** uses Intel Iris Xe 96 EUs for attention/softmax offload
5. **Speculative decoding** with a tiny draft model on E-cores for 1.5-2.5x effective throughput

## Target Hardware

| Component | Spec | Role |
|-----------|------|------|
| **CPU** | Intel i7-12700H (6P + 8E cores, AVX2 + AVX-VNNI) | Primary compute |
| **Memory** | 16GB DDR4-3200 (~40 GB/s real bandwidth) | THE bottleneck |
| **iGPU** | Intel Iris Xe 96 EUs (~2.2 TFLOPS FP32) | Attention offload |
| **Storage** | NVMe SSD PCIe 4.0 (~5-6 GB/s) | KV cache overflow |
| **OS** | Windows 11 | Runtime environment |

**Hard constraints:** No AVX-512 (fused off on Alder Lake). No AMX. Shared DDR4 bus between CPU and iGPU. Max 6-7 GB RAM budget for inference.

## Project Status

This project is currently in the **deep research phase** -- systematically investigating every optimization technique across 25 topic areas before writing a single line of engine code. The research covers hardware microarchitecture, SIMD kernel design, quantization, memory management, threading, and more.

### Research Progress: 2 / 25 sections complete

| Part | Sections | Status |
|------|----------|--------|
| I. Hardware Foundation | 01-07 | 2/7 complete |
| II. Model & Data | 08-12 | 0/5 |
| III. Transformer Forward Pass | 13 | 0/1 |
| IV. Core Kernels & Runtime | 14-18 | 0/5 |
| V. Advanced Techniques | 19-25 | 0/7 |

See [`research-progress.md`](research-progress.md) for the full checklist.

## Repository Structure

```
edgeLM/
├── README.md                        # You are here
├── implementation-plan.md           # Full implementation roadmap (6 phases), benchmarking
│                                      methodology, experiment schedule, paper outline
├── deep-dive.md                     # Exhaustive 25-section technical reference covering
│                                      the entire inference pipeline from hardware to software
├── research-progress.md             # Tracks completion status for all 25 research sections
├── research-papers-data.json        # 100 structured entries from prior paper/technique survey
├── CLAUDE.md                        # AI assistant context file
│
└── research/                        # Deep research files (one per section)
    ├── 01-intel-alder-lake-cpu-architecture.md
    ├── 02-avx2-vnni-simd-optimization.md
    └── ... (23 more sections in progress)
```

## Key Documents

| Document | Description |
|----------|-------------|
| [`implementation-plan.md`](implementation-plan.md) | The master plan: 6 implementation phases from naive C to 100+ tok/s, experiment schedule, benchmarking methodology, risk analysis, and research paper outline |
| [`deep-dive.md`](deep-dive.md) | 25-section technical deep dive covering every layer of the inference stack -- from CPU microarchitecture to end-to-end pipeline integration |
| [`research/`](research/) | Per-section extended research with findings beyond the deep dive: recent papers (2024-2026), open-source implementations, hardware-specific tricks, benchmark data, and community insights |
| [`research-papers-data.json`](research-papers-data.json) | Structured dataset of 100 techniques and papers from the initial literature survey |

## Architecture Overview

```
                         EdgeLM Engine
 ┌──────────────────────────────────────────────────────┐
 │                                                      │
 │  Tokenizer ──► Model Loader ──► Weight Manager       │
 │  (C BPE)       (GGUF parse)    (ternary 2-bit pack)  │
 │                                                      │
 │  ┌──────────── Transformer Pipeline ──────────────┐  │
 │  │                                                │  │
 │  │  RMSNorm ──► Attention ──► FFN/MLP             │  │
 │  │  (AVX2)      (CPU+iGPU)    (AVX-VNNI ternary)  │  │
 │  │                                                │  │
 │  │  KV Cache ──► Sampler ──► Speculative Decode   │  │
 │  │  (ring buf)   (top-p/k)   (draft on E-cores)   │  │
 │  │                                                │  │
 │  └────────────────────────────────────────────────┘  │
 │                                                      │
 │  Thread Pool          iGPU Scheduler                 │
 │  (P-core/E-core       (OpenCL / Level Zero)          │
 │   affinity pinned)                                   │
 │                                                      │
 │  Memory Manager                                      │
 │  (VirtualAlloc + Large Pages + NVMe streaming)       │
 └──────────────────────────────────────────────────────┘
```

## Implementation Phases

| Phase | Goal | Expected Performance |
|-------|------|---------------------|
| **0. Setup** | Environment, toolchain, baseline measurements | -- |
| **1. Minimal Engine** | Load model, naive C forward pass, correct output | 5-15 tok/s |
| **2. SIMD Kernels** | AVX2/AVX-VNNI ternary matmul, optimized norms | 30-50 tok/s |
| **3. Threading** | P-core/E-core affinity, work-stealing thread pool | 50-80 tok/s |
| **4. iGPU Offload** | Iris Xe for attention, double-buffered CPU-iGPU pipeline | 70-100 tok/s |
| **5. Advanced** | Speculative decoding, FP8 KV cache, NVMe streaming | 100-120+ tok/s |
| **6. Paper** | Benchmarks, ablation study, write and submit paper | -- |

## Key Research Findings So Far

From the completed research sections:

- **AVX-VNNI on Alder Lake matches AVX-512 VNNI throughput** -- no penalty for lacking AVX-512 (zero performance left on the table)
- **Intel researchers achieved 2.2x over bitnet.cpp** using VNNI-optimized ternary kernels on Alder Lake (arXiv:2508.06753)
- **VPDPBUSD eliminates the INT16 saturation bug** present in the standard VPMADDUBSW chain
- **TQ2_0 ternary format achieves 141 GB/s on AVX2** -- 2.2x faster than Q4_K quantization
- **Dense SIMD beats sparse skip-based approaches** at BitNet's 40% zero-weight sparsity
- **MAD-based kernels outperform LUT-based on x86 AVX2** (3.7ns vs 6.2ns -- opposite of ARM)
- **E-core avoidance yields 57% speedup** on Alder Lake (validated by llamafile)
- **No AVX2 frequency throttling** on Alder Lake (eliminated since Rocket Lake)
- **Golden Cove has 2 FMA/cycle + 3 loads/cycle** -- NOT load-bottlenecked from L1 cache
- **GFNI instructions available on Alder Lake** for single-instruction bit permutations in weight unpacking

## Design Principles

- **Zero dependencies** -- custom C engine, no Python, no frameworks, no external runtimes
- **Hardware-specific** -- written FOR the i7-12700H, not abstracting hardware away
- **Ternary-first** -- architecture assumes {-1, 0, +1} weights as the primary format
- **Memory hierarchy aware** -- explicitly manage L1/L2/L3/RAM/NVMe tiers
- **Async everything** -- overlap I/O, compute, and iGPU work
- **Measure before optimizing** -- Intel VTune profiling at every stage, one experiment at a time

## Target Models

| Model | Params | Size | Confidence for 100+ tok/s |
|-------|--------|------|--------------------------|
| BitNet-b1.58-2B-4T | 2.4B | 0.4 GB | HIGH |
| bitnet_b1_58-3B | 3.3B | 0.6 GB | MEDIUM (needs speculative decoding) |
| Llama-3.2-3B (custom ternary) | 3.2B | ~0.6 GB | Stretch goal |

## The Math

```
tokens/sec = memory_bandwidth / model_size_per_token

DDR4-3200 real bandwidth:  ~40 GB/s
BitNet 3B model size:      ~0.6 GB
Theoretical baseline:      ~67 tok/s

+ L3 cache hits for hot layers:     effectively 200+ GB/s for cached data
+ Software prefetching:              hides 30-50% of memory latency
+ iGPU offload:                      reduces CPU memory pressure 15-20%
+ Speculative decoding:              2x effective throughput
─────────────────────────────────────────────────────────────────
Estimated achievable:                100-120 tok/s
```

## Inspiration

- [**flash-moe**](https://github.com/danveloper/flash-moe) -- Achieved 5.7 tok/s on a 397B MoE model on MacBook using custom Objective-C + Metal shaders, NVMe streaming via pread(), and 2-bit requantization. We adapt their philosophy (not their code) to Intel/Windows.
- [**bitnet.cpp**](https://github.com/microsoft/BitNet) -- Microsoft's official ternary inference framework. Our primary benchmark baseline.
- [**T-MAC**](https://github.com/microsoft/T-MAC) -- LUT-based low-bit inference. Key insight: VPSHUFB as parallel 16-entry lookup table.
- [**llamafile**](https://github.com/Mozilla-Ocho/llamafile) -- tinyBLAS with 3x4 register blocking and E-core avoidance on Alder Lake.
- [**llama.cpp**](https://github.com/ggml-org/llama.cpp) -- TQ2_0 ternary format, block-interleaved packing.

## Contributing

This project is in early research phase. The engine code has not been written yet -- we are completing the research foundation first to ensure every optimization decision is backed by data.

If you're interested in:
- CPU-optimized LLM inference
- BitNet / ternary neural network deployment
- Intel Alder Lake microarchitecture optimization
- AVX2/AVX-VNNI kernel development

Feel free to open an issue or reach out.

## License

TBD -- will be determined before public release.

---

*Built by Chait. Research ongoing.*
=======
#edgeLM

We are trying to build architecture for running low-parameter (≤3B) language models on laptops with optimized power efficiency.

Our goal includes having an output of 25+ tokens / second from an 8GB memory & a normal intel CPU laptop without any GPU.
>>>>>>> 40b6ff3f36598164aede05ead96c6710a939f0b3





