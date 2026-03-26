# EdgeLM -- Custom Inference Engine for Consumer Intel Hardware

## What this is
Custom inference engine targeting 100-120 tokens/second on a 3B parameter LLM
running on consumer Intel hardware (i7-12700H, 16GB DDR4, Intel Iris Xe iGPU,
no discrete GPU). Written from scratch in C with inline AVX2/AVX-VNNI assembly.
Intended to produce a research paper targeting MLSys/EuroSys 2027.

## Hardware constraints
- **CPU:** i7-12700H (6 P-cores Golden Cove + 8 E-cores Gracemont, 20 threads, AVX2 + AVX-VNNI, NO AVX-512, NO AMX)
- **Memory:** 16GB DDR4-3200 dual-channel (~40 GB/s real bandwidth) -- THE fundamental bottleneck
- **iGPU:** Intel Iris Xe 96 EUs (~2.2 TFLOPS FP32) -- shares DDR4 bus with CPU
- **Storage:** NVMe SSD PCIe 4.0 (~5-6 GB/s sequential)
- **Budget:** max 6-7 GB RAM for inference. Always plugged in.

## The core insight
The memory bandwidth wall makes 100+ tok/s impossible with standard Q4 quantization
(~1.7GB model, capped at ~23 tok/s). The only viable path is BitNet 1.58-bit ternary
weights {-1, 0, +1} (~0.4-0.6GB for 3B) which turns matmul into conditional add/subtract,
combined with custom AVX2/AVX-VNNI kernels, hybrid CPU+iGPU compute, and speculative decoding.

**Bandwidth math:** `tok/s = bandwidth / model_size` => 40 GB/s / 0.6 GB = ~67 tok/s baseline,
plus optimizations to reach 100+.

## Architecture strategy
- **Zero dependencies** -- custom C engine, no Python, no frameworks
- **Hardware-specific** -- written FOR i7-12700H, not abstracting hardware away
- **Ternary-first** -- assumes {-1, 0, +1} weights as primary format
- **Memory hierarchy aware** -- explicitly manage L1/L2/L3/RAM/NVMe tiers
- **Async everything** -- overlap I/O, compute, and iGPU work
- **Language:** C with inline ASM for critical kernels (no STL, no exceptions, no RTTI)

## Target models
- **Primary:** BitNet-b1.58-2B-4T (2.4B params, 0.4 GB) -- highest confidence for 100+ tok/s
- **Secondary:** bitnet_b1_58-3B (3.3B params, 0.6 GB) -- true 3B target
- **Stretch:** Custom ternary quantization of Llama-3.2-3B
- **Baseline:** Llama-3.2-3B-Instruct Q4_K_M via llama.cpp (5-7 tok/s on this hardware)

## Key optimization tiers (ranked by impact)
1. **Custom AVX2/AVX-VNNI ternary matmul kernels** (2-3x over BitNet's kernels)
2. **Explicit memory prefetching** (20-40% bandwidth improvement)
3. **Thread affinity & P-core/E-core scheduling** (15-25% improvement)
4. **Hybrid CPU+iGPU pipeline** for attention offload (15-30%)
5. **Optimized KV cache** with FP8 quantization (10-20%)
6. **Speculative decoding** with draft model on E-cores (1.5-2.5x effective throughput)

## Key components
- Tokenizer (custom C BPE, <5ms for 1000 tokens)
- GGUF/custom model loader (mmap + weight repacking)
- Weight manager (ternary 2-bit packing, SIMD-aligned, prefetching)
- Transformer pipeline (RMSNorm, Attention w/ GQA, FFN w/ SiLU)
- KV cache (ring buffer, FP8 quantized, 64-byte aligned)
- Token sampler (top-p, top-k, temperature, repetition penalty)
- Thread pool (explicit P-core/E-core affinity, work-stealing)
- iGPU scheduler (OpenCL/Level Zero for attention offload)
- Memory manager (VirtualAlloc, 2MB large pages, arena allocation)

## Implementation phases
1. **Phase 0:** Environment setup (VS2022, Intel oneAPI, model downloads)
2. **Phase 1:** Minimal viable engine (naive C, ~5-15 tok/s)
3. **Phase 2:** AVX2 kernel optimization (~30-50 tok/s)
4. **Phase 3:** Multi-threading & core affinity (~50-80 tok/s)
5. **Phase 4:** iGPU offloading (~70-100 tok/s)
6. **Phase 5:** Advanced optimizations (~100-120+ tok/s)
7. **Phase 6:** Paper & polish

## Key documents
- `implementation-plan.md` -- Implementation roadmap, 6 phases, benchmarking methodology, paper structure
- `deep-dive.md` -- Exhaustive 25-section reference covering the entire inference pipeline
- `research-papers-data.json` -- 100 structured technique/paper entries from prior research
- `research-progress.md` -- Tracks research completion status for all 25 deep dive sections
- `research/` -- Per-section deep research files with optimizations beyond the deep dive

## Research workflow
Run `/edgelm-layer-research` to research the next pending section from the deep dive.
The command reads `research-progress.md`, identifies the next unresearched section, performs deep
internet research, writes a structured research file to `research/`, and updates progress.
Each session covers one section. 25 sections total.

## Critical rules
- NEVER run multiple experiments simultaneously -- max 6-7 GB RAM
- NEVER auto-launch heavy processes -- always ask before running
- ONE experiment at a time, record results, analyze, then proceed
- Do NOT target AVX-512 -- fused off on Alder Lake, will crash
- Do NOT use HyperThreading for SIMD workloads -- 6 physical P-core threads optimal
- Do NOT overlap CPU and iGPU memory-intensive work -- time-slice instead (shared DDR4 bus)
- Do NOT use Python in the hot path (3+ second startup penalty)
- Do NOT fork llama.cpp or BitNet -- custom engine, study their code for reference only
- Use 14 physical threads (6P + 8E), not 20 logical threads
- All weight buffers must be 64-byte aligned for AVX2
- Use 2MB large pages (VirtualAlloc + MEM_LARGE_PAGES) to reduce TLB pressure
