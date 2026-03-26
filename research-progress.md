# EdgeLM Research Progress

## Status
- **Total sections:** 25
- **Completed:** 2
- **Remaining:** 23
- **Last updated:** 2026-03-26

## Part I: Hardware Foundation

- [x] **Section 01** | CPU Architecture -- Intel Alder Lake | `research/01-intel-alder-lake-cpu-architecture.md`
- [x] **Section 02** | SIMD Instruction Sets | `research/02-avx2-vnni-simd-optimization.md`
- [ ] **Section 03** | Cache Hierarchy | `research/03-l1-l2-l3-cache-hierarchy-optimization.md`
- [ ] **Section 04** | Memory Subsystem -- The Bandwidth Wall | `research/04-ddr4-memory-bandwidth-wall.md`
- [ ] **Section 05** | iGPU -- Intel Iris Xe | `research/05-intel-iris-xe-igpu-compute.md`
- [ ] **Section 06** | NVMe SSD | `research/06-nvme-pcie4-storage-streaming.md`
- [ ] **Section 07** | Power & Thermal | `research/07-power-thermal-throttling-management.md`

## Part II: Model & Data

- [ ] **Section 08** | Model Architecture Choices | `research/08-bitnet-model-architecture-selection.md`
- [ ] **Section 09** | Model Format & Storage | `research/09-gguf-model-format-weight-packing.md`
- [ ] **Section 10** | Quantization -- The Complete Landscape | `research/10-ternary-quantization-landscape.md`
- [ ] **Section 11** | Model Loading & Memory Allocation | `research/11-model-loading-mmap-memory-mapping.md`
- [ ] **Section 12** | Tokenization | `research/12-bpe-tokenizer-c-implementation.md`

## Part III: The Transformer Forward Pass

- [ ] **Section 13** | The Transformer Forward Pass | `research/13-transformer-forward-pass-pipeline.md`

## Part IV: Core Kernels & Runtime

- [ ] **Section 14** | Matrix Multiplication -- The Core Compute Kernel | `research/14-ternary-matmul-kernel-optimization.md`
- [ ] **Section 15** | Attention Mechanisms | `research/15-grouped-query-attention-mechanisms.md`
- [ ] **Section 16** | KV Cache Management | `research/16-kv-cache-quantization-ring-buffer.md`
- [ ] **Section 17** | Memory Management | `research/17-virtual-memory-large-pages-arena.md`
- [ ] **Section 18** | Threading & Parallelism | `research/18-pcore-ecore-thread-affinity-scheduling.md`

## Part V: Advanced Techniques & Integration

- [ ] **Section 19** | iGPU Offloading | `research/19-opencl-igpu-attention-offload.md`
- [ ] **Section 20** | Decoding Strategies | `research/20-speculative-decoding-draft-verify.md`
- [ ] **Section 21** | Sampling | `research/21-token-sampling-topk-topp-temperature.md`
- [ ] **Section 22** | NVMe / Storage Streaming | `research/22-nvme-kvcache-overflow-streaming.md`
- [ ] **Section 23** | System-Level Tuning | `research/23-windows-kernel-system-tuning.md`
- [ ] **Section 24** | Inference Runtimes Compared | `research/24-llamacpp-bitnet-tmac-runtime-comparison.md`
- [ ] **Section 25** | End-to-End Pipeline Integration | `research/25-end-to-end-inference-pipeline-integration.md`
