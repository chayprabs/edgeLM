# The Complete Deep Dive: Running 3-5B Parameter LLMs on Consumer Intel Hardware

## Every Step, Every Approach, Every Tradeoff

**Target Hardware:** Intel i7-12700H | 16GB DDR4-3200 | Intel Iris Xe 96 EUs | NVMe SSD | Windows 11 | No discrete GPU

**Purpose:** Exhaustive reference covering the ENTIRE inference pipeline from silicon to output token, with ALL possible approaches at each step — not just the "best" one.

---

# PART I: HARDWARE FOUNDATION

---

## Section 1: CPU Architecture — Intel Alder Lake (12th Gen)

### 1.1 The Hybrid Architecture

The i7-12700H uses Intel's big.LITTLE-inspired hybrid design — the first x86 chip to ship two fundamentally different core types:

| Property | P-Cores (Golden Cove) | E-Cores (Gracemont) |
|---|---|---|
| Count | 6 physical (12 threads with HT) | 8 physical (8 threads, no HT) |
| Decode width | 6 uops/cycle | 2 clusters of 2-wide decode |
| Out-of-order window | 512-entry ROB | ~256-entry ROB |
| Pipeline depth | ~20+ stages | ~16 stages |
| Base clock | 2.3 GHz | 1.7 GHz |
| Max turbo (single) | 4.7 GHz | 3.5 GHz |
| Max turbo (all-core) | 4.1 GHz | 3.5 GHz |
| L1D cache | 48 KB per core | 32 KB per cluster (4 cores share) |
| L2 cache | 1.25 MB per core | 2 MB per cluster (4 cores share) |
| AVX2 support | Yes (256-bit, full speed) | Yes (256-bit, full speed) |
| AVX-VNNI | Yes | Yes |
| AVX-512 | **NO** (fused off) | **NO** |
| AMX | **NO** | **NO** |

**Total cores:** 6P + 8E = 14 cores, 20 threads

**Critical for LLM inference:** The P-cores are where heavy SIMD matmul work should run. E-cores are for orchestration, I/O, and lightweight tasks. Scheduling matters enormously.

### 1.2 Intel Thread Director

Hardware-assisted thread scheduling via the Enhanced Hardware Feedback Interface (EHFI):
- The CPU itself classifies workloads and hints the OS scheduler about P-core vs E-core preference
- Windows 11's scheduler integrates with Thread Director
- **Problem:** Thread Director does NOT understand LLM inference workloads well. It may assign SIMD-heavy threads to E-cores, causing 30-50% slowdowns
- **Solution:** Manual thread affinity pinning (see Section 18)

### 1.3 HyperThreading Impact on SIMD

Each P-core has 2 logical threads via HyperThreading (HT). For SIMD-heavy workloads:
- Two SIMD threads on the same P-core **share** the execution ports
- AVX2 FMA uses ports 0 and 1 — two threads competing for these ports get ~60% throughput each (not 50%, due to interleaving)
- **For matmul:** HT provides ~10-20% extra throughput per core at best, but adds scheduling complexity
- **Recommendation from BitNet benchmarks:** Using 6 threads (one per P-core, no HT) often outperforms 12 threads (two per P-core with HT) for SIMD-bound work

### 1.4 Execution Ports (Golden Cove P-Core)

Understanding which instructions go where determines how to interleave operations:

| Port | Units | Key Instructions |
|---|---|---|
| Port 0 | FMA, FADD, AES, AVX-VNNI | VFMADD231PS, VPDPBUSD |
| Port 1 | FMA, FADD, integer multiply | VFMADD231PS, VPMULLW |
| Port 2 | Load (AGU) | VMOVDQA load |
| Port 3 | Load (AGU) | VMOVDQA load |
| Port 4 | Store data | VMOVDQA store |
| Port 5 | Shuffle, permute, shift | VPSHUFB, VPERMD, VPSLLW |
| Port 6 | Branch, integer ALU | JMP, CMP |
| Port 7 | Store address (AGU) | Store address calc |
| Port 8 | Load (AGU) | VMOVDQA load |
| Port 9 | Store address (AGU) | Store address calc |
| Port 10 | Integer ALU, vector int ALU | VPADDB, VPAND, VPOR |

**Key insight for ternary kernels:**
- VPDPBUSD (AVX-VNNI dot product): Port 0 only, throughput = 1/cycle
- VPSHUFB (LUT lookup): Port 5 only, throughput = 1/cycle
- VPAND/VPOR (bitmask ops): Port 0 or Port 10, throughput = 2/cycle
- Loads: Ports 2, 3, 8 = 3 loads/cycle (96 bytes/cycle from L1)

This means you can theoretically do 1 VPDPBUSD + 1 VPSHUFB + 2 VPAND + 3 loads per cycle if the pipeline is fully fed.

---

## Section 2: SIMD Instruction Sets — Complete Inventory

### 2.1 What's Available on i7-12700H

| ISA Extension | Register Width | Available? | Notes |
|---|---|---|---|
| SSE4.2 | 128-bit (XMM) | Yes | Legacy, use for scalar/string ops |
| AVX2 | 256-bit (YMM) | Yes | **Primary workhorse** |
| FMA3 | 256-bit (YMM) | Yes | Fused multiply-add |
| AVX-VNNI | 256-bit (YMM) | Yes | **INT8 dot product** |
| AVX-512 | 512-bit (ZMM) | **NO** | Fused off on Alder Lake |
| AMX | Tile registers | **NO** | Server-only (Sapphire Rapids+) |
| BMI1/BMI2 | Scalar | Yes | Bit manipulation |
| POPCNT | Scalar | Yes | Population count |

### 2.2 AVX2 — The Workhorse Instructions for Inference

#### Integer Arithmetic
```
VPADDB/W/D/Q   — packed add (byte/word/dword/qword)
VPSUBB/W/D/Q   — packed subtract
VPMULLW/D      — packed multiply low (word/dword)
VPMULHW/UW     — packed multiply high (signed/unsigned word)
VPABSB/W/D     — packed absolute value
```

#### The Critical Multiply-Accumulate Chain (for INT8 quantized inference)
```
VPMADDUBSW ymm, ymm  — multiply 32 unsigned*signed bytes -> 16 signed words, add adjacent pairs
                        Throughput: 1/cycle (Port 0)
                        Latency: 5 cycles

VPMADDWD ymm, ymm    — multiply 16 signed words -> 8 signed dwords, add adjacent pairs
                        Throughput: 1/cycle (Port 0)
                        Latency: 5 cycles

VPADDD ymm, ymm      — add 8 packed dwords
                        Throughput: 2/cycle (Port 0, 10)
                        Latency: 1 cycle
```
**This 3-instruction chain processes 32 INT8 multiply-accumulates.**

#### Shuffle and Permute (Critical for LUT-based approaches)
```
VPSHUFB ymm, ymm     — byte-level shuffle within 128-bit lanes (THE LUT instruction)
                        Throughput: 1/cycle (Port 5)
                        Latency: 1 cycle
                        Performs 32 parallel 4-bit-indexed lookups from a 16-byte table

VPERMD ymm, ymm      — dword-level cross-lane permute
VPERMPS ymm, ymm     — float cross-lane permute
VPERM2I128 ymm, ymm  — swap/blend 128-bit lanes
```

#### Bitwise Operations (Critical for ternary mask-based approaches)
```
VPAND ymm, ymm       — AND (apply masks)
VPANDN ymm, ymm      — AND-NOT (apply inverted masks)
VPOR ymm, ymm        — OR (combine results)
VPXOR ymm, ymm       — XOR
                        All: Throughput 2/cycle (Port 0, 10), Latency 1 cycle
```

#### Comparison and Blend (Conditional operations for ternary)
```
VPCMPEQB/W/D         — compare equal, produce mask
VPCMPGTB/W/D         — compare greater than, produce mask
VPBLENDVB ymm, ymm, ymm — variable byte blend based on mask
                           Throughput: 1/cycle (Port 0 or 5)
                           Latency: 2 cycles
```

#### Float Operations (for activations, norms, softmax)
```
VFMADD231PS ymm, ymm — fused multiply-add: dst = dst + src1 * src2
                        Throughput: 2/cycle (Port 0, 1)
                        Latency: 4 cycles

VMULPS / VADDPS       — multiply / add floats
VRSQRTPS              — fast approximate reciprocal sqrt (for RMSNorm)
VRCPPS                — fast approximate reciprocal
```

#### Horizontal Reductions (for dot products, norms)
```
VHADDPS               — horizontal add floats (slow, avoid in hot loop)
VPHADDD               — horizontal add ints (slow)
Better approach:       — use VPERMD + VPADDD for cross-lane reduction
```

### 2.3 AVX-VNNI — The INT8 Dot Product Accelerator

The single most important instruction for quantized inference on this CPU:

```
VPDPBUSD ymm, ymm, ymm — Dot Product of unsigned Bytes and signed Bytes, Unsigned accumulate to Signed Dword

    For each of 8 dword lanes:
      dst[i] += src1.byte[4i+0] * src2.byte[4i+0]
              + src1.byte[4i+1] * src2.byte[4i+1]
              + src1.byte[4i+2] * src2.byte[4i+2]
              + src1.byte[4i+3] * src2.byte[4i+3]

    Throughput: 1/cycle (Port 0)
    Latency: 5 cycles
    Processes: 32 byte multiply-accumulates per instruction
```

**Why this matters for ternary:**
- Pack ternary weights as INT8 values {-1, 0, +1} in src2
- Pack activations as UINT8 (with bias offset) in src1
- One VPDPBUSD computes 32 elements of the dot product
- Replaces the 3-instruction VPMADDUBSW+VPMADDWD+VPADDD chain with 1 instruction

**Comparison: VPDPBUSD vs VPMADDUBSW chain**

| Metric | VPDPBUSD | VPMADDUBSW chain |
|---|---|---|
| Instructions needed | 1 | 3 |
| Elements per batch | 32 | 32 |
| Throughput | 1/cycle | 1/cycle (bottleneck on VPMADDUBSW) |
| Accumulator width | 32-bit (direct) | Need explicit VPADDD |
| Code complexity | Simple | More complex |

On this CPU, VPDPBUSD is ~2-3x better in instruction throughput for INT8 dot products.

### 2.4 What's NOT Available (and What It Would Give Us)

**AVX-512 (if it were available):**
- 512-bit registers = 64 INT8 ops per instruction
- VNNI-512: VPDPBUSD with ZMM = 64 multiply-accumulates per instruction
- Would roughly 2x throughput over AVX2
- Available on: Intel 11th gen (desktop Tiger Lake), server chips

**AMX (Advanced Matrix Extensions, Sapphire Rapids+):**
- Tile-based matrix multiply: 8 tile registers, each up to 1KB
- Can compute 16x64 * 64x16 INT8 matrix multiply in ~16 cycles
- ~2048 INT8 ops/cycle — would make inference compute-bound instead of memory-bound
- Available on: 4th gen Xeon Scalable only

**AVX10.2 (future, Granite Rapids+):**
- Unified 256-bit AVX-512 subset that works on both P and E cores
- Would give us AVX-512 features in 256-bit form

### 2.5 ARM NEON and Apple Silicon (for comparison)

Why Apple Silicon is fast at inference:
- **NEON SDOT:** 128-bit, processes 16 INT8 dot products per instruction
- **Apple AMX:** Proprietary matrix coprocessor, ~4096 INT8 ops/cycle
- **Unified memory:** 200+ GB/s on M3 Max vs our 40 GB/s DDR4
- **This is why llama.cpp on M2 gets 40+ tok/s** — bandwidth, not compute

---

## Section 3: Cache Hierarchy

### 3.1 Cache Structure on i7-12700H

| Level | Size (P-core) | Size (E-core) | Latency | Line Size | Associativity |
|---|---|---|---|---|---|
| L1 Data | 48 KB per core | 32 KB per 4-core cluster | ~4 cycles (~1 ns) | 64 bytes | 12-way |
| L1 Instruction | 32 KB per core | 64 KB per 4-core cluster | — | 64 bytes | 8-way |
| L2 Unified | 1.25 MB per core | 2 MB per 4-core cluster | ~14 cycles (~3.5 ns) | 64 bytes | 10-way |
| L3 (LLC) | 24 MB shared across ALL cores | Same | ~40-50 cycles (~10-13 ns) | 64 bytes | 12-way |

**Total L2 for 6 P-cores:** 6 x 1.25 MB = 7.5 MB
**Total L2 for 8 E-cores:** 2 x 2 MB = 4 MB

### 3.2 TLB (Translation Lookaside Buffer)

| TLB Level | Entries (P-core) | Coverage (4KB pages) | Coverage (2MB large pages) |
|---|---|---|---|
| L1 DTLB | 96 entries | 384 KB | 192 MB |
| L2 STLB | 2048 entries | 8 MB | 4 GB |

**Why TLB matters for LLM inference:**
- A 3B ternary model is ~600 MB. With 4KB pages, that's 153,600 pages
- L2 STLB has 2048 entries — can only cover 8 MB of the model at 4KB pages
- With 2 MB large pages: 300 pages needed, well within STLB capacity
- **TLB misses cost ~20-50 cycles each** — massive penalty for random-ish weight access
- **Using large pages can eliminate TLB misses entirely** for models under 4 GB

### 3.3 Cache Behavior During Inference

During single-token decode (GEMV):
- **Input activation vector:** ~12.8 KB (3200 floats at FP32) — fits in L1
- **Output activation vector:** ~12.8 KB — fits in L1
- **Weight matrix per layer:** Millions of elements — does NOT fit in any cache
- **KV cache per layer:** Depends on seq_len, typically 10s of KB to MBs

**The fundamental pattern:** Weights are streamed from RAM through cache, used once, then evicted. The cache hierarchy matters mainly for:
1. Keeping activation vectors hot (they're reused for each output element)
2. Keeping KV cache hot for attention
3. TLB coverage for weight access

### 3.4 Hardware Prefetcher Behavior

Golden Cove has several hardware prefetchers:
- **L2 Streamer:** Detects sequential access patterns, prefetches up to 16 cache lines ahead
- **L1 Streamer:** Prefetches 2 lines ahead into L1
- **Stride prefetcher:** Detects regular stride patterns (e.g., accessing every 3200th byte)
- **AMP (Adjacent Cache Line Prefetch):** Prefetches the other half of a 128-byte pair

**For GEMV (sequential weight scan):** Hardware prefetchers work well but cannot fully hide DDR4 latency (~50-70 ns). Software prefetch instructions can help by:
1. Prefetching further ahead than hardware does
2. Guiding prefetch to the correct cache level
3. Initiating prefetch for non-sequential patterns (e.g., next-layer weights)

---

## Section 4: Memory Subsystem — THE BANDWIDTH WALL

### 4.1 DDR4-3200 Specifications

| Spec | Value |
|---|---|
| Configuration | Dual-channel DDR4-3200 (PC4-25600) |
| Theoretical peak bandwidth | 2 channels x 8 bytes x 3200 MHz = **51.2 GB/s** |
| Real-world sequential read | **38-42 GB/s** (75-82% efficiency) |
| Real-world random read | **15-25 GB/s** (depends on pattern) |
| Latency (idle) | ~50-55 ns |
| Latency (loaded) | ~65-80 ns |
| Burst length | 8 (64 bytes per access = 1 cache line) |
| Capacity | 16 GB (2 x 8 GB, likely single rank) |

### 4.2 The Bandwidth Wall — Why This Is THE Constraint

During autoregressive decode, every generated token requires reading the ENTIRE model weights once:

| Quantization | Model Size (3B) | Theoretical Max tok/s at 40 GB/s |
|---|---|---|
| FP32 | 12 GB | Won't fit in RAM |
| FP16 | 6 GB | 6.7 tok/s |
| INT8 | 3 GB | 13.3 tok/s |
| INT4 (Q4_K_M) | ~1.9 GB | 21 tok/s |
| INT3 | ~1.4 GB | 28.6 tok/s |
| INT2 | ~0.95 GB | 42 tok/s |
| Ternary 1.58-bit | ~0.6 GB | **67 tok/s** |
| Binary 1-bit | ~0.38 GB | **105 tok/s** |

**This is why ternary quantization is the path to 100+ tok/s.** No amount of kernel optimization can overcome the bandwidth wall with larger quantizations. Q4_K_M is capped at ~21 tok/s even with perfect bandwidth utilization.

### 4.3 The Math in Detail

```
tokens_per_second = memory_bandwidth / model_bytes_per_token

For ternary 3B (0.6 GB):
  tokens_per_second = 40 GB/s / 0.6 GB = 66.7 tok/s (theoretical ceiling)

But model bytes includes more than just weights:
  - Weights: 0.6 GB
  - Embedding table (FP16): ~80 MB (if not ternary)
  - Output head (FP16): ~200 MB (if not ternary)
  - KV cache reads: ~1-5 MB per token (depends on seq_len)
  - Activation intermediates: negligible (fit in cache)

Adjusted: 40 GB/s / (0.6 + 0.08 + 0.2 + 0.003) GB ≈ 45 tok/s

To reach 100 tok/s, we need one or more of:
  1. Speculative decoding (1.5-2.5x multiplier → 67-112 tok/s effective)
  2. Ternary embedding + output head (reduce from 0.88 to 0.6 GB → 67 tok/s)
  3. Better bandwidth utilization (>42 GB/s real)
  4. iGPU handling some compute (freeing CPU bandwidth for weights)
```

### 4.4 Bandwidth Sharing Between CPU and iGPU

**Critical constraint:** CPU and Iris Xe iGPU share the same DDR4 bus.

- If both are active: bandwidth splits roughly proportional to demand
- CPU reading weights at 35 GB/s + iGPU at 5 GB/s = 40 GB/s total (OK)
- CPU at 30 GB/s + iGPU at 15 GB/s = both slow down (BAD for matmul)

**This means iGPU offloading must be done carefully** — it can only help if the compute it does saves more CPU time than the bandwidth it steals.

---

## Section 5: iGPU — Intel Iris Xe (96 EUs)

### 5.1 Architecture

| Spec | Value |
|---|---|
| Execution Units | 96 EUs |
| Subslices | 6 (16 EUs each) |
| FP32 throughput | ~1.5 TFLOPS |
| FP16 throughput | ~3 TFLOPS |
| INT8 throughput | ~6 TOPS |
| Shared L3 cache | 3.8 MB (separate from CPU L3) |
| Local shared memory | 64 KB per subslice |
| Memory | Shares DDR4 with CPU |
| Max allocatable memory | Configured in BIOS (64-512 MB DVMT) |

### 5.2 Available Compute APIs

| API | Maturity | Overhead | Notes |
|---|---|---|---|
| **OpenCL 3.0** | High | ~10-50 us/dispatch | Most stable, best documented for Intel |
| **SYCL/oneAPI (Level Zero)** | Medium | ~5-20 us/dispatch | Intel's preferred, requires oneAPI toolkit |
| **Vulkan Compute** | Medium | ~5-15 us/dispatch | Cross-platform, most complex to program |
| **DirectX 12 Compute** | Medium | ~5-20 us/dispatch | Windows-native, best DirectStorage integration |
| **OpenGL Compute** | Low | ~20-50 us/dispatch | Legacy, not recommended for new work |

### 5.3 When iGPU Helps vs Hurts

**Helps:**
- Operations that are compute-bound with small data transfer (softmax, RoPE, small reductions)
- Attention score computation for long sequences (>512 tokens)
- Overlapping GPU compute with CPU compute (pipeline parallelism)

**Hurts:**
- Anything bandwidth-bound (weight matmul) — iGPU steals CPU's bandwidth
- Small operations where dispatch overhead (~10-50 us) exceeds compute time
- When total CPU+iGPU bandwidth demand exceeds 40 GB/s

---

## Section 6: NVMe SSD

### 6.1 Typical PCIe 4.0 NVMe Specs

| Spec | Value |
|---|---|
| Sequential read | 3.5-7 GB/s |
| Random read (4KB, QD32) | ~500K-1M IOPS = 2-4 GB/s |
| Random read (4KB, QD1) | ~15K IOPS = 60 MB/s |
| Latency | ~30-80 us (vs ~50 ns for DRAM) |
| Interface | PCIe 4.0 x4 |

### 6.2 When NVMe Matters for Inference

- **3B ternary model (0.6 GB):** Fits entirely in RAM. NVMe only matters for initial load (~0.1s at 5 GB/s)
- **Larger models (7B+ at Q4):** May need NVMe streaming if model exceeds available RAM
- **KV cache overflow:** For very long contexts, old KV entries can be paged to NVMe
- **MoE models:** Only active experts loaded per token (flash-moe approach)

For the target 3B ternary case, NVMe is NOT on the critical path during inference.

---

## Section 7: Power & Thermal

### 7.1 i7-12700H Power Limits

| Parameter | Value |
|---|---|
| PL1 (sustained power) | 45W |
| PL2 (burst power) | 115W |
| Tau (burst duration) | 28-56 seconds |
| Package max temp | 100C (throttles above this) |

**Impact on inference:**
- Short benchmarks may run at PL2 (115W turbo) — giving inflated numbers
- Sustained inference at PL1 (45W) — P-cores drop to ~3.5-3.8 GHz all-core
- Thermal throttling in a laptop with poor cooling can drop further
- **Always benchmark sustained (>2 minutes) for real numbers**

### 7.2 Power Management Settings

- Set Windows power plan to "Best Performance"
- In BIOS: Disable Intel SpeedStep/SpeedShift if you want locked high frequency (increases power, may help consistency)
- Keep laptop plugged in (battery mode caps PL1 to ~35W on many models)

---

# PART II: MODEL & DATA

---

## Section 8: Model Architecture Choices

### 8.1 Standard Transformer (Decoder-Only)

The dominant architecture for LLMs. Used by GPT, Llama, Qwen, Phi, Mistral, etc.

**Structure per layer:**
1. RMSNorm
2. Multi-head self-attention (Q, K, V projections → attention → output projection)
3. Residual connection
4. RMSNorm
5. Feed-forward network (gate → up → activation → down)
6. Residual connection

**Properties:**
- Well-understood, battle-tested
- Attention is O(n²) in sequence length (n)
- All weight matrices are dense — bandwidth-bound during decode
- KV cache grows linearly with sequence length

### 8.2 BitNet / Ternary Transformer

Same architecture as standard transformer, but weights are constrained to {-1, 0, +1} during training.

**Key difference:** Weight matrices are ternary, so multiply-accumulate becomes conditional add/subtract. No floating-point multiply needed for weight × activation.

**Available models at ~3B scale:**
- BitNet-b1.58-2.4B (Microsoft Research)
- BitNet-b1.58-3.3B (Microsoft Research)
- Llama3-BitNet-3B (community conversion — lower quality)

**Tradeoffs:**
- 4-5x smaller than FP16 equivalent
- ~5-10% quality degradation vs FP16 at same parameter count
- Custom kernels needed — standard BLAS doesn't help
- Not as many model choices available

### 8.3 Mamba / State Space Models (SSMs)

**Architecture:** Replaces attention with selective state space blocks. No KV cache needed.

**Structure per layer:**
1. Norm
2. Linear projection (expand)
3. Conv1D (short convolution)
4. Selective SSM (the core innovation)
5. Linear projection (contract)
6. Residual connection

**Properties:**
- O(1) memory per token during generation (no KV cache)
- O(n) compute instead of O(n²) for long sequences
- Hardware-aware parallel scan algorithm (efficient on GPU, tricky on CPU)
- Quality: competitive with transformers up to ~3B, slightly behind at 7B+

**Available at ~3B:**
- Mamba-2.8B (original)
- Mamba-2 (improved architecture)
- Jamba (hybrid Mamba+Transformer, but only at 52B)

**For this project:**
- Pro: No KV cache → more RAM for other things, constant inference cost regardless of context length
- Con: Fewer model choices, less mature tooling, not ternary-trained, CPU scan kernel is complex

### 8.4 RWKV

**Architecture:** Linear attention alternative based on recurrent formulation.

**Core mechanism (WKV operator):**
```
WKV = Σ(e^(w*t + k) * v) / Σ(e^(w*t + k))
```
Can be computed recurrently: constant memory per token, like an RNN.

**Properties:**
- O(1) memory per token (fixed-size state)
- Training can be parallelized (unlike RNN)
- Quality: competitive with transformers at 1.5-7B
- Multiple versions: RWKV-4, RWKV-5 (Eagle), RWKV-6 (Finch)

**Available at ~3B:**
- RWKV-6-3B (Finch)
- RWKV-5-3B (Eagle)

**For this project:**
- Pro: No KV cache, constant inference cost, available at 3B, active community
- Con: Not ternary-trained, fewer GGUF quantizations available, quality slightly below Llama-3 at same size

### 8.5 Mixture of Experts (MoE)

**Architecture:** FFN layer replaced with multiple "expert" FFNs. A router selects top-K experts per token.

**Properties:**
- Total parameters >> active parameters per token
- e.g., Mixtral 8x7B: 47B total params, 13B active per token
- Only need to load active expert weights → reduces bandwidth per token

**For this project:**
- Pro: More parameters for the same inference cost (if experts fit in RAM)
- Con: Total model size is large, need all experts stored (even if only 2 active), complex routing, no ternary MoE models available at practical sizes
- flash-moe approach: stream experts from NVMe, but this adds I/O latency

### 8.6 Hybrid Architectures

Emerging: models that combine transformer layers with Mamba layers.

- **Jamba (AI21):** Transformer + Mamba hybrid — only at 52B
- **Zamba (Zyphra):** Hybrid with shared attention layers — 2.7B available
- **StripedHyena:** Hyena + attention hybrid

**For this project:** Interesting research direction, but very limited model availability at 3B, and no ternary variants.

### 8.7 Model Architecture Comparison for Edge Inference

| Architecture | KV Cache | Compute/Token | Quality (3B) | Ternary Available | Tooling Maturity | Edge Suitability |
|---|---|---|---|---|---|---|
| Transformer | Yes (grows) | O(n) per layer | Best | Yes (BitNet) | Excellent | Good (with GQA) |
| Mamba/SSM | No (fixed state) | O(1) per layer | Good | No | Medium | Very Good |
| RWKV | No (fixed state) | O(1) per layer | Good | No | Medium | Very Good |
| MoE | Yes + expert weights | O(1) active | Best (more params) | No | Medium | Complex |
| Hybrid | Reduced | Varies | Very Good | No | Low | Good |

---

## Section 9: Model Format & Storage

### 9.1 GGUF (GPT-Generated Unified Format)

**Origin:** Successor to GGML format, designed by llama.cpp project.

**Structure:**
```
┌─────────────────────┐
│ Magic: "GGUF"       │ 4 bytes
│ Version: 3          │ 4 bytes
│ Tensor count         │ 8 bytes
│ Metadata KV count    │ 8 bytes
├─────────────────────┤
│ Metadata Key-Values  │ Variable (model name, architecture,
│                      │  vocab, tokenizer data, quantization info)
├─────────────────────┤
│ Tensor Info Array    │ (name, dimensions, type, offset) per tensor
├─────────────────────┤
│ Alignment Padding    │ To 32-byte boundary
├─────────────────────┤
│ Tensor Data          │ Raw quantized weight data (bulk of file)
└─────────────────────┘
```

**Quantization types in GGUF:**
- Q4_0, Q4_1, Q5_0, Q5_1 — simple block quantization
- Q4_K_S, Q4_K_M, Q5_K_S, Q5_K_M, Q6_K — k-quant (higher quality, variable precision)
- Q8_0, Q8_1 — 8-bit quantization
- IQ1_S, IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S, IQ2_M, IQ3_XXS, IQ3_XS, IQ3_S, IQ4_XS, IQ4_NL — importance-based quantization (lowest bits)
- F16, F32, BF16 — full precision
- Q2_K, Q3_K_S, Q3_K_M, Q3_K_L — 2-3 bit k-quants

**Pros:** Ecosystem standard, self-contained (includes tokenizer), easy to mmap, wide tool support
**Cons:** No native ternary type, quantization is post-training (PTQ not QAT)

### 9.2 SafeTensors (HuggingFace)

**Structure:** Header (JSON) + raw tensor data. Simple, safe (no pickle).

```
┌─────────────────────┐
│ Header length (u64)  │ 8 bytes
├─────────────────────┤
│ JSON Header          │ {"tensor_name": {"dtype": "F16",
│                      │   "shape": [3200, 3200],
│                      │   "data_offsets": [start, end]}}
├─────────────────────┤
│ Raw tensor data      │ Contiguous, in declared dtype
└─────────────────────┘
```

**Used by:** HuggingFace hub (default format), BitNet models store weights as SafeTensors

**Pros:** Simple to parse, safe (no code execution), standard on HuggingFace
**Cons:** No built-in tokenizer, no quantization metadata, separate tokenizer.json needed

### 9.3 ONNX (Open Neural Network Exchange)

**Structure:** Protobuf-based computation graph with embedded weights.

**Pros:** Cross-framework, hardware-agnostic, many runtime options (ONNX Runtime, DirectML)
**Cons:** Large files, complex format, slower to load than binary formats, graph overhead

### 9.4 Custom Binary Format (.edgelm)

For a from-scratch engine, you can design an optimal format:

```
┌─────────────────────┐
│ Magic: "EDLM"       │
│ Version              │
│ Model architecture   │ (layer count, dims, head count, etc.)
├─────────────────────┤
│ Tokenizer data       │ (vocab + BPE merges, embedded)
├─────────────────────┤
│ Per-layer metadata   │ (weight shapes, quantization info, offsets)
├─────────────────────┤
│ Weight data          │ Pre-packed in SIMD-optimal layout:
│                      │ - 2-bit packed ternary with sign+zero masks
│                      │ - Aligned to 32/64 bytes for AVX2
│                      │ - Ordered for sequential streaming
│                      │ - Activation scales co-located with weights
└─────────────────────┘
```

**Pros:** Zero repacking at load (weights pre-packed for target CPU), fastest possible load, minimal parsing
**Cons:** Non-standard, needs converter tool, format tied to specific kernel implementation

### 9.5 Weight Packing Layouts

How ternary weights are physically stored in memory affects kernel performance:

**Layout A: Two-bitmask (sign + nonzero)**
```
For 128 ternary weights:
  nonzero_mask: 128 bits (16 bytes) — bit=1 if weight ≠ 0
  sign_mask:    128 bits (16 bytes) — bit=1 if weight = -1
  Total: 32 bytes for 128 weights = 2 bits/weight
```

**Layout B: Packed 2-bit encoding**
```
For 128 ternary weights:
  Each weight: 2 bits (00=zero, 01=+1, 10=-1)
  128 weights = 256 bits = 32 bytes
  Requires extraction logic (shift+mask) before use
```

**Layout C: INT8 expanded (for VPDPBUSD)**
```
For 32 ternary weights:
  Each weight: 1 byte with value -1, 0, or +1
  32 weights = 32 bytes = 1 YMM register
  Wastes 6 bits per weight but directly usable by VPDPBUSD
  8 bits/weight instead of 2 → 4x more memory bandwidth needed
```

**Layout D: LUT-indexed (for T-MAC approach)**
```
For groups of 4 ternary weights:
  Each group: index into precomputed partial sum table
  3^4 = 81 possible sums → needs 7-bit index
  Packed: 2 indices per byte (with padding)
```

---

## Section 10: Quantization — The Complete Landscape

### 10.1 Why Quantization Matters

Quantization reduces the number of bits per weight. This directly reduces:
1. **Model size** (less storage, faster load)
2. **Bandwidth per token** (THE bottleneck — directly maps to tok/s)
3. **Compute complexity** (INT ops cheaper than FP ops)

At the cost of:
- Quality degradation (lower perplexity, less coherent outputs)
- Potential outlier handling complexity

### 10.2 Full-Precision Formats

#### FP32 (32-bit float)
- **Bits per weight:** 32
- **3B model size:** 12 GB
- **Max tok/s (40 GB/s):** 3.3
- **Quality loss:** None (reference)
- **Use case:** Training only. Impractical for inference on this hardware.

#### FP16 (16-bit float, IEEE 754 half)
- **Bits per weight:** 16
- **3B model size:** 6 GB
- **Max tok/s:** 6.7
- **Quality loss:** ~0.01 perplexity increase (negligible)
- **Use case:** GPU inference with high bandwidth. Impractical on DDR4.

#### BF16 (Brain Float 16)
- **Bits per weight:** 16
- **3B model size:** 6 GB
- **Max tok/s:** 6.7
- **Quality loss:** Slightly more than FP16 (less mantissa precision, same exponent range)
- **Use case:** Training and GPU inference. Same bandwidth as FP16.

#### FP8 (E4M3 or E5M2)
- **Bits per weight:** 8
- **3B model size:** 3 GB
- **Max tok/s:** 13.3
- **Quality loss:** ~0.1-0.5 perplexity increase
- **Use case:** GPU inference (H100+). No native x86 CPU support — must emulate.

### 10.3 Integer Quantization Formats

#### INT8 Quantization
- **Bits per weight:** 8
- **3B model size:** 3 GB
- **Max tok/s:** 13.3
- **Quality loss:** 0.1-0.3 perplexity increase (well-studied)

**Variants:**
| Variant | How It Works | Quality | Compute |
|---|---|---|---|
| Per-tensor symmetric | One scale per entire tensor. w_q = round(w / scale) | Worst | Simplest |
| Per-tensor asymmetric | Scale + zero-point per tensor. w_q = round(w / scale) + zp | Better | +1 subtract |
| Per-channel symmetric | One scale per output row | Good | Same compute, more scales |
| Per-group (g=128) | One scale per 128 weights | Very good | Same compute, many more scales |

**Hardware support:** AVX-VNNI VPDPBUSD natively operates on INT8. This is the sweet spot for x86 CPU inference.

#### INT4 Quantization
- **Bits per weight:** 4
- **3B model size:** ~1.5-1.9 GB (depends on group size and scales overhead)
- **Max tok/s:** 21-27
- **Quality loss:** 0.3-1.0 perplexity increase (noticeable but usable)

**GGML Variants (llama.cpp):**

| Format | Bits | Block Size | Scales | Description |
|---|---|---|---|---|
| Q4_0 | 4.5 | 32 | 1 FP16 per block | Basic round-to-nearest |
| Q4_1 | 5.0 | 32 | 1 FP16 scale + 1 FP16 min | Asymmetric |
| Q4_K_S | 4.5 | 256 | Superblock scales | K-quant, smaller |
| Q4_K_M | 4.85 | 256 | Superblock + min scales | K-quant, medium — **llama.cpp default** |

**Note:** "4.5 bits" means 4 bits for weight + overhead for scale factors, averaging to 4.5 bits per weight.

#### INT3 / INT2 Quantization
- **INT3:** ~1.1 GB for 3B, max ~36 tok/s. Significant quality loss.
- **INT2:** ~0.75-0.95 GB for 3B, max ~42-53 tok/s. Severe quality loss with PTQ.

GGML variants: Q2_K, Q3_K_S, Q3_K_M, Q3_K_L, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_XS

### 10.4 Sub-2-Bit Quantization

#### BitNet 1.58-bit Ternary ({-1, 0, +1})
- **Bits per weight:** ~1.58 (log2(3)) effective, 2 bits stored
- **3B model size:** 0.4-0.6 GB
- **Max tok/s:** 67-100
- **Quality loss:** ~5-10% vs FP16 at same param count (but model is TRAINED this way — QAT not PTQ)

**How it works:**
- During training, weights are constrained to ternary values using a quantization function
- Not post-training quantization — the model learns to work with {-1, 0, +1}
- Activation quantization: inputs quantized to INT8 per-group with absmax scaling
- Result: no floating-point multiply needed. Just add, subtract, or skip.

**Key equations:**
```
w_ternary = RoundClip(w / (|w|_mean + epsilon), -1, +1)
where RoundClip maps to {-1, 0, +1}

Activation quantization (per-group):
x_int8 = Clip(round(x * 127 / |x|_max), -128, 127)
scale = |x|_max / 127
```

#### Binary 1-bit ({-1, +1})
- **Bits per weight:** 1
- **3B model size:** ~0.38 GB
- **Max tok/s:** 105+
- **Quality loss:** Severe. Only viable at very large scales (100B+ parameters)
- **Compute:** XNOR + popcount (extremely fast)
- **Not recommended** for 3B models — quality is unusable

#### IQ (Importance-based Quantization, llama.cpp)
Ultra-low bit quantization with importance-weighted code assignment:

| Format | Bits/weight | Quality | How It Works |
|---|---|---|---|
| IQ1_S | 1.56 | Very low | Lattice-based codebook, importance matrix |
| IQ1_M | 1.75 | Low | Extended codebook |
| IQ2_XXS | 2.06 | Moderate | Importance-weighted 2-bit |
| IQ2_XS | 2.31 | Moderate-good | Extended 2-bit |
| IQ2_S | 2.5 | Good | Standard 2-bit |
| IQ2_M | 2.7 | Good | Medium 2-bit |
| IQ3_XXS | 3.06 | Very good | 3-bit ultra-small |

These use a codebook approach: groups of weights map to entries in an optimized codebook, minimizing error for important weights.

### 10.5 Post-Training Quantization (PTQ) Methods

These take a trained FP16 model and quantize it after training:

#### RTN (Round-to-Nearest)
- **How:** Simply round each weight to nearest quantization level
- **Quality:** Worst PTQ method
- **Speed:** Instant (no calibration needed)
- **What uses it:** Q4_0 in llama.cpp is essentially RTN

#### GPTQ (Generalized Post-Training Quantization)
- **How:** Uses second-order (Hessian) information to minimize quantization error. Processes weight matrix column-by-column, compensating for each column's quantization error in remaining columns
- **Quality:** Very good, especially at 4-bit. Close to FP16 quality
- **Speed:** 3-4 hours for 7B model on GPU (one-time cost)
- **Calibration:** Requires ~128 example sequences
- **Output:** Standard quantized weights (compatible with many runtimes)

#### AWQ (Activation-Aware Weight Quantization)
- **How:** Identifies salient weight channels (those that process large activations) and protects them at higher precision. Only ~1% of channels need protection.
- **Quality:** Best-in-class for 4-bit. Slightly better than GPTQ
- **Speed:** Faster than GPTQ (no Hessian computation)
- **Key insight:** "Which weights matter" depends on activation magnitudes, not weight magnitudes
- **Variants:** AWQ with group quantization, per-channel scaling

#### SmoothQuant (W8A8)
- **How:** Migrates quantization difficulty from activations to weights via per-channel smoothing. Makes both weights AND activations quantizable to INT8
- **Quality:** Near-lossless at W8A8
- **Key insight:** Activations have outlier channels; smoothing shifts the outlier problem to weights (which are easier to quantize)
- **Intel-specific advantage:** AVX-VNNI VPDPBUSD directly handles W8A8

#### SpQR (Sparse Quantization and Representation)
- **How:** Identifies outlier weights, stores them at FP16 (sparse), quantizes rest to 3-4 bits
- **Quality:** Very good, especially for 3-bit
- **Tradeoff:** Sparse outlier storage adds complexity to kernels

#### QuIP / QuIP# (Quantization with Incoherence Processing)
- **How:** Applies random orthogonal transforms to make weights more "incoherent" (uniform) before quantization
- **Quality:** State-of-the-art at 2-bit. Enables usable 2-bit quantization
- **Complexity:** Very high — requires lattice codebooks and Kronecker product decomposition
- **Practical:** QuIP# achieves near-FP16 quality at 2 bits for 7B+ models

#### HQQ (Half-Quadratic Quantization)
- **How:** Solves quantization as a half-quadratic optimization problem. No calibration data needed
- **Quality:** Competitive with GPTQ/AWQ
- **Speed:** Very fast quantization (minutes, not hours)
- **Advantage:** Training-free, no calibration dataset required

### 10.6 Quantization-Aware Training (QAT)

These train the model WITH quantization in the loop:

#### BitNet QAT
- Weights constrained to {-1, 0, +1} during training
- Straight-Through Estimator (STE) for gradient propagation through quantization
- Model learns to compensate for quantization
- **Result:** Much better quality than PTQ at same bit width

#### OneBit (1-bit QAT)
- Trains with binary weights {-1, +1}
- Uses knowledge distillation from FP16 teacher
- Quality better than PTQ binary but still significantly degraded

### 10.7 Mixed-Precision Quantization

Not all layers need the same precision:

- **Embedding layer:** Often kept at FP16 (small relative to total weights, high sensitivity)
- **Output head (LM head):** Often kept at FP16 (directly affects token probabilities)
- **First/last transformer layers:** More sensitive, sometimes kept at higher precision
- **Middle layers:** Most compressible, can use lowest bit width
- **Attention vs FFN:** Some methods quantize attention more aggressively

**Approach:** Profile each layer's sensitivity (e.g., via calibration perplexity), assign precision accordingly.

### 10.8 Quantization Comparison Matrix

| Method | Bits | Quality (3B, ppl) | Calibration | Time to Quantize | Kernel Complexity |
|---|---|---|---|---|---|
| FP16 (baseline) | 16 | Reference | None | None | Standard BLAS |
| RTN Q4_0 | 4.5 | Poor | None | Instant | Simple |
| GPTQ 4-bit | 4 | Very good | 128 samples | Hours (GPU) | Medium |
| AWQ 4-bit | 4 | Best 4-bit | 128 samples | ~1 hour (GPU) | Medium |
| SmoothQuant W8A8 | 8+8 | Near-lossless | 128 samples | Minutes | VNNI direct |
| Q4_K_M (llama.cpp) | 4.85 | Good | None | Instant | GGML custom |
| QuIP# 2-bit | 2 | Good | 128 samples | Hours | Very complex |
| IQ2_XS (llama.cpp) | 2.31 | Moderate | Importance matrix | Minutes | Codebook lookup |
| **BitNet ternary** | **1.58** | **Good (QAT)** | **Part of training** | **Training cost** | **Custom ternary** |
| Binary 1-bit | 1 | Poor | Part of training | Training cost | XNOR+popcount |

---

## Section 11: Model Loading & Memory Allocation

### 11.1 File I/O Approaches

#### Approach A: Standard fread()
```c
FILE *f = fopen("model.bin", "rb");
fread(buffer, 1, file_size, f);
```
- **Speed:** ~2-4 GB/s (limited by C runtime buffering)
- **Pros:** Simple, portable
- **Cons:** Copies data into user buffer, can't take advantage of OS page cache

#### Approach B: Memory-Mapped I/O (mmap / CreateFileMapping)
```c
// Windows
HANDLE hFile = CreateFile("model.bin", GENERIC_READ, ...);
HANDLE hMap = CreateFileMapping(hFile, NULL, PAGE_READONLY, ...);
void *data = MapViewOfFile(hMap, FILE_MAP_READ, 0, 0, 0);
```
- **Speed:** Near-instant mapping, actual reads happen on page fault (~3-5 GB/s for first access, instant if cached)
- **Pros:** Zero-copy (OS maps file pages directly), lazy loading, automatic caching
- **Cons:** Page fault latency spikes, 4KB page granularity (TLB pressure)
- **What llama.cpp does:** mmap by default, optional mlock to prevent paging

**flash-moe key finding:** Trusting the OS page cache via mmap and removing application-level caching improved throughput by 38%.

#### Approach C: Async Overlapped I/O (Windows)
```c
HANDLE hFile = CreateFile("model.bin", GENERIC_READ, ..., FILE_FLAG_OVERLAPPED);
OVERLAPPED ov = {0};
ReadFile(hFile, buffer, chunk_size, NULL, &ov);
// Continue doing other work...
WaitForSingleObject(ov.hEvent, INFINITE);
```
- **Speed:** Same as synchronous, but non-blocking
- **Pros:** Can overlap I/O with computation (load layer N+1 while processing layer N)
- **Cons:** More complex code, mainly useful for streaming (not needed if model fits in RAM)

#### Approach D: Unbuffered Direct I/O
```c
HANDLE hFile = CreateFile("model.bin", GENERIC_READ, ..., FILE_FLAG_NO_BUFFERING);
```
- **Speed:** 4-7 GB/s (full NVMe speed, no OS cache overhead)
- **Pros:** Predictable latency, no page cache pollution
- **Cons:** Must use sector-aligned buffers, bypasses OS cache (no benefit on second load)

#### Approach E: DirectStorage
- Microsoft API for GPU-direct NVMe reads
- Bypasses CPU for data going to iGPU
- Best for: loading data directly into iGPU-accessible memory
- **Limitation:** Only available on Windows 11, requires DirectX 12

### 11.2 Memory Allocation Strategies

#### Standard 4KB Pages
```c
void *buf = malloc(size);  // or _aligned_malloc(size, 64)
```
- 16 GB = 4,194,304 pages
- TLB pressure: L2 STLB covers only 8 MB (2048 entries × 4KB)
- Every TLB miss costs ~20-50 cycles

#### Large Pages (2MB)
```c
// Windows — requires SeLockMemoryPrivilege
void *buf = VirtualAlloc(NULL, size,
    MEM_RESERVE | MEM_COMMIT | MEM_LARGE_PAGES, PAGE_READWRITE);
```
- 16 GB = 8,192 pages
- TLB: L2 STLB covers 4 GB (2048 entries × 2MB) — covers entire model
- **Measured improvement:** 10-25% for large sequential scans
- **Setup required:** Local Security Policy → User Rights → "Lock pages in memory" → add user → reboot

#### Alignment Requirements
- **Cache line (64 bytes):** Minimum for avoiding false sharing between threads
- **AVX2 (32 bytes):** Required for VMOVDQA (aligned loads). Unaligned loads (VMOVDQU) work but ~10-20% slower
- **Page (4096 bytes):** Required for mmap/VirtualAlloc compatibility

### 11.3 Weight Repacking at Load

Most model formats (GGUF, SafeTensors) store weights in a format that's not optimal for the target SIMD kernel. Repacking reorganizes weights at load time.

**Example for ternary:**
1. Source: SafeTensors with INT8 values of -1, 0, +1 (8 bits per weight)
2. Repack to: 2-bit packed sign+zero bitmasks (2 bits per weight)
3. Arrange in 256-bit blocks aligned for AVX2 loads
4. Co-locate per-group activation scales adjacent to their weight block

**Time cost:** ~50-200 ms for a 3B model (one-time at load)

**Optimization: Cache repacked weights to disk**
- First load: repack from SafeTensors (~200 ms), save as .edgelm file
- Subsequent loads: mmap the pre-packed .edgelm file (~0 ms effective)

### 11.4 Memory Budget Planning

Complete memory budget for 3B ternary inference:

| Component | Size | Notes |
|---|---|---|
| Model weights (ternary) | 0.4-0.6 GB | 2 bits × 3B params |
| Embedding table (FP16) | ~80 MB | 128K vocab × 3200 dim × 2 bytes |
| Output head (FP16) | ~200 MB | 3200 × 128K × 2 bytes |
| KV cache (FP16, 2048 ctx) | ~45-100 MB | Depends on n_kv_heads |
| Activation buffers | ~50-100 MB | Double-buffered per layer |
| Thread stacks + overhead | ~50 MB | 20 threads × 2.5 MB each |
| Tokenizer data | ~5-10 MB | Vocab + merge table |
| **Total** | **~1.0-1.5 GB** | |
| **Remaining from 16 GB** | **~5-6 GB** | For OS, apps, speculative draft model |

Compare with Q4_K_M 3B: model alone is 1.9 GB, total ~2.5-3 GB. Ternary gives 2x more headroom.

---

## Section 12: Tokenization

### 12.1 Algorithms

#### BPE (Byte Pair Encoding)
- **How:** Iteratively merge most frequent byte/character pairs to build a vocabulary
- **Used by:** GPT, Llama, Qwen, BitNet, most modern LLMs
- **Vocab size:** Typically 32K-128K tokens
- **Encoding:** Find longest match in vocab, iterate
- **Complexity:** O(n × m) naive, O(n × log n) with priority queue

#### SentencePiece (Unigram)
- **How:** Probabilistic subword segmentation — Viterbi search on subword lattice
- **Used by:** T5-family, some older models
- **More complex** than BPE, requires full probabilistic model

#### WordPiece
- **How:** Similar to BPE but uses likelihood-based merging
- **Used by:** BERT-family (not relevant for decoder-only LLMs)

#### Byte-level BPE
- **How:** Treats input as raw bytes, vocabulary built on byte sequences
- **Used by:** GPT-2, Llama (via tiktoken or sentencepiece with byte fallback)
- **Advantage:** Handles any Unicode input without UNK tokens

### 12.2 Implementation Approaches

| Approach | Startup | Speed | Complexity | Dependencies |
|---|---|---|---|---|
| HuggingFace tokenizers (Python) | ~3 sec | Fast (Rust backend) | Low | Python + tokenizers lib |
| sentencepiece (C++) | ~100 ms | Fast | Low | C++ lib + protobuf |
| Custom C (parse tokenizer.json) | ~5 ms | Fastest | Medium (~500-1000 LOC) | None |
| Extract from llama.cpp | ~10 ms | Fast | Medium (~2000 LOC C++) | Partial llama.cpp |

**For a custom C engine:** Parse tokenizer.json directly, build hash table for BPE merge lookups. ~500-1000 lines of C. Zero dependencies, <5 ms for 1000 tokens.

### 12.3 Performance Considerations

- **Tokenization is NOT a bottleneck for decode speed** — it runs once at the start
- **IS relevant for time-to-first-token (TTFT)** — Python tokenizer adds 3+ seconds
- **Memory:** Vocab table ~2-10 MB depending on vocab size
- **Custom C tokenizer:** flash-moe achieved 180 ms startup vs 3500 ms with Python

---

## Section 13: The Transformer Forward Pass — Step by Step

This section traces the COMPLETE path of a single token through the model during autoregressive decode.

### 13.1 Overview: One Token Through 28 Layers

```
Input token ID
    │
    ▼
[Embedding Lookup] ──→ activation vector (d=3200, FP32)
    │
    ▼
╔═══════════════════════════════════╗
║  For each of 28 transformer layers: ║
║                                     ║
║  [RMSNorm] ─→ normalized vector    ║
║      │                              ║
║  [Q,K,V Projections] ─→ 3 vectors  ║  ← MATMUL (ternary GEMV)
║      │                              ║
║  [RoPE] ─→ position-encoded Q,K    ║
║      │                              ║
║  [Attention: QK^T] ─→ scores       ║
║  [Softmax] ─→ probabilities        ║
║  [Score × V] ─→ context vector     ║
║      │                              ║
║  [Output Projection] ─→ output     ║  ← MATMUL (ternary GEMV)
║      │                              ║
║  [Residual Add]                     ║
║      │                              ║
║  [RMSNorm] ─→ normalized           ║
║      │                              ║
║  [FFN: Gate proj] ─→ gate          ║  ← MATMUL (ternary GEMV)
║  [FFN: Up proj]   ─→ up            ║  ← MATMUL (ternary GEMV)
║  [SiLU(gate) × up]                 ║
║  [FFN: Down proj] ─→ output        ║  ← MATMUL (ternary GEMV)
║      │                              ║
║  [Residual Add]                     ║
╚═══════════════════════════════════╝
    │
    ▼
[Final RMSNorm]
    │
    ▼
[Output Head (LM Head)] ──→ logits   ← MATMUL (possibly FP16 GEMV)
    │
    ▼
[Sampling] ──→ next token ID
```

**Total matmuls per token:** 7 per layer × 28 layers + 1 output head = **197 matmuls**

### 13.2 Embedding Lookup

**What:** token_id → embedding vector

```c
float *embedding = embedding_table + token_id * d;  // d = 3200
```

- Size: vocab_size × d × bytes = 128K × 3200 × 2 = **819 MB at FP16**
- Or: 128K × 3200 × 0.2 = **~80 MB at ternary** (but embeddings are typically FP16)
- Cache behavior: only ONE row accessed per token — perfectly cache-friendly
- Time: **<0.01 ms** — never a bottleneck
- Memory read: 6.4 KB (one row of 3200 × FP32 for computation)

### 13.3 RMSNorm

**What:** Normalize activation vector by root-mean-square, then scale.

```
output[i] = (x[i] / sqrt(mean(x²) + eps)) × gamma[i]
```

**Compute profile:**
1. Sum of squares: Σ(x[i]²) — reduction over d elements
2. Reciprocal square root: 1/sqrt(sum/d + eps)
3. Element-wise multiply: x[i] × scale × gamma[i]

**Implementation approaches:**

| Approach | Time (d=3200) | Notes |
|---|---|---|
| Naive C loops | ~200 ns | Straightforward |
| AVX2 (VFMADD + VRSQRTPS) | ~50 ns | 8 floats per instruction |
| Fused with next projection | ~30 ns effective | Avoid extra memory round-trip |
| iGPU offload | ~5-10 us (dispatch overhead) | NOT worth it — CPU faster |

**Performance:** ~0.1% of total time. Not a bottleneck, but should be SIMD-optimized to avoid waste.

### 13.4 QKV Projections

**What:** Three (or fused) matrix-vector multiplications:
```
Q = Wq × x    (d × d matrix × d vector → d vector)
K = Wk × x    (d × d_kv matrix × d vector → d_kv vector)
V = Wv × x    (d × d_kv matrix × d vector → d_kv vector)
```

For GQA (Grouped Query Attention):
- Llama-3.2-3B: 32 Q heads, 8 KV heads, head_dim=100
- d_kv = n_kv_heads × head_dim = 8 × 100 = 800
- Wq: 3200 × 3200, Wk: 3200 × 800, Wv: 3200 × 800
- Total weights: 3200 × (3200 + 800 + 800) = 15.36M parameters per layer

**This IS a matmul and IS on the critical path.** See Section 14 for kernel optimization.

### 13.5 Rotary Position Embeddings (RoPE)

**What:** Encode position by rotating pairs of dimensions in Q and K vectors.

```
For each pair (x[2i], x[2i+1]):
    x_rot[2i]   = x[2i] × cos(θ) - x[2i+1] × sin(θ)
    x_rot[2i+1] = x[2i] × sin(θ) + x[2i+1] × cos(θ)
where θ = position / 10000^(2i/d)
```

**Implementation:** Precompute sin/cos tables at startup. Per-token: just multiply and add.

**Time:** ~0.01 ms — element-wise operation on a d-dimensional vector. NOT a bottleneck.

**Variants:**
- Standard RoPE (used by Llama, BitNet)
- NTK-aware RoPE (for context length extension beyond training window)
- YaRN (Yet another RoPE extensioN)

### 13.6 Attention Score Computation (QK^T)

**What:** Dot product of query with all keys in KV cache.

For single-token decode:
- Q is a vector: (1 × head_dim) per head
- K is the KV cache: (seq_len × head_dim) per KV head
- Result: attention scores vector (1 × seq_len) per head

**Compute:** O(n_heads × seq_len × head_dim) per layer
- At seq_len=512: 32 × 512 × 100 = 1.6M multiply-adds → ~0.02 ms
- At seq_len=2048: 32 × 2048 × 100 = 6.6M multiply-adds → ~0.08 ms
- At seq_len=8192: ~0.3 ms (starts becoming significant)

**Implementation approaches:**

| Approach | Description | Best For |
|---|---|---|
| Direct dot products | Loop over KV cache rows | Short sequences |
| GEMV | Treat as matrix-vector multiply | Any length |
| FlashAttention-style | Tiled, fused with softmax | Long sequences (>2K) |
| iGPU offload | Run attention on Iris Xe | Long sequences if bandwidth available |

### 13.7 Softmax

**What:** Normalize attention scores to probabilities.

```
prob[i] = exp(score[i] - max(scores)) / Σ exp(score[j] - max(scores))
```

Two passes: (1) find max, (2) compute exp and sum.
Or: online softmax (single pass, FlashAttention method).

**Time:** O(seq_len) per head — negligible (<0.01 ms).

**SIMD:** AVX2 doesn't have a native exp instruction. Options:
- Polynomial approximation (Cephes or minimax polynomial, ~5-10 instructions)
- Table-based exp (lookup + interpolation)
- Use SVML `_mm256_exp_ps` if available (Intel compiler)

### 13.8 Weighted Sum (Score × V)

Same profile as QK^T: O(n_heads × seq_len × head_dim). Often fused with softmax in FlashAttention.

### 13.9 Output Projection

```
output = Wo × concatenated_heads    (d × d matrix-vector multiply)
```

Another ternary GEMV. Same optimization as QKV projections.

### 13.10 Feed-Forward Network (FFN/MLP) — THE DOMINANT COST

**Structure (SwiGLU variant, used by Llama/BitNet):**
```
gate = W_gate × x        (d × d_ffn GEMV)
up   = W_up × x          (d × d_ffn GEMV)
ffn_out = W_down × (SiLU(gate) ⊙ up)    (d_ffn × d GEMV)
```

**Sizes:**
- d_ffn is typically 2.67x to 4x d (e.g., d=3200, d_ffn=8640)
- W_gate and W_up: d × d_ffn = 3200 × 8640 = 27.6M params each
- W_down: d_ffn × d = 8640 × 3200 = 27.6M params
- **Total FFN per layer: 82.9M params — ~67% of all weights per layer**

**This is THE bottleneck.** The three FFN matmuls are the largest in the model.

**SiLU activation:** x × sigmoid(x)
- Compute: O(d_ffn), ~0.01 ms
- Polynomial approximation sufficient — NOT a bottleneck

### 13.11 Output Head (LM Head)

```
logits = W_lm × hidden_state    (d × vocab_size GEMV)
```

- Size: 3200 × 128K = 409M elements
- At FP16: **819 MB** — this can be the single biggest tensor
- At ternary: ~100 MB (if ternary-quantized)

**Optimization approaches:**

| Approach | Description | Speedup |
|---|---|---|
| Full logits | Compute all vocab_size logits | Baseline |
| Top-K only | Approximate via random projection or LSH | 5-10x for large vocabs |
| Tied embeddings | Share weights with embedding table | 2x memory savings |
| Draft model head | Only use in speculative decoding | N/A |

### 13.12 Complete Timing Breakdown (Target: 100 tok/s = 10 ms/token)

| Operation | Per Layer | × 28 Layers | Notes |
|---|---|---|---|
| RMSNorm (×2) | 0.01 ms | 0.28 ms | Negligible |
| QKV projection | 0.05 ms | 1.40 ms | Ternary GEMV |
| RoPE | 0.005 ms | 0.14 ms | Negligible |
| Attention (QK^T + softmax + SV) | 0.05 ms | 1.40 ms | At seq_len=512 |
| Output projection | 0.03 ms | 0.84 ms | Ternary GEMV |
| FFN (gate + up + down) | 0.15 ms | 4.20 ms | **Dominant** |
| Residual adds | 0.002 ms | 0.06 ms | Negligible |
| **Subtotal (layers)** | **0.30 ms** | **8.32 ms** | |
| Final RMSNorm | | 0.01 ms | |
| Output head | | 0.5-1.0 ms | Depends on precision |
| Sampling | | 0.05 ms | |
| **Total** | | **~9-10 ms** | **= 100-111 tok/s** |

This is tight but achievable if kernels achieve ~35 GB/s effective bandwidth.

---

# PART IV: CORE KERNELS & RUNTIME

---

## Section 14: Matrix Multiplication — The Core Compute Kernel

**This is the longest and most critical section.** 90%+ of inference time is spent in matmul.

### 14.1 GEMV vs GEMM

During single-token decode: **GEMV** (matrix × vector)
```
y = W × x    where W is (M × K), x is (K × 1), y is (M × 1)
```

During prefill (batch): **GEMM** (matrix × matrix)
```
Y = W × X    where X is (K × N) for N tokens
```

**Why this matters:**
- GEMV reads M×K weight elements to compute M output elements → arithmetic intensity = 1 op/element
- GEMM reads M×K weight elements to compute M×N output elements → arithmetic intensity = N ops/element
- At N≥10: GEMM becomes compute-bound. At N=1 (GEMV): always memory-bound on this hardware.

### 14.2 Roofline Analysis for i7-12700H

```
Peak compute (INT8, 6 P-cores, AVX2):
  6 cores × 1 VPDPBUSD/cycle × 32 ops × 4.1 GHz = ~787 GOPS

Peak bandwidth: ~40 GB/s

Arithmetic intensity crossover: 787 GOPS / 40 GB/s = ~20 ops/byte

Ternary GEMV at 2 bits/weight:
  Each byte = 4 weights = 4 add/sub operations = 4 ops
  Arithmetic intensity = 4 ops/byte

4 ops/byte << 20 ops/byte → FIRMLY MEMORY-BOUND

Implication: kernel speed is limited by memory bandwidth, NOT compute.
The goal is to maximize bandwidth utilization, not compute throughput.
```

### 14.3 Approach A: Naive C Loops (Baseline)

```c
void gemv_naive(float *y, const int8_t *W, const float *x, int M, int K) {
    for (int i = 0; i < M; i++) {
        float sum = 0;
        for (int k = 0; k < K; k++) {
            sum += W[i * K + k] * x[k];
        }
        y[i] = sum;
    }
}
```

- **Performance:** ~1-2 GB/s effective bandwidth (5% of peak)
- **Why slow:** No SIMD, no prefetch, poor cache use, branch prediction overhead
- **Value:** Correctness reference for testing optimized kernels

### 14.4 Approach B: BLAS Libraries

#### Intel MKL (oneMKL)
- Highly optimized SGEMM for Intel CPUs
- Automatically uses AVX2
- **Limitation:** No native ternary support. Would need to expand ternary to FP32/INT8 first
- **For GEMV:** MKL's cblas_sgemv is optimized for large matrices but has dispatch overhead

#### OpenBLAS
- Open-source, decent AVX2 support
- Same ternary limitation

#### BLIS
- Research BLAS, very configurable micro-kernel architecture
- Could theoretically be adapted for custom data types

**Why BLAS is NOT optimal for LLM inference:**
1. BLAS assumes standard data types (FP32/FP64/INT8)
2. BLAS is optimized for large GEMM, not GEMV
3. LLM decode is GEMV-dominant (batch=1)
4. Ternary weights need custom kernels — BLAS can't handle {-1,0,+1} natively
5. Dispatch and threading overhead for small matrices

### 14.5 Approach C: INT8 GEMV with AVX2

For standard INT8 quantized models (Q8_0):

```c
// Process 32 INT8 multiply-accumulates per iteration
__m256i acc = _mm256_setzero_si256();

for (int k = 0; k < K; k += 32) {
    __m256i w = _mm256_loadu_si256((__m256i*)(weights + k));  // 32 INT8 weights
    __m256i x = _mm256_loadu_si256((__m256i*)(act + k));      // 32 INT8 activations

    // VPMADDUBSW: multiply unsigned×signed bytes → signed words, add adjacent
    __m256i prod = _mm256_maddubs_epi16(x, w);  // 16 INT16 results

    // VPMADDWD: multiply INT16 pairs, add adjacent → INT32
    __m256i sum32 = _mm256_madd_epi16(prod, _mm256_set1_epi16(1));  // 8 INT32 results

    // Accumulate
    acc = _mm256_add_epi32(acc, sum32);
}
// Horizontal reduce acc to single INT32
```

- **Performance:** ~15-25 GB/s effective bandwidth
- **What llama.cpp does:** Q8_0 kernel uses this pattern

### 14.6 Approach D: INT4 GEMV with AVX2

For Q4_0/Q4_K_M quantized models:

```c
// Each byte contains 2 INT4 values
for (int k = 0; k < K; k += 32) {
    __m256i packed = _mm256_loadu_si256(weights + k/2);  // 32 INT4 in 16 bytes

    // Unpack lower nibbles
    __m256i lo = _mm256_and_si256(packed, _mm256_set1_epi8(0x0F));
    // Unpack upper nibbles
    __m256i hi = _mm256_and_si256(_mm256_srli_epi16(packed, 4), _mm256_set1_epi8(0x0F));

    // Subtract zero point (8 for unsigned 4-bit centered at 0)
    lo = _mm256_sub_epi8(lo, _mm256_set1_epi8(8));
    hi = _mm256_sub_epi8(hi, _mm256_set1_epi8(8));

    // Now use VPMADDUBSW chain as in INT8 path
    // ... (interleave lo and hi with activations)
}
```

- **Performance:** ~20-30 GB/s effective bandwidth (higher than INT8 because fewer bytes to read per weight)
- **What llama.cpp does:** Q4_0, Q4_1, Q4_K kernels use this approach

### 14.7 Approach E: Ternary GEMV — Sign-Mask Method

Pack ternary weights as two bitmasks:

```c
// For each group of 256 ternary weights:
//   nonzero_mask: 256 bits — bit=1 where weight ≠ 0
//   sign_mask:    256 bits — bit=1 where weight = -1

void ternary_gemv_signmask(float *y, const uint8_t *nonzero, const uint8_t *sign,
                            const float *x, int M, int K) {
    for (int i = 0; i < M; i++) {
        __m256 acc = _mm256_setzero_ps();

        for (int k = 0; k < K; k += 8) {
            __m256 xv = _mm256_loadu_ps(x + k);

            // Load bitmask bits for these 8 weights
            uint8_t nz_bits = nonzero[i * K/8 + k/8];
            uint8_t sg_bits = sign[i * K/8 + k/8];

            // Create float masks from bits
            // For each weight: if nonzero && positive → +1, if nonzero && negative → -1, else 0
            // This needs bit unpacking to float mask — complex but doable

            // Method: convert bits to +1/-1/0 float vector, multiply with x
            // Or: use VPBLENDVB to conditionally negate
        }
    }
}
```

**Better approach using integer domain:**

```c
// Work in INT8/INT16 to leverage SIMD integer ops
// 1. Load 32 activation values (INT8, pre-quantized)
// 2. AND with nonzero_mask → zeros out where weight=0
// 3. Use sign_mask with VPSUBB to negate where weight=-1
// 4. Accumulate

for (int k = 0; k < K; k += 256) {
    __m256i nz = _mm256_load_si256(nonzero_masks + ...);  // 256-bit nonzero mask
    __m256i sg = _mm256_load_si256(sign_masks + ...);      // 256-bit sign mask
    __m256i act = _mm256_load_si256(activations + ...);    // 32 INT8 activations

    // Zero out activations where weight=0
    __m256i masked = _mm256_and_si256(act, nz);  // applies mask byte-wise (needs expansion)

    // Negate where weight=-1: result = masked XOR sign_expanded, then add carry
    // Or: positive = masked AND NOT(sign), negative = masked AND sign
    // acc += positive - negative
}
```

- **Bits per weight:** 2 (optimal for bandwidth)
- **Compute:** Multiple instructions per 32 weights (mask expansion, conditional negate, accumulate)
- **Performance:** ~25-35 GB/s effective bandwidth (good bandwidth efficiency, moderate compute overhead)

### 14.8 Approach F: Ternary GEMV — Two-Bit Encoding

Encode each weight as 2 bits: 00=zero, 01=+1, 10=-1 (or 11=unused)

```c
// Use VPSHUFB as a 4-entry lookup table
// Group 4 weights → 8-bit index → lookup partial sum from table

// Precompute LUT for each group of 4 activations:
// For all 3^4 = 81 combinations of {-1,0,+1}^4:
//   LUT[idx] = sum of (weight[i] * activation[i]) for i=0..3

// But 81 > 16, so VPSHUFB can't hold it (only 16 entries)
// Solution: use 2 weights at a time (3^2 = 9 entries, fits in 16-entry LUT)

__m256i lut = build_lut_for_2_activations(x[k], x[k+1]);  // 9 used entries
__m256i indices = extract_2bit_pairs(weights);               // 16 lookups per VPSHUFB
__m256i results = _mm256_shuffle_epi8(lut, indices);         // 16 parallel lookups
```

- **Bits per weight:** 2
- **Key instruction:** VPSHUFB (Port 5, 1/cycle throughput)
- **Performance:** Similar to sign-mask, but different port utilization (Port 5 vs Port 0)

### 14.9 Approach G: Ternary GEMV — AVX-VNNI (VPDPBUSD)

Pack ternary weights directly as INT8 {-1, 0, +1} and use AVX-VNNI:

```c
// Pack 32 ternary weights as INT8 values: -1, 0, or +1
__m256i weights_i8 = _mm256_load_si256(weight_data);  // 32 bytes = 32 weights
// Pack 32 activations as UINT8 (biased: original_value + 128)
__m256i acts_u8 = _mm256_load_si256(act_data);

// VPDPBUSD: 32 UINT8 × INT8 multiply-accumulates → 8 INT32
acc = _mm256_dpbusd_epi32(acc, acts_u8, weights_i8);
```

- **Bits per weight:** 8 (wastes 6 bits — 4× more bandwidth than 2-bit packing!)
- **Instructions per 32 weights:** 1 (VPDPBUSD) — minimal compute
- **Performance:** Limited by the 8 bits/weight bandwidth cost
  - At 8 bits/weight: 3B × 1 byte = 3 GB model → max 13.3 tok/s (same as INT8!)
  - **This approach DESTROYS the ternary bandwidth advantage**

**Hybrid approach:** Pack 2-bit, expand to INT8 on-the-fly, then VPDPBUSD
```c
// Load 64 ternary weights packed as 2-bit (16 bytes)
__m256i packed = _mm256_load_si256(weight_2bit);
// Unpack to 32 INT8 values (expand 2-bit → 8-bit)
__m256i expanded = unpack_2bit_to_int8(packed);  // ~3-4 instructions
// Then VPDPBUSD
acc = _mm256_dpbusd_epi32(acc, acts_u8, expanded);
```

This gives: 2 bits/weight bandwidth + VPDPBUSD compute. ~4-5 instructions total per 32 weights.

**What BitNet does:** I2_S kernel uses 2-bit packing, TL2 kernel uses tiled approach.

### 14.10 Approach H: T-MAC LUT-Based Method

**Core idea:** Precompute ALL possible partial sums for groups of k weights.

```
For k=4 ternary weights: 3^4 = 81 possible partial sums
For k=2 ternary weights: 3^2 = 9 possible partial sums (fits in VPSHUFB's 16-entry LUT)

Step 1: For the current 2 activation values (a0, a1), compute the 9-entry LUT:
  LUT[0] = 0*a0 + 0*a1 = 0
  LUT[1] = 0*a0 + 1*a1 = a1
  LUT[2] = 0*a0 + (-1)*a1 = -a1
  LUT[3] = 1*a0 + 0*a1 = a0
  LUT[4] = 1*a0 + 1*a1 = a0 + a1
  LUT[5] = 1*a0 + (-1)*a1 = a0 - a1
  LUT[6] = (-1)*a0 + 0*a1 = -a0
  LUT[7] = (-1)*a0 + 1*a1 = -a0 + a1
  LUT[8] = (-1)*a0 + (-1)*a1 = -a0 - a1

Step 2: Pack weight-pair indices (each is 0-8, fits in 4 bits)

Step 3: VPSHUFB performs 16 parallel lookups from the 9-entry LUT
  → computes 16 partial dot products in 1 instruction

Step 4: Accumulate results
```

**Implementation:**
```c
// For each pair of activation values:
__m256i lut = compute_ternary_lut_2(act[k], act[k+1]);  // ~6 instructions

// Process 32 output elements (16 weight-pairs per VPSHUFB, 2 VPSHUFB per 32 outputs)
__m256i indices = _mm256_load_si256(weight_lut_indices);
__m256i partial = _mm256_shuffle_epi8(lut, indices);     // 16 lookups in 1 instruction

// Accumulate partials
acc = _mm256_add_epi16(acc, partial);
```

- **Bits per weight:** 2 (indices packed as 4-bit, 2 weights per index)
- **Key advantage:** Eliminates ALL multiplications. Only additions and table lookups
- **Key instruction:** VPSHUFB — 1/cycle on Port 5
- **Performance:** ~30-40 GB/s effective bandwidth (T-MAC paper reports ~40-50 tok/s on similar Intel hardware)
- **Limitation:** LUT construction has overhead. Amortized over many output elements

### 14.11 Approach I: XNOR + Popcount (Binary 1-bit only)

Only for {-1, +1} weights (NOT ternary — no zero):

```c
// Pack weights as bits, activations as sign bits
// XNOR of weight bits and activation sign bits
// Popcount gives dot product
uint64_t w_bits = weight_binary[offset];
uint64_t a_bits = activation_signs[offset];
int dot = 2 * __builtin_popcountll(~(w_bits ^ a_bits)) - 64;
```

- **1 bit per weight** — maximum bandwidth efficiency
- **Extremely fast** computation (XNOR + POPCNT)
- **Quality: too poor** for 3B models — only viable at 100B+ scale
- **Not recommended** for this project

### 14.12 Approach Comparison Matrix

| Approach | Bits/Weight | Instructions/32 weights | Port Pressure | GB/s Expected | Notes |
|---|---|---|---|---|---|
| Naive C | 8+ | ~100+ | All | 1-2 | Baseline only |
| BLAS (MKL) | 32 | Optimized | All | 15-20 | Wrong data type |
| INT8 GEMV | 8 | 3 | Port 0 | 15-25 | Standard approach |
| INT4 GEMV | 4 | 5-6 | Port 0,5 | 20-30 | llama.cpp default |
| Sign-mask ternary | 2 | 5-7 | Port 0,5,10 | 25-35 | Good balance |
| Two-bit encoded | 2 | 4-6 | Port 5 | 25-35 | Similar to sign-mask |
| VNNI direct (8-bit) | 8 | 1 | Port 0 | 15-20 | Wastes bandwidth |
| VNNI hybrid (2→8) | 2 | 4-5 | Port 0,5 | 28-35 | Best VNNI approach |
| T-MAC LUT | 2 | 3-4 | Port 5 | 30-40 | Eliminates multiply |
| XNOR+popcount | 1 | 2 | Port 0 | 35-45 | Binary only, bad quality |

### 14.13 Prefetching Strategies

#### Software Prefetch Instructions
```c
_mm_prefetch(addr, _MM_HINT_T0);   // Prefetch to L1 (+ all higher levels)
_mm_prefetch(addr, _MM_HINT_T1);   // Prefetch to L2 (+ L3)
_mm_prefetch(addr, _MM_HINT_T2);   // Prefetch to L3 only
_mm_prefetch(addr, _MM_HINT_NTA);  // Non-temporal: use once, don't pollute cache
```

#### Strategy A: Same-Row Prefetch
Prefetch next weight block while processing current:
```c
for (int k = 0; k < K; k += 32) {
    _mm_prefetch(weights + k + 256, _MM_HINT_T0);  // 8 cache lines ahead
    // ... process weights[k..k+31] ...
}
```
- Hides memory latency within the inner loop
- Hardware prefetcher usually handles this — software prefetch may add marginal benefit

#### Strategy B: Next-Layer Prefetch
Prefetch next layer's weights during the last portion of current layer:
```c
// During last 10% of layer L computation:
for (int i = 0; i < next_layer_prefetch_bytes; i += 64) {
    _mm_prefetch(next_layer_weights + i, _MM_HINT_T2);  // Prefetch to L3
}
```
- L3 is 24 MB — can prefetch a significant chunk of next layer
- **Measured impact:** 10-20% improvement when layer weights >> L3 size

#### Strategy C: Non-Temporal Stores
For activation outputs that won't be re-read immediately:
```c
_mm256_stream_ps(output + i, result);  // VMOVNTPS — bypass cache
```
- Frees L1/L2 capacity for weight streaming
- **Only useful** when output buffer is large and won't be re-read soon

#### Strategy D: Cross-Thread Prefetch (E-Core Prefetcher)
Dedicate an E-core thread to prefetching next layer's weights into L3:
```c
// E-core thread:
while (current_layer < total_layers) {
    wait_for_signal(layer_start);
    for (size_t i = 0; i < next_layer_size; i += 64) {
        _mm_prefetch(next_layer_weights + i, _MM_HINT_T2);
    }
}
```
- L3 is shared between all cores — E-core prefetch benefits P-core reads
- **Measured impact:** 5-15% on top of software prefetch

### 14.14 Tiling and Cache Blocking

For a weight matrix W of size (M × K):

**L1 tile:** Process 48 KB of weights at a time (fits in 48 KB L1D)
- At 2 bits/weight: 48 KB = 192K weights → ~190 rows × 1024 columns, or similar decomposition

**L2 tile:** 1.25 MB per P-core
- At 2 bits/weight: 1.25 MB = 5M weights → can hold ~1562 rows × 3200 columns (one full K-dimension sweep)

**L3 tile:** 24 MB shared
- At 2 bits/weight: 24 MB = 96M weights → can hold ~30K rows × 3200 columns (significant fraction of layer)

**For GEMV (decode):** Tiling is primarily over the output dimension M:
- Partition M rows across P-cores: each core processes M/6 rows
- Within each core: process rows in L2-sized blocks
- Input vector x (3200 × 4 = 12.8 KB at FP32) fits in L1 for all cores simultaneously

**Register blocking:** Unroll inner loop to keep results in YMM registers:
- 16 YMM registers available
- Optimal: 4 registers for accumulators (4 output elements), 4 for weight data, 4 for activation data, 4 for temporaries
- Process 4 output rows simultaneously to maximize register utilization

### 14.15 Multi-Threaded GEMV

**Parallelization:** Partition output rows across threads.

```
Thread 0: rows [0, M/6)
Thread 1: rows [M/6, 2M/6)
...
Thread 5: rows [5M/6, M)
```

Each thread reads its portion of weights + the FULL input vector (shared, read-only).

**Bandwidth scaling:**
- 1 P-core: ~15-20 GB/s
- 2 P-cores: ~25-30 GB/s
- 4 P-cores: ~35-38 GB/s
- 6 P-cores: ~38-42 GB/s (approaching DDR4 bandwidth limit)
- Adding E-cores beyond this point: minimal benefit for bandwidth-bound work

**Synchronization:** Barrier at end of each matmul (all threads must finish before the next operation can use the output).

| Barrier Type | Latency | Notes |
|---|---|---|
| pthread_barrier / Windows Event | ~1-2 us | Heavyweight, OS-mediated |
| Spin barrier (atomic counter) | ~0.1-0.5 us | Lowest latency, wastes power |
| Futex-based (spin then sleep) | ~0.2-1 us | Hybrid, good default |

**False sharing:** Ensure each thread's output buffer starts on a 64-byte cache line boundary:
```c
// BAD: thread i writes to output[i * chunk_size], chunk_size not cache-aligned
// GOOD: thread i writes to output[i * ALIGN_UP(chunk_size, 64)]
```

### 14.16 Prefill Batch GEMM Optimization

During prefill, all prompt tokens are processed at once:

```
Y = W × X    where X is (K × N) for N prompt tokens
```

**Advantage:** Weight reads amortized across N tokens:
- At N=1: 1 weight read per output element (memory-bound)
- At N=32: 1 weight read per 32 output elements (becoming compute-bound)
- At N=64: firmly compute-bound — can utilize full CPU compute throughput

**Optimal batch size on i7-12700H:** 16-64 tokens
- Below 16: still partially bandwidth-bound
- Above 64: no additional benefit, may exceed L2 cache for activations

**flash-moe insight:** For MoE models, skip heavy expert computation for all prefill tokens except the last. Only the last token's output determines the generated text.

### 14.17 Intrinsics vs Inline Assembly vs .asm Files

| Approach | Portability | Optimizer | Control | Maintenance |
|---|---|---|---|---|
| Compiler intrinsics | MSVC, GCC, Clang | Can rearrange, may be suboptimal | Medium | Easy |
| Inline asm (asm volatile) | GCC/Clang only | Cannot optimize around it | Full | Hard |
| Separate .asm (NASM/MASM) | Assembly-level | No compiler interaction | Full | Hardest |

**Recommendation:** Start with intrinsics. Profile. Only drop to inline assembly for the innermost matmul loop IF intrinsics produce suboptimal register allocation or instruction ordering.

---

## Section 15: Attention Mechanisms

### 15.1 Multi-Head Attention (MHA)

Standard: n_heads Q, K, V heads, each with head_dim dimensions.

KV cache per layer: 2 × n_heads × seq_len × head_dim × bytes_per_element

For 3B (32 heads, head_dim=100) at FP16, 2048 context:
```
2 × 32 × 2048 × 100 × 2 bytes = 25.6 MB per layer
× 28 layers = 716 MB total
```
**This is a LOT for 16 GB RAM.** That's why GQA is essential.

### 15.2 Grouped-Query Attention (GQA)

Fewer KV heads than Q heads: e.g., 32 Q heads, 8 KV heads.

```
KV cache: 2 × 8 × 2048 × 100 × 2 = 6.4 MB per layer
× 28 layers = 179 MB total
```
**4x reduction** vs MHA. Used by Llama 3, Mistral, Qwen2.5, most modern models.

Implementation: each KV head is shared by group_size (4) Q heads. During attention, broadcast the K/V from the shared head to its Q heads.

### 15.3 Multi-Query Attention (MQA)

Extreme: 1 KV head shared by ALL Q heads.

```
KV cache: 2 × 1 × 2048 × 100 × 2 = 0.8 MB per layer
× 28 layers = 22.4 MB total
```
**32x reduction** vs MHA. Used by Falcon, some CodeGen models.

Quality tradeoff: slight but measurable degradation. GQA is the sweet spot.

### 15.4 FlashAttention (CPU Adaptation)

**Original:** GPU optimization to avoid materializing the full N×N attention matrix (O(N²) memory).

**CPU adaptation:**
1. Process attention in blocks of query positions
2. Maintain running softmax statistics (online softmax: running max and running sum)
3. Never allocate the full seq_len × seq_len matrix

**Relevance for decode:**
- During single-token decode: attention is a vector-matrix product (Q is 1×d, K is seq_len×d)
- No N×N matrix exists — FlashAttention is primarily a **prefill** optimization
- For prefill with long prompts: FlashAttention reduces memory from O(N²) to O(N)

### 15.5 Linear Attention Variants

Replace softmax(QK^T)V with phi(Q) × (phi(K)^T × V):

| Model | Mechanism | Complexity | KV Cache | Quality vs Transformer |
|---|---|---|---|---|
| RWKV (WKV operator) | Linear recurrence | O(d) per token | Fixed-size state | ~5-10% worse |
| Mamba (selective SSM) | State space model | O(d×d_state) per token | Fixed-size state | ~3-5% worse |
| RetNet | Retention mechanism | O(d) per token | Fixed-size state | ~3-8% worse |
| GLA (Gated Linear Attention) | Gated linear attn | O(d) per token | Fixed-size state | ~2-5% worse |

**Advantage:** No growing KV cache. Constant memory per token regardless of context length.
**Disadvantage:** Quality gap (narrowing with research). Limited model availability at 3B.

### 15.6 Sliding Window Attention

Limit attention to last W tokens (e.g., W=4096):
- KV cache becomes a fixed-size ring buffer
- Memory: constant regardless of total conversation length
- Used by: Mistral 7B (W=4096)
- Cannot attend to tokens beyond the window (long-range recall limited)

### 15.7 Attention on iGPU

**What to offload:** QK^T computation and softmax (for long sequences)

**Decision framework:**
- For seq_len < 512: CPU is faster (iGPU dispatch overhead exceeds compute savings)
- For seq_len > 1024: iGPU may help IF bandwidth contention is managed
- Must keep KV cache in shared-memory region accessible by both CPU and iGPU

**Implementation with OpenCL:**
```c
// Launch attention kernel on iGPU
clEnqueueNDRangeKernel(queue, attention_kernel, 1, NULL, &global_size, &local_size, ...);
// Overlap with CPU work on next layer's FFN
```

---

## Section 16: KV Cache Management

### 16.1 Data Structures

#### Approach A: Contiguous Array (Simple)
```c
// Pre-allocate: [n_layers][2(K,V)][max_seq][n_kv_heads][head_dim]
float16 *kv_cache = aligned_alloc(64, total_kv_size);
```
- **Pros:** Simple, cache-friendly sequential access during attention
- **Cons:** Must pre-allocate max_seq_len, wastes memory if conversation is short

#### Approach B: Ring Buffer (Sliding Window)
```c
// Fixed-size circular buffer
int write_pos = seq_num % window_size;
kv_cache[layer][write_pos] = new_kv;
```
- **Pros:** Constant memory, perfect for sliding window attention
- **Cons:** Cannot attend beyond window

#### Approach C: Paged KV Cache (vLLM-style)
- Allocate in fixed-size pages (e.g., 256 tokens)
- Pages can be non-contiguous
- **Pros:** No wasted pre-allocation, dynamic
- **Cons:** Pointer chasing, worse locality, primarily a server optimization

#### Approach D: Attention Sinks (StreamingLLM)
- Keep only: first few tokens (attention sinks) + recent window
- Enables "infinite" context with fixed memory
- Quality: good for recent context, no long-range recall

### 16.2 KV Cache Quantization

| Precision | Size (per entry) | Compression | Quality Impact |
|---|---|---|---|
| FP32 | 4 bytes | 1x | Reference |
| FP16 | 2 bytes | 2x | Negligible |
| INT8 | 1 byte | 4x | Minimal (~0.1 ppl) |
| FP8 | 1 byte | 4x | Minimal |
| INT4 | 0.5 bytes | 8x | Noticeable (~0.5 ppl) |

**Per-channel quantization for keys:** Quantize each head_dim channel separately (keys have distinct value ranges per channel).

### 16.3 KV Cache Offloading to NVMe

For very long contexts (>8K tokens) that exceed RAM budget:
- Page out old KV cache blocks to NVMe
- Page in on demand with async I/O
- Latency: ~0.1-0.5 ms per page-in (vs ~0.001 ms from RAM)
- **For 3B ternary:** KV cache is small enough (~100-200 MB at FP16, 2048 context) that offloading is unnecessary

---

## Section 17: Memory Management

### 17.1 Arena Allocator Pattern

Avoid malloc/free during inference:

```c
typedef struct {
    uint8_t *base;
    size_t offset;
    size_t capacity;
} Arena;

void *arena_alloc(Arena *a, size_t size, size_t align) {
    a->offset = (a->offset + align - 1) & ~(align - 1);
    void *ptr = a->base + a->offset;
    a->offset += size;
    return ptr;
}
```

- Pre-allocate all buffers at startup
- Bump-pointer sub-allocation (zero overhead)
- Reset between layers or conversations
- **What llama.cpp does:** ggml_context with arena allocation

### 17.2 Double-Buffer Pattern

Two activation buffers, alternate between layers:

```c
float *buf_A = arena_alloc(arena, d * sizeof(float), 64);
float *buf_B = arena_alloc(arena, d * sizeof(float), 64);

for (int layer = 0; layer < n_layers; layer++) {
    float *input  = (layer % 2 == 0) ? buf_A : buf_B;
    float *output = (layer % 2 == 0) ? buf_B : buf_A;

    forward_layer(output, input, weights[layer], ...);
}
```

Avoids allocating separate buffers per layer. Total activation memory = 2 × buffer_size.

### 17.3 Cache-Line Padding for Thread Safety

```c
typedef struct {
    float accumulator[MAX_OUTPUT_PER_THREAD];
    char padding[64 - (MAX_OUTPUT_PER_THREAD * sizeof(float)) % 64];
} ThreadOutput;

ThreadOutput thread_outputs[NUM_THREADS] __attribute__((aligned(64)));
```

Ensures no two threads share a cache line → eliminates false sharing.

---

## Section 18: Threading & Parallelism

### 18.1 Thread Pool Designs

#### Approach A: OS Thread Pool (Windows Thread Pool API)
```c
PTP_WORK work = CreateThreadpoolWork(callback, context, NULL);
SubmitThreadpoolWork(work);
WaitForThreadpoolWorkCallbacks(work, FALSE);
```
- **Pros:** Managed by OS, handles core count dynamically
- **Cons:** No control over which cores, may assign SIMD work to E-cores

#### Approach B: Custom Thread Pool with Explicit Affinity
```c
typedef struct {
    HANDLE thread;
    HANDLE wake_event;
    HANDLE done_event;
    volatile int task_type;
    void *task_data;
} Worker;

Worker workers[6];  // One per P-core

for (int i = 0; i < 6; i++) {
    workers[i].thread = CreateThread(NULL, 0, worker_func, &workers[i], 0, NULL);
    // Pin to P-core i
    SetThreadAffinityMask(workers[i].thread, 1ULL << (i * 2));  // Skip HT sibling
}
```
- **Pros:** Full control, deterministic scheduling, guaranteed P-core for SIMD
- **Cons:** Must implement synchronization primitives

#### Approach C: OpenMP
```c
#pragma omp parallel for num_threads(6)
for (int i = 0; i < M; i++) {
    compute_output_row(i);
}
```
- **Pros:** Easy, compiler handles threading
- **Cons:** Less control over affinity, may not respect P/E topology
- **What llama.cpp does:** OpenMP or pthreads (configurable)

#### Approach D: Work-Stealing Scheduler
- Each thread has a deque of tasks
- Idle threads steal from busy threads' deques
- **Pros:** Dynamic load balancing (good when work is uneven)
- **Cons:** Complexity, steal overhead, overkill for LLM inference (work is evenly partitioned)

### 18.2 Core Affinity Strategy for Alder Lake

**Topology discovery (Windows):**
```c
// GetLogicalProcessorInformationEx(RelationProcessorCore, ...)
// Enumerate: which logical processors are P-cores vs E-cores
// Typical: P-cores = logical 0-11 (6 cores × 2 HT), E-cores = logical 12-19
```

**Recommended assignment:**
| Core Type | Thread Count | Purpose |
|---|---|---|
| 6 P-cores (no HT) | 6 threads | Heavy SIMD matmul work |
| E-cores (subset) | 1-2 threads | I/O, prefetching, iGPU submission |
| E-cores (subset) | 2-4 threads | Draft model for speculative decoding |

**Why avoid HT for SIMD:** Two SIMD threads on one P-core share execution ports, getting ~60% throughput each vs 100% with one thread.

**Measured improvement from affinity pinning:** 15-25% over default OS scheduling.

### 18.3 Parallelism Patterns

**Data parallelism (prefill):** Split batch of N prompt tokens across threads. Each thread processes N/T tokens through same layer.

**Tensor parallelism (decode):** Split weight matrix across threads. Each P-core computes M/6 output rows. Requires no reduction for independent rows.

**Pipeline parallelism:** Different layers on different cores. Hard to balance due to sequential dependency — each layer depends on previous layer's output.

### 18.4 E-Core Utilization

E-cores run SIMD at 50-70% of P-core single-thread performance. But if 6 P-cores already saturate DDR4 bandwidth, E-cores add NO value for bandwidth-bound matmul.

**Best uses during inference:**
1. Prefetching next layer weights into L3 (shared cache benefits P-cores)
2. Running draft model for speculative decoding
3. KV cache management (quantization, compression)
4. Tokenization/detokenization
5. iGPU synchronization and submission
6. Background OS work isolation (pin OS threads to E-cores)

---

# PART V: ADVANCED TECHNIQUES & INTEGRATION

---

## Section 19: iGPU Offloading

### 19.1 API Comparison

| API | Launch Latency | Setup Complexity | Driver Stability | DirectStorage | Recommendation |
|---|---|---|---|---|---|
| OpenCL 3.0 | 10-50 us | Low | Good | No | **Best starting point** |
| SYCL/Level Zero | 5-20 us | Medium | Good | No | Best performance |
| Vulkan Compute | 5-15 us | High | OK | No | Best portability |
| DX12 Compute | 5-20 us | Medium | Good | Yes | Best Windows integration |

### 19.2 What to Offload vs What to Keep on CPU

| Operation | CPU Time | iGPU Benefit | Bandwidth Contention | Verdict |
|---|---|---|---|---|
| Weight matmul (GEMV) | 8-10 ms | None (bandwidth-bound) | HIGH | **Keep on CPU** |
| Attention (QK^T) short | 0.02 ms | None (too small) | Low | **Keep on CPU** |
| Attention (QK^T) long | 0.1-0.3 ms | Possible | Medium | **Maybe offload** |
| Softmax | 0.01 ms | None (too small) | None | **Keep on CPU** |
| RMSNorm | 0.01 ms | None (too small) | None | **Keep on CPU** |
| RoPE | 0.005 ms | None | None | **Keep on CPU** |
| Output head (large vocab) | 0.5-1 ms | Possible | Medium | **Maybe offload** |

**Key insight:** On shared-memory architecture, the iGPU COMPETES with CPU for bandwidth. Any iGPU work that reads significant data from RAM slows down the CPU's weight streaming.

### 19.3 Pipeline Designs

#### Design A: CPU-Only (No iGPU)
- Simplest. All work on CPU.
- If 6 P-cores can achieve target speed without iGPU, this is best.
- **Valid outcome:** If shared-memory contention makes iGPU net-negative, skip it entirely.

#### Design B: Time-Sliced (Sequential)
```
Time: ─────CPU matmul──────┤──iGPU attention──┤─────CPU matmul──────
```
- No bandwidth contention (only one active at a time)
- Total time = CPU_time + iGPU_time
- Only helps if iGPU_time < CPU_attention_time

#### Design C: Overlapped (Pipelined)
```
CPU:  ─────Layer L FFN──────┤──────Layer L+1 FFN──────
iGPU: ───Layer L+1 attention─┤──────Layer L+2 attention──
```
- CPU processes layer L FFN while iGPU processes layer L+1 attention
- Risks bandwidth contention if both access RAM simultaneously
- Works if attention data is small relative to FFN weights

#### Design D: Hybrid with Double-Buffering
- Preload attention data into iGPU local memory during CPU FFN
- iGPU computes from local memory (no RAM contention during compute)
- Requires careful scheduling of data transfers

### 19.4 CPU-iGPU Data Transfer

**Zero-copy (shared memory):**
```c
// OpenCL: use host pointer directly
cl_mem buf = clCreateBuffer(ctx, CL_MEM_USE_HOST_PTR, size, host_ptr, NULL);
```
- No explicit copy — same physical RAM
- But: coherency protocol overhead, may flush caches

**USM (Unified Shared Memory) in SYCL:**
```c++
auto *ptr = sycl::malloc_shared<float>(size, queue);
// Accessible by both CPU and iGPU
```
- Simplest programming model
- Driver handles page migration and coherency

---

## Section 20: Decoding Strategies

### 20.1 Standard Autoregressive Decoding

One token at a time: forward pass → sample → append to context → repeat.

Each token requires reading ALL model weights from RAM. This is the baseline that all other strategies improve upon.

### 20.2 Speculative Decoding

**Core idea:** Small draft model proposes K tokens cheaply. Large target model verifies all K at once (batch forward pass).

```
Draft model (0.5B ternary):
  Proposes: [token1, token2, token3, token4]    (4 forward passes, each very fast)

Target model (3B ternary):
  Verifies: [token1, token2, token3, token4]    (1 batch forward pass with N=4)
  Accepts:  [token1, token2, token3]             (token4 rejected, regenerate)

Net: 3 tokens generated in time ≈ 1.2x single-token cost
Effective speedup: ~2.5x
```

**Acceptance rate:** Typically 60-80% for a well-matched draft model. Higher for simpler text (code comments), lower for creative/diverse text.

**Variants:**

| Variant | Draft Source | Extra Model? | Training Needed? | Typical Speedup |
|---|---|---|---|---|
| Independent draft | Separate small LLM | Yes | No (use existing) | 1.5-2.5x |
| Self-speculative | Early-exit from main model | No | No | 1.3-1.8x |
| Medusa | Multiple lightweight heads | No | Yes (train heads) | 1.5-2.5x |
| Lookahead (Jacobi) | Fixed-point iteration | No | No | 1.3-2x |
| EAGLE | Autoregressive draft head | No | Yes (train head) | 2-3x |

**Implementation on i7-12700H:**
- Draft model on E-cores (4 threads), target model on P-cores (6 threads)
- Must share memory bus: draft model bandwidth may slow target model
- Alternative: run draft and verify sequentially (no bandwidth contention)

### 20.3 Continuous Batching

Server optimization: batch requests from multiple users. NOT relevant for single-user edge inference.

### 20.4 Non-Autoregressive / Parallel Decoding

Generate multiple tokens simultaneously (diffusion-based, CMLM). Quality significantly worse than autoregressive. Not practical for production at current state.

---

## Section 21: Sampling

### 21.1 Complete Sampling Pipeline

```
raw_logits (vocab_size floats)
    │
    ▼
[Repetition Penalty] ─→ penalize recently generated tokens
    │
    ▼
[Temperature Scaling] ─→ logits / T
    │
    ▼
[Top-K Filter] ─→ keep only top K logits, -inf the rest
    │
    ▼
[Top-P (Nucleus)] ─→ keep smallest set with cumsum > P
    │
    ▼
[Min-P Filter] ─→ keep tokens with prob >= min_p × max_prob
    │
    ▼
[Softmax] ─→ convert to probabilities
    │
    ▼
[Random Sample] ─→ pick token according to probability distribution
    │
    ▼
next_token_id
```

### 21.2 Method Details

| Method | What It Does | Typical Value | Effect |
|---|---|---|---|
| Greedy | argmax(logits) | — | Deterministic, repetitive |
| Temperature | logits / T | T=0.7-1.0 | Lower = more deterministic |
| Top-K | Keep top K tokens | K=40-100 | Removes unlikely tokens |
| Top-P | Keep tokens summing to P probability | P=0.9-0.95 | Adaptive K per token |
| Min-P | Keep tokens with prob >= min_p × max_prob | min_p=0.05-0.1 | Simpler than top-P |
| Repetition penalty | Multiply recent token logits by 1/penalty | penalty=1.1-1.3 | Reduces repetition |

**Total sampling time: <0.1 ms.** Never a bottleneck. ~200 lines of C to implement.

---

## Section 22: NVMe / Storage Streaming

### 22.1 When NVMe Streaming Is Needed

| Scenario | Model Fits in RAM? | Need NVMe? |
|---|---|---|
| 3B ternary (0.6 GB) | Yes | No (only for initial load) |
| 3B Q4_K_M (1.9 GB) | Yes | No |
| 7B Q4_K_M (4.2 GB) | Yes (tight) | Maybe for KV cache |
| 13B Q4_K_M (~7.5 GB) | Barely | Yes |
| 70B Q4_K_M (~40 GB) | No | Yes (mandatory) |
| MoE (active experts) | Depends | Yes (flash-moe approach) |

### 22.2 Windows I/O Approaches Summary

| Approach | Speed | Latency | Caching | Complexity | Best For |
|---|---|---|---|---|---|
| fread (buffered) | 2-4 GB/s | Medium | OS cache | Low | Simple loads |
| mmap (CreateFileMapping) | 3-5 GB/s first, instant cached | Variable (page faults) | OS cache | Low | Repeated access |
| Unbuffered (NO_BUFFERING) | 4-7 GB/s | Predictable | Bypass | Medium | Streaming |
| Overlapped (async) | 4-7 GB/s | Async | Configurable | High | Pipeline overlap |
| IOCP | 4-7 GB/s | Async | Configurable | High | Many concurrent ops |
| DirectStorage | 5-7 GB/s | Low | Bypass | Medium | iGPU-direct loads |

### 22.3 Flash-MoE Key Findings on I/O

1. **Trust the OS page cache:** Removing application-level caching and using mmap improved throughput by 38%
2. **Multiple parallel reads:** 4 concurrent pread() calls via dispatch_apply maximized NVMe bandwidth
3. **2-bit requantization:** Reduced expert weights by 44%, reducing I/O bandwidth needed
4. **Session KV caching:** Persisting KV cache between conversation turns reduced second-turn TTFT by ~98%

---

## Section 23: System-Level Tuning

### 23.1 Compiler Flags

| Compiler | Flags | Notes |
|---|---|---|
| MSVC | `/O2 /arch:AVX2 /fp:fast /GL` | `/GL` enables whole-program optimization |
| Clang | `-O3 -march=alderlake -mavx2 -mavxvnni -mfma -flto` | Best codegen for SIMD (5-15% faster than MSVC) |
| GCC | `-O3 -march=alderlake -mavx2 -mavxvnni -mfma -flto` | Similar to Clang |

**Critical:** `-march=alderlake` (or `-march=native`) enables ALL Alder Lake features including AVX-VNNI.

**Clang vs MSVC:** Clang typically generates 5-15% faster code for SIMD-heavy workloads. Consider using clang-cl on Windows.

### 23.2 Profile-Guided Optimization (PGO)

```bash
# Step 1: Build with profiling instrumentation
clang -fprofile-generate -O3 -march=native -o edgelm_profile edgelm.c

# Step 2: Run representative workload
./edgelm_profile --benchmark

# Step 3: Rebuild using profile data
clang -fprofile-use=default.profdata -O3 -march=native -o edgelm edgelm.c
```

**Expected improvement:** 5-15% for branch-heavy code. Less dramatic for SIMD-heavy kernels (already predictable branches).

### 23.3 Link-Time Optimization (LTO)

Enables cross-file inlining and optimization:
- `-flto=thin`: Faster builds, nearly same optimization quality
- `-flto=full`: Maximum optimization, slower builds

**Expected improvement:** 3-10%

### 23.4 Large Pages Setup on Windows

1. Open Local Security Policy (secpol.msc)
2. Local Policies → User Rights Assignment → "Lock pages in memory"
3. Add your user account
4. Reboot
5. In code: `VirtualAlloc(NULL, size, MEM_RESERVE | MEM_COMMIT | MEM_LARGE_PAGES, PAGE_READWRITE)`

### 23.5 BIOS Settings

| Setting | Recommended | Why |
|---|---|---|
| XMP Profile | Enabled | Without XMP, DDR4 may default to 2400 MHz (25% less bandwidth) |
| iGPU DVMT | 256 MB | Enough for compute, preserves RAM for CPU |
| Turbo Boost | Enabled | Higher clocks = faster compute |
| C-States | Keep enabled | Minimal impact on sustained workloads |
| VT-d | Doesn't matter | No virtualization needed for inference |

### 23.6 Profiling Tools

| Tool | What It Measures | When to Use |
|---|---|---|
| Intel VTune | CPU hotspots, port utilization, cache misses, bandwidth | Primary profiling tool |
| Intel Advisor | Vectorization efficiency, roofline analysis | Checking SIMD utilization |
| Windows Performance Analyzer | System-level: scheduling, I/O, interrupts | OS-level issues |
| QueryPerformanceCounter | Sub-microsecond timing | Micro-benchmarks |
| __rdtsc() | Cycle-accurate timing | Innermost loop timing |

---

## Section 24: Inference Runtimes Compared

### 24.1 Comparison Matrix

| Feature | llama.cpp | bitnet.cpp | T-MAC | llamafile | ONNX Runtime | MLC-LLM | Custom Engine |
|---|---|---|---|---|---|---|---|
| **Language** | C/C++ | C/C++ (fork of llama.cpp) | C++ | C/C++ (cosmopolitan) | C++ | Python/TVM | C |
| **Model format** | GGUF | SafeTensors | Custom | GGUF | ONNX | TVM compiled | Custom |
| **Ternary support** | No (IQ1_S closest) | **Yes (native)** | **Yes** | No | No | Partial | **Yes** |
| **Quantization** | Q4_0 through IQ1_S | 1.58-bit native | 1-4 bit LUT | Same as llama.cpp | INT8/INT4 QDQ | INT4/INT8 | Ternary native |
| **AVX2 kernels** | Hand-written | Hand-written | Hand-written | tinyBLAS | MKL-DNN | Auto-generated | Hand-written |
| **AVX-VNNI** | Partial | Yes (TL2 kernel) | Via LUT | Partial | Via MKL-DNN | Via auto-tune | Yes |
| **iGPU support** | SYCL backend (experimental) | No | No | No | DirectML | Vulkan backend | OpenCL/SYCL |
| **Threading** | OpenMP/pthreads | OpenMP/pthreads | Custom | pthreads | Thread pool | TVM runtime | Custom affinity |
| **Core affinity** | No | No | No | No | No | No | **Yes (P/E aware)** |
| **Speculative decode** | Yes | No | No | Yes (via llama.cpp) | No | No | Yes (planned) |
| **tok/s on i7-12700H (3B)** | 5-7 (Q4_K_M) | **30-46 (ternary)** | ~40-50 (est.) | ~5-7 (Q4_K_M) | ~3-5 | ~5-10 | **Target: 100+** |

### 24.2 What to Learn From Each

**llama.cpp:** GGUF parser, Q4 kernel patterns, tokenizer, sampling pipeline, server infrastructure
**bitnet.cpp:** Ternary weight packing formats (I2_S, TL2), ternary kernel architectures, how to handle ternary inference
**T-MAC:** LUT-based computation approach, VPSHUFB utilization, group-based processing
**llamafile:** tinyBLAS kernel designs (CPU-tuned GEMM), single-binary deployment, cosmopolitan libc
**flash-moe:** I/O pipeline design, OS page cache trust, KV cache persistence, 2-bit requantization

---

## Section 25: End-to-End Pipeline Integration

### 25.1 Complete Data Flow With Timing

```
USER PROMPT: "What is the meaning of life?"
    │
    ▼
1. TOKENIZE (custom C BPE)                         [~0.5 ms]
   "What is the meaning of life?" → [1722, 374, 279, 7438, 315, 2324, 30]
    │
    ▼
2. EMBED (table lookup)                             [~0.01 ms]
   7 token IDs → 7 embedding vectors (7 × 3200 FP32)
    │
    ▼
3. PREFILL (batch forward pass, N=7 tokens)         [~15-30 ms]
   Process all 7 tokens through all 28 layers simultaneously
   ├── Batch GEMM (amortizes weight reads across 7 tokens)
   ├── Populate KV cache for positions 0-6
   └── Output: logits for position 6 (last token)
    │
    ▼
4. SAMPLE first output token                         [~0.05 ms]
   logits → top-p → sample → token_id
    │
    ▼
5. DECODE LOOP (autoregressive, one token at a time):
   ┌─────────────────────────────────────────────┐
   │ For each output token:                       │  [~10 ms per token]
   │                                              │
   │  a. Embed token_id → vector                  │  [0.01 ms]
   │  b. For each of 28 layers:                   │
   │     - RMSNorm                                │  [0.01 ms]
   │     - QKV ternary GEMV (6 P-core threads)    │  [0.05 ms]
   │     - RoPE                                   │  [0.005 ms]
   │     - Attention: QK^T + softmax + SV          │  [0.05 ms]
   │     - Output projection GEMV                 │  [0.03 ms]
   │     - Residual add                           │  [0.002 ms]
   │     - RMSNorm                                │  [0.01 ms]
   │     - FFN: gate+up+down GEMV (×3)            │  [0.15 ms] ← DOMINANT
   │     - Residual add                           │  [0.002 ms]
   │  c. Final RMSNorm                            │  [0.01 ms]
   │  d. Output head GEMV                         │  [0.5 ms]
   │  e. Sample next token                        │  [0.05 ms]
   │  f. Update KV cache                          │  [0.01 ms]
   │  g. Yield token → stream to user             │  [~0 ms]
   │                                              │
   │  Total per token: ~9-10 ms = 100-111 tok/s   │
   └─────────────────────────────────────────────┘
    │
    ▼
6. DETOKENIZE (table lookup)                        [~0.01 ms per token]
   token_id → text string → display to user
```

### 25.2 Thread Orchestration

```
P-Core 0-5 (6 threads):  ████ GEMV ████  ▒ barrier ▒  ████ GEMV ████  ▒ barrier ▒ ...
E-Core 0 (1 thread):     ░░░ prefetch next layer ░░░  ░░░ prefetch ░░░  ...
E-Core 1 (1 thread):     ░░░ iGPU submit/sync ░░░  ░░░ ...
E-Core 2-5 (4 threads):  ░░░ draft model (speculative) ░░░  ...
```

### 25.3 Startup Sequence

```
1. Load model file (mmap or read)                [~100 ms for 0.6 GB]
2. Parse header + metadata                        [~1 ms]
3. Repack weights to SIMD-optimal layout          [~200 ms first time, 0 if cached]
4. Initialize tokenizer (parse vocab + merges)    [~5 ms]
5. Allocate KV cache + activation buffers         [~1 ms]
6. Spawn thread pool, pin to cores                [~1 ms]
7. Initialize iGPU context (if used)              [~50-100 ms]
8. Compile iGPU kernels (if used)                 [~100-500 ms first time, cached after]
9. Warm up (1 dummy forward pass)                 [~10 ms]
──────────────────────────────────────────────────
Total cold start: ~0.5-1.0 seconds
Total warm start (model cached): ~0.1-0.3 seconds
```

### 25.4 Session Management

**Multi-turn conversation:**
- Preserve KV cache between turns
- flash-moe finding: KV cache persistence reduces second-turn TTFT by ~98%
- Only process NEW tokens through prefill (existing context already in KV cache)

**Context overflow handling:**
1. **Truncate oldest:** Drop tokens from the beginning (simplest)
2. **Sliding window:** Shift window forward, keeping last N tokens
3. **Attention sinks + window:** Keep first 4 tokens + last N tokens (StreamingLLM)
4. **External summarization:** Summarize old context, inject summary (requires API call)

### 25.5 Quality Assurance

**Numerical correctness:**
- Ternary matmul with INT32/FP32 accumulation is EXACT (no rounding in add/subtract)
- Compare output logits with reference implementation (BitNet's Python model)
- Tolerance: logits should match to FP32 precision

**Regression testing:**
- Fixed prompts with known outputs
- Verify token-for-token match with temperature=0 (greedy)

**Perplexity measurement:**
- WikiText-2 perplexity as quality metric
- Should match reference within 0.1 perplexity points

---

# APPENDICES

---

## Appendix A: Theoretical Performance Model

### Roofline Model

```
                    │
    GOPS            │           ┌──── Compute ceiling (787 GOPS INT8)
                    │          ╱
                    │         ╱
                    │        ╱
                    │       ╱
                    │      ╱   ← Bandwidth ceiling slope (40 GB/s)
                    │     ╱
                    │    ╱
                    │   ╱ ★ Ternary GEMV (4 ops/byte → 160 GOPS achievable)
                    │  ╱
                    │ ╱   ★ INT4 GEMV (2 ops/byte → 80 GOPS)
                    │╱    ★ INT8 GEMV (1 op/byte → 40 GOPS)
                    └──────────────────────────
                         Arithmetic Intensity (ops/byte)
```

The star positions show where each quantization format falls on the roofline. ALL are in the bandwidth-bound region (left of the ridge point at 20 ops/byte).

### Sensitivity Analysis

| Variable | Range | Impact on tok/s |
|---|---|---|
| DDR4 bandwidth (real) | 35-42 GB/s | ±10% |
| Model size (ternary) | 0.4-0.6 GB | ±20% |
| Bandwidth utilization | 70-95% | ±15% |
| Speculative decode rate | 1.0-2.5x | Up to +150% |
| Number of P-cores used | 4-6 | ±15% |
| Large pages vs 4KB | — | +10-25% |
| iGPU offload | — | -5% to +15% (uncertain) |

---

## Appendix B: AVX2/VNNI Instruction Quick Reference

### Most Used Instructions for Inference

| Instruction | What It Does | Throughput (P-core) | Port |
|---|---|---|---|
| VPDPBUSD | 32× UINT8×INT8 dot product + accumulate | 1/cycle | 0 |
| VPMADDUBSW | 32× UINT8×INT8 → 16 INT16 sums | 1/cycle | 0 |
| VPMADDWD | 16× INT16 → 8 INT32 sums | 1/cycle | 0 |
| VPADDD | 8× INT32 add | 2/cycle | 0, 10 |
| VPAND | 256-bit AND | 2/cycle | 0, 10 |
| VPOR | 256-bit OR | 2/cycle | 0, 10 |
| VPXOR | 256-bit XOR | 2/cycle | 0, 10 |
| VPSHUFB | 32× byte shuffle (LUT lookup) | 1/cycle | 5 |
| VPBLENDVB | Conditional byte blend | 1/cycle | 0 or 5 |
| VFMADD231PS | 8× float FMA: a = a + b×c | 2/cycle | 0, 1 |
| VMOVDQA | 256-bit aligned load | 3/cycle | 2, 3, 8 |
| VMOVDQU | 256-bit unaligned load | 3/cycle | 2, 3, 8 |
| VMOVNTDQ | 256-bit non-temporal store | 1/cycle | 4 |
| VRSQRTPS | 8× approx reciprocal sqrt | 1/cycle | 0 |

---

## Appendix C: Glossary

| Term | Definition |
|---|---|
| **GEMV** | General Matrix-Vector multiply (y = Wx) |
| **GEMM** | General Matrix-Matrix multiply (Y = WX) |
| **tok/s** | Tokens generated per second during decode |
| **TTFT** | Time To First Token (latency before first output) |
| **KV cache** | Stored key and value vectors from previous tokens |
| **Ternary** | Weight values restricted to {-1, 0, +1} |
| **BitNet** | Architecture with ternary weights (trained, not post-hoc) |
| **Roofline** | Performance model showing compute vs bandwidth limits |
| **P-core** | Performance core (Golden Cove, big) |
| **E-core** | Efficiency core (Gracemont, small) |
| **AVX2** | Advanced Vector Extensions 2 (256-bit SIMD) |
| **AVX-VNNI** | Vector Neural Network Instructions (INT8 dot product) |
| **VPDPBUSD** | The AVX-VNNI dot product instruction |
| **VPSHUFB** | Byte shuffle — used as 16-entry LUT |
| **PTQ** | Post-Training Quantization |
| **QAT** | Quantization-Aware Training |
| **GQA** | Grouped-Query Attention |
| **MQA** | Multi-Query Attention |
| **MoE** | Mixture of Experts |
| **SSM** | State Space Model (Mamba) |
| **SiLU** | Sigmoid Linear Unit activation: x × sigmoid(x) |
| **RMSNorm** | Root Mean Square Layer Normalization |
| **RoPE** | Rotary Position Embeddings |
| **mmap** | Memory-mapped file I/O |
| **TLB** | Translation Lookaside Buffer (page table cache) |
| **IOCP** | I/O Completion Port (Windows async I/O) |
| **PGO** | Profile-Guided Optimization |
| **LTO** | Link-Time Optimization |

---

## Appendix D: Cross-Reference Map

```
Section 4 (Bandwidth Wall) ──────→ Section 10 (Quantization) ──→ Section 14 (Kernels)
    │                                      │
    └──→ Section 5 (iGPU) ──────────→ Section 19 (iGPU Offloading)
                                           │
Section 2 (SIMD) ─────────────────→ Section 14 (Matmul Kernels)
    │                                      │
    └──→ Section 14.9 (AVX-VNNI) ──→ Section 14.10 (T-MAC LUT)
                                           │
Section 3 (Cache) ─────────────────→ Section 14.14 (Tiling)
    │                                      │
    └──→ Section 17 (Memory Mgmt) ──→ Section 14.13 (Prefetching)

Section 8 (Architectures) ─────────→ Section 13 (Forward Pass)
    │                                      │
    └──→ Section 15 (Attention) ────→ Section 16 (KV Cache)

Section 18 (Threading) ────────────→ Section 14.15 (Multi-threaded GEMV)
    │                                      │
    └──→ Section 25 (Pipeline) ─────→ Section 20 (Speculative Decode)

Section 6 (NVMe) ──────────────────→ Section 22 (NVMe Streaming)
    │
    └──→ Section 16.3 (KV Cache Offload)
```

---

*This document covers the complete landscape as of early 2026. The field moves fast — new quantization methods, architectures, and hardware emerge regularly. Use this as a reference framework, not a final answer.*
