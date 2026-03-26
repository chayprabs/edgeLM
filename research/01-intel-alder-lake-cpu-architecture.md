# Section 01: CPU Architecture -- Intel Alder Lake (12th Gen) -- Extended Research

## Overview

This section covers the Intel i7-12700H CPU microarchitecture, the foundational hardware that EdgeLM targets. Understanding every quirk of the Golden Cove P-cores and Gracemont E-cores is critical because our inference engine must be hand-tuned for this specific silicon. The CPU architecture dictates how we design our SIMD kernels, schedule threads, manage caches, and overlap computation -- every optimization decision downstream depends on the details documented here.

## What the Deep Dive Already Covers

- Hybrid architecture: 6 P-cores (Golden Cove) + 8 E-cores (Gracemont), 20 threads total
- P-core max turbo: 4.7 GHz single, 4.1 GHz all-core; E-core: 3.5 GHz
- AVX2 + AVX-VNNI available; AVX-512 and AMX fused off
- Thread Director doesn't understand LLM workloads; manual affinity needed
- HyperThreading: two SIMD threads per P-core get ~60% throughput each
- 11 execution ports listed with key instruction mappings (Port 0=FMA/VNNI, Port 5=shuffle, etc.)
- Recommendation: 6 threads (one per P-core) for SIMD work

---

## New Findings

### 1. Golden Cove Microarchitecture -- Deeper Pipeline Details

#### 1.1 Front-End: 6-Wide Decode with Eliminated Complex Decoder

- **Source:** [WikiChip - Golden Cove](https://en.wikichip.org/wiki/intel/microarchitectures/golden_cove), [WikiChip Fuse](https://fuse.wikichip.org/news/6111/intel-details-golden-cove-next-generation-big-core-for-client-and-server-socs/)
- **Key idea:** Golden Cove moved from 4+1 (4 simple + 1 complex) decoders to 6 simple decoders, with the complex decoder eliminated. Complex instructions are now handled by macro-op fusion in the instruction sequencer. This is the first 6-wide x86 decode.
- **Relevance to EdgeLM:** Our tight AVX2 kernel loops will primarily hit the uop cache (DSB path), not the decoders. But for cold-start paths (first invocation of a kernel), wider decode means faster ramp-up.
- **Estimated impact:** Minimal for hot loops (uop cache dominates), moderate for cold paths.
- **Implementation complexity:** Low -- this is passive; just be aware the decode width exists.
- **Details:**
  - Fetch bandwidth doubled to 32 bytes/cycle (from 16)
  - uop cache: **4K entries** (up from 2.25K in Sunny Cove, 1.5K in Skylake)
  - uop cache delivery: **8 uops/cycle** (up from 6)
  - Allocation width: 6 uops/cycle
  - Retirement width: increased (exact number not public, estimated ~8/cycle)

#### 1.2 Back-End: Expanded Out-of-Order Resources

- **Source:** [Wikipedia - Golden Cove](https://en.wikipedia.org/wiki/Golden_Cove), [Hardware Times - P-Core Comparison](https://hardwaretimes.com/skylake-vs-sunny-cove-vs-golden-cove-vs-redwood-cove-vs-lion-cove-intel-p-core-architectures/)
- **Key idea:** Golden Cove massively expanded OoO resources, enabling the CPU to find more instruction-level parallelism.
- **Relevance to EdgeLM:** Larger ROB and scheduler means interleaved prefetch + compute instructions can execute more efficiently without stalling.
- **Details:**
  - **ROB:** 512 entries (up from 352 in Sunny Cove) -- second only to Apple's ~630
  - **ALU Scheduler:** 97 entries (up from 80 in Sunny Cove)
  - **Load Buffer:** 192 entries (up from 128)
  - **Store Buffer:** 114 entries (up from 72)
  - **L1D Fill Buffers:** 16 (up from 12) -- more outstanding memory transactions
  - **Load ports:** 3 (Ports 2, 3, 8) -- up from 2, enabling 96 bytes/cycle from L1
  - **Partial store forwarding** supported (store-to-load forwarding even when load only partially overlaps store)

#### 1.3 Execution Ports -- 12 Total (Corrected from Deep Dive's 11)

- **Source:** [WikiChip - Golden Cove](https://en.wikichip.org/wiki/intel/microarchitectures/golden_cove)
- **Key idea:** Golden Cove has 12 execution ports (Ports 0-10 + Port 8 for load AGU), not 11 as listed in the deep dive. The deep dive was missing Port 8 detail.
- **Relevance to EdgeLM:** For our ternary kernel, we can theoretically sustain per cycle: 1 VPDPBUSD (Port 0) + 1 VPSHUFB (Port 5) + 2 VPAND/VPOR (Ports 0, 10) + 3 loads (Ports 2, 3, 8). Understanding this enables optimal instruction interleaving.
- **Critical port assignments for ternary kernels:**

| Instruction | Port(s) | Throughput | Latency |
|-------------|---------|------------|---------|
| VPDPBUSD (AVX-VNNI) | Port 0 only | 1/cycle | 5 cycles |
| VPSHUFB (LUT lookup) | Port 5 only | 1/cycle | 1 cycle |
| VPAND/VPOR/VPXOR | Port 0 or 10 | 2/cycle | 1 cycle |
| VPMADDUBSW | Port 0 | 1/cycle | 5 cycles |
| VPMADDWD | Port 0 | 1/cycle | 5 cycles |
| VPADDD | Port 0 or 10 | 2/cycle | 1 cycle |
| VMOVDQA (load) | Ports 2, 3, 8 | 3/cycle | ~4 cycles (L1) |
| VFMADD231PS | Ports 0, 1 | 2/cycle | 4 cycles |

- **Key insight:** VPDPBUSD and VPSHUFB are on different ports (0 vs 5), so they can execute in parallel. This means a hybrid kernel using both LUT (VPSHUFB) and dot-product (VPDPBUSD) paths simultaneously is theoretically possible if the instruction mix is balanced.

---

### 2. Gracemont E-Core -- AVX2 is Split 128-bit, Not Native 256-bit

#### 2.1 256-bit AVX2 Execution Method

- **Source:** [HWCooling - Gracemont Microarch Analysis](https://www.hwcooling.net/en/gracemont-the-not-so-little-alder-lake-core-microarch-analysis/3/), [Tom's Hardware - Architecture Day 2021](https://www.tomshardware.com/features/intel-architecture-day-2021-intel-unveils-alder-lake-golden-cove-and-gracemont-cores/3)
- **Key idea:** Gracemont handles 256-bit AVX2 by splitting operations into two 128-bit halves. The renamer performs micro-op splitting, turning each 256-bit vector operation into two 128-bit uops. This means AVX2 on E-cores runs at roughly half the throughput of P-cores per-clock, and E-cores also clock lower.
- **Relevance to EdgeLM:** This confirms E-cores should NEVER run our SIMD matmul kernels. A P-core at 4.1 GHz with native 256-bit AVX2 delivers roughly 4-6x the SIMD throughput of an E-core at 3.5 GHz with split 128-bit execution. The deep dive says "50-70%" throughput but this is misleading -- for 256-bit AVX2 specifically, it's much worse.
- **Estimated impact:** Critical -- wrong core assignment = 4-6x slower kernel execution.
- **Implementation complexity:** Low -- enforce via SetThreadAffinityMask.
- **Details:**
  - E-core AVX2 throughput (256-bit): ~1 op per 2 cycles (split into 128-bit halves)
  - P-core AVX2 throughput (256-bit): ~1 op per cycle (native 256-bit)
  - E-core AVX-VNNI (VPDPBUSD): Also split, so 1 per 2 cycles vs P-core's 1 per cycle
  - E-core ROB: ~256 entries (vs P-core's 512)
  - E-core L1D: 32 KB per cluster of 4 cores (shared!) vs P-core's 48 KB per core (private)
  - E-core L2: 2 MB per cluster of 4 cores (shared) vs P-core's 1.25 MB per core (private)
  - E-core L3 latency: **60+ cycles** (vs P-core's ~40-50 cycles) -- E-cores are further from L3 slices

#### 2.2 E-Core Practical Performance Ratio

- **Source:** [Overclock.net - Raptor Lake E-cores](https://www.overclock.net/threads/how-good-are-the-raptor-lake-e-cores.1801881/), [AnandTech Forums - Geekbench Comparison](https://forums.anandtech.com/threads/fyi-comparison-of-alder-lakes-performance-vs-efficiency-cores-geekbench-5.2606589/)
- **Key idea:** For heavily vectorized SIMD workloads, an E-core delivers approximately 30-50% of a P-core's multi-threaded performance. For scalar integer work, it's closer to 50-60%.
- **Relevance to EdgeLM:** E-cores are suitable for: tokenization, sampling, KV cache management, prefetching orchestration, and running a speculative draft model. They should never touch the SIMD matmul hot path.

---

### 3. Core-to-Core Latency -- Cross-Type Communication is Expensive

#### 3.1 Measured Latencies

- **Source:** [nviennot/core-to-core-latency GitHub](https://github.com/nviennot/core-to-core-latency), [Issue #48 - i5-12600K Results](https://github.com/nviennot/core-to-core-latency/issues/48)
- **Key idea:** Communication between P-cores and E-cores goes through the shared L3 ring bus and incurs significantly higher latency than same-type communication.
- **Relevance to EdgeLM:** When designing the CPU+iGPU pipeline or speculative decoding (draft model on E-cores, main model on P-cores), data handoff between core types adds ~100ns+ overhead per synchronization point. Minimize cross-type communication.
- **Estimated impact:** Design-critical for pipeline architecture.
- **Implementation complexity:** Medium -- requires careful data flow design.
- **Measured latencies (i5-12600K, same generation):**

| Communication Path | Latency |
|---|---|
| P-core to P-core (same group of 3) | ~40 ns |
| P-core to P-core (cross-group) | ~42-45 ns |
| E-core to E-core (same cluster of 4) | ~53 ns |
| E-core to E-core (cross-cluster) | ~55-60 ns |
| **P-core to E-core (cross-type)** | **~145 ns** |

- **Critical insight:** The P-to-E latency (~145ns) is comparable to multi-socket NUMA latency on server CPUs. This means P-core/E-core handoffs are expensive. For speculative decoding, the draft model (E-cores) should write results to a shared buffer, and P-cores should poll it -- avoid tight synchronization primitives.

---

### 4. llama.cpp Empirical Findings -- P-Core Only is 2.4-3x Faster

#### 4.1 Community Benchmark Data

- **Source:** [llama.cpp Discussion #572 - 3x better with P-cores only](https://github.com/ggml-org/llama.cpp/discussions/572)
- **Key idea:** Real-world llama.cpp benchmarks on i7-12700H show dramatic speedup when restricting inference to P-cores only. E-cores become a bottleneck because P-cores finish their partition and spin-wait for E-cores.
- **Relevance to EdgeLM:** Directly validates our architectural decision to pin SIMD work to P-cores. Also reveals that naive work-splitting across heterogeneous cores is worse than not using E-cores at all.
- **Estimated impact:** 2.4-3x (already factored into our plan, but now empirically confirmed).
- **Implementation complexity:** Low.
- **Benchmark data (i7-12700H, Linux, numactl):**

| Model | All Cores (14 threads) | P-Cores Only (6 threads) | Speedup |
|---|---|---|---|
| 7B q4_0 | 481.99 ms/run | 201.83 ms/run | **2.4x** |
| 65B q4_0 | 4068.66 ms/run | 1357.09 ms/run | **3.0x** |

- **Root cause identified:** P-cores finish their partition first and spin-lock waiting for E-cores. The overall speed is limited by the slowest core (E-core). This is a classic load-imbalance problem in heterogeneous systems.
- **Workaround used:** `numactl -C 0-5 ./main -m <model> -t 6`
- **Additional finding:** Even with affinity set, E-cores sometimes showed 60-70% load -- the OS scheduler can override soft affinity hints. Hard affinity (CPU sets) is more reliable.

---

### 5. AVX2 Frequency Throttling on Alder Lake

#### 5.1 Which Instructions Throttle and Which Don't

- **Source:** [Extensa Tech - AVX Throttling Part 1](https://extensa.tech/blog/avx-throttling-part1/), [Cloudflare Blog - Dangers of Intel Frequency Scaling](https://blog.cloudflare.com/on-the-dangers-of-intels-frequency-scaling/), [Travis Downs - AVX Freq](https://travisdowns.github.io/blog/2020/01/17/avxfreq1.html)
- **Key idea:** Not all AVX2 instructions cause frequency throttling. Only "heavy" AVX2 instructions (FP multiply, FMA) trigger the frequency offset. Integer shuffles, adds, bitwise operations, and loads/stores run at full frequency indefinitely.
- **Relevance to EdgeLM:** Our ternary kernels primarily use VPADDB/VPSUBB (integer add/subtract), VPAND/VPOR (bitwise), VPSHUFB (shuffle), and VPDPBUSD (VNNI dot product). These are all "light" or integer instructions. **We should NOT experience AVX2 frequency throttling** for our primary ternary matmul workload. This is a significant advantage over FP-heavy inference engines.
- **Estimated impact:** 0% penalty for ternary kernels (positive finding -- no throttling).
- **Implementation complexity:** None -- just avoid FP FMA in hot paths.
- **Details:**
  - **Heavy (cause throttling):** VFMADD* (FP FMA), VMULPS/PD (FP multiply), VDIVPS/PD (FP divide)
  - **Light (no throttling):** VPADDB/W/D (integer add), VPSUBB/W/D (integer subtract), VPAND/VPOR/VPXOR (bitwise), VPSHUFB (shuffle), VPMADDUBSW/VPMADDWD (integer MAC), VPDPBUSD (VNNI), VMOVDQA (load/store)
  - **Alder Lake specific:** AVX2 negative ratio offset applies to P-cores only; E-cores are unaffected
  - **Two throttling levels:** Level 1 (light frequency reduction) for heavy AVX2, Level 2 (stronger reduction) for AVX-512 only
  - **Recovery time:** Not precisely documented, estimated 1-10 microseconds after heavy instructions stop

---

### 6. Macro-Op Fusion Rules for Golden Cove

#### 6.1 What Fuses and What Doesn't

- **Source:** [Corsix - x86 Macro-Op Fusion Notes](https://www.corsix.org/content/x86-macro-op-fusion-notes), [EasyPerf - MacroFusion in Intel CPUs](https://easyperf.net/blog/2018/02/23/MacroFusion-in-Intel-CPUs)
- **Key idea:** Golden Cove inherits Ice Lake's fusion rules, which removed INC/DEC fusion and memory operand fusion. CMP/TEST/AND + conditional jumps still fuse.
- **Relevance to EdgeLM:** Our inner loops should use CMP/SUB + JNZ patterns (fusible) instead of DEC + JNZ (not fusible on Ice Lake+). This saves one uop per loop iteration.
- **Estimated impact:** 5-10% for tight inner loops with many iterations.
- **Implementation complexity:** Low -- just choose the right loop counter pattern.
- **Fusible on Golden Cove:**
  - CMP + any conditional jump (JZ, JNZ, JB, JA, JL, JGE, etc.)
  - TEST + conditional jumps
  - AND + conditional jumps
  - ADD/SUB + conditional jumps (unsigned and signed comparisons, zero test)
- **NOT fusible on Golden Cove (changed from Sandy Bridge):**
  - INC/DEC + conditional jumps (removed in Ice Lake, stays removed)
  - Instructions with memory operands (removed in Ice Lake)
  - JO/JNO (overflow jumps) -- never fused on any Intel
- **Optimization tip:** Transform loop counters from `for(i=0; i<N; i++)` to `for(i=-N; i<0; i++)` so the ADD/SUB producing the loop condition is the same instruction that updates the counter, enabling fusion with the branch.

---

### 7. Thread Director -- Practical Override Methods

#### 7.1 CoreDirector Software (BitSum)

- **Source:** [Tom's Hardware - CoreDirector](https://www.tomshardware.com/news/bitsun-coredirector-intel-cpu-e-cores), [BitSum - CoreDirector](https://bitsum.com/apps/coredirector/)
- **Key idea:** CoreDirector provides three enforcement methods for controlling P-core/E-core assignment without writing custom code: Efficiency Mode OFF (soft), CPU Affinities (hard), CPU Sets (medium).
- **Relevance to EdgeLM:** While we'll use SetThreadAffinityMask in our engine, CoreDirector is useful for testing/benchmarking phases to quickly experiment with core assignments.
- **Implementation complexity:** None (external tool).
- **Enforcement methods:**
  1. **Efficiency Mode OFF:** Prevents Thread Director from demoting threads to E-cores. Soft -- OS can still use E-cores if needed.
  2. **CPU Affinities:** Hard restriction to P-cores only. Most reliable. Uses SetProcessAffinityMask internally.
  3. **CPU Sets:** Medium enforcement, OS retains some scheduling discretion.

#### 7.2 Windows API: SetThreadAffinityMask Details

- **Source:** [Intel - Managing Performance with Heterogeneous Cores](https://www.intel.com/content/www/us/en/docs/onemkl/developer-guide-windows/2023-0/managing-performance-with-heterogeneous-cores.html), [ElevenForum - Forcing Performance Mode](https://www.elevenforum.com/t/forcing-performance-mode-for-background-app-on-intel-p-e-core-cpu.15763/)
- **Key idea:** For programmatic control, SetThreadAffinityMask with a bitmask selecting P-core logical processors is the most reliable approach. Intel recommends P-core-only for predictable performance.
- **Relevance to EdgeLM:** Our thread pool initialization should call SetThreadAffinityMask for each SIMD worker thread, pinning to P-core logical CPUs (typically 0-11 for 6 P-cores with HT).
- **Implementation complexity:** Low.
- **Details:**
  - Use `GetLogicalProcessorInformationEx()` to enumerate core types at runtime
  - P-cores have `EfficiencyClass = 1` (higher = more capable) on Windows 11
  - E-cores have `EfficiencyClass = 0`
  - Set EPP (Energy Performance Preference) to 0 via `powercfg` for maximum performance mode

#### 7.3 Hardware P-States (HWP) and EPP

- **Source:** [Linux Kernel - intel_pstate Documentation](https://docs.kernel.org/admin-guide/pm/intel_pstate.html), [Microsoft - Power Performance Tuning](https://learn.microsoft.com/en-us/windows-server/administration/performance-tuning/hardware/power/power-performance-tuning)
- **Key idea:** HWP allows the CPU to adjust frequency every 1ms (vs Windows PPM's default 30ms check interval). Setting EPP to 0 (performance) ensures the CPU ramps to max frequency immediately for SIMD workloads.
- **Relevance to EdgeLM:** Set Windows power plan to "High Performance" or "Ultimate Performance" to ensure EPP=0. This eliminates frequency ramp-up latency at the start of inference.
- **Estimated impact:** 5-15% for short inference bursts (eliminates ramp-up delay).
- **Implementation complexity:** Low -- power plan setting.

---

### 8. OS-Level LLM Inference Optimization

#### 8.1 Thread Pinning Reduces Jitter from 11ms to Microseconds

- **Source:** [Eunomia - OS-Level Challenges in LLM Inference](https://eunomia.dev/blog/2025/02/18/os-level-challenges-in-llm-inference-and-optimizations/)
- **Key idea:** Without CPU isolation, worst-case scheduling delays reach 11ms (more than an entire token generation at 100 tok/s). Pinning threads to dedicated cores brings worst-case to tens of microseconds.
- **Relevance to EdgeLM:** At 100 tok/s, each token must complete in 10ms. A single 11ms scheduling delay would cause a visible stutter. Thread pinning is mandatory, not optional.
- **Estimated impact:** Eliminates tail latency spikes (P99 improvement).
- **Implementation complexity:** Low.
- **Techniques:**
  - **Thread pinning:** `SetThreadAffinityMask()` on Windows
  - **IRQ redirection:** Move hardware interrupts off inference cores (Windows: use Device Manager IRQ assignment or `Set-NetAdapterAdvancedProperty`)
  - **Process priority:** Set inference process to `HIGH_PRIORITY_CLASS` or `REALTIME_PRIORITY_CLASS`
  - **Huge pages:** 2MB pages reduce TLB misses (already in deep dive, confirmed critical here)
  - **Memory locking:** Use `VirtualLock()` to prevent OS from paging out model weights
  - **Pre-touching:** Read through allocated memory before first inference to fault all pages in

#### 8.2 Context Switch Overhead Measurements

- **Source:** [Eunomia - OS-Level Challenges](https://eunomia.dev/blog/2025/02/18/os-level-challenges-in-llm-inference-and-optimizations/)
- **Key idea:** Direct context switch cost: hundreds of nanoseconds to a few microseconds. But indirect costs (cache pollution, TLB flush, branch predictor contamination) multiply the total impact significantly.
- **Relevance to EdgeLM:** Reinforces the need for dedicated cores. Even a brief context switch flushes L1 data that took many cycles to warm up.

---

### 9. PMCSched -- ML-Based Scheduling for Alder Lake

#### 9.1 Performance Monitoring Counter-Based Scheduling

- **Source:** [PMCSched Paper - Wiley 2023](https://onlinelibrary.wiley.com/doi/full/10.1002/cpe.7814)
- **Key idea:** PMCSched uses ML models trained on hardware performance counter data to predict optimal core assignment for each thread. The inference procedure runs in microseconds and can be loaded as a kernel module on unmodified Linux kernels.
- **Relevance to EdgeLM:** While we're on Windows and will use manual affinity, the PMCSched research validates that PMC-guided scheduling outperforms Thread Director for specialized workloads. If we ever port to Linux, PMCSched could dynamically reassign the draft model between P-cores and E-cores based on workload phase.
- **Estimated impact:** Research reference only (Linux-specific).
- **Implementation complexity:** N/A (academic reference).
- **Key finding:** Thread Director's hardware classification can be suboptimal because it uses a fixed set of heuristics, while PMCSched's ML models adapt to the actual workload.

---

### 10. P-Core Silicon Lottery -- Not All P-Cores Are Equal

#### 10.1 Frequency Stability Varies Per Core

- **Source:** [SillyCross Blog - Bizarre Performance Characteristics of Alder Lake](https://sillycross.github.io/2022/06/11/2022-06-11/)
- **Key idea:** Individual P-cores on the same die can behave differently. The author found 3 categories: "gold" cores (stable 4.7 GHz), "B-grade" cores (stable 4.5 GHz), and "wild" cores (unstable, fluctuating 4.05-4.55 GHz).
- **Relevance to EdgeLM:** For maximum consistent performance, benchmark each P-core individually and pin the most SIMD-intensive threads to the most stable/fastest cores. E-cores showed no such instability (stable 3.5 GHz).
- **Estimated impact:** Up to 15% variance between best and worst P-core on the same chip.
- **Implementation complexity:** Medium -- requires per-core benchmarking at startup.
- **Recommended approach:**
  1. At engine startup, run a quick AVX2 microbenchmark on each P-core (10ms per core)
  2. Rank P-cores by sustained throughput
  3. Pin the main matmul threads to the top-ranked cores
  4. Use lower-ranked P-cores for less critical work

---

### 11. CPU Outperforming GPU for Small Model Inference

#### 11.1 "Challenging GPU Dominance" Paper (2025)

- **Source:** [arxiv.org/abs/2505.06461](https://arxiv.org/abs/2505.06461)
- **Key idea:** On mobile devices, CPU inference with optimal thread count outperforms GPU for models ≤1B parameters. The key insight: optimal thread count = number of performance cores. Thread oversubscription causes degradation.
- **Relevance to EdgeLM:** Validates our P-core-only SIMD strategy. Also reinforces that the number of compute threads should match P-core count exactly (6 in our case), not logical thread count.
- **Key findings:**
  - CPU (2 threads, P-cores only) = 17 tok/s vs GPU = 12.8 tok/s for Qwen2-0.5B (1.33x CPU win)
  - Q4 quantization delivers 1.5-2.5x speedup on CPU
  - GEMM = 87.6% of prefill time, 76.2% of decode time
  - GPU memory transfer overhead + Metal buffer sync makes GPU slower for small batches
  - For larger models (7B+), GPU wins due to higher total compute

---

### 12. Power Throttling Reality on i7-12700H Laptops

#### 12.1 PL1/PL2 Limits and Thermal Reality

- **Source:** [Tom's Hardware Forum - 12700H Throttling](https://forums.tomshardware.com/threads/how-to-stop-i7-12700h-from-power-limit-throttling.3769453/), [Intel Community - i7-12700H Overheating](https://community.intel.com/t5/Mobile-and-Desktop-Processors/intel-i7-12700H-overheating-throttling/td-p/1470612)
- **Key idea:** The i7-12700H has PL1=45W (sustained) and PL2=115W (burst for 28-56 seconds). Most laptop cooling solutions cannot sustain PL2 indefinitely. After the tau period, frequency drops to PL1 levels (~3.5-3.8 GHz all-core).
- **Relevance to EdgeLM:** Our benchmarks MUST account for thermal throttling. Short benchmarks (a few seconds) will show inflated numbers at PL2 frequencies. Sustained inference (chatbot use) will run at PL1 frequencies. Our target of 100+ tok/s must be achievable at the lower PL1 frequency.
- **Estimated impact:** 15-25% frequency reduction from PL2 to PL1 sustained.
- **Implementation complexity:** Low -- just set realistic benchmark methodology.
- **Details:**
  - PL1 (sustained): 45W → ~3.5-3.8 GHz all-core P-cores
  - PL2 (burst): 115W → 4.1 GHz all-core for ~28-56 seconds
  - Temperatures often hit 95-100°C at PL2 on thin laptops
  - Undervolting often blocked by OEM BIOS on 12th gen
  - XTU workaround: Set custom PL1=65W, PL2=65W for stable sustained performance
  - Repasting: 7-9°C reduction (worthwhile for sustained workloads)

#### 12.2 RAPL Energy Monitoring

- **Source:** [SPEC ICPE 2024 - Alder Lake Energy Efficiency](https://dl.acm.org/doi/10.1145/3629526.3645040)
- **Key idea:** Intel's RAPL (Running Average Power Limit) interface provides per-domain power measurements (CPU, package, DRAM) accessible via MSRs or Intel Power Gadget. The 2024 ICPE paper provides a reproducible analysis of Alder Lake's energy efficiency.
- **Relevance to EdgeLM:** We should log RAPL power data during benchmarks to understand Watts-per-token alongside tok/s. This strengthens the research paper.
- **Implementation complexity:** Low -- read RAPL MSRs or use Intel Power Gadget API.

---

### 13. Windows 11 Scheduler Integration

#### 13.1 Thread Director + Windows 11 Synergy

- **Source:** [Intel - What Is Thread Director](https://www.intel.com/content/www/us/en/support/articles/000097053/processors/intel-core-processors.html), [Digital Trends - Thread Director and Windows 11](https://www.digitaltrends.com/computing/how-intel-thread-director-marries-alder-lake-windows-11/)
- **Key idea:** Windows 11's scheduler integrates deeply with Thread Director via the EHFI (Enhanced Hardware Feedback Interface). The CPU classifies each thread's workload type and provides hints to the OS about optimal core placement.
- **Relevance to EdgeLM:** Even though we override with SetThreadAffinityMask, understanding that Thread Director classifies "vector/SIMD-heavy" workloads for P-cores means our engine should "look like" a SIMD workload to the OS (in case any threads escape our affinity settings).
- **Details:**
  - Thread Director monitors per-thread instruction mix in real-time
  - Classifies threads into categories: scalar, vector, mixed
  - Windows 11 23H2+ improved scheduling decisions
  - 24H2 reported reduced stuttering on Alder Lake notebooks
  - Windows 11 outperforms Linux with Alder Lake specifically because of Thread Director integration

#### 13.2 GetLogicalProcessorInformationEx for Core Discovery

- **Source:** [Intel - Hybrid Architecture](https://www.intel.com/content/www/us/en/developer/articles/technical/hybrid-architecture.html)
- **Key idea:** Use `GetLogicalProcessorInformationEx()` with `RelationProcessorCore` to enumerate cores and their `EfficiencyClass`. On Alder Lake, P-cores have `EfficiencyClass = 1`, E-cores have `EfficiencyClass = 0`.
- **Relevance to EdgeLM:** Our engine should auto-detect P-cores vs E-cores at startup rather than hardcoding core IDs. This makes the code work on different Alder Lake SKUs (6P+8E, 4P+8E, 2P+8E variants).
- **Implementation complexity:** Low -- single API call at startup.

---

### 14. Energy Efficiency -- When E-Cores Beat P-Cores

#### 14.1 ICPE 2024 Energy Analysis

- **Source:** [SPEC ICPE 2024](https://dl.acm.org/doi/10.1145/3629526.3645040)
- **Key idea:** For light/scalar workloads, E-cores are 2-3x more energy efficient than P-cores. But for heavy SIMD workloads, P-cores achieve higher performance-per-watt because they complete the work faster and return to idle sooner.
- **Relevance to EdgeLM:** Running the draft model (speculative decoding) on E-cores is not just architecturally correct -- it's also energy-optimal. The main model on P-cores is also energy-optimal because the alternative (spreading across E-cores) would take longer and consume more total energy.
- **Details:**
  - C-state transition latencies: C1 (~1us), C6 (~100us)
  - Frequency transition latency: ~10-100us depending on magnitude
  - RAPL granularity: ~1ms sampling period
  - E-cores consume ~1/3 the power of P-cores at similar scalar workload levels

---

## Techniques Comparison Matrix

| Technique | Source | Impact | Complexity | In Deep Dive? |
|-----------|--------|--------|------------|---------------|
| P-core only SIMD (pin with affinity) | llama.cpp #572 | 2.4-3x speedup | Low | Yes (mentioned) |
| Per-core quality benchmarking | SillyCross blog | Up to 15% variance | Medium | No |
| Avoid INC/DEC in loop counters (fusion) | Corsix fusion notes | 5-10% inner loops | Low | No |
| VPDPBUSD+VPSHUFB parallel execution | Port analysis | Enables hybrid kernel | Medium | Partially |
| E-core AVX2 = split 128-bit (confirmed) | HWCooling analysis | Architecture-critical | Low | Mentioned vaguely |
| IRQ redirection off inference cores | Eunomia blog | Eliminates tail latency | Low | No |
| VirtualLock() model weights | Eunomia blog | Prevents page-out | Low | No |
| EPP=0 via power plan | Intel HWP docs | 5-15% for burst | Low | No |
| RAPL power monitoring | ICPE 2024 | Paper data | Low | No |
| GetLogicalProcessorInformationEx | Intel hybrid docs | Auto-detect cores | Low | No |
| Core-to-core latency awareness | nviennot tool | Pipeline design | Medium | No |
| Pre-touch memory at startup | Eunomia blog | Eliminates cold faults | Low | No |
| Process priority HIGH_PRIORITY_CLASS | OS optimization | Reduces preemption | Low | No |
| No AVX2 frequency throttle for ternary | Extensa/Cloudflare | 0% penalty (positive) | None | No |
| Custom PL1/PL2 via XTU | Community findings | Stable sustained freq | Low | No |

---

## Recommendations for EdgeLM

Ranked by impact-to-effort ratio:

1. **Pin SIMD matmul threads to P-cores with hard affinity** (2.4-3x, Low effort)
   Use `SetThreadAffinityMask()` with P-core bitmask derived from `GetLogicalProcessorInformationEx()`. This is the single highest-impact optimization for hybrid CPUs.

2. **Confirm ternary kernels don't trigger AVX2 throttling** (0% penalty, Zero effort)
   Our integer-only kernel (VPADDB, VPSHUFB, VPDPBUSD) avoids heavy FP instructions. Verify with Intel Power Gadget that frequency stays at max turbo during sustained inference.

3. **Set Windows power plan to High Performance / EPP=0** (5-15%, Low effort)
   Ensures immediate frequency ramp-up. Use `powercfg /setactive SCHEME_MIN` or "Ultimate Performance" plan.

4. **VirtualLock() model weights + pre-touch at startup** (Eliminates cold faults, Low effort)
   Prevents OS from ever paging out model data. Pre-touch by reading through all weight pages during initialization.

5. **Set process to HIGH_PRIORITY_CLASS** (Reduces jitter, Low effort)
   Minimizes scheduling preemption during inference. Combined with thread pinning, this virtually eliminates OS-induced latency spikes.

6. **Redirect IRQs off P-cores used for inference** (Eliminates interrupt jitter, Low effort)
   Network card and USB interrupts can cause microsecond-level stalls on inference cores. Redirect via Device Manager or registry.

7. **Design P-E data handoff for speculative decoding around 145ns cross-type latency** (Architecture, Medium effort)
   Use lock-free ring buffer in L3-resident memory for draft model → main model communication. Avoid spin-locks across core types.

8. **Benchmark individual P-cores at startup and rank them** (Up to 15% on best core, Medium effort)
   Run a quick AVX2 microbenchmark per P-core (10ms each). Pin the most critical threads to the fastest cores.

9. **Use CMP/SUB+JNZ loops, never INC/DEC+JNZ** (5-10% inner loops, Low effort)
   Ensure macro-op fusion in inner loops. Transform loop counters to count down toward zero using SUB.

10. **Interleave VPDPBUSD (Port 0) with VPSHUFB (Port 5) for parallel execution** (Potential 1.5x kernel, High effort)
    These instructions use different ports and can execute simultaneously. Explore a hybrid kernel that uses both paths.

11. **Log RAPL power data during benchmarks** (Paper quality, Low effort)
    Report Watts-per-token alongside tok/s. Strengthens the research paper with energy efficiency analysis.

12. **Account for PL1 thermal throttling in sustained benchmarks** (Honest benchmarks, Low effort)
    Warm up for 60+ seconds before recording to ensure PL1 steady state. Report both burst (PL2) and sustained (PL1) numbers.

---

## References

1. [WikiChip - Golden Cove Microarchitecture](https://en.wikichip.org/wiki/intel/microarchitectures/golden_cove)
2. [WikiChip Fuse - Intel Details Golden Cove](https://fuse.wikichip.org/news/6111/intel-details-golden-cove-next-generation-big-core-for-client-and-server-socs/)
3. [Wikipedia - Golden Cove](https://en.wikipedia.org/wiki/Golden_Cove)
4. [HWCooling - Gracemont Microarch Analysis](https://www.hwcooling.net/en/gracemont-the-not-so-little-alder-lake-core-microarch-analysis/3/)
5. [Tom's Hardware - Intel Architecture Day 2021](https://www.tomshardware.com/features/intel-architecture-day-2021-intel-unveils-alder-lake-golden-cove-and-gracemont-cores)
6. [Hardware Times - Intel P-Core Architectures Compared](https://hardwaretimes.com/skylake-vs-sunny-cove-vs-golden-cove-vs-redwood-cove-vs-lion-cove-intel-p-core-architectures/)
7. [nviennot/core-to-core-latency - GitHub](https://github.com/nviennot/core-to-core-latency)
8. [SillyCross - Bizarre Performance Characteristics of Alder Lake](https://sillycross.github.io/2022/06/11/2022-06-11/)
9. [llama.cpp Discussion #572 - P-core performance](https://github.com/ggml-org/llama.cpp/discussions/572)
10. [Extensa Tech - AVX Throttling Part 1](https://extensa.tech/blog/avx-throttling-part1/)
11. [Cloudflare - On the Dangers of Intel's Frequency Scaling](https://blog.cloudflare.com/on-the-dangers-of-intels-frequency-scaling/)
12. [Travis Downs - AVX Frequency Transitions](https://travisdowns.github.io/blog/2020/01/17/avxfreq1.html)
13. [Corsix - x86 Macro-Op Fusion Notes](https://www.corsix.org/content/x86-macro-op-fusion-notes)
14. [EasyPerf - MacroFusion in Intel CPUs](https://easyperf.net/blog/2018/02/23/MacroFusion-in-Intel-CPUs)
15. [BitSum - CoreDirector](https://bitsum.com/apps/coredirector/)
16. [Tom's Hardware - CoreDirector Software](https://www.tomshardware.com/news/bitsun-coredirector-intel-cpu-e-cores)
17. [Eunomia - OS-Level Challenges in LLM Inference](https://eunomia.dev/blog/2025/02/18/os-level-challenges-in-llm-inference-and-optimizations/)
18. [Intel - Managing Performance with Heterogeneous Cores](https://www.intel.com/content/www/us/en/docs/onemkl/developer-guide-windows/2023-0/managing-performance-with-heterogeneous-cores.html)
19. [Intel - What Is Thread Director](https://www.intel.com/content/www/us/en/support/articles/000097053/processors/intel-core-processors.html)
20. [Intel - Hybrid Architecture](https://www.intel.com/content/www/us/en/developer/articles/technical/hybrid-architecture.html)
21. [Linux Kernel - intel_pstate Documentation](https://docs.kernel.org/admin-guide/pm/intel_pstate.html)
22. [Wiley - PMCSched for Alder Lake (2023)](https://onlinelibrary.wiley.com/doi/full/10.1002/cpe.7814)
23. [ACM/SPEC ICPE 2024 - Alder Lake Energy Efficiency](https://dl.acm.org/doi/10.1145/3629526.3645040)
24. [arXiv - Challenging GPU Dominance (2025)](https://arxiv.org/abs/2505.06461)
25. [Agner Fog - Instruction Tables](https://www.agner.org/optimize/instruction_tables.pdf)
26. [Agner Fog - Microarchitecture Manual](https://www.agner.org/optimize/microarchitecture.pdf)
27. [uops.info - Instruction Characterization](https://uops.info/)
28. [Intel - Golden Cove Instruction Throughput and Latency](https://www.intel.com/content/www/us/en/content-details/723498/intel-processors-and-processor-cores-based-on-golden-cove-microarchitecture-instruction-throughput-and-latency.html)
29. [Intel - Optimization Reference Manual](https://cdrdv2-public.intel.com/821613/355308-Optimization-Reference-Manual-050-Changes-Doc.pdf)
30. [ElevenForum - Forcing Performance Mode](https://www.elevenforum.com/t/forcing-performance-mode-for-background-app-on-intel-p-e-core-cpu.15763/)
31. [LWN.net - Hybrid Scheduling Gets More Complicated](https://lwn.net/Articles/909611/)
32. [ACM SIGOPS - Thread Director Evaluation on Alder Lake](https://dl.acm.org/doi/10.1145/3546591.3547532)
