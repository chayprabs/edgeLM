# Section 15: P-core / E-core Workload Distribution -- Extended Research

## Overview

This section covers how EdgeLM should distribute work across the Intel i7-12700H's
6 Golden Cove P-cores and 8 Gracemont E-cores. This is not just a thread-count
question. Alder Lake is an asymmetric CPU: the P-cores have native 256-bit AVX2 and
AVX-VNNI throughput, while the E-cores split 256-bit vectors into 128-bit halves and
share L2 in clusters. A bad scheduling policy can erase most of the gains from the
custom ternary kernels; a good one can remove tail latency, reduce spin-wait time,
and keep the P-cores fed without letting helper work steal memory bandwidth.

For EdgeLM, the central question is not "how do we use all cores?" but "which work
must stay on P-cores, which work can safely move to E-cores, and which OS APIs give
us enough control without fighting Windows 11's scheduler?"

## What the Deep Dive Already Covers

`deep-dive.md` is currently empty, so the usable project context comes from
`implementation-plan.md` plus prior research sections.

- Use 14 physical threads, not 20 logical threads, for the main engine.
- Keep AVX2/AVX-VNNI ternary matmul on the 6 P-cores.
- Use E-cores for tokenization, sampling, I/O, prefetch orchestration, and the
  speculative draft model.
- Pin threads explicitly with `SetThreadAffinityMask()` and
  `SetThreadIdealProcessor()`.
- Expect roughly 15-25% improvement from eliminating migration and load imbalance.
- Avoid HyperThreading for SIMD-heavy workers.
- Prior sections already established two critical facts:
  1. Gracemont executes 256-bit AVX2 as split 128-bit uops, making it a poor fit
     for the hot ternary kernels.
  2. P-core to E-core communication is much slower than same-type communication, so
     cross-type handoff frequency must stay low.

`research-papers-data.json` does not contain a directly on-point P-core/E-core
scheduling entry beyond generic "hybrid scheduling" references, so it was not a
primary source for this section.

## New Findings

### 1. Runtime Topology Discovery Must Be Dynamic, Not Hard-Coded

#### 1.1 Windows exposes core capability directly through `EfficiencyClass`

- **Source:** Microsoft `PROCESSOR_RELATIONSHIP` docs, Microsoft CPU Sets docs, Intel 12th Gen game development guide
- **Key idea:** Windows already exposes heterogeneous core information. `GetLogicalProcessorInformationEx()` returns `PROCESSOR_RELATIONSHIP`, whose `EfficiencyClass` identifies relative core capability. CPU Sets APIs also expose per-CPU-set efficiency and allocation state.
- **Relevance to EdgeLM:** Do not hard-code "cores 0-5 are P-cores" or assume a fixed logical CPU map. BIOS updates, SMT exposure, core parking, and OEM firmware settings can change logical numbering.
- **Estimated impact:** High for correctness and portability; prevents invalid masks and brittle scheduling bugs.
- **Implementation complexity:** Low.
- **Details:** Build topology once at startup:
  - Enumerate physical cores with `GetLogicalProcessorInformationEx(RelationProcessorCore, ...)`
  - Record each core's logical processors and `EfficiencyClass`
  - Cross-check with `GetSystemCpuSetInformation()` so parked, reserved, or unavailable CPUs are not assigned
  - Build four masks:
    1. `p_core_primary_threads`
    2. `p_core_smt_siblings`
    3. `e_core_threads`
    4. `service_threads_allowed`

#### 1.2 CPU Sets are better than raw affinity masks for soft partitioning

- **Source:** Microsoft CPU Sets docs
- **Key idea:** Windows CPU Sets let a process or thread select a preferred subset of CPUs while still allowing the scheduler to manage placement within that subset. Thread-selected CPU sets override process default CPU sets; hard affinity masks override CPU sets entirely.
- **Relevance to EdgeLM:** This gives us a clean split:
  - Use hard affinity only for the 6 latency-critical P-core matmul workers
  - Use CPU Sets for E-core service pools, sampler threads, iGPU submission, and draft-model helpers
- **Estimated impact:** Medium; improves scheduler cooperation and reduces accidental thread drift without over-constraining every thread.
- **Implementation complexity:** Medium.
- **Details:** A practical policy is:
  - Set process-default CPU Sets to `e_core_threads + p_core_smt_siblings`
  - Explicitly hard-pin only the main SIMD workers to one primary logical thread per P-core
  - For non-hot threads, select E-core CPU Sets and let Windows choose the exact CPU inside that pool

### 2. Hard Affinity Is a Scalpel, Not a Default

#### 2.1 `SetThreadIdealProcessorEx()` is a hint; `SetThreadAffinityMask()` is a contract

- **Source:** Microsoft `SetThreadIdealProcessorEx` docs, Intel 12th Gen game development guide
- **Key idea:** `SetThreadIdealProcessorEx()` is only a placement preference. `SetThreadAffinityMask()` is a hard restriction. Intel's hybrid-core guidance warns against overusing hard affinities because they can fight the scheduler and reduce flexibility.
- **Relevance to EdgeLM:** The right answer is not "pin everything." Pin only threads whose performance is so sensitive that migration is unacceptable.
- **Estimated impact:** High for tail-latency stability; prevents over-constrained scheduling.
- **Implementation complexity:** Low.
- **Details:** Recommended split:
  - `SetThreadAffinityMask()` for the 6 matmul workers
  - CPU Sets or ideal-processor hints for E-core helpers
  - No hard affinity for short-lived utility threads unless measurement proves it is needed

#### 2.2 Intel's own guidance favors weak steering first, then hard pinning only when justified

- **Source:** Intel 12th Gen game development guide
- **Key idea:** Intel recommends giving the Windows 11 scheduler and Thread Director enough room to do the right thing, then applying stronger steering only where the workload is known and stable.
- **Relevance to EdgeLM:** This matches the engine architecture. The hot SIMD path is extremely stable and should be hard-pinned. Everything else should stay more flexible.
- **Estimated impact:** Medium.
- **Implementation complexity:** Low.
- **Details:** In practice, EdgeLM should use three scheduling classes:
  1. **Class A:** Hard-pinned P-core workers
  2. **Class B:** E-core-preferred service threads
  3. **Class C:** Unbound short-lived/control threads

### 3. Background Threads Should Be Deliberately "Cheap"

#### 3.1 EcoQoS is a good fit for helper threads that belong on E-cores anyway

- **Source:** Microsoft Quality of Service docs, Microsoft `SetThreadInformation` docs
- **Key idea:** Windows QoS lets threads request different scheduling and power behavior. EcoQoS is explicitly designed for background or efficiency-oriented work. The API is exposed via `SetThreadInformation()` and thread power throttling state.
- **Relevance to EdgeLM:** This is useful for threads we explicitly do not want stealing turbo headroom or scheduler priority from the P-core workers:
  - tokenizer
  - sampler
  - iGPU submission thread
  - low-urgency prefetch / staging helpers
- **Estimated impact:** Low to medium in average throughput, high in jitter reduction.
- **Implementation complexity:** Low.
- **Details:** EcoQoS is not for the decode hot path. It is for helper threads whose job is to stay out of the way while still making forward progress.

#### 3.2 Low-priority memory behavior matters for staging and streaming threads

- **Source:** Microsoft `SetThreadInformation` docs
- **Key idea:** Windows also exposes per-thread memory-priority hints. Lower-priority pages are reclaimed before higher-priority pages under pressure.
- **Relevance to EdgeLM:** If later phases add NVMe-backed KV spill or asynchronous staging buffers, helper threads should use lower memory priority so the model weights, KV cache, and hot activations remain favored.
- **Estimated impact:** Low today, potentially medium once long-context streaming is implemented.
- **Implementation complexity:** Low.
- **Details:** This is not a cache-control primitive; it is an OS memory-management hint. It belongs on future spill/staging workers, not on the main inference workers.

### 4. Intel's Official Heterogeneous-Core Guidance Strongly Supports P-Core-Only Decode

#### 4.1 Intel oneMKL explicitly recommends P-cores-only and one thread per core for predictable performance

- **Source:** Intel oneMKL developer guide: Managing Performance with Heterogeneous Cores
- **Key idea:** Intel's own guidance for hybrid CPUs recommends running on P-cores only and turning off HyperThreading when the goal is better load balancing and predictable throughput.
- **Relevance to EdgeLM:** This aligns almost perfectly with the decode path:
  - the hot kernels are latency-sensitive
  - work partitions are small
  - equal work splitting across P and E cores creates stragglers
  - SMT on the P-cores adds resource contention for vector code
- **Estimated impact:** High.
- **Implementation complexity:** Low.
- **Details:** Even though oneMKL is a library, the recommendation is architectural rather than library-specific. EdgeLM should treat "one primary hardware thread per P-core" as the default decode mode.

#### 4.2 If E-cores ever participate, the split must be weighted, not equal

- **Source:** Intel oneMKL heterogeneous-core guidance, prior Alder Lake microarchitecture findings
- **Key idea:** Hybrid execution only makes sense when work assignment reflects asymmetric throughput. Equal-size blocks are wrong on Alder Lake because a P-core and an E-core do not finish equal AVX2 work in equal time.
- **Relevance to EdgeLM:** If we later add an E-core-specific 128-bit path or let E-cores assist during large prefill workloads, task sizes must be weighted by measured throughput ratio, not thread count.
- **Estimated impact:** High if hybrid execution is attempted; otherwise zero.
- **Implementation complexity:** Medium.
- **Details:** A good initial model is:
  - decode: `P only`
  - prefill or non-hot scalar tasks: `weighted P + E`
  - never use equal row counts per worker across P and E cores

### 5. Recent LLM-Specific Research on Hybrid CPUs Agrees: Static Splits Waste Hardware

#### 5.1 Neural Speed's 2024 hybrid-CPU work argues for dynamic partitioning by core ratio and problem size

- **Source:** "Unlocking Hybrid CPU Potential for LLM Inference on NPU-less Platform", arXiv:2411.19542 / NeurIPS 2024 Efficient Natural Language and Speech Processing workshop
- **Key idea:** The paper reports that static workload distribution is unsatisfactory on hybrid CPUs and proposes dynamic parallel scheduling that adapts to both problem size and the performance ratio between core types. The reported goal is to push bandwidth utilization above 90 percent.
- **Relevance to EdgeLM:** This is the closest directly relevant recent paper for our use case. It supports a two-mode policy:
  - **Mode 1:** P-core-only decode for the latency-critical hot path
  - **Mode 2:** measured, weighted hybrid participation for larger or less latency-sensitive phases
- **Estimated impact:** Medium to high in phases where the workload is large enough to amortize coordination.
- **Implementation complexity:** High.
- **Details:** The important design lesson is not "always use both core types." It is "switch policies by phase and size." That is a better fit for EdgeLM than a single global thread-pool policy.

### 6. OS Heuristics Alone Are Not Enough for Specialized Workloads

#### 6.1 PMCSched shows performance-counter-guided scheduling can beat generic heuristics

- **Source:** "Flexible system software scheduling for asymmetric multicore systems with PMCSched", Concurrency and Computation: Practice and Experience, 2023, DOI: 10.1002/cpe.7814
- **Key idea:** PMC-guided schedulers can outperform generic heterogeneous scheduling because they react to actual workload behavior rather than a fixed heuristic model.
- **Relevance to EdgeLM:** We do not need a kernel module or ML scheduler to benefit from this insight. We can measure phase behavior ourselves and choose among a small number of hand-written policies.
- **Estimated impact:** Medium as a design principle; high if we later add adaptive policies.
- **Implementation complexity:** Medium.
- **Details:** A practical EdgeLM version is:
  - record per-phase wall time and per-core worker completion time
  - if E-core participation increases the P-core wait tail, disable it for that phase
  - if prefill scales with weighted hybrid participation, keep it only there

#### 6.2 Thread Director evaluation papers report that hardware guidance is helpful but not omniscient

- **Source:** "Evaluation of the Intel Thread Director technology on an Alder Lake processor", ACM APSys 2022
- **Key idea:** Thread Director improves default placement, but specialized applications can still benefit from software-aware policies because hardware classification cannot fully infer application intent.
- **Relevance to EdgeLM:** An LLM engine with custom AVX2 kernels, background streaming, and phase changes is exactly the kind of workload where intent matters.
- **Estimated impact:** Medium.
- **Implementation complexity:** Low.
- **Details:** Use Thread Director as the background behavior, not as the only policy layer.

### 7. Real LLM Inference Benchmarks on Alder Lake Show Why Naive All-Core Execution Fails

#### 7.1 llama.cpp users on i7-12700H reported 2.4x to 3.0x better performance with P-cores only

- **Source:** llama.cpp discussion #572
- **Key idea:** Community benchmarks on a 12700H reported that forcing inference onto P-cores only was dramatically faster than letting all 14 physical cores participate.
- **Relevance to EdgeLM:** This is the exact failure mode we want to avoid. It is not theoretical; it already shows up in similar CPU-bound LLM inference on this chip class.
- **Estimated impact:** Very high.
- **Implementation complexity:** Low.
- **Details:** Reported numbers on a 12700H:
  - 7B q4_0: `481.99 ms/run` on all cores vs `201.83 ms/run` on P-cores only
  - 65B q4_0: `4068.66 ms/run` on all cores vs `1357.09 ms/run` on P-cores only
  - Root cause reported by users: the P-cores finished first, then spin-waited on slower E-core partitions

#### 7.2 The failure is load imbalance first, not "E-cores are useless"

- **Source:** llama.cpp discussion #572, Intel heterogeneous-core guidance
- **Key idea:** The problem is not simply that E-cores exist. The problem is equal partitioning and barrier synchronization across unequal workers.
- **Relevance to EdgeLM:** This means E-cores still have value, but only for:
  - helper work with weak synchronization
  - separate pipelines
  - draft-model execution
  - hybrid execution with weighted chunking and coarse barriers
- **Estimated impact:** High.
- **Implementation complexity:** Medium.
- **Details:** This should directly shape queue design:
  - separate P-core and E-core pools
  - no per-layer barrier that waits on E-core completion unless the work was already weighted

### 8. Cross-Type Communication Must Stay Coarse-Grained

#### 8.1 P-to-E handoffs are expensive enough to justify separate queues

- **Source:** Prior Section 01 core-to-core latency findings, HybridDetect topology reference
- **Key idea:** Same-generation measurements show that P-core to E-core communication latency is much worse than same-type communication. On Alder Lake-style topologies, P<->E handoffs behave more like a cross-domain synchronization than a cheap thread wakeup.
- **Relevance to EdgeLM:** Fine-grained work stealing across core types is a bad match for the hot path.
- **Estimated impact:** High for tail latency and synchronization overhead.
- **Implementation complexity:** Medium.
- **Details:** Design implication:
  - one queue family for P-core compute
  - one queue family for E-core service/draft work
  - only exchange large, infrequent messages between the families

### 9. Shared DDR4 Bandwidth Changes What E-Cores Are Allowed To Do

#### 9.1 E-core helper work must be chosen by bandwidth cost, not just CPU cost

- **Source:** Intel hardware guidance, prior sections on DDR4 bandwidth and prefetching, Neural Speed hybrid-CPU paper
- **Key idea:** On this laptop, CPU cores and the iGPU share the same DDR4 pool. E-cores are not "free extra compute" if their helper work streams memory at the same time as the P-core matmul workers.
- **Relevance to EdgeLM:** E-core placement should obey a bandwidth budget:
  - good E-core jobs: tokenizer, sampler, control flow, queue management, draft model if time-sliced
  - risky E-core jobs: memcpy-heavy staging, aggressive software prefetch, memory-scanning helpers during decode
- **Estimated impact:** High, because memory bandwidth is the global bottleneck.
- **Implementation complexity:** Medium.
- **Details:** This is the most important correction to a naive "use the E-cores for prefetch" story. E-core work is only helpful if it does not steal the same DDR4 bandwidth the P-core kernels are trying to monetize.

### 10. Open-Source Hybrid Topology Tooling Gives a Good Implementation Template

#### 10.1 HybridDetect is a practical reference for Windows-side topology handling

- **Source:** Intel GameTechDev `HybridDetect` repository
- **Key idea:** HybridDetect shows how to discover P-cores, E-cores, SMT siblings, parking state, and related topology information in a lightweight Windows-oriented codebase.
- **Relevance to EdgeLM:** We do not need the project as a dependency, but it is a useful reference for:
  - topology enumeration
  - mask construction
  - validation during bring-up
- **Estimated impact:** Low for runtime speed, high for implementation correctness and debugging speed.
- **Implementation complexity:** Low.
- **Details:** This is especially useful during Phase 3 bring-up because it shortens the path to a correct topology map without tying the engine to an external runtime.

## Techniques Comparison Matrix

| Technique | Source | Impact | Complexity | Already in Deep Dive? |
|-----------|--------|--------|------------|-----------------------|
| Runtime core discovery via `EfficiencyClass` + CPU Sets | Microsoft docs | High | Low | No |
| Hard-pin only the 6 P-core decode workers | Intel oneMKL guidance, Intel game-dev guide, llama.cpp benchmarks | Very High | Low | Partially |
| Keep one thread per P-core and avoid SMT for SIMD hot path | Intel oneMKL guidance | High | Low | Partially |
| Use CPU Sets for E-core service pools instead of pinning everything | Microsoft CPU Sets docs | Medium | Medium | No |
| Use EcoQoS for helper threads | Microsoft QoS docs | Low-Medium | Low | No |
| Weighted hybrid chunking for mixed P+E phases | Neural Speed 2024 | Medium-High | High | No |
| Separate P-core and E-core queue families | Section 01 latency evidence + llama.cpp behavior | High | Medium | Partially |
| Treat memory-heavy E-core helpers as dangerous during decode | Bandwidth analysis + hybrid CPU evidence | High | Medium | No |
| Adaptive policy switching by phase (decode vs prefill vs draft) | Neural Speed 2024, PMCSched | High | Medium-High | No |
| Topology bring-up using HybridDetect as reference | HybridDetect repo | Low | Low | No |

## Recommendations for EdgeLM

1. Make **P-core-only decode** the baseline policy. Hard-pin exactly 6 matmul workers to one primary logical thread per P-core.

2. Do **not** put the hot ternary kernels on E-cores unless you later write a separate E-core-tuned path and can prove weighted chunking helps. With the current AVX2-first design, E-cores are a poor match.

3. Split scheduling policy into three classes:
   - Class A: hard-pinned P-core workers
   - Class B: E-core-preferred service threads using CPU Sets
   - Class C: flexible control threads with weak hints only

4. Use Windows topology APIs at startup and build masks dynamically. Never hard-code logical CPU IDs.

5. Give helper threads an explicit "stay cheap" policy:
   - E-core CPU Sets
   - EcoQoS where appropriate
   - lower memory priority for future spill/staging threads

6. Keep P/E communication coarse-grained. Use separate queues and exchange only big state transitions, not tiny per-block work items.

7. Treat bandwidth as a first-class scheduler input. E-core helper work that streams memory during decode should be disabled or time-sliced.

8. For future hybrid execution experiments, test only two controlled variants:
   - weighted P+E prefill on large prompts
   - E-core speculative draft model with coarse handoff to P-core verification

9. Add measurement hooks before implementing adaptive policies:
   - per-thread completion time
   - P-core barrier wait time
   - E-core service latency
   - end-to-end token latency and P99 jitter

10. In Phase 3, the first benchmark sweep should be:
    - `6P only`
    - `6P + E-core helpers only`
    - `weighted 6P+8E on large prefill`
    - `6P decode + E-core draft model`
    The engine should default to the fastest measured policy, not the most "fully utilized" one.

## References

1. Intel, "Managing Performance with Heterogeneous Cores," oneMKL Developer Guide for Windows.  
   https://www.intel.com/content/www/us/en/docs/onemkl/developer-guide-windows/2025-0/managing-performance-with-heterogeneous-cores.html

2. Intel, "Developing Games on 12th Generation Intel Core Processors."  
   https://www.intel.com/content/www/us/en/developer/articles/guide/12th-gen-intel-core-processor-gamedev-guide.html

3. Microsoft, "CPU Sets."  
   https://learn.microsoft.com/en-us/windows/win32/procthread/cpu-sets

4. Microsoft, "GetSystemCpuSetInformation function."  
   https://learn.microsoft.com/en-us/windows/win32/api/processtopologyapi/nf-processtopologyapi-getsystemcpusetinformation

5. Microsoft, "GetLogicalProcessorInformationEx function."  
   https://learn.microsoft.com/en-us/windows/win32/api/sysinfoapi/nf-sysinfoapi-getlogicalprocessorinformationex

6. Microsoft, "`PROCESSOR_RELATIONSHIP` structure."  
   https://learn.microsoft.com/en-us/windows/win32/api/winnt/ns-winnt-processor_relationship

7. Microsoft, "`SetThreadIdealProcessorEx` function."  
   https://learn.microsoft.com/en-us/windows/win32/api/processthreadsapi/nf-processthreadsapi-setthreadidealprocessorex

8. Microsoft, "`SetThreadInformation` function."  
   https://learn.microsoft.com/en-us/windows/win32/api/processthreadsapi/nf-processthreadsapi-setthreadinformation

9. Microsoft, "Quality of Service."  
   https://learn.microsoft.com/en-us/windows/win32/procthread/quality-of-service

10. Xin Liu et al., "Unlocking Hybrid CPU Potential for LLM Inference on NPU-less Platform," arXiv:2411.19542, 2024.  
    https://arxiv.org/abs/2411.19542

11. Edgar Mencias et al., "Flexible system software scheduling for asymmetric multicore systems with PMCSched," Concurrency and Computation: Practice and Experience, 2023.  
    https://doi.org/10.1002/cpe.7814

12. Xiaochen Liu et al., "Evaluation of the Intel Thread Director technology on an Alder Lake processor," ACM APSys 2022.

13. ggml-org, "llama.cpp discussion #572: 3x better performance with `numactl -C 0-5`?"  
    https://github.com/ggml-org/llama.cpp/discussions/572

14. Intel GameTechDev, "HybridDetect."  
    https://github.com/GameTechDev/HybridDetect

## Audit Addendum (2026-04-02)

- **Runtime calibration should pick the default schedule.** A very short startup
  benchmark can decide whether the machine today wants:
  - P-core-only decode,
  - helper E-cores,
  - or a more conservative all-CPU policy.
- **Thermal adaptation is the next missing layer.** A schedule that is optimal
  for the first `30` seconds may be wrong after several minutes of sustained
  decode on a mobile chassis.
- **Parking E-cores is a valid experimental baseline.** Because the shared DDR4
  bus is the project's dominant bottleneck, "use fewer cores" should remain a
  respectable measured option rather than a failure case.
