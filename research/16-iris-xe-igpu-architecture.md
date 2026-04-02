# Section 16: Intel Iris Xe iGPU Architecture -- Extended Research

## Overview

This section covers the integrated Intel Iris Xe graphics engine on the target
`i7-12700H` platform and asks a narrower question than a generic GPU survey:
what does the Xe-LP iGPU actually look like as a compute target for EdgeLM's
planned attention offload path?

That question matters because the EdgeLM plan does **not** target a discrete GPU
with dedicated VRAM, tensor cores, or HBM. It targets an integrated 96-EU
Xe-LP graphics block that shares DDR4 bandwidth with the CPU and lives inside a
system whose primary bottleneck is already memory traffic. If we misunderstand
this GPU as "a small Arc" or "free extra compute," we will build the wrong
pipeline. If we treat it as a many-threaded vector coprocessor with modest
on-chip memory, no dedicated VRAM, and a very specific software/runtime model,
we can make much better decisions about which parts of inference belong there.

For EdgeLM, the architecture question is therefore not "can Iris Xe run AI?"
Intel's own material shows that it can. The real question is:

- what the Xe-LP iGPU is structurally good at,
- what it is structurally bad at,
- how much of its behavior is fixed by hardware vs device-dependent at runtime,
- and what that implies for Phase 4's CPU+iGPU attention pipeline on a
  dual-channel DDR4 laptop.

## What the Deep Dive Already Covers

`deep-dive.md` is currently empty, so the usable project context comes from
`implementation-plan.md`, `AGENTS.md`, and earlier research sections.

- The target machine is an `i7-12700H` laptop with `16GB DDR4-3200` dual-channel
  memory and an integrated `Intel Iris Xe` iGPU.
- The project already treats memory bandwidth as the dominant limit and
  explicitly warns that CPU and iGPU should not overlap memory-intensive work on
  this platform.
- Phase 4 plans to offload attention to the iGPU:
  - `Q * K^T`
  - softmax
  - `softmax * V`
  - double-buffered CPU<->iGPU transfers
  - bandwidth contention profiling
  - time-slicing if overlap is too expensive
- Prior sections already established two important constraints that interact with
  this GPU research:
  1. DDR4 bandwidth is the global system bottleneck.
  2. The CPU hot path should remain centered on P-cores and tightly controlled
     scheduling.

What the project did **not** yet have was a concrete hardware model for the iGPU
itself:

- whether this is actually the full 96-EU Xe-LP configuration,
- what the execution unit hierarchy looks like,
- whether XMX-like matrix acceleration exists here,
- how much on-chip local memory is available,
- how subgroup execution works,
- how queue/context choices affect movement and concurrency,
- and how unified memory changes the economics of offload.

`research-papers-data.json` contains a few high-level references related to
Intel integrated GPUs and deployment stacks, but it does not contain the level
of architecture detail needed for this section. Official Intel product and
oneAPI documentation were therefore the primary sources.

## New Findings

### 1. The Target GPU Is the Full 96-EU Xe-LP Configuration, and Dual-Channel Memory Is Part of the Contract

#### 1.1 The i7-12700H exposes a 96-EU Iris Xe-capable integrated GPU at up to 1.40 GHz

- **Source:** Intel ARK page for `Intel Core i7-12700H`
- **Key idea:** Intel lists the `i7-12700H` as `Intel Iris Xe Graphics eligible`
  with `96` execution units, `1.40 GHz` graphics max dynamic frequency,
  `OpenCL 3.0`, `DirectX 12.1`, and `OpenGL 4.6`.
- **Relevance to EdgeLM:** This confirms the actual compute target for Phase 4.
  The offload path is not targeting an abstract "Intel iGPU"; it is targeting a
  specific `96 EU` Xe-LP implementation with an official OpenCL stack.
- **Estimated impact:** High for architecture correctness.
- **Implementation complexity:** Low.
- **Details:** This matters because a lot of online discussion about Iris Xe
  mixes together 80-EU, 96-EU, Tiger Lake, Alder Lake, and even Arc behavior.
  For this project, we can anchor on the exact SKU:
  - `GPU name`: Intel Iris Xe Graphics eligible
  - `Execution Units`: 96
  - `Graphics Max Dynamic Frequency`: 1.40 GHz
  - `OpenCL support`: 3.0
  - `DirectX support`: 12.1
  That is enough to treat the device as a legitimate compute target rather than
  a graphics-only afterthought.

#### 1.2 Dual-channel memory is not optional for getting the real Iris Xe configuration

- **Source:** Intel ARK footnote for Iris Xe branding requirements
- **Key idea:** Intel explicitly states that the `Intel Iris Xe` brand requires
  `128-bit (dual channel) memory`; otherwise the system falls back to the `Intel
  UHD` brand.
- **Relevance to EdgeLM:** The project hardware already specifies dual-channel
  DDR4-3200, so the architecture assumptions in this section are consistent with
  the intended machine. If the machine were accidentally running single-channel
  memory, both branding and performance assumptions would break.
- **Estimated impact:** Very High.
- **Implementation complexity:** Low.
- **Details:** This is not just a marketing footnote. On this project, the
  memory subsystem is the bottleneck. A single-channel mistake would simultaneously:
  - reduce CPU-side bandwidth,
  - hurt iGPU effective bandwidth,
  - degrade shared-memory behavior,
  - and undermine the entire CPU+iGPU plan.
  So before any Phase 4 bring-up, EdgeLM should explicitly log the observed
  memory-channel configuration as part of startup diagnostics and benchmark
  metadata.

### 2. Xe-LP Is a Many-Threaded Vector Machine, Not a Small Tensor-Core GPU

#### 2.1 The basic execution unit is a 7-thread SIMD engine with a narrow vector datapath

- **Source:** Intel oneAPI GPU Optimization Guide, `Intel Iris Xe GPU Architecture`
- **Key idea:** Intel describes the Xe-LP `Execution Unit (EU)` as the smallest
  thread-level building block. Each EU supports `7` hardware threads and uses
  an `8-wide SIMD` ALU for FP/INT plus a `2-wide SIMD` ALU for extended math.
  Each thread has `128` general-purpose registers of `32B` each.
- **Relevance to EdgeLM:** This tells us what kind of kernels the iGPU wants:
  structured, vector-friendly, lane-regular kernels. It is not architected like
  an NVIDIA SM or an Intel Arc XMX block.
- **Estimated impact:** High.
- **Implementation complexity:** Medium.
- **Details:** The most useful mental model for EdgeLM is:
  - Xe-LP likes throughput kernels with regular control flow.
  - It exposes high thread-level concurrency to hide latency.
  - It relies on vector/SIMD packing inside the execution unit.
  - Register pressure matters because spills go to slower memory.
  In practice, this means:
  - avoid branch-heavy attention masking logic inside hot kernels,
  - avoid kernels with too many temporaries,
  - prefer regular reductions and tiled compute,
  - and assume occupancy and register pressure will both matter.

#### 2.2 At full size, the architecture is 6 dual subslices x 16 EUs = 96 EUs and 672 hardware threads

- **Source:** Intel oneAPI GPU Optimization Guide architecture summary tables;
  Intel Xe-LP API Optimization Guide architecture highlights
- **Key idea:** Intel's Xe-LP hierarchy for the high-end integrated variant is:
  `6` dual subslices, `16` EUs per dual subslice, `7` threads per EU, for
  `96` total EUs and `672` total hardware threads. Intel also advertises up to
  `2.2 TFLOPS` of compute capability for Xe-LP.
- **Relevance to EdgeLM:** This is enough parallelism to matter, but it is still
  a relatively small integrated GPU. The right offload targets are medium-grain
  kernels like attention blocks, not wholesale "move the model to GPU" designs.
- **Estimated impact:** High.
- **Implementation complexity:** Low.
- **Details:** Intel's architecture table lists `1536` single-precision
  FLOPs/clock (MAD counted as two operations) for the full Xe-LP integrated
  configuration. Using the `1.40 GHz` graphics max dynamic frequency from the
  `i7-12700H` product page gives:

  `1536 FLOPs/clk * 1.40e9 clk/s ~= 2.15e12 FLOPs/s`

  or about `2.15 TFLOPS` peak FP32, which aligns with Intel's broader
  "`up to 2.2 TFLOPS`" claim for Xe-LP.

  This is enough raw arithmetic to accelerate the right part of the pipeline,
  but only if memory traffic is controlled. On this machine, arithmetic peak is
  not the gating constraint by itself.

### 3. Xe-LP Has Useful INT8/FP16 Vector Support, but It Does Not Have XMX or Arc-Style Matrix Engines

#### 3.1 Xe-LP exposes vector low-precision arithmetic, including INT8 DP4A-like throughput

- **Source:** Intel oneAPI Xe-LP EU throughput table
- **Key idea:** Intel documents FP16, INT16, and INT8 support on Xe-LP EUs, with
  INT8 throughput expressed via `32 (DP4A)` operations per clock per EU.
- **Relevance to EdgeLM:** This is relevant for quantized attention-side kernels,
  KV-cache transforms, and low-precision staging logic. There is meaningful
  low-precision vector arithmetic here.
- **Estimated impact:** Medium.
- **Implementation complexity:** Medium.
- **Details:** This makes the iGPU more promising for:
  - INT8 or mixed-precision KV-cache movement/transforms,
  - low-precision attention score preparation,
  - pre/post-processing around attention,
  - and potentially FP16 softmax/value paths where numerical behavior is safe.
  It does **not** automatically imply that all quantized LLM kernels should move
  to the iGPU. The shared-memory system still dominates the economics.

#### 3.2 The matrix-engine story begins with newer Xe families, not Xe-LP

- **Source:** Intel oneAPI GPU Optimization Guide sections contrasting Xe-LP with
  Xe-HPG / Xe-HPC
- **Key idea:** Intel explicitly states that newer Xe families use `Xe-core`
  structures with vector and matrix engines, while Xe-LP and earlier generations
  use the older EU-centric compute structure.
- **Relevance to EdgeLM:** This is the clearest architectural warning against
  designing for "hidden tensor cores" that do not exist on this target.
- **Estimated impact:** High.
- **Implementation complexity:** Low.
- **Details:** For EdgeLM, this means:
  - do not assume Arc-style XMX acceleration,
  - do not design kernels around matrix-engine tile shapes,
  - do not treat oneAPI or OpenCL on this device as if it were a datacenter AI
    accelerator.
  The correct target is a vector machine with local memory and lots of threads.
  Any useful speedup must come from:
  - vector-friendly kernels,
  - good subgroup design,
  - careful tiling,
  - minimized synchronization,
  - and disciplined memory movement.

### 4. The On-Chip Memory Hierarchy Is Real and Useful, but It Is Small Enough To Punish Bad Tiling

#### 4.1 Each dual subslice has local on-chip storage and fixed internal bandwidth characteristics

- **Source:** Intel oneAPI Xe-LP Dual Subslice and Slice descriptions
- **Key idea:** Intel describes each Xe-LP Dual Subslice (DSS) as containing
  `16 EUs`, a local thread dispatcher, instruction cache, `Shared Local Memory
  (SLM)`, and a `128B/cycle` data port. A Xe-LP slice provides up to `16MB L3`
  and `128B/cycle` bandwidth to both L3 and memory.
- **Relevance to EdgeLM:** This is the hardware reason tiled attention kernels
  are even worth considering. There is meaningful on-chip reuse available if the
  tiles are chosen well.
- **Estimated impact:** High.
- **Implementation complexity:** Medium.
- **Details:** The major practical implication is that the iGPU can benefit from:
  - loading query/key/value tiles once,
  - performing multiple operations while the tile is resident,
  - and reducing repeated global-memory traffic.
  This is especially important for attention, where naive kernels can thrash
  memory badly.

#### 4.2 Intel's documentation also makes clear that local-memory limits are device-dependent and must be queried

- **Source:** Intel oneAPI `Shared Local Memory` and `Considerations for Selecting Work-group Size`
- **Key idea:** Intel repeatedly describes local/shared memory size and maximum
  work-group size as device-dependent and recommends querying them at runtime via
  device properties.
- **Relevance to EdgeLM:** Do not hard-code SLM capacity assumptions from guide
  examples into production kernel launch logic.
- **Estimated impact:** High for correctness and portability across laptop SKUs.
- **Implementation complexity:** Low.
- **Details:** This section is one place where Intel's docs are easy to misuse.
  Architecture pages give a structural model of Xe-LP. Optimization pages then
  emphasize that actual `local_mem_size`, `sub_group_sizes`, and
  `max_work_group_size` should still be queried from the runtime.

  For EdgeLM, that should become a hard rule:

  - query device limits at startup,
  - log them,
  - feed them into kernel-launch heuristics,
  - and store them with benchmark output.

  In other words, the architecture guide gives the mental model; the runtime
  query gives the value you should actually trust for execution policy.

### 5. Sub-Groups Are the Real Programming Unit for Hot Compute Kernels

#### 5.1 Intel graphics execute subgroup work inside multithreaded vector engines, and subgroup size is central to performance

- **Source:** Intel oneAPI `Sub-groups and SIMD Vectorization`
- **Key idea:** Intel explains that work-items are packed into sub-groups that
  execute in the same vector-engine thread. The compiler may choose subgroup
  widths such as `8`, `16`, or `32`, and divergence inside a subgroup hurts
  performance.
- **Relevance to EdgeLM:** This is the core design rule for offloaded attention:
  write subgroup-friendly kernels, not scalar-per-thread kernels with irregular
  control flow.
- **Estimated impact:** Very High.
- **Implementation complexity:** Medium.
- **Details:** For attention kernels, this means:
  - treat subgroup width as a first-class tuning parameter,
  - try to keep lane behavior uniform,
  - avoid branch-heavy code paths inside the score and softmax loops,
  - and organize memory so adjacent lanes cooperate on contiguous data.

  The easiest way to lose performance on Xe-LP is to write "generic GPU code"
  that ignores subgroup structure and lets the compiler create wasteful lane
  behavior.

#### 5.2 A 16-lane subgroup is a strong default for memory-centric collectives and transposes

- **Source:** Intel oneAPI `Sub-groups and SIMD Vectorization`
- **Key idea:** Intel's examples show explicit subgroup control, note that block
  load/store instructions are optimized for subgroup use, and caution that block
  load/store does not work with subgroup size `32` on current Intel hardware.
- **Relevance to EdgeLM:** For the first generation of Xe-LP attention kernels,
  subgroup-16 is the safest default starting point.
- **Estimated impact:** High.
- **Implementation complexity:** Low to Medium.
- **Details:** A good initial heuristic for EdgeLM is:
  - start with subgroup size `16`,
  - use subgroup block operations where contiguous loads/stores exist,
  - use subgroup shuffle for small intra-tile exchanges,
  - only test subgroup `8` or `32` after a baseline exists.

  This is especially attractive for:
  - softmax row reductions,
  - small tile transposes,
  - loading contiguous K/V slices,
  - and per-head reductions where the head dimension is divisible by 16.

### 6. Occupancy and Work-Group Size Must Be Tuned Per Kernel, Not Chosen Once for the Whole iGPU Path

#### 6.1 Intel explicitly recommends using the largest supported work-group size when barriers/shared memory matter

- **Source:** Intel oneAPI `Considerations for Selecting Work-group Size`
- **Key idea:** Intel notes that work-group size affects utilization of compute
  resources, vector lanes, and inter-work-item communication. When barriers and
  local/shared memory are involved, larger work-group sizes can materially change
  performance.
- **Relevance to EdgeLM:** Attention kernels with synchronization, tiled
  reductions, or shared-memory staging must tune work-group size explicitly.
- **Estimated impact:** High.
- **Implementation complexity:** Medium.
- **Details:** The right policy is not "always max everything." It is:
  - query `max_work_group_size`,
  - generate a short candidate set,
  - benchmark on the real target,
  - and select per-kernel launch parameters.

  This is one of the reasons the iGPU path should start with a small
  microbenchmark harness before full model integration.

#### 6.2 100 percent occupancy is not always optimal for memory-bound kernels

- **Source:** Intel oneAPI occupancy guidance
- **Key idea:** Intel explicitly notes that while full occupancy may be best for
  compute-bound kernels, memory-bound kernels can benefit from using more local
  memory even at the cost of lower occupancy.
- **Relevance to EdgeLM:** This is directly applicable to attention. Large parts
  of attention offload on an integrated GPU will still be memory sensitive.
- **Estimated impact:** High.
- **Implementation complexity:** Medium.
- **Details:** This corrects a common tuning mistake. For EdgeLM, a kernel that:
  - uses more SLM,
  - reduces global-memory traffic,
  - and launches with lower nominal occupancy

  may still outperform a "100% occupancy" version.

  Therefore, the autotuning objective should be:
  - tokens/sec or kernel time,
  - not occupancy in isolation.

### 7. SLM Bank Conflicts Are a First-Order Risk for Attention Tiles

#### 7.1 Intel documents 16-bank SLM behavior with serialization on same-bank conflicts

- **Source:** Intel oneAPI `Shared Local Memory`
- **Key idea:** Intel states that SLM is banked, and at the time of writing
  `64 consecutive bytes` map across `16 consecutive banks` at `4-byte`
  granularity. Accesses to different addresses in the same bank serialize.
- **Relevance to EdgeLM:** A bad SLM layout can destroy the value of tiling for
  attention score staging, softmax scratch buffers, or small transposes.
- **Estimated impact:** High.
- **Implementation complexity:** Medium.
- **Details:** Intel's own example shows a worst-case pattern dropping effective
  bandwidth to `1/16` of peak due to bank conflicts. That is catastrophic enough
  that SLM usage should never be treated as "automatically fast."

#### 7.2 Bank-conflict avoidance should be built into the first kernel design, not added later

- **Source:** Intel oneAPI `Shared Local Memory`
- **Key idea:** Intel's examples show that layout and access stride determine
  whether SLM reaches full bandwidth or collapses under conflicts.
- **Relevance to EdgeLM:** This affects:
  - score-tile staging,
  - softmax scratch buffers,
  - K/V tile transposes,
  - and any work-group-local reductions.
- **Estimated impact:** High.
- **Implementation complexity:** Medium.
- **Details:** A practical design rule for EdgeLM is:
  - use padding when storing tiled matrices in SLM,
  - avoid stride patterns that make neighboring lanes hammer the same bank,
  - validate with microbenchmarks early,
  - and assume bank conflicts are likely until proven otherwise.

### 8. Keep Whole Attention Subpipelines on the GPU Once You Pay the Launch and Residency Cost

#### 8.1 Intel explicitly recommends avoiding host/device round-trips for intermediate data

- **Source:** Intel oneAPI `Avoid moving data back and forth between host and device`
- **Key idea:** Intel shows that intermediate results should remain on the device
  rather than being brought back to the host for small follow-up operations.
  Intel explicitly uses machine learning layer-style pipelines as an example of
  why this matters.
- **Relevance to EdgeLM:** This strongly argues against tiny fragmented offload.
  If the iGPU is used at all, it should own a coarse attention stage, not just
  one or two isolated arithmetic operations.
- **Estimated impact:** Very High.
- **Implementation complexity:** Medium.
- **Details:** For EdgeLM, the bad design is:
  - CPU computes some partial attention state,
  - send to iGPU,
  - return to CPU for normalization,
  - send back to iGPU,
  - return again for post-processing.

  The better design is:
  - send the required tile/state once,
  - compute score/mask/softmax/value on device as a chain,
  - return the final attention output only when the stage is complete.

#### 8.2 Kernel fusion is especially valuable when arithmetic intensity is modest

- **Source:** Intel oneAPI host/device movement guidance
- **Key idea:** Intel explicitly points out that keeping follow-up operations on
  the accelerator or fusing them can beat host round-trips even when the
  follow-up kernel itself is not highly parallel.
- **Relevance to EdgeLM:** Softmax and masking are exactly the kinds of
  operations that often get separated conceptually but should stay inside the
  same device-side pipeline once offload begins.
- **Estimated impact:** High.
- **Implementation complexity:** Medium to High.
- **Details:** A good first-generation offload boundary is likely:
  - input: Q tile, K tile/window, V tile/window, metadata
  - on device: score, scale, mask, max reduction, exp/sum, normalize, value
    accumulation
  - output: attention output tile

  That boundary is large enough to justify launch overhead and small enough to
  keep DDR4 pressure under control if scheduled carefully.

### 9. Queue and Context Design Matter Even on a Single Integrated GPU

#### 9.1 Independent kernels can overlap when resources are underutilized

- **Source:** Intel oneAPI `Executing Multiple Kernels on the Device at the Same Time`
- **Key idea:** Intel shows that out-of-order queues can reduce total time by
  letting multiple kernels execute concurrently when they do not fully consume
  machine resources.
- **Relevance to EdgeLM:** This creates an opportunity for overlap inside the
  iGPU path itself, but only if kernels leave headroom.
- **Estimated impact:** Medium.
- **Implementation complexity:** Medium.
- **Details:** This is useful mainly for:
  - overlapping independent housekeeping kernels,
  - staging/lightweight transforms,
  - and possibly pipelining head or tile work
  when the main attention kernel is not already saturating the device.

  It is **not** a license to assume "more queues means more performance." On a
  bandwidth-limited integrated GPU, extra concurrency can just create more
  contention.

#### 9.2 If multiple queues are used, they should share a context

- **Source:** Intel oneAPI `Submitting Kernels to Multiple Queues`
- **Key idea:** Intel shows that kernels submitted to different queues with a
  shared context perform similarly to one-queue submission, while using separate
  contexts can trigger extra transfers and extra JIT cost.
- **Relevance to EdgeLM:** If the eventual OpenCL or Level Zero runtime design
  uses multiple queues, those queues should stay inside one shared device
  context.
- **Estimated impact:** High for avoiding self-inflicted overhead.
- **Implementation complexity:** Low.
- **Details:** This gives EdgeLM a straightforward policy:
  - one device context,
  - as few queues as possible initially,
  - shared-context multi-queue only after measurement,
  - never separate contexts for independent kernels in the same pipeline unless
    there is a very specific reason.

### 10. Shared System Memory Is the Defining Constraint of the Architecture for EdgeLM

#### 10.1 Intel graphics use system memory rather than dedicated VRAM

- **Source:** Intel Graphics Memory FAQ for Windows 10/11
- **Key idea:** Intel states that integrated processor graphics do not use a
  separate memory bank; they use system memory. Intel also notes that the
  reported `Shared System Memory` is a limit, not a permanent reservation, and
  that Iris Xe graphics are limited by OS policy to up to one-half of system
  memory.
- **Relevance to EdgeLM:** This is the architectural reason the iGPU cannot be
  treated like a free extra device with its own memory pool.
- **Estimated impact:** Very High.
- **Implementation complexity:** Low.
- **Details:** On the target `16GB` laptop, this means:
  - the GPU can dynamically consume a large fraction of system RAM if needed,
  - but doing so directly competes with CPU-side weights, KV cache, and working
    buffers,
  - and the reported shared-memory ceiling is not the same as "preallocated VRAM."

  This distinction matters for memory budgeting. EdgeLM should budget for
  pressure, not just static reservation.

#### 10.2 The CPU+iGPU overlap problem is fundamentally a shared-DRAM problem, not just a scheduling problem

- **Source:** Intel shared-memory documentation plus prior EdgeLM bandwidth
  analysis
- **Key idea:** Because the iGPU uses shared system memory, CPU-side tensor
  traffic and iGPU-side tensor traffic ultimately compete for the same DRAM
  subsystem.
- **Relevance to EdgeLM:** This directly supports the project's existing rule
  that CPU and iGPU should not overlap memory-intensive work blindly.
- **Estimated impact:** Very High.
- **Implementation complexity:** Medium.
- **Details:** This is partly an inference from the sources rather than a single
  line in one Intel document, but it is a very strong inference:

  - Intel says the iGPU uses system memory.
  - Intel says there is no dedicated memory bank.
  - The project already measured/estimated DDR4 as the system bottleneck.

  Therefore, naive overlap such as:
  - CPU ternary matmul streaming weights from DDR4
  - while iGPU attention streams K/V/state from the same DDR4

  is likely to degrade both sides.

  This is the single most important architectural caution in this section.

### 11. Intel's Own AI Benchmarks Show That Xe-LP Can Help, but Only with the Right Software Path

#### 11.1 Intel's Xe-LP AI article shows meaningful inference gains from a tuned software stack

- **Source:** Intel article `Effective Deployment of AI Workloads with Intel Xe Architecture`
- **Key idea:** Intel reports that on an `i7-1165G7` system, OpenVINO GPU
  inference on Xe-LP substantially outperformed both CPU inference and DirectML
  GPU inference for its tested ResNet101 workload.
- **Relevance to EdgeLM:** This is a practical reminder that software stack
  quality matters enormously on Intel iGPUs. Generic paths can leave a lot of
  performance on the table.
- **Estimated impact:** Medium as evidence; High as a strategic lesson.
- **Implementation complexity:** Low.
- **Details:** This does **not** prove that LLM attention offload on the
  `i7-12700H` will work well. The workload shape is different. But it does prove
  three useful things:
  - Xe-LP is not hopeless for inference.
  - Intel's own optimized stacks can materially beat generic ones.
  - Software specialization matters enough that "test the official fast path"
    should be part of bring-up.

#### 11.2 EdgeLM should treat the first iGPU milestone as a kernel bring-up and measurement phase, not a guaranteed throughput win

- **Source:** Intel official benchmarking evidence plus all architectural
  findings above
- **Key idea:** The hardware has enough compute and enough local structure to be
  useful, but the shared-memory architecture makes the success of offload highly
  sensitive to kernel shape and scheduling policy.
- **Relevance to EdgeLM:** The right Phase 4 mindset is "measure, then trust."
- **Estimated impact:** Very High.
- **Implementation complexity:** Low.
- **Details:** The correct goal of the first iGPU milestone is:
  - establish whether a carefully chosen attention kernel can beat CPU-only
    attention on this machine,
  - under a bandwidth-aware scheduling policy,
  - with residency, subgroup size, and launch parameters logged.

  That is a more realistic milestone than "offload attention and throughput goes
  up automatically."

## Techniques Comparison Matrix

| Technique | Source | Impact | Complexity | Already in Deep Dive? |
|-----------|--------|--------|------------|-----------------------|
| Runtime-query device limits (`local_mem_size`, `sub_group_sizes`, `max_work_group_size`) | Intel oneAPI docs | High | Low | No |
| Treat Xe-LP as a vector iGPU, not a tensor-core GPU | Intel architecture docs | Very High | Low | No |
| Start with subgroup-16 kernels for attention tiles | Intel subgroup guidance | High | Medium | No |
| Avoid subgroup divergence in mask/softmax code | Intel subgroup guidance | High | Medium | No |
| Use SLM tiling only with bank-conflict-aware layouts | Intel SLM docs | High | Medium | No |
| Fuse score/mask/softmax/value on device after offload begins | Intel host/device movement guidance | Very High | Medium-High | Partially |
| Reuse one shared device context across queues | Intel multi-queue guidance | High | Low | No |
| Use out-of-order queue overlap only when kernels underutilize the GPU | Intel queue concurrency guidance | Medium | Medium | No |
| Time-slice or carefully stage CPU+iGPU memory-intensive phases | Intel shared-memory docs + project bandwidth model | Very High | Medium | Partially |
| Benchmark the Xe path as a measured milestone rather than assuming a win | Intel Xe inference article + architecture constraints | High | Low | Partially |

## Recommendations for EdgeLM

1. Treat the `i7-12700H` iGPU as a `96-EU Xe-LP vector device with shared DDR4`,
   not as a mini discrete GPU.

2. Make Phase 4 start with a device-query and logging step:
   - `local_mem_size`
   - `sub_group_sizes`
   - `max_work_group_size`
   - driver/runtime version
   - memory-channel confirmation

3. Assume **no XMX / tensor-engine path** on this hardware and design the first
   kernels around vector EUs, subgroup collectives, and SLM.

4. Use subgroup size `16` as the initial default for attention microkernels, and
   only expand to subgroup `8` or `32` after profiling.

5. Make SLM use conditional on measured benefit. If SLM is used:
   - pad tiles,
   - benchmark for bank conflicts,
   - and treat SLM layout as part of kernel design, not a later cleanup.

6. Choose a **coarse offload boundary**. Offload complete attention stages or
   substantial attention tiles, not tiny helper operations that bounce data back
   and forth between CPU and GPU.

7. Keep one device context and the simplest possible queue model at first. Add
   multi-queue or out-of-order execution only after the single-path baseline is
   working and measured.

8. Do **not** overlap CPU ternary matmul and iGPU attention blindly. The default
   experimental policy should be:
   - first test time-sliced execution,
   - then test carefully staged overlap,
   - and keep whichever wins in wall-clock token throughput.

9. Add a dedicated iGPU microbenchmark suite before integrating with the full
   transformer:
   - tiled `Q * K^T`
   - masked softmax
   - `softmax * V`
   - fused attention tile
   - host/iGPU staging overhead

10. Record the following for every iGPU experiment:
    - subgroup size
    - work-group size
    - SLM use and tile size
    - queue model
    - transfer volume
    - GPU kernel time
    - end-to-end token latency
    - whether CPU matmul overlapped or was time-sliced

11. Make the first success criterion modest and precise:
    - beat CPU-only attention time on realistic sequence lengths
    - without increasing end-to-end decode latency due to DDR4 contention
    - and without blowing the `6-7 GB` inference memory budget.

## References

1. Intel, "Intel Core i7-12700H Processor Specifications."  
   https://www.intel.com/content/www/us/en/products/sku/132228/intel-core-i712700h-processor-24m-cache-up-to-4-70-ghz/specifications.html

2. Intel, "Intel Iris Xe GPU Architecture," oneAPI GPU Optimization Guide.  
   https://www.intel.com/content/www/us/en/docs/oneapi/optimization-guide-gpu/2023-0/intel-iris-xe-gpu-architecture.html

3. Intel, "Sub-groups and SIMD Vectorization," oneAPI GPU Optimization Guide.  
   https://www.intel.com/content/www/us/en/docs/oneapi/optimization-guide-gpu/2023-0/sub-groups-and-simd-vectorization.html

4. Intel, "Shared Local Memory," oneAPI GPU Optimization Guide.  
   https://www.intel.com/content/www/us/en/docs/oneapi/optimization-guide-gpu/2023-0/shared-local-memory.html

5. Intel, "Considerations for Selecting Work-group Size," oneAPI GPU Optimization Guide.  
   https://www.intel.com/content/www/us/en/docs/oneapi/optimization-guide-gpu/2023-0/considerations-for-selecting-work-group-size.html

6. Intel, "Avoid moving data back and forth between host and device," oneAPI GPU Optimization Guide.  
   https://www.intel.com/content/www/us/en/docs/oneapi/optimization-guide-gpu/2023-0/avoid-moving-data-back-and-forth-between-host-and.html

7. Intel, "Executing Multiple Kernels on the Device at the Same Time," oneAPI GPU Optimization Guide.  
   https://www.intel.com/content/www/us/en/docs/oneapi/optimization-guide-gpu/2023-0/executing-multiple-kernels-on-the-device-at-the.html

8. Intel, "Submitting Kernels to Multiple Queues," oneAPI GPU Optimization Guide.  
   https://www.intel.com/content/www/us/en/docs/oneapi/optimization-guide-gpu/2023-0/submitting-kernels-to-multiple-queues.html

9. Intel, "Intel Processor Graphics Xe-LP API Developer and Optimization Guide."  
   https://www.intel.com/content/www/us/en/developer/articles/guide/lp-api-developer-optimization-guide.html

10. Intel, "Frequently Asked Questions for Intel Graphics Memory on Windows 10 and Windows 11."  
    https://www.intel.com/content/www/us/en/support/articles/000020962/graphics.html

11. Intel, "Effective Deployment of AI Workloads with Intel Xe Architecture."  
    https://www.intel.com/content/www/us/en/developer/articles/technical/effective-deployment-ai-workloads-with-xe.html

## Audit Addendum (2026-04-02)

- **Shared allocations and LLC interaction deserve explicit measurement.** On an
  iGPU this integrated, the practical question is not only device flops but
  whether the chosen memory path pollutes CPU-visible caches badly enough to
  negate the offload.
- **Offload thresholds should be sequence-length aware.** Some kernels may only
  become worthwhile above specific prompt lengths or head dimensions.
- **Kernel autotuning should record occupancy and bandwidth together.** A launch
  that looks compute-efficient but saturates shared DDR is not a real win for
  EdgeLM.
