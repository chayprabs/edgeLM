# Section 17: OpenCL & Level Zero Programming -- Extended Research

## Overview

This section covers the two realistic low-level compute APIs for EdgeLM's
planned Iris Xe offload path: `OpenCL 3.0` and `oneAPI Level Zero`.

This is not just an API-style preference question. For EdgeLM, the runtime
interface determines:

- how kernels are compiled and cached,
- how command submission overhead shows up in token latency,
- how explicit memory movement and coherency are managed,
- how much control the engine has over queues, engines, and synchronization,
- how painful debugging and profiling will be during bring-up,
- and how hard it is to keep the host codebase in plain C without importing a
  heavyweight framework.

The wrong framing is "which API is newer?" The right framing is:

- which API gets the first attention kernel working fastest,
- which API exposes enough control to make an Intel-specific engine worth the
  complexity,
- and whether EdgeLM should choose one API forever or deliberately stage from
  one to the other as optimization needs grow.

Section 16 established that the `i7-12700H` iGPU is a `96-EU Xe-LP` device with
shared DDR4 memory, subgroup-oriented execution, useful SLM, and no Arc/XMX
matrix engines. That means Section 17 is really about the host-side contract
with that hardware: queue submission, module creation, memory allocation,
barriers, events, profiling, and launch overhead.

## What the Deep Dive Already Covers

`deep-dive.md` is still empty, so the project context comes from
`implementation-plan.md`, `AGENTS.md`, and Section 16.

- The implementation plan already assumes Phase 4 will use `OpenCL or Level
  Zero (oneAPI)` for iGPU dispatch.
- The current plan explicitly mentions pre-compiling attention kernels as
  OpenCL `.cl` files, double-buffering CPU<->iGPU work, and benchmarking
  CPU-only vs CPU+iGPU execution.
- The repo instructions make the engine intentionally Intel-specific and
  dependency-light:
  - custom C engine
  - no framework-heavy runtime in the hot path
  - hardware-specific optimization is encouraged, not avoided
- Section 16 established the architectural constraints that dominate API choice:
  1. the target is a `96-EU Xe-LP` iGPU, not a discrete GPU,
  2. the iGPU uses shared system memory,
  3. subgroup-friendly kernels matter,
  4. coarse offload boundaries matter,
  5. CPU+iGPU overlap is dangerous if both sides stream DDR4 heavily.

The current implementation plan already leans slightly toward an `OpenCL-first`
bring-up path:

- `Set up OpenCL or Level Zero for Iris Xe`
- `Pre-compile attention kernels as OpenCL .cl files`
- likely initial files such as `opencl_init.c`, `attention.cl`, and
  `softmax.cl`

That matters because the best recommendation here should align with the
existing staged plan unless the sources strongly argue otherwise.

One local observation from this workspace is also relevant: a simple PATH check
did **not** find `clinfo`, `zeinfo`, or `sycl-ls`. That does not block research,
but it does mean Phase 0 / Phase 4 bring-up would currently benefit from adding
at least one lightweight device-query tool.

`research-papers-data.json` does not contain strong primary material on the API
selection question itself. Official Khronos, Intel, and Level Zero
specification material was therefore the main source base.

## New Findings

### 1. Both OpenCL and Level Zero Are First-Class Supported on Alder Lake, and Intel Ships Them as Part of the Same Compute Runtime Stack

#### 1.1 Alder Lake is listed as supporting both OpenCL 3.0 and Level Zero 1.15

- **Source:** Intel `compute-runtime` repository support matrix
- **Key idea:** Intel's public graphics compute runtime lists `Alder Lake`
  support for both `OpenCL 3.0` and `Level Zero 1.15`.
- **Relevance to EdgeLM:** This removes the biggest strategic uncertainty. On
  the target class of hardware, choosing between OpenCL and Level Zero is not a
  question of "is one officially supported?" Both are.
- **Estimated impact:** Very High.
- **Implementation complexity:** Low.
- **Details:** This is important because it means EdgeLM does **not** need to
  bet on an obscure or deprecated path to use Level Zero, and it does **not**
  need to abandon OpenCL if the fastest path to a working prototype starts
  there. Intel's own runtime stack supports both on the relevant platform.

#### 1.2 Applications should link to the loader, not directly to the runtime implementation

- **Source:** Intel `compute-runtime` repository README
- **Key idea:** Intel explicitly states that applications should not link
  directly to the runtime library:
  - `Level Zero` applications should link with the `Level Zero loader`
  - `OpenCL` applications should link with the `ICD loader library`
- **Relevance to EdgeLM:** This affects host-side architecture from day one.
  The engine should treat the driver stack as dynamically loaded system
  infrastructure, not as a private static dependency.
- **Estimated impact:** High.
- **Implementation complexity:** Low.
- **Details:** For a plain-C custom engine, this is actually helpful. It means
  the runtime boundary can stay thin:
  - loader-based initialization,
  - API handles stored in an EdgeLM backend layer,
  - no direct dependency on a private internal Intel library ABI.

### 2. OpenCL Is the Simpler Bring-Up API, but OpenCL 3.0 Must Be Treated as a Query-Driven Capability Model

#### 2.1 OpenCL remains a low-level, portable host API centered on contexts, devices, queues, kernels, and events

- **Source:** Khronos OpenCL API Specification
- **Key idea:** OpenCL's architecture is still the classic heterogeneous
  model:
  - context
  - device
  - command-queue
  - program
  - kernel
  - memory objects
  - synchronization commands and events
- **Relevance to EdgeLM:** For the first usable iGPU path, this model is
  simpler to stand up than Level Zero's more explicit driver/device/queue-group
  split.
- **Estimated impact:** High for bring-up velocity.
- **Implementation complexity:** Low.
- **Details:** This is why OpenCL remains attractive for Phase 4A:
  - easier minimal host scaffolding,
  - mature examples and tooling,
  - straightforward C API,
  - direct fit for "compile kernel, set args, enqueue NDRange, wait, inspect
    output."

#### 2.2 OpenCL 3.0 is not a guarantee of uniform functionality beyond the 1.2 baseline

- **Source:** Khronos OpenCL 3.0 announcement and OpenCL C 3.0 specification
- **Key idea:** Khronos explicitly states that OpenCL 3.0 makes functionality
  beyond OpenCL 1.2 optional and queryable. OpenCL C 3.0 likewise describes
  optional language features that may or may not be present on a given device.
- **Relevance to EdgeLM:** This is the single biggest OpenCL gotcha for a
  hardware-specific engine. Seeing `OpenCL 3.0` in a driver string does **not**
  mean EdgeLM can blindly assume:
  - subgroup support,
  - SVM behavior,
  - IL ingestion details,
  - or every 2.x-era feature.
- **Estimated impact:** Very High.
- **Implementation complexity:** Low.
- **Details:** The correct policy is:
  - query every feature the engine intends to use,
  - log the results at startup,
  - branch the backend accordingly,
  - and make those feature results part of benchmark metadata.

  This is especially relevant on a laptop where driver versions can vary across
  OEM packages and update cadences.

### 3. OpenCL's Execution Model Is Fully Capable of Expressing EdgeLM's First Attention Pipeline

#### 3.1 A single in-order queue plus explicit events is enough for the first correct implementation

- **Source:** Khronos OpenCL API Specification
- **Key idea:** OpenCL command queues can execute in-order or out-of-order, and
  commands pass through an explicit queued/submitted/ready/running/ended model.
- **Relevance to EdgeLM:** The first attention pipeline does **not** need a
  sophisticated runtime graph engine. A simple in-order queue is enough to prove
  correctness and gather first performance data.
- **Estimated impact:** High.
- **Implementation complexity:** Low.
- **Details:** For Phase 4 bring-up, the simplest viable OpenCL sequence is:
  - enqueue upload or map/unmap as needed,
  - enqueue `Q*K^T`,
  - enqueue softmax,
  - enqueue `softmax*V`,
  - enqueue readback or final handoff,
  - wait at a coarse boundary.

  That is easy to reason about and minimizes "runtime cleverness" during the
  phase when kernel correctness matters most.

#### 3.2 Multiple queues exist, but they should be added only after a single-queue baseline is understood

- **Source:** Khronos OpenCL API Specification
- **Key idea:** OpenCL allows multiple command-queues inside one context, and
  event objects can synchronize work across queues.
- **Relevance to EdgeLM:** This means the API is not too limited for later
  pipelining experiments, but it also means there is no need to start with a
  multi-queue design before the single-queue path is working.
- **Estimated impact:** Medium.
- **Implementation complexity:** Medium.
- **Details:** On this hardware, more queues do not automatically mean more
  performance:
  - they can increase complexity,
  - they can make profiling noisier,
  - and on a shared-memory iGPU they can simply create more contention.

  So the right ordering is:
  1. one queue, correct kernels
  2. one queue, measured kernels
  3. only then try multi-queue experiments if traces show dispatch bubbles

### 4. OpenCL's Compilation and Binary Model Makes It Good for Prototyping, Validation, and Early C-Only Integration

#### 4.1 OpenCL supports source build, compile/link workflows, asynchronous compilation, and build logs

- **Source:** Khronos OpenCL API Specification
- **Key idea:** OpenCL exposes `clBuildProgram`, `clCompileProgram`, link-time
  flows, callbacks for asynchronous compilation, and build-log queries.
- **Relevance to EdgeLM:** This is extremely useful during early kernel
  development when errors will be frequent and iteration speed matters more than
  the last few microseconds of launch overhead.
- **Estimated impact:** High for bring-up speed.
- **Implementation complexity:** Low.
- **Details:** OpenCL's build model is a good match for the current plan to keep
  kernels as `.cl` files initially:
  - readable kernel source
  - easy build-log extraction
  - easy option experimentation
  - no custom compiler front-end integration required on day one

#### 4.2 OpenCL can also ingest intermediate language, which makes it more production-friendly than "source-only JIT" suggests

- **Source:** Khronos OpenCL API Specification
- **Key idea:** OpenCL exposes `clCreateProgramWithIL`, and OpenCL devices can
  report supported intermediate languages such as `SPIR-V`.
- **Relevance to EdgeLM:** This means OpenCL does not force the engine into
  runtime source compilation forever. There is a real path toward precompiled
  kernels.
- **Estimated impact:** High.
- **Implementation complexity:** Medium.
- **Details:** This is one of the most important practical findings in this
  section:
  - start with `.cl` source for debugging,
  - then move to offline-compiled `SPIR-V`,
  - keep source around for development builds,
  - and load IL in release-style experiments to reduce runtime build cost.

### 5. OpenCL Has a Command-Buffer Story, but It Is Extension-Based Rather Than the Baseline Portability Path

#### 5.1 `cl_khr_command_buffer` gives OpenCL a record/replay style submission model

- **Source:** Khronos `cl_khr_command_buffer` man pages
- **Key idea:** The OpenCL command-buffer extension allows commands to be
  recorded, finalized, and then enqueued as an executable command-buffer object.
- **Relevance to EdgeLM:** This is important because it partially closes one of
  the most obvious gaps between OpenCL and Level Zero: reusable submission
  packets.
- **Estimated impact:** Medium.
- **Implementation complexity:** Medium.
- **Details:** If supported on the target driver, command buffers could
  eventually help with:
  - repeating similar attention kernels across tokens,
  - reducing some host-side submission overhead,
  - and capturing a stable micro-pipeline once shapes are known.

#### 5.2 EdgeLM should not make command buffers part of the initial OpenCL dependency surface

- **Source:** Khronos command-buffer extension status and OpenCL 3.0 capability model
- **Key idea:** Command buffers are extension-driven, not the OpenCL 3.0
  portability baseline.
- **Relevance to EdgeLM:** This makes them a possible optimization path, not a
  sane day-one assumption.
- **Estimated impact:** Medium.
- **Implementation complexity:** Low.
- **Details:** The right way to think about this feature is:
  - nice if present,
  - useful for future replay-style work,
  - but not a foundation the first backend should require.

  In other words, if EdgeLM wants a reusable command-recording model from the
  beginning, Level Zero is the cleaner baseline API for that idea.

### 6. Level Zero Exposes the Hardware and Submission Model Much More Directly Than OpenCL

#### 6.1 Level Zero is explicitly designed as the low-level interface beneath oneAPI libraries

- **Source:** Intel oneAPI GPU Optimization Guide, Level Zero overview
- **Key idea:** Intel describes Level Zero as the low-level interface used by
  oneAPI libraries to exploit hardware capabilities on target devices.
- **Relevance to EdgeLM:** This aligns unusually well with the project's goals.
  EdgeLM is intentionally hardware-specific and does not want abstraction for
  its own sake.
- **Estimated impact:** Very High.
- **Implementation complexity:** Medium.
- **Details:** The API design reflects that intent:
  - explicit driver discovery,
  - explicit context creation,
  - explicit module creation,
  - explicit queue groups,
  - explicit command lists,
  - explicit synchronization primitives,
  - explicit memory types.

  That is more work than OpenCL, but it is also much closer to the metal.

#### 6.2 Queue groups in Level Zero expose engine-class information OpenCL does not present as directly

- **Source:** Level Zero Core Programming Guide
- **Key idea:** Level Zero lets applications query command queue group
  properties, including how many queues exist in a group and what command types
  that group supports.
- **Relevance to EdgeLM:** This matters because the project explicitly cares
  about runtime structure, not just raw kernel launch.
- **Estimated impact:** High.
- **Implementation complexity:** Medium.
- **Details:** Intel's guide says a queue group represents a physical input
  stream backed by one or more physical engines. That gives EdgeLM a direct hook
  for:
  - discovering compute-capable queues,
  - distinguishing queue families,
  - and matching recorded command lists to the intended queue group.

  OpenCL exposes queues and devices, but not this engine-oriented model in the
  same explicit way.

### 7. Level Zero's Separation of Command Queues and Command Lists Is a Better Match for a Tuned Inference Engine Than for a First Prototype

#### 7.1 Level Zero separates "where work executes" from "how work is recorded"

- **Source:** Level Zero Core Programming Guide
- **Key idea:** The spec explicitly says command queues and command lists are
  separate because:
  - queues are tied to physical device properties and input streams,
  - queues provide near-zero-latency access to the device,
  - command lists are mostly associated with host threads,
  - command-list construction can occur independently from submission.
- **Relevance to EdgeLM:** This is exactly the kind of control a specialized
  engine can use once the kernel set stabilizes.
- **Estimated impact:** High.
- **Implementation complexity:** Medium.
- **Details:** This separation opens useful design space for EdgeLM:
  - prebuild or recycle command lists,
  - let a submission thread own queue interaction,
  - let worker threads prepare future work without touching the queue directly,
  - and model queue submission as a first-class latency source.

#### 7.2 Level Zero explicitly includes throughput-oriented and low-latency submission modes

- **Source:** Level Zero Core Programming Guide
- **Key idea:** The spec describes:
  - `ZE_COMMAND_LIST_FLAG_MAXIMIZE_THROUGHPUT` for throughput-oriented command
    lists,
  - and recommends immediate command lists for very low-latency usage models.
- **Relevance to EdgeLM:** This is unusually relevant to autoregressive decode,
  where both throughput and per-token tail latency matter.
- **Estimated impact:** Very High.
- **Implementation complexity:** Medium.
- **Details:** This is one of the strongest arguments for eventually having a
  Level Zero backend:
  - prefill-like or batched work may prefer regular command lists optimized for
    throughput,
  - token-at-a-time decode stages may prefer immediate submission if command
    latency becomes visible in traces.

### 8. Immediate Command Lists Make Level Zero Especially Interesting for Token-by-Token Inference

#### 8.1 Immediate command lists are designed for very low dispatch latency

- **Source:** Level Zero Core Programming Guide and Intel oneAPI Level Zero guide
- **Key idea:** Level Zero explicitly includes `zeCommandListCreateImmediate`
  and positions immediate command lists as the right choice for very low-latency
  usage models.
- **Relevance to EdgeLM:** Token generation is a low-latency usage model. That
  does not automatically mean immediate lists will win, but it means the API has
  a mechanism aimed directly at this shape of workload.
- **Estimated impact:** High.
- **Implementation complexity:** Medium.
- **Details:** This is particularly interesting for:
  - short attention kernels during decode,
  - command streams that are too small to justify elaborate batching,
  - and cases where host-side submission overhead starts to matter as kernels
    get faster.

#### 8.2 Immediate command lists are not guaranteed to be the best choice everywhere

- **Source:** Level Zero Core Programming Guide
- **Key idea:** The same specification that introduces immediate command lists
  also preserves regular command-list workflows for throughput-oriented
  optimization.
- **Relevance to EdgeLM:** This argues for measurement, not dogma.
- **Estimated impact:** Medium.
- **Implementation complexity:** Low.
- **Details:** A sensible EdgeLM policy would be:
  - regular command lists for repeated or batched tile sequences,
  - immediate command lists only for measured low-latency hotspots,
  - and no assumption that one submission style wins globally.

### 9. Level Zero's Synchronization Model Is More Powerful Than OpenCL's, but It Is Also Easier To Misuse

#### 9.1 There is no implicit memory or cache coherency between commands in Level Zero

- **Source:** Level Zero Core Programming Guide
- **Key idea:** The spec explicitly states that commands executed on a command
  list are not guaranteed to maintain memory coherency with other commands, and
  that there is no implicit memory or cache coherency.
- **Relevance to EdgeLM:** This is the most important Level Zero programming
  fact for correctness. The engine must treat synchronization and coherency as
  explicit program logic, not driver magic.
- **Estimated impact:** Very High.
- **Implementation complexity:** Medium.
- **Details:** The spec distinguishes:
  - fences as coarse-grain host-visible completion and coherency points,
  - events as fine-grain execution and/or memory dependencies with explicit
    scope control.

  That is excellent for performance tuning, but it also means the backend can
  silently become wrong if event scopes are chosen carelessly.

#### 9.2 Bad event design can introduce bubbles or deadlocks

- **Source:** Level Zero Core Programming Guide
- **Key idea:** The spec warns that events can create bubbles in the pipeline or
  deadlock situations if not scheduled correctly.
- **Relevance to EdgeLM:** A custom inference runtime with double buffering,
  host staging, and optional overlap is exactly the sort of code that can wander
  into these failure modes.
- **Estimated impact:** High.
- **Implementation complexity:** Medium.
- **Details:** This pushes EdgeLM toward a disciplined backend design:
  - one narrow submission layer,
  - a limited number of synchronization patterns,
  - clear ownership of event pools,
  - no ad hoc event graph scattered across the engine.

### 10. Level Zero Gives Much Finer Memory Control Than OpenCL, but Shared-Memory Laptop Economics Still Dominate

#### 10.1 Level Zero explicitly exposes host, device, shared, and shared-system allocations with different access guarantees

- **Source:** Level Zero Core Programming Guide
- **Key idea:** Level Zero defines several allocation classes:
  - host allocations,
  - device allocations,
  - shared allocations,
  - shared-system allocations,
  with explicit access-capability flags and optional concurrent/concurrent-atomic
  semantics.
- **Relevance to EdgeLM:** This is much richer than a naive "GPU memory vs CPU
  memory" model and lets the runtime choose memory placement deliberately.
- **Estimated impact:** High.
- **Implementation complexity:** Medium.
- **Details:** The most important nuance is that concurrent access capability
  does **not** automatically imply coherency or correctness. The spec is very
  explicit that concurrent access semantics must still be respected and that
  undefined behavior exists if unsupported patterns are used.

#### 10.2 Shared allocations are attractive on an iGPU, but they do not eliminate migration, caching, or contention costs

- **Source:** Level Zero Core Programming Guide plus Intel oneAPI memory
  performance guidance
- **Key idea:** Level Zero shared allocations are migratable and may be
  prefetched or given memory advice. Intel's GPU optimization guide also notes
  that shared-memory models can incur page-fault and migration overheads whose
  impact depends strongly on access pattern.
- **Relevance to EdgeLM:** This is where many integrated-GPU designs go wrong.
  "Shared allocation" does not mean "free zero-cost cross-processor sharing."
- **Estimated impact:** Very High.
- **Implementation complexity:** Medium.
- **Details:** Intel's memory guidance points out that:
  - page-fault servicing adds overhead,
  - sparse random access is worst,
  - linear scans are less harmful,
  - and explicit synchronization still matters.

  For EdgeLM, that means:
  - shared allocations may be good for coarse staging and persistent buffers,
  - but they are not a blanket excuse to let CPU and iGPU mutate the same data
    structure opportunistically.

#### 10.3 Level Zero's prefetch and memory-advice APIs are promising, but they must be judged against the DDR4 bottleneck

- **Source:** Level Zero Core Programming Guide
- **Key idea:** The spec exposes `zeCommandListAppendMemoryPrefetch` and
  `zeCommandListAppendMemAdvise` for shared allocations.
- **Relevance to EdgeLM:** These are attractive because the whole project is
  already obsessed with staged movement and memory timing.
- **Estimated impact:** Medium to High.
- **Implementation complexity:** Medium.
- **Details:** These APIs are worth testing for:
  - persistent attention buffers,
  - repeated K/V windows,
  - or page-migration smoothing across repeated kernels.

  But the project should remember the larger Section 16 truth: on this laptop,
  all such wins still cash out against a shared DDR4 budget.

### 11. A Shared SPIR-V Kernel Path Can Bridge OpenCL and Level Zero Cleanly

#### 11.1 OpenCL and Level Zero can both consume SPIR-V-based kernels

- **Source:** Khronos OpenCL API Specification and Level Zero Core Programming Guide
- **Key idea:** OpenCL devices can report support for IL such as `SPIR-V` and
  ingest it via `clCreateProgramWithIL`. Level Zero modules can also be created
  from `ZE_MODULE_FORMAT_IL_SPIRV`.
- **Relevance to EdgeLM:** This is one of the highest-leverage architectural
  findings in the whole section.
- **Estimated impact:** Very High.
- **Implementation complexity:** Medium.
- **Details:** This creates a clean staged development story:
  - author kernels in OpenCL C or a compatible frontend,
  - compile to SPIR-V offline,
  - use OpenCL for easy bring-up if desired,
  - and later feed the same IL into Level Zero modules for a lower-level host
    backend.

  That means EdgeLM does **not** need to choose between "portable kernel source"
  and "Intel-specific host runtime control." It can have both.

#### 11.2 Level Zero specialization constants make SPIR-V especially attractive for tuned kernels

- **Source:** Level Zero Core Programming Guide
- **Key idea:** Level Zero allows specialization constants to be overridden at
  module-creation time.
- **Relevance to EdgeLM:** This is directly useful for kernel tuning parameters
  such as:
  - tile sizes,
  - group sizes,
  - unroll thresholds,
  - and other compile-time constants that may differ between experiments.
- **Estimated impact:** High.
- **Implementation complexity:** Medium.
- **Details:** This can shorten the inner loop of optimization:
  - keep one kernel family,
  - vary constants at module creation,
  - measure,
  - and only regenerate source or SPIR-V when truly necessary.

### 12. Tooling Is Good on Both Sides, but the Tooling Story Is Different Enough To Matter

#### 12.1 OpenCL has a very friendly interception/debugging path

- **Source:** Intel Intercept Layer for OpenCL Applications
- **Key idea:** Intel's Intercept Layer can intercept and modify OpenCL calls
  for debugging and performance analysis without requiring application or driver
  modifications, by sitting in front of the ICD loader.
- **Relevance to EdgeLM:** This is a strong pragmatic advantage during bring-up.
- **Estimated impact:** High for debugging velocity.
- **Implementation complexity:** Low.
- **Details:** Intel's own example output includes timing summaries for calls
  such as:
  - `clBuildProgram`
  - `clCreateBuffer`
  - `clCreateContext`
  - `clCreateKernel`
  - `clCreateProgramWithIL`

  That is exactly the kind of visibility we want when separating compilation
  cost, setup cost, and steady-state dispatch cost.

#### 12.2 Level Zero has comparable tracing, but via a different tooling stack

- **Source:** Intel Level Zero Tracer and Intel profiling guidance
- **Key idea:** Intel describes `ze_tracer` as the analogue of the OpenCL
  intercept layer for Level Zero and also documents tracing tools that can
  collect host and device timing for Level Zero and OpenCL backends.
- **Relevance to EdgeLM:** The tooling gap between the two APIs is not fatal,
  but it changes bring-up ergonomics.
- **Estimated impact:** Medium.
- **Implementation complexity:** Low to Medium.
- **Details:** Intel's tracing guidance shows:
  - per-API timing summaries,
  - command-queue/device timelines,
  - JSON traces viewable in Chrome tracing tools,
  - and support for both Level Zero and OpenCL-oriented workflows.

  So the true tradeoff is not "OpenCL has tools, Level Zero doesn't." The real
  tradeoff is:
  - OpenCL: slightly simpler first-step tooling
  - Level Zero: equally serious tooling, but with a more explicit runtime model

### 13. The Best EdgeLM Strategy Is a Staged Backend Architecture, Not a One-Time API Bet

#### 13.1 OpenCL best fits the first functional backend

- **Source:** Khronos OpenCL model, current implementation plan, Intel OpenCL tooling
- **Key idea:** OpenCL matches the existing repo plan and minimizes host-side
  complexity during the first correctness phase.
- **Relevance to EdgeLM:** This is the best path to a working experiment quickly.
- **Estimated impact:** Very High.
- **Implementation complexity:** Low.
- **Details:** OpenCL is the best default for the earliest milestone because:
  - it matches the `.cl`-file workflow already in the roadmap,
  - the host-side API is easier to bootstrap,
  - the build/debug loop is friendlier,
  - and the first objective is correctness plus first measurements, not final
    submission latency.

#### 13.2 Level Zero best fits the final tuned backend if command submission and memory control become measurable bottlenecks

- **Source:** Level Zero Core Programming Guide and Intel runtime model
- **Key idea:** Level Zero's explicit queue groups, command lists, immediate
  command lists, memory classes, event scopes, and specialization constants make
  it a better fit for a mature Intel-specific engine.
- **Relevance to EdgeLM:** This is where the project's hardware-specific
  philosophy really points.
- **Estimated impact:** Very High.
- **Implementation complexity:** Medium to High.
- **Details:** The correct test is not ideological. It is empirical:
  - if OpenCL already meets the performance target for attention offload,
    keep it,
  - if traces show submission, synchronization, or memory-placement limits,
    Level Zero becomes the right next step.

## Techniques Comparison Matrix

| Technique | Source | Impact | Complexity | Already in Deep Dive? |
|-----------|--------|--------|------------|-----------------------|
| Start with OpenCL for first correctness-oriented backend | Khronos OpenCL spec + current implementation plan | Very High | Low | Partially |
| Treat OpenCL 3.0 as a queried feature set, not a guaranteed capability bundle | Khronos OpenCL 3.0 guidance | Very High | Low | No |
| Use OpenCL source build logs first, then move to IL/SPIR-V for steadier startup behavior | Khronos OpenCL spec | High | Medium | Partially |
| Keep OpenCL command buffers optional, not baseline | Khronos `cl_khr_command_buffer` | Medium | Medium | No |
| Use Level Zero when queue-group control or submission latency becomes measurable | Level Zero spec | Very High | Medium | No |
| Evaluate immediate command lists only after a regular-command-list baseline exists | Level Zero spec | High | Medium | No |
| Build one SPIR-V kernel pipeline that can serve both OpenCL and Level Zero backends | Khronos OpenCL spec + Level Zero spec | Very High | Medium | No |
| Use Level Zero specialization constants for kernel tuning rather than source rewrites when possible | Level Zero spec | High | Medium | No |
| Keep a narrow synchronization layer because Level Zero has no implicit memory/cache coherency | Level Zero spec | Very High | Medium | No |
| Use Intel tracing from the first backend iteration, not after optimization begins | Intel Intercept Layer + Level Zero Tracer | High | Low | No |

## Recommendations for EdgeLM

1. Make **OpenCL the first functional iGPU backend** because it best matches the
   existing implementation plan and minimizes host-side complexity.

2. Do **not** make runtime source JIT the long-term kernel strategy. Keep the
   kernel authoring flow compatible with offline `SPIR-V` generation from day
   one.

3. Build the host runtime behind a thin backend interface:
   - `gpu_backend_opencl.c`
   - `gpu_backend_level_zero.c`
   - shared backend-neutral kernel descriptors and launch metadata

4. Treat `OpenCL 3.0` as a feature-query API. At startup, log at least:
   - IL support
   - subgroup-related features you intend to use
   - SVM-related support if considered
   - queue capabilities
   - device memory limits

5. Keep the first OpenCL backend simple:
   - one context
   - one device
   - one in-order queue
   - coarse events only where measurement requires them

6. Add a `SPIR-V` build path early, even if the first experiments still use
   `.cl` source builds for convenience.

7. Add a Level Zero backend **only after** the OpenCL microbenchmarks answer two
   questions:
   - Is kernel math itself fast enough?
   - Is host-side submission/coherency overhead showing up materially?

8. When the Level Zero backend is built, start with:
   - one context
   - one compute queue group
   - one regular command list path
   - explicit fences/events only in a few reviewed patterns

9. Evaluate immediate command lists only for decode-style, small-kernel,
   low-latency loops. Do not assume they are universally better.

10. Be conservative with shared allocations. Benchmark:
    - shared allocation
    - host allocation + explicit copy
    - device allocation + explicit copy
    before standardizing on one memory policy.

11. Use tracing from the first real backend experiment:
    - OpenCL Intercept Layer for OpenCL bring-up
    - `ze_tracer` / PTI tooling for Level Zero
    - record submission cost separately from kernel time

12. Install at least one local inspection tool in Phase 0 or early Phase 4.
    This workspace currently had no `clinfo`, `zeinfo`, or `sycl-ls` in `PATH`,
    which slows down environment validation.

13. The best long-term shape is:
    - **OpenCL for early correctness and rapid kernel iteration**
    - **shared SPIR-V artifacts for portability across backends**
    - **Level Zero for the final Intel-specific tuned path if measurements justify it**

## References

1. Intel, `compute-runtime` README and support matrix.  
   https://github.com/intel/compute-runtime

2. Intel, "Level Zero," oneAPI GPU Optimization Guide.  
   https://www.intel.com/content/www/us/en/docs/oneapi/optimization-guide-gpu/2024-2/level-zero.html

3. oneAPI, "Level Zero Core Programming Guide."  
   https://oneapi-src.github.io/level-zero-spec/level-zero/latest/core/PROG.html

4. oneAPI, "Level Zero Core API."  
   https://oneapi-src.github.io/level-zero-spec/level-zero/latest/core/api.html

5. Khronos, "The OpenCL Specification."  
   https://registry.khronos.org/OpenCL/specs/3.0-unified/html/OpenCL_API.html

6. Khronos, "OpenCL 3.0 Specification Finalized and Initial Khronos Open Source OpenCL SDK Released."  
   https://www.khronos.org/blog/opencl-3.0-specification-finalized-and-initial-khronos-open-source-opencl-sdk-released

7. Khronos, "`clFinalizeCommandBufferKHR` manual page."  
   https://registry.khronos.org/OpenCL/sdk/3.0/docs/man/html/clFinalizeCommandBufferKHR.html

8. Intel, "Intel Intercept Layer for OpenCL Applications," oneAPI GPU Optimization Guide.  
   https://www.intel.com/content/www/us/en/docs/oneapi/optimization-guide-gpu/2024-0/intel-intercept-layer-for-opencltm-applications.html

9. Intel, "Level Zero Tracer," oneAPI GPU Optimization Guide.  
   https://www.intel.com/content/www/us/en/docs/oneapi/optimization-guide-gpu/2023-1/level-zero-tracer.html

10. Intel, "Performance Impact of USM and Buffers," oneAPI GPU Optimization Guide.  
    https://www.intel.com/content/www/us/en/docs/oneapi/optimization-guide-gpu/2025-2/performance-impact-of-usm-and-buffers.html

11. Intel, "oneAPI GPU Optimization Guide" overview.  
    https://www.intel.com/content/www/us/en/docs/oneapi/optimization-guide-gpu/2025-2/overview.html

## Audit Addendum (2026-04-02)

- **Offline binary caching should be built into the backend plan.** Recompiling
  kernels or rebuilding module state on every run would distort both startup and
  benchmark results.
- **Backend fallback order should be explicit.** EdgeLM should define a clean
  chain such as:
  - CPU-only,
  - OpenCL prototype path,
  - Level Zero tuned path,

  rather than treating backend selection as ad hoc bring-up logic.
- **Debug and validation modes should be benchmark-separable.** Tool-assisted
  correctness runs are valuable, but the runtime must also make it obvious when
  a slower instrumented path is active.
