# Section 18: CPU + iGPU Hybrid Pipeline -- Extended Research

## Overview

This section asks the most important practical question in the whole iGPU path:
what should the **actual** CPU+iGPU pipeline for EdgeLM look like on the target
`i7-12700H + Iris Xe + dual-channel DDR4` laptop?

That is a much harder question than "can we offload something to the iGPU?"
Sections 16 and 17 already showed that:

- the Iris Xe iGPU is a real compute target,
- OpenCL and Level Zero are both viable programming interfaces,
- the device has meaningful subgroup and SLM capabilities,
- and the hardware is constrained by shared system memory rather than dedicated
  VRAM.

So Section 18 is not a generic heterogeneous-computing discussion. It is about
the shape of the end-to-end inference pipeline:

- what stays on CPU,
- what moves to iGPU,
- whether overlap is actually beneficial,
- whether the current project sketch of layer-level CPU/iGPU pipelining is valid
  for autoregressive decode,
- how many handoff boundaries are acceptable,
- how buffering and submission should work,
- and how policy should differ between prompt prefill and token-by-token decode.

This section is where the optimistic roadmap has to meet the dependency graph
and the memory system. If the design is wrong here, the iGPU path will be
slower than CPU-only despite good kernels.

## What the Deep Dive Already Covers

`deep-dive.md` remains empty, so the relevant project context comes from
`implementation-plan.md`, `AGENTS.md`, and the immediately preceding research
sections.

- The implementation plan already proposes a hybrid path with expected impact of
  `15-30%` by offloading attention to the iGPU.
- The current sketch in `implementation-plan.md` is:
  - CPU handles FFN / MLP because ternary matmul is weight-bound and CPU-tuned
  - iGPU helps with attention because score computation / softmax appear more
    data-parallel
  - double buffering is used
  - bandwidth contention is a known risk
  - time-slicing may be necessary if overlap is too expensive
- The plan also contains a more aggressive visual sketch:

  ```text
  Token N:    [CPU: FFN layer L]  [iGPU: Attention layer L+1]
  Token N+1:  [CPU: FFN layer L+1]  [iGPU: Attention layer L+2]
  ```

- Section 15 established that P-cores should own the hot AVX2/AVX-VNNI decode
  work while E-cores should own orchestration, queueing, and other helper work.
- Section 16 established that the iGPU is a `96-EU Xe-LP` vector device with
  shared DDR4 memory and that CPU+iGPU overlap is fundamentally a shared-DRAM
  problem.
- Section 17 established that:
  - OpenCL is the best first backend for bring-up,
  - Level Zero is the most promising final low-level path if host-side control
    becomes a real bottleneck,
  - shared `SPIR-V` artifacts can bridge both backends,
  - and memory movement / synchronization policy must be explicit.

So the unresolved questions entering this section are not about architecture or
API viability anymore. They are specifically about pipeline design:

- Is the project's current layer-level diagram actually legal for single-stream
  decode?
- Should the same kernel ever be split across CPU and iGPU?
- What is the right handoff boundary inside an attention block?
- Is overlap or time-slicing the default?
- Does the answer differ between prefill and decode?

## New Findings

### 1. "Hybrid Pipeline" Can Mean Several Different Things, and Only Some of Them Are Good Fits for EdgeLM

#### 1.1 There are at least four distinct heterogeneous execution patterns hiding behind one phrase

- **Source:** Intel oneAPI `Using Multiple Heterogeneous Devices`, OpenVINO
  `Heterogeneous Execution`, HeteGen paper
- **Key idea:** In practice, "CPU+iGPU pipeline" can mean four different
  designs:
  1. **Host/device async control pipeline**: CPU prepares work, submits GPU
     kernels, waits, postprocesses.
  2. **Stage offload**: entire operator families or subgraphs are assigned to
     CPU or GPU.
  3. **True co-processing of the same operator**: CPU and GPU share one module's
     work simultaneously.
  4. **Pipeline parallelism across stages/microbatches**: different devices work
     on different stages of different microbatches at once.
- **Relevance to EdgeLM:** These modes have very different costs. Treating them
  as the same thing hides most of the real design tradeoffs.
- **Estimated impact:** Very High.
- **Implementation complexity:** Low to Medium.
- **Details:** EdgeLM should not ask "hybrid or not?" It should ask:
  - which of the four forms is appropriate for prefill,
  - which is appropriate for decode,
  - and which should be avoided entirely on a shared-memory laptop.

#### 1.2 For EdgeLM on Iris Xe, the best first candidates are host/device async control and coarse stage offload

- **Source:** Intel oneAPI guidance, OpenVINO heterogeneous execution model,
  prior EdgeLM sections
- **Key idea:** The two hybrid forms that best fit the project are:
  - a **CPU-controlled async pipeline** around the iGPU
  - a **coarse operator-family split** between CPU and iGPU
- **Relevance to EdgeLM:** These two models preserve control and minimize
  communication complexity.
- **Estimated impact:** Very High.
- **Implementation complexity:** Medium.
- **Details:** By contrast:
  - same-operator CPU+iGPU co-processing is harder to balance and synchronize,
  - deep pipeline parallelism across model stages is only conditionally useful,
  - and both are much riskier on a bandwidth-limited integrated-memory machine.

### 2. The Current Layer-Level Diagram in the Implementation Plan Is Too Optimistic for Single-Stream Decode

#### 2.1 The sketch `CPU: FFN layer L` in parallel with `iGPU: Attention layer L+1` is not a valid default model for single-request autoregressive decode

- **Source:** `implementation-plan.md`; inference from transformer dataflow and
  autoregressive decoding dependencies
- **Key idea:** In a standard transformer decode pass, layer `L+1` cannot start
  until layer `L` has produced its output for the same token, and token `N+1`
  cannot begin its model pass until token `N` has produced logits and sampling
  is complete.
- **Relevance to EdgeLM:** This means the current pipeline sketch should not be
  treated as the baseline execution model for single-stream decode.
- **Estimated impact:** Very High.
- **Implementation complexity:** Low.
- **Details:** This point is an inference rather than a direct quote from one
  source, but it follows directly from the model dependency graph:
  - same-token layers are strictly ordered,
  - next-token decode depends on current-token completion,
  - and therefore naive cross-layer device overlap is far more limited than the
    roadmap diagram suggests.

  The sketch can become more realistic only under one of the following:
  - prompt prefill with microbatching or chunking,
  - multiple simultaneous requests,
  - speculative decoding,
  - or a more carefully staged intra-layer pipeline.

#### 2.2 This does not kill the hybrid idea, but it changes where the available overlap lives

- **Source:** Intel oneAPI non-blocking submit guidance; OpenVINO async pipeline
  model; inference from decode dependencies
- **Key idea:** Once the strict decode dependency chain is acknowledged, the real
  overlap opportunities move away from "different model layers on different
  devices at once" and toward:
  - host-side preparation while GPU runs,
  - communication overlap,
  - prefetch/pinning overlap,
  - and chunked prefill pipelines.
- **Relevance to EdgeLM:** This is a design correction, not a rejection of the
  iGPU path.
- **Estimated impact:** Very High.
- **Implementation complexity:** Medium.
- **Details:** The hybrid pipeline for decode should be designed around
  **coarse stage offload plus narrow overlap**, not around an assumption of wide,
  stable parallel execution of unrelated transformer stages.

### 3. Prefill and Decode Must Use Different Hybrid Policies

#### 3.1 Prefill is the better place to exploit deeper CPU+iGPU pipelining

- **Source:** OpenVINO pipeline-parallelism guidance; Intel heterogeneous-device
  guidance; inference from transformer workload shape
- **Key idea:** Prompt prefill operates on larger token blocks and larger
  attention shapes, so kernel launch overheads and handoff costs are amortized
  better than in single-token decode.
- **Relevance to EdgeLM:** If the project wants to explore true stage-pipelined
  CPU+iGPU execution, prefill is the right place to start.
- **Estimated impact:** High.
- **Implementation complexity:** Medium.
- **Details:** This is again an inference from the workload structure rather
  than a single vendor sentence:
  - prefill has larger working sets per launch,
  - more arithmetic per submission,
  - more opportunity for chunk/microbatch pipelining,
  - and more headroom to hide host-side control overhead.

  That makes prefill the best target for experiments such as:
  - chunked prompt processing,
  - stage pipelines across prompt chunks,
  - and GPU-heavy attention blocks with CPU preparation of the next chunk.

#### 3.2 Decode is the stricter regime and should be optimized around latency-aware stage ownership, not deep stage pipelines

- **Source:** Intel low-latency submission guidance from Section 17; inference
  from autoregressive dependency chain
- **Key idea:** Decode has batch size close to one, hard sequential
  dependencies, and a tighter sensitivity to launch overhead and synchronization.
- **Relevance to EdgeLM:** This means the decode hybrid policy should default to:
  - minimal handoffs,
  - clear stage ownership,
  - and carefully measured overlap only where it is proven to help.
- **Estimated impact:** Very High.
- **Implementation complexity:** Medium.
- **Details:** The consequence is subtle but important:
  - prefill and decode may both use the iGPU,
  - but they should not necessarily use the **same** hybrid pipeline policy.

  A good design for EdgeLM is likely:
  - **Prefill:** more aggressive chunked or pipelined experimentation
  - **Decode:** conservative, stage-offloaded, low-handoff execution

### 4. Intel's Own Heterogeneous-Device Guidance Says the Application Must Manage Distribution and Overlap Explicitly

#### 4.1 Work submission to accelerators is non-blocking, which allows host-side overlap

- **Source:** Intel oneAPI `Using Multiple Heterogeneous Devices`
- **Key idea:** Intel states that command-group submissions to accelerators are
  non-blocking, so the host can continue doing useful work or submit work to
  other devices while the accelerator is running.
- **Relevance to EdgeLM:** This is the official basis for the CPU-side
  orchestration model:
  - a submission thread can enqueue iGPU work,
  - other CPU threads can continue pre/post work,
  - and the hot P-cores need not block just because a GPU kernel was launched.
- **Estimated impact:** High.
- **Implementation complexity:** Low.
- **Details:** This supports the current plan's idea of a dedicated iGPU
  scheduler thread, and it strongly suggests that thread should live on an
  E-core rather than stealing a hot P-core from ternary matmul.

#### 4.2 Intel also says balanced distribution is the programmer's responsibility

- **Source:** Intel oneAPI `Using Multiple Heterogeneous Devices`
- **Key idea:** Intel explicitly says the programmer is responsible for
  distributing work between devices in a balanced manner, taking their differing
  capabilities into account.
- **Relevance to EdgeLM:** This is a strong reminder that "use both devices" is
  not a scheduling policy.
- **Estimated impact:** Very High.
- **Implementation complexity:** Medium.
- **Details:** For EdgeLM, this means:
  - never split work evenly by bytes or rows without measurement,
  - do not assume the iGPU should always be active just because it exists,
  - and treat device assignment as a measured policy function of phase, sequence
    length, and operator type.

### 5. Overlap on a Single iGPU Is Narrower Than It First Appears

#### 5.1 Intel's overlap guidance is mainly about overlapping copies and compute, not about magically running all kernels in parallel

- **Source:** Intel oneAPI `Asynchronous and Overlapping Data Transfers Between Host and Device`
- **Key idea:** Intel's guide explains overlap in terms of transfer engines,
  compute engines, and queue orchestration. The example also notes that on a
  single GPU the two example kernels themselves cannot execute concurrently.
- **Relevance to EdgeLM:** This sharply limits what "async everything" should
  mean for the project.
- **Estimated impact:** High.
- **Implementation complexity:** Medium.
- **Details:** On this laptop, the realistic overlap modes are:
  - host-side CPU work while a GPU kernel runs,
  - copies or migrations while compute runs, if hardware/runtime permit it,
  - lightweight queue/control work in parallel with compute.

  The unrealistic expectation is:
  - multiple big attention kernels all making simultaneous forward progress on a
    single iGPU in a way that meaningfully hides the core costs.

#### 5.2 Some overlap techniques are more applicable to discrete-GPU-style setups than to this shared-memory laptop

- **Source:** Intel overlap guide; Section 16 shared-memory findings
- **Key idea:** Intel's overlap guidance frequently assumes the presence of
  copy/compute engine distinctions and explicit host/device transfers, while
  EdgeLM is targeting a shared-memory iGPU where both devices ultimately compete
  for the same DRAM.
- **Relevance to EdgeLM:** This means overlap should be treated as conditional,
  not as the baseline design goal.
- **Estimated impact:** Very High.
- **Implementation complexity:** Medium.
- **Details:** This is partly an inference from the sources:
  - the Intel docs show how overlap can exist,
  - Section 16 shows the iGPU uses shared system memory,
  - and the project's own bandwidth model says DRAM is the bottleneck.

  Therefore, even when overlap is technically possible, it may still be a net
  loss if both devices are streaming the same DDR4 subsystem hard enough.

### 6. OpenVINO's Official Heterogeneous Execution Docs Deliver a Strong Warning for Shared-Memory CPU+iGPU Splits

#### 6.1 OpenVINO represents heterogeneous execution as subgraph partitioning with explicit affinities and intermediate tensors

- **Source:** OpenVINO `Heterogeneous Execution`
- **Key idea:** OpenVINO's HETERO flow:
  - assigns affinities to devices,
  - splits the model into subgraphs,
  - injects intermediate tensors between those subgraphs,
  - and can execute them in a pipelined manner.
- **Relevance to EdgeLM:** This is an official, production-minded reference
  architecture for the same core problem EdgeLM is solving manually.
- **Estimated impact:** High.
- **Implementation complexity:** Medium.
- **Details:** Two lessons transfer directly:
  - the split boundary should be explicit and coarse,
  - and every additional boundary creates intermediate tensors and scheduling
    overhead that must be justified.

#### 6.2 OpenVINO explicitly warns that because `iGPU` and `CPU` share host memory, it is recommended to use at most one of them in the device list

- **Source:** OpenVINO `Heterogeneous Execution`
- **Key idea:** OpenVINO states that for pipeline parallelism, since `iGPU` and
  `CPU` share host memory, it is recommended to use at most one of them and put
  it at the end of the list.
- **Relevance to EdgeLM:** This is the strongest official warning encountered so
  far against naive CPU+iGPU co-execution on an integrated-memory Intel system.
- **Estimated impact:** Very High.
- **Implementation complexity:** Low.
- **Details:** This does **not** mean EdgeLM should abandon the hybrid path.
  It means the hybrid path must be:
  - conservative,
  - coarse-grained,
  - and heavily measured.

  If Intel's own production inference stack is this cautious about CPU+iGPU
  pipelining on shared host memory, EdgeLM should be at least as cautious.

#### 6.3 OpenVINO also recommends minimizing the number of subgraphs to optimize memory transfer overhead

- **Source:** OpenVINO `Heterogeneous Execution`
- **Key idea:** OpenVINO explicitly advises users to set affinities to minimize
  the number of subgraphs when optimizing transfer overhead.
- **Relevance to EdgeLM:** This translates into a direct design rule for the
  custom engine: keep CPU/iGPU handoff boundaries few and coarse.
- **Estimated impact:** Very High.
- **Implementation complexity:** Low.
- **Details:** This is one of the simplest and most important rules in this
  section:
  - **one coarse handoff is plausible**
  - **many fine-grain handoffs are almost certainly wrong**

### 7. The Right EdgeLM Split Is Probably Not "All Attention on GPU" but a Narrower Intra-Attention Boundary

#### 7.1 Weight-bound linear layers should stay on the CPU, including the attention projections

- **Source:** `implementation-plan.md`; Sections 07, 16, and 17; inference from
  the target architecture
- **Key idea:** The Q/K/V projections and output projection inside attention are
  still weight-bearing linear layers. On EdgeLM, those weights are the part of
  the model most directly matched to the CPU's custom ternary kernels.
- **Relevance to EdgeLM:** This suggests the current plan's shorthand phrase
  "attention on the iGPU" is too broad.
- **Estimated impact:** Very High.
- **Implementation complexity:** Medium.
- **Details:** This is an inference from the overall project design rather than
  a sentence from one official guide:
  - the CPU is being custom-built for ternary matmul,
  - weight traffic is the dominant issue,
  - the iGPU shares the same DDR4 pool,
  - and Xe-LP does not have specialized matrix engines for ternary inference.

  Therefore the most defensible split is:
  - **CPU:** RMSNorm, Q/K/V projections, O projection, FFN, sampling, control
  - **iGPU:** attention score matmul, masking, softmax, and value application

  That is a smaller and more realistic iGPU target than "the whole attention
  layer."

#### 7.2 The best first offload boundary is likely `CPU QKV -> iGPU score/softmax/value -> CPU O projection`

- **Source:** Intel `Avoid moving data back and forth between host and device`;
  OpenVINO subgraph-minimization guidance; prior EdgeLM sections
- **Key idea:** If the iGPU is used, the boundary should isolate the most
  data-parallel, reuse-heavy, weight-light part of attention while keeping the
  weight-bound projections on CPU.
- **Relevance to EdgeLM:** This gives the project a concrete first hybrid
  pipeline shape rather than a generic aspiration.
- **Estimated impact:** Very High.
- **Implementation complexity:** Medium.
- **Details:** This split has several advantages:
  - it gives the iGPU a coherent subpipeline,
  - it minimizes repeated CPU/iGPU bouncing of tiny intermediates,
  - it keeps ternary-packed weights on the CPU side,
  - and it preserves a single, understandable handoff inside the attention
    block.

  It is not the only possible split, but it is the one most aligned with the
  entire rest of the EdgeLM architecture.

### 8. Same-Operator CPU+iGPU Co-Processing Should Be a Late Experiment, Not the Default

#### 8.1 HeteGen shows that heterogeneous splitting needs explicit balancing of compute and communication, not just device availability

- **Source:** HeteGen, arXiv:2403.01164
- **Key idea:** HeteGen derives a distribution ratio `alpha` using CPU compute,
  GPU compute, and communication terms, and explicitly works to maximize the
  overlap of data transfer with CPU computation.
- **Relevance to EdgeLM:** This is the right mental model if the project ever
  experiments with CPU+iGPU co-processing of the same module.
- **Estimated impact:** Medium to High.
- **Implementation complexity:** High.
- **Details:** The value here is not the exact HeteGen formula, because EdgeLM's
  integrated-memory path differs from the paper's environment. The value is the
  principle:
  - collaborative split is a balancing problem,
  - communication cost must be modeled explicitly,
  - and naive "split 50/50" policies are wrong.

#### 8.2 EdgeLM should assume whole-stage ownership first, and split a single operator only if profiling proves it worthwhile

- **Source:** HeteGen balancing model; OpenVINO coarse subgraph guidance; Intel
  device-balancing guidance; Damschen et al. on fused CPU-GPU co-scheduling
- **Key idea:** The default policy should be exclusive ownership:
  - CPU owns an operator or substage
  - or iGPU owns an operator or substage
  - but not both simultaneously
- **Relevance to EdgeLM:** This keeps the first hybrid implementation simple,
  analyzable, and debuggable.
- **Estimated impact:** Very High.
- **Implementation complexity:** Low.
- **Details:** A same-operator split should become legal only after the project
  can answer all of these with measurements:
  - What is the CPU-only time?
  - What is the iGPU-only time?
  - What is the handoff cost?
  - What is the DRAM contention penalty?
  - Can communication be hidden enough to win overall?

  If not, the split should not exist.

  Damschen et al. provide especially relevant supporting evidence here: on fused
  CPU-GPU architectures with shared LLC, they report that for most kernels it
  was not beneficial to split one kernel between CPU and GPU, and they trace the
  loss to inefficient cache coherence rather than simple cache-miss growth. That
  is not a proof that EdgeLM can never benefit from collaborative execution, but
  it is a very strong reason to make exclusive stage ownership the default.

### 9. Double Buffering Is Still Useful, but It Must Be Applied to Workspaces and Handoffs, Not as a Generic Cure-All

#### 9.1 The best use of ping-pong buffers is to decouple producer/consumer stages at a single coarse handoff

- **Source:** Intel async transfer guidance; OpenVINO AsyncInferRequest design;
  current implementation plan
- **Key idea:** Double buffering is most defensible when it isolates a producer
  stage and a consumer stage around one stable boundary.
- **Relevance to EdgeLM:** This means buffer pairs should exist around the
  chosen CPU/iGPU handoff point inside attention, not around every minor step.
- **Estimated impact:** High.
- **Implementation complexity:** Medium.
- **Details:** For the first hybrid implementation, the ping-pong set should be
  small and purposeful:
  - one or two activation buffers for the Q/K/V-derived inputs the iGPU needs,
  - one or two output buffers for the attention result returned to CPU,
  - one iGPU-side scratch arena for score / softmax / value tiles.

  The goal is to overlap stage turnover and submission latency, not to duplicate
  half the model's working set.

#### 9.2 Double buffering does not justify bandwidth-heavy simultaneous execution by itself

- **Source:** Section 16 shared-memory findings; Intel overlap guidance
- **Key idea:** A ping-pong buffer can remove producer/consumer serialization,
  but it cannot repeal the DDR4 bottleneck.
- **Relevance to EdgeLM:** This corrects an easy design trap in the current
  roadmap.
- **Estimated impact:** Very High.
- **Implementation complexity:** Low.
- **Details:** If both sides still need to stream large volumes of data from the
  same DRAM at the same time, double buffering may improve structure without
  improving total time. In the worst case, it can make the implementation more
  complex while preserving the bottleneck unchanged.

### 10. The CPU Side of the Hybrid Pipeline Should Look More Like an Async Runtime Than Like a Monolithic Worker Thread

#### 10.1 OpenVINO's async inference pipeline separates submit, wait, and postprocess responsibilities for a reason

- **Source:** OpenVINO `AsyncInferRequest`
- **Key idea:** OpenVINO's async design uses different executors for tasks such
  as CPU preprocessing, pipelined start, wait, and CPU postprocessing, and
  explicitly warns that wait stages can lock a thread.
- **Relevance to EdgeLM:** This is directly applicable to the custom runtime.
- **Estimated impact:** High.
- **Implementation complexity:** Medium.
- **Details:** EdgeLM should copy the spirit of this design even though it is
  not using OpenVINO:
  - one narrow submission path,
  - a separate wait/completion path,
  - CPU-side pre/post work kept off the hot SIMD workers,
  - and clear ownership of buffer lifetime around the device boundary.

#### 10.2 The iGPU submission path belongs on helper cores, not on the hot P-core kernel workers

- **Source:** Section 15 scheduling findings; current implementation plan; Intel
  non-blocking accelerator-submit guidance
- **Key idea:** Since GPU submission and wait handling are control-path work,
  they should be scheduled as cheap helper activity rather than consuming prime
  vector-compute cores.
- **Relevance to EdgeLM:** This reinforces the roadmap's idea of a dedicated
  iGPU submission thread.
- **Estimated impact:** High.
- **Implementation complexity:** Low.
- **Details:** The likely shape is:
  - one E-core-bound submission thread,
  - optional E-core-bound wait/completion thread,
  - P-cores reserved for ternary matmul and other hot CPU kernels.

### 11. Memory Policy Can Make or Break the Hybrid Pipeline

#### 11.1 Intel's USM guidance shows that host/GPU cooperative compute can easily fall into page-fault and migration traps

- **Source:** Intel `Performance Impact of USM and Buffers`; Intel `Unified Shared Memory Allocations`
- **Key idea:** Intel documents that shared allocations can incur page faults,
  migration overhead, and blocking behavior depending on access pattern and API
  usage.
- **Relevance to EdgeLM:** This is one of the biggest hidden risks in a custom
  CPU+iGPU pipeline on an integrated GPU.
- **Estimated impact:** Very High.
- **Implementation complexity:** Medium.
- **Details:** Intel explicitly discusses cases where:
  - CPU and GPU cooperate on a shared allocation,
  - shared allocations are touched from both sides,
  - host access or migration behavior becomes visible,
  - and device allocations with explicit movement still outperform the more
    convenient shared-memory approach.

  That means the first EdgeLM hybrid backend should treat memory policy as an
  experimental dimension, not an implementation detail.

#### 11.2 Shared allocations are attractive for persistent state, but explicit stage ownership is still safer than fine-grain shared mutation

- **Source:** Intel USM guidance; Section 17 Level Zero memory findings
- **Key idea:** The right lesson from unified/shared memory is not "let both
  processors touch everything whenever they want." The safer lesson is "use
  shared memory where it simplifies coarse ownership without creating fine-grain
  ping-pong."
- **Relevance to EdgeLM:** This strongly shapes the hot-loop memory design.
- **Estimated impact:** Very High.
- **Implementation complexity:** Medium.
- **Details:** A defensible first policy is:
  - persistent buffers may be shared or shared-system backed,
  - but each stage has clear ownership while it is active,
  - and ownership changes happen only at explicit synchronization points.

  That is much safer than allowing CPU and GPU to repeatedly poke the same data
  structure during a hot decode loop.

### 12. Time-Slicing Should Be the Default Experimental Baseline, and Controlled Overlap Should Be the Challenge Phase

#### 12.1 The repo's own warning about shared DDR4 is consistent with the strongest external evidence

- **Source:** `AGENTS.md`; `implementation-plan.md`; Section 16; OpenVINO
  heterogeneous execution guidance
- **Key idea:** The project already warns not to overlap CPU and iGPU
  memory-intensive work on this machine, and the external evidence does not
  soften that warning.
- **Relevance to EdgeLM:** This validates a conservative baseline strategy.
- **Estimated impact:** Very High.
- **Implementation complexity:** Low.
- **Details:** There is now a consistent line across sources:
  - the project says shared DDR4 is the bottleneck,
  - Intel graphics use shared system memory,
  - OpenVINO warns against CPU+iGPU pipeline parallelism on shared host memory,
  - and Intel's overlap guidance does not promise free wins from concurrency.

  Taken together, this makes **time-sliced stage execution** the right baseline
  experiment rather than an overly cautious fallback.

#### 12.2 Controlled overlap should be tested only when the resource vectors are demonstrably complementary

- **Source:** Intel overlap guidance; HeteGen overlap model; inference from
  EdgeLM operator mix
- **Key idea:** Overlap is most plausible when one side is compute- or
  control-heavy while the other side is not simultaneously saturating DRAM.
- **Relevance to EdgeLM:** This gives the project a rational condition for
  trying overlap instead of a blanket rule.
- **Estimated impact:** High.
- **Implementation complexity:** Medium.
- **Details:** On this system, promising overlap candidates are narrower than the
  roadmap implies:
  - host-side submission/control while iGPU computes,
  - CPU pre/post operations that are small and cache-resident,
  - perhaps weight prefetch orchestration on helper cores.

  Risky overlap candidates are:
  - CPU ternary matmul streaming weights from DDR4
  - at the same time as
  - iGPU attention streaming K/V and activations from the same DDR4

  That combination should be assumed unsafe until measurement proves otherwise.

### 13. The Best Hybrid Scheduler Will Be Phase-Aware and Sequence-Length-Aware, Not Static

#### 13.1 Research on CPU/GPU inference scheduling argues against one fixed assignment policy

- **Source:** "The Best of Many Worlds: Scheduling Machine Learning Inference on
  CPU-GPU Integrated Architectures"; LaLaRAND; Intel device-balancing guidance
- **Key idea:** Adaptive device selection and fine-grained scheduling outperform
  one-size-fits-all placement policies because the best device depends on the
  workload, optimization target, and system state.
- **Relevance to EdgeLM:** The hybrid scheduler should be policy-based, not
  hard-coded forever around one static split.
- **Estimated impact:** High.
- **Implementation complexity:** Medium.
- **Details:** Even though these sources are not EdgeLM-specific, they all point
  in the same direction:
  - device choice depends on workload size,
  - resource availability changes over time,
  - and per-layer or per-substage decisions can beat monolithic assignment.

  For EdgeLM, that implies at least three scheduler dimensions:
  - `phase`: prefill vs decode
  - `sequence length`: short vs long contexts
  - `mode`: CPU-only, sequential offload, time-sliced hybrid, or limited overlap

#### 13.2 Phase-aware policy switching is likely more important than a universal "hybrid on/off" flag

- **Source:** adaptive scheduling literature plus EdgeLM workload structure;
  inference from sources
- **Key idea:** The most plausible winning scheduler is one that switches
  strategy by phase and context length.
- **Relevance to EdgeLM:** This is a direct input to the future runtime design.
- **Estimated impact:** Very High.
- **Implementation complexity:** Medium.
- **Details:** A practical first policy table might look like:

  - **Short-context decode:** CPU-only or sequential iGPU attention offload only
    if launch overhead is small
  - **Long-context decode:** consider iGPU attention stage, but still with
    conservative overlap
  - **Large-prompt prefill:** the best place to test chunked CPU+iGPU pipelining
    and more aggressive stage overlap

### 14. The First Hybrid Pipeline Should Be Designed as a Measurement Harness as Much as a Runtime

#### 14.1 HeteGen's measured communication-aware methodology is a good model for experimentation

- **Source:** HeteGen, arXiv:2403.01164
- **Key idea:** HeteGen repeatedly models communication, overlap, pinned-memory
  effects, and split ratios instead of assuming them.
- **Relevance to EdgeLM:** This is exactly the research posture the project
  needs for its hybrid pipeline paper contribution.
- **Estimated impact:** High.
- **Implementation complexity:** Medium.
- **Details:** The important lesson is methodological:
  - treat communication as measurable,
  - treat overlap as measurable,
  - treat memory policy as measurable,
  - and treat split ratio as measurable.

#### 14.2 The benchmark matrix for Section 18 should compare pipeline shapes, not just "CPU-only vs hybrid"

- **Source:** all sources above; inference from project goals
- **Key idea:** A good experiment set must compare multiple hybrid designs, not
  just one hand-picked hybrid against a CPU baseline.
- **Relevance to EdgeLM:** This determines whether the section produces real
  engineering guidance or just anecdotes.
- **Estimated impact:** Very High.
- **Implementation complexity:** Low.
- **Details:** The minimum benchmark matrix should include:
  1. `CPU-only baseline`
  2. `iGPU attention stage, sequential`
  3. `iGPU attention stage, time-sliced`
  4. `iGPU attention stage, controlled overlap`
  5. `prefill chunk pipeline, if implemented`
  6. `same-operator split`, only as a late experiment

## Techniques Comparison Matrix

| Technique | Source | Impact | Complexity | Already in Deep Dive? |
|-----------|--------|--------|------------|-----------------------|
| Use coarse stage offload instead of same-operator CPU+iGPU co-processing as the default | OpenVINO + HeteGen + Intel docs + Damschen et al. | Very High | Medium | Partially |
| Separate prefill and decode hybrid policies | workload-shape inference + Intel/OpenVINO guidance | Very High | Medium | No |
| Correct the roadmap's layer-overlap sketch for single-stream decode dependencies | transformer-dataflow inference | Very High | Low | No |
| Use one coarse handoff inside attention (`CPU QKV -> iGPU score/softmax/value -> CPU O`) | Intel host/device movement guidance + prior sections | Very High | Medium | No |
| Keep iGPU submission/wait on helper cores, not hot P-cores | Intel non-blocking submit guidance + Section 15 | High | Low | Partially |
| Treat ping-pong buffers as stage-handoff tools, not bandwidth cures | Intel async guidance + Section 16 | High | Medium | Partially |
| Make time-sliced execution the initial hybrid baseline | OpenVINO shared-memory warning + project bandwidth model | Very High | Low | Partially |
| Permit controlled overlap only for complementary CPU/GPU resource vectors | Intel overlap guidance + HeteGen | High | Medium | No |
| Make memory policy an experiment dimension (shared vs explicit device/host ownership) | Intel USM/buffer guidance | Very High | Medium | No |
| Use policy switching by phase and sequence length | adaptive scheduling literature + project workload structure | High | Medium | No |

## Recommendations for EdgeLM

1. Treat the hybrid pipeline as **two separate problems**:
   - `prefill`
   - `decode`
   They should not be forced to use one identical scheduling strategy.

2. Retire the current roadmap diagram as a literal decode execution model. For
   single-stream autoregressive decode, assume stricter serialization than that
   sketch implies.

3. Make the first hybrid decode boundary:
   - **CPU:** RMSNorm, Q/K/V projections, RoPE if convenient, O projection, FFN
   - **iGPU:** score matmul, mask, softmax, value accumulation

4. Keep CPU/iGPU handoffs to **one coarse boundary per attention block** in the
   first implementation.

5. Make **time-sliced stage execution** the baseline hybrid experiment, not the
   fallback:
   - CPU-heavy stage
   - then iGPU-heavy stage
   - then CPU-heavy stage

6. Add overlap only in this order:
   - host submission/control overlap
   - lightweight CPU pre/post overlap
   - carefully staged transfer overlap
   - only then consider heavier simultaneous compute overlap

7. Put the iGPU runtime on helper cores:
   - one E-core submission thread
   - optional E-core wait/completion thread
   - keep hot P-cores for CPU kernels

8. Use ping-pong buffers only around the chosen attention handoff and scratch
   workspace ownership. Do not proliferate them across the whole pipeline.

9. Benchmark memory policy explicitly:
   - shared allocation
   - device allocation + explicit movement
   - host allocation + explicit movement
   and keep the fastest measured policy.

10. Make the scheduler mode-aware from the beginning. At minimum, support:
    - CPU-only
    - sequential iGPU attention stage
    - time-sliced hybrid
    - limited-overlap hybrid

11. Gate more aggressive pipeline parallelism behind larger-granularity work:
    - prompt prefill chunks
    - multiple simultaneous requests
    - or speculative decoding
    Do not assume base decode will naturally expose enough independent work.

12. Collect the following measurements for every hybrid run:
    - CPU stage time
    - iGPU kernel time
    - handoff / copy / migration time
    - P-core stall time waiting on GPU
    - GPU idle time waiting on CPU
    - total token latency
    - tokens/sec
    - memory footprint

13. The most likely winning long-term policy is:
    - **CPU owns all ternary weight-heavy linear work**
    - **iGPU owns the tight score/softmax/value attention subpipeline**
    - **scheduler chooses sequential, time-sliced, or lightly overlapped execution by phase**

## References

1. Intel, "Using Multiple Heterogeneous Devices," oneAPI GPU Optimization Guide.  
   https://www.intel.com/content/www/us/en/docs/oneapi/optimization-guide-gpu/2025-0/using-multiple-heterogeneous-devices.html

2. Intel, "Asynchronous and Overlapping Data Transfers Between Host and Device," oneAPI GPU Optimization Guide.  
   https://www.intel.com/content/www/us/en/docs/oneapi/optimization-guide-gpu/2023-0/asynchronous-and-overlapping-data-transfers.html

3. Intel, "Performance Impact of USM and Buffers," oneAPI GPU Optimization Guide.  
   https://www.intel.com/content/www/us/en/docs/oneapi/optimization-guide-gpu/2025-2/performance-impact-of-usm-and-buffers.html

4. Intel, "Unified Shared Memory Allocations," oneAPI GPU Optimization Guide.  
   https://www.intel.com/content/www/us/en/docs/oneapi/optimization-guide-gpu/2025-2/unified-shared-memory-allocations.html

5. OpenVINO, "Heterogeneous Execution."  
   https://docs.openvino.ai/2024/openvino-workflow/running-inference/inference-devices-and-modes/hetero-execution.html

6. OpenVINO, "AsyncInferRequest."  
   https://docs.openvino.ai/2024/api/c_cpp_api/classov_1_1_async_infer_request.html

7. Kaixuan Zhou et al., "HeteGen: Heterogeneous Parallel Inference for Large Language Models on Resource-Constrained Devices," arXiv:2403.01164, 2024.  
   https://arxiv.org/abs/2403.01164

8. Matthew Damschen et al., "Co-Scheduling on Fused CPU-GPU Architectures With Shared Last Level Caches," IEEE TCAD, 2018.  
   https://arcb.csc.ncsu.edu/~mueller/ftp/pub/mueller/papers/cases18-2.pdf

9. Taesik Gong et al., "LaLaRAND: Flexible Layer-by-Layer CPU/GPU Scheduling for Real-Time DNN Tasks," IEEE IoTJ, 2021.  
   https://pure.skku.edu/en/publications/lalarand-flexible-layer-by-layer-cpugpu-scheduling-for-real-time-/

10. Yaozong Zheng et al., "The Best of Many Worlds: Scheduling Machine Learning Inference on CPU-GPU Integrated Architectures," ACM TECS, 2022.  
    https://zenodo.org/records/6410912

## Audit Addendum (2026-04-02)

- **The hybrid runtime needs a prompt-length gate.** EdgeLM should not assume
  that GPU offload is globally useful; it should switch policies only when the
  workload shape justifies the launch and bandwidth cost.
- **Per-token mode switches are worth planning for, even if v1 stays static.**
  Long-prefill then short-decode requests may want different execution policies
  within the same request lifecycle.
- **Back-pressure telemetry should be first-class.** If handoff queues, shared
  workspaces, or fence waits grow, the runtime should surface that explicitly so
  "GPU helped" versus "GPU stalled the CPU" is measurable.
