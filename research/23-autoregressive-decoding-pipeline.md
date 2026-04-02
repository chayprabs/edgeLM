# Section 23: Autoregressive Decoding Pipeline -- Extended Research

## Overview

This section is where the previous twenty-two sections have to stop being
isolated optimizations and become one coherent machine.

By this point, EdgeLM already has research guidance for:

- CPU topology and scheduling,
- ternary kernels,
- memory hierarchy,
- GGUF/model loading,
- KV cache design,
- iGPU offload boundaries,
- speculative decoding,
- tokenizer behavior,
- memory management,
- and token sampling.

Section 23 asks the integrative question:

- what is the **actual end-to-end autoregressive decoding pipeline** that ties
  all of that together for a single request on the target laptop?

This is not a trivial bookkeeping exercise.

The decode pipeline determines:

- where prefill ends and decode begins,
- what state is attached to a request,
- which thread owns orchestration,
- how model compute and host-side work interleave,
- how tokens become streamed text,
- when stopping decisions are evaluated,
- how speculative decoding would plug in later,
- and which system ideas from large-scale LLM serving do or do not belong in a
  single-user local engine.

The literature helps here, but not by giving EdgeLM a server design to copy.
Modern serving papers mostly study:

- continuous batching,
- distributed scheduling,
- prefill/decode disaggregation,
- and GPU utilization.

Those are important, but EdgeLM is not a datacenter serving system. It is a
custom Windows C runtime for a single laptop where:

- single-request latency matters more than fleet throughput,
- RAM is tight,
- shared DDR4 bandwidth dominates,
- and the orchestration thread must stay simple and predictable.

So the correct goal for this section is not "port Orca or vLLM to a laptop." It
is:

- extract the durable pipeline lessons from modern inference systems,
- combine them with the hardware-specific EdgeLM findings,
- and define the minimal, explicit decode state machine that this engine should
  actually implement.

## What the Deep Dive Already Covers

`deep-dive.md` is still empty, but earlier project materials already imply a lot
about the intended pipeline.

- `implementation-plan.md` already distinguishes prompt prefill from token-by-token
  decode in the benchmark matrix and performance goals.
- The implementation plan places tokenization, sampling, and orchestration on
  the main control thread and reserves helper E-cores for prefetch, iGPU
  submission, and possible speculative drafting.
- Section 18 established that prefill and decode are different pipeline
  problems, and that single-stream decode has much stricter dependency chains
  than optimistic layer-overlap diagrams imply.
- Section 19 established that speculative decoding plugs into the decode loop as
  an optional outer controller around the same target-model distribution.
- Section 20 established that prompt serialization, special-token handling, and
  streaming detokenization are part of the runtime contract.
- Section 21 established that the decode loop must not perform OS allocation or
  page-state churn.
- Section 22 established that the sampler should be factored into deterministic
  distribution-building plus stochastic draw, with a greedy fast path and
  sparse repetition state.

So the unresolved questions entering this section are:

- What should the request object actually contain?
- Where is the clean boundary between request setup, prefill, decode, and
  teardown?
- Which work should happen once per request versus once per token?
- How should stop conditions be ordered?
- How should streaming output and cancellation fit into the loop?
- Which server-style features matter now, and which belong only to a later
  multi-request mode?
- And what instrumentation should the pipeline expose so the paper can explain
  its own results?

## New Findings

### 1. Autoregressive inference is fundamentally a multi-iteration stateful workload

#### 1.1 Orca's framing generalizes beyond servers: generative inference is not a one-shot request

- **Source:** Yu et al., *Orca: A Distributed Serving System for
  Transformer-Based Generative Models* (OSDI 2022)
- **Key idea:** Orca emphasizes that autoregressive generation requires running
  the model multiple times for one inference request, with each iteration
  producing a single output token.
- **Relevance to EdgeLM:** This is the most important conceptual starting point
  for the whole section.
- **Estimated impact:** Very High.
- **Implementation complexity:** Low.
- **Details:** Even though Orca studies large-scale serving, its first systems
  observation applies directly to EdgeLM: a generative request is not "infer
  once, return result." It is a stateful sequence of dependent iterations.

#### 1.2 The pipeline therefore has to be built around explicit request state

- **Source:** Orca; vLLM; project architecture synthesis
- **Key idea:** Because each request evolves over many iterations, the runtime
  needs a first-class request object rather than a pile of ad hoc local
  variables.
- **Relevance to EdgeLM:** This is the direct software-design consequence of the
  multi-iteration workload shape.
- **Estimated impact:** Very High.
- **Implementation complexity:** Medium.
- **Details:** The request object must carry everything that persists across
  decode steps:
  - prompt tokens
  - generated tokens
  - current position
  - KV-cache ownership
  - sampler state
  - detokenization buffer
  - stop status
  - timing counters

  That state is the decode pipeline.

### 2. Prefill and decode must be treated as different execution phases, not just different sequence lengths

#### 2.1 Modern serving work explicitly distinguishes prefill and decode because their hardware behavior is different

- **Source:** Agrawal et al., *SARATHI: Efficient LLM Inference by Piggybacking
  Decodes with Chunked Prefills* (arXiv 2023)
- **Key idea:** SARATHI states that LLM inference has two distinct phases:
  prefill, which processes the input prompt, and decode, which generates output
  tokens autoregressively. It also emphasizes that the phases behave very
  differently in terms of utilization and scheduling.
- **Relevance to EdgeLM:** This strongly validates the split already emerging in
  earlier local research.
- **Estimated impact:** Very High.
- **Implementation complexity:** Low.

#### 2.2 For EdgeLM, the phase split should be visible in APIs, metrics, and scheduling policy

- **Source:** SARATHI; Section 18; project benchmark plan
- **Key idea:** Prefill and decode should not share one opaque "run inference"
  function with no phase visibility.
- **Relevance to EdgeLM:** The engine needs explicit phase boundaries for
  performance work and for paper-quality measurement.
- **Estimated impact:** Very High.
- **Implementation complexity:** Medium.
- **Details:** At minimum, the runtime should report:
  - prompt token count
  - prefill latency / tok-s
  - time to first token
  - decode tok-s
  - generated token count

  If those are not first-class, the pipeline is too opaque.

### 3. The base EdgeLM pipeline should optimize for single-request clarity, not server-style scheduling sophistication

#### 3.1 Server papers motivate many ideas that EdgeLM should consciously defer

- **Source:** Orca; vLLM; SARATHI
- **Key idea:** These systems are driven by multi-request batching,
  heterogeneous request lengths, and shared-cluster throughput.
- **Relevance to EdgeLM:** That makes them valuable reference points, but not
  literal implementation templates.
- **Estimated impact:** Very High.
- **Implementation complexity:** Low.
- **Details:** Features such as:
  - iteration-level scheduling across many requests,
  - continuous batching,
  - prompt-sharing across requests,
  - distributed load balancing,
  - and decode-maximal multi-request batch formation

  are intellectually important, but they are not the right baseline for a
  single-user local engine.

#### 3.2 The right v1 policy is "single active request, explicit state machine"

- **Source:** Project hardware constraints; Section 18; Section 21
- **Key idea:** The EdgeLM laptop target has enough constraints that the first
  decode pipeline should be maximally explicit and minimally concurrent.
- **Relevance to EdgeLM:** This is the central design recommendation of the
  section.
- **Estimated impact:** Very High.
- **Implementation complexity:** Medium.
- **Details:** That means:
  - one request owns the decode loop,
  - one orchestration thread advances it,
  - compute workers service that request,
  - and any later multi-request mode should be an extension, not the default.

### 4. The decode pipeline should be represented as an explicit request state machine

#### 4.1 A request lifecycle with named states is simpler and more debuggable than nested function flow

- **Source:** Project architecture synthesis; OpenVINO async-request docs
- **Key idea:** A request should move through named states such as:
  - `NEW`
  - `TOKENIZED`
  - `PREFILLING`
  - `READY_TO_DECODE`
  - `DECODING`
  - `STREAMING`
  - `FINISHED`
  - `CANCELLED`
  - `ERROR`
- **Relevance to EdgeLM:** This makes the control flow visible and testable.
- **Estimated impact:** Very High.
- **Implementation complexity:** Medium.
- **Details:** It is much easier to reason about stop conditions, cleanup, and
  cancellation when the pipeline has explicit state transitions instead of one
  giant monolithic "generate()" function.

#### 4.2 The state machine should expose both normal completion and exceptional exits

- **Source:** OpenVINO `InferRequest` docs; project runtime needs
- **Key idea:** Modern inference APIs explicitly distinguish normal completion,
  callback-based continuation, and cancellation/error handling.
- **Relevance to EdgeLM:** Even CPU-only bring-up benefits from explicit stop
  reasons.
- **Estimated impact:** High.
- **Implementation complexity:** Medium.
- **Details:** A request should terminate with one explicit reason such as:
  - `STOP_EOS`
  - `STOP_MAX_TOKENS`
  - `STOP_STOP_STRING`
  - `STOP_CANCELLED`
  - `STOP_CONTEXT_LIMIT`
  - `STOP_ERROR`

  That will matter a lot for benchmark logs and CLI UX.

### 5. The request object should own all mutable decode state in one place

#### 5.1 The request should be the home for prompt, generation, sampling, and streaming state

- **Source:** Orca multi-iteration framing; Sections 20-22; project synthesis
- **Key idea:** Everything that changes per request should live under one
  request-owned structure.
- **Relevance to EdgeLM:** This avoids pipeline logic leaking across modules.
- **Estimated impact:** Very High.
- **Implementation complexity:** Medium.
- **Details:** A good request object should include at least:
  - token buffers for prompt and generated output
  - prompt length and generated length
  - current decode position / absolute token index
  - KV-cache cursor ownership
  - sampler configuration and sampler state
  - repetition-tracking state
  - detokenization byte buffer
  - stop strings / stopping config
  - cancellation flag
  - timing / telemetry counters

#### 5.2 Request state should also record derived prompt metadata

- **Source:** Section 20 tokenizer research; project chat-format needs
- **Key idea:** The pipeline needs more than raw token IDs. It also needs to know
  how those tokens were formed and what semantics apply to them.
- **Relevance to EdgeLM:** This matters for repetition scope, stop handling, and
  correct streaming.
- **Estimated impact:** High.
- **Implementation complexity:** Medium.
- **Details:** Useful derived metadata includes:
  - prompt token count
  - number of assistant-visible generated tokens
  - whether the request used chat serialization
  - which special tokens should be suppressed or allowed during generation

### 6. Prefill should be modeled as a dedicated pipeline stage with its own policy and outputs

#### 6.1 Prefill is not "decode on a longer sequence"; it initializes the request's runtime state

- **Source:** SARATHI; Section 18
- **Key idea:** Prefill processes the prompt tokens, populates the KV cache, and
  produces the logits from which the first generated token will be chosen.
- **Relevance to EdgeLM:** This is the natural seam between request setup and the
  per-token loop.
- **Estimated impact:** Very High.
- **Implementation complexity:** Medium.

#### 6.2 Prefill should output a fully decode-ready request

- **Source:** Section 18; Sections 20-22; project pipeline synthesis
- **Key idea:** After prefill finishes, the request should already have:
  - a populated KV cache,
  - the last logits needed for first-token sampling,
  - initialized sampler state,
  - initialized stop-tracking state,
  - and streaming buffers ready.
- **Relevance to EdgeLM:** This keeps the decode loop tight and predictable.
- **Estimated impact:** Very High.
- **Implementation complexity:** Medium.
- **Details:** In other words, the transition from `PREFILLING` to
  `READY_TO_DECODE` should be a strong invariant:
  - no remaining setup work,
  - no lazy allocations,
  - no missing scratch state.

#### 6.3 Chunked prefill is a later optimization, not the baseline local path

- **Source:** SARATHI; Section 18
- **Key idea:** Chunked prefill is valuable in GPU-serving systems and may matter
  for EdgeLM later, especially for long prompts or future iGPU experiments, but
  it is not necessary for the first correct decode pipeline.
- **Relevance to EdgeLM:** This keeps the v1 pipeline simple.
- **Estimated impact:** Medium.
- **Implementation complexity:** Medium.
- **Details:** The first local path should be:
  - one request
  - one prompt serialization
  - one prefill phase
  - then decode

  Chunking can be added later for long prompts or hybrid experiments.

### 7. The decode loop itself should be an explicit ordered sequence of steps

#### 7.1 A single decode step has a fixed semantic order

- **Source:** Sections 18-22 synthesized
- **Key idea:** One decode iteration should do the following in order:
  1. prepare current model inputs and positions
  2. run the model forward for the current token step
  3. update request-visible logits/output state
  4. build the final next-token distribution
  5. select the next token
  6. append token and update repetition / stop state
  7. stream any newly printable text
  8. evaluate whether the request should terminate or continue
- **Relevance to EdgeLM:** This is the core of the section.
- **Estimated impact:** Very High.
- **Implementation complexity:** Medium.

#### 7.2 Stop checks should occur both before and after sampling, depending on the condition

- **Source:** Section 22 processor design; runtime synthesis
- **Key idea:** Not all stopping rules belong at the same point in the loop.
- **Relevance to EdgeLM:** Getting this wrong creates subtle bugs.
- **Estimated impact:** High.
- **Implementation complexity:** Medium.
- **Details:** Examples:
  - `max_new_tokens`: check before continuing another step
  - EOS suppression until minimum length: apply before sampling
  - EOS stop reason: evaluate after token selection
  - stop-string match: evaluate after detokenization buffer update

  So "stop check" is not one line of code. It is a small ordered policy.

### 8. Streaming output should be a pipeline stage, not an afterthought

#### 8.1 Token emission and text emission are not the same thing

- **Source:** Section 20 tokenizer research
- **Key idea:** Byte-level tokenization means a sampled token does not
  necessarily correspond to immediately printable user text.
- **Relevance to EdgeLM:** This affects the decode loop directly.
- **Estimated impact:** Very High.
- **Implementation complexity:** Medium.

#### 8.2 The decode pipeline should stream only complete UTF-8 text fragments

- **Source:** Section 20
- **Key idea:** The request should maintain a detokenization buffer and only
  flush complete UTF-8 sequences to the output sink.
- **Relevance to EdgeLM:** This is the correct bridge between token loop and user
  experience.
- **Estimated impact:** High.
- **Implementation complexity:** Medium.
- **Details:** This means:
  - sample token
  - append its byte fragment
  - decode what is safely printable
  - emit only that portion
  - retain unfinished bytes in the request buffer

  The pipeline should never assume that one token equals one printable fragment.

### 9. Stop conditions should be centralized and explicit

#### 9.1 EdgeLM needs more than EOS and max token count

- **Source:** Sections 20 and 22; runtime synthesis
- **Key idea:** Real local generation needs a broader stop policy than a toy
  demo.
- **Relevance to EdgeLM:** This is necessary for a usable engine.
- **Estimated impact:** High.
- **Implementation complexity:** Medium.
- **Details:** The stop system should at least support:
  - EOS token
  - max new tokens
  - minimum new tokens before EOS allowed
  - stop strings
  - user cancellation
  - context/window exhaustion
  - internal error termination

#### 9.2 Stop strings should be evaluated on decoded text, not token IDs alone

- **Source:** Section 20 tokenizer semantics; runtime inference
- **Key idea:** Because text boundaries do not align cleanly with byte-level
  tokens, stop-string matching should operate on the decoded output buffer.
- **Relevance to EdgeLM:** This avoids incorrect or missed matches.
- **Estimated impact:** High.
- **Implementation complexity:** Medium.
- **Details:** A token-ID-only stop matcher can be made to work for specific
  token sequences, but a text-buffer matcher is the more correct default for a
  general-purpose local engine.

### 10. The memory manager and decode pipeline should be tightly coupled by contract

#### 10.1 The token loop must never allocate from the OS

- **Source:** Section 21 memory-management research
- **Key idea:** The decode loop should not call `VirtualAlloc`, `VirtualFree`,
  `malloc`, or `free`.
- **Relevance to EdgeLM:** This is one of the strongest system invariants in the
  entire codebase.
- **Estimated impact:** Very High.
- **Implementation complexity:** Low.

#### 10.2 Per-request scratch should come from resettable arenas

- **Source:** Section 21; project request-state synthesis
- **Key idea:** Any transient buffers used for:
  - sampler candidate storage
  - streaming detokenization staging
  - temporary logits bookkeeping
  - or stop-string matching scratch

  should come from request-local or decode-local arenas.
- **Relevance to EdgeLM:** This keeps hot-path memory behavior deterministic.
- **Estimated impact:** High.
- **Implementation complexity:** Medium.

### 11. The orchestration thread should own the state machine, not the heavy compute

#### 11.1 The project's existing core split already implies the right control pattern

- **Source:** `implementation-plan.md`; Sections 15, 20, and 22
- **Key idea:** Tokenization, sampling, orchestration, and output assembly belong
  on the control side of the engine, while hot model compute belongs on the
  worker side.
- **Relevance to EdgeLM:** This is the correct division of labor for the
  `6P + 8E` Alder Lake target.
- **Estimated impact:** Very High.
- **Implementation complexity:** Medium.
- **Details:** The simplest pattern is:
  - orchestration thread owns request state machine
  - P-core workers run the heavy forward pass
  - helper threads prefetch or service optional side work

  This matches the earlier scheduling research.

#### 11.2 The decode loop should therefore look synchronous from the request's point of view even when some internals are async

- **Source:** OpenVINO `InferRequest`; Section 18
- **Key idea:** Host/device async execution can exist internally, but the request
  state machine should still advance through clear semantic stages like submit,
  wait, and postprocess.
- **Relevance to EdgeLM:** This is especially important if the iGPU path is
  enabled later.
- **Estimated impact:** High.
- **Implementation complexity:** Medium.
- **Details:** OpenVINO explicitly separates:
  - `start_async`
  - `wait` / `wait_for`
  - `set_callback`
  - `cancel`

  EdgeLM does not need to copy the exact API, but it should borrow the same
  semantic split for any future async offload path.

### 12. The base EdgeLM pipeline should support cancellation and interruption explicitly

#### 12.1 Cancellation is part of a real decode pipeline, not a UI extra

- **Source:** OpenVINO `InferRequest::cancel`; runtime synthesis
- **Key idea:** Modern inference runtimes expose cancellation because iterative
  inference is long-lived enough that interruption matters.
- **Relevance to EdgeLM:** This matters for CLI use, desktop UI integration, and
  future server mode.
- **Estimated impact:** High.
- **Implementation complexity:** Medium.
- **Details:** A simple cooperative design is enough:
  - UI or caller sets a cancel flag
  - orchestration thread checks it between major stages
  - request exits with `STOP_CANCELLED`

#### 12.2 Cancellation boundaries should be stage-based, not arbitrary

- **Source:** Runtime inference; Section 18 async-stage reasoning
- **Key idea:** The safest places to check cancellation are at clear stage
  boundaries:
  - before prefill
  - after prefill
  - after each decode step
  - before streaming flush
- **Relevance to EdgeLM:** This avoids corrupting request state mid-update.
- **Estimated impact:** Medium.
- **Implementation complexity:** Low.

### 13. Server-style batching ideas should be treated as later extensions, not as base-pipeline requirements

#### 13.1 Continuous batching and iteration scheduling are multi-request optimizations

- **Source:** Orca; vLLM
- **Key idea:** Systems like Orca and vLLM are primarily concerned with serving
  many requests efficiently by managing request interleaving and KV memory at
  scale.
- **Relevance to EdgeLM:** This is not the first problem EdgeLM needs to solve.
- **Estimated impact:** Very High.
- **Implementation complexity:** Low.

#### 13.2 The durable lessons to borrow are phase separation and explicit state, not the whole scheduler

- **Source:** Orca; SARATHI; vLLM; project synthesis
- **Key idea:** The right EdgeLM takeaway is:
  - model prefill and decode separately
  - keep request state explicit
  - and leave multi-request scheduling for later
- **Relevance to EdgeLM:** This keeps the local engine honest about its actual
  use case.
- **Estimated impact:** Very High.
- **Implementation complexity:** Low.

#### 13.3 Prompt sharing and cross-request prefix reuse are also later-stage features

- **Source:** Preble; project single-request focus
- **Key idea:** Prompt sharing is valuable in distributed serving systems because
  long prefixes often repeat across requests, but this is not the first-order
  concern in a single active local session.
- **Relevance to EdgeLM:** Good future research direction, wrong baseline.
- **Estimated impact:** Medium.
- **Implementation complexity:** Medium.

### 14. Context-window policy should be an explicit pipeline choice

#### 14.1 Request admission should check context feasibility before decode starts

- **Source:** Project benchmark plan; Section 11 KV-cache research
- **Key idea:** The pipeline should reject or adapt oversized requests early,
  before entering a doomed decode loop.
- **Relevance to EdgeLM:** This is basic robustness.
- **Estimated impact:** High.
- **Implementation complexity:** Medium.
- **Details:** Before or during prefill, the request should determine whether:
  - prompt length fits,
  - prompt + requested generation fits,
  - or a special long-context policy is needed.

#### 14.2 Sliding-window or KV-eviction behavior must not be silent

- **Source:** Section 11; pipeline synthesis
- **Key idea:** If EdgeLM later supports context shifting, sliding windows, or
  KV eviction, that policy must be surfaced explicitly in the request state and
  telemetry.
- **Relevance to EdgeLM:** Hidden context truncation would make results hard to
  interpret.
- **Estimated impact:** High.
- **Implementation complexity:** Medium.

### 15. The pipeline should emit rich telemetry, because this section feeds the paper directly

#### 15.1 The decode pipeline should be self-measuring

- **Source:** Project benchmarking goals; Sections 18, 19, and 22
- **Key idea:** The runtime should record enough stage timing to explain where
  time went.
- **Relevance to EdgeLM:** Without this, performance tuning becomes guesswork.
- **Estimated impact:** Very High.
- **Implementation complexity:** Medium.
- **Details:** Minimum per-request telemetry should include:
  - tokenize time
  - prefill time
  - TTFT
  - average decode step time
  - average sampler time
  - total generated tokens
  - stop reason
  - and any speculative/offload substage times if enabled

#### 15.2 The pipeline should also expose structural counters, not just timers

- **Source:** Project synthesis
- **Key idea:** Counts are often more explanatory than pure durations.
- **Relevance to EdgeLM:** This matters for later paper figures and debugging.
- **Estimated impact:** High.
- **Implementation complexity:** Medium.
- **Details:** Useful counters include:
  - prompt token count
  - new token count
  - context occupancy
  - streamed byte count
  - number of stop-string checks
  - number of cancellation polls
  - speculative accepted tokens if enabled

## Comparative Fit for EdgeLM

| Pipeline shape | Strength | Weakness | EdgeLM verdict |
| --- | --- | --- | --- |
| Single-request explicit state machine | Simple, debuggable, best local latency baseline | No multi-request throughput tricks | Best first implementation |
| Single-request async host/device pipeline | Preserves clear request semantics while allowing future iGPU overlap | More moving parts | Good later extension |
| Continuous-batching server pipeline | Excellent for many concurrent requests | Overkill and complexity-heavy for local laptop v1 | Not a baseline target |
| Chunked-prefill local pipeline | Useful for long prompts and later hybrid experiments | More control complexity | Later optimization only |

## Recommendations for EdgeLM

### 1. Build the runtime around a first-class request object

- The request owns:
  - tokens
  - KV/cache position
  - sampler state
  - detokenization buffer
  - stop state
  - telemetry

### 2. Make prefill and decode separate pipeline stages in code and metrics

- Distinguish them in APIs.
- Distinguish them in timers.
- Distinguish them in benchmark reports.

### 3. Keep the base pipeline single-request and explicit

- No server-style scheduler in v1.
- No continuous batching requirement.
- No hidden concurrency model that obscures request state.

### 4. Keep the decode loop stage-ordered and allocation-free

- No OS allocation in the token loop.
- No implicit stop logic scattered across modules.
- No token-to-text shortcut that bypasses byte-buffered detokenization.

### 5. Design async and speculative extensions to plug into the same state machine

- Async offload should insert submit/wait/postprocess boundaries.
- Speculative decoding should wrap the same distribution-building and stop
  semantics, not fork them.

## Suggested EdgeLM Implementation Shape

### Request state

```c
typedef enum {
    REQ_NEW = 0,
    REQ_TOKENIZED,
    REQ_PREFILLING,
    REQ_READY,
    REQ_DECODING,
    REQ_FINISHED,
    REQ_CANCELLED,
    REQ_ERROR
} req_phase_t;

typedef enum {
    STOP_NONE = 0,
    STOP_EOS,
    STOP_MAX_TOKENS,
    STOP_STOP_STRING,
    STOP_CONTEXT_LIMIT,
    STOP_CANCELLED,
    STOP_ERROR
} stop_reason_t;

typedef struct {
    req_phase_t    phase;
    stop_reason_t  stop_reason;

    int32_t       *prompt_ids;
    uint32_t       prompt_len;
    int32_t       *output_ids;
    uint32_t       output_len;
    uint32_t       abs_pos;

    void          *kv_handle;
    sampler_cfg_t  sampler_cfg;
    sampler_state_t sampler_state;

    uint8_t       *stream_buf;
    uint32_t       stream_len;

    volatile int   cancel_flag;

    uint64_t       t_tokenize_ns;
    uint64_t       t_prefill_ns;
    uint64_t       t_decode_ns;
    uint64_t       t_sample_ns;
} edge_request_t;
```

### Phase flow

1. Serialize prompt / chat messages.
2. Tokenize and validate context feasibility.
3. Initialize request state and attach KV region/scratch arenas.
4. Run prefill.
5. Sample first token from final prefill logits.
6. Enter decode loop.
7. Stream printable text fragments.
8. Stop on explicit reason and finalize telemetry.

### Decode step

1. Check cancellation / max-token guard.
2. Prepare current token input and absolute position.
3. Run forward pass with existing KV state.
4. Build final next-token distribution through processors + warpers.
5. Select token.
6. Update repetition / stop / output state.
7. Detokenize and stream printable bytes.
8. Transition to next step or terminal phase.

## Suggested Experiment Sequence

1. Implement synchronous single-request pipeline with greedy decode.
2. Add explicit telemetry for tokenize, prefill, decode, and TTFT.
3. Add stochastic sampling through the factored sampler.
4. Add explicit stop reasons and stop-string handling.
5. Add cancellation checks and clean teardown.
6. Only then integrate speculative decoding.
7. Only after that, explore async/iGPU boundaries or chunked prefill modes.

## Bottom Line

The right autoregressive pipeline for EdgeLM is not a miniature datacenter
serving stack. It is a small, explicit request state machine that:

- separates prefill from decode,
- keeps all mutable decode state in one request object,
- advances through a fixed per-token stage order,
- streams text through a byte-aware detokenization stage,
- centralizes stop and cancellation logic,
- and records enough telemetry to explain performance scientifically.

Modern serving papers are still extremely useful here, but mostly as guidance on
what matters:

- autoregressive inference is multi-iteration,
- prefill and decode are different phases,
- async boundaries should be explicit,
- and hidden scheduler complexity is easy to add and hard to justify.

For EdgeLM, the best first implementation is therefore:

- single request,
- explicit phase machine,
- no hot-path allocation,
- no server-style scheduler,
- and clean extension points for speculative decoding and optional async offload.

That gives the project a decode pipeline that is simple enough to implement,
strong enough to benchmark, and structured enough to become the backbone of the
later paper.

## Sources

- Yu et al., *Orca: A Distributed Serving System for Transformer-Based
  Generative Models* (OSDI 2022):
  `https://www.usenix.org/conference/osdi22/presentation/yu`
- Kwon et al., *vLLM: Easy, Fast, and Cheap LLM Serving with PagedAttention*
  (SOSP 2023 / arXiv 2023): `https://arxiv.org/abs/2309.06180`
- Agrawal et al., *SARATHI: Efficient LLM Inference by Piggybacking Decodes with
  Chunked Prefills* (arXiv 2023): `https://arxiv.org/abs/2308.16369`
- Srivatsa et al., *Preble: Efficient Distributed Prompt Scheduling for LLM
  Serving* (arXiv 2024): `https://arxiv.org/abs/2407.00023`
- OpenVINO `InferRequest` documentation:
  `https://docs.openvino.ai/2024/openvino-workflow/running-inference/integrate-openvino-with-your-application/inference-request.html`

## Audit Addendum (2026-04-02)

- **Session snapshot and resume hooks are worth preserving in the request model.**
  Even if they are not implemented immediately, the state machine should avoid
  making future session persistence impossible.
- **Streaming UX needs one more layer of policy.** Partial stop-string matches,
  token-healing edge cases, and buffered UTF-8 fragments should be explicit in
  the output contract.
- **Timeout/watchdog semantics should exist even for local mode.** A request that
  hangs on a backend fence or rare runtime error needs a bounded recovery path.
