# Section 24: Benchmarking Methodology -- Extended Research

## Overview

Benchmarking is where EdgeLM either becomes a convincing systems project or a
collection of anecdotes.

By this stage of the research, the project already has strong hypotheses about:

- why the laptop is bandwidth-bound,
- why ternary weights are necessary,
- how the CPU and iGPU should be used,
- how speculative decoding might help,
- and how the decode pipeline should be structured.

Section 24 is where those hypotheses have to be tested in a way that survives
review.

That requires much more than reporting a single `tok/s` number.

For EdgeLM, a paper-grade benchmarking methodology has to answer at least five
different questions:

1. **Is the engine correct?**
2. **How fast is prompt prefill?**
3. **How fast is steady-state decode?**
4. **How much latency does a real user feel before seeing output?**
5. **How fair are the comparisons being made?**

The current implementation plan already contains a benchmark skeleton, but it is
still too coarse in a few important ways:

- it mixes system benchmarking with model-quality evaluation,
- it treats some non-iso comparisons as if they were direct baselines,
- it does not fully separate cold-start from warm steady-state behavior,
- and it does not yet define how correctness gates speed claims.

The deeper literature and tool ecosystem now make the right methodology much
clearer.

## What the Deep Dive Already Covers

`deep-dive.md` is still empty, but the repo already contains a meaningful local
starting point.

- `implementation-plan.md` already defines:
  - a test matrix by prompt/generation length,
  - prefill speed,
  - decode speed,
  - TTFT,
  - RAM, CPU, iGPU, and power as target metrics,
  - and a one-experiment-at-a-time protocol.
- The plan also already proposes:
  - `10` recorded runs,
  - `3` warmup runs,
  - and reporting summary statistics.
- Section 18 established that prefill and decode are different workloads and
  should be benchmarked separately.
- Section 19 established that speculative decoding must expose subsystem-level
  counters such as accepted prefix lengths and verifier/draft timings.
- Section 22 established that the sampler should be measured separately enough
  to understand whether it becomes a hidden per-token tax.
- Section 23 established that the decode pipeline itself should emit telemetry
  for tokenize time, prefill time, TTFT, decode time, stop reason, and related
  structural counters.

So the unresolved benchmarking questions are now more specific:

- Which metrics are primary versus secondary?
- How should EdgeLM separate correctness tests from performance tests?
- Which comparisons are truly apples-to-apples and which are only practical
  reference points?
- How should cold-start, warm-start, and streaming latency be reported?
- Should request-throughput metrics even matter for a single-request local
  engine?
- And what should the final experiment matrix look like so the paper can make
  strong but fair claims?

## New Findings

### 1. Benchmarking must separate correctness evaluation from performance evaluation

#### 1.1 LLMPerf explicitly splits load testing from correctness testing

- **Source:** Ray `LLMPerf` official repository
- **Key idea:** LLMPerf implements two different test classes:
  - a load test for performance
  - and a correctness test for correctness
- **Relevance to EdgeLM:** This is the right methodological split for this
  project too.
- **Estimated impact:** Very High.
- **Implementation complexity:** Low.
- **Details:** That split matters because a faster engine is not publishable if
  it silently changes outputs or breaks stop conditions. Conversely, a perfect
  correctness suite does not tell you where the time went.

#### 1.2 For EdgeLM, correctness should be a gate that every speed claim must pass

- **Source:** LLMPerf; `implementation-plan.md`; project synthesis
- **Key idea:** Performance numbers should only count once the corresponding
  configuration passes correctness checks.
- **Relevance to EdgeLM:** This is the biggest methodological correction to the
  current roadmap.
- **Estimated impact:** Very High.
- **Implementation complexity:** Medium.
- **Details:** EdgeLM should treat correctness as a prerequisite for performance
  reporting, not as a nice-to-have appendix. At minimum:
  - same prompt serialization
  - same tokenizer IDs
  - same stop behavior
  - same greedy outputs for iso-checkpoint comparisons

  should be verified before benchmark numbers are considered valid.

### 2. The benchmark stack should be hierarchical: microbenchmarks, component benchmarks, and end-to-end benchmarks

#### 2.1 A single end-to-end `tok/s` number is too coarse for tuning

- **Source:** Project architecture; Sections 18-23
- **Key idea:** The engine has too many moving parts for one aggregate number to
  explain regressions or gains.
- **Relevance to EdgeLM:** Without layered benchmarks, optimization work becomes
  guesswork.
- **Estimated impact:** Very High.
- **Implementation complexity:** Medium.
- **Details:** EdgeLM should maintain three layers:
  1. **microbenchmarks** for kernels and primitives
  2. **component benchmarks** for tokenizer, loader, sampler, KV cache,
     attention kernels, speculative verifier/draft, etc.
  3. **end-to-end request benchmarks** for actual prompt-to-output generation

#### 2.2 The paper should primarily cite end-to-end results, but they should be backed by lower-level evidence

- **Source:** Project paper goals; systems-paper norms
- **Key idea:** Reviewers ultimately care about end-to-end value, but they also
  need to see why the system behaves that way.
- **Relevance to EdgeLM:** This directly shapes result presentation.
- **Estimated impact:** High.
- **Implementation complexity:** Low.

### 3. Prefill and decode must be benchmarked separately

#### 3.1 Modern LLM-serving work treats prefill and decode as distinct regimes for good reason

- **Source:** SARATHI
- **Key idea:** SARATHI explicitly states that LLM inference consists of two
  distinct phases: prefill and decode, and that they exhibit different hardware
  behavior.
- **Relevance to EdgeLM:** This validates the phase split already established in
  local research.
- **Estimated impact:** Very High.
- **Implementation complexity:** Low.

#### 3.2 EdgeLM should never collapse them into one ambiguous throughput number

- **Source:** Sections 18 and 23; project benchmark plan
- **Key idea:** A single combined token/s number hides too much.
- **Relevance to EdgeLM:** This is one of the main reporting rules for the final
  paper.
- **Estimated impact:** Very High.
- **Implementation complexity:** Low.
- **Details:** Every serious result table should distinguish at least:
  - prefill tokens/s
  - decode tokens/s
  - TTFT
  - end-to-end request latency

  without forcing readers to reverse-engineer which phase dominated.

### 4. TTFT, ITL/TPOT, and end-to-end latency need precise definitions

#### 4.1 Official benchmarking tools now define a common latency vocabulary

- **Source:** NVIDIA GenAI-Perf; NVIDIA NIM benchmarking docs
- **Key idea:** Official LLM benchmarking docs define:
  - **TTFT** as time from request submission to first token received
  - **ITL / TPOT** as average time between consecutive output tokens
  - **end-to-end latency** as time from request submission to final token
  - **output token throughput** as total output tokens over benchmark duration
- **Relevance to EdgeLM:** Adopting compatible definitions makes the results much
  easier to interpret.
- **Estimated impact:** Very High.
- **Implementation complexity:** Low.

#### 4.2 For a local engine, TTFT should be reported both externally and internally

- **Source:** NVIDIA NIM metrics docs; Section 23 telemetry design
- **Key idea:** NVIDIA's TTFT definition includes request queuing, prefill, and
  first-token delivery effects; on a local engine, there is little or no network
  but there is still a difference between user-visible TTFT and internal model
  timing.
- **Relevance to EdgeLM:** This is a strong methodological refinement for the
  paper.
- **Estimated impact:** High.
- **Implementation complexity:** Medium.
- **Details:** EdgeLM should report:
  - **external TTFT**: prompt submission to first printable output
  - **internal first-token latency**: end of tokenization/setup to first sampled
    token
  - **prefill latency** separately

  That decomposition will make local results much more informative than raw TTFT
  alone.

#### 4.3 Time-to-second-token is also useful when TTFT and steady-state diverge sharply

- **Source:** GenAI-Perf
- **Key idea:** GenAI-Perf now reports time to second token separately from TTFT
  and ITL.
- **Relevance to EdgeLM:** This is useful for identifying whether only the first
  decode step is unusual or the entire decode loop is slow.
- **Estimated impact:** Medium.
- **Implementation complexity:** Low.

### 5. Cold-start and warm-start must be reported separately

#### 5.1 The first user-visible run and the steady-state run are different experiments

- **Source:** Project loader/repacking design; benchmarking best practice
- **Key idea:** A local engine has meaningful one-time costs that should not be
  silently mixed into steady-state decode results.
- **Relevance to EdgeLM:** This matters a lot because model loading and repacking
  are major parts of the runtime story.
- **Estimated impact:** Very High.
- **Implementation complexity:** Low.
- **Details:** On EdgeLM, cold-start can include:
  - process startup
  - model file open / mapping
  - metadata parse
  - tokenizer asset load
  - weight repacking
  - large-page allocation success/fallback
  - first driver/runtime initialization for any iGPU path

  Those costs are real and publishable, but they are not the same as warm
  token-generation speed.

#### 5.2 Warm benchmarking should exclude one-time initialization and focus on the stable decode loop

- **Source:** Project architecture synthesis
- **Key idea:** The main systems claim of EdgeLM is about sustained local
  inference, not "Python import plus startup plus first run."
- **Relevance to EdgeLM:** This is the right way to make optimization progress
  legible.
- **Estimated impact:** Very High.
- **Implementation complexity:** Low.
- **Details:** The benchmark suite should therefore always report at least:
  - **cold load-to-first-token**
  - **warm TTFT**
  - **warm prefill tok/s**
  - **warm decode tok/s**

  with the cold path and warm path clearly labeled as different experiments.

### 6. The benchmark matrix should be workload-based, not just "one short prompt and one long prompt"

#### 6.1 Prompt-heavy and decode-heavy workloads expose different failure modes

- **Source:** SARATHI; project prefill/decode split; NVIDIA benchmark matrices
- **Key idea:** Varying input length and output length changes which part of the
  runtime dominates.
- **Relevance to EdgeLM:** This is necessary to understand whether a change
  helps prompt ingestion, steady-state generation, or only one narrow corner.
- **Estimated impact:** Very High.
- **Implementation complexity:** Low.
- **Details:** EdgeLM should structure workloads into at least four families:
  - **prompt-heavy**: long input, short generation
  - **decode-heavy**: short input, long generation
  - **balanced**: medium input, medium generation
  - **stress / long-context**: near-window prompts and/or long outputs

  This is better than thinking in terms of arbitrary sequence pairs.

#### 6.2 The current implementation-plan matrix is a good starting point, but it should become a named benchmark suite

- **Source:** `implementation-plan.md`
- **Key idea:** The existing matrix already contains the right raw shapes.
- **Relevance to EdgeLM:** The next step is to make those shapes a stable,
  reusable benchmark suite with fixed prompt corpora and reporting rules.
- **Estimated impact:** High.
- **Implementation complexity:** Low.
- **Details:** A good paper-facing suite would keep the current token lengths but
  group them into named classes such as:
  - `QA_SHORT`
  - `CHAT_BALANCED`
  - `PROMPT_HEAVY`
  - `GEN_HEAVY`
  - `CTX_STRESS`

  so future regressions compare the same workload families, not loosely similar
  ad hoc prompts.

#### 6.3 Prompt content diversity still matters even in a single-user local benchmark

- **Source:** Tokenizer research; practical benchmarking inference
- **Key idea:** Different prompt styles stress different subsystems.
- **Relevance to EdgeLM:** The paper should not accidentally overfit to one
  friendly prompt shape.
- **Estimated impact:** High.
- **Implementation complexity:** Medium.
- **Details:** At minimum, the fixed benchmark corpus should contain:
  - chat/instruction prompts
  - factual Q&A
  - code-heavy text
  - long structured text

  because tokenizer behavior, repetition dynamics, and stop behavior can vary
  meaningfully across these prompt types.

### 7. EdgeLM needs two comparison categories: iso-checkpoint fairness and practical target-relevance

#### 7.1 Apples-to-apples system claims require iso-checkpoint comparisons

- **Source:** Benchmarking methodology synthesis; LLMPerf correctness emphasis
- **Key idea:** If the paper claims one engine is faster than another for "the
  same model," then the comparison must hold model, tokenizer, prompt template,
  and decode policy fixed.
- **Relevance to EdgeLM:** This is the cleanest path to defensible system claims.
- **Estimated impact:** Very High.
- **Implementation complexity:** Medium.
- **Details:** For an iso-checkpoint comparison, EdgeLM should keep constant:
  - model checkpoint / weights
  - tokenizer and vocabulary
  - prompt serialization
  - stop conditions
  - generation policy
  - prompt set

  Any change in those variables means the comparison is no longer a pure engine
  comparison.

#### 7.2 Practical laptop baselines are still valuable, but they must be labeled differently

- **Source:** Project roadmap; baseline-model reality
- **Key idea:** Comparing EdgeLM's ternary BitNet target to a `Q4_K_M`
  llama.cpp baseline on the same hardware is useful, but it is not a controlled
  same-model comparison.
- **Relevance to EdgeLM:** This distinction protects the paper from overstating
  claims.
- **Estimated impact:** Very High.
- **Implementation complexity:** Low.
- **Details:** These comparisons should be framed as:
  - **practical local alternatives**
  - **same-laptop reference points**
  - or **target-relevance baselines**

  rather than "our engine is X times faster than baseline" with no qualification.

#### 7.3 Output quality should therefore be reported in a separate comparison layer

- **Source:** Methodology synthesis
- **Key idea:** If the models or quantizations differ, speed and output-quality
  comparisons should not be collapsed into one table.
- **Relevance to EdgeLM:** This is especially important for BitNet-versus-Q4
  comparisons.
- **Estimated impact:** High.
- **Implementation complexity:** Low.
- **Details:** The clean paper structure is:
  - **Table A:** same-model/same-decode system comparisons
  - **Table B:** practical local alternatives with clearly non-iso quality
    implications
  - **Table C:** separate output-quality or task-quality results

### 8. Token accounting and decoding policy must be standardized or the metrics drift

#### 8.1 Prompt and generation lengths should always be defined in tokens, not characters

- **Source:** Tokenizer research; standard LLM benchmarking practice
- **Key idea:** Token counts are the only unit that maps directly to transformer
  work.
- **Relevance to EdgeLM:** This prevents misleading prompt-length comparisons.
- **Estimated impact:** Very High.
- **Implementation complexity:** Low.
- **Details:** Character count is not a stable proxy because tokenization depends
  on:
  - language,
  - whitespace,
  - punctuation,
  - code formatting,
  - and model-specific tokenization rules.

#### 8.2 Same-checkpoint comparisons require the same tokenizer and chat template

- **Source:** Section 20 tokenizer research
- **Key idea:** Even when the weights are nominally the same, different prompt
  serialization or tokenization rules change both token counts and outputs.
- **Relevance to EdgeLM:** This is a major fairness constraint for Llama 3 / BitNet.
- **Estimated impact:** Very High.
- **Implementation complexity:** Medium.
- **Details:** A fair system comparison therefore requires:
  - identical BOS/EOS handling
  - identical special-token policy
  - identical chat template serialization
  - identical stop-string policy

  If those differ, TTFT and tok/s are no longer directly comparable.

#### 8.3 Greedy decode should be the primary benchmark mode unless the experiment is specifically about sampling

- **Source:** Sections 19 and 22; performance-measurement synthesis
- **Key idea:** Greedy decode removes sampling randomness and minimizes
  distribution-side variability.
- **Relevance to EdgeLM:** This should be the default path for correctness and
  baseline performance measurement.
- **Estimated impact:** Very High.
- **Implementation complexity:** Low.
- **Details:** Stochastic benchmarks still matter, but they should be a separate
  layer with fixed seed and fixed parameters so they do not contaminate the main
  throughput story.

### 9. Environment control matters more on a laptop than on a server

#### 9.1 Benchmark logs should capture hardware, firmware, driver, and OS details explicitly

- **Source:** Project hardware-specific philosophy
- **Key idea:** On a consumer laptop, changes in BIOS, GPU driver, Windows
  version, or microcode can materially alter performance.
- **Relevance to EdgeLM:** The project is explicitly targeting one machine.
- **Estimated impact:** Very High.
- **Implementation complexity:** Low.
- **Details:** Every serious result file should record:
  - CPU model
  - RAM capacity and speed
  - Windows build
  - compiler and flags
  - GPU driver version
  - BIOS version if known
  - power plan
  - thread configuration
  - large-page enabled/disabled

#### 9.2 Thermal and power conditioning must be part of the methodology, not an afterthought

- **Source:** Laptop benchmarking synthesis
- **Key idea:** Sustained laptop performance can drift due to thermals and power
  management even when the software is identical.
- **Relevance to EdgeLM:** This directly affects claims about steady-state tok/s.
- **Estimated impact:** Very High.
- **Implementation complexity:** Low.
- **Details:** The benchmark harness should document and control:
  - plugged-in state
  - selected power profile
  - ambient-ish repeatability where possible
  - cooldown between long runs when necessary
  - and whether runs are performed from a thermally cold or already-warm system

#### 9.3 AB comparison runs should be interleaved when possible

- **Source:** Statistical methodology synthesis
- **Key idea:** If configuration `A` is always tested before `B`, thermal drift
  can masquerade as a software win or loss.
- **Relevance to EdgeLM:** This matters for careful laptop benchmarking.
- **Estimated impact:** High.
- **Implementation complexity:** Low.
- **Details:** Better patterns include:
  - alternating `A, B, A, B`
  - randomized run order
  - or paired same-prompt comparisons under the same session conditions

### 10. Summary statistics should include percentiles and paired analysis, not just averages

#### 10.1 Mean and standard deviation are not enough for latency-sensitive interactive systems

- **Source:** LLMPerf v2 release notes; interactive latency reasoning
- **Key idea:** Current benchmarking tools now emphasize quantile reporting
  because tail latency matters.
- **Relevance to EdgeLM:** A local engine is judged by responsiveness, not just
  average throughput.
- **Estimated impact:** High.
- **Implementation complexity:** Low.
- **Details:** EdgeLM should report at least:
  - mean
  - median / p50
  - p95
  - p99 when enough samples exist
  - min / max
  - standard deviation

  especially for TTFT and per-request latency.

#### 10.2 Paired prompt-level comparisons are better than independent aggregate runs

- **Source:** Methodology synthesis
- **Key idea:** Running the same prompt set under two configurations produces a
  much cleaner comparison than comparing separate prompt batches.
- **Relevance to EdgeLM:** This is one of the best ways to isolate small gains.
- **Estimated impact:** High.
- **Implementation complexity:** Medium.
- **Details:** A good benchmark result should be able to answer:
  - for the same prompt,
  - on the same machine session,
  - how much did TTFT, prefill speed, decode speed, and memory change?

  That is far stronger than comparing unrelated aggregate means.

### 11. Correctness for a systems paper should be layered, not reduced to "looks okay"

#### 11.1 Tokenizer and prompt-serialization correctness should be tested before model-speed benchmarking

- **Source:** Section 20; correctness-methodology synthesis
- **Key idea:** If prompt bytes become different token IDs, everything downstream
  is already incomparable.
- **Relevance to EdgeLM:** This is the lowest-level correctness gate.
- **Estimated impact:** Very High.
- **Implementation complexity:** Low.
- **Details:** Before any major benchmark campaign, EdgeLM should verify:
  - tokenization round-trips on golden cases
  - special-token insertion behavior
  - chat template serialization
  - stop-string / detokenization corner cases

#### 11.2 Same-checkpoint greedy runs should be compared at the token-sequence level

- **Source:** LLMPerf correctness split; systems benchmarking synthesis
- **Key idea:** For a deterministic decode policy, the strongest correctness
  check is exact token agreement with a reference implementation.
- **Relevance to EdgeLM:** This is the benchmark gate that matters most.
- **Estimated impact:** Very High.
- **Implementation complexity:** Medium.
- **Details:** For iso-checkpoint greedy tests, EdgeLM should verify:
  - same output token IDs
  - same stop reason
  - same generated length
  - and, where practical, same intermediate logits within tolerance on selected
    checkpoints or micro-tests

#### 11.3 Stochastic correctness should be framed as reproducibility and policy consistency

- **Source:** Section 22 sampler design
- **Key idea:** For non-greedy decode, exact token matching to another engine is
  less informative unless RNG mapping is also aligned.
- **Relevance to EdgeLM:** This matters for future sampling and speculative tests.
- **Estimated impact:** High.
- **Implementation complexity:** Medium.
- **Details:** The right correctness goals here are:
  - seeded reproducibility within EdgeLM
  - consistent processor/warper ordering
  - correct stop behavior
  - and, where needed, distribution-level agreement against a reference builder

### 12. The benchmark harness should emit a rich machine-readable result schema

#### 12.1 Benchmarks are much more useful when every run is self-describing

- **Source:** `implementation-plan.md`; project telemetry design
- **Key idea:** A `tok/s` printout in a terminal is not enough for paper work.
- **Relevance to EdgeLM:** The benchmark system should generate publishable data,
  not just human memory.
- **Estimated impact:** Very High.
- **Implementation complexity:** Medium.

#### 12.2 The result schema should capture config, environment, metrics, and correctness together

- **Source:** Sections 18-23; benchmarking synthesis
- **Key idea:** The benchmark record needs enough context that results remain
  interpretable months later.
- **Relevance to EdgeLM:** This will save time during the paper phase.
- **Estimated impact:** Very High.
- **Implementation complexity:** Medium.
- **Details:** At minimum, each benchmark JSON should include:
  - **run metadata**: timestamp, git commit, benchmark name
  - **environment**: CPU, RAM, OS, compiler, driver, power plan
  - **model config**: checkpoint, quantization, tokenizer, context length
  - **runtime config**: threads, affinity, large pages, speculative on/off,
    iGPU on/off
  - **decode config**: greedy or sampler settings, seed, stop limits
  - **workload description**: prompt corpus ID, prompt tokens, target output
    tokens
  - **metrics**: cold load, warm TTFT, prefill tok/s, decode tok/s, total
    latency, peak RAM
  - **counters**: generated tokens, stop reason, speculative accepted tokens,
    offload counters
  - **correctness status**: pass/fail and failure reason

### 13. Power and energy methodology needs a correction: do not build around Intel Power Gadget

#### 13.1 Intel Power Gadget is no longer a dependable primary measurement path

- **Source:** Intel advisory `INTEL-SA-01037`
- **Key idea:** Intel issued a product discontinuation notice in 2024 related to
  Intel Power Gadget.
- **Relevance to EdgeLM:** This directly affects the current roadmap, which
  still mentions Intel Power Gadget.
- **Estimated impact:** High.
- **Implementation complexity:** Low.
- **Details:** That means EdgeLM should not base its long-term paper methodology
  around a tool Intel itself has discontinued.

#### 13.2 Wall-power measurement is the best end-to-end energy method if available

- **Source:** Systems-measurement synthesis
- **Key idea:** A wall meter measures the actual laptop energy cost seen by the
  user, including CPU, iGPU, memory, and platform overhead.
- **Relevance to EdgeLM:** This is the cleanest whole-system energy metric for a
  consumer laptop.
- **Estimated impact:** High.
- **Implementation complexity:** Medium.
- **Details:** If a wall meter is available, the paper should prefer:
  - joules per request
  - joules per generated token
  - and average system power during benchmark phases

  over package-only telemetry as the primary energy story.

#### 13.3 Intel PCM is a better optional software-side fallback than Power Gadget

- **Source:** Intel PCM official repository / release notes
- **Key idea:** Intel PCM remains actively maintained and exposes performance and
  energy metrics on supported Intel systems.
- **Relevance to EdgeLM:** This is the best current candidate for optional
  software-side telemetry.
- **Estimated impact:** Medium.
- **Implementation complexity:** Medium.
- **Details:** PCM should be treated as:
  - useful supplemental telemetry
  - helpful for CPU-package trends
  - but not a substitute for whole-laptop wall energy if the paper wants
    end-to-end claims

### 14. Throughput and goodput are secondary for v1, but the paper should still use the concepts correctly

#### 14.1 Single-stream latency is the primary benchmark target for the first EdgeLM engine

- **Source:** Project single-user focus; Section 23
- **Key idea:** EdgeLM's first real use case is one local user interacting with
  one prompt at a time.
- **Relevance to EdgeLM:** That makes single-request latency and steady-state
  decode the main performance story.
- **Estimated impact:** Very High.
- **Implementation complexity:** Low.

#### 14.2 Request throughput should be deferred until there is a real multi-request mode

- **Source:** Orca; vLLM; DistServe; project scope
- **Key idea:** Server papers care deeply about requests/second because they
  serve many concurrent users. EdgeLM v1 does not.
- **Relevance to EdgeLM:** This prevents the project from optimizing the wrong
  objective too early.
- **Estimated impact:** High.
- **Implementation complexity:** Low.

#### 14.3 Goodput is still a useful later concept: performance under latency constraints

- **Source:** DistServe
- **Key idea:** DistServe frames serving performance in terms of the maximum rate
  that stays within TTFT/TPOT constraints.
- **Relevance to EdgeLM:** This is not a v1 primary metric, but it is a useful
  conceptual bridge if the project later explores queued or multi-session modes.
- **Estimated impact:** Medium.
- **Implementation complexity:** Medium.
- **Details:** For now, the main takeaway is methodological:
  - raw throughput alone is insufficient
  - latency constraints matter
  - and any future concurrency story should be reported under explicit latency
    bounds, not only peak tokens/s

## Comparative Fit for EdgeLM

| Benchmark framing | Strength | Weakness | EdgeLM verdict |
| --- | --- | --- | --- |
| Single-number `tok/s` only | Very simple | Hides prefill, TTFT, correctness, and cold-start behavior | Not acceptable for paper-grade evaluation |
| Phase-separated single-request benchmarking | Matches local UX and hardware reality | More metrics to collect and explain | Best primary methodology |
| Iso-checkpoint system comparison | Strongest fairness for engine claims | Requires same tokenizer/template/decode policy | Mandatory for core speed claims |
| Practical same-laptop baseline comparison | Useful for real-world context | Not apples-to-apples quality or model comparison | Include, but label clearly |
| Multi-request throughput benchmarking | Useful for future server mode | Misaligned with current single-user target | Secondary / later only |
| End-to-end energy benchmarking with wall power | Best user-relevant efficiency story | Needs extra measurement hardware | Preferred if available |

## Recommendations for EdgeLM

### 1. Make correctness a hard prerequisite for published speed numbers

- No major benchmark table should include a configuration that fails:
  - tokenizer correctness
  - prompt-serialization correctness
  - deterministic greedy output checks
  - or stop-condition correctness

### 2. Standardize a three-layer benchmark stack

- **Microbenchmarks:** kernels, memory copy/repack, sampler core operations.
- **Component benchmarks:** tokenizer, loader, KV cache, prefill path, sampler,
  speculative controller, iGPU offload path.
- **End-to-end benchmarks:** prompt-to-output request measurements.

### 3. Treat warm phase-separated single-request benchmarks as the main paper baseline

- Primary metrics:
  - warm TTFT
  - warm prefill tok/s
  - warm decode tok/s
  - end-to-end request latency
  - peak RAM
- Secondary metrics:
  - cold-start latency
  - energy
  - CPU/iGPU utilization

### 4. Split comparisons into two explicitly labeled families

- **Iso-checkpoint / iso-decode:** the real engine-comparison tables.
- **Practical same-laptop alternatives:** useful reference points, but clearly
  non-iso.

### 5. Make greedy decode the default benchmark policy

- Use stochastic decoding only in dedicated experiments.
- Fix seed and parameter set when stochastic paths are benchmarked.
- Never mix greedy and stochastic numbers in one unlabeled performance table.

### 6. Promote environment metadata to first-class benchmark data

- Record:
  - power mode
  - thread placement
  - compiler flags
  - large-page success/fallback
  - GPU driver version
  - and OS build

### 7. Replace Intel Power Gadget in the roadmap

- Prefer external wall-power measurement for paper-quality energy claims.
- Use Intel PCM as an optional supplemental telemetry source.
- Do not design the benchmark pipeline around a discontinued tool.

### 8. Emit benchmark data in structured JSON from day one

- Store each run under a stable schema.
- Keep benchmark prompts versioned.
- Attach correctness status and failure reason to every run record.

## Suggested EdgeLM Result Schema

```json
{
  "run_id": "exp_0247",
  "timestamp_utc": "2026-04-02T16:20:00Z",
  "git_commit": "abc1234",
  "benchmark_name": "CHAT_BALANCED",
  "comparison_family": "iso_checkpoint",
  "environment": {
    "cpu": "Intel Core i7-12700H",
    "ram_gb": 16,
    "os": "Windows 11",
    "compiler": "MSVC 19.x /O2 /arch:AVX2",
    "gpu_driver": "intel-driver-version",
    "power_plan": "Best performance"
  },
  "model": {
    "name": "bitnet-b1.58-2b-4t",
    "format": "gguf",
    "weight_encoding": "ternary_2bit_packed",
    "tokenizer": "llama3_tiktoken_128256"
  },
  "runtime": {
    "threads_p": 6,
    "threads_e": 8,
    "hyperthreads_used": false,
    "large_pages": true,
    "igpu_enabled": false,
    "speculative_enabled": false
  },
  "decode": {
    "mode": "greedy",
    "temperature": 0.0,
    "top_p": 1.0,
    "top_k": 0,
    "seed": 1234,
    "max_new_tokens": 256
  },
  "workload": {
    "prompt_corpus_id": "chat_v1",
    "prompt_id": "chat_013",
    "prompt_tokens": 256,
    "target_output_tokens": 256
  },
  "correctness": {
    "status": "pass",
    "reference_engine": "golden_trace_v5",
    "notes": ""
  },
  "metrics": {
    "cold_start_ms": 0,
    "warm_ttft_ms": 0,
    "time_to_second_token_ms": 0,
    "prefill_tok_s": 0.0,
    "decode_tok_s": 0.0,
    "request_latency_ms": 0,
    "peak_ram_mb": 0
  },
  "counters": {
    "generated_tokens": 0,
    "stop_reason": "STOP_MAX_TOKENS",
    "spec_accept_tokens": 0,
    "igpu_dispatches": 0
  }
}
```

## Suggested Experiment Sequence

1. Build correctness-first golden tests for tokenizer, prompt formatting, and
   greedy output traces.
2. Implement microbenchmarks for kernels, KV reads/writes, and sampler cost.
3. Implement component benchmarks for loader, tokenizer, prefill, decode loop,
   and optional speculative/offload subpaths.
4. Run warm single-request greedy benchmarks across the fixed workload suite.
5. Run cold-start benchmarks separately.
6. Add ablations one at a time:
   - large pages on/off
   - affinity policy variants
   - prefetch on/off
   - iGPU path on/off
   - speculative on/off
7. Add practical same-laptop baseline comparisons with careful labeling.
8. Add energy measurements after the timing harness is stable.

## Bottom Line

Benchmarking is where EdgeLM's research either becomes a systems result or
stays a collection of promising engineering notes.

The central methodological conclusions are:

- correctness and performance must be separated, but correctness must gate
  performance claims,
- prefill, decode, TTFT, and end-to-end latency must be reported separately,
- cold-start and warm steady-state are different experiments,
- fair engine claims require iso-checkpoint comparisons,
- practical same-laptop baselines are still useful but must be labeled as
  non-iso,
- environment control and thermal discipline matter disproportionately on a
  consumer laptop,
- and the benchmark harness should emit rich structured data from the start.

For EdgeLM specifically, the right primary methodology is:

- single-request,
- phase-separated,
- greedy-first,
- correctness-gated,
- workload-suite based,
- and explicit about cold versus warm behavior.

That will let the later paper say something much stronger than "we got a good
token/s number once." It will let the project show:

- what part of the pipeline improved,
- under which workload,
- on what exact machine state,
- with what fairness guarantees,
- and at what latency, memory, and energy cost.

That is the difference between a benchmark dashboard and a publishable systems
evaluation.

## Sources

- NVIDIA NIM LLM benchmarking docs:
  `https://docs.nvidia.com/nim/benchmarking/llm/latest/performance.html`
- NVIDIA GenAI-Perf / Triton benchmarking docs:
  `https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/perf_analyzer/genai-perf/README.html`
- NVIDIA technical blog, *LLM Inference Benchmarking Guide: NVIDIA GenAI-Perf
  and NIM* (2025):
  `https://developer.nvidia.com/blog/llm-performance-benchmarking-measuring-nvidia-nim-performance-with-genai-perf/`
- Ray `LLMPerf` official repository:
  `https://github.com/ray-project/llmperf`
- Yu et al., *Orca: A Distributed Serving System for Transformer-Based
  Generative Models* (OSDI 2022):
  `https://www.usenix.org/conference/osdi22/presentation/yu`
- Kwon et al., *vLLM: Easy, Fast, and Cheap LLM Serving with PagedAttention*
  (SOSP 2023 / arXiv 2023): `https://arxiv.org/abs/2309.06180`
- Agrawal et al., *SARATHI: Efficient LLM Inference by Piggybacking Decodes with
  Chunked Prefills* (arXiv 2023): `https://arxiv.org/abs/2308.16369`
- Zhong et al., *DistServe: Disaggregating Prefill and Decoding for
  Goodput-optimized Large Language Model Serving* (OSDI 2024 / arXiv 2024):
  `https://arxiv.org/abs/2401.09670`
- Intel advisory `INTEL-SA-01037`:
  `https://www.intel.com/content/www/us/en/security-center/advisory/intel-sa-01037.html`
- Intel PCM official repository:
  `https://github.com/intel/pcm`

## Audit Addendum (2026-04-02)

- **Benchmark results should record git cleanliness and compile-time flags.**
  Small local build differences are too easy to lose otherwise.
- **Negative results deserve a home in the final paper package.** For EdgeLM,
  "the iGPU path hurt under these bandwidth conditions" is publishable systems
  evidence, not wasted effort.
- **Hardware-counter traces should be sampled for representative runs, not every
  run.** This keeps the harness lightweight while still giving the paper a few
  deep explanatory profiles.
