# Section 19: Speculative Decoding -- Extended Research

## Overview

Speculative decoding is one of the few optimization families that can plausibly
move EdgeLM from "good custom decode engine" into "paper-worthy throughput
regime" on this laptop. That is why it appears in both `implementation-plan.md`
and `AGENTS.md` as a late-phase multiplier rather than a micro-optimization.

But this section needs more than the usual draft-and-verify summary. On EdgeLM,
speculative decoding sits at the intersection of:

- a bandwidth-bound ternary target model,
- asymmetric P-core and E-core compute,
- shared DDR4 memory,
- a strict RAM budget,
- and a no-framework custom C runtime.

That means the important question is not "does speculative decoding exist?" It
is:

- which speculative family fits this engine,
- which one preserves exactness,
- whether the draft model should be external or self-derived,
- whether fixed `K` is good enough,
- how draft/verify should be scheduled on `6P + 8E`,
- and whether the implementation plan's current "`0.5B` draft on E-cores"
  assumption is even close to optimal.

The literature is now deep enough to answer those questions in a much more
concrete way.

## What the Deep Dive Already Covers

`deep-dive.md` remains empty, so the operative project baseline comes from
`implementation-plan.md`, `AGENTS.md`, and the earlier EdgeLM research notes.

- The implementation plan already assumes speculative decoding can provide a
  `1.5-2.5x` effective throughput multiplier.
- The current project sketch is:
  - a tiny draft model, suggested as roughly `0.5B` ternary
  - running on E-cores
  - generating `K` candidate tokens
  - with the main model verifying those tokens in one batch-style forward pass
  - and accepting `N <= K` tokens before resuming drafting
- The implementation plan also reserves `E-core 10-13` as the likely draft
  model execution pool.
- Section 01 established that Gracemont E-cores can run AVX2, but 256-bit work
  is internally split, so they are good helper compute but not equal to
  Golden Cove P-cores for hot SIMD-heavy kernels.
- Section 15 established that P/E handoff overhead is real and that helper work
  should use coarse synchronization rather than tight ping-pong.
- Section 16 established that the laptop is fundamentally constrained by shared
  DDR4 bandwidth, not just raw FLOPS.
- Section 18 established that on this hardware, time-slicing or very carefully
  constrained overlap is usually safer than assuming free CPU+iGPU concurrency.

So the unresolved questions entering this section are:

- Is classical external speculative decoding still the best first path?
- Should EdgeLM consider self-speculative decoding instead to avoid an extra
  draft checkpoint?
- Is a fixed lookahead `K` too naive?
- How small does the draft need to be before it actually helps on E-cores?
- And should draft and verify overlap at all on this shared-memory machine?

## New Findings

### 1. Classical speculative decoding is exact, but only if EdgeLM implements the full acceptance rule

#### 1.1 The original algorithm is not a heuristic shortcut

- **Source:** Leviathan et al., *Fast Inference from Transformers via
  Speculative Decoding* (ICML 2023)
- **Key idea:** The original algorithm is lossless: it preserves the target
  model's output distribution while using a smaller model to propose `gamma`
  draft tokens and the target model to verify them in parallel.
- **Relevance to EdgeLM:** This matters for a systems paper. EdgeLM can present
  speculative decoding as exact acceleration rather than as an approximation
  trick.
- **Estimated impact:** Very High.
- **Implementation complexity:** Medium.
- **Details:** Leviathan et al. show that each target-model verification pass
  produces at least one token and up to `gamma + 1` tokens, while keeping the
  same distribution as the target model alone. That makes speculative decoding
  unusually attractive for EdgeLM because it increases concurrency without
  requiring architecture changes or retraining in the classical external-draft
  form.

#### 1.2 Exactness for sampling is more subtle than "accept matching tokens"

- **Source:** Leviathan et al. (ICML 2023)
- **Key idea:** Exact speculative decoding supports general sampling, but the
  non-greedy path requires rejection logic and an adjusted target distribution,
  not just an equality check on token IDs.
- **Relevance to EdgeLM:** The implementation plan's current summary is
  directionally right but incomplete. If EdgeLM eventually supports `top-k`,
  `top-p`, temperature, and repetition penalties while preserving exactness, the
  speculative path must use the full `p/q` acceptance rule.
- **Estimated impact:** High.
- **Implementation complexity:** Medium to High.
- **Details:** This is easy to miss. A minimal greedy benchmark can start with
  the simpler accept-or-reject path, but the real production sampler in
  `sampler.c` and `speculative.c` will eventually need a mathematically correct
  "correction token" path for non-greedy decoding. That is a direct implication
  of the original paper, not an optional refinement.

### 2. Speculative decoding attacks the same bottleneck that dominates EdgeLM

#### 2.1 The literature frames autoregressive decode as a bandwidth problem, not only a FLOP problem

- **Source:** Leviathan et al. (ICML 2023); Yan et al., *Decoding Speculative
  Decoding* (arXiv 2024 / NAACL 2025)
- **Key idea:** Both papers explicitly describe autoregressive decode as
  hardware-inefficient, with large-model inference often bottlenecked by memory
  bandwidth and communication rather than pure arithmetic throughput.
- **Relevance to EdgeLM:** This matches the project's central thesis almost
  exactly. EdgeLM is not trying to accelerate a compute-bound server GPU; it is
  trying to survive a shared-DDR4 memory wall on a consumer laptop.
- **Estimated impact:** Very High.
- **Implementation complexity:** Low.
- **Details:** Leviathan et al. note that large-model inference is often limited
  by memory bandwidth and communication, leaving additional compute resources
  available. Yan et al. then restate the modern decode problem as bandwidth
  bound and hardware inefficient. That is the strongest possible conceptual
  alignment with EdgeLM's own design premise.

#### 2.2 Verification is effectively a prefill-like step, which is why speculative decoding helps

- **Source:** Yan et al., *Decoding Speculative Decoding*
- **Key idea:** The verification phase is much more hardware-friendly than pure
  one-token-at-a-time decode because the target model sees the draft tokens in
  advance and verifies them in a parallel, prefill-like pass.
- **Relevance to EdgeLM:** This is the real systems reason speculative decoding
  is promising here. It amortizes the expensive target-model weight traffic over
  multiple generated tokens.
- **Estimated impact:** Very High.
- **Implementation complexity:** Medium.
- **Details:** For a `0.4-0.6 GB` ternary target on `~40 GB/s` real DDR4
  bandwidth, the only way to move the needle is to reduce how often the full
  target path must be streamed for each emitted token. Verification does exactly
  that when acceptance is decent and draft latency is low enough.

### 3. Draft latency matters more than draft quality once the draft gets too large

#### 3.1 Higher acceptance rate does not automatically mean higher throughput

- **Source:** Yan et al., *Decoding Speculative Decoding*
- **Key idea:** The common intuition "pick the draft model with the highest
  acceptance rate" is incomplete and often wrong. Larger draft models can
  improve acceptance while still lowering overall throughput because their own
  latency dominates.
- **Relevance to EdgeLM:** This directly challenges the current roadmap's casual
  "`0.5B` draft model" assumption. The right draft is the one that maximizes end
  to end throughput, not the one that most resembles the target.
- **Estimated impact:** Very High.
- **Implementation complexity:** Low.
- **Details:** Yan et al. show smaller draft models with lower token acceptance
  rate can outperform larger draft models with higher acceptance. Their
  analytical model treats throughput as jointly dependent on:
  - draft latency,
  - target verification latency,
  - and accepted tokens per round.

  That is the right mental model for EdgeLM too.

#### 3.2 Hardware-efficient draft architecture matters more than "best small LM"

- **Source:** Yan et al., *Decoding Speculative Decoding*
- **Key idea:** Draft models should be designed for speculative throughput, not
  for standalone language-model quality. The paper specifically explores
  shallower, wider draft shapes that better match hardware behavior.
- **Relevance to EdgeLM:** This is especially important on Gracemont E-cores,
  where deep sequential drafts are likely to be a poor fit. A tiny
  hardware-efficient draft may beat a more accurate but heavier one.
- **Estimated impact:** Very High.
- **Implementation complexity:** Medium.
- **Details:** The paper reports that language-model accuracy correlates weakly
  with speculative usefulness, and that redesigning draft shape can materially
  improve throughput. For EdgeLM, that strongly suggests:
  - draft size should be chosen from a latency budget first,
  - draft width/depth tradeoffs matter,
  - and "`0.5B` ternary on E-cores" should be treated as an upper-bound
    experiment, not as the default configuration.

### 4. Fixed lookahead `K` is too naive for a serious implementation

#### 4.1 Candidate length is a control problem, not just a compile-time constant

- **Source:** Huang et al., *SpecDec++: Boosting Speculative Decoding via
  Adaptive Candidate Lengths* (OpenReview / ICLR 2025 submission)
- **Key idea:** The right candidate length is dynamic. SpecDec++ formulates the
  choice of `K` as a decision process and derives a threshold-style stopping
  policy based on rejection probability.
- **Relevance to EdgeLM:** A fixed `K = 4` or `K = 8` is good enough for a first
  prototype, but not for the final paper-quality system.
- **Estimated impact:** High.
- **Implementation complexity:** Medium.
- **Details:** SpecDec++ adds an acceptance-prediction head and stops the draft
  round when the predicted rejection risk passes a threshold. Even if EdgeLM
  does not copy that exact method, the core lesson is clear: the runtime should
  adapt speculation length to context and measured acceptance behavior.

#### 4.2 Acceptance is context-dependent, not only position-dependent

- **Source:** Li et al., *EAGLE-2: Faster Inference of Language Models with
  Dynamic Draft Trees* (ICML 2024)
- **Key idea:** Acceptance depends strongly on context, not just on draft token
  position. Static trees or fixed candidate counts therefore leave speed on the
  table.
- **Relevance to EdgeLM:** This reinforces the case for an adaptive controller.
  Hard prompts, code generation, math, and repetitive text will not all want the
  same speculation depth.
- **Estimated impact:** High.
- **Implementation complexity:** Medium.
- **Details:** EAGLE-2 shows draft-model confidence can approximate acceptance
  rate well enough to drive dynamic tree expansion. EdgeLM does not need to
  adopt EAGLE-2 wholesale to benefit from the same idea. A much simpler
  controller could use:
  - recent acceptance ratio,
  - rejection position histogram,
  - and draft confidence

  to shrink or grow `K` online.

### 5. The speculative-decoding design space has several families, and they are not equally suitable for EdgeLM

#### 5.1 External draft model

- **Source:** Leviathan et al.; Yan et al.; DistillSpec
- **Key idea:** A separate smaller model drafts tokens and the target model
  verifies them.
- **Relevance to EdgeLM:** This is the cleanest first implementation because it
  requires no target-model retraining and matches the current project roadmap.
- **Estimated impact:** Very High.
- **Implementation complexity:** Medium.

#### 5.2 Self-speculative / early-exit draft path

- **Source:** Zhang et al., *Draft & Verify*; Elhoushi et al., *LayerSkip*
- **Key idea:** The target model itself drafts by running only part of its own
  layers, then verifies with the remaining layers.
- **Relevance to EdgeLM:** This can remove the need for an auxiliary draft
  checkpoint and reduce memory overhead, but it changes the engineering and
  sometimes the training assumptions.
- **Estimated impact:** High.
- **Implementation complexity:** Medium to High.

#### 5.3 Multi-head / in-model speculative heads

- **Source:** Cai et al., *Medusa*
- **Key idea:** Extra decoding heads on top of the same backbone generate
  several candidate continuations in parallel, often with tree attention.
- **Relevance to EdgeLM:** Interesting academically, but not the easiest
  starting point for a from-scratch custom C engine.
- **Estimated impact:** Medium to High.
- **Implementation complexity:** High.

#### 5.4 Feature-level and tree-based methods

- **Source:** Li et al., *EAGLE* and *EAGLE-2*
- **Key idea:** Drafting can occur at a more structured internal level with
  tree-based verification and context-aware expansion.
- **Relevance to EdgeLM:** These methods contribute important scheduling and
  acceptance insights, but are more specialized than classical draft-and-verify.
- **Estimated impact:** High.
- **Implementation complexity:** High.

### 6. External draft-and-verify remains the best first EdgeLM implementation path

#### 6.1 It matches the current architecture and keeps the engine modular

- **Source:** Leviathan et al.; project implementation plan
- **Key idea:** External drafting gives EdgeLM a clean separation between:
  - target-model kernels,
  - speculative control logic,
  - and draft-model execution.
- **Relevance to EdgeLM:** That modularity matters in a custom C codebase with
  no framework support. It lets the team build and benchmark speculative decode
  without entangling target-model internals too early.
- **Estimated impact:** Very High.
- **Implementation complexity:** Medium.
- **Details:** EdgeLM can prototype this with:
  - a draft-model interface,
  - a verifier interface,
  - a shared token/probability buffer,
  - and a runtime policy module.

  That is much cleaner than immediately rewriting the target model around early
  exits or auxiliary decoding heads.

#### 6.2 Tokenizer and vocabulary compatibility are non-negotiable

- **Source:** Inference from Leviathan et al.'s token-level acceptance rule
- **Key idea:** External speculative decoding assumes draft and target operate in
  the same token space.
- **Relevance to EdgeLM:** Section 20 is not independent from Section 19. If the
  draft model uses a different tokenizer, vocabulary ordering, or chat-template
  semantics, classical token-level verification breaks or acceptance collapses.
- **Estimated impact:** Very High.
- **Implementation complexity:** Medium.
- **Details:** This is an inference rather than a quoted sentence, but it follows
  directly from the algorithm. For EdgeLM, that means the draft model should
  ideally share:
  - tokenizer,
  - vocabulary IDs,
  - special-token behavior,
  - and prompt formatting assumptions

  with the target model family.

### 7. Self-speculative decoding is attractive for memory, but its ceiling is more constrained

#### 7.1 Draft & Verify is appealing because it needs no auxiliary model

- **Source:** Zhang et al., *Draft & Verify: Lossless Large Language Model
  Acceleration via Self-Speculative Decoding* (ACL 2024)
- **Key idea:** Draft & Verify proposes self-speculative decoding by skipping
  intermediate layers during drafting and then validating the result with the
  original model in one forward pass. The paper emphasizes no extra training and
  no extra memory footprint.
- **Relevance to EdgeLM:** On a `16 GB` laptop with a strict inference memory
  budget, "no extra draft checkpoint" is a real advantage.
- **Estimated impact:** High.
- **Implementation complexity:** Medium to High.
- **Details:** This is the most interesting alternative to external drafting for
  EdgeLM because it attacks RAM pressure directly. If an external draft model
  turns out to create too much memory traffic or complexity, Draft & Verify is a
  credible second path.

#### 7.2 But early-exit style methods have a structural upper bound

- **Source:** Yan et al., *Decoding Speculative Decoding*; LayerSkip
- **Key idea:** Early-exit and self-speculative methods are limited because the
  draft path still executes a non-trivial fraction of the target model.
- **Relevance to EdgeLM:** This matters a lot on a bandwidth-bound CPU. If the
  draft already streams a large fraction of the target weights, the benefit may
  be smaller than an extremely tiny external draft.
- **Estimated impact:** High.
- **Implementation complexity:** Medium.
- **Details:** Yan et al. explicitly note that early-exit methods usually run at
  least about a quarter of the target model, which limits their possible gains
  relative to tiny external drafts that can be orders of magnitude smaller.
  LayerSkip improves the practical self-speculative path further, but it does so
  with a training recipe tailored to early exits. That is a real research
  commitment, not a free runtime-only improvement.

### 8. Medusa and EAGLE matter intellectually, but they are not the first EdgeLM build target

#### 8.1 Medusa is elegant, but it moves EdgeLM toward model modification and fine-tuning

- **Source:** Cai et al., *Medusa: Simple LLM Inference Acceleration Framework
  with Multiple Decoding Heads*
- **Key idea:** Medusa adds extra decoding heads, uses tree attention, and can
  verify multiple candidate continuations in parallel. It can use either exact
  rejection sampling or the faster "typical acceptance" path.
- **Relevance to EdgeLM:** Medusa is useful as a literature reference, but it is
  not the simplest first target for a hand-written C engine whose current plan
  assumes an external draft model.
- **Estimated impact:** Medium.
- **Implementation complexity:** High.
- **Details:** The important nuance is that Medusa's practical fast path often
  uses typical acceptance, which does not preserve exact distribution in the
  same strict sense as classical lossless speculative decoding. That does not
  make Medusa uninteresting, but it weakens its fit as EdgeLM's first exact
  systems path.

#### 8.2 EAGLE and EAGLE-2 provide valuable ideas, but they are a later-stage specialization

- **Source:** Li et al., *EAGLE* (ICML 2024); Li et al., *EAGLE-2* (ICML 2024)
- **Key idea:** EAGLE moves drafting from token level toward feature-level
  autoregression and uses structured draft trees; EAGLE-2 improves this further
  with context-aware dynamic trees and draft-confidence calibration.
- **Relevance to EdgeLM:** These papers are extremely useful for understanding
  dynamic acceptance control, but they are not the easiest path to a first
  implementation.
- **Estimated impact:** High.
- **Implementation complexity:** High.
- **Details:** EAGLE reports strong speedups while preserving output
  distribution, and EAGLE-2 pushes that further with dynamic trees. But from
  EdgeLM's perspective, they imply:
  - more specialized draft machinery,
  - more complex verifier logic,
  - and more distance from the implementation plan's current architecture.

  The right use of these papers now is not "implement EAGLE first." It is
  "borrow their acceptance-aware control ideas after the classical path works."

### 9. Better alignment beats blindly scaling the draft

#### 9.1 Distilling the draft can improve speculative payoff more efficiently than making it larger

- **Source:** Zhou et al., *DistillSpec: Improving Speculative Decoding via
  Knowledge Distillation* (ICLR 2024)
- **Key idea:** DistillSpec improves draft/target alignment through on-policy
  data generation and a tailored divergence function, yielding sizable speedups
  over standard speculative decoding.
- **Relevance to EdgeLM:** If the first draft model underperforms, the next move
  should not automatically be "use a bigger draft." Better-aligned drafting may
  be a better lever than more parameters.
- **Estimated impact:** High.
- **Implementation complexity:** High for training; Low relevance for v1 runtime.
- **Details:** DistillSpec is especially important conceptually because it
  reframes draft quality as an alignment problem, not a size problem. For
  EdgeLM, that means future research could focus on:
  - distilling a tiny draft into the target token distribution,
  - training on target-like prompts,
  - or co-designing the draft with the actual verifier workload.

#### 9.2 Confidence calibration is also useful as a runtime signal

- **Source:** EAGLE-2
- **Key idea:** Draft confidence can approximate acceptance probability well
  enough to drive better scheduling.
- **Relevance to EdgeLM:** Even without training a new acceptance head, EdgeLM
  may be able to use raw draft probabilities as a cheap heuristic for adaptive
  `K`, early verification, or speculative disablement.
- **Estimated impact:** High.
- **Implementation complexity:** Medium.

### 10. Quantization and speculative decoding can compose

#### 10.1 Recent work shows drafting and verification can use different precision regimes

- **Source:** Zhao et al., *QSpec: Speculative Decoding with Complementary
  Quantization Schemes* (arXiv 2024 / EMNLP 2025)
- **Key idea:** QSpec combines low-precision fast drafting with higher-precision
  verification and emphasizes that speculative decoding can be co-designed with
  quantization rather than treated as a separate layer.
- **Relevance to EdgeLM:** This is highly relevant to a ternary-first engine.
  EdgeLM should not assume the draft must use the exact same representation and
  kernel mix as the verifier.
- **Estimated impact:** Medium to High.
- **Implementation complexity:** High.
- **Details:** QSpec is not an immediate implementation target for EdgeLM, but
  it expands the design space in an important way. A future EdgeLM paper could
  plausibly compare:
  - ternary target + ternary external draft,
  - ternary target + more aggressively quantized draft,
  - or self-speculative paths that reuse target weights.

  The larger lesson is that speculative decoding and quantization should be
  co-designed, not optimized in isolation.

### 11. On this laptop, draft and verify should be scheduled conservatively

#### 11.1 A shared-DDR4 machine should assume draft/verify contention until proven otherwise

- **Source:** Inference from Sections 01, 15, 16, and 18 combined with the
  speculative-decoding literature
- **Key idea:** The papers establish why speculative decoding helps in general,
  but they do not target Alder Lake + Iris Xe + dual-channel DDR4 specifically.
  On EdgeLM's machine, the draft and verifier still compete for the same memory
  system.
- **Relevance to EdgeLM:** This is one of the most important practical
  implementation constraints in the entire section.
- **Estimated impact:** Very High.
- **Implementation complexity:** Medium.
- **Details:** This recommendation is an inference from local hardware research,
  not a direct quote from one speculative-decoding paper. The main target-model
  pass is already bandwidth-sensitive. So although an E-core draft appears
  "free" on paper, an overly large draft can still hurt end-to-end speed by
  consuming:
  - DDR4 bandwidth,
  - LLC capacity,
  - and shared memory-controller time.

  EdgeLM should therefore assume the following policy by default:
  - keep the draft genuinely tiny,
  - use coarse handoff boundaries,
  - and treat aggressive draft/verify overlap as an experiment, not as the
    baseline.

#### 11.2 P-cores should own verification; E-cores should own drafting only if the draft is truly cheap

- **Source:** Inference from Sections 01 and 15 plus Yan et al.
- **Key idea:** The verifier is the throughput-critical exact path and should
  stay on P-cores. E-cores are appropriate for drafting only when the draft is
  lightweight enough that it does not meaningfully starve the target pass.
- **Relevance to EdgeLM:** This narrows the hardware mapping considerably.
- **Estimated impact:** Very High.
- **Implementation complexity:** Medium.
- **Details:** For EdgeLM, the initial scheduling assumption should be:
  - `P-core cluster`: target verification
  - `E-core subset`: draft generation
  - one coarse mailbox or SPSC ring between them
  - minimal fine-grained locking

  This follows the earlier Alder Lake results that P/E handoff should be coarse.
  It also means the current "`0.5B on E-core 10-13`" plan should be validated
  very skeptically. If the draft is not cheap enough, it should be shrunk or
  disabled at runtime.

#### 11.3 The iGPU should not be part of the first speculative-decoding prototype

- **Source:** Inference from Sections 16, 17, and 18
- **Key idea:** Speculative decoding and CPU+iGPU hybrid attention are both
  substantial projects and both stress shared memory. Combining them too early
  will make attribution impossible.
- **Relevance to EdgeLM:** This is a project-management and systems-design
  recommendation.
- **Estimated impact:** High.
- **Implementation complexity:** Low.
- **Details:** The clean experimental sequence is:
  1. establish CPU-only speculative decoding,
  2. characterize acceptance and bandwidth behavior,
  3. then test whether iGPU offload composes with the speculative path.

  Reversing that order would make negative results hard to interpret.

### 12. EdgeLM needs a runtime controller, not just a speculative kernel

#### 12.1 Speculative decoding should be dynamically enabled, sized, or bypassed per request

- **Source:** SpecDec++; EAGLE-2; Yan et al.; inference for EdgeLM runtime
- **Key idea:** Speculation is not universally beneficial. Its value depends on
  draft latency, acceptance, workload type, and prompt context.
- **Relevance to EdgeLM:** The final engine should be able to fall back to
  vanilla decode automatically when speculative rounds stop paying for
  themselves.
- **Estimated impact:** Very High.
- **Implementation complexity:** Medium.
- **Details:** The runtime should track at least:
  - draft latency per round,
  - verifier latency per round,
  - accepted tokens per round,
  - rejection position,
  - and possibly average draft confidence.

  With those signals, EdgeLM can:
  - grow or shrink `K`,
  - disable speculation on bad contexts,
  - or switch to a different draft policy.

#### 12.2 The speculative path should be instrumented like a first-class subsystem

- **Source:** Inference from the literature and EdgeLM benchmarking goals
- **Key idea:** A research-grade implementation must expose why speculative
  decoding helped or hurt, not just whether overall tok/s moved.
- **Relevance to EdgeLM:** This is essential for the paper. Otherwise the
  project cannot explain results rigorously.
- **Estimated impact:** Very High.
- **Implementation complexity:** Medium.
- **Details:** Minimum counters should include:
  - accepted prefix length distribution,
  - average rounds per emitted token block,
  - draft tok/s,
  - verifier tok/s,
  - fallback frequency,
  - and memory-bandwidth-sensitive timing under speculation on/off.

## Comparative Fit for EdgeLM

| Approach | Exact distribution | Extra checkpoint | Training requirement | Memory overhead | Engineering risk | EdgeLM verdict |
| --- | --- | --- | --- | --- | --- | --- |
| Classical external draft + verify | Yes | Yes | No | Medium | Medium | Best first implementation |
| Draft & Verify self-speculation | Yes | No | No | Low | Medium to High | Strong second path if external draft is too costly |
| LayerSkip | Yes | No | Yes | Low | High | Good research direction only if training/retraining is on the table |
| Medusa | Sometimes; common fast mode is often not exact | No separate checkpoint, but extra heads | Usually yes | Low to Medium | High | Not first target for an exact C engine |
| EAGLE / EAGLE-2 | Yes | Specialized drafting machinery | More specialized than classical SD | Medium | High | Later-stage algorithmic stretch, not v1 |
| QSpec-style complementary quantization | Yes | No auxiliary model in the paper's form | No | Low to Medium | High | Interesting future co-design once the core engine exists |

## Recommendations for EdgeLM

### 1. Build classical exact speculative decoding first

- Implement the external draft-and-verify path before self-speculative or
  EAGLE-style variants.
- Start with greedy decoding for bring-up and correctness.
- Keep the exact non-greedy sampler path as an explicit phase-2 extension.

### 2. Treat draft size as a latency-budget problem, not a language-modeling problem

- Do not assume "`0.5B` ternary draft" is optimal.
- Start with the smallest tokenizer-compatible draft that can still accept more
  than one token on average in favorable contexts.
- Prefer hardware-efficient shallow/wide draft experiments over deeper drafts if
  model design is under your control.

### 3. Make `K` adaptive as soon as the fixed-`K` prototype works

- A fixed `K` is acceptable for first implementation only.
- The final system should adjust speculation depth using measured acceptance and
  draft confidence.
- The runtime must be able to disable speculation when it stops paying off.

### 4. Keep the first hardware mapping conservative

- P-cores own target verification.
- A small E-core subset owns draft generation.
- Use a coarse mailbox, not fine-grained locks.
- Assume draft/verify overlap is harmful until the measurements prove otherwise.

### 5. Keep speculative decoding CPU-only until the baseline is understood

- Do not combine iGPU attention offload with the first speculative prototype.
- First measure classical CPU-only speculative decode.
- Then decide whether speculative verification composes with any later hybrid
  CPU+iGPU plan.

### 6. Keep self-speculative decoding as the highest-value fallback path

- If external drafting adds too much RAM pressure, too much bandwidth pressure,
  or too much engineering complexity, investigate Draft & Verify next.
- Treat LayerSkip as a training-aware research branch, not a baseline
  implementation assumption.

## Suggested EdgeLM Experiment Sequence

1. Implement exact external speculative decoding with fixed `K` and greedy
   verification.
2. Benchmark multiple tiny draft sizes on the reserved E-cores and record:
   - draft latency,
   - verifier latency,
   - accepted tokens per round,
   - total tok/s,
   - and whether target throughput regresses from memory contention.
3. Add adaptive `K` or early-stop logic driven by recent acceptance.
4. Extend to exact non-greedy sampling.
5. Compare against a self-speculative Draft & Verify style prototype if memory
   or bandwidth overhead from the external draft is too high.
6. Only after that, investigate whether speculative decode composes with any
   iGPU offload path.

## Bottom Line

Speculative decoding is still one of the strongest late-phase bets in the whole
EdgeLM roadmap, but the current project sketch is too coarse in three ways:

- it treats draft quality as more important than draft latency,
- it treats fixed `K` as if it were enough,
- and it does not yet fully reflect the shared-memory cost of running a draft on
  E-cores next to a bandwidth-bound verifier.

The best first implementation for this codebase is still classical exact
external draft-and-verify. But it should be built with these assumptions:

- the draft must be much more latency-aware than accuracy-aware,
- the runtime must adapt speculation depth online,
- and the engine must be able to abandon speculation dynamically when the memory
  system says no.

If external drafting proves too costly on this laptop, self-speculative decoding
is the most credible next move. Medusa and EAGLE are important literature, but
they are better treated as stretch research directions after the classical path
has been implemented, measured, and understood.

## Sources

- Leviathan et al., *Fast Inference from Transformers via Speculative
  Decoding* (ICML 2023): `https://proceedings.mlr.press/v202/leviathan23a.html`
- Yan et al., *Decoding Speculative Decoding* (arXiv 2024 / NAACL 2025):
  `https://arxiv.org/abs/2402.01528`
- Zhang et al., *Draft & Verify: Lossless Large Language Model Acceleration via
  Self-Speculative Decoding* (ACL 2024): `https://arxiv.org/abs/2309.08168`
- Elhoushi et al., *LayerSkip: Enabling Early Exit Inference and
  Self-Speculative Decoding* (ACL 2024):
  `https://aclanthology.org/2024.acl-long.681/`
- Cai et al., *Medusa: Simple LLM Inference Acceleration Framework with
  Multiple Decoding Heads* (arXiv 2024): `https://arxiv.org/abs/2401.10774`
- Li et al., *EAGLE: Speculative Sampling Requires Rethinking Feature
  Uncertainty* (ICML 2024):
  `https://proceedings.mlr.press/v235/li24bt.html`
- Li et al., *EAGLE-2: Faster Inference of Language Models with Dynamic Draft
  Trees* (ICML 2024): `https://arxiv.org/abs/2406.16858`
- Huang et al., *SpecDec++: Boosting Speculative Decoding via Adaptive
  Candidate Lengths* (OpenReview / ICLR 2025 submission):
  `https://openreview.net/forum?id=NnExMNiTHw`
- Zhou et al., *DistillSpec: Improving Speculative Decoding via Knowledge
  Distillation* (ICLR 2024): `https://openreview.net/forum?id=rsY6J3ZaTF`
- Zhao et al., *QSpec: Speculative Decoding with Complementary Quantization
  Schemes* (arXiv 2024 / EMNLP 2025): `https://arxiv.org/abs/2410.11305`

## Audit Addendum (2026-04-02)

- **Sampler exactness must remain coupled to speculation.** Once EdgeLM supports
  top-p, repetition rules, or other processors, speculative verification has to
  preserve the post-processor distribution exactly, not just raw logits.
- **Acceptance-length histograms should be a first-class runtime output.** Mean
  speedup alone hides whether the controller is healthy or only occasionally
  lucky.
- **Draft/tokenizer/prompt-format compatibility should be treated as a hard
  correctness gate.** Even small serialization mismatches can silently poison a
  speculative path.
