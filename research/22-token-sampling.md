# Section 22: Token Sampling (top-p, top-k, repetition) -- Extended Research

## Overview

Token sampling is deceptively small in most LLM architecture diagrams.

Compared with:

- the output projection,
- attention,
- FFN matmuls,
- KV-cache traffic,
- and tokenizer/model-loading work,

the sampler looks like a thin final step at the end of each decode iteration.

That appearance is misleading.

The sampler is not the main throughput bottleneck in EdgeLM, but it still sits
on a critical boundary:

- it is the last mathematical transformation before a token becomes user-visible
  output,
- it defines the exact generation distribution that speculative decoding must
  preserve,
- it determines whether the engine degenerates into repetitive or bland text,
- and it runs once per generated token on the control side of the engine.

For EdgeLM, Section 22 is therefore not just "implement top-p and top-k." It is
a design study for the **distribution-shaping and token-selection subsystem** of
a custom C inference engine targeting:

- a `128,256`-token Llama 3 vocabulary,
- CPU-side orchestration,
- deterministic benchmarking,
- and later compatibility with exact speculative decoding.

That means the real questions are:

- which sampling method should be the default,
- how truncation and penalties should be composed,
- where repetition control should live,
- how to make the sampler numerically stable and reproducible,
- how to keep the per-token overhead small on a large vocabulary,
- and how to structure the API so future features do not force a rewrite.

## What the Deep Dive Already Covers

`deep-dive.md` is still empty, but the project baseline is clear enough from the
rest of the repo.

- `implementation-plan.md` currently gives token sampling only a brief Phase 1
  note: implement `top-p`, `top-k`, and temperature.
- The plan places sampling on the orchestration/main-thread side of the engine,
  not in the hot SIMD worker pool.
- `AGENTS.md` expands the scope slightly by also calling out repetition penalty
  as part of the token sampler.
- Section 19 already established a crucial constraint: if EdgeLM later supports
  exact speculative decoding under non-greedy generation, then the sampler is
  part of the target distribution `p` and must be treated mathematically, not as
  an afterthought.
- Section 20 established that the target BitNet model uses the Llama 3 tokenizer
  with vocabulary size `128,256`, so the sampler is working over a large output
  space and must also respect special-token and chat-format policy.

So the unresolved questions entering this section are:

- Should EdgeLM treat `top-k` or `top-p` as the main default?
- Where should temperature live in the pipeline?
- How should repetition penalty be defined and tracked efficiently?
- Should repetition policy include prompt tokens or only generated tokens?
- How expensive is top-p over a `128k` vocabulary on CPU?
- How should the sampler interact with EOS/min-length constraints and future
  structured processors?
- And what sampler architecture will still work cleanly once speculative
  decoding arrives?

## New Findings

### 1. Sampling is not the main throughput bottleneck, but it is still on the per-token critical path

#### 1.1 The sampler runs once for every output token and therefore has a hard latency budget

- **Source:** Project architecture synthesis; `implementation-plan.md`
- **Key idea:** Even if token sampling is much cheaper than the transformer
  forward pass, it still runs every step of decode and therefore cannot be
  allowed to grow unbounded with vocabulary size or feature creep.
- **Relevance to EdgeLM:** This makes sampling a control-path latency problem,
  not a throughput-dominant compute problem.
- **Estimated impact:** High.
- **Implementation complexity:** Low.
- **Details:** On a `100 tok/s` target, the total per-token budget is only about
  `10 ms`. The sampler should consume only a small fraction of that. A sampler
  that drifts into expensive full-vocabulary sorting and repeated history scans
  every token can quietly become a measurable tax.

#### 1.2 The large Llama 3 vocabulary makes naive implementations easier to notice

- **Source:** Microsoft BitNet model card; Section 20 tokenizer research
- **Key idea:** The primary BitNet target uses the Llama 3 tokenizer with a
  `128,256`-token vocabulary.
- **Relevance to EdgeLM:** That vocabulary size is large enough that naive
  `O(V log V)` work, repeated allocations, or repeated history scans are no
  longer "free."
- **Estimated impact:** High.
- **Implementation complexity:** Low.
- **Details:** The output projection still dominates total cost, but after those
  logits exist, the sampler must process them efficiently and predictably.

### 2. The cleanest sampler architecture separates "logits processors" from "logits warpers"

#### 2.1 Official generation libraries already distinguish score processors from sampling warpers

- **Source:** Hugging Face generation docs and generation logits-process code
- **Key idea:** Hugging Face explicitly separates:
  - **LogitsProcessor** objects, which modify or mask scores for policy reasons
  - **LogitsWarper** objects, which reshape the distribution for multinomial
    sampling
- **Relevance to EdgeLM:** This is the right architecture for a custom C
  sampler too.
- **Estimated impact:** Very High.
- **Implementation complexity:** Low.
- **Details:** In the HF generation stack:
  - temperature is a warper,
  - top-p and top-k are warpers,
  - repetition penalty is a processor,
  - min-length and bad-word suppression are processors.

  That split is conceptually useful because it keeps "what tokens are allowed"
  separate from "how to sample among allowed tokens."

#### 2.2 EdgeLM should copy the separation, not necessarily the full framework

- **Source:** HF official docs; EdgeLM zero-dependency constraints
- **Key idea:** EdgeLM does not need a heavyweight plugin framework, but it
  should preserve the same conceptual boundary.
- **Relevance to EdgeLM:** This keeps the sampler maintainable as features grow.
- **Estimated impact:** Very High.
- **Implementation complexity:** Low.
- **Details:** A good internal pipeline is:

```c
raw logits
  -> processors  (ban/mask/penalize/force)
  -> warpers     (temperature/top-k/top-p)
  -> normalize
  -> RNG draw
  -> token id
```

  That structure will matter later for speculative decoding as well.

### 3. Temperature is the simplest sampling control, but its position in the pipeline matters

#### 3.1 Temperature is just score scaling, not a standalone decoding strategy

- **Source:** Hugging Face generation logits-process code
- **Key idea:** The standard temperature warper divides scores by `temperature`,
  flattening or sharpening the next-token distribution.
- **Relevance to EdgeLM:** This makes temperature cheap to implement and easy to
  compose with other methods.
- **Estimated impact:** High.
- **Implementation complexity:** Low.
- **Details:** HF's reference implementation literally applies:

  - `scores = scores / temperature`

  before later sampling logic. That simplicity is useful for EdgeLM because it
  means temperature should be implemented as an in-place score transform rather
  than as a separate sampling mode.

#### 3.2 Temperature should be applied before top-k/top-p truncation

- **Source:** HF generation architecture; inference from distribution semantics
- **Key idea:** Since temperature changes the relative sharpness of the
  distribution, it should act before truncation-based support selection.
- **Relevance to EdgeLM:** This must be fixed as part of the sampler contract if
  the engine wants reproducible behavior across versions.
- **Estimated impact:** High.
- **Implementation complexity:** Low.
- **Details:** If truncation happens first and temperature second, the sampler is
  reshaping a different support set. That can materially change outputs. EdgeLM
  should standardize on:
  - processors first
  - temperature next
  - truncation after that

  This is partly an implementation inference, but it matches common generation
  stacks and is the most sensible semantic ordering.

#### 3.3 Extreme temperature values should dispatch to simpler fast paths

- **Source:** Inference from temperature semantics
- **Key idea:** Very low temperatures approach greedy decoding, and a sampler
  should not insist on paying full multinomial overhead when the effective mode
  is already almost argmax.
- **Relevance to EdgeLM:** This is a practical CPU optimization.
- **Estimated impact:** Medium.
- **Implementation complexity:** Low.
- **Details:** Reasonable policy:
  - `temperature <= tiny_threshold`: use greedy argmax
  - otherwise apply normal stochastic path

  This keeps the deterministic benchmark path simple and avoids numerical
  pathologies from dividing by near-zero temperatures.

### 4. Top-k is an important baseline because it is simple, cheap, and still useful

#### 4.1 Top-k sampling was an early practical answer to beam-search degeneration

- **Source:** Fan et al., *Hierarchical Neural Story Generation* (ACL 2018)
- **Key idea:** Fan et al. explicitly used top-k random sampling with `k = 10`,
  reporting that it worked substantially better than beam search for story
  generation, which tended to produce common phrases and repetitive text.
- **Relevance to EdgeLM:** This is the simplest strong baseline for open-ended
  sampling and a useful Phase 1 target.
- **Estimated impact:** High.
- **Implementation complexity:** Low.
- **Details:** Fan et al. also note the core intuition:
  - completely random sampling can pull very unlikely words,
  - beam search tends toward generic and repetitive text,
  - restricting to the top few candidates is a practical middle ground.

#### 4.2 Top-k is especially attractive on CPU because the implementation can be cheap

- **Source:** HF generation code; algorithmic inference
- **Key idea:** HF's top-k warper keeps the `k` highest-probability tokens and
  suppresses everything below the `k`-th threshold.
- **Relevance to EdgeLM:** This maps well onto efficient CPU partial-selection
  logic.
- **Estimated impact:** High.
- **Implementation complexity:** Low.
- **Details:** EdgeLM does not need a full sort for top-k. A practical
  implementation can use:
  - partial selection,
  - a fixed-size min-heap,
  - or a thresholding pass built around `top_k`.

  For small `k`, this is a very friendly CPU workload.

#### 4.3 But fixed `k` is a structural limitation, not just a tuning inconvenience

- **Source:** Holtzman et al., *The Curious Case of Neural Text Degeneration*
- **Key idea:** Fixed-support truncation cannot adapt to changing entropy from
  one context to the next.
- **Relevance to EdgeLM:** This is why top-k should be implemented and exposed,
  but not necessarily treated as the final default.
- **Estimated impact:** High.
- **Implementation complexity:** Low.
- **Details:** In low-entropy contexts, `k` may keep too many tokens. In
  high-entropy contexts, it may keep too few. That is the core reason nucleus
  sampling became the stronger general-purpose default for open-ended text.

### 5. Nucleus sampling (`top-p`) is the better default for open-ended text generation

#### 5.1 Holtzman et al. argue that text degeneration comes from the unreliable tail of the distribution

- **Source:** Holtzman et al., *The Curious Case of Neural Text Degeneration*
  (ICLR 2020)
- **Key idea:** Holtzman et al. show that maximizing or sampling naively from the
  full distribution leads to degeneration, and argue that generation should
  truncate the unreliable low-probability tail.
- **Relevance to EdgeLM:** This is the core theoretical justification for
  truncation sampling in modern open-ended text generation.
- **Estimated impact:** Very High.
- **Implementation complexity:** Low.
- **Details:** Their central proposal is nucleus sampling:
  - choose the smallest set of top tokens whose cumulative probability exceeds
    `p`
  - sample from that dynamic set

  This makes the support size adaptive to the actual entropy of the context.

#### 5.2 Nucleus sampling is dynamic in exactly the way top-k is not

- **Source:** Holtzman et al.; HF top-p code
- **Key idea:** HF's reference top-p implementation sorts scores in descending
  order, computes cumulative probabilities, and masks everything above the
  target mass threshold while keeping at least one token.
- **Relevance to EdgeLM:** This is the behavior EdgeLM should reproduce when it
  claims to support top-p.
- **Estimated impact:** Very High.
- **Implementation complexity:** Medium.
- **Details:** The crucial difference from top-k is that the retained support can
  vary from:
  - very small in low-entropy contexts
  - to much larger in high-entropy contexts

  This is exactly why top-p tends to feel more robust across different prompt
  types.

#### 5.3 For EdgeLM, top-p should be the default user-facing stochastic mode

- **Source:** Holtzman et al.; project workload inference
- **Key idea:** Because EdgeLM is intended for general local chat and open-ended
  generation, a dynamic truncation rule is a better default than a fixed-cardinality
  rule.
- **Relevance to EdgeLM:** This is the strongest user-facing recommendation in
  the section.
- **Estimated impact:** Very High.
- **Implementation complexity:** Medium.
- **Details:** Top-k should still be exposed. But the default stochastic mode
  should likely be:
  - `top_p` enabled
  - moderate temperature
  - optional repetition control

  rather than pure top-k.

### 6. Top-k and top-p are not mutually exclusive, but EdgeLM must define the composition order explicitly

#### 6.1 Combining top-k and top-p is a practical engineering choice, not a pure theory result

- **Source:** HF generation architecture; project implementation inference
- **Key idea:** Generation systems often allow both top-k and top-p to be active,
  effectively sampling from the intersection of two truncation rules.
- **Relevance to EdgeLM:** This can be a useful CPU optimization and a helpful
  safety cap on candidate-set size.
- **Estimated impact:** High.
- **Implementation complexity:** Medium.
- **Details:** On a `128k` vocabulary, a hard top-k cap can bound the amount of
  sorting work top-p needs to do in pathological high-entropy cases. But this is
  no longer identical to pure top-p if the cap actually binds.

#### 6.2 The order should be standardized for reproducibility

- **Source:** HF code structure; implementation inference
- **Key idea:** If both top-k and top-p are enabled, the engine needs one fixed,
  documented order.
- **Relevance to EdgeLM:** Otherwise benchmark outputs will drift across
  versions, and speculative-decoding integration becomes harder to reason about.
- **Estimated impact:** High.
- **Implementation complexity:** Low.
- **Details:** The most practical order for EdgeLM is:
  1. processors
  2. temperature
  3. optional top-k cap
  4. top-p on the retained candidates
  5. renormalize and sample

  This is partly an implementation inference, but it is a sensible choice
  because it makes candidate-set cost easier to control.

### 7. Repetition penalty should be treated as a logits processor, not as part of top-p/top-k

#### 7.1 Official generation libraries define repetition penalty as a separate processor

- **Source:** Hugging Face generation docs
- **Key idea:** HF documents `RepetitionPenaltyLogitsProcessor` separately from
  top-k/top-p warpers and states that the penalty is applied at most once per
  token.
- **Relevance to EdgeLM:** This is the right abstraction because repetition
  control changes token scores before the truncation/sampling stage.
- **Estimated impact:** Very High.
- **Implementation complexity:** Low.

#### 7.2 In common decoder-only practice, prompt tokens are part of the repetition domain

- **Source:** Hugging Face generation docs
- **Key idea:** HF explicitly notes that for decoder-only models, the considered
  tokens include the prompt.
- **Relevance to EdgeLM:** This is an important semantic choice that many quick
  implementations forget to document.
- **Estimated impact:** High.
- **Implementation complexity:** Low.
- **Details:** This means that if a user prompt contains a rare word, standard
  repetition penalty may discourage reusing that word in the answer. Sometimes
  that is desirable; sometimes it is clearly not.

#### 7.3 EdgeLM should make repetition scope configurable

- **Source:** HF docs; project/chat-format inference
- **Key idea:** Because EdgeLM targets instruct/chat models with explicit special
  tokens and structured prompts, repetition penalty should not blindly treat all
  historical tokens as equivalent.
- **Relevance to EdgeLM:** This is one of the most practical local recommendations
  in the section.
- **Estimated impact:** High.
- **Implementation complexity:** Medium.
- **Details:** A good policy split is:
  - **special/chat-template tokens:** never penalize
  - **system/user prompt tokens:** configurable
  - **generated assistant tokens:** penalize by default

  This avoids pathological suppression of structural tokens and gives the engine
  saner chat behavior.

### 8. Repetition control over the entire history is crude; recency-aware variants are worth planning for

#### 8.1 Recent work argues that naive repetition penalty is hard to tune

- **Source:** Zhu et al., *Penalty Decoding: Well Suppress the Self-Reinforcement
  Effect in Open-Ended Text Generation* (EMNLP 2023)
- **Key idea:** Zhu et al. identify a self-reinforcement effect in open-ended
  generation and argue that plain repetition penalty is difficult to tune,
  proposing a forgetting mechanism that disregards distant tokens.
- **Relevance to EdgeLM:** This is a strong hint that repetition policy should
  eventually become more nuanced than "penalize every previously seen token
  forever."
- **Estimated impact:** High.
- **Implementation complexity:** Medium.

#### 8.2 EdgeLM should start simple, but not bake in an infinite-memory assumption

- **Source:** Penalty Decoding; project architecture inference
- **Key idea:** The v1 engine can start with standard repetition penalty, but the
  state representation should leave room for recency windows or decayed counts.
- **Relevance to EdgeLM:** This avoids a rewrite later.
- **Estimated impact:** High.
- **Implementation complexity:** Medium.
- **Details:** A useful state structure is not just "set of seen tokens." It is
  something closer to:
  - token seen flag / count
  - last occurrence position
  - touched-token list

  That is enough to support both current and future repetition policies.

### 9. Efficient repetition penalty implementation is a sparse-state problem, not a full-vocabulary problem

#### 9.1 Do not scan the entire context every token

- **Source:** HF repetition-penalty semantics; project performance inference
- **Key idea:** Since the penalty is applied at most once per token, the runtime
  only needs to know which unique token IDs are currently "active" in the
  repetition state.
- **Relevance to EdgeLM:** This is the obvious CPU optimization for a `128k`
  vocabulary.
- **Estimated impact:** High.
- **Implementation complexity:** Low.
- **Details:** A naive implementation that rescans the entire prompt+generation
  history every step wastes work. The right design is:
  - a dense mark/count array indexed by token ID
  - plus a sparse touched-token list

  so the per-step penalty pass touches only active tokens.

#### 9.2 The memory cost of a dense token-state table is trivial compared with the model

- **Source:** BitNet vocab size; project memory budget inference
- **Key idea:** A `128,256`-entry table for counts or stamps is tiny relative to
  EdgeLM's memory budget.
- **Relevance to EdgeLM:** This is one of the rare places where a dense
  vocabulary-indexed structure is the right answer.
- **Estimated impact:** Medium.
- **Implementation complexity:** Low.
- **Details:** Even a `uint32_t` table is only about `512 KB`. That is
  negligible compared with model weights and KV cache, and it buys constant-time
  updates and simple logic.

### 10. The sampler must be numerically stable and must never filter away the whole distribution

#### 10.1 Stable normalization still matters even if the logits came from a precise model

- **Source:** HF top-p implementation; standard softmax inference
- **Key idea:** Top-p implementations compute softmax probabilities after score
  ordering, and this implicitly assumes numerically stable exponentiation.
- **Relevance to EdgeLM:** The sampling path should run in `FP32` and use a
  max-subtracted softmax or equivalent stable normalization.
- **Estimated impact:** High.
- **Implementation complexity:** Low.
- **Details:** This is one of those areas where a sampler can be "mostly right"
  but still occasionally emit NaNs or pathologically biased probabilities when
  logits are extreme.

#### 10.2 EdgeLM should always keep at least one candidate token

- **Source:** HF top-p and top-k code
- **Key idea:** HF's implementations explicitly carry a `min_tokens_to_keep`
  guard so that filtering never removes every candidate.
- **Relevance to EdgeLM:** This is a critical safety rule.
- **Estimated impact:** Very High.
- **Implementation complexity:** Low.
- **Details:** Practical failure cases include:
  - all logits masked by bad-word/min-length policies
  - extreme top-p or top-k settings
  - numerical edge cases after processors and warpers

  The sampler should always have a guaranteed fallback candidate, typically the
  best remaining token.

### 11. Top-p over a 128k vocabulary is the main algorithmic cost center in the sampler

#### 11.1 Pure top-p naively wants sorted probabilities

- **Source:** HF top-p code; Holtzman et al.
- **Key idea:** The canonical implementation sorts scores in descending order,
  computes cumulative probabilities, and truncates at the mass threshold.
- **Relevance to EdgeLM:** This is straightforward but potentially the most
  expensive part of the sampler on CPU.
- **Estimated impact:** High.
- **Implementation complexity:** Medium.
- **Details:** For `128,256` logits, full descending sort every token is not free.
  It may still be acceptable in early bring-up, but it should be treated as the
  main optimization target inside the sampler itself.

#### 11.2 EdgeLM should start correct, then optimize candidate construction

- **Source:** Project philosophy; HF reference implementation
- **Key idea:** The first implementation should prioritize exact semantics. Only
  after that should EdgeLM optimize away unnecessary full-vocabulary ordering.
- **Relevance to EdgeLM:** This is the safest development path.
- **Estimated impact:** Very High.
- **Implementation complexity:** Medium.
- **Details:** Recommended sequence:
  1. exact reference top-p with full sort
  2. profile actual per-token cost
  3. optimize with bounded candidate construction if needed

#### 11.3 A practical optimization path is "hard cap first, dynamic truncation second"

- **Source:** Project implementation inference
- **Key idea:** If pure top-p proves too expensive, the most practical CPU
  optimization is to first cap candidates with top-k, then run top-p over that
  shortlist.
- **Relevance to EdgeLM:** This gives a bounded-cost stochastic mode that still
  retains much of nucleus sampling's adaptive behavior.
- **Estimated impact:** High.
- **Implementation complexity:** Medium.
- **Details:** This is not semantically identical to pure top-p if the top-k cap
  binds, so it should be documented as a distinct mode or clearly defined
  combined mode.

### 12. The sampler should expose deterministic fast paths for benchmarking

#### 12.1 Greedy decoding deserves a direct argmax path

- **Source:** Project benchmarking needs; temperature/top-p semantics
- **Key idea:** Greedy decode is not just sampling with weird parameters. It is a
  simpler algorithm with different performance characteristics.
- **Relevance to EdgeLM:** This matters for kernel benchmarking, correctness
  checks, and deterministic regression tests.
- **Estimated impact:** High.
- **Implementation complexity:** Low.

#### 12.2 The stochastic path must still be reproducible when seeded

- **Source:** Project benchmarking methodology needs; sampling-system inference
- **Key idea:** A local inference engine needs deterministic replay of stochastic
  runs when seed and parameters are fixed.
- **Relevance to EdgeLM:** Without this, tuning and debugging get much harder.
- **Estimated impact:** Very High.
- **Implementation complexity:** Medium.
- **Details:** EdgeLM should avoid `rand()` and instead use a small, explicit RNG
  such as:
  - PCG32,
  - xoroshiro/xoshiro,
  - or another well-specified non-cryptographic generator.

  The sampler should define exactly how RNG outputs map to `[0, 1)` floating
  draws so runs are reproducible across builds.

### 13. The sampler should treat EOS, bad-token suppression, and no-repeat-ngram as adjacent policy processors

#### 13.1 The processor/warper split naturally accommodates more than the four settings in the roadmap

- **Source:** Hugging Face generation docs
- **Key idea:** HF's processor stack includes forced BOS/EOS, bad-word
  suppression, no-repeat-ngram, min-length, and other policies that are clearly
  distinct from top-k/top-p sampling itself.
- **Relevance to EdgeLM:** This gives a strong blueprint for extensibility.
- **Estimated impact:** High.
- **Implementation complexity:** Medium.

#### 13.2 EdgeLM should at least reserve API space for these controls even if v1 only ships a subset

- **Source:** Project architecture inference
- **Key idea:** Even if the first sampler only exposes temperature, top-k, top-p,
  and repetition penalty, the internal API should anticipate:
  - EOS suppression until minimum length,
  - bad-word suppression,
  - no-repeat-ngram,
  - and token forcing hooks.
- **Relevance to EdgeLM:** This prevents a future sampler rewrite.
- **Estimated impact:** High.
- **Implementation complexity:** Low.
- **Details:** This matters because no-repeat-ngram and repetition penalty solve
  different problems:
  - repetition penalty is token-local and soft
  - no-repeat-ngram is phrase-local and hard

  They should not be conflated.

### 14. Sampling semantics must be reusable by speculative decoding

#### 14.1 The post-processor distribution is the one speculative decoding must preserve

- **Source:** Section 19 speculative-decoding research; Leviathan et al.
- **Key idea:** For exact non-greedy speculative decoding, the relevant target
  distribution is not "raw model softmax." It is the distribution after all
  active processors and warpers.
- **Relevance to EdgeLM:** This is one of the biggest architectural constraints
  on the sampler.
- **Estimated impact:** Very High.
- **Implementation complexity:** Medium.
- **Details:** If repetition penalty, top-p, or temperature are applied
  differently in normal decode and speculative verification, exactness is lost.

#### 14.2 EdgeLM should factor the sampler into "build distribution" and "draw token"

- **Source:** Section 19 inference; sampler design synthesis
- **Key idea:** The sampler should be split into:
  - a deterministic distribution-building stage
  - and a stochastic draw stage
- **Relevance to EdgeLM:** This cleanly supports:
  - normal decode,
  - greedy decode,
  - speculative verification,
  - and later testing/debugging tools.
- **Estimated impact:** Very High.
- **Implementation complexity:** Medium.
- **Details:** That split makes it much easier to:
  - unit-test score transformations,
  - compare exact distributions against a reference implementation,
  - and reuse the same math in speculative correction logic.

### 15. EdgeLM now has a clear default sampler policy

#### 15.1 Best default for general local chat

- **Source:** Holtzman et al.; HF generation architecture; project inference
- **Key idea:** The best general default is likely:
  - temperature
  - nucleus sampling
  - optional light repetition penalty
- **Relevance to EdgeLM:** This is the most practical answer for everyday use.
- **Estimated impact:** Very High.
- **Implementation complexity:** Medium.
- **Details:** A plausible default profile is:
  - `temperature`: moderate
  - `top_p`: enabled
  - `top_k`: disabled or only used as a hidden cost cap
  - `repetition_penalty`: mild and configurable

  Exact default numeric values can be tuned later, but the shape of the policy
  is already clear.

#### 15.2 Best default for benchmarking and kernel bring-up

- **Source:** Project benchmarking needs
- **Key idea:** Benchmarking wants greedy decode first.
- **Relevance to EdgeLM:** This should be the default for correctness and
  throughput measurement unless the benchmark explicitly studies stochastic
  generation.
- **Estimated impact:** High.
- **Implementation complexity:** Low.
- **Details:** This avoids polluting performance measurements with:
  - RNG cost,
  - top-p sorting cost,
  - and output variability.

## Comparative Fit for EdgeLM

| Mode | Quality behavior | CPU cost | Determinism | Main risk | EdgeLM verdict |
| --- | --- | --- | --- | --- | --- |
| Greedy / argmax | Stable, conservative, often bland | Lowest | Highest | generic/repetitive text | Best benchmarking baseline |
| Top-k only | Good simple stochastic baseline | Low to Medium | High when seeded | fixed `k` mismatches entropy | Strong v1 mode, not best final default |
| Top-p only | Best open-ended default behavior | Medium | High when seeded | full-sort cost if naive | Best final default |
| Top-k + top-p | Practical bounded-cost hybrid | Medium | High when seeded | semantics differ from pure top-p when cap binds | Good optimized production mode |
| Repetition penalty alone | Helps local loops | Low | High | can distort prompt-sensitive outputs | Useful processor, not standalone sampler |
| No-repeat-ngram alone | Strong anti-loop protection | Medium | High | harsh for short answers/code | Optional adjunct, not a default |

## Recommendations for EdgeLM

### 1. Build the sampler around processors and warpers

- Processors:
  - repetition penalty
  - min-length EOS suppression
  - bad-token suppression
  - future no-repeat-ngram
- Warpers:
  - temperature
  - top-k
  - top-p

### 2. Make greedy decoding the first implementation and default benchmark path

- It is the simplest correctness target.
- It gives the cleanest kernel-throughput measurements.
- It provides the easiest baseline for later stochastic comparison.

### 3. Make top-p the main user-facing stochastic default

- Implement top-k too, but treat it as:
  - a simple baseline,
  - a low-cost fallback,
  - or a hard cap inside an optimized combined mode.

### 4. Treat repetition penalty as sparse token-state tracking

- Maintain a dense vocab-indexed state table plus a touched-token list.
- Never rescan the whole prompt/history every step.
- Exclude structural special tokens from repetition accounting by default.

### 5. Standardize the transformation order and never change it casually

- Recommended order:
  1. hard processors/masks
  2. repetition penalty
  3. temperature
  4. optional top-k cap
  5. top-p truncation
  6. renormalize
  7. draw token

### 6. Design the sampler for speculative-decoding reuse now

- Expose a deterministic "build final next-token distribution" function.
- Expose a separate "sample from this distribution" function.
- Reuse the same distribution builder for normal decode and speculative
  verification.

## Suggested EdgeLM Implementation Shape

### Core API

```c
typedef struct {
    float temperature;
    float top_p;
    int   top_k;
    float repetition_penalty;
    int   min_tokens_to_keep;
    int   min_new_tokens;
    int   seed;
    int   mode_flags;
} sampler_cfg_t;

typedef struct {
    uint32_t *token_counts;
    uint32_t *token_stamp;
    uint32_t *touched_ids;
    uint32_t  touched_len;
    uint64_t  rng_state[2];
} sampler_state_t;

typedef struct {
    int32_t  *candidate_ids;
    float    *candidate_scores;
    float    *candidate_probs;
    uint32_t  candidate_count;
} sample_dist_t;
```

### Pipeline

1. Start from final-token logits in `FP32`.
2. Apply hard processors and suppressions in place.
3. Apply repetition penalty over sparse touched-token state.
4. Dispatch to greedy fast path if configured.
5. Apply temperature scaling.
6. Construct truncated support (`top_k`, `top_p`, or both).
7. Renormalize probabilities on the retained support only.
8. Draw with seeded RNG.
9. Update repetition state with the selected token.

### State policy

- Track repetition state separately from the raw token history.
- Keep special tokens in a do-not-penalize mask.
- Allow configuration for whether prompt tokens participate in repetition
  accounting.

## Suggested Experiment Sequence

1. Implement greedy argmax and verify exact agreement with a reference engine.
2. Add temperature scaling and multinomial draw on full support.
3. Add top-k truncation with a partial-selection implementation.
4. Add exact top-p with a correctness-first full-sort path.
5. Benchmark sampler time at `128,256` vocab across:
   - greedy
   - top-k
   - top-p
   - top-k + top-p
6. Add repetition penalty with sparse touched-token tracking.
7. Add EOS/min-length and special-token handling.
8. Factor the distribution builder for future speculative-decoding reuse.

## Bottom Line

Sampling is not where EdgeLM wins or loses the entire `100 tok/s` target, but it
is absolutely where the engine wins or loses generation correctness,
reproducibility, and output quality.

The deeper design conclusions are:

- `top-k` is the simplest strong stochastic baseline and should be implemented
  early,
- `top-p` is the better general default for open-ended local generation,
- repetition penalty belongs in a separate logits-processor stage,
- the sampler must be sparse-state and numerically stable on a `128k` vocabulary,
- and the sampler API should already anticipate exact speculative decoding.

So the right EdgeLM sampler is not "just top-p." It is a small distribution
pipeline with:

- deterministic processors,
- explicit warpers,
- a greedy fast path,
- a seeded RNG path,
- sparse repetition tracking,
- and a fixed transformation order that the whole engine can depend on.

That gives the project a sampler that is cheap enough for CPU inference, strong
enough for real text generation, and clean enough to survive the later
speculative-decoding phases without semantic drift.

## Sources

- Fan et al., *Hierarchical Neural Story Generation* (ACL 2018):
  `https://aclanthology.org/P18-1082/`
- Holtzman et al., *The Curious Case of Neural Text Degeneration* (ICLR 2020):
  `https://openreview.net/forum?id=rygGQyrFvH`
- Hewitt et al., *Truncation Sampling as Language Model Desmoothing* (TMLR 2022):
  `https://arxiv.org/abs/2210.15191`
- Zhu et al., *Penalty Decoding: Well Suppress the Self-Reinforcement Effect in
  Open-Ended Text Generation* (EMNLP 2023):
  `https://arxiv.org/abs/2310.14971`
- Hugging Face generation logits-process reference code:
  `https://huggingface.co/transformers/v3.5.1/_modules/transformers/generation_logits_process.html`
- Hugging Face generation docs (`LogitsProcessor`, `LogitsWarper`,
  `RepetitionPenaltyLogitsProcessor`):
  `https://huggingface.co/docs/transformers.js/en/api/generation/logits_process`
- Microsoft BitNet b1.58 2B4T model card:
  `https://huggingface.co/microsoft/bitnet-b1.58-2B-4T-bf16`

## Audit Addendum (2026-04-02)

- **Grammar- or schema-constrained decoding should stay on the roadmap.** Even
  if v1 only ships free-form generation, the processor/warper split should leave
  room for structured constraints later.
- **Alternative truncation rules such as min-p or typical sampling are worth
  leaving API space for.** They are not priority-one, but they fit naturally in
  the same architecture if the abstractions stay clean.
- **Sampler benchmarks should report candidate-set size distributions.** This
  would make top-p cost and repetition-policy side effects much easier to
  understand.
