# Section 25: Research Paper Structure & MLSys/EuroSys Submission -- Extended Research

## Overview

By the time Section 25 arrives, the technical question has changed.

Earlier sections asked:

- what the hardware can do,
- which kernels matter,
- how memory should be managed,
- where the iGPU helps,
- how the decode pipeline should work,
- and how to benchmark the result.

Section 25 asks the final integrative question:

- how does all of that become a **paper that a serious venue might accept**?

That is not just a writing exercise.

For EdgeLM, paper structure and submission strategy determine:

- what the project is actually claiming,
- which results are central versus supporting,
- which venue is the best fit,
- how narrow or broad the story should be,
- whether public preprints or code release create anonymization problems,
- and whether the current timeline is even realistic.

The existing implementation plan already has a rough outline for a paper, but it
is still too generic and too optimistic in a few important ways.

Most notably:

- it assumes venue fit without really distinguishing MLSys from EuroSys,
- it treats all technical ideas as if they belong in the main contribution set,
- it does not yet align the structure to real page limits,
- and it does not yet account for the exact 2027 submission rules now posted.

As of `2026-04-02`, the official venue situation is:

- **EuroSys 2027** has a live CFP with exact dates and submission rules.
- **MLSys 2027** does **not** appear to have an official CFP posted yet.
- **MLSys 2025** and **MLSys 2026** official calls are available and are useful
  for planning, but any MLSys 2027 schedule inference is still only an
  inference.

That timing reality matters a lot for EdgeLM.

## What the Deep Dive Already Covers

`deep-dive.md` is still empty, but the repo already contains a first-pass paper
idea in `implementation-plan.md`.

- The implementation plan already proposes a working title:
  - `Edge LM: Achieving 100+ Tokens/Second for 3B LLMs on Consumer Intel Hardware`
- It already sketches:
  - abstract topics,
  - a seven-part paper outline,
  - target venues (`MLSys 2027` or `EuroSys 2027`),
  - and an immediate `arXiv` preprint idea.
- Earlier research sections now provide the actual technical material that could
  populate the paper:
  - hardware characterization,
  - ternary-kernel design,
  - memory hierarchy analysis,
  - iGPU constraints,
  - speculative decoding strategy,
  - tokenizer/runtime correctness,
  - and a much stronger benchmarking methodology.

So the unresolved questions entering this section are no longer "what should the
paper talk about?" in the abstract. They are:

- What is the **single main thesis** of the paper?
- Which contributions belong in the main claim set, and which belong in the
  supporting details?
- Is EdgeLM a better **MLSys** paper or **EuroSys** paper?
- Which exact 2027 deadlines are realistic as of `2026-04-02`?
- How should the paper be structured under `10` pages versus `12` pages?
- What should the preprint/code/anonymization strategy be?
- And what evidence must exist before the title can honestly say `100+ tok/s`?

## New Findings

### 1. The paper needs one thesis, not a bag of optimizations

#### 1.1 Both MLSys and EuroSys are judging a paper-level claim, not a project diary

- **Source:** MLSys 2026 CFP; EuroSys 2027 CFP
- **Key idea:** Both venues emphasize novelty, impact/significance, correctness,
  and rigorous evaluation.
- **Relevance to EdgeLM:** This means the submission cannot read like "we built
  tokenizer + loader + kernels + iGPU + sampler + benchmark harness."
- **Estimated impact:** Very High.
- **Implementation complexity:** Low.
- **Details:** Reviewers do not accept papers because a lot of engineering
  happened. They accept papers because the work makes one coherent point that is
  technically important and convincingly demonstrated.

#### 1.2 The strongest EdgeLM thesis is "bandwidth-aware local LLM inference on consumer Intel hardware"

- **Source:** Project synthesis across Sections 1-24
- **Key idea:** The paper's center of gravity should be the memory wall and the
  system design forced by it.
- **Relevance to EdgeLM:** This is the thesis that ties the project together.
- **Estimated impact:** Very High.
- **Implementation complexity:** Low.
- **Details:** The cleanest main story is not:
  - "we optimized everything,"

  but rather:

  - consumer Intel laptops are fundamentally bandwidth-constrained for local LLM
    inference,
  - standard quantization paths do not clear that wall,
  - ternary-first design plus hardware-specific runtime choices do,
  - and the resulting system shows what shared-memory CPU+iGPU local inference
    should and should not do.

#### 1.3 The contribution set should be explicitly limited to four or five paper-grade claims

- **Source:** Project synthesis; venue norms
- **Key idea:** The main paper should present a small number of durable
  contributions, not every implementation detail as a co-equal contribution.
- **Relevance to EdgeLM:** This is essential for narrative clarity.
- **Estimated impact:** Very High.
- **Implementation complexity:** Low.
- **Details:** A strong contribution list would look more like:
  1. a quantitative characterization of the local inference bottleneck on the
     target consumer Intel platform,
  2. a ternary-first runtime and kernel design tuned for that bottleneck,
  3. a bandwidth-aware heterogeneous execution policy for shared-memory CPU+iGPU
     inference,
  4. a rigorous evaluation methodology and reproducible benchmark package,
  5. and optionally a speculative-decoding integration story if it is mature
     enough to be cleanly defended.

### 2. MLSys and EuroSys are both plausible, but they reward different versions of the story

#### 2.1 MLSys is an explicitly topical fit for EdgeLM

- **Source:** MLSys 2026 CFP
- **Key idea:** MLSys explicitly solicits work on:
  - efficient inference and serving,
  - LLM inference,
  - ML compilers and runtimes,
  - specialized hardware for ML,
  - hardware-efficient ML methods,
  - and ML benchmarks/tooling.
- **Relevance to EdgeLM:** This is almost a direct description of the project.
- **Estimated impact:** Very High.
- **Implementation complexity:** Low.

#### 2.2 EuroSys is also viable, but only if EdgeLM is framed as a systems contribution first

- **Source:** EuroSys 2027 CFP
- **Key idea:** EuroSys explicitly welcomes:
  - AI/ML systems,
  - systems for emerging hardware,
  - and experience papers,

  but it also strongly emphasizes rigorous systems evaluation and generalizable
  lessons.
- **Relevance to EdgeLM:** Venue fit is real, but the bar is different.
- **Estimated impact:** Very High.
- **Implementation complexity:** Low.
- **Details:** A EuroSys paper would need to foreground:
  - systems principles,
  - design tradeoffs,
  - failure modes,
  - and general lessons from shared-memory local inference,

  rather than centering purely on "LLM inference got faster."

#### 2.3 Default venue fit today: MLSys first, EuroSys conditional

- **Source:** MLSys 2026 CFP; EuroSys 2027 CFP; project scope synthesis
- **Key idea:** On current evidence, EdgeLM is more naturally an MLSys paper
  than a EuroSys paper.
- **Relevance to EdgeLM:** This affects the entire writing strategy.
- **Estimated impact:** Very High.
- **Implementation complexity:** Low.
- **Details:** The reason is not prestige. It is fit.

- **MLSys** is the more natural home if the final paper is mainly about:
  - efficient LLM inference,
  - ternary kernels,
  - runtime design for ML inference,
  - and benchmarking on a constrained target.
- **EuroSys** becomes more compelling if the final paper can clearly generalize
  beyond "our one custom engine" into:
  - system-level design rules,
  - heterogeneous runtime policy,
  - memory-management lessons,
  - and experience-paper style insights that matter to the broader systems
    community.

### 3. The calendar now matters: EuroSys 2027 spring is probably unrealistic, EuroSys 2027 fall and MLSys 2027 are the real decisions

#### 3.1 EuroSys 2027 spring has exact official deadlines, and they are close

- **Source:** EuroSys 2027 CFP
- **Key idea:** The official EuroSys 2027 spring deadlines are:
  - titles/abstracts due **Thursday, May 7, 2026**
  - full papers due **Thursday, May 14, 2026**
- **Relevance to EdgeLM:** As of `2026-04-02`, that is only about five to six
  weeks away.
- **Estimated impact:** Very High.
- **Implementation complexity:** Low.
- **Details:** Unless EdgeLM already had:
  - a mostly complete engine,
  - a stable evaluation suite,
  - and benchmark-quality results,

  this is not the realistic target.

#### 3.2 EuroSys 2027 fall is the first truly live conference target for this project

- **Source:** EuroSys 2027 CFP
- **Key idea:** The official EuroSys 2027 fall deadlines are:
  - titles/abstracts due **Thursday, September 17, 2026**
  - full papers due **Thursday, September 24, 2026**
- **Relevance to EdgeLM:** This is the first realistic fixed conference target.
- **Estimated impact:** Very High.
- **Implementation complexity:** Low.
- **Details:** That gives roughly six months from `2026-04-02`, which is still
  aggressive, but plausible if implementation and evaluation start producing
  paper-quality evidence by summer.

#### 3.3 MLSys 2027 does not yet have an official CFP posted, so only a schedule inference is possible today

- **Source:** MLSys 2025 CFP; MLSys 2026 CFP; absence of MLSys 2027 CFP as of
  `2026-04-02`
- **Key idea:** The last two official MLSys research deadlines were:
  - **October 31, 2024** for MLSys 2025
  - **October 30, 2025** for MLSys 2026
- **Relevance to EdgeLM:** This suggests, but does not prove, that MLSys 2027
  may again be a late-October 2026 submission.
- **Estimated impact:** Very High.
- **Implementation complexity:** Low.
- **Details:** This is an inference, not a published rule. But it is a useful
  planning baseline because it likely gives EdgeLM roughly one extra month over
  EuroSys 2027 fall.

### 4. Page limits materially change what the paper should look like

#### 4.1 MLSys gives a tighter main-paper budget than EuroSys

- **Source:** MLSys 2026 CFP; EuroSys 2027 CFP
- **Key idea:** MLSys currently uses:
  - up to **10 pages** of technical content plus references,

  while EuroSys 2027 allows:

  - up to **12 pages** of technical content plus references.
- **Relevance to EdgeLM:** This changes the shape of the paper.
- **Estimated impact:** Very High.
- **Implementation complexity:** Low.

#### 4.2 The current implementation-plan outline is too wide for either venue in its present form

- **Source:** `implementation-plan.md`
- **Key idea:** The plan currently sketches a paper with:
  - long background,
  - long system design,
  - many optimization subsections,
  - evaluation,
  - discussion,
  - and future work,

  which is broader than either venue's main-paper budget really allows.
- **Relevance to EdgeLM:** The paper needs compression and prioritization.
- **Estimated impact:** Very High.
- **Implementation complexity:** Low.

#### 4.3 The right workflow is "one 12-page master argument, then compress to the MLSys 10-page variant if needed"

- **Source:** Venue-rule synthesis
- **Key idea:** Since EuroSys allows slightly more room, the most practical
  writing workflow is to build one fuller draft and then compress.
- **Relevance to EdgeLM:** This reduces rewriting overhead.
- **Estimated impact:** High.
- **Implementation complexity:** Medium.
- **Details:** In practice:
  - the **12-page master draft** can carry slightly richer systems motivation,
    implementation detail, and discussion,
  - while the **10-page MLSys draft** should be the compressed,
    claim-maximized version with more material pushed into appendix.

### 5. The title and abstract must not overclaim

#### 5.1 The current working title is only valid if the strongest claim is literally achieved

- **Source:** `implementation-plan.md`; paper-claim synthesis
- **Key idea:** The working title says:
  - `Achieving 100+ Tokens/Second for 3B LLMs on Consumer Intel Hardware`
- **Relevance to EdgeLM:** That is a very specific and very attackable claim.
- **Estimated impact:** Very High.
- **Implementation complexity:** Low.
- **Details:** That title is appropriate only if the final paper really shows:
  - `100+ tok/s`,
  - on the actual `3B` target class,
  - on the target consumer Intel hardware,
  - under clearly defined measurement conditions,
  - without quietly switching from raw decode to speculative effective rate
    midway through the narrative.

#### 5.2 Raw decode, effective speculative throughput, and smaller-model results must be separated

- **Source:** Benchmarking methodology synthesis
- **Key idea:** These are not interchangeable numbers.
- **Relevance to EdgeLM:** Reviewers will absolutely test this distinction.
- **Estimated impact:** Very High.
- **Implementation complexity:** Low.
- **Details:** The paper must clearly distinguish:
  - **raw steady-state decode tok/s**
  - **effective throughput with speculative decoding**
  - **results on the 2.4B model**
  - **results on the 3.3B model**

  If those are conflated, the paper will look evasive.

#### 5.3 A safer title is one that states the system idea, not the final number

- **Source:** Paper-positioning synthesis
- **Key idea:** The title should center the real durable contribution.
- **Relevance to EdgeLM:** This reduces the risk of title-to-results mismatch.
- **Estimated impact:** High.
- **Implementation complexity:** Low.
- **Details:** Stronger title shapes would be:
  - `EdgeLM: Bandwidth-Aware Ternary LLM Inference on Consumer Intel Hardware`
  - `EdgeLM: A Ternary-First Runtime for Local LLM Inference on Consumer Intel Platforms`
  - `Breaking the Memory Wall for Local LLM Inference on Consumer Intel Hardware`

  Those leave room for the abstract and results section to state the actual
  throughput achieved, whatever it ends up being.

### 6. The paper should be structured around claim flow, not component inventory

#### 6.1 The main narrative should move from bottleneck to design principle to system to evidence

- **Source:** Systems-paper norms; project synthesis
- **Key idea:** The reader should be able to answer:
  - what is the problem,
  - why current approaches fail on this target,
  - what design principle fixes that,
  - what system embodies it,
  - and what evidence proves it.
- **Relevance to EdgeLM:** This is the right backbone for the paper.
- **Estimated impact:** Very High.
- **Implementation complexity:** Low.

#### 6.2 The paper should not be organized as "module by module through the source tree"

- **Source:** Writing-structure synthesis
- **Key idea:** A paper is not repo documentation.
- **Relevance to EdgeLM:** This is a common trap in systems projects.
- **Estimated impact:** Very High.
- **Implementation complexity:** Low.
- **Details:** That means the main paper should not allocate equal narrative
  weight to:
  - tokenizer,
  - loader,
  - thread pool,
  - sampler,
  - memory manager,
  - and each kernel file.

  Those belong only insofar as they support the central claim.

#### 6.3 The best main-paper structure is "few design principles, few key mechanisms, strong evaluation"

- **Source:** Project synthesis
- **Key idea:** EdgeLM will read strongest if the design is compressed into a
  small number of principles.
- **Relevance to EdgeLM:** This is how the many earlier sections become one
  coherent paper.
- **Estimated impact:** Very High.
- **Implementation complexity:** Low.
- **Details:** A good systems-design section could be structured around
  principles like:
  1. **ternary-first weight representation to clear the memory wall**
  2. **bandwidth-aware memory layout and prefetch**
  3. **explicit P-core/E-core role separation**
  4. **time-sliced shared-memory CPU+iGPU execution instead of naive overlap**
  5. **measurement-first runtime instrumentation**

### 7. Evaluation must dominate the back half of the paper

#### 7.1 Both venues put unusual weight on rigorous evaluation

- **Source:** EuroSys 2027 CFP; EuroSys 2026 CFP; MLSys 2026 CFP
- **Key idea:** EuroSys explicitly says rigorous evaluation is a hallmark of a
  good systems paper, and MLSys explicitly encourages reproducibility and
  benchmarking/tooling support.
- **Relevance to EdgeLM:** The evaluation section is not a formality here.
- **Estimated impact:** Very High.
- **Implementation complexity:** Low.

#### 7.2 EdgeLM needs an evaluation stack that supports the paper's actual claims

- **Source:** Section 24 benchmarking methodology
- **Key idea:** The evaluation must map directly to the contribution bullets.
- **Relevance to EdgeLM:** Otherwise the paper's story and evidence drift apart.
- **Estimated impact:** Very High.
- **Implementation complexity:** Medium.
- **Details:** The minimum serious evaluation package should include:
  - correctness gates,
  - prefill/decode/TTFT breakdowns,
  - same-checkpoint system comparisons where possible,
  - practical same-laptop baselines where useful,
  - ablations for each claimed optimization,
  - memory and power results if available,
  - and limitations when the iGPU path hurts instead of helps.

#### 7.3 The paper will be much stronger if it contains a small number of indispensable figures and tables

- **Source:** Paper-structure synthesis
- **Key idea:** Strong systems papers usually have a compact visual backbone.
- **Relevance to EdgeLM:** This helps the narrative stay disciplined.
- **Estimated impact:** High.
- **Implementation complexity:** Medium.
- **Details:** EdgeLM should plan around a figure/table set like:
  - **Figure 1:** system overview and execution roles
  - **Figure 2:** bandwidth/bottleneck characterization of the target laptop
  - **Table 1:** hardware, models, benchmark workloads, baselines
  - **Table 2:** end-to-end performance results
  - **Figure 3:** ablation waterfall or contribution breakdown
  - **Figure 4:** prefill/decode/TTFT phase breakdown
  - **Figure 5:** CPU-only vs CPU+iGPU vs speculative variants, if mature

### 8. Artifact and reproducibility should be planned before the paper is written, not after acceptance

#### 8.1 MLSys explicitly encourages artifact submission and reproducibility badging

- **Source:** MLSys 2026 CFP; MLSys 2026 Artifact Evaluation call
- **Key idea:** MLSys artifact evaluation is optional, but the conference
  explicitly encourages authors to pursue reproducibility badges, including
  `Results Reproduced`.
- **Relevance to EdgeLM:** This should influence engineering process now.
- **Estimated impact:** High.
- **Implementation complexity:** Medium.

#### 8.2 A hardware-specific local inference paper needs a two-tier artifact plan

- **Source:** Artifact-strategy synthesis
- **Key idea:** Not every evaluator will own the exact target laptop, but the
  paper still needs a credible artifact story.
- **Relevance to EdgeLM:** This is especially important because the system is
  tightly tuned to one Intel laptop class.
- **Estimated impact:** High.
- **Implementation complexity:** Medium.
- **Details:** The clean plan is:
  - **Tier 1: universal validation artifact**
    - build instructions,
    - reduced-size correctness tests,
    - microbenchmarks,
    - benchmark harness,
    - small sample data,
    - and scripts that run on broadly available x86 systems
  - **Tier 2: target-hardware reproduction package**
    - exact hardware manifest,
    - driver/compiler versions,
    - benchmark prompts,
    - raw results,
    - and full scripts for the i7-12700H + Iris Xe target

  This makes the artifact useful even when exact end-to-end numbers cannot be
  reproduced off-machine.

#### 8.3 The artifact appendix should be treated as part of the research output, not post-hoc packaging

- **Source:** MLSys 2026 AE call
- **Key idea:** MLSys AE asks for minimal hardware/software requirements,
  validation instructions, and expected results.
- **Relevance to EdgeLM:** That content should be prepared while experiments are
  being designed.
- **Estimated impact:** High.
- **Implementation complexity:** Medium.

### 9. Double-blind review, preprints, and naming strategy differ materially across the two venues

#### 9.1 MLSys allows arXiv/public posting while remaining double-blind

- **Source:** MLSys 2026 CFP
- **Key idea:** MLSys explicitly states that submissions are double-blind, but
  authors may post papers on arXiv or other public forums.
- **Relevance to EdgeLM:** This makes preprint timing relatively straightforward
  if MLSys becomes the primary target.
- **Estimated impact:** High.
- **Implementation complexity:** Low.

#### 9.2 EuroSys allows non-peer-reviewed public posting too, but adds a stricter anonymization wrinkle

- **Source:** EuroSys 2027 CFP
- **Key idea:** EuroSys says arXiv papers, technical reports, talks, and social
  media do not count as concurrent submission, but the submitted version should
  have a substantially different title and use a different system/tool name, if
  applicable.
- **Relevance to EdgeLM:** This is a major practical submission constraint.
- **Estimated impact:** Very High.
- **Implementation complexity:** Low.
- **Details:** For EdgeLM, that means:
  - if EuroSys is the primary target,
  - a public preprint titled `EdgeLM: ...` before submission can create real
    anonymization friction,
  - and the paper may need either a different public identity or a delayed named
    release strategy.

#### 9.3 For EuroSys, public code links during submission must also avoid identity leaks

- **Source:** EuroSys 2027 CFP
- **Key idea:** EuroSys explicitly requires repository links and contents not to
  reveal author identity or affiliation during submission.
- **Relevance to EdgeLM:** This affects repo preparation.
- **Estimated impact:** High.
- **Implementation complexity:** Medium.
- **Details:** If EuroSys is the target, the project should be ready to:
  - scrub names and affiliations from the submission-facing repo,
  - separate public-dev history from anonymized review artifact if necessary,
  - and avoid leaving identifying breadcrumbs in README text, commit metadata
    references, or issue links.

### 10. AI-assistance policy is now an explicit part of venue compliance

#### 10.1 EuroSys 2027 explicitly requires AI-tool use disclosure

- **Source:** EuroSys 2027 CFP
- **Key idea:** EuroSys 2027 specifically highlights that ACM authorship policy
  requires any use of AI tools such as ChatGPT to be fully disclosed in the
  submission.
- **Relevance to EdgeLM:** This matters directly to the writing workflow.
- **Estimated impact:** High.
- **Implementation complexity:** Low.

#### 10.2 MLSys 2026 allows consultation with LLMs but prohibits papers generated by them alone

- **Source:** MLSys 2026 CFP
- **Key idea:** MLSys explicitly states that authors may consult LLMs while
  writing, but papers whose content is generated by LLMs alone are prohibited
  unless used illustratively or as part of experimental analysis.
- **Relevance to EdgeLM:** This is the current official MLSys guidance available.
- **Estimated impact:** High.
- **Implementation complexity:** Low.

#### 10.3 EdgeLM should keep a simple authorship-compliance log

- **Source:** Venue-policy synthesis
- **Key idea:** If AI assistance is used for editing, summarization, or drafting,
  it is better to have a lightweight record than to reconstruct it later.
- **Relevance to EdgeLM:** This reduces policy risk near submission time.
- **Estimated impact:** Medium.
- **Implementation complexity:** Low.
- **Details:** A simple private note can record:
  - where AI tools were used,
  - whether the output was edited by the authors,
  - and what final disclosure statement may be needed.

### 11. EdgeLM should maintain two venue-framed paper variants, but choose one primary submission path

#### 11.1 The MLSys version should be ML-inference-centric

- **Source:** MLSys CFP topics
- **Key idea:** For MLSys, the paper should foreground:
  - inference efficiency,
  - ternary runtime design,
  - hardware-efficient inference methods,
  - and benchmark methodology.
- **Relevance to EdgeLM:** This is the natural primary variant.
- **Estimated impact:** Very High.
- **Implementation complexity:** Medium.

#### 11.2 The EuroSys version should be systems-principles-centric

- **Source:** EuroSys 2027 CFP
- **Key idea:** For EuroSys, the paper should foreground:
  - bottleneck characterization,
  - heterogeneous runtime policy,
  - memory/scheduling tradeoffs,
  - and general lessons from shared-memory local inference.
- **Relevance to EdgeLM:** This is the only way the paper feels native to the
  venue rather than transplanted.
- **Estimated impact:** Very High.
- **Implementation complexity:** Medium.

#### 11.3 But the project should still pick one primary target by late summer 2026

- **Source:** Timeline synthesis
- **Key idea:** Trying to optimize the same draft equally for both venues for too
  long will slow the project down.
- **Relevance to EdgeLM:** The writeup needs one dominant center of gravity.
- **Estimated impact:** High.
- **Implementation complexity:** Low.
- **Details:** A practical decision rule is:
  - if by **August 2026** the story is mostly about efficient inference and
    strong benchmark results on the target platform, choose **MLSys**;
  - if by **August 2026** there are stronger cross-cutting systems lessons and a
    clearer general runtime story, consider **EuroSys 2027 fall**.

### 12. The final submission strategy should be milestone-driven, not wish-driven

#### 12.1 EdgeLM should treat EuroSys 2027 fall as a stretch deadline and MLSys 2027 as the default planning baseline

- **Source:** Current date plus official/inferred schedules
- **Key idea:** This is the most realistic planning posture on `2026-04-02`.
- **Relevance to EdgeLM:** It prevents premature writing before results exist.
- **Estimated impact:** Very High.
- **Implementation complexity:** Low.

#### 12.2 The submission decision should be gated by concrete readiness criteria

- **Source:** Project synthesis
- **Key idea:** Venue choice should depend on evidence, not hope.
- **Relevance to EdgeLM:** This keeps the project honest.
- **Estimated impact:** Very High.
- **Implementation complexity:** Low.
- **Details:** Before committing to a deadline, EdgeLM should already have:
  - a correct end-to-end engine,
  - a stable benchmark harness,
  - at least one strong same-checkpoint or clearly labeled baseline comparison,
  - a convincing ablation set,
  - polished figures/tables,
  - and a credible artifact plan.

## Comparative Fit for EdgeLM

| Venue / Track | Strength | Weakness | Best if | EdgeLM verdict |
| --- | --- | --- | --- | --- |
| MLSys 2027 Research Track | Direct topical fit for inference, runtimes, hardware-efficient ML, LLM serving | Tighter `10`-page budget; official 2027 CFP not yet posted | Final story is primarily about efficient LLM inference and runtime design | Best default target |
| EuroSys 2027 Research Track | Strong systems prestige; explicit AI/ML systems scope; `12` pages | Higher pressure for general systems lessons and broader design significance | Final story is a systems paper first, with durable lessons beyond one engine | Strong conditional target |
| EuroSys 2027 Experience Paper | Welcomes lessons from practical deployment with quantitative support | Still needs general lessons and rigorous analysis; weaker fit if novelty is the main claim | The final result is more about engineering lessons than new runtime ideas | Conditional fallback only |
| arXiv / non-archival workshop | Fast feedback and public timestamping | Not a substitute for conference acceptance; can complicate EuroSys naming/anonymity | The project wants early visibility or workshop discussion | Useful side channel, not the main goal |

## Recommendations for EdgeLM

### 1. Make the central paper claim about the memory wall and the system design forced by it

- Do not present EdgeLM as a random collection of optimizations.
- Present it as a bandwidth-constrained local inference problem with a
  ternary-first systems solution.

### 2. Treat MLSys as the default primary venue and EuroSys as a conditional alternative

- MLSys is currently the most natural fit.
- EuroSys becomes attractive only if the paper develops stronger generalized
  systems lessons and broader runtime insights.

### 3. Do not target EuroSys 2027 spring unless a full results-quality paper exists by late April 2026

- The official spring deadlines are **May 7, 2026** for title/abstract and
  **May 14, 2026** for the full paper.
- On the current project timeline, that should be treated as unrealistic unless
  the implementation is much further along than the repo currently suggests.

### 4. Use a safer title until the final throughput claim is proven

- Avoid locking the paper into `100+ tok/s for 3B` until the exact measurement
  claim is real.
- Prefer a title centered on `bandwidth-aware`, `ternary-first`, or `local LLM
  inference` as the durable contribution.

### 5. Maintain one fuller draft and two framing variants

- Keep one master paper argument and figure set.
- Maintain:
  - an **MLSys framing** centered on efficient inference/runtime design
  - a **EuroSys framing** centered on systems principles and local heterogeneous
    execution

### 6. Make evaluation the largest section of the paper

- The paper should win on evidence, not on adjectives.
- Plan the figures, tables, and ablations before writing prose.

### 7. Build the artifact package in parallel with experimentation

- Save exact configs, prompts, logs, benchmark JSON, and environment manifests.
- Prepare for both a lightweight validation artifact and a target-hardware
  reproduction bundle.

### 8. If EuroSys is the primary target, delay or sanitize public naming before submission

- EuroSys explicitly allows public preprints, but the submitted version should
  have a substantially different title and tool name if applicable.
- That means a public `EdgeLM` preprint before submission is not a neutral
  decision.

### 9. Track AI-assisted writing usage now

- EuroSys 2027 requires disclosure of AI-tool use.
- MLSys 2026 permits consultation but prohibits papers generated by LLMs alone.
- Keep a simple compliance log rather than reconstructing later.

## Suggested EdgeLM Paper Structure

### A. Core contribution statement

The paper should be able to summarize itself in four sentences:

1. Consumer Intel laptops are fundamentally bandwidth-limited for local LLM
   inference.
2. A ternary-first runtime changes the feasible operating point in a way that
   standard `Q4` paths do not.
3. The right local execution policy on shared-memory CPU+iGPU hardware is
   bandwidth-aware and not naive overlap.
4. EdgeLM demonstrates these principles with a custom C runtime and a rigorous,
   reproducible evaluation.

### B. Recommended section order for the MLSys-shaped paper

Because MLSys currently allows only `10` pages of technical content, the paper
should be compressed around the main claim:

1. **Introduction**
   - Local inference matters.
   - Consumer Intel hardware is bottlenecked by memory bandwidth.
   - EdgeLM's thesis and contributions.
2. **Problem Setting and Bottleneck Analysis**
   - Target hardware.
   - Why standard quantization is bottlenecked.
   - Why ternary changes the design space.
3. **EdgeLM Design**
   - Design principles only.
   - Ternary-first runtime.
   - Bandwidth-aware scheduling.
4. **Implementation**
   - Kernels, layout, memory management, orchestration.
   - Only the mechanisms needed to defend the claim.
5. **Evaluation**
   - Methodology.
   - Main results.
   - Ablations.
   - Failure cases / limitations.
6. **Discussion and Limitations**
   - Hardware specificity.
   - When iGPU helps or hurts.
   - Generality boundaries.
7. **Conclusion**

### C. Recommended section order for the EuroSys-shaped paper

Because EuroSys allows `12` pages and expects a stronger systems arc, the paper
can spend more room on design rationale and lessons:

1. **Introduction**
2. **Motivation and Bottleneck Characterization**
3. **Design Principles for Shared-Memory Local LLM Inference**
4. **EdgeLM Architecture**
5. **Implementation Details**
6. **Evaluation**
7. **Lessons and Limitations**
8. **Related Work**
9. **Conclusion**

### D. Suggested figure/table inventory

1. `Figure 1`: EdgeLM architecture overview.
2. `Figure 2`: bandwidth-driven bottleneck model for the target hardware.
3. `Table 1`: hardware, models, token workloads, and baselines.
4. `Table 2`: main end-to-end performance results.
5. `Figure 3`: ablation breakdown by optimization tier.
6. `Figure 4`: phase-separated timing breakdown (`prefill`, `TTFT`, `decode`).
7. `Figure 5`: CPU-only vs CPU+iGPU vs speculative results, if mature.

## Suggested Submission Playbook

### Near-term timeline from the current date (`2026-04-02`)

1. **April-May 2026**
   - Finish correctness-first minimal engine and benchmark harness.
2. **June-July 2026**
   - Stabilize kernels, scheduling, and evaluation methodology.
   - Start drafting figures and contribution language.
3. **August 2026**
   - Decide primary venue based on actual results quality and narrative fit.
4. **September 2026**
   - If strong systems story and polished evaluation are ready, consider
     **EuroSys 2027 fall** (`September 17/24, 2026`).
5. **October 2026**
   - If MLSys 2027 follows the previous pattern, this is the likely conference
     submission window. This is an inference, not a confirmed date.

### Submission readiness checklist

- Correct end-to-end engine on target models.
- Stable benchmark suite with saved raw JSON results.
- Clean baseline comparisons and ablations.
- Finalized claim wording that matches actual results.
- Draft figures/tables.
- Anonymization plan.
- Artifact plan.
- Venue-specific compliance notes:
  - preprints,
  - code links,
  - AI-assistance disclosure,
  - authorship/conflicts.

## Bottom Line

EdgeLM is now at the point where the paper cannot just be "about the whole
project." It has to be about one central systems claim.

The strongest current claim is:

- local LLM inference on consumer Intel hardware is fundamentally constrained by
  memory bandwidth,
- and a ternary-first, bandwidth-aware runtime is the right systems response to
  that constraint.

The venue picture is also much clearer now.

As of `2026-04-02`:

- **EuroSys 2027 spring** is almost certainly too soon.
- **EuroSys 2027 fall** is the first realistic fixed conference deadline.
- **MLSys 2027** is not yet officially posted, but based on `2025` and `2026`
  official calls, late `October 2026` is the most plausible planning baseline.

So the practical recommendation is:

- treat **MLSys 2027** as the default primary target,
- treat **EuroSys 2027 fall** as a conditional stretch target if the systems
  story becomes especially strong,
- use a safer claim-centered title until the exact throughput result is proven,
- and build the paper around a small set of durable contributions backed by
  unusually rigorous evaluation and a credible artifact package.

If EdgeLM follows that path, the final paper can be more than a fast demo.

It can become a publishable argument about what local LLM inference should look
like on the kind of hardware most people actually own.

## Sources

- MLSys 2026 Call for Research Papers:
  `https://mlsys.org/Conferences/2026/CallForResearchPapers`
- MLSys 2026 conference overview and dates:
  `https://mlsys.org/Conferences/2026`
- MLSys 2026 Call for Artifact Evaluations:
  `https://mlsys.org/Conferences/2026/CallForAEs`
- MLSys 2025 Call for Papers:
  `https://mlsys.org/Conferences/2025/CallForPapers`
- MLSys 2025 dates:
  `https://mlsys.org/Conferences/2025/Dates`
- EuroSys 2027 Call for Papers:
  `https://2027.eurosys.org/cfp.html`
- EuroSys 2026 Call for Papers:
  `https://2026.eurosys.org/cfp.html`

## Audit Addendum (2026-04-02)

- **A reviewer-risk register would be useful before drafting.** EdgeLM should
  proactively list likely objections such as:
  - hardware specificity,
  - comparison fairness,
  - title/result mismatch,
  - and generality beyond one laptop.
- **Rebuttal preparation should begin with the benchmark plan, not after reviews
  arrive.** Many likely reviewer questions can be anticipated directly from the
  evaluation design.
- **A workshop or poster path can be used strategically without replacing the
  main conference target.** If the project wants early feedback, that path
  should be chosen in a way that does not accidentally undermine the preferred
  blind-review strategy.
