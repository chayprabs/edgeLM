You are performing deep research for the EdgeLM project -- a custom inference engine
targeting 100+ tok/s on 3B LLMs on consumer Intel hardware (i7-12700H, 16GB DDR4,
Intel Iris Xe, no dGPU).

## Step 1: Determine the next section

Read `research-progress.md` in the project root. Find the FIRST line with `- [ ]` (unchecked).
Extract:
- The section number (NN)
- The section title
- The output file path

If ALL sections are checked `[x]`, tell the user all research is complete and stop.

## Step 2: Read the existing deep dive content for this section

Read the corresponding section from `deep-dive.md`. Also read the relevant
parts of `implementation-plan.md` for additional context on how this section fits into the
overall optimization strategy.

Also check `research-papers-data.json` for any entries relevant to this section's topic.

## Step 3: Perform deep internet research

Use WebSearch and WebFetch to find EVERY possible optimization approach, technique,
paper, library, and implementation strategy related to this section that goes BEYOND
what is already covered in the deep dive. Research specifically for:

1. **Recent papers (2024-2026)** -- new techniques published after the deep dive was written
2. **Open-source implementations** -- GitHub repos with relevant optimized code
3. **Hardware-specific tricks** -- Intel Alder Lake / Golden Cove / Gracemont specific optimizations
4. **Benchmark data** -- real-world performance numbers on similar hardware
5. **Alternative approaches** -- methods the deep dive mentions briefly or misses entirely
6. **Community findings** -- blog posts, forum threads, HackerNews discussions with practical insights
7. **Adjacent domains** -- techniques from audio/video/signal processing that could apply

For each area, explore at least 3-5 different search queries to ensure comprehensive coverage.
Do NOT just confirm what the deep dive already says -- the goal is to find NEW information.

## Step 4: Write the research file

Write a well-organized markdown file to the path from Step 1 (inside the `research/` directory).

Use this structure:

```
# Section NN: [Section Title] -- Extended Research

## Overview
Brief summary of what this section covers and why it matters for EdgeLM.

## What the Deep Dive Already Covers
Bullet-point summary of key points from deep-dive.md for this section.
(Keep brief -- this is just for context, not a copy of the deep dive.)

## New Findings

### [Topic Area 1]
#### [Specific technique/paper/approach]
- **Source:** [URL or paper citation]
- **Key idea:** [1-2 sentence summary]
- **Relevance to EdgeLM:** [How this applies to our specific hardware/goals]
- **Estimated impact:** [Performance improvement estimate if applicable]
- **Implementation complexity:** [Low/Medium/High]
- **Details:** [Deeper explanation, code snippets if relevant]

### [Topic Area 2]
...

## Techniques Comparison Matrix
| Technique | Source | Impact | Complexity | Already in Deep Dive? |
|-----------|--------|--------|------------|----------------------|
| ...       | ...    | ...    | ...        | Yes/No/Partially     |

## Recommendations for EdgeLM
Ordered list of the most promising findings, ranked by impact-to-effort ratio,
with specific guidance on how to integrate them into the EdgeLM engine.

## References
Numbered list of all sources consulted.
```

## Step 5: Update research-progress.md

Change the completed section's line from `- [ ]` to `- [x]`.
Update the **Completed** and **Remaining** counts in the Status section.
Update the **Last updated** date to today.

## Important guidelines

- Be EXHAUSTIVE. The goal is to find things the deep dive missed, not to summarize it.
- Every claim should have a source URL or paper citation.
- Focus on what is actionable for our specific hardware (i7-12700H, DDR4-3200, no AVX-512).
- Do not include techniques that require hardware we do not have (AVX-512, AMX, CUDA, etc.)
  unless noting them as "not applicable but worth understanding why."
- Search in multiple languages/communities if relevant (Chinese ML community often has
  cutting-edge optimization work).
- Minimum 10 substantive new findings per section. If you cannot find 10, explain why
  and what you searched for.
- STOP after completing the single section. Do NOT continue to the next section.
