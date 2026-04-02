# Section 21: Memory Management (Large Pages / Arena Allocator) -- Extended Research

## Overview

The memory manager is one of the quietest but most consequential subsystems in
EdgeLM.

It is easy to reduce Section 21 to one slogan:

- "use `VirtualAlloc` with `MEM_LARGE_PAGES`."

That is directionally correct, but it is far too shallow for the real system
design problem.

For EdgeLM, the memory manager has to solve all of the following at once:

- keep the model inside a practical `6-7 GB` inference budget on a `16 GB`
  Windows laptop,
- reduce TLB and page-fault overhead on bandwidth-bound weight streaming,
- avoid allocator churn during autoregressive decode,
- preserve strict alignment guarantees for AVX2/VNNI kernels,
- keep startup predictable even when large pages are unavailable,
- integrate with GGUF loading and repacking,
- support fixed-size long-lived arenas such as the KV cache,
- and leave enough flexibility for future features like speculative decoding,
  hybrid CPU+iGPU work, and possibly sliding-window contexts.

So this section is not a generic allocator overview. It is a design study for a
**Windows-native memory manager for a custom LLM inference engine**.

The right answer for EdgeLM is not one allocator. It is a small hierarchy:

- a handful of top-level virtual-memory regions allocated with Win32 APIs,
- large pages where they materially help and can be obtained safely,
- monotonic arena allocation for sub-buffers,
- a strict "no OS allocation in the token loop" rule,
- and an explicit fallback plan when privilege or fragmentation prevents large
  pages.

## What the Deep Dive Already Covers

`deep-dive.md` remains empty, but the project already has strong local guidance
from `implementation-plan.md`, `AGENTS.md`, and earlier research.

- `AGENTS.md` already names the memory manager as a first-class component:
  `VirtualAlloc`, `2MB` large pages, and arena allocation are all explicit
  requirements.
- The implementation plan already says:
  - use `VirtualAlloc` with `MEM_LARGE_PAGES` for weight buffers,
  - use `NUMA`-aware allocation where helpful,
  - and keep runtime buffers contiguous and `64-byte` aligned.
- Section 03 established the microarchitectural case for large pages on Alder
  Lake:
  - large pages dramatically reduce TLB pressure,
  - the Windows interface is `VirtualAlloc + MEM_LARGE_PAGES`,
  - and large-page allocation should happen once at startup rather than
    repeatedly.
- Section 05 established that model-loading and repacking are already likely to
  involve a two-stage path:
  - map or read model assets,
  - then write a runtime-friendly layout.
- Section 11 already applied large-page thinking to the KV cache and recommended
  contiguous arena-style allocation rather than fragmented per-layer buffers.
- Section 18 established a broader systems rule for this laptop: shared memory
  bandwidth is the central constraint, and the engine should prefer predictable,
  low-overhead orchestration over optimistic dynamic behavior.

So the unresolved questions entering this section are not "are large pages good?"
That part is already well supported.

The real unresolved questions are:

- Which allocations should use large pages, and which should not?
- How should EdgeLM balance `reserve/commit` flexibility against large-page
  restrictions?
- Can model files be mapped directly into large pages?
- What should the fallback path be when large pages fail?
- Is a single monolithic arena enough, or should the engine use several?
- How should transient load-time buffers differ from decode-time buffers?
- And which Windows APIs are worth using in v1 versus only later?

## New Findings

### 1. The memory manager is a policy engine, not just an allocation wrapper

#### 1.1 For EdgeLM, memory policy affects throughput, latency, and correctness

- **Source:** `implementation-plan.md`; prior EdgeLM memory and KV-cache
  research; Windows virtual-memory docs
- **Key idea:** The memory manager decides much more than where bytes live. It
  defines when memory is committed, how buffers are aligned, whether decode can
  run without page-management interference, and whether startup is stable across
  privilege and fragmentation conditions.
- **Relevance to EdgeLM:** This is why Section 21 belongs among the core systems
  sections rather than among minor utilities.
- **Estimated impact:** Very High.
- **Implementation complexity:** Medium.
- **Details:** In a custom inference engine, memory policy influences:
  - TLB behavior,
  - page-fault behavior,
  - prefetch effectiveness,
  - inter-thread contention,
  - and whether the runtime keeps making expensive OS calls after startup.

  EdgeLM therefore needs an explicit memory strategy, not scattered
  `malloc/free` replacements.

#### 1.2 The right design is phase-aware

- **Source:** Synthesis of project architecture and Windows allocation
  semantics
- **Key idea:** EdgeLM has three distinct memory regimes:
  1. **load/import time**
  2. **steady-state inference**
  3. **shutdown / optional reclamation**
- **Relevance to EdgeLM:** Those phases want different policies.
- **Estimated impact:** High.
- **Implementation complexity:** Medium.
- **Details:** Load-time can tolerate:
  - file mappings,
  - copies,
  - staging buffers,
  - and validation work.

  Steady-state decode wants the opposite:
  - fixed regions,
  - no OS allocator churn,
  - predictable address stability,
  - and zero fragmentation-sensitive calls.

### 2. Windows reserve/commit semantics should directly shape the EdgeLM allocator API

#### 2.1 Standard `VirtualAlloc` gives EdgeLM elastic reserve/commit behavior

- **Source:** Microsoft Learn `VirtualAlloc`
- **Key idea:** `VirtualAlloc` separates `MEM_RESERVE` from `MEM_COMMIT`, and
  committed pages are zero-initialized but physical pages are not allocated
  until the address is actually touched.
- **Relevance to EdgeLM:** This is useful for buffers whose final active size may
  vary, or for regions that EdgeLM wants to reserve early but populate later.
- **Estimated impact:** High.
- **Implementation complexity:** Low.
- **Details:** Microsoft documents that `MEM_RESERVE` claims virtual address
  space without allocating backing storage, while `MEM_COMMIT` consumes commit
  charge and guarantees zeroed contents on first access. This makes normal pages
  attractive for:
  - elastic request-local arenas,
  - future variable-context buffers,
  - and advanced placeholder-based layouts.

#### 2.2 Large pages remove much of that elasticity in exchange for better translation behavior

- **Source:** Microsoft Learn `Large-Page Support`; `VirtualAlloc`
- **Key idea:** `MEM_LARGE_PAGES` requires `MEM_RESERVE | MEM_COMMIT` together,
  and size/alignment must be a multiple of `GetLargePageMinimum()`.
- **Relevance to EdgeLM:** This is the central API tradeoff of the whole
  section.
- **Estimated impact:** Very High.
- **Implementation complexity:** Low.
- **Details:** With standard pages, EdgeLM can reserve first and commit later.
  With large pages, it cannot. That means large pages fit **known, long-lived,
  mostly fixed-size regions** much better than elastic or speculative ones.

### 3. Large pages are strong medicine: high value, but only for the right buffers

#### 3.1 Microsoft's large-page contract is stricter than many summaries imply

- **Source:** Microsoft Learn `Large-Page Support`
- **Key idea:** Microsoft documents that large-page memory:
  - requires `SeLockMemoryPrivilege`,
  - must be allocated in large-page multiples,
  - should be allocated once at startup,
  - is always read/write,
  - is nonpageable,
  - is part of process private bytes but not the working set,
  - is not subject to job limits,
  - and must be reserved and committed in one operation.
- **Relevance to EdgeLM:** Almost every one of those bullets changes design
  decisions.
- **Estimated impact:** Very High.
- **Implementation complexity:** Medium.
- **Details:** Two especially important consequences are easy to miss:
  1. **Large-page weights cannot rely on read-only page protection** for debug
     safety, because Microsoft explicitly describes large-page memory as always
     read/write.
  2. **Large pages are physical-memory policy**, not just address-space policy,
     because they are nonpageable and require contiguous physical backing.

#### 3.2 Large pages are best for large, frequently reused, CPU-hot regions

- **Source:** Microsoft large-page docs; prior EdgeLM Section 03 and Section 11
  findings
- **Key idea:** The allocations that benefit most are the ones that are:
  - large,
  - long-lived,
  - accessed repeatedly,
  - and central to decode throughput.
- **Relevance to EdgeLM:** This gives a clean prioritization rule.
- **Estimated impact:** Very High.
- **Implementation complexity:** Low.
- **Details:** For EdgeLM, that points to:
  - model weights,
  - KV cache,
  - and possibly the main activation arena.

  It does **not** automatically point to:
  - tokenizer metadata,
  - small control structures,
  - or every transient helper buffer.

### 4. Large-page allocation failure is normal enough that EdgeLM needs a first-class fallback path

#### 4.1 Privilege enablement is necessary but not sufficient

- **Source:** Microsoft Learn `Large-Page Support`; Microsoft Learn
  `AdjustTokenPrivileges`
- **Key idea:** Large pages require `SeLockMemoryPrivilege`, and
  `AdjustTokenPrivileges` can only enable a privilege that already exists in the
  token. It cannot grant a missing privilege on the fly.
- **Relevance to EdgeLM:** A runtime call can enable the privilege if the user
  already has it, but EdgeLM must still expect systems where the privilege is
  absent.
- **Estimated impact:** Very High.
- **Implementation complexity:** Low.
- **Details:** This means the engine should distinguish three states:
  1. privilege available and enabled
  2. privilege absent
  3. privilege present but allocation still fails

  The second and third cases must both have graceful fallback.

#### 4.2 Physical fragmentation is an independent failure mode

- **Source:** Microsoft Learn `Large-Page Support`
- **Key idea:** Microsoft warns that large-page memory can become difficult to
  obtain after the system has been running for a long time because contiguous
  physical regions become scarce.
- **Relevance to EdgeLM:** This is why large-page support must be treated as an
  optimization with fallback, not as an unconditional startup requirement.
- **Estimated impact:** High.
- **Implementation complexity:** Low.
- **Details:** This is especially important on a consumer laptop that may:
  - wake from sleep repeatedly,
  - have browsers and IDEs already running,
  - and have less predictable memory pressure than a clean benchmark server.

#### 4.3 The correct fallback is normal `VirtualAlloc`, not allocator panic

- **Source:** Microsoft virtual-memory docs; EdgeLM project goals
- **Key idea:** If large pages fail, EdgeLM should fall back to regular committed
  pages with the same region topology and alignment guarantees.
- **Relevance to EdgeLM:** This preserves functionality while isolating the
  performance delta to the paging mode.
- **Estimated impact:** Very High.
- **Implementation complexity:** Low.
- **Details:** A good fallback policy is:
  - attempt large pages for eligible regions,
  - on failure log the reason,
  - allocate the same region with normal pages,
  - prefetch or pre-touch it as appropriate,
  - and continue.

  This keeps the benchmarking story clean because the main changed variable is
  page size, not the entire region layout.

### 5. EdgeLM should not try to memory-map model files directly into large pages

#### 5.1 Windows large-page file mappings are not a drop-in solution for GGUF files

- **Source:** Microsoft Learn `CreateFileMapping`; `MapViewOfFile`
- **Key idea:** `SEC_LARGE_PAGES` and `FILE_MAP_LARGE_PAGES` apply to large-page
  file mappings, but Microsoft documents that `SEC_LARGE_PAGES` is supported
  only for paging-file-backed section objects, not normal data files.
- **Relevance to EdgeLM:** This answers an important design question directly.
- **Estimated impact:** Very High.
- **Implementation complexity:** Low.
- **Details:** EdgeLM therefore cannot simply:
  - open a GGUF file,
  - map it,
  - and expect the mapped bytes to land in `2 MB` large pages.

  That path is not the right abstraction on Windows for ordinary model files.

#### 5.2 The practical Windows path is "map/read normally, then copy or repack into runtime buffers"

- **Source:** Microsoft file-mapping docs; prior EdgeLM GGUF research
- **Key idea:** The runtime-friendly solution is two-stage:
  1. load model data using normal file mapping or buffered/direct reads
  2. copy/repack into a pagefile-backed large-page runtime layout
- **Relevance to EdgeLM:** This aligns perfectly with the project's existing
  repacked-cache direction.
- **Estimated impact:** Very High.
- **Implementation complexity:** Medium.
- **Details:** This is one of the strongest arguments for the custom repacked
  cache format discussed in Section 05. The repacked runtime artifact can be
  laid out exactly as the CPU kernels want, then copied into the final weights
  region once at startup.

### 6. `PrefetchVirtualMemory` is a useful complement, but not a replacement, for large pages

#### 6.1 Windows provides an explicit prefetch API for virtual address ranges

- **Source:** Microsoft Learn `PrefetchVirtualMemory`
- **Key idea:** `PrefetchVirtualMemory` lets a process ask Windows to bring one
  or more virtual ranges into memory efficiently, including discontiguous
  regions.
- **Relevance to EdgeLM:** This is useful in two concrete places:
  - after normal-page fallback allocations
  - and after mapping model files for load/repack
- **Estimated impact:** High.
- **Implementation complexity:** Low.
- **Details:** This is especially attractive because it is a hint-oriented API
  for exactly the kind of sequential or chunked memory bring-up EdgeLM already
  wants during startup.

#### 6.2 But prefetching and large pages solve different problems

- **Source:** Microsoft Learn `PrefetchVirtualMemory`; `Large-Page Support`
- **Key idea:** Prefetching improves residency and reduces first-touch stalls.
  Large pages reduce translation overhead and page-boundary churn.
- **Relevance to EdgeLM:** They are complementary, not interchangeable.
- **Estimated impact:** High.
- **Implementation complexity:** Low.
- **Details:** So the right policy is:
  - **large pages available:** allocate eligible long-lived regions with large
    pages
  - **large pages unavailable:** use normal pages and call
    `PrefetchVirtualMemory` or perform explicit page touching
  - **mapped file input:** prefetch the mapped input before or during repack

  That is much better than treating prefetch as a substitute for page-size
  control.

### 7. Arena allocation is an unusually strong fit for LLM inference workloads

#### 7.1 A monotonic/bump allocator matches EdgeLM's steady-state allocation pattern

- **Source:** LLVM `BumpPtrAllocator` documentation; project workload synthesis
- **Key idea:** LLVM describes a bump-pointer allocator as a monotonically
  growing pool where each allocation is found by taking the next bytes from the
  current slab or the next slab.
- **Relevance to EdgeLM:** That is almost exactly the pattern of request-local
  scratch and many inference-time temporaries.
- **Estimated impact:** Very High.
- **Implementation complexity:** Low.
- **Details:** In EdgeLM, many buffers are:
  - created in a burst,
  - reused for a layer or request,
  - then discarded all at once.

  That is the ideal case for arenas because it turns many general allocations
  into:
  - pointer bump,
  - optional alignment round-up,
  - and bulk reset.

#### 7.2 Arena semantics are more important than a specific library

- **Source:** LLVM allocator docs; project zero-dependency constraint
- **Key idea:** EdgeLM does not need LLVM itself. It needs the **allocator
  semantics**:
  - monotonic growth,
  - optional slab separation for large sub-allocations,
  - and bulk lifetime reset.
- **Relevance to EdgeLM:** This makes arena allocation easy to implement in pure
  C without taking on a dependency.
- **Estimated impact:** High.
- **Implementation complexity:** Low.
- **Details:** The simplest EdgeLM arena API can look like:

```c
typedef struct {
    uint8_t *base;
    size_t   size;
    size_t   head;
} arena_t;

void *arena_alloc(arena_t *a, size_t bytes, size_t align);
void  arena_reset(arena_t *a);
```

  That is enough for a large fraction of the runtime.

### 8. EdgeLM should use several arenas, not one global "everything heap"

#### 8.1 The top-level memory layout should reflect buffer lifetime, not module ownership

- **Source:** Prior EdgeLM memory and KV-cache research; Windows allocation
  semantics
- **Key idea:** A useful partition is by lifetime and mutability:
  - immutable long-lived weights
  - long-lived KV arenas
  - reused activation scratch
  - transient load/import scratch
  - tiny metadata/control allocations
- **Relevance to EdgeLM:** This leads to a simpler and more robust memory API.
- **Estimated impact:** Very High.
- **Implementation complexity:** Medium.
- **Details:** A module-centric design like "tokenizer allocates its own stuff,
  loader allocates its own stuff, attention allocates its own stuff" is much
  more likely to reintroduce fragmentation and uncontrolled OS allocations.

#### 8.2 A good memory manager exposes named regions, not a generic opaque heap

- **Source:** Project architecture synthesis
- **Key idea:** The runtime should know which region a buffer belongs to.
- **Relevance to EdgeLM:** This matters for:
  - diagnostics,
  - fallback reporting,
  - benchmarking,
  - and future policy tuning.
- **Estimated impact:** High.
- **Implementation complexity:** Medium.
- **Details:** A practical top-level structure is:

```c
typedef struct {
    void   *base;
    size_t  reserved;
    size_t  committed;
    int     page_mode;   // normal / large
} vm_region_t;

typedef struct {
    vm_region_t weights;
    vm_region_t kv_k;
    vm_region_t kv_v;
    vm_region_t act;
    vm_region_t load_scratch;
    arena_t     act_arena;
    arena_t     request_arena;
} memory_manager_t;
```

  The exact field names can differ, but the principle is important.

### 9. Not every region should use large pages

#### 9.1 Model weights are the highest-priority large-page target

- **Source:** Microsoft large-page docs; EdgeLM Section 03; Section 05
- **Key idea:** Weights are large, read-mostly, CPU-hot, and scanned repeatedly
  during decode.
- **Relevance to EdgeLM:** If only one region gets large pages, it should be the
  weights region.
- **Estimated impact:** Very High.
- **Implementation complexity:** Low.
- **Details:** This is where TLB coverage and page-boundary effects matter most,
  and where the engine has the clearest fixed-size startup allocation target.

#### 9.2 KV cache is the second-best large-page target, but it should still be sized conservatively

- **Source:** Section 11 KV-cache research; Microsoft large-page docs
- **Key idea:** The KV cache is long-lived and heavily reused, so it benefits
  from large pages, but it also competes directly with the project's RAM budget.
- **Relevance to EdgeLM:** Large-page KV allocation is good, but only if the
  context-size target is chosen realistically.
- **Estimated impact:** High.
- **Implementation complexity:** Medium.
- **Details:** Because large pages are nonpageable and committed up front, an
  overprovisioned KV cache is especially costly. EdgeLM should not reserve a
  heroic maximum context just because the virtual address space exists.

#### 9.3 Activation scratch is a judgment call, not an automatic yes

- **Source:** Project workload synthesis; Windows large-page constraints
- **Key idea:** Activation scratch is reused constantly and may benefit from
  large pages, but its access patterns are shorter-lived and more localized than
  the weight stream.
- **Relevance to EdgeLM:** This region should be large-page eligible, but it is
  lower priority than weights and KV.
- **Estimated impact:** Medium to High.
- **Implementation complexity:** Low.
- **Details:** A sensible bring-up order is:
  1. weights first
  2. KV second
  3. activation arena third, if budget and stability allow

#### 9.4 Small metadata, tokenizer tables, and control structs should stay on normal pages

- **Source:** Project architecture synthesis
- **Key idea:** Large pages are too blunt an instrument for tiny, low-frequency
  data structures.
- **Relevance to EdgeLM:** This avoids wasting large-page capacity and startup
  complexity on things that do not materially benefit.
- **Estimated impact:** Medium.
- **Implementation complexity:** Low.

#### 9.5 CPU+iGPU shared or interop buffers should start conservative

- **Source:** Inference from Sections 17 and 18 plus Windows memory constraints
- **Key idea:** Buffers meant for heterogeneous interop should not be mixed into
  the first large-page experiment unless there is a measured benefit and known
  interoperability path.
- **Relevance to EdgeLM:** This reduces debugging surface area.
- **Estimated impact:** Medium.
- **Implementation complexity:** Low.
- **Details:** The safest default is:
  - keep CPU-hot model and cache memory in the CPU memory-manager policy
  - and keep iGPU/shared staging policy separate until the hybrid path is
    independently measured.

### 10. Reserve/commit elasticity and large pages should coexist, not compete

#### 10.1 Some EdgeLM regions are fixed-size; others are better reserved large and committed selectively

- **Source:** Microsoft `VirtualAlloc`; EdgeLM architecture synthesis
- **Key idea:** The memory manager should support both:
  - one-shot committed regions
  - and reserve-then-commit regions
- **Relevance to EdgeLM:** This avoids forcing every buffer into the large-page
  model.
- **Estimated impact:** High.
- **Implementation complexity:** Medium.
- **Details:** A good split is:
  - **weights / fixed KV / fixed activation arena:** allocate once at startup
  - **load scratch / optional request arenas / future elastic buffers:** reserve
    a region and commit as needed with normal pages

#### 10.2 This is one of the strongest arguments for separate regions instead of one monolithic arena

- **Source:** Windows allocation semantics; project memory goals
- **Key idea:** A single giant arena would force one commitment policy on all
  memory.
- **Relevance to EdgeLM:** That is a poor fit for the real workload.
- **Estimated impact:** High.
- **Implementation complexity:** Medium.
- **Details:** The engine should avoid a false binary of:
  - "everything is large-page"
  - or "everything is malloc."

  The right answer is a small number of intentionally different regions.

### 11. `VirtualAlloc2` is interesting, but most of its power belongs to later phases

#### 11.1 `VirtualAlloc2` adds alignment restrictions, NUMA hints, and placeholders

- **Source:** Microsoft Learn `VirtualAlloc2`
- **Key idea:** Microsoft documents that `VirtualAlloc2` can express:
  - power-of-two alignment restrictions,
  - extended parameters such as preferred NUMA node,
  - and placeholder operations that support remapping-based layouts.
- **Relevance to EdgeLM:** This makes `VirtualAlloc2` the most capable Windows
  virtual-memory API in the toolbox.
- **Estimated impact:** Medium.
- **Implementation complexity:** Medium.

#### 11.2 NUMA hints are low priority on this laptop

- **Source:** Microsoft Learn `VirtualAlloc2`; prior EdgeLM Section 03 findings
- **Key idea:** If no NUMA hint is provided, Windows bases preferred physical
  page placement on the ideal processor of the thread that first accesses the
  memory. `VirtualAlloc2` can override this with a preferred NUMA node.
- **Relevance to EdgeLM:** On a single-socket, single-NUMA-node `i7-12700H`,
  this is real but minor.
- **Estimated impact:** Low.
- **Implementation complexity:** Low.
- **Details:** NUMA-aware page placement belongs in the "nice late optimization"
  bucket, not the "must-have for v1" bucket.

#### 11.3 Placeholders are most interesting for advanced ring-buffer designs

- **Source:** Microsoft Learn `VirtualAlloc2`
- **Key idea:** Microsoft explicitly notes that placeholders can implement
  arbitrarily extendable regions or virtual-memory ring buffers.
- **Relevance to EdgeLM:** This is highly relevant to advanced KV-cache or
  sliding-window designs.
- **Estimated impact:** Medium.
- **Implementation complexity:** High.
- **Details:** A placeholder-backed mirrored ring buffer could eventually let
  EdgeLM linearize wraparound logic in some contexts. But it is not necessary
  for the first implementation because the current KV design already has a
  workable explicit indexing model.

### 12. Working-set tuning is not a substitute for explicit memory policy

#### 12.1 Large-page memory is outside the normal working-set model

- **Source:** Microsoft Learn `Large-Page Support`; `QueryWorkingSetEx`
- **Key idea:** Microsoft documents that large pages are not part of the normal
  working set, and `QueryWorkingSetEx` specifically exists to query addresses
  such as AWE or large pages that fall outside ordinary working-set reporting.
- **Relevance to EdgeLM:** This matters both conceptually and for diagnostics.
- **Estimated impact:** High.
- **Implementation complexity:** Low.

#### 12.2 Inflating the working set is the wrong fallback strategy

- **Source:** Microsoft Learn `SetProcessWorkingSetSize`
- **Key idea:** Microsoft warns that increasing working-set size does not
  guarantee requested memory will remain resident, and that taking physical
  memory away from the rest of the system can degrade overall system
  performance.
- **Relevance to EdgeLM:** This is the clearest official reason not to treat
  working-set APIs as a replacement for large pages.
- **Estimated impact:** High.
- **Implementation complexity:** Low.
- **Details:** So the fallback ladder should be:
  1. large pages if available
  2. normal pages plus prefetch/pre-touch
  3. never "just crank up the working set and hope"

  That is a much cleaner and safer policy on a shared-use laptop.

### 13. Reclamation policy should differ sharply between startup scratch and decode-time memory

#### 13.1 Decode-time regions should not be decommitted or reallocated in the token loop

- **Source:** Windows `VirtualAlloc`/`VirtualFree` semantics; project hot-path
  requirements
- **Key idea:** Any page-state transitions during decode are almost certainly a
  bad trade on this hardware.
- **Relevance to EdgeLM:** This should be a hard design rule.
- **Estimated impact:** Very High.
- **Implementation complexity:** Low.
- **Details:** Once inference begins, the engine should avoid:
  - `VirtualAlloc`,
  - `VirtualFree`,
  - decommit/recommit cycles,
  - and OS-managed allocator churn.

  Arena reset is fine. OS-level memory state changes are not.

#### 13.2 `MEM_RESET` is for cold private scratch, not hot-path reuse

- **Source:** Microsoft Learn `VirtualAlloc2`
- **Key idea:** `MEM_RESET` tells Windows that data in a private committed range
  is no longer interesting and should not be written to the paging file, but the
  range may be reused later. It does not guarantee zero fill, and it cannot be
  used on file-mapped memory.
- **Relevance to EdgeLM:** This makes it potentially useful for cold load-time
  staging areas, but not for per-token decode buffers.
- **Estimated impact:** Medium.
- **Implementation complexity:** Medium.
- **Details:** Good possible use cases:
  - one-time load scratch after the repack completes
  - optional request-local scratch between independent requests

  Bad use cases:
  - activation buffers reused every layer
  - KV cache
  - hot token-loop workspaces

#### 13.3 `VirtualFree` rules reinforce the case for region ownership

- **Source:** Microsoft Learn `VirtualFree` / freeing virtual memory
- **Key idea:** Releasing reserved virtual memory is region-granular; whole
  regions must be managed intentionally.
- **Relevance to EdgeLM:** This is another reason the memory manager should own
  top-level regions explicitly and sub-allocate within them.
- **Estimated impact:** High.
- **Implementation complexity:** Low.
- **Details:** In practice, this means:
  - allocate a region once,
  - hand out aligned sub-slices,
  - reclaim sub-slices by arena reset,
  - release the region only at shutdown or major mode transitions.

### 14. Diagnostics and instrumentation should be built into the memory manager

#### 14.1 Page mode and residency should be observable

- **Source:** Microsoft Learn `QueryWorkingSetEx`; project benchmarking goals
- **Key idea:** Windows exposes enough information to verify whether addresses
  belong to large-page or non-working-set regions.
- **Relevance to EdgeLM:** This is important for research reproducibility.
- **Estimated impact:** High.
- **Implementation complexity:** Medium.
- **Details:** The memory manager should track and report at least:
  - region name,
  - requested bytes,
  - rounded bytes,
  - page mode (`normal` vs `large`),
  - allocation success/fallback reason,
  - and whether the region is expected to be pageable or not.

  `QueryWorkingSetEx` can be used in debug tooling to sanity-check some of this.

#### 14.2 Memory accounting should distinguish requested bytes from rounded bytes

- **Source:** Microsoft large-page alignment rules; project RAM budget
- **Key idea:** Large pages force `2 MB` rounding, which can add meaningful
  hidden overhead when several regions are rounded separately.
- **Relevance to EdgeLM:** This matters on a `16 GB` machine with a strict
  project budget.
- **Estimated impact:** High.
- **Implementation complexity:** Low.
- **Details:** Every region record should therefore store:
  - logical bytes needed by the model/runtime
  - committed bytes actually allocated
  - wasted bytes from page-size rounding

  Without that accounting, memory pressure will look mysteriously worse than the
  architectural estimates.

### 15. A concrete EdgeLM region plan is now clear

#### 15.1 Recommended top-level regions

- **Source:** Synthesis of Windows docs and prior EdgeLM research
- **Key idea:** The most coherent v1 memory layout is:
  1. `weights_region`
  2. `kv_k_region`
  3. `kv_v_region`
  4. `activation_region`
  5. `request_region`
  6. `load_scratch_region`
- **Relevance to EdgeLM:** This gives the project a directly implementable shape.
- **Estimated impact:** Very High.
- **Implementation complexity:** Medium.
- **Details:** Suggested policy:
  - `weights_region`: large-page preferred, startup-only allocation, immutable by convention
  - `kv_k_region` / `kv_v_region`: large-page preferred, startup-only allocation
  - `activation_region`: normal or large pages depending remaining budget and measured benefit
  - `request_region`: normal pages, reserve/commit capable, arena-reset between requests
  - `load_scratch_region`: normal pages, may use `MEM_RESET` or decommit after import

#### 15.2 Suggested startup order matters

- **Source:** Microsoft large-page fragmentation guidance; project startup flow
- **Key idea:** The order should front-load the most large-page-sensitive
  allocations before memory becomes more fragmented.
- **Relevance to EdgeLM:** This is a subtle but practical startup optimization.
- **Estimated impact:** High.
- **Implementation complexity:** Low.
- **Details:** Recommended order:
  1. enable privilege if available
  2. allocate weights region
  3. allocate KV regions
  4. allocate activation region
  5. map/read model file
  6. repack/copy into runtime regions
  7. reset or release load scratch

  That ordering maximizes the chance that the most important large-page requests
  succeed.

### 16. The right v1 memory manager is simple, explicit, and boring

#### 16.1 EdgeLM does not need a general-purpose allocator framework

- **Source:** Project workload synthesis; Windows API constraints
- **Key idea:** The workload is too structured to justify a complicated
  allocator stack.
- **Relevance to EdgeLM:** Over-abstraction here would be a regression.
- **Estimated impact:** High.
- **Implementation complexity:** Low.
- **Details:** The v1 memory manager should provide:
  - top-level region creation/destruction,
  - page-mode fallback,
  - alignment-aware arena suballocation,
  - debug accounting,
  - and optional warmup/prefetch hooks.

  That is enough.

#### 16.2 Complexity should only be added when a measured need appears

- **Source:** Project implementation philosophy; Windows advanced APIs
- **Key idea:** `VirtualAlloc2` placeholders, NUMA hints, mirrored ring buffers,
  and more elaborate commitment policies are valuable, but they belong after the
  baseline region plan exists and is benchmarked.
- **Relevance to EdgeLM:** This preserves momentum and keeps the paper story
  clean.
- **Estimated impact:** High.
- **Implementation complexity:** Low.

## Comparative Fit for EdgeLM

| Strategy | Throughput benefit | Memory flexibility | OS complexity | Main risk | EdgeLM verdict |
| --- | --- | --- | --- | --- | --- |
| Large-page top-level regions + arena suballocation | High | Medium | Medium | privilege/fragmentation fallback | Best first implementation |
| Normal pages only + arena suballocation + prefetch | Medium | High | Low | leaves TLB wins on the table | Best fallback path |
| Direct file mapping as final runtime layout | Medium for startup, low for decode | Low | Medium | cannot directly get normal data files into large pages | Good input path, bad final runtime path |
| General heap / `malloc` in hot path | Low | High | Low | allocator churn and unpredictability | Not acceptable |
| Aggressive working-set tuning as residency control | Unclear | Low | Medium | harms system, weak guarantees | Do not use as primary strategy |
| Placeholder-heavy virtual ring design in v1 | Potentially medium | High | High | premature complexity | Later-stage option only |

## Recommendations for EdgeLM

### 1. Build the memory manager around named top-level regions

- One region each for:
  - weights,
  - KV K,
  - KV V,
  - activations,
  - request-local scratch,
  - and load scratch.

### 2. Use large pages selectively, not dogmatically

- First priority: weights
- Second priority: KV cache
- Third priority: activation arena if budget and stability allow
- Keep small metadata and control allocations on normal pages

### 3. Make fallback behavior a first-class feature

- Try large pages.
- If privilege or fragmentation blocks them, fall back to standard pages with
  identical logical layout.
- Record which regions fell back and why.

### 4. Keep all hot-path allocation inside arenas

- No `VirtualAlloc`, `VirtualFree`, `malloc`, or `free` during decode.
- Use alignment-aware bump allocation and bulk reset.

### 5. Separate load-time policy from steady-state policy

- File mappings and staging buffers are fine during import.
- The final runtime layout should live in explicitly managed private regions.
- Release or reset staging memory before normal inference begins.

### 6. Use `VirtualAlloc2` features only when a measured need appears

- Ignore NUMA hints for v1.
- Keep placeholders and virtual ring buffers for later KV-cache experiments.
- Prefer the simpler allocator unless profiling proves otherwise.

## Suggested EdgeLM Implementation Shape

### Top-level API

```c
typedef enum {
    PAGE_MODE_NORMAL = 0,
    PAGE_MODE_LARGE  = 1
} page_mode_t;

typedef struct {
    const char *name;
    void       *base;
    size_t      requested;
    size_t      allocated;
    page_mode_t page_mode;
    int         is_nonpageable;
} vm_region_t;

typedef struct {
    uint8_t *base;
    size_t   size;
    size_t   head;
} arena_t;

typedef struct {
    vm_region_t weights;
    vm_region_t kv_k;
    vm_region_t kv_v;
    vm_region_t activations;
    vm_region_t request;
    vm_region_t load_scratch;
    arena_t     act_arena;
    arena_t     req_arena;
} edge_memory_t;
```

### Region policy

1. `weights`
   - startup-only
   - large-page preferred
   - read-only by convention, not page protection
2. `kv_k` / `kv_v`
   - startup-only
   - large-page preferred
3. `activations`
   - startup-only
   - large-page optional
   - arena reset every layer or token as appropriate
4. `request`
   - normal pages
   - reserve/commit friendly
   - reset between requests
5. `load_scratch`
   - normal pages
   - may be `MEM_RESET` or released after import

### Allocation sequence

1. Check `GetLargePageMinimum()`.
2. Attempt to enable `SeLockMemoryPrivilege`.
3. Allocate large-page-eligible regions in priority order.
4. Fall back to normal pages on failure, preserving region topology.
5. Map or read model files.
6. Repack/copy into runtime regions.
7. Prefetch or warm touched normal-page regions if needed.
8. Reset or free load scratch before serving requests.

## Suggested Experiment Sequence

1. Implement region-tracked normal-page allocator with arenas.
2. Add large-page attempts for weights only.
3. Extend large-page support to KV cache.
4. Measure whether the activation arena benefits enough to justify large pages.
5. Add instrumentation for rounded-vs-requested bytes and fallback reasons.
6. Evaluate `PrefetchVirtualMemory` on fallback and load-time file mappings.
7. Only later explore `VirtualAlloc2` placeholders for advanced KV designs.

## Bottom Line

The memory manager EdgeLM needs is not a fancy general allocator. It is a small,
explicit region manager that speaks native Windows virtual-memory semantics and
knows the difference between:

- startup versus decode,
- fixed-size versus elastic regions,
- large-page-worthy versus ordinary allocations,
- and arena-resettable scratch versus true top-level lifetime boundaries.

The main design conclusions are:

- large pages are absolutely worth pursuing, but only for the right long-lived
  CPU-hot regions,
- large-page failure must be treated as a normal runtime condition,
- model files should be loaded normally and copied or repacked into runtime
  buffers rather than mapped directly as final large-page storage,
- arenas are the right hot-path allocation model for EdgeLM,
- and working-set tuning is not a serious substitute for explicit memory policy.

So the correct v1 implementation is:

- a few named `VirtualAlloc` regions,
- large-page attempts with graceful fallback,
- alignment-aware arena suballocation,
- no OS allocation in the decode loop,
- and built-in accounting so the engine always knows where its RAM budget went.

That is simple, fast, and directly matched to the workload. Anything more
complicated should be justified by measurement, not by how low-level it sounds.

## Sources

- Microsoft Learn, `Large-Page Support`:
  `https://learn.microsoft.com/en-us/windows/win32/memory/large-page-support`
- Microsoft Learn, `VirtualAlloc`:
  `https://learn.microsoft.com/en-us/windows/win32/api/memoryapi/nf-memoryapi-virtualalloc`
- Microsoft Learn, `VirtualAlloc2`:
  `https://learn.microsoft.com/en-us/windows/win32/api/memoryapi/nf-memoryapi-virtualalloc2`
- Microsoft Learn, `PrefetchVirtualMemory`:
  `https://learn.microsoft.com/en-us/windows/win32/api/memoryapi/nf-memoryapi-prefetchvirtualmemory`
- Microsoft Learn, `CreateFileMapping`:
  `https://learn.microsoft.com/en-us/windows/win32/api/winbase/nf-winbase-createfilemappinga`
- Microsoft Learn, `MapViewOfFile`:
  `https://learn.microsoft.com/en-us/windows/win32/api/memoryapi/nf-memoryapi-mapviewoffile`
- Microsoft Learn, `VirtualFree` and freeing virtual memory:
  `https://learn.microsoft.com/en-us/windows/win32/api/memoryapi/nf-memoryapi-virtualfree`
- Microsoft Learn, `SetProcessWorkingSetSize`:
  `https://learn.microsoft.com/en-us/windows/win32/api/memoryapi/nf-memoryapi-setprocessworkingsetsize`
- Microsoft Learn, `QueryWorkingSetEx`:
  `https://learn.microsoft.com/en-us/windows/win32/api/psapi/nf-psapi-queryworkingsetex`
- Microsoft Learn, `AdjustTokenPrivileges` / changing privileges in a token:
  `https://learn.microsoft.com/en-us/windows/win32/secbp/changing-privileges-in-a-token`
- LLVM `BumpPtrAllocator` documentation:
  `https://llvm.org/doxygen/classllvm_1_1BumpPtrAllocatorImpl.html`

## Audit Addendum (2026-04-02)

- **Memory-failure injection is worth adding to the test plan.** The allocator
  should be able to simulate:
  - large-page allocation failure,
  - reserve failure,
  - commit failure,
  - and fallback-path exhaustion

  so the runtime's failure behavior is tested before it happens on a real user
  machine.
- **Commit-charge telemetry should be part of benchmark metadata.** Working-set
  size alone does not fully explain Windows memory pressure behavior.
- **Region ownership should remain debuggable.** A simple region map dump would
  make later leak and fragmentation debugging much easier.
