# Section 03: Cache Hierarchy Optimization -- Extended Research

## Overview

This section provides deep research into TLB (Translation Lookaside Buffer) microarchitecture, large page allocation on Windows, cache line alignment, false sharing avoidance, NUMA-like behavior, and cache coloring on the i7-12700H (Alder Lake). The cache/TLB subsystem is the second most critical bottleneck after raw memory bandwidth for our ternary inference engine, because every weight access requires address translation, and every multi-threaded kernel is vulnerable to false sharing if buffers are not properly aligned.

## What the Deep Dive Already Covers

- Cache hierarchy sizes: L1D 48KB (P-core), 32KB (E-core cluster), L2 1.25MB (P) / 2MB (E cluster), L3 24MB shared
- Cache latencies: L1 ~4cy, L2 ~14cy, L3 ~40-50cy
- TLB basics: L1 DTLB 96 entries, L2 STLB 2048 entries
- Large pages reduce TLB misses; VirtualAlloc with MEM_LARGE_PAGES
- 64-byte cache line alignment needed for false sharing avoidance
- Hardware prefetchers: L2 Streamer (16 lines ahead), L1 Streamer (2 lines ahead), stride prefetcher, AMP
- Arena allocator pattern, double-buffer pattern
- secpol.msc setup for SeLockMemoryPrivilege

---

## New Findings

### 1. TLB Microarchitecture on Golden Cove P-Cores

#### 1.1 Complete TLB Hierarchy

- **Source:** [Intel Architecture Day 2021](https://www.intel.com/content/www/us/en/newsroom/resources/press-kit-architecture-day-2021.html), [Chips and Cheese - Golden Cove](https://chipsandcheese.com/2021/12/02/popping-the-hood-on-golden-cove/), [Wikipedia - Golden Cove](https://en.wikipedia.org/wiki/Golden_Cove)
- **Key finding:** The Golden Cove TLB hierarchy is significantly improved over Sunny Cove/Ice Lake, with separate TLBs for different page sizes.

**L1 Data TLB (Golden Cove P-Core):**

| Page Size | Entries | Associativity | Coverage |
|-----------|---------|---------------|----------|
| 4 KB      | 96      | Fully assoc.  | 384 KB   |
| 2 MB      | 32      | Fully assoc.  | 64 MB    |
| 1 GB      | 8       | Fully assoc.  | 8 GB     |

**L1 Instruction TLB (Golden Cove P-Core):**

| Page Size | Entries | Associativity | Coverage |
|-----------|---------|---------------|----------|
| 4 KB      | 256     | 8-way         | 1 MB     |
| 2 MB      | 16      | Fully assoc.  | 32 MB    |

**L2 Shared TLB (STLB, Golden Cove P-Core):**

| Page Size | Entries | Associativity | Coverage |
|-----------|---------|---------------|----------|
| 4 KB + 2 MB (unified) | 2048 | 16-way | 8 MB (4KB) or 4 GB (2MB) |

- **Relevance to EdgeLM:** The L1 DTLB has dedicated 2MB entries (32) separate from 4KB entries (96). This means with large pages, our ~600MB ternary model requires only 300 x 2MB pages. The L1 DTLB can hold 32 of those (64 MB coverage), and the STLB can hold all 300 (600 MB < 4 GB). This effectively eliminates STLB misses for our entire model.
- **Critical detail:** The 1 GB page entries (8 entries, 8 GB coverage) could theoretically cover our entire model in a SINGLE TLB entry, but Windows does not support 1GB pages via VirtualAlloc (only Hyper-V and some server configurations use them).
- **Estimated impact:** With 2MB large pages, TLB miss rate drops from potentially 93%+ (4KB pages with large working sets) to near 0%. Based on benchmark data from rigtorp.se, this can mean a 25-35% reduction in memory access latency for streaming workloads.

#### 1.2 Page Walk Details and Latency

- **Source:** [blog.stuffedcow.net - Page Walk Coherence](https://blog.stuffedcow.net/2015/08/pagewalk-coherence/), Intel SDM Vol. 3A Chapter 4
- **Key finding:** Golden Cove uses speculative, coherent page walks with conflict detection.

**Page walk structure on x86-64:**
- 4-level page table walk (PML4 -> PDPT -> PD -> PT) for 4KB pages
- 3-level walk for 2MB pages (PML4 -> PDPT -> PD, skipping PT level)
- 2-level walk for 1GB pages (PML4 -> PDPT)
- Each level requires one memory access (potentially cached in higher TLB levels or paging structure caches)

**Page walk latency estimates for Golden Cove:**

| Scenario | Estimated Cycles | Notes |
|----------|-----------------|-------|
| L1 DTLB hit | 0 (pipelined) | Absorbed into load latency |
| L2 STLB hit | ~8-10 cycles | Penalty vs L1 DTLB hit |
| Full page walk (PT entries in L1D) | ~15-20 cycles | Best case TLB miss |
| Full page walk (PT entries in L2) | ~25-35 cycles | Typical case |
| Full page walk (PT entries in L3) | ~40-60 cycles | Worst realistic case |
| Full page walk (PT entries in DRAM) | ~150-300 cycles | Cold page tables |

- **Relevance to EdgeLM:** With 4KB pages and a 600MB model (153,600 pages), the STLB can only hold 2048 entries. During a full model weight scan, we will experience ~151,000 STLB misses per inference pass. At ~30-60 cycles each, that is 4.5M - 9M wasted cycles per token generation. At 4.1 GHz, that is 1.1 - 2.2 ms of pure TLB miss overhead per token -- enough to reduce throughput by 10-20%.
- **With 2MB pages:** Only 300 pages needed, all fit in STLB. Zero STLB misses. The L1 DTLB can hold 32 entries, so within any local region of ~64MB, there are no DTLB misses either.
- **Page walk caches:** Golden Cove includes paging structure caches (PSC) that cache intermediate page table entries (PML4, PDPT, PD entries). These reduce the effective page walk cost even when a full TLB miss occurs, because upper-level entries are often cached. For a 600MB working set, the PML4 and PDPT entries will always be cached (only 1 PML4 entry and 1 PDPT entry needed), so worst-case walks are typically 2-level, not 4-level.

#### 1.3 TLB Partitioning with HyperThreading

- **Source:** Intel SDM Vol. 3A, [Wikipedia - Golden Cove](https://en.wikipedia.org/wiki/Golden_Cove)
- **Key finding:** On Golden Cove, the L1 DTLB is statically partitioned between hyperthreads (each thread gets half), while the STLB is competitively shared.

**With HyperThreading enabled on a P-core:**

| TLB Level | Partitioning | Per-Thread Entries (4KB) |
|-----------|-------------|------------------------|
| L1 DTLB   | Static partition (50/50) | 48 entries per thread |
| L1 ITLB   | Static partition | 128 entries per thread |
| L2 STLB   | Competitively shared | Up to 2048 (contended) |

- **Relevance to EdgeLM:** Since we recommend 6 threads (one per P-core, no HT) for SIMD work, we get the FULL 96-entry L1 DTLB per core. If we were to use HT (12 threads), each thread would only get 48 L1 DTLB entries, halving coverage. This is another strong reason to avoid HT for bandwidth-bound inference.
- **Implementation note:** When using E-cores for lightweight tasks (I/O, sampling), E-cores do not have HT so there is no partitioning concern there.

#### 1.4 Hardware Prefetcher Interaction with TLB

- **Source:** Intel SDM Vol. 3A Section 11.3, Intel Optimization Reference Manual
- **Key finding:** Hardware prefetchers DO NOT cross page boundaries and DO require TLB lookups.

**Critical behaviors:**
1. **Hardware prefetchers cannot cross 4KB page boundaries.** The L2 streamer prefetcher detects sequential access patterns but stops at the end of each 4KB page. With the next page, it must "re-learn" the pattern. This creates a "prefetch stall" at every page boundary.
2. **With 2MB large pages, the prefetcher can run uninterrupted for 2MB** before hitting a boundary. For our sequential weight streaming pattern, this means the prefetcher is 512x more effective (2MB / 4KB = 512 fewer boundary stops).
3. **Software PREFETCH instructions (PREFETCHT0/T1/T2/NTA) DO require TLB translation.** If the prefetch address hits a TLB miss, the behavior is implementation-defined -- on Golden Cove, the prefetch is likely dropped silently rather than causing a page walk. This means software prefetch is less effective with 4KB pages because some prefetches will be silently dropped at page boundaries.
4. **Hardware prefetchers consume TLB entries** for the prefetched addresses. With 4KB pages, aggressive prefetching can cause TLB thrashing. With 2MB pages, this is a non-issue.

- **Relevance to EdgeLM:** This is a MAJOR finding. The prefetcher boundary issue means that with 4KB pages, our weight streaming throughput is degraded not just by TLB misses but also by prefetcher re-training at every 4KB boundary. With 2MB pages, the prefetcher can stream a full 2MB weight block without interruption. Combined with explicit software prefetching, this could improve effective bandwidth utilization by 15-25%.
- **Estimated impact:** HIGH -- prefetcher effectiveness is a multiplier on bandwidth utilization.

---

### 2. Gracemont E-Core TLB Details

#### 2.1 E-Core TLB Hierarchy (Different from P-Core)

- **Source:** [Chips and Cheese - Gracemont](https://chipsandcheese.com/2022/08/11/gracemont-revenge-of-the-atom-cores/), [Wikipedia - Gracemont](https://en.wikipedia.org/wiki/Gracemont_(microarchitecture))

**L1 Data TLB (Gracemont E-Core):**

| Page Size | Entries | Associativity | Coverage |
|-----------|---------|---------------|----------|
| 4 KB      | 48      | Fully assoc.  | 192 KB   |
| 2 MB      | ~16 (est.) | Fully assoc. | 32 MB  |

**L2 TLB (Gracemont E-Core):**

| Page Size | Entries | Associativity | Coverage |
|-----------|---------|---------------|----------|
| 4 KB + 2 MB (unified) | 2048 | 4-way | 8 MB (4KB) or 4 GB (2MB) |

- **Key difference from P-core:** Gracemont's L1 DTLB has only 48 entries (vs 96 on Golden Cove), and the L2 TLB is only 4-way associative (vs 16-way on Golden Cove). This means:
  - E-core L1 DTLB coverage: 192 KB (4KB pages) vs 384 KB (P-core)
  - E-core L2 TLB has same entry count but lower associativity, meaning more conflict misses
  - **L2 TLB access adds 2 extra cycles on Gracemont** compared to Golden Cove, attributed to power optimization

- **Gracemont cache latencies:**
  - L1D: 3 cycles (faster than Golden Cove's 5 cycles at equal clock speeds, but E-cores clock lower)
  - L2: 17 cycles (shared among 4 E-cores in a cluster)
  - L3: Same shared LLC as P-cores

- **Relevance to EdgeLM:** E-cores are used for lightweight tasks (I/O coordination, token sampling, speculative draft model). Their smaller TLB and lower associativity is acceptable because these tasks have small working sets. We should NOT run weight streaming on E-cores -- not only because of slower AVX2 (split 128-bit), but also because of worse TLB coverage.

---

### 3. Large Pages on Windows 11 -- Complete Implementation Guide

#### 3.1 VirtualAlloc with MEM_LARGE_PAGES

- **Source:** [Microsoft Learn - Large Page Support](https://learn.microsoft.com/en-us/windows/win32/memory/large-page-support), [Microsoft Learn - VirtualAlloc](https://learn.microsoft.com/en-us/windows/win32/api/memoryapi/nf-memoryapi-virtualalloc)

**Complete API usage:**
```c
#include <windows.h>

// Step 1: Enable SeLockMemoryPrivilege at runtime
BOOL EnableLargePagePrivilege(void) {
    HANDLE hToken;
    TOKEN_PRIVILEGES tp;

    if (!OpenProcessToken(GetCurrentProcess(),
                          TOKEN_ADJUST_PRIVILEGES | TOKEN_QUERY, &hToken))
        return FALSE;

    if (!LookupPrivilegeValue(NULL, SE_LOCK_MEMORY_NAME,
                              &tp.Privileges[0].Luid)) {
        CloseHandle(hToken);
        return FALSE;
    }

    tp.PrivilegeCount = 1;
    tp.Privileges[0].Attributes = SE_PRIVILEGE_ENABLED;

    BOOL status = AdjustTokenPrivileges(hToken, FALSE, &tp, 0, NULL, 0);
    DWORD error = GetLastError();
    CloseHandle(hToken);

    return status && (error == ERROR_SUCCESS);
}

// Step 2: Allocate with large pages
void* AllocateLargePages(size_t size) {
    SIZE_T largePageMin = GetLargePageMinimum();  // Returns 2097152 (2MB)
    if (largePageMin == 0) return NULL;  // Large pages not supported

    // Size MUST be a multiple of the large page minimum
    size = (size + largePageMin - 1) & ~(largePageMin - 1);

    // MEM_RESERVE | MEM_COMMIT | MEM_LARGE_PAGES -- all three required together
    void *ptr = VirtualAlloc(NULL, size,
        MEM_RESERVE | MEM_COMMIT | MEM_LARGE_PAGES,
        PAGE_READWRITE);

    return ptr;  // NULL on failure
}
```

**Critical constraints and caveats:**
1. **Must reserve AND commit in one call.** You cannot VirtualAlloc(MEM_RESERVE) first and then VirtualAlloc(MEM_COMMIT) later for large pages. It must be a single `MEM_RESERVE | MEM_COMMIT | MEM_LARGE_PAGES` call.
2. **Size and alignment must be multiples of GetLargePageMinimum()** (2MB on x86-64).
3. **Memory is always non-pageable** (pinned in physical RAM). It will NOT be paged to disk.
4. **Memory counts toward process private bytes but NOT working set** (because working set only tracks pageable memory).
5. **Large page allocations are NOT subject to job limits.**
6. **Physical memory fragmentation:** After the system has been running for a long time, 2MB contiguous physical regions become scarce. Allocate all large pages at startup, before memory fragments.
7. **Allocation failure is common** if requested after system has been running. Always have a fallback to standard 4KB pages.

- **Estimated impact:** For our ~600MB model, using large pages eliminates ~151,000 TLB misses per weight scan, saving 1-2ms per token. At 100 tok/s target, that is 10-20% of our time budget.
- **Implementation complexity:** Low -- the API is straightforward. The setup (SeLockMemoryPrivilege) is a one-time configuration.

#### 3.2 VirtualAlloc2 -- Advanced Features

- **Source:** [Microsoft Learn - VirtualAlloc2](https://learn.microsoft.com/en-us/windows/win32/api/memoryapi/nf-memoryapi-virtualalloc2)
- **Key finding:** VirtualAlloc2 (Windows 10+) provides extended parameters including NUMA node preference and alignment control.

**NUMA-aware allocation with large pages:**
```c
void* AllocateLargePagesOnNode(size_t size, ULONG numaNode) {
    MEM_EXTENDED_PARAMETER param = {0};
    param.Type = MemExtendedParameterNumaNode;
    param.ULong = numaNode;

    SIZE_T largePageMin = GetLargePageMinimum();
    size = (size + largePageMin - 1) & ~(largePageMin - 1);

    return VirtualAlloc2(NULL, NULL, size,
        MEM_RESERVE | MEM_COMMIT | MEM_LARGE_PAGES,
        PAGE_READWRITE,
        &param, 1);
}
```

**Alignment-controlled allocation:**
```c
void* AllocateAligned(size_t size, size_t alignment) {
    MEM_ADDRESS_REQUIREMENTS addressReqs = {0};
    MEM_EXTENDED_PARAMETER param = {0};

    addressReqs.Alignment = alignment;  // Must be power of 2

    param.Type = MemExtendedParameterAddressRequirements;
    param.Pointer = &addressReqs;

    return VirtualAlloc2(NULL, NULL, size,
        MEM_RESERVE | MEM_COMMIT,
        PAGE_READWRITE,
        &param, 1);
}
```

**MEM_64K_PAGES (NEW in recent Windows builds):**
- VirtualAlloc2 supports `MEM_64K_PAGES` flag (value 0x20400000) which hints the OS to use 64KB physically contiguous pages
- Unlike 2MB large pages, 64KB pages are pageable by default
- If physical memory is too fragmented for 64KB contiguous pages, the allocation falls back to 4KB pages silently
- Combined with `MEM_EXTENDED_PARAMETER_NONPAGED`, they become non-paged 64KB pages (fails if contiguous pages unavailable)
- Size and BaseAddress must be multiples of 64KB

- **Relevance to EdgeLM:** MEM_64K_PAGES is a middle ground between 4KB and 2MB pages. For buffers that don't need the full 2MB granularity (e.g., KV cache, activation buffers), 64KB pages could reduce TLB pressure without the physical contiguity requirements of 2MB pages. However, 2MB pages remain preferred for the main model weights.

#### 3.3 SeLockMemoryPrivilege Setup

- **Source:** [Microsoft Learn - Assigning Privileges](https://learn.microsoft.com/en-us/windows/win32/secbp/assigning-privileges-to-an-account)

**Method 1: Local Security Policy GUI**
1. Press Win+R, type `secpol.msc`, press Enter
2. Navigate: Local Policies -> User Rights Assignment -> "Lock pages in memory"
3. Click "Add User or Group..."
4. Add the user account that will run the inference engine
5. Click OK, close the policy editor
6. **REBOOT REQUIRED** -- the privilege only takes effect after logon

**Method 2: PowerShell (requires elevation)**
```powershell
# Grant SeLockMemoryPrivilege to a user
$sid = (New-Object System.Security.Principal.NTAccount("USERNAME")).Translate(
    [System.Security.Principal.SecurityIdentifier]).Value
$tempFile = [System.IO.Path]::GetTempFileName()
secedit /export /cfg $tempFile
$config = Get-Content $tempFile
$line = $config | Where-Object { $_ -match "SeLockMemoryPrivilege" }
if ($line) {
    $config = $config -replace "SeLockMemoryPrivilege.*",
        "SeLockMemoryPrivilege = *$sid,$($line.Split('=')[1].Trim())"
} else {
    $config += "SeLockMemoryPrivilege = *$sid"
}
Set-Content $tempFile $config
secedit /configure /db secedit.sdb /cfg $tempFile
Remove-Item $tempFile
```

**Security implications:**
- SeLockMemoryPrivilege allows any process run by that user to pin memory in physical RAM
- Pinned memory cannot be paged out, reducing available memory for other processes
- On a dedicated inference machine, this is acceptable
- On a shared machine, only grant to specific service accounts

**Programmatic check at runtime:**
```c
BOOL CanUseLargePages(void) {
    SIZE_T min = GetLargePageMinimum();
    if (min == 0) return FALSE;

    // Try to enable the privilege
    if (!EnableLargePagePrivilege()) return FALSE;

    // Try a small allocation
    void *test = VirtualAlloc(NULL, min,
        MEM_RESERVE | MEM_COMMIT | MEM_LARGE_PAGES,
        PAGE_READWRITE);
    if (test == NULL) return FALSE;
    VirtualFree(test, 0, MEM_RELEASE);
    return TRUE;
}
```

#### 3.4 Large Pages with Memory-Mapped Files (MapViewOfFile)

- **Source:** [Microsoft Learn - Creating File Mapping Using Large Pages](https://learn.microsoft.com/en-us/windows/win32/memory/creating-a-file-mapping-using-large-pages)

**Key API details:**
```c
// Create file mapping with SEC_LARGE_PAGES
HANDLE hMap = CreateFileMapping(
    INVALID_HANDLE_VALUE,  // pagefile-backed (not a real file)
    NULL,
    PAGE_READWRITE | SEC_COMMIT | SEC_LARGE_PAGES,
    0, size, NULL);

// Map with FILE_MAP_LARGE_PAGES (REQUIRED since Windows 10 1703)
void *view = MapViewOfFile(hMap,
    FILE_MAP_ALL_ACCESS | FILE_MAP_LARGE_PAGES,
    0, 0, size);
```

**Critical Windows version note:** Starting in Windows 10 version 1703, MapViewOfFile maps views using small pages by default, even for file mapping objects created with SEC_LARGE_PAGES. You MUST specify `FILE_MAP_LARGE_PAGES` flag in the MapViewOfFile call. This flag is ignored on older OS versions.

- **Relevance to EdgeLM:** For model loading via mmap, we could use SEC_LARGE_PAGES with CreateFileMapping. However, this only works for pagefile-backed sections (not real files). For model files, we would need to: (1) mmap the model file normally, (2) allocate large pages via VirtualAlloc, (3) memcpy model data into the large page allocation. The memcpy cost (~150-300ms for 600MB at ~2-4 GB/s memcpy speed) is acceptable as a one-time startup cost.

#### 3.5 2MB vs 1GB Pages on Windows

- **2MB pages:** Fully supported via VirtualAlloc with MEM_LARGE_PAGES. GetLargePageMinimum() returns 2097152.
- **1GB pages:** NOT available via standard Windows APIs. Only used internally by Hyper-V for VM memory. No user-mode API to allocate 1GB pages on Windows 11 consumer editions.
- **Transparent Huge Pages equivalent:** Windows does NOT have a direct equivalent of Linux's Transparent Huge Pages (THP). There is no automatic promotion of 4KB pages to 2MB pages. You must explicitly use MEM_LARGE_PAGES. The closest automatic optimization is the OS's use of 64KB "large page" regions internally for some allocations, but this is not controllable by user code.

#### 3.6 Practical TLB Miss Reduction Measurements

- **Source:** [rigtorp.se/hugepages](https://rigtorp.se/hugepages/)
- **Key benchmark data:** On AMD Ryzen 3900X (similar TLB structure to Intel), a hash table benchmark with random access pattern showed:
  - **4KB pages:** 93.43% dTLB miss rate, 0.71 seconds runtime
  - **2MB huge pages:** 0.07% dTLB miss rate, 0.54 seconds runtime
  - **Speedup:** 24% faster with huge pages (and this was only from TLB improvement)

- **Relevance to EdgeLM:** Our weight access pattern is sequential streaming, not random. TLB miss rates will be lower than the 93% seen in random-access benchmarks, but still significant because:
  1. Hardware prefetchers stop at page boundaries (reducing effective bandwidth)
  2. Software prefetch may be silently dropped on TLB miss
  3. With multi-threaded access, multiple cores compete for STLB entries
  - Expected improvement for EdgeLM: 10-20% from large pages alone, primarily from prefetcher boundary elimination and TLB miss elimination.

---

### 4. Cache Line Alignment and False Sharing

#### 4.1 The 64-Byte Cache Line on Alder Lake

- **Source:** Intel SDM, [Google Highway thread pool](https://github.com/google/highway)
- **Key facts:**
  - Cache line size is 64 bytes on ALL Intel processors since Pentium 4 (including Alder Lake P-cores and E-cores)
  - L1D, L2, and L3 all use 64-byte cache lines
  - The Adjacent Cache Line Prefetcher (AMP) on Golden Cove prefetches the "pair" line to form 128-byte aligned blocks, but the coherency unit is still 64 bytes
  - MESI/MESIF protocol operates on 64-byte cache lines

#### 4.2 False Sharing: The Hidden Performance Killer

**What it is:** When two threads write to different variables that happen to occupy the same 64-byte cache line, the coherency protocol forces the line to bounce between cores. Each bounce costs ~40-70 cycles (L3 round-trip) or more.

**Cost measurement:**
- Cache line bounce between two P-cores: ~40-50 cycles (via L3)
- Cache line bounce between P-core and E-core: ~50-70 cycles (different ring stop)
- Cache line bounce across sockets (if applicable): ~100-300 cycles
- For Alder Lake single-socket: ~40-70 cycles per bounce

**Common false sharing patterns in inference:**

Pattern 1: Thread-local accumulators in adjacent memory
```c
// BAD: accumulators packed together
float accum[NUM_THREADS];  // Thread 0 writes accum[0], Thread 1 writes accum[1]
                            // They're in the same cache line!

// GOOD: padded to cache line boundaries
typedef struct {
    float value;
    char padding[60];  // Pad to 64 bytes total
} PaddedAccum;
PaddedAccum accum[NUM_THREADS] __attribute__((aligned(64)));
```

Pattern 2: Shared counters and flags
```c
// BAD: work counters adjacent
struct {
    volatile int tasks_done;      // Written by workers
    volatile int tasks_total;     // Read by workers
    volatile int shutdown;        // Written by coordinator
} state;  // All in one cache line

// GOOD: each on its own cache line
struct alignas(64) { volatile int tasks_done; } done_counter;
struct alignas(64) { volatile int tasks_total; } total_tasks;
struct alignas(64) { volatile int shutdown; } shutdown_flag;
```

Pattern 3: Output buffer overlap in multi-threaded GEMV
```c
// BAD: output elements assigned without alignment consideration
// Thread 0: output[0..chunk-1], Thread 1: output[chunk..2*chunk-1]
// If chunk * sizeof(float) is not a multiple of 64, last elements overlap cache lines

// GOOD: ensure chunk boundaries align to cache lines
int elems_per_cacheline = 64 / sizeof(float);  // 16 floats per cache line
int chunk = ALIGN_UP(M / num_threads, elems_per_cacheline);
```

#### 4.3 Alignment Techniques in C

- **Source:** [llama.cpp ggml.c](https://github.com/ggerganov/llama.cpp), [Google Highway](https://github.com/google/highway)

**Method 1: Compiler attribute (GCC/Clang)**
```c
float buffer[1024] __attribute__((aligned(64)));

typedef struct __attribute__((aligned(64))) {
    float data[16];
} AlignedBlock;
```

**Method 2: MSVC-specific**
```c
__declspec(align(64)) float buffer[1024];

// Or with the alignas keyword (C11/C++11)
alignas(64) float buffer[1024];
```

**Method 3: Dynamic allocation**
```c
// _aligned_malloc (Windows/MSVC)
void *ptr = _aligned_malloc(size, 64);
// Must free with _aligned_free(ptr)

// posix_memalign (POSIX)
void *ptr;
posix_memalign(&ptr, 64, size);
// Free with free(ptr)

// _mm_malloc (Intel intrinsics, cross-platform)
void *ptr = _mm_malloc(size, 64);
// Must free with _mm_free(ptr)

// C11 aligned_alloc
void *ptr = aligned_alloc(64, ALIGN_UP(size, 64));
// Free with free(ptr)
```

**Method 4: VirtualAlloc (always page-aligned, so always 64-byte aligned)**
```c
void *ptr = VirtualAlloc(NULL, size, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
// VirtualAlloc returns page-aligned addresses (4096-byte aligned minimum)
// This is automatically 64-byte aligned
```

**What llama.cpp does:**
- Default alignment is 64 bytes (`GGML_MEM_ALIGN = 64`, or 256 on s390x)
- Uses `posix_memalign()` on POSIX, `_aligned_malloc()` on Windows
- Has HBM (High Bandwidth Memory) support via `hbwmalloc.h`
- macOS uses `vm_allocate()` for page-aligned allocation
- Assertion macro: `GGML_ASSERT_ALIGNED(ptr)` checks alignment at runtime

**What Google Highway does:**
- `alignas(HWY_ALIGNMENT)` on all worker structs (typically 64 bytes)
- `static_assert(sizeof(Worker) == HWY_ALIGNMENT)` ensures exact cache line sizing
- Per-thread statistics arrays padded: `uint64_t per_thread_[kMaxThreads * kU64PerLine]`
- Explicit padding bytes calculated: `padding_[HWY_ALIGNMENT - 56 - 6*sizeof(void*) - sizeof(victims_)]`

- **Relevance to EdgeLM:** Our custom arena allocator must enforce 64-byte alignment for ALL buffers. The `arena_alloc` function in the deep dive already does this, but we should validate with runtime assertions. For thread-local accumulators in the GEMV kernel, each thread's accumulator must start on a 64-byte boundary.
- **AVX2 specifics:** While AVX2 `VMOVDQA` requires 32-byte alignment, we should align to 64 bytes (cache line) to also prevent false sharing. The cost of 64 vs 32 byte alignment is zero (just 0-31 bytes of extra padding per allocation).

#### 4.4 Cache Line Bouncing in Producer-Consumer Patterns

- **Relevance to pipeline stages:** Our inference pipeline has producer-consumer relationships:
  1. Attention produces -> FFN consumes
  2. Layer N produces -> Layer N+1 consumes
  3. Weight loader produces -> Compute kernel consumes

**Key patterns:**
- **Single-writer principle:** Each cache line should have at most one writer. Readers can be on any core without causing bouncing (shared state in MESI).
- **Read-after-write latency:** If Thread A writes a cache line and Thread B reads it, the line must transfer via L3 (~40-50 cycles on Alder Lake). For pipeline stages, this is unavoidable but can be minimized by ensuring the handoff point is a small, aligned buffer.
- **Double-buffering eliminates most bouncing:** While Thread A writes to buffer_0, Thread B reads from buffer_1 (which Thread A wrote previously). No cache line sharing between threads for the actual data.

---

### 5. NUMA-like Behavior on Alder Lake

#### 5.1 Single-Socket, but NOT Uniform

- **Source:** [Microsoft Learn - NUMA Support](https://learn.microsoft.com/en-us/windows/win32/procthread/numa-support), [Chips and Cheese - Gracemont](https://chipsandcheese.com/2022/08/11/gracemont-revenge-of-the-atom-cores/)
- **Key finding:** The i7-12700H is a single-socket processor and Windows reports it as a single NUMA node. However, there ARE non-uniform memory access latencies due to the ring bus topology and P-core/E-core placement.

**Memory access latency variations:**

| Access from | L3 Latency | Reason |
|-------------|-----------|--------|
| P-core near memory controller | ~38-42 cycles | Short ring hop |
| P-core far from memory controller | ~45-52 cycles | Long ring hop |
| E-core cluster | ~50-60+ cycles | Different ring stop, may traverse more hops |

**Ring bus topology on Alder Lake-H:**
- The 12 L3 slices are distributed across the ring
- P-cores and E-core clusters are at different ring stops
- The memory controller is at a fixed ring stop
- Accesses from cores farther from the memory controller or the target L3 slice incur more ring hops
- Each ring hop adds ~1-2 cycles

- **Relevance to EdgeLM:** While the latency differences are small (10-15 cycles, not 100+ like real NUMA), they can accumulate during memory-intensive streaming. For optimal performance:
  1. Pin the main inference threads to P-cores that are topologically closer to the memory controller (if we can determine this via hardware enumeration)
  2. Use `VirtualAlloc2` with `MemExtendedParameterNumaNode` to place weight memory near the cores that will access it (even though it is a single NUMA node, the hint affects physical page placement)
  3. In practice, the difference is small enough (~3-5%) that it is a Phase 5 optimization

#### 5.2 Memory Controller Contention: P-Core vs E-Core

- **Key finding:** P-cores and E-cores share the SAME memory controller and DDR4 bus. There is no memory bandwidth partitioning.

**Implications:**
- If E-cores are running memory-intensive tasks while P-cores stream weights, they COMPETE for the shared ~40 GB/s DDR4 bandwidth
- This is why the deep dive says "Do NOT overlap CPU and iGPU memory-intensive work"
- The same applies to E-core memory-intensive work: keep E-cores doing compute-only or I/O-only tasks during weight streaming
- Intel Thread Director does NOT account for memory bandwidth contention

#### 5.3 Windows APIs for Topology-Aware Allocation

- **Source:** [Microsoft Learn - VirtualAlloc2](https://learn.microsoft.com/en-us/windows/win32/api/memoryapi/nf-memoryapi-virtualalloc2), [Microsoft Learn - CPU Sets](https://learn.microsoft.com/en-us/windows/win32/procthread/cpu-sets)

**Topology discovery:**
```c
// Enumerate NUMA topology
ULONG highestNode;
GetNumaHighestNodeNumber(&highestNode);
// On i7-12700H: highestNode = 0 (single NUMA node)

// Discover which processors are P-cores vs E-cores
DWORD bufferSize = 0;
GetLogicalProcessorInformationEx(RelationProcessorCore, NULL, &bufferSize);
// Parse the returned SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX structures
// to identify P-core and E-core logical processors

// CPU Sets API (Windows 10+) for soft affinity
ULONG cpuSetIds[MAX_CPUS];
ULONG cpuSetIdCount;
GetSystemCpuSetInformation(NULL, 0, &bufferSize, GetCurrentProcess(), 0);
// Each CPU Set has EfficiencyClass: 0 = E-core, 1 = P-core
```

**Windows 11 improvements:**
- CPU Sets span all processor groups by default (no more single-group limitation)
- `SetProcessDefaultCpuSetMasks` / `SetThreadSelectedCpuSetMasks` for multi-group affinity
- Processes are no longer constrained to a single processor group
- The "Primary Group" concept assigns threads an ideal processor but allows execution on any group

---

### 6. Cache Coloring and L3 Slice Selection

#### 6.1 L3 Organization on Alder Lake

- **Source:** [Chips and Cheese - Golden Cove](https://chipsandcheese.com/2021/12/02/popping-the-hood-on-golden-cove/)

**L3 structure:**
- 24 MB shared across all cores (P + E)
- Organized as 12 slices of 2 MB each
- Each slice delivers 32 bytes per cycle
- Total L3 bandwidth: 12 x 32 = 384 bytes/cycle theoretical
- Connected via ring bus

**Slice selection (hash function):**
- Intel uses a hash function on physical address bits to determine which L3 slice stores a cache line
- The exact hash function is not publicly documented but is known to use bits [16:6] of the physical address (bits above the 64-byte cache line offset)
- The goal is to distribute data evenly across all 12 slices to avoid hotspots

#### 6.2 Can We Influence L3 Slice Mapping?

- **Key finding:** On Windows, we cannot directly control which L3 slice our data maps to, because:
  1. The hash function operates on PHYSICAL addresses, and we only control VIRTUAL addresses
  2. Windows manages the virtual-to-physical mapping
  3. Large pages (2MB) are physically contiguous, which gives us more predictable physical address distributions
  4. There is no "page coloring" API on Windows

**Indirect influence strategies:**
1. **Use large pages:** 2MB physically contiguous pages span multiple L3 slices, ensuring even distribution
2. **Align allocations to large boundaries:** With 2MB pages, every 2MB block is physically contiguous, giving predictable L3 slice distribution across the block
3. **Avoid power-of-2 strides:** When multiple buffers have the same size and alignment, they can map to the same L3 slices, causing conflict misses. Add small offsets to break the pattern.
4. **L3 conflict avoidance for activation buffers:** If input and output activation buffers are both 12.8KB and start at the same alignment, they may alias in L3. Offset one by a non-power-of-2 amount.

- **Relevance to EdgeLM:** Cache coloring is a low-priority optimization for our workload because:
  - Our primary bottleneck is DRAM bandwidth, not L3 capacity
  - Weight streaming uses L3 as a pass-through (weights evict before reuse)
  - Activation buffers (12.8 KB) are small enough to fit in L1/L2
  - KV cache is the main L3-resident structure, but it is accessed sequentially
  - **Estimated impact:** 1-3% at most. Defer to Phase 5.

---

### 7. TLB Shootdown Costs for Multi-Threaded Inference

#### 7.1 What Causes TLB Shootdowns

- **Source:** Intel SDM Vol. 3A, Chapter 4
- **Key finding:** TLB shootdowns are Inter-Processor Interrupts (IPIs) that force all cores to invalidate specific TLB entries.

**Common causes:**
1. `VirtualFree()` or `VirtualProtect()` -- changing page mappings or protections
2. `munmap()` / `UnmapViewOfFile()` -- removing memory mappings
3. `madvise(MADV_DONTNEED)` or `DiscardVirtualMemory()` -- hinting pages are not needed
4. OS memory management (page frame reclamation, working set trimming)

**Cost per shootdown:**
- IPI delivery: ~1-5 microseconds
- Each receiving core must: stop execution, flush relevant TLB entries, acknowledge
- On Alder Lake with 14 cores (6P + 8E): ~5-15 microseconds total
- During a shootdown, ALL affected cores stall

#### 7.2 Avoiding TLB Shootdowns in Inference

**Strategy: Allocate once, never free during inference.**
1. Allocate all model weights, KV cache, and activation buffers at startup
2. Use arena allocator for sub-allocation (no VirtualAlloc/VirtualFree during inference)
3. Never call VirtualProtect during the hot path
4. Use large pages (fewer total pages = fewer potential shootdown targets)
5. `MEM_LARGE_PAGES` memory is non-pageable, so the OS will never reclaim these pages (eliminating OS-initiated shootdowns for model memory)

- **Relevance to EdgeLM:** With our arena allocation pattern (allocate at startup, never free during inference), TLB shootdowns should be zero during token generation. This is already the recommended approach.

---

### 8. PrefetchVirtualMemory API for Model Loading

#### 8.1 Windows-Specific Prefetch API

- **Source:** [Microsoft Learn - PrefetchVirtualMemory](https://learn.microsoft.com/en-us/windows/win32/api/memoryapi/nf-memoryapi-prefetchvirtualmemory)

```c
// Prefetch model weights into physical memory
WIN32_MEMORY_RANGE_ENTRY ranges[1];
ranges[0].VirtualAddress = model_weights;
ranges[0].NumberOfBytes = model_size;

BOOL success = PrefetchVirtualMemory(
    GetCurrentProcess(),
    1,          // number of entries
    ranges,
    0           // flags (must be 0)
);
```

**Key benefits:**
- Issues large, concurrent I/O requests to bring pages from disk/pagefile into physical memory
- More efficient than page-faulting (which does one page at a time)
- Non-blocking: purely a performance hint, does not add pages to working set
- Prefetched pages are cached in physical memory and will be added to working set on first access

**What llama.cpp does:** On Windows, llama-mmap.cpp uses `PrefetchVirtualMemory` after `MapViewOfFile` to pre-fault the entire model file mapping into memory.

- **Relevance to EdgeLM:** Use PrefetchVirtualMemory immediately after mmap'ing the model file, before copying weights to large-page buffers. This ensures the source data is in physical memory for the memcpy, avoiding page fault stalls during the weight repacking phase.

---

### 9. Practical Implementation Strategy for EdgeLM

#### 9.1 Memory Allocation Hierarchy (Recommended)

```
Startup sequence:
1. Check if large pages are available (CanUseLargePages())
2. If yes: VirtualAlloc with MEM_LARGE_PAGES for:
   - Model weights buffer (round up to 2MB multiple)
   - KV cache buffer (round up to 2MB multiple)
   - Arena for activation buffers (round up to 2MB multiple)
3. If no: Fall back to VirtualAlloc without MEM_LARGE_PAGES
   - Still get page-aligned (4096-byte) allocation
   - Use PrefetchVirtualMemory to pre-fault all pages
4. Ensure all allocations are 64-byte aligned (automatic with VirtualAlloc)
5. Use arena sub-allocation for all runtime buffers
6. Never call VirtualAlloc or VirtualFree during inference
```

#### 9.2 Recommended Buffer Layout

```c
// All sizes rounded up to 2MB boundaries for large page alignment
typedef struct {
    // Model weights -- largest allocation, benefits most from large pages
    void *weights;          // ~600 MB (2MB-aligned, large pages)
    size_t weights_size;

    // KV cache -- second largest, frequently accessed
    void *kv_cache;         // ~100 MB (2MB-aligned, large pages)
    size_t kv_size;

    // Activation arena -- double-buffered, reused per layer
    void *arena;            // ~100 MB (2MB-aligned, large pages)
    size_t arena_size;

    // Thread-local accumulators -- must be 64-byte aligned per thread
    void *thread_outputs;   // ~64 KB (64-byte aligned per thread)

    // Tokenizer data -- small, standard pages OK
    void *tokenizer;        // ~10 MB (standard allocation)
} InferenceMemory;
```

#### 9.3 Runtime Alignment Validation

```c
#define CACHE_LINE 64
#define ASSERT_ALIGNED(ptr, align) \
    do { \
        if (((uintptr_t)(ptr)) % (align) != 0) { \
            fprintf(stderr, "ALIGNMENT FAULT: %s = %p, expected %zu-byte aligned\n", \
                    #ptr, (ptr), (size_t)(align)); \
            abort(); \
        } \
    } while(0)

// Use at startup to validate all critical buffers
void ValidateAlignment(InferenceMemory *mem) {
    ASSERT_ALIGNED(mem->weights, 2 * 1024 * 1024);  // 2MB for large pages
    ASSERT_ALIGNED(mem->kv_cache, CACHE_LINE);
    ASSERT_ALIGNED(mem->arena, CACHE_LINE);
    ASSERT_ALIGNED(mem->thread_outputs, CACHE_LINE);
}
```

#### 9.4 VTune Profiling Methodology for TLB/Cache Analysis

**Recommended VTune analysis types:**
1. **Memory Access Analysis:** Detects DTLB misses, cache misses, bandwidth utilization
2. **Microarchitecture Exploration:** Shows port utilization, pipeline stalls from TLB misses
3. **Threading Analysis:** Detects false sharing and lock contention

**Key hardware counters for TLB analysis:**
- `DTLB_LOAD_MISSES.MISS_CAUSES_A_WALK` -- count of L1 DTLB misses causing page walks
- `DTLB_LOAD_MISSES.WALK_COMPLETED` -- completed page walks
- `DTLB_LOAD_MISSES.WALK_DURATION` -- cycles spent in page walks
- `DTLB_LOAD_MISSES.STLB_HIT` -- L1 DTLB miss served by STLB
- `MEM_INST_RETIRED.STLB_MISS_LOADS` -- loads that missed STLB

**Methodology:**
1. Baseline: Run inference with 4KB pages, measure DTLB miss rate
2. Optimized: Run with 2MB large pages, measure DTLB miss rate
3. Compare: Expect STLB miss rate to drop from >90% to <1%
4. Verify: Total page walk cycles should drop by >90%

---

### 10. Summary: Optimization Priority and Expected Impact

| Optimization | Expected Improvement | Implementation Phase | Complexity |
|-------------|---------------------|---------------------|------------|
| 2MB large pages for model weights | 10-20% | Phase 1 (do immediately) | Low |
| 64-byte alignment for all buffers | 5-15% (avoids false sharing) | Phase 1 | Low |
| Arena allocator (no malloc in hot path) | 5-10% (avoids TLB shootdowns) | Phase 1 | Low |
| Per-thread cache line padding | 5-15% (eliminates false sharing) | Phase 1 | Low |
| PrefetchVirtualMemory for model loading | 20-50% faster load time | Phase 1 | Low |
| Software prefetch across page boundaries | 5-10% | Phase 2 | Medium |
| MEM_64K_PAGES for KV cache/activations | 2-5% | Phase 3 | Low |
| VirtualAlloc2 NUMA node hints | 1-3% | Phase 5 | Low |
| Cache coloring / L3 slice optimization | 1-3% | Phase 5 | High |
| Topology-aware thread placement | 2-5% | Phase 3 | Medium |

**Total estimated improvement from all memory/TLB optimizations: 25-50% over naive implementation.**

The most impactful optimizations (large pages, alignment, arena allocation) are also the simplest to implement and should be done in Phase 1.

---

## Sources

1. [Microsoft Learn - Large Page Support](https://learn.microsoft.com/en-us/windows/win32/memory/large-page-support) -- VirtualAlloc with MEM_LARGE_PAGES requirements
2. [Microsoft Learn - VirtualAlloc](https://learn.microsoft.com/en-us/windows/win32/api/memoryapi/nf-memoryapi-virtualalloc) -- Complete API documentation
3. [Microsoft Learn - VirtualAlloc2](https://learn.microsoft.com/en-us/windows/win32/api/memoryapi/nf-memoryapi-virtualalloc2) -- NUMA node, alignment, MEM_64K_PAGES
4. [Microsoft Learn - Creating File Mapping Using Large Pages](https://learn.microsoft.com/en-us/windows/win32/memory/creating-a-file-mapping-using-large-pages) -- SEC_LARGE_PAGES, FILE_MAP_LARGE_PAGES
5. [Microsoft Learn - PrefetchVirtualMemory](https://learn.microsoft.com/en-us/windows/win32/api/memoryapi/nf-memoryapi-prefetchvirtualmemory) -- Pre-fault memory
6. [Microsoft Learn - Processor Groups](https://learn.microsoft.com/en-us/windows/win32/procthread/processor-groups) -- Windows 11 multi-group changes
7. [Microsoft Learn - NUMA Support](https://learn.microsoft.com/en-us/windows/win32/procthread/numa-support) -- Topology APIs
8. [Microsoft Learn - CPU Sets](https://learn.microsoft.com/en-us/windows/win32/procthread/cpu-sets) -- Hybrid processor affinity
9. [Microsoft Learn - Assigning Privileges](https://learn.microsoft.com/en-us/windows/win32/secbp/assigning-privileges-to-an-account) -- SeLockMemoryPrivilege setup
10. [Chips and Cheese - Gracemont: Revenge of the Atom Cores](https://chipsandcheese.com/2022/08/11/gracemont-revenge-of-the-atom-cores/) -- E-core TLB details: 48-entry L1 DTLB, 2048-entry L2 TLB 4-way, 3-cycle L1D
11. [Chips and Cheese - Popping the Hood on Golden Cove](https://chipsandcheese.com/2021/12/02/popping-the-hood-on-golden-cove/) -- P-core memory subsystem, L2 1.25MB, L3 12 slices
12. [Wikipedia - Golden Cove](https://en.wikipedia.org/wiki/Golden_Cove) -- 48KB L1D, 32KB L1I, 192 load / 114 store buffers, 512-entry ROB
13. [Wikipedia - Gracemont](https://en.wikipedia.org/wiki/Gracemont_(microarchitecture)) -- 32KB L1D, 64KB L1I, 2MB L2 per module
14. [blog.stuffedcow.net - Page Walk Coherence](https://blog.stuffedcow.net/2015/08/pagewalk-coherence/) -- Intel speculative page walks, 84-cycle Haswell overhead, Intel/AMD differences
15. [llama.cpp ggml.c](https://github.com/ggerganov/llama.cpp) -- 64-byte GGML_MEM_ALIGN, posix_memalign, platform-specific allocation
16. [llama.cpp llama-mmap.cpp](https://github.com/ggerganov/llama.cpp) -- VirtualLock, MapViewOfFile, PrefetchVirtualMemory on Windows
17. [Google Highway thread_pool.h](https://github.com/google/highway) -- False sharing avoidance: alignas(HWY_ALIGNMENT), static_assert sizeof == 64
18. [rigtorp.se/hugepages](https://rigtorp.se/hugepages/) -- TLB miss reduction: 93.43% -> 0.07% with 2MB pages, 24% speedup
19. [Raymond Chen - Windows Page Sizes](https://devblogs.microsoft.com/oldnewthing/20210510-00/?p=105200) -- Page size history, constraint on mixed page sizes
20. Intel SDM Vol. 3A Chapter 4 -- TLB hierarchy, page walks, INVLPG
21. Intel Optimization Reference Manual -- Prefetcher behavior, TLB-prefetch interaction, cache optimization

## Audit Addendum (2026-04-02)

- **Write traffic needs separate accounting from read traffic.** KV writes,
  quantize-on-write, and scratch-buffer clears are small relative to weight
  reads, but still large enough to perturb decode on a `~40 GB/s` laptop.
- **Non-temporal stores should be treated surgically.** They help for true
  streaming write-only paths, but they are harmful when the next stage reuses
  the data from cache.
- **Cold-start page-fault behavior is worth measuring explicitly.** A startup
  trace that separates:
  - file I/O,
  - page population,
  - repack CPU time,
  - and cache warming

  would make the load-path memory story much sharper.
