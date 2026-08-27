# metalscope: Roofline Analysis for Metal Compute Kernels Against Measured, Not Asserted, Ceilings

*Version 0.1.0 · August 2026*

## Abstract

Apple's GPU tooling can tell you what a Metal kernel did. It cannot tell you how
far that kernel is from the best its shape allows on the chip in front of you.
The two questions that matter when optimising an ML kernel — *is this
memory-bound or compute-bound, and how much is left on the table?* — require an
arithmetic intensity, which requires a FLOP count and a byte count, which
Instruments has no way to obtain because it does not know a GEMM from a
softmax. They also require honest ceilings, and the published peak-FLOPS figures
for Apple GPUs are community arithmetic that no real kernel approaches.

metalscope is a small, dependency-free Swift tool that supplies both. It
computes FLOPs and compulsory DRAM traffic *analytically* from user-supplied ML
kernel-shape annotations (`.gemm(m:n:k:)`, `.attention(b:h:s:d:)`, `.norm(n:)`,
`.elementwise(n:)`), measures the local chip's real compute and bandwidth
ceilings before scoring anything against them, and reports each dispatch's
position on the resulting roofline — plus a static occupancy analysis derived
from `MTLComputePipelineState` that works on chips exposing no occupancy
counters at all, which is every Apple chip shipped to date.

The design commitment throughout is that a profiler must never print a
confidently wrong number. This paper describes the mechanisms that enforce that
— a three-tier timing ladder that degrades rather than extrapolates, derived-on-
read occupancy so a trace cannot contradict itself, and a deliberate refusal to
flag low occupancy ratios — and evaluates them on an Apple M1 Pro, including two
real measurement bugs that the tool found in itself.

---

## 1. Motivation

Instruments' Metal System Trace is a good instrument-level profiler. It is not
an ML profiler, and the gap is structural rather than a matter of missing
features.

**It has no notion of a kernel's shape.** Roofline analysis needs arithmetic
intensity: FLOPs per byte of DRAM traffic. For a matmul of *m×n×k* that is
`2mnk / e(mk + kn + mn)` — exactly, from the shape alone, before the kernel has
run. Instruments sees a compute dispatch with a thread count, and cannot
distinguish 2*mnk* useful FLOPs from the same dispatch doing something else, so
it cannot place the kernel on a roofline. The alternatives are to instrument the
shader (invasive, and it perturbs what you are measuring) or to estimate from
counters (unavailable on Apple silicon; see §6).

**It has no chip ceilings.** Even given an intensity, "823 GFLOP/s" means nothing
without knowing what the chip can do. Apple does not publish peak FLOPS. The
figures in circulation — 5.2 TF for an M1 Pro, 10.4 TF for an M1 Max — are
`GPU cores × ALUs × 2 FLOP × boost clock`, which presumes an FMA issued every
cycle at boost. §5.2 shows that number is off by roughly 1.5–1.7× for real GEMM
on two different chips.

**It has no diff.** Kernel optimisation is a loop: measure, change, measure,
decide whether the change helped. Instruments produces a trace document per run
and leaves the comparison to the human — precisely the task where a 2%
regression hides.

metalscope targets that gap and nothing else. It is not a replacement for
Instruments; the two answer different questions, and §6 lists what stays
Instruments-only and why.

---

## 2. Related work

**Nsight Compute** (NVIDIA) is the reference point and the direct inspiration:
a roofline section, per-kernel occupancy against a hardware model, source-level
counter attribution, and a baseline/diff workflow. It can do all of that because
CUDA exposes a rich, documented counter surface and NVIDIA publishes the machine
model occupancy calculations require. Neither condition holds on Apple silicon —
`MTLDevice.counterSets` returns `timestamp` and nothing else on every Apple part
metalscope has run on, and the per-core thread and register budgets are
unpublished. metalscope therefore keeps Nsight Compute's *shape* while replacing
every mechanism underneath.

**Instruments / Xcode GPU capture** (Apple) provides Metal System Trace, shader
profiling with per-line cost attribution, and counter data through a private
path. It is the right tool for instruction-level stalls, bank conflicts and
memory-hierarchy behaviour, and offers no roofline against chip peaks, no
kernel-shape awareness, and no programmatic diff.

**Roofline analysis** originates with Williams, Waterman and Patterson (2009),
which framed performance as `min(peak FLOP/s, AI × peak bandwidth)` and made the
bandwidth/compute boundary the central diagnostic. Two later extensions matter
here: cache-aware and hierarchical rooflines (Ilic et al.), which model each
level of the memory hierarchy rather than DRAM alone, and GPU-specific empirical
roofline construction (Yang et al.; the Empirical Roofline Toolkit from LBNL),
which established that *measured* ceilings are the only defensible ones on
hardware whose vendor figures are theoretical. metalscope sits in that empirical
tradition — `calibrate` is a two-workload roofline toolkit specialised to a
laptop-scale run time — but implements the single-level DRAM roofline
deliberately, because the compulsory-traffic model (§3.2) bounds intensity
honestly without a cache model, and a half-calibrated cache roofline would be
worse than none.

**Analytic FLOP counting** from model shape is standard in the ML systems
literature — every FLOPs-per-token estimate in a transformer scaling paper does
it — but is rarely wired into a GPU profiler as the source of truth for
intensity. That connection is metalscope's specific contribution.

---

## 3. Design

### 3.1 Measured, not asserted, peaks

Every efficiency percentage metalscope prints is a fraction of a ceiling, so the
ceiling is the most load-bearing number in the tool. `calibrate` measures it:
a sweep of square MPS GEMMs for the fp32 and fp16 compute roofs, and a STREAM
triad (`a = b + s·c`, three streams touched per element) across several working
set sizes for the bandwidth roof. Results are cached per device name in
`~/.metalscope/peaks.json`.

Two methodological choices turned out to matter more than the choice of
benchmark:

**Run length.** Apple GPUs ramp their clocks over tens of milliseconds. The same
1024³ MPS GEMM measures 0.88 TF over 4 iterations and 3.3 TF over 120 — a
3.75× difference from iteration count alone. `calibrate` therefore probes once
to estimate per-iteration cost, then auto-sizes the iteration count so each run
is ~250 ms of GPU work (`--target-ms`).

**Best-of-repeats, not mean.** A ceiling is what the hardware *can* sustain.
Thermal throttling and scheduler contention only ever move a sample downward, so
the maximum over repeats is the estimator that converges on the truth while the
mean converges on the machine's current mood.

When no measurement exists, metalscope falls back to a table of community
figures and labels every number derived from them **"spec-sheet folklore"**, in
the header of every table. The label is not decoration; §5.2 shows what
believing those figures costs.

### 3.2 Analytic FLOPs and bytes, from annotation rather than estimation

A `KernelShape` annotation supplies the numerator and denominator of arithmetic
intensity directly:

| shape | FLOPs | bytes (*e* = element size) |
| --- | --- | --- |
| `gemm(m,n,k)` | `2mnk` | `e(mk + kn + mn)` |
| `attention(b,h,s,d)` | `4bhs²d` | `4e·bhsd` |
| `elementwise(n)` | `n` | `2en` |
| `norm(n)` | `5n` | `2en` |
| `opaque(flops,bytes)` | as given | as given |

Two decisions inside this table carry most of its value.

**Bytes are compulsory traffic, assuming perfect on-chip reuse** — each input
read once, each output written once. Real kernels move more. Using the
compulsory count as the denominator makes arithmetic intensity an *upper bound*,
which makes roofline placement conservative in the useful direction: a kernel
that looks bandwidth-bound under this model really is bandwidth-bound, because
any additional traffic only pushes intensity lower. The cost is that efficiency
can exceed 100% when a working set is partly cache-resident (§5.5), which is
informative rather than confusing once labelled.

**`attention` models a fused kernel.** Its byte count is Q, K, V in and O out;
the *s×s* score matrix never appears. An unfused implementation therefore
reports well under 100%, and that gap is a *measurement of the cost of not
fusing* rather than a modelling error. In §5.4 an MPS-composite attention
implementation lands at 43.8% of the compute ceiling, and the report attaches
the caller's own note explaining why.

The alternative — inferring shapes heuristically from dispatch dimensions — was
rejected. A profiler that guesses the shape and then reports a confident
intensity derived from the guess is exactly the failure mode this tool exists to
avoid. Annotation is more work for the user and is the only version that can be
trusted.

### 3.3 The three-tier timing ladder

Metal on Apple silicon supports counter sampling at `.atStageBoundary` only —
not `.atDispatchBoundary`, not `.atBlitBoundary`. The consequence shapes the
whole capture design: **there is no way to bracket someone else's encoders with
counter samples.** metalscope cannot inject into a framework's dispatches, so it
degrades instead, and every kernel record states which tier it reached:

1. **`counter-sample-buffer`** — per-encoder GPU timestamps, for encoders
   metalscope created itself.
2. **`command-buffer`** — `gpuStartTime`/`gpuEndTime`. Real GPU time at
   command-buffer granularity; used when a framework (MPS, MLX) creates the
   encoders. Each captured region is its own command buffer, committed and
   waited on, precisely so this tier still measures exactly the annotated work.
3. **`host`** — wall clock around commit and wait. Includes scheduling latency;
   a fallback, not a measurement.

The critical rule is that **a region claims tier 1 only if every encoder in it
was sampled.** Partial sampling demotes the whole region. This is not
fastidiousness: §5.3 documents the bug that made the rule necessary.

Timestamps are decoded as `UInt64`, never `Double`. GPU timestamps are
nanoseconds since boot; after ~104 days of uptime they exceed 2⁵³ and a `Double`
begins rounding them, silently shortening every kernel duration by an amount
that grows with the machine's uptime.

### 3.4 Occupancy derived on read

`MTLComputePipelineState` exposes `maxTotalThreadsPerThreadgroup`,
`threadExecutionWidth` and `staticThreadgroupMemoryLength`. Paired with the
threadgroup size a dispatch actually used, those four numbers support a static
occupancy analysis that needs **no counters at all** — and therefore works
identically on every chip in the matrix, including the ones that expose nothing.

The trace stores *only those raw inputs*. `simdGroupsPerThreadgroup`,
`threadgroupOccupancy`, `laneUtilization`, `idleLanesPerThreadgroup`,
`threadgroupMemoryPressure` and the limiter classification are all derived when
the trace is read. This is a correctness property, not an economy: a trace
storing both inputs and ratios could be edited, merged, or written by an older
tool into a state where they disagree, and a reader would have no way to know
which half to believe. Derived-on-read makes that state unrepresentable.
`report --json` emits the derived values too, so a consumer never has to
re-implement the arithmetic.

SIMD groups per threadgroup are rounded **up**, which is the entire point: a
100-thread threadgroup still costs four full 32-wide SIMD groups, and the 28
lanes in the fourth are launched idle in every threadgroup, forever.

Getting the pipeline at all is a design constraint rather than a detail. Metal
offers no way to ask an encoder which pipeline it is holding, and metalscope
does not swizzle. So `CaptureRegion` provides `dispatchThreads` /
`dispatchThreadgroups` wrappers that set the pipeline, encode the dispatch, and
record its occupancy in one call — the recorded shape is definitionally the
shape that ran. Work encoded by MPS carries no occupancy block, and the report
prints a dash rather than a guess. **Absent means "not observed", never "nothing
to report".**

### 3.5 Deliberately not flagging low occupancy ratio

The single most consequential judgement in the tool is a negative one: **a low
`threadgroupOccupancy` is not a finding, and metalscope does not report it as
one.** 256 threads out of a 1024-thread pipeline ceiling is 25% "occupancy" and
a perfectly good shape on Apple silicon. A profiler that flagged it would
generate a finding on nearly every well-written kernel, and the reliable human
response to a section that is mostly false positives is to stop reading it —
taking the true positives along. The occupancy analysis is worth having only if
its output is worth acting on.

Three things get flagged instead, all structural facts rather than ratios:

- **`executionWidthAlignment`** — the threadgroup is not a multiple of the SIMD
  width, so lanes idle in *every* threadgroup.
- **`tinyThreadgroup`** — a single SIMD group, so there is nothing for the core
  to interleave against memory latency.
- **`threadgroupMemory`** — static threadgroup memory over half the device
  limit, capping co-residency.

Note the asymmetry this produces, which is the design working as intended: in
§5.4 a kernel at 3.1% occupancy *is* flagged (32 threads is one SIMD group)
while a kernel at 25.0% is not. The number is context; the structure is the
verdict.

A second cross-check runs on top. Before printing an occupancy hint, the report
places the same kernel on the roofline; if it is already at ≥90% of its ceiling,
the hint gains the clause *"note it is already at X% of its bandwidth ceiling,
so this costs lanes, not time."* A ragged threadgroup in a bandwidth-saturated
kernel wastes lanes that were going to be waiting on DRAM regardless. Sending
someone to fix a defect that is not costing them anything is how a profiler
loses its reader.

---

## 4. Implementation

metalscope is 3,702 lines of Swift across three targets, with no external
package dependencies:

| target | lines | contents |
| --- | --- | --- |
| `MetalscopeCore` | 1,234 | trace schema, shape registry, roofline math, occupancy math, diff, table/number formatting. No Metal import; usable in analysis tools |
| `MetalscopeCapture` | 777 | the opt-in capture API, plus per-counter-set resolvers |
| `metalscope` (CLI) | 1,691 | `info`, `calibrate`, `bench`, `profile`, `report`, `diff` |

**Counter resolvers.** `CaptureSession` keeps a resolver per counter set rather
than hardcoding timestamps. Each of Apple's three common sets resolves to a
different C struct — `MTLCounterResultTimestamp` (1 field),
`...StageUtilization` (6), `...Statistic` (8) — and each resolver declares its
field order and takes `resolvedStride` from the real Metal struct. A protocol
extension exposes the invariant `resolvedStride == counterNames.count ×
MemoryLayout<UInt64>.stride`, which the tests assert, so the day Apple adds a
field to one of those structs the suite fails rather than the decode silently
sliding by eight bytes. Every exposed set with a resolver is attached to the
compute pass and summed into `kernels[].counters` with no code change; on an
M1 Pro that loop runs exactly once, over `timestamp`. A counter the GPU declined
to write is `MTLCounterErrorValue` (~0) and is **omitted** rather than recorded:
a missing key means "not written", and must never be readable as zero.

**Attach model.** metalscope does not inject into arbitrary processes. The
target links `MetalscopeCapture` and wraps annotated work in `capture { }`;
`metalscope profile -- <cmd>` runs it with `METALSCOPE_TRACE` set and reports on
whatever it writes. Swizzling and `DYLD_INSERT_LIBRARIES` were rejected — they
are among the properties that make a profiler untrustworthy, and they would not
have solved the underlying problem anyway, since without dispatch-boundary
sampling there is nothing useful to do with an intercepted encoder.

**Trace format.** One JSON schema (currently v2), pretty-printed with sorted keys
so two traces can be compared with plain `diff` as well as with `metalscope
diff`. Every v2 addition is optional, so v1 traces still read; readers reject
schema versions newer than they understand.

---

## 5. Evaluation

All measurements below are on an Apple M1 Pro (16 GB, macOS 26.5.1 build 25F80)
unless stated otherwise, with metalscope 0.1.0 built via `swift build -c
release`.

### 5.1 Test suite and coverage

149 XCTest cases, all passing:

| suite | cases | what it covers |
| --- | --- | --- |
| `TextTableTests` | 31 | table layout (alignment, column sizing, ragged rows, trailing whitespace, Character-vs-byte widths) and every `Fmt` numeric formatter, including NaN, infinity, unit boundaries and clamping |
| `CaptureTests` | 22 | GPU-dependent: capture path, timing ladder, occupancy recording, trace round trip, sample-buffer exhaustion, environment trace path |
| `TraceTests` | 19 | schema v1/v2 round trips, version gate in both directions, timestamp parsing and failure, derived kernel numbers |
| `OccupancyTests` | 19 | derived ratios, limiter classification, folding several dispatch shapes into one record |
| `CounterResolverTests` | 17 | struct-layout invariants and aggregation rules for all three counter sets, against synthetic resolved data |
| `DiffTests` | 16 | alignment by label+shape+precision, positional pairing of repeats, occupancy comparison gating |
| `KernelShapeTests` | 13 | analytic FLOP/byte models and stable JSON encoding |
| `RooflineTests` | 12 | placement, bound classification, ridge tolerance |

Coverage of the two library targets, via `swift test --enable-code-coverage` and
`xcrun llvm-cov report`:

| file | region | function | line |
| --- | --- | --- | --- |
| `MetalscopeCore/TextTable.swift` | 100.00% | 100.00% | 100.00% |
| `MetalscopeCore/Trace.swift` | 100.00% | 100.00% | 100.00% |
| `MetalscopeCore/PeaksStore.swift` | 100.00% | 100.00% | 100.00% |
| `MetalscopeCore/KernelShape.swift` | 98.75% | 100.00% | 100.00% |
| `MetalscopeCore/Diff.swift` | 96.49% | 100.00% | 99.20% |
| `MetalscopeCapture/CounterResolvers.swift` | 92.06% | 100.00% | 100.00% |
| `MetalscopeCore/Occupancy.swift` | 91.04% | 90.91% | 97.69% |
| `MetalscopeCore/Roofline.swift` | 80.95% | 93.75% | 93.52% |
| `MetalscopeCapture/CaptureSession.swift` | 76.97% | 85.29% | 90.69% |
| **total** | **90.45%** | **94.22%** | **96.38%** |

The largest remaining gap is deliberate and hardware-bound: 11 of
`CaptureSession`'s 35 uncovered regions are the auxiliary-counter resolution
path, which cannot execute on a chip that exposes only `timestamp`. Its
aggregation *rules* are covered synthetically in `CounterResolverTests`; only
the Metal plumbing around them is dark. The CLI target is not in this table —
XCTest does not link an executable target — and is exercised end-to-end by hand;
[USAGE.md](USAGE.md) is a transcript of that exercise.

For reference, coverage before this round of test work was 73.20% region /
79.11% function / 84.17% line across 102 cases; `TextTable.swift` — 71 regions
used by every table the tool prints — was at 0%.

### 5.2 Measured versus spec-sheet peaks, on two chips

`calibrate` on this M1 Pro, on an idle machine:

| | measured | spec-sheet | ratio |
| --- | --- | --- | --- |
| fp32 compute | 3.45 TF | 5.2 TF | **66%** |
| fp16 compute | 3.33 TF | 10.4 TF | 32% |
| memory bandwidth | 166.4 GB/s | 200 GB/s | **83%** |

A second machine (lab-02, a Mac Studio M1 Max, macOS 26.5.1) gives an
independent data point:

| | measured | spec-sheet | ratio |
| --- | --- | --- | --- |
| fp32 compute | 6.2 TF | 10.4 TF | **60%** |
| memory bandwidth | 371.5 GB/s | 400 GB/s | **93%** |

The two chips agree on the shape of the error and it is not uniform. **The
bandwidth figure is nearly honest — 83% on M1 Pro and 93% on M1 Max, both
ordinary STREAM efficiencies — while the compute figure overstates achievable
fp32 GEMM by roughly 1.5–1.7× on both.** That asymmetry has a direct
consequence for roofline work: scaling a spec-sheet number by a fudge factor
cannot fix it, because the two roofs are wrong by different amounts, which moves
the ridge point and therefore changes which kernels are classified
bandwidth-bound at all. On this M1 Pro the measured fp32 ridge is 20.7
FLOP/byte; the folklore ridge is 26.0.

The fp16 row deserves separate comment. The 10.4 TF figure assumes double-rate
half precision, which the measurement does not find: fp16 GEMM through MPS
reaches 3.33 TF, essentially the same as fp32. Whether that is a hardware
property or an MPS path choice, scoring against 10.4 TF is indefensible — and
the cost of doing so is concrete. Scoring one trace both ways, the same
`ffn.gemm.half` kernel at the same 701.79 µs reads as **91.8%** of the measured
ceiling and **29.4%** of the folklore one. Against measured peaks it is
essentially optimal and appears in no headroom list; against folklore it is the
second-worst kernel in the trace and is explicitly recommended for attention.
One of those two readings costs a week.

Calibration itself has a noise floor worth publishing. Two full `calibrate` runs
minutes apart on a *loaded* machine returned 3.24 TF / 147.8 GB/s and 3.17 TF /
155.1 GB/s against the idle machine's 3.45 TF / 166.4 GB/s — a ~7% spread on
compute and ~11% on bandwidth. Best-of-repeats mitigates this within a run;
running on an idle machine is still required between them.

### 5.3 Two measurement bugs found by self-application

metalscope was used to profile metalscope's own benchmark workloads throughout
development. Two bugs surfaced that way, and both are now the reason for a
design rule.

**The counter-prefix bug: "541 GB/s on a 200 GB/s machine."** The capture region
allocates a counter sample buffer with finite capacity. Early on, when a region
encoded more encoders than the buffer could sample, the timing path used the
span of the *sampled prefix* and divided it by the region's full iteration
count. The result was a streaming triad reporting **541 GB/s** on a machine
whose bus tops out at 200 GB/s — a number roughly 3× reality, produced with no
error and no warning.

The number was caught only because it was physically impossible — and that is
the important part. Had the workload been a GEMM rather than a bandwidth
benchmark, the same 3× error would have produced an efficiency figure that
looked merely excellent rather than absurd, and it would have shipped. The fix
is the tier-1 rule in §3.3, and the regression test
(`testOversubscribedSampleBufferFallsBackInsteadOfLying`) asserts both that the
starved region does not claim tier 1 and that its duration stays within 2.5× of
the well-sampled one. The general lesson: **degrading is a feature and
extrapolating is a bug.**

**The clock-ramp bug: 0.88 TF or 3.3 TF, same GEMM.** The first `calibrate`
implementation ran a fixed small number of iterations and reported 0.88 TF fp32
for a 1024³ MPS GEMM. The same GEMM over 120 iterations reports 3.3 TF. The
short run was measuring the GPU's clock ramp, not the GPU.

This one is more insidious than the first, because 0.88 TF is not absurd — it is
17% of the folklore ceiling, a number one could rationalise as "MPS overhead on
small matrices" and move on. The fix (probe, then auto-size to ~250 ms of GPU
work) is in §3.1. The rule it produced: a benchmark harness that does not control
run length is measuring the power manager.

The effect is not confined to calibration, as the tool re-demonstrated during
the writing of [USAGE.md](USAGE.md). The example transformer block initially ran
32 iterations per region, and its MPS QKV projection measured 55.4% of the
compute ceiling. Raising the two microsecond-scale kernels ahead of it from 32
to 2048 iterations — leaving the GEMM's own iteration count untouched — moved
that same GEMM to **97.8%**, because the GPU was now fully clocked by the time
it ran. A kernel's measured efficiency depends on what ran before it.

### 5.4 The `act.scale` occupancy finding

`metalscope bench --variant baseline` dispatches its elementwise kernel at 100
threads per threadgroup on purpose, so the occupancy analysis always has a
genuine structural defect to find on any chip. The analysis finds it:

```
  act.scale
    threadgroup      100 threads of a 1024 max (9.8% occupancy)
    simd groups      4 x 32 lanes, 28 idle (78.1% lane use)
    threadgroup mem  0 B of 32.0 KB
    dispatches       167773 threadgroups/dispatch, 172 encoded
    limiter          threadgroup of 100 is not a multiple of the 32-wide SIMD
                     group — 28 of 128 lanes idle in every threadgroup (78.1%
                     lane use); round to 96 or 128
```

Scored against peaks measured in the same machine state, the report adds the
clause that makes this the tool's signature output:

> *…round to 96 or 128; **note it is already at 100.9% of its bandwidth ceiling,
> so this costs lanes, not time**.*

The defect is real — 28 of every 128 lanes launch with nothing to do — and
fixing it wins back nothing, because the kernel is already saturating memory
bandwidth. Confirmed by the `tuned` variant, which fixes the threadgroup to 256
and moves the kernel by **1.02×**.

The contrasting case appears in the [USAGE.md](USAGE.md) worked example, whose
RMS-norm kernel dispatches 32-wide threadgroups (a `tinyThreadgroup` limiter) at
62.6% of its bandwidth ceiling — below the 90% caveat threshold, so no caveat is
printed. Widening it to 256 threads yields **1.25×** and clears the limiter.

Two structurally similar occupancy findings, two opposite recommendations,
distinguished automatically by the roofline cross-check. Neither Instruments nor
a bare occupancy calculator can make that distinction, because it requires
knowing the ceiling.

A third observation from the same example: the occupancy finding was the *only*
thing that survived bad benchmarking. Profiling the block at a 512 KB working
set instead of 4 MB moved every roofline efficiency (12.4% → 62.6%, 55.4% →
97.8%, 7.7% → 43.8%, 15.7% → 100.8%) while the `tinyThreadgroup` limiter
appeared identically in both. Structural facts are robust to methodology in a
way that timing is not — which is an argument for reporting them at all.

### 5.5 Observer effect

Stage-boundary sampling is not free. Ten back-to-back dispatches of an identical
kernel, timed with per-encoder sampling and with none (best of 12 runs each):

```
  buffer   unsampled     sampled   overhead
  ------  ----------  ----------  ---------
    1 MB    221.1 us    333.4 us     +50.8%
    4 MB    601.7 us    793.0 us     +31.8%
    8 MB   1117.5 us   1358.6 us     +21.6%
   16 MB   2180.8 us   2441.4 us     +12.0%
   32 MB   3934.9 us   4407.5 us     +12.0%
   64 MB   8080.1 us   8158.8 us      +1.0%
```

Across six repetitions of that experiment the overhead ranged 49–129% at 1 MB,
32–49% at 4 MB, 12–18% at 16 MB, 7–12% at 32 MB and 1–6% at 64 MB. It behaves as
an approximately fixed per-encoder cost, so it shrinks as a fraction of the work
as the dispatch grows.

The overhead is real GPU work rather than a measurement artefact: counter timing
and command-buffer time for the *same sampled run* agree to within a
microsecond. Practically, short kernels look slower under `counters` than they
are in production, which argues for larger working sets, like-for-like trace
comparison, or deliberately letting a region fall to command-buffer timing when
an unperturbed number matters more than per-encoder detail.

This re-measurement also corrected the project's own documentation. The figure
previously recorded in [TRACE-FORMAT.md](TRACE-FORMAT.md) — ~35% at 4 MB and no
difference beyond noise at 32 MB — holds at 4 MB but is too optimistic at 32 MB,
where six runs show a consistent 7–12%. The overhead does become negligible,
later than previously claimed.

### 5.6 Two incidental findings

**Metal caps a counter sample buffer at 32 KB on this chip** — 4,096 timestamps,
so roughly 2,048 sampled encoders per region. metalscope treats an over-large
request as a reason to abandon sampling for that region and time the command
buffer instead, rather than failing the capture.

**Single-run diffs of short kernels are noise-dominated.** `bench` and `profile`
capture one run per kernel with no best-of. The same unchanged RMS-norm baseline
measured between 77 µs and 195 µs across five runs, making one fix look like
anything from 1.02× to 2.16×; in a `bench`-to-`bench` diff, `ffn.gemm` moved
from 66.1% to 94.3% of the compute ceiling with no code change at all, because
MPS took a different path for the same shape. The `occupancy changes:` section
is the part of a diff that can be trusted from a single pair of runs, because it
reports a change in shape rather than a stopwatch reading.

---

## 6. Limitations and future work

**Every non-timestamp counter set is hardware-blocked.** No Apple silicon part
metalscope has been run on — M1 Pro and M1 Max, both on macOS 26.5.1 — exposes
`stageutilization` or `statistic`. `kernels[].counters` is therefore absent from
every real trace produced so far. The resolvers exist, are unit-tested against
synthetic resolved data, and will populate the field with no code change on a
chip that offers those sets; testing them synthetically is the only way to keep
a path that nobody's hardware exercises from rotting.

**The counter matrix has one fully measured row.** One machine can contribute
one row, and the recipe (`metalscope info --json`) takes ten seconds. The M1 Max
row is partially filled from a second machine — same timestamp-only situation,
same OS — but without a full `info --json` payload its sampling boundaries and
threadgroup-memory limit are recorded as unknown rather than assumed from the
M1 Pro. Assuming them would defeat the purpose of a table whose entire premise
is *measured, not documented*.

**There is no measured occupancy.** The classic number — threadgroups actually
resident per core — requires a per-core thread and register budget Apple does
not publish and Metal does not expose. metalscope reports what fits inside the
*threadgroup memory* limit, labels it an upper bound, and stops. The static
analysis is a statement about a dispatch's shape, not about its residency, and
conflating the two would be the same class of error as the two bugs in §5.3.

**Instruction-level stalls and bank conflicts stay Instruments-only.** These are
not deferred features; Metal exposes no programmatic path to them. metalscope
says so rather than estimating them.

**No best-of in `bench` and `profile`.** §5.6 shows the cost. The obvious fix —
repeat each region and keep the fastest, as `calibrate` already does — would
make single-run diffs meaningful for microsecond-scale kernels and is the
highest-value next change.

**The roofline is single-level.** A cache-aware or hierarchical roofline would
explain the >100% efficiencies directly rather than by inference. That needs
per-level bandwidth calibration, which is a larger `calibrate` and a real
research question on a chip whose cache hierarchy is undocumented.

**Shape annotation is manual.** Heuristic shape inference was rejected on
honesty grounds (§3.2), but a middle path exists: a Metal-library-aware
front-end could propose annotations from function names and buffer sizes for a
human to confirm, keeping the guess out of the trace while removing most of the
typing.

---

## 7. Conclusion

The useful question about a GPU kernel is not "what did it do" but "how far is
it from the best its shape allows on this chip". Answering it requires two
things Apple's tooling does not provide: an arithmetic intensity derived from
knowing what the kernel computes, and a ceiling derived from measuring the
hardware rather than multiplying its specifications.

metalscope supplies both in 3,702 lines of dependency-free Swift, and the
recurring theme of its design is a refusal to fill gaps with plausible numbers.
When the timing path cannot cover a whole region, it degrades a tier and says
so. When metalscope never saw a pipeline, the occupancy column prints a dash.
When peaks have not been measured, every percentage derived from them is
labelled folklore. When a structural defect turns up in a kernel already at its
ceiling, the report says the fix will win lanes rather than time. Each of those
rules costs information; each exists because the alternative is a number that
reads as knowledge and is not.

The two bugs in §5.3 are the argument in miniature. Both produced output wrong
by roughly 3×, and only one was caught by inspection — because 541 GB/s on a
200 GB/s machine is impossible, while 0.88 TF on a 5.2 TF machine is merely
disappointing. A profiler is trusted by default; that trust is the whole of its
value, and it is spent every time it prints a number it cannot stand behind.

---

## References

- S. Williams, A. Waterman, D. Patterson. "Roofline: An Insightful Visual
  Performance Model for Multicore Architectures." *Communications of the ACM*
  52(4), 2009.
- A. Ilic, F. Pratas, L. Sousa. "Cache-aware Roofline Model: Upgrading the
  Loft." *IEEE Computer Architecture Letters*, 2014.
- C. Yang, T. Kurth, S. Williams. "Hierarchical Roofline analysis for GPUs."
  *Concurrency and Computation: Practice and Experience*, 2020.
- Empirical Roofline Toolkit, Lawrence Berkeley National Laboratory.
- NVIDIA. *Nsight Compute Documentation* — roofline and occupancy sections.
- Apple. *Metal* documentation: `MTLCounterSampleBuffer`,
  `MTLComputePipelineState`, `MTLCommonCounterSet`; Instruments Metal System
  Trace.

## Companion documents

- [USAGE.md](USAGE.md) — practical guide, with every example executed and its
  output transcribed verbatim
- [ARCHITECTURE.md](ARCHITECTURE.md) — design and milestone status
- [TRACE-FORMAT.md](TRACE-FORMAT.md) — the JSON schema, field by field
- [COUNTER-MATRIX.md](COUNTER-MATRIX.md) — what each chip exposes
