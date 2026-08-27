# metalscope architecture

## Scope

Deliberately smaller than mccl/triton-metal: a solo-buildable CLI + report format.
Immediately useful for optimizing OIM / mlxMesh kernels, and later feeds occupancy
data into triton-metal's autotuner.

## Counter capture

`MTLCounterSampleBuffer` with `MTLCommandBuffer`/encoder sample points exposes,
per device counter set (varies by chip/OS):

- timestamp (stage boundaries → per-dispatch duration)
- stageutilization / statistic sets (ALU busy, memory unit busy, occupancy)

What Metal does **not** expose programmatically (only via Instruments/Xcode GPU
capture): instruction-level stalls, bank conflicts. Strategy: capture what the API
gives, and document per-chip which counters exist (`metalscope info` prints them).

**What the M1 Pro actually gives you (macOS 26.5, measured — not documentation):**

| capability | value |
| --- | --- |
| counter sets | `timestamp` only (single counter, `GPUTimestamp`) |
| `.atStageBoundary` sampling | yes |
| `.atDispatchBoundary` / `.atBlitBoundary` / `.atDrawBoundary` | **no** |
| GPU vs CPU timestamp domain | identical nanosecond timebase (1:1), though metalscope derives the ratio rather than assuming it |

Per-chip rows live in [COUNTER-MATRIX.md](COUNTER-MATRIX.md); `metalscope info
--json` emits one.

`CaptureSession` keeps a **resolver per counter set** rather than hardcoding
timestamps: each of Apple's three common sets resolves to a different C struct
(`MTLCounterResultTimestamp`, `...StageUtilization`, `...Statistic`), and each
gets a resolver that knows its layout and counter names. Every set a device
exposes and metalscope can decode is attached to the compute pass, resolved, and
(for the non-timing sets) summed into `kernels[].counters`. On this M1 Pro that
loop runs exactly once, over `timestamp`. The other two paths are unit-tested
against synthetic resolved data, because an untested path that only wakes up on
hardware nobody here owns is a path that will be broken when it does.

One trap worth writing down: the timing path decodes timestamps as `UInt64`, not
`Double`. GPU timestamps are nanoseconds since boot, so after ~104 days of uptime
they pass 2^53 and a `Double` starts rounding them — silently shortening every
kernel duration.

The absence of blit- and dispatch-boundary sampling is what shapes the capture
design: there is no way to bracket someone *else's* encoders with counter samples.
So `CaptureSession` uses a three-tier timing ladder, and every kernel record says
which tier it got (`timingSource`):

1. **stage-boundary counters** for encoders metalscope creates — per-encoder GPU
   timestamps, the good case;
2. **`MTLCommandBuffer.gpuStartTime/gpuEndTime`** when a framework (MPS, MLX)
   creates the encoders, which is why every captured region is its own command
   buffer, committed and waited on;
3. **host wall clock**, last resort, flagged as such.

A region only claims tier 1 if *every* encoder in it was sampled. Timing a
sampled prefix and dividing by the full iteration count over-reports throughput
several-fold — during development this produced "541 GB/s" on a 200 GB/s
machine, which is exactly the kind of confident wrong number a profiler must
never print.

Attach model: metalscope can't inject into arbitrary processes; instead ship a tiny
`MetalscopeCapture` library (swizzle-free: an explicit `capture { }` wrapper around
command buffer creation) that apps/benchmarks opt into, writing samples to a JSON
trace the CLI analyzes. `metalscope profile -- <cmd>` runs such a target with
`METALSCOPE_TRACE` set in its environment and reports on whatever it writes.

## ML-awareness

A kernel-shape registry: for known dispatch signatures (GEMM MxNxK, attention
B/H/S/D, norm/elementwise over N), compute FLOPs and bytes analytically. Matching
is by user annotation first (`capture(label: .gemm(m:n:k:))`), heuristic later.
This is the piece Instruments fundamentally doesn't have.

## Roofline honesty

Spec-sheet peaks are folklore for Apple GPUs. `calibrate` runs a large fp32/fp16
GEMM (MPS) and a streaming triad to measure the local chip's real ceilings, caches
them in `~/.metalscope/peaks.json`, and every efficiency % is against measured
numbers. The `ChipPeaks.known` table is just a fallback and says so.

Two things turned out to matter more than the choice of benchmark:

- **Run length.** Apple GPUs ramp their clocks over tens of milliseconds. The
  same 1024³ MPS GEMM measures 0.88 TF over 4 iterations and 3.3 TF over 120.
  `calibrate` therefore probes once, then auto-sizes the iteration count to
  ~250 ms of GPU work per run (`--target-ms`).
- **Best-of-repeats, not mean.** A ceiling is what the hardware *can* sustain;
  thermal and scheduling noise only ever moves a sample downward.

Measured on this M1 Pro: 3.41 TF fp32 (66% of the 5.2 TF folklore number),
3.32 TF fp16, 168 GB/s triad (84% of the 200 GB/s bus figure — a normal STREAM
result). Reporting against folklore instead would rate a 95%-of-achievable fp16
GEMM at 30%, which is how you end up optimizing the wrong kernel.

## Static occupancy

The highest-value part of milestone 5 turned out to need no counters at all.
`MTLComputePipelineState` gives up `maxTotalThreadsPerThreadgroup`,
`threadExecutionWidth` and `staticThreadgroupMemoryLength`; pair those with the
threadgroup size a dispatch actually used and you can say, statically, whether a
kernel's shape is leaving lanes on the floor. That works identically on every
chip in the matrix, including the ones that expose nothing.

metalscope stores only those raw inputs and derives every ratio on read, so a
trace can't contradict itself. What it will say:

- **SIMD groups per threadgroup**, rounded *up* — the whole point, since a
  100-thread threadgroup still costs four full 32-wide groups;
- **threadgroup occupancy**, dispatched size over the pipeline's ceiling;
- **execution-width alignment**, and how many lanes idle when it's wrong;
- **threadgroup-memory pressure** against `maxThreadgroupMemoryLength`.

The limiter classification is deliberately conservative. A low occupancy *ratio*
is not a finding: 256 threads out of a 1024 ceiling is a good shape, and flagging
it would teach people to skip the section. What gets flagged is a threadgroup
that isn't a multiple of the SIMD width, a threadgroup of a single SIMD group,
or threadgroup memory over half the limit.

Two honesty constraints shaped this more than the arithmetic did:

- **No residency estimate.** The classic occupancy number — threadgroups
  resident per core — needs a per-core thread and register budget Apple doesn't
  publish and Metal doesn't expose. metalscope reports what fits in the
  *threadgroup memory* limit, labels it an upper bound, and stops there.
- **The report cross-checks against the roofline.** `bench --variant baseline`
  dispatches its elementwise kernel at 100 threads/threadgroup on purpose, and
  the analysis duly flags it — but that kernel is also at 97% of the measured
  bandwidth ceiling, so fixing the shape wins back lanes, not time. The headroom
  hint says exactly that. A profiler that sends you off to fix a defect that
  isn't costing anything is a profiler you stop reading.

Getting the pipeline is a design constraint, not a detail: Metal cannot tell you
which pipeline an encoder is holding, and metalscope doesn't swizzle. So
`CaptureRegion` offers `dispatchThreads`/`dispatchThreadgroups` wrappers that set
the pipeline, encode the dispatch, and record it in one call — the recorded shape
is the shape that ran. Work encoded by MPS or MLX carries no occupancy block, and
the report prints a dash rather than a guess.

## Report format

One JSON schema (`traces/*.json`) shared by capture, `diff`, and future HTML
output — documented in [TRACE-FORMAT.md](TRACE-FORMAT.md). `diff` aligns kernels
by label + shape + precision, reports duration delta, efficiency delta, and which
resource bound (bandwidth vs. compute vs. ridge) each version sits on.

## Milestones

1. ✅ `info` + `calibrate` with measured GEMM/triad peaks.
2. ✅ `MetalscopeCapture` wrapper lib + JSON traces (`bench` self-test workload,
   `profile` for external targets).
3. ✅ Roofline report with analytic shapes for GEMM + attention (+ norm,
   elementwise, opaque).
4. ✅ `diff` command.
5. 🟡 Static occupancy analysis, per-set counter resolvers, per-chip counter
   matrix — done. Occupancy/stall *counters* remain blocked on hardware that
   exposes them.

### Milestone 5: what landed, and what is still blocked

Done, and unblocked on this machine:

- **Static occupancy analysis** — see [above](#static-occupancy). Recorded at
  capture time, carried in schema v2, surfaced as `report` columns, a
  `report --occupancy` detail block, a headroom hint, and a `diff` column.
- **Per-set counter resolvers** — `CaptureSession` no longer hardcodes
  `timestamp`. All three of Apple's common sets have resolvers; the ones a device
  exposes are sampled and resolved without a code change, and the ones it doesn't
  are tested synthetically.
- **[COUNTER-MATRIX.md](COUNTER-MATRIX.md)** — the M1 Pro row is filled from this
  machine, with a `metalscope info --json` recipe for other chips.

Genuinely blocked on hardware or on Apple:

- **Every non-timestamp counter set.** No Apple silicon part metalscope has been
  run on — M1 Pro and M1 Max, both on macOS 26.5.1 — exposes `stageutilization`
  or `statistic`. The code path is written and tested; it needs a chip that
  returns those sets from `MTLDevice.counterSets`. Until then
  `kernels[].counters` is absent from every real trace, and the matrix has one
  fully filled row.
- **Dispatch- and blit-boundary sampling.** Not supported here, which is why
  metalscope can't bracket someone else's encoders.
- **Measured occupancy** — threadgroups actually resident per core. No counter
  exposes it; the static analysis is an upper bound on the shape, not a
  measurement of residency, and it says so.
- **Stalls stay out of reach.** Instruction-level stalls and bank conflicts
  remain Instruments-only; metalscope keeps saying so rather than estimating
  them.
