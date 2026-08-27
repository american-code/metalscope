# metalscope

An **ML-native kernel profiler for Metal** — what Nsight Compute is for CUDA.

A competitive hardware claim is only as checkable as the instrument behind it,
and Apple silicon has had no ML-native one. metalscope computes each kernel's
arithmetic intensity analytically from its ML shape and scores it against
ceilings **measured on your chip**, because published fp32 peaks overstate
reachable GEMM by 1.5–1.7× — by different amounts on different chips, so no
single correction factor rescues a spec-based roofline. It answers the question
Instruments cannot: how far is this kernel from the best its shape allows here.

Apple's Instruments GPU tooling exists but isn't ML-aware: it doesn't understand
attention kernels, doesn't give you roofline analysis against your M-series chip's
theoretical peaks, and can't diff two kernel versions side by side. metalscope does.

## What it does

- **Roofline analysis** per kernel dispatch: arithmetic intensity computed
  *analytically* from known ML kernel shapes (GEMM, attention, norms, elementwise) —
  placed against measured (not spec-sheet) peaks for the local chip.
- **Counter capture** via `MTLCounterSampleBuffer`: per-encoder GPU timestamps
  where the chip supports stage-boundary sampling, with honest fallbacks where it
  doesn't.
- **Static occupancy analysis** per dispatch: SIMD groups per threadgroup, lane
  waste from threadgroups that aren't a multiple of the SIMD width, threadgroup
  memory pressure — derived from `MTLComputePipelineState`, so it works on chips
  that expose no occupancy counters at all (which is all of them).
- **Kernel diffing**: profile two versions, get a side-by-side of where the time,
  the roofline placement, and the threadgroup shape moved.
- **`calibrate`**: measures real peak GEMM/streaming numbers on your chip so
  efficiency percentages are honest.

## Status

Milestones 1–5 of [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) are implemented and
exercised end-to-end on an M1 Pro: `info`, `calibrate`, `bench`, `profile`,
`report`, and `diff` all work, with a documented JSON trace schema
([docs/TRACE-FORMAT.md](docs/TRACE-FORMAT.md)) and **156 XCTest cases** (90.69%
region / 94.27% function / 96.46% line coverage of the two library targets).

New to metalscope? [**docs/USAGE.md**](docs/USAGE.md) is the practical guide —
the CLI end to end, a complete compiled example of instrumenting your own app,
and how to read every column. Every transcript in it was produced by running the
command above it on this machine. [**docs/WHITEPAPER.md**](docs/WHITEPAPER.md)
is the argument and the evaluation: the design rationale, measured-vs-spec-sheet
peaks on two chips, and the two measurement bugs the tool found in itself.

What is *not* here, and why: no Apple chip metalscope has run on exposes an
occupancy or utilization counter. On this machine (macOS 26.5.1, M1 Pro) the only
counter set Metal offers is `timestamp` and the only sampling point is
`.atStageBoundary`. So the occupancy analysis is **static** — read off the
pipeline, honest about being an upper bound — and the resolvers for Apple's other
two counter sets ship tested but dormant, ready for a chip that exposes them.
`metalscope info` prints what your chip actually gives you;
[docs/COUNTER-MATRIX.md](docs/COUNTER-MATRIX.md) collects those answers per chip
and explains how to contribute a row. Instruction-level stalls and bank conflicts
remain Instruments-only, and metalscope says so rather than estimating them.

## Install / build

Swift only, no external package dependencies.

```
swift build -c release
swift test
```

## Use

### 1. Calibrate once per machine

Spec-sheet TFLOPS for Apple GPUs are folklore. Measure instead (~10 s):

```
$ metalscope calibrate
metalscope calibrate — Apple M1 Pro
  sizes=1024,2048,3072 repeats=3 target=250ms/run (iterations auto-sized)

  workload      shape      iters  time/iter    achieved
  ------------  ---------  -----  ---------  ----------
  gemm fp32     2048^3        52   5.034 ms     3.41 TF
  gemm fp16     3072^3        14  17.477 ms     3.32 TF
  triad scalar  128 MB x3     92   2.395 ms  168.1 GB/s
  ...

  measured peaks for Apple M1 Pro
    fp32 compute               3.41 TF  MPS GEMM 2048^3
    fp16 compute               3.32 TF  MPS GEMM 3072^3
    memory bandwidth        168.1 GB/s  triad scalar @ 128 MB
    ridge point (fp32)  20.3 FLOP/byte

    vs spec-sheet folklore: fp32 66% of 5.2 TF, bandwidth 84% of 200 GB/s
```

Results are cached in `~/.metalscope/peaks.json`; `info` and `report` use them
automatically and label them `measured`. Without them, metalscope falls back to
`ChipPeaks.known` and labels it **spec-sheet folklore** so you know not to trust
the percentages.

### 2. Capture a trace

Either run the built-in self-test workload:

```
metalscope bench --variant baseline --output baseline.json
```

…or instrument your own code with the `MetalscopeCapture` library and run it
under `metalscope profile -- ./your-benchmark` (see
[docs/TRACE-FORMAT.md](docs/TRACE-FORMAT.md)). Dispatch through the capture
region to get occupancy recorded alongside the timing:

```swift
try session.capture(label: "qkv", shape: .gemm(m: 1024, n: 1024, k: 1024)) { region in
    let encoder = try region.makeComputeCommandEncoder()
    encoder.setBuffer(out, offset: 0, index: 0)
    region.dispatchThreads(encoder, pipeline: pipeline,
                           threads: grid, threadsPerThreadgroup: group)
    encoder.endEncoding()
}
```

### 3. Report

```
$ metalscope report baseline.json
roofline report — baseline.json
  device:  Apple M1 Pro   captured 2026-08-26T19:33:41Z
  peaks:   3.45 TF fp32 / 3.33 TF fp16 / 166.4 GB/s  [measured]
  ridge:   20.7 FLOP/byte (fp32), 20.0 FLOP/byte (fp16)

  kernel         shape                    prec  time/iter   GFLOP/s   GB/s    AI  bound       ceiling    eff  tgroup    occ  timing
  -------------  -----------------------  ----  ---------  --------  -----  ----  ---------  --------  -----  ------  -----  --------
  ffn.gemm       gemm 1024x1024x1024      fp32   1.076 ms   2.00 TF   11.7   171  compute     3.45 TF  57.9%       -      -  cmdbuf
  ffn.gemm.half  gemm 1024x1024x1024      fp16  689.62 us   3.11 TF    9.1   341  compute     3.33 TF  93.4%       -      -  cmdbuf
  attn.sdpa      attn b1 h8 s512 d64      fp32  734.98 us  730.5 GF    5.7   128  compute     3.45 TF  21.2%       -      -  cmdbuf
  block.rmsnorm  norm n=16777216          fp32  851.22 us   98.5 GF  157.7  0.62  bandwidth  104.0 GF  94.8%     256  25.0%  counters
  act.scale      elem n=16777216          fp32  842.34 us   19.9 GF  159.3  0.12  bandwidth   20.8 GF  95.8%    100*   9.8%  counters
  stream.triad   opaque 33.55MF/201.33MB  fp32   1.233 ms   27.2 GF  163.3  0.17  bandwidth   27.7 GF  98.1%     256  25.0%  counters

  headroom:
  - attn.sdpa is compute-bound at 21.2% of the compute ceiling (unfused: s x s scores round-trip through DRAM)
  - act.scale: threadgroup of 100 is not a multiple of the 32-wide SIMD group — 28 of 128 lanes
    idle in every threadgroup (78.1% lane use); round to 96 or 128; note it is already at 95.8%
    of its bandwidth ceiling, so this costs lanes, not time
```

`tgroup` is threads per threadgroup, starred when it isn't a multiple of the SIMD
width; `occ` is that against the pipeline's own ceiling. A dash means metalscope
never saw the pipeline — MPS encodes its own dispatches, and a guess would be
worse than a blank.

That last headroom line is the shape of the whole tool: the analysis found a real
structural defect (the `bench` baseline plants one on purpose), *and* checked it
against the roofline before telling you to go fix it. A ragged threadgroup in a
kernel already at 95.8% of measured bandwidth is wasted lanes, not wasted time.

### 4. Occupancy detail

```
$ metalscope report baseline.json --occupancy
  occupancy detail (static, from MTLComputePipelineState — no counters involved):

  act.scale
    threadgroup      100 threads of a 1024 max (9.8% occupancy)
    simd groups      4 x 32 lanes, 28 idle (78.1% lane use)
    threadgroup mem  0 B of 32.0 KB
    dispatches       167773 threadgroups/dispatch, 192 encoded
    limiter          threadgroup of 100 is not a multiple of the 32-wide SIMD group ...

  block.rmsnorm
    threadgroup      256 threads of a 1024 max (25.0% occupancy)
    simd groups      8 x 32 lanes, fully packed
    threadgroup mem  128 B of 32.0 KB (0.4%) — at most 256 fit that limit
    dispatches       16384 threadgroups/dispatch, 171 encoded
    limiter          nothing structural stands out
```

25% occupancy on a 256-thread threadgroup is *not* a finding, and metalscope
doesn't report it as one — 256 out of a 1024 ceiling is a perfectly good shape.
What gets flagged is a threadgroup that isn't a multiple of the SIMD width, one
that's a single SIMD group, or threadgroup memory over half the device limit.

### 5. Diff two versions

```
$ metalscope diff baseline.json tuned.json
  kernel         shape                     baseline  candidate   delta  speedup        eff a->b     d eff  tgroup a->b  bound
  -------------  -----------------------  ---------  ---------  ------  -------  --------------  --------  -----------  ---------
  ffn.gemm       gemm 1024x1024x1024       1.076 ms  648.47 us  -39.7%    1.66x  57.9% -> 96.1%  +38.2 pp            -  compute
  act.scale      elem n=16777216          842.34 us  861.01 us   +2.2%    0.98x  95.8% -> 93.7%   -2.1 pp  100* -> 256  bandwidth
  stream.triad   opaque 33.55MF/201.33MB   1.233 ms   1.247 ms   +1.1%    0.99x  98.1% -> 97.1%   -1.1 pp          256  bandwidth
  ...
  matched 6/6 kernels — total 5.427 ms -> 5.017 ms (-7.6%)
  occupancy changes:
  - act.scale: 100 -> 256 threads/threadgroup (+15.2 pp of pipeline max), limiter executionWidthAlignment -> none
```

Kernels align by label + shape + precision; a kernel whose shape changed shows up
as present in only one trace rather than being compared misleadingly. Occupancy
is only compared when both sides have it — a v1 trace against a v2 one reports no
delta rather than announcing a fix that never happened.

Run `metalscope help` for the full flag list.

## Package layout

| target | what it is |
| --- | --- |
| `MetalscopeCore` | Trace schema, shape registry, roofline math, occupancy math, diff. No Metal needed. |
| `MetalscopeCapture` | The opt-in capture API you link into your own app or benchmark, plus the per-counter-set resolvers. |
| `metalscope` | The CLI. |

## Docs

| doc | what's in it |
| --- | --- |
| [docs/USAGE.md](docs/USAGE.md) | **start here** — CLI walkthrough with verified output, instrumenting your own app, reading every column, interpreting results, and how to add a counter-set resolver |
| [docs/WHITEPAPER.md](docs/WHITEPAPER.md) | the argument: motivation, related work, design rationale, evaluation with real numbers, limitations |
| [docs/COMPARISON.md](docs/COMPARISON.md) | capability coverage vs. Nsight Compute, and the two-chip measured-vs-spec quantification |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | design, the timing ladder, why the peaks are measured, milestone status |
| [docs/TRACE-FORMAT.md](docs/TRACE-FORMAT.md) | the JSON schema (v2), field by field |
| [docs/COUNTER-MATRIX.md](docs/COUNTER-MATRIX.md) | what each chip actually exposes, and how to add a row |

## License

Apache-2.0 — see [LICENSE](LICENSE).
