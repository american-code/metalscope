# metalscope usage

A practical guide: the CLI end to end, how to instrument your own code with
`MetalscopeCapture`, how to read every column of every table, and how to extend
metalscope for hardware it hasn't met yet.

**Every transcript in this document was produced by running the command above
it.** The machine is an Apple M1 Pro (16 GB, macOS 26.5.1 build 25F80),
metalscope 0.1.0, built with `swift build -c release`. The only edit made to any
output is that the working directory has been shortened to `~/work` and the
example checkout to `~/src/block-bench`; everything else — every number, every
column, every warning — is verbatim. Your numbers will differ; the shapes of the
conclusions should not.

---

## Contents

1. [Build](#1-build)
2. [The walkthrough: info → calibrate → bench → report → diff](#2-the-walkthrough)
3. [Reading the report table, column by column](#3-reading-the-report-table)
4. [Occupancy detail](#4-occupancy-detail)
5. [Choosing which peaks to score against](#5-choosing-which-peaks-to-score-against)
6. [`diff`](#6-diff)
7. [Instrumenting your own app](#7-instrumenting-your-own-app)
8. [Interpreting results](#8-interpreting-results)
9. [Implementing what's missing](#9-implementing-whats-missing)
10. [Command and error reference](#10-command-and-error-reference)

---

## 1. Build

Swift only, no external package dependencies.

```
$ swift build -c release
$ swift test
```

The release binary lands at `.build/release/metalscope`. The examples below
assume it is on your `PATH` as `metalscope`.

---

## 2. The walkthrough

### `metalscope info` — what this chip actually gives you

Start here. `info` answers three questions: which counter sets Metal exposes on
this machine, which tier of the timing ladder that buys you, and whether the
roofline ceilings you'll be scored against are measured or folklore.

```
$ metalscope info
metalscope 0.1.0
  device:            Apple M1 Pro
  unified memory:    11 GB working set
  max threadgroup:   1024x1024x1024
  threadgroup mem:   32.0 KB max per threadgroup
  sampling points:   stage
  kernel timing:     per-encoder GPU timestamps [counters]

  counter sets exposed by this device
    set        counters      metalscope decodes
    ---------  ------------  ------------------
    timestamp  GPUTimestamp  yes

  not exposed here:  stageutilization, statistic
                     (resolvers are in place; a chip that offers them needs no code change)

  roofline peaks [measured] (2026-08-26T19:09:50Z)
                                 value
    ------------------  --------------
    fp32 compute               3.45 TF
    fp16 compute               3.33 TF
    memory bandwidth        166.4 GB/s
    ridge point (fp32)  20.7 FLOP/byte
    ridge point (fp16)  20.0 FLOP/byte
    source: /Users/melton/.metalscope/peaks.json
```

Line by line:

- **sampling points** — which of `.atStageBoundary`, `.atDispatchBoundary`,
  `.atBlitBoundary` this device supports. Only `stage` here, which is why
  metalscope cannot bracket encoders it did not create.
- **kernel timing** — the best tier reachable. `[counters]` means per-encoder GPU
  timestamps; `[cmdbuf]` would mean whole-command-buffer GPU time.
- **counter sets** — the honest list. `metalscope decodes` says whether a
  resolver exists for the set's struct layout. A set marked `no — resolver
  missing` is the interesting case; see [§9](#9-implementing-whats-missing).
- **not exposed here** — counter sets metalscope *can* decode but this chip does
  not offer. On every Apple part to date this is `stageutilization, statistic`.
- **roofline peaks** — `[measured]` after you've run `calibrate`, `[spec-sheet
  folklore]` before. The ridge point is where the bandwidth roof meets the
  compute roof: below it a kernel is bandwidth-bound, above it compute-bound.

`metalscope info --json` emits the same facts in machine form. That payload is
exactly one row of [COUNTER-MATRIX.md](COUNTER-MATRIX.md) — see
[§9](#9-implementing-whats-missing).

### `metalscope calibrate` — measure the ceilings

Spec-sheet TFLOPS for Apple GPUs are community folklore. Measure instead. This
takes about ten seconds.

```
$ metalscope calibrate
metalscope calibrate — Apple M1 Pro
  sizes=1024,2048,3072 repeats=3 target=250ms/run (iterations auto-sized)

  workload      shape      iters  time/iter    achieved
  ------------  ---------  -----  ---------  ----------
  gemm fp32     1024^3        82  701.78 us     3.06 TF
  gemm fp32     2048^3        51   5.431 ms     3.16 TF
  gemm fp32     3072^3        14  17.898 ms     3.24 TF
  gemm fp16     1024^3       357  696.43 us     3.08 TF
  gemm fp16     2048^3        50   5.536 ms     3.10 TF
  gemm fp16     3072^3        13  18.683 ms     3.10 TF
  triad scalar  64 MB x3     132   1.475 ms  136.5 GB/s
  triad vec4    64 MB x3     178   1.386 ms  145.2 GB/s
  triad scalar  128 MB x3    101   2.724 ms  147.8 GB/s
  triad vec4    128 MB x3    104   2.996 ms  134.4 GB/s
  triad scalar  256 MB x3     50   5.901 ms  136.5 GB/s
  triad vec4    256 MB x3     48   5.964 ms  135.0 GB/s

  measured peaks for Apple M1 Pro
                                 value  via
    ------------------  --------------  ---------------------
    fp32 compute               3.24 TF  MPS GEMM 3072^3
    fp16 compute               3.10 TF  MPS GEMM 2048^3
    memory bandwidth        147.8 GB/s  triad scalar @ 128 MB
    ridge point (fp32)  21.9 FLOP/byte
    ridge point (fp16)  21.0 FLOP/byte

    vs spec-sheet folklore: fp32 62% of 5.2 TF, bandwidth 74% of 200 GB/s
    (spec-sheet numbers assume FMA issue every cycle at boost clock; real GEMM never gets there)
```

*(That run used `--peaks-file` to write somewhere other than the cache, so the
machine's existing measurement — 3.45 TF / 3.33 TF / 166.4 GB/s, taken when it
was idle — is what the reports below score against. See
[run-to-run variance](#calibration-varies-run-to-run) for why the two differ and
which one to keep.)*

Reading the sweep:

- **iters** is auto-sized so each run is ~250 ms of GPU work (`--target-ms`).
  This matters more than the choice of benchmark: Apple GPUs ramp their clocks
  over tens of milliseconds, and a short run measures the ramp rather than the
  chip.
- Each row is the **best** of `--repeats` runs, not the mean. A ceiling is what
  the hardware *can* sustain; thermal and scheduling noise only ever moves a
  sample downward.
- The summary keeps the best size per workload, so a sweep that finds 3072³ is
  fastest for fp32 and 2048³ for fp16 records both.

Results are cached in `~/.metalscope/peaks.json`, keyed by device name, so one
machine can hold measurements for several GPUs. `info`, `report` and `diff`
pick them up automatically and label them `measured`.

Useful flags: `--quick` (small fast sweep), `--skip-fp16`, `--sizes`,
`--triad-mb`, `--repeats`, `--target-ms`, `--no-cache`, `--peaks-file PATH`,
`--json`.

### `metalscope bench` — capture a trace without writing any code

`bench` runs a small ML-shaped workload through `MetalscopeCapture` — the same
path your own instrumented app would take — and writes a trace.

```
$ metalscope bench --variant baseline --output baseline.json
metalscope bench [baseline] — Apple M1 Pro
  timing: encoder stage timestamps + command-buffer GPU time
  captured 6 kernels -> ~/work/baseline.json

$ metalscope bench --variant tuned --output tuned.json
metalscope bench [tuned] — Apple M1 Pro
  timing: encoder stage timestamps + command-buffer GPU time
  captured 6 kernels -> ~/work/tuned.json
```

The two variants exist to give `report` and `diff` something real to chew on.
`baseline` dispatches its elementwise kernel at **100 threads per threadgroup on
purpose** — not a multiple of the 32-wide SIMD group — and uses scalar loads;
`tuned` fixes both. Add `--report` to print the roofline table immediately after
capture.

### `metalscope report` — the roofline table

```
$ metalscope report baseline.json
roofline report — ~/work/baseline.json
  device:  Apple M1 Pro   captured 2026-08-26T22:37:04Z
  peaks:   3.45 TF fp32 / 3.33 TF fp16 / 166.4 GB/s  [measured]
  ridge:   20.7 FLOP/byte (fp32), 20.0 FLOP/byte (fp16)

  kernel         shape                    prec  time/iter   GFLOP/s   GB/s    AI  bound       ceiling    eff  tgroup    occ  timing
  -------------  -----------------------  ----  ---------  --------  -----  ----  ---------  --------  -----  ------  -----  --------
  ffn.gemm       gemm 1024x1024x1024      fp32  942.17 us   2.28 TF   13.4   171  compute     3.45 TF  66.1%       -      -  cmdbuf
  ffn.gemm.half  gemm 1024x1024x1024      fp16  701.79 us   3.06 TF    9.0   341  compute     3.33 TF  91.8%       -      -  cmdbuf
  attn.sdpa      attn b1 h8 s512 d64      fp32  764.29 us  702.4 GF    5.5   128  compute     3.45 TF  20.4%       -      -  cmdbuf
  block.rmsnorm  norm n=16777216          fp32  903.71 us   92.8 GF  148.5  0.62  bandwidth  104.0 GF  89.3%     256  25.0%  counters
  act.scale      elem n=16777216          fp32  900.11 us   18.6 GF  149.1  0.12  bandwidth   20.8 GF  89.6%    100*   9.8%  counters
  stream.triad   opaque 33.55MF/201.33MB  fp32   1.343 ms   25.0 GF  149.9  0.17  bandwidth   27.7 GF  90.1%     256  25.0%  counters

  AI = analytic FLOPs / compulsory bytes. eff = achieved / roofline ceiling at that AI.
  tgroup = threads/threadgroup (* = not a multiple of the SIMD width). occ = that vs the pipeline's max.
  a dash means metalscope never saw the pipeline (MPS and friends encode their own dispatches).

  headroom:
  - attn.sdpa is compute-bound at 20.4% of the compute ceiling (unfused: s x s scores round-trip through DRAM)
  - act.scale: threadgroup of 100 is not a multiple of the 32-wide SIMD group — 28 of 128 lanes idle in every threadgroup (78.1% lane use); round to 96 or 128
```

### `metalscope diff` — what moved

```
$ metalscope diff baseline.json tuned.json
diff — baseline baseline.json  ->  candidate tuned.json
  device:  Apple M1 Pro
  peaks:   3.45 TF fp32 / 166.4 GB/s  [measured]
  variant: baseline -> tuned

  kernel         shape                     baseline  candidate   delta  speedup        eff a->b     d eff  tgroup a->b  bound
  -------------  -----------------------  ---------  ---------  ------  -------  --------------  --------  -----------  ---------
  ffn.gemm       gemm 1024x1024x1024      942.17 us  660.85 us  -29.9%    1.43x  66.1% -> 94.3%  +28.2 pp            -  compute
  ffn.gemm.half  gemm 1024x1024x1024      701.79 us  697.11 us   -0.7%    1.01x  91.8% -> 92.4%   +0.6 pp            -  compute
  attn.sdpa      attn b1 h8 s512 d64      764.29 us  770.00 us   +0.7%    0.99x  20.4% -> 20.2%   -0.2 pp            -  compute
  block.rmsnorm  norm n=16777216          903.71 us  926.62 us   +2.5%    0.98x  89.3% -> 87.0%   -2.2 pp          256  bandwidth
  act.scale      elem n=16777216          900.11 us  886.79 us   -1.5%    1.02x  89.6% -> 91.0%   +1.3 pp  100* -> 256  bandwidth
  stream.triad   opaque 33.55MF/201.33MB   1.343 ms   1.305 ms   -2.8%    1.03x  90.1% -> 92.7%   +2.6 pp          256  bandwidth

  matched 6/6 kernels — total 5.555 ms -> 5.246 ms (-5.6%)
  occupancy changes:
  - act.scale: 100 -> 256 threads/threadgroup (+15.2 pp of pipeline max), limiter executionWidthAlignment -> none
```

Note what the fp32 GEMM did: it moved from 66.1% to 94.3% of the measured compute
ceiling without any change to the kernel. That is MPS taking a different path for
the same shape between runs, and it is the single best argument for reading the
occupancy column rather than trusting one timing delta. See
[run-to-run variance](#calibration-varies-run-to-run).

---

## 3. Reading the report table

### Header

```
  peaks:   3.45 TF fp32 / 3.33 TF fp16 / 166.4 GB/s  [measured]
  ridge:   20.7 FLOP/byte (fp32), 20.0 FLOP/byte (fp16)
```

- **peaks** — the ceilings every `eff` is a fraction of. The tag is either
  `[measured]` or `[spec-sheet folklore]`, and it is load-bearing: percentages
  against folklore are meaningless (see [§8](#measured-vs-spec-sheet-peaks)).
- **ridge** — peak compute ÷ peak bandwidth, the arithmetic intensity at which
  the two roofs cross. Kernels with lower AI can never be compute-bound on this
  chip no matter how good the code is.

### Columns

| column | what it is | how to read it |
| --- | --- | --- |
| `kernel` | the `label` you passed to `capture` | the identity used to align traces in `diff` |
| `shape` | the annotated kernel shape | `gemm MxNxK`, `attn bB hH sS dD`, `elem n=N`, `norm n=N`, `opaque <flops>F/<bytes>B` |
| `prec` | element precision | picks the byte size for the analytic model *and* which compute ceiling to score against (`fp16`/`bf16`/`int8` use the half peak) |
| `time/iter` | seconds for **one** invocation | the measured region span divided by `iterations` — not the whole region |
| `GFLOP/s` | analytic FLOPs ÷ `time/iter` | what the kernel achieved, given the shape's FLOP count |
| `GB/s` | analytic compulsory bytes ÷ `time/iter` | what the kernel achieved against the *compulsory* traffic model, not measured DRAM traffic |
| `AI` | arithmetic intensity, FLOPs per byte | the x-axis of the roofline. Computed from the shape, so it is exact for the model and independent of how fast the kernel ran |
| `bound` | `compute`, `bandwidth`, or `ridge` | whichever roof is lower at this AI. `ridge` means within 5% of the crossing point — genuinely balanced |
| `ceiling` | the lower of the two roofs at this AI | `min(peak compute, AI × peak bandwidth)`. This is the number `eff` divides by |
| `eff` | achieved ÷ ceiling | the headline. Over 100% is legal and informative — see [§8](#efficiency-over-100) |
| `tgroup` | threads per threadgroup as dispatched | starred when it is not a multiple of the SIMD width. A dash means metalscope never saw the pipeline |
| `occ` | `tgroup` ÷ the pipeline's `maxTotalThreadsPerThreadgroup` | **low is not automatically bad.** See [§8](#a-low-occupancy-ratio-is-not-a-finding) |
| `timing` | which tier of the timing ladder produced `time/iter` | `counters` > `cmdbuf` > `host`, best first |

### The `timing` column in detail

| value | meaning | when you get it |
| --- | --- | --- |
| `counters` | `MTLCounterSampleBuffer` timestamps at compute-encoder stage boundaries | every encoder in the region was created by metalscope *and* sampled |
| `cmdbuf` | `MTLCommandBuffer.gpuStartTime/gpuEndTime` | someone else created the encoders (MPS, MLX), or the region encoded more encoders than its sample buffer could hold |
| `host` | wall clock around commit + wait | last resort; includes scheduling latency |

metalscope never mixes tiers within a region. A region that sampled only some of
its encoders is demoted to `cmdbuf` rather than reporting the sampled prefix's
span divided by the full iteration count — that mistake reported "541 GB/s" on a
200 GB/s machine during development, and it is exactly the kind of confident
wrong number a profiler must never print.

In the table above, the three MPS-backed kernels are `cmdbuf` and carry no
occupancy; the three hand-written kernels are `counters` and do. That split is
not a limitation to work around, it is the honest shape of what Metal exposes.

### The `headroom` section

Two kinds of line appear here, and they answer the same question — *what should I
look at next?*

1. **Roofline lines**, for any kernel under 40% of its ceiling. The `bound` tells
   you which wall it is under, and a `fusion` note (if the capture supplied one)
   explains a known structural reason.
2. **Occupancy lines**, for any dispatch with a structural limiter. These are
   cross-checked against the roofline first; see
   [§8](#the-occupancy-cross-check-lanes-versus-time).

If a kernel is fine on both counts, nothing is printed about it. Silence is a
result.

### Sorting

```
$ metalscope report baseline.json --sort efficiency
...
  kernel         shape                    prec  time/iter   GFLOP/s   GB/s    AI  bound       ceiling    eff  tgroup    occ  timing
  -------------  -----------------------  ----  ---------  --------  -----  ----  ---------  --------  -----  ------  -----  --------
  attn.sdpa      attn b1 h8 s512 d64      fp32  764.29 us  702.4 GF    5.5   128  compute     3.45 TF  20.4%       -      -  cmdbuf
  ffn.gemm       gemm 1024x1024x1024      fp32  942.17 us   2.28 TF   13.4   171  compute     3.45 TF  66.1%       -      -  cmdbuf
  block.rmsnorm  norm n=16777216          fp32  903.71 us   92.8 GF  148.5  0.62  bandwidth  104.0 GF  89.3%     256  25.0%  counters
  act.scale      elem n=16777216          fp32  900.11 us   18.6 GF  149.1  0.12  bandwidth   20.8 GF  89.6%    100*   9.8%  counters
  stream.triad   opaque 33.55MF/201.33MB  fp32   1.343 ms   25.0 GF  149.9  0.17  bandwidth   27.7 GF  90.1%     256  25.0%  counters
  ffn.gemm.half  gemm 1024x1024x1024      fp16  701.79 us   3.06 TF    9.0   341  compute     3.33 TF  91.8%       -      -  cmdbuf
```

`--sort efficiency` is worst-first, because that is the list you actually work
through. `--sort duration` is longest-first, `--sort intensity` is
highest-AI-first. Without `--sort`, kernels appear in capture order.

### Machine-readable output

`--json` emits the derived values too, so a consumer (an autotuner, a CI check)
never re-implements the arithmetic:

```
$ metalscope report baseline.json --json
...
    {
      "achievedBandwidthGBs" : 149.11206584898392,
      "achievedGFLOPS" : 18.63900823112299,
      "arithmeticIntensity" : 0.125,
      "bound" : "bandwidth",
      "ceilingGFLOPS" : 20.800587006346348,
      "durationSeconds" : 0.0009001131279069767,
      "efficiency" : 0.896080876248163,
      "label" : "act.scale",
      "occupancy" : {
        "dispatchCount" : 172,
        "executionWidthAligned" : false,
        "hint" : "threadgroup of 100 is not a multiple of the 32-wide SIMD group — 28 of 128 lanes idle in every threadgroup (78.1% lane use); round to 96 or 128",
        "idleLanesPerThreadgroup" : 28,
        "laneUtilization" : 0.78125,
        "limiter" : "executionWidthAlignment",
        "maxTotalThreadsPerThreadgroup" : 1024,
        "simdGroupsPerThreadgroup" : 4,
        "threadExecutionWidth" : 32,
        "threadgroupMemoryBytes" : 0,
        "threadgroupMemoryPressure" : 0,
        "threadgroupOccupancy" : 0.09765625,
        "threadgroupsPerGrid" : 167773,
        "threadsPerThreadgroup" : 100,
        "variantCount" : 1
      },
      "precision" : "fp32",
      "shape" : "elem n=16777216",
      "timingSource" : "counter-sample-buffer"
    },
```

The trace on disk stores only the raw inputs (`threadsPerThreadgroup`,
`maxTotalThreadsPerThreadgroup`, `threadExecutionWidth`,
`threadgroupMemoryBytes`, …). Every ratio above is derived when the trace is
read, so a trace can never disagree with itself. See
[TRACE-FORMAT.md](TRACE-FORMAT.md).

---

## 4. Occupancy detail

```
$ metalscope report baseline.json --occupancy
...
  occupancy detail (static, from MTLComputePipelineState — no counters involved):

  block.rmsnorm
    threadgroup      256 threads of a 1024 max (25.0% occupancy)
    simd groups      8 x 32 lanes, fully packed
    threadgroup mem  128 B of 32.0 KB (0.4%) — at most 256 fit that limit
    dispatches       16384 threadgroups/dispatch, 177 encoded
    limiter          nothing structural stands out

  act.scale
    threadgroup      100 threads of a 1024 max (9.8% occupancy)
    simd groups      4 x 32 lanes, 28 idle (78.1% lane use)
    threadgroup mem  0 B of 32.0 KB
    dispatches       167773 threadgroups/dispatch, 172 encoded
    limiter          threadgroup of 100 is not a multiple of the 32-wide SIMD group — 28 of 128 lanes idle in every threadgroup (78.1% lane use); round to 96 or 128

  stream.triad
    threadgroup      256 threads of a 1024 max (25.0% occupancy)
    simd groups      8 x 32 lanes, fully packed
    threadgroup mem  0 B of 32.0 KB
    dispatches       65536 threadgroups/dispatch, 118 encoded
    limiter          nothing structural stands out
```

Reading the block:

- **threadgroup** — the size you dispatched, against this *pipeline's*
  `maxTotalThreadsPerThreadgroup`. That ceiling is set by the kernel's register
  pressure, not by the device, so two kernels on the same chip can have
  different ones.
- **simd groups** — rounded **up**. A 100-thread threadgroup still costs four
  full 32-wide SIMD groups; the 28 lanes in the fourth group are launched and
  idle in *every* threadgroup, forever.
- **threadgroup mem** — `staticThreadgroupMemoryLength` against the device
  limit. "at most N fit that limit" is an **upper bound** on co-residency, not a
  measurement: Metal exposes the per-threadgroup limit, and the scheduler has
  other reasons to keep fewer threadgroups resident.
- **dispatches** — threadgroups per dispatch, and how many dispatches folded into
  this record. When a region dispatched several *different* shapes it says
  `across N shapes (worst shown)`; averaging threadgroup sizes would describe a
  dispatch that never ran.
- **limiter** — one of four classifications, or "nothing structural stands out":

| limiter | meaning |
| --- | --- |
| `executionWidthAlignment` | threadgroup is not a multiple of `threadExecutionWidth` — lanes idle in every threadgroup |
| `tinyThreadgroup` | a single SIMD group, so there is nothing for the core to interleave while it waits on memory |
| `threadgroupMemory` | static threadgroup memory over half the device limit |
| `none` | nothing structural stands out |

This analysis needs **no counters at all** — it comes from
`MTLComputePipelineState` plus the threadgroup size the dispatch used — so it
works identically on every chip, including the ones that expose nothing but
`timestamp`.

What it is **not**: a measurement of how many threadgroups were actually
resident per core. No Apple GPU exposes that counter, and metalscope does not
estimate it.

---

## 5. Choosing which peaks to score against

`report` and `diff` resolve peaks in this order:

1. `--peaks-file PATH`
2. the trace's own `peaks`, if they are `measured`
3. `~/.metalscope/peaks.json`
4. whatever the trace carried (possibly folklore)
5. otherwise: an error telling you to run `calibrate`

`--spec-peaks` forces the community numbers, which is worth doing exactly once,
to see what you would have been told:

```
$ metalscope report baseline.json --spec-peaks
roofline report — ~/work/baseline.json
  device:  Apple M1 Pro   captured 2026-08-26T22:37:04Z
  peaks:   5.20 TF fp32 / 10.40 TF fp16 / 200.0 GB/s  [spec-sheet folklore]
  ridge:   26.0 FLOP/byte (fp32), 52.0 FLOP/byte (fp16)

  kernel         shape                    prec  time/iter   GFLOP/s   GB/s    AI  bound       ceiling    eff  tgroup    occ  timing
  -------------  -----------------------  ----  ---------  --------  -----  ----  ---------  --------  -----  ------  -----  --------
  ffn.gemm       gemm 1024x1024x1024      fp32  942.17 us   2.28 TF   13.4   171  compute     5.20 TF  43.8%       -      -  cmdbuf
  ffn.gemm.half  gemm 1024x1024x1024      fp16  701.79 us   3.06 TF    9.0   341  compute    10.40 TF  29.4%       -      -  cmdbuf
  attn.sdpa      attn b1 h8 s512 d64      fp32  764.29 us  702.4 GF    5.5   128  compute     5.20 TF  13.5%       -      -  cmdbuf
  block.rmsnorm  norm n=16777216          fp32  903.71 us   92.8 GF  148.5  0.62  bandwidth  125.0 GF  74.3%     256  25.0%  counters
  act.scale      elem n=16777216          fp32  900.11 us   18.6 GF  149.1  0.12  bandwidth   25.0 GF  74.6%    100*   9.8%  counters
  stream.triad   opaque 33.55MF/201.33MB  fp32   1.343 ms   25.0 GF  149.9  0.17  bandwidth   33.3 GF  75.0%     256  25.0%  counters

  AI = analytic FLOPs / compulsory bytes. eff = achieved / roofline ceiling at that AI.
  tgroup = threads/threadgroup (* = not a multiple of the SIMD width). occ = that vs the pipeline's max.
  a dash means metalscope never saw the pipeline (MPS and friends encode their own dispatches).
  peaks are spec-sheet folklore — run `metalscope calibrate` before trusting eff%.

  headroom:
  - ffn.gemm.half is compute-bound at 29.4% of the compute ceiling
  - attn.sdpa is compute-bound at 13.5% of the compute ceiling (unfused: s x s scores round-trip through DRAM)
  - act.scale: threadgroup of 100 is not a multiple of the 32-wide SIMD group — 28 of 128 lanes idle in every threadgroup (78.1% lane use); round to 96 or 128
```

Compare the two `ffn.gemm.half` rows. Against measured peaks it is at **91.8%**
and there is nothing to win. Against folklore it is at **29.4%** and appears in
the headroom list as the second-worst kernel in the trace. Same trace, same
kernel, same microseconds — one of those two readings sends you off to spend a
week rewriting a GEMM that is already essentially optimal.

> **Caveat.** `--peaks-file` pointing at a file that has no measured entry for
> this device silently falls back to spec-sheet folklore (labelled as such)
> rather than erroring. The label is honest, but the flag is not as strict as it
> looks; check for `[measured]` in the header.

---

## 6. `diff`

Kernels align by **label + shape description + precision**. Change a kernel's
shape and it will not align — deliberately, since the comparison would be
meaningless; it shows up as present in only one trace instead. Repeated
identical keys pair up positionally (the Nth occurrence in the baseline matches
the Nth in the candidate).

| column | how to read it |
| --- | --- |
| `baseline` / `candidate` | per-iteration duration on each side, or `absent` |
| `delta` | signed relative change. Negative is faster |
| `speedup` | baseline ÷ candidate. Above 1.00x is faster |
| `eff a->b` | roofline efficiency on each side |
| `d eff` | efficiency change in percentage **points**, not percent |
| `tgroup a->b` | threads per threadgroup, or a single value when unchanged. `-` on a side means no occupancy data there |
| `bound` | the candidate's bound, or `from -> to` when it moved |

Occupancy is compared **only when both sides have it**. A v1 trace diffed
against a v2 one reports no occupancy delta rather than treating the missing
side as unchanged, which would announce a fix that never happened:

```
$ metalscope diff legacy-v1.json tuned.json
...
  block.rmsnorm  norm n=16777216          903.71 us  926.62 us   +2.5%    0.98x  89.3% -> 87.0%   -2.2 pp     - -> 256  bandwidth
  act.scale      elem n=16777216          900.11 us  886.79 us   -1.5%    1.02x  89.6% -> 91.0%   +1.3 pp     - -> 256  bandwidth
  stream.triad   opaque 33.55MF/201.33MB   1.343 ms   1.305 ms   -2.8%    1.03x  90.1% -> 92.7%   +2.6 pp     - -> 256  bandwidth

  matched 6/6 kernels — total 5.555 ms -> 5.246 ms (-5.6%)
```

There is no `occupancy changes:` section, because nothing can be known to have
changed. Reporting a v1 trace at all drops the occupancy columns entirely rather
than filling them with dashes:

```
$ metalscope report legacy-v1.json
...
  kernel         shape                    prec  time/iter   GFLOP/s   GB/s    AI  bound       ceiling    eff  timing
  -------------  -----------------------  ----  ---------  --------  -----  ----  ---------  --------  -----  --------
  ffn.gemm       gemm 1024x1024x1024      fp32  942.17 us   2.28 TF   13.4   171  compute     3.45 TF  66.1%  cmdbuf
  ffn.gemm.half  gemm 1024x1024x1024      fp16  701.79 us   3.06 TF    9.0   341  compute     3.33 TF  91.8%  cmdbuf
  ...
```

`diff --json` emits every field, including both sides' occupancy limiters:

```
$ metalscope diff baseline.json tuned.json --json
...
{
  "baselineBound": "bandwidth",
  "baselineEfficiency": 0.896080876248163,
  "baselineOccupancyLimiter": "executionWidthAlignment",
  "baselineSeconds": 0.0009001131279069767,
  "baselineThreadgroupOccupancy": 0.09765625,
  "baselineThreadsPerThreadgroup": 100,
  "candidateBound": "bandwidth",
  "candidateEfficiency": 0.9095411821658385,
  "candidateOccupancyLimiter": "none",
  "candidateSeconds": 0.000886792347826087,
  "candidateThreadgroupOccupancy": 0.25,
  "candidateThreadsPerThreadgroup": 256,
  "durationDeltaFraction": -0.014799006555836318,
  "efficiencyDeltaPoints": 1.3460305917675441,
  "label": "act.scale",
  "occupancyDeltaPoints": 15.234375,
  "precision": "fp32",
  "shape": "elem n=16777216",
  "speedup": 1.0150213069762553,
  "status": "matched"
}
```

Diffing traces from two different devices prints a warning to stderr and
continues; the efficiency comparison is meaningless, and it says so.

---

## 7. Instrumenting your own app

metalscope does not inject into arbitrary processes. Metal offers no supported
way to do it, and the alternatives — swizzling, `DYLD_INSERT_LIBRARIES` — are
exactly what makes a profiler untrustworthy. The contract instead is: your
target links `MetalscopeCapture`, wraps annotated work in `capture { }`, and
writes a trace.

### 7.1 A complete example

This is a single transformer block — rmsnorm, QKV projection, attention, GELU —
with two hand-written kernels and two MPS ones. It compiles and runs as shown.

**`Package.swift`**

```swift
// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "block-bench",
    platforms: [.macOS(.v14)],
    dependencies: [
        .package(path: "../metalscope"),        // point this at your checkout
    ],
    targets: [
        .executableTarget(
            name: "block-bench",
            dependencies: [
                .product(name: "MetalscopeCapture", package: "metalscope"),
                .product(name: "MetalscopeCore", package: "metalscope"),
            ]),
    ]
)
```

**`Sources/block-bench/main.swift`**

```swift
import Foundation
import Metal
import MetalPerformanceShaders
import MetalscopeCapture
import MetalscopeCore

// A single transformer block, instrumented with MetalscopeCapture:
//   rmsnorm -> QKV projection -> attention -> GELU
// Two of those are hand-written kernels, two are MPS. The trace shows both the
// analytic roofline placement and which tier of the timing ladder each region
// reached.

let source = """
#include <metal_stdlib>
using namespace metal;

// One threadgroup per row. Reduces within each SIMD group, then across them,
// so any threadgroup size that is a multiple of 32 is correct.
kernel void rmsnorm(device float *y [[buffer(0)]],
                    device const float *x [[buffer(1)]],
                    constant uint &width [[buffer(2)]],
                    uint row [[threadgroup_position_in_grid]],
                    uint lane [[thread_position_in_threadgroup]],
                    uint tgSize [[threads_per_threadgroup]],
                    uint simdLane [[thread_index_in_simdgroup]],
                    uint simdGroup [[simdgroup_index_in_threadgroup]]) {
    threadgroup float partials[32];
    device const float *xr = x + (size_t)row * width;
    device float *yr = y + (size_t)row * width;

    float acc = 0.0f;
    for (uint i = lane; i < width; i += tgSize) { float v = xr[i]; acc += v * v; }
    acc = simd_sum(acc);
    if (simdLane == 0) { partials[simdGroup] = acc; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simdGroup == 0) {
        uint groups = max(1u, tgSize / 32u);
        float v = (simdLane < groups) ? partials[simdLane] : 0.0f;
        v = simd_sum(v);
        if (simdLane == 0) { partials[0] = v; }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float scale = rsqrt(partials[0] / (float)width + 1e-6f);
    for (uint i = lane; i < width; i += tgSize) { yr[i] = xr[i] * scale; }
}

kernel void gelu(device float *y [[buffer(0)]],
                 device const float *x [[buffer(1)]],
                 uint i [[thread_position_in_grid]]) {
    float v = x[i];
    y[i] = 0.5f * v * (1.0f + precise::tanh(0.7978845608f * (v + 0.044715f * v * v * v)));
}
"""

// MARK: - Setup

guard let device = MTLCreateSystemDefaultDevice() else {
    fatalError("no Metal device")
}
let session = try CaptureSession(device: device)
let library = try device.makeLibrary(source: source, options: nil)

func pipeline(_ name: String) throws -> MTLComputePipelineState {
    guard let function = library.makeFunction(name: name) else { fatalError("no \(name)") }
    return try device.makeComputePipelineState(function: function)
}

let rmsnorm = try pipeline("rmsnorm")
let gelu = try pipeline("gelu")

// Two knobs: the sequence length (so the same binary can be profiled at a
// working set that fits in cache and at one that does not) and the norm
// kernel's threadgroup width (so a shape fix can be diffed against its
// baseline). Usage: block-bench [seq] [normThreadgroupWidth]
let arguments = Array(CommandLine.arguments.dropFirst())
let seq = arguments.count > 0 ? (Int(arguments[0]) ?? 256) : 256
let normWidth = arguments.count > 1 ? (Int(arguments[1]) ?? 32) : 32
let heads = 8, headDim = 64
let model = heads * headDim          // 512
let rows = seq                       // one sequence, unbatched
let elements = rows * model

func buffer(_ count: Int) -> MTLBuffer {
    guard let b = device.makeBuffer(length: count * MemoryLayout<Float>.size,
                                    options: .storageModeShared) else { fatalError("alloc") }
    let p = b.contents().bindMemory(to: Float.self, capacity: count)
    for i in 0..<count { p[i] = Float((i % 17)) * 0.05 - 0.4 }
    return b
}

let hidden = buffer(elements)
let normed = buffer(elements)
let qkv = buffer(elements * 3)
let scores = buffer(heads * seq * seq)
let context = buffer(elements)
let activated = buffer(elements)
let weights = buffer(model * model * 3)

// MARK: - Captured regions

// Apple GPUs ramp their clocks over tens of milliseconds, so a region has to
// contain enough work to reach a sustained clock or it measures the ramp. The
// two MPS regions are milliseconds each; the two elementwise-ish kernels are
// tens of microseconds and need far more repetitions to add up to the same.
let blockIterations = 32
let elementIterations = 2048

// 2 timestamps per encoder: 2048 sampled encoders needs a 4096-sample buffer,
// which is exactly the 32 KB Metal allows on this chip. Leave this at its
// default and the region silently (and correctly) falls back to command-buffer
// timing instead.
session.maxSamplesPerRegion = 4096

// 1. norm: a hand-written kernel, dispatched through the region so its static
//    occupancy is recorded alongside the timing.
try session.capture(label: "block.rmsnorm",
                    shape: .norm(n: elements),
                    precision: .fp32,
                    iterations: elementIterations) { region in
    var width = UInt32(model)
    for _ in 0..<elementIterations {
        let encoder = try region.makeComputeCommandEncoder(label: "rmsnorm")
        encoder.setBuffer(normed, offset: 0, index: 0)
        encoder.setBuffer(hidden, offset: 0, index: 1)
        encoder.setBytes(&width, length: MemoryLayout<UInt32>.size, index: 2)
        region.dispatchThreadgroups(encoder, pipeline: rmsnorm,
                                    threadgroupsPerGrid: MTLSize(width: rows, height: 1, depth: 1),
                                    threadsPerThreadgroup: MTLSize(width: normWidth, height: 1, depth: 1))
        encoder.endEncoding()
    }
}

// 2. QKV projection: MPS encodes its own dispatches into the region's command
//    buffer, so this region is timed at command-buffer granularity and carries
//    no occupancy block. metalscope prints a dash rather than a guess.
let rowBytes = MPSMatrixDescriptor.rowBytes(forColumns: model, dataType: .float32)
let outBytes = MPSMatrixDescriptor.rowBytes(forColumns: model * 3, dataType: .float32)
let xDesc = MPSMatrixDescriptor(rows: rows, columns: model, rowBytes: rowBytes, dataType: .float32)
let wDesc = MPSMatrixDescriptor(rows: model, columns: model * 3, rowBytes: outBytes, dataType: .float32)
let yDesc = MPSMatrixDescriptor(rows: rows, columns: model * 3, rowBytes: outBytes, dataType: .float32)
let project = MPSMatrixMultiplication(device: device,
                                      transposeLeft: false, transposeRight: false,
                                      resultRows: rows, resultColumns: model * 3,
                                      interiorColumns: model, alpha: 1, beta: 0)

try session.capture(label: "block.qkv",
                    shape: .gemm(m: rows, n: model * 3, k: model),
                    precision: .fp32,
                    iterations: blockIterations,
                    notes: ["backend": "MPSMatrixMultiplication"]) { region in
    let x = MPSMatrix(buffer: normed, descriptor: xDesc)
    let w = MPSMatrix(buffer: weights, descriptor: wDesc)
    let y = MPSMatrix(buffer: qkv, descriptor: yDesc)
    for _ in 0..<blockIterations {
        project.encode(commandBuffer: region.commandBuffer, leftMatrix: x, rightMatrix: w, resultMatrix: y)
    }
}

// 3. Attention, unfused: QK^T -> softmax -> PV, per head. The `.attention` shape
//    models a *fused* kernel, so the gap to 100% is largely the cost of spilling
//    the s x s scores to DRAM — which is the thing worth seeing.
let qkvDesc = MPSMatrixDescriptor(rows: seq, columns: headDim,
                                  rowBytes: MPSMatrixDescriptor.rowBytes(forColumns: headDim,
                                                                         dataType: .float32),
                                  dataType: .float32)
let scoreDesc = MPSMatrixDescriptor(rows: seq, columns: seq,
                                    rowBytes: MPSMatrixDescriptor.rowBytes(forColumns: seq,
                                                                           dataType: .float32),
                                    dataType: .float32)
let qk = MPSMatrixMultiplication(device: device,
                                 transposeLeft: false, transposeRight: true,
                                 resultRows: seq, resultColumns: seq, interiorColumns: headDim,
                                 alpha: 1 / Double(headDim).squareRoot(), beta: 0)
let pv = MPSMatrixMultiplication(device: device,
                                 transposeLeft: false, transposeRight: false,
                                 resultRows: seq, resultColumns: headDim, interiorColumns: seq,
                                 alpha: 1, beta: 0)
let softmax = MPSMatrixSoftMax(device: device)
let headBytes = qkvDesc.rowBytes * seq
let scoreBytes = scoreDesc.rowBytes * seq

try session.capture(label: "block.attention",
                    shape: .attention(b: 1, h: heads, s: seq, d: headDim),
                    precision: .fp32,
                    iterations: blockIterations,
                    notes: ["fusion": "unfused: s x s scores round-trip through DRAM"]) { region in
    for _ in 0..<blockIterations {
        for head in 0..<heads {
            let q = MPSMatrix(buffer: qkv, offset: head * headBytes, descriptor: qkvDesc)
            let k = MPSMatrix(buffer: qkv, offset: (heads + head) * headBytes, descriptor: qkvDesc)
            let v = MPSMatrix(buffer: qkv, offset: (2 * heads + head) * headBytes, descriptor: qkvDesc)
            let p = MPSMatrix(buffer: scores, offset: head * scoreBytes, descriptor: scoreDesc)
            let o = MPSMatrix(buffer: context, offset: head * headBytes, descriptor: qkvDesc)
            qk.encode(commandBuffer: region.commandBuffer, leftMatrix: q, rightMatrix: k, resultMatrix: p)
            softmax.encode(commandBuffer: region.commandBuffer, inputMatrix: p, resultMatrix: p)
            pv.encode(commandBuffer: region.commandBuffer, leftMatrix: p, rightMatrix: v, resultMatrix: o)
        }
    }
}

// 4. GELU: one read, one write per element — exactly what `.elementwise(n:)`
//    models.
try session.capture(label: "block.gelu",
                    shape: .elementwise(n: elements),
                    precision: .fp32,
                    iterations: elementIterations) { region in
    for _ in 0..<elementIterations {
        let encoder = try region.makeComputeCommandEncoder(label: "gelu")
        encoder.setBuffer(activated, offset: 0, index: 0)
        encoder.setBuffer(context, offset: 0, index: 1)
        region.dispatchThreads(encoder, pipeline: gelu,
                               threads: MTLSize(width: elements, height: 1, depth: 1),
                               threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        encoder.endEncoding()
    }
}

// MARK: - Output

// Honours $METALSCOPE_TRACE when run under `metalscope profile`, and falls back
// to ./metalscope-trace.json when run directly.
let url = try session.writeTrace(notes: ["app": "block-bench"])
FileHandle.standardError.write(Data("block-bench: wrote \(session.records.count) kernels to \(url.path)\n".utf8))
```

### 7.2 Running it

```
$ swift build -c release
$ metalscope profile --output block-32.json -- ./.build/release/block-bench 2048 32
block-bench: wrote 4 kernels to ~/work/block-32.json
metalscope profile — running ~/src/block-bench/.build/release/block-bench 2048 32
  METALSCOPE_TRACE=~/work/block-32.json

roofline report — ~/work/block-32.json
  device:  Apple M1 Pro   captured 2026-08-26T22:42:34Z
  peaks:   3.45 TF fp32 / 3.33 TF fp16 / 166.4 GB/s  [measured]
  ridge:   20.7 FLOP/byte (fp32), 20.0 FLOP/byte (fp16)

  kernel           shape                 prec  time/iter  GFLOP/s   GB/s    AI  bound       ceiling     eff  tgroup    occ  timing
  ---------------  --------------------  ----  ---------  -------  -----  ----  ---------  --------  ------  ------  -----  --------
  block.rmsnorm    norm n=1048576        fp32   80.50 us  65.1 GF  104.2  0.62  bandwidth  104.0 GF   62.6%      32   3.1%  counters
  block.qkv        gemm 2048x1536x512    fp32  956.18 us  3.37 TF   20.8   162  compute     3.45 TF   97.8%       -      -  cmdbuf
  block.attention  attn b1 h8 s2048 d64  fp32   5.690 ms  1.51 TF    2.9   512  compute     3.45 TF   43.8%       -      -  cmdbuf
  block.gelu       elem n=1048576        fp32   49.99 us  21.0 GF  167.8  0.12  bandwidth   20.8 GF  100.8%     256  25.0%  counters

  AI = analytic FLOPs / compulsory bytes. eff = achieved / roofline ceiling at that AI.
  tgroup = threads/threadgroup (* = not a multiple of the SIMD width). occ = that vs the pipeline's max.
  a dash means metalscope never saw the pipeline (MPS and friends encode their own dispatches).

  headroom:
  - block.rmsnorm: threadgroup of 32 is a single SIMD group — nothing for the core to interleave while it waits on memory (pipeline allows 1024)
```

`profile` sets `METALSCOPE_TRACE` in the child's environment, runs it, and
reports on whatever it wrote there. Arguments after `--` are passed through
verbatim, which is how `2048 32` reaches `block-bench`. Add `--no-report` to
collect the trace without printing.

Four kernels, four different stories, and every one of them is a thing
Instruments cannot tell you:

- **`block.qkv` at 97.8%** — MPS's GEMM is essentially at the measured compute
  ceiling for this shape. There is nothing here. Move on.
- **`block.gelu` at 100.8%** — above the ceiling, which is a *result*, not a bug;
  see [§8](#efficiency-over-100).
- **`block.attention` at 43.8%** — compute-bound, but less than half the ceiling,
  and the `fusion` note says why: the `.attention` shape models a fused kernel
  where the s×s scores never reach DRAM, and this implementation spills them.
  That gap is the price of not fusing, quantified.
- **`block.rmsnorm` at 62.6%** with a `tinyThreadgroup` limiter — the one
  structural finding, and the only line the headroom section bothered to print.

### 7.3 Fixing what it found, and proving it

The example takes the norm kernel's threadgroup width as an argument, so the fix
is a second run:

```
$ metalscope profile --output block-256.json --no-report -- ./.build/release/block-bench 2048 256
$ metalscope diff block-32.json block-256.json
diff — baseline block-32.json  ->  candidate block-256.json
  device:  Apple M1 Pro
  peaks:   3.45 TF fp32 / 166.4 GB/s  [measured]

  kernel           shape                  baseline  candidate   delta  speedup         eff a->b     d eff  tgroup a->b  bound
  ---------------  --------------------  ---------  ---------  ------  -------  ---------------  --------  -----------  ---------
  block.rmsnorm    norm n=1048576         80.50 us   64.38 us  -20.0%    1.25x   62.6% -> 78.3%  +15.7 pp    32 -> 256  bandwidth
  block.qkv        gemm 2048x1536x512    956.18 us  982.40 us   +2.7%    0.97x   97.8% -> 95.2%   -2.6 pp            -  compute
  block.attention  attn b1 h8 s2048 d64   5.690 ms   5.751 ms   +1.1%    0.99x   43.8% -> 43.4%   -0.5 pp            -  compute
  block.gelu       elem n=1048576         49.99 us   52.98 us   +6.0%    0.94x  100.8% -> 95.2%   -5.7 pp          256  bandwidth

  matched 4/4 kernels — total 6.777 ms -> 6.850 ms (+1.1%)
  occupancy changes:
  - block.rmsnorm: 32 -> 256 threads/threadgroup (+15.7 pp of pipeline max), limiter tinyThreadgroup -> none
```

The norm kernel got 1.25x faster and the limiter went away. The other three rows
moved by 1–6% in both directions and mean nothing — see
[run-to-run variance](#calibration-varies-run-to-run). The
`occupancy changes:` line is the part you can actually trust from a single pair
of runs, because it is a statement about shapes, not about a stopwatch.

### 7.4 The capture API

```swift
let session = try CaptureSession()                    // or CaptureSession(device:)

try session.capture(label: "attn.sdpa",
                    shape: .attention(b: 1, h: 8, s: 512, d: 64),
                    precision: .fp16,
                    iterations: 32,
                    notes: ["backend": "custom"]) { region in
    for _ in 0..<32 {
        let encoder = try region.makeComputeCommandEncoder()   // stage-sampled
        encoder.setBuffer(out, offset: 0, index: 0)
        region.dispatchThreads(encoder, pipeline: pipeline,     // records occupancy
                               threads: grid, threadsPerThreadgroup: group)
        encoder.endEncoding()
    }
}

try session.writeTrace()          // honours $METALSCOPE_TRACE
```

| member | what it does |
| --- | --- |
| `capture(label:shape:precision:iterations:notes:_:)` | one region = one command buffer, committed and waited on. `iterations` is how many times the body encodes the annotated work; the recorded duration is the measured span ÷ `iterations` |
| `captureCompute(...)` | shorthand for a region that is exactly one compute encoder |
| `region.makeComputeCommandEncoder(label:)` | a compute encoder whose start/end are sampled into the region's counter buffers |
| `region.dispatchThreads(_:pipeline:threads:threadsPerThreadgroup:)` | sets the pipeline, encodes the dispatch, **and** records its static occupancy in one call |
| `region.dispatchThreadgroups(_:pipeline:threadgroupsPerGrid:threadsPerThreadgroup:)` | the same, for a grid declared in threadgroups |
| `region.observe(pipeline:threadsPerThreadgroup:threadgroupsPerGrid:)` | record occupancy for a dispatch you encoded yourself |
| `region.commandBuffer` | hand this to MPS/MLX; those regions get command-buffer timing and no occupancy |
| `session.reset()` / `session.dropLast(_:)` | discard warmup regions captured through the same code path |
| `session.maxSamplesPerRegion` | cap on stage samples per region (default 1024 = 512 sampled encoders). Metal allows at most 4096 samples on this chip |
| `session.records` | the `KernelRecord`s so far |
| `session.makeTrace(peaks:notes:)` / `session.writeTrace(to:peaks:notes:)` | build or write the trace |
| `CaptureSession.environmentTraceURL` | `$METALSCOPE_TRACE` as a URL, or nil |

The pipeline has to be handed over explicitly because **Metal offers no way to
ask an encoder which pipeline it is holding**, and metalscope will not swizzle to
find out. Dispatching straight on the encoder still works and is still timed —
the kernel just carries no `occupancy` block.

### 7.5 Shape annotations

FLOPs and compulsory bytes are computed analytically from the shape. Bytes
assume perfect on-chip reuse (each input read once, each output written once),
which makes AI an upper bound and the roofline placement conservative: a kernel
that looks bandwidth-bound under this model really is.

| annotation | FLOPs | bytes (e = element size) | use it for |
| --- | --- | --- | --- |
| `.gemm(m:n:k:)` | `2mnk` | `e(mk + kn + mn)` | any matmul: projections, FFN, output heads |
| `.attention(b:h:s:d:)` | `4bhs²d` | `4e·bhsd` | scaled dot-product attention |
| `.elementwise(n:)` | `n` | `2en` | one input, one output: activations, scaling, casts |
| `.norm(n:)` | `5n` | `2en` | layer/RMS norm — a two-pass statistic over a cache-resident row |
| `.opaque(flops:bytes:)` | as given | as given | anything else; you supply the numbers |

Notes on the models:

- **`.attention`** counts the two matmuls (QK^T and PV) and excludes softmax,
  which is transcendental overhead rather than useful FLOPs. Bytes assume a
  **fused** kernel: Q, K, V in and O out, with the s×s score matrix never
  reaching DRAM. An unfused implementation will therefore report well under
  100% — that gap is the cost of not fusing, and showing it is the point.
- **`.norm`** models one read and one write, not two reads, because the two
  statistics passes run over a row that stays in cache.
- **`.opaque`** is the escape hatch. `bench`'s streaming triad uses it because
  2 reads + 1 write per element does not fit `.elementwise`'s single-input
  model. Precision does **not** rescale opaque bytes — you gave the byte count,
  so it is used as given.
- Precision affects the byte count and which compute ceiling is used
  (`fp16`/`bf16`/`int8` score against the half-precision peak), not the FLOP
  count.

### 7.6 When timing falls back a tier

The ladder, best first:

1. **`counters`** — per-encoder GPU timestamps. Requires the device to support
   `.atStageBoundary` sampling, the `timestamp` counter set, **and** every
   encoder in the region to have been created via
   `region.makeComputeCommandEncoder()` and sampled.
2. **`cmdbuf`** — `MTLCommandBuffer.gpuStartTime/gpuEndTime`. Real GPU time, at
   command-buffer granularity.
3. **`host`** — wall clock around commit + wait. Includes scheduling latency.

You drop from 1 to 2 in three situations:

| situation | why |
| --- | --- |
| **a framework encodes the work** (MPS, MLX, Core ML) | metalscope never created those encoders, and Metal has no dispatch- or blit-boundary sampling on Apple silicon to bracket them with |
| **the region has more encoders than sample slots** | a region only claims tier 1 if *every* encoder contributed a sample; timing a sampled prefix and dividing by the full iteration count would over-report throughput several-fold |
| **the sample buffer could not be allocated** | Metal caps a counter sample buffer at 32 KB — 4096 timestamps, so ~2048 sampled encoders. Ask for more and metalscope gives up on sampling entirely and times the command buffer instead |

The MPS→cmdbuf case is visible in every table above: `block.qkv` and
`block.attention` say `cmdbuf` and print `-` in both occupancy columns, because
metalscope saw neither the encoders nor the pipelines. That is not a
degradation to route around; command-buffer GPU time is *real GPU time*, and
each captured region is its own command buffer precisely so that it belongs to
exactly the annotated work.

To stay on tier 1 with many encoders, raise the cap before the first capture:

```swift
session.maxSamplesPerRegion = 4096      // 2048 sampled encoders, the device maximum
```

---

## 8. Interpreting results

### Give the kernel enough work

The single most common way to get a meaningless roofline is to profile a working
set that fits in cache and a region that never reaches sustained clocks. Here is
the same binary, the same kernels, the same annotations, at sequence length 256
(512 KB of activations) and 2048 (4 MB):

```
$ metalscope profile --output block-small.json -- ./.build/release/block-bench
  kernel           shape                prec  time/iter   GFLOP/s  GB/s    AI  bound       ceiling    eff  tgroup    occ  timing
  ---------------  -------------------  ----  ---------  --------  ----  ----  ---------  --------  -----  ------  -----  --------
  block.rmsnorm    norm n=131072        fp32   50.88 us   12.9 GF  20.6  0.62  bandwidth  104.0 GF  12.4%      32   3.1%  counters
  block.qkv        gemm 256x1536x512    fp32  211.09 us   1.91 TF  24.8  76.8  compute     3.45 TF  55.4%       -      -  cmdbuf
  block.attention  attn b1 h8 s256 d64  fp32  508.31 us  264.0 GF   4.1  64.0  compute     3.45 TF   7.7%       -      -  cmdbuf
  block.gelu       elem n=131072        fp32   40.25 us    3.3 GF  26.0  0.12  bandwidth   20.8 GF  15.7%     256  25.0%  counters
```

versus the `seq=2048` table in [§7.2](#72-running-it): 12.4% → 62.6%,
55.4% → 97.8%, 7.7% → 43.8%, 15.7% → 100.8%. Nothing about the code changed. At
the small size the numbers are dominated by dispatch launch overhead and by the
GPU never leaving its low-power clock state.

Two rules follow:

- **Size the working set past the caches.** For bandwidth-bound kernels, tens of
  megabytes per stream.
- **Give each region enough iterations to reach sustained clocks.** Apple GPUs
  ramp over tens of milliseconds; `calibrate` auto-sizes to ~250 ms of GPU work
  per run for exactly this reason, and your `iterations` should aim somewhere
  near that too. In the example, the two microsecond-scale kernels use 2048
  iterations while the millisecond-scale MPS regions use 32.

The one thing that did **not** change between the two sizes is the
`tinyThreadgroup` finding on `block.rmsnorm`. Structural facts survive bad
benchmarking; roofline percentages do not.

### The observer effect

Stage-boundary counter sampling is not free. Same kernel, same ten back-to-back
dispatches, timed with sampling and without:

```
$ ./.build/release/observer
observer effect — Apple M1 Pro, 10 back-to-back dispatches, best of 12

  buffer   unsampled     sampled   overhead
  ------  ----------  ----------  ---------
    1 MB    221.1 us    333.4 us     +50.8%
    4 MB    601.7 us    793.0 us     +31.8%
    8 MB   1117.5 us   1358.6 us     +21.6%
   16 MB   2180.8 us   2441.4 us     +12.0%
   32 MB   3934.9 us   4407.5 us     +12.0%
   64 MB   8080.1 us   8158.8 us      +1.0%
```

Across six runs of that measurement the overhead ranged 49–129% at 1 MB,
32–49% at 4 MB, 12–18% at 16 MB, 7–12% at 32 MB, and 1–6% at 64 MB. It is
approximately a fixed per-encoder cost — tens of microseconds for ten
dispatches — so it shrinks as a fraction of the work as the dispatch grows.

Three consequences:

1. **Short kernels look slower under `counters` than they are in production.**
   Prefer larger working sets, or compare like-for-like traces where both sides
   paid the same overhead.
2. The overhead is **real GPU work**, not a measurement artefact: counter timing
   and the command-buffer time for the *same sampled run* agree to within a
   microsecond.
3. If you need an unperturbed number, let the region fall to `cmdbuf` on
   purpose. It measures the same span with no sampling cost, at the price of
   per-encoder detail.

> This refines what [TRACE-FORMAT.md](TRACE-FORMAT.md) records. That document's
> "no difference beyond noise at 32 MB" was measured on a 32 MB buffer with a
> different repeat count; re-measured here across six runs, 32 MB shows a
> consistent 7–12%. The overhead becomes negligible, but later than previously
> claimed.

### Measured vs spec-sheet peaks

Covered with a worked example in [§5](#5-choosing-which-peaks-to-score-against).
The short version: spec-sheet numbers for Apple GPUs are `cores × ALUs × 2 FLOP ×
boost clock`, which assumes an FMA issued every cycle at boost — a thing no real
GEMM does. On this M1 Pro the measured fp32 ceiling is 62–66% of the 5.2 TF
folklore figure and the measured bandwidth is 74–84% of the 200 GB/s bus figure
(a normal STREAM result). Scoring against folklore rates a 91.8%-of-achievable
fp16 GEMM at 29.4%.

Every metalscope table tags its peaks. If it says `[spec-sheet folklore]`, run
`calibrate` before you believe any percentage in that table.

### Calibration varies run-to-run

Two full `calibrate` runs on this machine, minutes apart, while a build was
running in the background:

| | fp32 | fp16 | bandwidth |
| --- | --- | --- | --- |
| cached (idle machine) | 3.45 TF | 3.33 TF | 166.4 GB/s |
| run A (loaded) | 3.24 TF | 3.10 TF | 147.8 GB/s |
| run B (loaded) | 3.17 TF | 3.08 TF | 155.1 GB/s |

That is a ~7% spread on compute and ~11% on bandwidth. Because a ceiling is what
the hardware *can* do, the highest measurement is the right one to keep, which
is why `calibrate` takes the best of `--repeats` rather than the mean — and why
you should run it on an otherwise idle machine.

The same noise floor applies to `bench` and `profile`, which capture **one** run
per kernel with no best-of. A single `diff` of a single pair of runs is not
evidence:

- In [§7.3](#73-fixing-what-it-found-and-proving-it), `block.qkv` — unchanged
  code — moved by +2.7%, and `block.gelu` — also unchanged — by +6.0%.
- In [§2](#metalscope-diff--what-moved), `ffn.gemm` moved from 66.1% to 94.3%
  between two `bench` runs because MPS chose differently for the same shape.
- Before the example's iteration counts were raised, the `block.rmsnorm` baseline
  measured anywhere from 77 µs to 195 µs across five runs, making the same fix
  look like anything from 1.02x to 2.16x.

Practical advice: repeat the pair, look for the delta that survives, and weight
the `occupancy changes:` section heavily — it reports a change in *shape*, which
does not have a noise floor.

### Efficiency over 100%

`eff` above 100% is legal and worth reading, not a bug. `block.gelu` reports
100.8% and `stream.triad` has been seen at 101.4%. There are three causes, in
rough order of likelihood:

1. **The analytic byte model over-counts.** Bytes are *compulsory DRAM traffic*.
   A 4 MB working set partly resident in the system-level cache moves less real
   DRAM traffic than the model assumes, so the achieved "GB/s" — which is
   modelled bytes ÷ time — exceeds what DRAM could have delivered.
2. **The peaks are stale.** A ceiling measured on a hot or busy machine is too
   low; anything measured later on an idle one will beat it.
3. **The kernel genuinely beats the calibration workload.** The triad used for
   calibration is one access pattern; another kernel may stream slightly better.

In all three cases the reading is the same: *this kernel is at the wall.* Treat
anything above ~95% as "no headroom here" and go look elsewhere.

### The occupancy cross-check: lanes versus time

This is the part of the report that most distinguishes metalscope from a table
of counters, so it is worth seeing both outcomes.

**Case 1 — the defect costs lanes, not time.** Reporting the `baseline` trace
against peaks measured in the same machine state:

```
$ metalscope report baseline.json --peaks-file peaks-verify.json
  ...
  act.scale      elem n=16777216          fp32  900.11 us   18.6 GF  149.1  0.12  bandwidth  18.5 GF  100.9%    100*   9.8%  counters
  ...
  headroom:
  - act.scale: threadgroup of 100 is not a multiple of the 32-wide SIMD group — 28 of 128 lanes idle in every threadgroup (78.1% lane use); round to 96 or 128; note it is already at 100.9% of its bandwidth ceiling, so this costs lanes, not time
```

The threadgroup really is malformed — 28 of every 128 lanes are launched with
nothing to do — but the kernel is already saturating memory bandwidth. Those
idle lanes were going to be waiting on DRAM regardless. Fixing the shape wins
back lanes, not microseconds, and the report says so rather than sending you off
to fix a defect that is not costing anything. This caveat fires whenever the
kernel is at ≥90% of its ceiling.

**Case 2 — the defect costs time.** `block.rmsnorm` in
[§7.2](#72-running-it) has a `tinyThreadgroup` limiter and sits at 62.6% of its
bandwidth ceiling — well clear of the caveat threshold. No caveat is printed,
and the fix in [§7.3](#73-fixing-what-it-found-and-proving-it) does buy 1.25x.

The rule: **a structural occupancy finding is only actionable when the kernel has
roofline headroom.** metalscope does that check for you.

### A low occupancy ratio is not a finding

`block.gelu` runs at 25.0% occupancy — 256 threads out of a 1024 pipeline
ceiling — and metalscope reports exactly nothing about it. That is deliberate.
256 threads per threadgroup is a perfectly good shape on Apple silicon, and a
profiler that flagged it would train you to skip the occupancy section entirely.

What does get flagged:

- a threadgroup that is **not a multiple of the SIMD width** (lanes idle in every
  threadgroup, forever);
- a threadgroup that is **a single SIMD group** (nothing to interleave against
  memory latency);
- **threadgroup memory over half the device limit** (caps co-residency).

Note that `block.rmsnorm` at 3.1% occupancy *is* flagged — not because 3.1% is a
low number, but because 32 threads is one SIMD group. The number in the `occ`
column is context, not a verdict.

---

## 9. Implementing what's missing

### 9.1 Adding a counter-set resolver

metalscope ships resolvers for all three of Apple's common counter sets, so a
chip that exposes `stageutilization` or `statistic` gets them sampled, resolved,
and written to `kernels[].counters` with **no code change**. You only need to
write a resolver for a counter set outside `MTLCommonCounterSet` entirely — the
case `metalscope info` reports as `no — resolver missing`.

A sample buffer resolves to a packed array of the counter set's own C struct, so
a resolver's whole job is to know that struct's layout:

```swift
import Metal

public struct MyVendorCounterResolver: CounterSetResolver {
    public init() {}

    /// Must match MTLCounterSet.name exactly, as `metalscope info` prints it.
    public let counterSetName = "myvendorset"

    /// Counter names in the order their fields appear in the resolved struct.
    /// This order is the whole contract — get it wrong and every value is
    /// attributed to the wrong counter, silently.
    public let counterNames = [
        MTLCommonCounter.totalCycles.rawValue,
        "ALUBusyCycles",
        "MemoryUnitBusyCycles",
    ]

    /// MemoryLayout<MTLCounterResultMyVendor>.stride for this set.
    public var resolvedStride: Int { MemoryLayout<MTLCounterResultMyVendor>.stride }
}
```

Then add it to `CounterResolvers.all` in
`Sources/MetalscopeCapture/CounterResolvers.swift`. `CaptureSession` picks up
every exposed set that has a resolver, attaches a sample buffer per set (up to
four attachments, timestamp always first), and sums per-encoder deltas into the
kernel record.

Three rules the existing resolvers follow, and yours must too:

1. **`layoutIsConsistent` must hold** — `resolvedStride ==
   counterNames.count * MemoryLayout<UInt64>.stride`. The default `decode`
   walks `counterNames.count` little-endian `uint64_t`s per sample, one sample
   every `resolvedStride` bytes. Assert this in a test; the day Apple adds a
   field to one of these structs, the test fails instead of the decode silently
   sliding by eight bytes.
2. **A counter the GPU declined to write is `MTLCounterErrorValue` (~0) and must
   be omitted, not recorded.** A missing key means "not written"; it must never
   be mistaken for zero.
3. **Decode timestamps as `UInt64`, never `Double`.** GPU timestamps are
   nanoseconds since boot; after ~104 days of uptime they exceed 2⁵³ and a
   `Double` begins rounding them, silently shortening every kernel duration.

Test it against synthetic resolved data — `Tests/metalscopeTests/CounterResolverTests.swift`
does exactly this for the two sets no Apple chip exposes. An untested path that
only wakes up on hardware nobody here owns is a path that will be broken when it
does.

### 9.2 Contributing a COUNTER-MATRIX row

[COUNTER-MATRIX.md](COUNTER-MATRIX.md) collects what each chip actually exposes,
measured rather than read off a documentation page. One machine can contribute
one row, and the recipe takes about ten seconds:

```
$ metalscope info --json > my-chip.json
```

That payload carries everything a row needs:

```jsonc
{
  "counterSetDetail" : [
    { "counters" : [ "GPUTimestamp" ], "name" : "timestamp", "resolver" : true }
  ],
  "device" : {
    "counterSets" : [ "timestamp" ],
    "maxThreadgroupMemoryBytes" : 32768,
    "maxWorkingSetBytes" : 12713115648,
    "name" : "Apple M1 Pro",
    "registryID" : 4294969717,
    "supportsStageBoundarySampling" : true
  },
  "knownCounterSetsAbsentHere" : [ "stageutilization", "statistic" ],
  "maxThreadsPerThreadgroup" : [ 1024, 1024, 1024 ],
  "osVersion" : "Version 26.5.1 (Build 25F80)",
  "samplingPoints" : [ "stage" ],
  "timingLadderTier" : "counter-sample-buffer",
  "toolVersion" : "0.1.0"
}
```

Map it to the table like this:

| matrix column | JSON field |
| --- | --- |
| chip | `device.name` |
| OS | `osVersion` |
| counter sets | `counterSetDetail[].name` with `counters` |
| sampling boundaries | `samplingPoints` |
| timing ladder tier | `timingLadderTier` (1 = `counter-sample-buffer`, 2 = `command-buffer`, 3 = `host`) |
| max threadgroup mem | `device.maxThreadgroupMemoryBytes` |
| source | "measured, metalscope `toolVersion`" |

The payload contains no personal data beyond the GPU name and OS build;
`registryID` is a boot-local handle, not a serial number.

The row worth sending most is one where `counterSetDetail` has an entry with
`"resolver" : false`, or where `knownCounterSetsAbsentHere` is *shorter* than
`["stageutilization", "statistic"]` — that means a chip finally exposes
something beyond `timestamp`, and the dormant code paths in
[§9.1](#91-adding-a-counter-set-resolver) can be exercised for the first time.

### 9.3 What stays Instruments-only, and why

Some things are not missing features; they are outside what Metal exposes
programmatically at all. metalscope says so rather than estimating them.

| what | why it isn't here |
| --- | --- |
| **Instruction-level stalls** | No counter set exposes them. Instruments/Xcode GPU capture reads them through a private path metalscope has no supported access to |
| **Bank conflicts** | Same |
| **Measured occupancy** (threadgroups actually resident per core) | Needs a per-core thread and register budget Apple does not publish and Metal does not expose. The static analysis is an upper bound on the *shape*, not a measurement of residency, and it says so |
| **ALU-busy / memory-unit-busy** | Would come from `stageutilization`, which no Apple chip metalscope has run on exposes. The resolver exists and is tested; it needs hardware |
| **Bracketing another process's encoders** | Requires `.atDispatchBoundary` or `.atBlitBoundary` sampling, unsupported on Apple silicon. This is why the attach model is an opt-in library rather than injection |

For those, use Instruments' Metal System Trace. The two tools answer different
questions: Instruments tells you what the hardware did instruction by
instruction; metalscope tells you how far a kernel is from the ceiling its shape
allows, and whether the difference between two versions is real.

---

## 10. Command and error reference

```
metalscope info                        device, exposed counter sets, peaks
metalscope calibrate                   measure real GEMM/streaming ceilings
metalscope bench                       self-test: capture a real trace
metalscope profile -- <cmd> [args]     run an instrumented target, report
metalscope report <trace.json>         roofline table
metalscope diff <a.json> <b.json>      side-by-side comparison
metalscope help | version
```

Run `metalscope help` for the full flag list.

### Common flags

| flag | commands | effect |
| --- | --- | --- |
| `--json` | all | machine-readable output |
| `--peaks-file PATH` | `report`, `diff`, `profile`, `calibrate` | read (or, for `calibrate`, write) peaks here instead of `~/.metalscope/peaks.json` |
| `--spec-peaks` | `report`, `diff` | force spec-sheet folklore |
| `--sort duration\|efficiency\|intensity` | `report`, `profile` | row order |
| `--occupancy` | `report` | add the per-kernel occupancy detail block |
| `--output PATH` | `bench`, `profile` | trace destination |
| `--no-report` | `profile` | collect the trace without printing |
| `--variant baseline\|tuned` | `bench` | which self-test kernels to run |

### Errors

Every error is a sentence and an exit code; nothing is printed to stdout before
validation completes, so a typo cannot half-print a report.

```
$ metalscope report baseline.json --occupancyy
metalscope: unknown option '--occupancyy'                                  # exit 2

$ metalscope report baseline.json --sort speed
metalscope: bad value 'speed' for --sort: expected duration|efficiency|intensity   # exit 2

$ metalscope report
metalscope: missing argument: <trace.json>                                 # exit 2

$ metalscope repot
metalscope: unknown command 'repot' — run `metalscope help`                # exit 2

$ metalscope profile -- /bin/echo hello
hello
metalscope: child wrote no trace at ~/work/metalscope-trace.json.
  The target must link the MetalscopeCapture library and call
  CaptureSession.writeTrace() — see docs/TRACE-FORMAT.md.                  # exit 2
```

A trace whose `schemaVersion` is newer than this build understands is rejected
rather than partially read:

```
metalscope: trace schema version 3 is newer than supported version 2
```

---

## See also

- [ARCHITECTURE.md](ARCHITECTURE.md) — the design, the timing ladder, why the
  peaks are measured
- [TRACE-FORMAT.md](TRACE-FORMAT.md) — the JSON schema (v2), field by field
- [COUNTER-MATRIX.md](COUNTER-MATRIX.md) — what each chip exposes
- [WHITEPAPER.md](WHITEPAPER.md) — the argument, the evaluation, and the
  limitations in full
