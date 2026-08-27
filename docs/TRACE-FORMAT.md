# metalscope trace format

One JSON schema, shared by capture (`MetalscopeCapture`), `metalscope report`,
`metalscope diff`, and any future HTML output. Current `schemaVersion` is **3**.
A reader must reject a trace whose `schemaVersion` is greater than the one it
knows (`TraceIO.decode` does).

Traces are pretty-printed with sorted keys, so two traces can be compared with
plain `diff` as well as with `metalscope diff`.

### Version history

| version | change |
| --- | --- |
| 1 | initial: device, peaks, kernels with analytic FLOPs/bytes and the timing ladder |
| 2 | added `kernels[].occupancy`, `kernels[].counters`, `device.maxThreadgroupMemoryBytes` |
| 3 | added `kernels[].durationSamplesSeconds` — one per timed repeat |

Every addition since v1 is optional, so **v1 and v2 traces still read** — they
simply carry no occupancy block and no repeat samples, and `report`/`diff` drop
those columns rather than inventing values. The reverse is not true: an older
reader rejects a newer trace, which is the documented contract of the version
field.

## Top level

```jsonc
{
  "schemaVersion": 3,
  "tool": "metalscope",
  "toolVersion": "0.1.0",
  "createdAt": "2026-08-26T19:06:18.512Z",   // ISO-8601, milliseconds
  "device": { ... },
  "peaks": { ... },                          // optional
  "kernels": [ ... ],
  "notes": { "command": "bench", "variant": "baseline", "repeats": "5" }  // optional, free-form
}
```

### `device`

```jsonc
{
  "name": "Apple M1 Pro",
  "registryID": 4294970113,                  // optional
  "maxWorkingSetBytes": 11453251584,         // optional
  "counterSets": ["timestamp"],              // what this chip exposes
  "supportsStageBoundarySampling": true,     // optional
  "maxThreadgroupMemoryBytes": 32768         // optional, v2
}
```

`counterSets` is the honest record of what the OS/chip made available at capture
time. On the M-series today that is `timestamp` and nothing else; see
[ARCHITECTURE.md](ARCHITECTURE.md) for what Metal does not expose at all, and
[COUNTER-MATRIX.md](COUNTER-MATRIX.md) for the per-chip table.

### `peaks`

The roofline ceilings in effect when the trace was captured.

```jsonc
{
  "source": "measured",            // "measured" | "spec-sheet"
  "chip": "Apple M1 Pro",
  "fp32GFLOPS": 3412.73,
  "fp16GFLOPS": 3324.11,           // optional (absent if never measured)
  "bandwidthGBs": 168.13,
  "measuredAt": "2026-08-26T19:05:11.804Z",  // optional
  "details": { "gemmBestSize": 2048, "repeats": 3 }  // optional provenance
}
```

`source` is load-bearing: `"spec-sheet"` numbers are community folklore and every
tool that prints them must say so. `report` prefers, in order: `--peaks-file`,
the trace's own peaks when they are `measured`, the local
`~/.metalscope/peaks.json`, then whatever the trace carried.

`~/.metalscope/peaks.json` uses its own small schema — `{ "schemaVersion": 1,
"chips": { "<device name>": PeakSet } }` — so one machine can hold measurements
for several GPUs.

### `kernels[]`

```jsonc
{
  "label": "ffn.gemm",
  "shape": { "kind": "gemm", "m": 1024, "n": 1024, "k": 1024 },
  "precision": "fp32",             // fp32 | fp16 | bf16 | int8
  "durationSeconds": 0.00067830,   // ONE invocation; the median when repeated
  "iterations": 162,               // invocations encoded into ONE region
  "timingSource": "command-buffer",
  "flops": 2147483648,             // analytic, for one invocation
  "bytes": 12582912,               // analytic, for one invocation
  "hostDurationSeconds": 0.1531,   // optional: wall clock around commit+wait
  "stages": [                      // optional: per-encoder spans
    { "name": "triad_vec4", "durationSeconds": 0.0012 }
  ],
  "occupancy": { ... },            // optional, v2 — see below
  "counters": {                    // optional, v2 — absent on Apple silicon
    "stageutilization.TotalCycles": 184320
  },
  "durationSamplesSeconds": [      // optional, v3 — one per timed repeat
    0.00067303, 0.00068620, 0.00067830, 0.00067451, 0.00068102
  ],
  "notes": { "backend": "MPSMatrixMultiplication" }   // optional
}
```

`flops`/`bytes` are stored rather than derived so that a reader needs no copy of
the shape registry, and so `opaque` shapes survive a round trip.

Note that `durationSeconds` is per invocation while `stages[]` durations are raw
per-encoder spans, not divided by `iterations`.

#### `durationSamplesSeconds` (v3)

Per-invocation seconds for each *timed repeat*, in the order they ran. Present
only when the region was captured more than once; a single-run capture omits the
field entirely, because an array of one would read as a measured spread of zero.

`iterations` describes one repeat, not the total: five repeats of 162 iterations
is `"iterations": 162` and five samples, each already divided by 162.

**Only the samples are stored.** `min`, `median`, `mean`, `p95`, `max` and the
spread fraction are derived on read, for the same reason the occupancy ratios
are: a trace that carried both could be edited or merged into a state where the
summary and the samples disagree, and a reader would have no way to tell which
half to believe. `metalscope report --json` emits the derived values under
`runStatistics` so a consumer never has to re-implement the conventions — which
matter, and are:

- **median** is the *lower* of the two middle samples when the count is even, so
  `durationSeconds` is always a run that actually happened rather than the
  average of two that did.
- **p95** is nearest-rank — index `ceil(0.95 n) - 1` — never interpolated. With
  the default five repeats that makes p95 the slowest sample, which is the
  honest reading of "95th percentile of five points".

`durationSeconds` is the median of these samples, and every derived number in a
report — GFLOP/s, bandwidth, efficiency, roofline placement — is computed from
it. A wide spread is therefore a warning about the whole row, not just the
duration column.

The other per-region fields — `stages`, `counters`, `occupancy`,
`hostDurationSeconds` — come from the single repeat whose duration is that
median. Summing or averaging them across repeats would describe a run that never
happened. `timingSource`, by contrast, is the *worst* tier any repeat reached: a
duration is only as trustworthy as its weakest sample.

Warm-up runs are not among the samples. A capture with `repeats > 1` runs one
discarded region at the same iteration count first, so that every timed repeat
sees an already-ramped GPU rather than the first one absorbing the clock ramp
(§3.1 of the whitepaper, and the reason `calibrate` probes before it measures).

#### `occupancy` (v2)

Static occupancy, from `MTLComputePipelineState` plus the threadgroup size the
dispatch actually used. No counters are involved, so this block appears on every
chip — including the ones that expose nothing but `timestamp`.

```jsonc
{
  "threadsPerThreadgroup": 100,          // as dispatched (w x h x d)
  "maxTotalThreadsPerThreadgroup": 1024, // pipeline ceiling (register pressure)
  "threadExecutionWidth": 32,            // SIMD group width, read not assumed
  "threadgroupMemoryBytes": 0,           // staticThreadgroupMemoryLength
  "threadgroupMemoryLimitBytes": 32768,  // optional: device limit
  "threadgroupsPerGrid": 167773,         // optional: threadgroups per dispatch
  "dispatchCount": 192,                  // dispatches folded into this record
  "variantCount": 1                      // distinct dispatch shapes seen
}
```

**Only raw inputs are stored.** Every ratio a report shows —
`simdGroupsPerThreadgroup` (rounded *up*: a ragged threadgroup still costs a full
SIMD group), `threadgroupOccupancy` (dispatched / pipeline max),
`laneUtilization`, `idleLanesPerThreadgroup`, `threadgroupMemoryPressure`, and
the `limiter` classification — is derived on read, so a trace can never disagree
with itself. `metalscope report --json` emits the derived values too, for
consumers that would rather not re-implement the arithmetic.

`limiter` is one of `none`, `executionWidthAlignment` (threadgroup not a multiple
of the SIMD width — lanes idle in *every* threadgroup, forever),
`tinyThreadgroup` (a single SIMD group, so nothing to interleave against memory
latency), or `threadgroupMemory` (static threadgroup memory over half the device
limit). A low `threadgroupOccupancy` on its own is deliberately **not** a
limiter: 256 threads out of a 1024 ceiling is a perfectly good shape, and
flagging it would train people to ignore the section.

The block is absent when metalscope never saw the pipeline state — MPS, MLX and
anything else that encodes its own dispatches. Absent means "not observed", never
"nothing to report".

When a region dispatches several *different* shapes, the worst one is recorded
and `variantCount` says how many there were. Averaging threadgroup sizes would
describe a dispatch that never ran.

#### `counters` (v2)

Resolved hardware counters other than the timing ladder, keyed
`"<counter set>.<counter>"` and summed over the region's encoders (per-encoder
`end - start` deltas). For `stageutilization`, per-stage fractions of
`TotalCycles` are appended as `<counter>Fraction`.

Absent on every Apple chip shipped so far, which expose only `timestamp`.
`MetalscopeCapture` ships resolvers for all three of Apple's common counter sets
and samples any that a device exposes, so this field populates itself on hardware
that has them. See [COUNTER-MATRIX.md](COUNTER-MATRIX.md).

A counter the GPU declined to write (`MTLCounterErrorValue`) is **omitted**, not
recorded as its sentinel value: a missing key means "not written", never "zero".

#### `timingSource`

| value | meaning |
| --- | --- |
| `counter-sample-buffer` | `MTLCounterSampleBuffer` timestamps at compute-encoder stage boundaries. Most precise; requires the device to support `.atStageBoundary` sampling **and** every encoder in the region to have been sampled. |
| `command-buffer` | `MTLCommandBuffer.gpuStartTime/gpuEndTime`. Real GPU time at command-buffer granularity — used when someone else creates the encoders (MPS), or when a region encodes more encoders than its sample buffer could hold. |
| `host` | Wall clock around commit + wait. Includes scheduling latency; a fallback, not a measurement. |

metalscope never mixes these: a region reports the best source that covered
*all* of its work. Timing the sampled prefix of a region and dividing by the full
iteration count would over-report throughput several-fold, so partial sampling
demotes the region to `command-buffer` instead.

**Observer effect.** Stage-boundary sampling is not free. Measured on an M1 Pro,
ten back-to-back dispatches over 4 MB buffers took 600 µs unsampled and 809 µs
sampled (~35% overhead); the same test over 32 MB buffers showed no difference
beyond noise. Counter timing and the command-buffer time for the *same* sampled
run agree to within a microsecond, so the overhead is real work, not a
measurement artifact — but it means very short kernels look slower under
`counters` than they are in production. Prefer larger working sets, or compare
like-for-like traces.

## `shape` kinds

FLOPs and bytes are computed analytically from the shape — this is what
"ML-workload-aware" means, and it's what Instruments cannot do. Bytes are
*compulsory* DRAM traffic assuming perfect on-chip reuse, which makes arithmetic
intensity an upper bound and roofline placement conservative.

| kind | fields | FLOPs | bytes (e = element size) |
| --- | --- | --- | --- |
| `gemm` | `m`, `n`, `k` | `2mnk` | `e(mk + kn + mn)` |
| `attention` | `b`, `h`, `s`, `d` | `4bhs²d` | `4e·bhsd` |
| `elementwise` | `n` | `n` | `2en` |
| `norm` | `n` | `5n` | `2en` |
| `opaque` | `flops`, `bytes` | as given | as given (precision does not rescale) |

- **attention** counts the two matmuls (QK^T and PV); softmax is transcendental
  overhead and is excluded. Bytes assume a *fused* kernel: Q, K, V in, O out,
  with the s×s score matrix never reaching DRAM. An unfused implementation will
  therefore report well under 100% efficiency — that gap is the cost of not
  fusing, and showing it is the point.
- **norm** models a two-pass mean/variance over a cache-resident row, hence
  one read and one write rather than two reads.
- **opaque** is the escape hatch for kernels with no registered shape (the
  streaming triad uses it: 2 reads + 1 write per element doesn't fit
  `elementwise`'s single-input model).

## Diff alignment

`metalscope diff` aligns kernels by `label` + shape description + `precision`.
Occupancy is compared only when *both* sides have it: a v1 baseline against a v2
candidate reports no occupancy delta rather than treating the missing side as
unchanged, which would announce a fix that never happened.
Repeated identical keys pair up positionally (the Nth occurrence in the baseline
matches the Nth in the candidate); leftovers on either side are reported as
present in only one trace rather than being dropped. Change a kernel's shape and
it will *not* align — that's deliberate, since the comparison would be
meaningless.

## Diff verdicts

`diff` compares the two medians and adds a `verdict`, which is the one column
that says whether the delta beside it means anything:

| verdict | when |
| --- | --- |
| `faster` / `slower` | the two `[min, p95]` intervals are disjoint, and the candidate's median is the lower / higher one |
| `within-noise` (shown as `no call`) | the intervals overlap — a gap narrower than the run-to-run noise that produced it is not a result |
| `unmeasured` (shown as `-`) | one or both sides were captured once, so there is no spread to compare a delta against |

The interval runs from `min` to `p95` rather than `min` to `max`, so a single
scheduler hiccup does not widen it far enough to swallow every real result.
Intervals that touch at a point count as overlapping: the rule exists to refuse
marginal calls, so the boundary resolves toward refusing.

`--json` carries `verdict`, `spreadsOverlap`, each side's repeat count, and each
side's `min`/`p95`, plus the rule itself as `verdictRule` — a consumer reading a
withheld verdict needs to know what withholding one means.

This narrows the false-positive rate rather than abolishing it. Measured on an
M1 Pro over all six pairings of four captures of an *unchanged* baseline at
microsecond scale: single-run comparisons showed a >5% delta in 30 of 36
kernel pairs, worst 132%; the same captures at five repeats produced a verdict
at all in 4 of 36, worst median gap 7.2%.

## Writing traces from your own code

```swift
import MetalscopeCapture

let session = try CaptureSession()
try session.capture(label: "attn.sdpa",
                    shape: .attention(b: 1, h: 8, s: 512, d: 64),
                    precision: .fp16,
                    iterations: 32,
                    repeats: 5) { region in       // 5 timed runs + a warm-up
    for _ in 0..<32 {
        let encoder = try region.makeComputeCommandEncoder()   // stage-sampled
        encoder.setBuffer(...)
        // Dispatch through the region to get an occupancy record for free:
        region.dispatchThreads(encoder, pipeline: pipeline,
                               threads: grid, threadsPerThreadgroup: group)
        encoder.endEncoding()
    }
}
try session.writeTrace()      // honours $METALSCOPE_TRACE
```

`repeats` runs the closure that many times over, each into its own command
buffer, and appends **one** record holding every sample. It defaults to 1, so
existing callers are unchanged and record no spread — and `diff` will then
withhold every verdict rather than reading one point as a measurement.
`warmupRuns:` overrides the discarded run (default: one when repeating, none
when not) for callers whose workload is already warm.

`region.dispatchThreads` / `region.dispatchThreadgroups` set the pipeline state,
encode the dispatch, and record its static occupancy — so what the trace says was
dispatched is what ran. Dispatching straight on the encoder still works and still
gets timed; the kernel just carries no `occupancy` block. If you must encode the
dispatch yourself, call `region.observe(pipeline:threadsPerThreadgroup:)` to
supply it. Metal offers no way to ask an encoder which pipeline it holds, and
metalscope will not swizzle to find out.

Use `region.commandBuffer` directly for work encoded by a framework (MPS, MLX);
those regions fall back to command-buffer GPU time automatically and carry no
occupancy.

`metalscope profile -- <your-binary>` sets `METALSCOPE_TRACE` in the child's
environment and reports on whatever it writes there.
