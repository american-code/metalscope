# Per-chip counter matrix

What Metal actually exposes, per chip and OS — **measured**, not read off a
documentation page. Apple documents `MTLCommonCounterSet` as three sets, but
which of them a given device returns from `MTLDevice.counterSets` is a per-chip,
per-OS fact that nobody publishes. This table is where metalscope collects it.

One machine can only contribute one row. If you have a chip that isn't here,
[the recipe below](#contributing-a-row) takes about ten seconds.

## The table

| chip | OS | counter sets | sampling boundaries | timing ladder tier | max threadgroup mem | source |
| --- | --- | --- | --- | --- | --- | --- |
| Apple M1 Pro | macOS 26.5.1 (25F80) | `timestamp` (`GPUTimestamp`) | stage | 1 — `counter-sample-buffer` | 32 KB | measured, metalscope 0.1.0 |
| Apple M1 Max | macOS 26.5.1 | `timestamp` only | ? | ? | ? | **partial** — see below |
| Apple M1 / M1 Ultra | — | — | — | — | — | *unfilled* |
| Apple M2 family | — | — | — | — | — | *unfilled* |
| Apple M3 family | — | — | — | — | — | *unfilled* |
| Apple M4 family | — | — | — | — | — | *unfilled* |

Columns:

- **counter sets** — `MTLDevice.counterSets`, with each set's counter names.
- **sampling boundaries** — which of `.atStageBoundary`, `.atDispatchBoundary`,
  `.atBlitBoundary`, `.atDrawBoundary` return true from
  `MTLDevice.supportsCounterSampling(_:)`.
- **timing ladder tier** — the best tier of the ladder in
  [ARCHITECTURE.md](ARCHITECTURE.md#counter-capture) this device can reach:
  1 = `counter-sample-buffer` (per-encoder GPU timestamps), 2 = `command-buffer`,
  3 = `host`. Tier 1 needs the `timestamp` set *and* stage-boundary sampling.
- **max threadgroup mem** — `MTLDevice.maxThreadgroupMemoryLength`, the
  denominator for threadgroup-memory pressure in the occupancy analysis.

### The partial M1 Max row

A Mac Studio M1 Max (studio-b) running the same macOS 26.5.1 reports the **same
timestamp-only counter situation** as the M1 Pro: `MTLDevice.counterSets`
contains `timestamp` and nothing else. That is the one fact confirmed on that
machine, and it is the one recorded.

The `?` cells are not unknown-because-unsupported; they are unknown because no
`metalscope info --json` payload was collected there. They are deliberately
**not** copied from the M1 Pro row. Sampling boundaries and
`maxThreadgroupMemoryLength` are per-chip facts, and a table whose entire premise
is *measured, not documented* cannot start inferring cells from a sibling chip
without becoming exactly the kind of folklore it exists to replace. Tier 1 in
particular requires stage-boundary sampling to be confirmed, not assumed.

If you have an M1 Max, [the recipe below](#contributing-a-row) completes this row
in ten seconds.

## What the M1 Pro row means in practice

Only `timestamp` exists, so:

- kernel **durations** are precise (per-encoder GPU timestamps);
- there is no ALU-busy, no memory-unit-busy, no occupancy counter, and no
  instruction-level anything — see
  [ARCHITECTURE.md](ARCHITECTURE.md#counter-capture) for what Metal withholds
  entirely versus what this particular chip merely lacks;
- everything metalscope reports about occupancy is *static*, derived from
  `MTLComputePipelineState` rather than sampled. That analysis needs no counters
  and therefore works identically on every row of this table.

`stageutilization` and `statistic` resolvers ship anyway
(`Sources/MetalscopeCapture/CounterResolvers.swift`) and are unit-tested against
synthetic resolved data. A chip that exposes either set gets it sampled,
resolved, and written to `kernels[].counters` with **no code change** — which is
the only way to keep those paths from rotting on a machine that can't run them.

Note also that the two non-timestamp sets are graphics-shaped: their counters are
vertex/fragment/tessellation cycles and invocations. The one directly useful to a
compute profiler is `statistic`'s `KernelInvocations`.

## Contributing a row

```
$ metalscope info --json > my-chip.json
```

That payload carries everything a row needs — `device.counterSets`,
`counterSetDetail[]` (per-set counter names and whether metalscope can decode
them), `knownCounterSetsAbsentHere`, `samplingPoints`, `timingLadderTier`,
`device.maxThreadgroupMemoryBytes`, `osVersion`, and `toolVersion`. It contains
no personal data beyond the GPU name and OS build; `registryID` is a
boot-local handle, not a serial number.

The human-readable `metalscope info` prints the same facts, including a per-set
table and an explicit "not exposed here" line:

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
```

If your chip exposes a set metalscope reports as `no — resolver missing`, that's
the interesting case: a counter set outside `MTLCommonCounterSet` entirely. Send
the JSON.
