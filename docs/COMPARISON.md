# metalscope vs. Nsight Compute: a capability comparison

*2026-08-26. No efficiency fraction makes sense between profilers; the comparison is
capability coverage and measurement honesty. Local data from calibrate/bench on
M1 Pro and M1 Max.*

## Capability coverage

| capability | Nsight Compute (CUDA) | metalscope (Metal) |
|---|---|---|
| kernel wall time | hardware counters | GPU timestamps — 3-tier ladder (counters → cmdbuf → host), tier labeled per row |
| occupancy | measured (achieved-occupancy counter) | static, derived from pipeline state; **no Metal counter exposes measured occupancy** |
| memory throughput | measured per level (DRAM/L2/L1) | derived: analytic bytes ÷ measured time |
| stall reasons, bank conflicts | instruction-level sampling | not exposed by Metal — documented as Instruments-only, deliberately not estimated |
| roofline | speed-of-light % against spec ceilings | efficiency % against **measured** ceilings (`calibrate`) |
| kernel diffing | baseline comparison | built in; aligns by label+shape, refuses misleading comparisons |
| ML-shape awareness | none | analytic FLOPs/bytes from shape annotations — the differentiator |

## The measured-vs-spec argument, quantified on two chips

| chip | fp32 measured/spec | bandwidth measured/spec |
|---|---|---|
| M1 Pro | 3.45 / 5.2 TF = **66%** | 166.4 / 200 GB/s = **83%** |
| M1 Max | 6.2 / 10.4 TF = **60%** | 371.5 / 400 GB/s = **93%** |

Bandwidth specs are nearly honest; compute specs overstate reachable peak by 1.5–1.7×,
*by different amounts per chip*. No single correction factor rescues spec-based
rooflines: two roofs wrong by different ratios move the ridge point itself. This is the
same "speed-of-light vs. achievable" problem Nsight users know on CUDA — resolved here
by measuring the ceilings instead of asserting them.

## Where the CUDA side stays ahead

Utilization counter sets exist in Metal's API but no current Apple part exposes them;
instruction-level attribution is Instruments-only. metalscope's response is the
per-chip [COUNTER-MATRIX.md](COUNTER-MATRIX.md): document reality per chip, light up
automatically (the resolvers are written and unit-tested against synthetic data) when
Apple exposes more.

Validation in practice: metalscope's calibrated ceilings are the reference the
sibling projects score against — the triton-metal GEMM attribution table reports
its throughput as a fraction of metalscope's measured device peak, and its fused
attention comparison cites metalscope's measured SDPA composite as the check on
the composite it times itself, which is the direction that costs it margin.
(mccl's collectives numbers come from its own fabric benchmark, `mcclbench`,
under the same measured-denominator reporting standard.)
