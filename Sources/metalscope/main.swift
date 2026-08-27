import Foundation
import MetalscopeCore

// metalscope — an ML-native kernel profiler for Metal, in the spirit of
// Nsight Compute: roofline placement against the local chip's *measured* peaks,
// analytic FLOPs/bytes for known ML kernel shapes, and side-by-side kernel diffs.
//
//   metalscope info                        device, exposed counter sets, peaks
//   metalscope calibrate                   measure real GEMM/streaming ceilings
//   metalscope bench                       self-test: capture a real trace
//   metalscope profile -- <cmd> [args]     run an instrumented target, report
//   metalscope report <trace.json>         roofline table
//   metalscope diff <a.json> <b.json>      side-by-side comparison
//
// See docs/ARCHITECTURE.md for the design and docs/TRACE-FORMAT.md for the schema.

let helpText = """
metalscope \(MetalscopeVersion.current) — ML-native Metal kernel profiler

USAGE
  metalscope <command> [options]

COMMANDS
  info                         Device, exposed counter sets, and roofline peaks.
    --json                       Machine-readable output (one row of docs/COUNTER-MATRIX.md).

  calibrate                    Measure this chip's real ceilings (MPS GEMM + streaming triad).
    --sizes 1024,2048,3072       GEMM sizes to sweep (square).
    --iterations N               Fixed iterations (default: auto-sized to --target-ms).
    --target-ms N                GPU work per run, long enough to reach sustained clocks (250).
    --repeats N                  Runs per workload; the best one wins (default 3).
    --triad-mb 64,128,256        Triad working set per stream, in MB.
    --skip-fp16                  fp32 + bandwidth only.
    --quick                      Small, fast sweep.
    --no-cache                   Don't write ~/.metalscope/peaks.json.
    --peaks-file PATH            Write to this peaks file instead.
    --json                       Emit the measured PeakSet as JSON.

  bench                        Self-test: run a small ML workload under MetalscopeCapture.
    --variant baseline|tuned     Kernel variant (tuned = vectorized elementwise/triad).
    --size N                     GEMM size (default 1024).
    --seq N / --heads N          Attention sequence length / head count.
    --elements-mb N              Elementwise/triad working set per stream in MB.
    --iterations N               Fixed iterations per region (default: auto-sized).
    --target-ms N                Auto-size iterations to ~N ms of GPU work (default 150).
    --repeats N                  Timed runs per kernel after a discarded warm-up (default 5).
                                 The trace records every sample; `report` shows the
                                 median and its spread, and `diff` refuses to call a
                                 winner when two spreads overlap. --repeats 1 records
                                 no spread, and diff then withholds every verdict.
    --output PATH                Trace destination.
    --report                     Print the roofline report afterwards.

  profile -- <command> [args]  Run an instrumented child, then report its trace.
    --output PATH                Trace path handed to the child ($METALSCOPE_TRACE).
    --no-report                  Just collect the trace.

  report <trace.json>          Roofline table: median duration and its spread, GFLOP/s,
                               AI, bound, efficiency, and static occupancy where the
                               trace has it.
    --sort duration|efficiency|intensity
    --occupancy                  Per-kernel static occupancy detail block.
    --spec-peaks                 Force spec-sheet peaks instead of measured.
    --peaks-file PATH            Use peaks from this file.
    --json                       Machine-readable output.

  diff <baseline.json> <candidate.json>
                               Align kernels by label + shape and show what moved.
                               Compares medians; withholds a verdict when the two
                               traces' min-p95 spreads overlap.
    --json                       Machine-readable output.

  help, version

Peaks are measured, not asserted: run `calibrate` once per machine. Until then,
metalscope labels its numbers "spec-sheet folklore" and you should not trust
efficiency percentages.
"""

func runMain() -> Int32 {
    var argv = Array(CommandLine.arguments.dropFirst())
    let command = argv.first ?? "info"
    if !argv.isEmpty { argv.removeFirst() }

    do {
        switch command {
        case "info":
            try InfoCommand.run(try Arguments.parse(argv, valueOptions: []))
        case "calibrate":
            try CalibrateCommand.run(try Arguments.parse(argv, valueOptions: CalibrateCommand.valueOptions))
        case "bench":
            try BenchCommand.run(try Arguments.parse(argv, valueOptions: BenchCommand.valueOptions))
        case "profile":
            try ProfileCommand.run(try Arguments.parse(argv, valueOptions: ProfileCommand.valueOptions))
        case "report":
            try ReportCommand.run(try Arguments.parse(argv, valueOptions: ReportCommand.valueOptions))
        case "diff":
            try DiffCommand.run(try Arguments.parse(argv, valueOptions: DiffCommand.valueOptions))
        case "help", "--help", "-h":
            Terminal.out(helpText)
        case "version", "--version":
            Terminal.out("metalscope \(MetalscopeVersion.current)")
        default:
            if command.hasPrefix("-") {
                // `metalscope --json` etc. still means info, as the scaffold did.
                try InfoCommand.run(try Arguments.parse([command] + argv, valueOptions: []))
            } else {
                throw CLIError.unknownCommand(command)
            }
        }
        return 0
    } catch let error as CLIError {
        Terminal.error("metalscope: \(error.description)")
        return 2
    } catch let error as CustomStringConvertible {
        Terminal.error("metalscope: \(error.description)")
        return 1
    } catch {
        Terminal.error("metalscope: \(error.localizedDescription)")
        return 1
    }
}

exit(runMain())
