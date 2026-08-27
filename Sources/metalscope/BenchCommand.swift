import Foundation
import Metal
import MetalscopeCapture
import MetalscopeCore

/// Dev/self-test command: run a small ML-shaped workload under
/// `MetalscopeCapture` and write a real trace. This is what exercises the
/// capture path end-to-end on whatever chip you're on, and it gives `report`
/// and `diff` something real to chew on.
enum BenchCommand {
    static let known: Set<String> = ["output", "variant", "size", "iterations", "seq", "heads",
                                     "elements-mb", "target-ms", "repeats", "quiet", "report"]
    static let valueOptions: Set<String> = ["output", "variant", "size", "iterations", "seq",
                                            "heads", "elements-mb", "target-ms", "repeats"]

    /// Five timed runs per kernel. Enough that the median is not one sample's
    /// opinion and the nearest-rank p95 is the slowest of the five, while still
    /// keeping `bench` under ten seconds on an M1 Pro.
    static let defaultRepeats = 5

    static func run(_ args: Arguments) throws {
        try args.rejectUnknown(known)
        let variantName = args.string("variant") ?? "baseline"
        guard let variant = Variant(rawValue: variantName) else {
            throw CLIError.badValue("variant", variantName, "expected 'baseline' or 'tuned'")
        }
        let size = try args.int("size", default: 1024)
        let explicitIterations = args.string("iterations") != nil
            ? try args.int("iterations", default: 0) : nil
        let targetSeconds = try args.double("target-ms", default: 150) / 1000
        let seq = try args.int("seq", default: 512)
        let heads = try args.int("heads", default: 8)
        let elementsMB = try args.int("elements-mb", default: 64)
        let repeats = try args.int("repeats", default: defaultRepeats)
        guard repeats >= 1 else {
            throw CLIError.badValue("repeats", "\(repeats)", "expected at least 1")
        }
        let quiet = args.has("quiet")
        let outputPath = args.string("output")
            ?? CaptureSession.environmentTraceURL?.path
            ?? "metalscope-bench-\(variant.rawValue).json"
        let output = URL(fileURLWithPath: (outputPath as NSString).expandingTildeInPath)

        let session = try CaptureSession()
        let workloads = try Workloads(session: session)

        func log(_ text: String) { if !quiet { Terminal.out(text) } }

        log("metalscope bench [\(variant.rawValue)] — \(session.device.name)")
        log("  timing: \(session.capabilities.canSampleEncoderStages ? "encoder stage timestamps + command-buffer GPU time" : "command-buffer GPU time")")
        log("  repeats: \(repeats) timed run\(repeats == 1 ? "" : "s") per kernel"
            + (repeats > 1 ? ", after a discarded warm-up run" : " — no spread will be recorded"))

        let elements = elementsMB * 1024 * 1024 / MemoryLayout<Float>.size

        /// Warm up and size the run through the same path a real user would,
        /// then measure. Iteration counts are auto-sized so the GPU reaches its
        /// sustained clock — an 8-iteration GEMM measures the ramp, not the chip.
        func capture(_ runner: Workloads.Runner, label: String) throws {
            let iterations = try explicitIterations
                ?? workloads.autoIterations(runner, label: label, targetSeconds: targetSeconds)
            try workloads.run(runner, label: label, iterations: iterations, repeats: repeats)
        }

        try capture(try workloads.makeGEMM(m: size, n: size, k: size, precision: .fp32),
                    label: "ffn.gemm")
        do {
            try capture(try workloads.makeGEMM(m: size, n: size, k: size, precision: .fp16),
                        label: "ffn.gemm.half")
        } catch {
            log("  fp16 GEMM skipped: \(error)")
        }
        try capture(try workloads.makeAttention(b: 1, h: heads, s: seq, d: 64), label: "attn.sdpa")
        try capture(try workloads.makeRMSNorm(rows: elements / 1024, width: 1024),
                    label: "block.rmsnorm")
        // The baseline elementwise kernel is suboptimal in two independent ways:
        // scalar loads, and a threadgroup that isn't a multiple of the SIMD width.
        // The second one exists so `report`'s occupancy analysis and `diff`'s
        // occupancy column always have something real to show on any chip.
        try capture(try workloads.makeScale(elements: elements,
                                            vectorized: variant == .tuned,
                                            threadgroupWidth: variant == .tuned
                                                ? nil : raggedThreadgroupWidth),
                    label: "act.scale")
        try capture(try workloads.makeTriad(elements: elements,
                                            variant: variant == .tuned ? .vec4 : .scalar),
                    label: "stream.triad")

        let peaks = PeaksStore.default.resolve(for: session.device.name)
        let url = try session.writeTrace(to: output, peaks: peaks,
                                         notes: ["command": "bench", "variant": variant.rawValue,
                                                 "repeats": "\(repeats)"])
        log("  captured \(session.records.count) kernels -> \(url.path)")
        if peaks?.source != .measured {
            log("  note: peaks are spec-sheet folklore — run `metalscope calibrate` for honest efficiency numbers.")
        }

        if args.has("report") {
            log("")
            try ReportCommand.printReport(trace: session.makeTrace(peaks: peaks), path: url.path)
        }
    }

    /// 100 threads per threadgroup: not a multiple of any plausible SIMD width
    /// (32 on every Apple GPU so far), so the hardware launches 4 SIMD groups and
    /// idles 28 lanes in every single threadgroup. Deliberate — it gives the
    /// occupancy analysis a genuine defect to find in the baseline trace, and the
    /// `tuned` variant fixes it.
    static let raggedThreadgroupWidth = 100

    enum Variant: String {
        case baseline
        case tuned
    }
}
