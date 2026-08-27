import Foundation
import Metal
import MetalscopeCapture
import MetalscopeCore

/// Measure the local chip's real ceilings. Spec-sheet peaks are folklore; every
/// efficiency percentage metalscope prints should be against these numbers.
///
/// Method: sweep several GEMM sizes through MPS and several triad sizes through
/// a hand-written kernel, run each long enough for the GPU to reach its
/// sustained clock, repeat, and keep the best. Best-of-repeats (rather than a
/// mean) is deliberate — the ceiling is what the hardware *can* do; thermal and
/// scheduling noise only ever moves a sample downward.
enum CalibrateCommand {
    static let known: Set<String> = ["sizes", "iterations", "repeats", "target-ms", "triad-mb",
                                     "skip-fp16", "no-cache", "quick", "json", "peaks-file"]
    static let valueOptions: Set<String> = ["sizes", "iterations", "repeats", "target-ms",
                                            "triad-mb", "peaks-file"]

    static func run(_ args: Arguments) throws {
        try args.rejectUnknown(known)
        let quick = args.has("quick")
        let sizes = try args.intList("sizes", default: quick ? [1024, 2048] : [1024, 2048, 3072])
        let repeats = try args.int("repeats", default: quick ? 1 : 3)
        let targetSeconds = try args.double("target-ms", default: quick ? 80 : 250) / 1000
        let triadMB = try args.intList("triad-mb", default: quick ? [128] : [64, 128, 256])
        let explicitIterations = args.string("iterations") != nil
            ? try args.int("iterations", default: 0) : nil
        let skipFP16 = args.has("skip-fp16")
        let jsonOnly = args.has("json")

        let session = try CaptureSession()
        let workloads = try Workloads(session: session)
        let device = session.device

        func log(_ text: String) {
            if !jsonOnly { Terminal.out(text) }
        }

        log("metalscope calibrate — \(device.name)")
        log("  sizes=\(sizes.map(String.init).joined(separator: ",")) repeats=\(repeats) " +
            "target=\(Int(targetSeconds * 1000))ms/run" +
            (explicitIterations.map { " iterations=\($0) (fixed)" } ?? " (iterations auto-sized)"))
        log("")

        var table = TextTable(columns: [.init("workload"), .init("shape"), .init("iters", .right),
                                        .init("time/iter", .right), .init("achieved", .right)])

        /// Run one prepared workload: size the iteration count, repeat, keep the best.
        func best(_ runner: Workloads.Runner,
                  label: String,
                  workloadName: String,
                  shapeText: String,
                  metric: (KernelRecord) -> Double,
                  format: (Double) -> String) throws -> Double {
            let iterations = try explicitIterations
                ?? workloads.autoIterations(runner, label: label, targetSeconds: targetSeconds)
            var bestValue = 0.0
            var bestRecord: KernelRecord?
            for _ in 0..<repeats {
                let record = try workloads.run(runner, label: label, iterations: iterations, keep: false)
                let value = metric(record)
                if value > bestValue { bestValue = value; bestRecord = record }
            }
            table.addRow([workloadName, shapeText, "\(iterations)",
                          Fmt.duration(bestRecord?.durationSeconds ?? 0), format(bestValue)])
            return bestValue
        }

        func gflops(_ record: KernelRecord) -> Double {
            record.durationSeconds > 0 ? record.flops / record.durationSeconds / 1e9 : 0
        }
        func bandwidth(_ record: KernelRecord) -> Double {
            record.durationSeconds > 0 ? record.bytes / record.durationSeconds / 1e9 : 0
        }

        // fp32 compute ceiling.
        var bestFP32 = 0.0
        var bestFP32Size = 0
        for size in sizes {
            let runner = try workloads.makeGEMM(m: size, n: size, k: size, precision: .fp32)
            let value = try best(runner, label: "calibrate.gemm.fp32", workloadName: "gemm fp32",
                                 shapeText: "\(size)^3", metric: gflops, format: Fmt.gflops)
            if value > bestFP32 { bestFP32 = value; bestFP32Size = size }
        }

        // fp16 compute ceiling (same MPS path, half precision).
        var bestFP16: Double? = nil
        var bestFP16Size = 0
        if !skipFP16 {
            for size in sizes {
                do {
                    let runner = try workloads.makeGEMM(m: size, n: size, k: size, precision: .fp16)
                    let value = try best(runner, label: "calibrate.gemm.fp16", workloadName: "gemm fp16",
                                         shapeText: "\(size)^3", metric: gflops, format: Fmt.gflops)
                    if value > (bestFP16 ?? 0) { bestFP16 = value; bestFP16Size = size }
                } catch {
                    log("  fp16 GEMM unavailable (\(error)) — continuing with fp32 only")
                    break
                }
            }
        }

        // Bandwidth ceiling: both triad variants across working-set sizes.
        var bestBandwidth = 0.0
        var bestVariant = ""
        for mb in triadMB {
            let elements = mb * 1024 * 1024 / MemoryLayout<Float>.size
            for variant in [Workloads.TriadVariant.scalar, .vec4] {
                let runner = try workloads.makeTriad(elements: elements, variant: variant)
                let value = try best(runner, label: "calibrate.triad.\(variant.rawValue)",
                                     workloadName: "triad \(variant.rawValue)",
                                     shapeText: "\(mb) MB x3", metric: bandwidth, format: Fmt.bandwidth)
                if value > bestBandwidth {
                    bestBandwidth = value
                    bestVariant = "\(variant.rawValue) @ \(mb) MB"
                }
            }
        }

        log(table.rendered(indent: "  "))
        log("")

        var details: [String: Double] = [
            "gemmBestSize": Double(bestFP32Size),
            "repeats": Double(repeats),
            "targetSecondsPerRun": targetSeconds,
        ]
        if bestFP16 != nil { details["gemmFP16BestSize"] = Double(bestFP16Size) }

        let peaks = PeakSet(source: .measured,
                            chip: device.name,
                            fp32GFLOPS: bestFP32,
                            fp16GFLOPS: bestFP16,
                            bandwidthGBs: bestBandwidth,
                            measuredAt: Date(),
                            details: details)

        if jsonOnly {
            let data = try TraceIO.makeEncoder().encode(peaks)
            Terminal.out(String(data: data, encoding: .utf8) ?? "{}")
        } else {
            Terminal.out("  measured peaks for \(device.name)")
            var summary = TextTable(columns: [.init(""), .init("value", .right), .init("via")])
            summary.addRow(["fp32 compute", Fmt.gflops(bestFP32), "MPS GEMM \(bestFP32Size)^3"])
            if let bestFP16 {
                summary.addRow(["fp16 compute", Fmt.gflops(bestFP16), "MPS GEMM \(bestFP16Size)^3"])
            } else {
                summary.addRow(["fp16 compute", "not measured", "--skip-fp16"])
            }
            summary.addRow(["memory bandwidth", Fmt.bandwidth(bestBandwidth), "triad \(bestVariant)"])
            summary.addRow(["ridge point (fp32)", Fmt.intensity(peaks.ridgePoint(for: .fp32)) + " FLOP/byte", ""])
            if bestFP16 != nil {
                summary.addRow(["ridge point (fp16)", Fmt.intensity(peaks.ridgePoint(for: .fp16)) + " FLOP/byte", ""])
            }
            Terminal.out(summary.rendered(indent: "    "))

            if let folklore = ChipPeaks.folklore(for: device.name) {
                Terminal.out("")
                Terminal.out(String(format: "    vs spec-sheet folklore: fp32 %.0f%% of %.1f TF, bandwidth %.0f%% of %.0f GB/s",
                                    bestFP32 / (folklore.fp32TFLOPS * 1000) * 100, folklore.fp32TFLOPS,
                                    bestBandwidth / folklore.memoryBandwidthGBs * 100, folklore.memoryBandwidthGBs))
                Terminal.out("    (spec-sheet numbers assume FMA issue every cycle at boost clock; real GEMM never gets there)")
            }
        }

        if !args.has("no-cache") {
            let store = args.string("peaks-file").map { PeaksStore(url: URL(fileURLWithPath: $0)) } ?? .default
            let url = try store.save(peaks, for: device.name)
            log("")
            log("  cached to \(url.path) — `metalscope info` and `report` now use these.")
        }
    }
}
