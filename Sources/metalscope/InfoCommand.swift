import Foundation
import Metal
import MetalscopeCapture
import MetalscopeCore

enum InfoCommand {
    static let known: Set<String> = ["json"]

    static func run(_ args: Arguments) throws {
        try args.rejectUnknown(known)
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw CLIError.message("no Metal device found")
        }
        let capabilities = CaptureCapabilities(device: device)
        let store = PeaksStore.default
        let peaks = store.resolve(for: device.name)

        if args.has("json") {
            let payload = InfoPayload(
                device: DeviceInfo(name: device.name,
                                   registryID: device.registryID,
                                   maxWorkingSetBytes: device.recommendedMaxWorkingSetSize,
                                   counterSets: capabilities.counterSetNames,
                                   supportsStageBoundarySampling: capabilities.supportsStageBoundarySampling,
                                   maxThreadgroupMemoryBytes: capabilities.maxThreadgroupMemoryBytes),
                peaks: peaks,
                peaksSource: peaks?.source.label ?? "none",
                peaksPath: store.url.path,
                samplingPoints: samplingPoints(capabilities),
                counterSetDetail: capabilities.counterSets.map {
                    CounterSetPayload(name: $0.name, counters: $0.counterNames, resolver: $0.hasResolver)
                },
                knownCounterSetsAbsentHere: capabilities.absentKnownCounterSetNames,
                timingLadderTier: capabilities.timingLadderTier.rawValue,
                maxThreadsPerThreadgroup: [capabilities.maxThreadsPerThreadgroup.width,
                                           capabilities.maxThreadsPerThreadgroup.height,
                                           capabilities.maxThreadsPerThreadgroup.depth],
                osVersion: ProcessInfo.processInfo.operatingSystemVersionString,
                toolVersion: MetalscopeVersion.current)
            let data = try TraceIO.makeEncoder().encode(payload)
            Terminal.out(String(data: data, encoding: .utf8) ?? "{}")
            return
        }

        Terminal.out("metalscope \(MetalscopeVersion.current)")
        Terminal.out("  device:            \(device.name)")
        Terminal.out("  unified memory:    \(device.recommendedMaxWorkingSetSize / (1 << 30)) GB working set")
        Terminal.out("  max threadgroup:   \(device.maxThreadsPerThreadgroup.width)x\(device.maxThreadsPerThreadgroup.height)x\(device.maxThreadsPerThreadgroup.depth)")
        Terminal.out("  threadgroup mem:   \(Fmt.bytes(Double(capabilities.maxThreadgroupMemoryBytes))) max per threadgroup")
        Terminal.out("  sampling points:   \(samplingPoints(capabilities).joined(separator: ", "))")
        let timing = capabilities.canSampleEncoderStages
            ? "per-encoder GPU timestamps"
            : "command-buffer GPU time (no stage-boundary sampling here)"
        Terminal.out("  kernel timing:     \(timing) [\(capabilities.timingLadderTier.displayName)]")

        // Per-set detail, because "which counters exist here" is the question the
        // per-chip matrix in docs/COUNTER-MATRIX.md is built from.
        Terminal.out("")
        if capabilities.counterSets.isEmpty {
            Terminal.out("  counter sets:      (none exposed)")
        } else {
            Terminal.out("  counter sets exposed by this device")
            var table = TextTable(columns: [.init("set"), .init("counters"), .init("metalscope decodes")])
            for set in capabilities.counterSets {
                table.addRow([set.name,
                              set.counterNames.isEmpty ? "(none listed)" : set.counterNames.joined(separator: ", "),
                              set.hasResolver ? "yes" : "no — resolver missing"])
            }
            Terminal.out(table.rendered(indent: "    "))
        }
        let absent = capabilities.absentKnownCounterSetNames
        if !absent.isEmpty {
            Terminal.out("")
            Terminal.out("  not exposed here:  \(absent.joined(separator: ", "))")
            Terminal.out("                     (resolvers are in place; a chip that offers them needs no code change)")
        }

        if let peaks {
            let when = peaks.measuredAt.map { " (" + ISO8601DateFormatter().string(from: $0) + ")" } ?? ""
            Terminal.out("")
            Terminal.out("  roofline peaks [\(peaks.source.label)]\(when)")
            var table = TextTable(columns: [.init("", .left), .init("value", .right)])
            table.addRow(["fp32 compute", Fmt.gflops(peaks.fp32GFLOPS)])
            table.addRow(["fp16 compute", peaks.fp16GFLOPS.map(Fmt.gflops) ?? "not measured"])
            table.addRow(["memory bandwidth", Fmt.bandwidth(peaks.bandwidthGBs)])
            table.addRow(["ridge point (fp32)", Fmt.intensity(peaks.ridgePoint(for: .fp32)) + " FLOP/byte"])
            if peaks.fp16GFLOPS != nil {
                table.addRow(["ridge point (fp16)", Fmt.intensity(peaks.ridgePoint(for: .fp16)) + " FLOP/byte"])
            }
            Terminal.out(table.rendered(indent: "    "))
            if peaks.source == .measured {
                Terminal.out("    source: \(store.url.path)")
            } else {
                Terminal.out("    these are community spec-sheet numbers, not your chip — run `metalscope calibrate`")
            }
        } else {
            Terminal.out("  roofline peaks:    unknown chip — run `metalscope calibrate`")
        }
    }

    private static func samplingPoints(_ capabilities: CaptureCapabilities) -> [String] {
        var points: [String] = []
        if capabilities.supportsStageBoundarySampling { points.append("stage") }
        if capabilities.supportsDispatchBoundarySampling { points.append("dispatch") }
        if capabilities.supportsBlitBoundarySampling { points.append("blit") }
        return points.isEmpty ? ["(none)"] : points
    }

    private struct CounterSetPayload: Encodable {
        var name: String
        var counters: [String]
        var resolver: Bool
    }

    /// The payload docs/COUNTER-MATRIX.md asks contributors to paste: everything
    /// needed to fill one row of the per-chip table.
    private struct InfoPayload: Encodable {
        var device: DeviceInfo
        var peaks: PeakSet?
        var peaksSource: String
        var peaksPath: String
        var samplingPoints: [String]
        var counterSetDetail: [CounterSetPayload]
        var knownCounterSetsAbsentHere: [String]
        var timingLadderTier: String
        var maxThreadsPerThreadgroup: [Int]
        var osVersion: String
        var toolVersion: String
    }
}
