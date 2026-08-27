import Foundation
import MetalscopeCore

/// Side-by-side of two traces: where the time and the roofline placement moved.
enum DiffCommand {
    static let known: Set<String> = ["peaks-file", "spec-peaks", "json", "all"]
    static let valueOptions: Set<String> = ["peaks-file"]

    static func run(_ args: Arguments) throws {
        try args.rejectUnknown(known)
        guard args.positionals.count >= 2 else {
            throw CLIError.missingArgument("<baseline.json> <candidate.json>")
        }
        let baselineURL = URL(fileURLWithPath: (args.positionals[0] as NSString).expandingTildeInPath)
        let candidateURL = URL(fileURLWithPath: (args.positionals[1] as NSString).expandingTildeInPath)
        let baseline = try TraceIO.read(from: baselineURL)
        let candidate = try TraceIO.read(from: candidateURL)

        if baseline.device.name != candidate.device.name {
            Terminal.error("warning: traces are from different devices (\(baseline.device.name) vs \(candidate.device.name)) — efficiency comparison is meaningless")
        }

        let peaks = try ReportCommand.resolvePeaks(for: candidate, args: args)
        let diff = TraceDiff(baselineTrace: baseline, candidateTrace: candidate)

        if args.has("json") {
            let rows = diff.entries.map { entry -> JSONRow in
                let placements = entry.placements(peaks: peaks)
                return JSONRow(label: entry.label,
                               shape: entry.shape.descriptionText,
                               precision: entry.precision.rawValue,
                               status: entry.status.rawValue,
                               baselineSeconds: entry.baseline?.durationSeconds,
                               candidateSeconds: entry.candidate?.durationSeconds,
                               durationDeltaFraction: entry.durationDeltaFraction,
                               speedup: entry.speedup,
                               baselineEfficiency: placements.baseline?.efficiency,
                               candidateEfficiency: placements.candidate?.efficiency,
                               efficiencyDeltaPoints: entry.efficiencyDeltaPoints(peaks: peaks),
                               baselineBound: placements.baseline?.bound.rawValue,
                               candidateBound: placements.candidate?.bound.rawValue,
                               baselineThreadsPerThreadgroup: entry.baseline?.occupancy?.threadsPerThreadgroup,
                               candidateThreadsPerThreadgroup: entry.candidate?.occupancy?.threadsPerThreadgroup,
                               baselineThreadgroupOccupancy: entry.baseline?.occupancy?.threadgroupOccupancy,
                               candidateThreadgroupOccupancy: entry.candidate?.occupancy?.threadgroupOccupancy,
                               occupancyDeltaPoints: entry.occupancyDeltaPoints,
                               baselineOccupancyLimiter: entry.baseline?.occupancy?.limiter.rawValue,
                               candidateOccupancyLimiter: entry.candidate?.occupancy?.limiter.rawValue)
            }
            let data = try TraceIO.makeEncoder().encode(JSONDiff(peaks: peaks, kernels: rows))
            Terminal.out(String(data: data, encoding: .utf8) ?? "{}")
            return
        }

        Terminal.out("diff — baseline \(baselineURL.lastPathComponent)  ->  candidate \(candidateURL.lastPathComponent)")
        Terminal.out("  device:  \(candidate.device.name)")
        Terminal.out(String(format: "  peaks:   %@ fp32 / %@  [%@]",
                            Fmt.gflops(peaks.fp32GFLOPS),
                            Fmt.bandwidth(peaks.bandwidthGBs),
                            peaks.source.label))
        if let a = baseline.notes?["variant"], let b = candidate.notes?["variant"] {
            Terminal.out("  variant: \(a) -> \(b)")
        }
        Terminal.out("")

        // Only widen the table when at least one aligned kernel has occupancy on
        // one side — diffs of v1 traces keep the output they always had.
        let showsOccupancy = diff.entries.contains { $0.hasOccupancy }

        var columns: [TextTable.Column] = [
            .init("kernel"),
            .init("shape"),
            .init("baseline", .right),
            .init("candidate", .right),
            .init("delta", .right),
            .init("speedup", .right),
            .init("eff a->b", .right),
            .init("d eff", .right),
        ]
        if showsOccupancy { columns.append(.init("tgroup a->b", .right)) }
        columns.append(.init("bound"))
        var table = TextTable(columns: columns)

        /// "100* -> 256", or a single side when only one trace has the data.
        func threadgroupText(_ entry: DiffEntry) -> String {
            let a = entry.baseline?.occupancy?.tableText
            let b = entry.candidate?.occupancy?.tableText
            switch (a, b) {
            case let (.some(a), .some(b)): return a == b ? a : "\(a) -> \(b)"
            case let (.some(a), .none): return "\(a) -> -"
            case let (.none, .some(b)): return "- -> \(b)"
            case (.none, .none): return "-"
            }
        }

        for entry in diff.entries {
            let placements = entry.placements(peaks: peaks)
            var cells: [String]
            switch entry.status {
            case .matched:
                let boundText: String
                if let change = entry.boundChange(peaks: peaks) {
                    boundText = "\(change.from.displayName) -> \(change.to.displayName)"
                } else {
                    boundText = placements.candidate?.bound.displayName ?? "-"
                }
                cells = [
                    entry.label,
                    entry.shape.descriptionText,
                    Fmt.duration(entry.baseline?.durationSeconds ?? 0),
                    Fmt.duration(entry.candidate?.durationSeconds ?? 0),
                    entry.durationDeltaFraction.map(Fmt.signedPercent) ?? "-",
                    entry.speedup.map { String(format: "%.2fx", $0) } ?? "-",
                    String(format: "%@ -> %@",
                           placements.baseline.map { Fmt.percent($0.efficiency) } ?? "-",
                           placements.candidate.map { Fmt.percent($0.efficiency) } ?? "-"),
                    entry.efficiencyDeltaPoints(peaks: peaks).map(Fmt.signedPoints) ?? "-",
                ]
                if showsOccupancy { cells.append(threadgroupText(entry)) }
                cells.append(boundText)
            case .onlyInBaseline:
                cells = [entry.label, entry.shape.descriptionText,
                         Fmt.duration(entry.baseline?.durationSeconds ?? 0),
                         "absent", "-", "-", "-", "-"]
                if showsOccupancy { cells.append(threadgroupText(entry)) }
                cells.append(placements.baseline?.bound.displayName ?? "-")
            case .onlyInCandidate:
                cells = [entry.label, entry.shape.descriptionText, "absent",
                         Fmt.duration(entry.candidate?.durationSeconds ?? 0),
                         "-", "-", "-", "-"]
                if showsOccupancy { cells.append(threadgroupText(entry)) }
                cells.append(placements.candidate?.bound.displayName ?? "-")
            }
            table.addRow(cells)
        }
        Terminal.out(table.rendered(indent: "  "))

        let matched = diff.matched
        if !matched.isEmpty {
            let baselineTotal = matched.compactMap { $0.baseline?.durationSeconds }.reduce(0, +)
            let candidateTotal = matched.compactMap { $0.candidate?.durationSeconds }.reduce(0, +)
            let delta = baselineTotal > 0 ? (candidateTotal - baselineTotal) / baselineTotal : 0
            Terminal.out("")
            Terminal.out(String(format: "  matched %d/%d kernels — total %@ -> %@ (%@)",
                                matched.count, diff.entries.count,
                                Fmt.duration(baselineTotal), Fmt.duration(candidateTotal),
                                Fmt.signedPercent(delta)))
            let moved = matched.compactMap { entry -> String? in
                guard let change = entry.boundChange(peaks: peaks) else { return nil }
                return "  - \(entry.label): \(change.from.displayName) -> \(change.to.displayName)"
            }
            if !moved.isEmpty {
                Terminal.out("  bound changes:")
                moved.forEach(Terminal.out)
            }
            let occupancyMoves = matched.compactMap { entry -> String? in
                guard let o = entry.occupancies else { return nil }
                let sizeMoved = o.baseline.threadsPerThreadgroup != o.candidate.threadsPerThreadgroup
                guard sizeMoved || entry.limiterChange != nil else { return nil }
                var text = "  - \(entry.label): \(o.baseline.threadsPerThreadgroup) -> \(o.candidate.threadsPerThreadgroup) threads/threadgroup"
                if let points = entry.occupancyDeltaPoints, sizeMoved {
                    text += " (\(Fmt.signedPoints(points)) of pipeline max)"
                }
                if let change = entry.limiterChange {
                    text += ", limiter \(change.from.rawValue) -> \(change.to.rawValue)"
                }
                return text
            }
            if !occupancyMoves.isEmpty {
                Terminal.out("  occupancy changes:")
                occupancyMoves.forEach(Terminal.out)
            }
        }
        if !diff.unmatched.isEmpty {
            Terminal.out("  \(diff.unmatched.count) kernel(s) present in only one trace (aligned by label + shape + precision)")
        }
    }

    private struct JSONRow: Encodable {
        var label: String
        var shape: String
        var precision: String
        var status: String
        var baselineSeconds: Double?
        var candidateSeconds: Double?
        var durationDeltaFraction: Double?
        var speedup: Double?
        var baselineEfficiency: Double?
        var candidateEfficiency: Double?
        var efficiencyDeltaPoints: Double?
        var baselineBound: String?
        var candidateBound: String?
        var baselineThreadsPerThreadgroup: Int?
        var candidateThreadsPerThreadgroup: Int?
        var baselineThreadgroupOccupancy: Double?
        var candidateThreadgroupOccupancy: Double?
        var occupancyDeltaPoints: Double?
        var baselineOccupancyLimiter: String?
        var candidateOccupancyLimiter: String?
    }

    private struct JSONDiff: Encodable {
        var peaks: PeakSet
        var kernels: [JSONRow]
    }
}
