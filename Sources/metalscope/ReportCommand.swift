import Foundation
import Metal
import MetalscopeCore

/// Roofline report for a captured trace.
enum ReportCommand {
    static let known: Set<String> = ["peaks-file", "spec-peaks", "sort", "json", "occupancy"]
    static let valueOptions: Set<String> = ["peaks-file", "sort"]

    enum SortOrder: String {
        case duration, efficiency, intensity

        static func parse(_ raw: String?) throws -> SortOrder? {
            guard let raw else { return nil }
            guard let order = SortOrder(rawValue: raw) else {
                throw CLIError.badValue("sort", raw, "expected duration|efficiency|intensity")
            }
            return order
        }
    }

    static func run(_ args: Arguments) throws {
        try args.rejectUnknown(known)
        guard let path = args.positionals.first else {
            throw CLIError.missingArgument("<trace.json>")
        }
        let url = URL(fileURLWithPath: (path as NSString).expandingTildeInPath)
        let trace = try TraceIO.read(from: url)
        let peaks = try resolvePeaks(for: trace, args: args)
        // Validate before any output, so a typo doesn't half-print a report.
        let sort = try SortOrder.parse(args.string("sort"))

        if args.has("json") {
            let payload = trace.kernels.map { record -> JSONRow in
                let placement = Roofline.place(record, peaks: peaks)
                return JSONRow(label: record.label,
                               shape: record.shape.descriptionText,
                               precision: record.precision.rawValue,
                               durationSeconds: record.durationSeconds,
                               achievedGFLOPS: placement.achievedGFLOPS,
                               achievedBandwidthGBs: placement.achievedBandwidthGBs,
                               arithmeticIntensity: placement.arithmeticIntensity,
                               ceilingGFLOPS: placement.ceilingGFLOPS,
                               bound: placement.bound.rawValue,
                               efficiency: placement.efficiency,
                               timingSource: record.timingSource.rawValue,
                               occupancy: record.occupancy.map(JSONOccupancy.init),
                               counters: record.counters)
            }
            let data = try TraceIO.makeEncoder().encode(JSONReport(peaks: peaks, kernels: payload))
            Terminal.out(String(data: data, encoding: .utf8) ?? "{}")
            return
        }

        try printReport(trace: trace, path: url.path, peaks: peaks, sort: sort,
                        occupancyDetail: args.has("occupancy"))
    }

    /// Resolve which peaks to score against: an explicit file, the trace's own
    /// measured peaks, the local cache, or (last) spec-sheet folklore.
    static func resolvePeaks(for trace: Trace, args: Arguments) throws -> PeakSet {
        if let path = args.string("peaks-file") {
            let store = PeaksStore(url: URL(fileURLWithPath: (path as NSString).expandingTildeInPath))
            if let peaks = store.resolve(for: trace.device.name) { return peaks }
            throw CLIError.message("no peaks for '\(trace.device.name)' in \(path)")
        }
        if args.has("spec-peaks") {
            guard let folklore = ChipPeaks.folklore(for: trace.device.name)?.peakSet else {
                throw CLIError.message("no spec-sheet entry for '\(trace.device.name)'")
            }
            return folklore
        }
        if let traced = trace.peaks, traced.source == .measured { return traced }
        if let local = PeaksStore.default.resolve(for: trace.device.name) { return local }
        if let traced = trace.peaks { return traced }
        throw CLIError.message("no peaks available for '\(trace.device.name)' — run `metalscope calibrate`")
    }

    static func printReport(trace: Trace,
                            path: String,
                            peaks: PeakSet? = nil,
                            sort: SortOrder? = nil,
                            occupancyDetail: Bool = false) throws {
        let peaks = try peaks ?? resolvePeaks(for: trace, args: Arguments())
        Terminal.out("roofline report — \(path)")
        Terminal.out("  device:  \(trace.device.name)   captured \(ISO8601DateFormatter().string(from: trace.createdAt))")
        Terminal.out(String(format: "  peaks:   %@ fp32 / %@ fp16 / %@  [%@]",
                            Fmt.gflops(peaks.fp32GFLOPS),
                            peaks.fp16GFLOPS.map(Fmt.gflops) ?? "n/a",
                            Fmt.bandwidth(peaks.bandwidthGBs),
                            peaks.source.label))
        Terminal.out(String(format: "  ridge:   %@ FLOP/byte (fp32)%@",
                            Fmt.intensity(peaks.ridgePoint(for: .fp32)),
                            peaks.fp16GFLOPS != nil
                                ? ", \(Fmt.intensity(peaks.ridgePoint(for: .fp16))) FLOP/byte (fp16)" : ""))
        Terminal.out("")

        var rows = trace.kernels.enumerated().map { (index: $0.offset, record: $0.element) }
        switch sort {
        case .duration: rows.sort { $0.record.durationSeconds > $1.record.durationSeconds }
        case .efficiency:
            // Worst first: that's the list you actually want to work through.
            rows.sort {
                Roofline.place($0.record, peaks: peaks).efficiency
                    < Roofline.place($1.record, peaks: peaks).efficiency
            }
        case .intensity: rows.sort { $0.record.arithmeticIntensity > $1.record.arithmeticIntensity }
        case nil: break
        }

        // The occupancy columns only earn their width when something in the trace
        // carries occupancy data — a v1 trace, or one made entirely of MPS
        // regions, has none.
        let showsOccupancy = trace.kernels.contains { $0.occupancy != nil }

        var columns: [TextTable.Column] = [
            .init("kernel"),
            .init("shape"),
            .init("prec"),
            .init("time/iter", .right),
            .init("GFLOP/s", .right),
            .init("GB/s", .right),
            .init("AI", .right),
            .init("bound"),
            .init("ceiling", .right),
            .init("eff", .right),
        ]
        if showsOccupancy {
            columns.append(.init("tgroup", .right))
            columns.append(.init("occ", .right))
        }
        columns.append(.init("timing"))
        var table = TextTable(columns: columns)

        for row in rows {
            let record = row.record
            let placement = Roofline.place(record, peaks: peaks)
            var cells = [
                record.label,
                record.shape.descriptionText,
                record.precision.rawValue,
                Fmt.duration(record.durationSeconds),
                Fmt.gflops(placement.achievedGFLOPS),
                String(format: "%.1f", placement.achievedBandwidthGBs),
                Fmt.intensity(placement.arithmeticIntensity),
                placement.bound.displayName,
                Fmt.gflops(placement.ceilingGFLOPS),
                Fmt.percent(placement.efficiency),
            ]
            if showsOccupancy {
                cells.append(record.occupancy?.tableText ?? "-")
                cells.append(record.occupancy.map { Fmt.percent($0.threadgroupOccupancy) } ?? "-")
            }
            cells.append(record.timingSource.displayName)
            table.addRow(cells)
        }
        Terminal.out(table.rendered(indent: "  "))
        Terminal.out("")
        Terminal.out("  AI = analytic FLOPs / compulsory bytes. eff = achieved / roofline ceiling at that AI.")
        if showsOccupancy {
            Terminal.out("  tgroup = threads/threadgroup (* = not a multiple of the SIMD width). occ = that vs the pipeline's max.")
            Terminal.out("  a dash means metalscope never saw the pipeline (MPS and friends encode their own dispatches).")
        }
        if peaks.source != .measured {
            Terminal.out("  peaks are \(peaks.source.label) — run `metalscope calibrate` before trusting eff%.")
        }

        var notable = rows.compactMap { row -> String? in
            let record = row.record
            let placement = Roofline.place(record, peaks: peaks)
            guard placement.efficiency < 0.4, record.durationSeconds > 0 else { return nil }
            let ceiling = placement.bound == .compute ? "compute ceiling" : "bandwidth ceiling"
            return String(format: "  - %@ is %@-bound at %@ of the %@%@",
                          record.label, placement.bound.displayName,
                          Fmt.percent(placement.efficiency), ceiling,
                          record.notes?["fusion"].map { " (\($0))" } ?? "")
        }
        // Occupancy hints sit in the same section: both answer "what should I
        // look at next", and a ragged threadgroup is often why a kernel that
        // looks fine on the roofline still leaves time on the table.
        notable += rows.compactMap { row -> String? in
            guard let occupancy = row.record.occupancy, let hint = occupancy.hint else { return nil }
            // A kernel already sitting on its roofline ceiling has no time to win
            // back, whatever its threadgroup looks like. Say so, rather than
            // sending someone off to fix a shape that isn't costing them anything
            // — a ragged threadgroup in a bandwidth-saturated kernel wastes lanes
            // that were going to be waiting on DRAM regardless.
            let placement = Roofline.place(row.record, peaks: peaks)
            let caveat = placement.efficiency >= 0.9
                ? String(format: "; note it is already at %@ of its %@ ceiling, so this costs lanes, not time",
                         Fmt.percent(placement.efficiency), placement.bound.displayName)
                : ""
            return "  - \(row.record.label): \(hint)\(caveat)"
        }
        if !notable.isEmpty {
            Terminal.out("")
            Terminal.out("  headroom:")
            notable.forEach(Terminal.out)
        }

        if occupancyDetail {
            printOccupancyDetail(rows.map(\.record))
        }
    }

    /// Per-kernel static occupancy block (`report --occupancy`).
    static func printOccupancyDetail(_ records: [KernelRecord]) {
        Terminal.out("")
        Terminal.out("  occupancy detail (static, from MTLComputePipelineState — no counters involved):")
        var printedAny = false
        for record in records {
            guard let o = record.occupancy else { continue }
            printedAny = true
            Terminal.out("")
            Terminal.out("  \(record.label)")
            /// A fixed label column reads better here than a table: the limiter
            /// line is a sentence, and a rule sized to it would dwarf the block.
            func line(_ label: String, _ value: String) {
                Terminal.out("    " + label.padding(toLength: 17, withPad: " ", startingAt: 0) + value)
            }
            line("threadgroup",
                 "\(o.threadsPerThreadgroup) threads of a \(o.maxTotalThreadsPerThreadgroup) max"
                    + String(format: " (%@ occupancy)", Fmt.percent(o.threadgroupOccupancy)))
            line("simd groups",
                 "\(o.simdGroupsPerThreadgroup) x \(o.threadExecutionWidth) lanes"
                    + (o.idleLanesPerThreadgroup > 0
                       ? ", \(o.idleLanesPerThreadgroup) idle (\(Fmt.percent(o.laneUtilization)) lane use)"
                       : ", fully packed"))
            let limitText = o.threadgroupMemoryLimitBytes.map { " of \(Fmt.bytes(Double($0)))" } ?? ""
            var memoryText = Fmt.bytes(Double(o.threadgroupMemoryBytes)) + limitText
            if let pressure = o.threadgroupMemoryPressure, o.threadgroupMemoryBytes > 0 {
                memoryText += String(format: " (%@)", Fmt.percent(pressure))
            }
            if let resident = o.memoryLimitedThreadgroups {
                memoryText += " — at most \(resident) fit that limit"
            }
            line("threadgroup mem", memoryText)
            let grid = o.threadgroupsPerGrid.map { "\($0) threadgroups/dispatch, " } ?? ""
            line("dispatches",
                 grid + "\(o.dispatchCount) encoded"
                    + (o.variantCount > 1 ? " across \(o.variantCount) shapes (worst shown)" : ""))
            line("limiter", o.hint ?? "nothing structural stands out")
        }
        if !printedAny {
            Terminal.out("  (no kernel in this trace carries occupancy data)")
        }
    }

    private struct JSONRow: Encodable {
        var label: String
        var shape: String
        var precision: String
        var durationSeconds: Double
        var achievedGFLOPS: Double
        var achievedBandwidthGBs: Double
        var arithmeticIntensity: Double
        var ceilingGFLOPS: Double
        var bound: String
        var efficiency: Double
        var timingSource: String
        var occupancy: JSONOccupancy?
        var counters: [String: Double]?
    }

    /// Occupancy for machine consumers: the stored inputs *and* the derived
    /// numbers, so an autotuner reading this doesn't re-implement the math.
    struct JSONOccupancy: Encodable {
        var threadsPerThreadgroup: Int
        var maxTotalThreadsPerThreadgroup: Int
        var threadExecutionWidth: Int
        var simdGroupsPerThreadgroup: Int
        var threadgroupOccupancy: Double
        var laneUtilization: Double
        var idleLanesPerThreadgroup: Int
        var executionWidthAligned: Bool
        var threadgroupMemoryBytes: Int
        var threadgroupMemoryPressure: Double?
        var threadgroupsPerGrid: Int?
        var dispatchCount: Int
        var variantCount: Int
        var limiter: String
        var hint: String?

        init(_ o: OccupancyInfo) {
            threadsPerThreadgroup = o.threadsPerThreadgroup
            maxTotalThreadsPerThreadgroup = o.maxTotalThreadsPerThreadgroup
            threadExecutionWidth = o.threadExecutionWidth
            simdGroupsPerThreadgroup = o.simdGroupsPerThreadgroup
            threadgroupOccupancy = o.threadgroupOccupancy
            laneUtilization = o.laneUtilization
            idleLanesPerThreadgroup = o.idleLanesPerThreadgroup
            executionWidthAligned = o.isExecutionWidthAligned
            threadgroupMemoryBytes = o.threadgroupMemoryBytes
            threadgroupMemoryPressure = o.threadgroupMemoryPressure
            threadgroupsPerGrid = o.threadgroupsPerGrid
            dispatchCount = o.dispatchCount
            variantCount = o.variantCount
            limiter = o.limiter.rawValue
            hint = o.hint
        }
    }

    private struct JSONReport: Encodable {
        var peaks: PeakSet
        var kernels: [JSONRow]
    }
}
