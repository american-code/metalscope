import Foundation

/// One aligned row of a trace comparison.
public struct DiffEntry: Sendable, Equatable {
    public enum Status: String, Sendable {
        case matched
        case onlyInBaseline
        case onlyInCandidate
    }

    public var key: String
    public var label: String
    public var shape: KernelShape
    public var precision: Precision
    /// Index within the group of identically-keyed kernels (0 for the common case).
    public var occurrence: Int
    public var baseline: KernelRecord?
    public var candidate: KernelRecord?

    public var status: Status {
        switch (baseline, candidate) {
        case (.some, .some): return .matched
        case (.some, .none): return .onlyInBaseline
        case (.none, .some): return .onlyInCandidate
        case (.none, .none): return .matched // unreachable by construction
        }
    }

    /// Candidate minus baseline, in seconds. Negative means the candidate is faster.
    public var durationDeltaSeconds: Double? {
        guard let a = baseline, let b = candidate else { return nil }
        return b.durationSeconds - a.durationSeconds
    }

    /// Relative duration change; -0.25 means "25% faster".
    public var durationDeltaFraction: Double? {
        guard let a = baseline, let b = candidate, a.durationSeconds > 0 else { return nil }
        return (b.durationSeconds - a.durationSeconds) / a.durationSeconds
    }

    /// Speedup factor: baseline / candidate. > 1 means the candidate is faster.
    public var speedup: Double? {
        guard let a = baseline, let b = candidate, b.durationSeconds > 0 else { return nil }
        return a.durationSeconds / b.durationSeconds
    }

    // MARK: - Verdict

    /// Whether the median moved by more than the two sides' measurement noise.
    ///
    /// The point of the distinction is that a diff of microsecond-scale kernels
    /// will *always* show a delta, and most of them are the stopwatch rather
    /// than the code.
    public enum Verdict: String, Sendable {
        case faster
        case slower
        /// Medians differ, but the two [min, p95] intervals overlap, so the
        /// difference is inside the noise that produced it.
        case withinNoise = "within-noise"
        /// One or both sides were captured once. A single run has no spread, so
        /// there is nothing to compare a delta against.
        case unmeasured

        /// Table text. "no call" rather than "same": an overlap means the runs
        /// could not be told apart, not that they are equal.
        public var displayName: String {
            switch self {
            case .faster: return "faster"
            case .slower: return "slower"
            case .withinNoise: return "no call"
            case .unmeasured: return "-"
            }
        }
    }

    /// Both sides' run statistics, when both sides were repeated. A single-run
    /// trace on either side leaves this nil rather than presenting one point as
    /// a distribution.
    public var runStatistics: (baseline: RunStatistics, candidate: RunStatistics)? {
        guard let a = baseline?.runStatistics, let b = candidate?.runStatistics,
              a.hasSpread, b.hasSpread else { return nil }
        return (a, b)
    }

    /// nil when either side lacks repeat data.
    public var spreadsOverlap: Bool? {
        runStatistics.map { $0.baseline.overlaps($0.candidate) }
    }

    /// The honesty feature: a winner is named only when the two measurement
    /// intervals are disjoint.
    public var verdict: Verdict {
        guard let s = runStatistics else { return .unmeasured }
        guard !s.baseline.overlaps(s.candidate) else { return .withinNoise }
        return s.candidate.median < s.baseline.median ? .faster : .slower
    }

    /// True when either side carries repeat samples — the diff uses this to
    /// decide whether the verdict column is worth its width.
    public var hasRunStatistics: Bool {
        baseline?.runStatistics?.hasSpread == true || candidate?.runStatistics?.hasSpread == true
    }

    public func placements(peaks: PeakSet) -> (baseline: RooflinePlacement?, candidate: RooflinePlacement?) {
        (baseline.map { Roofline.place($0, peaks: peaks) },
         candidate.map { Roofline.place($0, peaks: peaks) })
    }

    /// Efficiency change in percentage points (candidate - baseline).
    public func efficiencyDeltaPoints(peaks: PeakSet) -> Double? {
        let p = placements(peaks: peaks)
        guard let a = p.baseline, let b = p.candidate else { return nil }
        return (b.efficiency - a.efficiency) * 100
    }

    /// nil when unmatched or when the bound type didn't move.
    public func boundChange(peaks: PeakSet) -> (from: BoundType, to: BoundType)? {
        let p = placements(peaks: peaks)
        guard let a = p.baseline, let b = p.candidate, a.bound != b.bound else { return nil }
        return (a.bound, b.bound)
    }

    // MARK: - Occupancy

    /// Both sides' occupancy blocks, when both sides have one. A v1 trace on
    /// either side (or an MPS region metalscope never saw the pipeline for)
    /// leaves this nil rather than pretending the missing side is unchanged.
    public var occupancies: (baseline: OccupancyInfo, candidate: OccupancyInfo)? {
        guard let a = baseline?.occupancy, let b = candidate?.occupancy else { return nil }
        return (a, b)
    }

    /// Threads per threadgroup, candidate minus baseline.
    public var threadgroupSizeDelta: Int? {
        guard let o = occupancies else { return nil }
        return o.candidate.threadsPerThreadgroup - o.baseline.threadsPerThreadgroup
    }

    /// Threadgroup-occupancy change in percentage points.
    public var occupancyDeltaPoints: Double? {
        guard let o = occupancies else { return nil }
        return (o.candidate.threadgroupOccupancy - o.baseline.threadgroupOccupancy) * 100
    }

    /// nil when unmatched, when either side lacks occupancy data, or when the
    /// limiter didn't move.
    public var limiterChange: (from: OccupancyLimiter, to: OccupancyLimiter)? {
        guard let o = occupancies, o.baseline.limiter != o.candidate.limiter else { return nil }
        return (o.baseline.limiter, o.candidate.limiter)
    }

    /// True when either side carries occupancy data at all — the report and diff
    /// use this to decide whether an occupancy column is worth its width.
    public var hasOccupancy: Bool {
        baseline?.occupancy != nil || candidate?.occupancy != nil
    }
}

public struct TraceDiff: Sendable {
    /// The refusal rule, printed verbatim wherever a verdict is shown. A rule
    /// the reader cannot see is a rule they cannot check.
    public static let verdictRule =
        "verdict compares medians and is withheld (\"no call\") when the two "
        + "min-p95 spreads overlap: a gap narrower than the run-to-run noise "
        + "that produced it is not a result."

    public var baselineTrace: Trace
    public var candidateTrace: Trace
    public var entries: [DiffEntry]

    public var matched: [DiffEntry] { entries.filter { $0.status == .matched } }
    public var unmatched: [DiffEntry] { entries.filter { $0.status != .matched } }

    /// Matched kernels whose two spreads overlapped, so no winner was called.
    public var withinNoise: [DiffEntry] { matched.filter { $0.verdict == .withinNoise } }
    /// Matched kernels that moved beyond both sides' measured spread.
    public var resolved: [DiffEntry] {
        matched.filter { $0.verdict == .faster || $0.verdict == .slower }
    }

    public init(baselineTrace: Trace, candidateTrace: Trace) {
        self.baselineTrace = baselineTrace
        self.candidateTrace = candidateTrace
        self.entries = TraceDiff.align(baseline: baselineTrace.kernels,
                                       candidate: candidateTrace.kernels)
    }

    /// Align kernels by label + shape + precision.
    ///
    /// Repeated identical keys (a kernel captured in a loop) are paired up
    /// positionally within their key group, so the Nth occurrence in the
    /// baseline lines up with the Nth in the candidate. Leftovers on either side
    /// become unmatched rows rather than being silently dropped. Baseline order
    /// is preserved; candidate-only rows are appended in candidate order.
    public static func align(baseline: [KernelRecord], candidate: [KernelRecord]) -> [DiffEntry] {
        func numbered(_ records: [KernelRecord]) -> [(occurrence: Int, record: KernelRecord)] {
            var counts: [String: Int] = [:]
            return records.map { r in
                let n = counts[r.alignmentKey, default: 0]
                counts[r.alignmentKey] = n + 1
                return (n, r)
            }
        }

        let a = numbered(baseline)
        let b = numbered(candidate)
        var candidateByKey: [String: KernelRecord] = [:]
        for item in b { candidateByKey["\(item.record.alignmentKey)#\(item.occurrence)"] = item.record }

        var entries: [DiffEntry] = []
        var consumed = Set<String>()

        for item in a {
            let slot = "\(item.record.alignmentKey)#\(item.occurrence)"
            let match = candidateByKey[slot]
            if match != nil { consumed.insert(slot) }
            entries.append(DiffEntry(key: item.record.alignmentKey,
                                     label: item.record.label,
                                     shape: item.record.shape,
                                     precision: item.record.precision,
                                     occurrence: item.occurrence,
                                     baseline: item.record,
                                     candidate: match))
        }

        for item in b {
            let slot = "\(item.record.alignmentKey)#\(item.occurrence)"
            guard !consumed.contains(slot) else { continue }
            entries.append(DiffEntry(key: item.record.alignmentKey,
                                     label: item.record.label,
                                     shape: item.record.shape,
                                     precision: item.record.precision,
                                     occurrence: item.occurrence,
                                     baseline: nil,
                                     candidate: item.record))
        }

        return entries
    }
}
