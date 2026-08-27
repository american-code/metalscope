import Foundation

/// Run-to-run statistics for a kernel captured more than once.
///
/// Derived on read from `KernelRecord.durationSamplesSeconds`, for the same
/// reason the occupancy ratios are (see `OccupancyInfo`): a trace that stored
/// both the samples and the summary could be edited, merged, or written by an
/// older tool into a state where the two disagree, and a reader would have no
/// way to know which half to believe.
///
/// Two definitions here are chosen so that every number printed is a
/// measurement rather than an interpolation:
///
/// - **median** is the *lower* of the two middle samples when `count` is even,
///   so the reported duration is always a run that actually happened rather
///   than the average of two that did.
/// - **p95** is nearest-rank (`ceil(0.95 n)`), not interpolated. With the
///   default five repeats that makes p95 the slowest sample, which is the
///   honest reading of "95th percentile of five points".
public struct RunStatistics: Sendable, Equatable {
    /// Number of timed repeats. Warm-up runs are not among them.
    public let count: Int
    public let min: Double
    /// Lower median: always an observed sample.
    public let median: Double
    public let mean: Double
    /// Nearest-rank 95th percentile.
    public let p95: Double
    public let max: Double

    /// nil for an empty sample set — there is no such thing as the median of
    /// nothing, and inventing one is how a profiler starts lying.
    public init?(samples: [Double]) {
        let usable = samples.filter { $0.isFinite && $0 > 0 }
        guard !usable.isEmpty else { return nil }
        let sorted = usable.sorted()
        let n = sorted.count
        count = n
        min = sorted[0]
        max = sorted[n - 1]
        median = sorted[(n - 1) / 2]
        mean = sorted.reduce(0, +) / Double(n)
        p95 = sorted[Swift.min(n - 1, Swift.max(0, Int((0.95 * Double(n)).rounded(.up)) - 1))]
    }

    /// A single sample has no spread, and a spread of one point must never be
    /// read as "this measurement is tight".
    public var hasSpread: Bool { count > 1 }

    /// The interval a comparison is allowed to resolve inside: `min` to `p95`.
    ///
    /// The low end is the fastest run seen and the high end is the nearest-rank
    /// p95 rather than the maximum, so one scheduler hiccup does not widen the
    /// interval far enough to swallow every real result.
    public var interval: ClosedRange<Double> { min...Swift.max(min, p95) }

    /// `(p95 - min) / median` — how wide this kernel's noise band is relative to
    /// the number the report prints.
    public var spreadFraction: Double {
        median > 0 ? (p95 - min) / median : 0
    }

    /// True when the two measurement intervals share any point at all.
    ///
    /// This is the whole of the diff's refusal rule: if the baseline could have
    /// been as fast as the candidate, or the candidate as slow as the baseline,
    /// then the difference between their medians is smaller than the noise that
    /// produced them and there is no winner to call.
    public func overlaps(_ other: RunStatistics) -> Bool {
        interval.overlaps(other.interval)
    }
}
