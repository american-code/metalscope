import Foundation
import Metal

/// One resolved counter sample, normalized to counter-name -> value.
///
/// Counters the GPU declined to write (`MTLCounterErrorValue`) are **omitted**
/// rather than reported as a huge number, so a missing key means "not written",
/// not "zero".
public struct ResolvedCounterSample: Sendable, Equatable {
    public var values: [String: Double]

    public init(values: [String: Double] = [:]) {
        self.values = values
    }

    public subscript(counter: String) -> Double? { values[counter] }
}

/// Decodes the resolved bytes of one `MTLCounterSet`.
///
/// A sample buffer resolves to a packed array of the counter set's own C struct,
/// and the struct differs per set — `MTLCounterResultTimestamp` (1 field),
/// `MTLCounterResultStageUtilization` (6), `MTLCounterResultStatistic` (8). Each
/// set therefore gets its own resolver that knows its layout; the byte-level
/// decode is shared because all three layouts are, today, a flat run of
/// `uint64_t` fields in a documented order. `resolvedStride` is taken from the
/// real Metal struct and cross-checked against the field list in tests, so the
/// day Apple adds a field to one of these structs the tests fail rather than the
/// decode silently sliding.
public protocol CounterSetResolver: Sendable {
    /// Matches `MTLCounterSet.name` (`MTLCommonCounterSet` raw value).
    var counterSetName: String { get }
    /// Counter names in the order their fields appear in the resolved struct.
    var counterNames: [String] { get }
    /// `MemoryLayout<MTLCounterResult...>.stride` for this set.
    var resolvedStride: Int { get }
    func decode(_ data: Data) -> [ResolvedCounterSample]
}

public extension CounterSetResolver {
    /// Shared field decode: `counterNames.count` little-endian `uint64_t`s per
    /// sample, one sample every `resolvedStride` bytes.
    func decode(_ data: Data) -> [ResolvedCounterSample] {
        CounterFieldDecoder.decode(data, names: counterNames, stride: resolvedStride)
    }

    /// True when the declared field list accounts for the whole struct. Every
    /// resolver here must satisfy this; the tests assert it.
    var layoutIsConsistent: Bool {
        resolvedStride == counterNames.count * MemoryLayout<UInt64>.stride
    }
}

enum CounterFieldDecoder {
    static func decode(_ data: Data, names: [String], stride: Int) -> [ResolvedCounterSample] {
        guard stride > 0, !names.isEmpty else { return [] }
        let fieldSize = MemoryLayout<UInt64>.stride
        let count = data.count / stride
        guard count > 0 else { return [] }
        return data.withUnsafeBytes { raw -> [ResolvedCounterSample] in
            (0..<count).map { index -> ResolvedCounterSample in
                var values: [String: Double] = [:]
                for (field, name) in names.enumerated() {
                    let offset = index * stride + field * fieldSize
                    guard offset + fieldSize <= raw.count else { continue }
                    let value = raw.loadUnaligned(fromByteOffset: offset, as: UInt64.self)
                    // ~0 is MTLCounterErrorValue: the GPU did not write this one.
                    guard value != MTLCounterErrorValue else { continue }
                    values[name] = Double(value)
                }
                return ResolvedCounterSample(values: values)
            }
        }
    }
}

/// `timestamp` — the only counter set Apple silicon exposes today, and the one
/// the whole timing ladder rests on.
public struct TimestampCounterResolver: CounterSetResolver {
    public init() {}

    public let counterSetName = MTLCommonCounterSet.timestamp.rawValue
    public let counterNames = [MTLCommonCounter.timestamp.rawValue]
    public var resolvedStride: Int { MemoryLayout<MTLCounterResultTimestamp>.stride }

    /// Raw ticks, not `Double`.
    ///
    /// GPU timestamps are nanoseconds since boot; after ~104 days of uptime they
    /// exceed 2^53 and a `Double` starts rounding them, which would show up as
    /// phantom nanoseconds in every kernel duration. The timing path uses this;
    /// `decode` exists for uniformity with the other sets.
    public func timestamps(_ data: Data) -> [MTLTimestamp] {
        let stride = resolvedStride
        guard stride > 0 else { return [] }
        let count = data.count / stride
        guard count > 0 else { return [] }
        return data.withUnsafeBytes { raw in
            (0..<count).compactMap { index -> MTLTimestamp? in
                let offset = index * stride
                guard offset + MemoryLayout<MTLTimestamp>.size <= raw.count else { return nil }
                return raw.loadUnaligned(fromByteOffset: offset, as: MTLTimestamp.self)
            }
        }
    }
}

/// `stageutilization` — per-stage cycle counts. Absent on M1/M2/M3; present on
/// some discrete/older parts and, if Apple ever ships it, on later M-series.
public struct StageUtilizationCounterResolver: CounterSetResolver {
    public init() {}

    public let counterSetName = MTLCommonCounterSet.stageUtilization.rawValue
    /// Field order of `MTLCounterResultStageUtilization`.
    public let counterNames = [
        MTLCommonCounter.totalCycles.rawValue,
        MTLCommonCounter.vertexCycles.rawValue,
        MTLCommonCounter.tessellationCycles.rawValue,
        MTLCommonCounter.postTessellationVertexCycles.rawValue,
        MTLCommonCounter.fragmentCycles.rawValue,
        MTLCommonCounter.renderTargetWriteCycles.rawValue,
    ]
    public var resolvedStride: Int { MemoryLayout<MTLCounterResultStageUtilization>.stride }

    /// Cycle counter used as the denominator for utilization fractions.
    public static let totalCycles = MTLCommonCounter.totalCycles.rawValue
}

/// `statistic` — invocation counts, including `computeKernelInvocations`, which
/// is the one that would let metalscope cross-check a dispatch's thread count
/// against what the GPU says it ran.
public struct StatisticCounterResolver: CounterSetResolver {
    public init() {}

    public let counterSetName = MTLCommonCounterSet.statistic.rawValue
    /// Field order of `MTLCounterResultStatistic`.
    public let counterNames = [
        MTLCommonCounter.tessellationInputPatches.rawValue,
        MTLCommonCounter.vertexInvocations.rawValue,
        MTLCommonCounter.postTessellationVertexInvocations.rawValue,
        MTLCommonCounter.clipperInvocations.rawValue,
        MTLCommonCounter.clipperPrimitivesOut.rawValue,
        MTLCommonCounter.fragmentInvocations.rawValue,
        MTLCommonCounter.fragmentsPassed.rawValue,
        MTLCommonCounter.computeKernelInvocations.rawValue,
    ]
    public var resolvedStride: Int { MemoryLayout<MTLCounterResultStatistic>.stride }

    public static let computeKernelInvocations = MTLCommonCounter.computeKernelInvocations.rawValue
}

/// Every counter set metalscope knows how to decode, whether or not the local
/// chip has it. `metalscope info` prints both lists so the gap is visible.
public enum CounterResolvers {
    public static let all: [any CounterSetResolver] = [
        TimestampCounterResolver(),
        StageUtilizationCounterResolver(),
        StatisticCounterResolver(),
    ]

    public static let timestamp = TimestampCounterResolver()

    public static func resolver(for counterSetName: String) -> (any CounterSetResolver)? {
        all.first { $0.counterSetName == counterSetName }
    }

    /// Known sets other than `timestamp` — the ones sampled for their values
    /// rather than for timing.
    public static var auxiliary: [any CounterSetResolver] {
        all.filter { $0.counterSetName != timestamp.counterSetName }
    }
}

/// Turning resolved samples into the per-kernel numbers a trace carries.
///
/// Kept free of Metal state on purpose: every rule here is unit-tested against
/// synthetic resolved data, so the code path an M4 would light up is exercised
/// on a machine that has only `timestamp`.
public enum CounterAggregation {
    /// `end - start` per counter, for counters present in both samples.
    ///
    /// Counters are monotonic within a command buffer; a decreasing value means
    /// a counter wrapped or the samples were taken out of order, and the safe
    /// answer is to drop it rather than emit a negative cycle count.
    public static func delta(from start: ResolvedCounterSample,
                             to end: ResolvedCounterSample) -> [String: Double] {
        var result: [String: Double] = [:]
        for (name, endValue) in end.values {
            guard let startValue = start.values[name] else { continue }
            let difference = endValue - startValue
            guard difference >= 0 else { continue }
            result[name] = difference
        }
        return result
    }

    /// Sum per-encoder deltas into one region total.
    public static func sum(_ deltas: [[String: Double]]) -> [String: Double] {
        var total: [String: Double] = [:]
        for delta in deltas {
            for (name, value) in delta { total[name, default: 0] += value }
        }
        return total
    }

    /// Prefix counter names with their set, so two sets that share a counter name
    /// can't collide in `KernelRecord.counters`.
    public static func namespaced(_ totals: [String: Double], set: String) -> [String: Double] {
        var result: [String: Double] = [:]
        for (name, value) in totals { result["\(set).\(name)"] = value }
        return result
    }

    /// Stage-utilization cycles as fractions of `totalCycles`, appended as
    /// `<counter>Fraction`. Returns the input unchanged when there is no total to
    /// divide by — a fabricated 0% busy is worse than no number.
    public static func utilizationFractions(_ totals: [String: Double]) -> [String: Double] {
        guard let total = totals[StageUtilizationCounterResolver.totalCycles], total > 0 else {
            return totals
        }
        var result = totals
        for (name, value) in totals where name != StageUtilizationCounterResolver.totalCycles {
            result["\(name)Fraction"] = value / total
        }
        return result
    }
}
