import Foundation

/// Which structural property of a dispatch is the most likely occupancy limiter.
///
/// This is a *static* judgement made from the pipeline's own numbers and the
/// threadgroup size the dispatch asked for — no counters involved. Apple exposes
/// none of the counters that would let metalscope measure real residency, so
/// these are the things that can be said without guessing.
public enum OccupancyLimiter: String, Codable, Sendable, CaseIterable {
    /// Nothing structural stands out.
    case none
    /// The threadgroup size is not a multiple of `threadExecutionWidth`, so the
    /// last SIMD group runs with idle lanes on every threadgroup, forever.
    case executionWidthAlignment
    /// Fewer than two SIMD groups per threadgroup: there is almost nothing for
    /// the scheduler to interleave while a SIMD group waits on memory.
    case tinyThreadgroup
    /// Static threadgroup memory takes a large share of the per-threadgroup
    /// limit, which caps how many threadgroups can be co-resident on a core.
    case threadgroupMemory

    /// Ordering used when one record folds several dispatch shapes: the worst
    /// shape is the one worth reporting.
    public var severity: Int {
        switch self {
        case .none: return 0
        case .threadgroupMemory: return 1
        case .tinyThreadgroup: return 2
        case .executionWidthAlignment: return 3
        }
    }
}

/// Static occupancy facts for one kernel dispatch.
///
/// Everything here comes from `MTLComputePipelineState`
/// (`maxTotalThreadsPerThreadgroup`, `threadExecutionWidth`,
/// `staticThreadgroupMemoryLength`) plus the threadgroup size the dispatch
/// actually used. Only those raw inputs are stored; every ratio below is derived,
/// so a trace can never disagree with itself and a v2 reader needs no table of
/// per-chip constants.
///
/// What this is *not*: a measurement of how many threadgroups were resident. No
/// Apple GPU exposes that counter, and metalscope does not estimate it.
public struct OccupancyInfo: Codable, Sendable, Equatable, Hashable {
    /// Threads per threadgroup as dispatched (width x height x depth).
    public var threadsPerThreadgroup: Int
    /// `MTLComputePipelineState.maxTotalThreadsPerThreadgroup` for this kernel —
    /// a per-pipeline ceiling set by register pressure, not a device constant.
    public var maxTotalThreadsPerThreadgroup: Int
    /// `MTLComputePipelineState.threadExecutionWidth` — the SIMD group width (32
    /// on every Apple GPU shipped so far, but read, never assumed).
    public var threadExecutionWidth: Int
    /// `MTLComputePipelineState.staticThreadgroupMemoryLength`, in bytes.
    public var threadgroupMemoryBytes: Int
    /// `MTLDevice.maxThreadgroupMemoryLength`, when the capturing device was known.
    public var threadgroupMemoryLimitBytes: Int?
    /// Threadgroups in the grid for one dispatch, when the dispatch declared it.
    public var threadgroupsPerGrid: Int?
    /// How many dispatches were folded into this record (an iteration loop
    /// encodes the same dispatch many times).
    public var dispatchCount: Int
    /// How many *distinct* dispatch shapes were seen. > 1 means this record shows
    /// the worst of them; the report says so.
    public var variantCount: Int

    public init(threadsPerThreadgroup: Int,
                maxTotalThreadsPerThreadgroup: Int,
                threadExecutionWidth: Int,
                threadgroupMemoryBytes: Int = 0,
                threadgroupMemoryLimitBytes: Int? = nil,
                threadgroupsPerGrid: Int? = nil,
                dispatchCount: Int = 1,
                variantCount: Int = 1) {
        self.threadsPerThreadgroup = threadsPerThreadgroup
        self.maxTotalThreadsPerThreadgroup = maxTotalThreadsPerThreadgroup
        self.threadExecutionWidth = threadExecutionWidth
        self.threadgroupMemoryBytes = threadgroupMemoryBytes
        self.threadgroupMemoryLimitBytes = threadgroupMemoryLimitBytes
        self.threadgroupsPerGrid = threadgroupsPerGrid
        self.dispatchCount = dispatchCount
        self.variantCount = variantCount
    }

    // MARK: - Derived

    /// SIMD groups the hardware must launch per threadgroup — rounded *up*,
    /// which is the whole point: a ragged threadgroup still costs a full group.
    public var simdGroupsPerThreadgroup: Int {
        let width = max(1, threadExecutionWidth)
        return max(1, (max(0, threadsPerThreadgroup) + width - 1) / width)
    }

    /// Dispatched threadgroup size as a fraction of what the pipeline allows.
    ///
    /// Low is not automatically bad — 256 threads out of a 1024 ceiling is a
    /// perfectly good shape — which is why the limiter rules below do not fire
    /// on this number alone.
    public var threadgroupOccupancy: Double {
        guard maxTotalThreadsPerThreadgroup > 0 else { return 0 }
        return Double(threadsPerThreadgroup) / Double(maxTotalThreadsPerThreadgroup)
    }

    public var isExecutionWidthAligned: Bool {
        threadExecutionWidth > 0 && threadsPerThreadgroup % threadExecutionWidth == 0
    }

    /// Lanes launched but doing nothing, per threadgroup, because the threadgroup
    /// size doesn't fill its last SIMD group.
    public var idleLanesPerThreadgroup: Int {
        max(0, simdGroupsPerThreadgroup * max(1, threadExecutionWidth) - max(0, threadsPerThreadgroup))
    }

    /// Fraction of launched lanes that carry a thread. 1.0 when aligned.
    public var laneUtilization: Double {
        let launched = simdGroupsPerThreadgroup * max(1, threadExecutionWidth)
        guard launched > 0 else { return 0 }
        return Double(max(0, threadsPerThreadgroup)) / Double(launched)
    }

    /// Static threadgroup memory as a fraction of the device limit.
    public var threadgroupMemoryPressure: Double? {
        guard let limit = threadgroupMemoryLimitBytes, limit > 0 else { return nil }
        return Double(threadgroupMemoryBytes) / Double(limit)
    }

    /// How many threadgroups' worth of static threadgroup memory fit inside the
    /// device limit. An **upper bound** on co-residency, not a measurement: Metal
    /// exposes the per-threadgroup limit, and the scheduler has other reasons to
    /// keep fewer threadgroups resident. nil when the kernel uses none.
    public var memoryLimitedThreadgroups: Int? {
        guard threadgroupMemoryBytes > 0, let limit = threadgroupMemoryLimitBytes, limit > 0 else {
            return nil
        }
        return max(0, limit / threadgroupMemoryBytes)
    }

    /// The structural property most likely to be holding this dispatch back.
    public var limiter: OccupancyLimiter {
        if !isExecutionWidthAligned { return .executionWidthAlignment }
        if simdGroupsPerThreadgroup < 2 { return .tinyThreadgroup }
        if let pressure = threadgroupMemoryPressure, pressure > 0.5 { return .threadgroupMemory }
        return .none
    }

    /// One sentence for the report's headroom section. nil when nothing to say.
    public var hint: String? {
        switch limiter {
        case .none:
            return nil
        case .executionWidthAlignment:
            // A threadgroup smaller than one SIMD group has only one sane target,
            // so don't print "round to 32 or 32".
            let advice = nearestAlignedBelow == nearestAlignedAbove
                ? "round to \(nearestAlignedBelow)"
                : "round to \(nearestAlignedBelow) or \(nearestAlignedAbove)"
            return String(format: "threadgroup of %d is not a multiple of the %d-wide SIMD group — %d of %d lanes idle in every threadgroup (%.1f%% lane use); %@",
                          threadsPerThreadgroup, threadExecutionWidth,
                          idleLanesPerThreadgroup,
                          simdGroupsPerThreadgroup * max(1, threadExecutionWidth),
                          laneUtilization * 100,
                          advice)
        case .tinyThreadgroup:
            return "threadgroup of \(threadsPerThreadgroup) is a single SIMD group — nothing for the core to interleave while it waits on memory (pipeline allows \(maxTotalThreadsPerThreadgroup))"
        case .threadgroupMemory:
            let pressure = threadgroupMemoryPressure ?? 0
            let resident = memoryLimitedThreadgroups.map { " — at most \($0) such threadgroup(s) fit the limit" } ?? ""
            return String(format: "static threadgroup memory is %.0f%% of the device limit%@",
                          pressure * 100, resident)
        }
    }

    /// Nearest width-aligned threadgroup sizes, for the alignment hint.
    var nearestAlignedBelow: Int {
        let width = max(1, threadExecutionWidth)
        return max(width, (threadsPerThreadgroup / width) * width)
    }

    var nearestAlignedAbove: Int {
        let width = max(1, threadExecutionWidth)
        let up = simdGroupsPerThreadgroup * width
        return min(max(width, maxTotalThreadsPerThreadgroup), up)
    }

    /// Compact cell for the report table: threads, starred when ragged.
    public var tableText: String {
        isExecutionWidthAligned ? "\(threadsPerThreadgroup)" : "\(threadsPerThreadgroup)*"
    }

    /// Identity that ignores the fold bookkeeping — two dispatches of the same
    /// shape are the same shape however many times they were encoded.
    var shapeKey: OccupancyShapeKey {
        OccupancyShapeKey(threadsPerThreadgroup: threadsPerThreadgroup,
                          maxTotalThreadsPerThreadgroup: maxTotalThreadsPerThreadgroup,
                          threadExecutionWidth: threadExecutionWidth,
                          threadgroupMemoryBytes: threadgroupMemoryBytes,
                          threadgroupMemoryLimitBytes: threadgroupMemoryLimitBytes,
                          threadgroupsPerGrid: threadgroupsPerGrid)
    }

    /// Fold every dispatch observed inside one captured region into a single
    /// record.
    ///
    /// The common case is N identical dispatches (an iteration loop), which folds
    /// to one shape with `dispatchCount = N`. When a region really did dispatch
    /// different shapes, the *worst* one is kept and `variantCount` records that
    /// something was dropped — averaging threadgroup sizes would invent a shape
    /// that never ran.
    public static func fold(_ observations: [OccupancyInfo]) -> OccupancyInfo? {
        guard !observations.isEmpty else { return nil }
        var distinct: [OccupancyShapeKey: OccupancyInfo] = [:]
        for observation in observations where distinct[observation.shapeKey] == nil {
            distinct[observation.shapeKey] = observation
        }
        let worst = distinct.values.sorted { a, b in
            if a.limiter.severity != b.limiter.severity { return a.limiter.severity > b.limiter.severity }
            if a.laneUtilization != b.laneUtilization { return a.laneUtilization < b.laneUtilization }
            if a.threadgroupOccupancy != b.threadgroupOccupancy {
                return a.threadgroupOccupancy < b.threadgroupOccupancy
            }
            return a.threadsPerThreadgroup < b.threadsPerThreadgroup
        }.first
        guard var result = worst else { return nil }
        result.dispatchCount = observations.reduce(0) { $0 + max(1, $1.dispatchCount) }
        result.variantCount = distinct.count
        return result
    }
}

/// Shape identity for folding — deliberately excludes `dispatchCount` and
/// `variantCount` so bookkeeping never splits an otherwise identical shape.
struct OccupancyShapeKey: Hashable {
    var threadsPerThreadgroup: Int
    var maxTotalThreadsPerThreadgroup: Int
    var threadExecutionWidth: Int
    var threadgroupMemoryBytes: Int
    var threadgroupMemoryLimitBytes: Int?
    var threadgroupsPerGrid: Int?
}
