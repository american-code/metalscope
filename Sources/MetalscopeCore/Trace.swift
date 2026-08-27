import Foundation

/// How a kernel's duration was obtained, best first.
public enum TimingSource: String, Codable, Sendable {
    /// `MTLCounterSampleBuffer` timestamps at compute-encoder stage boundaries.
    case counterSampleBuffer = "counter-sample-buffer"
    /// `MTLCommandBuffer.gpuStartTime/gpuEndTime` — real GPU time, command-buffer
    /// granularity. Used when the encoders are created by someone else (MPS).
    case commandBuffer = "command-buffer"
    /// Host wall clock around commit+wait. Includes scheduling latency.
    case host = "host"

    public var displayName: String {
        switch self {
        case .counterSampleBuffer: return "counters"
        case .commandBuffer: return "cmdbuf"
        case .host: return "host"
        }
    }

    /// Position on the timing ladder, 0 being the best. A record covering
    /// several runs claims the worst tier any of them reached, for the same
    /// reason a region claims tier 1 only if *every* encoder in it was sampled.
    public var tierRank: Int {
        switch self {
        case .counterSampleBuffer: return 0
        case .commandBuffer: return 1
        case .host: return 2
        }
    }
}

/// A sub-span inside a captured region (one compute encoder, typically).
public struct StageSample: Codable, Sendable, Equatable {
    public var name: String
    public var durationSeconds: Double

    public init(name: String, durationSeconds: Double) {
        self.name = name
        self.durationSeconds = durationSeconds
    }
}

/// One captured kernel region: an annotation plus its measured duration.
///
/// `flops`/`bytes` are stored, not just derived, so that analysis tools can read
/// a trace without re-implementing the shape registry — and so `opaque` shapes
/// survive a round trip.
public struct KernelRecord: Codable, Sendable, Equatable {
    public var label: String
    public var shape: KernelShape
    public var precision: Precision
    /// Seconds for ONE invocation (total measured span / iterations). When the
    /// region was repeated, this is the median of `durationSamplesSeconds` —
    /// the lower median, so it is a run that actually happened.
    public var durationSeconds: Double
    /// How many times the annotated work was encoded inside the region.
    public var iterations: Int
    public var timingSource: TimingSource
    /// Analytic FLOPs for one invocation.
    public var flops: Double
    /// Analytic compulsory bytes for one invocation.
    public var bytes: Double
    /// Host-side wall clock for the whole region, for scheduling-overhead sanity.
    public var hostDurationSeconds: Double?
    public var stages: [StageSample]?
    /// Static occupancy facts (schema v2). Absent when the encoders were created
    /// by someone else (MPS, MLX) and metalscope never saw the pipeline state.
    public var occupancy: OccupancyInfo?
    /// Resolved hardware counters other than the timing ladder (schema v2), keyed
    /// `"<counter set>.<counter>"` and summed over the region's encoders. Absent
    /// on every chip that exposes only the `timestamp` counter set.
    public var counters: [String: Double]?
    /// Per-invocation seconds for each timed repeat, in the order they ran
    /// (schema v3). Absent when the region was captured once, which is what a
    /// v1/v2 trace and `--repeats 1` both produce.
    ///
    /// Only the samples are stored; min/median/mean/p95 are derived on read via
    /// `runStatistics`, so a trace cannot contradict its own summary.
    public var durationSamplesSeconds: [Double]?
    public var notes: [String: String]?

    public init(label: String,
                shape: KernelShape,
                precision: Precision = .fp32,
                durationSeconds: Double,
                iterations: Int = 1,
                timingSource: TimingSource,
                flops: Double? = nil,
                bytes: Double? = nil,
                hostDurationSeconds: Double? = nil,
                stages: [StageSample]? = nil,
                occupancy: OccupancyInfo? = nil,
                counters: [String: Double]? = nil,
                durationSamplesSeconds: [Double]? = nil,
                notes: [String: String]? = nil) {
        self.label = label
        self.shape = shape
        self.precision = precision
        self.durationSeconds = durationSeconds
        self.iterations = iterations
        self.timingSource = timingSource
        self.flops = flops ?? shape.flops
        self.bytes = bytes ?? shape.bytes(precision: precision)
        self.hostDurationSeconds = hostDurationSeconds
        self.stages = stages
        self.occupancy = occupancy
        self.counters = counters
        self.durationSamplesSeconds = durationSamplesSeconds
        self.notes = notes
    }

    /// Identity used to align kernels across two traces: label + shape + precision.
    public var alignmentKey: String {
        "\(label)|\(shape.descriptionText)|\(precision.rawValue)"
    }

    public var arithmeticIntensity: Double {
        bytes > 0 ? flops / bytes : 0
    }

    /// Run-to-run statistics, derived from the stored samples. nil for a
    /// single-run capture — absent means "not measured", never "no spread".
    public var runStatistics: RunStatistics? {
        durationSamplesSeconds.flatMap(RunStatistics.init(samples:))
    }
}

/// Device identity recorded with every trace, so a report can tell you the peaks
/// it scored against belong to the machine that produced the trace.
public struct DeviceInfo: Codable, Sendable, Equatable {
    public var name: String
    public var registryID: UInt64?
    public var maxWorkingSetBytes: UInt64?
    public var counterSets: [String]
    public var supportsStageBoundarySampling: Bool?
    /// `MTLDevice.maxThreadgroupMemoryLength` (schema v2) — context for the
    /// threadgroup-memory pressure in each kernel's occupancy block.
    public var maxThreadgroupMemoryBytes: Int?

    public init(name: String,
                registryID: UInt64? = nil,
                maxWorkingSetBytes: UInt64? = nil,
                counterSets: [String] = [],
                supportsStageBoundarySampling: Bool? = nil,
                maxThreadgroupMemoryBytes: Int? = nil) {
        self.name = name
        self.registryID = registryID
        self.maxWorkingSetBytes = maxWorkingSetBytes
        self.counterSets = counterSets
        self.supportsStageBoundarySampling = supportsStageBoundarySampling
        self.maxThreadgroupMemoryBytes = maxThreadgroupMemoryBytes
    }
}

/// The single JSON schema shared by capture, `report`, and `diff`.
/// Documented in docs/TRACE-FORMAT.md.
public struct Trace: Codable, Sendable, Equatable {
    /// Bumped to 3 when `kernels[].durationSamplesSeconds` was added.
    public static let currentSchemaVersion = 3
    /// Oldest schema this build still reads. v1 and v2 traces decode unchanged:
    /// every addition since has been optional, so an old trace simply has no
    /// occupancy block and no repeat samples.
    public static let minimumReadableSchemaVersion = 1

    public var schemaVersion: Int
    public var tool: String
    public var toolVersion: String
    public var createdAt: Date
    public var device: DeviceInfo
    /// Peaks in effect when the trace was captured. `report` prefers the trace's
    /// peaks when they are `.measured`, otherwise the local cache.
    public var peaks: PeakSet?
    public var kernels: [KernelRecord]
    public var notes: [String: String]?

    public init(schemaVersion: Int = Trace.currentSchemaVersion,
                tool: String = "metalscope",
                toolVersion: String = MetalscopeVersion.current,
                createdAt: Date = Date(),
                device: DeviceInfo,
                peaks: PeakSet? = nil,
                kernels: [KernelRecord] = [],
                notes: [String: String]? = nil) {
        self.schemaVersion = schemaVersion
        self.tool = tool
        self.toolVersion = toolVersion
        self.createdAt = createdAt
        self.device = device
        self.peaks = peaks
        self.kernels = kernels
        self.notes = notes
    }
}

public enum MetalscopeVersion {
    public static let current = "0.1.0"
}

public enum TraceIO {
    /// ISO-8601 with milliseconds: traces from the same second still order, and
    /// dates survive a write/read round trip unchanged.
    private static let fractionalFormatter: ISO8601DateFormatter = {
        let f = ISO8601DateFormatter()
        f.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        return f
    }()

    private static let plainFormatter = ISO8601DateFormatter()

    public static func makeEncoder() -> JSONEncoder {
        let e = JSONEncoder()
        e.dateEncodingStrategy = .custom { date, encoder in
            var container = encoder.singleValueContainer()
            try container.encode(fractionalFormatter.string(from: date))
        }
        e.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
        return e
    }

    public static func makeDecoder() -> JSONDecoder {
        let d = JSONDecoder()
        d.dateDecodingStrategy = .custom { decoder in
            let text = try decoder.singleValueContainer().decode(String.self)
            // Accept both fractional and whole-second stamps.
            if let date = fractionalFormatter.date(from: text) ?? plainFormatter.date(from: text) {
                return date
            }
            throw DecodingError.dataCorrupted(
                .init(codingPath: decoder.codingPath,
                      debugDescription: "expected an ISO-8601 timestamp, got '\(text)'"))
        }
        return d
    }

    public static func encode(_ trace: Trace) throws -> Data {
        try makeEncoder().encode(trace)
    }

    public static func decode(_ data: Data) throws -> Trace {
        let trace = try makeDecoder().decode(Trace.self, from: data)
        guard trace.schemaVersion <= Trace.currentSchemaVersion else {
            throw TraceError.unsupportedSchema(found: trace.schemaVersion,
                                               supported: Trace.currentSchemaVersion)
        }
        guard trace.schemaVersion >= Trace.minimumReadableSchemaVersion else {
            throw TraceError.retiredSchema(found: trace.schemaVersion,
                                           oldestSupported: Trace.minimumReadableSchemaVersion)
        }
        return trace
    }

    public static func write(_ trace: Trace, to url: URL) throws {
        let dir = url.deletingLastPathComponent()
        if !dir.path.isEmpty {
            try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        }
        try encode(trace).write(to: url, options: .atomic)
    }

    public static func read(from url: URL) throws -> Trace {
        try decode(try Data(contentsOf: url))
    }
}

public enum TraceError: Error, CustomStringConvertible {
    case unsupportedSchema(found: Int, supported: Int)
    case retiredSchema(found: Int, oldestSupported: Int)

    public var description: String {
        switch self {
        case let .unsupportedSchema(found, supported):
            return "trace schema version \(found) is newer than supported version \(supported)"
        case let .retiredSchema(found, oldestSupported):
            return "trace schema version \(found) is older than the oldest readable version \(oldestSupported)"
        }
    }
}
