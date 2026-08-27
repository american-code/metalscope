import Foundation

/// Spec-sheet peak numbers per chip. **Fallback only.**
///
/// Apple doesn't publish official FLOPS; these are the community-derived
/// figures (GPU cores x ALUs x 2 FLOP x clock) and every consumer of this table
/// must label them as folklore. `metalscope calibrate` measures the real
/// ceilings on the local chip and those always win — see `PeakSet.Source`.
public struct ChipPeaks: Sendable {
    public let name: String
    public let fp32TFLOPS: Double
    public let fp16TFLOPS: Double
    public let memoryBandwidthGBs: Double

    public init(name: String, fp32TFLOPS: Double, fp16TFLOPS: Double, memoryBandwidthGBs: Double) {
        self.name = name
        self.fp32TFLOPS = fp32TFLOPS
        self.fp16TFLOPS = fp16TFLOPS
        self.memoryBandwidthGBs = memoryBandwidthGBs
    }

    public static let known: [ChipPeaks] = [
        ChipPeaks(name: "Apple M1", fp32TFLOPS: 2.6, fp16TFLOPS: 5.2, memoryBandwidthGBs: 68),
        ChipPeaks(name: "Apple M1 Pro", fp32TFLOPS: 5.2, fp16TFLOPS: 10.4, memoryBandwidthGBs: 200),
        ChipPeaks(name: "Apple M1 Max", fp32TFLOPS: 10.4, fp16TFLOPS: 20.8, memoryBandwidthGBs: 400),
        ChipPeaks(name: "Apple M1 Ultra", fp32TFLOPS: 20.8, fp16TFLOPS: 41.6, memoryBandwidthGBs: 800),
        ChipPeaks(name: "Apple M2", fp32TFLOPS: 3.6, fp16TFLOPS: 7.2, memoryBandwidthGBs: 100),
        ChipPeaks(name: "Apple M2 Pro", fp32TFLOPS: 6.8, fp16TFLOPS: 13.6, memoryBandwidthGBs: 200),
        ChipPeaks(name: "Apple M2 Max", fp32TFLOPS: 13.6, fp16TFLOPS: 27.2, memoryBandwidthGBs: 400),
        ChipPeaks(name: "Apple M3", fp32TFLOPS: 4.1, fp16TFLOPS: 8.2, memoryBandwidthGBs: 100),
        ChipPeaks(name: "Apple M3 Pro", fp32TFLOPS: 7.4, fp16TFLOPS: 14.8, memoryBandwidthGBs: 150),
        ChipPeaks(name: "Apple M3 Max", fp32TFLOPS: 16.4, fp16TFLOPS: 32.8, memoryBandwidthGBs: 400),
        ChipPeaks(name: "Apple M4", fp32TFLOPS: 4.6, fp16TFLOPS: 9.2, memoryBandwidthGBs: 120),
        ChipPeaks(name: "Apple M4 Pro", fp32TFLOPS: 9.2, fp16TFLOPS: 18.4, memoryBandwidthGBs: 273),
        ChipPeaks(name: "Apple M4 Max", fp32TFLOPS: 18.4, fp16TFLOPS: 36.8, memoryBandwidthGBs: 546),
    ]

    /// Longest-prefix match, so "Apple M1 Pro" doesn't get scored against "Apple M1".
    public static func folklore(for deviceName: String) -> ChipPeaks? {
        known.filter { deviceName.hasPrefix($0.name) }
            .max { $0.name.count < $1.name.count }
    }

    public var peakSet: PeakSet {
        PeakSet(source: .specSheet,
                chip: name,
                fp32GFLOPS: fp32TFLOPS * 1000,
                fp16GFLOPS: fp16TFLOPS * 1000,
                bandwidthGBs: memoryBandwidthGBs,
                measuredAt: nil,
                details: nil)
    }
}

/// The two ceilings a roofline is drawn from, plus where they came from.
public struct PeakSet: Codable, Sendable, Equatable {
    public enum Source: String, Codable, Sendable {
        case measured
        case specSheet = "spec-sheet"

        /// How the number should be described in output. Peaks must be measured;
        /// the fallback says so out loud.
        public var label: String {
            switch self {
            case .measured: return "measured"
            case .specSheet: return "spec-sheet folklore"
            }
        }
    }

    public var source: Source
    public var chip: String
    public var fp32GFLOPS: Double
    public var fp16GFLOPS: Double?
    public var bandwidthGBs: Double
    public var measuredAt: Date?
    /// Free-form calibration provenance (best GEMM size, iterations, ...).
    public var details: [String: Double]?

    public init(source: Source,
                chip: String,
                fp32GFLOPS: Double,
                fp16GFLOPS: Double?,
                bandwidthGBs: Double,
                measuredAt: Date? = nil,
                details: [String: Double]? = nil) {
        self.source = source
        self.chip = chip
        self.fp32GFLOPS = fp32GFLOPS
        self.fp16GFLOPS = fp16GFLOPS
        self.bandwidthGBs = bandwidthGBs
        self.measuredAt = measuredAt
        self.details = details
    }

    /// Compute ceiling for a precision, falling back to the fp32 number when a
    /// half-precision peak was never measured.
    public func peakGFLOPS(for precision: Precision) -> Double {
        precision.usesHalfPeak ? (fp16GFLOPS ?? fp32GFLOPS) : fp32GFLOPS
    }

    /// Arithmetic intensity where the bandwidth roof meets the compute roof.
    public func ridgePoint(for precision: Precision) -> Double {
        guard bandwidthGBs > 0 else { return .infinity }
        return peakGFLOPS(for: precision) / bandwidthGBs
    }
}

/// Which resource a kernel is sitting against.
public enum BoundType: String, Codable, Sendable {
    case bandwidth
    case compute
    /// Within `ridgeTolerance` of the ridge point — genuinely balanced.
    case ridge

    public var displayName: String {
        switch self {
        case .bandwidth: return "bandwidth"
        case .compute: return "compute"
        case .ridge: return "ridge"
        }
    }
}

/// One kernel's position on the roofline.
public struct RooflinePlacement: Sendable, Equatable {
    public var arithmeticIntensity: Double
    public var achievedGFLOPS: Double
    public var achievedBandwidthGBs: Double
    public var ceilingGFLOPS: Double
    public var ridgeIntensity: Double
    public var bound: BoundType
    /// achieved / ceiling, in [0, ~1]. Can exceed 1 if the peaks are stale or the
    /// analytic byte model under-counts cache-resident reuse.
    public var efficiency: Double

    public init(arithmeticIntensity: Double,
                achievedGFLOPS: Double,
                achievedBandwidthGBs: Double,
                ceilingGFLOPS: Double,
                ridgeIntensity: Double,
                bound: BoundType,
                efficiency: Double) {
        self.arithmeticIntensity = arithmeticIntensity
        self.achievedGFLOPS = achievedGFLOPS
        self.achievedBandwidthGBs = achievedBandwidthGBs
        self.ceilingGFLOPS = ceilingGFLOPS
        self.ridgeIntensity = ridgeIntensity
        self.bound = bound
        self.efficiency = efficiency
    }
}

public enum Roofline {
    /// Relative distance from the ridge point still counted as "ridge".
    public static let defaultRidgeTolerance = 0.05

    /// Place a measurement on the roofline defined by `peaks`.
    ///
    /// - Parameters:
    ///   - flops: analytic FLOPs for one invocation
    ///   - bytes: analytic compulsory bytes for one invocation
    ///   - seconds: measured duration of one invocation
    public static func place(flops: Double,
                             bytes: Double,
                             seconds: Double,
                             precision: Precision,
                             peaks: PeakSet,
                             ridgeTolerance: Double = defaultRidgeTolerance) -> RooflinePlacement {
        let intensity = bytes > 0 ? flops / bytes : 0
        let achievedGFLOPS = seconds > 0 ? flops / seconds / 1e9 : 0
        let achievedGBs = seconds > 0 ? bytes / seconds / 1e9 : 0
        let peak = peaks.peakGFLOPS(for: precision)
        let ridge = peaks.ridgePoint(for: precision)
        let bandwidthRoof = intensity * peaks.bandwidthGBs
        let ceiling = min(peak, bandwidthRoof)

        let bound: BoundType
        if ridge.isFinite, ridge > 0, abs(intensity - ridge) / ridge <= ridgeTolerance {
            bound = .ridge
        } else if intensity < ridge {
            bound = .bandwidth
        } else {
            bound = .compute
        }

        let efficiency = ceiling > 0 ? achievedGFLOPS / ceiling : 0
        return RooflinePlacement(arithmeticIntensity: intensity,
                                 achievedGFLOPS: achievedGFLOPS,
                                 achievedBandwidthGBs: achievedGBs,
                                 ceilingGFLOPS: ceiling,
                                 ridgeIntensity: ridge,
                                 bound: bound,
                                 efficiency: efficiency)
    }

    public static func place(_ record: KernelRecord,
                             peaks: PeakSet,
                             ridgeTolerance: Double = defaultRidgeTolerance) -> RooflinePlacement {
        place(flops: record.flops,
              bytes: record.bytes,
              seconds: record.durationSeconds,
              precision: record.precision,
              peaks: peaks,
              ridgeTolerance: ridgeTolerance)
    }
}

/// One profiled kernel dispatch, placed on the roofline.
///
/// Retained from the scaffold for source compatibility; new code should build a
/// `KernelRecord` and call `Roofline.place`.
public struct KernelSample: Sendable {
    public let name: String
    public let flopsPerByte: Double
    public let achievedGFLOPS: Double
    public let durationMicros: Double

    public init(name: String, flopsPerByte: Double, achievedGFLOPS: Double, durationMicros: Double) {
        self.name = name
        self.flopsPerByte = flopsPerByte
        self.achievedGFLOPS = achievedGFLOPS
        self.durationMicros = durationMicros
    }

    /// Fraction of the roofline this dispatch achieved on `chip`.
    public func efficiency(on chip: ChipPeaks) -> Double {
        let ridge = chip.fp32TFLOPS * 1000 / chip.memoryBandwidthGBs
        let ceilingGFLOPS = flopsPerByte < ridge
            ? flopsPerByte * chip.memoryBandwidthGBs        // bandwidth-bound
            : chip.fp32TFLOPS * 1000                        // compute-bound
        return achievedGFLOPS / ceilingGFLOPS
    }
}
