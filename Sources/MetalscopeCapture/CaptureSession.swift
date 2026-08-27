import Foundation
import Metal
import MetalscopeCore

public enum CaptureError: Error, CustomStringConvertible {
    case noMetalDevice
    case commandQueueCreationFailed
    case commandBufferCreationFailed
    case encoderCreationFailed

    public var description: String {
        switch self {
        case .noMetalDevice: return "no Metal device available"
        case .commandQueueCreationFailed: return "could not create a Metal command queue"
        case .commandBufferCreationFailed: return "could not create a Metal command buffer"
        case .encoderCreationFailed: return "could not create a compute command encoder"
        }
    }
}

/// One counter set the device exposes, and whether metalscope can decode it.
///
/// The `resolver` half is the interesting one: metalscope ships resolvers for
/// all three common counter sets, so a chip that exposes `stageutilization` or
/// `statistic` gets them captured with no code change. On every Apple part
/// shipped so far this list has exactly one entry.
public struct CounterSetInfo: Sendable, Equatable {
    public var name: String
    /// Counter names as the device reports them.
    public var counterNames: [String]
    /// metalscope knows this set's resolved struct layout.
    public var hasResolver: Bool

    public init(name: String, counterNames: [String], hasResolver: Bool) {
        self.name = name
        self.counterNames = counterNames
        self.hasResolver = hasResolver
    }
}

/// What this device can actually tell us about time — and, where the chip has
/// them, about anything else.
///
/// On Apple silicon today (M1/M2/M3 families) the only counter set is
/// `timestamp` and the only supported sampling point is `.atStageBoundary`;
/// `metalscope info` prints what the local chip exposes. Everything here
/// degrades to command-buffer GPU time rather than failing.
public struct CaptureCapabilities: Sendable {
    /// Every counter set the device exposes, with per-set detail.
    public var counterSets: [CounterSetInfo]
    public var supportsStageBoundarySampling: Bool
    public var supportsDispatchBoundarySampling: Bool
    public var supportsBlitBoundarySampling: Bool
    /// `MTLDevice.maxThreadgroupMemoryLength` — the denominator for threadgroup
    /// memory pressure in the occupancy analysis.
    public var maxThreadgroupMemoryBytes: Int
    public var maxThreadsPerThreadgroup: MTLSize

    public var counterSetNames: [String] { counterSets.map(\.name) }

    public var hasTimestampCounterSet: Bool {
        counterSets.contains { $0.name == MTLCommonCounterSet.timestamp.rawValue }
    }

    /// Sets that are present *and* decodable, other than `timestamp`: the ones
    /// sampled for their values rather than for timing.
    public var auxiliaryCounterSetNames: [String] {
        counterSets
            .filter { $0.hasResolver && $0.name != MTLCommonCounterSet.timestamp.rawValue }
            .map(\.name)
    }

    /// Counter sets metalscope can decode but this chip does not expose.
    public var absentKnownCounterSetNames: [String] {
        let present = Set(counterSetNames)
        return CounterResolvers.all.map(\.counterSetName).filter { !present.contains($0) }
    }

    /// True when we can time individual compute encoders rather than whole
    /// command buffers.
    public var canSampleEncoderStages: Bool {
        hasTimestampCounterSet && supportsStageBoundarySampling
    }

    /// Best timing-ladder tier this device can reach. Reported by `info` and
    /// used to fill a row of docs/COUNTER-MATRIX.md.
    public var timingLadderTier: TimingSource {
        canSampleEncoderStages ? .counterSampleBuffer : .commandBuffer
    }

    public init(device: MTLDevice) {
        let sets = device.counterSets ?? []
        counterSets = sets.map { set in
            CounterSetInfo(name: set.name,
                           counterNames: set.counters.map(\.name),
                           hasResolver: CounterResolvers.resolver(for: set.name) != nil)
        }
        supportsStageBoundarySampling = device.supportsCounterSampling(.atStageBoundary)
        supportsDispatchBoundarySampling = device.supportsCounterSampling(.atDispatchBoundary)
        supportsBlitBoundarySampling = device.supportsCounterSampling(.atBlitBoundary)
        maxThreadgroupMemoryBytes = device.maxThreadgroupMemoryLength
        maxThreadsPerThreadgroup = device.maxThreadsPerThreadgroup
    }
}

/// Explicit, swizzle-free capture: an app or benchmark opts in by creating a
/// session and wrapping annotated work in `capture`.
///
/// ```swift
/// let session = try CaptureSession()
/// try session.capture(label: "qkv", shape: .gemm(m: 1024, n: 1024, k: 1024)) { region in
///     let enc = try region.makeComputeCommandEncoder()
///     enc.setComputePipelineState(pipeline)
///     region.dispatchThreads(enc, pipeline: pipeline,          // records occupancy
///                            threads: grid, threadsPerThreadgroup: group)
///     enc.endEncoding()
/// }
/// try session.writeTrace(to: url)
/// ```
///
/// Each captured region is its own command buffer, committed and waited on, so
/// timestamps resolve immediately and the recorded duration belongs to exactly
/// the annotated work.
public final class CaptureSession {
    public let device: MTLDevice
    public let commandQueue: MTLCommandQueue
    public let capabilities: CaptureCapabilities
    /// Upper bound on stage samples per region (2 per compute encoder). A region
    /// that encodes more encoders than this can hold is still timed correctly —
    /// it falls back to command-buffer GPU time rather than reporting the span
    /// of the sampled prefix as if it covered everything.
    public var maxSamplesPerRegion: Int = 1024
    /// Also sample every non-timestamp counter set the chip exposes and metalscope
    /// can decode, recording the resolved values in `KernelRecord.counters`.
    /// A no-op on hardware that exposes only `timestamp` — which is all of Apple
    /// silicon so far.
    public var samplesAuxiliaryCounterSets: Bool = true

    public private(set) var records: [KernelRecord] = []

    /// Every exposed counter set with a resolver, timestamp first so it always
    /// lands on attachment 0 (the timing path reads that one).
    private let resolvableSets: [(resolver: any CounterSetResolver, counterSet: MTLCounterSet)]

    /// A compute pass can carry only a handful of sample-buffer attachments;
    /// Metal doesn't publish the number, so cap conservatively.
    static let maxSampleBufferAttachments = 4

    public init(device: MTLDevice? = nil, commandQueue: MTLCommandQueue? = nil) throws {
        guard let device = device ?? MTLCreateSystemDefaultDevice() else {
            throw CaptureError.noMetalDevice
        }
        self.device = device
        guard let queue = commandQueue ?? device.makeCommandQueue() else {
            throw CaptureError.commandQueueCreationFailed
        }
        queue.label = "metalscope.capture"
        self.commandQueue = queue
        self.capabilities = CaptureCapabilities(device: device)

        let exposed = device.counterSets ?? []
        let paired = exposed.compactMap { set -> (resolver: any CounterSetResolver, counterSet: MTLCounterSet)? in
            guard let resolver = CounterResolvers.resolver(for: set.name) else { return nil }
            return (resolver, set)
        }
        let timestampName = MTLCommonCounterSet.timestamp.rawValue
        self.resolvableSets = paired.filter { $0.resolver.counterSetName == timestampName }
            + paired.filter { $0.resolver.counterSetName != timestampName }
    }

    // MARK: - Capture

    /// Run `body`, timing everything it encodes, and record it under `shape`.
    ///
    /// - Parameter iterations: how many times `body` encodes the annotated work.
    ///   The recorded duration is the measured span divided by this, which is how
    ///   you get a stable number out of a kernel that runs in tens of microseconds.
    /// - Parameter repeats: how many separate timed regions to run. One record is
    ///   appended whatever this is: `durationSeconds` becomes the median and
    ///   `durationSamplesSeconds` carries every sample, so a reader can see the
    ///   spread instead of guessing at it. Dividing a long region by its
    ///   iteration count controls the GPU's clock ramp but says nothing about
    ///   run-to-run variance, which on a microsecond kernel is the larger term.
    /// - Parameter warmupRuns: regions to run and throw away first, so the timed
    ///   repeats all see an already-ramped GPU. Defaults to one when repeating
    ///   and none when not — a lone run has nothing to be consistent with.
    /// - Returns: the value from the last timed repeat.
    @discardableResult
    public func capture<T>(label: String,
                           shape: KernelShape,
                           precision: Precision = .fp32,
                           iterations: Int = 1,
                           repeats: Int = 1,
                           warmupRuns: Int? = nil,
                           notes: [String: String]? = nil,
                           _ body: (CaptureRegion) throws -> T) throws -> T {
        precondition(iterations >= 1, "iterations must be >= 1")
        precondition(repeats >= 1, "repeats must be >= 1")
        let warmups = warmupRuns ?? (repeats > 1 ? 1 : 0)
        precondition(warmups >= 0, "warmupRuns must be >= 0")

        for _ in 0..<warmups {
            _ = try runRegion(label: "\(label).warmup", iterations: iterations, body)
        }

        var runs: [RunResult<T>] = []
        runs.reserveCapacity(repeats)
        for _ in 0..<repeats {
            runs.append(try runRegion(label: label, iterations: iterations, body))
        }

        let samples = runs.map(\.perIterationSeconds)
        // Lower median, by index, so every non-aggregated field below comes from
        // one repeat that actually happened rather than a blend of several.
        let ranked = samples.indices.sorted { samples[$0] < samples[$1] }
        let representative = runs[ranked[(runs.count - 1) / 2]]
        // A record covering several runs claims the worst tier any of them
        // reached; a duration is only as trustworthy as its weakest sample.
        let source = runs.map(\.source).max { $0.tierRank < $1.tierRank } ?? representative.source

        records.append(KernelRecord(label: label,
                                    shape: shape,
                                    precision: precision,
                                    durationSeconds: representative.perIterationSeconds,
                                    iterations: iterations,
                                    timingSource: source,
                                    hostDurationSeconds: representative.hostSeconds,
                                    stages: representative.stages.isEmpty ? nil : representative.stages,
                                    occupancy: representative.occupancy,
                                    counters: representative.counters.isEmpty ? nil : representative.counters,
                                    durationSamplesSeconds: repeats > 1 ? samples : nil,
                                    notes: notes))
        return runs[runs.count - 1].value
    }

    /// Convenience for the common case of a single compute encoder that the
    /// caller dispatches into `iterations` times.
    @discardableResult
    public func captureCompute<T>(label: String,
                                  shape: KernelShape,
                                  precision: Precision = .fp32,
                                  iterations: Int = 1,
                                  repeats: Int = 1,
                                  warmupRuns: Int? = nil,
                                  notes: [String: String]? = nil,
                                  _ body: (MTLComputeCommandEncoder) throws -> T) throws -> T {
        try capture(label: label, shape: shape, precision: precision,
                    iterations: iterations, repeats: repeats, warmupRuns: warmupRuns,
                    notes: notes) { region in
            let encoder = try region.makeComputeCommandEncoder(label: label)
            defer { encoder.endEncoding() }
            return try body(encoder)
        }
    }

    /// One timed region: everything `capture` used to do before repeats existed.
    private func runRegion<T>(label: String,
                              iterations: Int,
                              _ body: (CaptureRegion) throws -> T) throws -> RunResult<T> {
        guard let commandBuffer = commandQueue.makeCommandBuffer() else {
            throw CaptureError.commandBufferCreationFailed
        }
        commandBuffer.label = label

        // Two samples per encoder, plus slack for kernels that use more than one
        // encoder per iteration. Capacity is capped; regions that overflow it are
        // timed from the command buffer instead.
        let capacity = min(maxSamplesPerRegion, max(4, 2 * iterations + 4))
        let region = CaptureRegion(commandBuffer: commandBuffer,
                                   sampleSets: makeSampleSets(label: label, sampleCount: capacity),
                                   capacity: capacity,
                                   iterations: iterations,
                                   threadgroupMemoryLimitBytes: capabilities.maxThreadgroupMemoryBytes)

        let result = try body(region)

        let (cpu0, gpu0) = device.sampleTimestamps()
        let hostStart = DispatchTime.now().uptimeNanoseconds
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        let hostEnd = DispatchTime.now().uptimeNanoseconds
        let (cpu1, gpu1) = device.sampleTimestamps()

        let hostSeconds = Double(hostEnd - hostStart) / 1e9
        let scale = Self.gpuTickScale(cpu0: cpu0, gpu0: gpu0, cpu1: cpu1, gpu1: gpu1)

        let timing = resolveTiming(region: region, commandBuffer: commandBuffer,
                                   scale: scale, hostSeconds: hostSeconds)

        return RunResult(value: result,
                         perIterationSeconds: timing.total / Double(iterations),
                         source: timing.source,
                         stages: timing.stages,
                         occupancy: OccupancyInfo.fold(region.occupancyObservations),
                         counters: resolveAuxiliaryCounters(region: region),
                         hostSeconds: hostSeconds)
    }

    /// One timed region's output, before repeats are folded into a record.
    private struct RunResult<T> {
        var value: T
        var perIterationSeconds: Double
        var source: TimingSource
        var stages: [StageSample]
        var occupancy: OccupancyInfo?
        var counters: [String: Double]
        var hostSeconds: Double
    }

    /// Discard recorded kernels (e.g. after warmup).
    public func reset() {
        records.removeAll()
    }

    /// Drop the most recent `count` records — used to throw away warmup regions
    /// that were captured through the same code path as the real ones.
    public func dropLast(_ count: Int) {
        records.removeLast(min(count, records.count))
    }

    // MARK: - Trace output

    public var deviceInfo: DeviceInfo {
        DeviceInfo(name: device.name,
                   registryID: device.registryID,
                   maxWorkingSetBytes: device.recommendedMaxWorkingSetSize,
                   counterSets: capabilities.counterSetNames,
                   supportsStageBoundarySampling: capabilities.supportsStageBoundarySampling,
                   maxThreadgroupMemoryBytes: capabilities.maxThreadgroupMemoryBytes)
    }

    public func makeTrace(peaks: PeakSet? = nil, notes: [String: String]? = nil) -> Trace {
        Trace(device: deviceInfo,
              peaks: peaks ?? PeaksStore.default.resolve(for: device.name),
              kernels: records,
              notes: notes)
    }

    /// Default trace destination: `$METALSCOPE_TRACE` if the process was launched
    /// by `metalscope profile`, otherwise nil.
    public static var environmentTraceURL: URL? {
        guard let path = ProcessInfo.processInfo.environment["METALSCOPE_TRACE"], !path.isEmpty else {
            return nil
        }
        return URL(fileURLWithPath: (path as NSString).expandingTildeInPath)
    }

    @discardableResult
    public func writeTrace(to url: URL? = nil,
                           peaks: PeakSet? = nil,
                           notes: [String: String]? = nil) throws -> URL {
        let destination = url ?? CaptureSession.environmentTraceURL
            ?? URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
                .appendingPathComponent("metalscope-trace.json")
        try TraceIO.write(makeTrace(peaks: peaks, notes: notes), to: destination)
        return destination
    }

    // MARK: - Internals

    /// One sample buffer per resolvable counter set, in attachment order.
    private func makeSampleSets(label: String, sampleCount: Int) -> [CaptureRegion.SampleSet] {
        guard capabilities.canSampleEncoderStages else { return [] }
        var sets: [CaptureRegion.SampleSet] = []
        for (resolver, counterSet) in resolvableSets {
            let isTimestamp = resolver.counterSetName == MTLCommonCounterSet.timestamp.rawValue
            if !isTimestamp && !samplesAuxiliaryCounterSets { continue }
            guard sets.count < Self.maxSampleBufferAttachments else { break }
            guard let buffer = makeSampleBuffer(counterSet: counterSet,
                                                label: "\(label).\(resolver.counterSetName)",
                                                sampleCount: sampleCount) else { continue }
            sets.append(CaptureRegion.SampleSet(resolver: resolver, buffer: buffer))
        }
        // The timing path assumes attachment 0 is timestamps; if that one failed
        // to allocate, the rest are useless for correlating anything.
        guard sets.first?.resolver.counterSetName == MTLCommonCounterSet.timestamp.rawValue else {
            return []
        }
        return sets
    }

    private func makeSampleBuffer(counterSet: MTLCounterSet,
                                  label: String,
                                  sampleCount: Int) -> MTLCounterSampleBuffer? {
        let descriptor = MTLCounterSampleBufferDescriptor()
        descriptor.counterSet = counterSet
        descriptor.sampleCount = sampleCount
        descriptor.storageMode = .shared
        descriptor.label = "metalscope.\(label)"
        return try? device.makeCounterSampleBuffer(descriptor: descriptor)
    }

    /// GPU ticks -> host nanoseconds. Both clocks are nanosecond-based on Apple
    /// silicon today, but derive the ratio rather than assuming it.
    static func gpuTickScale(cpu0: MTLTimestamp, gpu0: MTLTimestamp,
                             cpu1: MTLTimestamp, gpu1: MTLTimestamp) -> Double {
        guard gpu1 > gpu0, cpu1 > cpu0 else { return 1.0 }
        let scale = Double(cpu1 - cpu0) / Double(gpu1 - gpu0)
        guard scale.isFinite, scale > 1e-6, scale < 1e6 else { return 1.0 }
        return scale
    }

    private struct Timing {
        var total: Double
        var source: TimingSource
        var stages: [StageSample]
    }

    private func resolveTiming(region: CaptureRegion,
                               commandBuffer: MTLCommandBuffer,
                               scale: Double,
                               hostSeconds: Double) -> Timing {
        // Only trust counter timestamps when every encoder in the region was
        // sampled: the span of a sampled *prefix* divided by the full iteration
        // count would silently report several times the real throughput.
        let fullySampled = region.encoderCount > 0 && region.stages.count == region.encoderCount
        if fullySampled, let sampleBuffer = region.timestampSampleBuffer,
           let data = try? sampleBuffer.resolveCounterRange(0..<region.usedSampleCount) {
            let timestamps = CounterResolvers.timestamp.timestamps(data)
            var stages: [StageSample] = []
            var earliest: MTLTimestamp = .max
            var latest: MTLTimestamp = 0
            for stage in region.stages {
                guard stage.endIndex < timestamps.count else { continue }
                let start = timestamps[stage.startIndex]
                let end = timestamps[stage.endIndex]
                // MTLCounterErrorValue (~0) marks a sample the GPU didn't write.
                guard start != MTLCounterErrorValue, end != MTLCounterErrorValue, end > start else { continue }
                stages.append(StageSample(name: stage.name,
                                          durationSeconds: Double(end - start) * scale / 1e9))
                earliest = min(earliest, start)
                latest = max(latest, end)
            }
            if stages.count == region.encoderCount, latest > earliest {
                return Timing(total: Double(latest - earliest) * scale / 1e9,
                              source: .counterSampleBuffer,
                              stages: stages)
            }
        }

        let gpuSpan = commandBuffer.gpuEndTime - commandBuffer.gpuStartTime
        if gpuSpan > 0 {
            return Timing(total: gpuSpan, source: .commandBuffer, stages: [])
        }
        return Timing(total: hostSeconds, source: .host, stages: [])
    }

    /// Resolve every non-timestamp counter set sampled in this region, summing
    /// each encoder's (end - start) delta. Empty on every chip that exposes only
    /// `timestamp`; the aggregation rules themselves are tested against synthetic
    /// resolved data in `CounterResolverTests`.
    private func resolveAuxiliaryCounters(region: CaptureRegion) -> [String: Double] {
        guard region.usedSampleCount > 0, !region.stages.isEmpty else { return [:] }
        var result: [String: Double] = [:]
        for set in region.sampleSets.dropFirst() {
            guard let data = try? set.buffer.resolveCounterRange(0..<region.usedSampleCount) else {
                continue
            }
            let samples = set.resolver.decode(data)
            let deltas = region.stages.compactMap { stage -> [String: Double]? in
                guard stage.startIndex < samples.count, stage.endIndex < samples.count else { return nil }
                return CounterAggregation.delta(from: samples[stage.startIndex],
                                                to: samples[stage.endIndex])
            }
            guard !deltas.isEmpty else { continue }
            var totals = CounterAggregation.sum(deltas)
            if set.resolver.counterSetName == MTLCommonCounterSet.stageUtilization.rawValue {
                totals = CounterAggregation.utilizationFractions(totals)
            }
            result.merge(CounterAggregation.namespaced(totals, set: set.resolver.counterSetName)) { a, _ in a }
        }
        return result
    }
}

/// The command buffer being timed, plus encoder factories that wire up
/// stage-boundary counter sampling when the device supports it.
public final class CaptureRegion {
    /// One counter set being sampled across this region's encoders.
    struct SampleSet {
        let resolver: any CounterSetResolver
        let buffer: MTLCounterSampleBuffer
    }

    public let commandBuffer: MTLCommandBuffer
    public let iterations: Int

    let sampleSets: [SampleSet]
    private let capacity: Int
    /// `MTLDevice.maxThreadgroupMemoryLength`, stamped into each occupancy record.
    private let threadgroupMemoryLimitBytes: Int
    private(set) var stages: [(name: String, startIndex: Int, endIndex: Int)] = []
    /// Sample slots consumed. All sample sets advance in lockstep, so one cursor
    /// indexes every attached buffer.
    private(set) var usedSampleCount: Int = 0
    /// Every compute encoder handed out, sampled or not.
    private(set) var encoderCount: Int = 0
    /// Static occupancy facts for every dispatch the caller reported.
    public private(set) var occupancyObservations: [OccupancyInfo] = []

    var timestampSampleBuffer: MTLCounterSampleBuffer? {
        guard let first = sampleSets.first,
              first.resolver.counterSetName == MTLCommonCounterSet.timestamp.rawValue else {
            return nil
        }
        return first.buffer
    }

    init(commandBuffer: MTLCommandBuffer,
         sampleSets: [SampleSet],
         capacity: Int,
         iterations: Int,
         threadgroupMemoryLimitBytes: Int) {
        self.commandBuffer = commandBuffer
        self.sampleSets = sampleSets
        self.capacity = capacity
        self.iterations = iterations
        self.threadgroupMemoryLimitBytes = threadgroupMemoryLimitBytes
    }

    /// A compute encoder whose start/end are sampled into the region's counter
    /// buffers. Falls back to a plain encoder when the device can't sample stages
    /// or the region ran out of sample slots — timing then comes from the
    /// command buffer instead, and the trace says so.
    public func makeComputeCommandEncoder(label: String? = nil) throws -> MTLComputeCommandEncoder {
        encoderCount += 1
        if !sampleSets.isEmpty, usedSampleCount + 2 <= capacity {
            let startIndex = usedSampleCount
            let endIndex = usedSampleCount + 1
            let descriptor = MTLComputePassDescriptor()
            for (attachment, set) in sampleSets.enumerated() {
                descriptor.sampleBufferAttachments[attachment].sampleBuffer = set.buffer
                descriptor.sampleBufferAttachments[attachment].startOfEncoderSampleIndex = startIndex
                descriptor.sampleBufferAttachments[attachment].endOfEncoderSampleIndex = endIndex
            }
            if let encoder = commandBuffer.makeComputeCommandEncoder(descriptor: descriptor) {
                encoder.label = label ?? "metalscope.stage\(stages.count)"
                usedSampleCount += 2
                stages.append((name: encoder.label ?? "stage\(stages.count)",
                               startIndex: startIndex,
                               endIndex: endIndex))
                return encoder
            }
        }
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw CaptureError.encoderCreationFailed
        }
        encoder.label = label ?? "metalscope.stage\(stages.count)"
        return encoder
    }

    // MARK: - Occupancy

    /// Record the static occupancy of a dispatch metalscope did not encode itself.
    ///
    /// Metal has no way to ask an encoder what pipeline it is holding, and
    /// metalscope refuses to swizzle, so the pipeline has to be handed over
    /// explicitly. Prefer `dispatchThreads`/`dispatchThreadgroups`, which do this
    /// for you and cannot disagree with what was actually dispatched.
    @discardableResult
    public func observe(pipeline: MTLComputePipelineState,
                        threadsPerThreadgroup: MTLSize,
                        threadgroupsPerGrid: Int? = nil) -> OccupancyInfo {
        let info = OccupancyInfo(
            threadsPerThreadgroup: threadsPerThreadgroup.width
                * threadsPerThreadgroup.height
                * threadsPerThreadgroup.depth,
            maxTotalThreadsPerThreadgroup: pipeline.maxTotalThreadsPerThreadgroup,
            threadExecutionWidth: pipeline.threadExecutionWidth,
            threadgroupMemoryBytes: pipeline.staticThreadgroupMemoryLength,
            threadgroupMemoryLimitBytes: threadgroupMemoryLimitBytes > 0
                ? threadgroupMemoryLimitBytes : nil,
            threadgroupsPerGrid: threadgroupsPerGrid)
        occupancyObservations.append(info)
        return info
    }

    /// `dispatchThreads` plus an occupancy record. Sets the pipeline state so the
    /// recorded numbers describe the kernel that actually ran.
    public func dispatchThreads(_ encoder: MTLComputeCommandEncoder,
                                pipeline: MTLComputePipelineState,
                                threads: MTLSize,
                                threadsPerThreadgroup: MTLSize) {
        encoder.setComputePipelineState(pipeline)
        observe(pipeline: pipeline,
                threadsPerThreadgroup: threadsPerThreadgroup,
                threadgroupsPerGrid: CaptureRegion.threadgroupCount(threads: threads,
                                                                   threadsPerThreadgroup: threadsPerThreadgroup))
        encoder.dispatchThreads(threads, threadsPerThreadgroup: threadsPerThreadgroup)
    }

    /// `dispatchThreadgroups` plus an occupancy record.
    public func dispatchThreadgroups(_ encoder: MTLComputeCommandEncoder,
                                     pipeline: MTLComputePipelineState,
                                     threadgroupsPerGrid: MTLSize,
                                     threadsPerThreadgroup: MTLSize) {
        encoder.setComputePipelineState(pipeline)
        observe(pipeline: pipeline,
                threadsPerThreadgroup: threadsPerThreadgroup,
                threadgroupsPerGrid: threadgroupsPerGrid.width
                    * threadgroupsPerGrid.height
                    * threadgroupsPerGrid.depth)
        encoder.dispatchThreadgroups(threadgroupsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
    }

    /// Threadgroups a non-uniform `dispatchThreads` grid rounds up to.
    static func threadgroupCount(threads: MTLSize, threadsPerThreadgroup group: MTLSize) -> Int {
        func ceilDiv(_ a: Int, _ b: Int) -> Int { b > 0 ? (a + b - 1) / b : 0 }
        return ceilDiv(threads.width, group.width)
            * ceilDiv(threads.height, group.height)
            * ceilDiv(threads.depth, group.depth)
    }
}
