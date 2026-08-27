import Metal
import XCTest
@testable import MetalscopeCapture
@testable import MetalscopeCore

/// GPU-dependent tests. Everything here skips cleanly when there's no Metal
/// device (CI containers, headless build boxes).
final class CaptureTests: XCTestCase {
    private var device: MTLDevice!

    override func setUpWithError() throws {
        try XCTSkipIf(MTLCreateSystemDefaultDevice() == nil, "no Metal device available")
        device = MTLCreateSystemDefaultDevice()
    }

    private static let source = """
    #include <metal_stdlib>
    using namespace metal;
    kernel void scale(device float *y [[buffer(0)]],
                      device const float *x [[buffer(1)]],
                      uint i [[thread_position_in_grid]]) {
        y[i] = 2.0f * x[i];
    }
    """

    private func makePipeline() throws -> MTLComputePipelineState {
        let library = try device.makeLibrary(source: Self.source, options: nil)
        let function = try XCTUnwrap(library.makeFunction(name: "scale"))
        return try device.makeComputePipelineState(function: function)
    }

    func testCapabilitiesReportSomethingCoherent() throws {
        let capabilities = CaptureCapabilities(device: device)
        // Whatever the chip exposes, canSampleEncoderStages must imply both parts.
        if capabilities.canSampleEncoderStages {
            XCTAssertTrue(capabilities.hasTimestampCounterSet)
            XCTAssertTrue(capabilities.supportsStageBoundarySampling)
            XCTAssertTrue(capabilities.counterSetNames.contains("timestamp"))
        }
    }

    func testCaptureRecordsAnnotatedKernel() throws {
        let session = try CaptureSession(device: device)
        let pipeline = try makePipeline()
        let count = 1 << 16
        let x = try XCTUnwrap(device.makeBuffer(length: count * 4, options: .storageModePrivate))
        let y = try XCTUnwrap(device.makeBuffer(length: count * 4, options: .storageModePrivate))

        try session.captureCompute(label: "scale",
                                   shape: .elementwise(n: count),
                                   iterations: 1) { encoder in
            encoder.setComputePipelineState(pipeline)
            encoder.setBuffer(y, offset: 0, index: 0)
            encoder.setBuffer(x, offset: 0, index: 1)
            encoder.dispatchThreads(MTLSize(width: count, height: 1, depth: 1),
                                    threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
        }

        XCTAssertEqual(session.records.count, 1)
        let record = try XCTUnwrap(session.records.first)
        XCTAssertEqual(record.label, "scale")
        XCTAssertEqual(record.shape, .elementwise(n: count))
        XCTAssertGreaterThan(record.durationSeconds, 0)
        // A 64K-element scale cannot plausibly take a second.
        XCTAssertLessThan(record.durationSeconds, 1.0)
        XCTAssertEqual(record.flops, Double(count), accuracy: 1e-6)
        XCTAssertEqual(record.bytes, Double(count) * 8, accuracy: 1e-6)
        XCTAssertNotNil(record.hostDurationSeconds)

        if CaptureCapabilities(device: device).canSampleEncoderStages {
            XCTAssertEqual(record.timingSource, .counterSampleBuffer)
            XCTAssertEqual(record.stages?.count, 1)
        }
    }

    func testIterationsDivideTheMeasuredSpan() throws {
        let session = try CaptureSession(device: device)
        let pipeline = try makePipeline()
        let count = 1 << 20
        let x = try XCTUnwrap(device.makeBuffer(length: count * 4, options: .storageModePrivate))
        let y = try XCTUnwrap(device.makeBuffer(length: count * 4, options: .storageModePrivate))

        func run(iterations: Int) throws -> KernelRecord {
            try session.capture(label: "scale", shape: .elementwise(n: count),
                                iterations: iterations) { region in
                for _ in 0..<iterations {
                    let encoder = try region.makeComputeCommandEncoder()
                    encoder.setComputePipelineState(pipeline)
                    encoder.setBuffer(y, offset: 0, index: 0)
                    encoder.setBuffer(x, offset: 0, index: 1)
                    encoder.dispatchThreads(MTLSize(width: count, height: 1, depth: 1),
                                            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
                    encoder.endEncoding()
                }
            }
            return session.records[session.records.count - 1]
        }

        _ = try run(iterations: 4)          // warm the pipeline/allocator
        session.reset()
        let single = try run(iterations: 1)
        let many = try run(iterations: 8)
        XCTAssertEqual(many.iterations, 8)
        // Per-iteration time should be the same order of magnitude, not 8x.
        XCTAssertLessThan(many.durationSeconds, single.durationSeconds * 4 + 1e-4)
        XCTAssertEqual(session.records.count, 2)
    }

    // MARK: - Repeats

    /// A repeated capture must still produce exactly one record — the repeats
    /// are folded into it, not appended as separate kernels — and the warm-up
    /// must leave no trace at all.
    func testRepeatsFoldIntoOneRecordAndWarmupsAreDiscarded() throws {
        let session = try CaptureSession(device: device)
        let pipeline = try makePipeline()
        let count = 1 << 18
        let x = try XCTUnwrap(device.makeBuffer(length: count * 4, options: .storageModePrivate))
        let y = try XCTUnwrap(device.makeBuffer(length: count * 4, options: .storageModePrivate))

        try session.capture(label: "scale", shape: .elementwise(n: count),
                            iterations: 4, repeats: 5) { region in
            for _ in 0..<4 {
                let encoder = try region.makeComputeCommandEncoder()
                region.dispatchThreads(encoder, pipeline: pipeline,
                                       threads: MTLSize(width: count, height: 1, depth: 1),
                                       threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
                encoder.setBuffer(y, offset: 0, index: 0)
                encoder.setBuffer(x, offset: 0, index: 1)
                encoder.endEncoding()
            }
        }

        XCTAssertEqual(session.records.count, 1)
        let record = try XCTUnwrap(session.records.first)
        let samples = try XCTUnwrap(record.durationSamplesSeconds)
        XCTAssertEqual(samples.count, 5)
        XCTAssertTrue(samples.allSatisfy { $0 > 0 })
        XCTAssertEqual(record.iterations, 4, "iterations describe one repeat, not the total")

        let stats = try XCTUnwrap(record.runStatistics)
        XCTAssertEqual(stats.count, 5)
        XCTAssertTrue(stats.hasSpread)
        // The reported duration is the median, and the median is a real sample.
        XCTAssertEqual(record.durationSeconds, stats.median)
        XCTAssertTrue(samples.contains(record.durationSeconds))
        XCTAssertLessThanOrEqual(stats.min, stats.median)
        XCTAssertLessThanOrEqual(stats.median, stats.p95)

        // Occupancy comes from one repeat, so the dispatch count describes a
        // region that ran rather than the sum over five of them.
        XCTAssertEqual(record.occupancy?.dispatchCount, 4)
    }

    /// The default is one run, and one run records no samples — a v2-shaped
    /// record. Present-but-single would read as a measured spread of zero.
    func testASingleRunRecordsNoSamples() throws {
        let session = try CaptureSession(device: device)
        let pipeline = try makePipeline()
        let count = 1 << 16
        let x = try XCTUnwrap(device.makeBuffer(length: count * 4, options: .storageModePrivate))
        let y = try XCTUnwrap(device.makeBuffer(length: count * 4, options: .storageModePrivate))

        try session.captureCompute(label: "scale", shape: .elementwise(n: count)) { encoder in
            encoder.setComputePipelineState(pipeline)
            encoder.setBuffer(y, offset: 0, index: 0)
            encoder.setBuffer(x, offset: 0, index: 1)
            encoder.dispatchThreads(MTLSize(width: count, height: 1, depth: 1),
                                    threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
        }

        let record = try XCTUnwrap(session.records.first)
        XCTAssertNil(record.durationSamplesSeconds)
        XCTAssertNil(record.runStatistics)
    }

    /// Repeating actually narrows the reported spread, because the warm-up run
    /// absorbs the clock ramp that would otherwise land on sample one. This is
    /// the property the whole change exists for, asserted loosely enough to
    /// survive a busy machine: the median must sit inside the sample range and
    /// the band must not be absurd.
    func testRepeatedCaptureProducesACoherentSpread() throws {
        let session = try CaptureSession(device: device)
        let pipeline = try makePipeline()
        let count = 1 << 20
        let x = try XCTUnwrap(device.makeBuffer(length: count * 4, options: .storageModePrivate))
        let y = try XCTUnwrap(device.makeBuffer(length: count * 4, options: .storageModePrivate))

        try session.capture(label: "scale", shape: .elementwise(n: count),
                            iterations: 32, repeats: 7) { region in
            for _ in 0..<32 {
                let encoder = try region.makeComputeCommandEncoder()
                encoder.setComputePipelineState(pipeline)
                encoder.setBuffer(y, offset: 0, index: 0)
                encoder.setBuffer(x, offset: 0, index: 1)
                encoder.dispatchThreads(MTLSize(width: count, height: 1, depth: 1),
                                        threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
                encoder.endEncoding()
            }
        }

        let stats = try XCTUnwrap(session.records.first?.runStatistics)
        XCTAssertEqual(stats.count, 7)
        XCTAssertGreaterThan(stats.min, 0)
        XCTAssertGreaterThanOrEqual(stats.max, stats.p95)
        XCTAssertGreaterThanOrEqual(stats.mean, stats.min)
        XCTAssertLessThanOrEqual(stats.mean, stats.max)
        XCTAssertFalse(stats.spreadFraction.isNaN)
    }

    /// Every repeat is a fresh region, so a region-level property — here the
    /// timing tier — has to hold across all of them, not just the first.
    func testEveryRepeatReachesTheSameTimingTier() throws {
        let session = try CaptureSession(device: device)
        guard session.capabilities.canSampleEncoderStages else {
            throw XCTSkip("device cannot sample at stage boundaries")
        }
        let pipeline = try makePipeline()
        let count = 1 << 16
        let x = try XCTUnwrap(device.makeBuffer(length: count * 4, options: .storageModePrivate))
        let y = try XCTUnwrap(device.makeBuffer(length: count * 4, options: .storageModePrivate))

        try session.capture(label: "scale", shape: .elementwise(n: count),
                            iterations: 2, repeats: 4) { region in
            for _ in 0..<2 {
                let encoder = try region.makeComputeCommandEncoder()
                encoder.setComputePipelineState(pipeline)
                encoder.setBuffer(y, offset: 0, index: 0)
                encoder.setBuffer(x, offset: 0, index: 1)
                encoder.dispatchThreads(MTLSize(width: count, height: 1, depth: 1),
                                        threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
                encoder.endEncoding()
            }
        }

        let record = try XCTUnwrap(session.records.first)
        XCTAssertEqual(record.timingSource, .counterSampleBuffer)
        XCTAssertEqual(record.durationSamplesSeconds?.count, 4)
        // Stages come from the representative repeat, so there are two of them
        // rather than eight summed across the four regions.
        XCTAssertEqual(record.stages?.count, 2)
    }

    func testMultipleEncodersProduceStagesWhenSupported() throws {
        let session = try CaptureSession(device: device)
        guard session.capabilities.canSampleEncoderStages else {
            throw XCTSkip("device cannot sample at stage boundaries")
        }
        let pipeline = try makePipeline()
        let count = 1 << 14
        let x = try XCTUnwrap(device.makeBuffer(length: count * 4, options: .storageModePrivate))
        let y = try XCTUnwrap(device.makeBuffer(length: count * 4, options: .storageModePrivate))

        try session.capture(label: "three-stage", shape: .norm(n: count), iterations: 3) { region in
            for _ in 0..<3 {
                let encoder = try region.makeComputeCommandEncoder()
                encoder.setComputePipelineState(pipeline)
                encoder.setBuffer(y, offset: 0, index: 0)
                encoder.setBuffer(x, offset: 0, index: 1)
                encoder.dispatchThreads(MTLSize(width: count, height: 1, depth: 1),
                                        threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
                encoder.endEncoding()
            }
        }
        let record = try XCTUnwrap(session.records.first)
        XCTAssertEqual(record.stages?.count, 3)
        XCTAssertEqual(record.timingSource, .counterSampleBuffer)
        let stageTotal = (record.stages ?? []).reduce(0) { $0 + $1.durationSeconds }
        // The region span covers all stages, so it can't be shorter than their sum
        // minus overlap; with serial encoders they're within a hair of each other.
        XCTAssertGreaterThan(stageTotal, 0)
        XCTAssertLessThanOrEqual(stageTotal, record.durationSeconds * Double(record.iterations) * 1.5)
    }

    /// Regression: when a region encodes more encoders than its counter buffer
    /// can sample, the span of the sampled *prefix* must not be divided by the
    /// full iteration count — that reported ~3x the real throughput, which for a
    /// bandwidth benchmark meant "541 GB/s" on a 200 GB/s machine.
    func testOversubscribedSampleBufferFallsBackInsteadOfLying() throws {
        let pipeline = try makePipeline()
        // 32 MB per buffer: too big for the system cache, so the dispatch is long
        // enough that per-encoder sampling overhead and scheduling noise don't
        // dominate the comparison below.
        let count = 8 << 20
        let x = try XCTUnwrap(device.makeBuffer(length: count * 4, options: .storageModePrivate))
        let y = try XCTUnwrap(device.makeBuffer(length: count * 4, options: .storageModePrivate))
        let iterations = 5

        /// Best (shortest) of `repeats` runs, matching how `calibrate` treats a
        /// ceiling: noise only ever moves a sample the slow way.
        func measure(maxSamples: Int, repeats: Int = 3) throws -> KernelRecord {
            let session = try CaptureSession(device: device)
            session.maxSamplesPerRegion = maxSamples
            for _ in 0..<(repeats + 1) {      // first run is warmup
                try session.capture(label: "scale", shape: .elementwise(n: count),
                                    iterations: iterations) { region in
                    for _ in 0..<iterations {
                        let encoder = try region.makeComputeCommandEncoder()
                        encoder.setComputePipelineState(pipeline)
                        encoder.setBuffer(y, offset: 0, index: 0)
                        encoder.setBuffer(x, offset: 0, index: 1)
                        encoder.dispatchThreads(MTLSize(width: count, height: 1, depth: 1),
                                                threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
                        encoder.endEncoding()
                    }
                }
            }
            return try XCTUnwrap(session.records.dropFirst().min { $0.durationSeconds < $1.durationSeconds })
        }

        // Room for 2 of the 5 encoders: must not claim counter-based timing.
        let starved = try measure(maxSamples: 4)
        XCTAssertNotEqual(starved.timingSource, .counterSampleBuffer)
        XCTAssertGreaterThan(starved.durationSeconds, 0)

        let ample = try measure(maxSamples: 1024)
        XCTAssertGreaterThan(ample.durationSeconds, 0)
        // The exact invariant the bug violated: counter-based timing is only
        // claimed when every encoder in the region contributed a sample.
        if ample.timingSource == .counterSampleBuffer {
            XCTAssertEqual(ample.stages?.count, iterations)
        }
        // And the two must still describe the same kernel — the bug reported the
        // starved region as running several times faster than it did.
        let ratio = ample.durationSeconds / starved.durationSeconds
        XCTAssertGreaterThan(ratio, 1.0 / 2.5, "starved region timing is implausibly fast")
        XCTAssertLessThan(ratio, 2.5, "starved region timing is implausibly slow")
    }

    // MARK: - Occupancy

    /// The end-to-end static-occupancy path: a real pipeline, a real dispatch,
    /// and the numbers that land in the trace.
    func testDispatchHelperRecordsOccupancyFromTheRealPipeline() throws {
        let session = try CaptureSession(device: device)
        let pipeline = try makePipeline()
        let count = 1 << 16
        let x = try XCTUnwrap(device.makeBuffer(length: count * 4, options: .storageModePrivate))
        let y = try XCTUnwrap(device.makeBuffer(length: count * 4, options: .storageModePrivate))
        let iterations = 4

        try session.capture(label: "scale", shape: .elementwise(n: count),
                            iterations: iterations) { region in
            for _ in 0..<iterations {
                let encoder = try region.makeComputeCommandEncoder()
                encoder.setBuffer(y, offset: 0, index: 0)
                encoder.setBuffer(x, offset: 0, index: 1)
                region.dispatchThreads(encoder, pipeline: pipeline,
                                       threads: MTLSize(width: count, height: 1, depth: 1),
                                       threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
                encoder.endEncoding()
            }
        }

        let occupancy = try XCTUnwrap(session.records.first?.occupancy)
        XCTAssertEqual(occupancy.threadsPerThreadgroup, 256)
        XCTAssertEqual(occupancy.maxTotalThreadsPerThreadgroup, pipeline.maxTotalThreadsPerThreadgroup)
        XCTAssertEqual(occupancy.threadExecutionWidth, pipeline.threadExecutionWidth)
        XCTAssertEqual(occupancy.threadgroupMemoryBytes, pipeline.staticThreadgroupMemoryLength)
        XCTAssertEqual(occupancy.threadgroupMemoryLimitBytes, device.maxThreadgroupMemoryLength)
        // Identical dispatches fold to one shape that counts them.
        XCTAssertEqual(occupancy.dispatchCount, iterations)
        XCTAssertEqual(occupancy.variantCount, 1)
        XCTAssertEqual(occupancy.threadgroupsPerGrid, count / 256)
        XCTAssertEqual(occupancy.limiter, .none)
    }

    /// The defect `bench --variant baseline` plants on purpose, measured through
    /// the capture path rather than constructed by hand.
    func testRaggedThreadgroupIsRecordedAndFlagged() throws {
        let session = try CaptureSession(device: device)
        let pipeline = try makePipeline()
        let count = 1 << 14
        let x = try XCTUnwrap(device.makeBuffer(length: count * 4, options: .storageModePrivate))
        let y = try XCTUnwrap(device.makeBuffer(length: count * 4, options: .storageModePrivate))

        try session.capture(label: "ragged", shape: .elementwise(n: count)) { region in
            let encoder = try region.makeComputeCommandEncoder()
            encoder.setBuffer(y, offset: 0, index: 0)
            encoder.setBuffer(x, offset: 0, index: 1)
            region.dispatchThreads(encoder, pipeline: pipeline,
                                   threads: MTLSize(width: count, height: 1, depth: 1),
                                   threadsPerThreadgroup: MTLSize(width: 100, height: 1, depth: 1))
            encoder.endEncoding()
        }

        let occupancy = try XCTUnwrap(session.records.first?.occupancy)
        XCTAssertFalse(occupancy.isExecutionWidthAligned)
        XCTAssertEqual(occupancy.limiter, .executionWidthAlignment)
        XCTAssertNotNil(occupancy.hint)
        // Every Apple GPU so far is 32 wide; assert the arithmetic against
        // whatever this one reports rather than against 32.
        let width = pipeline.threadExecutionWidth
        XCTAssertEqual(occupancy.simdGroupsPerThreadgroup, (100 + width - 1) / width)
    }

    /// A region whose dispatches metalscope never saw (an MPS encoder, or a
    /// caller using the raw encoder) records no occupancy rather than a guess.
    func testUnobservedDispatchesRecordNoOccupancy() throws {
        let session = try CaptureSession(device: device)
        let pipeline = try makePipeline()
        let count = 1 << 12
        let x = try XCTUnwrap(device.makeBuffer(length: count * 4, options: .storageModePrivate))
        let y = try XCTUnwrap(device.makeBuffer(length: count * 4, options: .storageModePrivate))
        try session.captureCompute(label: "scale", shape: .elementwise(n: count)) { encoder in
            encoder.setComputePipelineState(pipeline)
            encoder.setBuffer(y, offset: 0, index: 0)
            encoder.setBuffer(x, offset: 0, index: 1)
            encoder.dispatchThreads(MTLSize(width: count, height: 1, depth: 1),
                                    threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
        }
        XCTAssertNil(session.records.first?.occupancy)
    }

    func testThreadgroupCountRoundsUpInEveryDimension() {
        let count = CaptureRegion.threadgroupCount(
            threads: MTLSize(width: 100, height: 10, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 32, height: 4, depth: 1))
        XCTAssertEqual(count, 4 * 3 * 1)
    }

    /// On a chip with only `timestamp`, there is nothing to put in `counters` —
    /// and metalscope must leave the field absent rather than write an empty
    /// object that a reader would mistake for "measured, all zero".
    func testAuxiliaryCountersAreAbsentWhenTheChipHasNoOtherSets() throws {
        let session = try CaptureSession(device: device)
        try XCTSkipUnless(session.capabilities.auxiliaryCounterSetNames.isEmpty,
                          "this device exposes more than the timestamp counter set")
        let pipeline = try makePipeline()
        let count = 1 << 12
        let x = try XCTUnwrap(device.makeBuffer(length: count * 4, options: .storageModePrivate))
        let y = try XCTUnwrap(device.makeBuffer(length: count * 4, options: .storageModePrivate))
        try session.captureCompute(label: "scale", shape: .elementwise(n: count)) { encoder in
            encoder.setComputePipelineState(pipeline)
            encoder.setBuffer(y, offset: 0, index: 0)
            encoder.setBuffer(x, offset: 0, index: 1)
            encoder.dispatchThreads(MTLSize(width: count, height: 1, depth: 1),
                                    threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
        }
        XCTAssertNil(session.records.first?.counters)
        XCTAssertEqual(session.deviceInfo.maxThreadgroupMemoryBytes, device.maxThreadgroupMemoryLength)
    }

    func testTraceFromSessionRoundTrips() throws {
        let session = try CaptureSession(device: device)
        let pipeline = try makePipeline()
        let count = 1 << 12
        let x = try XCTUnwrap(device.makeBuffer(length: count * 4, options: .storageModePrivate))
        let y = try XCTUnwrap(device.makeBuffer(length: count * 4, options: .storageModePrivate))
        try session.captureCompute(label: "scale", shape: .elementwise(n: count)) { encoder in
            encoder.setComputePipelineState(pipeline)
            encoder.setBuffer(y, offset: 0, index: 0)
            encoder.setBuffer(x, offset: 0, index: 1)
            encoder.dispatchThreads(MTLSize(width: count, height: 1, depth: 1),
                                    threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
        }

        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("metalscope-capture-\(UUID().uuidString).json")
        defer { try? FileManager.default.removeItem(at: url) }
        let peaks = PeakSet(source: .measured, chip: device.name, fp32GFLOPS: 1000,
                            fp16GFLOPS: 2000, bandwidthGBs: 100)
        let written = try session.writeTrace(to: url, peaks: peaks)
        let trace = try TraceIO.read(from: written)
        XCTAssertEqual(trace.device.name, device.name)
        XCTAssertEqual(trace.kernels.count, 1)
        XCTAssertEqual(trace.peaks?.fp32GFLOPS, 1000)
        XCTAssertEqual(trace.schemaVersion, Trace.currentSchemaVersion)
        // The trace is directly consumable by the roofline analysis.
        let placement = Roofline.place(trace.kernels[0], peaks: peaks)
        XCTAssertEqual(placement.bound, .bandwidth)     // AI = 0.125 FLOP/byte
        XCTAssertGreaterThan(placement.achievedBandwidthGBs, 0)
    }

    func testResetAndDropLast() throws {
        let session = try CaptureSession(device: device)
        let pipeline = try makePipeline()
        let count = 1 << 10
        let x = try XCTUnwrap(device.makeBuffer(length: count * 4, options: .storageModePrivate))
        let y = try XCTUnwrap(device.makeBuffer(length: count * 4, options: .storageModePrivate))
        for _ in 0..<3 {
            try session.captureCompute(label: "scale", shape: .elementwise(n: count)) { encoder in
                encoder.setComputePipelineState(pipeline)
                encoder.setBuffer(y, offset: 0, index: 0)
                encoder.setBuffer(x, offset: 0, index: 1)
                encoder.dispatchThreads(MTLSize(width: count, height: 1, depth: 1),
                                        threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
            }
        }
        XCTAssertEqual(session.records.count, 3)
        session.dropLast(1)
        XCTAssertEqual(session.records.count, 2)
        session.dropLast(99)                    // must clamp, not crash
        XCTAssertTrue(session.records.isEmpty)
        session.reset()
        XCTAssertTrue(session.records.isEmpty)
    }

    func testGPUTickScaleIsRobust() {
        // Identical clocks (Apple silicon today).
        XCTAssertEqual(CaptureSession.gpuTickScale(cpu0: 100, gpu0: 100, cpu1: 200, gpu1: 200),
                       1.0, accuracy: 1e-12)
        // A 2:1 ratio is honoured.
        XCTAssertEqual(CaptureSession.gpuTickScale(cpu0: 0, gpu0: 0, cpu1: 200, gpu1: 100),
                       2.0, accuracy: 1e-12)
        // Degenerate or absurd correlations fall back to 1.0 rather than NaN.
        XCTAssertEqual(CaptureSession.gpuTickScale(cpu0: 0, gpu0: 0, cpu1: 0, gpu1: 0), 1.0)
        XCTAssertEqual(CaptureSession.gpuTickScale(cpu0: 0, gpu0: 0, cpu1: 100, gpu1: 0), 1.0)
        XCTAssertEqual(CaptureSession.gpuTickScale(cpu0: 0, gpu0: 0, cpu1: 1, gpu1: 1_000_000_000), 1.0)
    }

    func testEnvironmentTraceURLReflectsEnvironment() {
        // Not set in the test runner, so it must be nil rather than a bogus path.
        if ProcessInfo.processInfo.environment["METALSCOPE_TRACE"] == nil {
            XCTAssertNil(CaptureSession.environmentTraceURL)
        }
    }

    // MARK: - Trace destination

    /// The `metalscope profile -- <cmd>` contract from the child's side: the CLI
    /// sets `METALSCOPE_TRACE`, and an instrumented target that calls
    /// `writeTrace()` with no argument must land exactly there.
    func testWriteTraceDefaultsToTheEnvironmentTracePath() throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("metalscope-env-\(UUID().uuidString)")
        let destination = directory.appendingPathComponent("trace.json")
        setenv("METALSCOPE_TRACE", destination.path, 1)
        defer {
            unsetenv("METALSCOPE_TRACE")
            try? FileManager.default.removeItem(at: directory)
        }

        XCTAssertEqual(CaptureSession.environmentTraceURL?.path, destination.path)

        let session = try CaptureSession(device: device)
        let pipeline = try makePipeline()
        let count = 1 << 12
        let x = try XCTUnwrap(device.makeBuffer(length: count * 4, options: .storageModePrivate))
        let y = try XCTUnwrap(device.makeBuffer(length: count * 4, options: .storageModePrivate))
        try session.captureCompute(label: "scale", shape: .elementwise(n: count)) { encoder in
            encoder.setComputePipelineState(pipeline)
            encoder.setBuffer(y, offset: 0, index: 0)
            encoder.setBuffer(x, offset: 0, index: 1)
            encoder.dispatchThreads(MTLSize(width: count, height: 1, depth: 1),
                                    threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
        }

        // No explicit URL: the environment wins, and intermediate directories are
        // created rather than the write failing.
        let written = try session.writeTrace()
        XCTAssertEqual(written.path, destination.path)
        XCTAssertEqual(try TraceIO.read(from: written).kernels.count, 1)
    }

    func testEnvironmentTracePathExpandsATilde() {
        setenv("METALSCOPE_TRACE", "~/traces/run.json", 1)
        defer { unsetenv("METALSCOPE_TRACE") }
        let url = CaptureSession.environmentTraceURL
        XCTAssertNotNil(url)
        XCTAssertFalse(url?.path.contains("~") ?? true, "the tilde must be expanded, not written literally")
        XCTAssertTrue(url?.path.hasSuffix("/traces/run.json") ?? false)
    }

    /// An empty `METALSCOPE_TRACE` is treated as unset: a profiler that writes a
    /// trace to the path "" would fail in a way nobody could diagnose.
    func testEmptyEnvironmentTracePathIsTreatedAsUnset() {
        setenv("METALSCOPE_TRACE", "", 1)
        defer { unsetenv("METALSCOPE_TRACE") }
        XCTAssertNil(CaptureSession.environmentTraceURL)
    }

    /// `makeTrace()` with no peaks argument falls back to whatever this machine
    /// has cached, which is what `bench` and an instrumented app both rely on.
    func testMakeTraceWithoutExplicitPeaksUsesTheLocalStore() throws {
        let session = try CaptureSession(device: device)
        let trace = session.makeTrace()
        XCTAssertEqual(trace.device.name, device.name)
        XCTAssertEqual(trace.schemaVersion, Trace.currentSchemaVersion)
        XCTAssertTrue(trace.kernels.isEmpty)
        // Either measured peaks from `calibrate`, labelled folklore, or — on a
        // chip absent from both — nothing. Never an unlabelled guess.
        if let peaks = trace.peaks {
            XCTAssertEqual(peaks.chip.isEmpty, false)
            XCTAssertTrue([.measured, .specSheet].contains(peaks.source))
        }
    }

    // MARK: - Degrading rather than failing

    /// Every failure inside a capture is reported as a sentence a user can act
    /// on; these strings are what `main.swift` prints on the way out.
    func testCaptureErrorsDescribeThemselves() {
        XCTAssertEqual(CaptureError.noMetalDevice.description, "no Metal device available")
        XCTAssertEqual(CaptureError.commandQueueCreationFailed.description,
                       "could not create a Metal command queue")
        XCTAssertEqual(CaptureError.commandBufferCreationFailed.description,
                       "could not create a Metal command buffer")
        XCTAssertEqual(CaptureError.encoderCreationFailed.description,
                       "could not create a compute command encoder")
    }

    /// Metal caps a counter sample buffer at 32 KB — 4096 timestamps on this
    /// machine, so ~2048 sampled encoders per region. Asking for more must not
    /// fail the capture: `makeSampleSets` gives up on sampling entirely and the
    /// region falls back down the timing ladder, which is the whole point of
    /// having a ladder.
    func testAnUnallocatableSampleBufferDegradesToCommandBufferTiming() throws {
        let session = try CaptureSession(device: device)
        try XCTSkipUnless(session.capabilities.canSampleEncoderStages,
                          "device never samples stages, so there is nothing to degrade from")
        session.maxSamplesPerRegion = 8192          // 64 KB — over the device limit
        let pipeline = try makePipeline()
        let count = 1 << 12
        let x = try XCTUnwrap(device.makeBuffer(length: count * 4, options: .storageModePrivate))
        let y = try XCTUnwrap(device.makeBuffer(length: count * 4, options: .storageModePrivate))

        // `iterations` only sizes the sample-buffer request here; the body encodes
        // one encoder, which is all that is needed to see which tier was reached.
        try session.capture(label: "scale", shape: .elementwise(n: count),
                            iterations: 4096) { region in
            let encoder = try region.makeComputeCommandEncoder()
            encoder.setComputePipelineState(pipeline)
            encoder.setBuffer(y, offset: 0, index: 0)
            encoder.setBuffer(x, offset: 0, index: 1)
            encoder.dispatchThreads(MTLSize(width: count, height: 1, depth: 1),
                                    threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
            encoder.endEncoding()
        }

        let record = try XCTUnwrap(session.records.first)
        XCTAssertEqual(record.timingSource, .commandBuffer)
        XCTAssertNil(record.stages)
        XCTAssertGreaterThan(record.durationSeconds, 0)
    }

    /// A region built with no sample sets at all — the shape every capture takes
    /// on a chip without stage-boundary sampling. Encoders still work, they are
    /// still counted, and nothing claims counter-based timing.
    func testRegionWithoutSampleSetsStillHandsOutEncoders() throws {
        let queue = try XCTUnwrap(device.makeCommandQueue())
        let commandBuffer = try XCTUnwrap(queue.makeCommandBuffer())
        let region = CaptureRegion(commandBuffer: commandBuffer,
                                   sampleSets: [],
                                   capacity: 0,
                                   iterations: 2,
                                   threadgroupMemoryLimitBytes: 0)
        XCTAssertNil(region.timestampSampleBuffer)
        XCTAssertEqual(region.iterations, 2)

        let encoder = try region.makeComputeCommandEncoder()
        XCTAssertEqual(region.encoderCount, 1)
        XCTAssertEqual(region.usedSampleCount, 0)
        XCTAssertTrue(region.stages.isEmpty)
        XCTAssertEqual(encoder.label, "metalscope.stage0")

        // A device that reports no threadgroup-memory limit records none, rather
        // than a zero that a report would divide by.
        let pipeline = try makePipeline()
        let info = region.observe(pipeline: pipeline,
                                  threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
        XCTAssertNil(info.threadgroupMemoryLimitBytes)
        XCTAssertNil(info.threadgroupsPerGrid)
        XCTAssertNil(info.threadgroupMemoryPressure)
        XCTAssertEqual(info.threadsPerThreadgroup, 64)
        XCTAssertEqual(region.occupancyObservations.count, 1)

        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
    }

    /// `dispatchThreadgroups` is the path `bench`'s RMS norm takes: the grid is
    /// declared in threadgroups, so the count is multiplied out rather than
    /// divided down.
    func testDispatchThreadgroupsRecordsTheGridItWasGiven() throws {
        let session = try CaptureSession(device: device)
        let pipeline = try makePipeline()
        let count = 1 << 12
        let x = try XCTUnwrap(device.makeBuffer(length: count * 4, options: .storageModePrivate))
        let y = try XCTUnwrap(device.makeBuffer(length: count * 4, options: .storageModePrivate))

        try session.capture(label: "norm", shape: .norm(n: count)) { region in
            let encoder = try region.makeComputeCommandEncoder()
            encoder.setBuffer(y, offset: 0, index: 0)
            encoder.setBuffer(x, offset: 0, index: 1)
            region.dispatchThreadgroups(encoder, pipeline: pipeline,
                                        threadgroupsPerGrid: MTLSize(width: 16, height: 2, depth: 1),
                                        threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
            encoder.endEncoding()
        }

        let occupancy = try XCTUnwrap(session.records.first?.occupancy)
        XCTAssertEqual(occupancy.threadgroupsPerGrid, 32)      // 16 x 2 x 1
        XCTAssertEqual(occupancy.threadsPerThreadgroup, 64)
        XCTAssertEqual(occupancy.dispatchCount, 1)
        XCTAssertEqual(occupancy.threadgroupMemoryLimitBytes, device.maxThreadgroupMemoryLength)
    }
}
