import Metal
import XCTest
@testable import MetalscopeCapture
@testable import MetalscopeCore

/// Per-counter-set resolvers, exercised against **synthetic resolved data**.
///
/// The point of this file: this M1 Pro exposes only the `timestamp` counter set,
/// so the `stageutilization` and `statistic` paths would otherwise be untested
/// code that quietly rots until someone runs metalscope on a chip that has them.
/// Feeding each resolver bytes shaped like its own C struct tests the decode
/// without the hardware.
final class CounterResolverTests: XCTestCase {
    /// Pack `values` as one resolved sample of `stride` bytes.
    private func sample(_ values: [UInt64], stride: Int) -> Data {
        var data = Data()
        for value in values { withUnsafeBytes(of: value.littleEndian) { data.append(contentsOf: $0) } }
        XCTAssertLessThanOrEqual(data.count, stride)
        data.append(contentsOf: [UInt8](repeating: 0, count: stride - data.count))
        return data
    }

    private func samples(_ rows: [[UInt64]], stride: Int) -> Data {
        rows.reduce(into: Data()) { $0.append(sample($1, stride: stride)) }
    }

    // MARK: - Layout

    /// The shared field decode is only valid because all three resolved structs
    /// are a flat run of `uint64_t`. If Apple ever adds a field (or padding) to
    /// one of them, this fails instead of the decode silently sliding by 8 bytes.
    func testEveryResolverAccountsForItsWholeStruct() {
        for resolver in CounterResolvers.all {
            XCTAssertTrue(resolver.layoutIsConsistent,
                          "\(resolver.counterSetName): stride \(resolver.resolvedStride) "
                            + "vs \(resolver.counterNames.count) declared counters")
        }
    }

    func testStridesMatchTheMetalStructs() {
        XCTAssertEqual(TimestampCounterResolver().resolvedStride,
                       MemoryLayout<MTLCounterResultTimestamp>.stride)
        XCTAssertEqual(StageUtilizationCounterResolver().resolvedStride,
                       MemoryLayout<MTLCounterResultStageUtilization>.stride)
        XCTAssertEqual(StatisticCounterResolver().resolvedStride,
                       MemoryLayout<MTLCounterResultStatistic>.stride)
    }

    func testResolverRegistryIsKeyedByTheCommonCounterSetNames() {
        XCTAssertNotNil(CounterResolvers.resolver(for: "timestamp"))
        XCTAssertNotNil(CounterResolvers.resolver(for: "stageutilization"))
        XCTAssertNotNil(CounterResolvers.resolver(for: "statistic"))
        XCTAssertNil(CounterResolvers.resolver(for: "someFutureAppleCounterSet"))
        XCTAssertEqual(CounterResolvers.auxiliary.map(\.counterSetName),
                       ["stageutilization", "statistic"])
        // The names must be the ones MTLCounterSet.name actually reports.
        XCTAssertEqual(TimestampCounterResolver().counterSetName,
                       MTLCommonCounterSet.timestamp.rawValue)
        XCTAssertEqual(TimestampCounterResolver().counterNames,
                       [MTLCommonCounter.timestamp.rawValue])
    }

    // MARK: - Timestamp

    func testTimestampResolverKeepsFullTickPrecision() {
        let resolver = TimestampCounterResolver()
        // At 2^53: a Double round trip collapses these two onto the same value,
        // which is how a real duration turns into zero after ~104 days of uptime.
        let a: UInt64 = 9_007_199_254_740_992      // 2^53
        let b: UInt64 = 9_007_199_254_740_993      // 2^53 + 1
        let data = samples([[a], [b]], stride: resolver.resolvedStride)
        let ticks = resolver.timestamps(data)
        XCTAssertEqual(ticks, [a, b])
        XCTAssertEqual(ticks[1] - ticks[0], 1)
        XCTAssertEqual(Double(a), Double(b))    // the loss this avoids
    }

    func testTimestampResolverAlsoDecodesThroughTheGenericPath() {
        let resolver = TimestampCounterResolver()
        let data = samples([[1000], [4000]], stride: resolver.resolvedStride)
        let decoded = resolver.decode(data)
        XCTAssertEqual(decoded.count, 2)
        XCTAssertEqual(decoded[0][MTLCommonCounter.timestamp.rawValue], 1000)
        XCTAssertEqual(decoded[1][MTLCommonCounter.timestamp.rawValue], 4000)
    }

    // MARK: - Stage utilization (absent on this chip)

    func testStageUtilizationDecodesItsSixFieldsInOrder() {
        let resolver = StageUtilizationCounterResolver()
        let data = samples([[1000, 10, 20, 30, 40, 50]], stride: resolver.resolvedStride)
        let decoded = resolver.decode(data)
        XCTAssertEqual(decoded.count, 1)
        XCTAssertEqual(decoded[0]["TotalCycles"], 1000)
        XCTAssertEqual(decoded[0]["VertexCycles"], 10)
        XCTAssertEqual(decoded[0]["TessellationCycles"], 20)
        // Note the name: Apple's constant is postTessellationVertexCycles but the
        // string it carries is "PostTessellationCycles".
        XCTAssertEqual(decoded[0][MTLCommonCounter.postTessellationVertexCycles.rawValue], 30)
        XCTAssertEqual(decoded[0]["PostTessellationCycles"], 30)
        XCTAssertEqual(decoded[0]["FragmentCycles"], 40)
        XCTAssertEqual(decoded[0]["RenderTargetWriteCycles"], 50)
    }

    func testStatisticDecodesItsEightFieldsInOrder() {
        let resolver = StatisticCounterResolver()
        let data = samples([[1, 2, 3, 4, 5, 6, 7, 8]], stride: resolver.resolvedStride)
        let decoded = resolver.decode(data)
        XCTAssertEqual(decoded[0]["TessellationInputPatches"], 1)
        XCTAssertEqual(decoded[0]["VertexInvocations"], 2)
        XCTAssertEqual(decoded[0]["FragmentsPassed"], 7)
        // The one that matters for a compute profiler.
        XCTAssertEqual(decoded[0][StatisticCounterResolver.computeKernelInvocations], 8)
    }

    func testCounterErrorValueIsOmittedRatherThanReportedAsAHugeNumber() {
        let resolver = StageUtilizationCounterResolver()
        let data = samples([[1000, MTLCounterErrorValue, 20, 30, 40, 50]],
                           stride: resolver.resolvedStride)
        let decoded = resolver.decode(data)
        XCTAssertEqual(decoded[0]["TotalCycles"], 1000)
        // Absent, not 1.8e19 — a missing key means "the GPU didn't write this".
        XCTAssertNil(decoded[0]["VertexCycles"])
        XCTAssertEqual(decoded[0].values.count, 5)
    }

    func testTruncatedOrEmptyBuffersDecodeToWhateverWholeSamplesExist() {
        let resolver = StatisticCounterResolver()
        XCTAssertTrue(resolver.decode(Data()).isEmpty)
        // One and a half samples' worth of bytes yields exactly one sample.
        var data = samples([[1, 2, 3, 4, 5, 6, 7, 8]], stride: resolver.resolvedStride)
        data.append(contentsOf: [UInt8](repeating: 0, count: resolver.resolvedStride / 2))
        XCTAssertEqual(resolver.decode(data).count, 1)
    }

    // MARK: - Aggregation

    func testDeltaSubtractsMatchingCountersOnly() {
        let start = ResolvedCounterSample(values: ["TotalCycles": 1000, "VertexCycles": 10])
        let end = ResolvedCounterSample(values: ["TotalCycles": 2500, "FragmentCycles": 7])
        let delta = CounterAggregation.delta(from: start, to: end)
        XCTAssertEqual(delta, ["TotalCycles": 1500])
    }

    func testDecreasingCountersAreDroppedRatherThanReportedNegative() {
        let start = ResolvedCounterSample(values: ["TotalCycles": 5000])
        let end = ResolvedCounterSample(values: ["TotalCycles": 100])   // wrapped
        XCTAssertTrue(CounterAggregation.delta(from: start, to: end).isEmpty)
    }

    func testPerEncoderDeltasSumIntoARegionTotal() {
        let total = CounterAggregation.sum([["TotalCycles": 100, "FragmentCycles": 1],
                                            ["TotalCycles": 250],
                                            [:]])
        XCTAssertEqual(total["TotalCycles"], 350)
        XCTAssertEqual(total["FragmentCycles"], 1)
    }

    func testUtilizationFractionsUseTotalCyclesAsTheDenominator() {
        let totals: [String: Double] = ["TotalCycles": 1000, "FragmentCycles": 250, "VertexCycles": 0]
        let withFractions = CounterAggregation.utilizationFractions(totals)
        XCTAssertEqual(withFractions["FragmentCyclesFraction"]!, 0.25, accuracy: 1e-12)
        XCTAssertEqual(withFractions["VertexCyclesFraction"]!, 0, accuracy: 1e-12)
        XCTAssertEqual(withFractions["TotalCycles"], 1000)
        XCTAssertNil(withFractions["TotalCyclesFraction"])          // 100% of itself is noise
    }

    func testUtilizationFractionsRefuseToDivideByAMissingOrZeroTotal() {
        XCTAssertEqual(CounterAggregation.utilizationFractions(["FragmentCycles": 250]),
                       ["FragmentCycles": 250])
        XCTAssertEqual(CounterAggregation.utilizationFractions(["TotalCycles": 0, "FragmentCycles": 5]),
                       ["TotalCycles": 0, "FragmentCycles": 5])
    }

    func testNamespacingKeepsTwoSetsFromCollidingInOneRecord() {
        let stage = CounterAggregation.namespaced(["TotalCycles": 5], set: "stageutilization")
        let statistic = CounterAggregation.namespaced(["KernelInvocations": 9], set: "statistic")
        let merged = stage.merging(statistic) { a, _ in a }
        XCTAssertEqual(merged["stageutilization.TotalCycles"], 5)
        XCTAssertEqual(merged["statistic.KernelInvocations"], 9)
        XCTAssertEqual(merged.count, 2)
    }

    /// The full path an M4 would exercise: two encoders sampled start/end into a
    /// `stageutilization` buffer, resolved and folded into `KernelRecord.counters`.
    func testEndToEndAggregationOfSyntheticStageUtilizationSamples() {
        let resolver = StageUtilizationCounterResolver()
        // start0, end0, start1, end1
        let data = samples([[1000, 0, 0, 0, 100, 0],
                            [1600, 0, 0, 0, 250, 0],
                            [2000, 0, 0, 0, 300, 0],
                            [2400, 0, 0, 0, 350, 0]],
                           stride: resolver.resolvedStride)
        let decoded = resolver.decode(data)
        XCTAssertEqual(decoded.count, 4)
        let deltas = [CounterAggregation.delta(from: decoded[0], to: decoded[1]),
                      CounterAggregation.delta(from: decoded[2], to: decoded[3])]
        let totals = CounterAggregation.utilizationFractions(CounterAggregation.sum(deltas))
        let namespaced = CounterAggregation.namespaced(totals, set: resolver.counterSetName)
        XCTAssertEqual(namespaced["stageutilization.TotalCycles"], 1000)     // 600 + 400
        XCTAssertEqual(namespaced["stageutilization.FragmentCycles"], 200)   // 150 + 50
        XCTAssertEqual(namespaced["stageutilization.FragmentCyclesFraction"]!, 0.2, accuracy: 1e-12)

        // ...and that dictionary is exactly what a v2 KernelRecord carries.
        let record = KernelRecord(label: "k", shape: .norm(n: 10), durationSeconds: 1,
                                  timingSource: .counterSampleBuffer, counters: namespaced)
        XCTAssertEqual(record.counters?["stageutilization.TotalCycles"], 1000)
    }

    // MARK: - Capabilities

    func testCapabilitiesSeparatePresentSetsFromKnownButAbsentOnes() throws {
        try XCTSkipIf(MTLCreateSystemDefaultDevice() == nil, "no Metal device available")
        let device = MTLCreateSystemDefaultDevice()!
        let capabilities = CaptureCapabilities(device: device)

        // Whatever this chip exposes, the two lists must partition the known sets.
        let present = Set(capabilities.counterSetNames)
        let absent = Set(capabilities.absentKnownCounterSetNames)
        XCTAssertTrue(present.isDisjoint(with: absent))
        XCTAssertTrue(absent.isSubset(of: Set(CounterResolvers.all.map(\.counterSetName))))

        for set in capabilities.counterSets where set.hasResolver {
            XCTAssertNotNil(CounterResolvers.resolver(for: set.name))
        }
        // Auxiliary sets are the decodable ones that aren't the timing set.
        XCTAssertFalse(capabilities.auxiliaryCounterSetNames.contains("timestamp"))
        XCTAssertGreaterThan(capabilities.maxThreadgroupMemoryBytes, 0)
        XCTAssertEqual(capabilities.timingLadderTier,
                       capabilities.canSampleEncoderStages ? .counterSampleBuffer : .commandBuffer)
    }
}
