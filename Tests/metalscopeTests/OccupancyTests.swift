import XCTest
@testable import MetalscopeCore

/// Static occupancy math. No Metal here: everything is derived from four numbers
/// a pipeline hands over, which is exactly why this analysis works on chips that
/// expose no counters at all.
final class OccupancyTests: XCTestCase {
    private func info(threads: Int,
                      max: Int = 1024,
                      width: Int = 32,
                      memory: Int = 0,
                      limit: Int? = 32768,
                      grid: Int? = nil) -> OccupancyInfo {
        OccupancyInfo(threadsPerThreadgroup: threads,
                      maxTotalThreadsPerThreadgroup: max,
                      threadExecutionWidth: width,
                      threadgroupMemoryBytes: memory,
                      threadgroupMemoryLimitBytes: limit,
                      threadgroupsPerGrid: grid)
    }

    // MARK: - SIMD groups and lanes

    func testSimdGroupCountRoundsUp() {
        XCTAssertEqual(info(threads: 256).simdGroupsPerThreadgroup, 8)
        XCTAssertEqual(info(threads: 32).simdGroupsPerThreadgroup, 1)
        // The point of rounding up: 100 threads still costs four full SIMD groups.
        XCTAssertEqual(info(threads: 100).simdGroupsPerThreadgroup, 4)
        XCTAssertEqual(info(threads: 1).simdGroupsPerThreadgroup, 1)
        // A non-32 execution width is read, never assumed.
        XCTAssertEqual(info(threads: 100, width: 64).simdGroupsPerThreadgroup, 2)
    }

    func testIdleLanesAndLaneUtilization() {
        let aligned = info(threads: 256)
        XCTAssertEqual(aligned.idleLanesPerThreadgroup, 0)
        XCTAssertEqual(aligned.laneUtilization, 1.0, accuracy: 1e-12)
        XCTAssertTrue(aligned.isExecutionWidthAligned)

        let ragged = info(threads: 100)
        XCTAssertFalse(ragged.isExecutionWidthAligned)
        XCTAssertEqual(ragged.idleLanesPerThreadgroup, 28)          // 4 x 32 - 100
        XCTAssertEqual(ragged.laneUtilization, 100.0 / 128.0, accuracy: 1e-12)
    }

    func testThreadgroupOccupancyIsDispatchedOverPipelineMax() {
        XCTAssertEqual(info(threads: 256, max: 1024).threadgroupOccupancy, 0.25, accuracy: 1e-12)
        XCTAssertEqual(info(threads: 1024, max: 1024).threadgroupOccupancy, 1.0, accuracy: 1e-12)
        // A pipeline whose register pressure caps it low isn't penalised for it.
        XCTAssertEqual(info(threads: 256, max: 256).threadgroupOccupancy, 1.0, accuracy: 1e-12)
    }

    func testDegenerateInputsDoNotProduceNaNOrDivideByZero() {
        let zeroed = OccupancyInfo(threadsPerThreadgroup: 0,
                                   maxTotalThreadsPerThreadgroup: 0,
                                   threadExecutionWidth: 0,
                                   threadgroupMemoryBytes: 0,
                                   threadgroupMemoryLimitBytes: 0)
        XCTAssertEqual(zeroed.threadgroupOccupancy, 0)
        XCTAssertEqual(zeroed.simdGroupsPerThreadgroup, 1)
        XCTAssertFalse(zeroed.laneUtilization.isNaN)
        XCTAssertNil(zeroed.threadgroupMemoryPressure)
        XCTAssertNil(zeroed.memoryLimitedThreadgroups)
    }

    // MARK: - Threadgroup memory

    func testThreadgroupMemoryPressureAndResidencyBound() {
        let heavy = info(threads: 256, memory: 24576, limit: 32768)
        XCTAssertEqual(heavy.threadgroupMemoryPressure!, 0.75, accuracy: 1e-12)
        XCTAssertEqual(heavy.memoryLimitedThreadgroups, 1)

        let light = info(threads: 256, memory: 128, limit: 32768)
        XCTAssertEqual(light.threadgroupMemoryPressure!, 128.0 / 32768.0, accuracy: 1e-12)
        XCTAssertEqual(light.memoryLimitedThreadgroups, 256)

        // No threadgroup memory means no residency claim to make.
        XCTAssertNil(info(threads: 256, memory: 0).memoryLimitedThreadgroups)
        // No known device limit means no pressure fraction to invent.
        XCTAssertNil(info(threads: 256, memory: 4096, limit: nil).threadgroupMemoryPressure)
    }

    // MARK: - Limiter classification

    func testWellShapedDispatchHasNoLimiterAndNoHint() {
        let good = info(threads: 256, memory: 128)
        XCTAssertEqual(good.limiter, .none)
        XCTAssertNil(good.hint)
        // A quarter of the pipeline max is a perfectly good threadgroup: the
        // occupancy *ratio* alone must never trigger a hint.
        XCTAssertEqual(good.threadgroupOccupancy, 0.25, accuracy: 1e-12)
    }

    func testRaggedThreadgroupIsFlaggedWithRoundingAdvice() {
        let ragged = info(threads: 100)
        XCTAssertEqual(ragged.limiter, .executionWidthAlignment)
        let hint = try? XCTUnwrap(ragged.hint)
        XCTAssertTrue(hint?.contains("28 of 128 lanes idle") ?? false, hint ?? "no hint")
        // Advice must name reachable sizes on both sides.
        XCTAssertEqual(ragged.nearestAlignedBelow, 96)
        XCTAssertEqual(ragged.nearestAlignedAbove, 128)
    }

    func testSubSimdThreadgroupGetsASingleRoundingTarget() throws {
        // 20 threads: below and above both land on the SIMD width, so the advice
        // must not read "round to 32 or 32".
        let sliver = info(threads: 20)
        XCTAssertEqual(sliver.limiter, .executionWidthAlignment)
        XCTAssertEqual(sliver.nearestAlignedBelow, 32)
        XCTAssertEqual(sliver.nearestAlignedAbove, 32)
        let hint = try XCTUnwrap(sliver.hint)
        XCTAssertTrue(hint.hasSuffix("round to 32"), hint)
    }

    func testRoundingAdviceStaysInsideThePipelineCeiling() {
        // 1000 threads of a 1024 max: rounding up lands exactly on the ceiling.
        XCTAssertEqual(info(threads: 1000, max: 1024).nearestAlignedAbove, 1024)
        // A pipeline capped at 512 must never be told to try 1024.
        XCTAssertEqual(info(threads: 500, max: 512).nearestAlignedAbove, 512)
    }

    func testSingleSimdGroupIsFlaggedAsTiny() {
        let tiny = info(threads: 32)
        XCTAssertEqual(tiny.limiter, .tinyThreadgroup)
        XCTAssertNotNil(tiny.hint)
        // Two SIMD groups is enough to stop complaining.
        XCTAssertEqual(info(threads: 64).limiter, .none)
    }

    func testThreadgroupMemoryPressureIsFlaggedOnlyWhenDominant() {
        XCTAssertEqual(info(threads: 256, memory: 20000).limiter, .threadgroupMemory)
        XCTAssertEqual(info(threads: 256, memory: 4096).limiter, .none)
        XCTAssertTrue(info(threads: 256, memory: 20000).hint?.contains("threadgroup memory") ?? false)
    }

    func testAlignmentOutranksEveryOtherLimiter() {
        // Ragged *and* memory-heavy: the alignment defect is the one to fix first,
        // and it's the only one that's unambiguously a defect.
        let both = info(threads: 100, memory: 20000)
        XCTAssertEqual(both.limiter, .executionWidthAlignment)
        XCTAssertGreaterThan(OccupancyLimiter.executionWidthAlignment.severity,
                             OccupancyLimiter.threadgroupMemory.severity)
        XCTAssertGreaterThan(OccupancyLimiter.tinyThreadgroup.severity,
                             OccupancyLimiter.threadgroupMemory.severity)
        XCTAssertEqual(OccupancyLimiter.none.severity, 0)
    }

    // MARK: - Folding a region's dispatches

    func testFoldOfNothingIsNil() {
        XCTAssertNil(OccupancyInfo.fold([]))
    }

    func testIdenticalDispatchesFoldToOneShapeWithACount() {
        let folded = try? XCTUnwrap(OccupancyInfo.fold(Array(repeating: info(threads: 256, grid: 64), count: 120)))
        XCTAssertEqual(folded?.threadsPerThreadgroup, 256)
        XCTAssertEqual(folded?.dispatchCount, 120)
        XCTAssertEqual(folded?.variantCount, 1)
        // threadgroupsPerGrid stays per-dispatch — summing it would describe a
        // grid that never ran.
        XCTAssertEqual(folded?.threadgroupsPerGrid, 64)
    }

    func testMixedShapesKeepTheWorstAndSayHowManyWereDropped() throws {
        let folded = try XCTUnwrap(OccupancyInfo.fold([info(threads: 256),
                                                       info(threads: 100),
                                                       info(threads: 512)]))
        XCTAssertEqual(folded.threadsPerThreadgroup, 100)   // the ragged one
        XCTAssertEqual(folded.limiter, .executionWidthAlignment)
        XCTAssertEqual(folded.variantCount, 3)
        XCTAssertEqual(folded.dispatchCount, 3)
    }

    func testWithNoDefectsTheLeastOccupiedShapeWins() throws {
        let folded = try XCTUnwrap(OccupancyInfo.fold([info(threads: 512), info(threads: 64)]))
        XCTAssertEqual(folded.threadsPerThreadgroup, 64)
        XCTAssertEqual(folded.limiter, .none)
        XCTAssertEqual(folded.variantCount, 2)
    }

    func testFoldIsDeterministicAcrossOrderings() throws {
        let shapes = [info(threads: 256), info(threads: 100), info(threads: 64), info(threads: 512)]
        let forward = try XCTUnwrap(OccupancyInfo.fold(shapes))
        let backward = try XCTUnwrap(OccupancyInfo.fold(shapes.reversed()))
        XCTAssertEqual(forward, backward)
    }

    func testTableTextStarsOnlyRaggedThreadgroups() {
        XCTAssertEqual(info(threads: 256).tableText, "256")
        XCTAssertEqual(info(threads: 100).tableText, "100*")
    }

    // MARK: - Serialization

    func testOccupancySurvivesAJSONRoundTrip() throws {
        let original = info(threads: 100, max: 1024, width: 32, memory: 128, limit: 32768, grid: 4096)
        let data = try TraceIO.makeEncoder().encode(original)
        let decoded = try TraceIO.makeDecoder().decode(OccupancyInfo.self, from: data)
        XCTAssertEqual(original, decoded)
        // Derived values are recomputed, never stored — so they cannot drift.
        let json = String(data: data, encoding: .utf8) ?? ""
        XCTAssertFalse(json.contains("laneUtilization"))
        XCTAssertFalse(json.contains("simdGroups"))
        XCTAssertEqual(decoded.laneUtilization, 100.0 / 128.0, accuracy: 1e-12)
    }
}
