import XCTest
@testable import MetalscopeCore

final class DiffTests: XCTestCase {
    private let peaks = PeakSet(source: .measured, chip: "Test Chip",
                                fp32GFLOPS: 1000, fp16GFLOPS: 2000, bandwidthGBs: 100)

    private func record(_ label: String,
                        _ shape: KernelShape,
                        seconds: Double,
                        precision: Precision = .fp32) -> KernelRecord {
        KernelRecord(label: label, shape: shape, precision: precision,
                     durationSeconds: seconds, timingSource: .counterSampleBuffer)
    }

    private func trace(_ kernels: [KernelRecord], notes: [String: String]? = nil) -> Trace {
        Trace(device: DeviceInfo(name: "Test Chip"), peaks: peaks, kernels: kernels, notes: notes)
    }

    func testAlignmentMatchesByLabelShapeAndPrecision() {
        let a = [record("gemm", .gemm(m: 8, n: 8, k: 8), seconds: 2),
                 record("norm", .norm(n: 100), seconds: 1)]
        let b = [record("norm", .norm(n: 100), seconds: 0.5),      // reordered
                 record("gemm", .gemm(m: 8, n: 8, k: 8), seconds: 1)]
        let entries = TraceDiff.align(baseline: a, candidate: b)
        XCTAssertEqual(entries.count, 2)
        XCTAssertTrue(entries.allSatisfy { $0.status == .matched })
        // Baseline order is preserved regardless of candidate ordering.
        XCTAssertEqual(entries.map(\.label), ["gemm", "norm"])
        XCTAssertEqual(entries[0].candidate?.durationSeconds, 1)
    }

    func testShapeChangeBreaksAlignment() {
        let a = [record("gemm", .gemm(m: 8, n: 8, k: 8), seconds: 2)]
        let b = [record("gemm", .gemm(m: 16, n: 16, k: 16), seconds: 2)]
        let entries = TraceDiff.align(baseline: a, candidate: b)
        XCTAssertEqual(entries.count, 2)
        XCTAssertEqual(entries[0].status, .onlyInBaseline)
        XCTAssertEqual(entries[1].status, .onlyInCandidate)
        XCTAssertNil(entries[0].durationDeltaFraction)
    }

    func testPrecisionChangeBreaksAlignment() {
        let a = [record("gemm", .gemm(m: 8, n: 8, k: 8), seconds: 2, precision: .fp32)]
        let b = [record("gemm", .gemm(m: 8, n: 8, k: 8), seconds: 1, precision: .fp16)]
        let entries = TraceDiff.align(baseline: a, candidate: b)
        XCTAssertEqual(Set(entries.map(\.status)), [.onlyInBaseline, .onlyInCandidate])
    }

    func testRepeatedKernelsPairUpPositionally() {
        let a = [record("k", .norm(n: 10), seconds: 1),
                 record("k", .norm(n: 10), seconds: 2),
                 record("k", .norm(n: 10), seconds: 3)]
        let b = [record("k", .norm(n: 10), seconds: 10),
                 record("k", .norm(n: 10), seconds: 20)]
        let entries = TraceDiff.align(baseline: a, candidate: b)
        XCTAssertEqual(entries.count, 3)
        XCTAssertEqual(entries[0].candidate?.durationSeconds, 10)
        XCTAssertEqual(entries[1].candidate?.durationSeconds, 20)
        XCTAssertEqual(entries[2].status, .onlyInBaseline)
        XCTAssertEqual(entries.map(\.occurrence), [0, 1, 2])
    }

    func testExtraCandidateOccurrencesAppearAsCandidateOnly() {
        let a = [record("k", .norm(n: 10), seconds: 1)]
        let b = [record("k", .norm(n: 10), seconds: 1),
                 record("k", .norm(n: 10), seconds: 2)]
        let entries = TraceDiff.align(baseline: a, candidate: b)
        XCTAssertEqual(entries.count, 2)
        XCTAssertEqual(entries[0].status, .matched)
        XCTAssertEqual(entries[1].status, .onlyInCandidate)
        XCTAssertEqual(entries[1].occurrence, 1)
    }

    func testEmptyTracesAlignToNothing() {
        XCTAssertTrue(TraceDiff.align(baseline: [], candidate: []).isEmpty)
        XCTAssertEqual(TraceDiff.align(baseline: [], candidate: [record("k", .norm(n: 1), seconds: 1)]).count, 1)
    }

    func testDurationDeltasAndSpeedup() {
        let entries = TraceDiff.align(baseline: [record("k", .norm(n: 10), seconds: 2)],
                                      candidate: [record("k", .norm(n: 10), seconds: 1.5)])
        let entry = entries[0]
        XCTAssertEqual(entry.durationDeltaSeconds!, -0.5, accuracy: 1e-12)
        XCTAssertEqual(entry.durationDeltaFraction!, -0.25, accuracy: 1e-12)
        XCTAssertEqual(entry.speedup!, 4.0 / 3.0, accuracy: 1e-12)
    }

    func testEfficiencyDeltaInPercentagePoints() {
        // AI = 1 FLOP/byte (bandwidth-bound, ceiling 100 GFLOP/s).
        // 100 GFLOP in 2 s = 50 GFLOP/s = 50%; in 1 s = 100%.
        let baseline = KernelRecord(label: "k", shape: .opaque(flops: 100e9, bytes: 100e9),
                                    durationSeconds: 2, timingSource: .host)
        let candidate = KernelRecord(label: "k", shape: .opaque(flops: 100e9, bytes: 100e9),
                                     durationSeconds: 1, timingSource: .host)
        let entry = TraceDiff.align(baseline: [baseline], candidate: [candidate])[0]
        XCTAssertEqual(entry.efficiencyDeltaPoints(peaks: peaks)!, 50, accuracy: 1e-9)
        XCTAssertNil(entry.boundChange(peaks: peaks))   // same shape, same bound
    }

    func testBoundChangeIsReportedWhenIntensityMoves() {
        // Same label, but the candidate keeps more data on chip: AI 1 -> 100.
        let baseline = KernelRecord(label: "k", shape: .opaque(flops: 100e9, bytes: 100e9),
                                    durationSeconds: 1, timingSource: .host)
        let candidate = KernelRecord(label: "k", shape: .opaque(flops: 100e9, bytes: 1e9),
                                     durationSeconds: 1, timingSource: .host)
        // Different shapes don't align, so compare placements directly.
        XCTAssertEqual(Roofline.place(baseline, peaks: peaks).bound, .bandwidth)
        XCTAssertEqual(Roofline.place(candidate, peaks: peaks).bound, .compute)

        // A same-shape kernel that got faster keeps its bound type.
        let entry = TraceDiff.align(baseline: [baseline], candidate: [baseline])[0]
        XCTAssertNil(entry.boundChange(peaks: peaks))
    }

    func testUnmatchedEntriesHaveNoDeltas() {
        let entries = TraceDiff.align(baseline: [record("only-a", .norm(n: 1), seconds: 1)],
                                      candidate: [])
        XCTAssertNil(entries[0].durationDeltaSeconds)
        XCTAssertNil(entries[0].speedup)
        XCTAssertNil(entries[0].efficiencyDeltaPoints(peaks: peaks))
        XCTAssertNil(entries[0].boundChange(peaks: peaks))
        XCTAssertNotNil(entries[0].placements(peaks: peaks).baseline)
        XCTAssertNil(entries[0].placements(peaks: peaks).candidate)
    }

    // MARK: - Occupancy

    private func occupancy(_ threads: Int, max: Int = 1024, memory: Int = 0) -> OccupancyInfo {
        OccupancyInfo(threadsPerThreadgroup: threads,
                      maxTotalThreadsPerThreadgroup: max,
                      threadExecutionWidth: 32,
                      threadgroupMemoryBytes: memory,
                      threadgroupMemoryLimitBytes: 32768)
    }

    private func record(_ label: String, seconds: Double, occupancy: OccupancyInfo?) -> KernelRecord {
        KernelRecord(label: label, shape: .elementwise(n: 1024), precision: .fp32,
                     durationSeconds: seconds, timingSource: .counterSampleBuffer,
                     occupancy: occupancy)
    }

    func testOccupancyDeltasAcrossAFixedThreadgroup() {
        // The exact `bench baseline -> tuned` case: 100 ragged threads -> 256.
        let entry = TraceDiff.align(baseline: [record("scale", seconds: 2, occupancy: occupancy(100))],
                                    candidate: [record("scale", seconds: 1, occupancy: occupancy(256))])[0]
        XCTAssertTrue(entry.hasOccupancy)
        XCTAssertEqual(entry.threadgroupSizeDelta, 156)
        // 25% - 9.765625% = 15.234 pp of the pipeline max.
        XCTAssertEqual(entry.occupancyDeltaPoints!, 15.234375, accuracy: 1e-9)
        let change = try? XCTUnwrap(entry.limiterChange)
        XCTAssertEqual(change?.from, .executionWidthAlignment)
        XCTAssertEqual(change?.to, OccupancyLimiter.none)
    }

    func testUnchangedOccupancyReportsNoLimiterChange() {
        let entry = TraceDiff.align(baseline: [record("k", seconds: 2, occupancy: occupancy(256))],
                                    candidate: [record("k", seconds: 1, occupancy: occupancy(256))])[0]
        XCTAssertTrue(entry.hasOccupancy)
        XCTAssertEqual(entry.threadgroupSizeDelta, 0)
        XCTAssertEqual(entry.occupancyDeltaPoints!, 0, accuracy: 1e-12)
        XCTAssertNil(entry.limiterChange)
    }

    /// One side lacking occupancy (a v1 trace, or an MPS region) must not be
    /// treated as "unchanged" — that would report a fix that never happened.
    func testMissingOccupancyOnEitherSideYieldsNoDeltas() {
        let onlyBaseline = TraceDiff.align(baseline: [record("k", seconds: 2, occupancy: occupancy(100))],
                                           candidate: [record("k", seconds: 1, occupancy: nil)])[0]
        XCTAssertTrue(onlyBaseline.hasOccupancy)        // still worth a column
        XCTAssertNil(onlyBaseline.occupancies)
        XCTAssertNil(onlyBaseline.threadgroupSizeDelta)
        XCTAssertNil(onlyBaseline.occupancyDeltaPoints)
        XCTAssertNil(onlyBaseline.limiterChange)

        let neither = TraceDiff.align(baseline: [record("k", seconds: 2, occupancy: nil)],
                                      candidate: [record("k", seconds: 1, occupancy: nil)])[0]
        XCTAssertFalse(neither.hasOccupancy)
        XCTAssertNil(neither.threadgroupSizeDelta)
    }

    func testUnmatchedEntryCarriesNoOccupancyDeltasButStillReportsPresence() {
        let entries = TraceDiff.align(baseline: [record("gone", seconds: 1, occupancy: occupancy(100))],
                                      candidate: [])
        XCTAssertEqual(entries[0].status, .onlyInBaseline)
        XCTAssertTrue(entries[0].hasOccupancy)
        XCTAssertNil(entries[0].occupancyDeltaPoints)
    }

    func testLimiterChangeIsReportedWhenOnlyTheMemoryPressureMoves() {
        let entry = TraceDiff.align(
            baseline: [record("k", seconds: 1, occupancy: occupancy(256, memory: 20000))],
            candidate: [record("k", seconds: 1, occupancy: occupancy(256, memory: 4096))])[0]
        XCTAssertEqual(entry.threadgroupSizeDelta, 0)
        XCTAssertEqual(entry.limiterChange?.from, .threadgroupMemory)
        XCTAssertEqual(entry.limiterChange?.to, OccupancyLimiter.none)
    }

    // MARK: - Verdicts and the overlap-refusal rule

    private func repeated(_ label: String, samples: [Double]) -> KernelRecord {
        let sorted = samples.sorted()
        return KernelRecord(label: label, shape: .norm(n: 1024), precision: .fp32,
                            durationSeconds: sorted[(sorted.count - 1) / 2],
                            timingSource: .counterSampleBuffer,
                            durationSamplesSeconds: samples)
    }

    private func entry(baseline: KernelRecord, candidate: KernelRecord) -> DiffEntry {
        TraceDiff.align(baseline: [baseline], candidate: [candidate])[0]
    }

    /// The whole point: two captures of the *same* unchanged kernel, each noisy,
    /// must not produce a winner just because their medians differ.
    func testOverlappingSpreadsRefuseToCallAWinner() {
        // The §5.6 numbers: an unchanged RMS-norm baseline measured twice.
        let a = repeated("block.rmsnorm", samples: [77e-6, 195e-6, 91e-6, 120e-6, 88e-6])
        let b = repeated("block.rmsnorm", samples: [82e-6, 160e-6, 103e-6, 95e-6, 130e-6])
        let e = entry(baseline: a, candidate: b)
        XCTAssertEqual(e.verdict, .withinNoise)
        XCTAssertEqual(e.spreadsOverlap, true)
        XCTAssertTrue(e.hasRunStatistics)
        // The medians still differ and the speedup is still computed — the diff
        // reports the number and declines to interpret it.
        XCTAssertNotEqual(e.baseline?.durationSeconds, e.candidate?.durationSeconds)
        XCTAssertNotNil(e.speedup)
    }

    func testDisjointSpreadsCallTheCandidateFaster() {
        let a = repeated("k", samples: [200e-6, 210e-6, 205e-6, 220e-6, 215e-6])
        let b = repeated("k", samples: [100e-6, 104e-6, 101e-6, 110e-6, 106e-6])
        let e = entry(baseline: a, candidate: b)
        XCTAssertEqual(e.verdict, .faster)
        XCTAssertEqual(e.spreadsOverlap, false)
        XCTAssertEqual(e.verdict.displayName, "faster")
    }

    func testDisjointSpreadsCallTheCandidateSlower() {
        let a = repeated("k", samples: [100e-6, 104e-6, 101e-6, 110e-6, 106e-6])
        let b = repeated("k", samples: [200e-6, 210e-6, 205e-6, 220e-6, 215e-6])
        let e = entry(baseline: a, candidate: b)
        XCTAssertEqual(e.verdict, .slower)
        XCTAssertEqual(e.verdict.displayName, "slower")
    }

    /// A single-run capture on either side leaves nothing to compare a delta
    /// against, and the diff says so rather than treating one point as tight.
    func testSingleRunOnEitherSideWithholdsTheVerdict() {
        let repeatedSide = repeated("k", samples: [100e-6, 104e-6, 101e-6, 110e-6, 106e-6])
        let single = record("k", .norm(n: 1024), seconds: 500e-6)

        let candidateSingle = entry(baseline: repeatedSide, candidate: single)
        XCTAssertEqual(candidateSingle.verdict, .unmeasured)
        XCTAssertNil(candidateSingle.spreadsOverlap)
        XCTAssertNil(candidateSingle.runStatistics)
        XCTAssertTrue(candidateSingle.hasRunStatistics)   // one side still earns a column

        let baselineSingle = entry(baseline: single, candidate: repeatedSide)
        XCTAssertEqual(baselineSingle.verdict, .unmeasured)

        let bothSingle = entry(baseline: single, candidate: record("k", .norm(n: 1024), seconds: 1e-6))
        XCTAssertEqual(bothSingle.verdict, .unmeasured)
        XCTAssertFalse(bothSingle.hasRunStatistics)
        XCTAssertEqual(bothSingle.verdict.displayName, "-")
    }

    /// A one-element samples array is still one run. It must not be dressed up
    /// as a distribution just because the field is present.
    func testAOneSampleArrayIsStillASingleRun() {
        let a = repeated("k", samples: [100e-6])
        let b = repeated("k", samples: [500e-6])
        let e = entry(baseline: a, candidate: b)
        XCTAssertEqual(a.runStatistics?.count, 1)
        XCTAssertEqual(e.verdict, .unmeasured)
        XCTAssertFalse(e.hasRunStatistics)
    }

    func testUnmatchedEntriesHaveNoVerdict() {
        let entries = TraceDiff.align(
            baseline: [repeated("gone", samples: [1e-6, 2e-6, 3e-6, 4e-6, 5e-6])],
            candidate: [])
        XCTAssertEqual(entries[0].status, .onlyInBaseline)
        XCTAssertEqual(entries[0].verdict, .unmeasured)
        XCTAssertNil(entries[0].spreadsOverlap)
    }

    func testTraceDiffPartitionsResolvedAndWithinNoise() {
        let a = trace([repeated("moved", samples: [200e-6, 210e-6, 205e-6, 220e-6, 215e-6]),
                       repeated("noise", samples: [77e-6, 195e-6, 91e-6, 120e-6, 88e-6])])
        let b = trace([repeated("moved", samples: [100e-6, 104e-6, 101e-6, 110e-6, 106e-6]),
                       repeated("noise", samples: [82e-6, 160e-6, 103e-6, 95e-6, 130e-6])])
        let diff = TraceDiff(baselineTrace: a, candidateTrace: b)
        XCTAssertEqual(diff.matched.count, 2)
        XCTAssertEqual(diff.resolved.map(\.label), ["moved"])
        XCTAssertEqual(diff.withinNoise.map(\.label), ["noise"])
    }

    /// The rule is printed with the table, so its wording is part of the
    /// interface rather than a comment.
    func testVerdictRuleNamesBothHalvesOfTheTest() {
        XCTAssertTrue(TraceDiff.verdictRule.contains("medians"))
        XCTAssertTrue(TraceDiff.verdictRule.contains("min-p95"))
        XCTAssertTrue(TraceDiff.verdictRule.contains("overlap"))
        XCTAssertTrue(TraceDiff.verdictRule.contains("no call"))
    }

    func testVerdictRawValuesAreStableForJSONConsumers() {
        XCTAssertEqual(DiffEntry.Verdict.faster.rawValue, "faster")
        XCTAssertEqual(DiffEntry.Verdict.slower.rawValue, "slower")
        XCTAssertEqual(DiffEntry.Verdict.withinNoise.rawValue, "within-noise")
        XCTAssertEqual(DiffEntry.Verdict.unmeasured.rawValue, "unmeasured")
        XCTAssertEqual(DiffEntry.Verdict.withinNoise.displayName, "no call")
    }

    func testTraceDiffPartitionsMatchedAndUnmatched() {
        let a = trace([record("shared", .norm(n: 10), seconds: 1),
                       record("gone", .norm(n: 20), seconds: 1)], notes: ["variant": "baseline"])
        let b = trace([record("shared", .norm(n: 10), seconds: 0.5),
                       record("new", .norm(n: 30), seconds: 1)], notes: ["variant": "tuned"])
        let diff = TraceDiff(baselineTrace: a, candidateTrace: b)
        XCTAssertEqual(diff.matched.count, 1)
        XCTAssertEqual(diff.unmatched.count, 2)
        XCTAssertEqual(diff.matched[0].speedup!, 2.0, accuracy: 1e-12)
        XCTAssertEqual(diff.baselineTrace.notes?["variant"], "baseline")
    }
}
