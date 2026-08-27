import XCTest
@testable import MetalscopeCore

/// The arithmetic behind every spread the report prints and every verdict the
/// diff withholds. Percentile conventions are the sort of thing that quietly
/// differs between two implementations, so each one is pinned here.
final class RunStatisticsTests: XCTestCase {
    private func stats(_ samples: [Double]) throws -> RunStatistics {
        try XCTUnwrap(RunStatistics(samples: samples))
    }

    // MARK: - Construction

    func testEmptySamplesYieldNoStatistics() {
        XCTAssertNil(RunStatistics(samples: []))
    }

    /// Non-positive and non-finite samples are not measurements. Dropping them
    /// is right; averaging them in would poison every number in the row.
    func testNonMeasurementsAreDiscardedAndAnAllBadSetYieldsNil() throws {
        let s = try stats([1, 0, -3, .nan, .infinity, 2])
        XCTAssertEqual(s.count, 2)
        XCTAssertEqual(s.min, 1)
        XCTAssertEqual(s.max, 2)
        XCTAssertNil(RunStatistics(samples: [0, -1, .nan]))
    }

    func testSingleSampleHasNoSpread() throws {
        let s = try stats([0.000_1])
        XCTAssertEqual(s.count, 1)
        XCTAssertFalse(s.hasSpread)
        XCTAssertEqual(s.min, 0.000_1)
        XCTAssertEqual(s.median, 0.000_1)
        XCTAssertEqual(s.mean, 0.000_1)
        XCTAssertEqual(s.p95, 0.000_1)
        XCTAssertEqual(s.max, 0.000_1)
        XCTAssertEqual(s.spreadFraction, 0, accuracy: 1e-12)
    }

    func testSamplesNeedNotArriveSorted() throws {
        XCTAssertEqual(try stats([5, 1, 4, 2, 3]), try stats([1, 2, 3, 4, 5]))
    }

    // MARK: - Median

    func testMedianOfAnOddCountIsTheMiddleSample() throws {
        XCTAssertEqual(try stats([3, 1, 2]).median, 2)
        XCTAssertEqual(try stats([9, 1, 2, 3, 100]).median, 3)
    }

    /// The lower median, deliberately: the reported duration must be a run that
    /// actually happened, not the average of two that did.
    func testMedianOfAnEvenCountIsTheLowerMiddleSample() throws {
        XCTAssertEqual(try stats([1, 2, 3, 4]).median, 2)
        XCTAssertEqual(try stats([10, 20]).median, 10)
    }

    func testMedianIsAlwaysOneOfTheSamples() throws {
        for n in 1...12 {
            let samples = (0..<n).map { Double($0) * 1.7 + 1 }.shuffled()
            let s = try stats(samples)
            XCTAssertTrue(samples.contains(s.median), "median of \(n) samples must be observed")
        }
    }

    // MARK: - p95

    /// Nearest-rank: index ceil(0.95 n) - 1, with no interpolation between
    /// samples. With five repeats that is the slowest of the five.
    func testP95IsNearestRankAndNeverInterpolated() throws {
        XCTAssertEqual(try stats([1, 2, 3, 4, 5]).p95, 5)
        XCTAssertEqual(try stats([1, 2]).p95, 2)
        XCTAssertEqual(try stats([7]).p95, 7)
        // n = 20: ceil(19) - 1 = index 18, the second-slowest.
        let twenty = (1...20).map(Double.init)
        XCTAssertEqual(try stats(twenty).p95, 19)
        // n = 100: ceil(95) - 1 = index 94.
        let hundred = (1...100).map(Double.init)
        XCTAssertEqual(try stats(hundred).p95, 95)
    }

    func testP95IsAlwaysOneOfTheSamplesAndNeverBelowTheMedian() throws {
        for n in 1...40 {
            let samples = (0..<n).map { _ in Double.random(in: 1...100) }
            let s = try stats(samples)
            XCTAssertTrue(samples.contains(s.p95))
            XCTAssertGreaterThanOrEqual(s.p95, s.median)
            XCTAssertLessThanOrEqual(s.p95, s.max)
            XCTAssertGreaterThanOrEqual(s.median, s.min)
        }
    }

    // MARK: - Mean and spread

    func testMeanIsTheArithmeticMean() throws {
        XCTAssertEqual(try stats([1, 2, 3, 4]).mean, 2.5, accuracy: 1e-12)
        // The mean is the one summary a single outlier drags, which is why the
        // report leads with the median instead.
        XCTAssertEqual(try stats([1, 1, 1, 1, 96]).mean, 20, accuracy: 1e-12)
        XCTAssertEqual(try stats([1, 1, 1, 1, 96]).median, 1)
    }

    /// The §5.6 measurement that motivated all of this: an unchanged RMS-norm
    /// baseline read between 77 and 195 µs across five runs.
    func testSpreadFractionOfTheNoisyBaselineFromTheWhitepaper() throws {
        let s = try stats([77e-6, 195e-6, 91e-6, 120e-6, 88e-6])
        XCTAssertEqual(s.count, 5)
        XCTAssertEqual(s.min, 77e-6, accuracy: 1e-12)
        XCTAssertEqual(s.median, 91e-6, accuracy: 1e-12)
        XCTAssertEqual(s.p95, 195e-6, accuracy: 1e-12)
        // (195 - 77) / 91 = 1.297 — a band wider than the median itself.
        XCTAssertEqual(s.spreadFraction, (195.0 - 77) / 91, accuracy: 1e-9)
    }

    func testSpreadFractionIsZeroForIdenticalSamples() throws {
        XCTAssertEqual(try stats([2, 2, 2, 2]).spreadFraction, 0, accuracy: 1e-12)
    }

    // MARK: - Overlap, the refusal rule

    func testDisjointIntervalsDoNotOverlap() throws {
        let slow = try stats([100, 105, 110, 108, 102])   // min 100, p95 110
        let fast = try stats([10, 12, 11, 13, 10])        // min 10,  p95 13
        XCTAssertFalse(slow.overlaps(fast))
        XCTAssertFalse(fast.overlaps(slow))
    }

    func testOverlappingIntervalsOverlapInBothDirections() throws {
        let a = try stats([77, 195, 91, 120, 88])         // 77 .. 195
        let b = try stats([80, 150, 95, 110, 99])         // 80 .. 150
        XCTAssertTrue(a.overlaps(b))
        XCTAssertTrue(b.overlaps(a))
    }

    /// Touching at a single point counts as overlapping. The rule exists to
    /// refuse marginal calls, so the boundary case resolves toward refusing.
    func testIntervalsThatMeetAtAPointOverlap() throws {
        let a = try stats([10, 20, 15, 18, 12])           // 10 .. 20
        let b = try stats([20, 30, 25, 28, 22])           // 20 .. 30
        XCTAssertEqual(a.p95, 20)
        XCTAssertEqual(b.min, 20)
        XCTAssertTrue(a.overlaps(b))
    }

    /// One side entirely inside the other is the classic phantom delta: the
    /// medians differ, and the difference is noise.
    func testAnIntervalContainedInAnotherOverlaps() throws {
        let wide = try stats([50, 250, 100, 200, 150])    // 50 .. 250
        let narrow = try stats([120, 130, 125, 128, 122]) // 120 .. 130
        XCTAssertTrue(wide.overlaps(narrow))
        XCTAssertTrue(narrow.overlaps(wide))
    }

    func testIntervalRunsFromMinToP95NotToMax() throws {
        let s = try stats([10, 11, 12, 13, 14, 15, 16, 17, 18, 900])
        XCTAssertEqual(s.max, 900)
        // n = 10: ceil(9.5) - 1 = index 9 — with ten samples p95 *is* the max.
        XCTAssertEqual(s.p95, 900)
        XCTAssertEqual(s.interval.lowerBound, 10)

        // With twenty, the single outlier falls outside the interval.
        let twenty = try stats((1...19).map(Double.init) + [900])
        XCTAssertEqual(twenty.max, 900)
        XCTAssertEqual(twenty.p95, 19)
        XCTAssertEqual(twenty.interval, 1...19)
    }
}
