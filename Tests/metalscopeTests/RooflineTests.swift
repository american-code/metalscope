import XCTest
@testable import MetalscopeCore

final class RooflineTests: XCTestCase {
    /// 1 TFLOP/s over 100 GB/s: ridge at 10 FLOP/byte.
    private let peaks = PeakSet(source: .measured,
                                chip: "Test Chip",
                                fp32GFLOPS: 1000,
                                fp16GFLOPS: 2000,
                                bandwidthGBs: 100)

    func testRidgePoint() {
        XCTAssertEqual(peaks.ridgePoint(for: .fp32), 10, accuracy: 1e-9)
        // fp16 doubles the compute roof but not the memory roof.
        XCTAssertEqual(peaks.ridgePoint(for: .fp16), 20, accuracy: 1e-9)
    }

    func testRidgePointWithoutMeasuredHalfPeakFallsBackToFP32() {
        let noHalf = PeakSet(source: .measured, chip: "x", fp32GFLOPS: 1000,
                             fp16GFLOPS: nil, bandwidthGBs: 100)
        XCTAssertEqual(noHalf.peakGFLOPS(for: .fp16), 1000, accuracy: 1e-9)
        XCTAssertEqual(noHalf.ridgePoint(for: .fp16), 10, accuracy: 1e-9)
    }

    func testBandwidthBoundClassificationAndCeiling() {
        // AI = 1 FLOP/byte, well left of the ridge.
        // 1e9 bytes in 1 s at 1 FLOP/byte -> 1 GFLOP/s achieved, ceiling 100 GFLOP/s.
        let placement = Roofline.place(flops: 1e9, bytes: 1e9, seconds: 1,
                                       precision: .fp32, peaks: peaks)
        XCTAssertEqual(placement.arithmeticIntensity, 1, accuracy: 1e-9)
        XCTAssertEqual(placement.bound, .bandwidth)
        XCTAssertEqual(placement.ceilingGFLOPS, 100, accuracy: 1e-9)
        XCTAssertEqual(placement.achievedGFLOPS, 1, accuracy: 1e-9)
        XCTAssertEqual(placement.achievedBandwidthGBs, 1, accuracy: 1e-9)
        XCTAssertEqual(placement.efficiency, 0.01, accuracy: 1e-9)
    }

    func testComputeBoundClassificationAndCeiling() {
        // AI = 100 FLOP/byte, far right of the ridge; ceiling is the compute roof.
        let placement = Roofline.place(flops: 100e9, bytes: 1e9, seconds: 1,
                                       precision: .fp32, peaks: peaks)
        XCTAssertEqual(placement.bound, .compute)
        XCTAssertEqual(placement.ceilingGFLOPS, 1000, accuracy: 1e-9)
        XCTAssertEqual(placement.efficiency, 0.1, accuracy: 1e-9)
    }

    func testRidgeClassificationWithinTolerance() {
        let atRidge = Roofline.place(flops: 10e9, bytes: 1e9, seconds: 1,
                                     precision: .fp32, peaks: peaks)
        XCTAssertEqual(atRidge.bound, .ridge)

        // 4% below the ridge is still "ridge" at the 5% default tolerance...
        let nearRidge = Roofline.place(flops: 9.6e9, bytes: 1e9, seconds: 1,
                                       precision: .fp32, peaks: peaks)
        XCTAssertEqual(nearRidge.bound, .ridge)

        // ...but 10% below is bandwidth-bound.
        let below = Roofline.place(flops: 9e9, bytes: 1e9, seconds: 1,
                                   precision: .fp32, peaks: peaks)
        XCTAssertEqual(below.bound, .bandwidth)

        // Tolerance is configurable.
        let strict = Roofline.place(flops: 9.6e9, bytes: 1e9, seconds: 1,
                                    precision: .fp32, peaks: peaks, ridgeTolerance: 0.01)
        XCTAssertEqual(strict.bound, .bandwidth)
    }

    func testEfficiencyAtTheCeilingIsOneHundredPercent() {
        // Bandwidth-bound kernel running at exactly 100 GB/s.
        let saturated = Roofline.place(flops: 100e9, bytes: 100e9, seconds: 1,
                                       precision: .fp32, peaks: peaks)
        XCTAssertEqual(saturated.efficiency, 1.0, accuracy: 1e-9)
        // Compute-bound kernel running at exactly 1 TFLOP/s.
        let peaked = Roofline.place(flops: 1000e9, bytes: 1e9, seconds: 1,
                                    precision: .fp32, peaks: peaks)
        XCTAssertEqual(peaked.efficiency, 1.0, accuracy: 1e-9)
    }

    func testHalfPrecisionUsesHalfPeakCeiling() {
        // AI = 100: compute-bound in both precisions, but fp16's roof is 2x.
        let fp32 = Roofline.place(flops: 100e9, bytes: 1e9, seconds: 1,
                                  precision: .fp32, peaks: peaks)
        let fp16 = Roofline.place(flops: 100e9, bytes: 1e9, seconds: 1,
                                  precision: .fp16, peaks: peaks)
        XCTAssertEqual(fp32.ceilingGFLOPS, 1000, accuracy: 1e-9)
        XCTAssertEqual(fp16.ceilingGFLOPS, 2000, accuracy: 1e-9)
        XCTAssertEqual(fp16.efficiency, fp32.efficiency / 2, accuracy: 1e-12)
    }

    func testDegenerateInputsDoNotProduceNaN() {
        let zeroTime = Roofline.place(flops: 1e9, bytes: 1e9, seconds: 0,
                                      precision: .fp32, peaks: peaks)
        XCTAssertEqual(zeroTime.achievedGFLOPS, 0)
        XCTAssertEqual(zeroTime.efficiency, 0)

        let zeroBytes = Roofline.place(flops: 1e9, bytes: 0, seconds: 1,
                                       precision: .fp32, peaks: peaks)
        XCTAssertEqual(zeroBytes.arithmeticIntensity, 0)
        XCTAssertEqual(zeroBytes.efficiency, 0)
        XCTAssertFalse(zeroBytes.efficiency.isNaN)
    }

    func testPlaceFromKernelRecordUsesStoredAnalyticNumbers() {
        let record = KernelRecord(label: "gemm",
                                  shape: .gemm(m: 100, n: 100, k: 100),
                                  precision: .fp32,
                                  durationSeconds: 1e-3,
                                  timingSource: .counterSampleBuffer)
        let placement = Roofline.place(record, peaks: peaks)
        // 2*100^3 = 2e6 FLOP in 1 ms = 2 GFLOP/s.
        XCTAssertEqual(placement.achievedGFLOPS, 2, accuracy: 1e-9)
    }

    func testFolkloreLookupPrefersLongestPrefix() {
        XCTAssertEqual(ChipPeaks.folklore(for: "Apple M1 Pro")?.name, "Apple M1 Pro")
        XCTAssertEqual(ChipPeaks.folklore(for: "Apple M1 Max")?.name, "Apple M1 Max")
        XCTAssertEqual(ChipPeaks.folklore(for: "Apple M1")?.name, "Apple M1")
        XCTAssertNil(ChipPeaks.folklore(for: "Intel Iris"))
    }

    func testFolklorePeakSetIsLabelledAsFolklore() throws {
        let peaks = try XCTUnwrap(ChipPeaks.folklore(for: "Apple M1 Pro")).peakSet
        XCTAssertEqual(peaks.source, .specSheet)
        XCTAssertEqual(peaks.source.label, "spec-sheet folklore")
        XCTAssertEqual(peaks.fp32GFLOPS, 5200, accuracy: 1e-6)
    }

    func testLegacyKernelSampleEfficiencyStillWorks() {
        let chip = ChipPeaks(name: "T", fp32TFLOPS: 1, fp16TFLOPS: 2, memoryBandwidthGBs: 100)
        let sample = KernelSample(name: "k", flopsPerByte: 1, achievedGFLOPS: 50, durationMicros: 10)
        XCTAssertEqual(sample.efficiency(on: chip), 0.5, accuracy: 1e-9)
    }
}
