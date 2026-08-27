import XCTest
@testable import MetalscopeCore

/// The analytic shape registry is the whole ML-awareness claim; if these numbers
/// are wrong every efficiency percentage downstream is wrong too.
final class KernelShapeTests: XCTestCase {
    func testGEMMFlops() {
        // One multiply-add per (m, n, k) triple = 2 FLOPs.
        XCTAssertEqual(KernelShape.gemm(m: 2, n: 3, k: 4).flops, 48, accuracy: 1e-9)
        XCTAssertEqual(KernelShape.gemm(m: 1024, n: 1024, k: 1024).flops,
                       2 * 1024.0 * 1024 * 1024, accuracy: 1e-3)
    }

    func testGEMMBytesCountsThreeMatrices() {
        // A(m*k) + B(k*n) + C(m*n) at 4 bytes each.
        let shape = KernelShape.gemm(m: 2, n: 3, k: 4)
        XCTAssertEqual(shape.bytes(precision: .fp32), Double((8 + 12 + 6) * 4), accuracy: 1e-9)
        XCTAssertEqual(shape.bytes(precision: .fp16), Double((8 + 12 + 6) * 2), accuracy: 1e-9)
        XCTAssertEqual(shape.bytes(precision: .int8), Double(8 + 12 + 6), accuracy: 1e-9)
    }

    func testGEMMIntensityGrowsWithSizeAndHalvesForFP16Bytes() {
        let small = KernelShape.gemm(m: 128, n: 128, k: 128)
        let large = KernelShape.gemm(m: 2048, n: 2048, k: 2048)
        XCTAssertLessThan(small.arithmeticIntensity(precision: .fp32),
                          large.arithmeticIntensity(precision: .fp32))
        // Square GEMM: 2n^3 FLOP / 3n^2*e bytes = 2n / 3e.
        XCTAssertEqual(large.arithmeticIntensity(precision: .fp32),
                       2 * 2048.0 / (3 * 4), accuracy: 1e-6)
        XCTAssertEqual(large.arithmeticIntensity(precision: .fp16),
                       2 * large.arithmeticIntensity(precision: .fp32), accuracy: 1e-6)
    }

    func testAttentionFlopsCountBothMatmuls() {
        // QK^T (2*b*h*s*s*d) + PV (2*b*h*s*s*d).
        let shape = KernelShape.attention(b: 2, h: 4, s: 8, d: 16)
        XCTAssertEqual(shape.flops, 4.0 * 2 * 4 * 8 * 8 * 16, accuracy: 1e-9)
    }

    func testAttentionBytesAssumeFusedScores() {
        // Q, K, V in and O out; the s x s score matrix never touches DRAM.
        let shape = KernelShape.attention(b: 2, h: 4, s: 8, d: 16)
        XCTAssertEqual(shape.bytes(precision: .fp16), 2.0 * 4 * 2 * 4 * 8 * 16, accuracy: 1e-9)
    }

    func testAttentionIntensityScalesWithSequenceLength() {
        // FLOPs go as s^2, bytes as s, so AI is linear in s: 4*s*d / (4*d*e) = s/e.
        let short = KernelShape.attention(b: 1, h: 8, s: 128, d: 64)
        let long = KernelShape.attention(b: 1, h: 8, s: 1024, d: 64)
        XCTAssertEqual(short.arithmeticIntensity(precision: .fp32), 128.0 / 4, accuracy: 1e-6)
        XCTAssertEqual(long.arithmeticIntensity(precision: .fp32), 1024.0 / 4, accuracy: 1e-6)
        XCTAssertEqual(long.arithmeticIntensity(precision: .fp32),
                       8 * short.arithmeticIntensity(precision: .fp32), accuracy: 1e-6)
    }

    func testElementwiseIsOneReadOneWrite() {
        let shape = KernelShape.elementwise(n: 1000)
        XCTAssertEqual(shape.flops, 1000, accuracy: 1e-9)
        XCTAssertEqual(shape.bytes(precision: .fp32), 8000, accuracy: 1e-9)
        // 1 FLOP per 8 bytes — hopelessly bandwidth-bound, by construction.
        XCTAssertEqual(shape.arithmeticIntensity(precision: .fp32), 0.125, accuracy: 1e-9)
    }

    func testNormIsFiveFlopsPerElementOverTwoStreams() {
        let shape = KernelShape.norm(n: 1000)
        XCTAssertEqual(shape.flops, 5000, accuracy: 1e-9)
        XCTAssertEqual(shape.bytes(precision: .fp32), 8000, accuracy: 1e-9)
        XCTAssertEqual(shape.arithmeticIntensity(precision: .fp16), 5000.0 / 4000, accuracy: 1e-9)
    }

    func testOpaqueShapePassesThroughSuppliedNumbers() {
        let shape = KernelShape.opaque(flops: 123, bytes: 456)
        XCTAssertEqual(shape.flops, 123, accuracy: 1e-9)
        // Precision must not rescale explicitly-supplied bytes.
        XCTAssertEqual(shape.bytes(precision: .fp32), 456, accuracy: 1e-9)
        XCTAssertEqual(shape.bytes(precision: .fp16), 456, accuracy: 1e-9)
    }

    func testPrecisionElementSizes() {
        XCTAssertEqual(Precision.fp32.bytesPerElement, 4)
        XCTAssertEqual(Precision.fp16.bytesPerElement, 2)
        XCTAssertEqual(Precision.bf16.bytesPerElement, 2)
        XCTAssertEqual(Precision.int8.bytesPerElement, 1)
        XCTAssertFalse(Precision.fp32.usesHalfPeak)
        XCTAssertTrue(Precision.fp16.usesHalfPeak)
    }

    func testShapeDescriptionsAreStableAndDistinct() {
        XCTAssertEqual(KernelShape.gemm(m: 1, n: 2, k: 3).descriptionText, "gemm 1x2x3")
        XCTAssertEqual(KernelShape.attention(b: 1, h: 2, s: 3, d: 4).descriptionText, "attn b1 h2 s3 d4")
        XCTAssertEqual(KernelShape.elementwise(n: 5).descriptionText, "elem n=5")
        XCTAssertEqual(KernelShape.norm(n: 5).descriptionText, "norm n=5")
        // A norm and an elementwise of the same size must not collide in diff keys.
        XCTAssertNotEqual(KernelShape.elementwise(n: 5).descriptionText,
                          KernelShape.norm(n: 5).descriptionText)
    }

    func testShapeCodableRoundTripForEveryCase() throws {
        let shapes: [KernelShape] = [
            .gemm(m: 64, n: 128, k: 256),
            .attention(b: 2, h: 8, s: 512, d: 64),
            .elementwise(n: 1 << 20),
            .norm(n: 4096),
            .opaque(flops: 1.5e9, bytes: 2.5e9),
        ]
        for shape in shapes {
            let data = try TraceIO.makeEncoder().encode(shape)
            let decoded = try TraceIO.makeDecoder().decode(KernelShape.self, from: data)
            XCTAssertEqual(shape, decoded)
            XCTAssertEqual(shape.flops, decoded.flops, accuracy: 1e-6)
        }
    }

    func testUnknownShapeKindIsRejected() {
        let json = Data(#"{"kind":"convolution","n":1}"#.utf8)
        XCTAssertThrowsError(try TraceIO.makeDecoder().decode(KernelShape.self, from: json))
    }
}
