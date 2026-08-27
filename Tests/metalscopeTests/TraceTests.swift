import XCTest
@testable import MetalscopeCore

final class TraceTests: XCTestCase {
    private func makeTrace() -> Trace {
        var trace = Trace(createdAt: Date(timeIntervalSince1970: 1_724_700_000.25),
                          device: DeviceInfo(name: "Apple M1 Pro",
                                             registryID: 12345,
                                             maxWorkingSetBytes: 11_453_246_566,
                                             counterSets: ["timestamp"],
                                             supportsStageBoundarySampling: true),
                          peaks: PeakSet(source: .measured, chip: "Apple M1 Pro",
                                         fp32GFLOPS: 4321.5, fp16GFLOPS: 8123.25,
                                         bandwidthGBs: 178.5,
                                         measuredAt: Date(timeIntervalSince1970: 1_700_000_000),
                                         details: ["gemmBestSize": 2048]),
                          notes: ["command": "bench"])
        trace.kernels = [
            KernelRecord(label: "ffn.gemm", shape: .gemm(m: 1024, n: 1024, k: 1024),
                         precision: .fp32, durationSeconds: 0.000_512, iterations: 8,
                         timingSource: .commandBuffer, hostDurationSeconds: 0.0061,
                         notes: ["backend": "MPSMatrixMultiplication"]),
            KernelRecord(label: "attn.sdpa", shape: .attention(b: 1, h: 8, s: 512, d: 64),
                         precision: .fp16, durationSeconds: 0.001_25, iterations: 2,
                         timingSource: .counterSampleBuffer,
                         stages: [StageSample(name: "qk", durationSeconds: 0.0007),
                                  StageSample(name: "pv", durationSeconds: 0.00055)]),
            KernelRecord(label: "stream.triad", shape: .opaque(flops: 3.3e7, bytes: 2.01e8),
                         durationSeconds: 0.0011, timingSource: .host),
        ]
        return trace
    }

    func testTraceRoundTripPreservesEverything() throws {
        let trace = makeTrace()
        let data = try TraceIO.encode(trace)
        let decoded = try TraceIO.decode(data)
        XCTAssertEqual(trace, decoded)
    }

    func testRoundTripPreservesAnalyticNumbersAndStages() throws {
        let decoded = try TraceIO.decode(try TraceIO.encode(makeTrace()))
        let gemm = decoded.kernels[0]
        XCTAssertEqual(gemm.flops, 2 * 1024.0 * 1024 * 1024, accuracy: 1e-3)
        XCTAssertEqual(gemm.bytes, 3 * 1024.0 * 1024 * 4, accuracy: 1e-3)
        XCTAssertEqual(gemm.iterations, 8)
        XCTAssertEqual(gemm.timingSource, .commandBuffer)
        XCTAssertEqual(decoded.kernels[1].stages?.count, 2)
        XCTAssertEqual(decoded.kernels[1].stages?.first?.name, "qk")
        // Opaque numbers survive: nothing recomputes them from a shape.
        XCTAssertEqual(decoded.kernels[2].bytes, 2.01e8, accuracy: 1e-3)
        XCTAssertEqual(decoded.peaks?.details?["gemmBestSize"], 2048)
    }

    func testFileRoundTripThroughDisk() throws {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("metalscope-test-\(UUID().uuidString)")
            .appendingPathComponent("trace.json")
        defer { try? FileManager.default.removeItem(at: url.deletingLastPathComponent()) }
        let trace = makeTrace()
        try TraceIO.write(trace, to: url)   // must create intermediate directories
        XCTAssertEqual(try TraceIO.read(from: url), trace)
    }

    func testJSONIsStableAndHumanReadable() throws {
        let json = String(data: try TraceIO.encode(makeTrace()), encoding: .utf8) ?? ""
        XCTAssertTrue(json.contains("\"schemaVersion\" : 2"))
        XCTAssertTrue(json.contains("\"kind\" : \"gemm\""))
        XCTAssertTrue(json.contains("\"source\" : \"measured\""))
        XCTAssertTrue(json.contains("\"timingSource\" : \"command-buffer\""))
        // Sorted keys make traces diffable with plain `diff`.
        let first = json.range(of: "\"createdAt\"")
        let second = json.range(of: "\"device\"")
        XCTAssertNotNil(first)
        XCTAssertNotNil(second)
        XCTAssertLessThan(first!.lowerBound, second!.lowerBound)
    }

    func testNewerSchemaVersionIsRejected() throws {
        var trace = makeTrace()
        trace.schemaVersion = Trace.currentSchemaVersion + 1
        let data = try TraceIO.makeEncoder().encode(trace)
        XCTAssertThrowsError(try TraceIO.decode(data)) { error in
            guard case TraceError.unsupportedSchema = error else {
                return XCTFail("expected unsupportedSchema, got \(error)")
            }
        }
    }

    // MARK: - Schema v2

    /// A v1 trace, byte for byte as metalscope 0.1.0 wrote them. Kept as a
    /// literal rather than generated, so a future change to the encoder can't
    /// quietly redefine what "a v1 trace" means.
    private static let schemaV1JSON = """
    {
      "createdAt" : "2026-08-26T19:06:18.512Z",
      "device" : {
        "counterSets" : [
          "timestamp"
        ],
        "maxWorkingSetBytes" : 11453251584,
        "name" : "Apple M1 Pro",
        "registryID" : 4294970113,
        "supportsStageBoundarySampling" : true
      },
      "kernels" : [
        {
          "bytes" : 12582912,
          "durationSeconds" : 0.00092634,
          "flops" : 2147483648,
          "hostDurationSeconds" : 0.1531,
          "iterations" : 162,
          "label" : "ffn.gemm",
          "notes" : {
            "backend" : "MPSMatrixMultiplication"
          },
          "precision" : "fp32",
          "shape" : {
            "k" : 1024,
            "kind" : "gemm",
            "m" : 1024,
            "n" : 1024
          },
          "timingSource" : "command-buffer"
        },
        {
          "bytes" : 134217728,
          "durationSeconds" : 0.000843,
          "flops" : 16777216,
          "iterations" : 178,
          "label" : "act.scale",
          "precision" : "fp32",
          "shape" : {
            "kind" : "elementwise",
            "n" : 16777216
          },
          "stages" : [
            {
              "durationSeconds" : 0.0008,
              "name" : "scale"
            }
          ],
          "timingSource" : "counter-sample-buffer"
        }
      ],
      "notes" : {
        "command" : "bench",
        "variant" : "baseline"
      },
      "peaks" : {
        "bandwidthGBs" : 168.13,
        "chip" : "Apple M1 Pro",
        "fp16GFLOPS" : 3324.11,
        "fp32GFLOPS" : 3412.73,
        "source" : "measured"
      },
      "schemaVersion" : 1,
      "tool" : "metalscope",
      "toolVersion" : "0.1.0"
    }
    """

    func testSchemaV1TracesStillRead() throws {
        let trace = try TraceIO.decode(Data(Self.schemaV1JSON.utf8))
        XCTAssertEqual(trace.schemaVersion, 1)
        XCTAssertEqual(trace.kernels.count, 2)
        XCTAssertEqual(trace.device.name, "Apple M1 Pro")
        XCTAssertEqual(trace.kernels[0].label, "ffn.gemm")
        XCTAssertEqual(trace.kernels[1].stages?.first?.name, "scale")
        XCTAssertEqual(trace.peaks?.source, .measured)
        // Everything v2 added is simply absent — no defaults invented.
        XCTAssertNil(trace.kernels[0].occupancy)
        XCTAssertNil(trace.kernels[1].occupancy)
        XCTAssertNil(trace.kernels[0].counters)
        XCTAssertNil(trace.device.maxThreadgroupMemoryBytes)
    }

    /// A v1 trace stays analysable: report and diff must not need occupancy.
    func testV1TraceIsStillFullyAnalysable() throws {
        let trace = try TraceIO.decode(Data(Self.schemaV1JSON.utf8))
        let peaks = try XCTUnwrap(trace.peaks)
        let placement = Roofline.place(trace.kernels[0], peaks: peaks)
        XCTAssertEqual(placement.bound, .compute)
        XCTAssertGreaterThan(placement.efficiency, 0)
        let entries = TraceDiff.align(baseline: trace.kernels, candidate: trace.kernels)
        XCTAssertTrue(entries.allSatisfy { $0.status == .matched })
        XCTAssertFalse(entries.contains { $0.hasOccupancy })
        XCTAssertNil(entries[0].occupancyDeltaPoints)
    }

    /// Re-writing a v1 trace stamps it v2; the v1 content is untouched.
    func testReadingV1AndWritingBackProducesV2() throws {
        var trace = try TraceIO.decode(Data(Self.schemaV1JSON.utf8))
        trace.schemaVersion = Trace.currentSchemaVersion
        let round = try TraceIO.decode(try TraceIO.encode(trace))
        XCTAssertEqual(round.schemaVersion, 2)
        XCTAssertEqual(round.kernels.map(\.label), ["ffn.gemm", "act.scale"])
        XCTAssertNil(round.kernels[0].occupancy)
    }

    func testSchemaV2RoundTripsOccupancyAndCounters() throws {
        var trace = makeTrace()
        trace.device.maxThreadgroupMemoryBytes = 32768
        trace.kernels[1].occupancy = OccupancyInfo(threadsPerThreadgroup: 100,
                                                   maxTotalThreadsPerThreadgroup: 1024,
                                                   threadExecutionWidth: 32,
                                                   threadgroupMemoryBytes: 128,
                                                   threadgroupMemoryLimitBytes: 32768,
                                                   threadgroupsPerGrid: 4096,
                                                   dispatchCount: 162,
                                                   variantCount: 1)
        trace.kernels[2].counters = ["stageutilization.TotalCycles": 1000,
                                     "stageutilization.FragmentCyclesFraction": 0.25]

        let decoded = try TraceIO.decode(try TraceIO.encode(trace))
        XCTAssertEqual(decoded, trace)
        XCTAssertEqual(decoded.kernels[1].occupancy?.threadsPerThreadgroup, 100)
        XCTAssertEqual(decoded.kernels[1].occupancy?.dispatchCount, 162)
        XCTAssertEqual(decoded.kernels[1].occupancy?.limiter, .executionWidthAlignment)
        XCTAssertEqual(decoded.kernels[2].counters?["stageutilization.TotalCycles"], 1000)
        XCTAssertEqual(decoded.device.maxThreadgroupMemoryBytes, 32768)
        // Kernels without the new fields still serialize without them.
        XCTAssertNil(decoded.kernels[0].occupancy)
        let json = String(data: try TraceIO.encode(trace), encoding: .utf8) ?? ""
        XCTAssertTrue(json.contains("\"occupancy\""))
        XCTAssertTrue(json.contains("\"schemaVersion\" : 2"))
    }

    func testAlignmentKeyIncludesLabelShapeAndPrecision() {
        let a = KernelRecord(label: "gemm", shape: .gemm(m: 1, n: 1, k: 1),
                             precision: .fp32, durationSeconds: 1, timingSource: .host)
        let b = KernelRecord(label: "gemm", shape: .gemm(m: 1, n: 1, k: 1),
                             precision: .fp16, durationSeconds: 1, timingSource: .host)
        let c = KernelRecord(label: "gemm", shape: .gemm(m: 2, n: 1, k: 1),
                             precision: .fp32, durationSeconds: 1, timingSource: .host)
        XCTAssertNotEqual(a.alignmentKey, b.alignmentKey)
        XCTAssertNotEqual(a.alignmentKey, c.alignmentKey)
        XCTAssertEqual(a.alignmentKey, "gemm|gemm 1x1x1|fp32")
    }

    // MARK: - Timing ladder

    /// The `timing` column in every report. These strings are the whole
    /// user-visible record of *how* a duration was obtained, so they are part of
    /// the interface, not decoration.
    func testTimingSourceRawValuesAndDisplayNames() {
        XCTAssertEqual(TimingSource.counterSampleBuffer.rawValue, "counter-sample-buffer")
        XCTAssertEqual(TimingSource.commandBuffer.rawValue, "command-buffer")
        XCTAssertEqual(TimingSource.host.rawValue, "host")
        XCTAssertEqual(TimingSource.counterSampleBuffer.displayName, "counters")
        XCTAssertEqual(TimingSource.commandBuffer.displayName, "cmdbuf")
        XCTAssertEqual(TimingSource.host.displayName, "host")
    }

    // MARK: - Derived kernel numbers

    func testArithmeticIntensityIsFLOPsOverBytes() {
        let gemm = KernelRecord(label: "g", shape: .gemm(m: 1024, n: 1024, k: 1024),
                                precision: .fp32, durationSeconds: 1, timingSource: .host)
        // 2mnk / 4(mk + kn + mn) = 2*2^30 / (3 * 4 * 2^20).
        XCTAssertEqual(gemm.arithmeticIntensity, 2 * 1073741824.0 / 12582912.0, accuracy: 1e-9)

        let scale = KernelRecord(label: "s", shape: .elementwise(n: 1000),
                                 precision: .fp32, durationSeconds: 1, timingSource: .host)
        XCTAssertEqual(scale.arithmeticIntensity, 0.125, accuracy: 1e-12)
    }

    /// An unmodeled `opaque` kernel carries zero bytes; intensity must be 0
    /// rather than a division by zero that reaches a report as `nan` or `inf`.
    func testArithmeticIntensityOfAnUnmodeledKernelIsZeroNotNaN() {
        let unmodeled = KernelRecord(label: "mystery", shape: .opaque(flops: 0, bytes: 0),
                                     durationSeconds: 0.001, timingSource: .host)
        XCTAssertEqual(unmodeled.bytes, 0)
        XCTAssertEqual(unmodeled.arithmeticIntensity, 0)
        XCTAssertFalse(unmodeled.arithmeticIntensity.isNaN)
    }

    // MARK: - Schema gate

    /// The reject-newer half of the version contract is tested above; this is the
    /// other end. Nothing is retired yet (v1 still reads), so the guard is driven
    /// with a version below the floor.
    func testSchemaVersionBelowTheReadableFloorIsRejected() throws {
        var trace = makeTrace()
        trace.schemaVersion = Trace.minimumReadableSchemaVersion - 1
        let data = try TraceIO.makeEncoder().encode(trace)
        XCTAssertThrowsError(try TraceIO.decode(data)) { error in
            guard case let TraceError.retiredSchema(found, oldest) = error else {
                return XCTFail("expected retiredSchema, got \(error)")
            }
            XCTAssertEqual(found, 0)
            XCTAssertEqual(oldest, Trace.minimumReadableSchemaVersion)
        }
    }

    /// Schema errors are printed straight to the terminal by `main.swift`, so
    /// their text is the whole diagnosis a user gets.
    func testTraceErrorDescriptionsNameBothVersions() {
        XCTAssertEqual(TraceError.unsupportedSchema(found: 3, supported: 2).description,
                       "trace schema version 3 is newer than supported version 2")
        XCTAssertEqual(TraceError.retiredSchema(found: 0, oldestSupported: 1).description,
                       "trace schema version 0 is older than the oldest readable version 1")
    }

    // MARK: - Timestamps

    /// Traces are written with millisecond precision, but a hand-edited trace or
    /// one from another tool may carry whole-second stamps. Both must read.
    func testWholeSecondTimestampsAreAccepted() throws {
        let json = Self.schemaV1JSON.replacingOccurrences(of: "2026-08-26T19:06:18.512Z",
                                                          with: "2026-08-26T19:06:18Z")
        let trace = try TraceIO.decode(Data(json.utf8))
        let fractional = try TraceIO.decode(Data(Self.schemaV1JSON.utf8))
        XCTAssertEqual(trace.createdAt.timeIntervalSince1970,
                       fractional.createdAt.timeIntervalSince1970 - 0.512, accuracy: 1e-3)
    }

    func testAnUnparseableTimestampIsADecodingErrorNotADefaultDate() throws {
        let json = Self.schemaV1JSON.replacingOccurrences(of: "2026-08-26T19:06:18.512Z",
                                                          with: "last Tuesday")
        XCTAssertThrowsError(try TraceIO.decode(Data(json.utf8))) { error in
            guard case let DecodingError.dataCorrupted(context) = error else {
                return XCTFail("expected dataCorrupted, got \(error)")
            }
            XCTAssertTrue(context.debugDescription.contains("last Tuesday"),
                          "the error should quote the value it could not parse")
        }
    }

    /// Sub-millisecond precision is not carried: the format is documented as
    /// milliseconds, and a round trip must be idempotent rather than drifting.
    func testTimestampRoundTripIsStableToTheMillisecond() throws {
        var trace = makeTrace()
        trace.createdAt = Date(timeIntervalSince1970: 1_724_700_000.123_456)
        let once = try TraceIO.decode(try TraceIO.encode(trace))
        let twice = try TraceIO.decode(try TraceIO.encode(once))
        XCTAssertEqual(once.createdAt, twice.createdAt)
        XCTAssertEqual(once.createdAt.timeIntervalSince1970, 1_724_700_000.123, accuracy: 1e-6)
    }

    func testPeaksStoreRoundTripAndFallback() throws {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("metalscope-peaks-\(UUID().uuidString)")
            .appendingPathComponent("peaks.json")
        defer { try? FileManager.default.removeItem(at: url.deletingLastPathComponent()) }
        let store = PeaksStore(url: url)

        XCTAssertNil(store.measured(for: "Apple M1 Pro"))
        // With nothing measured, resolve() falls back to labeled folklore.
        XCTAssertEqual(store.resolve(for: "Apple M1 Pro")?.source, .specSheet)
        XCTAssertNil(store.resolve(for: "Nonexistent GPU"))

        let measured = PeakSet(source: .measured, chip: "Apple M1 Pro", fp32GFLOPS: 4000,
                               fp16GFLOPS: 7000, bandwidthGBs: 180, measuredAt: Date())
        try store.save(measured, for: "Apple M1 Pro")
        XCTAssertEqual(store.measured(for: "Apple M1 Pro")?.fp32GFLOPS, 4000)
        XCTAssertEqual(store.resolve(for: "Apple M1 Pro")?.source, .measured)

        // A second chip must not clobber the first.
        try store.save(PeakSet(source: .measured, chip: "Apple M2", fp32GFLOPS: 3000,
                               fp16GFLOPS: nil, bandwidthGBs: 100), for: "Apple M2")
        XCTAssertEqual(store.loadFile()?.chips.count, 2)
        XCTAssertEqual(store.measured(for: "Apple M1 Pro")?.bandwidthGBs, 180)
    }
}
