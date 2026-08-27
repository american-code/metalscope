import XCTest
@testable import MetalscopeCore
@testable import metalscope

/// `resolvePeaks` is shared by `report`, `diff`, and `profile`, so these tests
/// cover all three. The device name is one with a `ChipPeaks.known` entry, so
/// any accidental folklore fallback would *succeed* — the strict-branch tests
/// below fail loudly if that fallback ever comes back.
final class PeaksResolutionTests: XCTestCase {
    private let deviceName = "Apple M1 Pro"
    private var tempDir: URL!

    override func setUpWithError() throws {
        tempDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("metalscope-tests-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        try FileManager.default.removeItem(at: tempDir)
    }

    private func trace(peaks: PeakSet? = nil) -> Trace {
        Trace(device: DeviceInfo(name: deviceName), peaks: peaks)
    }

    private func args(_ argv: [String]) throws -> Arguments {
        try Arguments.parse(argv, valueOptions: ReportCommand.valueOptions)
    }

    private func measuredPeaks(chip: String) -> PeakSet {
        PeakSet(source: .measured, chip: chip,
                fp32GFLOPS: 4321.5, fp16GFLOPS: 8123.25, bandwidthGBs: 178.5)
    }

    private func peaksFile(_ name: String) -> URL {
        tempDir.appendingPathComponent(name)
    }

    private func assertResolveThrows(_ argv: [String],
                                     trace: Trace? = nil,
                                     messageContains needle: String,
                                     file: StaticString = #filePath,
                                     line: UInt = #line) throws {
        let trace = trace ?? self.trace()
        XCTAssertThrowsError(try ReportCommand.resolvePeaks(for: trace, args: args(argv)),
                             file: file, line: line) { error in
            guard let cliError = error as? CLIError else {
                XCTFail("expected CLIError, got \(error)", file: file, line: line)
                return
            }
            XCTAssertTrue(cliError.description.contains(needle),
                          "expected '\(needle)' in '\(cliError.description)'",
                          file: file, line: line)
        }
    }

    // MARK: - `--peaks-file` is strict

    func testExplicitFileWithMeasuredEntryIsUsed() throws {
        let url = peaksFile("peaks.json")
        let saved = measuredPeaks(chip: deviceName)
        try PeaksStore(url: url).save(saved, for: deviceName)

        let resolved = try ReportCommand.resolvePeaks(for: trace(),
                                                      args: args(["--peaks-file", url.path]))
        XCTAssertEqual(resolved, saved)
    }

    func testExplicitFileMissingErrorsInsteadOfFolklore() throws {
        // The original bug: this fell through to the 5.2 TF spec-sheet entry.
        try assertResolveThrows(["--peaks-file", peaksFile("does-not-exist.json").path],
                                messageContains: "cannot read peaks file")
    }

    func testExplicitFileUnparseableErrorsWithParseMessage() throws {
        let url = peaksFile("garbage.json")
        try Data("not a peaks file".utf8).write(to: url)
        try assertResolveThrows(["--peaks-file", url.path],
                                messageContains: "cannot parse")
    }

    func testExplicitFileWithoutEntryForDeviceErrors() throws {
        let url = peaksFile("peaks.json")
        try PeaksStore(url: url).save(measuredPeaks(chip: "Apple M3 Max"), for: "Apple M3 Max")
        try assertResolveThrows(["--peaks-file", url.path],
                                messageContains: "no entry for '\(deviceName)'")
    }

    func testExplicitFileWithFolkloreEntryErrors() throws {
        // A hand-edited file can hold a spec-sheet entry; an explicit
        // `--peaks-file` still demands a measured one.
        let url = peaksFile("peaks.json")
        let folklore = try XCTUnwrap(ChipPeaks.folklore(for: deviceName)?.peakSet)
        try PeaksStore(url: url).save(folklore, for: deviceName)
        try assertResolveThrows(["--peaks-file", url.path],
                                messageContains: "not measured")
    }

    // MARK: - the rest of the resolution order is unchanged

    func testSpecPeaksFlagStillForcesFolklore() throws {
        let resolved = try ReportCommand.resolvePeaks(for: trace(), args: args(["--spec-peaks"]))
        XCTAssertEqual(resolved.source, .specSheet)
        XCTAssertEqual(resolved.fp32GFLOPS, 5200)
    }

    func testTraceMeasuredPeaksWinWithoutExplicitFile() throws {
        let traced = measuredPeaks(chip: deviceName)
        let resolved = try ReportCommand.resolvePeaks(for: trace(peaks: traced), args: args([]))
        XCTAssertEqual(resolved, traced)
    }
}
