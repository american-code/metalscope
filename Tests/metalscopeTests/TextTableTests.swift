import XCTest
@testable import MetalscopeCore

/// `TextTable` and `Fmt` are the last thing between an analysis and the person
/// reading it: every table `report`, `diff`, `info` and `calibrate` print goes
/// through here. A misaligned column or a silently truncated label is a
/// correctness bug in a profiler, because it is how a wrong number gets read as
/// a right one.
final class TextTableTests: XCTestCase {

    /// Spaces, spelled out — hand-typed runs of whitespace in an expectation are
    /// how an alignment test ends up asserting the bug.
    private func sp(_ n: Int) -> String { String(repeating: " ", count: n) }
    private func dash(_ n: Int) -> String { String(repeating: "-", count: n) }

    private func lines(_ table: TextTable, indent: String = "") -> [String] {
        table.rendered(indent: indent).split(separator: "\n", omittingEmptySubsequences: false)
            .map(String.init)
    }

    // MARK: - Layout

    /// The whole contract in one small table: column widths are the widest of
    /// title and cells, cells are padded to that width, columns are joined by
    /// `gap`, and a rule of dashes sits under the header.
    func testRendersHeaderRuleAndRowExactly() {
        var table = TextTable(columns: [.init("a"), .init("b", .right)])
        table.addRow(["xx", "y"])
        XCTAssertEqual(table.rendered(), """
        a   b
        --  -
        xx  y
        """)
    }

    func testIndentPrefixesEveryLineIncludingTheRule() {
        var table = TextTable(columns: [.init("a"), .init("b", .right)])
        table.addRow(["xx", "y"])
        XCTAssertEqual(lines(table, indent: "  "), ["  a   b", "  --  -", "  xx  y"])
        // A rule that starts at column 0 under an indented header is the classic
        // "looks fine until you paste it somewhere" bug.
        XCTAssertTrue(lines(table, indent: "    ").allSatisfy { $0.hasPrefix("    ") })
    }

    func testColumnWidthIsTheWidestOfTitleAndCells() {
        var table = TextTable(columns: [.init("kernel"), .init("n", .right)])
        table.addRow(["act.scale", "1"])          // cell wider than its title
        table.addRow(["x", "16777216"])           // and in the other column
        let out = lines(table)
        XCTAssertEqual(out[0], "kernel" + sp(3) + sp(2) + sp(7) + "n")
        XCTAssertEqual(out[1], dash(9) + sp(2) + dash(8))
        XCTAssertEqual(out[2], "act.scale" + sp(2) + sp(7) + "1")
        XCTAssertEqual(out[3], "x" + sp(8) + sp(2) + "16777216")
    }

    func testLeftAndRightAlignmentPutPaddingOnOppositeSides() {
        var left = TextTable(columns: [.init("wide-title"), .init("z")])
        left.addRow(["ab", "z"])
        // Left-aligned: content first, padding after, so the next column always
        // starts at the same offset.
        XCTAssertEqual(lines(left)[2], "ab" + sp(8) + sp(2) + "z")

        var right = TextTable(columns: [.init("wide-title", .right), .init("z")])
        right.addRow(["ab", "z"])
        XCTAssertEqual(lines(right)[2], sp(8) + "ab" + sp(2) + "z")
    }

    func testDefaultAlignmentIsLeft() {
        XCTAssertEqual(TextTable.Column("x").alignment, .left)
        XCTAssertEqual(TextTable.Column("x", .right).alignment, .right)
    }

    func testCustomGapSeparatesColumnsAndTheRule() {
        var table = TextTable(columns: [.init("a"), .init("b")], gap: " | ")
        table.addRow(["xx", "yy"])
        XCTAssertEqual(table.rendered(), """
        a  | b
        -- | --
        xx | yy
        """)
    }

    // MARK: - Ragged input

    /// `report` builds its rows conditionally (the occupancy columns only appear
    /// when a trace carries them), so a short row must pad rather than crash.
    func testShortRowsArePaddedNotDropped() {
        var table = TextTable(columns: [.init("a"), .init("b"), .init("c")])
        table.addRow(["xx"])
        table.addRow(["xx", "yy", "zz"])
        let out = lines(table)
        XCTAssertEqual(out[2], "xx")              // trailing blanks trimmed away
        XCTAssertEqual(out[3], "xx  yy  zz")
        // The short row still contributes its width to column 0.
        XCTAssertEqual(out[1], dash(2) + sp(2) + dash(2) + sp(2) + dash(2))
    }

    func testEmptyCellsInTheMiddleKeepTheirColumnWidth() {
        var table = TextTable(columns: [.init("a"), .init("bbbb"), .init("c")])
        table.addRow(["x", "", "z"])
        // An empty middle cell must hold its column open — this is the "-" case
        // in `report`, where a kernel has no occupancy block.
        XCTAssertEqual(lines(table)[2], "x" + sp(2) + sp(4) + sp(2) + "z")
    }

    /// Extra cells are ignored rather than appended as a phantom column: the
    /// header declares the schema, and a row that disagrees must not silently
    /// widen the table.
    func testCellsBeyondTheColumnCountAreIgnored() {
        var table = TextTable(columns: [.init("a")])
        table.addRow(["xx", "this cell has no column"])
        XCTAssertEqual(table.rendered(), """
        a
        --
        xx
        """)
    }

    func testTableWithNoRowsIsJustHeaderAndRule() {
        let table = TextTable(columns: [.init("kernel"), .init("eff", .right)])
        XCTAssertEqual(table.rendered(), """
        kernel  eff
        ------  ---
        """)
    }

    func testTableWithNoColumnsRendersBlankLinesRatherThanCrashing() {
        var table = TextTable(columns: [])
        table.addRow(["ignored"])
        XCTAssertEqual(lines(table), ["", "", ""])
    }

    // MARK: - Trailing whitespace

    /// Terminal output with trailing spaces is invisible until someone diffs it,
    /// pastes it into a doc, or a linter rejects it. The last column is trimmed.
    func testTrailingPaddingIsStripped() {
        var table = TextTable(columns: [.init("aaaa"), .init("bbbb")])
        table.addRow(["a", "b"])
        for line in lines(table) {
            XCTAssertFalse(line.hasSuffix(" "), "line '\(line)' has trailing whitespace")
        }
        XCTAssertEqual(lines(table)[2], "a" + sp(3) + sp(2) + "b")
    }

    func testInteriorWhitespaceInACellIsPreserved() {
        var table = TextTable(columns: [.init("a"), .init("b")])
        table.addRow(["x  y", "z"])
        XCTAssertEqual(lines(table)[2], "x  y  z")
    }

    // MARK: - Sizing edge cases

    /// There is deliberately no truncation: a kernel label is an identifier, and
    /// a table that quietly turns `attn.sdpa.qkv.projection` into `attn.sdp…`
    /// makes two different kernels look like one. Wide content widens the column.
    func testLongCellsWidenTheColumnAndAreNeverTruncated() {
        let long = String(repeating: "x", count: 120)
        var table = TextTable(columns: [.init("k"), .init("n", .right)])
        table.addRow([long, "1"])
        let out = lines(table)
        XCTAssertEqual(out[2], long + sp(2) + "1")
        XCTAssertEqual(out[1], dash(120) + sp(2) + "-")
    }

    /// Widths are counted in Characters, not UTF-8 bytes: "café" is 4 columns
    /// wide on screen and 5 bytes on disk, and padding to the byte count would
    /// push every following column one place right.
    func testWidthsAreCountedInCharactersNotBytes() {
        var table = TextTable(columns: [.init("name"), .init("x")])
        table.addRow(["café", "1"])
        XCTAssertEqual("café".count, 4)
        XCTAssertEqual("café".utf8.count, 5)
        XCTAssertEqual(lines(table)[2], "café" + sp(2) + "1")
    }

    func testEmptyTitleColumnStillSizesToItsCells() {
        // `info` and `calibrate` use an unnamed first column for row labels.
        var table = TextTable(columns: [.init(""), .init("value", .right)])
        table.addRow(["fp32 compute", "3.45 TF"])
        let out = lines(table)
        XCTAssertEqual(out[0], sp(12) + sp(2) + sp(2) + "value")
        XCTAssertEqual(out[1], dash(12) + sp(2) + dash(7))
        XCTAssertEqual(out[2], "fp32 compute" + sp(2) + "3.45 TF")
    }

    func testRowsArePrintedInInsertionOrder() {
        var table = TextTable(columns: [.init("k")])
        for label in ["ffn.gemm", "attn.sdpa", "act.scale"] { table.addRow([label]) }
        XCTAssertEqual(table.rows.map(\.first), ["ffn.gemm", "attn.sdpa", "act.scale"])
        XCTAssertEqual(Array(lines(table).dropFirst(2)), ["ffn.gemm", "attn.sdpa", "act.scale"])
    }

    /// The property every report depends on: each column starts at the same
    /// offset on every line, header and rule included.
    func testColumnsStartAtTheSameOffsetOnEveryLine() {
        var table = TextTable(columns: [.init("kernel"), .init("time/iter", .right), .init("timing")])
        table.addRow(["ffn.gemm", "1.076 ms", "cmdbuf"])
        table.addRow(["act.scale", "842.34 us", "counters"])
        table.addRow(["stream.triad", "1.233 ms", "counters"])
        // Widest cell per column: "stream.triad" (12) and "842.34 us" (9).
        let gapAfterFirst = 12
        let gapAfterSecond = 12 + 2 + 9
        for line in lines(table) {
            let chars = Array(line)
            XCTAssertGreaterThan(chars.count, gapAfterSecond + 1)
            XCTAssertEqual(chars[gapAfterFirst], " ")
            XCTAssertEqual(chars[gapAfterFirst + 1], " ")
            XCTAssertEqual(chars[gapAfterSecond], " ")
            XCTAssertEqual(chars[gapAfterSecond + 1], " ")
        }
    }

    // MARK: - Fmt.duration

    func testDurationPicksAUnitPerMagnitude() {
        XCTAssertEqual(Fmt.duration(0.000_9), "900.00 us")
        XCTAssertEqual(Fmt.duration(0.000_842_34), "842.34 us")
        XCTAssertEqual(Fmt.duration(0.001_076), "1.076 ms")
        XCTAssertEqual(Fmt.duration(0.5), "500.000 ms")
        XCTAssertEqual(Fmt.duration(1.5), "1.500 s")
    }

    func testDurationUnitBoundariesAreExact() {
        // 1e-3 is not *less than* 1e-3, so it renders as milliseconds.
        XCTAssertEqual(Fmt.duration(1e-3), "1.000 ms")
        XCTAssertEqual(Fmt.duration(9.99e-4), "999.00 us")
        XCTAssertEqual(Fmt.duration(1.0), "1.000 s")
    }

    /// A duration of zero means "never measured", not "infinitely fast", and a
    /// dash is the only honest rendering. `diff` prints these for absent kernels.
    func testDurationRefusesToRenderNonPositiveOrNaN() {
        XCTAssertEqual(Fmt.duration(0), "-")
        XCTAssertEqual(Fmt.duration(-1), "-")
        XCTAssertEqual(Fmt.duration(.nan), "-")
    }

    // MARK: - Fmt.durationRange

    /// The `spread` column. Both ends are rendered in the unit the high end
    /// picks, so the two numbers in a cell can be compared without conversion.
    func testDurationRangeUsesOneUnitForBothEnds() {
        XCTAssertEqual(Fmt.durationRange(77e-6, 195e-6), "77.0-195.0 us")
        XCTAssertEqual(Fmt.durationRange(5e-6, 5e-6), "5.0-5.0 us")
        // A sub-millisecond low end is still rendered in the high end's unit,
        // at the same precision `Fmt.duration` uses there.
        XCTAssertEqual(Fmt.durationRange(0.000_9, 0.001_2), "0.900-1.200 ms")
        XCTAssertEqual(Fmt.durationRange(0.001_28, 0.001_293), "1.280-1.293 ms")
        XCTAssertEqual(Fmt.durationRange(0.4, 1.5), "0.400-1.500 s")
    }

    func testDurationRangeRefusesNonPositiveNaNOrInvertedBounds() {
        XCTAssertEqual(Fmt.durationRange(0, 1e-3), "-")
        XCTAssertEqual(Fmt.durationRange(1e-3, 0), "-")
        XCTAssertEqual(Fmt.durationRange(-1, 1), "-")
        XCTAssertEqual(Fmt.durationRange(.nan, 1e-3), "-")
        XCTAssertEqual(Fmt.durationRange(1e-3, .infinity), "-")
        // A high end below the low end is a bug upstream, not a range.
        XCTAssertEqual(Fmt.durationRange(200e-6, 100e-6), "-")
    }

    // MARK: - Fmt.gflops / bandwidth / intensity

    func testGFLOPSSwitchesToTeraflopsAtAThousand() {
        XCTAssertEqual(Fmt.gflops(999.94), "999.9 GF")
        XCTAssertEqual(Fmt.gflops(1000), "1.00 TF")
        XCTAssertEqual(Fmt.gflops(3445.73), "3.45 TF")
        XCTAssertEqual(Fmt.gflops(98.5), "98.5 GF")
        XCTAssertEqual(Fmt.gflops(0), "-")
        XCTAssertEqual(Fmt.gflops(.nan), "-")
    }

    func testBandwidthAlwaysUsesGBPerSecond() {
        XCTAssertEqual(Fmt.bandwidth(166.404_696), "166.4 GB/s")
        XCTAssertEqual(Fmt.bandwidth(0), "-")
        XCTAssertEqual(Fmt.bandwidth(-3), "-")
        XCTAssertEqual(Fmt.bandwidth(.nan), "-")
    }

    /// Arithmetic intensity spans four orders of magnitude in one table (0.12 for
    /// a scale kernel, 341 for a half-precision GEMM), so the precision has to
    /// follow the magnitude or every small number renders as "0".
    func testIntensityPrecisionFollowsMagnitude() {
        XCTAssertEqual(Fmt.intensity(341.3), "341")
        XCTAssertEqual(Fmt.intensity(100), "100")
        XCTAssertEqual(Fmt.intensity(99.94), "99.9")
        XCTAssertEqual(Fmt.intensity(20.7), "20.7")
        XCTAssertEqual(Fmt.intensity(10), "10.0")
        XCTAssertEqual(Fmt.intensity(9.994), "9.99")
        XCTAssertEqual(Fmt.intensity(0.125), "0.12")
        XCTAssertEqual(Fmt.intensity(0), "0.00")
        XCTAssertEqual(Fmt.intensity(-1), "-1.00")
        XCTAssertEqual(Fmt.intensity(.nan), "-")
    }

    /// `PeakSet.ridgePoint` returns infinity when bandwidth is unknown; that has
    /// to survive formatting as infinity rather than as a plausible number.
    func testIntensityOfInfinityRendersAsInfinity() {
        XCTAssertEqual(Fmt.intensity(.infinity).lowercased(), "inf")
    }

    // MARK: - Fmt percentages

    func testPercentAndItsEdges() {
        XCTAssertEqual(Fmt.percent(0.579), "57.9%")
        XCTAssertEqual(Fmt.percent(0), "0.0%")
        XCTAssertEqual(Fmt.percent(1), "100.0%")
        // Over 100% is legal and must print: it means the peaks are stale or the
        // analytic byte model under-counted cache-resident reuse.
        XCTAssertEqual(Fmt.percent(1.234), "123.4%")
        XCTAssertEqual(Fmt.percent(.nan), "-")
    }

    /// A diff column is unreadable without an explicit sign: "+2.2%" is a
    /// regression and "-39.7%" is a win, and the reader must not have to work out
    /// which direction the metric runs.
    func testSignedPercentAlwaysCarriesASign() {
        XCTAssertEqual(Fmt.signedPercent(-0.397), "-39.7%")
        XCTAssertEqual(Fmt.signedPercent(0.022), "+2.2%")
        XCTAssertEqual(Fmt.signedPercent(0), "+0.0%")
        XCTAssertEqual(Fmt.signedPercent(.nan), "-")
    }

    /// Efficiency deltas are percentage *points*, labelled "pp" so they can't be
    /// misread as a relative change.
    func testSignedPointsAreLabelledPP() {
        XCTAssertEqual(Fmt.signedPoints(38.2), "+38.2 pp")
        XCTAssertEqual(Fmt.signedPoints(-2.1), "-2.1 pp")
        XCTAssertEqual(Fmt.signedPoints(0), "+0.0 pp")
        XCTAssertEqual(Fmt.signedPoints(.nan), "-")
    }

    // MARK: - Fmt.bytes

    func testBytesScaleThroughBinaryUnits() {
        XCTAssertEqual(Fmt.bytes(0), "0 B")
        XCTAssertEqual(Fmt.bytes(128), "128 B")
        XCTAssertEqual(Fmt.bytes(1023), "1023 B")
        XCTAssertEqual(Fmt.bytes(1024), "1.0 KB")
        XCTAssertEqual(Fmt.bytes(32768), "32.0 KB")         // the M1 Pro threadgroup limit
        XCTAssertEqual(Fmt.bytes(Double(1 << 20)), "1.0 MB")
        XCTAssertEqual(Fmt.bytes(11_453_251_584), "10.7 GB")
        XCTAssertEqual(Fmt.bytes(Double(1 << 40)), "1.0 TB")
    }

    /// The unit table ends at TB; anything larger keeps counting in TB rather
    /// than walking off the end of the array.
    func testBytesClampAtTheLargestUnit() {
        XCTAssertEqual(Fmt.bytes(Double(1 << 50)), "1024.0 TB")
        XCTAssertTrue(Fmt.bytes(1e30).hasSuffix(" TB"))
    }

    /// Byte counts are non-negative in every real caller, but the formatter must
    /// not loop forever or crash if one ever arrives negative.
    func testBytesHandleNegativeAndSubUnitValues() {
        XCTAssertEqual(Fmt.bytes(-5), "-5 B")
        XCTAssertEqual(Fmt.bytes(0.4), "0 B")
    }
}
