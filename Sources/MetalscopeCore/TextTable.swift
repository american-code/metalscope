import Foundation

/// Minimal column-aligned terminal table. No dependencies, no ANSI.
public struct TextTable {
    public enum Alignment {
        case left, right
    }

    public struct Column {
        public var title: String
        public var alignment: Alignment

        public init(_ title: String, _ alignment: Alignment = .left) {
            self.title = title
            self.alignment = alignment
        }
    }

    public var columns: [Column]
    public var rows: [[String]] = []
    public var gap: String

    public init(columns: [Column], gap: String = "  ") {
        self.columns = columns
        self.gap = gap
    }

    public mutating func addRow(_ cells: [String]) {
        rows.append(cells)
    }

    public func rendered(indent: String = "") -> String {
        let widths = (0..<columns.count).map { i -> Int in
            let cellWidths = rows.map { $0.count > i ? $0[i].count : 0 }
            return max(columns[i].title.count, cellWidths.max() ?? 0)
        }

        func pad(_ s: String, _ width: Int, _ alignment: Alignment) -> String {
            let padding = String(repeating: " ", count: max(0, width - s.count))
            return alignment == .left ? s + padding : padding + s
        }

        func line(_ cells: [String]) -> String {
            let parts = (0..<columns.count).map { i in
                pad(i < cells.count ? cells[i] : "", widths[i], columns[i].alignment)
            }
            return indent + parts.joined(separator: gap)
                .replacingOccurrences(of: "\\s+$", with: "", options: .regularExpression)
        }

        var out = [line(columns.map(\.title))]
        out.append(indent + widths.map { String(repeating: "-", count: $0) }.joined(separator: gap))
        out.append(contentsOf: rows.map(line))
        return out.joined(separator: "\n")
    }
}

/// Number formatting shared by `report`, `diff`, `info`, and `calibrate`.
public enum Fmt {
    public static func duration(_ seconds: Double) -> String {
        if seconds.isNaN || seconds <= 0 { return "-" }
        if seconds < 1e-3 { return String(format: "%.2f us", seconds * 1e6) }
        if seconds < 1 { return String(format: "%.3f ms", seconds * 1e3) }
        return String(format: "%.3f s", seconds)
    }

    public static func gflops(_ v: Double) -> String {
        if v.isNaN || v <= 0 { return "-" }
        if v >= 1000 { return String(format: "%.2f TF", v / 1000) }
        return String(format: "%.1f GF", v)
    }

    public static func bandwidth(_ gbs: Double) -> String {
        if gbs.isNaN || gbs <= 0 { return "-" }
        return String(format: "%.1f GB/s", gbs)
    }

    public static func intensity(_ v: Double) -> String {
        if v.isNaN { return "-" }
        if v >= 100 { return String(format: "%.0f", v) }
        if v >= 10 { return String(format: "%.1f", v) }
        return String(format: "%.2f", v)
    }

    public static func percent(_ fraction: Double) -> String {
        if fraction.isNaN { return "-" }
        return String(format: "%.1f%%", fraction * 100)
    }

    public static func signedPercent(_ fraction: Double) -> String {
        if fraction.isNaN { return "-" }
        return String(format: "%+.1f%%", fraction * 100)
    }

    public static func signedPoints(_ points: Double) -> String {
        if points.isNaN { return "-" }
        return String(format: "%+.1f pp", points)
    }

    public static func bytes(_ b: Double) -> String {
        let units = ["B", "KB", "MB", "GB", "TB"]
        var v = b
        var i = 0
        while v >= 1024, i < units.count - 1 { v /= 1024; i += 1 }
        return String(format: i == 0 ? "%.0f %@" : "%.1f %@", v, units[i])
    }
}
