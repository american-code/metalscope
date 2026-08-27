import Foundation

/// On-disk cache of measured peaks, keyed by device name, at
/// `~/.metalscope/peaks.json`.
public struct PeaksFile: Codable, Sendable, Equatable {
    public static let currentSchemaVersion = 1

    public var schemaVersion: Int
    public var chips: [String: PeakSet]

    public init(schemaVersion: Int = PeaksFile.currentSchemaVersion, chips: [String: PeakSet] = [:]) {
        self.schemaVersion = schemaVersion
        self.chips = chips
    }
}

public struct PeaksStore: Sendable {
    public let url: URL

    public static var defaultDirectory: URL {
        FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent(".metalscope", isDirectory: true)
    }

    public static var defaultURL: URL {
        defaultDirectory.appendingPathComponent("peaks.json")
    }

    public static let `default` = PeaksStore(url: PeaksStore.defaultURL)

    public init(url: URL) {
        self.url = url
    }

    public func loadFile() -> PeaksFile? {
        guard let data = try? Data(contentsOf: url) else { return nil }
        return try? TraceIO.makeDecoder().decode(PeaksFile.self, from: data)
    }

    /// Measured peaks for a device, if `calibrate` has been run here.
    public func measured(for deviceName: String) -> PeakSet? {
        guard let entry = loadFile()?.chips[deviceName], entry.source == .measured else { return nil }
        return entry
    }

    @discardableResult
    public func save(_ peaks: PeakSet, for deviceName: String) throws -> URL {
        var file = loadFile() ?? PeaksFile()
        file.schemaVersion = PeaksFile.currentSchemaVersion
        file.chips[deviceName] = peaks
        try FileManager.default.createDirectory(at: url.deletingLastPathComponent(),
                                                withIntermediateDirectories: true)
        try TraceIO.makeEncoder().encode(file).write(to: url, options: .atomic)
        return url
    }

    /// Measured peaks if available, otherwise the labeled spec-sheet fallback.
    public func resolve(for deviceName: String) -> PeakSet? {
        measured(for: deviceName) ?? ChipPeaks.folklore(for: deviceName)?.peakSet
    }

    /// Measured peaks from this exact file, or a typed error saying why not.
    /// For a file the user named explicitly (`--peaks-file`) there is no
    /// fallback of any kind, and "the file could not be read" is a different
    /// failure than "the file has no entry for this device".
    public func requireMeasured(for deviceName: String) throws -> PeakSet {
        guard let data = try? Data(contentsOf: url) else {
            throw PeaksFileError.unreadable(path: url.path)
        }
        guard let file = try? TraceIO.makeDecoder().decode(PeaksFile.self, from: data) else {
            throw PeaksFileError.undecodable(path: url.path)
        }
        guard let entry = file.chips[deviceName] else {
            throw PeaksFileError.noEntry(deviceName: deviceName, path: url.path)
        }
        guard entry.source == .measured else {
            throw PeaksFileError.notMeasured(deviceName: deviceName,
                                             sourceLabel: entry.source.label,
                                             path: url.path)
        }
        return entry
    }
}

/// Why an explicitly named peaks file failed to yield measured peaks.
public enum PeaksFileError: Error, Equatable, CustomStringConvertible {
    case unreadable(path: String)
    case undecodable(path: String)
    case noEntry(deviceName: String, path: String)
    case notMeasured(deviceName: String, sourceLabel: String, path: String)

    public var description: String {
        switch self {
        case let .unreadable(path):
            return "cannot read peaks file \(path)"
        case let .undecodable(path):
            return "cannot parse \(path) as a peaks file — expected the JSON `metalscope calibrate` writes"
        case let .noEntry(deviceName, path):
            return "no entry for '\(deviceName)' in \(path) — run `metalscope calibrate --peaks-file \(path)` on that machine"
        case let .notMeasured(deviceName, sourceLabel, path):
            return "peaks for '\(deviceName)' in \(path) are \(sourceLabel), not measured — run `metalscope calibrate --peaks-file \(path)`"
        }
    }
}
