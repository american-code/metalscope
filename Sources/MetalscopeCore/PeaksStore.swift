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
}
