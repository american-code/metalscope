import Foundation
import MetalscopeCore

/// Run a child process that links `MetalscopeCapture`, collect its trace, and
/// report on it.
///
/// metalscope deliberately does not inject into arbitrary processes — Metal
/// offers no supported way to do that, and the alternatives (swizzling,
/// DYLD_INSERT_LIBRARIES) are exactly the kind of thing that makes a profiler
/// untrustworthy. The contract instead is: the target links MetalscopeCapture
/// and writes to `$METALSCOPE_TRACE`, which this command sets and then reads.
enum ProfileCommand {
    static let known: Set<String> = ["output", "no-report", "sort", "peaks-file"]
    static let valueOptions: Set<String> = ["output", "sort", "peaks-file"]

    static func run(_ args: Arguments) throws {
        try args.rejectUnknown(known)
        // Validate before spending a child process run on a typo'd flag.
        let sort = try ReportCommand.SortOrder.parse(args.string("sort"))
        guard let executable = args.positionals.first else {
            throw CLIError.missingArgument("<command> [args...] — e.g. `metalscope profile -- ./my-bench`")
        }
        let outputPath = args.string("output") ?? "metalscope-trace.json"
        let output = URL(fileURLWithPath: (outputPath as NSString).expandingTildeInPath)
        try? FileManager.default.removeItem(at: output)

        let process = Process()
        process.executableURL = resolve(executable)
        process.arguments = Array(args.positionals.dropFirst())
        var environment = ProcessInfo.processInfo.environment
        environment["METALSCOPE_TRACE"] = output.path
        environment["METALSCOPE_ACTIVE"] = "1"
        process.environment = environment

        Terminal.out("metalscope profile — running \(([executable] + process.arguments!).joined(separator: " "))")
        Terminal.out("  METALSCOPE_TRACE=\(output.path)")
        try process.run()
        process.waitUntilExit()

        guard process.terminationStatus == 0 else {
            throw CLIError.message("child exited with status \(process.terminationStatus)")
        }
        guard FileManager.default.fileExists(atPath: output.path) else {
            throw CLIError.message("""
                child wrote no trace at \(output.path).
                  The target must link the MetalscopeCapture library and call
                  CaptureSession.writeTrace() — see docs/TRACE-FORMAT.md.
                """)
        }

        Terminal.out("")
        guard !args.has("no-report") else { return }
        let trace = try TraceIO.read(from: output)
        let peaks = try ReportCommand.resolvePeaks(for: trace, args: args)
        try ReportCommand.printReport(trace: trace, path: output.path, peaks: peaks,
                                      sort: sort)
    }

    private static func resolve(_ executable: String) -> URL {
        if executable.contains("/") {
            return URL(fileURLWithPath: (executable as NSString).expandingTildeInPath)
        }
        let paths = (ProcessInfo.processInfo.environment["PATH"] ?? "").split(separator: ":")
        for path in paths {
            let candidate = URL(fileURLWithPath: String(path)).appendingPathComponent(executable)
            if FileManager.default.isExecutableFile(atPath: candidate.path) { return candidate }
        }
        return URL(fileURLWithPath: executable)
    }
}
