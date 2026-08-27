import Foundation

/// Hand-rolled argument parsing — metalscope has no external dependencies.
///
/// Supported forms: `--name value`, `--name=value`, bare `--flag`, and
/// positionals. Options that take a value must be declared per command so a
/// bare flag followed by a positional never swallows it.
struct Arguments {
    var positionals: [String] = []
    var options: [String: String] = [:]
    var flags: Set<String> = []

    static func parse(_ argv: [String], valueOptions: Set<String>) throws -> Arguments {
        var result = Arguments()
        var index = 0
        while index < argv.count {
            let token = argv[index]
            index += 1
            guard token.hasPrefix("--") else {
                result.positionals.append(token)
                continue
            }
            let body = String(token.dropFirst(2))
            if body.isEmpty {                      // "--" ends option parsing
                result.positionals.append(contentsOf: argv[index...])
                break
            }
            if let eq = body.firstIndex(of: "=") {
                result.options[String(body[body.startIndex..<eq])] = String(body[body.index(after: eq)...])
                continue
            }
            if valueOptions.contains(body) {
                guard index < argv.count else { throw CLIError.missingValue(token) }
                result.options[body] = argv[index]
                index += 1
            } else {
                result.flags.insert(body)
            }
        }
        return result
    }

    func string(_ name: String) -> String? { options[name] }

    func has(_ name: String) -> Bool { flags.contains(name) || options[name] != nil }

    func int(_ name: String, default defaultValue: Int) throws -> Int {
        guard let raw = options[name] else { return defaultValue }
        guard let value = Int(raw) else { throw CLIError.badValue(name, raw, "expected an integer") }
        return value
    }

    func double(_ name: String, default defaultValue: Double) throws -> Double {
        guard let raw = options[name] else { return defaultValue }
        guard let value = Double(raw) else { throw CLIError.badValue(name, raw, "expected a number") }
        return value
    }

    func intList(_ name: String, default defaultValue: [Int]) throws -> [Int] {
        guard let raw = options[name] else { return defaultValue }
        let parts = raw.split(whereSeparator: { $0 == "," || $0 == " " }).map(String.init)
        var values: [Int] = []
        for part in parts {
            guard let v = Int(part), v > 0 else {
                throw CLIError.badValue(name, raw, "expected comma-separated positive integers")
            }
            values.append(v)
        }
        guard !values.isEmpty else { throw CLIError.badValue(name, raw, "empty list") }
        return values
    }

    /// Reject typos rather than silently ignoring them.
    func rejectUnknown(_ known: Set<String>) throws {
        for name in flags.union(options.keys) where !known.contains(name) {
            throw CLIError.unknownOption(name)
        }
    }
}

enum CLIError: Error, CustomStringConvertible {
    case unknownCommand(String)
    case unknownOption(String)
    case missingValue(String)
    case badValue(String, String, String)
    case missingArgument(String)
    case message(String)

    var description: String {
        switch self {
        case let .unknownCommand(name): return "unknown command '\(name)' — run `metalscope help`"
        case let .unknownOption(name): return "unknown option '--\(name)'"
        case let .missingValue(token): return "option '\(token)' needs a value"
        case let .badValue(name, raw, why): return "bad value '\(raw)' for --\(name): \(why)"
        case let .missingArgument(what): return "missing argument: \(what)"
        case let .message(text): return text
        }
    }
}

enum Terminal {
    static func out(_ text: String) {
        print(text)
    }

    static func error(_ text: String) {
        FileHandle.standardError.write((text + "\n").data(using: .utf8)!)
    }
}
