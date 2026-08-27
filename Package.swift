// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "metalscope",
    platforms: [.macOS(.v14)],
    products: [
        .executable(name: "metalscope", targets: ["metalscope"]),
        // Opt-in capture library: link this into a benchmark/app to emit traces.
        .library(name: "MetalscopeCapture", targets: ["MetalscopeCapture"]),
        // Trace schema + roofline math, no Metal required (usable in analysis tools).
        .library(name: "MetalscopeCore", targets: ["MetalscopeCore"]),
    ],
    targets: [
        .target(name: "MetalscopeCore"),
        .target(name: "MetalscopeCapture", dependencies: ["MetalscopeCore"]),
        .executableTarget(name: "metalscope", dependencies: ["MetalscopeCore", "MetalscopeCapture"]),
        .testTarget(name: "metalscopeTests", dependencies: ["MetalscopeCore", "MetalscopeCapture"]),
    ]
)
