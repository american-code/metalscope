import Foundation
import Metal
import MetalPerformanceShaders
import MetalscopeCapture
import MetalscopeCore

/// The workloads `calibrate` and `bench` run, all captured through
/// `MetalscopeCapture` so both commands measure via the same path a user's own
/// instrumented app would.
///
/// Each `make*` call allocates and fills its buffers once and returns a
/// `Runner`; `run` then encodes it N times into a single captured region. That
/// split matters for `calibrate`, which probes once to pick an iteration count
/// and then re-runs the same prepared workload several times.
struct Workloads {
    let session: CaptureSession
    let library: MetalKernels.Library
    var device: MTLDevice { session.device }

    init(session: CaptureSession) throws {
        self.session = session
        self.library = try MetalKernels.Library(device: session.device)
    }

    enum WorkloadError: Error, CustomStringConvertible {
        case allocationFailed(String)
        case mpsUnavailable

        var description: String {
            switch self {
            case let .allocationFailed(what): return "could not allocate \(what)"
            case .mpsUnavailable: return "MetalPerformanceShaders is not available on this device"
            }
        }
    }

    /// A prepared workload: buffers are already resident and filled.
    struct Runner {
        let shape: KernelShape
        let precision: Precision
        let notes: [String: String]
        let encode: (CaptureRegion, Int) throws -> Void
    }

    /// Encode `runner` `iterations` times into one captured region, `repeats`
    /// times over.
    /// - Parameter keep: false discards the record (warmup / probe runs).
    @discardableResult
    func run(_ runner: Runner,
             label: String,
             iterations: Int,
             repeats: Int = 1,
             keep: Bool = true) throws -> KernelRecord {
        try session.capture(label: label, shape: runner.shape, precision: runner.precision,
                            iterations: iterations, repeats: repeats,
                            notes: runner.notes) { region in
            try runner.encode(region, iterations)
        }
        let record = session.records[session.records.count - 1]
        if !keep { session.dropLast(1) }
        return record
    }

    /// Warm up (first dispatch pays pipeline/allocator costs), then pick an
    /// iteration count that keeps the GPU busy for ~`targetSeconds` — long
    /// enough for Apple's GPU to ramp its clocks, which is the difference
    /// between measuring 0.9 TF and 3.4 TF on the same GEMM.
    func autoIterations(_ runner: Runner,
                        label: String,
                        targetSeconds: Double,
                        minimum: Int = 3,
                        maximum: Int = 500) throws -> Int {
        let probe = try run(runner, label: label, iterations: 1, keep: false)
        guard probe.durationSeconds > 0 else { return minimum }
        let estimate = Int((targetSeconds / probe.durationSeconds).rounded())
        return max(minimum, min(maximum, estimate))
    }

    private func makeBuffer(_ length: Int, _ what: String) throws -> MTLBuffer {
        guard let buffer = device.makeBuffer(length: length, options: .storageModePrivate) else {
            throw WorkloadError.allocationFailed("\(what) (\(Fmt.bytes(Double(length))))")
        }
        return buffer
    }

    // MARK: - GEMM (MPSMatrixMultiplication)

    /// Square-ish GEMM through MPS. MPS creates its own encoders, so these
    /// regions are timed with command-buffer GPU time — real GPU time, just at
    /// command-buffer granularity, which is why each region is its own buffer.
    func makeGEMM(m: Int, n: Int, k: Int, precision: Precision) throws -> Runner {
        guard MPSSupportsMTLDevice(device) else { throw WorkloadError.mpsUnavailable }
        let dataType: MPSDataType = precision == .fp16 ? .float16 : .float32
        let isHalf = precision == .fp16

        func descriptor(rows: Int, columns: Int) -> MPSMatrixDescriptor {
            let rowBytes = MPSMatrixDescriptor.rowBytes(forColumns: columns, dataType: dataType)
            return MPSMatrixDescriptor(rows: rows, columns: columns, rowBytes: rowBytes, dataType: dataType)
        }

        let descA = descriptor(rows: m, columns: k)
        let descB = descriptor(rows: k, columns: n)
        let descC = descriptor(rows: m, columns: n)
        let bufA = try makeBuffer(descA.rowBytes * m, "GEMM A")
        let bufB = try makeBuffer(descB.rowBytes * k, "GEMM B")
        let bufC = try makeBuffer(descC.rowBytes * m, "GEMM C")

        let elementSize = precision.bytesPerElement
        try MetalKernels.fill(buffer: bufA, elementCount: descA.rowBytes * m / elementSize,
                              half: isHalf, library: library, queue: session.commandQueue, value: 0.25)
        try MetalKernels.fill(buffer: bufB, elementCount: descB.rowBytes * k / elementSize,
                              half: isHalf, library: library, queue: session.commandQueue, value: 0.5)

        let multiply = MPSMatrixMultiplication(device: device,
                                               transposeLeft: false, transposeRight: false,
                                               resultRows: m, resultColumns: n, interiorColumns: k,
                                               alpha: 1.0, beta: 0.0)
        let matA = MPSMatrix(buffer: bufA, descriptor: descA)
        let matB = MPSMatrix(buffer: bufB, descriptor: descB)
        let matC = MPSMatrix(buffer: bufC, descriptor: descC)

        return Runner(shape: .gemm(m: m, n: n, k: k),
                      precision: precision,
                      notes: ["backend": "MPSMatrixMultiplication"]) { region, count in
            for _ in 0..<count {
                multiply.encode(commandBuffer: region.commandBuffer,
                                leftMatrix: matA, rightMatrix: matB, resultMatrix: matC)
            }
        }
    }

    // MARK: - Streaming triad (custom MSL)

    enum TriadVariant: String {
        case scalar
        case vec4

        var functionName: String { self == .scalar ? "triad_scalar" : "triad_vec4" }
        var elementsPerThread: Int { self == .scalar ? 1 : 4 }
    }

    /// a = b + s*c over `elements` floats: 2 reads + 1 write per element, so the
    /// bytes are modeled explicitly (`.opaque`) rather than by `.elementwise`,
    /// which assumes a single input stream.
    func makeTriad(elements rawElements: Int, variant: TriadVariant) throws -> Runner {
        let elements = (rawElements / 4) * 4
        let byteLength = elements * MemoryLayout<Float>.size
        let bufA = try makeBuffer(byteLength, "triad a")
        let bufB = try makeBuffer(byteLength, "triad b")
        let bufC = try makeBuffer(byteLength, "triad c")
        try MetalKernels.fill(buffer: bufB, elementCount: elements, half: false,
                              library: library, queue: session.commandQueue, value: 1.0)
        try MetalKernels.fill(buffer: bufC, elementCount: elements, half: false,
                              library: library, queue: session.commandQueue, value: 2.0)

        let pipeline = try library.pipeline(variant.functionName)
        let threads = elements / variant.elementsPerThread

        return Runner(shape: .opaque(flops: 2.0 * Double(elements), bytes: 3.0 * Double(byteLength)),
                      precision: .fp32,
                      notes: ["variant": variant.rawValue, "kernel": variant.functionName]) { region, count in
            var scalar: Float = 3.0
            for _ in 0..<count {
                let encoder = try region.makeComputeCommandEncoder(label: "triad_\(variant.rawValue)")
                encoder.setComputePipelineState(pipeline)
                encoder.setBuffer(bufA, offset: 0, index: 0)
                encoder.setBuffer(bufB, offset: 0, index: 1)
                encoder.setBuffer(bufC, offset: 0, index: 2)
                encoder.setBytes(&scalar, length: MemoryLayout<Float>.size, index: 3)
                MetalKernels.dispatch(encoder, pipeline: pipeline, threads: threads, region: region)
                encoder.endEncoding()
            }
        }
    }

    // MARK: - Elementwise scale

    /// - Parameter threadgroupWidth: threads per threadgroup, or nil for the
    ///   default 256. `bench --variant baseline` passes a deliberately ragged
    ///   width so the occupancy analysis has a real dispatch to flag; see
    ///   `BenchCommand.raggedThreadgroupWidth`.
    func makeScale(elements rawElements: Int,
                   vectorized: Bool,
                   threadgroupWidth: Int? = nil) throws -> Runner {
        let elements = (rawElements / 4) * 4
        let byteLength = elements * MemoryLayout<Float>.size
        let bufY = try makeBuffer(byteLength, "scale y")
        let bufX = try makeBuffer(byteLength, "scale x")
        try MetalKernels.fill(buffer: bufX, elementCount: elements, half: false,
                              library: library, queue: session.commandQueue, value: 1.5)

        let pipeline = try library.pipeline(vectorized ? "scale_vec4" : "scale_scalar")
        let threads = vectorized ? elements / 4 : elements
        let width = threadgroupWidth ?? MetalKernels.defaultThreadgroupWidth(pipeline)

        return Runner(shape: .elementwise(n: elements),
                      precision: .fp32,
                      notes: ["variant": vectorized ? "vec4" : "scalar",
                              "threadgroupWidth": "\(width)"]) { region, count in
            var scalar: Float = 1.125
            for _ in 0..<count {
                let encoder = try region.makeComputeCommandEncoder(label: "scale")
                encoder.setComputePipelineState(pipeline)
                encoder.setBuffer(bufY, offset: 0, index: 0)
                encoder.setBuffer(bufX, offset: 0, index: 1)
                encoder.setBytes(&scalar, length: MemoryLayout<Float>.size, index: 2)
                MetalKernels.dispatch(encoder, pipeline: pipeline, threads: threads,
                                      threadgroupWidth: width, region: region)
                encoder.endEncoding()
            }
        }
    }

    // MARK: - RMS norm

    func makeRMSNorm(rows: Int, width: Int) throws -> Runner {
        let elements = rows * width
        let byteLength = elements * MemoryLayout<Float>.size
        let bufY = try makeBuffer(byteLength, "norm y")
        let bufX = try makeBuffer(byteLength, "norm x")
        try MetalKernels.fill(buffer: bufX, elementCount: elements, half: false,
                              library: library, queue: session.commandQueue, value: 0.75)

        let pipeline = try library.pipeline("rmsnorm_f32")
        let threadgroupWidth = min(pipeline.maxTotalThreadsPerThreadgroup, 256)

        return Runner(shape: .norm(n: elements),
                      precision: .fp32,
                      notes: ["rows": "\(rows)", "width": "\(width)"]) { region, count in
            var widthValue = UInt32(width)
            for _ in 0..<count {
                let encoder = try region.makeComputeCommandEncoder(label: "rmsnorm")
                encoder.setComputePipelineState(pipeline)
                encoder.setBuffer(bufY, offset: 0, index: 0)
                encoder.setBuffer(bufX, offset: 0, index: 1)
                encoder.setBytes(&widthValue, length: MemoryLayout<UInt32>.size, index: 2)
                region.dispatchThreadgroups(encoder, pipeline: pipeline,
                                            threadgroupsPerGrid: MTLSize(width: rows, height: 1, depth: 1),
                                            threadsPerThreadgroup: MTLSize(width: threadgroupWidth,
                                                                           height: 1, depth: 1))
                encoder.endEncoding()
            }
        }
    }

    // MARK: - Attention (unfused MPS composite)

    /// Scaled dot-product attention built from MPS matmuls + softmax.
    ///
    /// The `.attention` shape models a *fused* kernel, so the s x s scores never
    /// appear in its byte count. This composite does spill them to DRAM — the
    /// gap between its efficiency here and 100% is largely the cost of not
    /// fusing, which is the sort of thing the report exists to show.
    func makeAttention(b: Int, h: Int, s: Int, d: Int) throws -> Runner {
        guard MPSSupportsMTLDevice(device) else { throw WorkloadError.mpsUnavailable }
        let heads = b * h
        let qkvRowBytes = MPSMatrixDescriptor.rowBytes(forColumns: d, dataType: .float32)
        let scoreRowBytes = MPSMatrixDescriptor.rowBytes(forColumns: s, dataType: .float32)
        let qkvBytesPerHead = qkvRowBytes * s
        let scoreBytesPerHead = scoreRowBytes * s

        let bufQ = try makeBuffer(qkvBytesPerHead * heads, "attention Q")
        let bufK = try makeBuffer(qkvBytesPerHead * heads, "attention K")
        let bufV = try makeBuffer(qkvBytesPerHead * heads, "attention V")
        let bufScores = try makeBuffer(scoreBytesPerHead * heads, "attention scores")
        let bufOut = try makeBuffer(qkvBytesPerHead * heads, "attention out")

        for (buffer, value) in [(bufQ, Float(0.1)), (bufK, Float(0.2)), (bufV, Float(0.3))] {
            try MetalKernels.fill(buffer: buffer,
                                  elementCount: qkvBytesPerHead * heads / MemoryLayout<Float>.size,
                                  half: false, library: library, queue: session.commandQueue, value: value)
        }

        let qkvDesc = MPSMatrixDescriptor(rows: s, columns: d, rowBytes: qkvRowBytes, dataType: .float32)
        let scoreDesc = MPSMatrixDescriptor(rows: s, columns: s, rowBytes: scoreRowBytes, dataType: .float32)
        let scores = MPSMatrixMultiplication(device: device,
                                             transposeLeft: false, transposeRight: true,
                                             resultRows: s, resultColumns: s, interiorColumns: d,
                                             alpha: 1.0 / Double(d).squareRoot(), beta: 0.0)
        let apply = MPSMatrixMultiplication(device: device,
                                            transposeLeft: false, transposeRight: false,
                                            resultRows: s, resultColumns: d, interiorColumns: s,
                                            alpha: 1.0, beta: 0.0)
        let softmax = MPSMatrixSoftMax(device: device)

        return Runner(shape: .attention(b: b, h: h, s: s, d: d),
                      precision: .fp32,
                      notes: ["backend": "MPS matmul + softmax",
                              "fusion": "unfused: s x s scores round-trip through DRAM"]) { region, count in
            for _ in 0..<count {
                for head in 0..<heads {
                    let qkvOffset = head * qkvBytesPerHead
                    let scoreOffset = head * scoreBytesPerHead
                    let q = MPSMatrix(buffer: bufQ, offset: qkvOffset, descriptor: qkvDesc)
                    let k = MPSMatrix(buffer: bufK, offset: qkvOffset, descriptor: qkvDesc)
                    let v = MPSMatrix(buffer: bufV, offset: qkvOffset, descriptor: qkvDesc)
                    let p = MPSMatrix(buffer: bufScores, offset: scoreOffset, descriptor: scoreDesc)
                    let o = MPSMatrix(buffer: bufOut, offset: qkvOffset, descriptor: qkvDesc)
                    scores.encode(commandBuffer: region.commandBuffer,
                                  leftMatrix: q, rightMatrix: k, resultMatrix: p)
                    softmax.encode(commandBuffer: region.commandBuffer, inputMatrix: p, resultMatrix: p)
                    apply.encode(commandBuffer: region.commandBuffer,
                                 leftMatrix: p, rightMatrix: v, resultMatrix: o)
                }
            }
        }
    }
}
