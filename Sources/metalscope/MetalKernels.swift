import Foundation
import Metal
import MetalscopeCapture

/// Runtime-compiled MSL used by `calibrate` and `bench`. Kept as source (rather
/// than a .metal file in the bundle) so the executable stays a single artifact
/// and `swift build` needs no Metal toolchain integration.
enum MetalKernels {
    static let source = """
    #include <metal_stdlib>
    using namespace metal;

    // STREAM triad: a = b + s*c. Three streams touched per element.
    kernel void triad_scalar(device float *a [[buffer(0)]],
                             device const float *b [[buffer(1)]],
                             device const float *c [[buffer(2)]],
                             constant float &s [[buffer(3)]],
                             uint i [[thread_position_in_grid]]) {
        a[i] = b[i] + s * c[i];
    }

    // Same traffic, 4 elements per thread: fewer threads, wider loads.
    kernel void triad_vec4(device float4 *a [[buffer(0)]],
                           device const float4 *b [[buffer(1)]],
                           device const float4 *c [[buffer(2)]],
                           constant float &s [[buffer(3)]],
                           uint i [[thread_position_in_grid]]) {
        a[i] = b[i] + s * c[i];
    }

    // Elementwise scale: one read, one write -- matches .elementwise(n:) exactly.
    kernel void scale_scalar(device float *y [[buffer(0)]],
                             device const float *x [[buffer(1)]],
                             constant float &s [[buffer(2)]],
                             uint i [[thread_position_in_grid]]) {
        y[i] = s * x[i];
    }

    kernel void scale_vec4(device float4 *y [[buffer(0)]],
                           device const float4 *x [[buffer(1)]],
                           constant float &s [[buffer(2)]],
                           uint i [[thread_position_in_grid]]) {
        y[i] = s * x[i];
    }

    // RMS norm, one threadgroup per row. Two passes over a cache-resident row,
    // which is what .norm(n:)'s 2n-byte model assumes.
    kernel void rmsnorm_f32(device float *y [[buffer(0)]],
                            device const float *x [[buffer(1)]],
                            constant uint &width [[buffer(2)]],
                            uint row [[threadgroup_position_in_grid]],
                            uint lane [[thread_position_in_threadgroup]],
                            uint tgSize [[threads_per_threadgroup]],
                            uint simdLane [[thread_index_in_simdgroup]],
                            uint simdGroup [[simdgroup_index_in_threadgroup]]) {
        threadgroup float partials[32];
        device const float *xr = x + (size_t)row * width;
        device float *yr = y + (size_t)row * width;

        float acc = 0.0f;
        for (uint i = lane; i < width; i += tgSize) {
            float v = xr[i];
            acc += v * v;
        }
        acc = simd_sum(acc);
        if (simdLane == 0) { partials[simdGroup] = acc; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (simdGroup == 0) {
            uint groups = max(1u, tgSize / 32u);
            float v = (simdLane < groups) ? partials[simdLane] : 0.0f;
            v = simd_sum(v);
            if (simdLane == 0) { partials[0] = v; }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float scale = rsqrt(partials[0] / (float)width + 1e-6f);
        for (uint i = lane; i < width; i += tgSize) {
            yr[i] = xr[i] * scale;
        }
    }

    kernel void fill_f32(device float *x [[buffer(0)]],
                         constant float &v [[buffer(1)]],
                         uint i [[thread_position_in_grid]]) {
        x[i] = v + (float)(i & 15u) * 0.0125f;
    }

    kernel void fill_f16(device half *x [[buffer(0)]],
                         constant float &v [[buffer(1)]],
                         uint i [[thread_position_in_grid]]) {
        x[i] = (half)(v + (float)(i & 15u) * 0.0125f);
    }
    """

    /// Compiled pipelines, built once per process.
    final class Library {
        let device: MTLDevice
        private let library: MTLLibrary
        private var cache: [String: MTLComputePipelineState] = [:]

        init(device: MTLDevice) throws {
            self.device = device
            // Default compile options: Metal's default math mode is already the
            // fast one, and setting it explicitly costs an availability dance.
            self.library = try device.makeLibrary(source: MetalKernels.source, options: nil)
        }

        func pipeline(_ name: String) throws -> MTLComputePipelineState {
            if let cached = cache[name] { return cached }
            guard let function = library.makeFunction(name: name) else {
                throw MetalKernelsError.missingFunction(name)
            }
            let pipeline = try device.makeComputePipelineState(function: function)
            cache[name] = pipeline
            return pipeline
        }
    }

    enum MetalKernelsError: Error, CustomStringConvertible {
        case missingFunction(String)

        var description: String {
            switch self {
            case let .missingFunction(name): return "MSL function '\(name)' not found"
            }
        }
    }

    /// Fill a private buffer so measurements never run over uninitialized memory.
    static func fill(buffer: MTLBuffer,
                     elementCount: Int,
                     half: Bool,
                     library: Library,
                     queue: MTLCommandQueue,
                     value: Float = 0.5) throws {
        let pipeline = try library.pipeline(half ? "fill_f16" : "fill_f32")
        guard let commandBuffer = queue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        var v = value
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(buffer, offset: 0, index: 0)
        encoder.setBytes(&v, length: MemoryLayout<Float>.size, index: 1)
        dispatch(encoder, pipeline: pipeline, threads: elementCount)
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
    }

    /// Default 1-D threadgroup width: 256, or the pipeline's ceiling if lower.
    static func defaultThreadgroupWidth(_ pipeline: MTLComputePipelineState) -> Int {
        min(pipeline.maxTotalThreadsPerThreadgroup, 256)
    }

    /// 1-D dispatch with a threadgroup size the pipeline actually supports.
    ///
    /// Pass `region` to have the dispatch's static occupancy recorded into the
    /// trace; without it (the `fill` path, which is setup, not measurement) the
    /// dispatch is encoded plainly.
    static func dispatch(_ encoder: MTLComputeCommandEncoder,
                         pipeline: MTLComputePipelineState,
                         threads: Int,
                         threadgroupWidth: Int? = nil,
                         region: CaptureRegion? = nil) {
        let width = min(threadgroupWidth ?? defaultThreadgroupWidth(pipeline),
                        pipeline.maxTotalThreadsPerThreadgroup)
        let grid = MTLSize(width: threads, height: 1, depth: 1)
        let group = MTLSize(width: max(1, width), height: 1, depth: 1)
        if let region {
            region.dispatchThreads(encoder, pipeline: pipeline, threads: grid, threadsPerThreadgroup: group)
        } else {
            encoder.dispatchThreads(grid, threadsPerThreadgroup: group)
        }
    }
}
