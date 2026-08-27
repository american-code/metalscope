import Foundation

/// Element precision of a kernel's tensors — drives the analytic byte count.
public enum Precision: String, Codable, Sendable, CaseIterable {
    case fp32
    case fp16
    case bf16
    case int8

    public var bytesPerElement: Int {
        switch self {
        case .fp32: return 4
        case .fp16, .bf16: return 2
        case .int8: return 1
        }
    }

    /// Which measured ceiling this precision should be scored against.
    public var usesHalfPeak: Bool {
        switch self {
        case .fp16, .bf16, .int8: return true
        case .fp32: return false
        }
    }
}

/// An ML kernel shape whose FLOPs and compulsory memory traffic are known
/// *analytically* — this is the piece Instruments fundamentally doesn't have.
///
/// Byte counts are compulsory DRAM traffic assuming perfect on-chip reuse
/// (each input read once, each output written once). Real kernels move more;
/// treating the compulsory count as the denominator makes arithmetic intensity
/// an upper bound and keeps the roofline placement conservative — a kernel that
/// looks bandwidth-bound under this model really is.
public enum KernelShape: Hashable, Sendable {
    /// C[m,n] = A[m,k] * B[k,n]
    case gemm(m: Int, n: Int, k: Int)
    /// Scaled dot-product attention: batch, heads, sequence, head dim.
    /// FLOPs count the two matmuls (QK^T and PV); softmax is transcendental
    /// overhead and is deliberately excluded. Bytes assume a fused kernel where
    /// the s x s score matrix never reaches DRAM.
    case attention(b: Int, h: Int, s: Int, d: Int)
    /// Unary/binary elementwise op over n elements: one read, one write.
    case elementwise(n: Int)
    /// Layer/RMS norm over n elements: two-pass statistics, then normalize.
    case norm(n: Int)
    /// Escape hatch for kernels with no registered shape: caller supplies the
    /// analytic numbers (or zeros, which reports as "unmodeled").
    case opaque(flops: Double, bytes: Double)

    /// Total floating-point operations for one invocation of this shape.
    public var flops: Double {
        switch self {
        case let .gemm(m, n, k):
            return 2.0 * Double(m) * Double(n) * Double(k)
        case let .attention(b, h, s, d):
            // QK^T: 2*b*h*s*s*d, then P*V: 2*b*h*s*s*d.
            return 4.0 * Double(b) * Double(h) * Double(s) * Double(s) * Double(d)
        case let .elementwise(n):
            return Double(n)
        case let .norm(n):
            // sum, sum-of-squares, subtract, scale, shift ~ 5 ops/element.
            return 5.0 * Double(n)
        case let .opaque(flops, _):
            return flops
        }
    }

    /// Compulsory bytes moved to/from DRAM for one invocation.
    public func bytes(precision: Precision) -> Double {
        let e = Double(precision.bytesPerElement)
        switch self {
        case let .gemm(m, n, k):
            return e * (Double(m) * Double(k) + Double(k) * Double(n) + Double(m) * Double(n))
        case let .attention(b, h, s, d):
            // Q, K, V in and O out.
            return e * 4.0 * Double(b) * Double(h) * Double(s) * Double(d)
        case let .elementwise(n):
            return e * 2.0 * Double(n)
        case let .norm(n):
            return e * 2.0 * Double(n)
        case let .opaque(_, bytes):
            return bytes
        }
    }

    /// FLOPs per byte at the given precision — the roofline x-axis.
    public func arithmeticIntensity(precision: Precision) -> Double {
        let b = bytes(precision: precision)
        guard b > 0 else { return 0 }
        return flops / b
    }

    /// Compact human-readable form used in tables and as part of the diff key.
    public var descriptionText: String {
        switch self {
        case let .gemm(m, n, k): return "gemm \(m)x\(n)x\(k)"
        case let .attention(b, h, s, d): return "attn b\(b) h\(h) s\(s) d\(d)"
        case let .elementwise(n): return "elem n=\(n)"
        case let .norm(n): return "norm n=\(n)"
        case let .opaque(f, b): return "opaque \(KernelShape.si(f))F/\(KernelShape.si(b))B"
        }
    }

    /// Deterministic SI-ish rendering — `descriptionText` is part of the diff
    /// alignment key, so this must produce identical text for identical inputs.
    private static func si(_ value: Double) -> String {
        let units: [(Double, String)] = [(1e12, "T"), (1e9, "G"), (1e6, "M"), (1e3, "k")]
        for (scale, suffix) in units where abs(value) >= scale {
            return String(format: "%.2f%@", value / scale, suffix)
        }
        return String(format: "%.0f", value)
    }

    public var kindName: String {
        switch self {
        case .gemm: return "gemm"
        case .attention: return "attention"
        case .elementwise: return "elementwise"
        case .norm: return "norm"
        case .opaque: return "opaque"
        }
    }
}

// MARK: - Stable JSON encoding

extension KernelShape: Codable {
    private enum CodingKeys: String, CodingKey {
        case kind, m, n, k, b, h, s, d, flops, bytes
    }

    public func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(kindName, forKey: .kind)
        switch self {
        case let .gemm(m, n, k):
            try c.encode(m, forKey: .m); try c.encode(n, forKey: .n); try c.encode(k, forKey: .k)
        case let .attention(b, h, s, d):
            try c.encode(b, forKey: .b); try c.encode(h, forKey: .h)
            try c.encode(s, forKey: .s); try c.encode(d, forKey: .d)
        case let .elementwise(n):
            try c.encode(n, forKey: .n)
        case let .norm(n):
            try c.encode(n, forKey: .n)
        case let .opaque(f, b):
            try c.encode(f, forKey: .flops); try c.encode(b, forKey: .bytes)
        }
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        let kind = try c.decode(String.self, forKey: .kind)
        switch kind {
        case "gemm":
            self = .gemm(m: try c.decode(Int.self, forKey: .m),
                         n: try c.decode(Int.self, forKey: .n),
                         k: try c.decode(Int.self, forKey: .k))
        case "attention":
            self = .attention(b: try c.decode(Int.self, forKey: .b),
                              h: try c.decode(Int.self, forKey: .h),
                              s: try c.decode(Int.self, forKey: .s),
                              d: try c.decode(Int.self, forKey: .d))
        case "elementwise":
            self = .elementwise(n: try c.decode(Int.self, forKey: .n))
        case "norm":
            self = .norm(n: try c.decode(Int.self, forKey: .n))
        case "opaque":
            self = .opaque(flops: try c.decode(Double.self, forKey: .flops),
                           bytes: try c.decode(Double.self, forKey: .bytes))
        default:
            throw DecodingError.dataCorruptedError(forKey: .kind, in: c,
                                                   debugDescription: "unknown kernel shape kind '\(kind)'")
        }
    }
}
