public struct InnerProduct : Layer {
    public static let type = LayerType.innerProduct
    public let name: String
    public let input: [String]
    public let output: [String]
    public let weights: [Float]
    public let bias: [Float]

    public let inputChannels: UInt
    public let outputChannels: UInt
    public let updatable: Bool
}

public struct EdgeSizes : Codable {
    public let startEdgeSize: UInt
    public let endEdgeSize: UInt
}

public struct PaddingAmount : Codable {
    public let borderAmounts: [EdgeSizes]
    public let offset: [UInt]
}

public enum PaddingMode : String, Codable {
    case bottomRightHeavy
    case topLeftHeavy
}

public enum PaddingType {
    case valid(amount: PaddingAmount)
    case same(mode: PaddingMode)
}

public enum PaddingCodableError : Error {
    case errorDecoding
}

extension PaddingType : Codable {
    enum CodingKeys: String, CodingKey {
        case valid, same
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        if let value = try container.decodeIfPresent(PaddingAmount.self, forKey: .valid) {
            self = .valid(amount: value)
        }
        else if let value = try container.decodeIfPresent(PaddingMode.self, forKey: .same) {
            self = .same(mode: value)
        }
        else {
            throw PaddingCodableError.errorDecoding
        }
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)

        switch self {
        case .valid(let amount):
            try container.encode(amount, forKey: .valid)
        case .same(let mode):
            try container.encode(mode, forKey: .same)
        }
    }
}

public struct Convolution : Layer {
    public static let type = LayerType.convolution
    public let name: String
    public let input: [String]
    public let output: [String]
    public let weights: [Float]
    public let bias: [Float]

    public let outputChannels: UInt
    public let kernelChannels: UInt
    public let nGroups: UInt
    public let kernelSize: [UInt]
    public let stride: [UInt]
    public let dilationFactor: [UInt]
    public let paddingType: PaddingType
    public let outputShape: [UInt]
    public let deconvolution: Bool
    public let updatable: Bool
}

public struct Pooling : Layer {
    public static let type = LayerType.pooling
    public let name: String
    public let input: [String]
    public let output: [String]
    public let weights: [Float]
    public let bias: [Float]
}

public struct Embedding : Layer {
    public static let type = LayerType.embedding
    public let name: String
    public let input: [String]
    public let output: [String]
    public let weights: [Float]
    public let bias: [Float]
}

public struct SimpleRecurrent : Layer {
    public static let type = LayerType.simpleRecurrent
    public let name: String
    public let input: [String]
    public let output: [String]
    public let weights: [Float]
    public let bias: [Float]
}

public struct Gru : Layer {
    public static let type = LayerType.gru
    public let name: String
    public let input: [String]
    public let output: [String]
    public let weights: [Float]
    public let bias: [Float]
}

public struct UniDirectionalLstm : Layer {
    public static let type = LayerType.uniDirectionalLstm
    public let name: String
    public let input: [String]
    public let output: [String]
    public let weights: [Float]
    public let bias: [Float]
}

public struct BiDirectionalLstm : Layer {
    public static let type = LayerType.biDirectionalLstm
    public let name: String
    public let input: [String]
    public let output: [String]
    public let weights: [Float]
    public let bias: [Float]
}
