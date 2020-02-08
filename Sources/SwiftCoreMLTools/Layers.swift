public struct InnerProduct : TrainableLayer {
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

public enum PaddingMode : String, Codable {
    case bottomRightHeavy
    case topLeftHeavy
}

public enum PaddingType {
    case valid(borderAmounts: [EdgeSizes])
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
        if let value = try container.decodeIfPresent([EdgeSizes].self, forKey: .valid) {
            self = .valid(borderAmounts: value)
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
        case .valid(let borderAmounts):
            try container.encode(borderAmounts, forKey: .valid)
        case .same(let mode):
            try container.encode(mode, forKey: .same)
        }
    }
}

public struct Convolution : TrainableLayer {
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

// TODO
public struct Pooling : Layer {
    public static let type = LayerType.pooling
    public let name: String
    public let input: [String]
    public let output: [String]
}

// TODO
public struct Embedding : TrainableLayer {
    public static let type = LayerType.embedding
    public let name: String
    public let input: [String]
    public let output: [String]
    public let weights: [Float]
    public let bias: [Float]
}

// TODO
public struct SimpleRecurrent : TrainableLayer {
    public static let type = LayerType.simpleRecurrent
    public let name: String
    public let input: [String]
    public let output: [String]
    public let weights: [Float]
    public let bias: [Float]
}

// TODO
public struct Gru : TrainableLayer {
    public static let type = LayerType.gru
    public let name: String
    public let input: [String]
    public let output: [String]
    public let weights: [Float]
    public let bias: [Float]
}

// TODO
public struct UniDirectionalLstm : TrainableLayer {
    public static let type = LayerType.uniDirectionalLstm
    public let name: String
    public let input: [String]
    public let output: [String]
    public let weights: [Float]
    public let bias: [Float]
}

// TODO
public struct BiDirectionalLstm : TrainableLayer {
    public static let type = LayerType.biDirectionalLstm
    public let name: String
    public let input: [String]
    public let output: [String]
    public let weights: [Float]
    public let bias: [Float]
}
