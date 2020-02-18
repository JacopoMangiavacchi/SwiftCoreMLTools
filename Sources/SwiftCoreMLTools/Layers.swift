public struct InnerProduct : TrainableLayer {
    public static let type = LayerType.innerProduct
    public let name: String
    public let input: [String]
    public let output: [String]
    public let weight: [Float]
    public let bias: [Float]

    public let inputChannels: UInt
    public let outputChannels: UInt
    public let updatable: Bool

    public init(name: String, input: [String], output: [String], weight: [Float]? = nil, bias: [Float]? = nil, inputChannels: UInt, outputChannels: UInt, updatable: Bool = false) {
        self.name = name
        self.input = input
        self.output = output
    
        if let weight = weight, let bias = bias {
            self.weight = weight
            self.bias = bias
        }
        else {
            let (weight, bias) = Self.getUniformWeigthsAndBias(inputChannels: inputChannels, outputChannels: outputChannels)
            self.weight = weight
            self.bias = bias
        }

        self.inputChannels = inputChannels
        self.outputChannels = outputChannels
        self.updatable = updatable
    }
}

public struct EdgeSizes : Codable {
    public let startEdgeSize: UInt
    public let endEdgeSize: UInt

    public init(startEdgeSize: UInt, endEdgeSize: UInt) {
        self.startEdgeSize = startEdgeSize
        self.endEdgeSize = endEdgeSize
    }
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
    public let weight: [Float]
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

    public init(name: String, input: [String], output: [String], weight: [Float], bias: [Float], outputChannels: UInt, kernelChannels: UInt, 
                nGroups: UInt, kernelSize: [UInt], stride: [UInt], dilationFactor: [UInt], paddingType: PaddingType, outputShape: [UInt],
                deconvolution: Bool, updatable: Bool = false) {
        self.name = name
        self.input = input
        self.output = output
        self.weight = weight
        self.bias = bias
        
        self.outputChannels = outputChannels
        self.kernelChannels = kernelChannels
        self.nGroups = nGroups
        self.kernelSize = kernelSize
        self.stride = stride
        self.dilationFactor = dilationFactor
        self.paddingType = paddingType
        self.outputShape = outputShape
        self.deconvolution = deconvolution
        self.updatable = updatable
    }
}

// TODO
public struct Pooling : BaseLayer {
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
    public let weight: [Float]
    public let bias: [Float]
}

// TODO
public struct SimpleRecurrent : TrainableLayer {
    public static let type = LayerType.simpleRecurrent
    public let name: String
    public let input: [String]
    public let output: [String]
    public let weight: [Float]
    public let bias: [Float]
}

// TODO
public struct Gru : TrainableLayer {
    public static let type = LayerType.gru
    public let name: String
    public let input: [String]
    public let output: [String]
    public let weight: [Float]
    public let bias: [Float]
}

// TODO
public struct UniDirectionalLstm : TrainableLayer {
    public static let type = LayerType.uniDirectionalLstm
    public let name: String
    public let input: [String]
    public let output: [String]
    public let weight: [Float]
    public let bias: [Float]
}

// TODO
public struct BiDirectionalLstm : TrainableLayer {
    public static let type = LayerType.biDirectionalLstm
    public let name: String
    public let input: [String]
    public let output: [String]
    public let weight: [Float]
    public let bias: [Float]
}
