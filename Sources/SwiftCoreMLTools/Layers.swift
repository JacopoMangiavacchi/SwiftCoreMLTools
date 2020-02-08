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

public enum PaddingType : String, Codable {
//   enum OneOf_ConvolutionPaddingType: Equatable {
//     case valid(CoreML_Specification_ValidPadding)
//     case same(CoreML_Specification_SamePadding)
    case valid, same
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
