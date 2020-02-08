public struct Linear : Activation {
    public static let type = LayerType.linear
    public let name: String
    public let input: [String]
    public let output: [String]
    public let inputChannels: UInt
    public let outputChannels: UInt

    public let alpha: Float = 0.0
    public let beta: Float = 0.0
}

public struct ReLu : Activation {
    public static let type = LayerType.reLu
    public let name: String
    public let input: [String]
    public let output: [String]
    public let inputChannels: UInt
    public let outputChannels: UInt
}

public struct LeakyReLu : Activation {
    public static let type = LayerType.leakyReLu
    public let name: String
    public let input: [String]
    public let output: [String]
    public let inputChannels: UInt
    public let outputChannels: UInt

    public let alpha: Float = 0.0
}

public struct ThresholdedReLu : Activation {
    public static let type = LayerType.thresholdedReLu
    public let name: String
    public let input: [String]
    public let output: [String]
    public let inputChannels: UInt
    public let outputChannels: UInt
}

public struct PreLu : Activation {
    public static let type = LayerType.preLu
    public let name: String
    public let input: [String]
    public let output: [String]
    public let inputChannels: UInt
    public let outputChannels: UInt
}

public struct Tanh : Activation {
    public static let type = LayerType.tanh
    public let name: String
    public let input: [String]
    public let output: [String]
    public let inputChannels: UInt
    public let outputChannels: UInt
}

public struct ScaledTanh : Activation {
    public static let type = LayerType.tanh
    public let name: String
    public let input: [String]
    public let output: [String]
    public let inputChannels: UInt
    public let outputChannels: UInt

    public let alpha: Float = 0.0
    public let beta: Float = 0.0
}

public struct Sigmoid : Activation {
    public static let type = LayerType.sigmoid
    public let name: String
    public let input: [String]
    public let output: [String]
    public let inputChannels: UInt
    public let outputChannels: UInt
}

public struct SigmoidHard : Activation {
    public static let type = LayerType.sigmoidHard
    public let name: String
    public let input: [String]
    public let output: [String]
    public let inputChannels: UInt
    public let outputChannels: UInt

    public let alpha: Float = 0.0
    public let beta: Float = 0.0
}

public struct Elu : Activation {
    public static let type = LayerType.elu
    public let name: String
    public let input: [String]
    public let output: [String]
    public let inputChannels: UInt
    public let outputChannels: UInt

    public let alpha: Float = 0.0
}

public struct Softsign : Activation {
    public static let type = LayerType.softsign
    public let name: String
    public let input: [String]
    public let output: [String]
    public let inputChannels: UInt
    public let outputChannels: UInt
}

public struct Softplus : Activation {
    public static let type = LayerType.softplus
    public let name: String
    public let input: [String]
    public let output: [String]
    public let inputChannels: UInt
    public let outputChannels: UInt
}

public struct ParametricSoftplus : Activation {
    public static let type = LayerType.parametricSoftplus
    public let name: String
    public let input: [String]
    public let output: [String]
    public let inputChannels: UInt
    public let outputChannels: UInt
}



// struct CoreML_Specification_ActivationReLU {
//   // SwiftProtobuf.Message conformance is added in an extension below. See the
//   // `Message` and `Message+*Additions` files in the SwiftProtobuf library for
//   // methods supported on all messages.

//   var unknownFields = SwiftProtobuf.UnknownStorage()

//   init() {}
// }

// CoreML_Specification_NeuralNetworkLayer.with {
//                         $0.name = layer.name
//                         $0.input = layer.input
//                         $0.output = layer.output
//                         $0.isUpdatable = layer.updatable

//                         switch

//                         $0.innerProduct = CoreML_Specification_InnerProductLayerParams.with {
//                             $0.inputChannels = UInt64(layer.inputChannels)
//                             $0.outputChannels = UInt64(layer.outputChannels)
//                             ...
//                         }
//                         $0.activation = CoreML_Specification_ActivationParams.with {

//                             switch 

//                             $0.linear = CoreML_Specification_ActivationLinear.with {
//                             $0.reLu = CoreML_Specification_ActivationReLU.with {
//                             $0.leakyReLu = CoreML_Specification_ActivationLeakyReLU.with {
//                             $0.thresholdedReLu = CoreML_Specification_ActivationThresholdedReLU.with {
//                             $0.preLu = CoreML_Specification_ActivationPReLU.with {
//                             $0.tanh = CoreML_Specification_ActivationTanh.with {
//                             $0.scaledTanh = CoreML_Specification_ActivationScaledTanh.with {
//                             $0.sigmoid = CoreML_Specification_ActivationSigmoid.with {
//                             $0.sigmoidHard = CoreML_Specification_ActivationSigmoidHard.with {
//                             $0.elu = CoreML_Specification_ActivationELU.with {
//                             $0.softsign = CoreML_Specification_ActivationSoftsign.with {
//                             $0.softplus = CoreML_Specification_ActivationSoftplus.with {
//                             $0.parametricSoftplus = CoreML_Specification_ActivationParametricSoftplus.with {
//                         }
