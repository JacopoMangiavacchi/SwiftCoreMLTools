public struct Linear : Layer, Codable {
    public static let type = LayerType.linear
}

public struct ReLu : Layer, Codable {
    public static let type = LayerType.reLu
}

public struct LeakyReLu : Layer, Codable {
    public static let type = LayerType.leakyReLu
}

public struct ThresholdedReLu : Layer, Codable {
    public static let type = LayerType.thresholdedReLu
}

public struct PreLu : Layer, Codable {
    public static let type = LayerType.preLu
}

public struct Tanh : Layer, Codable {
    public static let type = LayerType.tanh
}

public struct ScaledTanh : Layer, Codable {
    public static let type = LayerType.tanh
}

public struct Sigmoid : Layer, Codable {
    public static let type = LayerType.sigmoid
}

public struct SigmoidHard : Layer, Codable {
    public static let type = LayerType.sigmoidHard
}

public struct Elu : Layer, Codable {
    public static let type = LayerType.elu
}

public struct Softsign : Layer, Codable {
    public static let type = LayerType.softsign
}

public struct Softplus : Layer, Codable {
    public static let type = LayerType.softplus
}

public struct ParametricSoftplus : Layer, Codable {
    public static let type = LayerType.parametricSoftplus
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
