public enum LayerType : String, Codable {
    case innerProduct, convolution, pooling, embedding, simpleRecurrent, gru, uniDirectionalLstm, biDirectionalLstm, permute, flatten, concat 
    case linear, reLu, leakyReLu, thresholdedReLu, pReLu, tanh, scaledTanh, sigmoid, sigmoidHard, elu, softsign, softplus, parametricSoftplus
    case softmax

    var metatype: BaseLayer.Type {
        switch self {
        // Real Layers
        case .innerProduct:
            return InnerProduct.self
        case .convolution:
            return Convolution.self
        case .pooling:
            return Pooling.self
        case .embedding:
            return Embedding.self
        case .simpleRecurrent:
            return SimpleRecurrent.self
        case .gru:
            return Gru.self
        case .uniDirectionalLstm:
            return UniDirectionalLstm.self
        case .biDirectionalLstm:
            return BiDirectionalLstm.self
        case .permute:
            return Permute.self
        case .flatten:
            return Flatten.self
        case .concat:
            return Concat.self

        //Activation Functions
        case .linear:
            return Linear.self
        case .reLu:
            return ReLu.self
        case .leakyReLu:
            return LeakyReLu.self
        case .thresholdedReLu:
            return ThresholdedReLu.self
        case .pReLu:
            return PReLu.self
        case .tanh:
            return Tanh.self
        case .scaledTanh:
            return ScaledTanh.self
        case .sigmoid:
            return Sigmoid.self
        case .sigmoidHard:
            return SigmoidHard.self
        case .elu:
            return Elu.self
        case .softsign:
            return Softsign.self
        case .softplus:
            return Softplus.self
        case .parametricSoftplus:
            return ParametricSoftplus.self
            
        case .softmax:
            return Softmax.self
        }
    }
}

public protocol BaseLayer : Codable {
    static var type: LayerType { get }
    var name: String { get }
    var input: [String] { get }
    var output: [String] { get }
}

public protocol TrainableLayer : BaseLayer {
    var weight: [Float] { get }
    var bias: [Float] { get }
}

extension TrainableLayer {
    static func getUniformWeigthsAndBias(inputChannels: UInt, outputChannels: UInt) -> (weight: [Float], bias: [Float]) {
        let limit = Float(6.0 / Float(inputChannels + outputChannels)).squareRoot()

        let weight = (0..<(inputChannels*outputChannels)).map{ _ in Float.random(in: -limit...limit) }
        let bias = (0..<outputChannels).map{ _ in Float.random(in: -limit...limit) }

        return (weight, bias)        
    }
}

public protocol Activation : BaseLayer {
}


struct AnyLayer : Codable {
    var base: BaseLayer

    init(_ base: BaseLayer) {
        self.base = base
    }

    private enum CodingKeys : CodingKey {
        case type, base
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)

        let type = try container.decode(LayerType.self, forKey: .type)
        self.base = try type.metatype.init(from: container.superDecoder(forKey: .base))
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)

        try container.encode(type(of: base).type, forKey: .type)
        try base.encode(to: container.superEncoder(forKey: .base))
    }
}
