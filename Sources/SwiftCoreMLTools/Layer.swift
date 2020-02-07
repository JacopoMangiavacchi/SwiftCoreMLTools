public enum LayerType : String, Codable {
    case innerProduct
    case linear, reLu, leakyReLu, thresholdedReLu, preLu, tanh, scaledTanh, sigmoid, sigmoidHard, elu, softsign, softplus, parametricSoftplus

    var metatype: Layer.Type {
        switch self {
        // Real Layers
        case .innerProduct:
            return InnerProduct.self

        //Activation Functions
        case .linear:
            return Linear.self

        case .reLu:
            return ReLu.self

        case .leakyReLu:
            return LeakyReLu.self

        case .thresholdedReLu:
            return ThresholdedReLu.self

        case .preLu:
            return PreLu.self

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
        }
    }
}

public protocol Layer : Codable {
    static var type: LayerType { get }
}

struct AnyLayer : Codable {
    var base: Layer

    init(_ base: Layer) {
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
