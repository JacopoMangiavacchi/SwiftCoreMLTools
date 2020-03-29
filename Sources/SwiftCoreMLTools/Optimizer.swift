public enum OptimizerType : String, Codable {
    case sgd, adam

    var metatype: Optimizer.Type {
        switch self {
        case .sgd:
            return SGD.self
        case .adam:
            return Adam.self
        }
    }
}

public protocol Optimizer : Codable {
    static var type: OptimizerType { get }
}

public struct SGD : Optimizer {
    public static let type = OptimizerType.sgd

    public let learningRateDefault: Double
    public let learningRateMax: Double
    public let miniBatchSizeDefault: UInt
    public let miniBatchSizeRange: [UInt]
    public let momentumDefault: Double
    public let momentumMax: Double

    public init(learningRateDefault: Double, learningRateMax: Double, miniBatchSizeDefault: UInt, miniBatchSizeRange: [UInt], momentumDefault: Double, momentumMax: Double) {
        self.learningRateDefault = learningRateDefault
        self.learningRateMax = learningRateMax
        self.miniBatchSizeDefault = miniBatchSizeDefault
        self.miniBatchSizeRange = miniBatchSizeRange
        self.momentumDefault = momentumDefault
        self.momentumMax = momentumMax
    }
}

public struct Adam : Optimizer {
    public static let type = OptimizerType.adam

    public let learningRateDefault: Double
    public let learningRateMax: Double
    public let miniBatchSizeDefault: UInt
    public let miniBatchSizeRange: [UInt]
    public let beta1Default: Double
    public let beta1Max: Double
    public let beta2Default: Double
    public let beta2Max: Double
    public let epsDefault: Double
    public let epsMax: Double

    public init(learningRateDefault: Double, learningRateMax: Double, miniBatchSizeDefault: UInt, miniBatchSizeRange: [UInt], beta1Default: Double, beta1Max: Double, beta2Default: Double, beta2Max: Double, epsDefault: Double, epsMax: Double) {
        self.learningRateDefault = learningRateDefault
        self.learningRateMax = learningRateMax
        self.miniBatchSizeDefault = miniBatchSizeDefault
        self.miniBatchSizeRange = miniBatchSizeRange
        self.beta1Default = beta1Default
        self.beta1Max = beta1Max
        self.beta2Default = beta2Default
        self.beta2Max = beta2Max
        self.epsDefault = epsDefault
        self.epsMax = epsMax
    }
}

struct AnyOptimizer : Codable {
    var base: Optimizer

    init(_ base: Optimizer) {
        self.base = base
    }

    private enum CodingKeys : CodingKey {
        case type, base
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)

        let type = try container.decode(OptimizerType.self, forKey: .type)
        self.base = try type.metatype.init(from: container.superDecoder(forKey: .base))
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)

        try container.encode(type(of: base).type, forKey: .type)
        try base.encode(to: container.superEncoder(forKey: .base))
    }
}
