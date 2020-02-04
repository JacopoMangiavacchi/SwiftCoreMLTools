public enum OptimizerType : String, Codable {
    case sgd

    var metatype: Optimizer.Type {
        switch self {
        case .sgd:
            return SGD.self
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
