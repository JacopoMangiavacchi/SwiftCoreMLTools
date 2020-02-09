public enum LossType : String, Codable {
    case mse

    var metatype: Loss.Type {
        switch self {
        case .mse:
            return MSE.self
        }
    }
}

public protocol Loss : Codable {
    static var type: LossType { get }
}

public struct MSE : Loss {
    public static let type = LossType.mse

    public let name: String
    public let input: String
    public let target: String

    public init(name: String, input: String, target: String) {
        self.name = name
        self.input = input
        self.target = target
    }
}

struct AnyLoss : Codable {
    var base: Loss

    init(_ base: Loss) {
        self.base = base
    }

    private enum CodingKeys : CodingKey {
        case type, base
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)

        let type = try container.decode(LossType.self, forKey: .type)
        self.base = try type.metatype.init(from: container.superDecoder(forKey: .base))
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)

        try container.encode(type(of: base).type, forKey: .type)
        try base.encode(to: container.superEncoder(forKey: .base))
    }
}
