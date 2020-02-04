public enum LayerType : String, Codable {
    case innerProduct

    var metatype: Layer.Type {
        switch self {
        case .innerProduct:
            return InnerProduct.self
        }
    }
}

public protocol Layer : Codable {
    static var type: LayerType { get }
}

public struct InnerProduct : Layer, Codable {
    public static let type = LayerType.innerProduct

    public let name: String
    public let input: [String]
    public let output: [String]
    public let inputChannels: UInt
    public let outputChannels: UInt
    public let updatable: Bool
    public let weights: [Float]
    public let bias: [Float]
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
