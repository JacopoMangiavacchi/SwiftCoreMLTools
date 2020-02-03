public protocol NetworkLayer {
}

public struct InnerProduct : NetworkLayer, Codable {
    public let name: String
    public let input: [String]
    public let output: [String]
    public let inputChannels: UInt
    public let outputChannels: UInt
    public let updatable: Bool
    public let weights: [Float]
    public let bias: [Float]
}

@_functionBuilder
public struct LayerBuilder {
    public static func buildBlock(_ children: NetworkLayer...) -> [NetworkLayer] {
        children.compactMap{ $0 }
    }
}

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






public protocol Optimizer {
}

public struct SGD : Optimizer, Codable {
    public let learningRateDefault: Double
    public let learningRateMax: Double
    public let miniBatchSizeDefault: UInt
    public let miniBatchSizeRange: [UInt]
    public let momentumDefault: Double
    public let momentumMax: Double
}







public struct NeuralNetwork : ModelItems {
    public let losses: [Loss]?
    public let optimizer: Optimizer?
    public let epochDefault: UInt?
    public let epochSet: [UInt]?
    public let shuffle: Bool?
    public var layers: [NetworkLayer]

    fileprivate init(losses: [Loss]?,
         optimizer: Optimizer?,
         epochDefault: UInt?,
         epochSet: [UInt]?,
         shuffle: Bool?,
         layers: [NetworkLayer]) {
        self.layers = layers
        self.losses = losses
        self.optimizer = optimizer
        self.epochDefault = epochDefault
        self.epochSet = epochSet
        self.shuffle = shuffle
    }

    public init(losses: [Loss]? = nil,
                optimizer: Optimizer? = nil,
                epochDefault: UInt? = nil,
                epochSet: [UInt]? = nil,
                shuffle: Bool? = nil) {
        self.init(losses: losses,
                  optimizer: optimizer,
                  epochDefault: epochDefault,
                  epochSet: epochSet,
                  shuffle: shuffle,
                  layers: [NetworkLayer]())
    }

    public init(losses: [Loss]? = nil,
                optimizer: Optimizer? = nil,
                epochDefault: UInt? = nil,
                epochSet: [UInt]? = nil,
                shuffle: Bool? = nil,
                @LayerBuilder _ builder: () -> NetworkLayer) {
        self.init(losses: losses,
                  optimizer: optimizer,
                  epochDefault: epochDefault,
                  epochSet: epochSet,
                  shuffle: shuffle,
                  layers: [builder()])
    }

    public init(losses: [Loss]? = nil,
                optimizer: Optimizer? = nil,
                epochDefault: UInt? = nil,
                epochSet: [UInt]? = nil,
                shuffle: Bool? = nil,
                @LayerBuilder _ builder: () -> [NetworkLayer]) {
        self.init(losses: losses,
                  optimizer: optimizer,
                  epochDefault: epochDefault,
                  epochSet: epochSet,
                  shuffle: shuffle,
                  layers: builder())
    }

    public mutating func addLayer(_ layer: NetworkLayer) {
        layers.append(layer)
    }
}


extension NeuralNetwork : Codable {
    private enum CodingKeys : CodingKey {
    // public let losses: [Loss]?
    // public let optimizer: Optimizer?
    // public let epochDefault: UInt?
    // public let epochSet: [UInt]?
    // public let shuffle: Bool?
    // public var layers: [NetworkLayer]
        case loss //, epochDefault, epochSet, shuffle
    }

    public init(from decoder: Decoder) throws {

        let container = try decoder.container(keyedBy: CodingKeys.self)

        self.losses = try container.decode([AnyLoss].self, forKey: .loss).map { $0.base }
        // self.title = try container.decode(String.self, forKey: .title)

        self.optimizer = nil
        self.epochDefault = nil
        self.epochSet = nil
        self.shuffle = nil
        self.layers = [NetworkLayer]()
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)

        try container.encode(losses?.map(AnyLoss.init), forKey: .loss)
        // try container.encode(title, forKey: .title)
    }
}