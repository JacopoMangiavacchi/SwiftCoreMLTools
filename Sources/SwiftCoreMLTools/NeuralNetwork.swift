@_functionBuilder
public struct LayerBuilder {
    public static func buildBlock(_ children: Layer...) -> [Layer] {
        children.compactMap{ $0 }
    }
}

public struct NeuralNetwork : Items {
    public let losses: [Loss]?
    public let optimizer: Optimizer?
    public let epochDefault: UInt?
    public let epochSet: [UInt]?
    public let shuffle: Bool?
    public var layers: [Layer]

    fileprivate init(losses: [Loss]?,
         optimizer: Optimizer?,
         epochDefault: UInt?,
         epochSet: [UInt]?,
         shuffle: Bool?,
         layers: [Layer]) {
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
                  layers: [Layer]())
    }

    public init(losses: [Loss]? = nil,
                optimizer: Optimizer? = nil,
                epochDefault: UInt? = nil,
                epochSet: [UInt]? = nil,
                shuffle: Bool? = nil,
                @LayerBuilder _ builder: () -> Layer) {
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
                @LayerBuilder _ builder: () -> [Layer]) {
        self.init(losses: losses,
                  optimizer: optimizer,
                  epochDefault: epochDefault,
                  epochSet: epochSet,
                  shuffle: shuffle,
                  layers: builder())
    }

    public mutating func addLayer(_ layer: Layer) {
        layers.append(layer)
    }
}

extension NeuralNetwork : Codable {
    private enum CodingKeys : CodingKey {
        case losses, optimizer, epochDefault, epochSet, shuffle, layers
    }

    public init(from decoder: Decoder) throws {

        let container = try decoder.container(keyedBy: CodingKeys.self)

        self.losses = try container.decode([AnyLoss].self, forKey: .losses).map { $0.base }
        self.optimizer = try container.decode(AnyOptimizer.self, forKey: .optimizer).base
        self.epochDefault = try container.decode(UInt?.self, forKey: .epochDefault)
        self.epochSet = try container.decode([UInt]?.self, forKey: .epochSet)
        self.shuffle = try container.decode(Bool?.self, forKey: .shuffle)
        self.layers = try container.decode([AnyLayer].self, forKey: .layers).map { $0.base }
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)

        if let losses = losses {
            try container.encode(losses.map(AnyLoss.init), forKey: .losses)
        }
        if let optimizer = optimizer {
            try container.encode(AnyOptimizer(optimizer), forKey: .optimizer)
        }
        if let epochDefault = epochDefault {
            try container.encode(epochDefault, forKey: .epochDefault)
        }
        if let epochSet = epochSet {
            try container.encode(epochSet, forKey: .epochSet)
        }
        if let shuffle = shuffle {
            try container.encode(shuffle, forKey: .shuffle)
        }
        try container.encode(layers.map(AnyLayer.init), forKey: .layers)
    }
}
