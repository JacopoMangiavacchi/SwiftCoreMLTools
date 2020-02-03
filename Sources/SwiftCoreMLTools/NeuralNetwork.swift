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

public protocol Loss : Codable {
    static var type: LossType { get }
}

public struct MSE : Loss {
    public static let type = LossType.mse

    public let name: String
    public let input: String
    public let target: String
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
