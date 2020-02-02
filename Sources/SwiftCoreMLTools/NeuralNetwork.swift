public protocol NetworkLayer {
}

public struct InnerProductLayer : NetworkLayer {
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

public protocol Loss {
}

public struct MSE : Loss {
    public let name: String
    public let input: String
    public let target: String
}

public protocol Optimizer {
}

public struct SGD : Optimizer {
    public let learningRateDefault: Double
    public let learningRateMax: Double
    public let miniBatchSizeDefault: UInt
    public let miniBatchSizeRange: [UInt]
    public let momentumDefault: Double
    public let momentumMax: Double
}

public struct NeuralNetwork : ModelItems {
    public let loss: [Loss]?
    public let optimizer: Optimizer?
    public let epochDefault: UInt?
    public let epochSet: [UInt]?
    public let shuffle: Bool?
    let layers: [NetworkLayer]?

    init(loss: [Loss]?,
         optimizer: Optimizer?,
         epochDefault: UInt?,
         epochSet: [UInt]?,
         shuffle: Bool?,
         layers: [NetworkLayer]) {
        self.layers = layers
        self.loss = loss
        self.optimizer = optimizer
        self.epochDefault = epochDefault
        self.epochSet = epochSet
        self.shuffle = shuffle
    }

    public init(loss: [Loss]? = nil,
                optimizer: Optimizer? = nil,
                epochDefault: UInt? = nil,
                epochSet: [UInt]? = nil,
                shuffle: Bool? = nil,
                @LayerBuilder _ builder: () -> NetworkLayer) {
        self.init(loss: loss,
                  optimizer: optimizer,
                  epochDefault: epochDefault,
                  epochSet: epochSet,
                  shuffle: shuffle,
                  layers: [builder()])
    }

    public init(loss: [Loss]? = nil,
                optimizer: Optimizer? = nil,
                epochDefault: UInt? = nil,
                epochSet: [UInt]? = nil,
                shuffle: Bool? = nil,
                @LayerBuilder _ builder: () -> [NetworkLayer]) {
        self.init(loss: loss,
                  optimizer: optimizer,
                  epochDefault: epochDefault,
                  epochSet: epochSet,
                  shuffle: shuffle,
                  layers: builder())
    }
}
