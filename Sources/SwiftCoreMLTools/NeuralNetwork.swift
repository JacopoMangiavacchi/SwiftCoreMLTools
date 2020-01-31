public protocol NetworkLayers {
}

public struct InnerProductLayer : NetworkLayers {
    public let name: String
    public let input: [String]
    public let output: [String]
    public let updatable: Bool
    public let weights: [Float]
    public let bias: [Float]
}

@_functionBuilder
public struct LayerBuilder {
    public static func buildBlock(_ children: NetworkLayers...) -> [NetworkLayers] {
        children.compactMap{ $0 }
    }
}

public struct MSE {
    public let name: String
    public let input: String
    public let target: String
}

public struct SGD {
    public let learningRateDefault: Double
    public let learningRateMax: Double
    public let miniBatchSizeDefault: UInt
    public let miniBatchSizeRange: [UInt]
    public let momentumDefault: Double
    public let momentumMax: Double
}

public struct NeuralNetwork : ModelItems {
    public let loss: [MSE]?
    public let optimizer: SGD?
    public let epochDefault: UInt?
    public let epochSet: [UInt]?
    public let shuffle: Bool?
    let layers: [NetworkLayers]?

    init(loss: [MSE]?,
         optimizer: SGD?,
         epochDefault: UInt?,
         epochSet: [UInt]?,
         shuffle: Bool?,
         layers: [NetworkLayers]) {
        self.layers = layers
        self.loss = loss
        self.optimizer = optimizer
        self.epochDefault = epochDefault
        self.epochSet = epochSet
        self.shuffle = shuffle
    }

    public init(loss: [MSE]? = nil,
                optimizer: SGD? = nil,
                epochDefault: UInt? = nil,
                epochSet: [UInt]? = nil,
                shuffle: Bool? = nil,
                @LayerBuilder _ builder: () -> NetworkLayers) {
        self.init(loss: loss,
                  optimizer: optimizer,
                  epochDefault: epochDefault,
                  epochSet: epochSet,
                  shuffle: shuffle,
                  layers: [builder()])
    }

    public init(loss: [MSE]? = nil,
                optimizer: SGD? = nil,
                epochDefault: UInt? = nil,
                epochSet: [UInt]? = nil,
                shuffle: Bool? = nil,
                @LayerBuilder _ builder: () -> [NetworkLayers]) {
        self.init(loss: loss,
                  optimizer: optimizer,
                  epochDefault: epochDefault,
                  epochSet: epochSet,
                  shuffle: shuffle,
                  layers: builder())
    }
}
