public protocol NetworkLayers {
}

public struct InnerProductLayer : NetworkLayers {
    public let name: String
    public let input: [String]
    public let output: [String]
    public let updatable: Bool
    public let weights: [Double]
    public let bias: [Double]
}

@_functionBuilder
public struct LayerBuilder {
    public static func buildBlock(_ children: NetworkLayers...) -> [NetworkLayers] {
        children.compactMap{ $0 }
    }
}

public struct NeuralNetwork : ModelItems {
    let layers: [NetworkLayers]?

    public init(@LayerBuilder _ builder: () -> NetworkLayers) {
        self.layers = [builder()]
    }

    public init(@LayerBuilder _ builder: () -> [NetworkLayers]) {
        self.layers = builder()
    }
}
