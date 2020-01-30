public protocol NetworkLayers {
}

public struct InnerProductLayer : NetworkLayers {
    public let name: String
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
