public enum FeatureType {
    case Double
}

public enum FeatureDescriptionType {
    case Input, Outupt, TrainingInput
}

public protocol InputOutputDescriptionProtocol {
    var name: String { get }
    var shape: [Int] { get }
    var featureType: FeatureType { get }
    var descriptionType: FeatureDescriptionType { get }
}

public struct Input : InputOutputDescriptionProtocol {
    public let name: String
    public let shape: [Int]
    public let featureType: FeatureType
    public let descriptionType = FeatureDescriptionType.Input
}

public struct Output : InputOutputDescriptionProtocol {
    public let name: String
    public let shape: [Int]
    public let featureType: FeatureType
    public let descriptionType = FeatureDescriptionType.Outupt
}

public struct TrainingInput : InputOutputDescriptionProtocol {
    public let name: String
    public let shape: [Int]
    public let featureType: FeatureType
    public let descriptionType = FeatureDescriptionType.TrainingInput
}

@_functionBuilder
public struct InputOutputDescriptionBuilder {
    public static func buildBlock(_ children: InputOutputDescriptionProtocol...) -> [InputOutputDescriptionProtocol] {
        children.compactMap{ $0 }
    }
}

public struct Model {
    public let descriptions: [InputOutputDescriptionProtocol]

    public init(@InputOutputDescriptionBuilder _ builder: () -> InputOutputDescriptionProtocol) {
        self.descriptions = [builder()]
    }

    public init(@InputOutputDescriptionBuilder _ builder: () -> [InputOutputDescriptionProtocol]) {
        self.descriptions = builder()
    }
}
