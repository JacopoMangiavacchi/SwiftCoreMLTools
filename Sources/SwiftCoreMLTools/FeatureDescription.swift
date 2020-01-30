public enum FeatureType {
    case Double
}

public struct InputOutputDescription {
    public let name: String
    public let shape: [Int]
    public let type: FeatureType
}

public enum FeatureDescriptionType {
    case Input, Outupt, TrainingInput
}

public struct FeatureDescription {
    public let type: FeatureDescriptionType
    public let inputOutputDescription: InputOutputDescription
}

@_functionBuilder
public struct InputOutputDescriptionBuilder {
    public static func buildBlock(_ children: InputOutputDescription...) -> [InputOutputDescription] {
        children.compactMap{ $0 }
    }
}

// public func Inputs(@InputOutputDescriptionBuilder _ makeInput: () -> [InputOutputDescription]) -> [FeatureDescription] {
//     makeInput().map{ FeatureDescription(type: .Input, inputOutputDescription: $0) }
// }

// public func Outputs(@InputOutputDescriptionBuilder _ makeInput: () -> [InputOutputDescription]) -> [FeatureDescription] {
//     makeInput().map{ FeatureDescription(type: .Outupt, inputOutputDescription: $0) }
// }

// public func TrainingInputs(@InputOutputDescriptionBuilder _ makeInput: () -> [InputOutputDescription]) -> [FeatureDescription] {
//     makeInput().map{ FeatureDescription(type: .TrainingInput, inputOutputDescription: $0) }
// }

public protocol DescriptionProtocol {
    var descriptions: [FeatureDescription] { get }
}

public struct Inputs : DescriptionProtocol {
    public let descriptions: [FeatureDescription]

    public init(@InputOutputDescriptionBuilder _ builder: () -> [InputOutputDescription]) {
        self.descriptions = builder().map{ FeatureDescription(type: .Input, inputOutputDescription: $0) }
    }

    init(@InputOutputDescriptionBuilder _ builder: () -> InputOutputDescription) {
        self.descriptions = [builder()].map{ FeatureDescription(type: .Input, inputOutputDescription: $0) }
    }
}

public struct Outputs : DescriptionProtocol {
    public let descriptions: [FeatureDescription]

    public init(@InputOutputDescriptionBuilder _ builder: () -> [InputOutputDescription]) {
        self.descriptions = builder().map{ FeatureDescription(type: .Outupt, inputOutputDescription: $0) }
    }

    public init(@InputOutputDescriptionBuilder _ builder: () -> InputOutputDescription) {
        self.descriptions = [builder()].map{ FeatureDescription(type: .Outupt, inputOutputDescription: $0) }
    }
}

public struct TrainingInputs : DescriptionProtocol {
    public let descriptions: [FeatureDescription]

    public init(@InputOutputDescriptionBuilder _ builder: () -> [InputOutputDescription]) {
        self.descriptions = builder().map{ FeatureDescription(type: .TrainingInput, inputOutputDescription: $0) }
    }

    public init(@InputOutputDescriptionBuilder _ builder: () -> InputOutputDescription) {
        self.descriptions = [builder()].map{ FeatureDescription(type: .TrainingInput, inputOutputDescription: $0) }
    }
}
