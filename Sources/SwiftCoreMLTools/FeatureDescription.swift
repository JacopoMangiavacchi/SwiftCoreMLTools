public enum FeatureType {
    case Double
}

public struct InputOutputDescription {
    let name: String
    let shape: [Int]
    let type: FeatureType = .Double
}

public enum FeatureDescriptionType {
    case Input, Outupt, TrainingInput
}

public struct FeatureDescription {
    let type: FeatureDescriptionType
    let inputOutputDescription: InputOutputDescription
}

@_functionBuilder
public struct InputOutputDescriptionBuilder {
    static func buildBlock(_ children: InputOutputDescription...) -> [InputOutputDescription] {
        children.compactMap{ $0 }
    }
}

func Inputs(@InputOutputDescriptionBuilder _ makeInput: () -> [InputOutputDescription]) -> [FeatureDescription] {
    makeInput().map{ FeatureDescription(type: .Input, inputOutputDescription: $0) }
}

func Outputs(@InputOutputDescriptionBuilder _ makeInput: () -> [InputOutputDescription]) -> [FeatureDescription] {
    makeInput().map{ FeatureDescription(type: .Outupt, inputOutputDescription: $0) }
}

func TrainingInputs(@InputOutputDescriptionBuilder _ makeInput: () -> [InputOutputDescription]) -> [FeatureDescription] {
    makeInput().map{ FeatureDescription(type: .TrainingInput, inputOutputDescription: $0) }
}
