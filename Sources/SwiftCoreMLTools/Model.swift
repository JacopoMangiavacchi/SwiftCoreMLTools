public enum FeatureType {
    case Double
}

public protocol ModelItems {
}

public struct Input : ModelItems {
    public let name: String
    public let shape: [UInt]
    public let featureType: FeatureType
}

public struct Output : ModelItems {
    public let name: String
    public let shape: [UInt]
    public let featureType: FeatureType
}

public struct TrainingInput : ModelItems {
    public let name: String
    public let shape: [UInt]
    public let featureType: FeatureType
}

@_functionBuilder
public struct ItemBuilder {
    public static func buildBlock(_ children: ModelItems...) -> [ModelItems] {
        children.compactMap{ $0 }
    }
}

public struct Model {
    public let version: UInt
    public let shortDescription: String?
    public let author: String?
    public let license: String?
    public let userDefined: [String : String]?

    public var inputs = [String : Input]()
    public var outputs = [String : Output]()
    public var trainingInputs = [String : TrainingInput]()
    public var neuralNetwork = NeuralNetwork()

    var items: [ModelItems] {
        didSet {
            inputs.removeAll()
            outputs.removeAll()
            trainingInputs.removeAll()
            neuralNetwork = NeuralNetwork()
            for item in items {
                switch item {
                case let input as Input:
                    self.inputs[input.name] = input

                case let output as Output:
                    self.outputs[output.name] = output

                case let trainingInput as TrainingInput:
                    self.trainingInputs[trainingInput.name] = trainingInput

                case let neuralNetwork as NeuralNetwork:
                    self.neuralNetwork = neuralNetwork

                default:
                    break
                }
            }
        }
    }

    fileprivate init(version: UInt,
         shortDescription: String?,
         author: String?,
         license: String?,
         userDefined: [String : String]?,
         items: [ModelItems]) {
        self.items = items
        self.version = version
        self.shortDescription = shortDescription
        self.author = author
        self.license = license
        self.userDefined = userDefined
    }

    public init(version: UInt = 4,
                shortDescription: String? = nil,
                author: String? = nil,
                license: String? = nil,
                userDefined: [String : String]? = [:]) {
        self.init(version: version,
                  shortDescription: shortDescription,
                  author: author,
                  license: license,
                  userDefined: userDefined,
                  items: [ModelItems]())
    }

    public init(version: UInt = 4,
                shortDescription: String? = nil,
                author: String? = nil,
                license: String? = nil,
                userDefined: [String : String]? = [:],
                @ItemBuilder _ builder: () -> ModelItems) {
        self.init(version: version,
                  shortDescription: shortDescription,
                  author: author,
                  license: license,
                  userDefined: userDefined,
                  items: [builder()])
    }

    public init(version: UInt = 4,
                shortDescription: String? = nil,
                author: String? = nil,
                license: String? = nil,
                userDefined: [String : String]? = [:],
                @ItemBuilder _ builder: () -> [ModelItems]) {
        self.init(version: version,
                  shortDescription: shortDescription,
                  author: author,
                  license: license,
                  userDefined: userDefined,
                  items: builder())
    }

    public mutating func addInput(_ input: Input) {
        inputs[input.name] = input
    }

    public mutating func addOutput(_ output: Output) {
        outputs[output.name] = output
    }

    public mutating func addTrainingInput(_ trainingInput: TrainingInput) {
        trainingInputs[trainingInput.name] = trainingInput
    }
}
