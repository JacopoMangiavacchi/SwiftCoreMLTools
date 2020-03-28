public enum FeatureType : String, Codable {
    case Float
}

public protocol Items : Codable {
}

public protocol InputOutputItems : Items {
    var name: String { get }
    var shape: [UInt] { get }
    var featureType: FeatureType { get }
}

public struct Input : InputOutputItems {
    public let name: String
    public let shape: [UInt]
    public let featureType: FeatureType

    public init(name: String, shape: [UInt], featureType: FeatureType = .Float) {
        self.name = name
        self.shape = shape
        self.featureType = featureType
    }
}

public struct Output : InputOutputItems {
    public let name: String
    public let shape: [UInt]
    public let featureType: FeatureType

    public init(name: String, shape: [UInt], featureType: FeatureType = .Float) {
        self.name = name
        self.shape = shape
        self.featureType = featureType
    }
}

public struct TrainingInput : InputOutputItems {
    public let name: String
    public let shape: [UInt]
    public let featureType: FeatureType

    public init(name: String, shape: [UInt], featureType: FeatureType = .Float) {
        self.name = name
        self.shape = shape
        self.featureType = featureType
    }
}

@_functionBuilder
public struct ItemBuilder {
    public static func buildBlock(_ children: Items...) -> [Items] {
        children.compactMap{ $0 }
    }
}

public struct Model {
    public let version: UInt
    public let shortDescription: String?
    public let author: String?
    public let license: String?
    public let userDefined: [String : String]?

    public var inputs: [String : Input]
    public var outputs: [String : Output]
    public var trainingInputs: [String : TrainingInput]
    public var neuralNetwork: NeuralNetwork

    fileprivate init(version: UInt,
         shortDescription: String?,
         author: String?,
         license: String?,
         userDefined: [String : String]?,
         items: [Items]) {
        self.version = version
        self.shortDescription = shortDescription
        self.author = author
        self.license = license
        self.userDefined = userDefined
        self.inputs = [String : Input]()
        self.outputs = [String : Output]()
        self.trainingInputs = [String : TrainingInput]()
        self.neuralNetwork = NeuralNetwork()

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
                  items: [Items]())
    }

    public init(version: UInt = 4,
                shortDescription: String? = nil,
                author: String? = nil,
                license: String? = nil,
                userDefined: [String : String]? = [:],
                @ItemBuilder _ builder: () -> Items) {
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
                @ItemBuilder _ builder: () -> [Items]) {
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

extension Model : Codable {
    private enum CodingKeys : CodingKey {
        case version, shortDescription, author, license, userDefined, inputs, outputs, trainingInputs, neuralNetwork
    }

    public init(from decoder: Decoder) throws {

        let container = try decoder.container(keyedBy: CodingKeys.self)

        self.version = try container.decode(UInt.self, forKey: .version)
        self.shortDescription = try container.decode(String?.self, forKey: .shortDescription)
        self.author = try container.decode(String?.self, forKey: .author)
        self.license = try container.decode(String?.self, forKey: .license)
        self.userDefined = try container.decode([String : String]?.self, forKey: .userDefined)
        self.inputs = try container.decode([String : Input].self, forKey: .inputs)
        self.outputs = try container.decode([String : Output].self, forKey: .outputs)
        self.trainingInputs = try container.decode([String : TrainingInput].self, forKey: .trainingInputs)
        self.neuralNetwork = try container.decode(NeuralNetwork.self, forKey: .neuralNetwork)
    }
}
