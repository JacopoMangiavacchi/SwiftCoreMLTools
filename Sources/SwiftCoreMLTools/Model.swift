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
    public let version: Int
    public let shortDescription: String?
    public let author: String?
    public let license: String?
    public let userDefined: [String : String]?
    public let descriptions: [InputOutputDescriptionProtocol]?

    init(version: Int,
         shortDescription: String?,
         author: String?,
         license: String?,
         userDefined: [String : String]?,
         descriptions: [InputOutputDescriptionProtocol]) {
        self.descriptions = descriptions
        self.version = version
        self.shortDescription = shortDescription
        self.author = author
        self.license = license
        self.userDefined = userDefined
    }

    public init(version: Int = 4,
                shortDescription: String? = nil,
                author: String? = nil,
                license: String? = nil,
                userDefined: [String : String]? = [:],
                @InputOutputDescriptionBuilder _ builder: () -> InputOutputDescriptionProtocol) {
        self.init(version: version,
                  shortDescription: shortDescription,
                  author: author,
                  license: license,
                  userDefined: userDefined,
                  descriptions: [builder()])
    }

    public init(version: Int = 4,
                shortDescription: String? = nil,
                author: String? = nil,
                license: String? = nil,
                userDefined: [String : String]? = [:],
                @InputOutputDescriptionBuilder _ builder: () -> [InputOutputDescriptionProtocol]) {
        self.init(version: version,
                  shortDescription: shortDescription,
                  author: author,
                  license: license,
                  userDefined: userDefined,
                  descriptions: builder())
    }
}
