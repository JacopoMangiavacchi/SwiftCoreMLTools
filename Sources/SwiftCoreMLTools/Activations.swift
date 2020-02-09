public struct Linear : Activation {
    public static let type = LayerType.linear
    public let name: String
    public let input: [String]
    public let output: [String]

    public let alpha: Float
    public let beta: Float

    public init(name: String, input: [String], output: [String], alpha: Float = 0.0, beta: Float = 0.0) {
        self.name = name
        self.input = input
        self.output = output

        self.alpha = alpha
        self.beta = beta
    }
}

public struct ReLu : Activation {
    public static let type = LayerType.reLu
    public let name: String
    public let input: [String]
    public let output: [String]

    public init(name: String, input: [String], output: [String]) {
        self.name = name
        self.input = input
        self.output = output
    }
}

public struct LeakyReLu : Activation {
    public static let type = LayerType.leakyReLu
    public let name: String
    public let input: [String]
    public let output: [String]

    public let alpha: Float

    public init(name: String, input: [String], output: [String], alpha: Float = 0.0) {
        self.name = name
        self.input = input
        self.output = output

        self.alpha = alpha
    }
}

public struct ThresholdedReLu : Activation {
    public static let type = LayerType.thresholdedReLu
    public let name: String
    public let input: [String]
    public let output: [String]

    public let alpha: Float

    public init(name: String, input: [String], output: [String], alpha: Float = 0.0) {
        self.name = name
        self.input = input
        self.output = output

        self.alpha = alpha
    }
}

public struct PReLu : Activation {
    public static let type = LayerType.pReLu
    public let name: String
    public let input: [String]
    public let output: [String]

    public init(name: String, input: [String], output: [String]) {
        self.name = name
        self.input = input
        self.output = output
    }
}

public struct Tanh : Activation {
    public static let type = LayerType.tanh
    public let name: String
    public let input: [String]
    public let output: [String]

    public init(name: String, input: [String], output: [String]) {
        self.name = name
        self.input = input
        self.output = output
    }
}

public struct ScaledTanh : Activation {
    public static let type = LayerType.tanh
    public let name: String
    public let input: [String]
    public let output: [String]

    public let alpha: Float
    public let beta: Float

    public init(name: String, input: [String], output: [String], alpha: Float = 0.0, beta: Float = 0.0) {
        self.name = name
        self.input = input
        self.output = output

        self.alpha = alpha
        self.beta = beta
    }
}

public struct Sigmoid : Activation {
    public static let type = LayerType.sigmoid
    public let name: String
    public let input: [String]
    public let output: [String]

    public init(name: String, input: [String], output: [String]) {
        self.name = name
        self.input = input
        self.output = output
    }
}

public struct SigmoidHard : Activation {
    public static let type = LayerType.sigmoidHard
    public let name: String
    public let input: [String]
    public let output: [String]

    public let alpha: Float
    public let beta: Float

    public init(name: String, input: [String], output: [String], alpha: Float = 0.0, beta: Float = 0.0) {
        self.name = name
        self.input = input
        self.output = output

        self.alpha = alpha
        self.beta = beta
    }
}

public struct Elu : Activation {
    public static let type = LayerType.elu
    public let name: String
    public let input: [String]
    public let output: [String]

    public let alpha: Float

    public init(name: String, input: [String], output: [String], alpha: Float = 0.0) {
        self.name = name
        self.input = input
        self.output = output

        self.alpha = alpha
    }
}

public struct Softsign : Activation {
    public static let type = LayerType.softsign
    public let name: String
    public let input: [String]
    public let output: [String]

    public init(name: String, input: [String], output: [String]) {
        self.name = name
        self.input = input
        self.output = output
    }
}

public struct Softplus : Activation {
    public static let type = LayerType.softplus
    public let name: String
    public let input: [String]
    public let output: [String]

    public init(name: String, input: [String], output: [String]) {
        self.name = name
        self.input = input
        self.output = output
    }
}

public struct ParametricSoftplus : Activation {
    public static let type = LayerType.parametricSoftplus
    public let name: String
    public let input: [String]
    public let output: [String]

    public init(name: String, input: [String], output: [String]) {
        self.name = name
        self.input = input
        self.output = output
    }
}
