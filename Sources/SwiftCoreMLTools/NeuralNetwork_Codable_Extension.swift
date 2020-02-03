public enum LossType : String, Codable {
    case mse

    var metatype: Loss.Type {
        switch self {
        case .mse:
            return MSE.self
        }
    }
}

public enum OptimizerType : String, Codable {
    case sgd

    var metatype: Optimizer.Type {
        switch self {
        case .sgd:
            return SGD.self
        }
    }
}

struct AnyLoss : Codable {
    var base: Loss

    init(_ base: Loss) {
        self.base = base
    }

    private enum CodingKeys : CodingKey {
        case type, base
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)

        let type = try container.decode(LossType.self, forKey: .type)
        self.base = try type.metatype.init(from: container.superDecoder(forKey: .base))
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)

        try container.encode(type(of: base).type, forKey: .type)
        try base.encode(to: container.superEncoder(forKey: .base))
    }
}

struct AnyOptimizer : Codable {
    var base: Optimizer

    init(_ base: Optimizer) {
        self.base = base
    }

    private enum CodingKeys : CodingKey {
        case type, base
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)

        let type = try container.decode(OptimizerType.self, forKey: .type)
        self.base = try type.metatype.init(from: container.superDecoder(forKey: .base))
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)

        try container.encode(type(of: base).type, forKey: .type)
        try base.encode(to: container.superEncoder(forKey: .base))
    }
}

extension NeuralNetwork : Codable {
    private enum CodingKeys : CodingKey {
        case losses, optimizer //, epochDefault, epochSet, shuffle, layers
    }

    public init(from decoder: Decoder) throws {

        let container = try decoder.container(keyedBy: CodingKeys.self)

        self.losses = try container.decode([AnyLoss].self, forKey: .losses).map { $0.base }
        self.optimizer = try container.decode(AnyOptimizer.self, forKey: .optimizer).base

        // self.title = try container.decode(String.self, forKey: .title)
        self.epochDefault = nil
        self.epochSet = nil
        self.shuffle = nil
        self.layers = [NetworkLayer]()
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)

        if let losses = losses {
            try container.encode(losses.map(AnyLoss.init), forKey: .losses)
        }
        if let optimizer = optimizer {
            try container.encode(AnyOptimizer(optimizer), forKey: .optimizer)
        }

        // try container.encode(title, forKey: .title)
    }
}
