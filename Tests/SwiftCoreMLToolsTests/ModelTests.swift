import XCTest
@testable import SwiftCoreMLTools

final class ModelTests: XCTestCase {
    func testSingleInputs() {
        let model = Model {
            Input(name: "dense_input1", shape: [1], featureType: .Double)
        }

        XCTAssertEqual(model.items.count, 1)
    }

    func testMultipleInputs() {
        let model = Model {
            Input(name: "dense_input1", shape: [1], featureType: .Double)
            Input(name: "dense_input2", shape: [1], featureType: .Double)
            Output(name: "output", shape: [1], featureType: .Double)
            TrainingInput(name: "dense_input", shape: [1], featureType: .Double)
            TrainingInput(name: "output_true", shape: [1], featureType: .Double)
        }

        XCTAssertEqual(model.items.count, 5)
    }

    func testWithMetadata() {
        let model = Model(version: 4,
                          shortDescription: "Trivial linear classifier",
                          author: "Jacopo Mangiavacchi",
                          license: "MIT",
                          userDefined: ["SwiftCoremltoolsVersion" : "0.1"]) {
            Input(name: "dense_input", shape: [1], featureType: .Double)
            Output(name: "output", shape: [1], featureType: .Double)
            TrainingInput(name: "dense_input", shape: [1], featureType: .Double)
            TrainingInput(name: "output_true", shape: [1], featureType: .Double)
        }

        XCTAssertEqual(model.items.count, 4)
    }

    func testWithNeuralNetwork() {
        let model = Model(version: 4,
                          shortDescription: "Trivial linear classifier",
                          author: "Jacopo Mangiavacchi",
                          license: "MIT",
                          userDefined: ["SwiftCoremltoolsVersion" : "0.1"]) {
            Input(name: "dense_input", shape: [1], featureType: .Double)
            Output(name: "output", shape: [1], featureType: .Double)
            TrainingInput(name: "dense_input", shape: [1], featureType: .Double)
            TrainingInput(name: "output_true", shape: [1], featureType: .Double)
            NeuralNetwork {
                InnerProductLayer(name: "layer1",
                                  input: ["dense_input"],
                                  output: ["output"],
                                  inputChannels: 1,
                                  outputChannels: 1,
                                  updatable: true,
                                  weights: [0.0],
                                  bias: [0.0])
            }
        }

        XCTAssertEqual(model.items.count, 5)
    }

    func testWithPersonazibleNeuralNetwork() {
        let model = Model(version: 4,
                          shortDescription: "Trivial linear classifier",
                          author: "Jacopo Mangiavacchi",
                          license: "MIT",
                          userDefined: ["SwiftCoremltoolsVersion" : "0.1"]) {
            Input(name: "dense_input", shape: [1], featureType: .Double)
            Output(name: "output", shape: [1], featureType: .Double)
            TrainingInput(name: "dense_input", shape: [1], featureType: .Double)
            TrainingInput(name: "output_true", shape: [1], featureType: .Double)
            NeuralNetwork(loss: [MSE(name: "lossLayer",
                                     input: "output",
                                     target: "output_true")],
                          optimizer: SGD(learningRateDefault: 0.01,
                                         learningRateMax: 0.3,
                                         miniBatchSizeDefault: 5,
                                         miniBatchSizeRange: [5],
                                         momentumDefault: 0,
                                         momentumMax: 1.0),
                          epochDefault: 2,
                          epochSet: [2],
                          shuffle: true) {
                InnerProductLayer(name: "layer1",
                                  input: ["dense_input"],
                                  output: ["output"],
                                  inputChannels: 1,
                                  outputChannels: 1,
                                  updatable: true,
                                  weights: [0.0],
                                  bias: [0.0])
            }
        }

        XCTAssertEqual(model.items.count, 5)
    }

    func testRealModelExport() {
        let model = Model(version: 4,
                          shortDescription: "Trivial linear classifier",
                          author: "Jacopo Mangiavacchi",
                          license: "MIT",
                          userDefined: ["SwiftCoremltoolsVersion" : "0.1"]) {
            Input(name: "dense_input", shape: [1], featureType: .Double)
            Output(name: "output", shape: [1], featureType: .Double)
            TrainingInput(name: "dense_input", shape: [1], featureType: .Double)
            TrainingInput(name: "output_true", shape: [1], featureType: .Double)
            NeuralNetwork(loss: [MSE(name: "lossLayer",
                                     input: "output",
                                     target: "output_true")],
                          optimizer: SGD(learningRateDefault: 0.01,
                                         learningRateMax: 0.3,
                                         miniBatchSizeDefault: 5,
                                         miniBatchSizeRange: [5],
                                         momentumDefault: 0,
                                         momentumMax: 1.0),
                          epochDefault: 2,
                          epochSet: [2],
                          shuffle: true) {
                InnerProductLayer(name: "layer1",
                                  input: ["dense_input"],
                                  output: ["output"],
                                  inputChannels: 1,
                                  outputChannels: 1,
                                  updatable: true,
                                  weights: [0.0],
                                  bias: [0.0])
            }
        }

        let coreMLData = model.coreMLData

        XCTAssert(coreMLData != nil, "Failed exporting CoreML protobuf")
        XCTAssert(coreMLData!.count > 0, "Exporting CoreML protobuf empty")
    }

    static var allTests = [
        ("testSingleInputs", testSingleInputs),
        ("testMultipleInputs", testMultipleInputs),
        ("testWithMetadata", testWithMetadata),
        ("testWithNeuralNetwork", testWithNeuralNetwork),
        ("testWithPersonazibleNeuralNetwork", testWithPersonazibleNeuralNetwork),
        ("testRealModelExport", testRealModelExport),
    ]
}
