import XCTest
@testable import SwiftCoreMLTools

final class ModelTests: XCTestCase {
    func testSingleInputs() {
        let model = Model {
            Input(name: "dense_input1", shape: [1])
        }

        XCTAssertEqual(model.items.count, 1)
    }

    func testMultipleInputs() {
        let model = Model {
            Input(name: "dense_input1", shape: [1])
            Input(name: "dense_input2", shape: [1])
            Output(name: "output", shape: [1])
            TrainingInput(name: "dense_input", shape: [1])
            TrainingInput(name: "output_true", shape: [1])
        }

        XCTAssertEqual(model.items.count, 5)
    }

    func testWithMetadata() {
        let model = Model(version: 4,
                          shortDescription: "Trivial linear classifier",
                          author: "Jacopo Mangiavacchi",
                          license: "MIT",
                          userDefined: ["SwiftCoremltoolsVersion" : "0.1"]) {
            Input(name: "dense_input", shape: [1])
            Output(name: "output", shape: [1])
            TrainingInput(name: "dense_input", shape: [1])
            TrainingInput(name: "output_true", shape: [1])
        }

        XCTAssertEqual(model.items.count, 4)
    }

    func testWithNeuralNetwork() {
        let model = Model(version: 4,
                          shortDescription: "Trivial linear classifier",
                          author: "Jacopo Mangiavacchi",
                          license: "MIT",
                          userDefined: ["SwiftCoremltoolsVersion" : "0.1"]) {
            Input(name: "dense_input", shape: [1])
            Output(name: "output", shape: [1])
            TrainingInput(name: "dense_input", shape: [1])
            TrainingInput(name: "output_true", shape: [1])
            NeuralNetwork {
                InnerProduct(name: "layer1",
                             input: ["dense_input"],
                             output: ["output"],
                             weight: [0.0],
                             bias: [0.0],
                             inputChannels: 1,
                             outputChannels: 1,
                             updatable: true)
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
            Input(name: "dense_input", shape: [1])
            Output(name: "output", shape: [1])
            TrainingInput(name: "dense_input", shape: [1])
            TrainingInput(name: "output_true", shape: [1])
            NeuralNetwork(losses: [MSE(name: "lossLayer",
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
                InnerProduct(name: "layer1",
                             input: ["dense_input"],
                             output: ["output"],
                             weight: [0.0],
                             bias: [0.0],
                             inputChannels: 1,
                             outputChannels: 1,
                             updatable: true)
            }
        }

        XCTAssertEqual(model.items.count, 5)
    }

    func testModelExtraction() {
        let model = Model(version: 4,
                          shortDescription: "Trivial linear classifier",
                          author: "Jacopo Mangiavacchi",
                          license: "MIT",
                          userDefined: ["SwiftCoremltoolsVersion" : "0.1"]) {
            Input(name: "dense_input", shape: [1])
            Output(name: "output", shape: [1])
            TrainingInput(name: "dense_input", shape: [1])
            TrainingInput(name: "output_true", shape: [1])
            NeuralNetwork(losses: [MSE(name: "lossLayer",
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
                InnerProduct(name: "layer1",
                             input: ["dense_input"],
                             output: ["output"],
                             weight: [0.0],
                             bias: [0.0],
                             inputChannels: 1,
                             outputChannels: 1,
                             updatable: true)
            }
        }

        XCTAssert(model.inputs.count == 1, "Failed extracting Input")
        XCTAssert(model.outputs.count == 1, "Failed extracting Output")
        XCTAssert(model.trainingInputs.count == 2, "Failed extracting TrainingInput")
        XCTAssert(model.neuralNetwork.layers.count == 1, "Failed extracting NeuralNetwork")
    }

    func testModelAPI() {
        let model = Model(version: 4,
                          shortDescription: "Trivial linear classifier",
                          author: "Jacopo Mangiavacchi",
                          license: "MIT",
                          userDefined: ["SwiftCoremltoolsVersion" : "0.1"]) {
            Input(name: "dense_input", shape: [1])
            Output(name: "output", shape: [1])
            TrainingInput(name: "dense_input", shape: [1])
            TrainingInput(name: "output_true", shape: [1])
            NeuralNetwork(losses: [MSE(name: "lossLayer",
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
                InnerProduct(name: "layer1",
                             input: ["dense_input"],
                             output: ["output"],
                             weight: [0.0],
                             bias: [0.0],
                             inputChannels: 1,
                             outputChannels: 1,
                             updatable: true)
            }
        }

        XCTAssert(model.inputs.count == 1, "Failed extracting Input")
        XCTAssert(model.outputs.count == 1, "Failed extracting Output")
        XCTAssert(model.trainingInputs.count == 2, "Failed extracting TrainingInput")
        XCTAssert(model.neuralNetwork.layers.count == 1, "Failed extracting NeuralNetwork")
    }

    func testRealModelExport() {
        var model = Model(version: 4,
                          shortDescription: "Trivial linear classifier",
                          author: "Jacopo Mangiavacchi",
                          license: "MIT",
                          userDefined: ["SwiftCoremltoolsVersion" : "0.1"])
                          
        model.addInput(Input(name: "dense_input", shape: [1]))
        model.addOutput(Output(name: "output", shape: [1]))
        model.addTrainingInput(TrainingInput(name: "dense_input", shape: [1]))
        model.addTrainingInput(TrainingInput(name: "output_true", shape: [1]))
        model.neuralNetwork = NeuralNetwork(losses: [MSE(name: "lossLayer",
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
                                            shuffle: true)
                                            
        model.neuralNetwork.addLayer(InnerProduct(name: "layer1",
                                                 input: ["dense_input"],
                                                 output: ["output"],
                                                 weight: [0.0],
                                                 bias: [0.0],
                                                 inputChannels: 1,
                                                 outputChannels: 1,
                                                 updatable: true))

        let coreMLData = model.coreMLData

        XCTAssert(coreMLData != nil, "Failed Convert to CoreML Data")
        XCTAssert(coreMLData!.count > 0, "Failed Convert to CoreML Data (empty)")

        // TODO: Add more test validating the generated coreMLData
    }

    func testCodable() {
        let model = Model(version: 4,
                          shortDescription: "Trivial linear classifier",
                          author: "Jacopo Mangiavacchi",
                          license: "MIT",
                          userDefined: ["SwiftCoremltoolsVersion" : "0.1"]) {
            Input(name: "dense_input", shape: [1])
            Output(name: "output", shape: [1])
            TrainingInput(name: "dense_input", shape: [1])
            TrainingInput(name: "output_true", shape: [1])
            NeuralNetwork(losses: [MSE(name: "lossLayer",
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
                InnerProduct(name: "layer1",
                             input: ["dense_input"],
                             output: ["output"],
                             weight: [0.0],
                             bias: [0.0],
                             inputChannels: 1,
                             outputChannels: 1,
                             updatable: true)
            }
        }

        XCTAssert(model.inputs.count == 1, "Failed extracting Input")
        XCTAssert(model.outputs.count == 1, "Failed extracting Output")
        XCTAssert(model.trainingInputs.count == 2, "Failed extracting TrainingInput")
        XCTAssert(model.neuralNetwork.layers.count == 1, "Failed extracting NeuralNetwork")

        let encoder = JSONEncoder()
        // encoder.outputFormatting = .prettyPrinted

        let jsonData = try! encoder.encode(model)
        // let jsonString = String(data: jsonData, encoding: .utf8)
        // print(jsonString!)

        let decodedModel = try! JSONDecoder().decode(Model.self, from: jsonData)

        XCTAssert(decodedModel.inputs.count == 1, "Failed extracting Input after encoding/decoding")
        XCTAssert(decodedModel.outputs.count == 1, "Failed extracting Output after encoding/decoding")
        XCTAssert(decodedModel.trainingInputs.count == 2, "Failed extracting TrainingInput after encoding/decoding")
        XCTAssert(decodedModel.neuralNetwork.layers.count == 1, "Failed extracting NeuralNetwork after encoding/decoding")
    }

    static var allTests = [
        ("testSingleInputs", testSingleInputs),
        ("testMultipleInputs", testMultipleInputs),
        ("testWithMetadata", testWithMetadata),
        ("testWithNeuralNetwork", testWithNeuralNetwork),
        ("testWithPersonazibleNeuralNetwork", testWithPersonazibleNeuralNetwork),
        ("testModelExtraction", testModelExtraction),
        ("testModelAPI", testModelAPI),
        ("testRealModelExport", testRealModelExport),
        ("testCodable", testCodable),
    ]
}
