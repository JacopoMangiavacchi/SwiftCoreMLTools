import XCTest
@testable import SwiftCoreMLTools

final class NeuralNetworkTests: XCTestCase {
    func testSingleLayer() {
        let network = NeuralNetwork {
            InnerProduct(name: "layer1",
                         input: ["dense_input"],
                         output: ["output"],
                         weights: [0.0],
                         bias: [0.0],
                         inputChannels: 1,
                         outputChannels: 1,
                         updatable: true)
        }

        XCTAssertEqual(network.layers.count, 1)
    }

    func testMultipleLayers() {
        let network = NeuralNetwork {
            InnerProduct(name: "layer1",
                         input: ["dense_input"],
                         output: ["output"],
                         weights: [0.0],
                         bias: [0.0],
                         inputChannels: 1,
                         outputChannels: 1,
                         updatable: true)
            InnerProduct(name: "layer2",
                         input: ["dense_input"],
                         output: ["output"],
                         weights: [0.0],
                         bias: [0.0],
                         inputChannels: 1,
                         outputChannels: 1,
                         updatable: true)
        }

        XCTAssertEqual(network.layers.count, 2)
    }

    func testCodable() {
         let network = NeuralNetwork(losses: [MSE(name: "lossLayer",
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
                         weights: [0.0],
                         bias: [0.0],
                         inputChannels: 1,
                         outputChannels: 1,
                         updatable: true)
            InnerProduct(name: "layer2",
                         input: ["dense_input"],
                         output: ["output"],
                         weights: [0.0],
                         bias: [0.0],
                         inputChannels: 1,
                         outputChannels: 1,
                         updatable: true)
        }

        XCTAssertEqual(network.layers.count, 2)

        let jsonData = try! JSONEncoder().encode(network)
        let decodedNetwork = try! JSONDecoder().decode(NeuralNetwork.self, from: jsonData)

        XCTAssertEqual(decodedNetwork.layers.count, 2)
    }

    static var allTests = [
        ("testSingleLayer", testSingleLayer),
        ("testMultipleLayers", testMultipleLayers),
        ("testCodable", testCodable),
    ]
}
