import XCTest
@testable import SwiftCoreMLTools

final class NeuralNetworkTests: XCTestCase {
    func testSingleLayer() {
        let network = NeuralNetwork {
            InnerProductLayer(name: "layer1",
                              input: ["dense_input"],
                              output: ["output"],
                              updatable: true,
                              weights: [0.0],
                              bias: [0.0])
        }

        XCTAssertEqual(network.layers?.count, 1)
    }

    func testMultipleLayers() {
        let network = NeuralNetwork {
            InnerProductLayer(name: "layer1",
                              input: ["dense_input"],
                              output: ["output"],
                              updatable: true,
                              weights: [0.0],
                              bias: [0.0])
            InnerProductLayer(name: "layer2",
                              input: ["dense_input"],
                              output: ["output"],
                              updatable: true,
                              weights: [0.0],
                              bias: [0.0])
        }

        XCTAssertEqual(network.layers?.count, 2)
    }

    static var allTests = [
        ("testSingleLayer", testSingleLayer),
        ("testMultipleLayers", testMultipleLayers),
    ]
}
