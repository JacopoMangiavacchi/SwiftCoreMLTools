import XCTest
@testable import SwiftCoreMLTools

final class NeuralNetworkTests: XCTestCase {
    func testSingleLayer() {
        let network = NeuralNetwork {
            InnerProductLayer(name: "layer1")
        }

        XCTAssertEqual(network.layers?.count, 1)
    }

    func testMultipleLayers() {
        let network = NeuralNetwork {
            InnerProductLayer(name: "layer1")
            InnerProductLayer(name: "layer2")
        }

        XCTAssertEqual(network.layers?.count, 2)
    }

    static var allTests = [
        ("testSingleLayer", testSingleLayer),
        ("testMultipleLayers", testMultipleLayers),
    ]
}
