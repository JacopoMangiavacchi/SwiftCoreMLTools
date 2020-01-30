import XCTest
@testable import SwiftCoreMLTools

final class ModelTests: XCTestCase {
    func testSingleInputs() {
        let model = Model {
            Input(name: "dense_input1", shape: [1], featureType: .Double)
        }

        XCTAssertEqual(model.descriptions.count, 1)
    }

    func testMultipleInputs() {
        let model = Model {
            Input(name: "dense_input1", shape: [1], featureType: .Double)
            Input(name: "dense_input2", shape: [1], featureType: .Double)
            Output(name: "output", shape: [1], featureType: .Double)
            TrainingInput(name: "dense_input", shape: [1], featureType: .Double)
            TrainingInput(name: "output_true", shape: [1], featureType: .Double)
        }

        XCTAssertEqual(model.descriptions.count, 5)
    }

    static var allTests = [
        ("testSingleInputs", testSingleInputs),
        ("testMultipleInputs", testMultipleInputs),
    ]
}
