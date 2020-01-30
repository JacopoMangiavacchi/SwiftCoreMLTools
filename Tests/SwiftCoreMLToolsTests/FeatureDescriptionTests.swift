import XCTest
@testable import SwiftCoreMLTools

final class FeatureDescriptionTests: XCTestCase {
    func testSingleInputs() {
        let featureDescriptionArray = Inputs {
            InputOutputDescription(name: "dense_input1", shape: [1], type: .Double)
        }

        XCTAssertEqual(featureDescriptionArray.descriptions.count, 1)
    }

    func testMultipleInputs() {
        let featureDescriptionArray = Inputs {
            InputOutputDescription(name: "dense_input1", shape: [2, 3], type: .Double)
            InputOutputDescription(name: "dense_input2", shape: [4], type: .Double)
        }

        XCTAssertEqual(featureDescriptionArray.descriptions.count, 2)
    }

    func testOutputs() {
        print("Test Outputs")
    }

    func testTrainingInputs() {
        print("Test TrainingInputs")
    }

    static var allTests = [
        ("testSingleInputs", testSingleInputs),
        ("testMultipleInputs", testMultipleInputs),
        ("testOutputs", testOutputs),
        ("testTrainingInputs", testTrainingInputs),
    ]
}
