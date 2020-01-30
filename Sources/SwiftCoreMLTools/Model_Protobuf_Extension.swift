import Foundation
import SwiftProtobuf

extension Model {
    public var coreMLData: Data? {
        let coreMLModel = CoreML_Specification_Model()

        let binaryModelData: Data? = try? coreMLModel.serializedData()

        return binaryModelData
    }
}