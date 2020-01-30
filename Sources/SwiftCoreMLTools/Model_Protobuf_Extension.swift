import Foundation
import SwiftProtobuf

extension Model {
    public var coreMLData: Data? {
        let coreMLModel = convertToCoreML()

        let binaryModelData: Data? = try? coreMLModel.serializedData()

        return binaryModelData
    }

    func convertToCoreML() -> CoreML_Specification_Model {
        return CoreML_Specification_Model.with {
            $0.specificationVersion = Int32(self.version)
            $0.description_p = CoreML_Specification_ModelDescription.with {
                $0.input = [CoreML_Specification_FeatureDescription.with {
                    $0.name = "dense_input"
                    $0.type = CoreML_Specification_FeatureType.with {
                        $0.multiArrayType = CoreML_Specification_ArrayFeatureType.with {
                            $0.shape = [1]
                            $0.dataType = CoreML_Specification_ArrayFeatureType.ArrayDataType.double
                        }
                    }
                }]
                $0.output = [CoreML_Specification_FeatureDescription.with {
                    $0.name = "output"
                    $0.type = CoreML_Specification_FeatureType.with {
                        $0.multiArrayType = CoreML_Specification_ArrayFeatureType.with {
                            $0.shape = [1]
                            $0.dataType = CoreML_Specification_ArrayFeatureType.ArrayDataType.double
                        }
                    }
                }]
                $0.trainingInput = [CoreML_Specification_FeatureDescription.with {
                    $0.name = "dense_input"
                    $0.type = CoreML_Specification_FeatureType.with {
                        $0.multiArrayType = CoreML_Specification_ArrayFeatureType.with {
                            $0.shape = [1]
                            $0.dataType = CoreML_Specification_ArrayFeatureType.ArrayDataType.double
                        }
                    }
                }, CoreML_Specification_FeatureDescription.with {
                    $0.name = "output_true"
                    $0.type = CoreML_Specification_FeatureType.with {
                        $0.multiArrayType = CoreML_Specification_ArrayFeatureType.with {
                            $0.shape = [1]
                            $0.dataType = CoreML_Specification_ArrayFeatureType.ArrayDataType.double
                        }
                    }

                }]
                $0.metadata = CoreML_Specification_Metadata.with {
                    $0.shortDescription = "Trivial linear classifier"
                    $0.author = "Jacopo Mangiavacchi"
                    $0.license = "MIT"
                    $0.userDefined = ["coremltoolsVersion" : "3.1"]
                }
            }
            $0.isUpdatable = true
            $0.neuralNetwork = CoreML_Specification_NeuralNetwork.with {
                $0.layers = [CoreML_Specification_NeuralNetworkLayer.with {
                    $0.name = "dense_1"
                    $0.input = ["dense_input"]
                    $0.output = ["output"]
                    $0.isUpdatable = true
                    $0.innerProduct = CoreML_Specification_InnerProductLayerParams.with {
                        $0.inputChannels = 1
                        $0.outputChannels = 1
                        $0.hasBias_p = true
                        $0.weights = CoreML_Specification_WeightParams.with {
                            $0.floatValue = [0.0]
                            $0.isUpdatable = true
                        }
                        $0.bias = CoreML_Specification_WeightParams.with {
                            $0.floatValue = [0.0]
                            $0.isUpdatable = true
                        }
                    }
                }]
                $0.updateParams = CoreML_Specification_NetworkUpdateParameters.with {
                    $0.lossLayers = [CoreML_Specification_LossLayer.with {
                        $0.name = "lossLayer"
                        $0.meanSquaredErrorLossLayer = CoreML_Specification_MeanSquaredErrorLossLayer.with {
                            $0.input = "output"
                            $0.target = "output_true"
                        }
                    }]
                    $0.optimizer = CoreML_Specification_Optimizer.with {
                        $0.sgdOptimizer = CoreML_Specification_SGDOptimizer.with {
                            $0.learningRate = CoreML_Specification_DoubleParameter.with {
                                $0.defaultValue = 0.01
                                $0.range = CoreML_Specification_DoubleRange.with {
                                    $0.maxValue = 1.0
                                }
                            }
                            $0.miniBatchSize = CoreML_Specification_Int64Parameter.with {
                                $0.defaultValue = 5
                                $0.set = CoreML_Specification_Int64Set.with {
                                    $0.values = [5]
                                }
                            }
                            $0.momentum = CoreML_Specification_DoubleParameter.with {
                                $0.defaultValue = 0
                                $0.range = CoreML_Specification_DoubleRange.with {
                                    $0.maxValue = 1.0
                                }
                            }
                        }
                    }
                    $0.epochs = CoreML_Specification_Int64Parameter.with {
                        $0.defaultValue = 2
                        $0.set = CoreML_Specification_Int64Set.with {
                            $0.values = [2]
                        }
                    }
                    $0.shuffle = CoreML_Specification_BoolParameter.with {
                        $0.defaultValue = true
                    }
                }
            }
        }
    }
}