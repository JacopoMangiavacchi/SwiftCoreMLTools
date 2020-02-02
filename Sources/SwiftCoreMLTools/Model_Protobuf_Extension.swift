import Foundation
import SwiftProtobuf

extension Model {
    public var coreMLData: Data? {
        let coreMLModel = convertToCoreML()

        let binaryModelData: Data? = try? coreMLModel.serializedData()

        return binaryModelData
    }

    func convertToCoreML() -> CoreML_Specification_Model {
        guard self.inputs.count > 0 &&
              self.outputs.count > 0 &&
              self.neuralNetwork.layers.count > 0 else { 
            return CoreML_Specification_Model() 
        }

        let trainable:Bool = self.trainingInputs.count > 0

        return CoreML_Specification_Model.with { model in 
            model.specificationVersion = Int32(self.version)

            model.description_p = CoreML_Specification_ModelDescription.with {
                $0.input = self.inputs.values.map{ input in 
                    CoreML_Specification_FeatureDescription.with {
                        $0.name = input.name
                        $0.type = CoreML_Specification_FeatureType.with {
                            $0.multiArrayType = CoreML_Specification_ArrayFeatureType.with {
                                $0.shape = input.shape.map{ Int64($0) }
                                $0.dataType = CoreML_Specification_ArrayFeatureType.ArrayDataType.double
                            }
                        }
                    }
                }

                $0.output = self.outputs.values.map{ output in 
                    CoreML_Specification_FeatureDescription.with {
                        $0.name = output.name
                        $0.type = CoreML_Specification_FeatureType.with {
                            $0.multiArrayType = CoreML_Specification_ArrayFeatureType.with {
                                $0.shape = output.shape.map{ Int64($0) }
                                $0.dataType = CoreML_Specification_ArrayFeatureType.ArrayDataType.double
                            }
                        }
                    }
                }

                $0.trainingInput = self.trainingInputs.values.map{ trainingInput in 
                    CoreML_Specification_FeatureDescription.with {
                        $0.name = trainingInput.name
                        $0.type = CoreML_Specification_FeatureType.with {
                            $0.multiArrayType = CoreML_Specification_ArrayFeatureType.with {
                                $0.shape = trainingInput.shape.map{ Int64($0) }
                                $0.dataType = CoreML_Specification_ArrayFeatureType.ArrayDataType.double
                            }
                        }
                    }
                }

                $0.metadata = CoreML_Specification_Metadata.with {
                    $0.shortDescription = self.shortDescription ?? ""
                    $0.author = self.author ?? ""
                    $0.license = self.license ?? ""
                    $0.userDefined = self.userDefined ?? [:]
                }
            }

            if trainable {
                model.isUpdatable = true
            }

            model.neuralNetwork = CoreML_Specification_NeuralNetwork.with {
                $0.layers = (self.neuralNetwork.layers.filter{ $0 is InnerProduct } as! [InnerProduct]).map{ layer in 
                    CoreML_Specification_NeuralNetworkLayer.with {
                        $0.name = layer.name
                        $0.input = layer.input
                        $0.output = layer.output
                        $0.isUpdatable = layer.updatable
                        $0.innerProduct = CoreML_Specification_InnerProductLayerParams.with {
                            $0.inputChannels = UInt64(layer.inputChannels)
                            $0.outputChannels = UInt64(layer.outputChannels)
                            $0.hasBias_p = true
                            $0.weights = CoreML_Specification_WeightParams.with {
                                $0.floatValue = layer.weights
                                $0.isUpdatable = layer.updatable
                            }
                            $0.bias = CoreML_Specification_WeightParams.with {
                                $0.floatValue = layer.bias
                                $0.isUpdatable = layer.updatable
                            }
                        }
                    }
                }

                if trainable, 
                   let loss = self.neuralNetwork.loss,
                   loss.count > 0,
                   let optimizer = self.neuralNetwork.optimizer,
                   let epochDefault = self.neuralNetwork.epochDefault,
                   let epochSet = self.neuralNetwork.epochSet,
                   let shuffle = self.neuralNetwork.shuffle {
                    $0.updateParams = CoreML_Specification_NetworkUpdateParameters.with {
                        $0.lossLayers = loss.map{ loss in 
                            CoreML_Specification_LossLayer.with {
                                switch loss {
                                case let mse as MSE:
                                    $0.name = mse.name
                                    $0.meanSquaredErrorLossLayer = CoreML_Specification_MeanSquaredErrorLossLayer.with {
                                        $0.input = mse.input
                                        $0.target = mse.target
                                    }
                                default:
                                    break
                                }
                            }
                        }
                                                
                        $0.optimizer = CoreML_Specification_Optimizer.with {
                            switch optimizer {
                            case let sgd as SGD:
                                $0.sgdOptimizer = CoreML_Specification_SGDOptimizer.with {
                                    $0.learningRate = CoreML_Specification_DoubleParameter.with {
                                        $0.defaultValue = sgd.learningRateDefault
                                        $0.range = CoreML_Specification_DoubleRange.with {
                                            $0.maxValue = sgd.learningRateMax
                                        }
                                    }
                                    $0.miniBatchSize = CoreML_Specification_Int64Parameter.with {
                                        $0.defaultValue = Int64(sgd.miniBatchSizeDefault)
                                        $0.set = CoreML_Specification_Int64Set.with {
                                            $0.values = sgd.miniBatchSizeRange.map{ Int64($0) }
                                        }
                                    }
                                    $0.momentum = CoreML_Specification_DoubleParameter.with {
                                        $0.defaultValue = sgd.momentumDefault
                                        $0.range = CoreML_Specification_DoubleRange.with {
                                            $0.maxValue = sgd.momentumMax
                                        }
                                    }
                                }
                            default:
                                break
                            }
                        }
                        
                        $0.epochs = CoreML_Specification_Int64Parameter.with {
                            $0.defaultValue = Int64(epochDefault)
                            $0.set = CoreML_Specification_Int64Set.with {
                                $0.values = epochSet.map{ Int64($0) }
                            }
                        }
                        
                        $0.shuffle = CoreML_Specification_BoolParameter.with {
                            $0.defaultValue = shuffle
                        }
                    }
                }
            }
        }
    }
}