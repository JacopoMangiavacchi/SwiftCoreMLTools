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
                $0.layers = self.neuralNetwork.layers.compactMap{ layer in 
                    switch layer {
                    case let innerProduct as InnerProduct:
                        return CoreML_Specification_NeuralNetworkLayer.with {
                            $0.name = innerProduct.name
                            $0.input = innerProduct.input
                            $0.output = innerProduct.output
                            $0.isUpdatable = innerProduct.updatable
                            $0.innerProduct = CoreML_Specification_InnerProductLayerParams.with {
                                $0.inputChannels = UInt64(innerProduct.inputChannels)
                                $0.outputChannels = UInt64(innerProduct.outputChannels)
                                $0.hasBias_p = true
                                $0.weights = CoreML_Specification_WeightParams.with {
                                    $0.floatValue = innerProduct.weights
                                    $0.isUpdatable = innerProduct.updatable
                                }
                                $0.bias = CoreML_Specification_WeightParams.with {
                                    $0.floatValue = innerProduct.bias
                                    $0.isUpdatable = innerProduct.updatable
                                }
                            }
                        }

                    case let activation as Activation:
                        return CoreML_Specification_NeuralNetworkLayer.with {
                            $0.name = activation.name
                            $0.input = activation.input
                            $0.output = activation.output
                            $0.activation = CoreML_Specification_ActivationParams.with { activationParam in
                                switch activation {
                                case let linear as Linear:
                                    activationParam.linear = CoreML_Specification_ActivationLinear.with {
                                        $0.alpha = linear.alpha
                                        $0.beta = linear.beta
                                    }

                                case _ as ReLu:
                                    activationParam.reLu = CoreML_Specification_ActivationReLU()

                                case let leakyReLu as LeakyReLu:
                                    activationParam.leakyReLu = CoreML_Specification_ActivationLeakyReLU.with {
                                        $0.alpha = leakyReLu.alpha
                                    }

                                case let thresholdedReLu as ThresholdedReLu:
                                    activationParam.thresholdedReLu = CoreML_Specification_ActivationThresholdedReLU.with {
                                        $0.alpha = thresholdedReLu.alpha
                                    }

                                case _ as PReLu:
                                    activationParam.preLu = CoreML_Specification_ActivationPReLU()

                                case _ as Tanh:
                                    activationParam.tanh = CoreML_Specification_ActivationTanh()

                                case let scaledTanh as ScaledTanh:
                                    activationParam.scaledTanh = CoreML_Specification_ActivationScaledTanh.with {
                                        $0.alpha = scaledTanh.alpha
                                        $0.beta = scaledTanh.beta
                                    }

                                case _ as Sigmoid:
                                    activationParam.sigmoid = CoreML_Specification_ActivationSigmoid()

                                case let sigmoidHard as SigmoidHard:
                                    activationParam.sigmoidHard = CoreML_Specification_ActivationSigmoidHard.with {
                                        $0.alpha = sigmoidHard.alpha
                                        $0.beta = sigmoidHard.beta
                                    }

                                case let elu as Elu:
                                    activationParam.elu = CoreML_Specification_ActivationELU.with {
                                        $0.alpha = elu.alpha
                                    }

                                case _ as Softsign:
                                    activationParam.softsign = CoreML_Specification_ActivationSoftsign()

                                case _ as Softplus:
                                    activationParam.softplus = CoreML_Specification_ActivationSoftplus()

                                case _ as ParametricSoftplus:
                                    activationParam.parametricSoftplus = CoreML_Specification_ActivationParametricSoftplus()

                                default:
                                    break
                                }
                            }
                        }

                    default:
                        return nil
                    }
                }

                if trainable, 
                   let losses = self.neuralNetwork.losses,
                   losses.count > 0,
                   let optimizer = self.neuralNetwork.optimizer,
                   let epochDefault = self.neuralNetwork.epochDefault,
                   let epochSet = self.neuralNetwork.epochSet,
                   let shuffle = self.neuralNetwork.shuffle {
                    $0.updateParams = CoreML_Specification_NetworkUpdateParameters.with {
                        $0.lossLayers = losses.map{ loss in 
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
