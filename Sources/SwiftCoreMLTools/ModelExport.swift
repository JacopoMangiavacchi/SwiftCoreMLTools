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
                $0.input = convertToFeatureDescriptionList(dictionary: self.inputs)
                $0.output = convertToFeatureDescriptionList(dictionary: self.outputs)
                $0.trainingInput = convertToFeatureDescriptionList(dictionary: self.trainingInputs)
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
                $0.layers = self.neuralNetwork.layers.map{ layer in 
                    return CoreML_Specification_NeuralNetworkLayer.with {
                        $0.name = layer.name
                        $0.input = layer.input
                        $0.output = layer.output

                        switch layer {
                        case let innerProduct as InnerProduct:
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

                        case let convolution as Convolution:
                            $0.isUpdatable = convolution.updatable
                            $0.convolution = CoreML_Specification_ConvolutionLayerParams.with {
                                $0.outputChannels = UInt64(convolution.outputChannels)
                                $0.kernelChannels = UInt64(convolution.kernelChannels)
                                $0.nGroups = UInt64(convolution.nGroups)
                                $0.kernelSize = convolution.kernelSize.map{ UInt64($0) }
                                $0.stride = convolution.stride.map{ UInt64($0) }
                                $0.dilationFactor = convolution.stride.map{ UInt64($0) }
  
                                switch convolution.paddingType {
                                case .valid(let borderAmounts):
                                    $0.valid = CoreML_Specification_ValidPadding.with {
                                        $0.paddingAmounts = CoreML_Specification_BorderAmounts.with {
                                            $0.borderAmounts = borderAmounts.map{
                                                var edge = CoreML_Specification_BorderAmounts.EdgeSizes()
                                                edge.startEdgeSize = UInt64($0.startEdgeSize)
                                                edge.endEdgeSize = UInt64($0.endEdgeSize)
                                                return edge
                                            }
                                        }
                                    }

                                case .same(let mode):
                                    $0.same = CoreML_Specification_SamePadding.with {
                                        switch mode {
                                        case .bottomRightHeavy:
                                            $0.asymmetryMode = .bottomRightHeavy

                                        case .topLeftHeavy:
                                            $0.asymmetryMode = .topLeftHeavy
                                        }
                                    }
                                }
  
                                $0.isDeconvolution = convolution.deconvolution
                                $0.hasBias_p = true
                                $0.weights = CoreML_Specification_WeightParams.with {
                                    $0.floatValue = convolution.weights
                                    $0.isUpdatable = convolution.updatable
                                }
                                $0.bias = CoreML_Specification_WeightParams.with {
                                    $0.floatValue = convolution.bias
                                    $0.isUpdatable = convolution.updatable
                                }

                                $0.outputShape = convolution.outputShape.map{ UInt64($0) }
                            }

                        case let activation as Activation:
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

                        default:
                            break
                        }
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

    func convertToFeatureDescriptionList(dictionary: [String : InputOutputItems]) -> [CoreML_Specification_FeatureDescription] {
        return dictionary.values.map{ input in 
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
    }
}
