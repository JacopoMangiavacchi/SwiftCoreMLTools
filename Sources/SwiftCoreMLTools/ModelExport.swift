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

            model.description_p = CoreML_Specification_ModelDescription.with { descriptionSpec in
                descriptionSpec.input = convertToFeatureDescriptionList(dictionary: self.inputs)
                descriptionSpec.output = convertToFeatureDescriptionList(dictionary: self.outputs)
                descriptionSpec.trainingInput = convertToFeatureDescriptionList(dictionary: self.trainingInputs)
                descriptionSpec.metadata = CoreML_Specification_Metadata.with { metadataSpec in
                    metadataSpec.shortDescription = self.shortDescription ?? ""
                    metadataSpec.author = self.author ?? ""
                    metadataSpec.license = self.license ?? ""
                    metadataSpec.userDefined = self.userDefined ?? [:]
                }
            }

            if trainable {
                model.isUpdatable = true
            }

            model.neuralNetwork = CoreML_Specification_NeuralNetwork.with { neuralNetworkSpec in
                neuralNetworkSpec.layers = self.neuralNetwork.layers.values.map{ layer in 
                    return CoreML_Specification_NeuralNetworkLayer.with { layerSpec in 
                        layerSpec.name = layer.name
                        layerSpec.input = layer.input
                        layerSpec.output = layer.output

                        switch layer {
                        case let innerProduct as InnerProduct:
                            layerSpec.isUpdatable = innerProduct.updatable
                            layerSpec.innerProduct = convertToInnerProduct(innerProduct: innerProduct)

                        case let convolution as Convolution:
                            layerSpec.isUpdatable = convolution.updatable
                            layerSpec.convolution = convertToConvolution(convolution: convolution)

                        case let activation as Activation:
                            layerSpec.activation = convertToActivation(activation: activation)

                        case let embedding as Embedding:
                            layerSpec.embedding = convertToEmbedding(embedding: embedding)

                        case let permute as Permute:
                            layerSpec.permute = convertToPermute(permute: permute)

                        case let flatten as Flatten:
                            layerSpec.flatten = convertToFlatten(flatten: flatten)

                        case let concat as Concat:
                            layerSpec.concat = convertToConcat(concat: concat)

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
                    neuralNetworkSpec.updateParams = convertToUpdateParam(losses: losses, optimizer: optimizer, epochDefault: epochDefault, epochSet: epochSet, shuffle: shuffle)
                }
            }
        }
    }

    func convertToFeatureDescriptionList(dictionary: [String : InputOutputItems]) -> [CoreML_Specification_FeatureDescription] {
        return dictionary.values.map{ input in 
            CoreML_Specification_FeatureDescription.with { featureSpec in
                featureSpec.name = input.name
                featureSpec.type = CoreML_Specification_FeatureType.with { featureTypeSpec in
                    featureTypeSpec.multiArrayType = CoreML_Specification_ArrayFeatureType.with { featureArrayTypeSpec in
                        featureArrayTypeSpec.shape = input.shape.map{ Int64($0) }
                        featureArrayTypeSpec.dataType = CoreML_Specification_ArrayFeatureType.ArrayDataType.float32
                    }
                }
            }
        }
    }

    func convertToInnerProduct(innerProduct: InnerProduct) -> CoreML_Specification_InnerProductLayerParams {
        return CoreML_Specification_InnerProductLayerParams.with { innerProductSpec in
            innerProductSpec.inputChannels = UInt64(innerProduct.inputChannels)
            innerProductSpec.outputChannels = UInt64(innerProduct.outputChannels)
            innerProductSpec.hasBias_p = true
            innerProductSpec.weights = CoreML_Specification_WeightParams.with { weightsSpec in
                weightsSpec.floatValue = innerProduct.weight
                weightsSpec.isUpdatable = innerProduct.updatable
            }
            innerProductSpec.bias = CoreML_Specification_WeightParams.with { biasSpec in
                biasSpec.floatValue = innerProduct.bias
                biasSpec.isUpdatable = innerProduct.updatable
            }
        }
    }

    func convertToConvolution(convolution: Convolution) -> CoreML_Specification_ConvolutionLayerParams {
        return CoreML_Specification_ConvolutionLayerParams.with { convolutionSpec in
            convolutionSpec.outputChannels = UInt64(convolution.outputChannels)
            convolutionSpec.kernelChannels = UInt64(convolution.kernelChannels)
            convolutionSpec.nGroups = UInt64(convolution.nGroups)
            convolutionSpec.kernelSize = convolution.kernelSize.map{ UInt64($0) }
            convolutionSpec.stride = convolution.stride.map{ UInt64($0) }
            convolutionSpec.dilationFactor = convolution.stride.map{ UInt64($0) }

            switch convolution.paddingType {
            case .valid(let borderAmounts):
                convolutionSpec.valid = CoreML_Specification_ValidPadding.with { paddingSpec in
                    paddingSpec.paddingAmounts = CoreML_Specification_BorderAmounts.with {  borderSpec in
                        borderSpec.borderAmounts = borderAmounts.map{
                            var edge = CoreML_Specification_BorderAmounts.EdgeSizes()
                            edge.startEdgeSize = UInt64($0.startEdgeSize)
                            edge.endEdgeSize = UInt64($0.endEdgeSize)
                            return edge
                        }
                    }
                }

            case .same(let mode):
                convolutionSpec.same = CoreML_Specification_SamePadding.with { paddingSpec in
                    switch mode {
                    case .bottomRightHeavy:
                        paddingSpec.asymmetryMode = .bottomRightHeavy

                    case .topLeftHeavy:
                        paddingSpec.asymmetryMode = .topLeftHeavy
                    }
                }
            }

            convolutionSpec.isDeconvolution = convolution.deconvolution
            convolutionSpec.hasBias_p = true
            convolutionSpec.weights = CoreML_Specification_WeightParams.with { weightsSpec in
                weightsSpec.floatValue = convolution.weight
                weightsSpec.isUpdatable = convolution.updatable
            }
            convolutionSpec.bias = CoreML_Specification_WeightParams.with { biasSpec in
                biasSpec.floatValue = convolution.bias
                biasSpec.isUpdatable = convolution.updatable
            }

            convolutionSpec.outputShape = convolution.outputShape.map{ UInt64($0) }
        }
    }

    func convertToActivation(activation: Activation) -> CoreML_Specification_ActivationParams {
        return CoreML_Specification_ActivationParams.with { activationParam in
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

    func convertToUpdateParam(losses: [Loss], optimizer: Optimizer, epochDefault: UInt, epochSet: [UInt], shuffle: Bool) -> CoreML_Specification_NetworkUpdateParameters {
        return CoreML_Specification_NetworkUpdateParameters.with { updateSpec in
            updateSpec.lossLayers = losses.map{ loss in 
                CoreML_Specification_LossLayer.with { lossSpec in
                    switch loss {
                    case let mse as MSE:
                        lossSpec.name = mse.name
                        lossSpec.meanSquaredErrorLossLayer = CoreML_Specification_MeanSquaredErrorLossLayer.with { mseSpec in
                            mseSpec.input = mse.input
                            mseSpec.target = mse.target
                        }

                    case let categoricalCrossEntropy as CategoricalCrossEntropy:
                        lossSpec.name = categoricalCrossEntropy.name
                        lossSpec.categoricalCrossEntropyLossLayer = CoreML_Specification_CategoricalCrossEntropyLossLayer.with { categoricalCrossEntropySpec in
                            categoricalCrossEntropySpec.input = categoricalCrossEntropy.input
                            categoricalCrossEntropySpec.target = categoricalCrossEntropy.target
                        }

                    default:
                        break
                    }
                }
            }
                                    
            updateSpec.optimizer = CoreML_Specification_Optimizer.with { optimizerSpec in
                switch optimizer {
                case let sgd as SGD:
                    optimizerSpec.sgdOptimizer = CoreML_Specification_SGDOptimizer.with { sgdSpec in
                        sgdSpec.learningRate = CoreML_Specification_DoubleParameter.with { learningRateSpec in
                            learningRateSpec.defaultValue = sgd.learningRateDefault
                            learningRateSpec.range = CoreML_Specification_DoubleRange.with {
                                $0.maxValue = sgd.learningRateMax
                            }
                        }
                        sgdSpec.miniBatchSize = CoreML_Specification_Int64Parameter.with { miniBatchSizeSpec in
                            miniBatchSizeSpec.defaultValue = Int64(sgd.miniBatchSizeDefault)
                            miniBatchSizeSpec.set = CoreML_Specification_Int64Set.with {
                                $0.values = sgd.miniBatchSizeRange.map{ Int64($0) }
                            }
                        }
                        sgdSpec.momentum = CoreML_Specification_DoubleParameter.with { momentumSpec in
                            momentumSpec.defaultValue = sgd.momentumDefault
                            momentumSpec.range = CoreML_Specification_DoubleRange.with {
                                $0.maxValue = sgd.momentumMax
                            }
                        }
                    }
                    
                case let adam as Adam:
                    optimizerSpec.adamOptimizer = CoreML_Specification_AdamOptimizer.with { adamSpec in
                        adamSpec.learningRate = CoreML_Specification_DoubleParameter.with { learningRateSpec in
                            learningRateSpec.defaultValue = adam.learningRateDefault
                            learningRateSpec.range = CoreML_Specification_DoubleRange.with {
                                $0.maxValue = adam.learningRateMax
                            }
                        }
                        adamSpec.miniBatchSize = CoreML_Specification_Int64Parameter.with { miniBatchSizeSpec in
                            miniBatchSizeSpec.defaultValue = Int64(adam.miniBatchSizeDefault)
                            miniBatchSizeSpec.set = CoreML_Specification_Int64Set.with {
                                $0.values = adam.miniBatchSizeRange.map{ Int64($0) }
                            }
                        }
                        adamSpec.beta1 = CoreML_Specification_DoubleParameter.with { beta1Spec in
                            beta1Spec.defaultValue = adam.beta1Default
                            beta1Spec.range = CoreML_Specification_DoubleRange.with {
                                $0.maxValue = adam.beta1Max
                            }
                        }
                        adamSpec.beta2 = CoreML_Specification_DoubleParameter.with { beta2Spec in
                            beta2Spec.defaultValue = adam.beta2Default
                            beta2Spec.range = CoreML_Specification_DoubleRange.with {
                                $0.maxValue = adam.beta2Max
                            }
                        }
                        adamSpec.eps = CoreML_Specification_DoubleParameter.with { epsSpec in
                            epsSpec.defaultValue = adam.epsDefault
                            epsSpec.range = CoreML_Specification_DoubleRange.with {
                                $0.maxValue = adam.epsMax
                            }
                        }
                    }
                    
                default:
                    break
                }
            }
            
            updateSpec.epochs = CoreML_Specification_Int64Parameter.with { epochsSpec in
                epochsSpec.defaultValue = Int64(epochDefault)
                epochsSpec.set = CoreML_Specification_Int64Set.with {
                    $0.values = epochSet.map{ Int64($0) }
                }
            }
            
            updateSpec.shuffle = CoreML_Specification_BoolParameter.with {
                $0.defaultValue = shuffle
            }
        }
    }

    func convertToEmbedding(embedding: Embedding) -> CoreML_Specification_EmbeddingLayerParams {
        return CoreML_Specification_EmbeddingLayerParams.with {
            $0.inputDim = UInt64(embedding.inputDim)
            $0.outputChannels = UInt64(embedding.outputChannels)
            $0.weights = CoreML_Specification_WeightParams.with { weightsSpec in
                weightsSpec.floatValue = embedding.weight
            }
        }
    }

    func convertToPermute(permute: Permute) -> CoreML_Specification_PermuteLayerParams {
        return CoreML_Specification_PermuteLayerParams.with {
            $0.axis = permute.axis.map{ UInt64($0) }
        }
    }

    func convertToFlatten(flatten: Flatten) -> CoreML_Specification_FlattenLayerParams {
        return CoreML_Specification_FlattenLayerParams.with {
            switch flatten.mode {
            case .first:
                $0.mode = .channelFirst
            case .last:
                $0.mode = .channelLast
            }
        }
    }

    func convertToConcat(concat: Concat) -> CoreML_Specification_ConcatLayerParams {
        return CoreML_Specification_ConcatLayerParams()
    }
}
