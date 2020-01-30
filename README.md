# SwiftCoreMLTools

A Swift Library defining a (function builder based) DSL for creating and exporting CoreML Models in Swift.

This library use the Apple Swift Protocol Buffer package and compile and import to Swift the CoreML ProtoBuf datastructures defined from the GitHub Apple CoreMLTools repo - https://github.com/apple/coremltools/tree/master/mlmodel/format

This package could be used to export Swift For TensorFlow models or to generate new CoreML models from scratch providing a much swifty interface compared to directly using the Swift compiled CoreML protobuf data structures.

CoreML models generated with this library could be potentially personalized (trained) partially or entirely using the CoreML runtime.

## Export Swift for TensorFlow sample scenario

### Trivial Swift for TensorFlow model
```
struct LinearRegression: Layer {
    var layer1 = Dense<Float>(inputSize: 1, outputSize: 1, activation: identity)
    
    @differentiable
    func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        return layer1(input)
    }
}

var s4tfModel = LinearRegression()
// Training Loop ...
```

### Export a CoreML personalizable (re-trainable) model using this DSL approach
```
let coremlModel = Model(version: 4,
                        shortDescription: "Trivial linear classifier",
                        author: "Jacopo Mangiavacchi",
                        license: "MIT",
                        userDefined: ["SwiftCoremltoolsVersion" : "0.1"]) {
                            Inputs {
                                InputOutputDescription(name: "dense_input", shape: [1], type: Double.Type)
                            }
                            Outputs {
                                InputOutputDescription(name: "output", shape: [1], type: Double.Type)
                            }
                            TrainingInputs {
                                InputOutputDescription(name: "dense_input", shape: [1], type: Double.Type)
                                InputOutputDescription(name: "output_true", shape: [1], type: Double.Type)
                            }
                            NeuralNetwork {
                                InnerProductLayer(name: "dense_1",
                                                  input: ["dense_input"],
                                                  output: ["output"],
                                                  updatable: true,
                                                  weights: s4tfModel.layer1.weight[0],
                                                  bias: s4tfModel.layer1.bias)
                            }
}
```

### Verbouse alternative approach to explicitly use Swift version of the CoreML ProtoBuf data structure to export the model
```
func convertToCoreML(weights: Float, bias: Float) -> CoreML_Specification_Model {
    return CoreML_Specification_Model.with {
        $0.specificationVersion = 4
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
                        $0.floatValue = [weights]
                        $0.isUpdatable = true
                    }
                    $0.bias = CoreML_Specification_WeightParams.with {
                        $0.floatValue = [bias]
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
```

