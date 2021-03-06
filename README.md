# SwiftCoreMLTools

![Swift](https://github.com/JacopoMangiavacchi/SwiftCoreMLTools/workflows/Swift/badge.svg)
[![Swift Package Manager compatible](https://img.shields.io/badge/Swift%20Package%20Manager-compatible-brightgreen.svg)](https://github.com/apple/swift-package-manager)
[![ver](https://img.shields.io/github/v/release/JacopoMangiavacchi/SwiftCoreMLTools?include_prereleases&label=version)](https://github.com/JacopoMangiavacchi/SwiftCoreMLTools)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) 
![Documentation](https://github.com/JacopoMangiavacchi/SwiftCoreMLTools/workflows/Documentation/badge.svg)

A Swift Library for creating CoreML models in Swift.

**Work in progress**

This library expose a (function builder based) DSL as well as a programmatic API (see examples below).

The library also implement Codable protocol allowing to print and edit CoreML model in JSON format.

**The library is not "official" - it is not part of Apple CoreML and it is not maintained.**

This library use the Apple Swift Protocol Buffer package and compile and import to Swift the CoreML ProtoBuf datastructures defined from the GitHub Apple CoreMLTools repo - https://github.com/apple/coremltools/tree/master/mlmodel/format

This package could be used to export Swift For TensorFlow models or to generate new CoreML models from scratch providing a much swifty interface compared to directly using the Swift compiled CoreML protobuf data structures.

CoreML models generated with this library could be potentially personalized (trained) partially or entirely using the CoreML runtime.

CoreML support much more then Neural Network models but this experimental library is only focused, at the moment, on Neural Network support.

> End to end test case exporting a real S4TF model at:
> https://github.com/JacopoMangiavacchi/TestSwiftCoreMLTools

## Sample projects using this library to create and train Core ML models on device

### LeNet Convolutional Neural Network for MNIST dataset

- GitHub project: https://github.com/JacopoMangiavacchi/MNIST-CoreML-Training
- Documentation: https://medium.com/@JMangia/mnist-cnn-core-ml-training-c0f081014fa6

### Transfer Learning with Categorical Embedding

- GitHub project: https://github.com/JacopoMangiavacchi/CoreML-TransferLearning-Demo
- Documentation: https://heartbeat.fritz.ai/core-ml-on-device-training-with-transfer-learning-from-swift-for-tensorflow-models-1264b444e18d


## Documentation

[Documentation](https://jacopomangiavacchi.github.io/SwiftCoreMLTools/Documentation/)


## Neural Network Support (work in progress)

### Layers

- InnerProduct
- Convolution
- Embedding
- Flatten
- Pooling
- Permute
- Concat

### Activation Functions

- Linear
- ReLu
- LeakyReLu
- ThresholdedReLu
- PReLu
- Tanh
- ScaledTanh
- Sigmoid
- SigmoidHard
- Elu
- Softsign
- Softplus
- ParametricSoftplus
- Softmax

### Loss Functions

- MSE
- CategoricalCrossEntropy

### Optimizers

- SGD
- Adam

## Export Swift for TensorFlow sample scenario

### Trivial Swift for TensorFlow model

```swift
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

### Export to CoreML using DSL approach

```swift
let coremlModel = Model(version: 4,
                        shortDescription: "Trivial linear classifier",
                        author: "Jacopo Mangiavacchi",
                        license: "MIT",
                        userDefined: ["SwiftCoremltoolsVersion" : "0.1"]) {
    Input(name: "dense_input", shape: [1])
    Output(name: "output", shape: [1])
    NeuralNetwork {
        InnerProduct(name: "dense_1",
                     input: ["dense_input"],
                     output: ["output"],
                     weight: s4tfModel.layer1.weight.transposed().flattened().scalars,
                     bias: s4tfModel.layer1.bias.flattened().scalars,
                     inputChannels: 1,
                     outputChannels: 1)
    }
}
```

### Export a CoreML personalizable (re-trainable) model using DSL approach

```swift
let coremlModel = Model(version: 4,
                        shortDescription: "Trivial linear classifier",
                        author: "Jacopo Mangiavacchi",
                        license: "MIT",
                        userDefined: ["SwiftCoremltoolsVersion" : "0.1"]) {
    Input(name: "dense_input", shape: [1])
    Output(name: "output", shape: [1])
    TrainingInput(name: "dense_input", shape: [1])
    TrainingInput(name: "output_true", shape: [1])
    NeuralNetwork(losses: [MSE(name: "lossLayer",
                               input: "output",
                               target: "output_true")],
                  optimizer: SGD(learningRateDefault: 0.01,
                                 learningRateMax: 0.3,
                                 miniBatchSizeDefault: 5,
                                 miniBatchSizeRange: [5],
                                 momentumDefault: 0,
                                 momentumMax: 1.0),
                  epochDefault: 2,
                  epochSet: [2],
                  shuffle: true) {
        InnerProduct(name: "dense_1",
                     input: ["dense_input"],
                     output: ["output"],
                     weight: s4tfModel.layer1.weight.transposed().flattened().scalars,
                     bias: s4tfModel.layer1.bias.flattened().scalars,
                     inputChannels: 1,
                     outputChannels: 1,
                     updatable: true)
    }
}
```

## Example code to export and save to a CoreML model data file

```swift
let model = Model(...){ ... }
let coreMLData = model.coreMLData
try! coreMLData.write(to: URL(fileURLWithPath: "model.mlmodel"))
```

## CoreML model creation with programmatic API

```swift
var model = Model(version: 4,
                  shortDescription: "Trivial linear classifier",
                  author: "Jacopo Mangiavacchi",
                  license: "MIT",
                  userDefined: ["SwiftCoremltoolsVersion" : "0.1"])

model.addInput(Input(name: "dense_input", shape: [1]))
model.addOutput(Output(name: "output", shape: [1]))
model.addTrainingInput(TrainingInput(name: "dense_input", shape: [1]))
model.addTrainingInput(TrainingInput(name: "output_true", shape: [1]))
model.neuralNetwork = NeuralNetwork(losses: [MSE(name: "lossLayer",
                                                 input: "output",
                                                 target: "output_true")],
                                    optimizer: SGD(learningRateDefault: 0.01,
                                                   learningRateMax: 0.3,
                                                   miniBatchSizeDefault: 5,
                                                   miniBatchSizeRange: [5],
                                                   momentumDefault: 0,
                                                   momentumMax: 1.0),
                                    epochDefault: 2,
                                    epochSet: [2],
                                    shuffle: true)

model.neuralNetwork.addLayer(InnerProduct(name: "layer1",
                                         input: ["dense_input"],
                                         output: ["output"],
                                         weight: [0.0],
                                         bias: [0.0],
                                         inputChannels: 1,
                                         outputChannels: 1,
                                         updatable: true))
```

## YAML / JSON Format model persistence (Codable)

### Example CoreML model in YAML format

```yaml
version: 4
shortDescription: Trivial linear classifier
author: Jacopo Mangiavacchi
license: MIT
userDefined:
  SwiftCoremltoolsVersion: '0.1'
inputs:
  dense_input:
    name: dense_input
    shape:
    - 1
    featureType: float
outputs:
  output:
    name: output
    shape:
    - 1
    featureType: float
trainingInputs:
  dense_input:
    name: dense_input
    shape:
    - 1
    featureType: float
  output_true:
    name: output_true
    shape:
    - 1
    featureType: float
neuralNetwork:
  losses:
  - type: mse
    base:
      name: lossLayer
      input: output
      target: output_true
  optimizer:
    type: sgd
    base:
      learningRateDefault: 1e-2
      learningRateMax: 3e-1
      miniBatchSizeDefault: 5
      miniBatchSizeRange:
      - 5
      momentumDefault: 0e+0
      momentumMax: 1e+0
  epochDefault: 2
  epochSet:
  - 2
  shuffle: true
  layers:
  - type: innerProduct
    base:
      name: layer1
      input:
      - dense_input
      output:
      - output
      weight:
      - 0e+0
      bias:
      - 0e+0
      inputChannels: 1
      outputChannels: 1
      updatable: true
```

### Example CoreML model in JSON format

```json
{
  "author" : "Jacopo Mangiavacchi",
  "shortDescription" : "Trivial linear classifier",
  "version" : 4,
  "license" : "MIT",
  "userDefined" : {
    "SwiftCoremltoolsVersion" : "0.1"
  },
  "inputs" : {
    "dense_input" : {
      "name" : "dense_input",
      "shape" : [1],
      "featureType" : "float"
    }
  },
  "outputs" : {
    "output" : {
      "name" : "output",
      "shape" : [1],
      "featureType" : "float"
    }
  },
  "trainingInputs" : {
    "output_true" : {
      "name" : "output_true",
      "shape" : [1],
      "featureType" : "float"
    },
    "dense_input" : {
      "name" : "dense_input",
      "shape" : [1],
      "featureType" : "float"
    }
  },
  "neuralNetwork" : {
    "layers" : [
      {
        "type" : "innerProduct",
        "base" : {
          "output" : [
            "output"
          ],
          "outputChannels" : 1,
          "weight" : [0],
          "input" : [
            "dense_input"
          ],
          "bias" : [0],
          "inputChannels" : 1,
          "updatable" : true,
          "name" : "layer1"
        }
      }
    ],
    "optimizer" : {
      "type" : "sgd",
      "base" : {
        "momentumDefault" : 0,
        "momentumMax" : 1,
        "learningRateMax" : 0.3,
        "miniBatchSizeRange" : [5],
        "miniBatchSizeDefault" : 5,
        "learningRateDefault" : 0.01
      }
    },
    "losses" : [
      {
        "type" : "mse",
        "base" : {
          "name" : "lossLayer",
          "input" : "output",
          "target" : "output_true"
        }
      }
    ],
    "shuffle" : true,
    "epochDefault" : 2,
    "epochSet" : [2]
  }
}
```
