# SwiftCoreMLTools

![Swift](https://github.com/JacopoMangiavacchi/SwiftCoreMLTools/workflows/Swift/badge.svg)

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

## JSON Format model persistence (Codable)

### Example code to Encode a CoreML model to JSON String

```swift
let model = Model(...){...}

let encoder = JSONEncoder()
encoder.outputFormatting = .prettyPrinted

let jsonData = try! encoder.encode(model)
let jsonModel = String(data: jsonData, encoding: .utf8)

print(jsonModel!)
```

### Example code to Decode a JSON String to a CoreML model

```swift
let jsonModel = "{...}"
let jsonData = Data(jsonModel.utf8)
let model = try! JSONDecoder().decode(Model.self, from: jsonData)
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
