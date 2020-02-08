public struct InnerProduct : Layer {
    public static let type = LayerType.innerProduct
    public let name: String
    public let input: [String]
    public let output: [String]
    public let weights: [Float]
    public let bias: [Float]

    public let inputChannels: UInt
    public let outputChannels: UInt
    public let updatable: Bool
}

// struct CoreML_Specification_InnerProductLayerParams {
//   // SwiftProtobuf.Message conformance is added in an extension below. See the
//   // `Message` and `Message+*Additions` files in the SwiftProtobuf library for
//   // methods supported on all messages.

//   //// Input size: C_in.
//   var inputChannels: UInt64 {
//     get {return _storage._inputChannels}
//     set {_uniqueStorage()._inputChannels = newValue}
//   }

//   //// Output size: C_out.
//   var outputChannels: UInt64 {
//     get {return _storage._outputChannels}
//     set {_uniqueStorage()._outputChannels = newValue}
//   }

//   //// Whether a bias is added or not.
//   var hasBias_p: Bool {
//     get {return _storage._hasBias_p}
//     set {_uniqueStorage()._hasBias_p = newValue}
//   }

//   //// Weight matrix [C_out, C_in].
//   var weights: CoreML_Specification_WeightParams {
//     get {return _storage._weights ?? CoreML_Specification_WeightParams()}
//     set {_uniqueStorage()._weights = newValue}
//   }
//   /// Returns true if `weights` has been explicitly set.
//   var hasWeights: Bool {return _storage._weights != nil}
//   /// Clears the value of `weights`. Subsequent reads from it will return its default value.
//   mutating func clearWeights() {_uniqueStorage()._weights = nil}

//   //// Bias vector [C_out].
//   var bias: CoreML_Specification_WeightParams {
//     get {return _storage._bias ?? CoreML_Specification_WeightParams()}
//     set {_uniqueStorage()._bias = newValue}
//   }
//   /// Returns true if `bias` has been explicitly set.
//   var hasBias: Bool {return _storage._bias != nil}
//   /// Clears the value of `bias`. Subsequent reads from it will return its default value.
//   mutating func clearBias() {_uniqueStorage()._bias = nil}

//   var unknownFields = SwiftProtobuf.UnknownStorage()

//   init() {}

//   fileprivate var _storage = _StorageClass.defaultInstance
// }


public struct Convolution : Layer {
    public static let type = LayerType.convolution
    public let name: String
    public let input: [String]
    public let output: [String]
    public let weights: [Float]
    public let bias: [Float]
}

// struct CoreML_Specification_ConvolutionLayerParams {
//   // SwiftProtobuf.Message conformance is added in an extension below. See the
//   // `Message` and `Message+*Additions` files in the SwiftProtobuf library for
//   // methods supported on all messages.

//   ///*
//   /// The number of kernels.
//   /// Same as ``C_out`` used in the layer description.
//   var outputChannels: UInt64 {
//     get {return _storage._outputChannels}
//     set {_uniqueStorage()._outputChannels = newValue}
//   }

//   ///*
//   /// Channel dimension of the kernels.
//   /// Must be equal to ``inputChannels / nGroups``, if isDeconvolution == False
//   /// Must be equal to ``inputChannels``, if isDeconvolution == True
//   var kernelChannels: UInt64 {
//     get {return _storage._kernelChannels}
//     set {_uniqueStorage()._kernelChannels = newValue}
//   }

//   ///*
//   /// Group convolution, i.e. weight reuse along channel axis.
//   /// Input and kernels are divided into g groups
//   /// and convolution / deconvolution is applied within the groups independently.
//   /// If not set or 0, it is set to the default value 1.
//   var nGroups: UInt64 {
//     get {return _storage._nGroups}
//     set {_uniqueStorage()._nGroups = newValue}
//   }

//   ///*
//   /// Must be length 2 in the order ``[H, W]``.
//   /// If not set, default value ``[3, 3]`` is used.
//   var kernelSize: [UInt64] {
//     get {return _storage._kernelSize}
//     set {_uniqueStorage()._kernelSize = newValue}
//   }

//   ///*
//   /// Must be length 2 in the order ``[H, W]``.
//   /// If not set, default value ``[1, 1]`` is used.
//   var stride: [UInt64] {
//     get {return _storage._stride}
//     set {_uniqueStorage()._stride = newValue}
//   }

//   ///*
//   /// Must be length 2 in order ``[H, W]``.
//   /// If not set, default value ``[1, 1]`` is used.
//   /// It is ignored if ``isDeconvolution == true``.
//   var dilationFactor: [UInt64] {
//     get {return _storage._dilationFactor}
//     set {_uniqueStorage()._dilationFactor = newValue}
//   }

//   ///*
//   /// The type of padding.
//   var convolutionPaddingType: OneOf_ConvolutionPaddingType? {
//     get {return _storage._convolutionPaddingType}
//     set {_uniqueStorage()._convolutionPaddingType = newValue}
//   }

//   var valid: CoreML_Specification_ValidPadding {
//     get {
//       if case .valid(let v)? = _storage._convolutionPaddingType {return v}
//       return CoreML_Specification_ValidPadding()
//     }
//     set {_uniqueStorage()._convolutionPaddingType = .valid(newValue)}
//   }

//   var same: CoreML_Specification_SamePadding {
//     get {
//       if case .same(let v)? = _storage._convolutionPaddingType {return v}
//       return CoreML_Specification_SamePadding()
//     }
//     set {_uniqueStorage()._convolutionPaddingType = .same(newValue)}
//   }

//   ///*
//   /// Flag to specify whether it is a deconvolution layer.
//   var isDeconvolution: Bool {
//     get {return _storage._isDeconvolution}
//     set {_uniqueStorage()._isDeconvolution = newValue}
//   }

//   ///*
//   /// Flag to specify whether a bias is to be added or not.
//   var hasBias_p: Bool {
//     get {return _storage._hasBias_p}
//     set {_uniqueStorage()._hasBias_p = newValue}
//   }

//   ///*
//   /// Weights associated with this layer.
//   /// If convolution (``isDeconvolution == false``), weights have the shape
//   /// ``[outputChannels, kernelChannels, kernelHeight, kernelWidth]``, where kernelChannels == inputChannels / nGroups
//   /// If deconvolution (``isDeconvolution == true``) weights have the shape
//   /// ``[kernelChannels, outputChannels / nGroups, kernelHeight, kernelWidth]``, where kernelChannels == inputChannels
//   var weights: CoreML_Specification_WeightParams {
//     get {return _storage._weights ?? CoreML_Specification_WeightParams()}
//     set {_uniqueStorage()._weights = newValue}
//   }
//   /// Returns true if `weights` has been explicitly set.
//   var hasWeights: Bool {return _storage._weights != nil}
//   /// Clears the value of `weights`. Subsequent reads from it will return its default value.
//   mutating func clearWeights() {_uniqueStorage()._weights = nil}

//   //// Must be of size [outputChannels].
//   var bias: CoreML_Specification_WeightParams {
//     get {return _storage._bias ?? CoreML_Specification_WeightParams()}
//     set {_uniqueStorage()._bias = newValue}
//   }
//   /// Returns true if `bias` has been explicitly set.
//   var hasBias: Bool {return _storage._bias != nil}
//   /// Clears the value of `bias`. Subsequent reads from it will return its default value.
//   mutating func clearBias() {_uniqueStorage()._bias = nil}

//   ///*
//   /// The output shape, which has length 2 ``[H_out, W_out]``.
//   /// This is used only for deconvolution (``isDeconvolution == true``).
//   /// If not set, the deconvolution output shape is calculated
//   /// based on ``ConvolutionPaddingType``.
//   var outputShape: [UInt64] {
//     get {return _storage._outputShape}
//     set {_uniqueStorage()._outputShape = newValue}
//   }

//   var unknownFields = SwiftProtobuf.UnknownStorage()

//   ///*
//   /// The type of padding.
//   enum OneOf_ConvolutionPaddingType: Equatable {
//     case valid(CoreML_Specification_ValidPadding)
//     case same(CoreML_Specification_SamePadding)

//   #if !swift(>=4.1)
//     static func ==(lhs: CoreML_Specification_ConvolutionLayerParams.OneOf_ConvolutionPaddingType, rhs: CoreML_Specification_ConvolutionLayerParams.OneOf_ConvolutionPaddingType) -> Bool {
//       switch (lhs, rhs) {
//       case (.valid(let l), .valid(let r)): return l == r
//       case (.same(let l), .same(let r)): return l == r
//       default: return false
//       }
//     }
//   #endif
//   }

//   init() {}

//   fileprivate var _storage = _StorageClass.defaultInstance
// }




public struct Pooling : Layer {
    public static let type = LayerType.pooling
    public let name: String
    public let input: [String]
    public let output: [String]
    public let weights: [Float]
    public let bias: [Float]
}

public struct Embedding : Layer {
    public static let type = LayerType.embedding
    public let name: String
    public let input: [String]
    public let output: [String]
    public let weights: [Float]
    public let bias: [Float]
}

public struct SimpleRecurrent : Layer {
    public static let type = LayerType.simpleRecurrent
    public let name: String
    public let input: [String]
    public let output: [String]
    public let weights: [Float]
    public let bias: [Float]
}

public struct Gru : Layer {
    public static let type = LayerType.gru
    public let name: String
    public let input: [String]
    public let output: [String]
    public let weights: [Float]
    public let bias: [Float]
}

public struct UniDirectionalLstm : Layer {
    public static let type = LayerType.uniDirectionalLstm
    public let name: String
    public let input: [String]
    public let output: [String]
    public let weights: [Float]
    public let bias: [Float]
}

public struct BiDirectionalLstm : Layer {
    public static let type = LayerType.biDirectionalLstm
    public let name: String
    public let input: [String]
    public let output: [String]
    public let weights: [Float]
    public let bias: [Float]
}
