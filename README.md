
# MxNet to TensorFlow converter

| **Discontinuation Notice** |
|-----------------------|
| This converter was done with specific architectures in mind when most converters were rather crunchy. Now most frameworks support [ONNX](https://github.com/onnx/onnx) format, plesase use it for transition between frameworks. |

This is small project for converting some pretrained CNN models from MxNet format to TensorFlow.

## Supported Layers

* **Activations**: ReLU
* **Batch normalization** without `use_global` flag
* **Convolution** without bias
* **Elementwise:** add
* **Flatten**
* **Fully connected**
* **Normalization:** l2
* **Pooling**: max, global pooling
* **Softmax** for output
