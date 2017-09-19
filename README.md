# MxNet to TensorFlow converter

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