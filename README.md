# vConv

vConv is designed as a variant of the canonical convolutional kernel which could adjust the kernel length adaptively during the training. For more information, see the manuscript [Identifying complex sequence patterns with a variable-convolutional layer effectively and efficiently](https://doi.org/10.1101/508242). A repository for reproducing figures and tables in the manuscript is accessible at [https://github.com/gao-lab/vConv-Figures_and_Tables].

The current Class VConv1D is implemented based on the [original Keras Conv1D layer](https://keras.io/api/layers/convolution_layers/convolution1d/).

## Prerequisites

### Software

- Python 2 and its packages:
  - numpy
  - h5py
  - pandas
  - seaborn
  - scipy
  - keras (version 2.2.4)
  - tensorflow (version 1.3.0)
  - sklearn

Alternatively, if you want to guarantee working versions of each dependency, you can install via a fully pre-specified environment.
```{bash}
conda env create -f environment_vConv.yml
```

# Quick start

The class is implemented at [./corecode/vConv_core.py](/corecode/vConv_core.py).

As demonstrated below, VConv1D can be added in the same way as Conv1D layers to the model.

When using the layer, you need pass at least 2 parameters: filters (the number of filters/kernels in the layer) and kernel_init_len (the initial unmasked length of each filter/kernel). In addition (and as identical to Conv1D and most other keras layers), parameter "input_shape" is required if this is the first layer.

```{python}
from vConv_core import VConv1D

model_tmp = keras.models.Sequential()
model_template.add(VConv1D(
        input_shape=input_shape,
        kernel_size=(kernel_init_len),
        filters=number_of_kernel,
        padding='same',
        strides=1))
```

# Run demo code

Clone this repository and run demo codes under the directory ./demo/:

```{bash}
python Demo.py
```
This script trains a vConv-based network with a vConv layer, a Maxpooling layer, two dense layers to classify the sequence data.
It will output the accuracy and model parameter in "./demo/Output/test/vCNN".

# Notes

Although the kernel-to-PWM transformation in vConv’s MSL assumes that the input sequence is one-hot encoded (Ding, et al., 2018), all types of layers that can precede or succeed the convolutional layer apply to vConv as well in practice, and thus one can always try to improve a given CNN model by replacing any of its convolutional layer(s) accepting arbitrary real-value input sequences with vConv. Violation of the assumption above can be avoided by either (1) setting lambda to 0 to disable MSL, or (2) assuming that the input sequence is a weighted sum of one-hot encoded sequences.
