# encoding: UTF-8
import os
import pdb
import keras
import h5py
import numpy as np
import pandas as pd
from keras import backend as K
from keras.engine.topology import Layer
import tensorflow as tf
import sys
from keras.regularizers import l1
from keras.callbacks import LearningRateScheduler
import glob
import tensorflow

from keras.layers.convolutional import *
import pickle
import sklearn.metrics as Metrics
import keras.backend.tensorflow_backend as KTF
import copy

"""
vCNN class
Functions and classes used in the training process
"""


class VConv1D(Conv1D):

    '''1D Convolution layer supports changing the valid length of kernel in run time
     This keras convolution layer supports changing the valid length of kernel by
     using a mask to multiply the kernel. The value on the mask is based on the logistic function
     During the training time, The mask parameter is updated by the BP algorithm using a gradient.
     There are some parameters determine the mask as below:
         Each mask is a  matrix, there are 2 values for each kernel. As for
         the kernel in 1D sequence Detection, the kernel has the shape of (kernel_size,4,filters).
         For the i-th kernel (kernel[:,:,i]) and the corresponding mask is mask[:,:,i].
        Each kernel corresponding mask has two parameters, leftvalue and rightvalue,
        which respectively control the position of the endpoints at both ends of the kernel.

         # Argument:
             filters: the number of kernel
             kernel_init_len: the init length kernel's valid part. By default, the valid part of
                 the kernel is placed in the middle of the kernel.
             kernel_max_len: the max length of the kernel (including the valid and invalid part).
                 By default is 50
             verbose: a bool, if the message will be printed in the concole. The messages including:
                 the masks' states and lengths.
             "padding": is set to "same", because VCNN have invalid part of kernel, where the value is zero.
                 In order to prevent the edge of each sequence be ignored.
             "dataformat": is set to "channels_last" for the convenience of implementation
                 (this can be changed in future version)
             "kernel_initializer": is set to "RandomUniform", in order to calculated the IC threshold's
                 initial distribution. Also unnecessary limitation just for the convenience of implementation
             other parameters are chosen only for the implementation convenience. Can be changed in future version
             "average_IC_update": a bool variable, using average IC as threshold when updating mask edges
         # Reference:
             The algorithm is described in doc: {to fix!}
     '''

    def __init__(self, filters,
                 kernel_size,
                 strides=1,
                 padding='valid',
                 dilation_rate=1,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):

        super(Conv1D, self).__init__(
            rank=1,
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format='channels_last',
            dilation_rate=dilation_rate,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            **kwargs)
        self.input_spec = InputSpec(ndim=3)
        self.k_weights_3d_left = K.cast(0, dtype='float32')
        self.k_weights_3d_right = K.cast(0, dtype='float32')
        self.MaskSize = 0
        self.KernerShape = ()
        self.MaskFinal = 0
        self.KernelSize = 0
        self.LossKernel = K.zeros(shape=self.kernel_size + (4, self.filters))

    def build(self, input_shape):
        if self.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1
        if input_shape[channel_axis] is None:
            raise ValueError('The channel dimension of the inputs '
                             'should be defined. Found `None`.')
        input_dim = input_shape[channel_axis]
        kernel_shape = self.kernel_size + (input_dim, self.filters)
        k_weights_shape = (2,) + (1, self.filters)

        self.kernel = self.add_weight(shape=kernel_shape,
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)

        self.k_weights = self.add_weight(shape=k_weights_shape,
                                         initializer=self.kernel_initializer,
                                         name='k_weights',
                                         regularizer=self.kernel_regularizer)

        if self.use_bias:
            self.bias = self.add_weight(shape=(self.filters,),
                                        initializer=keras.initializers.Zeros(),
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None
        # Set input spec.
        self.input_spec = InputSpec(ndim=self.rank + 2,
                                    axes={channel_axis: input_dim})
        self.built = True

    def init_left(self):
        """
        Used to generate a leftmask
        :return:
        """
        K.set_floatx('float32')
        k_weights_tem_2d_left = K.arange(self.kernel.shape[0])
        k_weights_tem_2d_left = tf.expand_dims(k_weights_tem_2d_left, 1)
        k_weights_tem_3d_left = K.cast(K.repeat_elements(k_weights_tem_2d_left, self.kernel.shape[2], axis=1),
                                       dtype='float32') - self.k_weights[0, :, :]
        self.k_weights_3d_left = tf.expand_dims(k_weights_tem_3d_left, 1)

    def init_right(self):
        """
        Used to generate a rightmask
        :return:
        """
        k_weights_tem_2d_right = K.arange(self.kernel.shape[0])
        k_weights_tem_2d_right = tf.expand_dims(k_weights_tem_2d_right, 1)
        k_weights_tem_3d_right = -(K.cast(K.repeat_elements(k_weights_tem_2d_right, self.kernel.shape[2], axis=1),
                                          dtype='float32') - self.k_weights[1, :, :])
        self.k_weights_3d_right = tf.expand_dims(k_weights_tem_3d_right, 1)

    def regularzeMask(self, maskshape, slip):

        Masklevel = keras.backend.zeros(shape=maskshape)
        for i in range(slip):
            TemMatrix = K.sigmoid(self.MaskSize-float(i)/slip * maskshape[0])
            Matrix = K.repeat_elements(TemMatrix, maskshape[0], axis=0)

            MatrixOut = tf.expand_dims(Matrix, 1)
            Masklevel = Masklevel + MatrixOut
        Masklevel = Masklevel/float(slip) + 1
        return Masklevel


    def call(self, inputs):
        if self.rank == 1:
            self.init_left()
            self.init_right()
            k_weights_left = K.sigmoid(self.k_weights_3d_left)
            k_weights_right = K.sigmoid(self.k_weights_3d_right)
            MaskFinal = k_weights_left + k_weights_right - 1
            mask = K.repeat_elements(MaskFinal, 4, axis=1)
            self.MaskFinal = K.sigmoid(self.k_weights_3d_left) + K.sigmoid(self.k_weights_3d_right) - 1
            kernel = self.kernel * mask
            outputs = K.conv1d(
                inputs,
                kernel,
                strides=self.strides[0],
                padding=self.padding,
                data_format=self.data_format,
                dilation_rate=self.dilation_rate[0])
        if self.use_bias:
            outputs = K.bias_add(
                outputs,
                self.bias,
                data_format=self.data_format)
        if self.activation is not None:
            return self.activation(outputs)
        return outputs

    def get_config(self):
        config = super(Conv1D, self).get_config()
        config.pop('rank')
        config.pop('data_format')
        return config

class TrainMethod(keras.callbacks.Callback):
    """
    mask and kernel train crossover
    """
    def on_epoch_begin(self, epoch, logs={}):

        evenTrain = [self.model.layers[0].kernel, self.model.layers[0].bias]
        even_non_Train = [self.model.layers[0].k_weights]
        AllTrain = [self.model.layers[0].kernel, self.model.layers[0].bias, self.model.layers[0].k_weights]
        All_non_Train = []
        if epoch <= 10:
            self.model.layers[0].trainable_weights = evenTrain
            self.model.layers[0].non_trainable_weights = even_non_Train
        else:
            self.model.layers[0].trainable_weights = AllTrain
            self.model.layers[0].non_trainable_weights = All_non_Train
    def on_train_batch_begin(self, batch):
        """
        Assignment kernel
        """
        self.model.layers[0].LossKernel = copy.deepcopy(self.model.layers[0].kernel)


def ShanoyLoss(KernelWeights, MaskWeight, mu):
    """
    Constructing a loss function with Shannon entropy
    :param KernelWeights: kernel parameters in the model
    :param MaskWeight: mask parameters in the model
    :param mu:  coefficient for Shannon loss
    :return:
    """

    def DingYTransForm(KernelWeights):
        """
        Generate PWM
        :param KernelWeights:
        :return:
        """
        ExpArrayT = K.exp(KernelWeights * K.log(K.cast(2, dtype='float32')))
        ExpArray = K.sum(ExpArrayT, axis=1, keepdims=True)
        ExpTensor = K.repeat_elements(ExpArray, 4, axis=1)
        PWM = tf.divide(ExpArrayT, ExpTensor)

        return PWM

    def CalShanoyE(PWM):
        """
        Calculating the Shannon Entropy of PWM
        :param PWM:
        :return:
        """
        Shanoylog = -K.log(PWM) / K.log(K.cast(2, dtype='float32'))
        ShanoyE = K.sum(Shanoylog * PWM, axis=1, keepdims=True)
        ShanoyMean = tf.divide(K.sum(ShanoyE, axis=0, keepdims=True), K.cast(ShanoyE.shape[0], dtype='float32'))
        ShanoyMeanRes = K.repeat_elements(ShanoyMean, ShanoyE.shape[0], axis=0)

        return ShanoyE, ShanoyMeanRes

    def lossFunction(y_true,y_pred):
        """
        Output loss function
        :param y_true:
        :param y_pred:
        :return:
        """

        loss = keras.losses.binary_crossentropy(y_true, y_pred)
        PWM = DingYTransForm(KernelWeights)
        ShanoyE,ShanoyMeanRes = CalShanoyE(PWM)
        MaskValue = K.cast(0.25, dtype='float32') - (MaskWeight - K.cast(0.5, dtype='float32')) * (MaskWeight - K.cast(0.5, dtype='float32'))
        ShanoylossValue= K.sum((ShanoyE * MaskValue - K.cast(0.3, dtype='float32'))
                               * (ShanoyE * MaskValue - K.cast(0.3, dtype='float32'))
                               )
        loss += mu * ShanoylossValue
        return loss

    return lossFunction

class KMaxPooling(Layer):
    def __init__(self, K, mode=0, **kwargs):
        super(KMaxPooling, self).__init__(**kwargs)
        self.K = K
        self.mode = mode

    def compute_output_shape(self,input_shape):
        shape = list(input_shape)
        shape[1] = self.K
        return tuple(shape)

    def call(self,x):
        k = K.cast(self.K, dtype="int32")
        #sorted_tensor = K.sort(x, axis=1)
        #output = sorted_tensor[:, -k:, :]
        if self.mode == 0:
          output = tensorflow.nn.top_k(tensorflow.transpose(x,[0,2,1]), k)
        elif self.mode ==1:
          output = tensorflow.nn.top_k(x, k)
        else:
          print("not support this mode: ",self.mode)
        return output.values

    def get_config(self):
        config = {"pool_size": self.K}
        base_config = super(KMaxPooling, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

def get_mask(model):
    param = model.layers[0].get_weights()
    return param[1]

def get_kernel(model):
    param = model.layers[0].get_weights()
    return param[0]

def init_mask_final(model, init_len_dict, KernelLen):
    """
    Initialize the mask parameter
    :param model:
    :param init_len:The length of the initialization corresponds to the number of dict format, the corresponding length and the number of corresponding lengths
    :return:
    """
    param =model.layers[0].get_weights()
    k_weights_shape = param[1].shape
    k_weights = np.zeros(k_weights_shape)
    init_len_list = init_len_dict.keys()
    index_start = 0
    for init_len in init_len_list:
        init_num = init_len_dict[init_len]
        init_len = int(init_len)
        init_part_left = np.zeros([1, k_weights_shape[1], init_num]) + (KernelLen - init_len) / 2
        init_part_right = np.zeros((1, k_weights_shape[1], init_num))+ (KernelLen + init_len)/2
        k_weights[0,:,index_start:(index_start+init_num)] = init_part_left
        k_weights[1,:,index_start:(index_start+init_num)] = init_part_right
        index_start = index_start + init_num
    param[1] = k_weights
    model.set_weights(param)
    return model

def load_kernel_mask(model_path, conv_layer=None):
    param_file = h5py.File(model_path)
    param = param_file['model_weights']['v_conv1d_1']['v_conv1d_1']

    k_weights = param[param.keys()[1]].value

    kernel = param[param.keys()[2]].value

    mask_left_tem = np.repeat(np.arange(kernel.shape[0]).reshape(kernel.shape[0],1), 4, axis=1)
    mask_right_tem = np.repeat(np.arange(kernel.shape[0]).reshape(kernel.shape[0],1), 4, axis=1)
    mask = np.zeros(kernel.shape)
    for i in range(kernel.shape[2]):
        mask_left = np.zeros(mask_left_tem.shape)
        mask_right = np.zeros(mask_right_tem.shape)
        for j in range(mask_left_tem.shape[0]):
            for k in range(mask_left_tem.shape[1]):
                mask_left[j,k] = sigmoid(mask_left_tem[j,k] - mask[0,:,i])
                mask_right[j,k] = sigmoid(-mask_right_tem[j,k] + mask[1,:,i])
        mask[:,:,i] =mask_left + mask_right -1



    return kernel, k_weights, mask

if __name__ == '__main__':

    pass