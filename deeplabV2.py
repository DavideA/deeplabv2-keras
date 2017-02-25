# -*- coding: utf-8 -*-
"""DeeplabV2 model for Keras.
- This model uses VGG16 for encoding.

# Reference:


- [DeepLab: Semantic Image Segmentation with Deep Convolutional Nets, Atrous Convolution,
    and Fully Connected CRFs](https://arxiv.org/pdf/1606.00915v1.pdf)

"""

import warnings

import theano.tensor as T

from keras import backend as K
from keras.models import Model
from keras.layers import Input, Convolution2D, AtrousConvolution2D, MaxPooling2D, merge, ZeroPadding2D, Dropout
from keras.layers.core import Layer, InputSpec
from keras.utils.layer_utils import convert_all_kernels_in_model
from keras.utils.data_utils import get_file
from keras.engine.topology import get_source_inputs

from utils import softmax


class BilinearUpsampling(Layer):
    '''Just a simple bilinear upsampling layer. This works only in theano.

    # Arguments
        upsampling: Integer > 0. The upsampling ratio for h and w.
        name: the name of the layer
    '''
    def __init__(self, upsampling, name='', **kwargs):
        self.name = name
        self.upsampling = upsampling
        self.channels = None

        self.input_spec = [InputSpec(ndim=4)]
        super(BilinearUpsampling, self).__init__(**kwargs)

    def build(self, input_shape):
        self.channels = input_shape[1]  # todo this works only in theano
        super(BilinearUpsampling, self).build(input_shape)

    def call(self, x, mask=None):
        output = T.nnet.abstract_conv.bilinear_upsampling(x, self.upsampling,
                                                          batch_size=None, num_input_channels=self.channels)
        return output

    def get_output_shape_for(self, input_shape):
        return None, input_shape[1], input_shape[2]*self.upsampling, input_shape[3]*self.upsampling


TH_WEIGHTS_PATH = 'http://imagelab.ing.unimore.it/files/deeplabV2_weights/deeplabV2_weights_th.h5'


def DeeplabV2(input_shape, upsampling=8, apply_softmax=True,
              weights='voc2012', input_tensor=None,
              classes=21):
    """Instantiate the DeeplabV2 architecture with VGG16 encoder,
    optionally loading weights pre-trained on VOC2012 segmentation.
    Note that pre-trained model is only available for Theano dim ordering.

    The model and the weights should be compatible with both
    TensorFlow and Theano backends.

    # Arguments
        input_shape: shape tuple. It should have exactly 3 inputs channels,
            and the axis ordering should be coherent with what specified in
            your keras.json (e.g. use (3, 512, 512) for 'th' and (512, 512, 3)
            for 'tf').
        upsampling: final front end upsampling (default is 8x).
        apply_softmax: whether to apply softmax or return logits.
        weights: one of `None` (random initialization)
            or `voc2012` (pre-training on VOC2012 segmentation).
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.

    # Returns
        A Keras model instance.
    """

    if weights not in {'voc2012', None}:
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization) or `voc2012` '
                         '(pre-training on VOC2012 segmentation).')

    if weights == 'voc2012' and classes != 21:
        raise ValueError('If using `weights` as voc2012 `classes` should be 21')

    if input_shape is None:
        raise ValueError('Please provide a valid input_shape to deeplab')

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    # Block 1
    h = ZeroPadding2D(padding=(1, 1))(img_input)
    h = Convolution2D(64, 3, 3, activation='relu', name='conv1_1')(h)
    h = ZeroPadding2D(padding=(1, 1))(h)
    h = Convolution2D(64, 3, 3, activation='relu', name='conv1_2')(h)
    h = ZeroPadding2D(padding=(1, 1))(h)
    h = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(h)

    # Block 2
    h = ZeroPadding2D(padding=(1, 1))(h)
    h = Convolution2D(128, 3, 3, activation='relu', name='conv2_1')(h)
    h = ZeroPadding2D(padding=(1, 1))(h)
    h = Convolution2D(128, 3, 3, activation='relu', name='conv2_2')(h)
    h = ZeroPadding2D(padding=(1, 1))(h)
    h = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(h)

    # Block 3
    h = ZeroPadding2D(padding=(1, 1))(h)
    h = Convolution2D(256, 3, 3, activation='relu', name='conv3_1')(h)
    h = ZeroPadding2D(padding=(1, 1))(h)
    h = Convolution2D(256, 3, 3, activation='relu', name='conv3_2')(h)
    h = ZeroPadding2D(padding=(1, 1))(h)
    h = Convolution2D(256, 3, 3, activation='relu', name='conv3_3')(h)
    h = ZeroPadding2D(padding=(1, 1))(h)
    h = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(h)

    # Block 4
    h = ZeroPadding2D(padding=(1, 1))(h)
    h = Convolution2D(512, 3, 3, activation='relu', name='conv4_1')(h)
    h = ZeroPadding2D(padding=(1, 1))(h)
    h = Convolution2D(512, 3, 3, activation='relu', name='conv4_2')(h)
    h = ZeroPadding2D(padding=(1, 1))(h)
    h = Convolution2D(512, 3, 3, activation='relu', name='conv4_3')(h)
    h = ZeroPadding2D(padding=(1, 1))(h)
    h = MaxPooling2D(pool_size=(3, 3), strides=(1, 1))(h)

    # Block 5
    h = ZeroPadding2D(padding=(2, 2))(h)
    h = AtrousConvolution2D(512, 3, 3, atrous_rate=(2, 2), activation='relu', name='conv5_1')(h)
    h = ZeroPadding2D(padding=(2, 2))(h)
    h = AtrousConvolution2D(512, 3, 3, atrous_rate=(2, 2), activation='relu', name='conv5_2')(h)
    h = ZeroPadding2D(padding=(2, 2))(h)
    h = AtrousConvolution2D(512, 3, 3, atrous_rate=(2, 2), activation='relu', name='conv5_3')(h)
    h = ZeroPadding2D(padding=(1, 1))(h)
    p5 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1))(h)

    # branching for Atrous Spatial Pyramid Pooling
    # hole = 6
    b1 = ZeroPadding2D(padding=(6, 6))(p5)
    b1 = AtrousConvolution2D(1024, 3, 3, atrous_rate=(6, 6), activation='relu', name='fc6_1')(b1)
    b1 = Dropout(0.5)(b1)
    b1 = Convolution2D(1024, 1, 1, activation='relu', name='fc7_1')(b1)
    b1 = Dropout(0.5)(b1)
    b1 = Convolution2D(21, 1, 1, activation='relu', name='fc8_voc12_1')(b1)

    # hole = 12
    b2 = ZeroPadding2D(padding=(12, 12))(p5)
    b2 = AtrousConvolution2D(1024, 3, 3, atrous_rate=(12, 12), activation='relu', name='fc6_2')(b2)
    b2 = Dropout(0.5)(b2)
    b2 = Convolution2D(1024, 1, 1, activation='relu', name='fc7_2')(b2)
    b2 = Dropout(0.5)(b2)
    b2 = Convolution2D(21, 1, 1, activation='relu', name='fc8_voc12_2')(b2)

    # hole = 18
    b3 = ZeroPadding2D(padding=(18, 18))(p5)
    b3 = AtrousConvolution2D(1024, 3, 3, atrous_rate=(18, 18), activation='relu', name='fc6_3')(b3)
    b3 = Dropout(0.5)(b3)
    b3 = Convolution2D(1024, 1, 1, activation='relu', name='fc7_3')(b3)
    b3 = Dropout(0.5)(b3)
    b3 = Convolution2D(21, 1, 1, activation='relu', name='fc8_voc12_3')(b3)

    # hole = 24
    b4 = ZeroPadding2D(padding=(24, 24))(p5)
    b4 = AtrousConvolution2D(1024, 3, 3, atrous_rate=(24, 24), activation='relu', name='fc6_4')(b4)
    b4 = Dropout(0.5)(b4)
    b4 = Convolution2D(1024, 1, 1, activation='relu', name='fc7_4')(b4)
    b4 = Dropout(0.5)(b4)
    b4 = Convolution2D(21, 1, 1, activation='relu', name='fc8_voc12_4')(b4)

    s = merge([b1, b2, b3, b4], mode='sum')
    logits = BilinearUpsampling(upsampling=upsampling)(s)

    if apply_softmax:
        out = softmax(logits)
    else:
        out = logits

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input

    # Create model.
    model = Model(inputs, out, name='deeplabV2')

    # load weights
    if weights == 'voc2012':
        if K.image_dim_ordering() == 'th':
            weights_path = get_file('deeplabV2_weights_th.h5',
                                    TH_WEIGHTS_PATH,
                                    cache_subdir='models')

            model.load_weights(weights_path)

            if K.backend() == 'tensorflow':
                warnings.warn('You are using the TensorFlow backend, yet you '
                              'are using the Theano '
                              'image dimension ordering convention '
                              '(`image_dim_ordering="th"`). '
                              'For best performance, set '
                              '`image_dim_ordering="tf"` in '
                              'your Keras config '
                              'at ~/.keras/keras.json.')
                convert_all_kernels_in_model(model)
        else:
            raise NotImplementedError('Pretrained DeeplabV2 model is not available for'
                                      'voc2012 dataset and tensorflow dim ordering')

    return model
