import numpy as np

from keras.layers import Activation, Permute, Reshape


# voc12 palette
palette = np.array([[0, 0, 0],
                    [128, 0, 0],
                    [0, 128, 0],
                    [128, 128, 0],
                    [0, 0, 128],
                    [128, 0, 128],
                    [0, 128, 128],
                    [128, 128, 128],
                    [64, 0, 0],
                    [192, 0, 0],
                    [64, 128, 0],
                    [192, 128, 0],
                    [64, 0, 128],
                    [192, 0, 128],
                    [64, 128, 128],
                    [192, 128, 128],
                    [0, 64, 0],
                    [128, 64, 0],
                    [0, 192, 0],
                    [128, 192, 0],
                    [0, 64, 128]], dtype='uint8')


def softmax(x, restore_shape=True):
    """
    Softmax activation for a tensor x. No need to unroll the input first.
    :param x: x is a tensor with shape (None, channels, h, w)
    :param restore_shape: if False, output is returned unrolled (None, h * w, channels)
    :return: softmax activation of tensor x
    """
    _, c, h, w = x._keras_shape
    x = Permute(dims=(2, 3, 1))(x)
    x = Reshape(target_shape=(h * w, c))(x)

    x = Activation('softmax')(x)

    if restore_shape:
        x = Reshape(target_shape=(h, w, c))(x)
        x = Permute(dims=(3, 1, 2))(x)

    return x