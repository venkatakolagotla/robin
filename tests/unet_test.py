from __future__ import print_function
from robin import unet
from keras.layers import Input
import tensorflow as tf


def test_unet():
    input = Input((448, 448, 1))
    conv_layer = unet.double_conv_layer(input, 32)
    print(type(conv_layer))
    assert type(conv_layer) == tf.Tensor
