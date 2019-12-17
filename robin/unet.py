#!/usr/bin/python3

from keras.models import Model
from keras.layers.merge import concatenate
from keras.layers.normalization import BatchNormalization
from keras.layers.core import SpatialDropout2D, Activation
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from tensorflow import Tensor as tf_tensor


def double_conv_layer(inputs: tf_tensor, filter: int) -> tf_tensor:
    """Create a double convolution layer with
    mentioned inputs and filters.

    Parameters
    ----------
    inputs: tf_tensor
        input keras layer
    filter: int
        filter size/ kernal size for the tf_tensor

    Returns
    -------
    tf_tensor
        tensor with mentioned number of filters

    Example
    -------
    robin.unet.double_conv_layer(Input, 32)

    """
    conv = Conv2D(
        filter,
        (3, 3),
        padding="same",
        kernel_initializer="he_normal")(inputs)

    conv = BatchNormalization(axis=3)(conv)
    conv = Activation("relu")(conv)
    conv = Conv2D(
        filter,
        (3, 3),
        padding="same",
        kernel_initializer="he_normal")(conv)
    conv = BatchNormalization(axis=3)(conv)
    conv = Activation("relu")(conv)
    conv = SpatialDropout2D(0.1)(conv)
    return conv


def down_layer(inputs: tf_tensor, filter: int) -> tf_tensor:
    """Create downsampling layer.

    Parameters
    ----------
    inputs: tf_tensor
        input keras layer
    filter: int
        filter size/ kernal size for the tf_tensor

    Returns
    -------
    tf_tensor
        tensor with mentioned number of filters

    See Also
    --------
    double_conv_layer()

    Example
    -------
    robin.unet.down_layer(tf_tensor, 64)

    """
    conv = double_conv_layer(inputs, filter)
    pool = MaxPooling2D(pool_size=(2, 2))(conv)
    return conv, pool


def up_layer(
    inputs: tf_tensor,
    concats: tf_tensor,
    filter: int
) -> tf_tensor:
    """Create upsampling layer.

    Parameters
    ----------
    inputs: tf_tensor
        input keras layer
    concats: tf_tensor
        keras layer to concat
    filter: int
        filter size/ kernal size for the tf_tensor

    Returns
    -------
    tf_tensor
        tensor with mentioned number of filters

    See Also
    --------
    double_conv_layer()

    Example
    -------
    robin.unet.up_layer(input, pool_layer, 128)

    """
    return double_conv_layer(
        concatenate([UpSampling2D(size=(2, 2))(inputs), concats], axis=3),
        filter
    )


def unet(
    width: int = 128,
    height: int = 128,
    channels: int = 1
) -> Model:
    """Create U-net for input image/tensor size.

    Parameters
    ----------
    width : int
        width of input tensor, (default is `128`)
    height : int
        height of input tensor, (default is `128`)
    channels : int
        number of channels in image, (default is `1`)

    Returns
    -------
    Model
        keras Model with mentioned number of filters and layers

    See Also
    --------
    double_conv_layer(), down_layer(), up_layer()

    Example
    -------
    robin.unet.unet()

    """
    inputs = Input((128, 128, 1))

    # Downsampling.
    down1, pool1 = down_layer(inputs, 32)
    down2, pool2 = down_layer(pool1, 64)
    down3, pool3 = down_layer(pool2, 128)
    down4, pool4 = down_layer(pool3, 256)
    down5, pool5 = down_layer(pool4, 512)

    # Bottleneck.
    bottleneck = double_conv_layer(pool5, 1024)

    # Upsampling.
    up5 = up_layer(bottleneck, down5, 512)
    up4 = up_layer(up5, down4, 256)
    up3 = up_layer(up4, down3, 128)
    up2 = up_layer(up3, down2, 64)
    up1 = up_layer(up2, down1, 32)

    outputs = Conv2D(1, (1, 1))(up1)
    outputs = Activation("sigmoid")(outputs)

    model = Model(inputs, outputs)

    return model


if __name__ == "__main__":
    unet()
