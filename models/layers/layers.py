# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

""" High level definition of layers for model construction """
import tensorflow as tf


def _normalization(inputs, name, mode):
    """ Choose a normalization layer

    :param inputs: Input node from the graph
    :param name: Name of layer
    :param mode: Estimator's execution mode
    :return: Normalized output
    """
    training = mode == tf.estimator.ModeKeys.TRAIN

    if name == 'groupnorm':
        return tf.keras.layers.GroupNormalization(
                                            groups=16
                                            )(inputs, training=training)

    if name == 'batchnorm':
        return tf.keras.layers.BatchNormalization(axis=-1,
                                                  trainable=True,
                                                  virtual_batch_size=None)(inputs, training=training)
    if name == 'none':
        return inputs

    raise ValueError('Invalid normalization layer')


def _activation(out, activation):
    """ Choose an activation layer

    :param out: Input node from the graph
    :param activation: Name of layer
    :return: Activation output
    """
    if activation == 'relu':
        return tf.nn.relu(out)
    if activation == 'leaky_relu':
        return tf.nn.leaky_relu(out, alpha=0.01)
    if activation == 'sigmoid':
        return tf.nn.sigmoid(out)
    if activation == 'softmax':
        return tf.nn.softmax(out, axis=-1)
    if activation == 'none':
        return out

    raise ValueError("Unknown activation {}".format(activation))


def convolution(inputs,  # pylint: disable=R0913
                out_channels,
                kernel_size=3,
                stride=1,
                mode=tf.estimator.ModeKeys.TRAIN,
                normalization='batchnorm',
                activation='leaky_relu',
                transpose=False):
    """ Create a convolution layer

    :param inputs: Input node from graph
    :param out_channels: Output number of channels
    :param kernel_size: Size of the kernel
    :param stride: Stride of the kernel
    :param mode: Estimator's execution mode
    :param normalization: Name of the normalization layer
    :param activation: Name of the activation layer
    :param transpose: Select between regular and transposed convolution
    :return: Convolution output
    """
    if transpose:
        conv = tf.keras.layers.Conv3DTranspose
    else:
        conv = tf.keras.layers.Conv3D
    regularizer = None  # tf.keras.regularizers.l2(1e-5)

    use_bias = normalization == "none"
    inputs = conv(filters=out_channels,
                  kernel_size=kernel_size,
                  strides=stride,
                  activation=None,
                  padding='same',
                  data_format='channels_last',
                  kernel_initializer="he_normal",
                  kernel_regularizer=regularizer,
                  bias_initializer=tf.zeros_initializer(),
                  bias_regularizer=regularizer,
                  use_bias=use_bias)(inputs)

    inputs = _normalization(inputs, normalization, mode)

    return _activation(inputs, activation)


def upsample_block(inputs, skip_connection, out_channels, normalization, mode):
    """ Create a block for upsampling

    :param inputs: Input node from the graph
    :param skip_connection: Choose whether or not to use skip connection
    :param out_channels: Number of output channels
    :param normalization: Name of the normalizaiton layer
    :param mode: Estimator's execution mode
    :return: Output from the upsample block
    """
    inputs = convolution(inputs, kernel_size=2, out_channels=out_channels, stride=2,
                         normalization='none', activation='none', transpose=True)
    inputs = tf.keras.layers.Concatenate(axis=-1)([inputs, skip_connection])

    inputs = convolution(inputs, out_channels=out_channels, normalization=normalization, mode=mode)
    inputs = convolution(inputs, out_channels=out_channels, normalization=normalization, mode=mode)
    return inputs


def input_block(inputs, out_channels, normalization, mode):
    """ Create the input block

    :param inputs: Input node from the graph
    :param out_channels: Number of output channels
    :param normalization:  Name of the normalization layer
    :param mode: Estimator's execution mode
    :return: Output from the input block
    """
    inputs = convolution(inputs, out_channels=out_channels, normalization=normalization, mode=mode)
    inputs = convolution(inputs, out_channels=out_channels, normalization=normalization, mode=mode)
    return inputs


def downsample_block(inputs, out_channels, normalization, mode):
    """ Create a downsample block

    :param inputs: Input node from the graph
    :param out_channels: Number of output channels
    :param normalization:  Name of the normalization layer
    :param mode: Estimator's execution mode
    :return: Output from the downsample block
    """
    inputs = convolution(inputs, out_channels=out_channels, normalization=normalization, mode=mode, stride=2)
    return convolution(inputs, out_channels=out_channels, normalization=normalization, mode=mode)


def output_layer(inputs, out_channels, activation):
    """ Create the output layer

    :param inputs: Input node from the graph
    :param out_channels: Number of output channels
    :param activation:  Name of the activation layer
    :return: Output from the output block
    """
    return convolution(inputs, out_channels=out_channels, kernel_size=3, normalization='none', activation=activation)

def input_layer(shape):
    """ Create the intput layer

    :param inputs: Input node from the graph
    :param out_channels: Number of output channels
    :param activation:  Name of the activation layer
    :return: Output from the output block
    """
    return tf.keras.layers.Input(shape=shape)


def classification_layer(inputs, out_channels):
    """Creates a block that reduces a 3D layer to a 1D layer

    Args:
        inputs: Input node
        outchannels: Number of output channels

    Returns:
        node: Output from classification layer
    """
    inputs = tf.keras.layers.GlobalAveragePooling3D()(inputs)
    inputs = tf.keras.layers.Dense(units=512, activation="relu")(inputs)
    inputs = tf.keras.layers.Dropout(0.2)(inputs)
    inputs = tf.keras.layers.Dense(units=out_channels, activation="sigmoid")(inputs)
    
    return inputs
