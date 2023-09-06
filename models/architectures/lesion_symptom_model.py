from models.layers.layers import downsample_block, classification_layer, input_block, input_layer

import tensorflow as tf
import keras

def create_lesion_symptom_img_model(img_size, n_channels = 1):
    inputs = input_layer(shape=(img_size, img_size, img_size, n_channels))
    skip_128 = input_block(inputs=inputs,
                            out_channels=32,
                            normalization='groupnorm',
                            mode=tf.estimator.ModeKeys.TRAIN)

    skip_64 = downsample_block(inputs=skip_128,
                                out_channels=64,
                                normalization='groupnorm',
                                mode=tf.estimator.ModeKeys.TRAIN)

    skip_32 = downsample_block(inputs=skip_64,
                                out_channels=128,
                                normalization='groupnorm',
                                mode=tf.estimator.ModeKeys.TRAIN)

    skip_16 = downsample_block(inputs=skip_32,
                                out_channels=256,
                                normalization='groupnorm',
                                mode=tf.estimator.ModeKeys.TRAIN)

    skip_8 = downsample_block(inputs=skip_16,
                                out_channels=320,
                                normalization='groupnorm',
                                mode=tf.estimator.ModeKeys.TRAIN)

    out = downsample_block(inputs=skip_8,
                            out_channels=320,
                            normalization='groupnorm',
                            mode=tf.estimator.ModeKeys.TRAIN)
    out = classification_layer(out, 1)
    return keras.Model(inputs=inputs, outputs=out)