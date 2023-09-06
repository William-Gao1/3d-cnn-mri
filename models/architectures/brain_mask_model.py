from models.layers.layers import downsample_block, upsample_block, output_layer, input_block, input_layer

import tensorflow as tf
import keras

def create_brain_mask_model(img_size):
    inputs = input_layer(shape=(img_size, img_size, img_size, 1))
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

    out = upsample_block(out, skip_8,
                            out_channels=320,
                            normalization='groupnorm',
                            mode=tf.estimator.ModeKeys.TRAIN)

    out = upsample_block(out, skip_16,
                            out_channels=256,
                            normalization='groupnorm',
                            mode=tf.estimator.ModeKeys.TRAIN)

    out = upsample_block(out, skip_32,
                            out_channels=128,
                            normalization='groupnorm',
                            mode=tf.estimator.ModeKeys.TRAIN)

    out = upsample_block(out, skip_64,
                            out_channels=64,
                            normalization='groupnorm',
                            mode=tf.estimator.ModeKeys.TRAIN)

    out = upsample_block(out, skip_128,
                            out_channels=32,
                            normalization='groupnorm',
                            mode=tf.estimator.ModeKeys.TRAIN)

    out = output_layer(out,
                        out_channels=2,
                        activation='softmax')
    return keras.Model(inputs=inputs, outputs=out)