import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers


def label_condition_disc(in_shape=(28, 28, 1), n_classes=10, embedding_dim=100):  # 100
    # label input
    con_label = layers.Input(shape=(1,))
    # embedding for categorical input
    label_embedding = layers.Embedding(n_classes, embedding_dim)(con_label)
    # scale up to image dimensions with linear activation
    nodes = in_shape[0] * in_shape[1] * in_shape[2]
    label_dense = layers.Dense(nodes)(label_embedding)
    # reshape to additional channel
    label_reshape_layer = layers.Reshape((in_shape[0], in_shape[1], 1))(label_dense)
    # image input
    return con_label, label_reshape_layer


def image_disc(in_shape=(28, 28, 1)):
    inp_image = layers.Input(shape=in_shape)
    return inp_image


def make_discriminator_model():

    con_label, label_condition_output = label_condition_disc()
    inp_image_output = image_disc()
    # concat label as a channel
    merge = layers.Concatenate()([inp_image_output, label_condition_output])
    # downsample
    x = layers.Conv2D(128, (3, 3), strides=(2, 2), padding="same")(merge)
    x = layers.LeakyReLU(alpha=0.2)(x)
    # downsample
    x = layers.Conv2D(128, (3, 3), strides=(2, 2), padding="same")(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    # flatten feature maps
    flattened_out = layers.Flatten()(x)
    # dropout
    dropout = layers.Dropout(0.4)(flattened_out)
    # output
    dense_out = layers.Dense(1, activation="sigmoid")(dropout)
    # define model
    model = tf.keras.Model(
        [inp_image_output, con_label], dense_out, name="cDiscriminator"
    )
    return model


def get_discriminator_model():

    discriminator = make_discriminator_model()
    print(discriminator.summary())

    return discriminator
