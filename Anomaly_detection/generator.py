import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers

# label input
con_label = layers.Input(shape=(1,))
# image generator input
latent_vector = layers.Input(shape=(100,))


def label_conditioned_gen(n_classes=10, embedding_dim=100):
    # embedding for categorical input
    label_embedding = layers.Embedding(n_classes, embedding_dim)(con_label)
    # linear multiplication
    n_nodes = 7 * 7
    label_dense = layers.Dense(n_nodes)(label_embedding)
    # reshape to additional channel
    label_reshape_layer = layers.Reshape((7, 7, 1))(label_dense)
    return label_reshape_layer


def latent_gen(latent_dim=100):
    # image generator input
    n_nodes = 7 * 7 * 128
    latent_dense = layers.Dense(n_nodes)(latent_vector)
    latent_dense = layers.LeakyReLU(alpha=0.2)(latent_dense)
    latent_reshape = layers.Reshape((7, 7, 128))(latent_dense)
    return latent_reshape


def make_generator_model():
    latent_vector_output = label_conditioned_gen()
    label_output = latent_gen()

    # merge image gen and label input
    merge = layers.Concatenate()([latent_vector_output, label_output])
    # upsample to 14x14
    x = layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding="same")(merge)
    x = layers.LeakyReLU(alpha=0.2)(x)
    assert x.shape == (None, 14, 14, 128)
    # upsample to 28x28
    x = layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding="same")(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    assert x.shape == (None, 28, 28, 128)
    # output
    out_layer = layers.Conv2D(1, (7, 7), activation="tanh", padding="same")(x)
    assert out_layer.shape == (None, 28, 28, 1)
    # define model
    model = tf.keras.Model([latent_vector, con_label], out_layer, name="cGenerator")
    return model


def get_generator_model():
    generator = make_generator_model()
    print(generator.summary())
    return generator
