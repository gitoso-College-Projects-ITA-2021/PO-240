import tensorflow as tf
import numpy as np

import tensorflow.keras.models as models
import tensorflow.keras.losses as losses
import tensorflow.keras.layers as layers
import tensorflow.keras.metrics as metrics
import tensorflow.keras.optimizers as optimizers
import tensorflow.keras.activations as activations


alpha = 0.01
dropout_rate = 0.5


def custom_loss(output_len):
    weights_dict = {
        0: 1.00,
        1: 0.75,
        2: 0.60,
        3: 0.50,
        4: 0.43,
        5: 0.38,
        6: 0.33
    }
    weights = [weights_dict[i // 16] for i in range(output_len)]
    sum_weights = np.sum(weights)

    def compute_loss(Y_true, Y_pred):
        squares = tf.square(Y_true - Y_pred)
        weighted = squares * weights
        mean = tf.reduce_sum(weighted, axis=-1) / sum_weights
        return tf.sqrt(mean)
    return compute_loss


def basic_model(input_len, output_len):
    input_layer = layers.Input(shape=(input_len,))
    x = layers.Dense(8 * 1024, activation=activations.linear)(input_layer)
    x = layers.LeakyReLU(alpha)(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(2 * 1024, activation=activations.linear)(x)
    x = layers.LeakyReLU(alpha)(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(1024, activation=activations.linear)(x)
    x = layers.LeakyReLU(alpha)(x)
    x = layers.Dropout(dropout_rate)(x)
    output_layer = layers.Dense(
        output_len, activation=activations.linear)(x)
    return models.Model(inputs=input_layer, outputs=output_layer)


def convolutional_model(input_len, output_len):
    input_layer = layers.Input(shape=(input_len,))

    general_input = input_layer[:, 0:5]
    temporal_input = input_layer[:, 5:]
    temporal_input = tf.reshape(temporal_input, [-1, 553, 14, 1])

    x = layers.Conv1D(124, 5, activation='relu')(temporal_input)
    x = layers.MaxPool2D(pool_size=(54, 10))(x)
    x = layers.Flatten()(x)

    x = layers.Concatenate(axis=1)([general_input, x])
    x = layers.Dense(16*1024, activation=activations.linear)(x)
    x = layers.LeakyReLU(alpha)(x)
    x = layers.Dense(1024, activation=activations.linear)(x)
    x = layers.LeakyReLU(alpha)(x)
    output_layer = layers.Dense(
        output_len,
        activation=activations.linear
    )(x)
    return models.Model(inputs=input_layer, outputs=output_layer)
