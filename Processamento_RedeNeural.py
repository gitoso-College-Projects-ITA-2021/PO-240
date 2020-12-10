import os

from datetime import datetime
import pandas as pd
import seaborn as sn
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

import tensorflow.keras.models as models
import tensorflow.keras.losses as losses
import tensorflow.keras.layers as layers
import tensorflow.keras.metrics as metrics
import tensorflow.keras.optimizers as optimizers
import tensorflow.keras.activations as activations

from dataset import separa_dataset


from tensorflow.python.client import device_lib
print("Num GPUs Available: ", len(
    tf.config.experimental.list_physical_devices('GPU')))
print(device_lib.list_local_devices())


dataset_folder = 'KaggleDatasets/RAW/'
preprocess_folder = 'KaggleDatasets/PRE/'
filename = 'KaggleDatasets/PRE/preprocessado_small.csv'
dataset = pd.read_csv(filename)
print(dataset.head())


def custom_loss(Y_true, Y_pred):
    weights_dict = {
        0: 1.00,
        1: 0.75,
        2: 0.60,
        3: 0.50,
        4: 0.43,
        5: 0.38,
        6: 0.33
    }
    weights = []


def basic_model():
    input_layer = layers.Input(shape=(input_shape,))
    x = layers.Dense(1024, activation=activations.linear)(input_layer)
    x = layers.LeakyReLU(alpha)(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(1024, activation=activations.linear)(x)
    x = layers.LeakyReLU(alpha)(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(1024, activation=activations.linear)(x)
    x = layers.LeakyReLU(alpha)(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(1024, activation=activations.linear)(x)
    x = layers.LeakyReLU(alpha)(x)
    x = layers.Dropout(dropout_rate)(x)
    output_layer = layers.Dense(
        output_shape, activation=activations.linear)(x)
    return models.Model(inputs=input_layer, outputs=output_layer)


def convolutional_model():
    input_layer = layers.Input(shape=(7747,))

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
        output_shape,
        activation=activations.linear
    )(x)
    return models.Model(inputs=input_layer, outputs=output_layer)


if __name__ == "__main__":
    inputs, outputs = separa_dataset(dataset)

    print('Inputs')
    print(inputs.head())

    print('Outputs')
    print(outputs.head())

    X_train, X_val, Y_train, Y_val = train_test_split(
        inputs, outputs, test_size=0.30, random_state=57)
    # Sung, sung, háa, pbaet never gets old

    print('# of training images:', X_train.shape)
    print('# of cross-validation images:', X_val.shape)

    # ### Criando o Modelo

    alpha = 0.01
    dropout_rate = 0.5

    input_shape = X_train.shape[1]
    output_shape = Y_train.shape[1]

    model = basic_model()
    # model = convolutional_model()

    opt = optimizers.Adam()
    model.compile(
        optimizer=opt,
        loss=losses.mean_squared_error,
        metrics=[metrics.mean_squared_error])
    print(model.summary())

    # Treinamento

    batch_size = 1024

    history = model.fit(
        X_train,
        Y_train,
        batch_size=batch_size,
        epochs=100,
        # We pass some validation for
        # monitoring validation loss and metrics
        # at the end of each epoch
        validation_data=(X_val, Y_val),
        validation_batch_size=batch_size,
    )

    fig_format = 'png'

    # Plotting cost function convergence
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Cost Function Convergence')
    plt.legend(['Treinamento', 'Validação'])
    plt.grid()
    plt.savefig('convergence_imitation' + '.' + fig_format, format=fig_format)

    # model.save_weights(filepath='./KaggleDatasets/Weights/final_model.hdf5')
    model.save(filepath='./KaggleDatasets/Weights/final_model.hdf5')

    #load_model(model_name + '.hdf5')
