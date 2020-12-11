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
from model import basic_model, convolutional_model, custom_loss


from tensorflow.python.client import device_lib
print("Num GPUs Available: ", len(
    tf.config.experimental.list_physical_devices('GPU')))
print(device_lib.list_local_devices())


dataset_folder = 'KaggleDatasets/RAW/'
preprocess_folder = 'KaggleDatasets/PRE/'
filename = 'KaggleDatasets/PRE/preprocessado.csv'
dataset = pd.read_csv(filename)
print(dataset.head())


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

    model = basic_model(input_shape, output_shape)
    # model = convolutional_model(input_shape, output_shape)

    opt = optimizers.Adam()
    model.compile(
        optimizer=opt,
        loss=custom_loss(output_shape),
        metrics=[metrics.mean_squared_error])
    print(model.summary())

    # Treinamento

    batch_size = 1024

    history = model.fit(
        X_train,
        Y_train,
        batch_size=batch_size,
        epochs=150,
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
