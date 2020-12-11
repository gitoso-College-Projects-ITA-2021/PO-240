import os

import numpy as np
import pandas as pd
import seaborn as sn
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model

from dataset import prepara_dataset
from model import basic_model, custom_loss


input_folder = 'KaggleDatasets/RAW/'
output_folder = 'KaggleDatasets/PRE/answers.csv'
model_folder = 'KaggleDatasets/Weights/final_model.hdf5'

print('Openning file:', output_folder)
answers = open(output_folder, "w")
answers.write("id,value\n")

sample = pd.read_csv(input_folder + '201701.csv', nrows=0)
output_cols = [col for col in sample.columns if col.startswith("output")]

print('Loading model from:', model_folder)
model = load_model(
    model_folder,
    custom_objects={
        # Tensorflow doesn't save the function, so it must be passed here (same name)
        'compute_loss': custom_loss(len(output_cols))
    }
)


print('Arquivos:')
files = []
for entry in os.listdir(input_folder):
    if entry.startswith("public"):
        print(entry)
        files.append(entry)


def avalia_teste(model, dataset):
    print('Iniciando Inferencia...')
    predictions = model.predict(dataset)
    print('Terminando Inferencia!')
    return predictions


def escreve_resposta(ids, predictions):
    lines = []
    for i in range(predictions.shape[0]):
        for j in range(predictions.shape[1]):
            line = '{}_{},{}\n'.format(
                ids[i],
                output_cols[j],
                predictions[i, j]
            )
            lines.append(line)
    answers.writelines(lines)


if __name__ == "__main__":
    for file_name in files:
        print('Processando:', file_name)
        input_path = os.path.join(input_folder, file_name)
        dataset = pd.read_csv(input_path)

        dataset, ids = prepara_dataset(dataset, verbose=False)
        predictions = avalia_teste(model, dataset)

        escreve_resposta(ids, predictions)
