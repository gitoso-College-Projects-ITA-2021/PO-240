import os
import sys

import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt

from dataset import prepara_dataset


input_folder = 'KaggleDatasets/RAW/'
output_folder = 'KaggleDatasets/PRE/'
filename = 'preprocessado.csv'
number_files = 12

file_names = ['201801.csv', '201802.csv', '201803.csv',
              '201804.csv', '201805.csv', '201806.csv',
              '201807.csv', '201808.csv', '201809.csv',
              '201810.csv', '201811.csv', '201812.csv']
files = []
for file_name in file_names:
    file_path = os.path.join(input_folder, file_name)
    if os.path.isfile(file_path):
        files.append(file_path)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        number_files = int(sys.argv[1])

    if len(sys.argv) > 2:
        filename = sys.argv[2]

    datasets = []
    chosen = np.random.choice(files, size=number_files, replace=False)
    print('PROCESSING:', chosen)

    for file_path in chosen:
        ds = pd.read_csv(file_path)
        datasets.append(ds)

    dataset = pd.concat(datasets, sort=False)
    dataset = prepara_dataset(dataset)
    dataset.to_csv(output_folder + filename, index=False)
