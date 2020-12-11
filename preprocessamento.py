import os
import sys

import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt

from dataset import prepara_dataset


input_folder = 'KaggleDatasets/RAW/'
output_folder = 'KaggleDatasets/PRE/'

number_files = 12
fraction = 0.5
filename = 'preprocessado.csv'

file_names = ['201701.csv', '201702.csv', '201703.csv',
              '201704.csv', '201705.csv', '201706.csv',
              '201707.csv', '201708.csv', '201709.csv',
              '201710.csv', '201711.csv', '201712.csv',
              '201801.csv', '201802.csv', '201803.csv',
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
        fraction = float(sys.argv[2])

    if len(sys.argv) > 3:
        filename = sys.argv[3]

    datasets = []
    chosen = np.random.choice(files, size=number_files, replace=False)
    print('PROCESSING:', chosen)

    for file_path in chosen:
        ds = pd.read_csv(file_path)
        ds = ds.sample(frac=.5)
        datasets.append(ds)

    dataset = pd.concat(datasets, sort=True, copy=False)
    dataset, _ = prepara_dataset(dataset)
    print(dataset.head())
    dataset.to_csv(output_folder + filename, index=False)
