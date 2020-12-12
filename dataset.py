import pandas as pd
import numpy as np

from datetime import datetime


# def prepara_dataset(dataset, verbose=True):
#     ids = dataset['id'].copy()
#     dataset = dataset.drop(['id'], axis=1)
#     dataset['day'] = dataset['date'].apply(
#         lambda x: datetime.strptime(x, "%Y-%m-%d").weekday() / 6)
#     dataset['week'] = dataset['date'].apply(
#         lambda x: datetime.strptime(x, "%Y-%m-%d").isocalendar()[1] / 53)
#     dataset = dataset.drop(['date'], axis=1)

#     for column in dataset:
#         num_na = dataset[column].isnull().sum()
#         if num_na > 0:
#             placeholder = dataset[column].median()
#             if np.isnan(placeholder):
#                 if verbose:
#                     print(
#                         'Empty column {:12s}... Filling with zero'.format(column))
#                 placeholder = 0
#             dataset[column] = dataset[column].fillna(placeholder)

#     for column in dataset:
#         num_na = dataset[column].isnull().sum()
#         assert num_na == 0
#     return dataset, ids


def prepara_dataset(dataset, verbose=True):
    ids = dataset['id'].copy()
    dataset = dataset.drop(['id'], axis=1)
    dataset['day'] = dataset['date'].apply(
        lambda x: datetime.strptime(x, "%Y-%m-%d").weekday() / 6)
    dataset['week'] = dataset['date'].apply(
        lambda x: datetime.strptime(x, "%Y-%m-%d").isocalendar()[1] / 53)
    dataset = dataset.drop(['date'], axis=1)

    # Parâmetros
    x = 55  # x%
    y = 35  # y%
    z = 20  # z%

    total = len(dataset)
    delete_list = []
    median_list = []

    variance = dataset.var()
    mean = dataset.mean()

    for column in dataset:
        num_na = dataset[column].isnull().sum()
        per_na = num_na * 100 / total
        aux = variance[column] * 100 / mean[column]
        if per_na > x:
            delete_list.append(column)
        elif per_na > y and aux > z:
            delete_list.append(column)
        elif num_na > 0:
            median_list.append(column)

    print('\n\n\n\',)
    dataset = dataset.drop(delete_list, axis=1)

    median = dataset.median()
    for column in median_list:
        dataset[column] = dataset[column].fillna(median[column])

    for column in dataset:
        num_na = dataset[column].isnull().sum()
        assert num_na == 0
    return dataset, ids


def separa_dataset(dataset, com_date=True):
    cols_input = []
    cols_output = []
    for col in dataset.columns:
        if col.startswith("input") or ((col.startswith('day') or col.startswith('week')) and com_date):
            cols_input.append(col)
        elif col.startswith("output"):
            cols_output.append(col)
        else:
            print("Unexpected column name:", col)
            continue

    inputs = dataset[cols_input]
    outputs = dataset[cols_output]
    return inputs, outputs
