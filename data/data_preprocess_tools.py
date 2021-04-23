import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns


def standardize_data_type(data, d_type="float32"):
    return np.array(data).astype(d_type)


def split_data(data, train_ratio=0.7, val_ratio=0.2, verbose=False):
    n = len(data)
    train_data = data[0 : int(n * train_ratio)]
    val_data = data[int(n * train_ratio) : int(n * (train_ratio + val_ratio))]
    test_data = data[int(n * (train_ratio + val_ratio)) :]

    if verbose:
        print("> check data shape")
        print("\ttrain shape: {}".format(train_data.shape))
        print("\tval shape: {}".format(val_data.shape))
        print("\ttest shape: {}".format(test_data.shape))

    return train_data, val_data, test_data


def normalize_data(data, train_data, val_data, test_data, verbose=False):
    data = pd.DataFrame(data)
    train_data = pd.DataFrame(train_data)
    val_data = pd.DataFrame(val_data)
    test_data = pd.DataFrame(test_data)

    if verbose:
        plt.figure(figsize=(12, 6))
        plt.title("before normalization")
        sns.violinplot(data=train_data)

    # normalization
    scaler = MinMaxScaler()
    scaler.fit(X=train_data)
    train_data = scaler.transform(train_data)
    val_data = scaler.transform(val_data)
    test_data = scaler.transform(test_data)

    if verbose:
        plt.figure(figsize=(12, 6))
        plt.title("after normalization")
        sns.violinplot(data=train_data)

    # standardize the data type
    train_data = standardize_data_type(train_data)
    val_data = standardize_data_type(val_data)
    test_data = standardize_data_type(test_data)

    # check is nan exist
    assert np.isnan(np.sum(train_data)) == False
    assert np.isnan(np.sum(val_data)) == False
    assert np.isnan(np.sum(test_data)) == False

    return train_data, val_data, test_data
