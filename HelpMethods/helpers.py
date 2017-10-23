# -*- coding: utf-8 -*-
"""some helper functions for project 1."""
import csv
import numpy as np

#Method to use if we need to standardize the data
def standardize(x):
    """Standardize the original data set."""
    a = len(x[0])
    mean_x = np.zeros(a)
    std_x = np.zeros(a)
    for i in range(a):
        mean_x[i] = np.mean(x[:,i])
        x[:,i] = x[:,i] - mean_x[i]
        std_x[i] = np.std(x[:,i])
        x[:,i] = x[:,i] / std_x[i]
    return x, mean_x, std_x

def clean_matrice(x):
    # Mean-adjust entries with -999. We adjust them to the mean of the column (without the -999 values)
    means = np.zeros(len(x[0]))
    number_of_entries = np.zeros(len(x[0]))
    sums = np.zeros(len(x[0]))

    # Summing over all values per column that are not -999
    for j in range(len(means)):
        for i in range(len(x)):
            if (x[i][j] != -999):
                sums[j] += x[i][j]
                number_of_entries[j] += 1

    #Calculating mean value for every column of x
    for i in range(len(means)):
        means[i] = (sums[i] / number_of_entries[i])

    # Replacing -999 with the mean of the column
    for j in range(len(means)):
        for i in range(len(x)):
            if (x[i][j] == -999):
                x[i][j] = means[j]

    return x

def load_csv_data(data_path, sub_sample=False):
    """Loads data and returns y (class labels), tX (features) and ids (event ids)"""
    y = np.genfromtxt(data_path, delimiter=",", skip_header=1, dtype=str, usecols=1)
    x = np.genfromtxt(data_path, delimiter=",", skip_header=1)
    ids = x[:, 0].astype(np.int)
    input_data = x[:, 2:]

    # convert class labels from strings to binary (-1,1)
    yb = np.ones(len(y))
    yb[np.where(y == 'b')] = -1

    # sub-sample
    if sub_sample:
        yb = yb[::50]
        input_data = input_data[::50]
        ids = ids[::50]

    return yb, input_data, ids


def predict_labels(weights, data):
    """Generates class predictions given weights, and a test data matrix"""
    y_pred = np.dot(data, weights)
    y_pred[np.where(y_pred <= 0)] = -1
    y_pred[np.where(y_pred > 0)] = 1

    return y_pred


def create_csv_submission(ids, y_pred, name):
    """
    Creates an output file in csv format for submission to kaggle
    Arguments: ids (event ids associated with each prediction)
               y_pred (predicted class labels)
               name (string name of .csv output file to be created)
    """
    with open(name, 'w') as csvfile:
        fieldnames = ['Id', 'Prediction']
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({'Id': int(r1), 'Prediction': int(r2)})