"""
Some helper methods used in our project.

This file is divided into 4 sections:

1. Helper methods for the regression functions (func 1-4). Begins on line 20
2. Helper methods for the classification functions (func 5-6). Begins on line 68
3. Model selection methods (Split data and Cross Validation). Begins on line 96
4. Other helper methods. Begins on line 141
"""

import numpy as np
import math

from Implementations import implementations




# HELPER METHODS FOR THE REGRESSION FUNCTIONS (FUNC 1-4)

# function to compute the Mean Squared Error
def compute_mse(y, tx, w):
    e = y - np.dot(tx,w)
    squared = e**2
    mse = np.mean(squared)/2
    return mse

def compute_rmse(y, tx, w):
    mse = compute_mse(y, tx, w)
    return ((mse*2)**0.5)

# function to compute the Mean Absolute Error
def compute_mae(y,tx,w):
    e = abs(y - np.dot(tx,w))
    mae = np.mean(e)/2
    return mae


# computes the gradient, needed for Gradient Descent and Stochastic Gradient Descent (FUNC 3 and 4)
def compute_gradient(y, tx, w):
    e = y - np.dot(tx, w)
    gradient = -(np.dot(tx.T, e))/len(e)
    return gradient

# helper method for Stochastic Gradient Descent
def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    # Generate a minibatch iterator for a dataset.
    # Takes as input two iterables (here the output desired values 'y' and the input data 'tx').
    # Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    # Data can be randomly shuffled to avoid ordering in the original data messing with
    # the randomness of the minibatches.
    data_size = len(y)
    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]


# HELPER METHODS FOR THE CLASSIFICATION FUNCTIONS (FUNC 5 AND 6)

# apply sigmoid function on t
def sigmoid(t):
    return np.exp(-np.logaddexp(0, -t))

# compute the cost by negative log likelihood
def compute_log_likelihood(y, tx, w):
    vector = np.dot(tx, w)
    first = np.logaddexp(0, vector)
    second = y * vector
    log_likelihood = np.sum(first - second)
    return log_likelihood

# compute the gradient of loss
def compute_log_gradient(y, tx, w):
    vector = np.dot(tx, w)
    calc = sigmoid(vector) - y
    gradient = tx.T.dot(calc)
    return gradient

# compute the loss and gradient of the Regularized Logistic Regression method (FUNCTION 6)
def penalized_logistic_regression(y, tx, w, lambda_):
    loss = compute_log_likelihood(y, tx, w)
    gradient = compute_log_gradient(y, tx, w) + 2 * lambda_ * w
    return loss, gradient


# MODEL SELECTION METHODS (Split Data and Cross Validation)

def split_data(x, y, ratio, seed=1):
    np.random.seed(seed)
    shuffled_indices = np.random.permutation(len(x))
    training_data_x = x[shuffled_indices[: math.floor(len(x) * ratio)]]
    training_data_y = y[shuffled_indices[: math.floor(len(x) * ratio)]]
    test_data_x = x[shuffled_indices[math.floor(len(x) * ratio):]]
    test_data_y = y[shuffled_indices[math.floor(len(x) * ratio):]]
    return training_data_x, training_data_y, test_data_x, test_data_y

# used in cross validation. Builds k_indices to the k_fold cross validation
def build_k_indices(y, k_fold, seed):
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval] for k in range(k_fold)]
    return np.array(k_indices)

# example where lambda_ and degree are the hyper parameters
def cross_validation(y, x, k_indices, k, lambda_, degree):

    # get k'th subgroup in test, others in train
    testindex = k_indices[k]
    trainindex = np.delete(k_indices, k, 0).ravel()

    # split the data, and return train and test data:
    test_x = x[testindex]
    test_y = y[testindex]
    train_x = x[trainindex]
    train_y = y[trainindex]

    # form data with polynomial degree
    training = build_poly(train_x, degree)
    test = build_poly(test_x, degree)

    # implemented with ridge regression
    weight_tr, MSE_tr = implementations.ridge_regression(train_y, training, lambda_)
    loss_te = compute_rmse(test_y, test, weight_tr)
    return loss_te


# OTHER HELPER METHODS

# standardizes the data. Returns the standardized input matrix.
def standardize(x):
    return (x - np.mean(x, axis=0)) / np.std(x, axis=0)


# builds and returns a polynomial expansion in given degree of input x.
def build_poly(x, degree):
    polynomial_expansion = np.ones([len(x), 1])
    for i in range(1, degree + 1):
        x_temp = np.power(x, i)
        polynomial_expansion = np.c_[polynomial_expansion, x_temp]
    return polynomial_expansion


# mean-adjust entries with -999. We adjust them to the mean of
# the column (without the -999 values). Outputs the adjusted matrix
def clean_matrix(x):
    means = np.zeros(len(x[0]))
    number_of_entries = np.zeros(len(x[0]))
    sums = np.zeros(len(x[0]))

    # summing over all values per column that are not -999
    for j in range(len(means)):
        for i in range(len(x)):
            if (x[i][j] != -999):
                sums[j] += x[i][j]
                number_of_entries[j] += 1

    # calculating mean value for every column of x
    for j in range(len(means)):
        means[j] = sums[j] / number_of_entries[j]

    # replacing -999 with the mean of the column
    for j in range(len(means)):
        for i in range(len(x)):
            if (x[i][j] == -999):
                x[i][j] = means[j]
    return x


# excludes features of the input matrix which has more -999 values than the given limit
def exclude_features(features, limit):
    nr_of_values = np.zeros(len(features[0]))
    indexes_to_include = []

    # count number of -999 values in each feature
    for col in range(len(features[0])):
        for row in range(len(features)):
            if features[row][col] == -999:
                nr_of_values[col] += 1

    # append indexes of features which shall be outputted
    for i, count in enumerate(nr_of_values):
        if count < limit * len(features):
            indexes_to_include.append(i)
    return features[:, indexes_to_include]

# takes a data set and a list of indexes and removes features with these indexes from the data set.
# Returns the rest
def remove_features(features,remove_featurelist):
    return np.delete(features, remove_featurelist, 0)