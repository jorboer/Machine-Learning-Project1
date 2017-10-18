import numpy as np
import math

#Method to use if we need to standardize the data
def standardize(x):
    """Standardize the original data set."""
    mean_x = np.mean(x)
    x = x - mean_x
    std_x = np.std(x)
    x = x / std_x
    return x, mean_x, std_x

#Helper function for all three regression methods
def compute_loss(y, tx, w):
    e = y - np.dot(tx, np.transpose(w))
    squared = e**2
    MSE = np.mean(squared)/2
    return MSE

#Computes the gradient, needed for gradient descent and stochastic gradient descent
def compute_gradient(y, tx, w):
    e = y - np.dot(tx, w)
    gradient = -1*(np.dot(np.transpose(tx), e))/len(e)
    return gradient


#Helper function for gradient descent and stochastic gradient descent
def gradient_descent(y, tx, initial_w, max_iters, gamma):
    """Gradient descent algorithm."""
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        loss = compute_loss(y, tx, w)
        gradient = compute_gradient(y, tx, w)
        w = w - gamma*gradient
        ws.append(w)
        losses.append(loss)
        print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
              bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))
    return losses, ws

#Helper function for stochastic gradient descent
def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
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



def stochastic_gradient_descent(y, tx, initial_w, batch_size, max_iters, gamma):
    ws = [initial_w]
    losses = []
    w = initial_w
    teller = 0
    for minibatch_y, minibatch_tx in helpers.batch_iter(y, tx, batch_size, max_iters):
        gradient = compute_gradient(minibatch_y, minibatch_tx, w)
        loss = compute_loss(minibatch_y, minibatch_tx, w)
        w = w - gamma * gradient
        ws.append(w)
        losses.append(loss)
        print("Stochastic Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
            bi=teller, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))
        teller += 1
    return losses, ws


def least_squares(y, tx):
    A = np.dot(np.transpose(tx), tx)
    W = np.linalg.solve(A, np.dot(np.transpose(tx),y))
    MSE = compute_loss(y, tx, W)
    return W, MSE


#Helper method for polynomial regression and
def build_poly(x, degree):
    outmatrix = np.zeros((len(x), degree + 1))
    for i in range(degree + 1):
        outmatrix[:, i] = x**i
    return outmatrix


#Function we have created earlier has no input parameters (uses public variables instead)
def polynomial_regression(x, y):
    """Constructing the polynomial basis function expansion of the data,
       and then running least squares regression."""
    # define parameters
    degrees = [1, 3, 7, 12]

    # define the structure of the figure
    num_row = 2
    num_col = 2
    #f, axs = plt.subplots(num_row, num_col)

    for ind, degree in enumerate(degrees):
        temp_x = build_poly(x, degree)
        weights, mse = least_squares(y, temp_x)

        rmse = (2 * mse) ** 0.5
        print("Processing {i}th experiment, degree={d}, rmse={loss}".format(
            i=ind + 1, d=degree, loss=rmse))
        # plot fit
        # plot_fitted_curve(
            #  y, x, weights, degree, axs[ind // num_col][ind % num_col])
    #plt.tight_layout()
    #plt.savefig("visualize_polynomial_regression")
    #plt.show()


def ridge_regression(y, tx, lambda_):
    A = np.dot(np.transpose(tx), tx)
    B = np.dot(np.array(np.identity(len(A))), lambda_ * 2 * len(A))

    W = np.linalg.solve(A+B, np.transpose(tx).dot(y))
    MSE = compute_loss(y, tx, W)
    return W, MSE


#Method needed to test ridge regression, it splits data in two sets, one for testing and one for training
def split_data(x, y, ratio, seed=1):
    """
    split the dataset based on the split ratio. If ratio is 0.8
    you will have 80% of your data set dedicated to training
    and the rest dedicated to testing
    """
    # set seed
    np.random.seed(seed)
    shuffled_indices = np.random.permutation(len(x))
    training_data_x = x[shuffled_indices[: math.floor(len(x) * ratio)]]
    training_data_y = y[shuffled_indices[: math.floor(len(x) * ratio)]]
    test_data_x = x[shuffled_indices[math.floor(len(x) * ratio):]]
    test_data_y = y[shuffled_indices[math.floor(len(x) * ratio):]]

    return training_data_x, training_data_y, test_data_x, test_data_y


def train_test_split_demo(x, y, degree, ratio, seed):
    """polynomial regression with different split ratios and different degrees."""
    # ***************************************************
    train_x, train_y, test_x, test_y = split_data(x, y, ratio, seed)
    # split the data, and return train and test data: TODO
    # ***************************************************
    training = build_poly(train_x, degree)
    test = build_poly(test_x, degree)
    # form train and test data with polynomial basis function: TODO
    # ***************************************************
    weight_train, MSE_train = least_squares(train_y, training)
    loss_test = compute_loss(test_y, test, weight_train)
    # calcualte weight through least square.: TODO
    # ***************************************************
    rmse_tr = (2*MSE_train)**0.5
    rmse_te = (2*loss_test)**0.5
    # calculate RMSE for train and test data,
    # and store them in rmse_tr and rmse_te respectively: TODO
    # ***************************************************
    print("proportion={p}, degree={d}, Training RMSE={tr:.3f}, Testing RMSE={te:.3f}".format(
          p=ratio, d=degree, tr=rmse_tr, te=rmse_te))


def ridge_regression_demo(x, y, degree, ratio, seed):
    """ridge regression demo."""
    # define parameter
    lambdas = np.logspace(-5, 0, 15)
    # ***************************************************
    train_x, train_y, test_x, test_y = split_data(x, y, ratio, seed)
    # split the data, and return train and test data: TODO
    # ***************************************************
    training = build_poly(train_x, degree)
    test = build_poly(test_x, degree)

    weight_train, MSE_train = least_squares(train_y, training)
    loss_test = compute_loss(test_y, test, weight_train)
    # form train and test data with polynomial basis function: TODO
    # ***************************************************
    rmse_tr = []
    rmse_te = []
    for ind, lambda_ in enumerate(lambdas):
        # ***************************************************
        weight_tr, MSE_tr = ridge_regression(train_y, training, lambda_)
        loss_test = compute_loss(test_y, test, weight_tr)
        rmse_tr.append((MSE_tr * 2) ** 0.5)
        rmse_te.append((loss_test * 2) ** 0.5)
        # ridge regression with a given lambda
        # ***************************************************
        print("proportion={p}, degree={d}, lambda={l:.3f}, Training RMSE={tr:.3f}, Testing RMSE={te:.3f}".format(
            p=ratio, d=degree, l=lambda_, tr=rmse_tr[ind], te=rmse_te[ind]))




#Does not work
def cross_validation(y, x, k_indices, k, lambda_, degree):
    """return the loss of ridge regression."""
    loss_tr = 0
    loss_te = 0
    for i in range(k):
        test = k_indices[i]
        train = np.delete(k_indices,[i],axis=0)
        # ***************************************************
        # get k'th subgroup in test, others in train: TODO
        # ***************************************************
        test_x = x[test]
        test_y = y[test]
        train_x = x[train].ravel()
        train_y = y[train].ravel()
        # split the data, and return train and test data: TODO
        # ***************************************************
        test = build_poly(test_x, degree)
        training = build_poly(train_x, degree)
        # form data with polynomial degree: TODO
        weight_tr, MSE_tr = ridge_regression(train_y, training, lambda_)
        # ridge regression: TODO
        # ***************************************************
        loss_tr += MSE_tr
        loss_te += compute_loss(test_y, test, weight_tr)
    loss_tr /= k
    loss_te /= k
    return loss_tr, loss_te

def cross_validation_demo():
    seed = 1
    degree = 7
    k_fold = 4
    lambdas = np.logspace(-4, 0, 30)
    # split data in k fold
    k_indices = build_k_indices(y, k_fold, seed)
    # define lists to store the loss of training data and test data
    rmse_tr = []
    rmse_te = []
    for ind, lambda_ in enumerate(lambdas):
        MSE_tr, MSE_te = cross_validation(y, x, k_indices, k_fold, lambda_, degree)
        rmse_tr.append((MSE_tr*2)**0.5)
        rmse_te.append((MSE_te*2)**0.5)


def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    return np.array(k_indices)