import numpy as np
import implementations
import least_squares as ls


def ridge_regression(y, tx, lambda_):
    A = np.dot(np.transpose(tx), tx)
    B = np.dot(np.array(np.identity(len(A))), lambda_ * 2 * len(A))
    C = np.linalg.inv(A + B)
    D = np.dot(C, np.transpose(tx))
    W = np.dot(D, y)

    MSE = implementations.compute_loss(y, tx, W)
    return W, MSE

def ridge_regression_demo(x, y, degree, ratio, seed):
    """ridge regression demo."""
    # define parameter
    lambdas = np.logspace(-5, 0, 15)
    # ***************************************************
    train_x, train_y, test_x, test_y = implementations.split_data(x, y, ratio, seed)
    # split the data, and return train and test data: TODO
    # ***************************************************
    training = implementations.build_poly(train_x, degree)
    test = implementations.build_poly(test_x, degree)

    weight_train, MSE_train = ls.least_squares(train_y, training)
    loss_test = implementations.compute_loss(test_y, test, weight_train)
    # form train and test data with polynomial basis function: TODO
    # ***************************************************
    rmse_tr = []
    rmse_te = []
    for ind, lambda_ in enumerate(lambdas):
        # ***************************************************
        weight_tr, MSE_tr = ridge_regression(train_y, training, lambda_)
        loss_test = implementations.compute_loss(test_y, test, weight_tr)
        rmse_tr.append((MSE_tr * 2) ** 0.5)
        rmse_te.append((loss_test * 2) ** 0.5)
        # ridge regression with a given lambda
        # ***************************************************
        print("proportion={p}, degree={d}, lambda={l:.3f}, Training RMSE={tr:.3f}, Testing RMSE={te:.3f}".format(
            p=ratio, d=degree, l=lambda_, tr=rmse_tr[ind], te=rmse_te[ind]))

