import numpy as np

from HelpMethods import helpers

#FUNCTION 1 - Least Squares
def least_squares(y, tx):
    tx_squared = np.dot(tx.T, tx)
    w = np.linalg.solve(tx_squared, np.dot(tx.T, y))
    loss = helpers.compute_mse(y, tx, w)
    #loss = helpers.compute_mae(y, tx, w)
    return w, loss


#FUNCTION 2 - Ridge regression
def ridge_regression(y, tx, lambda_):
    tx_squared = np.dot(tx.T, tx)
    lamda_identity = np.identity(len(tx[0])) * lambda_ * 2 * len(tx)
    sum = tx_squared + lamda_identity
    w = np.linalg.solve(sum, tx.T.dot(y))
    loss = helpers.compute_mse(y, tx, w)
    #loss = helpers.compute_mae(y, tx, w)
    return w, loss


#FUNCTION 3 - Least Squares Grad Desc
def gradient_descent(y, tx, initial_w, max_iters, gamma):
    """Gradient descent algorithm."""
    w = initial_w
    for n_iter in range(max_iters):
        loss = helpers.compute_mse(y, tx, w)
        #loss = helpers.compute_mae(y, tx, w)
        gradient = helpers.compute_gradient(y, tx, w)
        w = w - gamma*gradient
    return w, loss


#FUNCTION 4 - Least squares Stoch Grad Desc
def stochastic_gradient_descent(y, tx, initial_w, batch_size, max_iters, gamma):
    w = initial_w
    for minibatch_y, minibatch_tx in helpers.batch_iter(y, tx, batch_size, max_iters):
        gradient = helpers.compute_gradient(minibatch_y, minibatch_tx, w)
        loss = helpers.compute_mse(minibatch_y, minibatch_tx, w)
        #loss = helpers.compute_mae(y, tx, w)
        w = w - gamma * gradient
    return w, loss


#FUNCTION 5 - Logistic regression
def logistic_regression(y, tx, initial_w, max_iters, gamma):
    # set desired stop! criterion
    threshold = 1e-3

    losses = []
    w = initial_w
    for iter in range(max_iters):
        loss = helpers.compute_log_likelihood(y, tx, w)
        gradient = helpers.compute_log_gradient(y, tx, w)
        w = w - gamma * gradient
        losses.append(loss)

        if ((len(losses) > 1) and (np.abs(losses[-1] - losses[-2]) < threshold)):
            break
    return w, loss


#FUNCTION 6 - Regularized logistic regression
def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    # set desired stop! criterion
    threshold = 1e-3

    losses = []
    w = initial_w
    for iter in range(max_iters):
        loss, gradient = helpers.penalized_logistic_regression(y, tx, w, lambda_)
        w = w - gamma * gradient
        losses.append(loss)

        if ((len(losses) > 1) and (np.abs(losses[-1] - losses[-2]) < threshold)):
            break
    return w, loss