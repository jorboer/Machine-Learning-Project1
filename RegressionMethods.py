import numpy as np

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
    B = np.linalg.inv(A)
    C = np.dot(B, np.transpose(tx))
    W = np.dot(C, y)
    lowest_cost = compute_loss(y, tx, W)
    return W, lowest_cost