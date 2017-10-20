import numpy as np


def least_squares(y, tx):
    A = np.dot(np.transpose(tx), tx)
    W = np.linalg.solve(A, np.dot(np.transpose(tx),y))
    MSE = compute_loss(y, tx, W)
    return W, MSE