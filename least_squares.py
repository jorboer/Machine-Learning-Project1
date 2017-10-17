import numpy as np
import implementations

def least_squares(y, tx):
    A = np.dot(np.transpose(tx), tx)
    B = np.linalg.inv(A)
    C = np.dot(B, np.transpose(tx))
    W = np.dot(C, y)
    lowest_cost = implementations.compute_loss(y, tx, W)
    return W, lowest_cost