import numpy as np
from Implementations import implementations
import math
from HelpMethods import *

def load_data():
    """load data."""
    data = np.loadtxt("C:\\Users\\Emil\\Documents\\NTNU\\ML\\project01\\data\\test1.csv", delimiter=",", skiprows=1, unpack=True)
    x1 = data[0]
    x2 = data[1]
    tx = np.c_[x1,x2]
    y = data[2]

    return tx, y


# #
# tx,y = load_data()
# tx = build_poly(tx,2)


def compute_log_loss(y,tx,w):
    X_w_vector = tx.dot(w)
    log_add_vector = np.logaddexp(0,X_w_vector)
    y_term = y*X_w_vector

    loss =np.sum(log_add_vector-y_term)
    return loss

def sigmoid(x):
    return np.exp(-np.logaddexp(0, -x))


#
# def sigmoid(vector):
#     """ Apply sigmoid function on the vector. """
#     return (np.exp(vector))/(1+np.exp(vector))
#
#

def compute_log_gradient(y, tx, w):
    """Compute the gradient."""
    X_w_vector = tx.dot(w)


    trans_X_w_minus_y = sigmoid(X_w_vector) - y

    gradient = tx.T.dot(trans_X_w_minus_y)

    return gradient

def compute_hessian(tx,w,lambda_):
    X_w_vector = tx.dot(w)
    trans_X = sigmoid(X_w_vector)
    trans_X_1 = (1-trans_X)
    # S = np.diag(trans_X*trans_X_1)
    wierd_x = trans_X*trans_X_1
    first_m = tx.T.copy()
    for i,row in enumerate(first_m):
        new_row = row*wierd_x

        first_m[i] = new_row

    H = first_m.dot(tx)
    H = H+np.eye(len(H)) * 2*lambda_
    # H2 = (tx.T.dot(S)).dot(tx)
    #print(H,"\n+hei",H2)
    # print(H.shape)
    return H
import time
def newton_decent(y, tx, initial_w, max_iters, gamma, lambda_):

    """Gradient descent algorithm."""
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    time_n = time.time()
    for n_iter in range(max_iters):

        # ***************************************************
        # INSERT YOUR CODE HERE
        # TODO: compute gradient and loss
        # ***************************************************
        # g = compute_log_reg_gradient(y, tx, w, lambda_)
        # loss = compute_log_loss(y, tx, w)
        # g = compute_log_gradient(y, tx, w)
        h = compute_hessian(tx,w,lambda_)
        g = compute_log_reg_gradient(y, tx, w, lambda_)
        #h_inv = np.linalg.inv(h)
        # loss = compute_log_loss(y,tx,w)
        # g = compute_log_reg_gradient(y, tx, w, 0.1)
        # loss = compute_log_loss(y, tx, w)
        # ***************************************************
        # INSERT YOUR CODE HERE
        # TODO: update w by gradient
        # ***************************************************
        w = w - gamma * np.linalg.solve(h,g)
        # store w and loss
        ws.append(w)
        # losses.append(loss)
    #     print(np.linalg.norm(g))
    #     if (n_iter % 100 == 0):
    #         print("Gradient Descent({bi}/{ti}): loss={l}".format(
    #             bi=n_iter, ti=max_iters - 1, l=compute_log_loss(y, tx, w)))
    #         print(np.linalg.norm(g))
    # print((time.time()-time_n)/60)
    return losses, np.array(ws[-1])

def compute_log_reg_gradient(y, tx, w, lamda_):
    return compute_log_gradient(y, tx, w) + (2 * lamda_ * w)

def gradient_descent_log(y, tx, initial_w, max_iters, gamma, lambda_):
    """Gradient descent algorithm."""
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        # ***************************************************
        # INSERT YOUR CODE HERE
        # TODO: compute gradient and loss
        # ***************************************************
        # g = compute_log_gradient(y,tx,w)
        # loss = compute_log_loss(y,tx,w)
        g = compute_log_reg_gradient(y,tx,w, lambda_)
        # loss = compute_log_loss(y,tx,w)
        # ***************************************************
        # INSERT YOUR CODE HERE
        # TODO: update w by gradient
        # ***************************************************
        w = w - gamma*g
        # store w and loss
        ws.append(w)
        # losses.append(loss)
        #if(n_iter%100 ==0):
        #    print("Gradient Descent({bi}/{ti}): loss={l}".format(
        #          bi=n_iter, ti=max_iters - 1, l=compute_log_loss(y,tx,w)))
        #    print(np.linalg.norm(g))
    return losses, np.array(ws[-1])
#
# losses,ws = newton_decent(y,tx,np.array([1,1,1,1,1]),5000,0.001)
# print(ws)
# h =compute_hessian(tx,np.array([-1,0.2,-0.11,3,3]))
# print(h)
# h_inv = np.linalg.inv(h)



