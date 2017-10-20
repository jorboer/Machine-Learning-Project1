from HelpMethods import helpers

from Implementations import implementations

y_test, x_test, ids_test = helpers.load_csv_data("C:\\Users\\magnu\\Documents\\NTNU\\3 (Utveksling EPFL)\\Machine Learning\\Prosjekt1\\Data\\test\\test.csv")
y_train, x_train, ids_train = helpers.load_csv_data("C:\\Users\\magnu\\Documents\\NTNU\\3 (Utveksling EPFL)\\Machine Learning\\Prosjekt1\\Data\\train\\train.csv")


#std_x, mean, std = helpers.standardize(x_train)
#standardize_test = helpers.standardize(x_test)

lambdas = [10000000,1000,100,10,1,0.1,0.01,0.001,0.0001,0.00001]

def grid_search_for_lambda(y, tx, y_test, x_test, lambdas):
    lowest_loss = 100000000
    best_lambda = 0
    for lambda_ in lambdas:
        weights, mse_train = implementations.ridge_regression(y, tx, lambda_)
        mse_test = implementations.compute_loss(y_test, x_test, weights)
        if (mse_test < lowest_loss):
            lowest_loss = mse_test
            best_lambda = lambda_
    return lowest_loss, best_lambda

print(grid_search_for_lambda(y_train, x_train, y_test, x_test, lambdas))

"""
def grid_search_for_lambda(y, tx, lambdas):
    seed = 1
    degree = 3
    k_fold = 10
    # split data in k fold
    k_indices = impl.build_k_indices(y, k_fold, seed)
    # define lists to store the loss of training data and test data
    loss = 100000000000
    lowest_loss = loss
    rmse_tr = loss
    rmse_te = loss
    best_lambda = 0
    for lambda_ in lambdas:
        MSE_tr, MSE_te = impl.cross_validation(y, tx, k_indices, k_fold, lambda_, degree)
        rmse_tr = (MSE_tr * 2) ** 0.5
        rmse_te = (MSE_te * 2) ** 0.5
        if (rmse_te < lowest_loss):
            lowest_loss = rmse_te
            best_lambda = lambda_
    return lowest_loss, best_lambda

lowest_loss, best_lambda = grid_search_for_lambda(yb, standardize_train, lambdas)
print(lowest_loss)
print(best_lambda)

"""
