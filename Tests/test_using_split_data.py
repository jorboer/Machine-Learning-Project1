"""
Example of how we used split data to search for optimal hyper-parameters using a grid search.
"""


from HelpMethods import helpers, given

DATA_PATH = ""
SPLIT_RATIO = 0.8

# load the data
y_train, x_train, ids = given.load_csv_data(DATA_PATH)

#split the data with given ratio
train_x, train_y, test_x, test_y = helpers.split_data(x_train, y_train, SPLIT_RATIO)

# standardize the train and test data
stand_train = helpers.standardize(train_x)
stand_test = helpers.standardize(test_x)

#Initializing lists of hyper-parameters we want to test
degrees = []
hyper_params = []

#Initializing decision variables
best_rmse = 1000000
best_degree = -1
best_hyperparam = -1


# grid search for best combination of hyperparameters. Can be extended with more for loops.
# change "REGRESSION_METHOD" with desired method.
for degree in degrees:
    #building polynomials
    train = helpers.build_poly(stand_train, degree)
    test = helpers.build_poly(stand_test, degree)
    for hyper_param in hyper_params:
        #computes the weights with a regression method and calculates the loss.
        weights, mse = REGRESSION_METHOD(train_y, train, hyper_param)
        rmse_tmp = helpers.compute_rmse(test_y, test, weights)

        #updates best hyper parameters if the error is less than the best registered so far.
        if (rmse_tmp < best_rmse):
            best_rmse = rmse_tmp
            best_degree = degree
            best_hyperparam = hyper_param