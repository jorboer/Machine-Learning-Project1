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
hyper_params1 = []
hyper_params2 = []

#Initializing decision variables
best_rmse = 1000000
best_hyperparam1 = -1
best_hyperparam2 = -1


# grid search for best combination of hyperparameters. Can be extended with more for loops.
# change "REGRESSION_METHOD" with desired method.
for hyper_param1 in hyper_params1:
    #hyper_param1 was often degree, and we did polynomial expansion here
    for hyper_param2 in hyper_params2:
        weights, mse = REGRESSION_METHOD(train_y, stand_train, hyper_param1)
        rmse_tmp = helpers.compute_rmse(test_y, stand_test, weights)
        if (rmse_tmp < best_rmse):
            best_rmse = rmse_tmp
            best_hyperparam1 = hyper_param1
            best_hyperparam2 = hyper_param2