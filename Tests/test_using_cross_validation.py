"""
Example of how we used cross validation to search for optimal hyper-parameters using a grid search.
"""

from HelpMethods import helpers, given

DATA_PATH = ""
SEED = 1

# loading data
y_train, x_train, ids = given.load_csv_data(DATA_PATH)

#Initializing lists of hyper-parameters we want to test
hyper_params1 = []
hyper_params2 = []

#Initializing decision variables
best_rmse = 1000000
best_hyperparam1 = -1
best_hyperparam2 = -1
k_fold = 4

# split data in k fold
k_indices = helpers.build_k_indices(y_train, k_fold, SEED)

# standardizing the input data
stand_train = helpers.standardize(x_train)

#grid search for best combination of hyperparameters. Can be extended with more for loops.
for hyper_param1 in hyper_params1:
    for hyper_param2 in hyper_params2:
        rmse_tmp = 0
        for i in range(k_fold):
            rmse_te = helpers.cross_validation(y_train, stand_train, k_indices, i, hyper_param1, hyper_param2)

            #sums all the error (k times)
            rmse_tmp += rmse_te

        #computes the average of the errors
        rmse_tmp /= k_fold

        #updates best hyper parameters if the error is less than the best registered so far
        if (rmse_tmp < best_rmse):
            best_rmse = rmse_tmp
            best_hyperparam1 = hyper_param1
            best_hyperparam2 = hyper_param2