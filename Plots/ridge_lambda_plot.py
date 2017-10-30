"""
Used to create the plot of ridge regression with lambdas against rmse in the report.
"""

import matplotlib.pyplot as plt

from HelpMethods import helpers, given
from Implementations import implementations

DATA_PATH = ""
DEGREE = 6
RATIO = 0.8

lambdas = [0, 0.1, 0.2, 0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]

# load the data
y_train, x_train, ids = given.load_csv_data(DATA_PATH)

# split the data
train_x, train_y, test_x, test_y = helpers.split_data(x_train, y_train, RATIO)

# build polynomial
train = helpers.build_poly(train_x, DEGREE)
test = helpers.build_poly(test_x, DEGREE)

# creating data set for rmse
rmse = []
for lambda_ in lambdas:
    w, mse = implementations.ridge_regression(train_y, train, lambda_)
    rmse_tmp = helpers.compute_rmse(test_y, test, w)
    rmse.append(rmse_tmp)

plt.plot(lambdas, rmse)
plt.ylabel("RMSE")
plt.xlabel("Lambda")
plt.savefig("testing" + " ridgeplot.png")

plt.show()