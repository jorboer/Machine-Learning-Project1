"""
Used to create the plot of least squares with rmse against degrees in the report.
"""

import matplotlib.pyplot as plt

from HelpMethods import helpers, given
from Implementations import implementations

DATA_PATH = ""
RATIO = 0.8

degrees = [1,2,3,4,5,6]

# load the data
y_train, x_train, ids = given.load_csv_data(DATA_PATH)

# split the data
train_x, train_y, test_x, test_y = helpers.split_data(x_train, y_train, RATIO)


# creating data set of rmse
rmse = []
for degree in degrees:
    train = helpers.build_poly(train_x, degree)
    test = helpers.build_poly(test_x, degree)

    w, mse = implementations.least_squares(train_y, train)
    rmse_tmp = helpers.compute_rmse(test_y, test, w)
    rmse.append(rmse_tmp)


plt.scatter(degrees, rmse)
plt.ylabel("RMSE")
plt.xlabel("Degree")
plt.savefig("testing" + " scatterplot.png")
plt.show()