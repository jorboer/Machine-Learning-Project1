"""
Plots the features against each other. Used for exploring the data,
and picking out possible features to leave out.
"""

import matplotlib.pyplot as plt

from HelpMethods import helpers, given

DATA_PATH = ""
SAVE_PATH = ""

#load training data
yb, input_data, ids = given.load_csv_data(DATA_PATH)

#standardize inputdata
stand_data = helpers.standardize(input_data)


def feature_plot(data, save):
    # blue = signal, red = background
    c1 = ['blue' if x == 1 else 'red' for x in yb]

    for x1 in range(2,data.shape[1]):
        for x2 in range(data.shape[1]):
            plt.scatter(data[:,x1], data[:,x2], color=c1, s=0.2)
            plt.xlabel(x1+1)
            plt.ylabel(x2+1)
            plt.savefig(save + str(x1+1) + "-" + str(x2+1) + " plot.png", dpi=200)
            plt.clf()

feature_plot(stand_data, SAVE_PATH)