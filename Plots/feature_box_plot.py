"""
Creates a box plot of the features on the x axis against the respective values on the y axis
The shape and layout varies with plt.boxplot()
"""


import matplotlib.pyplot as plt
from HelpMethods import helpers, given

DATA_PATH = ""
SAVE_PATH = ""

yb, input_data, ids = given.load_csv_data(DATA_PATH)

cleaned = helpers.clean_matrix(input_data)

stand_data = helpers.standardize(cleaned)

def box_plot_features_values(data, save):
    plt.figure()
    plt.boxplot(data, 0, '')

    plt.xlabel("Feature")
    plt.ylabel("Value")
    plt.savefig(save + str("Values") + "-" + str("Features") + " plot.png", dpi=200)
    plt.clf()

box_plot_features_values(stand_data, SAVE_PATH)