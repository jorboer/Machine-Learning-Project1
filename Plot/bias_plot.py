import helpers
import matplotlib.pyplot as plt
import numpy as np

from Implementations.implementations import compute_loss, build_poly, least_squares

yb, input_data, ids = helpers.load_csv_data("C:\\Users\\magnu\\Documents\\NTNU\\3 (Utveksling EPFL)\\Machine Learning\\Prosjekt1\\Data\\train\\train.csv")
yb_test, input_data_test, ids_test = helpers.load_csv_data("C:\\Users\\magnu\\Documents\\NTNU\\3 (Utveksling EPFL)\\Machine Learning\\Prosjekt1\\Data\\test\\test.csv")


def bias_variance_decomposition_visualization(degrees, rmse_tr, rmse_te):
    """visualize the bias variance decomposition."""
    rmse_tr_mean = np.expand_dims(np.mean(rmse_tr, axis=0), axis=0)
    rmse_te_mean = np.expand_dims(np.mean(rmse_te, axis=0), axis=0)
    plt.plot(
        degrees,
        rmse_tr.T,
        'b',
        linestyle="-",
        color=([0.7, 0.7, 1]),
        label='train',
        linewidth=0.3)
    plt.plot(
        degrees,
        rmse_te.T,
        'r',
        linestyle="-",
        color=[1, 0.7, 0.7],
        label='test',
        linewidth=0.3)
    plt.plot(
        degrees,
        rmse_tr_mean.T,
        'b',
        linestyle="-",
        label='train',
        linewidth=3)
    plt.plot(
        degrees,
        rmse_te_mean.T,
        'r',
        linestyle="-",
        label='test',
        linewidth=3)
    plt.ylim(0.2, 0.7)
    plt.xlabel("degree")
    plt.ylabel("error")
    plt.title("Bias-Variance Decomposition")
    plt.savefig("bias_variance")




def bias_variance_demo():
    """The entry."""
    # define parameters
    seeds = range(100)
    num_data = 10000
    ratio_train = 0.005
    degrees = range(1, 10)

    # define list to store the variable
    rmse_tr = np.empty((len(seeds), len(degrees)))
    rmse_te = np.empty((len(seeds), len(degrees)))

    for index_seed, seed in enumerate(seeds):
        np.random.seed(seed)
        #x = np.linspace(0.1, 2 * np.pi, num_data)
        #y = np.sin(x) + 0.3 * np.random.randn(num_data).T

        # split data with a specific seed: TODO
        #x_train, y_train, x_test, y_test = split_data(x, y, ratio_train, seed)

        # bias_variance_decomposition: TODO
        for index_degree, degree in enumerate(degrees):
            temp_x_test = build_poly(input_data_test[:,0], degree)
            temp_x_train = build_poly(input_data[:,0], degree)

            w, train_mse = least_squares(yb, temp_x_train)
            rmse_tr[index_seed, index_degree] = np.sqrt(2 * train_mse)
            rmse_te[index_seed, index_degree] = np.sqrt(2 * compute_loss(yb_test, temp_x_test, w))

    bias_variance_decomposition_visualization(degrees, rmse_tr, rmse_te)


bias_variance_demo()