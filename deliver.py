import helpers
import numpy as np
import implementations

yb, input_data, ids = helpers.load_csv_data("train.csv")
yb_test, input_data_test, ids_test = helpers.load_csv_data("test.csv")


weights, mse = implementations.ridge_regression(yb, input_data, 0.1)
prediction = helpers.predict_labels(weights, input_data_test)

helpers.create_csv_submission(ids_test, prediction, "Prediction using Ridge Regression")