import helpers

from Implementations import implementations

y_train, x_train, ids = helpers.load_csv_data("C:\\Users\\magnu\\Documents\\NTNU\\3 (Utveksling EPFL)\\Machine Learning\\Prosjekt1\\Data\\train\\train.csv")
y_test, x_test, ids_test = helpers.load_csv_data("C:\\Users\\magnu\\Documents\\NTNU\\3 (Utveksling EPFL)\\Machine Learning\\Prosjekt1\\Data\\test\\test.csv")

stand_x, mean, std = helpers.standardize(x_train)
#standardize_test = helpers.standardize(x_test)


weights, mse_train = implementations.ridge_regression(y_train, x_train, 10000000)

prediction = helpers.predict_labels(weights, x_test)

#print(weights, ', ', weights_test)
#print(mse, ', ', mse_test)


helpers.create_csv_submission(ids_test, prediction, "Prediction using Ridge Regression and lambda 10 million")