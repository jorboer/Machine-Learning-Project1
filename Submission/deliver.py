import HelpMethods.helpers as helpers

from Implementations import implementations

y_train, x_train, ids = helpers.load_csv_data("C:\\Users\\magnu\\Documents\\NTNU\\3 (Utveksling EPFL)\\Machine Learning\\Prosjekt1\\Data\\train\\train.csv")
y_test, x_test, ids_test = helpers.load_csv_data("C:\\Users\\magnu\\Documents\\NTNU\\3 (Utveksling EPFL)\\Machine Learning\\Prosjekt1\\Data\\test\\test.csv")

#weightsLS, mse_train2 = implementations.least_squares(y_train, x_train)
#print("Non-standardized and cleaned loss: ", implementations.compute_loss(y_test, x_test, weightsLS))

#Replacing -999 with mean of the respective column
clean_train = helpers.clean_matrice(x_train)
clean_test = helpers.clean_matrice(x_test)

#Standardizing the matrix, changing all previous -999 to 0
stand_train, mean, std = helpers.standardize(clean_train)
stand_test, mean_test, std_test = helpers.standardize(clean_test)

#Calculating weights
weights, mse_train = implementations.least_squares(y_train, stand_train)
#print("Standardized and cleaned, loss: ", implementations.compute_loss(y_test, stand_test, weights))


prediction = helpers.predict_labels(weights, stand_test)

helpers.create_csv_submission(ids_test, prediction, "Least Squares, cleaned and standardized train and test data")