import helpers
import numpy as np
import implementations

y_train, x_train, ids = helpers.load_csv_data("C:\\Users\\magnu\\Documents\\NTNU\\3 (Utveksling EPFL)\\Machine Learning\\Prosjekt1\\Data\\train\\train.csv")
y_test, x_test, ids_test = helpers.load_csv_data("C:\\Users\\magnu\\Documents\\NTNU\\3 (Utveksling EPFL)\\Machine Learning\\Prosjekt1\\Data\\test\\test.csv")

degrees = [1, 2, 3, 4, 5, 6, 7]

teller = 1
for ind, degree in enumerate(degrees):
    temp_x_train = implementations.build_poly(x_train[:,0], degree)
    temp_x_test = implementations.build_poly(x_test[:,0], degree)

    w, train_mse = implementations.least_squares(y_train, temp_x_train)

    mse_test = implementations.compute_loss(y_test, temp_x_test, w)

    print(teller, "Test MSE: ", mse_test, '\n',teller, 'Train MSE: ', train_mse)
    teller +=1

#prediction = helpers.predict_labels(weights, input_data_test)

#print(weights, ', ', weights_test)
#print(mse, ', ', mse_test)


#helpers.create_csv_submission(ids_test, prediction, "Prediction using Ridge Regression")