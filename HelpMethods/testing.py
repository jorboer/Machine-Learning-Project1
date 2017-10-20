from HelpMethods import helpers
import numpy as np
from Implementations import implementations as impl

#y_test, x_test, ids_test = helpers.load_csv_data("C:\\Users\\magnu\\Documents\\NTNU\\3 (Utveksling EPFL)\\Machine Learning\\Prosjekt1\\Data\\test\\test.csv")
y_train, x_train, ids = helpers.load_csv_data("C:\\Users\\magnu\\Documents\\NTNU\\3 (Utveksling EPFL)\\Machine Learning\\Prosjekt1\\Data\\train\\train.csv")



#Mean-adjust entries with -999. We adjust them to the mean of the column (without the -999 values)
means = np.zeros(len(x_train[0]))
number_of_entries = np.zeros(len(x_train[0]))
sums = np.zeros(len(x_train[0]))

print(x_train[1])

#Summing over all values per column that are not -999
for j in range(len(means)):
    for i in range(len(x_train)):
        if (x_train[i][j] != -999):
            sums[j] += x_train[i][j]
            number_of_entries[j] += 1
    print(sums[j])
    print(number_of_entries[j])
    print("\n")

for i in range(len(means)):
    means[i] = (sums[i]/number_of_entries[i])
    print(means[i])

#Replacing -999 with the mean of the column
for j in range(len(means)):
    for i in range(len(x_train)):
        if (x_train[i][j] == -999):
            x_train[i][j] = means[j]

#Standardizing the x-training data
standardize_x, mean, std = helpers.standardize(x_train)

print("\n")
print(x_train[1])
print("\n")
print(standardize_x[1])


"""
#Finding the optimal weights of the standardized train data using least_squares
weights, mse_train = impl.least_squares(y_train, standardize_x)

#Using weights to find loss on test_data
mse_test = impl.compute_loss(y_test, x_test, weights)

#Returns a very high loss, don't know why
print(mse_test)



#Options to improve

#1 Find optimal regularizer parameter using grid search
#2 Split data using cross validation to improve
#3 Find optimal level of degree using grid search, i.e. search for
#4 Remove features: can write input_data[:,:15] for instance. Look for correlation between features.







#prediction = help.predict_labels(weights, input_data_test)


#help.create_csv_submission(ids_test, prediction, "Some prediction...")"""
