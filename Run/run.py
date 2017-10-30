from HelpMethods import helpers, given
from Implementations import implementations

# Our best submission, least squares using a polynomial expansion of 6.

TRAIN_DATA_PATH = ""
TEST_DATA_PATH = ""
PREDICTION_NAME = "Least Squares with degree 6."
degree = 6

# load training data
y_train, x_train, _ = given.load_csv_data(TRAIN_DATA_PATH)

# build the polynomial for training
training = helpers.build_poly(x_train, degree)

# calculating weights
weights, mse_train = implementations.least_squares(y_train, training)


# load test data
y_test, x_test, ids_test = helpers.load_csv_data(TEST_DATA_PATH)

# build the polynomial for testing
test = helpers.build_poly(x_test, degree)


prediction = given.predict_labels(weights, test)

given.create_csv_submission(ids_test, prediction, PREDICTION_NAME)