import math
import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.datasets import fetch_california_housing


class LinearRegression:
    def __init__(self, learning_rate=1e-3):
        self.predictions = None
        self.weights = None
        self.bias = None
        self.learning_rate = learning_rate

    def fit(self, X, y, epochs=3):
        num_examples = X.shape[0]
        num_features = X.shape[1]
        self.weights = np.zeros(shape=(num_features, 1))
        self.bias = 0

        for epoch in range(epochs):
            self.predict(X)
            dW, db = self.gradient_descent(X, y, num_examples)
            self.weights -= self.learning_rate * dW
            self.bias -= self.learning_rate * db

    def predict(self, X):
        self.predictions = np.matmul(X, self.weights) + self.bias

    def means_squared_error(self, labels):
        num_examples = labels.shape[0]
        return np.sum(np.subtract(self.predictions, labels)**2)/num_examples

    def gradient_descent(self, X, y, num_examples):
        dWdl = (2 / num_examples) * np.dot(X.T, self.predictions - y)
        dbdl = (2 / num_examples) * np.sum(self.predictions - y)
        return dWdl, dbdl

    def evaluate(self, X, y):
        self.predict(X)
        y_mean = np.mean(y)
        tss = np.sum(np.square(y - y_mean))
        rss = np.sum(np.square(y - self.predictions))
        return 1 - (rss / tss)


def test_train_split(data, tst_perc=0.2, seed_state=3):
    num_examples = data.shape[0]

    tst_size = math.ceil(num_examples * tst_perc)
    np.random.seed(seed_state)
    shuffled_data = np.random.permutation(data)

    X_tst = shuffled_data[:tst_size, :-1]
    y_tst = shuffled_data[:tst_size, -1:].reshape(-1, 1)

    X_trn = shuffled_data[tst_size:, :-1]
    y_trn = shuffled_data[tst_size:, -1:].reshape(-1, 1)

    return X_trn, X_tst, y_trn, y_tst


def first_quartile(data: np.ndarray):
    shape = data.shape
    num_examples = shape[0]
    num_features = shape[1]

    sorted_data = np.sort(data, axis=0)
    lq = np.zeros(shape=num_features)

    for ft in range(num_features):
        lower_position = 0.25 * (num_examples - 1)
        lower_index = math.floor(lower_position)
        lower_fraction = lower_position - lower_index

        if lower_fraction == 0.0:
            lower_quartile = sorted_data[lower_index][ft]
        else:
            lower_quartile = (sorted_data[lower_index][ft] * (1 - lower_fraction)) + (
                    sorted_data[lower_index + 1][ft] * lower_fraction)

        lq[ft] = lower_quartile

    return lq


def third_quartile(data: np.ndarray):
    shape = data.shape
    num_examples = shape[0]
    num_features = shape[1]

    sorted_data = np.sort(data, axis=0)
    uq = np.zeros(shape=num_features)

    for ft in range(num_features):
        upper_position = 0.75 * (num_examples - 1)
        upper_index = math.floor(upper_position)
        upper_fraction = upper_position - upper_index
        if upper_fraction == 0.0:
            upper_quartile = sorted_data[upper_index][ft]
        else:
            upper_quartile = (sorted_data[upper_index][ft] * (1 - upper_fraction)) + (
                    sorted_data[upper_index + 1][ft] * upper_fraction)

        uq[ft] = upper_quartile
    return uq


def interquartile_range(data):
    return third_quartile(data) - first_quartile(data)


def robust_scalar(data):
    fq = first_quartile(data)
    iqr = interquartile_range(data)
    scaled_data = (data - fq) / iqr
    return scaled_data


def create_data():
    housing_bunch = fetch_california_housing(as_frame=True)
    housing_df_w_label = housing_bunch.frame
    housing_df_w_label.drop(housing_bunch.feature_names[1:], axis=1, inplace=True)
    return housing_df_w_label.values


housing_with_labels = create_data()
housing_no_labels = housing_with_labels[:, :-1]
means = np.mean(housing_with_labels[:, :-1], axis=0)
std = np.std(housing_with_labels[:, :-1], axis=0)

distance = np.subtract(housing_no_labels, means)
z_score = distance / std

outlier_indices = []
for i, score in enumerate(z_score):
    if abs(score) >= 3.5:
        outlier_indices.append(i)

mask = np.ones(len(housing_no_labels), dtype=bool)
mask[outlier_indices] = False

housing_with_labels_filtered = housing_with_labels[mask]

X_train, X_test, y_train, y_test = test_train_split(housing_with_labels_filtered)

regression = LinearRegression()
regression.fit(X_train, y_train, epochs=10000)
evaluation = regression.evaluate(X_test, y_test)

sk_regression = linear_model.LinearRegression()
sk_regression.fit(X_train, y_train)
sk_evaluation = sk_regression.score(X_test, y_test)

print(evaluation)
print(sk_evaluation)

error = np.subtract(regression.predictions, y_test)

# error_df = pd.DataFrame({"Error": error})
predicted_actual_df = pd.DataFrame({"Predictions": regression.predictions.squeeze(), "Actual": y_test.squeeze()})

sb.scatterplot(x="Predictions", y="Actual", data=predicted_actual_df)
plt.show()

print(np.sqrt(regression.means_squared_error(y_test)))
