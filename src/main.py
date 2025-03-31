
import numpy as np
from sklearn.datasets import fetch_california_housing

from data_loader import create_housing_data, test_train_split
from model import LinearRegression
from stats import mask_outliers_using_zscore
from data_visualization import vis_correlation_matrix, vis_prediction_label

if __name__ == '__main__':
    bunch = fetch_california_housing(as_frame=True)
    vis_correlation_matrix(bunch)

    # Load and preprocess the California housing data
    housing = create_housing_data(bunch)
    housing = mask_outliers_using_zscore(housing)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = test_train_split(housing)

    # Train your custom Linear Regression model
    regression = LinearRegression()
    regression.fit(X_train, y_train, epochs=10000)

    # Calculate and print R-squared (R2) scores
    r2_score = regression.score(X_test, y_test)

    # Calculate and print Root Mean Squared Error (RMSE)
    rmse = np.sqrt(regression.means_squared_error(y_test))

    vis_prediction_label(X_test, y_test, regression.predictions)
    print("R-squared (R2) score:", r2_score)
    print("Root Mean Squared Error (RMSE):", rmse)
