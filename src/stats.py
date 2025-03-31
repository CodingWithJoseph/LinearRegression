import math  # Import math module for floor function
import numpy as np  # Import NumPy for numerical operations


def first_quartile(data: np.ndarray):
    """Calculates the first quartile (Q1) for each feature in a NumPy array."""
    shape = data.shape  # Get the shape of the input data (number of examples and features)
    num_examples = shape[0]  # Get the number of examples (rows)
    num_features = shape[1]  # Get the number of features (columns)

    sorted_data = np.sort(data, axis=0)  # Sort the data along the example axis (rows) for quartile calculation
    lq = np.zeros(shape=num_features)  # Initialize an array to store the lower quartiles for each feature

    for ft in range(num_features):  # Iterate through each feature
        lower_position = 0.25 * (num_examples - 1)  # Calculate the position of the first quartile
        lower_index = math.floor(lower_position)  # Get the integer index of the lower position
        lower_fraction = lower_position - lower_index  # Get the fractional part of the lower position

        if lower_fraction == 0.0:  # If the lower position is an integer
            lower_quartile = sorted_data[lower_index][ft]  # The quartile is the value at the index
        else:  # If the lower position is not an integer
            lower_quartile = (sorted_data[lower_index][ft] * (1 - lower_fraction)) + (
                    sorted_data[lower_index + 1][ft] * lower_fraction)  # Interpolate between the two adjacent values

        lq[ft] = lower_quartile  # Store the calculated lower quartile for the current feature

    return lq  # Return the array of lower quartiles


def third_quartile(data: np.ndarray):
    """Calculates the third quartile (Q3) for each feature in a NumPy array."""
    shape = data.shape  # Get the shape of the input data
    num_examples = shape[0]  # Get the number of examples
    num_features = shape[1]  # Get the number of features

    sorted_data = np.sort(data, axis=0)  # Sort the data along the example axis
    uq = np.zeros(shape=num_features)  # Initialize an array to store the upper quartiles for each feature

    for ft in range(num_features):  # Iterate through each feature
        upper_position = 0.75 * (num_examples - 1)  # Calculate the position of the third quartile
        upper_index = math.floor(upper_position)  # Get the integer index of the upper position
        upper_fraction = upper_position - upper_index  # Get the fractional part of the upper position
        if upper_fraction == 0.0:  # If the upper position is an integer
            upper_quartile = sorted_data[upper_index][ft]  # The quartile is the value at the index
        else:  # If the upper position is not an integer
            upper_quartile = (sorted_data[upper_index][ft] * (1 - upper_fraction)) + (
                    sorted_data[upper_index + 1][ft] * upper_fraction)  # Interpolate between the two adjacent values

        uq[ft] = upper_quartile  # Store the calculated upper quartile for the current feature
    return uq  # Return the array of upper quartiles


def interquartile_range(data):
    """Calculates the interquartile range (IQR) for each feature."""
    return third_quartile(data) - first_quartile(data)  # Calculate IQR by subtracting Q1 from Q3


def z_score(data):
    """Calculates the Z-scores for each data point (excluding the last column)."""
    d = data[:, :-1]  # Select all rows and all columns except the last one (assuming last column is labels)
    means = np.mean(d, axis=0)  # Calculate the mean of each feature
    std = np.std(d, axis=0)  # Calculate the standard deviation of each feature
    distance = np.subtract(d, means)  # Calculate the difference between each data point and the mean
    return distance / std  # Calculate the Z-scores by dividing the distance by the standard deviation


def mask_outliers_using_zscore(data, threshold=3.0):
    """Removes outliers from the data using Z-score method."""
    s = z_score(data)  # Calculate the Z-scores
    outlier_indices = []  # Initialize a list to store the indices of outliers
    for i, s in enumerate(s):  # Iterate through each Z-score
        if abs(s) >= threshold:  # If the absolute Z-score is greater than or equal to the threshold
            outlier_indices.append(i)  # Add the index to the list of outliers
    mask = np.ones(len(data), dtype=bool)  # Create a boolean mask initially set to True for all rows
    mask[outlier_indices] = False  # Set the mask to False for outlier rows
    return data[mask]  # Return the data with outliers removed
