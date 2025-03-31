import math  # Import math for ceiling function
import numpy as np  # Import NumPy for numerical operations
from sklearn.datasets import fetch_california_housing  # Import the California housing dataset from scikit-learn


def create_housing_data(bunch):
    """
    Fetches the California housing dataset and prepares it for use.

    Returns:
        np.ndarray: A NumPy array containing the housing data, with only the 'MedInc' feature and the target.
    """
    df = bunch.frame  # Access the DataFrame from the Bunch object.
    df.drop(bunch.feature_names[1:], axis=1, inplace=True)  # Drop all features except 'MedInc' (Median Income).
    return df.values  # Return the DataFrame values as a NumPy array.


def test_train_split(data, tst_perc=0.2, seed_state=42):
    """
    Splits the data into training and testing sets.

    Args:
        data (np.ndarray): The dataset to split.
        tst_perc (float, optional): The percentage of data to use for testing. Defaults to 0.2.
        seed_state (int, optional): The random seed for reproducibility. Defaults to 42.

    Returns:
        tuple: A tuple containing the training and testing sets (X_trn, X_tst, y_trn, y_tst).
    """
    num_examples = data.shape[0]  # Get the number of examples in the dataset.

    tst_size = math.ceil(num_examples * tst_perc)  # Calculate the number of examples for the test set.
    np.random.seed(seed_state)  # Set the random seed for reproducibility.
    shuffled_data = np.random.permutation(data)  # Shuffle the data randomly.

    X_tst = shuffled_data[:tst_size, :-1]  # Extract features for the test set.
    y_tst = shuffled_data[:tst_size, -1:].reshape(-1, 1)  # Extract target values for the test set and reshape.

    X_trn = shuffled_data[tst_size:, :-1]  # Extract features for the training set.
    y_trn = shuffled_data[tst_size:, -1:].reshape(-1, 1)  # Extract target values for the training set and reshape.

    return X_trn, X_tst, y_trn, y_tst  # Return the training and testing sets.
