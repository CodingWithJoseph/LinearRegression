import math
from typing import Tuple
import numpy as np
import pandas as pd


def count_missing_values(data, num_exs: int, num_fts: int) -> np.ndarray:
    """
    Computes missing values for each feature in a NumPy array.
    """
    if not isinstance(data, np.ndarray):
        raise AssertionError('Input variables should be ndarray')

    missing = np.zeros(num_fts)
    for ft in range(num_fts):
        for ex in range(num_exs):
            if data[ex, ft] is None or math.isnan(data[ex, ft]):
                missing[ft] += 1
        missing[ft] = missing[ft] / num_exs
    return np.array(missing)


def fill_na_mean(data, mean, num_exs: int, num_fts: int) -> np.ndarray:
    """
    Fills NaN (Not a Number) values in a NumPy array with the mean of the respective feature.
    """
    if not isinstance(data, np.ndarray):
        raise AssertionError('Input variables should be ndarray')

    for ft in range(num_fts):
        for ex in range(num_exs):
            if pd.isna(data[ex, ft]):
                data[ex, ft] = mean[ft]
    return data


def split_train_validation_test(data, train_percent: float = 0.6, validation_percent: float = 0.2,
                                test_percent: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Splits a NumPy array into training, validation, and test sets.
    """
    if not isinstance(data, np.ndarray):
        raise AssertionError('Input variables should be ndarray')
    elif np.isclose(abs(train_percent + validation_percent + test_percent), 1):
        raise AssertionError('The sum (train_percent + validation_percent + test_percent) is not close to 1')

    values = data
    np.random.seed(42)
    np.random.shuffle(values)

    num_trn_examples = int(len(values) * train_percent)
    num_val_examples = int(len(values) * validation_percent)

    trn = pd.DataFrame(data=values[:num_trn_examples])
    val = pd.DataFrame(data=values[num_trn_examples:(num_trn_examples + num_val_examples)])
    test = pd.DataFrame(data=values[(num_trn_examples + num_val_examples):])
    return trn, val, test



