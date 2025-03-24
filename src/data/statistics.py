import numpy as np


def compute_column_range(data, num_fts: int) -> np.ndarray:
    sorted_data = np.sort(data, axis=0)
    column_ranges = np.zeros(num_fts)
    for ft in range(num_fts):
        column_ranges = sorted_data[-1] - sorted_data[0]
    return column_ranges


def compute_column_min_max(data, num_fts: int) -> np.ndarray:
    sorted_data = np.sort(data, axis=0)
    min_max = np.array([(0.0, 0.0)] * num_fts)

    for ft in range(num_fts):
        min_max[ft] = (sorted_data[0, ft], sorted_data[-1, ft])

    return min_max


def compute_column_means(data, num_exs: int, num_fts: int) -> np.ndarray:
    """
    Computes the mean of each feature in a NumPy array.
    """
    if not isinstance(data, np.ndarray):
        raise AssertionError('Input variables should be ndarray')

    mn = [0.0] * num_fts
    for ft in range(num_fts):
        for ex in range(num_exs):
            mn[ft] += data[ex, ft]
        mn[ft] = mn[ft] / num_exs
    return np.array(mn)


def compute_column_median(data, num_exs: int, num_fts: int) -> np.ndarray:
    """
    Computes the median of each feature in a NumPy array.
    """
    if not isinstance(data, np.ndarray):
        raise AssertionError('Input variables should be ndarray')

    sorted_data = np.sort(data, axis=0)
    median = np.zeros(num_fts)
    for ft in range(num_fts):
        if num_exs % 2 == 0:
            median[ft] = (sorted_data[num_exs // 2 - 1, ft] + sorted_data[num_exs // 2, ft]) / 2
        else:
            median[ft] = sorted_data[num_exs // 2, ft]
    return median


def compute_column_variance(data, means, num_exs: int, num_fts: int) -> np.ndarray:
    if not isinstance(data, np.ndarray):
        raise AssertionError('Input variables should be ndarray')

    var = np.zeros(num_fts)
    sample = 1 / (num_exs - 1)
    for ft in range(num_fts):
        dev_sum = 0
        for ex in range(num_exs):
            dev_sum += (means[ft] - data[ex, ft]) ** 2
        var[ft] = dev_sum * sample
    return var


def compute_column_std(data, means, num_exs: int, num_fts: int) -> np.ndarray:
    if not isinstance(data, np.ndarray):
        raise AssertionError('Input variables should be ndarray')

    var = compute_column_variance(data, means, num_exs, num_fts)
    stds = np.zeros(num_fts)
    for ft in range(num_fts):
        stds[ft] = np.sqrt(var[ft])
    return stds


def compute_covariance(data, means, num_exs: int, num_fts: int) -> np.ndarray:
    if not isinstance(data, np.ndarray):
        raise AssertionError('Input variables should be ndarray')

    cov = np.zeros((num_fts, num_fts))
    sample = 1 / (num_exs - 1)

    for ft1 in range(num_fts):
        for ft2 in range(num_fts):
            prd = 0
            for ex in range(num_exs):
                prd += (data[ex, ft1] - means[ft1]) * (data[ex, ft2] - means[ft2])
            cov[ft1, ft2] = prd * sample
    return cov


def compute_z_scores(data, means, num_exs: int, num_fts: int) -> np.ndarray:
    stds = compute_column_std(data=data, means=means, num_exs=num_exs, num_fts=num_fts)
    scores = np.zeros(num_fts)
    for ft in range(num_fts):
        for ex in range(num_exs):
            scores[ft] = (data[ex, ft] - means[ft]) / stds
    return scores


def compute_column_iqr(data, num_exs: int, num_fts: int) -> np.ndarray:
    if not isinstance(data, np.ndarray):
        raise AssertionError('Input variables should be ndarray')

    sorted_data = np.sort(data, axis=0)

    lqt = num_exs // 4
    uqt = 3 * (num_exs // 4)

    iqr = np.zeros(num_fts)

    for ft in range(num_fts):
        if lqt % 2 == 0:
            lqr = (sorted_data[lqt - 1, ft] + sorted_data[lqt, ft]) / 2
        else:
            lqr = sorted_data[lqt, ft]

        if uqt % 2 == 0:
            uqr = (sorted_data[uqt - 1, ft] + sorted_data[uqt, ft]) / 2
        else:
            uqr = sorted_data[uqt, ft]
        iqr[ft] = uqr - lqr

    return iqr
