import numpy as np

from data.linear_algebra import perform_eigen_decomposition
from data.statistics import compute_z_scores, compute_covariance


def compute_principle_components(data, means, num_exs: int, num_fts: int) -> np.ndarray:
    # step 1 compute z-scores
    # step 2 eigen decomposition
    # step 3 dimensionality reduction
    scores = compute_z_scores(data, means, num_exs, num_fts)
    compute_covariance(scores, means, num_exs, num_fts)
    return perform_eigen_decomposition(scores, means)
