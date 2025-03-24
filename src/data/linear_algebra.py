import numpy as np
import sympy as sp


def construct_identity_matrix(shape: tuple) -> np.ndarray:
    I = np.zeros(shape=shape).astype(float)
    for i in range(I.shape[0]):
        I[i, i] = 1.0
    return I


def compute_determinant(A, evenSwap=True):
    length = A.shape[0]
    determinate = 1
    for i in range(length):
        determinate *= A[i, i]

    return determinate if evenSwap else (-1 * determinate)


def compute_row_echelon_form(A: np.ndarray) -> (np.ndarray, int):
    n = A.shape[0]
    A = A.astype(float)
    swap_scount = 0

    for i in range(n):
        pivot_row = i  # assign pivot row index
        m = abs(A[i, i])

        for k in range(i + 1, n):
            if m < abs(A[k, i]):
                pivot_row = k
                m = abs(A[k, i])

        if pivot_row != i:
            swap_scount += 1
            A[[i, pivot_row]] = A[[pivot_row, i]]

        pivot = A[i, i]
        scaled_pivot_row = (A[i] / pivot) if pivot != 0 else A[i]

        for k in range(i + 1, n):
            factor = A[k, i]
            A[k] = A[k] - scaled_pivot_row * factor

        A[i] = np.where(np.isclose(A[i], 0), 0, A[i])
    return A, swap_scount


def compute_eigenvalues(A: np.ndarray) -> np.ndarray:
    I = construct_identity_matrix(A.shape)
    _lambda = sp.symbols('λ') * I
    _A_lambda = A - _lambda
    characteristic_poly = compute_determinant(_A_lambda)
    eigenvalues = sp.solve(characteristic_poly, sp.symbols('λ'))
    return eigenvalues


def compute_eigenvectors(A: np.ndarray) -> np.ndarray:
    eigenvalues = compute_eigenvalues(A)
    eigenvectors = construct_identity_matrix(A.shape) * eigenvalues
    print(eigenvectors)


def perform_eigen_decomposition(data, means) -> np.ndarray:
    return np.zeros(data.shape[0])


A = np.array([[2, 3, 5], [0, 9, 8], [0, 0, 1]])
compute_eigenvectors(A)
