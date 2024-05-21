from numpy import (
    mean,
    isnan,
    logical_not,
    float64,
    sum,
    flatnonzero,
    array,
    full,
    floating,
    nan,
    divide,
)
from numpy.typing import NDArray
from numbers import Integral
from typing import Any
from sklearn.utils.validation import check_array
from warnings import warn


def _residual(
    X, indices, row_indices, col_indices, total_mean, row_means, col_means
):
    Y = total_mean + row_means + col_means
    return (
        mean(X.flat[indices] - Y.flat[indices]) ** 2
        + sum(
            [
                (
                    mean(X[i, :].flat[idx] - Y[i, :].flat[idx]) ** 2
                    if len(idx) > 0
                    else 0.0
                )
                for i, idx in enumerate(row_indices)
            ]
        )
        + sum(
            [
                (
                    mean(X[:, i].flat[idx] - Y[:, i].flat[idx]) ** 2
                    if len(idx) > 0
                    else 0.0
                )
                for i, idx in enumerate(col_indices)
            ]
        )
    )


def bicenter(X: NDArray[float64], tol: float = 1e-16, max_iter: int = 10):
    """Bicenter the input matrix allowing for missing values.

    Computes a total mean as well as row and column means.

    Implements the centering algorithm described in

    Hastie et al. (2015) Matrix completion and low-rank SVD via fast
    alternating least squares. Journal of Machine Learning Research,
    16(104):3367--3402, 2015.

    Parameters
    ----------
    X : ndarray
        The input matrix
    tol : float
        Convergence tolerance
    max_iter : int
        Maximum number of iterations to perform.

    Returns
    -------
    (Y, m, rm, cm) : (ndarray, float, ndarray, ndarray)
        Returns the bi-centered matrix, the overall mean, as well as
        row-means and column-means.
    """
    X = check_array(X, force_all_finite="allow-nan")
    assert tol > 0, "'tol' needs to be positive"
    assert (
        isinstance(max_iter, Integral) and max_iter > 0
    ), "'max_iter' needs to be a positive integer"

    n, p = X.shape

    mask = logical_not(isnan(X))
    indices = flatnonzero(mask)
    row_indices = [flatnonzero(mask[i, :]) for i in range(n)]
    col_indices = [flatnonzero(mask[:, i]) for i in range(p)]

    # Initialization
    total_mean = mean(X.flat[indices])
    row_means = array(
        [
            mean(X[i, :].flat[idx]) if len(idx) > 0 else 0.0
            for i, idx in enumerate(row_indices)
        ]
    )[:, None]
    col_means = array(
        [
            mean(X[:, i].flat[idx]) if len(idx) > 0 else 0.0
            for i, idx in enumerate(col_indices)
        ]
    )[None, :]

    # Iterate
    for it in range(max_iter):
        total_mean = mean(
            X.flat[indices] - (row_means + col_means).flat[indices]
        )
        row_means = array(
            [
                (
                    mean(
                        X[i, :].flat[idx] - (total_mean + col_means).flat[idx]
                    )
                    if len(idx) > 0
                    else 0.0
                )
                for i, idx in enumerate(row_indices)
            ]
        )[:, None]
        col_means = array(
            [
                (
                    mean(
                        X[:, i].flat[idx] - (total_mean + row_means).flat[idx]
                    )
                    if len(idx) > 0
                    else 0.0
                )
                for i, idx in enumerate(col_indices)
            ]
        )[None, :]
        r_crit = _residual(
            X,
            indices,
            row_indices,
            col_indices,
            total_mean,
            row_means,
            col_means,
        )

        if r_crit <= tol:
            break

    if it + 1 == max_iter:
        warn(f"Bi-centering did not converge in {max_iter} iterations")

    Y = X.copy()
    Y[mask] -= (total_mean + row_means + col_means)[mask]

    return Y, total_mean, row_means, col_means


def nanscale(X: NDArray[floating[Any]], scale: float):
    Y = full(X.shape, nan, dtype=X.dtype)
    divide(
        X,
        scale,
        out=Y,
        where=logical_not(isnan(X)),
    )

    return Y
