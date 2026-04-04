"""Functions to preprocess data.

This module contains functions that can be used to preprocess the data.
"""

from numbers import Integral
from typing import Any
from warnings import warn

from numpy import (
    array,
    divide,
    flatnonzero,
    floating,
    full,
    isnan,
    logical_not,
    mean,
    nan,
    sum,
)
from numpy.typing import NDArray
from sklearn.utils.validation import check_array


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


def bicenter(
    X: NDArray[floating[Any]], tol: float = 1e-16, max_iter: int = 10
) -> tuple[
    NDArray[floating[Any]],
    floating[Any],
    NDArray[floating[Any]],
    NDArray[floating[Any]],
]:
    """Bicenter the input matrix allowing for missing values.

    Instead of simply centering all elements around a total mean value, this
    function models the data as `X = m + rm + cm + Y`, where m is the total
    mean (shape ()), rm are row means (shape (n, 1)), cm are column means
    (shape (1, m)), and Y are residuals (shape (n, m)).

    Implements the centering algorithm described in
    > Hastie et al. (2015) Matrix completion and low-rank SVD via fast
    > alternating least squares. Journal of Machine Learning Research,
    > 16(104):3367--3402, 2015.

    Args:
        X: The input matrix
        tol: Convergence tolerance
        max_iter: Maximum number of iterations to perform.

    Returns:
        A tuple (Y, m, rm, cm) containing the bi-centered matrix Y, the
            overall mean m, as well as row-means rm and column-means cm.

    """
    X = check_array(X, ensure_all_finite="allow-nan")
    if tol <= 0:
        raise ValueError(f"{tol=} needs to be positive")
    if not (isinstance(max_iter, Integral) and max_iter > 0):
        raise ValueError(f"{max_iter=} needs to be a positive integer")

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


def nanscale(
    X: NDArray[floating[Any]], scale: float
) -> NDArray[floating[Any]]:
    """Scale all non-nan values in an array.

    Args:
        X: Input array to be scaled, possibly containing numpy.nan values.
        scale: Positive scale parameter.

    Returns:
        A scaled version of the input array.

    """
    X = check_array(X, ensure_all_finite="allow-nan")
    if scale <= 0.0:
        raise ValueError(f"{scale=} is required to be positive")

    Y = full(X.shape, nan, dtype=X.dtype)
    divide(
        X,
        scale,
        out=Y,
        where=logical_not(isnan(X)),
    )

    return Y
