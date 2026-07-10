"""Metrics to assess data fidelity of a multi-matrix estimate."""

from numpy import float64, intp, isnan, logical_not, nansum, nanvar, sum
from numpy.typing import NDArray

from .base import ViewDesc


def neg_mean_squared_error(
    xs: dict[ViewDesc, NDArray[float64]],
    xhats: dict[ViewDesc, NDArray[float64]],
    *,
    indices: dict[ViewDesc, NDArray[intp]] | None = None,
) -> float:
    """Compute the negative Mean-Squared-Error (MSE).

    nan-values are ignored during computations.

    Args:
        xs: ground-truth data
        xhats: estimate containing same entries as `xs` with matching shapes
        indices: indices into `xs` and `xhats`. If provided, the negative MSE
            is only computed on these indices.

    Returns:
        the negative MSE

    """
    if indices is None:
        n_sums = [
            (sum(logical_not(isnan(xs[k]))), nansum((xs[k] - xhat) ** 2))
            for k, xhat in xhats.items()
        ]
    else:
        n_sums = [
            (
                sum(logical_not(isnan(xs[k].flat[indices[k]]))),
                nansum((xs[k].flat[indices[k]] - xhat.flat[indices[k]]) ** 2),
            )
            for k, xhat in xhats.items()
        ]

    return -float(sum([s for _, s in n_sums]) / sum([n for n, _ in n_sums]))


def weighted_neg_mean_squared_error(
    xs: dict[ViewDesc, NDArray[float64]],
    xhats: dict[ViewDesc, NDArray[float64]],
    *,
    indices: dict[ViewDesc, NDArray[intp]] | None = None,
) -> float:
    """Compute the variance-weighted negative sum of squared errors.

    For each matrix, the sum of squared errors between `xs[k]` and
    `xhats[k]` is divided by the variance of the entries of `xs[k]`.
    The weighted sums are added up across matrices; no division by the
    number of entries takes place.

    nan-values are ignored during computations.

    Args:
        xs: ground-truth data
        xhats: estimate containing same entries as `xs` with matching shapes
        indices: indices into `xs` and `xhats`. If provided, the error
            is only computed on these indices.

    Returns:
        the variance-weighted negative sum of squared errors

    """
    if indices is None:
        sums = [
            nansum((xs[k] - xhat) ** 2) / nanvar(xs[k])
            for k, xhat in xhats.items()
        ]
    else:
        sums = [
            nansum((xs[k].flat[indices[k]] - xhat.flat[indices[k]]) ** 2)
            / nanvar(xs[k].flat[indices[k]])
            for k, xhat in xhats.items()
        ]

    return -sum(sums)


def neg_sum_squared_error(
    xs: dict[ViewDesc, NDArray[float64]],
    xhats: dict[ViewDesc, NDArray[float64]],
    *,
    indices: dict[ViewDesc, NDArray[intp]] | None = None,
) -> float:
    """Compute the negative squared error.

    nan-values are ignored during computations.

    Args:
        xs: ground-truth data
        xhats: estimate containing same entries as `xs` with matching shapes
        indices: indices into `xs` and `xhats`. If provided, the negative MSE
            is only computed on these indices.

    Returns:
        the negative squared error

    """
    if indices is None:
        sums = [nansum((xs[k] - xhat) ** 2) for k, xhat in xhats.items()]
    else:
        sums = [
            nansum((xs[k].flat[indices[k]] - xhat.flat[indices[k]]) ** 2)
            for k, xhat in xhats.items()
        ]

    return -sum(sums)
