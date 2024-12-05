from numpy.typing import NDArray
from numpy import float64, intp, nansum, sum, isnan, logical_not, nanvar

from .base import ViewDesc


def neg_mean_squared_error(
    xs: dict[ViewDesc, NDArray[float64]],
    xhats: dict[ViewDesc, NDArray[float64]],
    *,
    indices: dict[ViewDesc, NDArray[intp]] | None = None,
):
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
):
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
):
    if indices is None:
        sums = [nansum((xs[k] - xhat) ** 2) for k, xhat in xhats.items()]
    else:
        sums = [
            nansum((xs[k].flat[indices[k]] - xhat.flat[indices[k]]) ** 2)
            for k, xhat in xhats.items()
        ]

    return -sum(sums)
