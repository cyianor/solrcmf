from numpy.typing import ArrayLike
from numpy import diag, sum, sqrt, argsort, floor, atleast_1d
from numpy.linalg import qr
from numpy.random import Generator, default_rng
from collections.abc import Hashable

from .base import ViewDesc


def _sparse_v(
    p: int,
    max_rank: int,
    sparsity: float,
    rng: Generator,
):
    # Random matrix
    v = rng.standard_normal((p, max_rank))
    # Set smallest values in each column to zero
    order = argsort(abs(v), axis=0)
    zero_indices = order[: int(floor((1.0 - sparsity) * p)), :]
    for i in range(max_rank):
        v[zero_indices[:, i], i] = 0.0

    # Orthonormalise the columns while keeping their respective zero pattern
    for i in range(max_rank):
        for j in range(i):
            mask = v[:, i] != 0.0
            if all(v[mask, j] == 0.0):
                continue

            v[mask, i] -= (
                sum(v[:, i] * v[:, j]) / sum(v[mask, j] ** 2) * v[mask, j]
            )

        v[:, i] /= sqrt(sum(v[:, i] ** 2))

    return v


def simulate(
    *,
    viewdims: dict[Hashable, int],
    factor_scales: dict[ViewDesc, ArrayLike],
    scales: dict[ViewDesc, float] | None = None,
    snr: dict[ViewDesc, float] | float = 1.0,
    factor_sparsity: dict[Hashable, float] | None = None,
    rng: Generator | None = None,
):
    if rng is None:
        rng = default_rng()

    factor_scales = {
        k: atleast_1d(v) for k, v in factor_scales.items()
    }
    shapes = [s.shape for s in factor_scales.values()]
    assert all(
        [len(s) == 1 and s == shapes[0] for s in shapes]
    ), "Each value in 'factor_scales' needs to be a of shape (max_rank,)"
    max_rank = shapes[0][0]
    assert all(len(k) >= 2 for k in factor_scales.keys()), (
        "Each key in 'factor_scales' needs to be a tuple of two"
        " or more integers"
    )

    views = set([k[i] for k in factor_scales.keys() for i in range(2)])
    assert views == viewdims.keys(), (
        "The keys of 'viewdims' need to appear in the first two entries"
        " of the keys of 'factor_scales'"
    )

    if scales is None:
        scales = {k: 1.0 for k in factor_scales.keys()}

    assert (
        scales.keys() == factor_scales.keys()
    ), "'scales' needs to be compatible with 'factor_scales'"
    assert all(
        s > 0.0 for s in scales.values()
    ), "Each value in 'scales' needs to be positive"

    if isinstance(snr, dict):
        assert (
            snr.keys() == factor_scales.keys()
        ), "'snr' needs to be compatible with 'factor_scales'"
        assert all(
            s > 0.0 for s in snr.values()
        ), "Each value in 'snr' needs to be positive"
    else:
        assert snr > 0.0, "'snr' needs to be positive"
        snr = {k: snr for k in factor_scales.keys()}

    if factor_sparsity is None:
        vs = {
            k: qr(rng.standard_normal((p, max_rank))).Q
            for k, p in viewdims.items()
        }
    else:
        assert (
            len(factor_sparsity) == len(views)
            and factor_sparsity.keys() == views
        ), "'factor_sparsity' needs to be provided for each view"

        vs = {
            k: _sparse_v(p, max_rank, factor_sparsity[k], rng)
            for k, p in viewdims.items()
        }

    xs_truth = {
        k: scales[k] * vs[k[0]] @ diag(d) @ vs[k[1]].T
        for k, d in factor_scales.items()
    }

    xs = {
        k: x
        + sqrt(sum(x**2) / (snr[k] * x.size))
        * rng.standard_normal(size=x.shape)
        for k, x in xs_truth.items()
    }

    return {
        "xs_truth": xs_truth,
        "xs": xs,
        "vs": vs,
    }
