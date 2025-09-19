"""Functions to simulate synthetic data.

This module provides functions to simulate synthetic data.
"""

from collections.abc import Hashable
from typing import Mapping

from numpy import argsort, atleast_1d, diag, float64, floor, sqrt, sum
from numpy.linalg import qr
from numpy.random import Generator, default_rng
from numpy.typing import ArrayLike, NDArray

from .base import ViewDesc


def _sparse_v(
    p: int,
    max_rank: int,
    sparsity: float,
    rng: Generator,
) -> NDArray[float64]:
    """Generate orthogonal sparse factors.

    Generate max_rank orthgonal columns of sparse factors of dimension p.
    Initially, each column is generated element-wise from a standard normal
    distribution. The (1 - sparsity) proportion of smallest values is then set
    to zero and a masked Gram-Schmidt algorithm is used to generate orthogonal
    factors which retain the initial zero pattern.

    Caveat: Note that if sparsity * p < max_rank, then there is a non-zero
    chance that the same set of non-zero entries is chosen across the max_rank
    columns. It is then impossible to generate max_rank orthogonal factors.
    Since SolrCMF operates in the p >> max_rank regime, this case is not
    explicitely checked for.

    Args:
        p: The dimension of the factors.
        max_rank: The number of factors.
        sparsity: The proportion of non-zero entries.
        rng: A random number generator.

    Returns:
        The generated factors as a numpy.ndarray of shape (p, max_rank).

    """
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
    viewdims: Mapping[Hashable, int],
    factor_scales: Mapping[ViewDesc, ArrayLike],
    scales: Mapping[ViewDesc, float] | None = None,
    snr: Mapping[ViewDesc, float] | float = 1.0,
    factor_sparsity: Mapping[Hashable, float] | None = None,
    rng: Generator | None = None,
) -> Mapping[
    str,
    Mapping[ViewDesc, NDArray[float64]] | Mapping[Hashable, NDArray[float64]],
]:
    """Simulate synthetic data confirming to the SolrCMF model.

    Args:
        viewdims: A mapping of views to view dimensions.
        factor_scales: A mapping of view descriptors to scalars or 1D arrays
            describing the strength of each factor.
        scales: A mapping of view descriptors to positive scalars which scale
            the factor_scales of the corresponding view descriptor.
        snr: A mapping of view descriptors to signal-to-noise ratios.
        factor_sparsity: `None` if factors should be simulated without sparsity
            and a mapping of views to sparsity proportions otherwise.
        rng: A random number generator or `None` to use the default random
            number generator.

    Returns:
        A dictionary containing the following keys

            - "xs_truth": Containing the groundtruth (noise-less) data for each
            view descriptor.
            - "xs": Containing the simulated data for each view descriptor.
            - "vs": Containing the groundtruth factors for each view.

    """
    if rng is None:
        rng = default_rng()

    factor_scales_ = {k: atleast_1d(v) for k, v in factor_scales.items()}
    shapes = [s.shape for s in factor_scales_.values()]
    assert all([len(s) == 1 and s == shapes[0] for s in shapes]), (
        "Each value in 'factor_scales' needs to be a of shape (max_rank,)"
    )
    max_rank = shapes[0][0]
    assert all(len(k) >= 2 for k in factor_scales_.keys()), (
        "Each key in 'factor_scales' needs to be a tuple of two"
        " or more integers"
    )

    views = set([k[i] for k in factor_scales_.keys() for i in range(2)])
    assert views == viewdims.keys(), (
        "The keys of 'viewdims' need to appear in the first two entries"
        " of the keys of 'factor_scales'"
    )

    if scales is None:
        scales = {k: 1.0 for k in factor_scales_.keys()}

    assert scales.keys() == factor_scales_.keys(), (
        "'scales' needs to be compatible with 'factor_scales'"
    )
    assert all(s > 0.0 for s in scales.values()), (
        "Each value in 'scales' needs to be positive"
    )

    if isinstance(snr, Mapping):
        assert snr.keys() == factor_scales_.keys(), (
            "'snr' needs to be compatible with 'factor_scales'"
        )
        assert all(s > 0.0 for s in snr.values()), (
            "Each value in 'snr' needs to be positive"
        )
    else:
        assert snr > 0.0, "'snr' needs to be positive"
        snr = {k: snr for k in factor_scales_.keys()}

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
        for k, d in factor_scales_.items()
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
