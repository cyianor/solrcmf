"""Functions to simulate synthetic data.

This module provides functions to simulate synthetic data.
"""

from collections.abc import Mapping
from numbers import Integral, Real
from typing import TypedDict

from numpy import (
    abs,
    asarray,
    atleast_1d,
    ceil,
    diag,
    eye,
    float64,
    isfinite,
    ix_,
    sqrt,
    sum,
    zeros,
)
from numpy.linalg import qr
from numpy.random import Generator, default_rng
from numpy.typing import ArrayLike, NDArray

from .base import Entity, ViewDesc


def _sparse_v(
    p: int,
    max_rank: int,
    sparsity: float,
    rng: Generator,
) -> NDArray[float64]:
    """Generate orthogonal sparse factors.

    Columns are split among disjoint row-support blocks. Within each
    support block, a thin QR decomposition produces orthonormal
    columns. This construction preserves the exact generated support
    and makes columns from different blocks orthogonal by construction.

    Args:
        p: The dimension of the factors.
        max_rank: The number of factors.
        sparsity: The proportion of non-zero entries.
        rng: A random number generator.

    Returns:
        The generated factors as a numpy.ndarray of shape
        (p, max_rank).

    Raises:
        ValueError: If the requested number of nonzero entries cannot be
            represented by disjoint support blocks.
        RuntimeError: If numerical postconditions fail unexpectedly.

    """
    n_nonzero = int(ceil(sparsity * p))
    min_support_blocks = (max_rank + n_nonzero - 1) // n_nonzero
    max_support_blocks = p // n_nonzero
    if min_support_blocks > max_support_blocks:
        raise ValueError(
            "Cannot construct sparse orthogonal factors with"
            f" dimension={p}, rank={max_rank}, and sparsity={sparsity}."
            " Increase 'factor_sparsity' or the view dimension, or decrease"
            " the rank."
        )

    n_support_blocks = min(max_rank, max_support_blocks)
    row_order = rng.permutation(p)
    column_order = rng.permutation(max_rank)
    base_group_size, remainder = divmod(max_rank, n_support_blocks)

    v = zeros((p, max_rank), dtype=float64)
    column_offset = 0
    for block_idx in range(n_support_blocks):
        group_size = base_group_size + (block_idx < remainder)
        support = row_order[
            block_idx * n_nonzero : (block_idx + 1) * n_nonzero
        ]
        columns = column_order[column_offset : column_offset + group_size]
        column_offset += group_size

        for _ in range(10):
            q = qr(rng.standard_normal((n_nonzero, group_size))).Q
            if (q != 0.0).all() and isfinite(q).all():
                break
        else:
            raise RuntimeError(
                "Failed to generate finite sparse factors with the intended"
                " support."
            )
        v[ix_(support, columns)] = q

    orthogonality_error = abs(v.T @ v - eye(max_rank)).max()
    if (
        not isfinite(v).all()
        or orthogonality_error > 1e-12
        or not ((v != 0.0).sum(axis=0) == n_nonzero).all()
    ):
        raise RuntimeError(
            "Sparse factor generation failed its orthogonality or support"
            " postcondition."
        )

    return v


class SimulationResult(TypedDict):
    """Return type of simulate."""

    xs_truth: dict[ViewDesc, NDArray[float64]]
    xs: dict[ViewDesc, NDArray[float64]]
    vs: dict[Entity, NDArray[float64]]


def simulate(
    *,
    viewdims: Mapping[Entity, int],
    factor_scales: Mapping[ViewDesc, ArrayLike],
    scales: Mapping[ViewDesc, float] | None = None,
    snr: Mapping[ViewDesc, float] | float = 1.0,
    factor_sparsity: Mapping[Entity, float] | None = None,
    rng: Generator | None = None,
) -> SimulationResult:
    """Simulate synthetic data confirming to the SolrCMF model.

    Args:
        viewdims: A mapping of views to view dimensions.
        factor_scales: A mapping of view descriptors to scalars or
            1D arrays describing the strength of each factor.
        scales: A mapping of view descriptors to positive scalars
            which scale the factor_scales of the corresponding
            view descriptor.
        snr: A mapping of view descriptors to signal-to-noise ratios.
        factor_sparsity: `None` if factors should be simulated without
            sparsity and a mapping of views to sparsity proportions
            otherwise.
        rng: A random number generator or `None` to use the default
            random number generator.

    Returns:
        A dictionary containing the following keys

            - "xs_truth": A dictionary of view descriptors to groundtruth
            (noise-less) data.
            - "xs": A dictionary of view descriptors to the simulated
            data.
            - "vs": A dictionary of views to the the groundtruth factors.

    """
    if rng is None:
        rng = default_rng()

    if not factor_scales:
        raise ValueError("'factor_scales' must contain at least one matrix")
    if not all(isinstance(k, tuple) and len(k) >= 2 for k in factor_scales):
        raise ValueError(
            "Each key in 'factor_scales' needs to be a tuple of two"
            " or more entries"
        )

    factor_scales_ = {
        k: asarray(atleast_1d(v), dtype=float64)
        for k, v in factor_scales.items()
    }
    shapes = [s.shape for s in factor_scales_.values()]
    if not all(len(s) == 1 and s == shapes[0] for s in shapes):
        raise ValueError(
            "Each value in 'factor_scales' needs to be of shape (max_rank,)"
        )
    max_rank = shapes[0][0]
    if max_rank == 0:
        raise ValueError("'factor_scales' values must not be empty")
    if not all(isfinite(s).all() for s in factor_scales_.values()):
        raise ValueError("Each value in 'factor_scales' needs to be finite")

    views = {k[i] for k in factor_scales_ for i in range(2)}
    if views != viewdims.keys():
        raise ValueError(
            "The keys of 'viewdims' need to appear in the first two entries"
            " of the keys of 'factor_scales'"
        )
    invalid_viewdims = {
        k: p
        for k, p in viewdims.items()
        if not isinstance(p, Integral) or p < max_rank
    }
    if invalid_viewdims:
        raise ValueError(
            "Each view dimension must be an integer greater than or equal to"
            f" max_rank={max_rank}. Invalid dimensions: {invalid_viewdims}"
        )

    if scales is None:
        scales = {k: 1.0 for k in factor_scales_.keys()}

    if scales.keys() != factor_scales_.keys():
        raise ValueError(
            "'scales' needs to be compatible with 'factor_scales'"
        )
    if not all(isfinite(s) and s > 0.0 for s in scales.values()):
        raise ValueError(
            "Each value in 'scales' needs to be positive and finite"
        )

    if isinstance(snr, (int, float)):
        if snr <= 0.0:
            raise ValueError("'snr' needs to be positive")
        snr = {k: float(snr) for k in factor_scales_.keys()}
    else:
        if snr.keys() != factor_scales_.keys():
            raise ValueError(
                "'snr' needs to be compatible with 'factor_scales'"
            )
        if not all(s > 0.0 for s in snr.values()):
            raise ValueError("Each value in 'snr' needs to be positive")
        snr = dict(snr)

    if factor_sparsity is None:
        vs = {
            k: qr(rng.standard_normal((p, max_rank))).Q
            for k, p in viewdims.items()
        }
    else:
        if not (
            len(factor_sparsity) == len(views)
            and factor_sparsity.keys() == views
        ):
            raise ValueError(
                "'factor_sparsity' needs to be provided for each view"
            )
        invalid_sparsities = {
            k: s
            for k, s in factor_sparsity.items()
            if not isinstance(s, Real) or not isfinite(s) or not 0.0 < s <= 1.0
        }
        if invalid_sparsities:
            raise ValueError(
                "Each value in 'factor_sparsity' must be finite and in"
                f" (0, 1]. Invalid values: {invalid_sparsities}"
            )

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

    if not all(isfinite(x).all() for x in (*xs_truth.values(), *xs.values())):
        raise ValueError(
            "Simulation produced non-finite values; check scales and signal"
            " magnitudes"
        )

    return {
        "xs_truth": xs_truth,
        "xs": xs,
        "vs": vs,
    }
