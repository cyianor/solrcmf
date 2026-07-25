from itertools import permutations

import numpy as np
import pytest
from numpy.linalg import norm
from numpy.random import default_rng
from numpy.testing import assert_allclose
from numpy.typing import NDArray

from solrcmf import SolrCMF, best_random_init, simulate
from solrcmf.base import Entity, ViewDesc, update


@pytest.mark.parametrize("use_nan", [False, True])
def test_z_update_does_not_shrink_unobserved_entries(use_nan):
    """Unobserved Z entries equal the current low-rank/dual value."""
    X: dict[ViewDesc, NDArray[np.float64]] = {
        (0, 1): np.arange(1, 7, dtype=np.float64).reshape(2, 3)
    }
    indices: dict[ViewDesc, NDArray[np.intp]] | None = None
    if use_nan:
        X[(0, 1)][0, 1:] = np.nan
        X[(0, 1)][1, :] = np.nan
    else:
        indices = {(0, 1): np.array([0], dtype=np.intp)}

    est = SolrCMF(
        structure_penalty=0.0,
        max_rank=1,
        factor_pruning=False,
        init_kwargs={"rng": default_rng(0)},
    )
    ctx = est._setup(X, indices=indices)

    ctx.blocks.v[0].value = np.array([[1.0], [2.0]])
    ctx.blocks.v[1].value = np.array([[1.0], [-1.0], [0.5]])
    ctx.blocks.d[(0, 1)].value = np.array([2.0])
    ctx.constraints.mean_structure[(0, 1)].value = np.array(
        [[0.5, 1.0, -0.5], [0.0, -1.0, 0.25]]
    )

    low_rank = (
        ctx.blocks.v[0].value
        * ctx.blocks.d[(0, 1)].value
        @ ctx.blocks.v[1].value.T
        - ctx.constraints.mean_structure[(0, 1)].value
    )
    expected = low_rank.copy()
    observed = ctx.params.flat_indices[(0, 1)]
    expected.flat[observed] = (
        ctx.params.rho * low_rank.flat[observed]
        + X[(0, 1)].flat[observed]
    ) / (1.0 + ctx.params.rho)

    update(ctx.blocks.z[(0, 1)], ctx)

    assert_allclose(ctx.blocks.z[(0, 1)].value, expected)
    unobserved = np.setdiff1d(np.arange(expected.size), observed)
    assert_allclose(
        ctx.blocks.z[(0, 1)].value.flat[unobserved],
        low_rank.flat[unobserved],
    )


def test_missing_data_recovers_truth_from_true_initialization():
    """An exact solution remains a fixed point with missing entries."""
    factor_scales: dict[ViewDesc, NDArray[np.float64]] = {
        (0, 1): np.array([5.0, 3.0, 0.0]),
        (0, 2): np.array([4.0, 0.0, 2.0]),
        (1, 2): np.array([0.0, 3.5, 2.5]),
    }
    simulated = simulate(
        viewdims={0: 24, 1: 22, 2: 20},
        factor_scales=factor_scales,
        snr=10.0,
        rng=default_rng(1),
    )

    rng = default_rng(2)
    X = {k: x.copy() for k, x in simulated["xs_truth"].items()}
    for x in X.values():
        x[rng.random(x.shape) < 0.2] = np.nan

    est = SolrCMF(
        structure_penalty=0.0,
        max_rank=3,
        factor_pruning=False,
        init="custom",
        max_iter=100,
        abs_tol=1e-12,
        rel_tol=1e-12,
    ).fit(
        X,
        vs=simulated["vs"],
        ds=factor_scales,
    )

    reconstructed = est.transform(X)
    relative_error = np.sqrt(
        sum(
            norm(reconstructed[k] - simulated["xs_truth"][k]) ** 2
            for k in factor_scales
        )
        / sum(norm(x) ** 2 for x in simulated["xs_truth"].values())
    )

    assert est.converged_
    assert relative_error < 1e-10
    for k, truth in factor_scales.items():
        assert_allclose(est.ds_[k], truth, atol=1e-10)


def test_missing_noisy_data_recovers_estimated_quantities():
    """Randomly initialized estimation recovers scales and factors."""
    viewdims: dict[Entity, int] = {0: 40, 1: 35, 2: 30}
    factor_scales: dict[ViewDesc, NDArray[np.float64]] = {
        (0, 1): np.array([5.0, 3.0, 0.0]),
        (0, 2): np.array([4.0, 0.0, 2.0]),
        (1, 2): np.array([0.0, 3.5, 2.5]),
    }
    rng = default_rng(0)
    simulated = simulate(
        viewdims=viewdims,
        factor_scales=factor_scales,
        snr=10.0,
        rng=rng,
    )

    X = {k: x.copy() for k, x in simulated["xs"].items()}
    heldout = {}
    for k, x in X.items():
        heldout[k] = rng.random(x.shape) < 0.2
        x[heldout[k]] = np.nan

    est = best_random_init(
        X,
        max_rank=3,
        n_inits=3,
        n_jobs=1,
        rng=0,
        max_iter=3000,
        abs_tol=1e-8,
        rel_tol=1e-8,
    )

    permutation = max(
        permutations(range(3)),
        key=lambda perm: sum(
            abs(
                simulated["vs"][view][:, component]
                @ est.vs_[view][:, perm[component]]
            )
            for view in viewdims
            for component in range(3)
        ),
    )

    truth_d = np.concatenate(
        [np.abs(d) for d in factor_scales.values()]
    )
    estimated_d = np.concatenate(
        [np.abs(est.ds_[k][list(permutation)]) for k in factor_scales]
    )
    active = truth_d != 0.0
    relative_d_error = norm(
        estimated_d[active] - truth_d[active]
    ) / norm(truth_d[active])

    minimum_factor_correlation = min(
        abs(
            simulated["vs"][view][:, component]
            @ est.vs_[view][:, permutation[component]]
        )
        for view in viewdims
        for component in range(3)
    )

    reconstructed = est.transform(X)
    heldout_relative_error = np.sqrt(
        sum(
            norm(
                (
                    reconstructed[k] - simulated["xs_truth"][k]
                )[heldout[k]]
            )
            ** 2
            for k in factor_scales
        )
        / sum(
            norm(simulated["xs_truth"][k][heldout[k]]) ** 2
            for k in factor_scales
        )
    )

    assert est.converged_
    assert relative_d_error < 0.03
    assert minimum_factor_correlation > 0.98
    assert heldout_relative_error < 0.15
