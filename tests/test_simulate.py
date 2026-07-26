import numpy as np
import pytest
from numpy import diag, eye
from numpy.linalg import norm
from numpy.random import default_rng
from numpy.testing import assert_allclose
from numpy.typing import NDArray

from solrcmf.base import Entity, ViewDesc
from solrcmf.simulate import simulate

VIEWDIMS: dict[Entity, int] = {0: 10, 1: 20, 2: 30}
FACTOR_SCALES: dict[ViewDesc, list[float]] = {(0, 1): [1, 2], (0, 2): [3, 4]}
MAX_RANK = 2


def _check_output_structure(data, viewdims, factor_scales):
    """Assert keys and shapes are correct."""
    assert data["xs"].keys() == factor_scales.keys()
    assert data["xs_truth"].keys() == factor_scales.keys()

    views = set(k[i] for k in factor_scales for i in range(2))
    assert data["vs"].keys() == views

    max_rank = len(next(iter(factor_scales.values())))

    for k in factor_scales:
        expected_shape = (viewdims[k[0]], viewdims[k[1]])
        assert data["xs"][k].shape == expected_shape
        assert data["xs_truth"][k].shape == expected_shape

    for v in views:
        assert data["vs"][v].shape == (viewdims[v], max_rank)


def _check_orthonormality(data):
    """Assert factor matrices have orthonormal columns."""
    for v, mat in data["vs"].items():
        assert_allclose(
            mat.T @ mat,
            eye(mat.shape[1]),
            atol=1e-10,
            err_msg=f"vs[{v}] columns are not orthonormal",
        )


def _check_xs_truth(data, factor_scales):
    """Assert xs_truth matches the factor model exactly."""
    vs = data["vs"]
    for k, d in factor_scales.items():
        expected = vs[k[0]] @ diag(d) @ vs[k[1]].T
        assert_allclose(data["xs_truth"][k], expected, atol=1e-10)


def test_simulate_dense_no_external_rng():
    data = simulate(
        viewdims=VIEWDIMS,
        factor_scales=FACTOR_SCALES,
        rng=None,
    )

    _check_output_structure(data, VIEWDIMS, FACTOR_SCALES)
    _check_orthonormality(data)
    _check_xs_truth(data, FACTOR_SCALES)


def test_simulate_dense():
    data = simulate(
        viewdims=VIEWDIMS,
        factor_scales=FACTOR_SCALES,
        rng=default_rng(42),
    )

    _check_output_structure(data, VIEWDIMS, FACTOR_SCALES)
    _check_orthonormality(data)
    _check_xs_truth(data, FACTOR_SCALES)


def test_simulate_dense_varying_snr():
    rng1, rng2 = default_rng(0), default_rng(1)

    data_high_snr = simulate(
        viewdims=VIEWDIMS,
        factor_scales=FACTOR_SCALES,
        snr={(0, 1): 10, (0, 2): 10},
        rng=rng1,
    )
    data_low_snr = simulate(
        viewdims=VIEWDIMS,
        factor_scales=FACTOR_SCALES,
        snr={(0, 1): 0.1, (0, 2): 0.1},
        rng=rng2,
    )

    _check_output_structure(data_high_snr, VIEWDIMS, FACTOR_SCALES)
    _check_output_structure(data_low_snr, VIEWDIMS, FACTOR_SCALES)

    for k in FACTOR_SCALES:
        noise_high = norm(
            data_high_snr["xs"][k] - data_high_snr["xs_truth"][k]
        )
        noise_low = norm(data_low_snr["xs"][k] - data_low_snr["xs_truth"][k])
        signal_high = norm(data_high_snr["xs_truth"][k])
        signal_low = norm(data_low_snr["xs_truth"][k])
        assert noise_high / signal_high < noise_low / signal_low


def test_simulate_dense_explicit_scale():
    data = simulate(
        viewdims=VIEWDIMS,
        factor_scales=FACTOR_SCALES,
        scales={(0, 1): 1, (0, 2): 1},
        rng=default_rng(42),
    )

    _check_output_structure(data, VIEWDIMS, FACTOR_SCALES)
    _check_orthonormality(data)
    _check_xs_truth(data, FACTOR_SCALES)


def test_simulate_sparse():
    factor_sparsity: dict[Entity, float] = {0: 0.5, 1: 0.6, 2: 0.7}
    data = simulate(
        viewdims=VIEWDIMS,
        factor_scales=FACTOR_SCALES,
        factor_sparsity=factor_sparsity,
        rng=default_rng(42),
    )

    _check_output_structure(data, VIEWDIMS, FACTOR_SCALES)
    _check_orthonormality(data)

    # Each column should have approximately sparsity * p non-zero entries
    for v, mat in data["vs"].items():
        p = VIEWDIMS[v]
        expected_nnz = factor_sparsity[v] * p
        for col in range(MAX_RANK):
            nnz = (mat[:, col] != 0.0).sum()
            assert abs(nnz - expected_nnz) <= 2, (
                f"vs[{v}] col {col}: expected ~{expected_nnz} non-zeros,"
                f" got {nnz}"
            )


@pytest.mark.parametrize(
    ("p", "rank", "sparsity"),
    [(15, 3, 0.4), (20, 5, 0.3), (50, 10, 0.2)],
)
@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_simulate_sparse_high_rank_postconditions(p, rank, sparsity, seed):
    """Higher-rank sparse simulations are exactly sparse and orthogonal."""
    factor_scales: dict[ViewDesc, NDArray[np.float64]] = {
        (0, 1): np.arange(1, rank + 1, dtype=np.float64)
    }
    data = simulate(
        viewdims={0: p, 1: p},
        factor_scales=factor_scales,
        factor_sparsity={0: sparsity, 1: sparsity},
        rng=default_rng(seed),
    )

    expected_nnz = int(np.ceil(sparsity * p))
    for factors in data["vs"].values():
        assert_allclose(factors.T @ factors, eye(rank), atol=1e-12)
        assert_allclose(
            (factors != 0.0).sum(axis=0),
            np.full(rank, expected_nnz),
        )


@pytest.mark.parametrize("sparsity", [0.0, -0.1, 1.1, np.inf, np.nan])
def test_simulate_rejects_invalid_factor_sparsity(sparsity):
    with pytest.raises(ValueError, match=r"must be finite and in \(0, 1\]"):
        simulate(
            viewdims={0: 5, 1: 5},
            factor_scales={(0, 1): [1.0]},
            factor_sparsity={0: sparsity, 1: 0.5},
            rng=default_rng(0),
        )


def test_simulate_rejects_view_dimension_below_rank():
    with pytest.raises(ValueError, match="greater than or equal to max_rank"):
        simulate(
            viewdims={0: 2, 1: 3},
            factor_scales={(0, 1): [1.0, 2.0, 3.0]},
            rng=default_rng(0),
        )


def test_simulate_rejects_empty_factor_scales():
    with pytest.raises(ValueError, match="at least one matrix"):
        simulate(
            viewdims={},
            factor_scales={},
            rng=default_rng(0),
        )


def test_simulate_rejects_infeasible_sparse_support():
    with pytest.raises(ValueError, match="Cannot construct"):
        simulate(
            viewdims={0: 5, 1: 5},
            factor_scales={(0, 1): [1.0, 2.0, 3.0, 4.0]},
            factor_sparsity={0: 0.6, 1: 0.6},
            rng=default_rng(0),
        )
