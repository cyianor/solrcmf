from numpy import diag, eye
from numpy.linalg import norm
from numpy.random import default_rng
from numpy.testing import assert_allclose

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
