from numpy import zeros_like
from numpy.random import default_rng
from numpy.testing import assert_allclose

from solrcmf import SolrCMF, best_random_init, simulate
from solrcmf.base import Entity, ViewDesc
from solrcmf.metrics import neg_mean_squared_error

VIEWDIMS: dict[Entity, int] = {0: 20, 1: 30, 2: 40}
FACTOR_SCALES: dict[ViewDesc, list[float]] = {
    (0, 1): [2.0, 1.0],
    (0, 2): [3.0, 1.5],
}


def _simdata(seed=0, snr=10.0):
    return simulate(
        viewdims=VIEWDIMS,
        factor_scales=FACTOR_SCALES,
        snr=snr,
        rng=default_rng(seed),
    )


def test_smoke():
    """Fit converges and produces vs_ and ds_ with correct keys and shapes."""
    X = _simdata()["xs"]
    est = SolrCMF(structure_penalty=0.1, max_rank=3)
    est.fit(X)

    assert set(est.vs_.keys()) == set(VIEWDIMS.keys())
    assert set(est.ds_.keys()) == set(FACTOR_SCALES.keys())
    for v, mat in est.vs_.items():
        assert mat.shape == (VIEWDIMS[v], est.est_max_rank_)


def test_score_beats_zero():
    """Penalized fit with random init scores better than zero prediction."""
    X = _simdata()["xs"]
    est = SolrCMF(
        structure_penalty=0.01, max_rank=2, init_kwargs={"rng": default_rng(0)}
    )
    est.fit(X)

    zero_score = neg_mean_squared_error(
        X, {k: zeros_like(v) for k, v in X.items()}
    )
    assert est.score(X) > zero_score


def test_score_beats_zero_best_random_init():
    """Penalized fit warm-started from best_random_init beats zero."""
    X = _simdata()["xs"]
    init_est = best_random_init(X, max_rank=2, n_inits=5, rng=default_rng(0))

    est = SolrCMF(
        structure_penalty=0.01,
        max_rank=2,
        init="custom",
    )
    est.fit(X, vs=init_est.vs_, ds=init_est.ds_)

    zero_score = neg_mean_squared_error(
        X, {k: zeros_like(v) for k, v in X.items()}
    )
    assert est.score(X) > zero_score


def test_structure_pattern_keys():
    """structure_pattern() returns one entry per data matrix."""
    X = _simdata()["xs"]
    est = SolrCMF(structure_penalty=0.1, max_rank=3)
    est.fit(X)

    assert est.structure_pattern().keys() == X.keys()


def test_max_rank_one():
    """max_rank=1 produces a single-component decomposition."""
    X = _simdata()["xs"]
    est = SolrCMF(structure_penalty=0.1, max_rank=1)
    est.fit(X)

    assert est.est_max_rank_ <= 1
    for v, mat in est.vs_.items():
        assert mat.shape == (VIEWDIMS[v], est.est_max_rank_)


def test_convergence():
    """Algorithm converges well before max_iter with a fixed seed."""
    X = _simdata(seed=42)["xs"]
    est = SolrCMF(structure_penalty=0.1, max_rank=3, max_iter=1000)
    est.fit(X)

    assert est.converged_, f"Did not converge after {est.n_iter_} iterations"
    assert est.n_iter_ < 1000


def test_random_init_accepts_integer_seed():
    """Integer random seeds are accepted and reproduce direct fits."""
    X = _simdata()["xs"]
    first = SolrCMF(
        structure_penalty=0.1,
        max_rank=2,
        init_kwargs={"rng": 42},
    ).fit(X)
    second = SolrCMF(
        structure_penalty=0.1,
        max_rank=2,
        init_kwargs={"rng": 42},
    ).fit(X)

    for k in first.ds_:
        assert_allclose(first.ds_[k], second.ds_[k])


def test_refit_without_factor_sparsity_removes_us():
    """A non-sparse refit does not retain sparse factors from the prior fit."""
    X = _simdata()["xs"]
    est = SolrCMF(
        structure_penalty=0.1,
        max_rank=2,
        factor_penalty=0.01,
        max_iter=1,
    ).fit(X)
    assert hasattr(est, "us_")

    est.set_params(factor_penalty=None)
    est.fit(X)

    assert not hasattr(est, "us_")
    assert est.factor_pattern() is None


def test_refit_without_context_saving_removes_ctx():
    """A refit with save_ctx=False does not retain the prior context."""
    X = _simdata()["xs"]
    est = SolrCMF(
        structure_penalty=0.1,
        max_rank=2,
        max_iter=1,
        save_ctx=True,
    ).fit(X)
    assert hasattr(est, "ctx_")

    est.set_params(save_ctx=False)
    est.fit(X)

    assert not hasattr(est, "ctx_")
