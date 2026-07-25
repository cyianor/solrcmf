from numpy import zeros_like
from numpy.random import default_rng

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


def test_missing_data_unbiased():
    """Signal strengths are recovered without bias under missing data.

    Regression test: the Z update used to treat unobserved entries as
    observed zeros, shrinking the d estimates proportionally to the
    fraction of missing entries.
    """
    from numpy import abs, nan, sort

    sim = simulate(
        viewdims={0: 40, 1: 50},
        factor_scales={(0, 1): [8.0, 5.0]},
        snr=100.0,
        rng=default_rng(7),
    )
    X = {k: x.copy() for k, x in sim["xs"].items()}
    mask = default_rng(2).random(X[(0, 1)].shape) < 0.4
    X[(0, 1)][mask] = nan

    est = SolrCMF(
        structure_penalty=1e-8,
        max_rank=2,
        factor_pruning=False,
        init_kwargs={"rng": default_rng(1)},
        max_iter=5000,
    )
    est.fit(X)

    d = sort(abs(est.ds_[(0, 1)]))[::-1]
    assert abs(d[0] - 8.0) / 8.0 < 0.1
    assert abs(d[1] - 5.0) / 5.0 < 0.1


def test_random_init_rng_seed():
    """init_kwargs 'rng' accepts a plain integer seed and is reproducible."""
    from numpy.testing import assert_allclose

    X = _simdata()["xs"]
    est1 = SolrCMF(structure_penalty=0.1, max_rank=2, init_kwargs={"rng": 42})
    est1.fit(X)
    est2 = SolrCMF(structure_penalty=0.1, max_rank=2, init_kwargs={"rng": 42})
    est2.fit(X)

    for k in est1.ds_:
        assert_allclose(est1.ds_[k], est2.ds_[k])


def test_debiased_refit_reduces_inactive_factors():
    """Custom init with reduce_max_rank drops globally inactive factors.

    Regression test: swapped pattern guards in FromFormerInitializer used
    to leave the structure pattern unreduced, crashing the debiased refit
    with a broadcasting error.
    """
    from numpy import array
    from numpy.linalg import qr

    X = _simdata()["xs"]
    rng = default_rng(0)
    vs = {v: qr(rng.standard_normal((p, 3))).Q for v, p in VIEWDIMS.items()}
    ds = {
        (0, 1): array([2.0, 0.0, 1.0]),
        (0, 2): array([3.0, 0.0, 1.5]),
    }
    structure_pattern = {k: d != 0.0 for k, d in ds.items()}

    est = SolrCMF(
        init="custom",
        init_kwargs={"reduce_max_rank": True},
        factor_pruning=False,
    )
    est.fit(X, structure_pattern=structure_pattern, vs=vs, ds=ds)

    assert est.est_max_rank_ == 2
    for v, mat in est.vs_.items():
        assert mat.shape == (VIEWDIMS[v], 2)
