import pytest
from numpy.random import default_rng

from solrcmf import SolrCMF, SolrCMFCV, best_random_init, simulate
from solrcmf.base import Entity, ViewDesc

VIEWDIMS: dict[Entity, int] = {0: 20, 1: 30}
FACTOR_SCALES: dict[ViewDesc, list[float]] = {
    (0, 1): [2.0, 1.0],
}


def _simdata(seed=0, snr=10.0):
    return simulate(
        viewdims=VIEWDIMS,
        factor_scales=FACTOR_SCALES,
        snr=snr,
        rng=default_rng(seed),
    )


def test_custom_init_requires_vs_ds():
    """SolrCMFCV with init='custom' raises ValueError when vs and ds are not provided."""
    X = _simdata()["xs"]
    cv = SolrCMFCV(
        structure_penalty=0.1,
        max_rank=2,
        cv=2,
        init="custom",
    )
    with pytest.raises(ValueError):
        cv.fit(X)


def test_smoke_random_init():
    """SolrCMFCV with random init fits and exposes best_estimator_."""
    X = _simdata()["xs"]
    cv = SolrCMFCV(
        structure_penalty=[0.1, 0.5],
        max_rank=2,
        cv=2,
        init="random",
        init_kwargs={"rng": default_rng(0)},
        max_iter=100,
    )
    cv.fit(X)

    assert hasattr(cv, "best_estimator_")
    assert isinstance(cv.best_estimator_, SolrCMF)


def test_smoke_custom_init():
    """SolrCMFCV with init='custom' accepts vs and ds from a prior fit."""
    X = _simdata()["xs"]
    init_est = best_random_init(X, max_rank=2, n_inits=3, rng=default_rng(0))

    cv = SolrCMFCV(
        structure_penalty=[0.1, 0.5],
        max_rank=2,
        cv=2,
        init="custom",
        max_iter=1000,
    )
    cv.fit(X, vs=[init_est.vs_], ds=[init_est.ds_])

    assert hasattr(cv, "best_estimator_")


def test_cv_result_keys():
    """cv_results_ contains entries for each parameter combination."""
    X = _simdata()["xs"]
    cv = SolrCMFCV(
        structure_penalty=[0.1, 0.5],
        max_rank=2,
        cv=2,
        init="random",
        init_kwargs={"rng": default_rng(0)},
        max_iter=1000,
    )
    cv.fit(X)

    assert "structure_penalty" in cv.cv_results_
    assert len(cv.cv_results_["structure_penalty"]) == 2
