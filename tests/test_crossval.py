import numpy as np
import pytest
from numpy.random import default_rng
from numpy.testing import assert_allclose

from solrcmf import (
    ElementwiseFolds,
    SolrCMF,
    SolrCMFCV,
    best_random_init,
    simulate,
)
from solrcmf.base import Entity, ViewDesc
from solrcmf.crossval import _select_best_index, _select_best_penalized_runs

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
    """Require vs and ds for custom CV initialization."""
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


def test_penalized_cv_selects_highest_scoring_restart():
    """Negative-error scores are maximized when selecting a restart."""
    scores = np.array(
        [
            -1.0,
            -1.0,
            -0.5,
            -0.4,
            -0.7,
            -0.6,
            -3.0,
            -2.0,
            -1.0,
            -1.2,
            -0.9,
            -1.5,
        ]
    )

    best_runs, selected_scores = _select_best_penalized_runs(
        scores,
        n_params=2,
        n_reps=3,
        n_folds=2,
    )

    assert_allclose(best_runs, [1, 1])
    assert_allclose(selected_scores, [[-0.5, -0.4], [-1.0, -1.2]])


def test_one_standard_error_selection_uses_sem():
    """The 1-SE candidate threshold uses standard error, not deviation."""
    results = {
        "mean_neg_mean_squared_error": np.array([1.0, 0.9]),
        "std_neg_mean_squared_error": np.array([0.2, 0.2]),
        "sem_neg_mean_squared_error": np.array([0.05, 0.05]),
        "structural_zeros": np.array([0, 10]),
        "factor_zeros": np.array([0, 0]),
    }

    assert (
        _select_best_index(
            results,
            score="neg_mean_squared_error",
            refit="1se_debiased",
        )
        == 0
    )


def test_init_kwargs_not_mutated():
    """Fitting CV leaves caller-owned initialization settings unchanged."""
    X = _simdata()["xs"]
    init_kwargs = {"rng": 0, "repetitions": 2}
    cv = SolrCMFCV(
        structure_penalty=0.1,
        max_rank=2,
        cv=2,
        init_kwargs=init_kwargs,
        max_iter=200,
        n_jobs=1,
    )

    cv.fit(X)

    assert init_kwargs == {"rng": 0, "repetitions": 2}
    assert cv.init_kwargs == init_kwargs


@pytest.mark.parametrize("repetitions", [0, -1, 1.5, True])
def test_invalid_repetitions_raise(repetitions):
    """The number of random restarts must be a positive integer."""
    X = _simdata()["xs"]
    cv = SolrCMFCV(
        structure_penalty=0.1,
        max_rank=2,
        cv=2,
        init_kwargs={"rng": 0, "repetitions": repetitions},
        n_jobs=1,
    )

    with pytest.raises(ValueError, match="repetitions"):
        cv.fit(X)


def test_penalized_cv_integer_seed_is_reproducible():
    """Integer seeds reproduce fold scores and the final penalized fit."""
    X = _simdata()["xs"]

    def fit_cv():
        return SolrCMFCV(
            structure_penalty=0.1,
            max_rank=2,
            cv=ElementwiseFolds(2, shuffle=False),
            cv_strategy="penalized_cv",
            refit="mean_penalized",
            init_kwargs={"rng": 42, "repetitions": 2},
            max_iter=500,
            n_jobs=1,
        ).fit(X)

    first = fit_cv()
    second = fit_cv()

    for fold in range(2):
        key = f"neg_mean_squared_error_fold{fold}"
        assert_allclose(first.cv_results_[key], second.cv_results_[key])
    assert_allclose(
        first.cv_results_["sem_neg_mean_squared_error"],
        first.cv_results_["std_neg_mean_squared_error"] / np.sqrt(2),
    )
    for k in first.best_estimator_.ds_:
        assert_allclose(
            first.best_estimator_.ds_[k],
            second.best_estimator_.ds_[k],
        )
