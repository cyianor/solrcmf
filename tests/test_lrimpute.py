import numpy as np
from numpy.testing import assert_allclose

import solrcmf.lrimpute as lrimpute
from solrcmf import LowRankImputation


def test_zero_penalty_handles_unobserved_and_rank_deficient_designs():
    """Unregularised least squares handles singular factor updates."""
    X = np.array(
        [
            [np.nan, np.nan, np.nan],
            [np.nan, 1.0, 2.0],
            [np.nan, 2.0, 4.0],
        ]
    )

    estimator = LowRankImputation(
        penalty=0.0,
        max_rank=2,
        max_iter=20,
        random_state=0,
    ).fit(X)

    reconstruction = estimator.U_ @ estimator.V_.T
    observed = ~np.isnan(X)

    assert_allclose(estimator.U_[0], 0.0)
    assert_allclose(estimator.V_[0], 0.0)
    assert_allclose(reconstruction[observed], X[observed], atol=1e-12)
    assert np.isfinite(estimator.U_).all()
    assert np.isfinite(estimator.V_).all()
    assert np.isfinite(estimator.loss_)


def test_fully_missing_row_and_column_are_stable():
    """Positive regularisation maps unobserved rows and columns to zero."""
    X = np.array(
        [
            [np.nan, np.nan, np.nan],
            [np.nan, 1.0, 2.0],
            [np.nan, 2.0, 4.0],
        ]
    )

    estimator = LowRankImputation(
        penalty=1.0,
        max_rank=2,
        max_iter=20,
        random_state=0,
    ).fit(X)

    assert_allclose(estimator.U_[0], 0.0)
    assert_allclose(estimator.V_[0], 0.0)
    assert np.isfinite(estimator.U_).all()
    assert np.isfinite(estimator.V_).all()
    assert np.isfinite(estimator.loss_)


def test_loss_increase_is_not_convergence(monkeypatch):
    """A negative objective improvement must not satisfy the stopping rule."""
    losses = iter((1.0, 2.0))
    monkeypatch.setattr(
        lrimpute,
        "_compute_loss",
        lambda *args: next(losses),
    )

    estimator = LowRankImputation(
        init="custom",
        max_rank=1,
        max_iter=1,
        tol=1e-6,
    ).fit(
        np.array([[1.0]]),
        U=np.ones((1, 1)),
        V=np.ones((1, 1)),
    )

    assert not estimator.converged_
    assert estimator.n_iter_ == 1
    assert estimator.loss_ == 2.0


def test_incomplete_observation_indices_are_precomputed(monkeypatch):
    """Missingness scans occur once per row and column, not per iteration."""
    flatnonzero = lrimpute.flatnonzero
    calls = 0

    def counted_flatnonzero(values):
        nonlocal calls
        calls += 1
        return flatnonzero(values)

    losses = iter((4.0, 3.0, 2.0, 1.0))
    monkeypatch.setattr(lrimpute, "flatnonzero", counted_flatnonzero)
    monkeypatch.setattr(
        lrimpute,
        "_compute_loss",
        lambda *args: next(losses),
    )

    X = np.array([[1.0, np.nan, 2.0], [np.nan, 3.0, 4.0]])
    estimator = LowRankImputation(
        max_rank=1,
        max_iter=3,
        random_state=0,
    ).fit(X)

    assert not estimator.converged_
    assert estimator.n_iter_ == 3
    assert calls == sum(X.shape)


def test_complete_penalized_updates_use_two_batched_solves(monkeypatch):
    """Complete data share one normal-equation solve per factor update."""
    X = np.array(
        [
            [1.0, 2.0, -1.0],
            [0.5, -2.0, 3.0],
            [4.0, 1.0, 0.0],
            [-1.0, 2.5, 1.5],
        ]
    )
    U = np.array([[1.0, 0.5], [-0.5, 1.0], [2.0, -1.0], [0.25, 0.75]])
    V = np.array([[0.5, 1.0], [1.5, -0.5], [-1.0, 2.0]])
    penalty = 0.5

    identity = np.eye(2)
    expected_U = np.linalg.solve(
        V.T @ V + penalty * identity,
        (X @ V).T,
    ).T
    expected_V = np.linalg.solve(
        expected_U.T @ expected_U + penalty * identity,
        (X.T @ expected_U).T,
    ).T

    solve = lrimpute.solve
    calls = 0

    def counted_solve(A, b):
        nonlocal calls
        calls += 1
        return solve(A, b)

    monkeypatch.setattr(lrimpute, "solve", counted_solve)

    estimator = LowRankImputation(
        penalty=penalty,
        max_rank=2,
        init="custom",
        max_iter=1,
    ).fit(X, U=U.copy(), V=V.copy())

    assert calls == 2
    assert_allclose(estimator.U_, expected_U)
    assert_allclose(estimator.V_, expected_V)


def test_complete_zero_penalty_uses_two_batched_least_squares(monkeypatch):
    """Unregularised complete data use stable batched least squares."""
    X = np.array(
        [
            [1.0, 2.0, -1.0],
            [0.5, -2.0, 3.0],
            [4.0, 1.0, 0.0],
            [-1.0, 2.5, 1.5],
        ]
    )
    U = np.ones((4, 2))
    V = np.array([[0.5, 0.5], [1.5, 1.5], [-1.0, -1.0]])

    expected_U = np.linalg.lstsq(V, X.T, rcond=None)[0].T
    expected_V = np.linalg.lstsq(expected_U, X, rcond=None)[0].T

    least_squares = lrimpute.lstsq
    calls = 0

    def counted_least_squares(A, b, *, rcond):
        nonlocal calls
        calls += 1
        return least_squares(A, b, rcond=rcond)

    monkeypatch.setattr(lrimpute, "lstsq", counted_least_squares)

    estimator = LowRankImputation(
        penalty=0.0,
        max_rank=2,
        init="custom",
        max_iter=1,
    ).fit(X, U=U.copy(), V=V.copy())

    assert calls == 2
    assert_allclose(estimator.U_, expected_U)
    assert_allclose(estimator.V_, expected_V)
