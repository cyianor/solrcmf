"""Tools for low-rank imputation.

This module provides functionality to compute estimates to single
matrices with missing values under a low-rank assumption. This can
be used to impute the missing values.
"""

from numbers import Integral, Real
from warnings import warn

from numpy import (
    asarray,
    flatnonzero,
    float32,
    float64,
    isnan,
    nansum,
)
from numpy.linalg import solve
from numpy.random import RandomState
from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator
from sklearn.utils._param_validation import Interval, StrOptions
from sklearn.utils.validation import check_array, check_random_state


class LowRankImputation(BaseEstimator):
    """Low-rank matrix imputation via alternating ridge regression.

    Factorises an incomplete matrix X as U @ V.T, where U and V are
    estimated by alternating ridge regression on the observed entries.
    Missing values (NaN) are excluded from all computations and can be
    recovered from the fitted factors via U_ @ V_.T.

    Attributes:
        U_ (NDArray[float64]): Left factor matrix of shape
            (n_samples, max_rank).
        V_ (NDArray[float64]): Right factor matrix of shape
            (n_features, max_rank).
        converged_ (bool): Whether the convergence criterion was met.
        n_iter_ (int): Number of iterations performed.
        loss_ (float): Final objective value.

    """

    _parameter_constraints = {
        "penalty": [Interval(Real, 0, None, closed="neither")],
        "max_rank": [Interval(Integral, 1, None, closed="left")],
        "init": [StrOptions({"random", "custom"})],
        "warm_start": ["boolean"],
        "max_iter": [Interval(Integral, 1, None, closed="left")],
        "tol": [Interval(Real, 0, None, closed="left")],
        "random_state": ["random_state"],
    }

    def __init__(
        self,
        *,
        penalty: float = 1.0,
        max_rank: int = 10,
        init: str = "random",
        warm_start: bool = False,
        max_iter: int = 1000,
        tol: float = 1e-6,
        random_state: int | RandomState | None = None,
    ):
        """Initialize LowRankImputation.

        Args:
            penalty: Strictly positive ridge regularisation weight applied to
                both U and V.
            max_rank: Number of latent factors.
            init: Initialisation strategy. "random" draws U and V from a
                standard normal distribution; "custom" uses the U and V
                provided to fit.
            warm_start: If True, reuse U_ and V_ from a previous fit as the
                starting point instead of reinitialising.
            max_iter: Maximum number of alternating update iterations.
            tol: Convergence tolerance; stops when the relative decrease in
                loss falls below tol.
            random_state: Seed or RandomState instance for reproducible random
                initialisation.

        """
        self.penalty = penalty
        self.max_rank = max_rank
        self.init = init
        self.warm_start = warm_start
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    def _more_tags(self):
        return {"allow_nan": True}

    def fit(
        self,
        X: ArrayLike,
        y: object | None = None,
        *,
        U: ArrayLike | None = None,
        V: ArrayLike | None = None,
    ) -> "LowRankImputation":
        """Fit the low-rank factorisation to X.

        Args:
            X: Input matrix of shape (n_samples, n_features), possibly
                containing NaN for missing entries.
            y: Ignored.
            U: Initial left factor matrix of shape (n_samples, max_rank).
                Used when init='custom' or warm_start=True.
            V: Initial right factor matrix of shape (n_features, max_rank).
                Used when init='custom' or warm_start=True.

        Returns:
            self

        """
        self._validate_params()

        X = check_array(
            X, dtype=[float64, float32], ensure_all_finite="allow-nan"
        )

        if U is not None or V is not None:
            if self.init != "custom":
                warn(
                    "When init!='custom', provided U or V are ignored. Set"
                    " init='custom' to use them as initialization.",
                    stacklevel=2,
                )

        if self.warm_start and hasattr(self, "U_") and hasattr(self, "V_"):
            U, V = _validate_init(X, self.U_, self.V_, self.max_rank)
        elif self.init == "random":
            U, V = _random_init(X, self.max_rank, self.random_state)
        else:  # init == "custom"
            U, V = _validate_init(X, U, V, self.max_rank)

        penalty = self.penalty
        max_iter = self.max_iter
        tol = self.tol

        observed = ~isnan(X)
        complete = observed.all()
        if not complete:
            observed_by_row = tuple(flatnonzero(row) for row in observed)
            observed_by_column = tuple(
                flatnonzero(column) for column in observed.T
            )

        loss_old = _compute_loss(X, U, V, penalty)

        converged = False
        for _i in range(max_iter):
            # We will solve
            # min_{u, v} 0.5 sum_{i, j obs.} (x^(i, j) - u^(i, :) v^(j, :))^2
            #            + lambda / 2 * ||u||_F^2
            #            + lambda / 2 * ||v||_F^2

            if complete:
                # Every row (and then every column) shares a normal matrix,
                # so solve all right-hand sides together.
                A = _ridge_normal_matrix(V, penalty)
                U[...] = solve(A, (X @ V).T).T

                A = _ridge_normal_matrix(U, penalty)
                V[...] = solve(A, (X.T @ U).T).T
            else:
                # Given fixed v this is a ridge regression problem for each
                # u^(i, :) for the observed rows of v.
                for r, indices in enumerate(observed_by_row):
                    V_observed = V[indices, :]
                    A = _ridge_normal_matrix(V_observed, penalty)
                    b = V_observed.T @ X[r, indices]
                    U[r, :] = solve(A, b)

                # Given fixed u this is a ridge regression problem for each
                # v^(j, :) for the observed rows of u.
                for c, indices in enumerate(observed_by_column):
                    U_observed = U[indices, :]
                    A = _ridge_normal_matrix(U_observed, penalty)
                    b = U_observed.T @ X[indices, c]
                    V[c, :] = solve(A, b)

            loss = _compute_loss(X, U, V, penalty)

            loss_decrease = loss_old - loss
            if loss_decrease < 0:
                loss_old = loss
                continue

            if loss_decrease <= tol * loss_old:
                converged = True
                break

            loss_old = loss

        self.converged_ = converged
        self.U_ = U
        self.V_ = V
        self.n_iter_ = _i + 1
        self.n_features_in_ = X.shape[1]
        self.loss_ = loss

        return self


def _ridge_normal_matrix(factors, penalty):
    """Return the ridge-regularised normal matrix for factors."""
    A = factors.T @ factors
    A.flat[:: A.shape[0] + 1] += penalty
    return A


def _random_init(X, max_rank, random_state):
    """Return randomly initialised factor matrices matching X's dtype."""
    rnd = check_random_state(random_state)
    U = asarray(rnd.standard_normal((X.shape[0], max_rank)), dtype=X.dtype)
    V = asarray(rnd.standard_normal((X.shape[1], max_rank)), dtype=X.dtype)
    return U, V


def _validate_init(X, U, V, max_rank):
    """Validate and cast U and V to X's dtype, raising on shape mismatch."""
    U = check_array(U, dtype=X.dtype)
    V = check_array(V, dtype=X.dtype)
    if U.shape != (X.shape[0], max_rank):
        raise ValueError(
            "U must be a 2d-array of shape"
            f" {(X.shape[0], max_rank)}, got {U.shape}"
        )
    if V.shape != (X.shape[1], max_rank):
        raise ValueError(
            "V must be a 2d-array of shape"
            f" {(X.shape[1], max_rank)}, got {V.shape}"
        )
    return U, V


def _compute_loss(X, U, V, penalty) -> float:
    """Return 0.5 * (||X - U V^T||_F^2 + penalty * (||U||_F^2 + ||V||_F^2)).

    Missing entries in X (NaN) are excluded from the reconstruction term.
    """
    return 0.5 * (
        nansum((X - U @ V.T) ** 2)
        + penalty * (U**2).sum()
        + penalty * (V**2).sum()
    )
