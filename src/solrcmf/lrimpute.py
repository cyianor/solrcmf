from numbers import Integral, Real
from warnings import warn

from numpy import (
    asarray,
    diagonal,
    fill_diagonal,
    flatnonzero,
    float32,
    float64,
    isnan,
    nansum,
)
from numpy.linalg import solve
from numpy.random import RandomState
from sklearn.base import BaseEstimator
from sklearn.utils._param_validation import Interval, StrOptions
from sklearn.utils.validation import check_array, check_random_state


class LowRankImputation(BaseEstimator):
    _parameter_constraints = {
        "penalty": [Interval(Real, 0, None, closed="left")],
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
        self.penalty = penalty
        self.max_rank = max_rank
        self.init = init
        self.warm_start = warm_start
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    def _more_tags(self):
        return {"allow_nan": True}

    def fit(self, X, y=None, *, U=None, V=None):
        self._validate_params()

        X = check_array(
            X, dtype=[float64, float32], ensure_all_finite="allow-nan"
        )

        if U is not None or V is not None:
            if self.init != "custom":
                warn(
                    "When init!='custom', provided U or V are ignored. Set"
                    " init='custom' to use them as initialization."
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

        loss_old = _compute_loss(X, U, V, penalty)

        converged = False
        for i in range(max_iter):
            # We will solve
            # min_{u, v} 0.5 sum_{i, j obs.} (x^(i, j) - u^(i, :) v^(j, :))^2
            #            + lambda / 2 * ||u||_F^2
            #            + lambda / 2 * ||v||_F^2

            # Given fixed v this is a ridge regression problem for each
            # u^(i, :) for a subset of the rows of v
            for r in range(X.shape[0]):
                indices = flatnonzero(1 - isnan(X[r, :]))
                A = V[indices, :].T @ V[indices, :]
                fill_diagonal(A, diagonal(A) + penalty)
                b = V[indices, :].T @ X[r, :][indices]
                U[r, :] = solve(A, b)

            # Given fixed u this is a ridge regression problem for each
            # v^(j, :) for a subset of the rows of u
            for c in range(X.shape[1]):
                indices = flatnonzero(1 - isnan(X[:, c]))
                A = U[indices, :].T @ U[indices, :]
                fill_diagonal(A, diagonal(A) + penalty)
                b = U[indices, :].T @ X[:, c][indices]
                V[c, :] = solve(A, b)

            loss = _compute_loss(X, U, V, penalty)

            if (loss_old - loss) < tol * loss_old:
                converged = True
                break

            loss_old = loss

        self.converged_ = converged
        self.U_ = U
        self.V_ = V
        self.n_iter_ = i + 1
        self.n_features_in_ = X.shape[1]
        self.loss_ = loss

        return self


def _random_init(X, max_rank, random_state):
    rnd = check_random_state(random_state)
    U = asarray(rnd.standard_normal((X.shape[0], max_rank)), dtype=X.dtype)
    V = asarray(rnd.standard_normal((X.shape[1], max_rank)), dtype=X.dtype)
    return U, V


def _validate_init(X, U, V, max_rank):
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
    return 0.5 * (
        nansum((X - U @ V.T) ** 2)
        + penalty * (U**2).sum()
        + penalty * (V**2).sum()
    )
