"""Cross-validate parameter grids and find best SolrCMF solution.

This module contains functionality for performing cross-validation (CV) across
parameter grids for the SolrCMF problem. CV is performed in parallel across
independent parameter combinations.
"""

from collections.abc import Generator
from numbers import Integral, Real
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from warnings import warn

from joblib import Parallel, delayed, dump, load
from numpy import (
    argmax,
    argmin,
    array,
    asarray,
    atleast_1d,
    broadcast_arrays,
    flatnonzero,
    float64,
    full,
    intp,
    mean,
    ndim,
    reshape,
    split,
    sqrt,
    std,
    sum,
    vstack,
)
from numpy.random import default_rng
from numpy.typing import ArrayLike, NDArray
from sklearn.base import BaseEstimator, clone
from sklearn.utils._param_validation import Interval, StrOptions

from .base import Entity, ViewDesc
from .metrics import (
    neg_mean_squared_error,
    neg_sum_squared_error,
    weighted_neg_mean_squared_error,
)
from .solrcmf import SolrCMF
from .splits import BaseSplitter, ElementwiseFolds

type InitsGeneratorType = Generator[
    tuple[
        int,
        tuple[
            dict[Entity, NDArray[float64]] | None,
            dict[ViewDesc, NDArray[float64]] | None,
            dict[Entity, NDArray[float64]] | None,
        ],
    ],
    Any,
    None,
]


class SolrCMFCV(BaseEstimator):
    """Cross-validated hyperparameter selection for SolrCMF.

    Fits SolrCMF over a grid of `structure_penalty`, `max_rank`, and
    `factor_penalty` values, selects the best combination by
    cross-validation, and refits on the full data.

    Two cross-validation strategies are supported.
    "structure_first_debiased_cv" first estimates the structure pattern on
    the full data for each parameter combination, then cross-validates a
    debiased (unpenalized) refit using that fixed pattern. "penalized_cv"
    cross-validates the full penalized estimation directly on held-out entries.

    Attributes:
        cv_results_ (dict[str, list]): Per-parameter-combination diagnostics.
            Each list has one entry per parameter combination. Keys:

            - "structure_penalty", "max_rank", "factor_penalty":
              the parameter values for each combination.
            - "<score>_fold<i>": held-out score on fold i, where
              <score> is the chosen scoring function.
            - "mean_<score>", "std_<score>", "sem_<score>": mean,
              standard deviation, and standard error of the per-fold scores.
            - "est_max_rank": effective rank of the best run.
            - "structural_zeros": number of zero entries in d[k] summed
              over all view pairs (proxy for structure sparsity).
            - "factor_zeros": number of zero entries in U summed over all
              views (proxy for factor sparsity).

            Additional keys for cv_strategy="structure_first_debiased_cv":

            - "objective_value_penalized": objective of the penalized fit.
            - "mean_elapsed_process_time_penalized",
              "std_elapsed_process_time_penalized": CPU time statistics
              for the penalized structure estimation step.
            - "mean_elapsed_process_time_fixed",
              "std_elapsed_process_time_fixed": CPU time statistics for
              the debiased cross-validation step.

            Additional keys for cv_strategy="penalized_cv":

            - "mean_elapsed_process_time",
              "std_elapsed_process_time": CPU time statistics across all
              penalized CV fits.
        best_index_ (int): Index into cv_results_ of the selected parameter
            combination.
        best_estimator_ (SolrCMF): Estimator refit on all data with the
            selected parameters.
        best_max_rank_ (int): Effective rank of the best estimator after
            fitting.

    """

    _parameter_constraints = {
        "structure_penalty": [
            Interval(Real, 0, None, closed="left"),
            "array-like",
        ],
        "max_rank": [Interval(Integral, 1, None, closed="left"), "array-like"],
        "factor_penalty": [
            Interval(Real, 0, None, closed="neither"),
            "array-like",
            None,
        ],
        "factor_pruning": ["boolean"],
        "cv": [Interval(Integral, 2, None, closed="left"), BaseSplitter],
        "cv_strategy": [
            StrOptions({"structure_first_debiased_cv", "penalized_cv"})
        ],
        "score": [
            StrOptions(
                {
                    "neg_mean_squared_error",
                    "neg_sum_squared_error",
                    "weighted_neg_mean_squared_error",
                }
            )
        ],
        "refit": [
            StrOptions(
                {
                    "mean_debiased",
                    "mean_penalized",
                    "1se_debiased",
                    "1se_penalized",
                }
            )
        ],
        "init": [StrOptions({"random", "custom"})],
        "init_kwargs": [dict, None],
        "rho": [Interval(Real, 0.0, None, closed="neither"), None],
        "alpha": [Interval(Real, 0.0, None, closed="left"), None],
        "mu": [Interval(Real, 0.0, None, closed="neither"), None],
        "max_iter": [Interval(Integral, 1, None, closed="left")],
        "abs_tol": [Interval(Real, 0.0, None, closed="neither")],
        "rel_tol": [Interval(Real, 0.0, None, closed="neither")],
        "verbose": ["boolean"],
        "n_jobs": [Integral, None],
    }

    def __init__(
        self,
        *,
        structure_penalty: float | ArrayLike = 1.0,
        max_rank: int | ArrayLike = 10,
        factor_penalty: float | ArrayLike | None = None,
        factor_pruning: bool = True,
        cv: int | BaseSplitter = 10,
        cv_strategy: str = "structure_first_debiased_cv",
        score: str = "neg_mean_squared_error",
        refit: str = "1se_debiased",
        init: str = "random",
        init_kwargs: dict | None = None,
        rho: float | None = None,
        alpha: float | None = None,
        mu: float | None = None,
        max_iter: int = 1000,
        abs_tol: float = 1e-6,
        rel_tol: float = 1e-6,
        verbose: bool = False,
        n_jobs: int | None = None,
    ):
        """Initialize SolrCMFCV.

        Args:
            structure_penalty: L1 penalty weight on d[k], or an array-like of
                values to search over.
            max_rank: Maximum number of factors, or an array-like of values to
                search over.
            factor_penalty: L1 penalty weight on sparse factor loadings U, or
                an array-like of values to search over. `None` disables factor
                sparsity.
            factor_pruning: Whether to prune globally inactive factors during
                fitting.
            cv: Number of cross-validation folds, or a BaseSplitter instance.
            cv_strategy: Cross-validation strategy. See class docstring.
            score: Scoring function used to evaluate held-out predictions.
            refit: Strategy for selecting and refitting the final estimator.
                Prefix "mean" selects by mean CV score; "1se" applies the
                one-standard-error rule. Suffix "debiased" refits without
                penalty using the estimated structure pattern; "penalized"
                refits with the full penalty.
            init: Initialisation strategy passed to each SolrCMF fit.
            init_kwargs: Additional keyword arguments for the initialiser.
                For random initialization, "rng" accepts an integer seed or
                Generator and "repetitions" controls the number of restarts.
            rho: ADMM penalty parameter passed to each SolrCMF fit.
            alpha: Ridge regularisation weight on V passed to each SolrCMF fit.
            mu: V'-slack penalty weight passed to each SolrCMF fit.
            max_iter: Maximum number of ADMM iterations per fit.
            abs_tol: Absolute convergence tolerance per fit.
            rel_tol: Relative convergence tolerance per fit.
            verbose: Whether to print progress messages during fitting.
            n_jobs: Number of parallel jobs. Passed to joblib.Parallel.

        """
        self.structure_penalty = structure_penalty
        self.max_rank = max_rank
        self.factor_penalty = factor_penalty
        self.factor_pruning = factor_pruning
        self.cv = cv
        self.cv_strategy = cv_strategy
        self.score = score
        self.refit = refit
        self.init = init
        self.init_kwargs = init_kwargs
        self.rho = rho
        self.alpha = alpha
        self.mu = mu
        self.max_iter = max_iter
        self.abs_tol = abs_tol
        self.rel_tol = rel_tol
        self.verbose = verbose
        self.n_jobs = n_jobs

    def _check_parameter_grid(self):
        if self.factor_penalty is None:
            factor_penalty = array([None])
        else:
            factor_penalty = atleast_1d(self.factor_penalty)

        # Scalars to 1d-arrays
        structure_penalty, max_rank = atleast_1d(
            self.structure_penalty, self.max_rank
        )

        # Check that all are indeed 1d
        if not (
            ndim(structure_penalty)
            == ndim(max_rank)
            == ndim(factor_penalty)
            == 1
        ):
            raise ValueError(
                f"In {self.__class__.__name__} arguments 'structure_penalty',"
                " 'max_rank', and 'factor_penalty' need to be one-dimensional"
                " or equal to a single number (or 'None' for 'factor_penalty')"
            )

        structure_penalty, max_rank, factor_penalty = broadcast_arrays(
            structure_penalty, max_rank, factor_penalty
        )

        return list(
            zip(
                structure_penalty,
                max_rank,
                factor_penalty,
                strict=True,
            )
        )

    def fit(
        self,
        X: dict[ViewDesc, NDArray[float64]],
        y: object | None = None,
        *,
        structure_weights: (
            dict[ViewDesc, NDArray[float64] | float64] | None
        ) = None,
        factor_weights: dict[Entity, NDArray[float64] | float64] | None = None,
        vs: list[dict[Entity, NDArray[float64]]] | None = None,
        ds: list[dict[ViewDesc, NDArray[float64]]] | None = None,
        us: list[dict[Entity, NDArray[float64]]] | None = None,
    ) -> "SolrCMFCV":
        """Cross-validate the parameter grid and refit on the best combination.

        Args:
            X: Data matrices, one per view pair.
            y: Ignored.
            structure_weights: Per-element L1 weights on d[k], one array per
                view pair. Passed through to each SolrCMF fit.
            factor_weights: Per-element L1 weights on U, one array per view.
                Passed through to each SolrCMF fit.
            vs: List of initial factor matrices, one dict per repetition.
                Required when init="custom".
            ds: List of initial scaling vectors, one dict per repetition.
                Required when init="custom".
            us: List of initial sparse loading matrices, one dict per
                repetition. Required when init="custom" and factor sparsity
                is used.

        Returns:
            self

        """
        self._validate_params()

        parameter_grid = self._check_parameter_grid()

        n_params = len(parameter_grid)

        if isinstance(self.cv, Integral):
            cv = ElementwiseFolds(int(self.cv))
        elif isinstance(self.cv, BaseSplitter):
            cv = self.cv

        if self.score == "neg_mean_squared_error":
            score_fn = neg_mean_squared_error
        elif self.score == "neg_sum_squared_error":
            score_fn = neg_sum_squared_error
        elif self.score == "weighted_neg_mean_squared_error":
            score_fn = weighted_neg_mean_squared_error

        results = {
            "structure_penalty": [s for s, _, _ in parameter_grid],
            "max_rank": [m for _, m, _ in parameter_grid],
            "factor_penalty": [f for _, _, f in parameter_grid],
        }

        init_kwargs = dict(self.init_kwargs or {})

        if self.init == "random":
            n_reps = init_kwargs.pop("repetitions", 1)
            if (
                not isinstance(n_reps, Integral)
                or isinstance(n_reps, bool)
                or n_reps < 1
            ):
                raise ValueError(
                    "'init_kwargs[\"repetitions\"]' needs to be a positive"
                    " integer"
                )

            def inits() -> InitsGeneratorType:
                for i in range(n_reps):
                    yield i, (None, None, None)

            # If an rng or seed is supplied, extract it
            if "rng" in init_kwargs:
                rng = default_rng(init_kwargs["rng"])
            else:
                rng = default_rng()
        elif self.init == "custom":
            # If one of these is provided all need to be the same length
            # (if only vs and ds are provided then us is treated as
            # a list of None)
            if not (
                vs is not None and ds is not None and len(vs) == len(ds) >= 1
            ):
                raise ValueError(
                    "If initial values are provided to"
                    f" {self.__class__.__name__}.fit(), then 'vs' and 'ds'"
                    " both need to be provided and have to be the same length"
                )
            if us is not None and len(us) != len(vs):
                raise ValueError(
                    "If initial values for 'u' are provided to"
                    f" {self.__class__.__name__}.fit(), then 'us' needs to"
                    " have the same length as 'vs' and 'ds'"
                )

            n_reps = len(vs)

            def inits() -> InitsGeneratorType:
                for i in range(n_reps):
                    yield i, (vs[i], ds[i], us[i] if us is not None else None)

        else:
            raise ValueError(f"Unknown init method {self.init}")

        base_est = SolrCMF(
            factor_pruning=self.factor_pruning,
            init=self.init,
            init_kwargs=init_kwargs,
            rho=self.rho,
            alpha=self.alpha,
            mu=self.mu,
            max_iter=self.max_iter,
            abs_tol=self.abs_tol,
            rel_tol=self.rel_tol,
        )

        if self.cv_strategy == "structure_first_debiased_cv":
            tmpdir = TemporaryDirectory()
            tmppath = Path(tmpdir.name)

            def _estimate_structure(
                idx_params,
                idx_init,
                structure_penalty,
                max_rank,
                factor_penalty,
                vs,
                ds,
                us,
                rng,
            ):
                est: SolrCMF = clone(base_est)
                est.set_params(
                    structure_penalty=structure_penalty,
                    max_rank=max_rank,
                    factor_penalty=factor_penalty,
                )

                if est.init == "random":
                    est.init_kwargs = {
                        **(est.init_kwargs or {}),
                        "rng": default_rng(rng),
                    }

                est.fit(
                    X,
                    structure_weights=structure_weights,
                    factor_weights=factor_weights,
                    vs=vs,
                    ds=ds,
                    us=us,
                )

                if not est.converged_:
                    warn(
                        "Penalized estimation with parameters"
                        f" (structure_penalty={structure_penalty},"
                        f" max_rank={max_rank},"
                        f" factor_penalty={factor_penalty}):"
                        f" {est.__class__.__name__} did not converge"
                        f" after {est.n_iter_} iterations.",
                        stacklevel=2,
                    )

                # Save estimator for later
                dump(est, tmppath / f"{idx_params}_{idx_init}.pkl")

                factor_pattern = est.factor_pattern()

                return (
                    est.objective_value_,
                    est.elapsed_process_time_,
                    est.est_max_rank_,
                    # compute relative to supplied max_rank;
                    # est_max_rank_ could be less in case of
                    # factor pruning
                    (max_rank - est.est_max_rank_) * len(X)
                    + sum(
                        [sum(1 - p) for p in est.structure_pattern().values()]
                    ),
                    (
                        sum(
                            [
                                (max_rank - est.est_max_rank_)
                                * (p.shape[0] - 1)
                                + sum(1 - p)
                                for p in factor_pattern.values()
                            ]
                        )
                        if factor_pattern is not None
                        else 0
                    ),
                )

            if self.verbose:
                print(
                    f"Perform structure estimation ({n_reps * n_params} tasks)"
                )

            if self.init == "random":
                # We need to split the randomness for random initialization
                child_states = reshape(
                    rng.bit_generator.spawn(n_reps * n_params),
                    (n_params, n_reps),
                )
            else:
                # Dummy otherwise
                child_states = full((n_params, n_reps), None)

            out = Parallel(
                n_jobs=self.n_jobs, verbose=10 if self.verbose else 0
            )(
                delayed(_estimate_structure)(
                    idx_params,
                    idx_init,
                    structure_penalty,
                    max_rank,
                    factor_penalty,
                    vs,
                    ds,
                    us,
                    child_states[idx_params, idx_init],
                )
                for idx_params, (
                    structure_penalty,
                    max_rank,
                    factor_penalty,
                ) in enumerate(parameter_grid)
                for idx_init, (vs, ds, us) in inits()
            )

            (
                objective_values,
                elapsed_process_times,
                est_max_rank,
                structural_zeros,
                factor_zeros,
            ) = zip(*out, strict=True)

            if self.verbose:
                print("Determine best runs")

            # Rely on the fact that joblib returns results in the same
            # order as the inputs
            objective_values = split(asarray(objective_values), n_params)
            best_runs = [int(argmin(vals)) for vals in objective_values]
            results["objective_value_penalized"] = [
                vals[idx]
                for idx, vals in zip(best_runs, objective_values, strict=True)
            ]

            elapsed_process_times = split(
                asarray(elapsed_process_times), n_params
            )
            results["mean_elapsed_process_time_penalized"] = [
                mean(ts) for ts in elapsed_process_times
            ]
            results["std_elapsed_process_time_penalized"] = [
                std(ts) for ts in elapsed_process_times
            ]

            est_max_rank = split(asarray(est_max_rank), n_params)
            results["est_max_rank"] = [
                rks[idx]
                for idx, rks in zip(best_runs, est_max_rank, strict=True)
            ]
            structural_zeros = split(asarray(structural_zeros), n_params)
            results["structural_zeros"] = [
                zs[idx]
                for idx, zs in zip(best_runs, structural_zeros, strict=True)
            ]
            factor_zeros = split(asarray(factor_zeros), n_params)
            results["factor_zeros"] = [
                zs[idx]
                for idx, zs in zip(best_runs, factor_zeros, strict=True)
            ]

            def _debiased_cv_score(
                est_in: SolrCMF,
                train_indices: dict[ViewDesc, NDArray[intp]],
                test_indices: dict[ViewDesc, NDArray[intp]],
            ):
                est: SolrCMF = clone(base_est)
                est.set_params(
                    init="custom",
                    init_kwargs={"reduce_max_rank": True},
                    factor_pruning=False,  # Set to False always
                )
                est.fit(
                    X,
                    indices=train_indices,
                    structure_pattern=est_in.structure_pattern(),
                    factor_pattern=est_in.factor_pattern(),
                    vs=est_in.vs_,
                    ds=est_in.ds_,
                    us=est_in.us_ if hasattr(est_in, "us_") else None,
                )

                if not est.converged_:
                    warn(
                        "Fixed structure estimation of"
                        f" {est.__class__.__name__} did not converge after"
                        f" {est.n_iter_} iterations.",
                        stacklevel=2,
                    )

                return (
                    score_fn(X, est.transform(X), indices=test_indices),
                    est.elapsed_process_time_,
                )

            # Reads fitted penalized estimators from cache and
            # extracts structure/factor patterns
            def solrcmf_estimators():
                for idx_params, idx_init in zip(
                    range(n_params), best_runs, strict=True
                ):
                    yield load(tmppath / f"{idx_params}_{idx_init}.pkl")

            # We want exactly the same splits for all parameter combinations,
            # so we produce the splits once and then reuse them.
            cv_splits = list(cv.split(X))
            n_folds = cv.get_n_splits(X)

            if self.verbose:
                print(
                    "Perform debiased cross-validation"
                    f" ({n_params * n_folds} tasks)"
                )

            out = Parallel(
                n_jobs=self.n_jobs, verbose=10 if self.verbose else 0
            )(
                delayed(_debiased_cv_score)(
                    est,
                    train_indices,
                    test_indices,
                )
                for est in solrcmf_estimators()
                for train_indices, test_indices in cv_splits
            )
            (
                scores,
                elapsed_process_times,
            ) = zip(*out, strict=True)

            for i in range(n_folds):
                results[f"{self.score}_fold{i}"] = [
                    scores[j * n_folds + i] for j in range(n_params)
                ]

            elapsed_process_times = split(
                asarray(elapsed_process_times), n_params
            )
            results["mean_elapsed_process_time_fixed"] = [
                mean(ts) for ts in elapsed_process_times
            ]
            results["std_elapsed_process_time_fixed"] = [
                std(ts) for ts in elapsed_process_times
            ]
        elif self.cv_strategy == "penalized_cv":

            def _penalized_cv_score(
                structure_penalty,
                max_rank,
                factor_penalty,
                vs,
                ds,
                us,
                train_indices,
                test_indices,
                rng,
            ):
                est: SolrCMF = clone(base_est)
                est.set_params(
                    structure_penalty=structure_penalty,
                    max_rank=max_rank,
                    factor_penalty=factor_penalty,
                )

                if est.init == "random":
                    est.init_kwargs = {
                        **(est.init_kwargs or {}),
                        "rng": default_rng(rng),
                    }

                est.fit(
                    X,
                    indices=train_indices,
                    structure_weights=structure_weights,
                    factor_weights=factor_weights,
                    vs=vs,
                    ds=ds,
                    us=us,
                )

                if not est.converged_:
                    warn(
                        "Penalized estimation with parameters"
                        f" (structure_penalty={structure_penalty},"
                        f" max_rank={max_rank},"
                        f" factor_penalty={factor_penalty}):"
                        f" {est.__class__.__name__} did not converge"
                        f" after {est.n_iter_} iterations.",
                        stacklevel=2,
                    )

                factor_pattern = est.factor_pattern()

                return (
                    score_fn(X, est.transform(X), indices=test_indices),
                    est.elapsed_process_time_,
                    est.est_max_rank_,
                    # compute relative to supplied max_rank;
                    # est_max_rank_ could be less in case of
                    # factor pruning
                    (max_rank - est.est_max_rank_) * len(X)
                    + sum(
                        [sum(1 - p) for p in est.structure_pattern().values()]
                    ),
                    (
                        sum(
                            [
                                (max_rank - est.est_max_rank_)
                                * (p.shape[0] - 1)
                                + sum(1 - p)
                                for p in factor_pattern.values()
                            ]
                        )
                        if factor_pattern is not None
                        else 0
                    ),
                )

            # We want exactly the same splits for all parameter combinations,
            # so we produce the splits once and then reuse them.
            cv_splits = list(cv.split(X))
            n_folds = cv.get_n_splits(X)

            if self.verbose:
                print(
                    "Perform penalized cross-validation"
                    f" ({n_params * n_reps * n_folds} tasks)"
                )

            if self.init == "random":
                # We need to split the randomness for random initialization
                child_states = reshape(
                    rng.bit_generator.spawn(n_reps * n_params * n_folds),
                    (n_params, n_reps, n_folds),
                )
            else:
                # Dummy otherwise
                child_states = full((n_params, n_reps, n_folds), None)

            out = Parallel(
                n_jobs=self.n_jobs, verbose=10 if self.verbose else 0
            )(
                delayed(_penalized_cv_score)(
                    structure_penalty,
                    max_rank,
                    factor_penalty,
                    vs,
                    ds,
                    us,
                    train_indices,
                    test_indices,
                    child_states[idx_param, idx_init, idx_fold],
                )
                for idx_param, (
                    structure_penalty,
                    max_rank,
                    factor_penalty,
                ) in enumerate(parameter_grid)
                for idx_init, (vs, ds, us) in inits()
                for idx_fold, (train_indices, test_indices) in enumerate(
                    cv_splits
                )
            )

            (
                scores,
                elapsed_process_times,
                est_max_rank,
                structural_zeros,
                factor_zeros,
            ) = zip(*out, strict=True)

            best_runs, selected_scores = _select_best_penalized_runs(
                scores,
                n_params=n_params,
                n_reps=n_reps,
                n_folds=n_folds,
            )
            for i in range(n_folds):
                results[f"{self.score}_fold{i}"] = selected_scores[
                    :, i
                ].tolist()

            elapsed_process_times = split(
                asarray(elapsed_process_times), n_params
            )
            results["mean_elapsed_process_time"] = [
                mean(ts) for ts in elapsed_process_times
            ]
            results["std_elapsed_process_time"] = [
                std(ts) for ts in elapsed_process_times
            ]

            results["est_max_rank"] = [
                mean(
                    est_max_rank[
                        (
                            idx_params * n_reps * n_folds
                            + best_runs[idx_params] * n_folds
                        ) : (
                            idx_params * n_reps * n_folds
                            + (best_runs[idx_params] + 1) * n_folds
                        )
                    ]
                )
                for idx_params in range(n_params)
            ]
            results["structural_zeros"] = [
                mean(
                    structural_zeros[
                        (
                            idx_params * n_reps * n_folds
                            + best_runs[idx_params] * n_folds
                        ) : (
                            idx_params * n_reps * n_folds
                            + (best_runs[idx_params] + 1) * n_folds
                        )
                    ]
                )
                for idx_params in range(n_params)
            ]

            results["factor_zeros"] = [
                mean(
                    factor_zeros[
                        (
                            idx_params * n_reps * n_folds
                            + best_runs[idx_params] * n_folds
                        ) : (
                            idx_params * n_reps * n_folds
                            + (best_runs[idx_params] + 1) * n_folds
                        )
                    ]
                )
                for idx_params in range(n_params)
            ]

        # Post-processing on the full dictionary. Same for both cases
        scores = vstack(
            [results[f"{self.score}_fold{i}"] for i in range(n_folds)]
        )
        score_std = scores.std(0)
        results.update(
            {
                f"mean_{self.score}": scores.mean(0),
                f"std_{self.score}": score_std,
                f"sem_{self.score}": score_std / sqrt(n_folds),
            }
        )

        self.cv_results_ = results

        if self.verbose:
            print("Re-fit final estimator")

        self.best_index_ = _select_best_index(
            results,
            score=self.score,
            refit=self.refit,
        )

        structure_penalty, max_rank, factor_penalty = parameter_grid[
            self.best_index_
        ]

        if self.verbose:
            print(
                "Best fit with\n"
                f"  structure_penalty = {structure_penalty}\n"
                f"  max_rank = {max_rank}\n"
                f"  factor_penalty = {factor_penalty}\n\n"
                "  estimated max_rank = "
                f"{results['est_max_rank'][self.best_index_]}"
            )

        # Re-fit best run on all data
        if self.cv_strategy == "structure_first_debiased_cv":
            # Load respective penalized estimator from cache
            est = load(
                tmppath
                / f"{self.best_index_}_{best_runs[self.best_index_]}.pkl"
            )

            self.best_estimator_: SolrCMF = clone(base_est)

            if self.refit.endswith("debiased"):
                self.best_estimator_.set_params(
                    init="custom",
                    init_kwargs={"reduce_max_rank": True},
                    factor_pruning=False,  # Set to False always
                )
                self.best_estimator_.fit(
                    X,
                    structure_pattern=est.structure_pattern(),
                    factor_pattern=est.factor_pattern(),
                    vs=est.vs_,
                    ds=est.ds_,
                    us=est.us_ if hasattr(est, "us_") else None,
                )
            elif self.refit.endswith("penalized"):
                self.best_estimator_.set_params(
                    structure_penalty=structure_penalty,
                    max_rank=max_rank,
                    factor_penalty=factor_penalty,
                    init="custom",
                )
                self.best_estimator_.fit(
                    X,
                    vs=est.vs_,
                    ds=est.ds_,
                    us=est.us_ if hasattr(est, "us_") else None,
                )

            tmpdir.cleanup()
        elif self.cv_strategy == "penalized_cv":
            # A penalized fit needs to be performed irrespectively.
            # Either because it is the final fit or because we need the
            # structure/factor pattern.
            if self.init == "custom":
                assert vs is not None and ds is not None
                vs_init = vs[best_runs[self.best_index_]]
                ds_init = ds[best_runs[self.best_index_]]
                if us is not None:
                    us_init = us[best_runs[self.best_index_]]
                else:
                    us_init = None
            else:
                vs_init = None
                ds_init = None
                us_init = None

            final_est: SolrCMF = clone(base_est)
            final_est.set_params(
                structure_penalty=structure_penalty,
                max_rank=max_rank,
                factor_penalty=factor_penalty,
            )
            final_est.fit(
                X,
                vs=vs_init,
                ds=ds_init,
                us=us_init,
            )

            if self.refit.endswith("debiased"):
                final_est_debiased: SolrCMF = clone(base_est)
                final_est_debiased.set_params(
                    init="custom",
                    init_kwargs={"reduce_max_rank": True},
                    factor_pruning=False,  # Set to False always
                )
                final_est_debiased.fit(
                    X,
                    structure_pattern=final_est.structure_pattern(),
                    factor_pattern=final_est.factor_pattern(),
                    vs=final_est.vs_,
                    ds=final_est.ds_,
                    us=final_est.us_ if hasattr(final_est, "us_") else None,
                )

                self.best_estimator_ = final_est_debiased
            elif self.refit.endswith("penalized"):
                self.best_estimator_ = final_est

        self.best_max_rank_ = self.best_estimator_.est_max_rank_

        return self


def _select_best_penalized_runs(
    scores: ArrayLike,
    *,
    n_params: int,
    n_reps: int,
    n_folds: int,
) -> tuple[NDArray[intp], NDArray[float64]]:
    """Select the highest-scoring restart for each parameter combination."""
    scores_array = asarray(scores, dtype=float64).reshape(
        n_params, n_reps, n_folds
    )
    best_runs = argmax(scores_array.mean(axis=2), axis=1)
    selected_scores = asarray(
        [
            scores_array[idx_params, idx_run, :]
            for idx_params, idx_run in enumerate(best_runs)
        ]
    )
    return best_runs, selected_scores


def _select_best_index(
    results: dict[str, Any],
    *,
    score: str,
    refit: str,
) -> int:
    """Select the best parameter index using mean score or the 1-SE rule."""
    mean_scores = asarray(results[f"mean_{score}"])
    max_index = int(argmax(mean_scores))
    if refit.startswith("mean"):
        return max_index

    candidates = flatnonzero(
        mean_scores
        >= mean_scores[max_index] - asarray(results[f"sem_{score}"])[max_index]
    )

    structural_zeros = asarray(results["structural_zeros"])[candidates]
    most_structural_zeros = candidates[
        flatnonzero(structural_zeros == max(structural_zeros))
    ]
    factor_zeros = asarray(results["factor_zeros"])[most_structural_zeros]
    return int(most_structural_zeros[argmax(factor_zeros)])
