"""Generic multi-block ADMM functionality.

This module provides a class that can be used as a generic base class
for a multi-block ADMM algorithm. It implements the basic machinery
necessary to run the algorithm and relies on the building blocks
defined in base.py.
"""

from abc import ABC, abstractmethod
from dataclasses import fields
from numbers import Integral, Real
from time import process_time
from typing import Any

from numpy import inf
from sklearn.base import BaseEstimator
from sklearn.utils._param_validation import Interval

from .base import (
    Block,
    BlockDesc,
    Constraint,
    Context,
    DataclassInstance,
    objective,
    update,
)


class ADMM[
    BT: DataclassInstance,
    CT: DataclassInstance,
    PT: DataclassInstance,
](BaseEstimator, ABC):
    """Base class for multi-block ADMM algorithms.

    Subclasses implement `_setup` to construct the `Context` (blocks,
    constraints, and parameters) for a specific problem, and override
    `score` and `transform` for evaluation and reconstruction.

    The iteration alternates between updating all primal blocks in
    `ctx.block_order`, updating all dual (constraint multiplier) variables,
    and evaluating the augmented Lagrangian objective. Convergence is
    declared when the absolute change in objective satisfies

        |obj_old - obj| <= max(rel_tol * |obj_old|, abs_tol)

    skipping the first iteration to avoid spurious convergence from the
    initialisation.

    Attributes:
        objs_ (list[float]): Objective value at each iteration.
        gaps_ (list[float]): Change in objective (obj_old - obj) at each
            iteration.
        converged_ (bool): Whether the convergence criterion was met.
        objective_value_ (float): Final objective value.
        n_iter_ (int): Number of iterations performed.
        elapsed_process_time_ (float): CPU time consumed by the iteration loop.
        ctx_ (Context): The context object (only when save_ctx=True).

    """

    _parameter_constraints = {
        "max_iter": [Interval(Integral, 1, None, closed="left")],
        "abs_tol": [Interval(Real, 0, None, closed="neither")],
        "rel_tol": [Interval(Real, 0, None, closed="neither")],
        "save_ctx": ["boolean"],
    }

    def __init__(
        self,
        max_iter: int = 1000,
        abs_tol: float = 1e-6,
        rel_tol: float = 1e-6,
        *,
        save_ctx: bool = False,
    ):
        """Initialize a new instance of the ADMM algorithm.

        Args:
            max_iter: Maximum number of iterations
            abs_tol: Absolute convergence tolerance
            rel_tol: Relative convergence tolerance
            save_ctx: Whether or not context object should be saved upon
                      convergence.

        """
        self.max_iter = max_iter
        self.abs_tol = abs_tol
        self.rel_tol = rel_tol

        self.save_ctx = save_ctx

    @abstractmethod
    def _setup(self, X, **kwargs) -> Context[BT, CT, PT]:
        """Set up the estimation problem.

        Called after data is available.
        """
        raise NotImplementedError(
            f"_setup method on {self.__class__.__name__} not implemented"
        )

    def fit(self, X, y=None, **kwargs):
        """Run the ADMM iteration until convergence or max_iter.

        Calls `_setup` to build the context, then alternates between primal
        block updates and dual (constraint multiplier) updates. The augmented
        Lagrangian objective is evaluated after each full sweep and checked
        against the convergence criterion.
        """
        # Validate parameters; should check parameters of
        # derived classes as well
        self._validate_params()

        # Setup ADMM context
        ctx = self._setup(X, **kwargs)

        start_time = process_time()

        objs = []
        gaps = []

        converged = False
        obj_old = inf
        for i in range(self.max_iter):
            # Update variable blocks
            for name, idx in ctx.block_order:
                bgroup = getattr(ctx.blocks, name)
                update(bgroup[idx], ctx)

            # Update constraints
            for cnstrnt in fields(ctx.constraints):
                cgroup = getattr(ctx.constraints, cnstrnt.name)
                for c in cgroup.values():
                    update(c, ctx)

            obj = _objective(ctx)
            gap = obj_old - obj

            objs.append(obj)
            gaps.append(gap)

            if i > 0 and abs(gap) <= max(
                self.rel_tol * abs(obj_old), self.abs_tol
            ):
                converged = True
                break

            obj_old = obj

        end_time = process_time()

        self.objs_ = objs
        self.gaps_ = gaps
        self.converged_ = converged
        self.objective_value_ = objs[-1]
        self.n_iter_ = i + 1
        self.elapsed_process_time_ = end_time - start_time

        for k, v in self._extra_attrs(ctx).items():
            setattr(self, k, v)

        if self.save_ctx:
            self.ctx_ = ctx

        return self

    @abstractmethod
    def score(self, X, **kwargs):
        """Evaluate the fit quality on X."""
        pass

    @abstractmethod
    def transform(self, X, y=None, **kwargs):
        """Return the low-rank reconstruction of X."""
        pass

    def _extra_attrs(self, ctx: Context[BT, CT, PT]) -> dict[str, Any]:
        """Return additional attributes to set on the estimator after fitting.

        Subclasses override this to expose problem-specific fitted quantities
        (e.g. factor matrices, structure patterns) without touching `fit`.
        """
        return {}


def _objective[
    BT: DataclassInstance,
    CT: DataclassInstance,
    PT: DataclassInstance,
](ctx: Context[BT, CT, PT]) -> float:
    """Compute the full augmented Lagrangian objective.

    Sums contributions from all primal blocks (data fidelity and
    regularisation) and all constraint dual variables (augmented penalty
    terms).
    """
    val = 0.0

    for name, idx in ctx.block_order:
        bgroup: dict[
            BlockDesc,
            Block[Any],
        ] = getattr(ctx.blocks, name)
        val += objective(bgroup[idx], ctx)

    for cnstrnt in fields(ctx.constraints):
        cgroup: dict[
            BlockDesc,
            Constraint[Any],
        ] = getattr(ctx.constraints, cnstrnt.name)
        for c in cgroup.values():
            val += objective(c, ctx)

    return val
