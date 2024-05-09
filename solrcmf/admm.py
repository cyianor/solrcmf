from abc import ABC, abstractmethod
from numpy import inf
from time import process_time
from sklearn.base import BaseEstimator
from sklearn.utils._param_validation import Interval
from numbers import Real, Integral
from typing import Any

from .base import Context


class ADMM(BaseEstimator, ABC):
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
        self.max_iter = max_iter
        self.abs_tol = abs_tol
        self.rel_tol = rel_tol

        self.save_ctx = save_ctx

    @abstractmethod
    def _setup(self, X, **kwargs) -> Context:
        """Setup the estimation problem.

        Called after data is available.
        """
        raise NotImplementedError(
            f"_setup method on {self.__class__.__name__} not implemented"
        )

    def fit(self, X, y=None, **kwargs):
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
                ctx.blocks[name][idx].update(ctx)

            # Update constraints
            for cgroup in ctx.constraints.values():
                for c in cgroup.values():
                    c.update(ctx)

            obj = _objective(ctx)
            gap = obj_old - obj

            objs.append(obj)
            gaps.append(gap)

            if gap <= max(self.rel_tol * obj, self.abs_tol):
                converged = True
                break

            obj_old = obj

        end_time = process_time()

        self.objs_ = objs
        self.gaps_ = gaps
        self.converged_ = converged
        self.objective_value_ = obj
        self.n_iter_ = i + 1
        self.elapsed_process_time_ = end_time - start_time

        for k, v in self._extra_attrs(ctx).items():
            setattr(self, k, v)

        if self.save_ctx:
            self.ctx_ = ctx

        return self

    def score(self, X, **kwargs):
        pass

    def transform(self, X, y=None, **kwargs):
        pass

    def _extra_attrs(self, ctx: Context) -> dict[str, Any]:
        return {}


def _objective(ctx: Context):
    val = 0.0

    for name, idx in ctx.block_order:
        val += ctx.blocks[name][idx].objective(ctx)

    for cgroup in ctx.constraints.values():
        for c in cgroup.values():
            val += c.objective(ctx)

    return val
