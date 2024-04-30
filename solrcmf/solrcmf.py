from numpy.typing import NDArray
from numpy import (
    float64,
    bool_,
    diag,
    intp,
    isnan,
    flatnonzero,
    logical_not,
    logical_and,
    vstack,
    sqrt,
    zeros,
    ones,
)
from typing import Any
from warnings import warn
from sklearn.utils.validation import check_is_fitted, check_array
from sklearn.utils._param_validation import Interval, StrOptions
from numbers import Real, Integral
from collections.abc import Hashable

from .base import Context, ViewDesc
from .admm import ADMM
from .initializer import RandomInitializer, FromFormerInitializer
from .blocks import VBlock, DBlock, ZBlock, UBlock, VpBlock
from .constraints import MeanStructureConstraint, FactorConstraint
from .metrics import neg_mean_squared_error


class SolrCMF(ADMM):
    _parameter_constraints = {
        **ADMM._parameter_constraints,
        "structure_penalty": [Interval(Real, 0, None, closed="left"), None],
        "max_rank": [Interval(Integral, 1, None, closed="left"), None],
        "factor_penalty": [Interval(Real, 0, None, closed="neither"), None],
        "factor_pruning": ["boolean"],
        "init": [StrOptions({"random", "custom"})],
        "init_kwargs": [dict, None],
        "rho": [Interval(Real, 0.0, None, closed="neither"), None],
        "alpha": [Interval(Real, 0.0, None, closed="left"), None],
        "mu": [Interval(Real, 0.0, None, closed="neither"), None],
    }

    def __init__(
        self,
        *,
        structure_penalty: float | None = None,
        max_rank: int | None = None,
        factor_penalty: float | None = None,
        factor_pruning: bool = True,
        init: str = "random",
        init_kwargs: dict[str, Any] | None = None,
        rho: float | None = None,
        alpha: float | None = None,
        mu: float | None = None,
        max_iter: int = 1000,
        abs_tol: float = 1e-6,
        rel_tol: float = 1e-6,
        save_ctx: bool = False,
    ):
        super().__init__(
            max_iter=max_iter,
            abs_tol=abs_tol,
            rel_tol=rel_tol,
            save_ctx=save_ctx,
        )

        self.structure_penalty = structure_penalty
        self.max_rank = max_rank
        self.factor_penalty = factor_penalty
        self.factor_pruning = factor_pruning
        self.init = init
        self.init_kwargs = init_kwargs
        self.rho = rho
        self.alpha = alpha
        self.mu = mu

    def _setup(
        self,
        X: dict[ViewDesc, NDArray[float64]],
        *,
        indices: dict[ViewDesc, NDArray[intp]] | None = None,
        structure_weights: (
            dict[ViewDesc, NDArray[float64] | float64] | None
        ) = None,
        structure_pattern: dict[ViewDesc, NDArray[bool_]] | None = None,
        factor_weights: dict[Hashable, NDArray[float64] | float64]
        | None = None,
        factor_pattern: dict[Hashable, NDArray[bool_]] | None = None,
        vs: dict[Hashable, NDArray[float64]] | None = None,
        ds: dict[ViewDesc, NDArray[float64]] | None = None,
        us: dict[Hashable, NDArray[float64]] | None = None,
    ):
        # A context will be populated throughout setup
        ctx = Context()

        assert isinstance(
            X, dict
        ), "'X' needs to be a dictionary of data matrices"

        for x in X.values():
            x = check_array(x, force_all_finite="allow-nan")

        layout = X.keys()
        # The first two tuple indices indicate the views.
        # The rest are arbitrary to make the indices different if the
        # same views appear.
        views = set([k[i] for k in layout for i in range(2)])
        viewdims_set = set(
            [(k[i], x.shape[i]) for k, x in X.items() for i in range(2)]
        )
        assert len(views) == len(viewdims_set), (
            "Views do not have consistent dimensions across layout. Received"
            f" matrices with dimensions {viewdims_set}."
        )
        viewdims = dict(viewdims_set)

        ctx.data = X

        assert bool(
            self.structure_penalty is None and self.max_rank is None
        ) ^ bool(structure_pattern is None), (
            "Either both structure_penalty and max_rank, or "
            " structure_pattern need to be provided. The respective"
            " other(s) need to be None."
        )

        assert self.factor_penalty is None or factor_pattern is None, (
            "One or both of `factor_penalty` and `factor_pattern`"
            " need to be None."
        )

        if self.structure_penalty is not None:
            ctx.params["structure_penalty"] = self.structure_penalty
            max_rank = self.max_rank

            if structure_weights is None:
                ctx.params["structure_weights"] = {k: 1.0 for k in layout}
            else:
                # TODO: Add argument check
                ctx.params["structure_weights"] = structure_weights

            ctx.params["fixed_structure_pattern"] = False
        else:
            assert (
                not self.factor_pruning
            ), "Set 'factor_pruning' to False to use 'structure_pattern'"
            assert structure_pattern.keys() == X.keys(), (
                "'structure_pattern' must contain one pattern for each data"
                f" matrix. Expected: {X.keys()}, observed:"
                f" {structure_pattern.keys()}"
            )

            rks = set(p.shape[0] for p in structure_pattern.values())
            assert len(rks) == 1, (
                "All patterns in 'structure_pattern' should have the same"
                f" length. Observed lengths: {rks}"
            )
            # Extract the only element
            max_rank = list(rks)[0]

            ctx.params["structure_pattern"] = structure_pattern
            ctx.params["fixed_structure_pattern"] = True

        ctx.params["max_rank"] = max_rank

        for v, p in viewdims.items():
            assert p >= max_rank, (
                f"View {v} has dimension {p} which is less than the maximum"
                f" requested rank {max_rank}"
            )

        if self.factor_penalty is not None:
            if self.mu is None:
                ctx.params["mu"] = 10.0
            else:
                ctx.params["mu"] = self.mu

            # assert self.mu is not None, (
            #     f"mu needs to be provided in {self.__class__.__name__} when"
            #     " factor_penalty is not None"
            # )
            # ctx.params["mu"] = self.mu

            ctx.params["factor_penalty"] = self.factor_penalty
            ctx.params["factor_sparsity"] = True

            if factor_weights is None:
                ctx.params["factor_weights"] = {
                    k: 1.0 / sqrt(p) for k, p in viewdims.items()
                }
            else:
                # TODO: Add argument check
                ctx.params["factor_weights"] = factor_weights
        else:
            ctx.params["factor_sparsity"] = False

        if factor_pattern is not None:
            assert (
                not self.factor_pruning
            ), "Set 'factor_pruning' to False to use 'factor_pattern'"

            if self.mu is None:
                ctx.params["mu"] = 10.0
            else:
                ctx.params["mu"] = self.mu

            # assert self.mu is not None, (
            #     f"mu needs to be provided in {self.__class__.__name__} when"
            #     " factor_pattern is not None"
            # )
            # ctx.params["mu"] = self.mu

            # Check factor pattern's correctness
            assert factor_pattern.keys() == viewdims.keys(), (
                "'factor_pattern' needs to contain a pattern for each view."
                f" Views = {viewdims.keys()}, Patterns available for views ="
                f" {factor_pattern.keys()}"
            )
            dims = {k: p.shape[0] for k, p in factor_pattern.items()}
            assert dims == viewdims, (
                f"View dimensions in 'factor_pattern' ({dims}) do not agree"
                f" with view dimensions in data ({viewdims})."
            )
            rks = set(p.shape[1] for p in factor_pattern.values())
            assert len(rks) == 1, (
                "The patterns in 'factor_pattern' need to have the same"
                f" number of columns. Observed sizes = {rks}"
            )
            # Extract the only element
            rk = list(rks)[0]
            assert rk == max_rank, (
                "Number of columns in 'factor_pattern' needs to match"
                " 'max_rank' or number of elements in each"
                f" 'structure_pattern'. Expected: {max_rank}, observed:"
                f" {rk}"
            )

            ctx.params["factor_pattern"] = factor_pattern
            ctx.params["fixed_factor_pattern"] = True
        else:
            ctx.params["fixed_factor_pattern"] = False

        if ctx.params["factor_sparsity"] or ctx.params["fixed_factor_pattern"]:
            ctx.params["vp_weights"] = {
                k: 1.0 / sqrt(p) for k, p in viewdims.items()
            }
            max_vp_w = max([w for w in ctx.params["vp_weights"].values()])
            min_vp_w = min([w for w in ctx.params["vp_weights"].values()])

            rho_lb = _rho_lower_bound(ctx.params["mu"], min_vp_w, max_vp_w)
        else:
            rho_lb = _rho_lower_bound()

        if self.rho is None:
            rho = rho_lb + 0.1
        else:
            rho = self.rho

        assert rho > rho_lb, (
            f"rho needs to be greater than {rho_lb} in"
            f" {self.__class__.__name__}; now it is {rho}"
        )

        if ctx.params["factor_sparsity"]:
            u_edge_cases = {
                k: self.factor_penalty * w * sqrt(viewdims[k]) / rho
                for k, w in ctx.params["factor_weights"].items()
            }
            if any([u >= 1 for u in u_edge_cases.values()]):
                warn(
                    "For numerical stability, factor_penalty * weight[k] *"
                    " sqrt(viewdims[k]) / rho < 1 should hold for all views k."
                    f" Here: {u_edge_cases}"
                )

        ctx.params["rho"] = rho
        if self.alpha is None:
            ctx.params["alpha"] = 1e-3 * ctx.params["rho"]
        else:
            ctx.params["alpha"] = self.alpha

        # print(f"alpha = {ctx.params['alpha']}")

        ctx.params["factor_pruning"] = self.factor_pruning

        ctx.params["vidx_ridx"] = {
            (v,): [(k, (k[0],)) for k in layout if k[1] == v] for v in views
        }
        ctx.params["vidx_cidx"] = {
            (v,): [(k, (k[1],)) for k in layout if k[0] == v] for v in views
        }

        # Set up ADMM blocks and constraints
        for v in views:
            ctx.add_block("v", (v,), VBlock, (viewdims[v], max_rank))
        for k in layout:
            ctx.add_block("d", k, DBlock, (max_rank,))

            if self.factor_pruning:
                ctx.blocks["d"][k].active_factors = ones(
                    (max_rank,), dtype=bool_
                )

        if self.factor_penalty is not None or factor_pattern is not None:
            for v in views:
                ctx.add_block("u", (v,), UBlock, (viewdims[v], max_rank))

            # Important to keep vp blocks just before z blocks
            for v in views:
                ctx.add_block("vp", (v,), VpBlock, (viewdims[v], max_rank))
                ctx.add_constraint(
                    "factor",
                    (v,),
                    FactorConstraint,
                    (viewdims[v], max_rank),
                )

        # Important to keep z blocks last
        for k in layout:
            ctx.add_block("z", k, ZBlock, (viewdims[k[0]], viewdims[k[1]]))
            ctx.add_constraint(
                "mean_structure",
                k,
                MeanStructureConstraint,
                (viewdims[k[0]], viewdims[k[1]]),
            )

        if self.init_kwargs is None:
            init_kwargs = {}
        else:
            init_kwargs = self.init_kwargs

        # Initialize blocks and constraints
        if self.init == "random":
            init_fn = RandomInitializer(**init_kwargs)
        elif self.init == "custom":
            assert vs is not None and ds is not None, (
                f"If 'init' is \"custom\" in {self.__class__.__name__} then"
                " 'vs' and 'ds' need to be provided to method 'fit' as"
                " keyword arguments."
            )
            if (
                ctx.params["factor_sparsity"]
                or ctx.params["fixed_factor_pattern"]
            ):
                assert us is not None, (
                    "If 'init' is \"custom\" in"
                    f" {self.__class__.__name__} then 'us' needs to be"
                    " provided to method 'fit' as a keyword argument."
                )

            init_fn = FromFormerInitializer(
                vs=vs,
                ds=ds,
                us=us,
                **init_kwargs,
            )

        # Call initializer
        init_fn(ctx)

        # Remove nan entries from `flat_indices` if indices provided.
        # Otherwise have non-nan indices as `flat_indices`
        ctx.params["flat_indices"] = {}
        for k, x in ctx.data.items():
            if indices is None:
                ctx.params["flat_indices"][k] = flatnonzero(
                    logical_not(isnan(x))
                )
            else:
                indices_mask = zeros(x.size, dtype=bool_)
                indices_mask[indices[k]] = True
                not_nan_mask = logical_not(isnan(x)).ravel()
                ctx.params["flat_indices"][k] = flatnonzero(
                    logical_and(indices_mask, not_nan_mask)
                )

        return ctx

    def transform(
        self,
        X: dict[ViewDesc, NDArray[float64]],
        y=None,
    ):
        check_is_fitted(self)

        return {
            k: self.vs_[k[0]] @ diag(d) @ self.vs_[k[1]].T
            for k, d in self.ds_.items()
        }

    def score(
        self,
        X: dict[ViewDesc, NDArray[float64]],
        *,
        indices: dict[ViewDesc, NDArray[intp]] | None = None,
    ):
        check_is_fitted(self)

        return neg_mean_squared_error(X, self.transform(X), indices=indices)

    def structure_pattern(self):
        check_is_fitted(self)

        return {k: d != 0.0 for k, d in self.ds_.items()}

    def factor_pattern(self):
        check_is_fitted(self)

        if hasattr(self, "us_"):
            return {k: u != 0.0 for k, u in self.us_.items()}
        else:
            None

    def _extra_attrs(self, ctx: Context):
        out = {}
        out["vs_"] = {k[0]: b.value for k, b in ctx.blocks["v"].items()}
        out["ds_"] = {k: b.value for k, b in ctx.blocks["d"].items()}
        if ctx.params["factor_sparsity"] or ctx.params["fixed_factor_pattern"]:
            out["us_"] = {k[0]: b.value for k, b in ctx.blocks["u"].items()}

        out["est_max_rank_"] = sum(
            vstack([d != 0.0 for d in out["ds_"].values()]).sum(0) != 0
        )

        return out

    def _more_tags(self):
        return {
            "X_types": "dict",
        }


def _rho_lower_bound(
    mu: float | None = None,
    min_vp_w: float | None = None,
    max_vp_w: float | None = None,
):
    if mu is not None and min_vp_w is not None and max_vp_w is not None:
        return max(
            2.0,
            max(
                2.0 * mu * max_vp_w**2 / min_vp_w,
                0.5
                * (1.0 + mu * max_vp_w)
                * (1.0 + 2.0 * max_vp_w / min_vp_w) ** 2,
            ),
        )
    else:
        return 2.0
