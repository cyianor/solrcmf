from numpy import float64, zeros_like, diag, vstack, flatnonzero, s_
from numpy.linalg import qr
from numpy.random import Generator, default_rng
from numpy.typing import NDArray
from collections.abc import Hashable

from .base import Context, ViewDesc


class RandomInitializer:
    def __init__(self, rng: Generator | None = None):
        if rng is None:
            rng = default_rng()

        self.rng = rng

    def __call__(self, ctx: Context):
        for v in ctx.blocks["v"].keys():
            ctx.blocks["v"][v].value = qr(
                self.rng.standard_normal(ctx.blocks["v"][v].shape)
            ).Q
            ctx.blocks["v"][v].initialized = True

            if (
                ctx.params["factor_sparsity"]
                or ctx.params["fixed_factor_pattern"]
            ):
                ctx.blocks["u"][v].value = ctx.blocks["v"][v].value.copy()
                ctx.blocks["u"][v].initialized = True

                ctx.blocks["vp"][v].value = zeros_like(
                    ctx.blocks["v"][v].value
                )
                ctx.blocks["vp"][v].initialized = True

                ctx.constraints["factor"][v].value = zeros_like(
                    ctx.blocks["v"][v].value
                )
                ctx.constraints["factor"][v].initialized = True

        for k in ctx.blocks["d"].keys():
            ctx.blocks["d"][k].value = self.rng.uniform(
                -1.0, 1.0, ctx.blocks["d"][k].shape
            )
            ctx.blocks["d"][k].initialized = True

            ctx.blocks["z"][k].value = (
                ctx.blocks["v"][(k[0],)].value
                @ diag(ctx.blocks["d"][k].value)
                @ ctx.blocks["v"][(k[1],)].value.T
            )
            ctx.blocks["z"][k].initialized = True

            ctx.constraints["mean_structure"][k].value = zeros_like(
                ctx.blocks["z"][k].value
            )
            ctx.constraints["mean_structure"][k].initialized = True


class FromFormerInitializer:
    """Initializes from old context.

    Performs extension to properly initialized state with factor sparsity
    even if the original object was run without.

    Can also reduce the rank of the former setup by removing all ranks which
    were zero across any DBlock.
    """

    def __init__(
        self,
        vs: dict[Hashable, NDArray[float64]],
        ds: dict[
            ViewDesc,
            NDArray[float64],
        ],
        us: dict[Hashable, NDArray[float64]] | None,
        reduce_max_rank: bool = False,
    ):
        self.vs = vs
        self.ds = ds
        self.us = us
        self.reduce_max_rank = reduce_max_rank

    def __call__(self, ctx: Context):
        if self.reduce_max_rank:
            if "structure_pattern" in ctx.params:
                structure_pattern = ctx.params["structure_pattern"]
            else:
                structure_pattern = {k: d != 0.0 for k, d in self.ds.items()}

            active_factors = flatnonzero(
                vstack(
                    [structure_pattern[k] for k in structure_pattern.keys()]
                ).sum(0)
                != 0
            )

            if "structure_pattern" in ctx.params:
                ctx.params["structure_pattern"] = {
                    k: p[active_factors]
                    for k, p in ctx.params["structure_pattern"].items()
                }
            if "factor_pattern" in ctx.params:
                ctx.params["factor_pattern"] = {
                    k: p[:, active_factors]
                    for k, p in ctx.params["factor_pattern"].items()
                }
        else:
            active_factors = s_[:]

        for k in ctx.blocks["v"].keys():
            ctx.blocks["v"][k].value = self.vs[k[0]][:, active_factors].copy()
            ctx.blocks["v"][k].shape = ctx.blocks["v"][k].value.shape
            ctx.blocks["v"][k].initialized = True

            if (
                ctx.params["factor_sparsity"]
                or ctx.params["fixed_factor_pattern"]
            ):
                if self.us is not None:
                    ctx.blocks["u"][k].value = self.us[k[0]][
                        :, active_factors
                    ].copy()
                    ctx.blocks["u"][k].shape = ctx.blocks["u"][k].value.shape
                    ctx.blocks["u"][k].initialized = True

                    ctx.blocks["vp"][k].value = (
                        ctx.blocks["u"][k].value - ctx.blocks["v"][k].value
                    )
                    ctx.blocks["vp"][k].shape = ctx.blocks["vp"][k].value.shape
                    ctx.blocks["vp"][k].initialized = True

                    ctx.constraints["factor"][k].value = zeros_like(
                        ctx.blocks["v"][k].value
                    )
                    ctx.constraints["factor"][k].shape = ctx.constraints[
                        "factor"
                    ][k].value.shape
                    ctx.constraints["factor"][k].initialized = True

        for k in ctx.blocks["d"].keys():
            ctx.blocks["d"][k].value = self.ds[k][active_factors].copy()
            ctx.blocks["d"][k].shape = ctx.blocks["d"][k].value.shape
            ctx.blocks["d"][k].initialized = True

            ctx.blocks["z"][k].value = (
                ctx.blocks["v"][(k[0],)].value
                @ diag(ctx.blocks["d"][k].value)
                @ ctx.blocks["v"][(k[1],)].value.T
            )
            ctx.blocks["z"][k].initialized = True

            ctx.constraints["mean_structure"][k].value = zeros_like(
                ctx.blocks["z"][k].value
            )
            ctx.constraints["mean_structure"][k].initialized = True
