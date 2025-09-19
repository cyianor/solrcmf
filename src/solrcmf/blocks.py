from __future__ import annotations

from dataclasses import dataclass, field
from warnings import warn

from numpy import (
    abs,
    argmin,
    bool_,
    diag,
    float64,
    intp,
    maximum,
    ndarray,
    sign,
    sqrt,
    sum,
    vstack,
)
from numpy.linalg import svd
from numpy.typing import NDArray

from .base import Block, Constraint, Context, Entity, ViewDesc


@dataclass
class SolrCMFBlocks:
    z: dict[ViewDesc, ZBlock] = field(default_factory=dict)
    d: dict[ViewDesc, DBlock] = field(default_factory=dict)
    v: dict[Entity, VBlock] = field(default_factory=dict)
    u: dict[Entity, UBlock] = field(default_factory=dict)
    vp: dict[Entity, VpBlock] = field(default_factory=dict)


@dataclass
class SolrCMFConstraints:
    factor: dict[Entity, FactorConstraint] = field(default_factory=dict)
    mean_structure: dict[ViewDesc, MeanStructureConstraint] = field(
        default_factory=dict
    )


@dataclass
class SolrCMFParams:
    rho: float = 0.0
    alpha: float = 0.0
    flat_indices: dict[ViewDesc, NDArray[intp]] = field(default_factory=dict)
    fixed_structure_pattern: bool = False
    structure_pattern: dict[ViewDesc, NDArray[bool_]] = field(
        default_factory=dict
    )
    structure_penalty: float = 0.0
    structure_weights: dict[ViewDesc, NDArray[float64] | float64] = field(
        default_factory=dict
    )
    factor_pruning: bool = False
    max_rank: int = 0
    factor_sparsity: bool = False
    fixed_factor_pattern: bool = False
    factor_pattern: dict[Entity, NDArray[bool_]] = field(default_factory=dict)
    factor_penalty: float = 0.0
    factor_weights: dict[Entity, NDArray[float64] | float64] = field(
        default_factory=dict
    )
    vidx_ridx: dict[Entity, list[tuple[ViewDesc, Entity]]] = field(
        default_factory=dict
    )
    vidx_cidx: dict[Entity, list[tuple[ViewDesc, Entity]]] = field(
        default_factory=dict
    )
    mu: float = 0.0
    vp_weights: dict[Entity, float] = field(default_factory=dict)


SolrCMFContext = Context[SolrCMFBlocks, SolrCMFConstraints, SolrCMFParams]


class ZBlock(Block[SolrCMFContext, ViewDesc]):
    def update(self, ctx: SolrCMFContext):
        self.value = (1.0 - 1.0 / (1.0 + ctx.params.rho)) * (
            ctx.blocks.v[self.idx[0]].value
            @ diag(ctx.blocks.d[self.idx].value)
            @ ctx.blocks.v[self.idx[1]].value.T
            - ctx.constraints.mean_structure[self.idx].value
        )

        self.value.flat[ctx.params.flat_indices[self.idx]] += (
            1.0
            / (1.0 + ctx.params.rho)
            * ctx.data[self.idx].flat[ctx.params.flat_indices[self.idx]]
        )

    def objective(self, ctx: SolrCMFContext) -> float:
        return 0.5 * sum(
            (
                ctx.data[self.idx].flat[ctx.params.flat_indices[self.idx]]
                - self.value.flat[ctx.params.flat_indices[self.idx]]
            )
            ** 2
        )


class DBlock(Block[SolrCMFContext, ViewDesc]):
    active_factors: NDArray[bool_]

    def update(self, ctx: SolrCMFContext):
        tmp = diag(
            ctx.blocks.v[self.idx[0]].value.T
            @ (
                (
                    ctx.blocks.z[self.idx].value
                    + ctx.constraints.mean_structure[self.idx].value
                )
                @ ctx.blocks.v[self.idx[1]].value
            )
        )
        if ctx.params.fixed_structure_pattern:
            # If zero pattern is known
            self.value = tmp * ctx.params.structure_pattern[self.idx]
        else:
            # Soft-thresholding
            self.value = sign(tmp) * maximum(
                (
                    abs(tmp)
                    - ctx.params.structure_penalty
                    * ctx.params.structure_weights[self.idx]
                    / ctx.params.rho
                ),
                0.0,
            )

        if ctx.params.factor_pruning:
            self.active_factors = self.value != 0.0

    def objective(self, ctx: SolrCMFContext) -> float64:
        if ctx.params.fixed_structure_pattern:
            return float64(0.0)

        return (
            ctx.params.structure_penalty
            * ctx.params.structure_weights[self.idx]
            * abs(self.value)
        ).sum()


class VBlock(Block[SolrCMFContext, Entity]):
    def update(self, ctx: SolrCMFContext):
        if ctx.params.factor_pruning:
            active_factors: NDArray[bool_] = (
                vstack([d.active_factors for d in ctx.blocks.d.values()]).sum(
                    axis=0
                )
                != 0
            )

            if sum(active_factors) < ctx.params.max_rank:
                # warn(
                #     "Reducing dimension of integration problem to maximum"
                #     f" rank {sum(active_factors)}"
                # )
                for d in ctx.blocks.d.values():
                    d.value = d.value[active_factors]
                if any(
                    isinstance(s, ndarray) and len(s) > 1
                    for s in ctx.params.structure_weights.values()
                ):
                    ctx.params.structure_weights = {
                        k: s[active_factors] if isinstance(s, ndarray) else s
                        for k, s in ctx.params.structure_weights.items()
                    }
                for k, v in ctx.blocks.v.items():
                    v.value = v.value[:, active_factors]
                    if (
                        ctx.params.factor_sparsity
                        or ctx.params.fixed_factor_pattern
                    ):
                        ctx.blocks.u[k].value = ctx.blocks.u[k].value[
                            :, active_factors
                        ]
                        ctx.blocks.vp[k].value = ctx.blocks.vp[k].value[
                            :, active_factors
                        ]
                        ctx.constraints.factor[
                            k
                        ].value = ctx.constraints.factor[k].value[
                            :, active_factors
                        ]

                ctx.params.max_rank = active_factors.sum()

        tmp = ctx.params.alpha / ctx.params.rho * self.value
        if ctx.params.factor_sparsity or ctx.params.fixed_factor_pattern:
            tmp += (
                ctx.blocks.u[self.idx].value
                - ctx.blocks.vp[self.idx].value
                + ctx.constraints.factor[self.idx].value
            )

        for vidx, cidx in ctx.params.vidx_cidx[self.idx]:
            tmp += (
                (
                    ctx.blocks.z[vidx].value
                    + ctx.constraints.mean_structure[vidx].value
                )
                @ ctx.blocks.v[cidx].value
                @ diag(ctx.blocks.d[vidx].value)
            )

        for vidx, ridx in ctx.params.vidx_ridx[self.idx]:
            tmp += (
                (
                    ctx.blocks.z[vidx].value
                    + ctx.constraints.mean_structure[vidx].value
                ).T
                @ ctx.blocks.v[ridx].value
                @ diag(ctx.blocks.d[vidx].value)
            )

        u, _, vt = svd(tmp, full_matrices=False)
        self.value = u @ vt

    def objective(self, ctx: SolrCMFContext) -> float64:
        return float64(0.0)


class UBlock(Block[SolrCMFContext, Entity]):
    def update(self, ctx: SolrCMFContext):
        m = (
            ctx.blocks.v[self.idx].value
            + ctx.blocks.vp[self.idx].value
            - ctx.constraints.factor[self.idx].value
            + ctx.params.alpha / ctx.params.rho * self.value
        )

        if ctx.params.fixed_factor_pattern:
            # If 0-pattern is known
            m *= ctx.params.factor_pattern[self.idx]
        else:
            # Soft-thresholding
            m = sign(m) * maximum(
                abs(m)
                - ctx.params.factor_penalty
                * ctx.params.factor_weights[self.idx]
                / ctx.params.rho,
                0.0,
            )

            # Deal with edge cases
            for i in (m == 0.0).all(0).nonzero()[0]:
                warn(
                    f"Edge case occurred in U subproblem for index {self.idx}"
                    f" - maximum value in m is {abs(m[:, i]).max()}"
                )
                tmp = (
                    -abs(m[:, i])
                    + ctx.params.factor_penalty
                    * ctx.params.factor_weights[self.idx]
                    / ctx.params.rho
                )
                idx = argmin(tmp)
                sgn = sign(tmp[idx])
                # Set to +/- unit vector
                m[:, i] = 0.0
                m[idx, i] = sgn

        # Column-normalize
        self.value = m / sqrt((m**2).sum(0))

    def objective(self, ctx: SolrCMFContext) -> float:
        if ctx.params.fixed_factor_pattern:
            return 0.0

        return (
            ctx.params.factor_penalty
            * ctx.params.factor_weights[self.idx]
            * abs(self.value)
        ).sum()


class VpBlock(Block[SolrCMFContext, Entity]):
    def update(self, ctx: SolrCMFContext):
        self.value = (
            ctx.params.rho
            / (
                ctx.params.rho
                + ctx.params.mu * ctx.params.vp_weights[self.idx]
            )
            * (
                ctx.blocks.u[self.idx].value
                - ctx.blocks.v[self.idx].value
                + ctx.constraints.factor[self.idx].value
            )
        )

    def objective(self, ctx: SolrCMFContext) -> float:
        return (
            0.5
            * ctx.params.mu
            * ctx.params.vp_weights[self.idx]
            * (self.value**2).sum()
        )


class MeanStructureConstraint(Constraint[SolrCMFContext, ViewDesc]):
    def constraint(self, ctx: SolrCMFContext) -> NDArray[float64]:
        return (
            ctx.blocks.z[self.idx].value
            - ctx.blocks.v[self.idx[0]].value
            @ diag(ctx.blocks.d[self.idx].value)
            @ ctx.blocks.v[self.idx[1]].value.T
        )


class FactorConstraint(Constraint[SolrCMFContext, Entity]):
    def constraint(self, ctx: SolrCMFContext) -> NDArray[float64]:
        return (
            ctx.blocks.u[self.idx].value
            - ctx.blocks.v[self.idx].value
            - ctx.blocks.vp[self.idx].value
        )
