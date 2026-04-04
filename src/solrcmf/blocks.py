from __future__ import annotations

from dataclasses import dataclass, field
from warnings import warn

from numpy import (
    abs,
    argmin,
    bool_,
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

from .base import (
    Block,
    Constraint,
    Context,
    Entity,
    ViewDesc,
    constraint,
    objective,
    update,
)


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


type SolrCMFContext = Context[SolrCMFBlocks, SolrCMFConstraints, SolrCMFParams]


class ZBlock(Block[ViewDesc]):
    """Low-rank reconstruction variable Z for a given data matrix.

    Z absorbs the data fidelity term. Its update is the proximal step
    that blends the current low-rank estimate V D V^T with the observed
    entries of X, leaving missing entries equal to the estimate.
    """


@update.register
def _(block: ZBlock, ctx: SolrCMFContext):
    """Proximal step blending observed data with the low-rank estimate."""
    block.value = (1.0 - 1.0 / (1.0 + ctx.params.rho)) * (
        (ctx.blocks.v[block.idx[0]].value * ctx.blocks.d[block.idx].value)
        @ ctx.blocks.v[block.idx[1]].value.T
        - ctx.constraints.mean_structure[block.idx].value
    )

    block.value.flat[ctx.params.flat_indices[block.idx]] += (
        1.0
        / (1.0 + ctx.params.rho)
        * ctx.data[block.idx].flat[ctx.params.flat_indices[block.idx]]
    )


@objective.register
def _(block: ZBlock, ctx: SolrCMFContext) -> float:
    """Half the squared reconstruction error on observed entries."""
    return 0.5 * sum(
        (
            ctx.data[block.idx].flat[ctx.params.flat_indices[block.idx]]
            - block.value.flat[ctx.params.flat_indices[block.idx]]
        )
        ** 2
    )


@dataclass
class DBlock(Block[ViewDesc]):
    """Diagonal scaling matrix D for a given view pair.

    D encodes the per-factor signal strengths. Its update applies
    soft-thresholding to learn the structure pattern or masks by a
    fixed pattern when one is provided.
    """

    active_factors: NDArray[bool_] = field(init=False, repr=False)


@update.register
def _(block: DBlock, ctx: SolrCMFContext):
    """Soft-threshold or pattern-mask the projected residual."""
    M = (
        ctx.blocks.z[block.idx].value
        + ctx.constraints.mean_structure[block.idx].value
    )
    tmp = (
        ctx.blocks.v[block.idx[0]].value
        * (M @ ctx.blocks.v[block.idx[1]].value)
    ).sum(axis=0)
    if ctx.params.fixed_structure_pattern:
        # If zero pattern is known
        block.value = tmp * ctx.params.structure_pattern[block.idx]
    else:
        # Soft-thresholding
        block.value = sign(tmp) * maximum(
            (
                abs(tmp)
                - ctx.params.structure_penalty
                * ctx.params.structure_weights[block.idx]
                / ctx.params.rho
            ),
            0.0,
        )

    if ctx.params.factor_pruning:
        block.active_factors = block.value != 0.0


@objective.register
def _(block: DBlock, ctx: SolrCMFContext) -> float64:
    """L1 structure penalty weighted by structure_weights."""
    if ctx.params.fixed_structure_pattern:
        return float64(0.0)

    return (
        ctx.params.structure_penalty
        * ctx.params.structure_weights[block.idx]
        * abs(block.value)
    ).sum()


class VBlock(Block[Entity]):
    """Orthogonal factor matrix V for a given view.

    V is updated via a Procrustes step: SVD of the gradient aggregated
    from all data matrices that involve this view yields the nearest
    column-orthonormal matrix. Factor pruning (removing globally inactive
    columns across all D blocks) is also handled here.
    """


@update.register
def _(block: VBlock, ctx: SolrCMFContext):
    """Procrustes step via SVD; prune inactive columns if enabled."""
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
                    k: s if isinstance(s, float64) else s[active_factors]
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
                    ctx.constraints.factor[k].value = ctx.constraints.factor[
                        k
                    ].value[:, active_factors]

            ctx.params.max_rank = active_factors.sum()

    tmp = ctx.params.alpha / ctx.params.rho * block.value
    if ctx.params.factor_sparsity or ctx.params.fixed_factor_pattern:
        tmp += (
            ctx.blocks.u[block.idx].value
            - ctx.blocks.vp[block.idx].value
            + ctx.constraints.factor[block.idx].value
        )

    for vidx, cidx in ctx.params.vidx_cidx[block.idx]:
        tmp += (
            ctx.blocks.z[vidx].value
            + ctx.constraints.mean_structure[vidx].value
        ) @ (ctx.blocks.v[cidx].value * ctx.blocks.d[vidx].value)

    for vidx, ridx in ctx.params.vidx_ridx[block.idx]:
        tmp += (
            (
                ctx.blocks.z[vidx].value
                + ctx.constraints.mean_structure[vidx].value
            ).T
            @ ctx.blocks.v[ridx].value
        ) * ctx.blocks.d[vidx].value

    u, _, vt = svd(tmp, full_matrices=False)
    block.value = u @ vt


@objective.register
def _(block: VBlock, ctx: SolrCMFContext) -> float64:
    """Return 0 since V carries no direct penalty term."""
    return float64(0.0)


class UBlock(Block[Entity]):
    """Sparse factor variable U for a given view.

    U decouples factor sparsity from the orthogonality constraint on V.
    Its update applies soft-thresholding (or fixed-pattern masking)
    followed by column normalization.
    """


@update.register
def _(block: UBlock, ctx: SolrCMFContext):
    """Soft-threshold or pattern-mask then column-normalize."""
    m = (
        ctx.blocks.v[block.idx].value
        + ctx.blocks.vp[block.idx].value
        - ctx.constraints.factor[block.idx].value
        + ctx.params.alpha / ctx.params.rho * block.value
    )

    if ctx.params.fixed_factor_pattern:
        # If 0-pattern is known
        m *= ctx.params.factor_pattern[block.idx]
    else:
        # Soft-thresholding
        m = sign(m) * maximum(
            abs(m)
            - ctx.params.factor_penalty
            * ctx.params.factor_weights[block.idx]
            / ctx.params.rho,
            0.0,
        )

        # Deal with edge cases
        for i in (m == 0.0).all(0).nonzero()[0]:
            warn(
                f"Edge case occurred in U subproblem for index {block.idx}"
                f" - maximum value in m is {abs(m[:, i]).max()}"
            )
            tmp = (
                -abs(m[:, i])
                + ctx.params.factor_penalty
                * ctx.params.factor_weights[block.idx]
                / ctx.params.rho
            )
            idx = argmin(tmp)
            sgn = sign(tmp[idx])
            # Set to +/- unit vector
            m[:, i] = 0.0
            m[idx, i] = sgn

    # Column-normalize
    block.value = m / sqrt((m**2).sum(0))


@objective.register
def _(block: UBlock, ctx: SolrCMFContext) -> float:
    """L1 factor sparsity penalty weighted by factor_weights."""
    if ctx.params.fixed_factor_pattern:
        return 0.0

    return (
        ctx.params.factor_penalty
        * ctx.params.factor_weights[block.idx]
        * abs(block.value)
    ).sum()


class VpBlock(Block[Entity]):
    """Auxiliary slack variable V' that decouples the U and V subproblems.

    Introducing V' turns the coupling constraint U = V into two separate
    constraints, U - V' = 0 and V' - V = 0, so the sparse U update and the
    orthogonal V update can each be solved independently. The mu penalty
    (0.5 * mu * w * ||V'||_F^2) controls how tightly V' tracks V.
    """


@update.register
def _(block: VpBlock, ctx: SolrCMFContext):
    """Proximal step for the ridge penalty 0.5 * mu * w * ||V'||_F^2."""
    block.value = (
        ctx.params.rho
        / (ctx.params.rho + ctx.params.mu * ctx.params.vp_weights[block.idx])
        * (
            ctx.blocks.u[block.idx].value
            - ctx.blocks.v[block.idx].value
            + ctx.constraints.factor[block.idx].value
        )
    )


@objective.register
def _(block: VpBlock, ctx: SolrCMFContext) -> float:
    """Ridge penalty 0.5 * mu * w * ||V'||_F^2."""
    return (
        0.5
        * ctx.params.mu
        * ctx.params.vp_weights[block.idx]
        * (block.value**2).sum()
    )


class MeanStructureConstraint(Constraint[ViewDesc]):
    """ADMM dual variable for the constraint Z = V D V^T."""


@constraint.register
def _(block: MeanStructureConstraint, ctx: SolrCMFContext) -> NDArray[float64]:
    """Return the primal residual Z - V D V^T."""
    return (
        ctx.blocks.z[block.idx].value
        - (ctx.blocks.v[block.idx[0]].value * ctx.blocks.d[block.idx].value)
        @ ctx.blocks.v[block.idx[1]].value.T
    )


class FactorConstraint(Constraint[Entity]):
    """ADMM dual variable for the constraint U = V + V'."""


@constraint.register
def _(block: FactorConstraint, ctx: SolrCMFContext) -> NDArray[float64]:
    """Return the primal residual U - V - V'."""
    return (
        ctx.blocks.u[block.idx].value
        - ctx.blocks.v[block.idx].value
        - ctx.blocks.vp[block.idx].value
    )
