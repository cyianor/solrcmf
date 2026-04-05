"""Basic building blocks used in multi-block ADMM.

This module contains the basic building blocks such as Block,
Constraint, or Context which are used in multi-block ADMM algorithms.

Note that blocks require the implementation of the `singledispatch`
methods `update` and `objective`, whereas constraints require
implementation of the `singledispatch` function `constraint`.

Pre-defined functionality is purposefully generic to allow
implementation of any multi-block ADMM algorithm using this framework.
"""

from __future__ import annotations

from dataclasses import Field, dataclass, field
from functools import singledispatch
from typing import Protocol

from numpy import float64
from numpy.typing import NDArray

type Entity = str | int
type BlockDesc = tuple[Entity, *tuple[Entity, ...]] | Entity
type ViewDesc = tuple[Entity, Entity, *tuple[Entity, ...]]


class DataclassInstance(Protocol):
    """Protocol for dataclass instances.

    Matches the DataclassInstance definition in typeshed.
    """

    __dataclass_fields__: dict[str, Field]


class ConstraintParams(DataclassInstance, Protocol):
    """Protocol for parameters for Constraint blocks."""

    rho: float


@dataclass
class Block[IdxT]:
    """A single block of ADMM primal variables.

    Each block owns one array `value` and contributes to both the primal
    update (via `update`) and the objective function (via `objective`).
    It is identified by a name and an index `idx`. The `update` and
    `objective` singledispatch functions are registered separately for
    each concrete subclass.

    Attributes:
        name: Attribute name on the blocks dataclass (e.g. "z", "d", "v").
        idx: Key under which this block is stored in the dict on that attr.
        shape: Expected shape of `value`.
        value: The current iterate; initialised by the concrete update.

    """

    name: str
    idx: IdxT
    shape: tuple[int, ...]
    value: NDArray[float64] = field(init=False, repr=False)


@dataclass
class Constraint[IdxT](Block[IdxT]):
    """A block of ADMM dual variables enforcing a multi-affine constraint.

    `value` holds the dual multiplier. `update` performs the dual ascent
    step value += residual. `objective` computes the augmented Lagrangian
    penalty term. Both use the `constraint` singledispatch function to
    obtain the primal residual, which is cached in `residual` after each
    `update` to avoid recomputation in `objective`.

    Attributes:
        residual: The most recently computed primal residual; set by
            `update` and consumed by `objective` within the same iteration.

    """

    residual: NDArray[float64] = field(init=False, repr=False)


@dataclass
class Context[
    BT: DataclassInstance,
    CT: DataclassInstance,
    PT: DataclassInstance,
]:
    """Shared state passed to every block and constraint update.

    Holds the primal blocks, dual constraints, algorithm parameters,
    observed data, and the ordered list of blocks to update each iteration.

    Attributes:
        blocks: Dataclass holding all primal block dicts.
        constraints: Dataclass holding all constraint (dual variable) dicts.
        params: Algorithm parameters (rho, penalties, etc.).
        data: Observed data matrices keyed by view descriptor.
        block_order: Ordered list of (name, idx) pairs defining the update
            sequence within each ADMM iteration.

    """

    blocks: BT
    constraints: CT
    params: PT
    data: dict[ViewDesc, NDArray[float64]] = field(default_factory=dict)
    block_order: list[tuple[str, BlockDesc]] = field(default_factory=list)

    def add_block(
        self,
        name: str,
        idx: BlockDesc,
        block_type: type[Block],
        shape: tuple[int, ...],
    ):
        """Instantiate a primal block and register it in the update order."""
        self.block_order.append((name, idx))
        getattr(self.blocks, name)[idx] = block_type(name, idx, shape)

    def add_constraint(
        self,
        name: str,
        idx: BlockDesc,
        constraint_type: type[Constraint],
        shape: tuple[int, ...],
    ):
        """Instantiate a constraint (dual variable) block."""
        getattr(self.constraints, name)[idx] = constraint_type(
            name, idx, shape
        )


@singledispatch
def update[
    BT: DataclassInstance,
    CT: DataclassInstance,
    PT: DataclassInstance,
](block: Block, ctx: Context[BT, CT, PT]):
    """Update the block variables."""
    raise NotImplementedError(f"update() not implemented for {type(block)}.")


@singledispatch
def constraint[
    BT: DataclassInstance,
    CT: DataclassInstance,
    PT: DataclassInstance,
](block: Constraint, _ctx: Context[BT, CT, PT]) -> NDArray[float64]:
    """Return the lhs of a constraint f(x) = 0."""
    raise NotImplementedError(
        f"constraint() not implemented for {type(block)}."
    )


@update.register
def _[
    BT: DataclassInstance,
    CT: DataclassInstance,
    PT: DataclassInstance,
](block: Constraint, ctx: Context[BT, CT, PT]):
    """Update the multipliers."""
    block.residual = constraint(block, ctx)
    block.value += block.residual


@singledispatch
def objective[
    BT: DataclassInstance,
    CT: DataclassInstance,
    PT: DataclassInstance,
](block: Block, ctx: Context[BT, CT, PT]) -> float:
    """Compute the contribution to the objective."""
    return 0.0


@objective.register
def _[
    BT: DataclassInstance,
    CT: DataclassInstance,
    PT: ConstraintParams,
](block: Constraint, ctx: Context[BT, CT, PT]) -> float:
    """Compute the contribution to the objective."""
    return (
        0.5
        * ctx.params.rho
        * (
            ((block.residual + block.value) ** 2).sum()
            - (block.value**2).sum()
        )
    )
