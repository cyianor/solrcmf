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
    name: str
    idx: IdxT
    shape: tuple[int, ...]
    value: NDArray[float64] = field(init=False, repr=False)


class Constraint[IdxT](Block[IdxT]):
    """Base class for (multi-)affine constraints."""


@dataclass
class Context[
    BT: DataclassInstance,
    CT: DataclassInstance,
    PT: DataclassInstance,
]:
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
        self.block_order.append((name, idx))
        getattr(self.blocks, name)[idx] = block_type(name, idx, shape)

    def add_constraint(
        self,
        name: str,
        idx: BlockDesc,
        constraint_type: type[Constraint],
        shape: tuple[int, ...],
    ):
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
    block.value += constraint(block, ctx)


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
            ((constraint(block, ctx) + block.value) ** 2).sum()
            - (block.value**2).sum()
        )
    )
