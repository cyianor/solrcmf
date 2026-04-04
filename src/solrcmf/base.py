from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import Field
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


class Context[BT: DataclassInstance, CT: DataclassInstance, PT: DataclassInstance]:
    def __init__(
        self,
        blocks: BT,
        constraints: CT,
        params: PT,
    ):
        self.blocks = blocks
        self.constraints = constraints
        self.params = params
        self.data: dict[
            ViewDesc,
            NDArray[float64],
        ] = {}
        self.block_order: list[tuple[str, BlockDesc]] = []

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


class Block[CtxT: Context, IdxT](ABC):
    value: NDArray[float64]

    def __init__(
        self,
        name: str,
        idx: IdxT,
        shape: tuple[int, ...],
    ):
        self.name = name
        self.idx = idx
        self.shape = shape

        self.initialized = False

    @abstractmethod
    def update(self, ctx: CtxT):
        """Update the block variables."""
        pass

    @abstractmethod
    def objective(self, ctx: CtxT) -> float:
        """Compute the contribution to the objective."""
        return 0.0


class Constraint[CtxT: Context, IdxT](Block[CtxT, IdxT], ABC):
    """Base class for (multi-)affine constraints."""

    @abstractmethod
    def constraint(self, ctx: CtxT) -> NDArray[float64]:
        """Return the lhs of a constraint f(x) = 0."""
        pass

    def update(self, ctx: CtxT):
        """Update the multipliers."""
        self.value += self.constraint(ctx)

    def objective(self, ctx: CtxT) -> float:
        """Compute the contribution to the objective."""
        return (
            0.5
            * ctx.params.rho
            * (
                ((self.constraint(ctx) + self.value) ** 2).sum()
                - (self.value**2).sum()
            )
        )
