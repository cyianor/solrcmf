from __future__ import annotations

from abc import ABCMeta, abstractmethod
from numpy.typing import NDArray
from numpy import float64
from typing import Any, TypeVar, Generic

try:
    from typing import Unpack
except ImportError:
    from typing_extensions import Unpack

Entity = str | int
BlockDesc = tuple[Entity, Unpack[tuple[Entity, ...]]] | Entity
ViewDesc = tuple[Entity, Entity, Unpack[tuple[Entity, ...]]]

Blocks = TypeVar("Blocks")
Constraints = TypeVar("Constraints")
Index = TypeVar("Index")


class Context[Blocks, Constraints]:
    def __init__(self, blocks: Blocks, constraints: Constraints):
        self.blocks = blocks
        self.constraints = constraints
        self.data: dict[
            ViewDesc,
            NDArray[float64],
        ] = {}
        self.params: dict[str, Any] = {}
        self.block_order: list[tuple] = []

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


class Block(Generic[Blocks, Constraints, Index], metaclass=ABCMeta):
    value: NDArray[float64]

    def __init__(
        self,
        name: str,
        idx: Index,
        shape: tuple[int, ...],
    ):
        self.name = name
        self.idx = idx
        self.shape = shape

        self.initialized = False

    @abstractmethod
    def update(self, ctx: Context[Blocks, Constraints]):
        pass

    @abstractmethod
    def objective(self, ctx: Context[Blocks, Constraints]) -> float:
        return 0.0


class Constraint(Block[Blocks, Constraints, Index], metaclass=ABCMeta):
    """Base class for (multi-)affine constraints"""

    @abstractmethod
    def constraint(
        self, ctx: Context[Blocks, Constraints]
    ) -> NDArray[float64]:
        """Returns the lhs of a constraint f(x) = 0"""
        pass

    def update(self, ctx: Context[Blocks, Constraints]):
        """Update the multipliers"""
        self.value += self.constraint(ctx)

    def objective(self, ctx: Context[Blocks, Constraints]) -> float:
        return (
            0.5
            * ctx.params["rho"]
            * (
                ((self.constraint(ctx) + self.value) ** 2).sum()
                - (self.value**2).sum()
            )
        )
