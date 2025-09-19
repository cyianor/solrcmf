from __future__ import annotations

from abc import ABCMeta, abstractmethod
from typing import Generic, TypeVar

from numpy import float64
from numpy.typing import NDArray

try:
    from typing import Unpack
except ImportError:
    from typing_extensions import Unpack

Entity = str | int
BlockDesc = tuple[Entity, Unpack[tuple[Entity, ...]]] | Entity
ViewDesc = tuple[Entity, Entity, Unpack[tuple[Entity, ...]]]

BlocksType = TypeVar("BlocksType")
ConstraintsType = TypeVar("ConstraintsType")
ParamsType = TypeVar("ParamsType")
IndexType = TypeVar("IndexType")


class Context[BlocksType, ConstraintsType, ParamsType]:
    def __init__(
        self,
        blocks: BlocksType,
        constraints: ConstraintsType,
        params: ParamsType,
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


ContextType = TypeVar("ContextType", bound=Context)


class Block(Generic[ContextType, IndexType], metaclass=ABCMeta):
    value: NDArray[float64]

    def __init__(
        self,
        name: str,
        idx: IndexType,
        shape: tuple[int, ...],
    ):
        self.name = name
        self.idx = idx
        self.shape = shape

        self.initialized = False

    @abstractmethod
    def update(self, ctx: ContextType):
        """Update the block variables."""
        pass

    @abstractmethod
    def objective(self, ctx: ContextType) -> float:
        """Compute the contribution to the objective."""
        return 0.0


class Constraint(Block[ContextType, IndexType], metaclass=ABCMeta):
    """Base class for (multi-)affine constraints."""

    @abstractmethod
    def constraint(self, ctx: ContextType) -> NDArray[float64]:
        """Return the lhs of a constraint f(x) = 0."""
        pass

    def update(self, ctx: ContextType):
        """Update the multipliers."""
        self.value += self.constraint(ctx)

    def objective(self, ctx: ContextType) -> float:
        """Compute the contribution to the objective."""
        return (
            0.5
            * ctx.params.rho
            * (
                ((self.constraint(ctx) + self.value) ** 2).sum()
                - (self.value**2).sum()
            )
        )
