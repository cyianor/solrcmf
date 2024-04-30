from __future__ import annotations

from abc import ABCMeta, abstractmethod
from numpy.typing import NDArray
from numpy import float64
from typing import Any
from collections.abc import Hashable

try:
    from typing import Unpack
except ImportError:
    from typing_extensions import Unpack


ViewDesc = tuple[Hashable, Hashable, Unpack[tuple[Hashable, ...]]]


class Context:
    def __init__(self):
        self.blocks: dict[str, dict[ViewDesc, Block]] = {}
        self.constraints: dict[str, dict[ViewDesc, Constraint]] = {}
        self.data: dict[
            ViewDesc,
            NDArray[float64],
        ] = {}
        self.params: dict[str, Any] = {}
        self.block_order = []

    def add_block(
        self,
        name: str,
        idx: ViewDesc,
        block_type: type[Block],
        shape: tuple[int, ...],
    ):
        self.block_order.append((name, idx))
        if name not in self.blocks:
            self.blocks.update({name: {}})

        self.blocks[name][idx] = block_type(name, idx, shape)

    def add_constraint(
        self,
        name: str,
        idx: ViewDesc,
        constraint_type: type[Constraint],
        shape: tuple[int, ...],
    ):
        if name not in self.constraints:
            self.constraints.update({name: {}})

        self.constraints[name][idx] = constraint_type(name, idx, shape)


class Block(metaclass=ABCMeta):
    def __init__(
        self,
        name: str,
        idx: ViewDesc,
        shape: tuple[int, ...],
    ):
        self.name = name
        self.idx = idx
        self.shape = shape

        self.initialized = False

    @abstractmethod
    def update(self, ctx: Context):
        pass

    @abstractmethod
    def objective(self, ctx: Context) -> float:
        return 0.0


class Constraint(Block, metaclass=ABCMeta):
    """Base class for (multi-)affine constraints"""

    @abstractmethod
    def constraint(self, ctx: Context) -> NDArray[float64]:
        """Returns the lhs of a constraint f(x) = 0"""
        pass

    def update(self, ctx: Context):
        """Update the multipliers"""
        self.value += self.constraint(ctx)

    def objective(self, ctx: Context) -> float:
        return (
            0.5
            * ctx.params["rho"]
            * (
                ((self.constraint(ctx) + self.value) ** 2).sum()
                - (self.value**2).sum()
            )
        )
