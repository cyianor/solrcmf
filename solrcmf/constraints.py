from numpy.typing import NDArray
from numpy import float64, diag

from .base import Constraint, Context


class MeanStructureConstraint(Constraint):
    def constraint(self, ctx: Context) -> NDArray[float64]:
        return (
            ctx.blocks["z"][self.idx].value
            - ctx.blocks["v"][(self.idx[0],)].value
            @ diag(ctx.blocks["d"][self.idx].value)
            @ ctx.blocks["v"][(self.idx[1],)].value.T
        )


class FactorConstraint(Constraint):
    def constraint(self, ctx: Context) -> NDArray[float64]:
        return (
            ctx.blocks["u"][self.idx].value
            - ctx.blocks["v"][self.idx].value
            - ctx.blocks["vp"][self.idx].value
        )
