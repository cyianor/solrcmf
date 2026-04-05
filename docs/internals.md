# Internals

The package is built around a generic multi-block ADMM interface that can serve
as the basis for other ADMM-based algorithms.

The basic loop of any ADMM algorithm looks as follows:

0. Initialize state
1. Repeat until convergence or maximum iterations
     1. Update primal blocks in specified order
     2. Update dual variables (constraint multipliers)
     3. Evaluate augmented Lagrangian objective
     4. Check for convergence

## ADMM base class

::: solrcmf.admm.ADMM

## Building blocks

::: solrcmf.base.Block
::: solrcmf.base.Constraint
::: solrcmf.base.Context

### Single-dispatch functions

The following functions drive the main update logic in the ADMM algorithm.
They are defined with the [functools.singledispatch][] decorator and subclasses
of [solrcmf.base.Block][] need to register their polymorphic `update` and
`objective` functions. Subclasses of [solrcmf.base.Constraint][] need to
register their polymorphic `constraint` function.

::: solrcmf.base.update
::: solrcmf.base.objective
::: solrcmf.base.constraint
