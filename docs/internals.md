# Internals

The package is built around a generic ADMM interface that should be capable of
serving as the basis for other ADMM-based algorithms as well.

The basic loop of any ADMM algorithm looks as follows

0. Initialize state
1. Repeat until convergence or maximum iterations
     1. Update state variables in specified order
     2. Update constraints
     3. Check for convergence