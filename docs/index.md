# Overview

Matrix decomposition techniques are well-established and widely used for
decades and common examples for decomposing a single matrix are the
[Eigendecomposition](https://en.wikipedia.org/wiki/Eigendecomposition_of_a_matrix),
[LU decomposition](https://en.wikipedia.org/wiki/LU_decomposition),
[Cholesky decomposition](https://en.wikipedia.org/wiki/Cholesky_decomposition),
[Singular Value Decomposition (SVD)](https://en.wikipedia.org/wiki/Singular_value_decomposition),
or [Non-negative Matrix Factorization (NMF)](https://en.wikipedia.org/wiki/Non-negative_matrix_factorization).

## Integrated matrix decomposition

Decomposing a single matrix can be very valueable to learn 

## Layout description

### Views

_**Views**_ are abstractions for observational units or other views on the data,
such as data types, layer index, time steps, and so on.
Typically, they are represented by integers or strings, however, it is allowed
to use any hashable type.

Each input data matrix is associated with two primary entities, a *row view*
and a *column view*. It is possible for a data matrix to be associated with
additional entities, such as a *layer view* in a tensor-like layout.

/// note
Additional entities are used to organize the input data and allow, e.g.,
repeated observations of the same row/column view combination.
Data integration is however only performed for row and column entities.
///

solrCMF uses the type alias `ViewDesc`, short for *view description*, to
describe view relationships. A `ViewDesc` is simply a tuple of two or more
hashable types.

/// details | Examples of view relationships
    type: example
    open: False
The following examples are valid view relationship descriptions:

//// tab | Numeric
```python
(0, 1), (10, 2), (1, 2)
```
Integers can be used as convenient short-hands for views.
////

//// tab | Strings
```python
("A", "B"), ("genes", "samples")
```
Strings can provide additional descriptions to the views.
////

//// tab | Additional
```python
("x", "y", "channel"), (0, 1, "a", "01:12"), (0, 1, "a", "10:50")
```
More than two views can be specified, where additional views are used to provide additional context for a data source, e.g., to integrate repeated measurements
of a view relationship.
////
///

/// warning | Important
Any hashable data types can be used to represent views. The only importance is
that every appearance of view `0`, say, represents the same view, no matter at
which position in the `ViewDesc` tuple it appears. For example, in `(0, 1)` and
`(5, 0)` the `0` represents the same view within a data layout.
This allows, e.g. for a view to appear in the rows of one data source,
but in the columns of another.
///

### Layouts

A _**layout**_ is a collection of view descriptions and can be seen as a
Python list containing entries of type `ViewDesc`.

/// details | Example layout
    type: example
A simple multi-view layout can be described as
```python
layout = [
    ("user", "datatype1"),
    ("user", "datatype2"),
    ("user", "datatype3"),
    ("user", "datatype4"),
]
```
///

Defining a layout indirectly also defines the views present in a collection of
data sources and establishes relationships between them.