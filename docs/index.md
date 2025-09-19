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

A single view is associated with type `Entity`, which is either a `str`
or `int`. solrCMF then uses the type alias `ViewDesc`, short for
*view description*, to describe view relationships. A `ViewDesc` is simply
a tuple of two or more entries of type `Entity`.

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

//// tab | Advanced
```python
("x", "y", "channel"), (0, 1, "a", "01:12"), (0, 1, "a", "10:50")
```
More than two views can be specified, where additional views are used to provide additional context for a data source, e.g., to integrate repeated measurements
of a view relationship.
////
///

/// warning | Important
Strings and integers can be used to represent views. Is is important
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
layout: list[ViewDesc] = [
    ("user", "datatype1"),
    ("user", "datatype2"),
    ("user", "datatype3", "layer1"),
    ("user", "datatype3", "layer2"),
]
```
///

Defining a layout establishes relationships between views and indirectly also
defines which views are present in a collection of
data sources.