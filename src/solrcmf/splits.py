"""Data splitting for dictionary datasets.

Provides functionality for splitting datasets defined as dictionaries of
arrays into cross-validation folds.
"""

from abc import ABCMeta, abstractmethod
from typing import override

from numpy import (
    arange,
    bool_,
    flatnonzero,
    float64,
    full,
    int_,
    isnan,
    logical_not,
    logical_or,
    zeros,
)
from numpy.random import Generator, default_rng
from numpy.typing import NDArray

from .base import ViewDesc


class BaseSplitter(metaclass=ABCMeta):
    """Abstract base class for element-wise cross-validation splitters.

    Subclasses implement `_iter_test_indices` to define how flat element
    indices are assigned to folds, and `get_n_splits` to report the number
    of folds. The public interface is `split`, which converts index sets into
    train/test index dicts suitable for passing to SolrCMF.
    """

    @abstractmethod
    def _iter_test_indices(self, xs: dict[ViewDesc, NDArray[float64]]):
        """Yield a dict of flat test indices, one array per view pair."""
        yield

    def _iter_test_masks(self, xs: dict[ViewDesc, NDArray[float64]]):
        """Yield a dict of boolean test masks, one array per view pair."""
        for test_indices in self._iter_test_indices(xs):
            test_mask = {k: zeros(x.size, dtype=bool_) for k, x in xs.items()}
            for k, m in test_mask.items():
                m[test_indices[k]] = True
            yield test_mask

    def split(self, xs: dict[ViewDesc, NDArray[float64]]):
        """Yield (train_indices, test_indices) pairs for each fold.

        Missing entries (NaN) are excluded from both train and test sets.

        Args:
            xs: Data matrices to split, one per view pair.

        Yields:
            A tuple (train_indices, test_indices), each a dict mapping view
            pairs to flat index arrays into the corresponding data matrix.

        """
        for test_mask in self._iter_test_masks(xs):
            train_indices = {
                k: flatnonzero(logical_not(logical_or(m, isnan(xs[k]).flat)))
                for k, m in test_mask.items()
            }
            test_indices = {k: flatnonzero(m) for k, m in test_mask.items()}
            yield train_indices, test_indices

    @abstractmethod
    def get_n_splits(self, xs: dict[ViewDesc, NDArray[float64]]):
        """Return the number of splits."""
        return 0


class ElementwiseFolds(BaseSplitter):
    """Element-wise k-fold splitter for collections of data matrices.

    Observed entries (non-NaN) across all data matrices are independently
    partitioned into k roughly equal folds. Each call to `split` yields k
    (train_indices, test_indices) pairs, where indices are flat into the
    corresponding data matrix.
    """

    def __init__(
        self,
        n_splits: int,
        *,
        shuffle: bool = True,
        rng: Generator | None = None,
    ):
        """Initialize ElementwiseFolds.

        Args:
            n_splits: Number of folds. Must be at least 2.
            shuffle: Whether to shuffle observed entries before assigning
                them to folds.
            rng: Random number generator used for shuffling. Must be `None`
                when shuffle is False.

        """
        if n_splits <= 1:
            raise ValueError("n_splits needs to be an integer >= 2")
        self.n_splits = n_splits

        if shuffle is False and rng is not None:
            raise ValueError("rng should be None if shuffle is False")
        self.shuffle = shuffle

        if rng is None:
            rng = default_rng()

        self.rng = rng

    @override
    def _iter_test_indices(self, xs: dict[ViewDesc, NDArray[float64]]):
        """Yield a dict of flat test indices for each fold.

        Observed entries are split into n_splits roughly equal folds
        independently per data matrix.

        """
        # Exclude entries that are already nan
        indices = {
            k: arange(x.size)[flatnonzero(logical_not(isnan(x)))]
            for k, x in xs.items()
        }
        if self.shuffle:
            for idx in indices.values():
                self.rng.shuffle(idx)

        fold_sizes = {
            k: full(self.n_splits, idx.size // self.n_splits, dtype=int_)
            for k, idx in indices.items()
        }
        for k, s in fold_sizes.items():
            s[: indices[k].size % self.n_splits] += 1

        current = {k: 0 for k in fold_sizes.keys()}
        for i in range(self.n_splits):
            test_indices = {
                k: idx[current[k] : current[k] + fold_sizes[k][i]]
                for k, idx in indices.items()
            }
            yield test_indices
            current = {k: idx + fold_sizes[k][i] for k, idx in current.items()}

    @override
    def get_n_splits(self, xs: dict[ViewDesc, NDArray[float64]]):
        """Return the number of splits."""
        return self.n_splits
