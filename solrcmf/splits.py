from numpy.typing import NDArray
from numpy import (
    float64,
    int_,
    bool_,
    zeros,
    arange,
    flatnonzero,
    logical_not,
    logical_or,
    isnan,
    full,
)
from numpy.random import Generator, default_rng
from abc import ABCMeta, abstractmethod

from .base import ViewDesc


class BaseSplitter(metaclass=ABCMeta):
    @abstractmethod
    def _iter_test_indices(self, xs: dict[ViewDesc, NDArray[float64]]):
        yield

    def _iter_test_masks(self, xs: dict[ViewDesc, NDArray[float64]]):
        for test_indices in self._iter_test_indices(xs):
            test_mask = {k: zeros(x.size, dtype=bool_) for k, x in xs.items()}
            for k, m in test_mask.items():
                m[test_indices[k]] = True
            yield test_mask

    def split(self, xs: dict[ViewDesc, NDArray[float64]]):
        """Element-wise k-fold split across data matrices."""
        for test_mask in self._iter_test_masks(xs):
            train_indices = {
                k: flatnonzero(logical_not(logical_or(m, isnan(xs[k]).flat)))
                for k, m in test_mask.items()
            }
            test_indices = {k: flatnonzero(m) for k, m in test_mask.items()}
            yield train_indices, test_indices

    @abstractmethod
    def get_n_splits(self, xs: dict[ViewDesc, NDArray[float64]]):
        return 0


class ElementwiseFolds(BaseSplitter):
    def __init__(
        self,
        n_splits: int,
        *,
        shuffle: bool = True,
        rng: Generator | None = None,
    ):
        if n_splits <= 1:
            raise ValueError("n_splits needs to be an integer >= 2")
        self.n_splits = n_splits

        if shuffle is False and rng is not None:
            raise ValueError("rng should be None if shuffle is False")
        self.shuffle = shuffle

        if rng is None:
            rng = default_rng()

        self.rng = rng

    # Take some inspiration from sklearn
    def _iter_test_indices(self, xs: dict[ViewDesc, NDArray[float64]]):
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

    def get_n_splits(self, xs: dict[ViewDesc, NDArray[float64]]):
        return self.n_splits
