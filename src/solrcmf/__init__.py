"""Sparse orthogonal low-rank collective matrix factorization.

This package implements a method for sparse orthogonal low-rank
collective matrix factorization (solrCMF) implemented via a
multi-affine multi-block ADMM algorithm.
"""

from solrcmf.crossval import SolrCMFCV
from solrcmf.initstrategies import best_random_init, multiview_init
from solrcmf.lrimpute import LowRankImputation
from solrcmf.preprocess import bicenter, nanscale
from solrcmf.simulate import simulate
from solrcmf.solrcmf import SolrCMF
from solrcmf.splits import ElementwiseFolds

__all__ = [
    "SolrCMF",
    "SolrCMFCV",
    "ElementwiseFolds",
    "simulate",
    "multiview_init",
    "best_random_init",
    "LowRankImputation",
    "bicenter",
    "nanscale",
]
