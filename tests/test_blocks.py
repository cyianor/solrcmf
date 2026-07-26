import numpy as np
import pytest
from numpy.testing import assert_allclose
from numpy.typing import NDArray

from solrcmf import SolrCMF
from solrcmf.base import Entity, ViewDesc, update


@pytest.mark.parametrize(
    ("weights", "expected"),
    [
        (1.0, np.array([[0.0], [-1.0], [0.0]])),
        (
            np.array([[0.6], [2.0], [0.8]]),
            np.array([[0.0], [0.0], [-1.0]]),
        ),
    ],
)
def test_u_zero_threshold_fallback_uses_original_objective(weights, expected):
    """All-zero sparse updates retain the best coordinate and its sign."""
    X: dict[ViewDesc, np.ndarray] = {
        (0, 1): np.zeros((3, 3), dtype=np.float64)
    }
    ctx = SolrCMF(
        structure_penalty=0.0,
        max_rank=1,
        factor_penalty=0.1,
        factor_pruning=False,
    )._setup(X)

    ctx.params.alpha = 0.0
    ctx.params.rho = 1.0
    ctx.params.factor_penalty = 1.0
    ctx.params.factor_weights[0] = weights
    ctx.blocks.v[0].value = np.array([[0.4], [-0.8], [-0.7]])
    ctx.blocks.vp[0].value.fill(0.0)
    ctx.constraints.factor[0].value.fill(0.0)
    ctx.blocks.u[0].value.fill(0.0)

    with pytest.warns(
        UserWarning, match="maximum pre-threshold magnitude is 0.8"
    ):
        update(ctx.blocks.u[0], ctx)

    assert_allclose(ctx.blocks.u[0].value, expected)


def test_fixed_factor_pattern_rejects_empty_column():
    """Every fixed sparse-factor column must allow a nonzero coordinate."""
    X: dict[ViewDesc, np.ndarray] = {
        (0, 1): np.zeros((3, 3), dtype=np.float64)
    }
    factor_pattern: dict[Entity, NDArray[np.bool_]] = {
        0: np.array(
            [[True, False], [True, False], [False, False]], dtype=np.bool_
        ),
        1: np.ones((3, 2), dtype=np.bool_),
    }

    est = SolrCMF(
        structure_penalty=0.0,
        max_rank=2,
        factor_pruning=False,
    )
    with pytest.raises(ValueError, match="at least one allowed entry"):
        est._setup(X, factor_pattern=factor_pattern)
