from numpy.typing import NDArray
from numpy import float64, hstack, vstack, diag, inf
from numpy.linalg import svd
from numpy.random import Generator, default_rng
from joblib import Parallel, delayed

from .base import ViewDesc, Entity
from .solrcmf import SolrCMF


def multiview_init(
    xs: dict[ViewDesc, NDArray[float64]],
    max_rank: int,
) -> tuple[dict[Entity, NDArray[float64]], dict[ViewDesc, NDArray[float64]]]:
    layout = list(xs.keys())
    if len({k[0] for k in layout}) == 1:
        x_joint = hstack([x for x in xs.values()]).T
        jx = 0
        ix = 1
    elif len({k[1] for k in layout}) == 1:
        x_joint = vstack([x for x in xs.values()])
        jx = 1
        ix = 0
    else:
        raise ValueError("'xs' does not follow a multiview layout")

    u, _, vt = svd(x_joint)

    vs = {layout[0][jx]: vt.T[:, :max_rank]}

    current = 0
    for k, x in xs.items():
        vs.update({k[ix]: u[current : current + x.shape[ix], :max_rank]})
        current += x.shape[ix]

    ds = {k: diag(vs[k[0]].T @ x @ vs[k[1]]) for k, x in xs.items()}

    return vs, ds


def best_random_init(
    xs: dict[ViewDesc, NDArray[float64]],
    max_rank: int,
    *,
    n_inits: int = 1,
    n_jobs: int = -1,
    rng: Generator | int | None = None,
    **kwargs,
) -> SolrCMF:
    """Generate best unpenalized solution from random starting points

    Parameters
    ----------
    xs
        Input data
    max_rank
        Maximum rank
    n_inits
        Number of random starting points to test
    n_jobs
        Number of jobs to run concurrently, use as in [joblib.Parallel](https://joblib.readthedocs.io/en/stable/generated/joblib.Parallel.html#joblib.Parallel)
    rng
        Random number generator ([numpy.random.Generator](https://numpy.org/doc/stable/reference/random/generator.html)), random seed, or `None`
        to choose the default random number generator.

    Returns
    -------
    fit : SolrCMF
        The solution found with minimal objective value among all solutions
        obtained from the `n_inits` random starting points.
    """
    assert n_inits > 0, "'n_init' needs to be a positive integer"

    rng = default_rng(rng)

    def init_run(
        xs: dict[ViewDesc, NDArray[float64]], rng: Generator
    ) -> SolrCMF:
        return SolrCMF(
            structure_penalty=0.0,
            max_rank=max_rank,
            factor_pruning=False,
            init="random",
            init_kwargs={"rng": rng},
            **kwargs,
        ).fit(xs)

    rng_inits = rng.spawn(n_inits)

    ests_init: list[SolrCMF] = Parallel(n_jobs=n_jobs)(
        delayed(init_run)(xs, ri) for ri in rng_inits
    )

    best_obj = inf

    for i in range(n_inits):
        if ests_init[i].objective_value_ < best_obj:
            best_obj = ests_init[i].objective_value_
            best_est_init = ests_init[i]

    return best_est_init
