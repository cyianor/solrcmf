# Sparse and orthogonal low-rank Collective Matrix Factorization (solrCMF)

This is a package describing the data integration methodology from ["Sparse and orthogonal low-rank Collective Matrix Factorization (solrCMF): Efficient data integration in flexible layouts" (Held et al., 2024, arXiv:2405.10067)](https://arxiv.org/abs/2405.10067).

To install the package run
```sh
pip install git+https://github.com/cyianor/solrcmf.git
```

A simple usage example is shown below:


```python
from sklearn import set_config

set_config(display="text")  # show text representation of sklearn estimators

import solrcmf
import numpy as np
from numpy.random import default_rng

# Control randomness
rng = default_rng(42)

# Simulate some data
# - `viewdims`: Dimensions of each view
# - `factor_scales`: The strength/singular value of each factor.
#                    The diagonal of the D matrices in the paper.
#                    Tuples are used to name data matrices. The first two
#                    entries describe the relationship between views observed
#                    in the data matrix. The third and following entries
#                    are used to make the index unique which is relevant
#                    in case of repeated layers of an observed relationship.
# - `snr`: Signal-to-noise ratio of the noise added to each true signal
#          (can be different for each data matrix)
# - `factor_sparsity`: Controls how sparse factors are generated in each
#                      view V_i.
#                      Example: 0.25 means 25% non-zero entries for each factor
#                      in a view.
#
# The function below generates sparse orthogonal matrices V_i and uses the
# supplied D_ij to form signal matrices V_i D_ij V_j^T. Noise with
# residual variance controlled by the signal-to-noise ratio is added.
xs_sim = solrcmf.simulate(
    viewdims={0: 100, 1: 50, 2: 50},
    factor_scales={
        (0, 1): [7.0, 4.5, 3.9, 0.0, 0.0],
        (0, 2): [8.3, 0.0, 0.0, 5.5, 0.0],
        (1, 2, 0): [6.3, 0.0, 4.7, 0.0, 5.1],
        (1, 2, 1): [0.0, 8.6, 4.9, 0.0, 0.0],
    },
    factor_sparsity={0: 0.25, 1: 0.25, 2: 0.25},
    snr=0.5,
    rng=rng,
)


# `xs_sim` is a dictionary containing
# - "xs_truth", the true signal matrices
# - "xs", the noisy data
# - "vs", the simulated orthogonal factors
```

Estimation via multi-block ADMM is encapsulated in the class `SolrCMF` which has a convenient scikit-learn interface.


```python
# It is recommended to center input matrices along rows and columns as
# well as to scale their Frobenius norm to 1.
xs_centered = {k: solrcmf.bicenter(x)[0] for k, x in xs_sim["xs"].items()}
xs_scaled = {k: x / np.sqrt((x**2).sum()) for k, x in xs_centered.items()}

# To determine good starting values, different strategies can be employed.
# In the paper, the algorithm was run repeatedly on random starting values
# without penalization and the best result is used as a starting point
# for hyperparameter selection.
# The data needs to be provided and `max_rank` sets a maximum rank for the
# low-rank matrix factorization.
# - `n_inits` controls how often new random starting values are selected.
# - `rng` controls the random number generation
# - `n_jobs` allows to parallize the search for initial values and is used
#   like in joblib https://joblib.readthedocs.io/en/stable/generated/joblib.Parallel.html
est_init = solrcmf.best_random_init(
    xs_scaled, max_rank=10, n_inits=50, n_jobs=4, rng=rng
)
```


```python
# Create a SolrCMF estimator
# - `max_rank` is the maximum rank of the low-rank matrix factorization
# - `structure_penalty` controls the integration penalty on the
#   diagonal entries of D_ij
# - `factor_penalty` controls the factor sparsity penalty on the
#   entries of V_i (indirectly through U_i in the ADMM algorithm)
# - `mu` controls how similar factors in U_i and V_i should be. A larger value
#   forces them together more closely
# - `init` can be set to "random" which constructs a random starting state
#   or "custom". In the latter case, a starting state for
#       * `vs`: the V_i matrices
#       * `ds`: the D_ij matrices
#       * `us`: the U_i matrices
#   needs to be supplied when calling the `fit` method. See the example below.
# - `factor_pruning` whether or not factors without any contribution should be
#   removed during estimation.
# - `max_iter`: Maximum number of iterations
est = solrcmf.SolrCMF(
    max_rank=10,
    structure_penalty=0.05,
    factor_penalty=0.08,
    mu=10,
    init="custom",
    factor_pruning=False,
    max_iter=100000,
)
```

The estimation is then performed by fitting the model to data. Use the
final values of the initial runs as starting values. Penalty parameters are not chosen optimally here.


```python
est.fit(xs_scaled, vs=est_init.vs_, ds=est_init.ds_, us=est_init.vs_)
```




    SolrCMF(factor_penalty=0.08, factor_pruning=False, init='custom',
            max_iter=100000, max_rank=10, mu=10, structure_penalty=0.05)



Estimates for $D_{ij}$ are then in `est.ds_` and estimates for $V_i$ in `est.vs_`.

Scale back to original scale.


```python
for k, d in est.ds_.items():
    rescaled_d = d * np.sqrt((xs_centered[k] ** 2).sum())
    print(
        f"{str(k):10s}: "
        f"{np.array2string(rescaled_d, precision=2, floatmode='fixed')}"
    )
```

    (0, 1)    : [ 0.00  6.55 -1.44  3.52  0.00 -0.00 -0.00 -0.00  0.00 -0.00]
    (0, 2)    : [ 0.00 -7.52 -0.00 -0.00 -2.88 -0.00  0.00 -0.00  0.00  0.00]
    (1, 2, 0) : [ 0.00 -5.67  2.92  0.00 -0.00 -0.00 -2.81 -0.00 -0.00  0.00]
    (1, 2, 1) : [-0.00  0.00  3.64 -8.23  0.00 -0.00  0.00  0.00  0.00  0.00]


Shrinkage can be clearly seen in the singular value estimates compared to the groundtruth.

Setting the right hyperparameters is non-trivial and
more rigorous method is necessary. The class `SolrCMFCV` is provided for this
purpose to perform cross-validation automatically.

Cross-validation performs a two-step procedure:

1. Possible model structures are determined by estimating the model for all supplied pairs of hyperparameters. Zero patterns in singular values and factors are recorded.
2. Cross-validation is then performed by fixing each zero pattern obtained in Step 1 and estimating model parameters on all $K$ combinations of training folds. Test errors are computed on the respective left-out test fold.

The final solution is found by determining the pair of hyperparameters that leads to the minimal CV error and to pick those parameters that are within one standard error of the minimal CV error with most sparsity in the singular values.


```python
# Lists of structure and factor penalties are supplied containing the
# parameter combinations to be tested. Lists need to be of the same length
# or one needs to be a scalar.
# - `cv` number of folds
est_cv = solrcmf.SolrCMFCV(
    max_rank=10,
    structure_penalty=np.exp(rng.uniform(np.log(5e-2), np.log(1.0), 100)),
    factor_penalty=np.exp(rng.uniform(np.log(5e-2), np.log(1.0), 100)),
    mu=10,
    cv=10,
    init="custom",
    max_iter=100000,
    n_jobs=4,
)
```

Perform hyperparameter selection. This step can be time-intensive.


```python
# Initial values are supplied as lists. If length 1 then they are reused.
# If same length as hyperparameters then different initial values can be used
# for each pair of hyperparameters.
est_cv.fit(xs_scaled, vs=[est_init.vs_], ds=[est_init.ds_], us=[est_init.vs_])
```




    SolrCMFCV(factor_penalty=array([0.36792423, 0.06941587, 0.10752516, 0.43376266, 0.53883355,
           0.95854522, 0.10676321, 0.62188201, 0.08373001, 0.07400256,
           0.44430964, 0.19013555, 0.53937187, 0.11657228, 0.09469818,
           0.25880558, 0.07063052, 0.11524739, 0.08227149, 0.13042509,
           0.49534386, 0.07178925, 0.3126294 , 0.28824926, 0.25066523,
           0.2132999 , 0.36531926, 0.64150673, 0.08124273, 0...
           0.71015324, 0.1375872 , 0.10718306, 0.73360059, 0.0774336 ,
           0.05972806, 0.12817684, 0.48768923, 0.40007808, 0.96196336,
           0.14680267, 0.11424985, 0.15524923, 0.52084544, 0.09501248,
           0.85510326, 0.23217319, 0.52223399, 0.59602222, 0.2098567 ,
           0.46080418, 0.14908991, 0.56755986, 0.59005505, 0.27265958,
           0.09611405, 0.91465952, 0.85313787, 0.32016594, 0.95285913,
           0.22548781, 0.15398784, 0.19865442, 0.05737153, 0.25905621]))



CV results can be found in the attribute `est_cv.cv_results_` and can be easily converted to a Pandas `DataFrame`. The best result corresponds to the row with index `est_cv.best_index_`.


```python
import pandas as pd

cv_res = pd.DataFrame(est_cv.cv_results_)
cv_res.loc[est_cv.best_index_, :]
```




    structure_penalty                         0.062040
    max_rank                                 10.000000
    factor_penalty                            0.069416
    objective_value_penalized                 1.934370
    mean_elapsed_process_time_penalized       6.977537
    std_elapsed_process_time_penalized        0.000000
    est_max_rank                              5.000000
    structural_zeros                         30.000000
    factor_zeros                           1746.000000
    neg_mean_squared_error_fold0             -0.000193
    neg_mean_squared_error_fold1             -0.000179
    neg_mean_squared_error_fold2             -0.000181
    neg_mean_squared_error_fold3             -0.000185
    neg_mean_squared_error_fold4             -0.000189
    neg_mean_squared_error_fold5             -0.000184
    neg_mean_squared_error_fold6             -0.000190
    neg_mean_squared_error_fold7             -0.000184
    neg_mean_squared_error_fold8             -0.000182
    neg_mean_squared_error_fold9             -0.000189
    mean_elapsed_process_time_fixed           1.265778
    std_elapsed_process_time_fixed            0.094385
    mean_neg_mean_squared_error              -0.000186
    std_neg_mean_squared_error                0.000004
    Name: 1, dtype: float64




```python
for k, d in est_cv.best_estimator_.ds_.items():
    rescaled_d = d * np.sqrt((xs_centered[k] ** 2).sum())
    print(
        f"{str(k):10s}: "
        f"{np.array2string(rescaled_d, precision=2, floatmode='fixed')}"
    )
```

    (0, 1)    : [ 7.40 -3.27  4.59  0.00 -0.00]
    (0, 2)    : [-8.43 -0.00 -0.00 -5.15  0.00]
    (1, 2, 0) : [-6.60  4.34  0.00 -0.00 -4.49]
    (1, 2, 1) : [ 0.00  4.82 -9.21  0.00  0.00]


Due to the small size of the data sources and signal-to-noise ratio of 0.5, it is not possible to recover singular values perfectly. However, thanks to unpenalized re-estimation, the strong shrinkage seen in the manual solution above is not present here.

The factor estimates are in `est_cv.best_estimator_.vs_`, however, sparse factors can be found in `est_cv.best_estimator_.us_`. In this particular run, factor 0 of view 0 in the groundtruth corresponds to factor 0 in view 0 of the estimate. Note that in general factor order is arbitrary.


```python
np.sum(xs_sim["vs"][0][:, 0] * est_cv.best_estimator_.us_[0][:, 0])
```




    0.9886157937798741



The correctness of the estimated sparsity pattern can be analysed by looking at true positive and false positive rate.


```python
def true_positive_rate(estimate, truth):
    return sum(np.logical_and(estimate != 0.0, truth != 0.0)) / sum(
        truth != 0.0
    )


def false_positive_rate(estimate, truth):
    return sum(np.logical_and(estimate != 0.0, truth == 0.0)) / sum(
        truth == 0.0
    )
```


```python
(
    true_positive_rate(
        xs_sim["vs"][0][:, 0], est_cv.best_estimator_.us_[0][:, 0]
    ),
    false_positive_rate(
        xs_sim["vs"][0][:, 0], est_cv.best_estimator_.us_[0][:, 0]
    ),
)
```




    (0.6578947368421053, 0.0)


