# Sparse and orthogonal low-rank Collective Matrix Factorization (solrCMF)

This is a package describing the data integration methodology from ["Sparse and orthogonal low-rank Collective Matrix Factorization (solrCMF): Efficient data integration in flexible layouts" (Held et al., 2024, arXiv:2405.10067)](https://arxiv.org/abs/2405.10067).

To install the package run
```sh
pip install git+https://github.com/cyianor/solrcmf.git
```

A simple usage example is shown below:


```python
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

Estimation via multi-block ADMM is encapsulated in the class `SolrCMF` which has
a convenient scikit-learn interface.


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




<style>#sk-container-id-1 {
  /* Definition of color scheme common for light and dark mode */
  --sklearn-color-text: black;
  --sklearn-color-line: gray;
  /* Definition of color scheme for unfitted estimators */
  --sklearn-color-unfitted-level-0: #fff5e6;
  --sklearn-color-unfitted-level-1: #f6e4d2;
  --sklearn-color-unfitted-level-2: #ffe0b3;
  --sklearn-color-unfitted-level-3: chocolate;
  /* Definition of color scheme for fitted estimators */
  --sklearn-color-fitted-level-0: #f0f8ff;
  --sklearn-color-fitted-level-1: #d4ebff;
  --sklearn-color-fitted-level-2: #b3dbfd;
  --sklearn-color-fitted-level-3: cornflowerblue;

  /* Specific color for light theme */
  --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));
  --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, white)));
  --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));
  --sklearn-color-icon: #696969;

  @media (prefers-color-scheme: dark) {
    /* Redefinition of color scheme for dark theme */
    --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));
    --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, #111)));
    --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));
    --sklearn-color-icon: #878787;
  }
}

#sk-container-id-1 {
  color: var(--sklearn-color-text);
}

#sk-container-id-1 pre {
  padding: 0;
}

#sk-container-id-1 input.sk-hidden--visually {
  border: 0;
  clip: rect(1px 1px 1px 1px);
  clip: rect(1px, 1px, 1px, 1px);
  height: 1px;
  margin: -1px;
  overflow: hidden;
  padding: 0;
  position: absolute;
  width: 1px;
}

#sk-container-id-1 div.sk-dashed-wrapped {
  border: 1px dashed var(--sklearn-color-line);
  margin: 0 0.4em 0.5em 0.4em;
  box-sizing: border-box;
  padding-bottom: 0.4em;
  background-color: var(--sklearn-color-background);
}

#sk-container-id-1 div.sk-container {
  /* jupyter's `normalize.less` sets `[hidden] { display: none; }`
     but bootstrap.min.css set `[hidden] { display: none !important; }`
     so we also need the `!important` here to be able to override the
     default hidden behavior on the sphinx rendered scikit-learn.org.
     See: https://github.com/scikit-learn/scikit-learn/issues/21755 */
  display: inline-block !important;
  position: relative;
}

#sk-container-id-1 div.sk-text-repr-fallback {
  display: none;
}

div.sk-parallel-item,
div.sk-serial,
div.sk-item {
  /* draw centered vertical line to link estimators */
  background-image: linear-gradient(var(--sklearn-color-text-on-default-background), var(--sklearn-color-text-on-default-background));
  background-size: 2px 100%;
  background-repeat: no-repeat;
  background-position: center center;
}

/* Parallel-specific style estimator block */

#sk-container-id-1 div.sk-parallel-item::after {
  content: "";
  width: 100%;
  border-bottom: 2px solid var(--sklearn-color-text-on-default-background);
  flex-grow: 1;
}

#sk-container-id-1 div.sk-parallel {
  display: flex;
  align-items: stretch;
  justify-content: center;
  background-color: var(--sklearn-color-background);
  position: relative;
}

#sk-container-id-1 div.sk-parallel-item {
  display: flex;
  flex-direction: column;
}

#sk-container-id-1 div.sk-parallel-item:first-child::after {
  align-self: flex-end;
  width: 50%;
}

#sk-container-id-1 div.sk-parallel-item:last-child::after {
  align-self: flex-start;
  width: 50%;
}

#sk-container-id-1 div.sk-parallel-item:only-child::after {
  width: 0;
}

/* Serial-specific style estimator block */

#sk-container-id-1 div.sk-serial {
  display: flex;
  flex-direction: column;
  align-items: center;
  background-color: var(--sklearn-color-background);
  padding-right: 1em;
  padding-left: 1em;
}


/* Toggleable style: style used for estimator/Pipeline/ColumnTransformer box that is
clickable and can be expanded/collapsed.
- Pipeline and ColumnTransformer use this feature and define the default style
- Estimators will overwrite some part of the style using the `sk-estimator` class
*/

/* Pipeline and ColumnTransformer style (default) */

#sk-container-id-1 div.sk-toggleable {
  /* Default theme specific background. It is overwritten whether we have a
  specific estimator or a Pipeline/ColumnTransformer */
  background-color: var(--sklearn-color-background);
}

/* Toggleable label */
#sk-container-id-1 label.sk-toggleable__label {
  cursor: pointer;
  display: block;
  width: 100%;
  margin-bottom: 0;
  padding: 0.5em;
  box-sizing: border-box;
  text-align: center;
}

#sk-container-id-1 label.sk-toggleable__label-arrow:before {
  /* Arrow on the left of the label */
  content: "▸";
  float: left;
  margin-right: 0.25em;
  color: var(--sklearn-color-icon);
}

#sk-container-id-1 label.sk-toggleable__label-arrow:hover:before {
  color: var(--sklearn-color-text);
}

/* Toggleable content - dropdown */

#sk-container-id-1 div.sk-toggleable__content {
  max-height: 0;
  max-width: 0;
  overflow: hidden;
  text-align: left;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-1 div.sk-toggleable__content.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-1 div.sk-toggleable__content pre {
  margin: 0.2em;
  border-radius: 0.25em;
  color: var(--sklearn-color-text);
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-1 div.sk-toggleable__content.fitted pre {
  /* unfitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-1 input.sk-toggleable__control:checked~div.sk-toggleable__content {
  /* Expand drop-down */
  max-height: 200px;
  max-width: 100%;
  overflow: auto;
}

#sk-container-id-1 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {
  content: "▾";
}

/* Pipeline/ColumnTransformer-specific style */

#sk-container-id-1 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-1 div.sk-label.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator-specific style */

/* Colorize estimator box */
#sk-container-id-1 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-1 div.sk-estimator.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

#sk-container-id-1 div.sk-label label.sk-toggleable__label,
#sk-container-id-1 div.sk-label label {
  /* The background is the default theme color */
  color: var(--sklearn-color-text-on-default-background);
}

/* On hover, darken the color of the background */
#sk-container-id-1 div.sk-label:hover label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

/* Label box, darken color on hover, fitted */
#sk-container-id-1 div.sk-label.fitted:hover label.sk-toggleable__label.fitted {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator label */

#sk-container-id-1 div.sk-label label {
  font-family: monospace;
  font-weight: bold;
  display: inline-block;
  line-height: 1.2em;
}

#sk-container-id-1 div.sk-label-container {
  text-align: center;
}

/* Estimator-specific */
#sk-container-id-1 div.sk-estimator {
  font-family: monospace;
  border: 1px dotted var(--sklearn-color-border-box);
  border-radius: 0.25em;
  box-sizing: border-box;
  margin-bottom: 0.5em;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-1 div.sk-estimator.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

/* on hover */
#sk-container-id-1 div.sk-estimator:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-1 div.sk-estimator.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Specification for estimator info (e.g. "i" and "?") */

/* Common style for "i" and "?" */

.sk-estimator-doc-link,
a:link.sk-estimator-doc-link,
a:visited.sk-estimator-doc-link {
  float: right;
  font-size: smaller;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-background);
  border-radius: 1em;
  height: 1em;
  width: 1em;
  text-decoration: none !important;
  margin-left: 1ex;
  /* unfitted */
  border: var(--sklearn-color-unfitted-level-1) 1pt solid;
  color: var(--sklearn-color-unfitted-level-1);
}

.sk-estimator-doc-link.fitted,
a:link.sk-estimator-doc-link.fitted,
a:visited.sk-estimator-doc-link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
div.sk-estimator:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover,
div.sk-label-container:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

div.sk-estimator.fitted:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover,
div.sk-label-container:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

/* Span, style for the box shown on hovering the info icon */
.sk-estimator-doc-link span {
  display: none;
  z-index: 9999;
  position: relative;
  font-weight: normal;
  right: .2ex;
  padding: .5ex;
  margin: .5ex;
  width: min-content;
  min-width: 20ex;
  max-width: 50ex;
  color: var(--sklearn-color-text);
  box-shadow: 2pt 2pt 4pt #999;
  /* unfitted */
  background: var(--sklearn-color-unfitted-level-0);
  border: .5pt solid var(--sklearn-color-unfitted-level-3);
}

.sk-estimator-doc-link.fitted span {
  /* fitted */
  background: var(--sklearn-color-fitted-level-0);
  border: var(--sklearn-color-fitted-level-3);
}

.sk-estimator-doc-link:hover span {
  display: block;
}

/* "?"-specific style due to the `<a>` HTML tag */

#sk-container-id-1 a.estimator_doc_link {
  float: right;
  font-size: 1rem;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-background);
  border-radius: 1rem;
  height: 1rem;
  width: 1rem;
  text-decoration: none;
  /* unfitted */
  color: var(--sklearn-color-unfitted-level-1);
  border: var(--sklearn-color-unfitted-level-1) 1pt solid;
}

#sk-container-id-1 a.estimator_doc_link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
#sk-container-id-1 a.estimator_doc_link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

#sk-container-id-1 a.estimator_doc_link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
}
</style><div id="sk-container-id-1" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>SolrCMF(factor_penalty=0.08, factor_pruning=False, init=&#x27;custom&#x27;,
        max_iter=100000, max_rank=10, mu=10, structure_penalty=0.05)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-1" type="checkbox" checked><label for="sk-estimator-id-1" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">&nbsp;SolrCMF<span class="sk-estimator-doc-link fitted">i<span>Fitted</span></span></label><div class="sk-toggleable__content fitted"><pre>SolrCMF(factor_penalty=0.08, factor_pruning=False, init=&#x27;custom&#x27;,
        max_iter=100000, max_rank=10, mu=10, structure_penalty=0.05)</pre></div> </div></div></div></div>



Estimates for $D_{ij}$ are then in `est.ds_` and estimates for $V_i$ in `est.vs_`.

Scale back to original scale


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




<style>#sk-container-id-2 {
  /* Definition of color scheme common for light and dark mode */
  --sklearn-color-text: black;
  --sklearn-color-line: gray;
  /* Definition of color scheme for unfitted estimators */
  --sklearn-color-unfitted-level-0: #fff5e6;
  --sklearn-color-unfitted-level-1: #f6e4d2;
  --sklearn-color-unfitted-level-2: #ffe0b3;
  --sklearn-color-unfitted-level-3: chocolate;
  /* Definition of color scheme for fitted estimators */
  --sklearn-color-fitted-level-0: #f0f8ff;
  --sklearn-color-fitted-level-1: #d4ebff;
  --sklearn-color-fitted-level-2: #b3dbfd;
  --sklearn-color-fitted-level-3: cornflowerblue;

  /* Specific color for light theme */
  --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));
  --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, white)));
  --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));
  --sklearn-color-icon: #696969;

  @media (prefers-color-scheme: dark) {
    /* Redefinition of color scheme for dark theme */
    --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));
    --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, #111)));
    --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));
    --sklearn-color-icon: #878787;
  }
}

#sk-container-id-2 {
  color: var(--sklearn-color-text);
}

#sk-container-id-2 pre {
  padding: 0;
}

#sk-container-id-2 input.sk-hidden--visually {
  border: 0;
  clip: rect(1px 1px 1px 1px);
  clip: rect(1px, 1px, 1px, 1px);
  height: 1px;
  margin: -1px;
  overflow: hidden;
  padding: 0;
  position: absolute;
  width: 1px;
}

#sk-container-id-2 div.sk-dashed-wrapped {
  border: 1px dashed var(--sklearn-color-line);
  margin: 0 0.4em 0.5em 0.4em;
  box-sizing: border-box;
  padding-bottom: 0.4em;
  background-color: var(--sklearn-color-background);
}

#sk-container-id-2 div.sk-container {
  /* jupyter's `normalize.less` sets `[hidden] { display: none; }`
     but bootstrap.min.css set `[hidden] { display: none !important; }`
     so we also need the `!important` here to be able to override the
     default hidden behavior on the sphinx rendered scikit-learn.org.
     See: https://github.com/scikit-learn/scikit-learn/issues/21755 */
  display: inline-block !important;
  position: relative;
}

#sk-container-id-2 div.sk-text-repr-fallback {
  display: none;
}

div.sk-parallel-item,
div.sk-serial,
div.sk-item {
  /* draw centered vertical line to link estimators */
  background-image: linear-gradient(var(--sklearn-color-text-on-default-background), var(--sklearn-color-text-on-default-background));
  background-size: 2px 100%;
  background-repeat: no-repeat;
  background-position: center center;
}

/* Parallel-specific style estimator block */

#sk-container-id-2 div.sk-parallel-item::after {
  content: "";
  width: 100%;
  border-bottom: 2px solid var(--sklearn-color-text-on-default-background);
  flex-grow: 1;
}

#sk-container-id-2 div.sk-parallel {
  display: flex;
  align-items: stretch;
  justify-content: center;
  background-color: var(--sklearn-color-background);
  position: relative;
}

#sk-container-id-2 div.sk-parallel-item {
  display: flex;
  flex-direction: column;
}

#sk-container-id-2 div.sk-parallel-item:first-child::after {
  align-self: flex-end;
  width: 50%;
}

#sk-container-id-2 div.sk-parallel-item:last-child::after {
  align-self: flex-start;
  width: 50%;
}

#sk-container-id-2 div.sk-parallel-item:only-child::after {
  width: 0;
}

/* Serial-specific style estimator block */

#sk-container-id-2 div.sk-serial {
  display: flex;
  flex-direction: column;
  align-items: center;
  background-color: var(--sklearn-color-background);
  padding-right: 1em;
  padding-left: 1em;
}


/* Toggleable style: style used for estimator/Pipeline/ColumnTransformer box that is
clickable and can be expanded/collapsed.
- Pipeline and ColumnTransformer use this feature and define the default style
- Estimators will overwrite some part of the style using the `sk-estimator` class
*/

/* Pipeline and ColumnTransformer style (default) */

#sk-container-id-2 div.sk-toggleable {
  /* Default theme specific background. It is overwritten whether we have a
  specific estimator or a Pipeline/ColumnTransformer */
  background-color: var(--sklearn-color-background);
}

/* Toggleable label */
#sk-container-id-2 label.sk-toggleable__label {
  cursor: pointer;
  display: block;
  width: 100%;
  margin-bottom: 0;
  padding: 0.5em;
  box-sizing: border-box;
  text-align: center;
}

#sk-container-id-2 label.sk-toggleable__label-arrow:before {
  /* Arrow on the left of the label */
  content: "▸";
  float: left;
  margin-right: 0.25em;
  color: var(--sklearn-color-icon);
}

#sk-container-id-2 label.sk-toggleable__label-arrow:hover:before {
  color: var(--sklearn-color-text);
}

/* Toggleable content - dropdown */

#sk-container-id-2 div.sk-toggleable__content {
  max-height: 0;
  max-width: 0;
  overflow: hidden;
  text-align: left;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-2 div.sk-toggleable__content.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-2 div.sk-toggleable__content pre {
  margin: 0.2em;
  border-radius: 0.25em;
  color: var(--sklearn-color-text);
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-2 div.sk-toggleable__content.fitted pre {
  /* unfitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-2 input.sk-toggleable__control:checked~div.sk-toggleable__content {
  /* Expand drop-down */
  max-height: 200px;
  max-width: 100%;
  overflow: auto;
}

#sk-container-id-2 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {
  content: "▾";
}

/* Pipeline/ColumnTransformer-specific style */

#sk-container-id-2 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-2 div.sk-label.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator-specific style */

/* Colorize estimator box */
#sk-container-id-2 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-2 div.sk-estimator.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

#sk-container-id-2 div.sk-label label.sk-toggleable__label,
#sk-container-id-2 div.sk-label label {
  /* The background is the default theme color */
  color: var(--sklearn-color-text-on-default-background);
}

/* On hover, darken the color of the background */
#sk-container-id-2 div.sk-label:hover label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

/* Label box, darken color on hover, fitted */
#sk-container-id-2 div.sk-label.fitted:hover label.sk-toggleable__label.fitted {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator label */

#sk-container-id-2 div.sk-label label {
  font-family: monospace;
  font-weight: bold;
  display: inline-block;
  line-height: 1.2em;
}

#sk-container-id-2 div.sk-label-container {
  text-align: center;
}

/* Estimator-specific */
#sk-container-id-2 div.sk-estimator {
  font-family: monospace;
  border: 1px dotted var(--sklearn-color-border-box);
  border-radius: 0.25em;
  box-sizing: border-box;
  margin-bottom: 0.5em;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-2 div.sk-estimator.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

/* on hover */
#sk-container-id-2 div.sk-estimator:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-2 div.sk-estimator.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Specification for estimator info (e.g. "i" and "?") */

/* Common style for "i" and "?" */

.sk-estimator-doc-link,
a:link.sk-estimator-doc-link,
a:visited.sk-estimator-doc-link {
  float: right;
  font-size: smaller;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-background);
  border-radius: 1em;
  height: 1em;
  width: 1em;
  text-decoration: none !important;
  margin-left: 1ex;
  /* unfitted */
  border: var(--sklearn-color-unfitted-level-1) 1pt solid;
  color: var(--sklearn-color-unfitted-level-1);
}

.sk-estimator-doc-link.fitted,
a:link.sk-estimator-doc-link.fitted,
a:visited.sk-estimator-doc-link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
div.sk-estimator:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover,
div.sk-label-container:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

div.sk-estimator.fitted:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover,
div.sk-label-container:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

/* Span, style for the box shown on hovering the info icon */
.sk-estimator-doc-link span {
  display: none;
  z-index: 9999;
  position: relative;
  font-weight: normal;
  right: .2ex;
  padding: .5ex;
  margin: .5ex;
  width: min-content;
  min-width: 20ex;
  max-width: 50ex;
  color: var(--sklearn-color-text);
  box-shadow: 2pt 2pt 4pt #999;
  /* unfitted */
  background: var(--sklearn-color-unfitted-level-0);
  border: .5pt solid var(--sklearn-color-unfitted-level-3);
}

.sk-estimator-doc-link.fitted span {
  /* fitted */
  background: var(--sklearn-color-fitted-level-0);
  border: var(--sklearn-color-fitted-level-3);
}

.sk-estimator-doc-link:hover span {
  display: block;
}

/* "?"-specific style due to the `<a>` HTML tag */

#sk-container-id-2 a.estimator_doc_link {
  float: right;
  font-size: 1rem;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-background);
  border-radius: 1rem;
  height: 1rem;
  width: 1rem;
  text-decoration: none;
  /* unfitted */
  color: var(--sklearn-color-unfitted-level-1);
  border: var(--sklearn-color-unfitted-level-1) 1pt solid;
}

#sk-container-id-2 a.estimator_doc_link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
#sk-container-id-2 a.estimator_doc_link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

#sk-container-id-2 a.estimator_doc_link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
}
</style><div id="sk-container-id-2" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>SolrCMFCV(factor_penalty=array([0.36792423, 0.06941587, 0.10752516, 0.43376266, 0.53883355,
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
       0.22548781, 0.15398784, 0.19865442, 0.05737153, 0.25905621]))</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-2" type="checkbox" checked><label for="sk-estimator-id-2" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">&nbsp;SolrCMFCV<span class="sk-estimator-doc-link fitted">i<span>Fitted</span></span></label><div class="sk-toggleable__content fitted"><pre>SolrCMFCV(factor_penalty=array([0.36792423, 0.06941587, 0.10752516, 0.43376266, 0.53883355,
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
       0.22548781, 0.15398784, 0.19865442, 0.05737153, 0.25905621]))</pre></div> </div></div></div></div>



CV results can be found in the attribute `est_cv.cv_results_` and can be easily converted to a Pandas `DataFrame`. The best result corresponds to the row with index `est_cv.best_index_`. 


```python
import pandas as pd

cv_res = pd.DataFrame(est_cv.cv_results_)
cv_res.loc[est_cv.best_index_, :]
```




    structure_penalty                         0.114250
    max_rank                                 10.000000
    factor_penalty                            0.058822
    objective_value_penalized                 1.999925
    mean_elapsed_process_time_penalized       9.922670
    std_elapsed_process_time_penalized        0.000000
    est_max_rank                              5.000000
    structural_zeros                         31.000000
    factor_zeros                           1764.000000
    neg_mean_squared_error_fold0             -0.000172
    neg_mean_squared_error_fold1             -0.000192
    neg_mean_squared_error_fold2             -0.000196
    neg_mean_squared_error_fold3             -0.000185
    neg_mean_squared_error_fold4             -0.000182
    neg_mean_squared_error_fold5             -0.000190
    neg_mean_squared_error_fold6             -0.000201
    neg_mean_squared_error_fold7             -0.000202
    neg_mean_squared_error_fold8             -0.000200
    neg_mean_squared_error_fold9             -0.000181
    mean_elapsed_process_time_fixed           1.284826
    std_elapsed_process_time_fixed            0.132348
    mean_neg_mean_squared_error              -0.000190
    std_neg_mean_squared_error                0.000009
    Name: 76, dtype: float64




```python
for k, d in est_cv.best_estimator_.ds_.items():
    rescaled_d = d * np.sqrt((xs_centered[k] ** 2).sum())
    print(
        f"{str(k):10s}: "
        f"{np.array2string(rescaled_d, precision=2, floatmode='fixed')}"
    )
```

    (0, 1)    : [ 7.40 -0.00  4.59  0.00 -0.00]
    (0, 2)    : [-8.43 -0.00 -0.00 -5.00  0.00]
    (1, 2, 0) : [-6.59  4.25  0.00 -0.00 -3.85]
    (1, 2, 1) : [ 0.00  4.86 -9.22  0.00  0.00]


Due to the small size of the data sources and signal-to-noise ratio of 0.5, it is not possible to recover singular values perfectly. However, thanks to unpenalized re-estimation, the strong shrinkage seen in the manual solution above is not present here.

The factor estimates are in `est_cv.best_estimator_.vs_`, however, sparse factors can be found in `est_cv.best_estimator_.us_`. In this particular run, factor 0 of view 0 in the groundtruth corresponds to factor 0 in view 0 of the estimate. Note that in general factor order is arbitrary.


```python
np.sum(xs_sim["vs"][0][:, 0] * est_cv.best_estimator_.us_[0][:, 0])
```




    0.9896296784962579



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
    true_positive_rate(xs_sim["vs"][0][:, 0], est_cv.best_estimator_.us_[0][:, 0]),
    false_positive_rate(xs_sim["vs"][0][:, 0], est_cv.best_estimator_.us_[0][:, 0]),
)
```




    (0.6578947368421053, 0.0)


