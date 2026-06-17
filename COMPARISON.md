# Comparative Analysis: `solrcmf` (Python) vs `solrcmf-r` (R/C++)

*Two implementations of Sparse Orthogonal Low-Rank Collective Matrix Factorization*

---

Both repositories implement variants of the same core algorithm: sparse and orthogonal
low-rank Collective Matrix Factorization (solrCMF), described in Held et al. (2024,
arXiv:2405.10067). They are sibling implementations, not competing alternatives.
`cyianor/solrcmf-r` is the earlier research prototype (R + C++, 2020); `cyianor/solrcmf`
is the production-quality Python package released alongside the 2024 paper. Understanding
their relationship is the main goal of this document.

---

## Table of Contents

1. [Quick-Reference Summary](#1-quick-reference-summary)
2. [Purpose and Problem Statement](#2-purpose-and-problem-statement)
3. [Algorithmic Design](#3-algorithmic-design)
4. [API and Interface Design](#4-api-and-interface-design)
5. [Feature Comparison Matrix](#5-feature-comparison-matrix)
6. [Packaging and Dependency Management](#6-packaging-and-dependency-management)
7. [Testing Philosophy](#7-testing-philosophy)
8. [Documentation](#8-documentation)
9. [Maturity and Development Trajectory](#9-maturity-and-development-trajectory)
10. [When to Use Each](#10-when-to-use-each)
11. [Shared Gaps](#11-shared-gaps)
12. [Appendix: Repository Metadata](#12-appendix-repository-metadata)

---

## 1. Quick-Reference Summary

| Attribute | `solrcmf` (Python) | `solrcmf-r` (R/C++) |
|---|---|---|
| Primary language | Python 3.12 | R + C++ (Rcpp / RcppArmadillo) |
| Version / activity | Active (Apr 2024 – Apr 2026) | v0.2.2 (Jan 2020, research prototype) |
| Optimization algorithm | Multi-block ADMM | Two-block ADMM + proximal gradient |
| Distribution support | Gaussian only | Gaussian, Bernoulli, Poisson |
| Sparsity model | Two-level (structure L1 + factor L1) | Three-level hierarchical λ triplet |
| API style | Class-based (sklearn-compatible) | Function-based |
| Cross-validation | `SolrCMFCV` class | `run_solrcmf_cv()` function |
| Imputation | `LowRankImputation` class | Partial (missing-at-random handling) |
| Testing | Formal pytest suite (3 files) | Example / script-style R files |
| Documentation | MkDocs + GitHub Pages | Roxygen2 man pages + README |
| Parallel execution | Yes (joblib) | Not explicit |
| Static types | Full (PEP 561 / `py.typed`) | N/A |
| License | MIT | MIT |

---

## 2. Purpose and Problem Statement

Both packages address **Collective Matrix Factorization (CMF)**: given a collection of
matrices `{X[k]}`, find shared latent structure by jointly decomposing them. The solrCMF
model writes each matrix as

```
X[k]  ≈  V[i]  diag(d[k])  V[j]^T
```

where:
- `V[i]`, `V[j]` are **column-orthonormal** factor matrices shared across views (data
  types, time points, omics layers, etc.),
- `d[k]` is a vector of per-factor **signal strengths** for matrix `k`.

"Collective" means multiple matrices can share the same factor matrices, enabling
information transfer across views. Orthonormality of `V` is enforced as a hard constraint
(via a Procrustes step), making the factorization identifiable. Sparsity is imposed on
`d[k]` (and optionally on `V`) to produce interpretable, compact representations.

The method is particularly suited to multi-omics integration, multi-modal data fusion, and
any setting where different observed matrices are believed to share an underlying latent
space.

---

## 3. Algorithmic Design

### 3.1 Shared Foundation

Both implementations rest on the same conceptual backbone:

- An **ADMM (Alternating Direction Method of Multipliers)** outer loop that separates
  the data-fidelity objective from the orthonormality constraint.
- A **Procrustes / SVD step** that projects factor matrices onto the Stiefel manifold
  (column-orthonormal matrices) after each update.
- A **soft-thresholding step** that enforces sparsity on the scaling vectors `d[k]`.
- An **initialization phase** that runs without the orthogonality constraint before
  switching to the constrained updates.

### 3.2 ADMM Block Decomposition

This is the most significant algorithmic difference.

**Python (`solrcmf`)** uses a **multi-block ADMM** decomposition:
- `Z` block (data fidelity): closed-form least-squares update.
- `D` block (structure sparsity): soft-thresholding on `d[k]`.
- `V` block (orthonormality): SVD-based Procrustes projection.
- Optional `U` / `V'` blocks for factor-level sparsity.

The multi-block design works cleanly because the Gaussian likelihood admits a closed-form
data-fidelity update. All blocks operate in closed form, making each iteration cheap.

**R (`solrcmf-r`)** uses a **two-block ADMM** with a **proximal gradient step** embedded
inside the data-fidelity block:
- Block 1 (data fidelity + sparsity): proximal gradient descent — the gradient of the
  negative log-likelihood is computed, then a proximal operator is applied.
- Block 2 (orthonormality): Procrustes projection.

The two-block design is more general: swapping the proximal operator for a different
likelihood (Bernoulli, Poisson) requires only changing the gradient and proximal
computation, not restructuring the algorithm. This is why the R package supports
non-Gaussian distributions while the Python package does not yet.

### 3.3 Distribution Support

| Distribution | Python | R/C++ | Notes |
|---|---|---|---|
| Gaussian / Normal | Yes | Yes | Least-squares objective; closed-form updates |
| Bernoulli (logistic) | No | Yes | Requires proximal gradient approach |
| Poisson (log-link) | No | Yes | Requires proximal gradient approach |

The restriction of the Python package to Gaussian is an architectural consequence of the
multi-block ADMM design, not an oversight. Extending it to non-Gaussian likelihoods would
require either embedding a proximal gradient sub-loop (converging toward the two-block R
design) or adopting a linearization approach.

### 3.4 Sparsity Structure

**Python** enforces two levels of sparsity:
1. **Structure sparsity** — L1 penalty on the scaling vectors `d[k]`, controlled by
   `structure_penalty`. Drives entire factors to zero globally.
2. **Factor sparsity** — Optional L1 penalty on factor loadings `V` (or `U`), controlled
   by `factor_penalty`. Makes individual loadings sparse.

**R** uses a **three-level hierarchical λ triplet** `(λ₁, λ₂, λ₃)`:
1. `λ₁` — global factor activity (analogous to structure penalty).
2. `λ₂` — view-level factor activity.
3. `λ₃` — element-level factor sparsity.

The Python two-level model can be viewed as a simplified version of the R triplet where
`λ₂` is not explicitly separated from `λ₁`.

### 3.5 Computational Implementation

**Python** relies on **NumPy** (backed by BLAS/LAPACK) for all numerical operations.
Installation is pure-Python — no C++ compiler required.

**R** offloads computationally intensive loops to **C++ via Rcpp and RcppArmadillo**.
The `src/` directory contains the hot-path implementations (`models.cpp`,
`distributions.cpp`, `data_structures.cpp`). Installing from source requires a C++11
compiler (Rtools on Windows, Xcode CLT or GCC on macOS/Linux).

The performance implication is nuanced: NumPy/BLAS is highly optimised for dense
matrix operations and can match or exceed hand-written C++ for the matrix algebra
in question. The C++ advantage in the R package is primarily in the loop overhead
around those operations.

---

## 4. API and Interface Design

### 4.1 Entry Points

**Python** follows the **scikit-learn estimator pattern**:

```python
from solrcmf import SolrCMF, SolrCMFCV

# Fit
model = SolrCMF(rank=5, structure_penalty=0.1)
model.fit(matrices, layout)

# Cross-validate
cv = SolrCMFCV(rank=5, structure_penalties=[0.01, 0.1, 1.0])
cv.fit(matrices, layout)
```

**R** uses a **functional style**:

```r
library(solrcmf)

# Fit
result <- run_solrcmf(X = matrices, S = layout, K = 5,
                      lambda = c(0.1, 0.1, 0.1))

# Cross-validate
cv_result <- run_solrcmf_cv(X = matrices, S = layout, K = 5,
                             lambdas = lambda_grid)
```

Neither style is objectively superior; each reflects the idiomatic convention of its
ecosystem (sklearn-compatible objects in Python, named-list returns in R).

### 4.2 Data Structure Conventions

**Python** accepts a Python list (or dict) of NumPy arrays as input matrices, with a
separate layout specification. Results are stored as attributes on the fitted object
(`model.Vs_`, `model.ds_`, `model.score_`).

**R** uses `create_data_solrcmf()` to assemble a `solrcmf_data` object from a named list
of matrices, weight matrices, and distribution specifications. Results are returned as
named R lists with S3 print methods.

The R API is more explicit about data assembly (requiring a dedicated constructor)
but provides finer control, including per-matrix weight matrices and
per-matrix distribution families.

### 4.3 Initialization

| Strategy | Python | R/C++ |
|---|---|---|
| Random initialization | `best_random_init()` | `initialise_randomly_solrcmf()` |
| Multi-view warm start | `multiview_init()` | — |
| Non-orthogonal pre-phase | Implicit (built into fit) | `settings$non_orth_phase` |

Both packages run a non-orthogonal initialization phase before applying the orthogonality
constraint. Python exposes `best_random_init()` as a public function that runs multiple
random starts and returns the best, directly usable as a warm start for `SolrCMF`. The
equivalent in R is controlled via `run_solrcmf()` settings parameters rather than a
standalone function.

### 4.4 Utility Functions

**Python** exports a focused set of high-level utilities:

| Function | Purpose |
|---|---|
| `simulate()` | Generate synthetic data with controllable SNR |
| `bicenter()` | Row + column centering (Toeplitz-style) |
| `nanscale()` | Frobenius-norm scaling, NaN-aware |
| `multiview_init()` | Multi-view warm start |
| `best_random_init()` | Best-of-N random starts |

**R** exposes a broader set of lower-level matrix utilities:

| Function | Purpose |
|---|---|
| `frob()` | Frobenius norm |
| `normalise()` | Matrix normalization |
| `row_apply()` / `col_apply()` | Row/column-wise function application |
| `row_center()` / `col_center()` / `row_col_center()` | Centering |
| `threepart_center()` | Three-part decomposition centering |
| `no_na()` | NA removal helper |
| `create_data_solrcmf()` | Construct input data object |
| `create_settings_solrcmf()` | Construct settings object with defaults |
| `create_matrix_folds()` | Create CV fold indices |
| `create_random_search_matrix()` | Build random hyperparameter search grid |

The R utility surface is larger and more granular, reflecting the lower-level control
the package offers. The Python package delegates most preprocessing to the user (or
to pandas/numpy directly) and exposes only the operations that are non-trivial.

### 4.5 Cross-Validation and Imputation

**Python** has a dedicated `LowRankImputation` class (separate from `SolrCMFCV`) and an
`ElementwiseFolds` splitter that creates element-wise held-out masks for matrix entries.
`SolrCMFCV` implements a two-step procedure: grid search followed by best-solution
selection with a one-standard-error rule.

**R** provides `create_matrix_folds()` (analogous to `ElementwiseFolds`) and
`create_random_search_matrix()` for building a random hyperparameter grid. The
`run_solrcmf_cv()` function orchestrates the cross-validation loop. Missing-value
handling is built into `create_data_solrcmf()` rather than a separate imputation class.

---

## 5. Feature Comparison Matrix

| Feature | Python | R/C++ | Notes |
|---|---|---|---|
| Gaussian likelihood | Yes | Yes | Closed-form updates in both |
| Bernoulli likelihood | No | Yes | Requires proximal gradient design |
| Poisson likelihood | No | Yes | Requires proximal gradient design |
| Structure sparsity on d[k] | Yes | Yes | L1 on scaling vectors |
| Factor sparsity on V | Yes | Yes | Different parameterization |
| Hierarchical λ triplet | No | Yes | Python uses 2-level; R uses 3-level |
| Cross-validation | Yes (class) | Yes (function) | |
| Missing data imputation | Yes (dedicated class) | Partial (built-in handling) | |
| Multi-view warm start | Yes | No | Python's `multiview_init()` |
| Best-of-N random starts | Yes | Yes | Different packaging |
| Parallel fitting | Yes (joblib) | Not explicit | |
| sklearn-compatible estimator | Yes | N/A | |
| Type annotations (static) | Full (py.typed) | N/A | |
| GPU acceleration | No | No | |
| CRAN / PyPI published | No | No | Install from source only |
| Hosted documentation site | Yes (GitHub Pages) | No | |

---

## 6. Packaging and Dependency Management

### Python (`solrcmf`)

- **Layout:** Modern `src/` layout — package lives in `src/solrcmf/`.
- **Build backend:** `hatchling` (PEP 517/518 compliant).
- **Package manager:** [uv](https://github.com/astral-sh/uv) with a `uv.lock` file for
  fully reproducible environments.
- **Runtime dependencies:** only `numpy>=1.25`, `joblib>=1.3.2`, `scikit-learn>=1.1.0`.
- **Type safety:** `py.typed` marker file enables downstream type checking with mypy/pyright.
- **Installation:** `pip install .` or `uv sync` — no compiler required.

### R (`solrcmf-r`)

- **Layout:** Standard CRAN-compatible package layout (`R/`, `src/`, `man/`, `tests/`,
  `DESCRIPTION`, `NAMESPACE`).
- **Build system:** `R CMD BUILD` + `R CMD INSTALL`; a convenience script
  `build-and-install.sh` wraps these steps.
- **Runtime dependencies:** `Rcpp (>= 1.0.2)` and `RcppArmadillo` only — but both
  require the package to compile C++ code at install time.
- **C++ compilation:** `Makevars` and `Makevars.win` configure platform-specific flags.
  Installation requires Rtools (Windows) or a GCC/Clang toolchain (macOS/Linux).
- **Documentation generation:** `roxygen2` (RoxygenNote 7.0.2) used for man pages.

### Portability Trade-off

The Python package installs with a single command on any machine with Python 3.12
and pip/uv. The R package requires a C++ toolchain, which can be a barrier in
locked-down environments or on machines without developer tools. In practice, most
R users on academic computing clusters already have the necessary tools.

---

## 7. Testing Philosophy

### Python — Formal pytest Suite

Three pytest files with well-defined scope:

| File | What it tests |
|---|---|
| `tests/test_solrcmf.py` | Core algorithm: convergence, output structure, score quality, rank constraints |
| `tests/test_simulate.py` | Data simulation: dense/sparse factors, SNR control, RNG seeding |
| `tests/test_crossval.py` | CV pipeline: random init, custom init, result structure |

Tests use `numpy.testing` for numerical assertions and controlled random seeds for
reproducibility. The suite can be executed in CI with `pytest` and is configured in
`pyproject.toml` (importlib mode).

### R — Example / Script Style

The `tests/` directory contains R scripts oriented around concrete examples:

| File | Purpose |
|---|---|
| `gen_test_data.R` | Generate synthetic Gaussian test data |
| `gen_bernoulli_test_data.R` | Generate Bernoulli test data |
| `gen_poisson_test_data.R` | Generate Poisson test data |
| `test_setup.R` / `bernoulli_test_setup.R` / etc. | Run algorithm on test data |
| `movielens.R` | MovieLens dataset application |
| `jonatans_bioex.R` | Biological data example |

These are closer to integration demonstrations than unit tests. There is no `testthat`
framework and no automated assertions — correctness is checked by visual inspection of
results. The scripts are valuable as worked examples but do not provide a regression-
detection safety net.

**Summary:** The Python package has a significantly stronger automated testing story.
Adding a `testthat`-based test suite would be the most impactful improvement to the R
package.

---

## 8. Documentation

### Python

- **Format:** MkDocs with the Material theme and the `mkdocstrings[python]` plugin.
- **Source of truth:** Google-style docstrings in source code; rendered automatically
  into the site.
- **Hosted at:** https://cyianor.github.io/solrcmf (GitHub Pages, `gh-pages` branch).
- **Content:** Overview, Getting Started notebook, full API Reference, Internals page.
- **Notebooks:** `README.ipynb` and `docs/getting-started.ipynb` rendered inline via
  `mkdocs-jupyter`.

The documentation site is accessible without any R/Python knowledge and provides a
"getting started" experience suitable for new users.

### R

- **Format:** Roxygen2-generated `.Rd` man pages (standard R documentation).
- **Access:** `?run_solrcmf`, `?create_data_solrcmf`, etc. from within an R session.
- **Hosted:** README.md on GitHub; no dedicated documentation site.
- **Content:** Function-level `@param`, `@return`, `@examples` tags in source; README
  provides a project overview.

The R documentation is thorough at the function level but requires familiarity with the
`?function` workflow and does not offer the narrative "getting started" experience that
the Python site provides.

---

## 9. Maturity and Development Trajectory

| Milestone | Date | Repository |
|---|---|---|
| R package v0.2.2 released | January 2020 | `solrcmf-r` |
| Python package created | April 2024 | `solrcmf` |
| arXiv preprint (Held et al.) | May 2024 | — |
| Python package last updated | April 2026 | `solrcmf` |

The timeline makes the relationship clear: **`solrcmf-r` is the research prototype** that
was used to develop and validate the algorithm during the research phase. **`solrcmf`
(Python) is the production implementation** written for the paper release, incorporating
everything learned from the R prototype: cleaner API, formal packaging, type hints, a
proper test suite, and hosted documentation.

The R package is complete for its research purpose and its multi-distribution capability
remains a unique advantage. The Python package is actively maintained and reflects the
current state of the algorithm and API design.

Neither package has CI/CD configured, which is a shared gap for both.

---

## 10. When to Use Each

**Use `solrcmf` (Python) if:**
- Your data follows a Gaussian / Normal distribution (least-squares objective).
- You want a `pip install`-able package with no compiler requirements.
- You work in the Python / scikit-learn ecosystem and want estimator compatibility.
- You want type-checked code, hosted documentation, or a formal test suite.
- You are reproducing or building on results from Held et al. (2024) — this is the
  canonical implementation associated with the paper.

**Use `solrcmf-r` (R/C++) if:**
- Your data follows a Bernoulli or Poisson distribution (binary / count data).
- You prefer R's ecosystem and the `?function` documentation workflow.
- You need the three-level hierarchical sparsity parameterization (λ triplet).
- You are extending the algorithm to new distribution families (the proximal gradient
  design makes this straightforward).

**If you need multi-distribution support in Python**, the R package's two-block ADMM +
proximal gradient design is the reference architecture to follow.

---

## 11. Shared Gaps

Both repositories share the following limitations:

- **No CI/CD:** Neither has GitHub Actions workflows. Adding a minimal CI pipeline
  (lint + test on push) would be a low-effort, high-value improvement.
- **No CRAN / PyPI publication:** Both must be installed from source. The Python package
  is closer to PyPI-ready (proper `pyproject.toml`, MIT license, `py.typed`).
- **No cross-language numerical equivalence tests:** There are no shared test fixtures
  that verify both implementations produce the same results on the same Gaussian input.
  Such tests would increase confidence that the implementations are consistent.
- **No GPU support:** Both use CPU-only numerical backends (NumPy/BLAS and
  RcppArmadillo respectively).
- **Non-Gaussian distributions in Python:** The most actionable functional gap — the
  Python package does not support Bernoulli or Poisson likelihoods.

---

## 12. Appendix: Repository Metadata

| | `solrcmf` (Python) | `solrcmf-r` (R/C++) |
|---|---|---|
| GitHub URL | https://github.com/cyianor/solrcmf | https://github.com/cyianor/solrcmf-r |
| Default branch | `main` | — |
| Feature branch (this report) | `claude/compare-repositories-report-DwE9q` | — |
| Language(s) | Python 3.12 | R, C++ |
| License | MIT | MIT |
| Stars | 1 | — |
| Created | 2024-04-30 | — |
| Last commit | 2026-04-05 | 2020-01-27 (v0.2.2) |
| Runtime deps | numpy, joblib, scikit-learn | Rcpp, RcppArmadillo |

### Paper Citation

```bibtex
@misc{held2024solrcmf,
  title     = {Sparse and orthogonal low-rank Collective Matrix Factorization (solrCMF):
               Efficient data integration in flexible layouts},
  author    = {Held, Felix and others},
  year      = {2024},
  eprint    = {2405.10067},
  archivePrefix = {arXiv},
  primaryClass  = {stat.ML}
}
```
