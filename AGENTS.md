# Repository workflow

## Python environment and dependencies

- This project uses `uv` for Python version management, dependency management,
  lockfile maintenance, and command execution.
- Use the Python version declared in `.python-version`.
- Synchronize the environment with `uv sync --locked --all-groups`.
- Add runtime dependencies with `uv add <package>` and development dependencies
  with `uv add --dev <package>`.
- Keep `pyproject.toml` and `uv.lock` in sync. Do not edit `uv.lock` manually.
- Run project tools through `uv run`; do not invoke executables in `.venv`
  directly or add `src` to `PYTHONPATH`.

## Required validation

Run the complete local quality gate before committing:

```sh
./scripts/check
```

It synchronizes the locked environment and requires all of the following to
pass:

```sh
uv run --locked --no-sync ruff check .
uv run --locked --no-sync ruff format --check .
uv run --locked --no-sync ty check
uv run --locked --no-sync pytest
```

To run one test module or test while developing:

```sh
uv run --locked pytest tests/test_solrcmf.py
uv run --locked pytest tests/test_solrcmf.py::test_convergence
```

If formatting changes are needed, run:

```sh
uv run --locked ruff format .
```

## Git hooks

Install the tracked pre-commit hook once per clone:

```sh
./scripts/install-git-hooks
```

The hook runs `./scripts/check`. GitHub Actions runs the same command for every
pull request and every push to `main`.

## Testing expectations

- Add regression tests for every bug fix.
- For estimation changes, test recovery of the simulated quantities rather
  than only checking that fitting completes.
- Use fixed random seeds for reproducibility and compare quantities with
  tolerances appropriate to the numerical method.
- Run performance measurements through `uv run` so they use the locked project
  environment.
