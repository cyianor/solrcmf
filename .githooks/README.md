# Git hooks

Install the tracked hooks once in each clone:

```sh
./scripts/install-git-hooks
```

This configures the repository-local `core.hooksPath` to `.githooks`. The
pre-commit hook runs the same Ruff, ty, and pytest gate used by GitHub Actions.

To verify the hook without creating a commit:

```sh
.githooks/pre-commit
```

Git's `--no-verify` option can bypass a local hook, so GitHub Actions remains
the authoritative required check.
