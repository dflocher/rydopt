# AGENTS.md

Use `uv` to build and test the package. Run tests only selectively to save time.

## Tips for Formatting and Linting

```bash
uvx pre-commit run --files $(git diff --name-only HEAD)
```

## Tips for Building the Documentation

```bash
(cd docs && uv run make html)
```
