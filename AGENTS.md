# AGENTS.md

Use `uv` to build and test the package. Run tests only selectively to save time.

## Tips for Formatting and Linting

```bash
uvx pre-commit run --files $(git diff --name-only HEAD)
```

## Tips for Building the Documentation

On Linux or macOS:
```bash
(cd docs && uv run make html)
```

On Windows:
```powershell
try { Push-Location docs; uv run .\make.bat html } finally { Pop-Location }
```
