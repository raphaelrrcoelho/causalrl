# Contributing

`causalrl` accepts focused fixes, reference-validated causal algorithms, benchmark
improvements, and documentation corrections.

## Development Setup

```bash
git clone https://github.com/raphaelrrcoelho/causalrl.git
cd causalrl
uv sync --extra dev --extra docs
uv run --extra dev pytest
uv run --extra dev ruff check .
uv run --extra dev pyright src
uv run --extra docs mkdocs build --strict
```

## Expectations

- Add tests before changing behavior.
- Cite a primary source for implemented causal algorithms and include oracle fixtures or
  reproducible benchmark evidence where applicable.
- State method assumptions in public APIs and documentation; do not present experimental
  helpers as validated estimators.
- Keep new public API changes typed and documented.

## Pull Requests

Describe the causal or software contract changed, the evidence used for validation, and the
commands run. Small, reviewable changes are preferred.
