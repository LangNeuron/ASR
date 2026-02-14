# Contributing Guide

## Branching
- Create feature branches from `main`.
- Keep pull requests focused and small.

## Local Setup
1. Install dependencies:
   ```bash
   poetry install --with dev
   ```
2. Install pre-commit hooks:
   ```bash
   poetry run pre-commit install
   ```

## Code Quality Requirements
Run all checks before opening a pull request:

```bash
poetry run ruff check .
poetry run mypy src tests
poetry run pytest
poetry run pre-commit run --all-files
```

## Testing
- Add or update tests for any behavior change.
- Prefer unit tests for logic and integration tests for workflow-level validation.

## Documentation
- Keep documentation synchronized with code changes.
- Update both language sections when behavior or architecture changes:
  - `docs/en/`
  - `docs/ru/`

## Pull Request Checklist
- [ ] Linting passes (`ruff`)
- [ ] Type checks pass (`mypy`)
- [ ] Tests pass (`pytest`)
- [ ] Documentation updated when needed
- [ ] No large artifacts committed
