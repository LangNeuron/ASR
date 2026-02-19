# ASR Project

## Project Name
ASR Project is a production-oriented workspace for building custom Automatic Speech Recognition (ASR) models and related audio ML tooling.

## Short Description
This repository provides a practical foundation for end-to-end speech ML workflows:
- dataset preparation and versioning;
- feature engineering and model training;
- evaluation with ASR metrics (WER/CER and related checks);
- reproducible experiments and inference-ready packaging.

## Project Structure
- `src/asr/` - application source code (`data`, `features`, `models`, `training`, `inference`, `utils`).
- `data/` - local datasets (`raw`, `interim`, `processed`, `external`).
- `notebooks/` - exploratory and experiment notebooks.
- `configs/` - experiment and pipeline configuration files.
- `scripts/` - CLI and automation scripts.
- `models/`, `checkpoints/`, `artifacts/`, `logs/` - training outputs and runtime artifacts.
- `tests/` - unit and integration tests.
- `docs/` - project and code documentation in multiple languages.

## Documentation
Documentation is organized by language and purpose.

Language support:
- `docs/en/` - English documentation.
- `docs/ru/` - Russian documentation.

Detailed dataset installation and downloader docs:
- English: `docs/en/code-documentation/scripts_data_download.md`
- Russian: `docs/ru/code-documentation/scripts_data_download.md`

## Installation
Requirements:
- Python 3.12+
- Poetry

Install project dependencies:

```bash
poetry install --with dev
```

Install ML stack when needed:

```bash
poetry install --with dev,ml
```

## Usage
Run quality checks:

```bash
poetry run ruff check .
poetry run mypy src tests
poetry run pytest
```

Quick dataset download example:

```bash
bash scripts/data_download.sh --datasets all
```

Install SOVA from local pre-downloaded archive:

```bash
bash scripts/data_download.sh --sova-archive /path/to/RuDevices.tar
```

Source for manual SOVA archive download:
- `https://disk.yandex.ru/d/jz3k7pnzTpnTgw`

Resume interrupted or verify existing dataset downloads:

```bash
bash scripts/data_download.sh --datasets golos
```

Force full re-download of built-in downloadable datasets:

```bash
bash scripts/data_download.sh reinstall-all
```

Retry policy overview:
- request-level retries: `5`
- dataset-level retries: `3`
- global pipeline retries: `5`

Start notebooks:

```bash
poetry run jupyter lab
```

## Pre-commit
Pre-commit is configured for `ruff`, `mypy`, and `pytest`.

Setup:

```bash
poetry run pre-commit install
poetry run pre-commit run --all-files
```

## Testing
Run all tests:

```bash
poetry run pytest
```

Run integration tests only:

```bash
poetry run pytest -m integration
```

## Contributing
Contribution guide: [CONTRIBUTING.md](CONTRIBUTING.md).

## Security
Security policy and responsible disclosure contacts: [SECURITY.md](SECURITY.md).

## License
This project is licensed under Apache License 2.0.
See [LICENSE](LICENSE).
