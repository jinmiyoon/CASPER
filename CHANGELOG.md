# Change Log

### Unreleased

### Removed
- Removed all unnecessary comments from files
- Removed all unused functions and moved them to `casper/utils/not_used/` for archival purposes


### Changed
- Updated `pyproject.toml` with updated dependencies, dev, and docs
- Updated dependencies in `casper_requirements.yml`
- Disabled `"python.analysis.typeCheckingMode"` in `.vscode/settings.json` to prevent type checking warnings
- Adjusted format to meet Ruff standards


### Added
- Updated `main.py`, `batch.py`, and `io_paths.py` so users do not have to manually create their own folders to run CASPER
- When CASPER runs, `logs/`, `npsave/`, and `outputs/` are automatically created, and data is stored within the appropriate files
- Added docstrings to each function
- Added type hints to each function
- Utilized the Open Astronomy cookiecutter package template for the CASPER package
- Added pytest to many functions of `gisic` and `interface` in `tests/`
- Set up Sphinx autodoc
- Added pytest and Sphinx autodoc installation instructions to the readme
- Updated readme instructions so they are releveant and accurate
