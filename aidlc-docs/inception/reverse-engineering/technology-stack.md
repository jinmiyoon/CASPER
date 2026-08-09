# Technology Stack

## Programming Languages
- Python — ≥3.11 (per `pyproject.toml` `requires-python`) — sole implementation language

## Frameworks / Scientific Libraries
- numpy 2.3.0 — array operations, numerical routines
- scipy 1.15.3 — spline fitting, optimization, integration, statistical distributions
- pandas 2.3.0 — tabular data (spectra frames, parameter tables, outputs)
- astropy 7.1.0 — FITS I/O
- emcee 3.1.6 — MCMC ensemble sampler
- statsmodels 0.14.4 — kernel density estimation (posterior smoothing)
- matplotlib 3.10.3 — plotting
- corner 2.2.3 — MCMC corner plots
- texttable 1.7.0 — formatted text table output
- tqdm ≥4.67.1 — progress bars

## Infrastructure
- None — no cloud services; runs as a local script/package.
- Git LFS — used to store large pickled spectral/isochrone grid files outside normal git history.

## Build Tools
- setuptools ≥62.1 + setuptools_scm ≥8.0.0 — build backend and git-tag-based versioning
- pip — package installation (`pip install .`, `pip install -e .`)
- tox 4.x — multi-version test/docs orchestration ([tox.ini](../../../tox.ini))
- ruff 0.15.5 — linting/formatting
- pre-commit 4.5.1 — git hook enforcement of lint rules

## Testing Tools
- pytest — test runner (`casper/tests/`)
- pytest-cov — coverage reporting
- pytest-doctestplus — doctest integration (also runs `--doctest-rst` per `pyproject.toml`)

## Documentation Tools
- Sphinx — documentation generator
- sphinx-astropy, sphinx-automodapi, sphinx-rtd-theme, sphinx-book-theme — Sphinx themes/extensions
- myst-parser — Markdown support in Sphinx
- sphinx-copybutton, sphinx_click, sphinxcontrib-spelling, linkify-it-py — supporting Sphinx extensions
- Napoleon (via Sphinx) — parses NumPy-style docstrings used throughout the codebase

## CI/CD
- GitHub Actions — [.github/workflows/ci.yml](../../../.github/workflows/ci.yml) (lint, test, docs build) and [.github/workflows/release.yml](../../../.github/workflows/release.yml) (tag-triggered GitHub Release creation)
