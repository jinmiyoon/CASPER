# Components

Per Application Design Q1 (Answer: A), this document uses the same 6 logical groupings identified during Reverse Engineering as the formal "components" for this refactor cycle.

## Component: Batch Orchestrator
- **Maps to**: `casper/interface/batch.py` (`Batch` class)
- **Purpose**: Coordinates the entire multi-spectrum processing pipeline end-to-end, from raw input to final output artifacts.
- **Responsibilities**: I/O path resolution, parameter/spectra loading, invoking every downstream scientific stage per star in the correct order, output file generation.
- **Interfaces**: Public interface is the sequence of `Batch` methods called by `main.py` (see [component-methods.md](component-methods.md)). No REST/network interface.
- **Risk/Test Status**: No existing unit test coverage (highest priority per requirements.md NFR-3).

## Component: Spectrum Domain Model
- **Maps to**: `casper/interface/spectrum.py` (`Spectrum` class)
- **Purpose**: In-memory representation of a single star's spectrum and every derived scientific quantity produced during analysis.
- **Responsibilities**: Store raw/processed spectral data, photometry, S/N statistics, MCMC results, and adopted parameters; expose getters consumed by `Batch` and output generation.
- **Interfaces**: Constructed from FITS/CSV input; consumed via getter methods and direct attribute access by `Batch` and `interface_main`.
- **Risk/Test Status**: No existing unit test coverage (highest priority per requirements.md NFR-3).

## Component: GISIC Continuum Normalization
- **Maps to**: `casper/interface/gisic/` (`normalize.py`, `spectrum.py`, `segment.py`, `norm_functions.py`)
- **Purpose**: Removes the stellar continuum from observed/synthetic flux via inflection-point segmentation so spectral features can be measured on a common scale.
- **Responsibilities**: Segment generation, robust (MAD-based) continuum point selection, spline continuum fitting, molecular-band exclusion.
- **Interfaces**: `normalize(wave, flux, sigma, k, cahk, band_check, flux_min, boost, return_points)` → `(wave, norm, cont[, points])`.
- **Risk/Test Status**: Already has unit test coverage (`casper/tests/gisic/`).

## Component: Temperature & Carbon Diagnostics
- **Maps to**: `casper/interface/temp_calibrations.py`, `casper/interface/EW.py`, `casper/interface/ac.py`, `casper/interface/MAD.py`
- **Purpose**: Derive photometric temperature estimates and carbon-mode/abundance-conversion diagnostics used as priors and classifiers.
- **Responsibilities**: Multi-calibration Teff estimation (Bergeat, Hernandez, Casagrande, Fukugita), equivalent-width measurement, carbon-mode classification, A(C)/[C/Fe] conversion, robust statistics (MAD).
- **Interfaces**: Pure functions taking photometric colors/spectral arrays, returning scalar/array estimates (see [component-methods.md](component-methods.md)).
- **Risk/Test Status**: Already has unit test coverage (`casper/tests/interface/test_temp_calibrations.py`, `test_EW.py`, `test_ac.py`, `test_MAD.py`).

## Component: Parameter Estimation Engine
- **Maps to**: `casper/interface/MCMC_interface.py`, `casper/interface/interface_main.py`, `casper/interface/MLE_priors.py`, `casper/interface/synthetic_functions.py`
- **Purpose**: Performs the Bayesian (MCMC) fit of stellar parameters against synthetic spectral models, including archetype classification and surface gravity estimation.
- **Responsibilities**: Coarse/refine MCMC sampling (via `emcee`), log-likelihood and prior computation, KDE-based posterior peak extraction, archetype (GI/GII/GIII) classification, synthetic spectrum interpolation/generation.
- **Interfaces**: Operates on a `Spectrum` object and pickled spectral/isochrone interpolators; mutates `Spectrum` state (`MCMC_COARSE`, `MCMC_REFINE`, `logg`, `synth_spectrum`, `LL_DICT`).
- **Risk/Test Status**: `interface_main.py` has no existing unit test coverage (highest priority per requirements.md NFR-3); `MCMC_interface.py` and `synthetic_functions.py` already have test coverage.

## Component: Reporting (Output & Plotting)
- **Maps to**: `casper/interface/plot_functions.py`, plus output-writing methods on `Batch`/`Spectrum`
- **Purpose**: Produce the final human/scientist-facing artifacts used to interpret and publish results.
- **Responsibilities**: Generate parameter/SNR/calibration/archetype tables, observed-vs-synthetic spectra CSV, diagnostic PDFs (spectral fits, corner plots, trace plots).
- **Interfaces**: Consumes fully-processed `Spectrum` objects from `Batch`; writes files to the configured output directory.
- **Risk/Test Status**: No existing unit test coverage; lower priority (output-only, visual, no scientific parameter computation).

## Out of Scope
- `casper/interface/libraries/` (pickled data grids) — read-only external data, not refactored.
- `casper/utils/not_used/` — legacy/deprecated code, not part of the active pipeline; may be used as an oracle reference for property-based/regression comparisons (per requirements.md NFR-6) but is not itself refactored.
