# Services

## Orchestration Model Decision (Application Design Q3, Answer: C)

The user deferred the final orchestration-structure decision to Construction, to be resolved **per unit** based on how invasive each unit's internal refactor turns out to be. The following default applies unless a specific unit's Code Generation step documents a reason to deviate:

- **Default**: `Batch` remains the single top-level orchestrating service, matching current behavior. `main.py` continues to call `Batch` methods in the same fixed sequence.
- **Per-unit override criterion**: If refactoring a given unit's internal logic naturally produces a small, self-contained coordinator function/object (e.g., a `TemperatureCalibrationService` that `Batch.calibrate_temperatures()` simply delegates to), that extraction is acceptable *as long as*:
  1. `Batch`'s public method (e.g., `calibrate_temperatures()`) still exists with the same externally-observable effect, and
  2. No new component boundary crosses more than one of the 6 components defined in [components.md](components.md).
- This decision must be re-confirmed (not silently assumed) in each unit's Code Generation step if the unit's author considers introducing a coordinator — per the deferral, it is not pre-approved wholesale.

## Service: Batch (Top-Level Orchestration)
- **Responsibilities**: Sequence every pipeline stage across all loaded spectra in the fixed order established in `main.py`; own I/O path resolution and final output writing.
- **Orchestrates**: Spectrum Domain Model, GISIC Normalization, Temperature & Carbon Diagnostics, Parameter Estimation Engine, Reporting.
- **Interaction Pattern**: Batch method → loops over `self.spectra_array` → delegates to component-level function/method per spectrum → (optionally) aggregates results for batch-level output (SNR table, output CSV, spectra CSV).

## Service: Parameter Estimation Engine (Internal Sub-Orchestration)
- **Responsibilities**: Within `interface_main.py`, coordinates the coarse → KDE → refine → KDE MCMC sequence and archetype classification/log g/synthetic-spectrum generation for a single `Spectrum`.
- **Orchestrates**: `MCMC_interface` (likelihood functions), `MLE_priors` (priors/bounds), `synthetic_functions` (interpolation/normalization), the pickled spectral/isochrone grids.
- **Interaction Pattern**: Called once per spectrum by `Batch`; internally sequences multiple sampling/smoothing steps before returning control to `Batch`.

## No New Services Introduced
Per Application Design Q3/Q4, this refactor does not introduce new top-level services — it formalizes documentation of the two orchestration layers that already exist (`Batch` at the pipeline level, `interface_main` at the per-spectrum parameter-estimation level).
