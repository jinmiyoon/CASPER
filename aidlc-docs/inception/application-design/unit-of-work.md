# Unit of Work

Per Units Generation Q1 (Answer: A), this finalizes the 6 units originally proposed in [../plans/execution-plan.md](../plans/execution-plan.md) and [components.md](../application-design/components.md), unchanged.

## Update Strategy (from Q2, Q3, Q4)
- **Sequencing**: Strictly sequential — each unit is fully completed (code + tests + regression diff passing) before the next unit begins (Q2 = A).
- **Team model**: Solo — user + AI assistant, one unit at a time, no parallel branches this cycle (Q3 = A).
- **Commit granularity**: Each unit corresponds to its own commit (or small commit set), independently reviewable/revertible (Q4 = A).
- **`batch.py` handling**: Touched incrementally within each unit (only the methods relevant to that unit) rather than as a separate 7th cross-cutting unit — consistent with Q1 = A (no further split) and the Application Design services.md orchestration model.

## Units

### Unit 1 — I/O & Configuration
- **Responsibilities**: `user_config.py`, `io_paths.py`, `config.py`
- **Batch touchpoints**: `set_io_paths()`
- **Test coverage status**: None currently — first tests to be added here
- **Dependencies**: None (foundational)

### Unit 2 — Spectrum Loading & Preprocessing
- **Responsibilities**: `spectrum.py` (full `Spectrum` class)
- **Batch touchpoints**: `load_params()`, `load_spectra()`, `set_params()`, `radial_correct()`, `build_frames()`
- **Test coverage status**: None currently (highest priority, per requirements.md NFR-3) — first tests for `spectrum.py`
- **Dependencies**: Unit 1

### Unit 3 — GISIC Normalization
- **Responsibilities**: `gisic/normalize.py`, `gisic/spectrum.py`, `gisic/segment.py`, `gisic/norm_functions.py`
- **Batch touchpoints**: `normalize()`
- **Test coverage status**: Existing (`casper/tests/gisic/`) — extend as needed
- **Dependencies**: Unit 2 (needs frame data)

### Unit 4 — Temperature Calibration & Extinction
- **Responsibilities**: `temp_calibrations.py`, `EW.py`, `MAD.py`, `ac.py`
- **Batch touchpoints**: `ebv_correction()`, `calibrate_temperatures()`, `set_KP_bounds()`, `set_carbon_mode()`, `estimate_sn()`, `get_sn()`
- **Test coverage status**: Existing (`casper/tests/interface/test_temp_calibrations.py`, `test_EW.py`, `test_ac.py`, `test_MAD.py`) — extend as needed
- **Dependencies**: Units 2, 3

### Unit 5 — Archetype Classification & MCMC
- **Responsibilities**: `interface_main.py`, `MCMC_interface.py`, `MLE_priors.py`, `synthetic_functions.py`
- **Batch touchpoints**: `archetype_classification()`, `mcmc_determination()`, `estimate_logg()`, `generate_synthetic()`
- **Test coverage status**: `interface_main.py` has none currently (highest priority, per requirements.md NFR-3) — first tests for `interface_main.py`; `MCMC_interface.py`/`synthetic_functions.py` have existing tests to extend
- **Dependencies**: Units 3, 4
- **Risk note**: Highest scientific-risk unit — core Bayesian parameter estimation engine

### Unit 6 — Output Generation & Plotting
- **Responsibilities**: `plot_functions.py`
- **Batch touchpoints**: `generate_output_spectra()`, `generate_output_files()`, `generate_plots()`
- **Test coverage status**: None currently — lower priority (output-only, no scientific computation) but still gains basic tests
- **Dependencies**: Units 1–5 (consumes final state of everything)

## Cross-Cutting Final Step (not a separate unit)
After Unit 6, a final integration pass adds tests directly for `Batch` as a whole (its orchestration/sequencing logic), since no single unit above owns `batch.py` in its entirety — each unit only touches the specific methods relevant to it. This is tracked as part of Build and Test, not as a 7th unit (per Q1 = A).
