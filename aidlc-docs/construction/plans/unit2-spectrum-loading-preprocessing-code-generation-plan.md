# Unit 2: Spectrum Loading & Preprocessing — Code Generation Plan

## Unit Context
- **Files in scope**: `casper/interface/spectrum.py` (full `Spectrum` class — owned entirely by this unit; no later unit revisits this file structurally)
- **Batch touchpoints** (in `casper/interface/batch.py`, per `unit-of-work.md`): `load_params()`, `load_spectra()`, `set_params()`, `radial_correct()`, `build_frames()`
- **Dependencies**: Unit 1 (I/O & Configuration) — COMPLETE
- **Test coverage status**: Zero existing tests for `spectrum.py` (highest priority per requirements.md NFR-3 — first tests for this module); `Batch` orchestration methods above also have no direct tests yet
- **Governing rules**: `.github/instructions/casper-python-style.instructions.md` (NumPy/astropy docstrings, preserve existing comments, output equivalence, README policy); Application Design Q2 (internal signatures may change freely), Q5 (testability seams/DI authorized)
- **Available test fixtures**: `casper/inputs/spectra/test_spectra/` already contains real sample spectra covering both supported formats: `he0017_m1b_casper.fits` (FITS), `g77-61_coadd_spectrum.csv`, `hd198269_coadd_spectrum.csv`, `smss1738_coadd_spectrum.csv` (CSV) — to be reused as test inputs rather than fabricating synthetic ones

## Finding to resolve before generation
`Spectrum.get_frame_norm()` (line ~789) returns `self.norm["norm"]`, but the only method that sets `self.norm` — `set_norm()` — assigns it a plain `np.ndarray` (not a dict/DataFrame), so `self.norm["norm"]` would raise a `TypeError` if ever called. Confirmed via workspace-wide search: **neither `get_frame_norm()` nor `set_norm()` is called anywhere** in `batch.py` or elsewhere — this is dead/latent-broken code, analogous to the `io_paths.py` finding in Unit 1. (Note: `set_frame_norm()` / `get_frame_norm()`'s likely intended counterpart, which writes/reads the `"norm"` column on `self.frame`, is what's actually used by the pipeline — see `batch.py:276`.)

**Question**: How should Unit 2 handle `get_frame_norm()` / `set_norm()`?
- A) Fix `get_frame_norm()` to return `self.frame["norm"]` (matching the pattern of `get_frame_wave()`/`get_frame_flux()` and its actual likely intent), keep `set_norm()` as-is since it's a separate (also-unused) attribute
- B) Remove both `set_norm()` and `get_frame_norm()` entirely as dead code
- C) Leave both untouched, out of scope for this unit (just add a regression-pinning test that documents the current — broken — behavior)
- D) Other (describe)

**Resolved: Option B** — user confirmed `set_norm()` and `get_frame_norm()` will be removed entirely from `spectrum.py` (2026-08-10).

## Plan Steps

- [x] Step 1: Resolve the `get_frame_norm()`/`set_norm()` question above — **Answer: B (remove both)**
- [x] Step 2: Business Logic Generation
  - [x] 2a. Review all docstrings in `spectrum.py` against NumPy/astropy style (most are already present and well-formed from prior work) — fill in any gaps, preserve all existing content, do not remove or reword unnecessarily
  - [x] 2b. Review testability seams (Application Design Q5): the nested `spectra_input()` closure inside `Batch.load_spectra()` bundles file-format dispatch with I/O, making isolated testing harder — extract as a module-level or static helper if it simplifies testing, without changing behavior (internal signatures may change per Q2)
  - [x] 2c. Remove `set_norm()` and `get_frame_norm()` from `spectrum.py` per Step 1 answer (B)
- [x] Step 3: Business Logic Unit Testing
  - [x] 3a. Add `casper/tests/interface/test_spectrum.py`:
    - Constructor, FITS path: load `he0017_m1b_casper.fits`, verify wavelength derivation (`CD1_1`/`CDELT1` selection, linear vs. log10 `CRVAL1` branch), flux extraction (`obtain_flux()` 1D/2D branches), byte-order swap branch
    - Constructor, CSV path: load one of the sample CSVs, verify `wavelength`/`flux`/`original_wavelength` populated correctly
    - `radial_correction()`: verify Doppler-shift formula against `original_wavelength`
    - `ebv_correct()`: verify dereddening math for `EBV_SFD > 0` and the "already corrected" branch
    - `trim_frame()`: verify inclusive wavelength-bound filtering
    - `estimate_sn()` / `get_sn()`: verify S/N and XI statistics on a constructed frame, including the out-of-coverage NaN branch
    - `set_params()`: verify attribute assignment plus both `AssertionError` cases (invalid `CLASS`, invalid `MODE`)
    - `prepare_regions()`: verify CA/CH region slicing and the conditional C2 region under `"CH+C2"` mode
    - Remaining setters/getters (`set_KP_bounds`/`get_KP_bounds`, `set_carbon_mode`/`get_carbon_mode`, `set_temperature`/`get_photo_temp`, `set_frame`/`get_frame*`, `set_flux`/`get_flux`, `set_synth_spectrum`, `set_mcmc_args`, `set_mcmc_results`/`get_mcmc_dict`, `set_sampler`, `set_kde_functions`/`get_kde_dict`, `set_group_ll`, `get_sequence`/`get_filename`/`get_starname`): lightweight pass-through/round-trip tests
    - `set_norm()`/`get_frame_norm()`: removed per Step 1 (B) — no test needed
  - [x] 3b. Add `casper/tests/interface/test_batch.py`:
    - `load_params()`: verify dtype coercion (`sequence`/`mode`/`class`/`carbon_mode` as strings) and `self.sequence` list population, using a small fixture parameter file
    - `load_spectra()`: verify correct dispatch to `Spectrum` for both `.fits` and `.csv` inputs (using the real sample files) and that an unsupported extension raises the documented exception
    - `set_params()`: verify parameter propagation to each `Spectrum` and the filename-mismatch assertion
    - `radial_correct()`: verify RV lookup per sequence and application to each `Spectrum`
    - `build_frames()`: verify `frame` construction and bounds-trimming across the batch
- [x] Step 4: Business Logic Summary — brief summary of what changed and why
  - Added module docstring to `spectrum.py`; removed dead `set_norm()`/`get_frame_norm()` (Step 1, Option B); extracted `Batch.load_spectra()`'s nested closure to a module-level `_load_spectrum_file()` helper for testability (no behavior change).
  - Added 41 new tests: 33 in `casper/tests/interface/test_spectrum.py` (constructor FITS/CSV paths, radial correction, EBV correction, frame trimming, S/N estimation, param validation, region prep, and all remaining setters/getters) and 8 in `casper/tests/interface/test_batch.py` (`load_params`, `load_spectra` incl. FITS/CSV dispatch and unsupported-extension error, `set_params` incl. mismatch assertion, `radial_correct`, `build_frames`). Full suite: 130/130 passing (89 pre-existing + 41 new).
- [x] Step 5: Documentation Generation — README update only if user-visible behavior changed (expected: **no**, internal-only per FR-5)
  - Confirmed no user-visible behavior changed; README left untouched.
- [x] Step 6: Regression Verification
  - [x] 6a. Fast dev-loop check: ran with `param_file_single_star.dat`, confirmed no errors through loading/preprocessing and the full pipeline
  - [x] 6b. Gate run (per user instruction, 2026-08-10): ran single-star `param_file_single_star.dat` (G77-61) only, diffed against the G77-61 row of `regression_baseline/unit0_baseline/scatter2_*`.
    - **Tier 1 (deterministic)**: `temp_cal_table.txt` — identical (3764.368, 3786.950, nan, nan, 4127, 4127). `snr.csv` — all 10 shared numeric columns identical (SN_AVG_CA=82.0, XI_AVG_CA=0.0125, XI_AVG_CH=0.0075, etc.); the only difference is that the 4-star baseline additionally carries `XI_C2`/`XI_C2_ERR` columns (NaN for this DWARF/CH star) because another star in that batch used `CH+C2` mode — this is a batch-composition artifact of comparing a 1-star run to a 4-star run, not a regression.
    - **Tier 2 (stochastic, 2σ tolerance)**: TEFF ratio=0.025σ, FEH ratio=0.017σ, CFE ratio=0.000σ, AC ratio=0.015σ, LOGG ratio=0.000σ — all far inside the 2σ threshold.
    - **Result: PASS.** **Coverage caveat**: per user instruction this gate covered only 1 of 4 stars (DWARF/CH mode) instead of the documented full 4-star official gate (`regression_baseline/unit0_baseline/README.md`, NFR-1 Q4 = A) — no GIANT-class or CH+C2-mode star was exercised in this run.
- [x] Step 7: Commit — one commit for this unit (per Units Generation Q4 = A), only after Step 6b passes

## Story/Requirement Traceability
Per `unit-of-work-story-map.md`: FR-2, FR-4, NFR-1, NFR-2, NFR-3 (first tests for `spectrum.py`, the flagged high-risk-priority module for this unit).
