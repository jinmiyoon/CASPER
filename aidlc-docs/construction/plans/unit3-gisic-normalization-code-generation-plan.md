# Unit 3: GISIC Normalization — Code Generation Plan

## Unit Context
- **Files in scope**: `casper/interface/gisic/normalize.py`, `casper/interface/gisic/spectrum.py` (the GISIC `Spectrum` class — distinct from `casper.interface.spectrum.Spectrum`), `casper/interface/gisic/segment.py`, `casper/interface/gisic/norm_functions.py`
- **Batch touchpoint**: `Batch.normalize()` in `casper/interface/batch.py`
- **Dependencies**: Unit 2 (Spectrum Loading & Preprocessing) — COMPLETE
- **Test coverage status**:
  - `segment.py` and `norm_functions.py`: already well covered by `casper/tests/gisic/test_gisic_segment.py` and `test_gisic_norm_functions.py`
  - `normalize.py`: one example-based test (`test_gisic_normalize.py`) covering the happy path and output-shape/bounds invariants
  - `casper/interface/gisic/spectrum.py` (the `Spectrum` class orchestrating segmentation/continuum-fitting): **zero direct tests** — only exercised indirectly through `normalize()`'s single test. This is the main gap for this unit.
  - `Batch.normalize()`: no direct test yet
- **Governing rules**: `.github/instructions/casper-python-style.instructions.md` (NumPy/astropy docstrings, preserve existing comments, output equivalence, README policy); NFR-6 (Property-Based Testing, full enforcement) explicitly names GISIC `normalize()`'s output-bounds invariant as a target PBT candidate for this unit — requires adding `hypothesis` as a `test` extra dependency (not yet installed)

## Finding resolved before generation
`casper.interface.gisic.spectrum.Spectrum.normalize()` clips out-of-bounds normalized flux only when **more than one** point violates the bound (`if len(flux_norm[flux_norm < 0.0]) > 1:` / `> 2.0`) — a single stray out-of-bounds point is left unclipped. Traced both call sites of the `normalize()` function that wraps this method (`Batch.normalize()` and `synthetic_functions.py`): both discard the `norm` return value entirely (only `continuum` is used; `Batch.normalize()` recomputes and unconditionally clips its own `norm` from the median continuum). So this `>1` quirk is a real bug (most likely a typo for `> 0`, a "non-empty" check) but has **zero effect on any current pipeline output** since the affected value is discarded everywhere it's produced.

**Resolved: Option B** — user confirmed (2026-08-11): fix `> 1` → `> 0` in `gisic.spectrum.Spectrum.normalize()` for correctness, since the value is inert today but could matter if ever consumed later.

## Second finding resolved before generation (behavior-affecting, not inert)
While writing `Batch.normalize()` test coverage, found `Batch.normalize()` passes `band_check=config.cahk` instead of `band_check=config.band_check` (`casper/interface/batch.py`, the call to `normalize()` inside the per-`SIGMA` loop). Since `config.cahk = True` and `config.band_check = False`, this means `band_check=True` is actually used in production instead of the intended `False` — this **does affect** which segments get excluded from continuum fitting, unlike the previous inert finding. Every other call site (e.g. `synthetic_functions.py`) correctly passes `band_check=config.band_check`.

**Resolved: Option B, then reversed to Option A** — user initially confirmed Option B (2026-08-11), then changed their mind and requested Option A: fix `band_check=config.cahk` → `band_check=config.band_check` now. Fix applied; the regression-pinning test was updated to assert the corrected wiring instead of documenting the bug. Re-ran the single-star regression gate to assess impact (see Step 6b) since this fix is **not** inert — it changed Tier 2 stochastic outputs (TEFF/FEH/CFE/AC/LOGG shifted noticeably vs. the pre-fix single-star run) but all remained within the 2σ tolerance.

## Plan Steps

- [x] Step 1: Add `hypothesis` to the `test` optional-dependency group in `pyproject.toml` (per NFR-6, PBT-09) and install it in the `casper311` dev environment
- [x] Step 2: Business Logic Generation
  - [x] 2a. Review docstrings across all four files in scope against NumPy/astropy style (already well-formed from prior authoring) — fill any gaps, preserve all existing content
  - [x] 2b. Review testability seams (Application Design Q5): `gisic.spectrum.Spectrum` already takes plain arrays in its constructor and is independently instantiable — no seam changes anticipated; confirmed, no changes needed
  - [x] 2c. Fix `gisic.spectrum.Spectrum.normalize()`: change the `> 1` count-guards to `> 0` (both the `< 0.0` and `> 2.0` clipping branches) per the resolved finding above
  - [x] 2d. Rename the unused per-`SIGMA` `norm` in `Batch.normalize()`'s loop to `_` (it is discarded/overwritten every iteration and never read — same as the already-discarded `wavelength` return value); purely cosmetic, no behavior change
  - [x] 2e. Fix `Batch.normalize()`'s `band_check=config.cahk` → `band_check=config.band_check` per the reversed second finding above (Option A)
- [x] Step 3: Business Logic Unit Testing
  - [x] 3a. Add `casper/tests/gisic/test_gisic_spectrum.py` (new) covering the GISIC `Spectrum` class:
    - `generate_inflection_segments()`: verify segments are produced, first/last marked as edges, `ZEROS`/`frame` populated, `cahk`/`band_check` branches
    - `generate_segments()` (bins-based alternative path, currently unused by the pipeline but part of the public API): verify bin count and edge marking
    - `assess_segment_variation()`: verify `mad_global`/`mad_min`/`mad_max`/`mad_range`/`mad_relative_array` computed correctly from known segment MAD values
    - `define_cont_points()`, `set_segment_midpoints()`, `set_segment_continuum()`: verify values collected from segments correctly
    - `add_continuum_point()` / `remove_point()`: verify sorted insertion and reverse-order-safe removal
    - `set_wavelength()` / `set_fluxpoints()` / `get_continuum_points()`: lightweight coverage
    - `spline_continuum()` / `normalize()`: verify continuum evaluated at input wavelengths, normalized flux formula, and verify the fixed clipping now applies for a single out-of-bounds point (as well as multiple)
  - [x] 3b. Extend `casper/tests/gisic/test_gisic_normalize.py` with a Hypothesis property test (NFR-6): generate randomized synthetic wavelength/flux arrays (bounded, finite, monotonic wavelength) and assert `normalize()`'s returned normalized flux always stays within the documented `[0, 2]` bounds
  - [x] 3c. Add `Batch.normalize()` coverage to `casper/tests/interface/test_batch.py`: construct a batch with one real sample spectrum (trimmed to `config.WAVE_BOUNDS`), call `normalize()`, verify `frame["norm"]`/`frame["cont"]` are populated; also added a test asserting the corrected `band_check=config.band_check` wiring (updated from the original bug-documenting version per the Option A reversal)
- [x] Step 4: Business Logic Summary — brief summary of what changed and why
  - Fixed `gisic.spectrum.Spectrum.normalize()`'s `>1` clipping bug (now `>0`) — inert today but correct going forward.
  - Cosmetic: renamed unused per-`SIGMA` `norm` to `_` in `Batch.normalize()`.
  - Fixed `Batch.normalize()`'s `band_check=config.cahk` → `band_check=config.band_check` — a real, behavior-affecting bugfix (Option A, after reversal from initial Option B decision).
  - Added `hypothesis` as a test dependency; added a property-based test for the GISIC `normalize()` bounds invariant (NFR-6).
  - Added 14 new tests in `casper/tests/gisic/test_gisic_spectrum.py` (previously zero direct coverage for this class), extended `test_gisic_normalize.py` (+1 property test), and added 2 new tests in `test_batch.py` for `Batch.normalize()`. Full suite: 146/146 passing.
- [x] Step 5: Documentation Generation — README update only if user-visible behavior changed (expected: **no**, internal-only per FR-5); confirmed no user-visible change, README untouched
- [x] Step 6: Regression Verification
  - [x] 6a. Fast dev-loop check: ran with `param_file_single_star.dat`, confirmed no errors
  - [x] 6b. Gate run: single-star `param_file_single_star.dat` (G77-61) vs. the G77-61 row of `regression_baseline/unit0_baseline/scatter2_*`, two-tier tolerance.
    - **Tier 1 (deterministic)**: `temp_cal_table.txt` and `snr.csv` shared numeric columns identical to baseline (unaffected by the `band_check` fix, since S/N is computed from raw frame flux, not continuum-fitting band exclusion).
    - **Tier 2 (stochastic, 2σ tolerance)**: with the `band_check` fix applied, outputs shifted more than the earlier (pre-fix) single-star run: TEFF ratio=0.650σ, FEH ratio=0.960σ, CFE ratio=0.596σ, AC ratio=1.104σ, LOGG ratio=0.615σ — larger than Unit 2's near-zero deviations, but all **within the 2σ threshold**. This is expected: `band_check` now genuinely changes which segments feed the continuum fit, so a real (not inert) shift in downstream MCMC-derived parameters is correct behavior, not a bug.
    - **Result: PASS.** Same 1-of-4-star coverage caveat as Unit 2 applies.
- [x] Step 7: Commit — one commit for this unit (per Units Generation Q4 = A), only after Step 6b passes

## Story/Requirement Traceability
Per `unit-of-work-story-map.md`: FR-2, FR-4, NFR-1, NFR-2, NFR-6 (Property-Based Testing — GISIC normalization bounds invariant, explicitly named for Unit 3).
