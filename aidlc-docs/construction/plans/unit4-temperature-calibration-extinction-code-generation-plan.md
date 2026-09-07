# Unit 4: Temperature Calibration & Extinction — Code Generation Plan

## Unit Context
- **Files in scope**: `casper/interface/temp_calibrations.py`, `casper/interface/EW.py`, `casper/interface/MAD.py`, `casper/interface/ac.py`
- **Batch touchpoints**: `Batch.ebv_correction()`, `Batch.calibrate_temperatures()`, `Batch.set_KP_bounds()`, `Batch.set_carbon_mode()`, `Batch.estimate_sn()`, `Batch.get_sn()`
- **Dependencies**: Units 2 (Spectrum Loading & Preprocessing), 3 (GISIC Normalization) — both COMPLETE
- **Test coverage status**:
  - `temp_calibrations.py`, `EW.py`, `MAD.py`, `ac.py`: all already have existing example-based tests (`test_temp_calibrations.py`, `test_EW.py`, `test_MAD.py`, `test_ac.py`) — reasonably thorough, no major gaps found
  - `Batch`'s 6 touchpoint methods above: **zero direct tests** — this is the main gap for this unit (same pattern as Unit 3's `Batch.normalize()` gap)
- **Governing rules**: `.github/instructions/casper-python-style.instructions.md` (NumPy/astropy docstrings, preserve existing comments, output equivalence, README policy); NFR-6 (Property-Based Testing, full enforcement) explicitly names two candidates for this unit: `ac.py`'s `ac()`/`cfe()` round-trip (PBT-02) and `MAD.py`'s `MAD`/`S_MAD` non-negativity (PBT-03) — **descoped per user decision (2026-09-07)**, see below

## Observation (no action needed)
`Batch.calibrate_temperatures()` has two parameters, `default: bool = True` and `teff_sigma: int = 250`, neither of which is read anywhere in the method body (`default` is already honestly documented as "not currently used"; `teff_sigma` is undocumented as unused — the method reads `spec.T_SIGMA` per-star instead). This is vestigial/dead parameter surface, not a bug — no output depends on it, so no decision needed; noted here for visibility only.

## NFR-6 descoping decision (2026-09-07)
Hypothesis property-based tests were implemented and passing for both PBT-02 (`ac`/`cfe` round-trip) and PBT-03 (`MAD`/`S_MAD` non-negativity), the two candidates explicitly named for this unit in requirements.md. The user then replaced both with expanded example-based tests instead (adding single-element-array and constant-array edge cases to `test_MAD.py`, and docstrings to `test_ac.py`'s existing cases) and explicitly chose **Option B**: keep the example-based version, and formally record that NFR-6 is not fully satisfied for Unit 4 rather than silently dropping it. Updated `requirements.md` and `unit-of-work-story-map.md` accordingly. Full suite: 153/153 passing (2 fewer than the earlier 155, reflecting the 2 removed property tests).

## Plan Steps

- [x] Step 1: Business Logic Generation
  - [x] 1a. Review docstrings across all four files in scope against NumPy/astropy style (already well-formed) — no gaps found, no changes needed
  - [x] 1b. Review testability seams (Application Design Q5): all four modules are already pure functions or take explicit arguments — confirmed, no seam changes needed
- [x] Step 2: Business Logic Unit Testing
  - [x] 2a. ~~Extended `casper/tests/interface/test_ac.py` with a Hypothesis round-trip property test~~ — originally implemented (NFR-6, PBT-02), then replaced by user with docstring-annotated example-based tests on `test_ac()`/`test_cfe()` (Option B descoping decision above)
  - [x] 2b. ~~Extended `casper/tests/interface/test_MAD.py` with a Hypothesis invariant property test~~ — originally implemented (NFR-6, PBT-03), then replaced by user with expanded example-based tests covering single-element and constant-array edge cases (Option B descoping decision above)
  - [x] 2c. Added 7 `Batch` touchpoint tests to `casper/tests/interface/test_batch.py`: `ebv_correction()`, `calibrate_temperatures()` (hard-Teff and photometric-Teff branches), `set_KP_bounds()`, `set_carbon_mode()`, `estimate_sn()`, `get_sn()`
- [x] Step 3: Business Logic Summary — brief summary of what changed and why
  - No production code changes this unit (no bugs or dead code found in `temp_calibrations.py`/`EW.py`/`MAD.py`/`ac.py`, unlike Units 2-3).
  - Added 9 new tests total: 7 new `Batch` touchpoint tests (previously zero coverage for `ebv_correction`, `calibrate_temperatures`, `set_KP_bounds`, `set_carbon_mode`, `estimate_sn`, `get_sn`) + 2 expanded example-based tests in `test_ac.py`/`test_MAD.py` (NFR-6 property-based versions descoped per user decision, see above).
  - Full suite: 153/153 passing.
- [x] Step 4: Documentation Generation — README update only if user-visible behavior changed (expected: **no**, internal-only per FR-5); confirmed no user-visible change, README untouched
- [x] Step 5: Regression Verification
  - [x] 5a. Fast dev-loop check: ran with `param_file_single_star.dat`, confirmed no errors
  - [x] 5b. Gate run: single-star `param_file_single_star.dat` (G77-61) vs. the G77-61 row of `regression_baseline/unit0_baseline/scatter2_*`.
    - **Tier 1 (deterministic)**: `snr.csv` and `temp_cal_table.txt` identical to baseline.
    - **Tier 2 (stochastic, 2σ tolerance)**: TEFF ratio=1.100σ, FEH ratio=1.203σ, CFE ratio=0.289σ, AC ratio=1.121σ, LOGG ratio=1.091σ — all within the 2σ threshold. Notably, this unit made **zero production code changes**, so these deviations are purely natural run-to-run stochastic scatter (confirms the 2σ threshold correctly accounts for inherent MCMC noise, independent of any actual code change).
    - **Result: PASS.** Same 1-of-4-star coverage caveat as prior units applies.
- [x] Step 6: Commit — one commit for this unit (per Units Generation Q4 = A), only after Step 5b passes

## Story/Requirement Traceability
Per `unit-of-work-story-map.md`: FR-2, FR-4, NFR-1, NFR-2, NFR-6 (Property-Based Testing — `ac()`/`cfe()` round-trip and MAD invariants, explicitly named for Unit 4, **descoped per user decision 2026-09-07** in favor of expanded example-based tests; formally recorded in requirements.md and the story map rather than silently dropped).
