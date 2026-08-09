# Unit 1: I/O & Configuration — Code Generation Plan

## Unit Context
- **Files in scope**: `casper/user_config.py`, `casper/interface/config.py`, `casper/interface/io_paths.py`
- **Batch touchpoints**: `Batch.set_io_paths()`
- **Dependencies**: None (foundational unit, per unit-of-work-dependency.md)
- **Test coverage status**: `user_config.py` and `config.py` have zero existing tests; `io_paths.py` has none either
- **Governing rules**: `.github/instructions/casper-python-style.instructions.md` (NumPy/astropy docstrings, preserve existing comments, output equivalence, README policy); Application Design Q2 (internal signatures may change freely), Q5 (testability seams/DI authorized)

## Finding to resolve before generation
`casper/interface/io_paths.py` is **dead code** — confirmed via workspace-wide search that no Python module imports this file. It appears to be a legacy leftover superseded by `user_config.py`/`user_config.json`. It contains a bare Python dict (no docstring, no tests, unused).

**Question**: How should Unit 1 handle `io_paths.py`?
- A) Move it to `casper/utils/not_used/` (matches the existing project convention for deprecated/superseded code)
- B) Delete it outright
- C) Leave it in place, untouched (out of scope for this unit)
- D) Other (describe)

## Plan Steps

- [ ] Step 1: Resolve the `io_paths.py` question above (await user answer)
- [ ] Step 2: Business Logic Generation
  - [ ] 2a. Add NumPy-style module docstring to `config.py` (currently has only inline `#` comments, no docstrings — preserve all existing comments per the instructions file)
  - [ ] 2b. Review `user_config.py` for testability seams (Application Design Q5): the module currently does file I/O and env var resolution at **import time** (module-level side effects), making it hard to unit test in isolation. Refactor into explicit, testable functions where possible (e.g., ensure `_load_user_config()`, `_resolve_path()`, `_resolve_input_subpath()` are cleanly callable/mockable) — internal signatures may change (Q2 = A), but `USER_CONFIG`, `get()`, and all env var behavior must produce identical results
  - [ ] 2c. Apply/verify NumPy-style docstrings on all functions in `user_config.py` (some already exist — preserve and extend, do not remove)
  - [ ] 2d. Handle `io_paths.py` per Step 1 answer
- [ ] Step 3: Business Logic Unit Testing
  - [ ] 3a. Add `casper/tests/test_user_config.py`: test `_resolve_path()` fallback behavior (env var unset, env var set to valid path, env var set to invalid path), `_deep_merge()`, `get()` dot-path lookup, env var overrides for `CASPER_OUTPUT_DIR`/`CASPER_NPSAVE_DIR`
  - [ ] 3b. Add `casper/tests/interface/test_config.py`: verify `WAVE_BOUNDS`, `SYNTH_WAVE`, `id_start_wave`/`id_end_wave` derivation, `ARCHETYPE_PARAMS` structure, `KP_BOUNDS`/`SIDEBANDS` values match current constants exactly (regression-pin the existing constants so future edits can't silently change them)
- [ ] Step 4: Business Logic Summary — brief summary of what changed and why
- [ ] Step 5: Documentation Generation — README update only if user-visible behavior changed (expected: **no**, since this unit is internal-only per FR-5)
- [ ] Step 6: Regression Verification
  - [ ] 6a. Fast dev-loop check: run with `param_file_single_star.dat`, confirm no errors, sanity-check `Batch.set_io_paths()` behavior
  - [ ] 6b. Official gate: run full 4-star `param_file_test.dat`, diff all output files against `regression_baseline/unit0_baseline/` using the two-tier tolerance (Tier 1 exact, Tier 2 within 2σ)
- [ ] Step 7: Commit — one commit for this unit (per Units Generation Q4 = A), only after Step 6b passes

## Story/Requirement Traceability
Per `unit-of-work-story-map.md`: FR-2, FR-4, NFR-1, NFR-2, NFR-3 (first tests for a previously-untested module), NFR-4 (SECURITY-05 partial — config/path validation).
