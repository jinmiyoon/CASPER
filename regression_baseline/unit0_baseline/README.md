# Unit 0 Baseline (Post-Fix, Pre-Construction)

This is the **active regression baseline** for all Construction units (Units 1-6). It supersedes the raw `pre_refactor/` baseline because two approved, verified-safe fixes were applied before Construction formally began:

1. **Caching fix** (`casper/interface/synthetic_functions.py`): Added `functools.lru_cache` to `get_interp()`/`get_grav_interp()` — previously these re-read and unpickled a 125MB+62MB file from disk on every call (thousands of times per run, since called on every MCMC likelihood evaluation). Pure performance fix; cut a 4-star run from ~819s to ~597s.
2. **Append-mode bugfix** (`casper/interface/batch.py`): `_temp_cal_table.txt` and `_archetype_likelihood_table.txt` were opened in append mode (`"a"`), causing them to accumulate duplicate tables across repeated runs with the same output name. Fixed to overwrite mode (`"w"`).

## Verification methodology (two-tier)

### Tier 1 — Deterministic outputs (exact/tight-tolerance diff)
Applies to: `*_snr.csv`, `*_temp_cal_table.txt` (when `HARD_TEFF` is set, as for all 4 test stars), normalized flux/continuum.

These outputs do not depend on any random draw and were confirmed **bit-identical** across all runs (`sample`, `sample_cached`, `scatter1`, `scatter2`) regardless of code state. Any difference here is a real regression — must match exactly (or within floating-point tolerance for numeric noise from operation reordering, if any).

### Tier 2 — Stochastic (MCMC-derived) outputs (statistical tolerance)
Applies to: `*_out.csv` (TEFF/FEH/CFE/AC/LOGG + error columns), `*_archetype_likelihood_table.txt`, `*_spectra_output.csv` (synthetic flux), corner/trace/spec plots.

**Root cause**: No random seed exists anywhere in the codebase (`np.random.normal` for archetype temp draws, unseeded `emcee` walker initialization). Two runs of *identical, unmodified* code produce different results here — this is inherent to the current pipeline, not a bug introduced by refactoring.

**Empirical scatter characterization** (3 independent runs — `sample`, `scatter1`, `scatter2` — same fixed code, 4 test stars × 5 parameters = 20 data points, expressed as `max|diff| / avg_reported_error`):

| Parameter | Mean scatter | Max observed | 95th percentile |
|---|---|---|---|
| TEFF | 0.29σ | 0.64σ | 0.59σ |
| FEH | 0.34σ | 0.87σ | 0.79σ |
| CFE | 0.35σ | 0.81σ | 0.73σ |
| AC | 0.44σ | 0.89σ | 0.86σ |
| LOGG | 0.12σ | 0.32σ | 0.28σ |

**Approved tolerance criterion**: A Construction run's point estimate for any MCMC-derived parameter must fall within **2σ of the reported error** relative to this Unit 0 baseline. This gives ~2x margin above the observed natural max (0.89σ), catching genuine regressions while not false-alarming on expected stochastic noise.

## Files in this directory
Reference outputs from the `scatter2` run (fixed code: caching + append-mode fix), used as the point-of-comparison for the 2σ tolerance check:
- `scatter2_out.csv` — stellar parameters (Tier 2)
- `scatter2_snr.csv` — S/N summary (Tier 1)
- `scatter2_temp_cal_table.txt` — photometric temperature calibration (Tier 1)
- `scatter2_archetype_likelihood_table.txt` — archetype classification (Tier 2)
- `scatter2_spectra_output.csv` — observed + synthetic spectra (Tier 2)
- `scatter2parameters_output.npy` / `scatter2spectra_output.npy` — binary equivalents

## Fast-iteration dataset (dev-loop only, not the official gate)
For quick sanity checks during active development of a unit, use `casper/inputs/params/param_file_single_star.dat` (1 star: G77-61, DWARF/CH mode, CSV format) instead of the full 4-star `param_file_test.dat` — cuts run time to roughly 1/4. This is **not** a substitute for the official per-unit regression gate, which per requirements.md NFR-1 (Q4 = A) still requires the full 4-star `param_file_test.dat` before a unit is marked complete.
