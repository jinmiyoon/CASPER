# Requirements: CASPER Refactoring Cycle (AI-DLC Hybrid Rollout)

## Intent Analysis Summary

- **User Request**: Adopt AI-DLC methodology for CASPER to enable safe refactoring — ensure scientific output does not change unexpectedly, enforce NumPy/astropy docstring conventions, preserve existing documentation during refactors, and apply a considered README-update policy.
- **Request Type**: Refactoring (maintainability/quality improvement of existing pipeline)
- **Initial Scope Estimate**: Multiple Components — spans the full `casper.interface` pipeline (batch orchestration, spectrum model, GISIC normalization, calibration, MCMC/classification, plotting) plus supporting test/CI infrastructure
- **Initial Complexity Estimate**: Moderate-to-Complex — no new business logic, but high scientific-risk surface (Bayesian MCMC fitting, multi-stage numerical pipeline) where subtle regressions are easy to introduce and hard to detect without rigorous output comparison
- **Rollout Mode**: Hybrid — full-rigor Inception (this document + Reverse Engineering already complete), lightweight per-unit Construction governed by [.github/instructions/casper-python-style.instructions.md](../../../.github/instructions/casper-python-style.instructions.md), comprehensive Build & Test

## Functional Requirements

1. **FR-1 — No New Features This Cycle**: This AI-DLC cycle is scoped to refactoring and maintainability only. No new scientific analysis features, calibrations, or output types are in scope. (Resolved via clarification: user selected "not sure yet, treat as refactor-only for now.")
2. **FR-2 — Preserve Pipeline Behavior**: Every existing `Batch` pipeline stage (I/O, spectra loading, RV correction, normalization, S/N estimation, temperature calibration, archetype classification, MCMC determination, log g estimation, synthetic spectrum generation, output/plot generation) must continue to produce the same results after refactoring.
3. **FR-3 — Preserve Public Behavior of Legacy/Deprecated Code**: `casper/utils/not_used/` remains out of scope for functional refactor (no behavior expectations) but may be referenced as an oracle for property-based/regression comparison during related module refactors (see NFR-6).
4. **FR-4 — Documentation Consistency**: All refactored functions/methods/classes must carry NumPy/astropy-style docstrings; existing docstrings/comments are updated, not silently deleted (per existing instructions file).
5. **FR-5 — README Currency**: README updates are mandatory only if a change is user-visible (e.g., CLI usage, config format, install steps); optional for internal-only refactors.

## Non-Functional Requirements

### NFR-1 — Scientific Output Equivalence (Regression Safety)
- **Requirement**: Refactors must not change scientific output values for existing workflows unless explicitly approved in advance (clarified: Q2 = B).
- **Baseline dataset**: `casper/inputs/spectra/test_spectra/` with its corresponding parameter file (Q4 = A) is the mandatory baseline regression dataset for the official per-unit gate. A single-star dataset (`casper/inputs/params/param_file_single_star.dat`) may be used for fast dev-loop iteration only — it is not a substitute for the official gate.
- **Baseline revision (Unit 0)**: Before Construction began, two verified-safe fixes were applied and adopted as the active baseline (see `regression_baseline/unit0_baseline/README.md`): (1) `functools.lru_cache` on `synthetic_functions.get_interp()`/`get_grav_interp()` (pure performance fix — eliminated redundant re-reads of a 125MB+62MB pickle file on every MCMC likelihood evaluation), (2) fixed `batch.py` opening `_temp_cal_table.txt`/`_archetype_likelihood_table.txt` in append mode instead of overwrite mode (pre-existing bug causing duplicate tables to accumulate across runs). Both were verified to not change deterministic outputs before being adopted.
- **Two-tier tolerance policy** (revised from the original single floating-point-tolerance policy, based on empirical findings — see below):
  - **Tier 1 (deterministic outputs)**: `*_snr.csv`, `*_temp_cal_table.txt` (when `HARD_TEFF` is set, as for all current test stars), normalized flux/continuum. These must match exactly (or within tight floating-point tolerance) — confirmed reproducible bit-for-bit across repeated runs of the same code.
  - **Tier 2 (stochastic/MCMC-derived outputs)**: `*_out.csv` (TEFF/FEH/CFE/AC/LOGG + errors), `*_archetype_likelihood_table.txt`, `*_spectra_output.csv`, corner/trace/spec plots. **Root cause discovered**: no random seed exists anywhere in the codebase (`np.random.normal` for archetype temp draws, unseeded `emcee` walker initialization) — two runs of *identical, unmodified* code produce different results here. This is inherent to the pipeline, not a refactor-introduced bug. Empirical scatter characterization (3 independent runs, 20 data points across 4 stars × 5 parameters) found natural scatter never exceeds ~0.9σ of the reported error bar. **Approved criterion**: a Construction run's point estimate must fall within **2σ of the reported error** relative to the Unit 0 baseline.
- **Verification method**: Automated numeric diff of `*_out.csv`, `*_snr.csv`, `*_temp_cal_table.txt`, `*_archetype_likelihood_table.txt`, `*_spectra_output.csv` against the Unit 0 baseline, using the tier-appropriate tolerance above. Passing unit tests alone is not sufficient proof.

### NFR-2 — Code Style Consistency
- NumPy/astropy docstring style enforced for all touched functions/methods/classes (per `.github/instructions/casper-python-style.instructions.md`).

### NFR-3 — Refactor Prioritization / Risk Ordering
- Construction should prioritize the highest-risk core pipeline modules first — `batch.py`, `interface_main.py`, `spectrum.py` — before lower-risk, already-tested modules (Q8 = A). This aligns with the Reverse Engineering finding that these three modules currently have **no unit test coverage**.

### NFR-4 — Security Baseline (Scoped)
Enabled per user decision (Q5 = A), scoped to what's applicable to a local, non-networked scientific package. Most Baseline Security rules are **N/A** (no deployed service, no network endpoints, no auth, no data store) — marked N/A with rationale, not blocking:

| Rule | Applicability to CASPER | Status |
|---|---|---|
| SECURITY-01 (encryption at rest/in transit) | No data store/network | N/A |
| SECURITY-02 (access logging on network intermediaries) | No network intermediaries | N/A |
| SECURITY-03 (application-level logging) | **Applicable** — CASPER has `logger_config.py`; verify no sensitive data (file paths are not sensitive here) logged, structured format maintained | Applicable |
| SECURITY-04 (HTTP security headers) | No web endpoints | N/A |
| SECURITY-05 (API input validation) | No API endpoints; however `user_config.json` parsing should still validate expected types/paths | Partially applicable |
| SECURITY-06 (least-privilege IAM) | No cloud IAM | N/A |
| SECURITY-07 (network configuration) | No network infra | N/A |
| SECURITY-08 (app-level access control) | No multi-user auth model | N/A |
| SECURITY-09 (hardening/misconfiguration) | Local tool; still worth avoiding leaking internal paths in error messages | Partially applicable |
| SECURITY-10 (software supply chain) | **Applicable** — `pyproject.toml` already pins exact versions; no vulnerability scanning currently configured in CI; no SBOM generation | Applicable — actionable gap |
| SECURITY-11 (secure design principles) | No auth/payment logic | N/A |
| SECURITY-12 (auth/credential management) | No authentication system | N/A |
| SECURITY-13 (integrity verification) | CI/CD pipeline integrity (GitHub Actions) worth reviewing; no untrusted deserialization identified | Partially applicable |
| SECURITY-14/15 (alerting/monitoring, ops) | No production monitoring target for a local script | N/A |

**Actionable items for Build & Test**: add a dependency vulnerability scan step to CI (SECURITY-10), review `user_config.py` path/type validation (SECURITY-05 partial), confirm no internal paths/stack traces leak in user-facing CLI error output (SECURITY-09 partial).

### NFR-5 — Resiliency Baseline: Not Enabled
- Declined per user decision (Q7 clarification = A / No). Rationale: CASPER has no deployed service, SLA, DR target, or CI/CD deployment pipeline — the extension's RTO/RPO, change-management, and multi-region questions do not map to a local batch-processing scientific script.

### NFR-6 — Property-Based Testing (Full Enforcement)
Enabled per user decision (Q6 = A, full enforcement). Applicable opportunities identified during Reverse Engineering:

| PBT Category | Candidate(s) | Notes |
|---|---|---|
| Round-trip (PBT-02) | `ac.py`: `ac(cfe, feh)` ↔ `cfe(ac, feh)` | Exact mathematical inverses — ideal round-trip property test. **Descoped for Unit 4** (user decision, 2026-09-07): a Hypothesis round-trip test was implemented and passing, then replaced by the user with expanded example-based tests (added single-element/constant-array-style edge cases) instead of property-based coverage. |
| Invariant (PBT-03) | `MAD.py`: `MAD`/`S_MAD` non-negativity; GISIC `normalize()`: normalized flux stays within documented bounds; `temp_calibrations.py`: Teff within calibration's valid color range | GISIC `normalize()` bounds invariant delivered via Hypothesis in Unit 3. **`MAD`/`S_MAD` non-negativity descoped for Unit 4** (user decision, 2026-09-07): same as above — replaced by example-based edge-case tests instead of a Hypothesis invariant test. |
| Oracle (PBT-05) | Refactored implementations vs. current (pre-refactor) implementations of the same function, and — where relevant — vs. the deprecated `casper/utils/not_used/` originals | Directly supports NFR-1 output-equivalence goal |
| Framework (PBT-09) | Hypothesis (Python) | To be added as a `test` extra dependency |
| Complementary strategy (PBT-10) | Existing example-based tests in `casper/tests/` remain; PBT is additive | |

Property identification (PBT-01) and stateful/generator-quality rules (PBT-04, PBT-06, PBT-07, PBT-08) will be assessed per-unit during Construction as each module is refactored, since this is a Code-Generation-stage (per-unit) activity in the lightweight Construction flow.

## Summary

This cycle is a **refactor-only, output-preserving maintainability effort** across CASPER's scientific pipeline, prioritizing the three currently-untested, highest-risk modules first. Success is defined by: (1) zero unintended numerical output changes against the repository's test-spectra baseline, (2) consistent NumPy/astropy documentation, (3) new property-based tests (Hypothesis) added alongside existing unit tests for round-trip/invariant/oracle-style scientific functions, and (4) a scoped security review focused on dependency supply-chain scanning and config validation — with cloud/resiliency-oriented practices explicitly out of scope given CASPER's local, non-deployed nature.
