# Unit of Work — Requirement Map

**Note**: User Stories was skipped for this refactor-only cycle (see requirements.md, Requirements Analysis Q1). This document maps [requirements.md](../requirements/requirements.md) functional (FR) and non-functional (NFR) requirements to units, in place of a story map.

| Requirement | Applies To |
|---|---|
| FR-1 — No New Features This Cycle | All units (scope boundary, not unit-specific) |
| FR-2 — Preserve Pipeline Behavior | All units (each unit's regression diff verifies this for its own scope) |
| FR-3 — Preserve Legacy/Deprecated Code Behavior | Unit 5 (may reference `casper/utils/not_used/` originals as an oracle for MCMC/classification-related functions) |
| FR-4 — Documentation Consistency (NumPy/astropy docstrings) | All units |
| FR-5 — README Currency | All units (evaluated per-unit: mandatory only if that unit's change is user-visible) |
| NFR-1 — Scientific Output Equivalence | All units — each unit's completion gate (per unit-of-work-dependency.md) requires a passing regression diff |
| NFR-2 — Code Style Consistency | All units |
| NFR-3 — Refactor Prioritization / Risk Ordering | Units 2 and 5 specifically (first tests for `spectrum.py` and `interface_main.py`); reflected in the unit sequence itself |
| NFR-4 — Security Baseline (scoped) | Unit 1 (config/path validation, SECURITY-05 partial) and Build & Test phase (SECURITY-10 dependency scanning, applies repo-wide) |
| NFR-5 — Resiliency Baseline: Not Enabled | N/A — no unit affected |
| NFR-6 — Property-Based Testing | Unit 4 (`ac()`/`cfe()` round-trip, MAD invariants — **descoped per user decision 2026-09-07**, see requirements.md NFR-6; replaced with expanded example-based tests instead) and Unit 3 (GISIC normalization bounds invariant — delivered via Hypothesis); oracle-style tests especially relevant to Unit 5 (refactored vs. pre-refactor MCMC/likelihood functions) |

## Coverage Check
All 5 functional requirements and 6 non-functional requirements from requirements.md are mapped to at least one unit or explicitly marked N/A (NFR-5). No requirement is left unassigned.
