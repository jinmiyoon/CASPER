# Application Design (Consolidated)

This document consolidates the Application Design artifacts for the CASPER refactor cycle. See individual files for full detail: [components.md](components.md), [component-methods.md](component-methods.md), [services.md](services.md), [component-dependency.md](component-dependency.md).

## Design Decisions (from application-design-plan.md)

| # | Question | Answer | Implication |
|---|---|---|---|
| 1 | Component grouping | A — use the same 6 groupings from Reverse Engineering | Components: Batch Orchestrator, Spectrum Domain Model, GISIC Normalization, Temperature & Carbon Diagnostics, Parameter Estimation Engine, Reporting |
| 2 | Method signature stability | A — internal signatures may change freely; only CLI behavior/output files must stay stable | Frees Code Generation to restructure method internals as needed for testability, as long as NFR-1 (output equivalence) holds |
| 3 | Orchestration layer | C — decide per unit during Construction, based on invasiveness | `Batch` remains the default single orchestrator; a unit may extract a coordinator only if it stays within one component boundary and `Batch`'s public method still exists with the same effect |
| 4 | Component dependencies | A — new shared internal utilities allowed if they reduce duplication and don't change public behavior | Refactor may introduce e.g. a shared validation helper, but must not add reverse/circular dependencies to the matrix in component-dependency.md |
| 5 | Testability design patterns | A — introduce seams/dependency injection where needed | Explicitly authorizes adding dependency injection (e.g., for the pickled library loader) and extracting pure functions from I/O-mixed methods in `batch.py`, `spectrum.py`, `interface_main.py` to enable their first unit tests |

## Component Summary

| Component | Test Status | Refactor Priority (per requirements.md NFR-3) |
|---|---|---|
| Batch Orchestrator | Untested | Critical |
| Spectrum Domain Model | Untested | Critical |
| Parameter Estimation Engine (`interface_main.py` portion) | Untested | Critical |
| GISIC Normalization | Tested | Important |
| Temperature & Carbon Diagnostics | Tested | Important |
| Reporting | Untested | Optional |

## Services Summary
Two existing orchestration layers are formalized, not replaced: `Batch` (pipeline-level) and the Parameter Estimation Engine's internal MCMC/classification sequencing (spectrum-level, within `interface_main.py`). No new top-level services are introduced.

## Dependency Summary
The existing dependency graph (Batch → all components; Parameter Estimation Engine → GISIC + Spectrum + pickled grids) is preserved. New shared internal utilities are permitted only if they don't introduce new reverse dependencies and don't change observable behavior.

## Relationship to Units Generation
This design (6 components, dependency graph, and the per-unit orchestration-decision deferral) directly maps onto the 6-unit sequence already proposed in [../plans/execution-plan.md](../plans/execution-plan.md) and will be formalized next in Units Generation.
