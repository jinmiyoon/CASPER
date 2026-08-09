# Component Dependency

## Dependency Policy (Application Design Q4, Answer: A)
New shared internal utility modules **are allowed** if they reduce duplication, provided they do not change any component's public/observable behavior. New dependencies must still respect the direction of the existing dependency graph below (no new reverse/circular dependencies).

## Dependency Matrix

| Component | Depends On | Depended On By |
|---|---|---|
| Batch Orchestrator | Spectrum Domain Model, GISIC Normalization, Temperature & Carbon Diagnostics, Parameter Estimation Engine, Reporting, `user_config`, `config`, `logger_config` | `main.py` |
| Spectrum Domain Model | (none — leaf domain object; uses numpy/pandas/astropy only) | Batch Orchestrator, Parameter Estimation Engine |
| GISIC Normalization | (none internal — uses scipy/numpy only) | Batch Orchestrator, Parameter Estimation Engine (`synthetic_functions.normalize_synth_spectrum`) |
| Temperature & Carbon Diagnostics | (EW.py depends on MAD.py; temp_calibrations.py depends on ac.py for downstream A(C) conversion) | Batch Orchestrator |
| Parameter Estimation Engine | GISIC Normalization, Spectrum Domain Model, pickled data grids (`libraries/*.pkl`) | Batch Orchestrator |
| Reporting | Spectrum Domain Model (reads processed state), Batch Orchestrator (invoked by it) | `main.py` (indirectly via Batch) |

## Communication Patterns
- **In-process function/method calls only** — no message queues, events, or network calls between components.
- **Shared mutable state via `Spectrum` object**: components communicate primarily by reading/writing attributes on a shared `Spectrum` instance (e.g., `frame`, `SN_DICT`, `MCMC_COARSE`) rather than passing immutable data structures. This is a **pre-existing pattern** the refactor does not need to change (Q2 = internal signatures may change, but no requirement to eliminate shared-state style), though a unit's Code Generation step may reduce implicit coupling if it doesn't affect observable behavior.
- **Batch → component**: batch methods loop over `self.spectra_array` and call a component function/method per `Spectrum`.
- **Component → external data**: Parameter Estimation Engine reads pickled interpolator grids from `casper/interface/libraries/` (Git LFS) — read-only, not part of this refactor.

## Data Flow (reference)
See [../reverse-engineering/architecture.md](../reverse-engineering/architecture.md) for the full sequence diagram of data flow through the pipeline — this refactor does not change that flow, only the internal implementation of each stage.
