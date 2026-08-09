# Application Design Plan

## Plan Steps

- [ ] Step 1: Analyze requirements.md and Reverse Engineering artifacts (architecture.md, code-structure.md, api-documentation.md) as design input
- [ ] Step 2: Ask clarifying questions below (component identification, methods, services, dependencies, design patterns)
- [ ] Step 3: Analyze answers for ambiguity; issue follow-ups if needed
- [ ] Step 4: Generate `aidlc-docs/inception/application-design/components.md`
- [ ] Step 5: Generate `aidlc-docs/inception/application-design/component-methods.md`
- [ ] Step 6: Generate `aidlc-docs/inception/application-design/services.md`
- [ ] Step 7: Generate `aidlc-docs/inception/application-design/component-dependency.md`
- [ ] Step 8: Generate `aidlc-docs/inception/application-design/application-design.md` (consolidated)
- [ ] Step 9: Present completion message, await approval

**Note (brownfield)**: Since this is a refactor-only cycle with no new components, "design" here means formally documenting the *existing* component/method/service boundaries (already reverse-engineered) at a level of rigor sufficient to safely guide Construction — not inventing new architecture.

---

## Clarifying Questions

### Question 1 — Component Identification
Reverse Engineering identified these logical components: `Batch` (orchestrator), `Spectrum` (domain model), GISIC normalization, temperature/carbon diagnostics, MCMC/archetype estimation engine, and reporting/plotting. Should Application Design use these same 6 groupings as its "components," or regroup differently?

A) Use the same 6 groupings — they already match natural responsibility boundaries

B) Regroup differently (describe after [Answer]: tag below)

C) Merge into fewer, broader components (e.g., "Pipeline Core" + "Reporting")

X) Other (please describe after [Answer]: tag below)

[Answer]: A

### Question 2 — Method Signature Stability
Should internal method signatures (parameters/return types) of `Batch` and `Spectrum` methods be allowed to change during refactoring, as long as the external CLI behavior and output files are unchanged?

A) Yes — internal method signatures may change freely; only CLI behavior and output files must stay stable

B) No — keep all existing method signatures unchanged too, even internally (safest, most conservative)

C) Case-by-case — signatures may change only within a single unit's own methods, not across unit boundaries

X) Other (please describe after [Answer]: tag below)

[Answer]: A

### Question 3 — Service/Orchestration Layer
Should `Batch` remain the single orchestrator class calling every pipeline stage in sequence (as today), or should orchestration be split into multiple smaller coordinators (e.g., one per unit)?

A) Keep `Batch` as the single orchestrator (matches current design, lowest risk)

B) Split orchestration into per-unit coordinator objects/functions, with `Batch` composing them

C) Not sure — decide during Construction based on how invasive each unit's refactor turns out to be

X) Other (please describe after [Answer]: tag below)

[Answer]: C

### Question 4 — Component Dependencies
Should this refactor be allowed to introduce new internal dependencies between components (e.g., extracting a shared utility module used by multiple units), or must the existing dependency graph be preserved exactly?

A) Allow new shared internal utility modules if they reduce duplication, as long as they don't change public behavior

B) Preserve the existing dependency graph exactly — no new inter-module dependencies

X) Other (please describe after [Answer]: tag below)

[Answer]: A

### Question 5 — Design Pattern Preference for Testability
Reverse Engineering flagged `batch.py`, `spectrum.py`, and `interface_main.py` as having zero test coverage, partly because they're tightly coupled to file I/O and pickled library loading. Should the refactor introduce lightweight seams (e.g., dependency injection for the spectral library loader, extracting pure functions from methods that currently mix I/O and computation) to make these modules independently testable?

A) Yes — introduce seams/dependency injection where needed to enable unit testing, as long as CLI behavior/output is unchanged

B) No — avoid structural seams; add tests only where possible without changing method structure

C) Not sure — let each unit's Code Generation step decide based on what's needed to add meaningful tests

X) Other (please describe after [Answer]: tag below)

[Answer]: A
