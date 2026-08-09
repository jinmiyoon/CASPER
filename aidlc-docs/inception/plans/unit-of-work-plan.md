# Unit of Work Plan

## Plan Steps

- [ ] Step 1: Confirm unit boundaries (using components.md + execution-plan.md as the starting basis)
- [ ] Step 2: Ask clarifying questions below (grouping confirmation, dependency/testing checkpoints, team alignment, requirement mapping)
- [ ] Step 3: Analyze answers for ambiguity; issue follow-ups if needed
- [ ] Step 4: Generate `aidlc-docs/inception/application-design/unit-of-work.md`
- [ ] Step 5: Generate `aidlc-docs/inception/application-design/unit-of-work-dependency.md`
- [ ] Step 6: Generate `aidlc-docs/inception/application-design/unit-of-work-story-map.md` (requirement-map, since User Stories was skipped)
- [ ] Step 7: Validate unit boundaries and dependencies
- [ ] Step 8: Present completion message, await approval

**Note (brownfield, no User Stories)**: Since User Stories was skipped for this refactor-only cycle, `unit-of-work-story-map.md` will map **requirements.md functional/non-functional requirements** to units instead of user stories.

---

## Clarifying Questions

### Question 1 — Unit Boundary Confirmation
`execution-plan.md` and `components.md` already propose 6 units: (1) I/O & Configuration, (2) Spectrum Loading & Preprocessing, (3) GISIC Normalization, (4) Temperature Calibration & Extinction, (5) Archetype Classification & MCMC, (6) Output Generation & Plotting. Should Units Generation finalize these as-is?

A) Yes — finalize these 6 units as-is

B) Adjust the grouping (describe after [Answer]: tag below)

C) Split further (e.g., separate `batch.py` orchestration changes into their own cross-cutting unit rather than folding them into each of the 6)

X) Other (please describe after [Answer]: tag below)

[Answer]: A

### Question 2 — Testing Checkpoint / Update Strategy
Should each unit be fully completed (code + tests + regression diff passing) before starting the next unit, or can multiple units be worked on in parallel with a final integrated regression pass at the end?

A) Strictly sequential — complete and verify each unit (including its regression diff) before starting the next

B) Allow parallel work across units, with regression diffing only at the very end

C) Sequential for the 3 highest-risk units (Spectrum, MCMC/Archetype, and cross-cutting Batch integration), parallel-allowed for the 3 lower-risk units

X) Other (please describe after [Answer]: tag below)

[Answer]: A

### Question 3 — Team Alignment
The README lists 3 contributors (Whitten, Yoon, Webb). Will this refactor be done solely by you working with the AI assistant, or could units be split across multiple contributors working in parallel branches?

A) Solo — just me and the AI assistant, one unit at a time

B) Multiple contributors may pick up different units in parallel (describe coordination expectations after [Answer]: tag below)

X) Other (please describe after [Answer]: tag below)

[Answer]: A

### Question 4 — Rollback/Checkpoint Granularity
Should each unit correspond to its own git commit (or small set of commits) so it can be reviewed/reverted independently, or is a single combined commit/PR at the end of Construction acceptable?

A) One commit (or small commit set) per unit — independently reviewable/revertible

B) Single combined commit/PR at the end of all Construction work

C) Not sure — decide per unit based on how large the diff turns out to be

X) Other (please describe after [Answer]: tag below)

[Answer]: A
