# Requirements Analysis — Clarification Questions

Your answers to Questions 1 and 7 need follow-up before requirements.md can be finalized. Please answer both below.

---

## Ambiguity 1: "Mixed goal" (Question 1)

You selected **C) Mixed goal: refactor some modules and add selected new features** for the primary goal of this AI-DLC cycle.

This is ambiguous because "mix of X and Y" doesn't tell us which specific new features are in scope, and it also creates tension with Question 2 (Answer: B — output changes for *existing* workflows require explicit approval in advance). Everything in this conversation so far (the docstring/output-equivalence rules, the Reverse Engineering findings, the whole rationale for choosing AI-DLC) has been framed around **safe refactoring of the existing pipeline** — no new scientific feature has been named yet.

### Clarification Question 1
For this specific AI-DLC cycle, what new feature(s) — if any — should be included alongside the refactoring work?

A) None right now — keep this cycle scoped to refactoring/maintainability only; new features become a separate future cycle

B) One or more specific new features ARE in scope for this cycle (describe them after [Answer]: tag below)

C) Not sure yet — decide after refactoring is underway, treat as refactor-only for now and revisit later

X) Other (please describe after [Answer]: tag below)

[Answer]: C

---

## Ambiguity 2: Resiliency Baseline (Question 7)

You asked for a summary before deciding. Here it is:

**What the Resiliency Baseline extension actually covers**: It's a set of AWS Well-Architected Reliability Pillar practices across 6 areas — Business Goals (criticality classification, RTO/RPO recovery targets), Change Management & Automation (CI/CD tooling, rollback/deployment strategy), Integrated Observability, High Availability (multi-zone/multi-region topology), Disaster Recovery, and Continuous Improvement (incident response, resiliency testing).

**Why it's very unlikely to apply to CASPER**:
- CASPER is a **local, single-machine batch script** (`python main.py`) — there is no deployed service, no uptime SLA, no production environment, no CI/CD deployment pipeline, no multi-region/multi-zone infrastructure.
- The extension's clarifying questions ask things like "What's your RTO/RPO?", "What's your DR strategy (Backup & Restore → Multi-site Active/Active)?", "What CI/CD tool deploys this workload?" — none of these have a meaningful answer for a scientific analysis script run on a researcher's own machine.
- Enabling it would mean documenting N/A justifications for nearly every rule (RTO/RPO targets, DR strategy, blue-green deployment, etc.) without any of it changing what code actually gets written — pure process overhead.

**My recommendation**: **No** — this doesn't fit CASPER's deployment model. If CASPER later becomes a hosted/deployed service (e.g. a web API), it would be worth revisiting then.

### Clarification Question 2
Given the summary above, should the Resiliency Baseline extension be enabled for this project?

A) No — skip the resiliency baseline (recommended given CASPER has no deployment/production environment)

B) Yes — apply it anyway as directional guidance, even though most rules will be marked N/A

X) Other (please describe after [Answer]: tag below)

[Answer]: A
