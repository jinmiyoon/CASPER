---
description: "Use when writing, editing, or refactoring Python code in CASPER (casper/). Covers NumPy/astropy docstring style, preserving existing docstrings/comments during refactors, README update policy, and the requirement that refactors must not change scientific outputs."
applyTo: "casper/**/*.py"
---

# CASPER Python Coding Conventions

## Docstring Style (NumPy/astropy)

All functions, methods, and classes must use NumPy/astropy-style docstrings — matching the existing convention in modules like `casper/interface/batch.py`.

```python
def foo(x: float, y: float = 0.0) -> float:
    """Short one-line summary.

    Parameters
    ----------
    x : float
        Description of x.
    y : float, optional
        Description of y, by default 0.0.

    Returns
    -------
    float
        Description of the return value.

    Raises
    ------
    ValueError
        When x is negative.
    """
```

- Sections used as needed: `Parameters`, `Returns`, `Raises`, `Notes`, `Examples` — each underlined with `----------`/`------`.
- One-line summary first, blank line, optional extended description before `Parameters`.
- Do not use Google-style (`Args:`/`Returns:`) or reST `:param:` docstrings.

## Preserving Existing Docstrings/Comments

When refactoring a function or method that already has a docstring or inline comments:

- Update the docstring/comments to reflect what the code actually does after the change.
- Do not completely remove an existing docstring or comment unless there is a strong reason (e.g. it documents dead code being deleted, or it is factually wrong and superseded by the new docstring).
- If there is a strong reason to remove one, ask the user for confirmation before removing it — do not remove silently.

## Refactoring Safety

Refactors (renaming internals, restructuring functions/classes, changing control flow, splitting modules) must **not** change CASPER's scientific outputs.

- Before refactoring, generate baseline outputs from a run on the repo's test spectra (`*_out.csv`, `*_snr.csv`, `*_temp_cal_table.txt`, `*_spectra_output.csv`, `*_archetype_likelihood_table.txt`).
- After refactoring, re-run the same inputs (same RNG seeds/config) and numerically diff against the baseline within floating-point tolerance.
- Passing unit tests alone is not sufficient proof — outputs must be directly compared.
- If an output value is intended to change, state that explicitly and get confirmation before proceeding; never let a refactor silently alter results.

## README Update Policy

- README updates are mandatory for any user-visible change.
- README updates are optional for internal-only refactors.
