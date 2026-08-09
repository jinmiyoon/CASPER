# Component Inventory

## Application Packages
- `casper` (root package) — entry point (`main.py`), version, top-level init
- `casper.interface` — orchestration (`batch.py`), domain model (`spectrum.py`), scientific processing modules, plotting, configuration
- `casper.interface.gisic` — continuum normalization subpackage
- `casper.utils` — logging utilities

## Infrastructure Packages
- None — CASPER has no cloud/deployment infrastructure packages (CDK/Terraform/CloudFormation not present).

## Shared Packages
- `casper.interface.libraries` — Models/Data — pickled synthetic spectral grid, isochrone grid, and archetype reference grid (Git LFS), consumed by `synthetic_functions.py` and `interface_main.py`
- `casper.inputs` — Models/Fixtures — parameter file format documentation and example/test spectra used as default inputs

## Test Packages
- `casper.tests` — Unit — top-level test package init
- `casper.tests.interface` — Unit — tests for `ac.py`, `MAD.py`, `EW.py`, `temp_calibrations.py`, `MCMC_interface.py`, `synthetic_functions.py`
- `casper.tests.gisic` — Unit — tests for `norm_functions.py`, `normalize.py`, `segment.py`

## Legacy / Deprecated
- `casper.utils.not_used` — quarantined legacy implementations superseded by current modules (not imported by the active pipeline): `not_used_EW.py`, `not_used_GISIC_C_spectrum.py`, `not_used_MCMC_interface.py`, `not_used_io_functions.py`, `not_used_plots_functions.py`, `not_used_temp_calibrations.py`, `segment.py`, `carbon_veiling.ipynb`

## Total Count
- **Total Packages**: 6 active (`casper`, `casper.interface`, `casper.interface.gisic`, `casper.utils`, `casper.tests`, `casper.tests.interface`, `casper.tests.gisic`) + 1 legacy (`casper.utils.not_used`)
- **Application**: 2 (`casper`, `casper.interface` + subpackage `gisic`)
- **Infrastructure**: 0
- **Shared**: 2 (`libraries`, `inputs`)
- **Test**: 3 (`tests`, `tests.interface`, `tests.gisic`)
