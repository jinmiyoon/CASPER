# Component Methods

Per Application Design Q2 (Answer: A), internal method signatures may change freely during refactoring — only external CLI behavior and output file contents/formats must remain stable. The signatures below reflect **current** behavior (from Reverse Engineering) as the baseline; Code Generation may restructure them within a unit as long as observable behavior is preserved.

## Batch Orchestrator (`batch.py`)

| Method | Current Signature | High-Level Purpose |
|---|---|---|
| `__init__` | `(io_paths: dict[str, str] \| str) -> None` | Initialize with I/O configuration |
| `set_io_paths` | `() -> None` | Resolve param/spectra/output paths |
| `load_params` | `() -> None` | Read parameter CSV into `self.param_file` |
| `load_spectra` | `() -> None` | Load FITS/CSV spectra into `self.spectra_array` |
| `set_params` | `() -> None` | Assign parameter-file row values onto each `Spectrum` |
| `radial_correct` | `() -> None` | Apply per-star RV correction |
| `build_frames` | `(bounds: tuple = config.WAVE_BOUNDS) -> None` | Build/trim per-star wave/flux frame |
| `normalize` | `(default: bool = True) -> None` | GISIC multi-sigma continuum normalization |
| `set_KP_bounds` | `() -> None` | Determine Ca II K integration bounds per star |
| `set_carbon_mode` | `() -> None` | Classify CH vs CH+C2 carbon mode per star |
| `estimate_sn` | `() -> None` | Compute S/N and XI per band |
| `get_sn` | `() -> None` | Write `*_snr.csv` |
| `ebv_correction` | `() -> None` | Apply E(B-V) reddening correction |
| `calibrate_temperatures` | `(default: bool = True, teff_sigma: int = 250) -> None` | Compute/adopt photometric Teff; write `*_temp_cal_table.txt` |
| `archetype_classification` | `() -> None` | Classify CEMP archetype group; write `*_archetype_likelihood_table.txt` |
| `mcmc_determination` | `() -> None` | Run coarse + refine MCMC for all stars |
| `estimate_logg` | `() -> None` | Interpolate log g from Teff/[Fe/H] |
| `generate_synthetic` | `() -> None` | Generate best-fit synthetic spectra |
| `generate_output_spectra` | `() -> None` | Write `*_spectra_output.csv` |
| `generate_output_files` | `() -> None` | Write `*_out.csv` |
| `generate_plots` | `() -> None` | Generate spectral/corner/trace PDFs |

## Spectrum Domain Model (`spectrum.py`)

| Method | Current Signature (approximate) | High-Level Purpose |
|---|---|---|
| `__init__` | `(spec, filename: str, is_fits: bool) -> None` | Load from FITS HDU or CSV |
| `radial_correction` | `(velocity_km_s: float) -> None` | Doppler-shift wavelength |
| `ebv_correct` | `(row) -> None` | Apply SFD reddening correction to colors |
| `set_frame` / `trim_frame` | `(wave, flux) -> None` / `(bounds: tuple) -> None` | Build/trim working DataFrame |
| `estimate_sn` | `() -> None` | Compute S/N and XI from sidebands |
| `get_sn` | `() -> DataFrame` | Return S/N summary as single-row DataFrame |
| `set_params` | `(**kwargs) -> None` | Assign stellar parameters from param file row |
| `set_KP_bounds` / `set_carbon_mode` | `() -> None` | Configure line bounds / carbon mode |
| `set_mcmc_args` | `() -> None` | Prepare spectral regions and kwargs for MCMC |
| `get_mcmc_dict` | `(mode: str) -> dict` | Retrieve MCMC results (COARSE/REFINE) |
| `get_output_row` / `get_spectra_row` | `() -> DataFrame` | Format output rows |
| Getters | `get_wave/get_flux/get_frame_*/get_filename/get_sequence/get_starname/get_gravity_class/get_rv/get_carbon_mode` | Accessors |

## GISIC Normalization (`gisic/normalize.py`)

| Function | Current Signature | High-Level Purpose |
|---|---|---|
| `normalize` | `(wave, flux, sigma, k, cahk, band_check, flux_min, boost, return_points=False) -> (wave, norm, cont[, points])` | Top-level continuum normalization entry point |

## Temperature & Carbon Diagnostics

| Function | Module | Current Signature | Purpose |
|---|---|---|---|
| `Hernandez` / `Casagrande` / `Fukugita` / `Bergeat` | `temp_calibrations.py` | `(color, feh, class_) -> teff` | Photometric Teff calibrations |
| `GBAND_QUAD` | `EW.py` | `(wave, flux, bounds=[4222,4322]) -> (EW, correction)` | CH G-band equivalent width |
| `CAII_K6` | `EW.py` | `(wave, flux) -> EW` | Ca II K-line equivalent width |
| `get_KP_band` | `EW.py` | `(spectrum) -> bounds` | Determine KP integration bounds |
| `set_CH_procedure` | `EW.py` | `(spectrum) -> None` | Classify carbon mode |
| `ac` / `cfe` | `ac.py` | `(cfe, feh) -> ac` / `(ac, feh) -> cfe` | A(C) ↔ [C/Fe] conversion (round-trip pair) |
| `MAD` / `S_MAD` | `MAD.py` | `(array) -> float` | Robust dispersion statistics |

## Parameter Estimation Engine

| Function | Module | Current Signature | Purpose |
|---|---|---|---|
| `kde_param` | `MCMC_interface.py` | `(distribution, x0) -> {result, kde}` | KDE peak-finding on posterior chain |
| `chi_likelihood` / `chi_likelihood_C2` | `MCMC_interface.py` | `(theta, observed_spec_regions, synth_wave, photo_teff, photo_teff_unc, SN_DICT, G_CLASS) -> float` | Coarse-stage log-likelihood |
| `chi_ll_refine` / `chi_ll_refine_C2` | `MCMC_interface.py` | `(theta, observed_spec_regions, synth_wave, PARAMS, G_CLASS) -> float` | Refine-stage log-likelihood |
| `teff_lnprior` / `sigma_lnprior` / `default_param_edges` | `MLE_priors.py` | varies | Prior probability / bounds checking |
| `get_interp` / `get_grav_interp` | `synthetic_functions.py` | `() -> interpolator` | Load pickled spectral/isochrone interpolators |
| `normalize_synth_spectrum` | `synthetic_functions.py` | `(synth_wave, synth_flux) -> norm_flux` | GISIC-normalize synthetic spectrum |
| `ln_chi_square_sigma` / `CAII_CH_CHI_LH` | `synthetic_functions.py` | varies | Chi-squared likelihood helpers |
| `archetype_classify_MC` | `interface_main.py` | `(spectrum) -> None` (mutates `spectrum.LL_DICT`) | Monte Carlo archetype classification |
| `mcmc_determination` | `interface_main.py` | `(spectrum, mode: str) -> None` | Run one MCMC stage |
| `generate_kde_params` | `interface_main.py` | `(spectrum, mode: str) -> None` | KDE-smooth posterior chains |
| `estimate_logg` | `interface_main.py` | `(spectrum) -> None` | Interpolate surface gravity |
| `generate_synthetic` | `interface_main.py` | `(spectrum) -> None` | Generate best-fit synthetic spectrum |

## Reporting (`plot_functions.py`)

| Function | Current Signature | Purpose |
|---|---|---|
| `produce_title` | `(spectrum) -> str` | Format plot title |
| `plot_spectra` | `(spectra_batch) -> None` | Multi-page spectral comparison PDF |
| `plot_corner_array` | `(batch) -> None` | MCMC corner plots |
| `plot_mcmc_trace_array` | `(batch) -> None` | MCMC trace plots |

**Note**: Detailed business-rule-level design (e.g., exact chi-squared formula changes, exact prior tuning) is explicitly deferred to Functional Design — which is **skipped** in this lightweight Construction flow per the approved execution plan. Any such detail needed will be handled directly during Code Generation for the relevant unit.
