# API Documentation

CASPER exposes no REST/network API — it is a local Python library/script. This document covers its **internal APIs**: the primary classes and functions that make up its programmatic surface.

## REST APIs
Not applicable — no HTTP endpoints exist in this codebase.

## Internal APIs

### `Batch` (`casper/interface/batch.py`)

| Method | Parameters | Return | Purpose |
|---|---|---|---|
| `__init__` | `io_paths: dict[str, str] \| str` | `None` | Initialize with I/O configuration |
| `set_io_paths()` | — | `None` | Resolve param/spectra/output paths |
| `load_params()` | — | `None` | Read parameter CSV into `self.param_file` |
| `load_spectra()` | — | `None` | Load FITS/CSV spectra into `self.spectra_array` |
| `set_params()` | — | `None` | Assign parameter-file row values onto each `Spectrum` |
| `radial_correct()` | — | `None` | Apply per-star RV correction |
| `build_frames(bounds=config.WAVE_BOUNDS)` | `bounds: tuple` | `None` | Build/trim per-star wave/flux frame |
| `normalize(default=True)` | `default: bool` | `None` | GISIC multi-sigma continuum normalization |
| `set_KP_bounds()` | — | `None` | Determine Ca II K integration bounds per star |
| `set_carbon_mode()` | — | `None` | Classify CH vs CH+C2 carbon mode per star |
| `estimate_sn()` | — | `None` | Compute S/N and XI (inverse S/N) per band |
| `get_sn()` | — | `None` | Write `*_snr.csv` |
| `ebv_correction()` | — | `None` | Apply E(B-V) reddening correction to photometry |
| `calibrate_temperatures(default=True, teff_sigma=250)` | `default: bool`, `teff_sigma: int` | `None` | Compute/adopt photometric Teff; write `*_temp_cal_table.txt` |
| `archetype_classification()` | — | `None` | Classify CEMP archetype group; write `*_archetype_likelihood_table.txt` |
| `mcmc_determination()` | — | `None` | Run coarse + refine MCMC for all stars |
| `estimate_logg()` | — | `None` | Interpolate log g from Teff/[Fe/H] |
| `generate_synthetic()` | — | `None` | Generate best-fit synthetic spectra |
| `generate_output_spectra()` | — | `None` | Write `*_spectra_output.csv` |
| `generate_output_files()` | — | `None` | Write `*_out.csv` |
| `generate_plots()` | — | `None` | Generate spectral/corner/trace PDFs |

### `Spectrum` (`casper/interface/spectrum.py`)

**Key attributes**: `filename`, `wavelength`, `flux`, `frame` (DataFrame[wave, flux, norm, cont]), `SEQUENCE`, `STARNAME`, `G_CLASS`, `JK`, `MODE`, `INPUT_CARBON_MODE`, `MCMC_iterations`, `T_SIGMA`, `HARD_TEFF`, `PHOTO_0`, `TEMP_FRAME`, `KP_bounds`, `carbon_mode`, `SN_DICT`, `MCMC_COARSE`, `MCMC_REFINE`, `logg`, `logg_err`, `synth_spectrum`, `LL_DICT`.

**Key methods**: `__init__(spec, filename, is_fits)`, `radial_correction(velocity_km_s)`, `ebv_correct(row)`, `set_frame(wave, flux)`, `trim_frame(bounds)`, `estimate_sn()`, `get_sn()`, `set_params(...)`, `set_KP_bounds()`, `set_carbon_mode()`, `set_temp_frame(df)`, `set_temperature(teff, sigma)`, `set_mcmc_args()`, `get_mcmc_dict(mode)`, `get_output_row()`, `get_spectra_row()`, plus getters (`get_wave`, `get_flux`, `get_frame_wave`, `get_frame_flux`, `get_frame_norm`, `get_frame_cont`, `get_filename`, `get_sequence`, `get_starname`, `get_gravity_class`, `get_rv`, `get_carbon_mode`).

### Scientific Utility Functions

| Function | Module | Purpose |
|---|---|---|
| `ac(cfe, feh)` / `cfe(ac, feh)` | `ac.py` | Convert between [C/Fe] and A(C) |
| `MAD(array)` / `S_MAD(array)` | `MAD.py` | Robust dispersion statistics |
| `GBAND_QUAD(wave, flux, bounds)` | `EW.py` | CH G-band equivalent width |
| `CAII_K6(wave, flux)` | `EW.py` | Ca II K-line equivalent width |
| `get_KP_band(spectrum)` | `EW.py` | Determine KP integration bounds |
| `set_CH_procedure(spectrum)` | `EW.py` | Classify carbon mode |
| `Hernandez/Casagrande/Fukugita/Bergeat(...)` | `temp_calibrations.py` | Photometric Teff calibrations |
| `kde_param(distribution, x0)` | `MCMC_interface.py` | KDE peak-finding on posterior chain |
| `chi_likelihood(...)` / `chi_likelihood_C2(...)` | `MCMC_interface.py` | Coarse-stage log-likelihood |
| `chi_ll_refine(...)` / `chi_ll_refine_C2(...)` | `MCMC_interface.py` | Refine-stage log-likelihood |
| `teff_lnprior(...)` / `sigma_lnprior(...)` / `default_param_edges(...)` | `MLE_priors.py` | Prior probability / bounds checking |
| `get_interp()` / `get_grav_interp()` | `synthetic_functions.py` | Load pickled spectral/isochrone interpolators |
| `normalize_synth_spectrum(...)` | `synthetic_functions.py` | Normalize synthetic spectrum via GISIC |
| `ln_chi_square_sigma(...)` / `CAII_CH_CHI_LH(...)` | `synthetic_functions.py` | Chi-squared likelihood helpers |
| `archetype_classify_MC(spectrum)` | `interface_main.py` | Monte Carlo archetype classification |
| `mcmc_determination(spectrum, mode)` | `interface_main.py` | Run one MCMC stage |
| `generate_kde_params(spectrum, mode)` | `interface_main.py` | KDE-smooth posterior chains |
| `estimate_logg(spectrum)` | `interface_main.py` | Interpolate surface gravity |
| `generate_synthetic(spectrum)` | `interface_main.py` | Generate best-fit synthetic spectrum |

## Data Models

### Input: Parameter File (CSV)

| Field | Type | Description |
|---|---|---|
| sequence | str | Unique identifier |
| filename | str | Spectrum file name |
| starname | str | Star designation |
| J-K, H-K, g-r | float | Photometric colors |
| EBV_SFD | float | SFD-map reddening |
| TEFF_SET | float | Hard/fixed Teff (optional) |
| T_SIGMA | float | Teff prior uncertainty (K) |
| RV | float | Radial velocity (km/s) |
| class | str | Gravity class (DWARF/GIANT) |
| mode | str | Galactic environment (HALO/UFD) |
| carbon_mode | str | CH or CH+C2 |
| MCMC_iter | int | MCMC iteration count |

### Output: Parameter Results (`*_out.csv`)
Fields: SEQUENCE, FILENAME, STARNAME, G_CLASS_ADOPTED, TEFF/FEH/CFE/A(C) coarse+refine values and errors, logg, logg_err, XI parameters.

### Output: SNR Table (`*_snr.csv`)
Fields: SEQUENCE, FILENAME, SN_AVG/STD and XI_AVG/STD per band (CA, CH, optionally C2).

### Output: Temperature Calibration Table (`*_temp_cal_table.txt`)
Fields: NAME, Bergeat, Hernandez, Casagrande, Fukugita, HARD_TEFF, ADOPTED.

### Output: Archetype Likelihood Table (`*_archetype_likelihood_table.txt`)
Fields: NAME, GI, GII, GIII (log-likelihood scores).

### Output: Spectra Table (`*_spectra_output.csv`)
Fields: SEQUENCE, FILENAME, WAVELENGTH, FLUX_OBS, FLUX_SYNTH, NORM_OBS, NORM_SYNTH.

**Validation**: Field types/ranges are enforced implicitly by the pipeline's own logic (e.g. bounds checks in `MLE_priors.default_param_edges`) rather than a formal schema.
