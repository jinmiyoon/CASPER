from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from astropy.io import fits

from casper.interface import MAD, config
from casper.interface.spectrum import Spectrum, obtain_flux

TEST_SPECTRA_DIR = Path(__file__).resolve().parents[2] / "inputs" / "spectra" / "test_spectra"
FITS_PATH = TEST_SPECTRA_DIR / "he0017_m1b_casper.fits"
CSV_PATH = TEST_SPECTRA_DIR / "g77-61_coadd_spectrum.csv"


def make_csv_spectrum(wave, flux, filename="test.csv"):
    """Build a Spectrum from an in-memory wave/flux DataFrame (CSV code path)."""
    frame = pd.DataFrame({"wave": wave, "flux": flux})
    return Spectrum(frame, filename=filename, is_fits=False)


# ---------------------------------------------------------------------------
# obtain_flux()
# ---------------------------------------------------------------------------


def test_obtain_flux_1d():
    data = np.array([1.0, 2.0, 3.0])
    assert np.array_equal(obtain_flux(data), data)


def test_obtain_flux_multidim_takes_first_row():
    data = np.array([[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]])
    assert np.array_equal(obtain_flux(data), np.array([1.0, 2.0, 3.0]))


# ---------------------------------------------------------------------------
# Constructor: FITS path
# ---------------------------------------------------------------------------


def test_constructor_fits_wavelength_and_flux():
    with fits.open(FITS_PATH) as hdu:
        header = hdu[0].header
        expected_flux = obtain_flux(hdu[0].data)
        naxis1 = header["NAXIS1"]
        crval1 = header["CRVAL1"]
        delta = header["CD1_1"] if "CD1_1" in header else header["CDELT1"]

    assert crval1 > 10.0, "This fixture is expected to exercise the linear-wavelength branch"

    with fits.open(FITS_PATH) as hdu:
        spec = Spectrum(hdu, filename="he0017_m1b_casper.fits", is_fits=True)

    expected_wave = (np.arange(0, naxis1, 1) * delta) + crval1
    assert np.allclose(spec.wavelength, expected_wave)
    assert np.allclose(spec.original_wavelength, expected_wave)
    assert len(spec.flux) == naxis1
    assert np.allclose(spec.flux, expected_flux)
    # byte-order correction: flux must be native-endian after loading
    assert spec.flux.dtype.byteorder in ("=", "<", "|")
    assert spec.segments is None
    assert spec.mad_global is None


# ---------------------------------------------------------------------------
# Constructor: CSV path
# ---------------------------------------------------------------------------


def test_constructor_csv_wavelength_and_flux():
    df = pd.read_csv(CSV_PATH)
    spec = Spectrum(df, filename="g77-61_coadd_spectrum.csv", is_fits=False)

    assert np.allclose(spec.wavelength, df["wave"].to_numpy(dtype=float))
    assert list(spec.flux) == list(df["flux"])
    assert np.allclose(spec.original_wavelength, spec.wavelength)
    assert spec.segments is None
    assert spec.mad_global is None


# ---------------------------------------------------------------------------
# radial_correction()
# ---------------------------------------------------------------------------


def test_radial_correction_zero_velocity_leaves_wavelength_unchanged():
    spec = make_csv_spectrum([4000.0, 4001.0, 4002.0], [1.0, 2.0, 3.0])
    spec.radial_correction()
    assert spec.rv == 0.0
    assert np.allclose(spec.wavelength, spec.original_wavelength)


def test_radial_correction_applies_doppler_formula():
    original = np.array([4000.0, 4001.0, 4002.0])
    spec = make_csv_spectrum(original, [1.0, 2.0, 3.0])
    velocity = 120.0
    spec.radial_correction(velocity)

    expected = original / ((velocity / 2.99792e5) + 1)
    assert spec.rv == velocity
    assert np.allclose(spec.wavelength, expected)
    # original_wavelength must remain untouched (used as the base for correction)
    assert np.allclose(spec.original_wavelength, original)


# ---------------------------------------------------------------------------
# ebv_correct()
# ---------------------------------------------------------------------------


def test_ebv_correct_applies_dereddening_when_ebv_positive():
    spec = make_csv_spectrum([4000.0], [1.0])
    row = pd.Series({"J-K": 1.0, "H-K": 0.5, "g-r": 0.8, "EBV_SFD": 0.1})
    spec.ebv_correct(row)

    expected_jk = 1.0 - (config.A_EBV["A_J"] - config.A_EBV["A_K"]) * 0.1
    expected_hk = 0.5 - (config.A_EBV["A_H"] - config.A_EBV["A_K"]) * 0.1
    expected_gr = 0.8 - (config.A_EBV["A_g"] - config.A_EBV["A_r"]) * 0.1

    assert spec.PHOTO_0["J-K"] == pytest.approx(expected_jk)
    assert spec.PHOTO_0["H-K"] == pytest.approx(expected_hk)
    assert spec.PHOTO_0["g-r"] == pytest.approx(expected_gr)


def test_ebv_correct_leaves_values_when_ebv_zero():
    spec = make_csv_spectrum([4000.0], [1.0])
    row = pd.Series({"J-K": 1.0, "H-K": 0.5, "g-r": 0.8, "EBV_SFD": 0.0})
    spec.ebv_correct(row)

    assert spec.PHOTO_0["J-K"] == pytest.approx(1.0)
    assert spec.PHOTO_0["H-K"] == pytest.approx(0.5)
    assert spec.PHOTO_0["g-r"] == pytest.approx(0.8)


# ---------------------------------------------------------------------------
# set_frame() / trim_frame()
# ---------------------------------------------------------------------------


def test_trim_frame_filters_inclusive_bounds():
    spec = make_csv_spectrum([4000.0], [1.0])
    wave = np.array([3799.0, 3800.0, 4500.0, 5000.0, 5001.0])
    flux = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    spec.set_frame(wave=wave, flux=flux)
    spec.trim_frame(bounds=(3800.0, 5000.0))

    assert list(spec.frame["wave"]) == [3800.0, 4500.0, 5000.0]


def test_trim_frame_default_bounds_uses_config_wave_bounds():
    spec = make_csv_spectrum([4000.0], [1.0])
    wave = np.array([3000.0, 4000.0, 6000.0])
    flux = np.array([1.0, 2.0, 3.0])
    spec.set_frame(wave=wave, flux=flux)
    spec.trim_frame()

    assert list(spec.frame["wave"]) == [4000.0]


# ---------------------------------------------------------------------------
# set_params()
# ---------------------------------------------------------------------------


def _set_params_kwargs(**overrides):
    kwargs = dict(
        SEQUENCE="1",
        STARNAME="test_star",
        CLASS="GIANT",
        JK=0.5,
        MODE="HALO",
        INPUT_CARBON_MODE="CH",
        iter=100,
        T_SIGMA=50.0,
        HARD_TEFF=5000.0,
    )
    kwargs.update(overrides)
    return kwargs


def test_set_params_assigns_attributes():
    spec = make_csv_spectrum([4000.0], [1.0])
    spec.set_params(**_set_params_kwargs())

    assert spec.SEQUENCE == "1"
    assert spec.STARNAME == "test_star"
    assert spec.G_CLASS == "GIANT"
    assert spec.JK == 0.5
    assert spec.MODE == "HALO"
    assert spec.INPUT_CARBON_MODE == "CH"
    assert spec.MCMC_iterations == 100
    assert spec.T_SIGMA == 50.0
    assert spec.HARD_TEFF == 5000.0


def test_set_params_rejects_invalid_class():
    spec = make_csv_spectrum([4000.0], [1.0])
    with pytest.raises(AssertionError):
        spec.set_params(**_set_params_kwargs(CLASS="INVALID"))


def test_set_params_rejects_invalid_mode():
    spec = make_csv_spectrum([4000.0], [1.0])
    with pytest.raises(AssertionError):
        spec.set_params(**_set_params_kwargs(MODE="INVALID"))


# ---------------------------------------------------------------------------
# KP bounds / carbon mode
# ---------------------------------------------------------------------------


def test_set_get_kp_bounds_roundtrip():
    spec = make_csv_spectrum([4000.0], [1.0])
    spec.set_KP_bounds(config.KP_BOUNDS["K6"])
    assert spec.get_KP_bounds() == config.KP_BOUNDS["K6"]


def test_print_kp_bounds_formats_string():
    spec = make_csv_spectrum([4000.0], [1.0])
    spec.set_KP_bounds([3930.7, 3936.7])
    assert spec.print_KP_bounds() == "3930.7 - 3936.7"


def test_set_get_carbon_mode_roundtrip():
    spec = make_csv_spectrum([4000.0], [1.0])
    spec.set_carbon_mode("CH+C2")
    assert spec.get_carbon_mode() == "CH+C2"


# ---------------------------------------------------------------------------
# set_group_ll()
# ---------------------------------------------------------------------------


def test_set_group_ll_selects_highest_likelihood_group():
    spec = make_csv_spectrum([4000.0], [1.0])
    spec.filename = "star.csv"
    input_dict = {"GI": (1.0, {}), "GII": (5.0, {}), "GIII": (2.0, {})}
    spec.set_group_ll(input_dict)

    assert spec.LL_DICT == input_dict
    assert spec.ARCH_GROUP == "GII"


# ---------------------------------------------------------------------------
# set_temperature() / get_photo_temp()
# ---------------------------------------------------------------------------


def test_set_temperature_and_get_photo_temp_roundtrip():
    spec = make_csv_spectrum([4000.0], [1.0])
    spec.set_temperature(5250.0, 80.0)
    assert spec.get_photo_temp() == (5250.0, 80.0)


# ---------------------------------------------------------------------------
# prepare_regions()
# ---------------------------------------------------------------------------


def test_prepare_regions_ch_mode_has_ca_and_ch_only():
    spec = make_csv_spectrum([4000.0], [1.0])
    wave = np.arange(3900.0, 4800.0, 1.0)
    flux = np.ones_like(wave)
    spec.set_frame(wave=wave, flux=flux)
    spec.set_KP_bounds([3930.7, 3936.7])
    spec.set_carbon_mode("CH")

    spec.prepare_regions()

    assert set(spec.regions.keys()) == {"CA", "CH"}
    assert spec.regions["CA"]["wave"].between(3930.7, 3936.7).all()
    assert spec.regions["CH"]["wave"].between(4222, 4322).all()


def test_prepare_regions_ch_c2_mode_adds_c2_region():
    spec = make_csv_spectrum([4000.0], [1.0])
    wave = np.arange(3900.0, 4800.0, 1.0)
    flux = np.ones_like(wave)
    spec.set_frame(wave=wave, flux=flux)
    spec.set_KP_bounds([3930.7, 3936.7])
    spec.set_carbon_mode("CH+C2")

    spec.prepare_regions()

    assert set(spec.regions.keys()) == {"CA", "CH", "C2"}


# ---------------------------------------------------------------------------
# estimate_sn() / get_sn()
# ---------------------------------------------------------------------------


def _band_frame():
    """A frame covering CA and CH sideband ranges but not C2 (out of coverage).

    estimate_sn()'s in-coverage check requires the frame's wavelength range to
    extend beyond both sidebands of a band (not just contain the sideband
    points), so padding points are added outside all band ranges.
    """
    wave = np.concatenate(
        [
            np.array([3800.0]),  # padding: pushes min(wave) below all CA/CH left sidebands
            np.array([3900.0, 3910.0]),  # CA left sideband [3884, 3923]
            np.array([3996.0, 3999.0]),  # CA right sideband [3995, 4045]
            np.array([4010.0, 4020.0]),  # CH left sideband [4000, 4080]
            np.array([4450.0, 4460.0]),  # CH right sideband [4440, 4500]
            np.array([4600.0]),  # padding: pushes max(wave) above CA/CH right sidebands, but not C2's
        ]
    )
    flux = np.array([1.0, 4.0, 16.0, 9.0, 25.0, 1.0, 9.0, 4.0, 16.0, 1.0])
    return wave, flux


def test_estimate_sn_computes_expected_statistics_for_covered_bands():
    wave, flux = _band_frame()
    spec = make_csv_spectrum([4000.0], [1.0])
    spec.set_frame(wave=wave, flux=flux)
    spec.set_carbon_mode("CH")
    spec.estimate_sn()

    for key, left_bounds, right_bounds in (
        ("CA", (3884, 3923), (3995, 4045)),
        ("CH", (4000, 4080), (4440, 4500)),
    ):
        left = np.sqrt(spec.frame["flux"][spec.frame["wave"].between(*left_bounds, inclusive="both")])
        right = np.sqrt(spec.frame["flux"][spec.frame["wave"].between(*right_bounds, inclusive="both")])
        expected_sn_avg = np.mean([np.median(left), np.median(right)])
        expected_sn_std = max(MAD.S_MAD(left), MAD.S_MAD(right))
        expected_xi_avg = np.mean([np.median(np.divide(1.0, left)), np.median(np.divide(1.0, right))])
        expected_xi_std = max(MAD.S_MAD(np.divide(1.0, left)), MAD.S_MAD(np.divide(1.0, right)))

        assert spec.SN_DICT[key]["SN_AVG"] == pytest.approx(expected_sn_avg)
        assert spec.SN_DICT[key]["SN_STD"] == pytest.approx(expected_sn_std)
        assert spec.SN_DICT[key]["XI_AVG"] == pytest.approx(expected_xi_avg)
        assert spec.SN_DICT[key]["XI_STD"] == pytest.approx(expected_xi_std)

        expected_alpha = ((expected_xi_avg**2) / np.square(expected_xi_std)) * (1 - expected_xi_avg) - expected_xi_avg
        expected_beta = (1 / expected_xi_avg - 1) * expected_alpha
        assert spec.SN_DICT[key]["alpha"] == pytest.approx(expected_alpha)
        assert spec.SN_DICT[key]["beta"] == pytest.approx(expected_beta)


def test_estimate_sn_out_of_coverage_band_is_nan():
    wave, flux = _band_frame()
    spec = make_csv_spectrum([4000.0], [1.0])
    spec.set_frame(wave=wave, flux=flux)
    spec.set_carbon_mode("CH")
    spec.estimate_sn()

    for stat in ("SN_AVG", "SN_STD", "XI_AVG", "XI_STD", "alpha", "beta"):
        assert np.isnan(spec.SN_DICT["C2"][stat])


def test_get_sn_returns_expected_columns_ch_mode():
    wave, flux = _band_frame()
    spec = make_csv_spectrum([4000.0], [1.0])
    spec.filename = "star.csv"
    spec.set_params(**_set_params_kwargs(SEQUENCE="42", INPUT_CARBON_MODE="CH"))
    spec.set_frame(wave=wave, flux=flux)
    spec.estimate_sn()

    sn_output = spec.get_sn()

    assert sn_output["SEQUENCE"].iloc[0] == "42"
    assert "XI_C2" not in sn_output.columns


def test_get_sn_includes_c2_columns_in_ch_c2_mode():
    wave, flux = _band_frame()
    spec = make_csv_spectrum([4000.0], [1.0])
    spec.filename = "star.csv"
    spec.set_params(**_set_params_kwargs(SEQUENCE="42", INPUT_CARBON_MODE="CH+C2"))
    spec.set_frame(wave=wave, flux=flux)
    spec.estimate_sn()

    sn_output = spec.get_sn()

    assert "XI_C2" in sn_output.columns
    assert "XI_C2_ERR" in sn_output.columns


# ---------------------------------------------------------------------------
# Remaining setters/getters: lightweight round-trip coverage
# ---------------------------------------------------------------------------


def test_set_get_flux_roundtrip():
    spec = make_csv_spectrum([4000.0], [1.0])
    new_flux = np.array([1.0, 2.0, 3.0])
    spec.set_flux(new_flux)
    assert np.array_equal(spec.get_flux(), new_flux)


def test_set_frame_norm_and_cont_update_frame_columns():
    spec = make_csv_spectrum([4000.0], [1.0])
    spec.set_frame(wave=np.array([4000.0, 4001.0]), flux=np.array([1.0, 2.0]))
    spec.set_frame_norm(np.array([0.9, 1.1]))
    spec.set_frame_cont(np.array([10.0, 20.0]))

    assert list(spec.frame["norm"]) == [0.9, 1.1]
    assert list(spec.frame["cont"]) == [10.0, 20.0]
    assert list(spec.get_frame_wave()) == [4000.0, 4001.0]
    assert list(spec.get_frame_flux()) == [1.0, 2.0]


def test_set_synth_spectrum_roundtrip():
    spec = make_csv_spectrum([4000.0], [1.0])
    synth = {"wave": [1, 2, 3], "norm": [0.1, 0.2, 0.3]}
    spec.set_synth_spectrum(synth)
    assert spec.synth_spectrum == synth


def test_set_mcmc_args_defaults_to_empty_dict():
    # Batch.set_mcmc_args() always calls spec.set_mcmc_args() with no argument,
    # so only the `input_dict is None` branch is ever exercised in production.
    spec = make_csv_spectrum([4000.0], [1.0])
    spec.set_mcmc_args()
    assert spec.mcmc_args == {}


def test_set_mcmc_results_coarse_and_refine():
    spec = make_csv_spectrum([4000.0], [1.0])
    coarse = {"TEFF": (5000.0, 100.0)}
    refine = {"TEFF": (5010.0, 90.0)}
    spec.set_mcmc_results(coarse, mode="COARSE")
    spec.set_mcmc_results(refine, mode="REFINE")

    assert spec.get_mcmc_dict("COARSE") == coarse
    assert spec.get_mcmc_dict("REFINE") == refine
    assert spec.get_mcmc_dict("BOTH") == (coarse, refine)


def test_set_sampler_coarse_and_refine():
    spec = make_csv_spectrum([4000.0], [1.0])
    spec.set_sampler("coarse_sampler", mode="COARSE")
    spec.set_sampler("refine_sampler", mode="REFINE")
    assert spec.MCMC_COARSE_sampler == "coarse_sampler"
    assert spec.MCMC_REFINE_sampler == "refine_sampler"


def test_set_kde_functions_coarse_and_refine():
    spec = make_csv_spectrum([4000.0], [1.0])
    spec.set_kde_functions({"TEFF": "coarse_kde"}, mode="COARSE")
    spec.set_kde_functions({"TEFF": "refine_kde"}, mode="REFINE")
    assert spec.get_kde_dict() == ({"TEFF": "coarse_kde"}, {"TEFF": "refine_kde"})


def test_get_sequence_filename_starname_formatting():
    spec = make_csv_spectrum([4000.0], [1.0])
    spec.filename = "star.csv"
    spec.set_params(**_set_params_kwargs(SEQUENCE="7", STARNAME="HD12345"))

    assert spec.get_sequence() == "7"
    assert spec.get_filename() == "{:<20}".format("star.csv")
    assert spec.get_starname() == "{:<25}".format("HD12345")
