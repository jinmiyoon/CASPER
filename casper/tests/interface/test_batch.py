from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from casper.interface import config
from casper.interface.batch import Batch, _load_spectrum_file
from casper.interface.spectrum import Spectrum

TEST_SPECTRA_DIR = Path(__file__).resolve().parents[2] / "inputs" / "spectra" / "test_spectra"
FITS_FILENAME = "he0017_m1b_casper.fits"
CSV_FILENAME = "g77-61_coadd_spectrum.csv"

PARAM_COLUMNS = [
    "sequence",
    "filename",
    "starname",
    "J-K",
    "H-K",
    "g-r",
    "EBV_SFD",
    "TEFF_SET",
    "T_SIGMA",
    "RV",
    "class",
    "mode",
    "carbon_mode",
    "MCMC_iter",
]


def make_batch() -> Batch:
    """Construct a Batch without going through set_io_paths()/param-file I/O."""
    return Batch(io_paths={"param_path": "unused", "spectra_dir_path": "unused", "output_file_name": "unused.csv"})


def make_param_row(**overrides):
    row = {
        "sequence": 1,
        "filename": CSV_FILENAME,
        "starname": "G77-61",
        "J-K": 0.9373,
        "H-K": 0.3267,
        "g-r": np.nan,
        "EBV_SFD": 0.1089,
        "TEFF_SET": 4127,
        "T_SIGMA": 250,
        "RV": -10.4,
        "class": "DWARF",
        "mode": "HALO",
        "carbon_mode": "CH",
        "MCMC_iter": 128,
    }
    row.update(overrides)
    return row


# ---------------------------------------------------------------------------
# load_params()
# ---------------------------------------------------------------------------


def test_load_params_coerces_dtypes_and_builds_sequence_list(tmp_path):
    param_path = tmp_path / "params.csv"
    pd.DataFrame([make_param_row(sequence=1), make_param_row(sequence=2, filename=FITS_FILENAME)]).to_csv(
        param_path, index=False
    )

    batch = make_batch()
    batch.param_path = str(param_path)
    batch.load_params()

    assert batch.sequence == ["1", "2"]
    assert batch.param_file["sequence"].tolist() == ["1", "2"]
    assert (batch.param_file["mode"] == "HALO").all()
    assert (batch.param_file["class"] == "DWARF").all()
    assert (batch.param_file["carbon_mode"] == "CH").all()


# ---------------------------------------------------------------------------
# load_spectra()
# ---------------------------------------------------------------------------


def test_load_spectra_dispatches_fits_and_csv():
    batch = make_batch()
    batch.spectra_path = str(TEST_SPECTRA_DIR)
    batch.param_file = pd.DataFrame({"filename": [FITS_FILENAME, CSV_FILENAME]})

    batch.load_spectra()

    assert batch.spectra_names == [FITS_FILENAME, CSV_FILENAME]
    assert batch.length == 2
    assert all(isinstance(spec, Spectrum) for spec in batch.spectra_array)
    assert batch.spectra_array[0].filename == FITS_FILENAME
    assert batch.spectra_array[1].filename == CSV_FILENAME


def test_load_spectra_rejects_unsupported_extension():
    batch = make_batch()
    batch.spectra_path = str(TEST_SPECTRA_DIR)
    batch.param_file = pd.DataFrame({"filename": ["not_a_real_format.txt"]})

    with pytest.raises(Exception, match="Invalid file format extension"):
        batch.load_spectra()


def test_load_spectrum_file_helper_directly_for_csv():
    spec = _load_spectrum_file(str(TEST_SPECTRA_DIR / CSV_FILENAME), CSV_FILENAME)
    assert isinstance(spec, Spectrum)
    assert spec.filename == CSV_FILENAME


# ---------------------------------------------------------------------------
# set_params()
# ---------------------------------------------------------------------------


def test_set_params_propagates_to_each_spectrum():
    batch = make_batch()
    batch.param_file = pd.DataFrame([make_param_row(sequence=1, filename="a.csv")])
    spec = make_csv_spectrum_stub("a.csv")
    batch.spectra_array = [spec]

    batch.set_params()

    assert spec.SEQUENCE == "1"
    assert spec.STARNAME == "G77-61"
    assert spec.G_CLASS == "DWARF"
    assert spec.MODE == "HALO"
    assert spec.INPUT_CARBON_MODE == "CH"
    assert spec.MCMC_iterations == 128
    assert spec.T_SIGMA == 250.0
    assert spec.HARD_TEFF == 4127.0


def test_set_params_raises_on_filename_mismatch():
    batch = make_batch()
    batch.param_file = pd.DataFrame([make_param_row(sequence=1, filename="a.csv")])
    batch.spectra_array = [make_csv_spectrum_stub("different_file.csv")]

    with pytest.raises(AssertionError):
        batch.set_params()


# ---------------------------------------------------------------------------
# radial_correct()
# ---------------------------------------------------------------------------


def test_radial_correct_applies_rv_per_sequence():
    batch = make_batch()
    batch.param_file = pd.DataFrame(
        [
            make_param_row(sequence="1", filename="a.csv", RV=-10.4),
            make_param_row(sequence="2", filename="b.csv", RV=5.0),
        ]
    )
    batch.sequence = ["1", "2"]
    spec_a = make_csv_spectrum_stub("a.csv")
    spec_b = make_csv_spectrum_stub("b.csv")
    batch.spectra_array = [spec_a, spec_b]

    batch.radial_correct()

    assert spec_a.rv == pytest.approx(-10.4)
    assert spec_b.rv == pytest.approx(5.0)
    assert np.allclose(spec_a.wavelength, spec_a.original_wavelength / ((-10.4 / 2.99792e5) + 1))
    assert np.allclose(spec_b.wavelength, spec_b.original_wavelength / ((5.0 / 2.99792e5) + 1))


# ---------------------------------------------------------------------------
# build_frames()
# ---------------------------------------------------------------------------


def test_build_frames_constructs_and_trims_each_spectrum():
    batch = make_batch()
    spec = make_csv_spectrum_stub("a.csv", wave=[3000.0, 4000.0, 4500.0, 6000.0], flux=[1.0, 2.0, 3.0, 4.0])
    batch.spectra_array = [spec]

    batch.build_frames(bounds=(3800.0, 5000.0))

    assert list(spec.frame["wave"]) == [4000.0, 4500.0]


# ---------------------------------------------------------------------------
# normalize()
# ---------------------------------------------------------------------------


def test_normalize_populates_frame_norm_and_cont():
    # Use real sample spectrum data (trimmed to the analysis wavelength bounds), since
    # Batch.normalize() smooths with much larger sigma values (config.SIGMA: 15-30) than
    # a narrow synthetic test spectrum can survive without losing all inflection points.
    df = pd.read_csv(TEST_SPECTRA_DIR / CSV_FILENAME)
    frame = df[df["wave"].between(*config.WAVE_BOUNDS, inclusive="both")].reset_index(drop=True)
    wave = frame["wave"].to_numpy(dtype=float)
    flux = frame["flux"].to_numpy(dtype=float)

    spec = make_csv_spectrum_stub(CSV_FILENAME, wave=wave, flux=flux)
    spec.set_frame(wave=wave, flux=flux)
    batch = make_batch()
    batch.spectra_array = [spec]

    batch.normalize()

    assert "norm" in spec.frame.columns
    assert "cont" in spec.frame.columns
    assert len(spec.frame["norm"]) == len(wave)
    assert len(spec.frame["cont"]) == len(wave)
    assert np.all(np.isfinite(spec.frame["norm"]))


def test_normalize_passes_correct_band_check_value():
    # Regression test for the band_check/cahk wiring fix (2026-08-11): Batch.normalize()
    # previously passed band_check=config.cahk (a copy-paste bug); now correctly passes
    # band_check=config.band_check. config.cahk is True and config.band_check is False,
    # so these must resolve to different values for this test to be meaningful.
    assert config.band_check != config.cahk

    spec = make_csv_spectrum_stub(CSV_FILENAME, wave=[4000.0, 4001.0], flux=[1.0, 2.0])
    spec.set_frame(wave=np.array([4000.0, 4001.0]), flux=np.array([1.0, 2.0]))
    batch = make_batch()
    batch.spectra_array = [spec]

    with patch("casper.interface.batch.normalize") as mock_normalize:
        mock_normalize.return_value = (spec.frame["wave"].to_numpy(), np.array([1.0, 1.0]), np.array([1.0, 1.0]))
        batch.normalize()

    assert mock_normalize.called
    _, kwargs = mock_normalize.call_args
    assert kwargs["cahk"] == config.cahk
    assert kwargs["band_check"] == config.band_check


def make_csv_spectrum_stub(filename: str, wave=None, flux=None) -> Spectrum:
    """Build a lightweight Spectrum instance via the CSV constructor path for Batch tests."""
    if wave is None:
        wave = [4000.0, 4001.0]
    if flux is None:
        flux = [1.0, 2.0]
    frame = pd.DataFrame({"wave": wave, "flux": flux})
    return Spectrum(frame, filename=filename, is_fits=False)


# ---------------------------------------------------------------------------
# ebv_correction()
# ---------------------------------------------------------------------------


def test_ebv_correction_dispatches_row_to_matching_spectrum():
    batch = make_batch()
    batch.param_file = pd.DataFrame(
        [
            {"J-K": 1.0, "H-K": 0.5, "g-r": 0.8, "EBV_SFD": 0.0},
            {"J-K": 2.0, "H-K": 1.0, "g-r": 1.6, "EBV_SFD": 0.0},
        ]
    )
    spec_a = make_csv_spectrum_stub("a.csv")
    spec_b = make_csv_spectrum_stub("b.csv")
    batch.spectra_array = [spec_a, spec_b]

    batch.ebv_correction()

    assert spec_a.PHOTO_0["J-K"] == pytest.approx(1.0)
    assert spec_b.PHOTO_0["J-K"] == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# calibrate_temperatures()
# ---------------------------------------------------------------------------


def test_calibrate_temperatures_adopts_hard_teff_when_finite(tmp_path):
    batch = make_batch()
    batch.output_name = str(tmp_path / "unit4_test")
    batch.param_file = pd.DataFrame([{"filename": "a.csv", "class": "DWARF"}])
    spec = make_csv_spectrum_stub("a.csv")
    spec.PHOTO_0 = {"J-K": 0.5, "H-K": 0.3, "g-r": 0.4}
    spec.T_SIGMA = 250.0
    spec.HARD_TEFF = 4127.0
    batch.spectra_array = [spec]

    batch.calibrate_temperatures()

    assert spec.TEMP_FRAME.loc["HARD_TEFF", "VALUE"] == pytest.approx(4127.0)
    assert spec.TEMP_FRAME.loc["ADOPTED", "VALUE"] == pytest.approx(4127.0)
    assert spec.teff_irfm == pytest.approx(4127.0)
    assert spec.teff_irfm_err == pytest.approx(250.0)
    assert Path(batch.output_name + "_temp_cal_table.txt").exists()


def test_calibrate_temperatures_adopts_photometric_teff_when_hard_teff_missing(tmp_path):
    batch = make_batch()
    batch.output_name = str(tmp_path / "unit4_test")
    batch.param_file = pd.DataFrame([{"filename": "a.csv", "class": "DWARF"}])
    spec = make_csv_spectrum_stub("a.csv")
    spec.PHOTO_0 = {"J-K": 0.5, "H-K": 0.3, "g-r": 0.4}
    spec.T_SIGMA = 250.0
    spec.HARD_TEFF = np.nan
    batch.spectra_array = [spec]

    batch.calibrate_temperatures()

    assert np.isnan(spec.TEMP_FRAME.loc["HARD_TEFF", "VALUE"])
    adopted = spec.TEMP_FRAME.loc["ADOPTED", "VALUE"]
    assert np.isfinite(adopted)
    assert spec.teff_irfm == pytest.approx(adopted)


# ---------------------------------------------------------------------------
# set_KP_bounds()
# ---------------------------------------------------------------------------


def test_set_kp_bounds_dispatches_get_kp_band_result_to_each_spectrum():
    batch = make_batch()
    spec_a = make_csv_spectrum_stub("a.csv")
    spec_b = make_csv_spectrum_stub("b.csv")
    batch.spectra_array = [spec_a, spec_b]

    with patch("casper.interface.batch.EW.get_KP_band", return_value=config.KP_BOUNDS["K6"]) as mock_get_kp_band:
        batch.set_KP_bounds()

    assert mock_get_kp_band.call_count == 2
    assert spec_a.KP_bounds == config.KP_BOUNDS["K6"]
    assert spec_b.KP_bounds == config.KP_BOUNDS["K6"]


# ---------------------------------------------------------------------------
# set_carbon_mode()
# ---------------------------------------------------------------------------


def test_set_carbon_mode_invokes_set_ch_procedure_per_spectrum():
    batch = make_batch()
    spec_a = make_csv_spectrum_stub("a.csv")
    spec_b = make_csv_spectrum_stub("b.csv")
    batch.spectra_array = [spec_a, spec_b]

    with patch("casper.interface.batch.EW.set_CH_procedure") as mock_set_ch:
        batch.set_carbon_mode()

    assert mock_set_ch.call_count == 2
    mock_set_ch.assert_any_call(spec_a)
    mock_set_ch.assert_any_call(spec_b)


# ---------------------------------------------------------------------------
# estimate_sn()
# ---------------------------------------------------------------------------


def test_estimate_sn_invokes_estimate_sn_on_each_spectrum():
    batch = make_batch()
    spec_a = make_csv_spectrum_stub("a.csv")
    spec_b = make_csv_spectrum_stub("b.csv")
    batch.spectra_array = [spec_a, spec_b]

    with patch.object(spec_a, "estimate_sn") as mock_a, patch.object(spec_b, "estimate_sn") as mock_b:
        batch.estimate_sn()

    mock_a.assert_called_once()
    mock_b.assert_called_once()


# ---------------------------------------------------------------------------
# get_sn()
# ---------------------------------------------------------------------------


def test_get_sn_concatenates_and_writes_csv(tmp_path):
    batch = make_batch()
    batch.output_name = str(tmp_path / "unit4_test")
    spec_a = make_csv_spectrum_stub("a.csv")
    spec_b = make_csv_spectrum_stub("b.csv")
    row_a = pd.DataFrame({"SEQUENCE": ["1"], "FILENAME": ["a.csv"]})
    row_b = pd.DataFrame({"SEQUENCE": ["2"], "FILENAME": ["b.csv"]})
    batch.spectra_array = [spec_a, spec_b]

    with patch.object(spec_a, "get_sn", return_value=row_a), patch.object(spec_b, "get_sn", return_value=row_b):
        result = batch.get_sn()

    assert list(result["SEQUENCE"]) == ["1", "2"]
    assert Path(batch.output_name + "_snr.csv").exists()
