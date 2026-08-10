from pathlib import Path

import numpy as np
import pandas as pd
import pytest

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


def make_csv_spectrum_stub(filename: str, wave=None, flux=None) -> Spectrum:
    """Build a lightweight Spectrum instance via the CSV constructor path for Batch tests."""
    if wave is None:
        wave = [4000.0, 4001.0]
    if flux is None:
        flux = [1.0, 2.0]
    frame = pd.DataFrame({"wave": wave, "flux": flux})
    return Spectrum(frame, filename=filename, is_fits=False)
