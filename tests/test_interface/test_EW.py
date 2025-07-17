import numpy as np
import pytest

from casper.interface import config
from casper.interface.EW import CAII_K6, CAII_K12, CAII_K18, CAII_KP, GBAND_QUAD, get_KP_band, set_CH_procedure


def test_gband_quad():
    wave = np.linspace(4200, 4322, 1200)  # range is from 4200A to 4322A using 1000 points
    flux = np.ones_like(wave)  # simulates a normalized flat continuum with no abosrption lines
    flux[(wave > 4290) & (wave < 4300)] -= 0.1
    ew, ew_subtract = GBAND_QUAD(wave, flux)
    assert ew > 0
    assert ew_subtract >= 0


def test_CAII_K6():
    wave = np.linspace(3925, 3940, 200)
    flux = np.ones_like(wave)
    flux[(wave > 3932) & (wave < 3935)] -= 0.2
    ew = CAII_K6(wave, flux)
    assert ew > 0
    assert not np.isnan(ew)


def test_CAII_K12():
    wave = np.linspace(3925, 3942, 200)
    flux = np.ones_like(wave)
    flux[(wave > 3930) & (wave < 3937)] -= 0.15
    ew = CAII_K12(wave, flux)
    assert ew > 0
    assert not np.isnan(ew)


def test_caii_k18():
    wave = np.linspace(3920, 3945, 200)
    flux = np.ones_like(wave)
    flux[(wave > 3928) & (wave < 3938)] -= 0.12
    ew = CAII_K18(wave, flux)
    assert ew > 0
    assert not np.isnan(ew)


# mock Spectrum class
class MockSpectrum:
    def __init__(self):
        self.frame = {
            "wave": np.linspace(3920, 3945, 1000),
            "norm": np.ones(1000),
        }


@pytest.mark.parametrize(
    "k6_val, k12_val, k18_val, expected",
    [
        (1.8, 3.0, 6.0, config.KP_BOUNDS["K6"]),  # K6 <= 2.0
        (2.1, 4.9, 6.0, config.KP_BOUNDS["K12"]),  # K6 > 2.0 and K12 <= 5.0
        (3.0, 6.0, 5.1, config.KP_BOUNDS["K18"]),  # K18 > 5.0
        (2.5, 5.5, 4.0, np.nan),  # None of the conditions are met
    ],
)
def test_get_KP_band(monkeypatch, k6_val, k12_val, k18_val, expected):
    spectrum = MockSpectrum()

    # Patch CAII_K6, K12, K18 to return controlled values
    monkeypatch.setattr("casper.interface.EW.CAII_K6", lambda w, f: k6_val)
    monkeypatch.setattr("casper.interface.EW.CAII_K12", lambda w, f: k12_val)
    monkeypatch.setattr("casper.interface.EW.CAII_K18", lambda w, f: k18_val)

    result = get_KP_band(spectrum)

    if isinstance(expected, float) and np.isnan(expected):
        assert isinstance(result, float) and np.isnan(result)
    else:
        np.testing.assert_array_equal(result, expected)


class MockSpectrum:
    def __init__(self, carbon_mode=None):
        self.frame = {
            "wave": [4200, 4210, 4220],  # dummy values (not used due to patch)
            "norm": [1.0, 0.9, 1.0],
        }
        self.INPUT_CARBON_MODE = carbon_mode
        self.gb_value = None
        self.ch_mode = None

    def set_GBAND(self, value):
        self.gb_value = value

    def set_carbon_mode(self, mode):
        self.ch_mode = mode


@pytest.mark.parametrize(
    "input_mode, ch_ew_return, expected_mode",
    [
        ("CH", 50.0, "CH"),
        ("CH+C2", 20.0, "CH+C2"),
        (None, 45.0, "CH+C2"),
        (None, 30.0, "CH"),
    ],
)
def test_set_CH_procedure(monkeypatch, input_mode, ch_ew_return, expected_mode):
    spectrum = MockSpectrum(carbon_mode=input_mode)

    monkeypatch.setattr("casper.interface.EW.GBAND_QUAD", lambda w, f: (ch_ew_return, 0.0))

    set_CH_procedure(spectrum)
    assert spectrum.gb_value == ch_ew_return
    assert spectrum.ch_mode == expected_mode


@pytest.mark.parametrize(
    "k6_val, k12_val, k18_val, expected",
    [
        (1.8, 3.0, 6.0, 1.8),  # K6 <= 2.0
        (2.1, 4.9, 6.0, 4.9),  # K6 > 2.0 and K12 <= 5.0
        (3.0, 6.0, 5.1, 5.1),  # K18 > 5.0
        (2.5, 5.5, 4.0, np.nan),
    ],
)
def test_CAII_KP(monkeypatch, k6_val, k12_val, k18_val, expected):
    monkeypatch.setattr("casper.interface.EW.CAII_K6", lambda w, f: k6_val)
    monkeypatch.setattr("casper.interface.EW.CAII_K12", lambda w, f: k12_val)
    monkeypatch.setattr("casper.interface.EW.CAII_K18", lambda w, f: k18_val)

    wave = np.linspace(3920, 3945, 1000)
    flux = np.ones_like(wave)

    result = CAII_KP(wave, flux)

    if isinstance(expected, float) and np.isnan(expected):
        assert isinstance(result, float) and np.isnan(result)
    else:
        assert result == expected
