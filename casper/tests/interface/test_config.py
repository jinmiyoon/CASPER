import numpy as np
import pytest

from casper.interface import config


def test_wave_bounds():
    assert config.WAVE_BOUNDS == [3800.0, 5000.0]


def test_synth_wave_matches_wave_bounds():
    assert config.SYNTH_WAVE[0] == config.WAVE_BOUNDS[0]
    assert config.SYNTH_WAVE[-1] == config.WAVE_BOUNDS[1]
    assert np.all(np.diff(config.SYNTH_WAVE) == 1.0)


def test_interpolator_wave_range():
    assert config.INTERPOLATOR_WAVE[0] == 3000.0
    assert config.INTERPOLATOR_WAVE[-1] == 5000.0


def test_id_start_end_wave_indices_match_wave_bounds():
    assert config.INTERPOLATOR_WAVE[config.id_start_wave] == config.WAVE_BOUNDS[0]
    assert config.INTERPOLATOR_WAVE[config.id_end_wave] == config.WAVE_BOUNDS[1]


def test_ch_bounds():
    assert config.CH_BOUNDS == [4222.0, 4322.0]


def test_kp_bounds_keys_and_values():
    assert set(config.KP_BOUNDS.keys()) == {"K6", "K12", "K18"}
    assert config.KP_BOUNDS["K6"] == [3930.7, 3936.7]


def test_sidebands_keys_and_values():
    assert set(config.SIDEBANDS.keys()) == {"CA", "CH", "C2"}
    assert config.SIDEBANDS["CA"] == [[3884, 3923], [3995, 4045]]


def test_archetype_params_structure():
    assert set(config.ARCHETYPE_PARAMS.keys()) == {"HALO", "UFD"}
    for mode in ("HALO", "UFD"):
        assert set(config.ARCHETYPE_PARAMS[mode].keys()) == {"GI", "GII", "GIII"}
        for group in ("GI", "GII", "GIII"):
            params = config.ARCHETYPE_PARAMS[mode][group]
            assert set(params.keys()) == {"FEH", "CFE", "AC"}


def test_archetype_params_known_values():
    assert config.ARCHETYPE_PARAMS["HALO"]["GI"] == {"FEH": -2.5, "CFE": 1.97, "AC": 7.9}
    assert config.ARCHETYPE_PARAMS["UFD"]["GIII"] == {"FEH": -3.5, "CFE": 2.37, "AC": 7.3}


def test_sigma_grid():
    assert config.SIGMA[0] == pytest.approx(15.0)
    assert config.SIGMA[-1] == pytest.approx(30.0)
    assert len(config.SIGMA) == 10


def test_gisic_scalar_constants():
    assert config.k == 1
    assert config.flux_min == 70
    assert config.band_check is False
    assert config.boost is True
    assert config.cahk is True


def test_a_ebv_extinction_coefficients():
    assert config.A_EBV == {"A_J": 0.709, "A_H": 0.449, "A_K": 0.302, "A_g": 3.303, "A_r": 2.285}
