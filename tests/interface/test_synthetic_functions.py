from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from casper.interface.synthetic_functions import CAII_CH_CHI_LH, ln_chi_square_sigma, normalize_synth_spectrum


@pytest.mark.parametrize(
    "synth_flux, fake_cont, expected_min, expected_max",
    [
        # Flat flux normalized by flat continuum: 1.0
        (np.ones(5), np.ones(5), 1.0, 1.0),
        # Flux = 2, continuum = 1: normalized = 2
        (np.full(5, 2.0), np.ones(5), 2.0, 2.0),
        # Flux = 3, continuum = 1: normalized = 3
        (np.full(5, 3.0), np.ones(5), 1.0, 1.0),
        # Flux = -1, continuum = 1: normalized = -1
        (np.full(5, -1.0), np.ones(5), 1.0, 1.0),
        # Mixed flux
        (np.array([0.5, 1.0, 1.5]), np.ones(3), 0.5, 1.5),
    ],
)
def test_normalize_syth_spectrum_mocked(synth_flux, fake_cont, expected_min, expected_max):
    synth_wave = np.linspace(3900, 4000, len(synth_flux))

    # Fake return value for normalize(): (placeholder1, placeholder2, continuum)
    fake_return = (None, None, fake_cont)

    with patch("casper.interface.synthetic_functions.normalize", return_value=fake_return):
        with patch("casper.interface.synthetic_functions.config.SIGMA", [15.0]):
            norm_flux = normalize_synth_spectrum(synth_wave, synth_flux)

    assert norm_flux.shape == synth_flux.shape

    assert np.all(norm_flux >= 0.0)
    assert np.all(norm_flux <= 2.0)

    assert np.min(norm_flux) >= expected_min
    assert np.max(norm_flux) <= expected_max


@pytest.mark.parametrize(
    "flux, synth, xi, expect_inf",
    [
        # Perfect match: returns -inf
        (np.array([1.0, 1.0, 1.0]), np.array([1.0, 1.0, 1.0]), 0.1, True),
        # Big mismatch: returns finite log-likelihood
        (np.array([1.0, 0.5, 1.5]), np.array([1.0, 1.0, 1.0]), 0.1, False),
        # Another chi^2 = 0 case
        (np.array([1.0, 1.0]), np.array([1.0, 1.0]), 1.0, True),
    ],
)
def test_ln_chi_square_sigma(flux, synth, xi, expect_inf):
    result = ln_chi_square_sigma(flux, synth, xi)

    if bool(expect_inf):
        assert result == -np.inf
    else:
        assert np.isfinite(result)


@pytest.mark.parametrize(
    "obs_flux, synth_flux, ca_bounds, ch_bounds, ca_xi, ch_xi, expect_finite",
    [
        # Perfect match: returns -inf
        (np.ones(100), np.ones(100), (3950, 3980), (4290, 4320), 0.1, 0.1, False),
        # Total mismatch
        (np.full(100, 0.0), np.full(100, 1.0), (3950, 3980), (4290, 4320), 0.1, 0.1, True),
        # No CH region data (so only Ca contributes which also returns -inf if it's a perfect match)
        (np.ones(100), np.ones(100), (3950, 3980), (5000, 5100), 0.1, 0.1, False),
    ],
)
def test_caii_ch_chi_lh(obs_flux, synth_flux, ca_bounds, ch_bounds, ca_xi, ch_xi, expect_finite):
    wave = np.linspace(3900, 4400, 100)
    obs_df = pd.DataFrame({"wave": wave, "norm": obs_flux})
    synth_dict = {"wave": wave, "norm": synth_flux}

    result = CAII_CH_CHI_LH(
        obs=obs_df, synth=synth_dict, CA_BOUNDS=ca_bounds, CH_BOUNDS=ch_bounds, CA_XI=ca_xi, CH_XI=ch_xi
    )

    if expect_finite:
        assert np.isfinite(result), "Expected a finite log-likelihood value"
    else:
        assert result == -np.inf
