import os
import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from casper.interface.MCMC_interface import chi_likelihood, chi_ll_refine, kde_param, likelihood_params


@pytest.mark.parametrize(
    "distribution, x0, expected_mode",
    [
        (np.random.normal(loc=5.0, scale=1.0, size=1000), 4.5, 5.0),
        (np.random.normal(loc=-2.0, scale=0.5, size=1000), -1.5, -2.0),
    ],
)
def test_kde_param(distribution, x0, expected_mode):
    result = kde_param(distribution, x0)
    peak = result["result"]

    assert isinstance(peak, float)
    assert np.isclose(peak, expected_mode, atol=0.3)


@pytest.mark.parametrize(
    "theta, include_C2, expected",
    [
        ((5000.0, -2.5, 0.0, 0.01, 0.02, 0.03), False, (5000.0, -2.5, 0.0, 0.01, 0.02)),
        ((5000.0, -2.5, 0.0, 0.01, 0.02, 0.03), True, (5000.0, -2.5, 0.0, 0.01, 0.02, 0.03)),
    ],
)
def test_likelihood_params(theta, include_C2, expected):
    result = likelihood_params(theta, include_C2=include_C2)
    assert result == expected
    assert isinstance(result, tuple)
    assert all(isinstance(val, float) for val in result)


@pytest.mark.parametrize(
    "theta, teff, feh, carbon, xi_ca, xi_ch, expected_result",
    [
        ((5000.0, -2.5, 0.0, 0.01, 0.02), 5000.0, -2.5, 0.0, 0.01, 0.02, 42.0),
        ((4800.0, -2.0, 0.2, 0.02, 0.03), 4800.0, -2.0, 0.2, 0.02, 0.03, 42.0),
    ],
)
def test_chi_likelihood(theta, teff, feh, carbon, xi_ca, xi_ch, expected_result):
    # Mock observed_spec_regions
    wave = np.linspace(3900, 4000, 100)
    flux = np.ones(100)
    df = pd.DataFrame({"wave": wave, "norm": flux})
    observed_spec_regions = {"CA": df.copy(), "CH": df.copy()}

    synth_wave = wave.copy()
    photo_teff = teff
    photo_teff_unc = 100.0
    SN_DICT = {
        "CA": {"alpha": 1.0, "beta": 1.0},
        "CH": {"alpha": 1.0, "beta": 1.0},
    }

    G_CLASS = "GIANT"

    dummy_interp_func = MagicMock(return_value=np.ones_like(wave))

    with (
        patch("casper.interface.MCMC_interface.interp1d_synth_flux", return_value=dummy_interp_func),
        patch("casper.interface.MCMC_interface.MLE_priors.ln_chi_square_sigma", return_value=10.0),
        patch("casper.interface.MCMC_interface.MLE_priors.teff_lnprior", return_value=5.0),
        patch("casper.interface.MCMC_interface.MLE_priors.sigma_lnprior", return_value=5.0),
        patch("casper.interface.MCMC_interface.MLE_priors.param_edges", return_value=7.0),
    ):
        result = chi_likelihood(
            theta=theta,
            observed_spec_regions=observed_spec_regions,
            synth_wave=synth_wave,
            photo_teff=photo_teff,
            photo_teff_unc=photo_teff_unc,
            SN_DICT=SN_DICT,
            G_CLASS=G_CLASS,
        )

        assert result == expected_result


@pytest.mark.parametrize(
    "theta, teff, xi_ca, xi_ch, feh, carbon, expected_ll",
    [
        ((-2.5, 0.0), 5000.0, 0.01, 0.02, -2.5, 0.0, 30.0),
        ((-1.8, 0.3), 5200.0, 0.02, 0.03, -1.8, 0.3, 30.0),
    ],
)
def test_chi_ll_refine(theta, teff, xi_ca, xi_ch, feh, carbon, expected_ll):
    wave = np.linspace(3900, 4000, 100)
    norm = np.ones_like(wave)
    spec_df = pd.DataFrame({"wave": wave, "norm": norm})

    observed_spec_regions = {"CA": spec_df.copy(), "CH": spec_df.copy()}

    PARAMS = {"TEFF": (teff,), "XI_CA": (xi_ca,), "XI_CH": (xi_ch,)}

    dummy_interp = MagicMock(return_value=np.ones_like(wave))

    with (
        patch("casper.interface.MCMC_interface.interp1d_synth_flux", return_value=dummy_interp),
        patch("casper.interface.MCMC_interface.MLE_priors.ln_chi_square_sigma", return_value=10.0),
        patch("casper.interface.MCMC_interface.MLE_priors.default_feh_cfe_param_edges", return_value=10.0),
    ):
        ll = chi_ll_refine(
            theta=theta, observed_spec_regions=observed_spec_regions, synth_wave=wave, PARAMS=PARAMS, G_CLASS="GIANT"
        )

        assert ll == expected_ll
