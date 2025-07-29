from typing import Any, Callable, Dict, Literal, Tuple

import numpy as np
import pandas as pd
import scipy
from scipy.interpolate import interp1d
from statsmodels.nonparametric.kde import KDEUnivariate

from casper.interface import MLE_priors, config
from casper.interface.synthetic_functions import get_interp, normalize_synth_spectrum
from casper.utils.logger_config import setup_logger

logger = setup_logger(__name__)


def get_interpolator():
    return get_interp()


def kde_param(distribution: np.ndarray, x0: float) -> Dict[str, Any]:
    """
    Estimate the peak of a (possibly multimodal) distribution using KDE.

    This function fits a kernel density estimation (KDE) model to the input
    distribution and then uses Powell's optimization method to find the
    maximum of the estimated density.

    Parameters
    ----------
    distribution : np.ndarray
        A 1D array representing the distribution to model.
    x0 : float
        Initial guess for the peak location in the distribution.

    Returns
    -------
    Dict[str, Any]
        A dictionary with:
        - "result": The x-value where the KDE reaches its maximum (float).
        - "kde": The fitted KDEUnivariate object.
    """

    KDE = KDEUnivariate(distribution)

    KDE.fit(bw=np.std(distribution) / 3.0)

    result = scipy.optimize.minimize(lambda x: -1 * KDE.evaluate(x), x0=x0, method="Powell")
    return {"result": float(result["x"]), "kde": KDE}


def interp1d_synth_flux(
    synth_wave: np.ndarray, G_CLASS: str, teff: float, feh: float, carbon: float
) -> Callable[[np.ndarray], np.ndarray] | None:
    """
    Interpolate normalized synthetic flux at a given wavelength range.

    This function retrieves synthetic flux values for given stellar parameters
    using a precomputed interpolator, normalizes them over a specified wavelength
    range, and returns a linear interpolating function for use in later modeling.

    Parameters
    ----------
    synth_wave : np.ndarray
        The wavelength grid for the synthetic spectrum.
    G_CLASS : str
        Spectral class key used to select the appropriate interpolator.
    teff : float
        Effective temperature of the star.
    feh : float
        Metallicity [Fe/H] of the star.
    carbon : float
        Carbon abundance [C/Fe] of the star.

    Returns
    -------
    Callable[[np.ndarray], np.ndarray] | None
        A 1D linear interpolating function over the synthetic spectrum,
        or None if the interpolated flux is not finite.
    """
    INTERPOLATOR = get_interpolator()
    if np.isfinite(INTERPOLATOR[G_CLASS]([teff, feh, carbon])).all():
        synth_flux = INTERPOLATOR[G_CLASS]([teff, feh, carbon])[0]
        norm_synth_flux = normalize_synth_spectrum(
            synth_wave, synth_flux[config.id_start_wave : config.id_end_wave + 1]
        )

        return interp1d(synth_wave, norm_synth_flux, kind="linear")

    else:
        logger.warning("interp1d_synth_flux: Interpolated synthetic flux is not finite")


def likelihood_params(theta: Tuple[float, ...], include_C2: bool = False) -> Tuple[float, ...]:
    """
    Extract model parameters from the input vector `theta`.

    This function returns either 5 or 6 parameters depending on whether the
    CH+C2 mode is enabled. Parameters include stellar properties and inverse
    noise terms used in likelihood calculations.

    Parameters
    ----------
    theta : Tuple[float, ...]
        A tuple containing model parameters in the following order:
        - Teff (effective temperature)
        - [Fe/H] (metallicity)
        - [C/Fe] (carbon abundance)
        - XI_CA (inverse noise variance for Ca II region)
        - XI_CH (inverse noise variance for CH region)
        - XI_C2 (optional; only included if `include_C2=True`)

    include_C2 : bool, optional
        If True, the function extracts all 6 parameters, including XI_C2.
        If False, only the first 5 parameters are returned. Default is False.

    Returns
    -------
    Tuple[float, ...]
        A tuple of 5 or 6 float values depending on the `include_C2` flag.
    """
    if include_C2:
        params = theta[0], theta[1], theta[2], theta[3], theta[4], theta[5]
    else:
        params = theta[0], theta[1], theta[2], theta[3], theta[4]

    return params


def chi_likelihood(
    theta: Tuple[float, ...],
    observed_spec_regions: Dict[str, pd.DataFrame],
    synth_wave: np.ndarray,
    photo_teff: float,
    photo_teff_unc: float,
    SN_DICT: Dict[str, float],
    G_CLASS: str,
    bounds: Literal["default", "final"] = "default",
) -> float:
    """
    Compute the log-likelihood of stellar model parameters given observed spectra.

    This function compares interpolated synthetic spectra to observed spectral
    regions (e.g., Ca II and CH bands) using a chi-squared likelihood, combined
    with Gaussian priors on Teff and log-normal priors on inverse variance terms.

    Parameters
    ----------
    theta : Tuple[float, ...]
        Tuple of model parameters:
        (Teff, [Fe/H], [C/Fe], XI_CA, XI_CH [, XI_C2]).
    observed_spec_regions : Dict[str, pd.DataFrame]
        Dictionary of observed spectra with region names ("CA", "CH") as keys.
        Each DataFrame must contain "wave" and "norm" columns.
    synth_wave : np.ndarray
        The wavelength grid for the synthetic model spectra.
    photo_teff : float
        External photometric estimate of effective temperature.
    photo_teff_unc : float
        Uncertainty in the photometric Teff estimate.
    SN_DICT : Dict[str, Dict[str, float]]
        Dictionary mapping region names ("CA", "CH") to noise model parameters
        (each must contain "alpha" and "beta").
    G_CLASS : str
        Spectral class key used to select the interpolator model.
    bounds : Literal["default", "final"], optional
        Specifies which set of parameter bounds to use. Default is "default".

    Returns
    -------
    float
        Log-likelihood value. Returns -np.inf if the interpolation fails
        or the likelihood evaluates to a non-finite value.
    """

    teff, feh, carbon, XI_CA, XI_CH = likelihood_params(theta)

    synth_flux_region = interp1d_synth_flux(synth_wave, G_CLASS, teff, feh, carbon)

    if not synth_flux_region:
        logger.warning(f"chi_likelihood: synth_flux_region (teff={teff}, feh={feh}, carbon={carbon}) returned -np.inf")
        return -np.inf

    LL = (
        MLE_priors.ln_chi_square_sigma(
            observed_spec_regions["CA"]["norm"].values,
            synth_flux_region(observed_spec_regions["CA"]["wave"].values),
            XI_CA,
        )
        + MLE_priors.ln_chi_square_sigma(
            observed_spec_regions["CH"]["norm"].values,
            synth_flux_region(observed_spec_regions["CH"]["wave"].values),
            XI_CH,
        )
        + MLE_priors.teff_lnprior(teff, photo_teff, photo_teff_unc)
        + MLE_priors.sigma_lnprior(XI_CA, SN_DICT["CA"]["alpha"], SN_DICT["CA"]["beta"])
        + MLE_priors.sigma_lnprior(XI_CH, SN_DICT["CH"]["alpha"], SN_DICT["CH"]["beta"])
        + MLE_priors.param_edges(teff, feh, carbon, [XI_CA, XI_CH], bounds)
    )

    if np.isfinite(LL):
        return LL

    else:
        logger.warning("chi_likelihood returned -np.inf")
        return -np.inf


def chi_likelihood_C2(
    theta: Tuple[float, ...],
    observed_spec_regions: Dict[str, pd.DataFrame],
    synth_wave: np.ndarray,
    photo_teff: float,
    photo_teff_unc: float,
    SN_DICT: Dict[str, Dict[str, float]],
    G_CLASS: str,
    bounds: Literal["default", "final"] = "default",
) -> float:
    """
    Compute the log-likelihood including the C2 band when AC > 8.

    This function extends `chi_likelihood` by incorporating the C2 band
    in addition to the Ca II and CH regions. It assumes the model includes
    an extra parameter (XI_C2) for the C2 noise level and balances the
    influence of CH and C2 via a 0.5 weighting factor.

    Parameters
    ----------
    theta : Tuple[float, ...]
        Model parameters tuple:
        (Teff, [Fe/H], [C/Fe], XI_CA, XI_CH, XI_C2).
    observed_spec_regions : Dict[str, pd.DataFrame]
        Dictionary of observed spectral regions ("CA", "CH", "C2").
        Each DataFrame must contain "wave" and "norm" columns.
    synth_wave : np.ndarray
        Wavelength grid for synthetic spectra.
    photo_teff : float
        Photometric estimate of effective temperature.
    photo_teff_unc : float
        Uncertainty in the photometric Teff estimate.
    SN_DICT : Dict[str, Dict[str, float]]
        Mapping from region name to noise model parameters ("alpha" and "beta").
    G_CLASS : str
        Spectral class key for selecting the synthetic model interpolator.
    bounds : Literal["default", "final"], optional
        Set of parameter bounds to use. Default is "default".

    Returns
    -------
    float
        Log-likelihood score incorporating Ca II, CH, and C2 bands.
        Returns -np.inf if interpolation fails or the result is not finite.
    """

    teff, feh, carbon, XI_CA, XI_CH, XI_C2 = likelihood_params(theta, include_C2=True)

    synth_flux_region = interp1d_synth_flux(synth_wave, G_CLASS, teff, feh, carbon)

    if not synth_flux_region:
        logger.warning(
            f"chi_likelihood_C2: synth_flux_region (teff={teff}, feh={feh}, carbon={carbon}) returned -np.inf"
        )
        return -np.inf

    LL = (
        MLE_priors.ln_chi_square_sigma(
            observed_spec_regions["CA"]["norm"].values,
            synth_flux_region(observed_spec_regions["CA"]["wave"].values),
            XI_CA,
        )
        + 0.5
        * MLE_priors.ln_chi_square_sigma(
            observed_spec_regions["CH"]["norm"].values,
            synth_flux_region(observed_spec_regions["CH"]["wave"].values),
            XI_CH,
        )
        + 0.5
        * MLE_priors.ln_chi_square_sigma(
            observed_spec_regions["C2"]["norm"].values,
            synth_flux_region(observed_spec_regions["C2"]["wave"].values),
            XI_C2,
        )
        + MLE_priors.teff_lnprior(teff, photo_teff, photo_teff_unc)
        + MLE_priors.sigma_lnprior(XI_CA, SN_DICT["CA"]["alpha"], SN_DICT["CA"]["beta"])
        + MLE_priors.sigma_lnprior(XI_CH, SN_DICT["CH"]["alpha"], SN_DICT["CH"]["beta"])
        + MLE_priors.sigma_lnprior(XI_C2, SN_DICT["C2"]["alpha"], SN_DICT["C2"]["beta"])
        + MLE_priors.param_edges(teff, feh, carbon, [XI_CA, XI_CH, XI_C2], bounds)
    )

    if np.isfinite(LL):
        return LL

    else:
        return -np.inf


def chi_ll_refine(
    theta: Tuple[float, float],
    observed_spec_regions: Dict[str, pd.DataFrame],
    synth_wave: np.ndarray,
    PARAMS: Dict[str, Tuple[float]],
    G_CLASS: str,
    bounds: Literal["default", "final"] = "default",
) -> float:
    """
    Refined log-likelihood using fixed Teff and sigma values from earlier sampling.

    This function refines the likelihood by evaluating only two parameters:
    [Fe/H] and [C/Fe], assuming Teff, XI_CA, and XI_CH have been fixed previously.
    Typically used after initial fitting, especially when the CH+C2 mode is active
    (e.g., AC > 8) and Teff/sigma values are considered well-constrained.

    Parameters
    ----------
    theta : Tuple[float, float]
        Model parameters to refine: (Fe/H, [C/Fe]).
    observed_spec_regions : Dict[str, pd.DataFrame]
        Dictionary of observed spectra by region ("CA", "CH").
        Each DataFrame must contain "wave" and "norm" columns.
    synth_wave : np.ndarray
        The wavelength grid for the synthetic spectrum.
    PARAMS : Dict[str, Tuple[float]]
        Dictionary of fixed values for Teff and sigma parameters.
        Required keys: "TEFF", "XI_CA", "XI_CH".
    G_CLASS : str
        Spectral class key used to select the interpolator.
    bounds : Literal["default", "final"], optional
        Parameter bounds to apply. Default is "default".

    Returns
    -------
    float
        Log-likelihood score for the refined parameters.
        Returns -np.inf if interpolation fails or likelihood is not finite.
    """

    teff = PARAMS["TEFF"][0]
    XI_CA = PARAMS["XI_CA"][0]
    XI_CH = PARAMS["XI_CH"][0]

    feh = theta[0]
    carbon = theta[1]

    synth_flux_region = interp1d_synth_flux(synth_wave, G_CLASS, teff, feh, carbon)
    if not synth_flux_region:
        logger.warning(f"chi_ll_refine: synth_flux_region (teff={teff}, feh={feh}, carbon={carbon}) returned -np.inf")
        return -np.inf

    LL = (
        MLE_priors.ln_chi_square_sigma(
            observed_spec_regions["CA"]["norm"].values,
            synth_flux_region(observed_spec_regions["CA"]["wave"].values),
            XI_CA,
        )
        + MLE_priors.ln_chi_square_sigma(
            observed_spec_regions["CH"]["norm"].values,
            synth_flux_region(observed_spec_regions["CH"]["wave"].values),
            XI_CH,
        )
        + MLE_priors.default_feh_cfe_param_edges(feh, carbon)
    )

    if np.isfinite(LL):
        return LL

    else:
        return -np.inf


def chi_ll_refine_C2(
    theta: np.ndarray,
    observed_spec_regions: Dict[str, pd.DataFrame],
    synth_wave: np.ndarray,
    PARAMS: Dict[str, np.ndarray],
    G_CLASS: str,
    bounds: str = "default",
) -> float:
    """
    Compute the log-likelihood (LL) score for observed vs. synthetic spectra
    using chi-squared loss across three molecular regions: Ca II, CH, and C₂.

    Parameters
    ----------
    theta : np.ndarray
        MCMC parameter array where::
            - theta[0] = [Fe/H] metallicity
            - theta[1] = [C/Fe] carbon abundance

    observed_spec_regions : dict
        Dictionary with keys "CA", "CH", and "C2", each mapping to a DataFrame
        containing observed spectral data with columns "wave" and "norm".

    synth_wave : np.ndarray
        Wavelength grid for the synthetic spectra.

    PARAMS : dict
        Dictionary of stellar parameters and inverse noise terms.
        Must contain:
            - "TEFF": np.ndarray of effective temperature
            - "XI_CA", "XI_CH", "XI_C2": np.ndarrays of inverse noise (1/SNR)
    G_CLASS : str
        The stellar class used to identify the appropriate synthetic model.
    bounds : str, optional
        Bound checking mode. Defaults to "default".

    Returns
    -------
    float
        The log-likelihood value. Returns -np.inf if synthetic flux generation fails
        or LL is non-finite.
    """

    teff = PARAMS["TEFF"][0]
    XI_CA = PARAMS["XI_CA"][0]
    XI_CH = PARAMS["XI_CH"][0]
    XI_C2 = PARAMS["XI_C2"][0]

    feh = theta[0]
    carbon = theta[1]

    synth_flux_region = interp1d_synth_flux(synth_wave, G_CLASS, teff, feh, carbon)

    if not synth_flux_region:
        logger.warning(
            f"chi_11_refine_C2: synth_flux_region (teff={teff}, feh={feh}, carbon={carbon}) returned -np.inf"
        )
        return -np.inf

    LL = (
        MLE_priors.ln_chi_square_sigma(
            observed_spec_regions["CA"]["norm"].values,
            synth_flux_region(observed_spec_regions["CA"]["wave"].values),
            XI_CA,
        )
        + 0.5
        * MLE_priors.ln_chi_square_sigma(
            observed_spec_regions["CH"]["norm"].values,
            synth_flux_region(observed_spec_regions["CH"]["wave"].values),
            XI_CH,
        )
        + 0.5
        * MLE_priors.ln_chi_square_sigma(
            observed_spec_regions["C2"]["norm"].values,
            synth_flux_region(observed_spec_regions["C2"]["wave"].values),
            XI_C2,
        )
        + MLE_priors.default_feh_cfe_param_edges(feh, carbon)
    )

    if np.isfinite(LL):
        return LL

    else:
        return -np.inf
