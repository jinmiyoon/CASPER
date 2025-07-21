import pickle as pkl
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from interface import config
from interface.GISIC_C.normalize import normalize
from numpy.typing import ArrayLike
from scipy.interpolate import interp1d


def get_interp():
    """
    Load and return the precomputed spectral interpolation library.

    This function reads a pickled file containing a precomputed interpolator
    used for synthetic spectra at a resolution of R~2000. The interpolator is
    used to estimate model spectra at arbitrary stellar parameters.

    Returns
    -------
    INTERPOLATOR : object
        The loaded interpolation object from the pickled file.

    """
    with open("interface/libraries/SYNTHETIC_SPEC_R2000_INTERP.pkl", "rb") as master_lib:
        INTERPOLATOR = pkl.load(master_lib)
    return INTERPOLATOR


def get_grav_interp():
    """
    Load and return the pre-computed surface-gravity interpolator.

    This utility opens the pickled file that stores an isochrone-based
    interpolator for stellar surface gravity (log g) and returns the
    loaded object.

    Returns
    -------
    GRAV_INTERP : object
        The surface-gravity interpolation object loaded from
        ``interface/libraries/grav_interp.pkl``.
    """
    with open("interface/libraries/grav_interp.pkl", "rb") as grav_lib:
        GRAV_INTERP = pkl.load(grav_lib)
        return GRAV_INTERP


def normalize_synth_spectrum(synth_wave: ArrayLike, synth_flux: ArrayLike) -> np.ndarray:
    """
    Normalize a synthetic spectrum using GISIC continuum fitting.

    This function applies the GISIC normalization algorithm across multiple
    `sigma` values (defined in the config) to generate continuum fits for a
    synthetic spectrum. The median of all continuum estimates is used to
    normalize the input flux.

    Parameters
    ----------
    synth_wave : array_like
        Wavelength values of the synthetic spectrum.
    synth_flux : array_like
        Corresponding flux values of the synthetic spectrum.

    Returns
    -------
    synth_norm : ndarray
        The continuum-normalized synthetic flux array. Flux values are scaled
        such that the continuum is near unity. Values below 0 or above 2 are
        clipped to 1.0 for stability.
    """
    cont_array = []
    for SIGMA in config.SIGMA:
        _, _, cont_synth_flux = normalize(
            synth_wave,
            synth_flux,
            sigma=SIGMA,
            k=config.k,
            cahk=config.cahk,
            band_check=config.band_check,
            flux_min=config.flux_min,
            boost=config.boost,
        )
        cont_array.append(cont_synth_flux)
    cont = np.median(np.array(cont_array), axis=0)
    synth_norm = np.divide(synth_flux, cont)

    if len(synth_norm[synth_norm < 0.0]) > 1:
        synth_norm[synth_norm < 0.0] = 1.0

    if len(synth_norm[synth_norm > 2.0]) > 1:
        synth_norm[synth_norm > 2.0] = 1.0

    return synth_norm


def ln_chi_square_sigma(flux: ArrayLike, synth: ArrayLike, xi: ArrayLike) -> float:
    """
    Calculate truncated log chi-square probability distribution function for an absorption feature.

    The function allows evaluating how well the synthetic spectrum matches the observed one by comparing observed flux values to synthetic (model) flux
    values using a version of the chi-square formula. The uncertainty is
    based on the signal-to-noise ratio and is adjusted using the model flux.

    Parameters
    ----------
    flux : array_like
        The observed flux values.
    synth : array_like
        The synthetic (model) flux values, matched in wavelength to the observed flux.
    xi : array_like or float
        The inverse signal-to-noise ratio (1 / SNR). This gets multiplied by the
        synthetic flux to get the uncertainty for each point.

    Returns
    -------
    float
        A number that tells us how likely the model is to match the data.
        If the result is not valid (e.g., negative), it returns -infinity.
    """
    dof = len(flux) - 1
    chi2 = np.square(np.divide(flux - synth, xi * synth)).sum()
    if chi2 > 0.0:
        return (0.5 * dof - 1) * np.log(chi2) - 0.5 * chi2
    else:
        return -np.inf


def CAII_CH_CHI_LH(
    obs: pd.DataFrame,
    synth: Dict[str, ArrayLike],
    CA_BOUNDS: Tuple[float, float],
    CH_BOUNDS: Tuple[float, float],
    CA_XI: float,
    CH_XI: float,
) -> float:
    """
    Calculate how well the synthetic spectrum matches the observed data
    in the Ca II and CH wavelength regions.

    Parameters
    ----------
    obs : DataFrame
        Observed spectrum with "wave" and "norm" columns.
    synth : dict
        Synthetic spectrum with:
        - "wave": array of wavelengths
        - "norm": array of normalized flux values
    CA_BOUNDS : tuple
        Wavelength range for the Ca II region (min, max).
    CH_BOUNDS : tuple
        Wavelength range for the CH region (min, max).
    CA_XI : float
        Inverse signal-to-noise ratio (1 / SNR) for the Ca II region.
    CH_XI : float
        Inverse signal-to-noise ratio (1 / SNR) for the CH region.

    Returns
    -------
    float
        The total log-likelihood score for both regions. Higher values
        mean the synthetic model matches the data better.
    """
    synth_function = interp1d(synth["wave"], synth["norm"])

    CA_FRAME = obs[obs["wave"].between(*CA_BOUNDS, inclusive="both")]
    CH_FRAME = obs[obs["wave"].between(*CH_BOUNDS, inclusive="both")]

    CHI_CA = ln_chi_square_sigma(CA_FRAME["norm"], synth_function(CA_FRAME["wave"]), CA_XI)
    CHI_CH = ln_chi_square_sigma(CH_FRAME["norm"], synth_function(CH_FRAME["wave"]), CH_XI)

    return CHI_CA + CHI_CH
