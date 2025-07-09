from typing import Tuple, Union

import numpy as np
import scipy.integrate as integrate
from numpy.typing import ArrayLike
from scipy.interpolate import interp1d

from casper.interface import config
from casper.utils.logger_config import setup_logger

logger = setup_logger(__name__)


def GBAND_QUAD(wave: ArrayLike, flux: ArrayLike, bounds: tuple[float, float] = config.CH_BOUNDS) -> tuple[float, float]:
    """
    Compute the equivalent width (EW) of the G-band (CH absorption) from a normalized spectrum.

    This function integrates the area of absorption in the CH band region (G-band),
    defined by `bounds`, using a normalized flux array. It also returns a correction
    factor (`EW_subtract`) in case the CH-band region is not fully normalized to 1.0.
    This correction is useful for estimating whether to use CH-only or CH+C2 mode in
    stellar carbon abundance analysis.

    Parameters
    ----------
    wave : ndarray
        Array of wavelength values corresponding to the spectrum.
    flux : ndarray
        Array of normalized flux values (should be near 1.0 in continuum).
    bounds : tuple of float
        The wavelength interval (min, max) defining the CH G-band region.

    Returns
    -------
    ew : float
        Equivalent width (EW) of the CH absorption feature.
    EW_subtract : float
        Correction factor to account for improper normalization of the CH-band.
        If the peak flux in the band is below 1.0, this value will be non-zero.
    """
    func = interp1d(wave, 1.0 - flux)
    func_bounds = interp1d(wave, flux)
    wave_bounds = np.arange(bounds[0], bounds[1], 0.01)
    flux_bounds = func_bounds(wave_bounds)
    flux_bounds_max = np.max(flux_bounds)

    logger.info(f"flux_bounds_max = {flux_bounds_max}")

    if flux_bounds_max < 1.0:
        EW_subtract = (1.0 - flux_bounds_max) * (bounds[1] - bounds[0])
    else:
        EW_subtract = 0.0
    logger.info(f"EW_subtract = {EW_subtract}")

    return integrate.quad(
        func, bounds[0], bounds[1], limit=1000, points=list(wave[(wave > bounds[0]) & (wave < bounds[1])])
    )[0], EW_subtract


def CAII_K6(wave: ArrayLike, flux: ArrayLike) -> float:
    """
    Compute the equivalent width of the Ca II K6 absorption feature.

    This function interpolates the inverted normalized flux (1 - flux)
    and integrates it over the wavelength range 3930.7 to 3936.7 Angstroms
    to estimate the equivalent width of the Ca II K6 line.

    Parameters
    ----------
    wave : array_like
        Wavelength values of the spectrum (in Angstroms).
    flux : array_like
        Normalized flux values corresponding to the wavelength array.

    Returns
    -------
    float
        The equivalent width of the Ca II K6 absorption feature, in Angstroms.
    """
    func = interp1d(wave, 1.0 - flux)
    return integrate.quad(func, 3930.7, 3936.7, limit=200, points=wave[(wave > 3930.7) & (wave < 3936.7)])[0]


def CAII_K12(wave: ArrayLike, flux: ArrayLike) -> float:
    """
    Compute the equivalent width of the Ca II K12 absorption feature.

    This function interpolates the inverted normalized flux (1 - flux)
    and integrates it over the wavelength range 3927.7 to 3939.7 Angstroms
    to estimate the equivalent width of the Ca II K12 line.

    Parameters
    ----------
    wave : array_like
        Wavelength values of the spectrum (in Angstroms).
    flux : array_like
        Normalized flux values corresponding to the wavelength array.

    Returns
    -------
    float
        The equivalent width of the Ca II K12 absorption feature, in Angstroms.
    """
    func = interp1d(wave, 1.0 - flux)
    return integrate.quad(func, 3927.7, 3939.7, limit=200, points=wave[(wave > 3927.7) & (wave < 3939.7)])[0]


def CAII_K18(wave: ArrayLike, flux: ArrayLike) -> float:
    """
    Compute the equivalent width of the Ca II K18 absorption feature.

    This function interpolates the inverted normalized flux (1 - flux)
    and integrates it over the wavelength range 3924.7 to 3942.7 Angstroms
    to estimate the equivalent width of the Ca II K18 line.

    Parameters
    ----------
    wave : ArrayLike
        Wavelength values of the spectrum (in Angstroms).
    flux : ArrayLike
        Normalized flux values corresponding to the wavelength array.

    Returns
    -------
    float
        The equivalent width of the Ca II K18 absorption feature, in Angstroms.
    """
    func = interp1d(wave, 1.0 - flux)
    return integrate.quad(func, 3924.7, 3942.7, limit=200, points=wave[(wave > 3924.7) & (wave < 3942.7)])[0]


def get_KP_band(spectrum) -> Tuple[float, float]:
    """
    Return the Ca II K-band wavelength range for chi-square fitting based on Beers (1999),
    using measurements K6, K12, and K18.

    Parameters
    ----------
    spectrum : Spectrum
        A Spectrum object with a 'frame' dictionary containing "wave" and "norm" arrays.

    Returns
    -------
    tuple of float
        Wavelength bounds (min, max) from config.KP_BOUNDS.

    Notes
    -----
    If an unexpected condition occurs, np.nan is returned,
    which does not match the declared return type.
    """
    KP_BOUNDS = config.KP_BOUNDS

    K6 = CAII_K6(spectrum.frame["wave"], spectrum.frame["norm"])
    K12 = CAII_K12(spectrum.frame["wave"], spectrum.frame["norm"])
    K18 = CAII_K18(spectrum.frame["wave"], spectrum.frame["norm"])

    if K6 <= 2.0:
        logger.info("Recommending K6 bounds")
        return KP_BOUNDS["K6"]

    elif (K6 > 2.0) and (K12 <= 5.0):
        logger.info("Recommending K12 bounds")
        return KP_BOUNDS["K12"]

    elif K18 > 5.0:
        logger.info("Recommending K18 bounds")
        return KP_BOUNDS["K18"]

    else:
        logger.warning("Warning: error in CAII_KP")
        return np.nan


def set_CH_procedure(spectrum) -> None:
    """
    Set the carbon analysis mode for the given spectrum based on G-band strength.

    This function measures the CH G-band equivalent width (CH_EW) using GBAND_QUAD.
    If a carbon mode is specified in `spectrum.INPUT_CARBON_MODE`, that mode is used.
    Otherwise, the function decides between "CH" and "CH+C2" based on the CH_EW threshold.

    Parameters
    ----------
    spectrum : Spectrum
        A Spectrum object containing a frame with "wave" and "norm",
        and methods for setting G-band and carbon mode.

    Returns
    -------
    None
        Modifies the spectrum in place.
    """
    CH_EW, EW_subtract = GBAND_QUAD(spectrum.frame["wave"], spectrum.frame["norm"])
    spectrum.set_GBAND(CH_EW)

    logger.info(f"CH_EW = {CH_EW:5.2f}")

    if spectrum.INPUT_CARBON_MODE == "CH":
        logger.info(f"Using the input {spectrum.INPUT_CARBON_MODE} carbon_mode")
        spectrum.set_carbon_mode("CH")

    elif spectrum.INPUT_CARBON_MODE == "CH+C2":
        logger.info(f"Using the input {spectrum.INPUT_CARBON_MODE} carbon_mode")
        spectrum.set_carbon_mode("CH+C2")

    else:
        if CH_EW > 40.0:
            logger.info("Recommending CH+C2 procedure")
            spectrum.set_carbon_mode("CH+C2")

        else:
            logger.info("Recommending CH procedure")
            spectrum.set_carbon_mode("CH")

    return


def CAII_KP(wave: np.ndarray, flux: np.ndarray) -> Union[float, np.float64]:
    """
    Compute the Ca II K-line strength index (KP) based on Beers et al. (1999).

    This function evaluates K6, K12, and K18 bandpasses and returns the appropriate
    KP value according to Beers' decision tree.

    Parameters
    ----------
    wave : np.ndarray
        Wavelength array of the spectrum.
    flux : np.ndarray
        Normalized flux array corresponding to the wavelengths.

    Returns
    -------
    float
        Ca II KP index value. Returns np.nan if selection logic fails.
    """
    K6 = CAII_K6(wave, flux)
    K12 = CAII_K12(wave, flux)
    K18 = CAII_K18(wave, flux)

    if K6 <= 2.0:
        return K6

    elif (K6 > 2.0) and (K12 <= 5):
        return K12

    elif K18 > 5.0:
        return K18

    else:
        logger.warning("Warning: error in CAII_KP")
        return np.nan
