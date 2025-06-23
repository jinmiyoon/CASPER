from typing import Optional

import numpy as np
from numpy.typing import ArrayLike
from scipy.stats import beta

native_bounds = {
    "teff": [4000.0, 5500.0],
    "feh": [-4.5, -1.0],
    "cfe": [-0.5, 4.5],
}


# we have multiple versions of this function instead of importing them
def ln_chi_square_sigma(flux: ArrayLike, synth: ArrayLike, sigma: float) -> float:
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
    chi = np.square(np.divide(flux - synth, sigma * synth)).sum()

    if (chi > 0) and np.isfinite(chi):
        return (0.5 * dof - 1) * np.log(chi) - 0.5 * chi
    else:
        return -np.inf


def teff_lnprior(X: float, mean: float, sigma: float) -> float:
    """
    Compute the logarithm of the Gaussian prior for effective temperature (Teff).

    This represents the log prior probability of Teff assuming a normal distribution
    centered on the photometric Teff with a given uncertainty.

    Parameters
    ----------
    X : float
        Trial effective temperature (Teff).
    mean : float
        Mean of the prior distribution (e.g., photometric Teff).
    sigma : float
        Standard deviation of the prior (e.g., uncertainty in photometric Teff).

    Returns
    -------
    float
        The log prior probability for the given Teff.
    """
    return -np.log(sigma) - 0.5 * np.square(np.divide(X - mean, sigma))


def sigma_lnprior(sigma: float, alpha_value: float, beta_value: float) -> float:
    """
    Compute the log prior probability of sigma using the Beta distribution.

    Uses the Beta PDF to estimate the prior probability of `sigma`, assuming
    `sigma` lies between 0 and 1 (or is normalized). This function returns the
    log of the probability, or negative infinity if the value is out of bounds
    or has zero probability.

    Parameters
    ----------
    sigma : float
        The standard deviation parameter to evaluate (should be in [0, 1]).
    alpha_value : float
        Alpha parameter of the Beta distribution.
    beta_value : float
        Beta parameter of the Beta distribution.

    Returns
    -------
    float
        Log of the prior probability. Returns -inf if the probability is zero or invalid.
    """
    prob = beta.pdf(abs(sigma), alpha_value, beta_value)

    if prob > 0.0:
        return np.log(prob)

    return -np.inf


def default_param_edges(teff: float, feh: float, carbon: float, sigma_array: ArrayLike) -> float:
    """
    Apply hard bounds to Teff, [Fe/H], [C/Fe], and sigma parameters using native grid limits.

    This function checks whether the input parameters fall within the predefined native bounds
    used by the MASTER grid. It also validates that all sigma values are within [0.0, 1.0].

    Parameters
    ----------
    teff : float
        Effective temperature to check.
    feh : float
        Metallicity [Fe/H] to check.
    carbon : float
        Carbon-to-iron ratio [C/Fe] to check.
    sigma_array : ArrayLike
        Array of sigma values (e.g., for Ca II, CH, and C2 features).
        Each value must be between 0.0 and 1.0.

    Returns
    -------
    float
        -np.inf if any parameter is out of bounds; otherwise 0.0 if all values are valid.
    """
    if (teff < native_bounds["teff"][0]) or (teff > native_bounds["teff"][1]):
        return -np.inf

    if (feh < native_bounds["feh"][0]) or (feh > native_bounds["feh"][1]):
        return -np.inf

    if (carbon < native_bounds["cfe"][0]) or (carbon > native_bounds["cfe"][1]):
        return -np.inf

    for item in sigma_array:
        if (item < 0.0) or (item > 1.0):
            return -np.inf
    # is else statment correct of can it be
    # return 0.0
    return 0.0


def param_edges(
    teff: float,
    feh: float,
    carbon: float,
    sigma_array: ArrayLike,
    mcmc_bounds: Optional[dict] = None,
    bounds: str = "default",
) -> float:
    """
    Apply parameter bounds based on search phase ("default" or "final").

    In the "final" phase, MCMC-specific bounds are applied to Teff, [Fe/H], and [C/Fe].
    Regardless of phase, this function checks hard bounds on all parameters using
    `default_param_edges`, including sigma restrictions.

    Parameters
    ----------
    teff : float
        Effective temperature to check.
    feh : float
        Metallicity [Fe/H] to check.
    carbon : float
        Carbon-to-iron ratio [C/Fe] to check.
    sigma_array : ArrayLike
        Array of sigma values to validate (each must be in the range [0.0, 1.0]).
    mcmc_bounds : dict, optional
        Dictionary of MCMC parameter bounds with keys "teff", "feh", and "cfe",
        each mapping to a (min, max) tuple.
    bounds : {"default", "final"}, optional
        Which bound strategy to use. If "final", apply MCMC bounds before
        falling back to default grid and sigma bounds.

    Returns
    -------
    float
        -np.inf if any parameter is out of bounds; otherwise 0.0 if all are valid.
    """
    if bounds == "final":
        if (teff < mcmc_bounds["teff"][0]) or (teff > mcmc_bounds["teff"][1]):
            return -np.inf

        if (feh < mcmc_bounds["feh"][0]) or (feh > mcmc_bounds["feh"][1]):
            return -np.inf

        if (carbon < mcmc_bounds["cfe"][0]) or (carbon > mcmc_bounds["cfe"][1]):
            return -np.inf

    return default_param_edges(teff, feh, carbon, sigma_array)


def default_feh_cfe_param_edges(feh: float, carbon: float) -> float:
    """
    Apply hard bounds to [Fe/H] and [C/Fe] parameters using native grid limits.

    Parameters
    ----------
    feh : float
        Metallicity [Fe/H] value to check.
    carbon : float
        Carbon-to-iron ratio [C/Fe] value to check.

    Returns
    -------
    float
        -np.inf if either parameter is out of bounds; otherwise 0.0.
    """
    if (feh < native_bounds["feh"][0]) or (feh > native_bounds["feh"][1]):
        return -np.inf

    if (carbon < native_bounds["cfe"][0]) or (carbon > native_bounds["cfe"][1]):
        return -np.inf

    return 0.0


def feh_cfe_param_edges(feh: float, carbon: float, mcmc_bounds: Optional[dict], bounds: str = "default") -> float:
    """
    Apply bounds to [Fe/H] and [C/Fe] based on MCMC or default limits.

    If bounds is set to "final", the function first checks against
    MCMC-provided parameter limits. Regardless of bounds mode, it
    falls back to checking hard-coded default limits.

    Parameters
    ----------
    feh : float
        Metallicity [Fe/H] value to check.
    carbon : float
        Carbon-to-iron ratio [C/Fe] value to check.
    mcmc_bounds : dict or None
        Dictionary with keys "feh" and "cfe" mapping to (min, max) tuples.
    bounds : str, optional
        Bounds mode. If "final", apply MCMC bounds before default checks.

    Returns
    -------
    float
        -np.inf if either parameter is out of bounds; otherwise 0.0.
    """
    if bounds == "final":
        if (feh < mcmc_bounds["feh"][0]) or (feh > mcmc_bounds["feh"][1]):
            return -np.inf

        if (carbon < mcmc_bounds["cfe"][0]) or (carbon > mcmc_bounds["cfe"][1]):
            return -np.inf
    return default_feh_cfe_param_edges(feh, carbon)
