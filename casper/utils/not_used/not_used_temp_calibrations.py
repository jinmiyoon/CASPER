# not used
from typing import Tuple

import numpy as np
import pandas as pd


def Alonso(Frame: pd.DataFrame) -> Tuple[float, ...]:
    """
    Estimate effective temperatures using empirical color-Teff calibrations.

    This function computes effective temperatures from several color indices:
    B to V, V to R, V to K, J-Ks, and J to H. It assumes a default metallicity
    of [Fe/H] = -3.5 and is based on photometric zero-point corrected magnitudes.

    Parameters
    ----------
    Frame : pd.DataFrame
        A DataFrame containing the following keys:
        - "BV0", "V0", "R0", "Kmag0", "Jmag0", "Hmag0"

    Returns
    -------
    Tuple[float, ...]
        A tuple of estimated Teff values (in Kelvin) from each color index:
        (Teff_BV, Teff_VR, Teff_VK, Teff_JK, Teff_JH)

    Notes
    -----
    Based on Alonso et al. (1995), "The empirical scale of temperatures of the
    low main sequence" for stars ranging from spectral types F0 to K5V.
    """
    FEH = -3.50

    # B - V
    BV = Frame["BV0"]

    if 0.30 <= BV and BV <= 0.8:
        Teff = 0.541 + 0.533 * BV + 0.007 * np.power(BV, 2) - 0.019 * BV * FEH - 0.047 * FEH - 0.011 * np.power(FEH, 2)

        Teff_BV = 5040.0 / Teff

    else:
        Teff_BV = np.nan

    # V - R
    # Sigma = 0.015
    VR = Frame["V0"] - Frame["R0"]
    if 0.40 <= VR and VR <= 0.6:
        Teff = 0.474 + 0.755 * VR + 0.005 * np.power(VR, 2) + 0.003 * VR * FEH - 0.027 * FEH - 0.007 * np.power(FEH, 2)
        Teff_VR = 5040.0 / Teff

    elif 0.6 < VR and VR <= 0.70:
        Teff = 0.524 + 0.724 * VR - 0.082 * np.power(VR, 2) - 0.166 * VR * FEH + 0.074 * FEH - 0.009 * np.power(FEH, 2)
        Teff_VR = 5040.0 / Teff

    else:
        Teff_VR = np.nan

    # V - K
    VK = 0.993 * (Frame["V0"] - Frame["Kmag0"]) + 0.050
    if 1.1 <= VK and VK <= 1.6:
        Teff = 0.555 + 0.195 * VK + 0.013 * np.power(VK, 2) - 0.008 * VK * FEH + 0.009 * FEH - 0.002 * np.power(FEH, 2)
        Teff_VK = 5040.0 / Teff

    elif 1.6 <= VK and VK <= 2.2:
        Teff = 0.566 + 0.217 * VK - 0.003 * np.power(VK, 2) - 0.024 * VK * FEH + 0.037 * FEH - 0.002 * np.power(FEH, 2)
        Teff_VK = 5040.0 / Teff

    else:
        Teff_VK = np.nan

    # J - K
    # Sigma = 0.025
    JK = 0.910 * (Frame["Jmag0"] - Frame["Kmag0"]) + 0.08
    if 0.2 <= JK and JK <= 0.6:
        Teff = 0.582 + 0.799 * JK + 0.085 * np.power(JK, 2)
        Teff_JK = 5040.0 / Teff

    else:
        Teff_JK = np.nan

    # J - H
    # Sigma = 0.030
    JH = 0.942 * (Frame["Jmag0"] - Frame["Hmag0"]) - 0.010
    if 0.15 <= JH and JH <= 0.45:
        Teff = 0.587 + 0.922 * JH + 0.218 * np.power(JH, 2) + 0.016 * JH * FEH
        Teff_JH = 5040.0 / Teff

    else:
        Teff_JH = np.nan

    return Teff_BV, Teff_VR, Teff_VK, Teff_JK, Teff_JH


def Bergeat_Frame(Frame: pd.DataFrame) -> np.ndarray:
    """
    Estimate effective temperatures using color-Teff calibrations for carbon-rich stars.

    This function calculates effective temperatures based on V-K, J-Ks, and H-K
    color indices using empirical log-linear relations.

    Parameters
    ----------
    Frame : pd.DataFrame
        A DataFrame containing the following zero-point corrected magnitudes:
        - "V0", "Jmag0", "Hmag0", "Kmag0"

    Returns
    -------
    np.ndarray
        Array of effective temperatures in Kelvin:
        [Teff_VK, Teff_JK, Teff_HK]

    Notes
    -----
    Based on Bergeat et al. (2001): ["Effective Temperatures of Carbon-rich Stars"](https://ui.adsabs.harvard.edu/abs/2001A%26A...369..178B/abstract)
    """

    # V - K
    CIj0 = Frame["V0"] - Frame["Kmag0"]
    if CIj0 <= 7.0:
        logT_VK = -0.079 * CIj0 + 3.91

    elif CIj0 >= 0.7:
        logT_VK = -0.061 * CIj0 + 3.79

    # J - K
    CIj0 = Frame["Jmag0"] - Frame["Kmag0"]
    if CIj0 <= 2.1:
        logT_JK = -0.184 * CIj0 + 3.74
    elif CIj0 >= 2.1:
        logT_JK = -0.109 * CIj0 + 3.59

    # H - K
    CIj0 = Frame["Hmag0"] - Frame["Kmag0"]
    if CIj0 <= 0.86:
        logT_HK = -0.287 * CIj0 + 3.60
    elif CIj0 >= 0.86:
        logT_HK = -0.169 * CIj0 + 3.50

    return np.power(10, [logT_VK, logT_JK, logT_HK])
