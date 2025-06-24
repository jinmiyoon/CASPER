from typing import Optional, Tuple

import numpy as np
import pandas as pd


def Hernandez(JK: float, FEH: float = -2.5, CLASS: Optional[str] = None) -> float:
    """
    Compute effective temperature (Teff) using the Hernandez Calibration.

    This function estimates stellar effective temperature from the (J to K) color and
    metallicity [Fe/H], using separate calibrations for GIANT and DWARF stars.

    Parameters
    ----------
    JK : float
        The (J to K) color index.
    FEH : float, optional
        Metallicity [Fe/H] of the star. Default is -2.5.
    CLASS : str, optional
        Stellar class: "GIANT" or "DWARF". If None or unrecognized, defaults to "GIANT".

    Returns
    -------
    float
        Effective temperature in Kelvin. Returns np.nan if JK is out of the valid range.

    Notes
    -----
    Based on the infrared flux method from Hernandez et al. (2009).
    Reference: J-Ks in Table 5 and Eq. (10) from https://ui.adsabs.harvard.edu/abs/2009A%26A...497..497G/abstract
    """
    if CLASS == "GIANT":
        print("\t\t using GIANT calibration in Hernandez")
        A0 = [0.6517, 0.6312, 0.0168, -0.0381, 0.0256, 0.0013]

    elif CLASS == "DWARF":
        print("\t\t using DWARF calibration in Hernandez")
        A0 = [0.6524, 0.5813, 0.1225, -0.0646, 0.0370, 0.0016]

    else:
        print("\t\t Can't handle input class:  ", CLASS)
        print("\t\t Defaulting to GIANT")
        A0 = [0.6517, 0.6312, 0.0168, -0.0381, 0.0256, 0.0013]

    if JK >= 0.1 and JK <= 0.90:
        Teff = (
            A0[0]
            + (JK * A0[1])
            + (A0[2] * np.power(JK, 2))
            + (A0[3] * JK * FEH)
            + A0[4] * FEH
            + A0[5] * np.power(FEH, 2)
        )
        T_JK = 5040.0 / Teff

    else:
        T_JK = np.nan

    return T_JK


def Casagrande(JK: float, FEH: float = -2.5, CLASS: Optional[str] = None) -> float:
    """
    Estimate effective temperature using the Casagrande Calibration.

    This function calculates stellar effective temperature (Teff) from the J to K
    color index and metallicity [Fe/H], using an empirical formula.
    It is valid for 0.07 ≤ JK ≤ 0.80.

    Parameters
    ----------
    JK : float
        The J to K color index.
    FEH : float, optional
        Metallicity [Fe/H] of the star. Default is -2.5.
    CLASS : str, optional
        Currently unused. Included for interface compatibility with Hernandez().

    Returns
    -------
    float
        Effective temperature in Kelvin. Returns np.nan if JK is out of bounds.

    Notes
    -----
    Based on the calibration from Casagrande et al. (2010).
    Reference: J-Ks in Table 4 and Eq. (3) from https://ui.adsabs.harvard.edu/abs/2010A%26A...512A..54C/abstract
    """

    if JK >= 0.07 and JK <= 0.80:
        Teff = (
            0.6393
            + (JK * 0.6104)
            + (0.0920 * np.power(JK, 2))
            + (-0.0330 * JK * FEH)
            + (0.0291 * FEH)
            + (0.0020 * np.power(FEH, 2))
        )
        T_JK = 5040.0 / Teff

    else:
        print("\t\t Casagrande Calibration out of bounds")
        T_JK = np.nan

    return T_JK


def Bergeat(JK: float) -> float:
    """
    Estimate effective temperature using the J to K color index.

    This function calculates Teff using a log-linear calibration based on
    the J to K color index, following the approach from Bergeat et al.

    Parameters
    ----------
    JK : float
        The J to K color index.

    Returns
    -------
    float
        Effective temperature in Kelvin.

    Notes
    -----
    Based on the calibration from Bergeat et al. (2001).
    """
    CIj0 = JK
    if CIj0 <= 2.1:
        logT_JK = -0.184 * CIj0 + 3.74
    elif CIj0 >= 2.1:
        logT_JK = -0.109 * CIj0 + 3.59

    return np.power(10, logT_JK)


def Alonso(Frame: pd.DataFrame) -> Tuple[float, ...]:
    """
    Estimate effective temperatures using empirical color-Teff calibrations.

    This function computes effective temperatures from several color indices:
    B to V, V to R, V to K, J to K, and J to H. It assumes a default metallicity
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

    This function calculates effective temperatures based on V to K, J to K, and H to K
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
    Based on Bergeat et al. (2001): "Effective Temperatures of Carbon-rich Stars".
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


def Fukugita(gr: float) -> float:
    """
    Estimate effective temperature from g to r color index
    using the Fukugita et al. (2011) relation.

    Parameters
    ----------
    gr : float
        The g to r color index.

    Returns
    -------
    float
        Estimated effective temperature in Kelvin.
        Returns np.nan if input is invalid.
    """
    try:
        return 1.09 * 10000 / (gr + 1.47)
    except:
        print("\t\t skipping (g-r)")
        return np.nan


def determine_effective(TEMP_FRAME: pd.DataFrame) -> pd.DataFrame:
    """
    Determine the adopted effective temperature from a table of temperature estimates.

    This function:
    - Sorts the input DataFrame by the "VALUE" column
    - Filters out non-finite values
    - Selects the median finite value as the adopted effective temperature
    - Appends this value to the DataFrame with the index "ADOPTED"

    Parameters
    ----------
    TEMP_FRAME : pd.DataFrame
        DataFrame containing a "VALUE" column with Teff estimates.

    Returns
    -------
    pd.DataFrame
        The original DataFrame with an additional row indexed as "ADOPTED",
        containing the median finite temperature value.

    Raises
    ------
    AssertionError
        If the selected temperature value is not finite.
    """
    print("\t\t setting effective photo temperature:")

    TEMP_FRAME = TEMP_FRAME.sort_values(by=["VALUE"])

    FINITE_FRAME = TEMP_FRAME[np.isfinite(TEMP_FRAME["VALUE"])]

    INDEX = int(len(FINITE_FRAME) / 2)

    value = float(FINITE_FRAME.iloc[INDEX]["VALUE"])

    assert np.isfinite(value), "\t\t ERROR, PHOTO TEMP NOT FINITE"
    TEMP_FRAME = pd.concat([TEMP_FRAME, pd.DataFrame(data=[value], columns=["VALUE"], index=["ADOPTED"])])
    return TEMP_FRAME


def calibrate_temp_frame(JK: float, gr: float, FEH: float = -2.5, CLASS: Optional[str] = None) -> pd.DataFrame:
    """
    Build a DataFrame of calibrated effective temperature estimates using multiple color indices.

    This function runs various photometric temperature calibrations (Casagrande, Hernandez,
    Bergeat, Fukugita) depending on the availability of valid (finite) input values for
    J to K and g to r color indices. It then attempts to determine an adopted Teff value.

    Parameters
    ----------
    JK : float
        J to K color index.
    gr : float
        g to r color index.
    FEH : float, optional
        Metallicity [Fe/H]. Default is -2.5.
    CLASS : str, optional
        Stellar class, e.g., "GIANT" or "DWARF". Default is None.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the Teff estimates from each calibration
        and an additional row "ADOPTED" with the median effective temperature.
        If adoption fails, "ADOPTED" will be set to NaN.
    """

    print("\t\t calibrating temperature frame")
    if np.isfinite(JK):
        TEMP_DICT = {
            "Casagrande": Casagrande(JK, FEH, CLASS),
            "Hernandez": Hernandez(JK, FEH, CLASS),
            "Bergeat": Bergeat(JK),
        }

    else:
        TEMP_DICT = {"Casagrande": np.nan, "Hernandez": np.nan, "Bergeat": np.nan}

    if np.isfinite(gr):
        TEMP_DICT["Fukugita"] = Fukugita(gr)
    else:
        TEMP_DICT["Fukugita"] = np.nan

    TEMP_FRAME = pd.DataFrame(data=list(TEMP_DICT.values()), columns=["VALUE"], index=TEMP_DICT.keys())

    try:
        TEMP_FRAME = determine_effective(TEMP_FRAME)
    except:
        TEMP_FRAME["ADOPTED"] = np.nan

    return TEMP_FRAME
