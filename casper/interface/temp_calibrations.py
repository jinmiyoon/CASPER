from typing import Optional

import numpy as np
import pandas as pd


def Hernandez(JK: float, FEH: float = -2.5, CLASS: Optional[str] = None) -> float:
    """
    Compute effective temperature (Teff) using the Hernandez Calibration.

    This function estimates stellar effective temperature from the (J-Ks) color and
    metallicity [Fe/H], using separate calibrations for GIANT and DWARF stars.

    Parameters
    ----------
    JK : float
        The (J-Ks) color index.
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
    Based on the infrared flux method from [Hernandez et al. (2009)](https://ui.adsabs.harvard.edu/abs/2009A%26A...497..497G/abstract).
    Reference: J-Ks in Table 5 and Eq. (10)
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

    This function calculates stellar effective temperature (Teff) from the J-Ks
    color index and metallicity [Fe/H], using an empirical formula.
    It is valid for 0.07 ≤ JK ≤ 0.80.

    Parameters
    ----------
    JK : float
        The J-Ks color index.
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
    Based on the calibration from [Casagrande et al. (2010)](https://ui.adsabs.harvard.edu/abs/2010A%26A...512A..54C/abstract).
    Reference: J-Ks in Table 4 and Eq. (3)
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
    Estimate effective temperature using the Bergeat calibration.

    This function calculates Teff using the J-Ks color and [Fe/H], following the approach from Bergeat et al.

    Parameters
    ----------
    JK : float
        The J-Ks color index.

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


def Fukugita(gr: float) -> float:
    """
    Estimate effective temperature from g-r color index
    using the Fukugita et al. (2011) relation.

    Parameters
    ----------
    gr : float
        The g-r color index.

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
    J-Ks and g-r color indices. It then attempts to determine an adopted Teff value.

    Parameters
    ----------
    JK : float
        J-Ks color index.
    gr : float
        g-r color index.
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
