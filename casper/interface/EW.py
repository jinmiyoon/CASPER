import config
import numpy as np
import scipy.integrate as integrate
from numpy.typing import ArrayLike
from scipy.interpolate import interp1d


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

    print("flux_bounds_max = ", flux_bounds_max)

    if flux_bounds_max < 1.0:
        EW_subtract = (1.0 - flux_bounds_max) * (bounds[1] - bounds[0])
    else:
        EW_subtract = 0.0
    print("EW_subtract=  ", EW_subtract)

    return integrate.quad(
        func, bounds[0], bounds[1], limit=1000, points=list(wave[(wave > bounds[0]) & (wave < bounds[1])])
    )[0], EW_subtract


def CAII_K6(wave, flux):
    func = interp1d(wave, 1.0 - flux)
    return integrate.quad(func, 3930.7, 3936.7, limit=200, points=wave[(wave > 3930.7) & (wave < 3936.7)])[0]


def CAII_K12(wave, flux):
    # 3927.7 - 3939.7
    func = interp1d(wave, 1.0 - flux)
    return integrate.quad(func, 3927.7, 3939.7, limit=200, points=wave[(wave > 3927.7) & (wave < 3939.7)])[0]


def CAII_K18(wave, flux):
    # 3924.7 - 3942.7
    func = interp1d(wave, 1.0 - flux)
    return integrate.quad(func, 3924.7, 3942.7, limit=200, points=wave[(wave > 3924.7) & (wave < 3942.7)])[0]


####### integrate.quad is being weird with the subdivision limit.
### here's the lame versions of CAII_K##


def CAII_K6_v(wave, flux):
    # 3930.7 - 3936.7

    trim = flux[(wave > 3930.7) & (wave < 3936.7)]
    return (1.0 - trim).sum()


def CAII_K12_v(wave, flux):
    # 3927.7 - 3939.7

    trim = flux[(wave > 3927.7) & (wave < 3939.7)]
    return (1.0 - trim).sum()


def CAII_K18_v(wave, flux):
    # 3924.7 - 3942.7

    trim = flux[(wave > 3927.7) & (wave < 3939.7)]
    return (1.0 - trim).sum()


###############################################################


def get_KP_band(spectrum):
    ### simply return the CAII band range for the chi fit, based on Beers 1999
    ### updated to utilize the spectrum.Spectrum() class
    KP_BOUNDS = config.KP_BOUNDS

    K6 = CAII_K6(spectrum.frame["wave"], spectrum.frame["norm"])
    K12 = CAII_K12(spectrum.frame["wave"], spectrum.frame["norm"])
    K18 = CAII_K18(spectrum.frame["wave"], spectrum.frame["norm"])

    if K6 <= 2.0:
        print("\t recommending K6 bounds")
        return KP_BOUNDS["K6"]

    elif (K6 > 2.0) and (K12 <= 5.0):
        print("\t recommending K12 bounds")
        return KP_BOUNDS["K12"]

    elif K18 > 5.0:
        print("\t recommending K18 bounds")
        return KP_BOUNDS["K18"]

    else:
        ### this shouldn't ever happen really
        print("warning: error in CAII_KP")

        return np.nan


def set_CH_procedure(spectrum):
    ## Measures Gband and sets carbon mode
    ##### This is intended to check whether C2 Swan band is necessary
    CH_EW, EW_subtract = GBAND_QUAD(spectrum.frame["wave"], spectrum.frame["norm"])
    spectrum.set_GBAND(CH_EW)
    # print("CH_EW, EW_subtract at EW.py = ", CH_EW, EW_subtract)

    ############################################################################
    # Revised by Jinmi Yoon, July 17 2020
    # The default CH_EW =40 was used for the Yoon+2020 paper,
    # but I realized that EW changes depending on the level of continuum.
    # So it has to change a bit to prevent an unnecessarily large EW value
    # to switch the mode. I meant to modify GBAND_QUAD calculation slightly to
    # tackle the problem with this issue.
    # However, the problem is that this function appears to be used other places.
    # So I decided to change critieria here by changing CH_EW value based on
    # the normalization level. First, I find a highest flux point, flux_max.
    # If flux_max does not reach 1.0, I subtract area from 1.0 to flux_max level
    # from CH_EW. To do so I define flux_bounds_max in GBAND_QUAD and calculate
    # this area and feed this number in this procedure.
    #
    # reduced_CH_EW = CH_EW - EW_subtract
    # if reduced_CH_EW > 45.:
    # I decided to keep Devin's procedure because the synthetic spectra at
    # this band indeed lower than 1.0 level.
    ############################################################################
    # Devin originally used CH_EW >40 for switching however, it depends on
    # the normalization though it is likely to be a minor difference.

    print("CH_EW= %5.2f" % CH_EW)

    ##########################################################################
    #  09/09/2020, J. Yoon
    # I modified this procedure because I want to have freedom to
    # set carbon_mode in input file for diagnosis of carbon mode.
    # If carbon_mode is missing in input files,
    # then it will use the CH_EW value for setting carbon_mode.
    ###########################################################################

    if spectrum.INPUT_CARBON_MODE == "CH":
        print("\t using the input {0} carbon_mode".format(spectrum.INPUT_CARBON_MODE))
        spectrum.set_carbon_mode("CH")

    elif spectrum.INPUT_CARBON_MODE == "CH+C2":
        print("\t using the input {0} carbon_mode: ".format(spectrum.INPUT_CARBON_MODE))
        spectrum.set_carbon_mode("CH+C2")

    else:
        # if CH_EW > 55.:
        if CH_EW > 40.0:
            print("\t recommending CH+C2 procedure")
            spectrum.set_carbon_mode("CH+C2")

        else:
            print("\t recommending CH procedure")
            spectrum.set_carbon_mode("CH")

    return


def CAII_KP(wave, flux):
    ## Following the Beers 1999
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
        print("warning: error in CAII_KP")
        return np.nan


def CAII_H(wave, flux):
    trim = flux[(wave > 3927.7) & (wave < 3939.7)]
