### Author: Devin Whitten
### Main driver for normalization routine, intended for SEGUE medium-resolution spectra.

# Jul 2020: Jinmi Yoon
# normalize() requires several parameters for both observed spectra (sigma) and continuum finding (k, s)
# Refer to scipy.interpolate.splrep for more details of k, s.
# s : A smoothing condition. This controls smoothing the synthetic fit.
#     Larger s means more smoothing while smaller values of s indicate less smoothing so more wiggly.
#     The default value for s is s=12.
# k : the degree of the spline fit. It is recommended to use cubic splines. Even values of k
#     should be avoided especially with small s values. 1 <= k <= 5
# sigma: a smoothing factor for the flux, with using a Gaussian filter.
# Refer to scipy.ndimage.gaussian_filter(sigma)


from typing import Any, Dict, Tuple, Union

import numpy as np

from .spectrum import Spectrum


def normalize(
    wavelength: np.ndarray,
    flux: np.ndarray,
    sigma: int = 30,
    k: int = 3,
    s: int = 12,
    cahk: bool = False,
    band_check: bool = True,
    flux_min: float = 70,
    boost: bool = True,
    return_points: bool = False,
) -> Union[Tuple[np.ndarray, np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]]:
    """
    Normalize a spectrum using spline fitting based on inflection segment detection.

    Parameters
    ----------
    wavelength : np.ndarray
        Array of wavelength values.
    flux : np.ndarray
        Array of flux values corresponding to the input wavelengths.
    sigma : int, optional
        Gaussian smoothing kernel width for segment generation.
    k : int, optional
        Degree of the spline for continuum fitting.
    s : int, optional
        Spline smoothing factor.
    cahk : bool, optional
        Whether to apply special treatment to the Ca H&K region.
    band_check : bool, optional
        Whether to exclude strong band regions during continuum estimation.
    flux_min : float, optional
        Minimum threshold for flux filtering.
    boost : bool, optional
        Whether to boost the flux points during continuum point definition.
    return_points : bool, optional
        If True, also return the continuum anchor points used for spline fitting.

    Returns
    -------
    tuple
        If `return_points` is False:
            - wavelength : np.ndarray
            - normalized_flux : np.ndarray
            - continuum : np.ndarray

        If `return_points` is True:
            - wavelength : np.ndarray
            - normalized_flux : np.ndarray
            - continuum : np.ndarray
            - points : dict
                Dictionary with keys "wavelength" and "flux" for the continuum anchor points.
    """

    spec = Spectrum(wavelength, flux)
    spec.generate_inflection_segments(sigma=sigma, cahk=cahk, band_check=band_check, flux_min=flux_min)
    spec.assess_segment_variation()
    spec.define_cont_points(boost=boost)
    spec.set_segment_continuum()
    spec.set_segment_midpoints()

    spec.spline_continuum(k=k, s=s)
    spec.normalize()

    if return_points:
        return (
            np.array(spec.wavelength),
            np.array(spec.flux_norm),
            np.array(spec.continuum),
            {"wavelength": spec.midpoints, "flux": spec.fluxpoints},
        )

    else:
        return np.array(spec.wavelength), np.array(spec.flux_norm), np.array(spec.continuum)
