### Author: Devin Whitten
### Main driver for normalization routine, intended for SEGUE medium-resolution spectra.

# Jul 2020: Jinmi Yoon
# normalize() requres several parameters.
# Refer to scipy.interpolate.splrep for more details of these parameters.
# s : A smoothing condition. This gives how smooth my fit would be.
#     Larger s means more smoothing while smaller values of s indicate less smoothing.
#     The default value for s is s=12.
# k : the degree of the spline fit. It is recommended to use cubic splines. Even values of k
#     should be avoided especially with small s values. 1 <= k <= 5

# I would like to set k and s outside of this code, perhaps in main.py or other parameter file or casper pa

import numpy as np
import pandas as pd
import sys, os
#sys.path.append("interface")
#from spectrum import Spectrum
from astropy.io import fits

from GISIC_C.spectrum import Spectrum

def normalize(wavelength, flux, sigma=30, k=3, s=12, cahk=False, band_check=True, flux_min=80, boost=True, return_points=False):
    # flux_min =70 percentile default where wavelength region
    spec = Spectrum(wavelength, flux)
    spec.generate_inflection_segments(sigma=sigma, cahk=cahk, band_check = band_check, flux_min=flux_min)
    spec.assess_segment_variation()
    spec.define_cont_points(boost=boost)
    spec.set_segment_continuum()
    spec.set_segment_midpoints()

    spec.spline_continuum(k=k, s=s)
    spec.normalize()

    if return_points:
        return np.array(spec.wavelength), np.array(spec.flux_norm), np.array(spec.continuum), {"wavelength" : spec.midpoints, "flux": spec.fluxpoints}

    else:
        return np.array(spec.wavelength), np.array(spec.flux_norm), np.array(spec.continuum)
