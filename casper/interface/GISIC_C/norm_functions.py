# Author: Devin Whitten
# Date: Nov 12, 2016

import numpy as np
import pandas as pd
import scipy.interpolate as interp


def in_molecular_band(wl, tol=10):
    # print("Checking wavelength for band", wl)
    ### Checks to see if wavelength is within unacceptable limits of known bands
    """
    bands = {"gband": [4200., 4400.],
             "C2_O":  [4100, 4200],
             "C2_N":    [5060., 5180.]}
    """
    bands = {"gband": [4250.0, 4318.0], "C2_N": [4100.0, 4220.0], "C2": [4550.0, 4750.0]}
    #             "C2_N":    [5060., 5180.]}

    for band in bands:
        # print(band, bands[band])
        if (wl > bands[band][0]) & (wl < bands[band][1]):
            # print("\t in ", band)
            return True

    return False
