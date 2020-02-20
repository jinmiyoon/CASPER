import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
from scipy.interpolate import interp1d
import sys



#########
KP_BOUNDS = {"K6"  : [3930.7, 3936.7],
             "K12" : [3927.7, 3939.7],
             "K18" : [3924.7, 3942.7]}




def GBAND_QUAD(wave, flux, bounds = [4222, 4322]):
    ## takes normalized flux and computes a quick G-band

    func = interp1d(wave, 1. - flux)


    return integrate.quad(func, bounds[0], bounds[1], limit=1000, points=list(wave[(wave > bounds[0]) & (wave <  bounds[1])]))[0]


def GBAND_vanilla(wave, flux, bounds = [4222, 4322]):
    trim = flux[(wave > bounds[0]) & (wave < bounds[1])]

    return (1-trim).sum()





def CAII_K6(wave, flux):
    #3930.7 - 3936.7
    func = interp1d(wave, 1.-flux)
    return integrate.quad(func, 3930.7, 3936.7, limit=200, points = wave[(wave > 3930.7) & (wave < 3936.7)])[0]

def CAII_K12(wave, flux):
    #3927.7 - 3939.7
    func = interp1d(wave, 1.-flux)
    return integrate.quad(func, 3927.7, 3939.7, limit=200, points = wave[(wave > 3927.7) & (wave < 3939.7)])[0]


def CAII_K18(wave, flux):
    #3924.7 - 3942.7
    func = interp1d(wave, 1.-flux)
    return integrate.quad(func, 3924.7, 3942.7, limit=200, points = wave[(wave > 3924.7) & (wave < 3942.7)])[0]


####### integrate.quad is being weird with the subdivision limit.
### here's the lame versions of CAII_K##


def CAII_K6_v(wave, flux):
    #3930.7 - 3936.7

    trim = flux[(wave > 3930.7) & (wave < 3936.7)]
    return (1.-trim).sum()

def CAII_K12_v(wave, flux):
    #3927.7 - 3939.7

    trim = flux[(wave > 3927.7) & (wave < 3939.7)]
    return (1. - trim).sum()


def CAII_K18_v(wave, flux):
    #3924.7 - 3942.7

    trim = flux[(wave > 3927.7) & (wave < 3939.7)]
    return (1. - trim).sum()

###############################################################

def get_KP_band(spectrum):
    ### simply return the CAII band range for the chi fit, based on Beers 1999
    ### updated to utilize the spectrum.Spectrum() class

    KP_BOUNDS = {"K6"  : [3930.7, 3936.7],
                 "K12" : [3927.7, 3939.7],
                 "K18" : [3924.7, 3942.7]}


    K6 =  CAII_K6(spectrum.frame['wave'], spectrum.frame['norm'])
    K12 = CAII_K12(spectrum.frame['wave'], spectrum.frame['norm'])
    K18 = CAII_K18(spectrum.frame['wave'], spectrum.frame['norm'])


    if K6 <= 2.:
        print('\t recommending K6 bounds')
        return KP_BOUNDS['K6']

    elif (K6 > 2.) and (K12 <= 5. ):
        print('\t recommending K12 bounds')
        return KP_BOUNDS['K12']

    elif K18 > 5.:
        print("\t recommending K18 bounds")
        return KP_BOUNDS['K18']

    else:
        ### this shouldn't ever happen really
        print("warning: error in CAII_KP")

        return np.nan



def set_CH_procedure(spectrum):
    ## Measures Gband and sets carbon mode
    ##### This is intended to check whether C2 Swan band is necessary
    CH_EW = GBAND_QUAD(spectrum.frame['wave'], spectrum.frame['norm'])
    spectrum.set_GBAND(CH_EW)

    if CH_EW > 40.:
        print("\t recommending CH+C2 procedure")
        spectrum.set_carbon_mode('CH+C2')

    else:
        print("\t recommending CH procedure")
        spectrum.set_carbon_mode('CH')

    return






def CAII_KP(wave, flux):
    ## Following the Beers 1999
    K6 =  CAII_K6(wave, flux)
    K12 = CAII_K12(wave, flux)
    K18 = CAII_K18(wave, flux)


    if K6 <= 2.:
        return K6

    elif (K6 > 2.) and (K12 <=5):
        return K12

    elif K18 > 5.:
        return K18

    else:
        print("warning: error in CAII_KP")
        return np.nan



def CAII_H(wave, flux):
    trim = flux[(wave > 3927.7) & (wave < 3939.7)]
