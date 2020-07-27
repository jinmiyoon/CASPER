################################################################################
### Author: Devin Whitten, revised by Jinmi Yoon
### Email: devin.d.whitten@gmail.com, jinmi.yoon@gmail.com
### Institute: University of Notre Dame
################################################################################
## Functions related to the equivalent-width determinations

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


    #devin's original func below.
    func = interp1d(wave, 1. - flux)


    #################################################################################################
    ## Revised by Jinmi Yoon, July 17 2020
    ## This revision needed to accomodate the problem of improper normalization of the CH-band area.
    ## If the CH-band is too low from the continuum level, the EW estimation is not reliable
    ## and in turn influences the set_CH_procedure whether to use the CH mode or CH+C2 mode.
    ## The below calculation is needed in set_CH_procedure()
    ##
    ## I eventually decided not to use reduced_CH_EW because the synthetic spectra at the CH band
    ## is indeed lower than 1.0 level. However, I just leave the below scripts as it is for 
    ## the future debugging.
    
    func_bounds = interp1d(wave, flux)
    wave_bounds = np.arange(bounds[0], bounds[1], 0.01)
    flux_bounds = func_bounds(wave_bounds)
    flux_bounds_max = np.max(flux_bounds)

    print("flux_bounds_max = ", flux_bounds_max)

    ## If flux_boounds_max is not close to 1. (normalized level),
    ## I will calculate area subtended from 1.0 to flux_bounds_max for later subtraction from CH_EW.
    ## I need to use SNR at CH band for noise = 1./SNR but it appears very small (~0.02) so at the moment 
    ##I ignore this.
    
    if flux_bounds_max < 1. :
        EW_subtract = (1.-flux_bounds_max)*(bounds[1]-bounds[0])
    else:
        EW_subtract = 0.
    print("EW_subtract=  ", EW_subtract)

    #############################################################################################

    ## J. Yoon, 07/17/2020
    ## I added EW_subtract in return so that I can use this subtraction for set_CH_procedure() 
    
    return integrate.quad(func, bounds[0], bounds[1], limit=1000, 
                          points=list(wave[(wave > bounds[0]) & (wave <  bounds[1])]))[0], EW_subtract


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
    CH_EW, EW_subtract = GBAND_QUAD(spectrum.frame['wave'], spectrum.frame['norm'])
    spectrum.set_GBAND(CH_EW)
    #print("CH_EW, EW_subtract at EW.py = ", CH_EW, EW_subtract)  # I can print this when I debug later for EW values.
    
    ########################################################################################################
    # Revised by Jinmi Yoon, July 17 2020
    # The default CH_EW =40 was used for the Yoon+2020 paper,
    # but I realized that EW changes depending on the level of continuum.
    # So it has to be changed a bit to prevent a false EW value to switch the mode.
    # I meant to modify GBAND_QUAD calculation slightly to accomodate the problem with this issue.
    # However, the problem is that this function appears to be used other places.
    # So I decided to change critieria here by changing CH_EW value based on the normalization level.
    # First, I find a highest flux point, flux_max.
    # If flux_max does not reach 1.0, I subtract area from 1.0 to flux_max level from CH_EW.
    # To do so I define flux_bounds_max in GBAND_QUAD and calculate this area and feed this number 
    # in this procedure.
    #
    # reduced_CH_EW = CH_EW - EW_subtract
    # if reduced_CH_EW > 45.:
    # I decided to keep Devin's procedure because the synthetic spectra at this band 
    # indeed lower than 1.0 level.
    ##########################################################################################################
    # Devin originally used CH_EW >40 for switching however, it depends on the normalization 
    # though it is likely to be a minor difference. 
    
    print("CH_EW= %5.2f" %CH_EW)
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
