import numpy as np
import pandas as pd

#This script will serve as the interface for the various temperature calibrations





### Hernandez et al 2009 - infrared flux method

def Hernandez(JK, FEH=-2.5, CLASS= None):
    #### Updated to Nov 2019

    if CLASS == 'GIANT':
        print("\t using GIANT calibration in Hernandez")
        #A0 = [0.6517, 0.6312, 0.0168, -0.0381, 0.0256, 0.0013]
        A0 = [0.6517, 0.6312, 0.0168, -0.0381, 0.0256, 0.0013]

    elif CLASS == 'DWARF':
        print("\t using DWARF calibration in Hernandez")
        #A0 = [0.6524, 0.5813, 0.1225, –0.0646, 0.0370, 0.0016]
        A0 = [0.6524, 0.5813, 0.1225, -0.0646, 0.0370, 0.0016]

    else:
        print("\t Can't handle input class:  ", CLASS)
        print("\t Defaulting to GIANT")
        A0 = [0.6517, 0.6312, 0.0168, -0.0381, 0.0256, 0.0013]

    if (JK >= 0.1 and JK <= 0.95): ### JK should be 0.9 !!!!

        Teff = A0[0]  + (JK * A0[1]) +  (A0[2]*np.power(JK,2)) + (A0[3]*JK*FEH)  +  A0[4]*FEH  + A0[5]*np.power(FEH, 2)
        T_JK = 5040./Teff

    else:
        T_JK = np.nan

    return T_JK


def Casagrande(JK, FEH=-2.5, CLASS=None):
    #### Updated to Nov 2019
    if (JK >= 0.07 and JK <=0.95):
        Teff = 0.6393 + (JK*0.6104) + (0.0920*np.power(JK, 2)) + (-0.0330*JK*FEH) + (0.0291*FEH) + (0.0020*np.power(FEH,2))
        T_JK = 5040./Teff

    else:
        print("\t Casagrande Calibration out of bounds")
        T_JK = np.nan

    return T_JK


def Bergeat(JK):
    ''' J - K '''
    CIj0 = JK
    if CIj0 <= 2.1:
        logT_JK = -0.184 *CIj0 + 3.74
    elif CIj0 >= 2.1:
        logT_JK = -0.109 * CIj0 + 3.59

    return np.power(10, logT_JK)

### Alonso et al. 1995 (F0 - K5V) The empirical scale of temperatures of the low main sequence

def Alonso(Frame):
    ## Default FEH
    FEH = -3.50


    ############################################################################
    ''' B - V '''  #Sigma = 0.023
    BV = Frame['BV0']

    if (0.30 <= BV and BV <= 0.8):

        Teff = 0.541 + 0.533*BV + 0.007*np.power(BV, 2) - 0.019*BV*FEH - 0.047 * FEH - 0.011 * np.power(FEH, 2)

        Teff_BV =  5040./Teff

    else:
        Teff_BV = np.nan

    ############################################################################
    ''' V - R ''' # Sigma = 0.015
    VR = Frame['V0'] - Frame['R0']
    if (0.40 <= VR and VR <= 0.6):
        Teff  = 0.474 + 0.755*VR + 0.005*np.power(VR,2) + 0.003*VR*FEH - 0.027*FEH - 0.007*np.power(FEH, 2)
        Teff_VR = 5040./Teff

    elif (0.6 < VR and VR <= 0.70):
        Teff = 0.524 + 0.724*VR - 0.082*np.power(VR, 2) - 0.166*VR*FEH + 0.074*FEH - 0.009*np.power(FEH,2)
        Teff_VR = 5040./Teff

    else:
        Teff_VR = np.nan


    ############################################################################
    ''' V - K '''
    VK = 0.993*(Frame['V0'] - Frame['Kmag0']) + 0.050
    if (1.1 <= VK  and VK <= 1.6):
        Teff = 0.555 + 0.195 * VK + 0.013 * np.power(VK, 2) - 0.008*VK* FEH + 0.009*FEH - 0.002*np.power(FEH, 2)
        Teff_VK = 5040./Teff

    elif (1.6 <= VK and VK <= 2.2):
        Teff = 0.566 + 0.217 * VK - 0.003*np.power(VK, 2) - 0.024*VK*FEH +0.037*FEH - 0.002*np.power(FEH,2)
        Teff_VK = 5040./Teff

    else:
        Teff_VK = np.nan



    ''' J - K '''   # Sigma = 0.025
    JK = 0.910*(Frame['Jmag0'] - Frame['Kmag0']) + 0.08
    if (0.2 <= JK and JK <= 0.6):
        Teff = 0.582 + 0.799 * JK + 0.085* np.power(JK, 2)
        Teff_JK = 5040./Teff

    else:
        Teff_JK = np.nan


    ''' J - H ''' # Sigma = 0.030
    JH = 0.942*(Frame['Jmag0'] - Frame['Hmag0']) - 0.010
    if (0.15 <= JH and JH <= 0.45):
        Teff = 0.587 + 0.922 * JH + 0.218* np.power(JH, 2) + 0.016*JH*FEH
        Teff_JH = 5040./Teff

    else:
        Teff_JH = np.nan

    return Teff_BV, Teff_VR, Teff_VK, Teff_JK, Teff_JH


### Bergeat et al. 2001 : Effective Temperatures of Carbon-rich stars
def Bergeat_Frame(Frame):
    # Preconditions: Need necessary formatting on following magnitudes:
    #                "Vmag0", "Jmag0", 'Hmag0', "Kmag0"
    ''' V - K '''
    CIj0 = Frame['V0'] - Frame['Kmag0']
    if CIj0 <= 7.0:
        logT_VK = -0.079 * CIj0 + 3.91

    elif CIj0 >= 0.7:
        logT_VK = -0.061 * CIj0 + 3.79


    ''' J - K '''
    CIj0 = Frame['Jmag0'] - Frame['Kmag0']
    if CIj0 <= 2.1:
        logT_JK = -0.184 *CIj0 + 3.74
    elif CIj0 >= 2.1:
        logT_JK = -0.109 * CIj0 + 3.59


    ''' H - K '''
    CIj0 = Frame['Hmag0'] - Frame['Kmag0']
    if CIj0 <= 0.86:
        logT_HK = -0.287 *CIj0 + 3.60
    elif CIj0 >= 0.86:
        logT_HK = -0.169 * CIj0 + 3.50

    return np.power(10, [logT_VK, logT_JK, logT_HK])


### Fukugita et al. 2011
def Fukugita(gr):
 try:
     return 1.09*10000/(gr + 1.47)
 except:
     print("\t skipping (g-r)")
     return np.nan

######-------------------------------------------------------------------------
def determine_effective(TEMP_FRAME):
    print("\t setting effective temperature:")

    TEMP_FRAME = TEMP_FRAME.sort_values(by=['VALUE'])

    FINITE_FRAME = TEMP_FRAME[np.isfinite(TEMP_FRAME['VALUE'])]

    INDEX = int(len(FINITE_FRAME)/2)

    print("\t adopting : ", FINITE_FRAME.index.values[INDEX])
    value = float(FINITE_FRAME.iloc[INDEX]["VALUE"])
    #print(value)

    assert np.isfinite(value), "\t ERROR, PHOTO TEMP NOT FINITE"
    TEMP_FRAME = TEMP_FRAME.append(pd.DataFrame(data = [value],
                                   columns = ['VALUE'], index=['ADOPTED']))

    #print(TEMP_FRAME)
    return TEMP_FRAME


def calibrate_temp_frame(JK, gr, FEH = -2.5, CLASS=None):
    ### Just run as many of them as you want
    print("\t calibrating temperature frame")
    if np.isfinite(JK):
        TEMP_DICT = {'Casagrande': Casagrande(JK, FEH, CLASS),
                    'Hernandez': Hernandez(JK, FEH, CLASS),
                    'Bergeat': Bergeat(JK)}

    else:
        TEMP_DICT = {'Casagrande': np.nan,
                    'Hernandez': np.nan,
                    'Bergeat': np.nan}


    if np.isfinite(gr):
        TEMP_DICT['Fukugita'] = Fukugita(gr)
    else:
        TEMP_DICT['Fukugita'] = np.nan

    ### It's easier to handle a dataframe..
    TEMP_FRAME = pd.DataFrame(data = list(TEMP_DICT.values()), columns = ["VALUE"], index = TEMP_DICT.keys())

    try:
        TEMP_FRAME = determine_effective(TEMP_FRAME)
    except:
        TEMP_FRAME['ADOPTED'] = np.nan

    return TEMP_FRAME
