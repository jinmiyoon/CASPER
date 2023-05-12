################################################################################
### Author: Devin Whitten
### Email: devin.d.whitten@gmail.com
### Institute: University of Notre Dame
################################################################################

#### Here are the functions for the synthetic minimization

import numpy as np
import scipy.interpolate as interp
import pandas as pd
from scipy.interpolate import interp1d
import os
import pickle as pkl
from scipy.interpolate import LinearNDInterpolator as NDLinear
import ac
import GISIC_C as GISIC

# Get synthetic flux interpolator 
def get_interp():
    ###########
    #return pkl.load(open("interface/libraries/MASTER_spec_interp.pkl", 'rb'))

    """
    # Devin's master spec interp
    with open("interface/libraries/MASTER_spec_interp.pkl", 'rb') as devin_master_lib:
        INTERPOLATOR = pkl.load(devin_master_lib)
    """
    # my New master spec interp

    with open("interface/libraries/SYNTHETIC_SPEC_R2000_INTERP.pkl", 'rb') as my_master_lib:
        INTERPOLATOR = pkl.load(my_master_lib)

    
    return INTERPOLATOR

# wave_region of synthetic spectra
def get_synth_wave():
    return np.arange(3000., 5001., 1)

# now let's normalize synthetic library with GISIC
def normalize(synth_wave, synth_flux):
    
    # print("\t\t synthetic_function: synth_wave ={}".format(synth_wave))
    # print("\t\t synthetic_function: synth_flux ={}".format(synth_flux))
    cont_array = []

    for SIGMA in np.linspace(15, 30, 10): #
        _, _, cont_synth_flux= GISIC.normalize(synth_wave, synth_flux, 
            sigma = SIGMA, k=1,cahk=False, band_check=True, flux_min=70, boost=True)
        cont_array.append(cont_synth_flux)
        
    ### average the sigma runs together
    cont = np.median(np.array(cont_array), axis=0)

    synth_norm = np.divide(synth_flux, cont)

    if len(synth_norm[synth_norm < 0.0])>1:

        synth_norm[synth_norm < 0.0] = 1.

    if len(synth_norm[synth_norm >2.0]) >1:

        synth_norm[synth_norm >2.0] = 1.
    
    #print("\t\t synthetic_function: synth_norm ={}".format(synth_norm))
    
    return synth_norm




################################################################################

def ln_chi_square_sigma(flux, synth, xi):
    ### J. Yoon 02/25/2022
    ### This function is truncated log chi square pdf.

    ### input
    #   flux : observed, synth: synthetic flux, xi: noise (the inverse of SNR for observed flux)
    #  assume the flux arrays are aligned in wavelength
    #  update: doing a transformation on here. sigma (xi) is the inverse of the signal-to-noise
    #  the sigma used in the chi square is the noise, so multiply by the current signal flux.
    #  smaller synth, means smaller effective sigma, which should prioritize the centers of the absorption features.

    dof = len(flux) - 1
    chi = np.square(np.divide(flux - synth, xi*synth)).sum()
    if chi > 0.:
        return (0.5 * dof - 1) * np.log(chi) - 0.5*chi
    else:
        return -np.inf



def CAII_CH_CHI_LH(obs, synth, CA_BOUNDS, CH_BOUNDS, CA_XI, CH_XI):
    # calculate log likelihood function for CA II and CH for MLE estimation

    #### CA_XI, CH_XI : the inverse signal to noise

    ### create linearly interpolated spectra J. Yoon 02/25/2022
    synth_function = interp1d(synth['wave'], synth['norm'])


    ### Break the segments
    CA_FRAME = obs[obs['wave'].between(*CA_BOUNDS, inclusive='both')]
    CH_FRAME = obs[obs['wave'].between(*CH_BOUNDS, inclusive='both')]

    CHI_CA = ln_chi_square_sigma(CA_FRAME['norm'], synth_function(CA_FRAME['wave']), CA_XI)
    CHI_CH = ln_chi_square_sigma(CH_FRAME['norm'], synth_function(CH_FRAME['wave']), CH_XI)


    return CHI_CA + CHI_CH


###########################################################################################
### The functions below are used in plot_functions only but not used in CASPER runs   #####
###########################################################################################



#unused func
def determine_rChi_2(spec, synth, bounds, type='both'):
    ### Just accept dataframes
    ### linearly interpolate the synthetic spectra


    #### Build the synthetic function
    synth_function = interp1d(synth['wave'], synth['norm'], kind='cubic')

    spec_trim = spec[spec['wave'].between(bounds[0], bounds[1], inclusive='both')]

    #residual = spec_trim['norm'] - synth_function(spec_trim['wave'])

    CHI = np.divide(np.square(spec_trim['norm'] - synth_function(spec_trim['wave'])), synth_function(spec_trim['wave']))
    trim = np.concatenate([CHI[spec_trim['wave'].between(3925, 3980, inclusive='both')], CHI[spec_trim['wave'].between(4222, 4322, inclusive='both')]])
    trim1 = CHI[spec_trim['wave'].between(3925, 3980, inclusive='both')]
    trim2 = CHI[spec_trim['wave'].between(4222, 4322, inclusive='both')]
    if type=="CAII":
        #FINAL = np.mean([np.mean(CHI[spec_trim['wave'].between(3925, 3980, inclusive='both')]),
        #        np.mean(CHI[spec_trim['wave'].between(4222, 4322, inclusive='both')])])
        FINAL = trim1.sum()/len(trim1)

        #print(CHI)
        return FINAL

    elif type=="both":
        FINAL = trim.sum()/len(trim)
        return FINAL

    elif type=="median":
        FINAL = np.median(trim[np.isfinite(trim)])
        return FINAL

    elif type=="mean":
        FINAL = np.mean(trim[np.isfinite(trim)])
        return FINAL

    elif type=="weight":
        FINAL = np.mean([trim1.sum()/len(trim1), trim2.sum()/len(trim2)])
        return FINAL


    elif type=="CH":


        FINAL = trim2.sum()/len(trim2)
        return FINAL


# unused func
def compute_synthetic_array(spectrum, synth_array, filenames, bounds, caHK_CH=False):
    ### Store the statistics in a dictionary or something
    temperatures = [float(name.split("T")[1].split("g")[0]) for name in filenames]

    feh = []
    for afile in filenames:
        feh.append(float(afile.split("z")[1].split("c")[0]))


    return pd.DataFrame({"temp": temperatures,
                         "chi2": [determine_rChi_2(spectrum, synth, bounds, caHK_CH) for synth in synth_array],
                         "name":filenames,
                         "feh":feh}).sort_values(by='feh')


# unused func
def determine_rChi(spec, wave, flux, bounds, type="norm"):
    ### Just accept dataframes
    ### linearly interpolate the synthetic spectra
    synth = pd.DataFrame({'wave': wave, "norm": flux})

    #### Build the synthetic function
    synth_function = interp1d(synth['wave'], synth['norm'], kind='cubic')

    spec_trim = spec[spec['wave'].between(bounds[0], bounds[1], inclusive='both')]

    CHI = np.divide(np.square(spec_trim['norm'] - synth_function(spec_trim['wave'])), synth_function(spec_trim['wave']))
    trim = np.concatenate([CHI[spec_trim['wave'].between(3925, 3980, inclusive='both')], CHI[spec_trim['wave'].between(4222, 4322, inclusive='both')]])
    trim1 = CHI[spec_trim['wave'].between(3925, 3980, inclusive='both')]
    trim2 = CHI[spec_trim['wave'].between(4222, 4322, inclusive='both')]

    if type ==  'norm':

        return trim.sum()/len(trim)

    elif type == 'weight':

        return np.average([trim1.sum()/len(trim1), trim2.sum()/len(trim2)])

    elif type == 'CAII':

        return np.median(trim1)

    elif type == "CH":

        return np.median(trim2)

    else:

        return np.mean(CHI)

# unused func
def compute_hd5f_array(spec, waves, fluxes, bounds, type='norm'):

    return [determine_rChi(spec, wave, flux, bounds, type=type) for wave, flux in zip(waves, fluxes)]


# unused func
def run(path, obs, group, bounds, type='both'):
    synth_array = [pd.read_csv(path + group + filename) for filename in os.listdir(path + group)]
    filenames = [filename for filename in os.listdir(path + group)]
    return compute_synthetic_array(obs, synth_array, filenames, bounds, type)



group_dict = {
            "GI":   {'FEH': -2.5, "CARBON": 7.9},
            "GII":  {'FEH': -3.5, "CARBON": 5.9},
            'GIII': {'FEH': -4.3, "CARBON": 7.0}
             }

group_bounds = {
            'GI'  : {"T": [4000, 5000], "FEH": [-3.5, -1.0],  'CARBON': [ 7.0, 9.0]},
            'GII' : {"T": [4000, 5000], "FEH": [-4.5, -2.0],  'CARBON': [-1.0, 1.5]},
            'GIII': {"T": [4000, 5000], "FEH": [-4.5, -3.0],  'CARBON': [ 6.0, 7.5]}
}

# unused func
### July 11th upgrade to run function
def interp_run(obs, tbounds = [4000, 5000], length=20, type='weight'):

    ### first we need to generate the archetype
    ### probably just easier to do all of the groups in the same function

    ### GI
    print("interp run")
    arch_lib = {}

    ### Load ever archetype spectrum from T= 4000 - 5000
    arch_lib['GI_D'] = arch_int['GI_D'](np.r_[[np.linspace(tbounds[0], tbounds[1], length),
                            np.ones(length) * group_dict['GI']['FEH'],
                            np.ones(length) * group_dict['GI']['CARBON']]].T
    )
    arch_lib['GI_G'] = arch_int['GI_G'](np.r_[[np.linspace(tbounds[0], tbounds[1], length),
                            np.ones(length) * group_dict['GI']['FEH'],
                            np.ones(length) * group_dict['GI']['CARBON']]].T
    )

    ### GII
    arch_lib['GII_D'] = arch_int['GII_D'](np.r_[[np.linspace(tbounds[0], tbounds[1], length),
                             np.ones(length) * group_dict['GII']['FEH'],
                             np.ones(length) * ac.cfe(group_dict['GII']['CARBON'], group_dict['GII']['FEH'])]].T
    )

    arch_lib['GII_G'] = arch_int['GII_G'](np.r_[[np.linspace(tbounds[0], tbounds[1], length),
                             np.ones(length) * group_dict['GII']['FEH'],
                             np.ones(length) * ac.cfe(group_dict['GII']['CARBON'], group_dict['GII']['FEH'])]].T
    )

    ### GIII
    arch_lib['GIII_D'] = arch_int['GIII_D'](np.r_[[np.linspace(tbounds[0], tbounds[1], length),
                              np.ones(length) * group_dict['GIII']['FEH'],
                              np.ones(length) * group_dict['GIII']['CARBON']]].T
    )
    arch_lib['GIII_G'] = arch_int['GIII_G'](np.r_[[np.linspace(tbounds[0], tbounds[1], length),
                              np.ones(length) * group_dict['GIII']['FEH'],
                              np.ones(length) * group_dict['GIII']['CARBON']]].T
    )

    print('running')

    chi_frame = {key : chi_interp_arch(obs, arch_lib[key], type=type) for key in ['GI_D', 'GI_G', 'GII_D', 'GII_G', 'GIII_D', 'GIII_G']}
    chi_frame['temp'] = np.linspace(tbounds[0], tbounds[1], length)
    return chi_frame
#    return {key: [chi_interp_arch(obs, spec, bounds = [3900, 4500]) for spec in arch_lib[key]] for key in ['GI_D', 'GI_G', 'GII_D', 'GII_G', 'GIII_D', 'GIII_G']}

# unused func
def chi_interp_arch(obs, specs, bounds = [3900, 4500], type='both'):
    ## Precondition: obs is a Dataframe, specs is an array of spectra straight from the interpolator
    #print('... doing')
    #print(len(specs))
    synth_frame = [pd.DataFrame({'wave' : get_synth_wave(), 'norm': spec}) for spec in specs]
    chi_array = []
    for synth in synth_frame:
        chi_array.append(determine_rChi_2(obs, synth, bounds, type=type))

    return chi_array


# unused func
def get_group_specs(tbounds = [4000, 5000], length=20):

    ### first we need to generate the archetype
    ### probably just easier to do all of the groups in the same function

    ### GI
    print("interp run")

    arch_lib = {}
    arch_lib['wave'] = get_synth_wave()
    ### Load ever archetype spectrum from T= 4000 - 5000
    arch_lib['GI_D'] = arch_int['GI_D'](np.r_[[np.linspace(tbounds[0], tbounds[1], length),
                            np.ones(length) * group_dict['GI']['FEH'],
                            np.ones(length) * group_dict['GI']['CARBON']]].T
    )
    arch_lib['GI_G'] = arch_int['GI_G'](np.r_[[np.linspace(tbounds[0], tbounds[1], length),
                            np.ones(length) * group_dict['GI']['FEH'],
                            np.ones(length) * group_dict['GI']['CARBON']]].T
    )

    ### GII
    arch_lib['GII_D'] = arch_int['GII_D'](np.r_[[np.linspace(tbounds[0], tbounds[1], length),
                             np.ones(length) * group_dict['GII']['FEH'],
                             np.ones(length) * ac.cfe(group_dict['GII']['CARBON'], group_dict['GII']['FEH'])]].T
    )

    arch_lib['GII_G'] = arch_int['GII_G'](np.r_[[np.linspace(tbounds[0], tbounds[1], length),
                             np.ones(length) * group_dict['GII']['FEH'],
                             np.ones(length) * ac.cfe(group_dict['GII']['CARBON'], group_dict['GII']['FEH'])]].T
    )

    ### GIII
    arch_lib['GIII_D'] = arch_int['GIII_D'](np.r_[[np.linspace(tbounds[0], tbounds[1], length),
                              np.ones(length) * group_dict['GIII']['FEH'],
                              np.ones(length) * group_dict['GIII']['CARBON']]].T
    )
    arch_lib['GIII_G'] = arch_int['GIII_G'](np.r_[[np.linspace(tbounds[0], tbounds[1], length),
                              np.ones(length) * group_dict['GIII']['FEH'],
                              np.ones(length) * group_dict['GIII']['CARBON']]].T
    )

    return arch_lib

# unused func
def determine_crit_params(spec, group_class, type='weight'):
    ## Precondition: group_class: ['GI_D', 'GI_G', 'GII_D', 'GII_G', 'etc.']
    ### assume the Group and logg class have been determined
    ### spec has already been normalized, and now we just have to find the critical Teff and [Fe/H] values.
    delta_c = 0.1
    delta_t = 100
    delta_f = 0.1
    ## Per the CVn Yoon et al. Method, we generated spectra according to each A(C) bound, in dex of 0.1
    group_bounds = {
                'GI'  : {"T": [4000, 5000], "FEH": [-3.5, -1.0],  'CARBON': [ 7.0, 9.0]},
                'GII' : {"T": [4000, 5000], "FEH": [-4.5, -2.0],  'CARBON': [-1.0, 1.5]},
                'GIII': {"T": [4000, 5000], "FEH": [-4.5, -3.0],  'CARBON': [ 6.0, 7.5]}
    }

    ## Grab the relevant interpolator bounds.
    bounds = group_bounds[group_class.split("_")[0]]

    ### Load the relevant interpolator
    interp = arch_int[group_class]

    ###  define FEH array
    FEH_ARRAY = np.arange(bounds['FEH'][0], bounds['FEH'][1] + delta_f, delta_f)
    frame_array = []
    ##
    for AC_VALUE in np.arange(bounds['CARBON'][0], bounds['CARBON'][1] + delta_c, delta_c):
        #print(AC_VALUE)
        ### for each AC_value generate FEH array for constant temperature

        for TEFF in np.arange(bounds['T'][0], bounds['T'][1] + delta_t, delta_t):
            #print(FEH_ARRAY)

            ## generate spectra of that AC and TEFF
            SYNTH_ARRAY = interp(np.c_[TEFF * np.ones(len(FEH_ARRAY)),
                                FEH_ARRAY,
                                AC_VALUE * np.ones(len(FEH_ARRAY)).T
                                ])



            frame_array.append(pd.DataFrame({"T"      : TEFF * np.ones(len(FEH_ARRAY)),
                                             'FEH'    : FEH_ARRAY,
                                             'CARBON' : AC_VALUE * np.ones(len(FEH_ARRAY)),
                                             'CHI'    : chi_interp_arch(spec, SYNTH_ARRAY, type='weight')}
                                             ))

    return pd.concat(frame_array)




