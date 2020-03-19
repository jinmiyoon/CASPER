### Name: Devin D. Whitten
### Email: dwhitten@nd.edu
### Institute: University of Notre Dame

### CCSLab_interface.py
from astropy.io import fits
from scipy.interpolate import LinearNDInterpolator as NDLinear
from scipy.interpolate import interp1d
from scipy.optimize import minimize
from multiprocessing import Pool, cpu_count
import pickle as pkl
import numpy as np
import ac
import pandas as pd
import matplotlib.pyplot as plt
import os, sys
import MAD
from os.path import isfile, join
from statsmodels.nonparametric.kde import KDEUnivariate

from collections import namedtuple
import GISIC
#custom class defs
import MCMC_interface
import spectrum
import EW
import emcee
import synthetic_functions


### GLOBAL ITEMS

ARCHETYPE_PARAMS = {"HALO" : {'GI'  : {'FEH': -2.5, 'CFE': 1.97, 'AC' : 7.9},
                              'GII' : {'FEH': -3.5, 'CFE': 0.97, 'AC' : 5.9},
                              'GIII': {'FEH': -4.3, 'CFE': 2.87, 'AC' : 7.0}},

                    "UFD"  : {'GI'  : {'FEH': -1.5, 'CFE': 1.07, 'AC' : 8.0},
                              'GII' : {'FEH': -3.0, 'CFE': 0.87, 'AC' : 6.3},
                              'GIII': {'FEH': -3.5, 'CFE': 2.37, 'AC' : 7.3}}}


LL_FUNCTION_DICT = {"COARSE": {"CH" : MCMC_interface.chi_likelihood, "CH+C2" : MCMC_interface.chi_likelihood_C2},
                    "REFINE": {"CH" : MCMC_interface.chi_ll_refine, "CH+C2" : MCMC_interface.chi_ll_refine_C2}
                    }

INTERPOLATOR     = pkl.load(open("interface/libraries/MASTER_spec_interp.pkl", 'rb'))

SYNTH_WAVE = np.arange(3000, 5001, 1)





def archetype_classify_MC(spectrum):
    ### Precondition: must have spectrum.frame with normalization defined
    ### spectrum: spectrum.Spectrum() object
    ### okay, we want this to run for whatever the class is set


    length = 100
    #temp_span = np.linspace(bounds[0], bounds[1], length)
    temp_values = np.random.normal(spectrum.teff_irfm, spectrum.teff_irfm_unc, length)
    span = np.ones(length)

    ### generate spectra
    ### GI


    GI_SYNTH = INTERPOLATOR[spectrum.gravity_class](np.column_stack((temp_values,
                                                span * ARCHETYPE_PARAMS[spectrum.MODE]['GI']['FEH'],
                                                span * ARCHETYPE_PARAMS[spectrum.MODE]['GI']['CFE'])))


    GII_SYNTH = INTERPOLATOR[spectrum.gravity_class](np.column_stack((temp_values,
                                                span * ARCHETYPE_PARAMS[spectrum.MODE]['GII']['FEH'],
                                                span * ARCHETYPE_PARAMS[spectrum.MODE]['GII']['CFE'])))


    GIII_SYNTH = INTERPOLATOR[spectrum.gravity_class](np.column_stack((temp_values,
                                                span * ARCHETYPE_PARAMS[spectrum.MODE]['GIII']['FEH'],
                                                span * ARCHETYPE_PARAMS[spectrum.MODE]['GIII']['CFE'])))


    GI_LLs = np.array([synthetic_functions.CAII_CH_CHI_LH(obs=spectrum.frame,
                                        synth=pd.DataFrame({'wave': SYNTH_WAVE, 'norm' : SYNTH}),
                                        CA_BOUNDS = spectrum.KP_bounds,
                                        CH_BOUNDS = [4222, 4322],
                                        CA_XI  = spectrum.SN_DICT['CA']['XI_AVG'],
                                        CH_XI  = spectrum.SN_DICT['CH']['XI_AVG']) for SYNTH in GI_SYNTH])

    GII_LLs = np.array([synthetic_functions.CAII_CH_CHI_LH(obs=spectrum.frame,
                                        synth=pd.DataFrame({'wave': SYNTH_WAVE, 'norm' : SYNTH}),
                                        CA_BOUNDS = spectrum.KP_bounds,
                                        CH_BOUNDS = [4222, 4322],
                                        CA_XI  = spectrum.SN_DICT['CA']['XI_AVG'],
                                        CH_XI  = spectrum.SN_DICT['CH']['XI_AVG']) for SYNTH in GII_SYNTH])

    GIII_LLs = np.array([synthetic_functions.CAII_CH_CHI_LH(obs=spectrum.frame,
                                        synth=pd.DataFrame({'wave': SYNTH_WAVE, 'norm' : SYNTH}),
                                        CA_BOUNDS = spectrum.KP_bounds,
                                        CH_BOUNDS = [4222, 4322],
                                        CA_XI  = spectrum.SN_DICT['CA']['XI_AVG'],
                                        CH_XI  = spectrum.SN_DICT['CH']['XI_AVG']) for SYNTH in GIII_SYNTH])

    #print(np.mean([GI_LLs, GII_LLs, GIII_LLs]))
    GI_LLs   = GI_LLs[np.isfinite(GI_LLs)]
    GII_LLs  = GII_LLs[np.isfinite(GII_LLs)]
    GIII_LLs = GIII_LLs[np.isfinite(GIII_LLs)]

    spectrum.set_group_ll({"GI" :  [np.median(GI_LLs),  np.std(GI_LLs)],
                           "GII":  [np.median(GII_LLs),  np.std(GII_LLs)],
                           "GIII": [np.median(GIII_LLs), np.std(GIII_LLs)]})


    return



def mcmc_determination(spectrum, mode='COARSE', pool=4):

    ### Precondition: must have run archetype_classification
    ### spectrum: spectrum.Spectrum() object

    print('\t ' + spectrum.get_name().ljust(20) + ":  " + spectrum.get_gravity_class() + " : " + spectrum.get_carbon_mode() + " : " + spectrum.print_KP_bounds())



    ## FOR initial FEH and CFE values, for temp use photo
    PARAMS = ARCHETYPE_PARAMS[spectrum.get_environ_mode()][spectrum.get_arch_group()]
    print('\t initializing with archetype parameters: ', PARAMS)

    #### MAIN MODE BRANCH

    if mode=='COARSE':
        ## if it's coarse, then you need the photo teff and the Sigma/Xi

        photo_teff = spectrum.get_photo_temp()
        initial = [photo_teff[0], PARAMS['FEH'], PARAMS['CFE']]

        ARGS = (spectrum.regions, SYNTH_WAVE, photo_teff[0], photo_teff[1],
                spectrum.get_SN_dict(), spectrum.get_gravity_class())

        initial = np.concatenate([initial,
                                 [spectrum.SN_DICT['CA']['XI_AVG'],
                                  spectrum.SN_DICT['CH']['XI_AVG']]])



        if spectrum.get_carbon_mode() == "CH+C2":
            print("\t running with carbon mode: CH+C2")

            ### add the beta params
            initial = np.concatenate([initial,
                                     [spectrum.SN_DICT['C2']['XI_AVG']]])


    elif mode == 'REFINE':
        ### In this case we want to use the params determined from the COARSE run
        PARAMS_0 = spectrum.get_mcmc_dict(mode = 'COARSE')

        ARGS = (spectrum.regions, SYNTH_WAVE,
                PARAMS_0, spectrum.get_gravity_class())

        initial = [spectrum.MCMC_COARSE['FEH'][0], spectrum.MCMC_COARSE['CFE'][0]]

    else:
        print("Invalid mode")

    ############################################################################
    ### PREPARE SPECTRA SLICES
    ############################################################################

    ### Select the correct likelihood function
    LL_FUNCTION = LL_FUNCTION_DICT[mode][spectrum.get_carbon_mode()]

    #if pool == 'MAX':
        #cpu_cores = cpu_count()

    #else:
    #    cpu_cores = pool

    #print("\t running on ", cpu_cores, " cores")
    #pool_init = Pool(cpu_cores)

    pos = initial + initial * (2e-2*np.random.rand(25, len(initial)))
    nwalkers, ndim = pos.shape
    bounds = 'default'


    print("\t running for ", spectrum.get_MCMC_iterations(), " iterations...")


    sampler = emcee.EnsembleSampler(nwalkers, ndim,
                                    LL_FUNCTION,
                                    args=(ARGS))
    ####

    _ = sampler.run_mcmc(pos, spectrum.get_MCMC_iterations())


    spectrum.set_sampler(sampler, mode=mode)

    print()


    return





def generate_synthetic(spectrum):
    ### Generates the best synth spectrum, given the KDE MCMC params

    ### grab those params
    ### NEED TO FINISH!!!!!!!!!!!!!!!!!

    SYNTH_FLUX = INTERPOLATOR[spectrum.get_gravity_class()](spectrum.MCMC_COARSE['TEFF'][0],
                                                            spectrum.MCMC_REFINE['FEH'][0],
                                                            spectrum.MCMC_REFINE['CFE'][0])


    spectrum.set_synth_spectrum(pd.DataFrame({'wave' : SYNTH_WAVE, 'norm' : SYNTH_FLUX.T}))


    return


def kde_param_relection(distro):
    ### this version is very susceptible to local maxima...
    ### kde_param tries to ensure correct handling of multimodal distributions

    distro = distro[np.isfinite(distro)]

    MIN, MAX = min(distro), max(distro)
    span = np.linspace(MIN, MAX, 200)

    ### create distribution reflection
    lower = MIN - abs(distro - MIN)
    upper = MAX + abs(distro - MAX)

    ### staple them together
    merge = np.concatenate([lower, distro, upper])

    ### compute kernal density estimation for both
    KDE_MAIN = KDEUnivariate(distro)
    KDE_FULL = KDEUnivariate(merge)

    ### fit distro, using the std from the main!

    KDE_MAIN.fit(bw = np.std(distro)/4.)
    KDE_FULL.fit(bw = np.std(distro)/4.)

    ### need to use the main KDE to scale the full
    scale = np.median(np.divide(KDE_MAIN.evaluate(span), KDE_FULL.evaluate(span)))


    ### now maximize the full KDE, using the maxed main as the starting guess
    result = minimize(lambda x: -1*KDE_FULL.evaluate(x),
    x0 = span[KDE_MAIN.evaluate(span) == max(KDE_MAIN.evaluate(span))], method='Powell')  ## Powell has been working pretty well.

    return {'result' : float(result['x']), 'kde' : KDE_MAIN, 'kde_reflect' : interp1d(span, KDE_FULL.evaluate(span) * scale)}


def generate_kde_params(spectrum, mode, burnin=0.25):
    ### main parameter extraction routine following mcmc determination
    ### get chain
    if   mode == 'COARSE':
        chain = spectrum.MCMC_COARSE_sampler.chain

    elif mode == 'REFINE':
        chain = spectrum.MCMC_REFINE_sampler.chain

    walkers, iter, ndim = chain.shape

    ### merge walkers and iterations for formatted chain
    chain = chain[:, int(burnin * iter):, :].reshape((-1, ndim))

    ### basic medians
    MEDIAN = [np.median(array) for array in chain.T]
    STD    = [np.std(array) for array in chain.T]

    ### Let's use the kde_params
    ### Note: kde is highly susceptible to errors at the boundaries of the grid
    ### I'm going to try a solution involving edge reflection

    results   =       [kde_param_relection(array) for array in chain.T]


    if ndim == 2:
        dict_keys = ['FEH', 'CFE']

    if ndim == 5:
        dict_keys = ['TEFF', 'FEH', 'CFE', 'XI_CA', 'XI_CH']

    elif ndim == 6:
        dict_keys = ['TEFF', 'FEH', 'CFE', 'XI_CA', 'XI_CH', 'XI_C2']


    ### build outputs
    OUTPUT = {key : [results[i]['result'], MAD.S_MAD(chain[:, i])] for i, key in enumerate(dict_keys)}

    OUTPUT['AC'] = [ac.ac(OUTPUT['CFE'][0], OUTPUT['FEH'][0]), np.sqrt(OUTPUT['CFE'][1]**2 + OUTPUT['FEH'][1]**2)]

    KDE_DICT = {key : [element['kde'], element['kde_reflect']] for key, element in zip(dict_keys, results)}


    spectrum.set_mcmc_results(OUTPUT, mode=mode)
    spectrum.set_kde_functions(KDE_DICT, mode=mode)


    return
