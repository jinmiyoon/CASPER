################################################################################
### Author: Devin Whitten, Jinmi Yoon
### Email: devin.d.whitten@gmail.com, jinmi.yoon@gmail.com
################################################################################
## Main parameter determination procedures


""" J Yoon 03/11/2022
To use multiprocessing module:
Some builds of NumPy (including the version included with Anaconda) will
automatically parallelize some operations using something like
the MKL linear algebra. This can cause problems when used with
the parallelization methods described here so it can be good to turn that off
(by setting the environment variable OMP_NUM_THREADS=1, for example).
"""
import os, sys
os.environ["OMP_NUM_THREADS"] = "1"

from scipy.interpolate import interp1d
from scipy.optimize import minimize
from multiprocessing import Pool, cpu_count,current_process
import numpy as np
import ac
import pandas as pd
import time
import MAD
from statsmodels.nonparametric.kde import KDEUnivariate
import MCMC_interface
import emcee
from synthetic_functions import get_interp, get_grav_interp, CAII_CH_CHI_LH, normalize 
#import spectrum
import config

### GLOBAL ITEMS
ARCHETYPE_PARAMS = config.ARCHETYPE_PARAMS

LL_FUNCTION_DICT = {"COARSE": {"CH" : MCMC_interface.chi_likelihood, "CH+C2" : MCMC_interface.chi_likelihood_C2},
                    "REFINE": {"CH" : MCMC_interface.chi_ll_refine, "CH+C2" : MCMC_interface.chi_ll_refine_C2}
                    }

# import synthetic library interpolator
INTERPOLATOR = get_interp()
GRAV_INTERP = get_grav_interp()

SYNTH_WAVE = config.SYNTH_WAVE

def synth_normalize(spectrum, group, temp):
    interp_flux = INTERPOLATOR[spectrum.gravity_class](temp, ARCHETYPE_PARAMS[spectrum.MODE][group]['FEH'], 
                                                                        ARCHETYPE_PARAMS[spectrum.MODE][group]['CFE'])
    if np.isfinite(interp_flux).all(): 
        return normalize(SYNTH_WAVE, interp_flux[config.id_start_wave:])
    else: 
        print("Interpolated synthetic flux is not finite, params = ",temp, ARCHETYPE_PARAMS[spectrum.MODE][group]['FEH'], ARCHETYPE_PARAMS[spectrum.MODE][group]['CFE'])



def archetype_classify_MC(spectrum):
    ### Precondition: must have spectrum.frame with normalization defined
    ### spectrum: spectrum.Spectrum() object
    ### okay, we want this to run for whatever the class is set


    length = 100
    temp_values = np.random.normal(spectrum.teff_irfm, spectrum.teff_irfm_err, length)

    ### Generate spectra (GI_NORM_SYNTH, GII_NORM_SYNTH,GIII_NORM_SYNTH)
    # J. Yoon Feb 25 2022
    #
    #    Here is buiding arrays of spectral parameters/grids within a Teff range
    #    ([TEFF_HARD - T_SIGMA, TEFF_HARD + T_SIGMA] or
    #    [TEFF_ADT - T_SIGMA, TEFF_ADT + T_SIGMA]) to generate synthetic spectra.
    #    for example, GI_NORM_SYNTH will create len(span) of arrays, each value looks
    #    like [4715.6, -2.5, 1.97] depending on gravity_class and galactic env mode.

    ### GI
    """
    def synth_normalize(group, temp):
        interp_flux = INTERPOLATOR[spectrum.gravity_class](temp, ARCHETYPE_PARAMS[spectrum.MODE][group]['FEH'], 
                                                                            ARCHETYPE_PARAMS[spectrum.MODE][group]['CFE'])
        if np.isfinite(interp_flux).all(): 
            return normalize(SYNTH_WAVE, interp_flux[config.id_start_wave:])
        else: 
            print("Interpolated synthetic flux is not finite, params = ",temp, ARCHETYPE_PARAMS[spectrum.MODE][group]['FEH'], ARCHETYPE_PARAMS[spectrum.MODE][group]['CFE'])
    """
    
    # *****  NEW SYNTH 
 
    start = time.time()

    GI_NORM_SYNTH =[synth_normalize(spectrum, 'GI', temp) for temp in temp_values]
    GII_NORM_SYNTH =[synth_normalize(spectrum, 'GII', temp) for temp in temp_values]
    GIII_NORM_SYNTH =[synth_normalize(spectrum,'GIII', temp) for temp in temp_values]


    """
    GI_temps = [['GI', temp] for temp in temp_values]
    GII_temps = [['GII', temp] for temp in temp_values]
    GIII_temps = [['GIII', temp] for temp in temp_values]
    with Pool() as pool:
        GI_NORM_SYNTH =pool.map(synth_normalize,  GI_temps)
        GII_NORM_SYNTH =pool.map(synth_normalize,  GII_temps)
        GIII_NORM_SYNTH =pool.map(synth_normalize,  GIII_temps)
    """

    end1 = time.time()
    print("\t\t interface_main: archetype_classify_MC SYNTH: took {0:.1f} seconds".format(end1 - start))

    """ # When using Devin's library
    #span = np.ones(length)

    GI_NORM_SYNTH = INTERPOLATOR[spectrum.gravity_class](np.column_stack((temp_values,
                                                span * ARCHETYPE_PARAMS[spectrum.MODE]['GI']['FEH'],
                                                span * ARCHETYPE_PARAMS[spectrum.MODE]['GI']['CFE'])))


    GII_NORM_SYNTH = INTERPOLATOR[spectrum.gravity_class](np.column_stack((temp_values,
                                                span * ARCHETYPE_PARAMS[spectrum.MODE]['GII']['FEH'],
                                                span * ARCHETYPE_PARAMS[spectrum.MODE]['GII']['CFE'])))


    GIII_NORM_SYNTH = INTERPOLATOR[spectrum.gravity_class](np.column_stack((temp_values,
                                                span * ARCHETYPE_PARAMS[spectrum.MODE]['GIII']['FEH'],
                                                span * ARCHETYPE_PARAMS[spectrum.MODE]['GIII']['CFE'])))
    
    """

    # calculate log likelihood function for CA II and CH for MLE estimation for each group
    GI_LLs = np.array([CAII_CH_CHI_LH(obs=spectrum.frame,
                                        synth=pd.DataFrame({'wave': SYNTH_WAVE, 'norm' : SYNTH}),
                                        CA_BOUNDS = spectrum.KP_bounds,
                                        CH_BOUNDS = config.CH_BOUNDS,
                                        CA_XI  = spectrum.SN_DICT['CA']['XI_AVG'],
                                        CH_XI  = spectrum.SN_DICT['CH']['XI_AVG']) for SYNTH in GI_NORM_SYNTH])

    GII_LLs = np.array([CAII_CH_CHI_LH(obs=spectrum.frame,
                                        synth=pd.DataFrame({'wave': SYNTH_WAVE, 'norm' : SYNTH}),
                                        CA_BOUNDS = spectrum.KP_bounds,
                                        CH_BOUNDS = config.CH_BOUNDS,
                                        CA_XI  = spectrum.SN_DICT['CA']['XI_AVG'],
                                        CH_XI  = spectrum.SN_DICT['CH']['XI_AVG']) for SYNTH in GII_NORM_SYNTH])

    GIII_LLs = np.array([CAII_CH_CHI_LH(obs=spectrum.frame,
                                        synth=pd.DataFrame({'wave': SYNTH_WAVE, 'norm' : SYNTH}),
                                        CA_BOUNDS = spectrum.KP_bounds,
                                        CH_BOUNDS = config.CH_BOUNDS,
                                        CA_XI  = spectrum.SN_DICT['CA']['XI_AVG'],
                                        CH_XI  = spectrum.SN_DICT['CH']['XI_AVG']) for SYNTH in GIII_NORM_SYNTH])
    #print(np.mean([GI_LLs, GII_LLs, GIII_LLs]))
    GI_LLs   = GI_LLs[np.isfinite(GI_LLs)]
    GII_LLs  = GII_LLs[np.isfinite(GII_LLs)]
    GIII_LLs = GIII_LLs[np.isfinite(GIII_LLs)]

    spectrum.set_group_ll({"GI" :  [np.median(GI_LLs),  np.std(GI_LLs)],
                           "GII":  [np.median(GII_LLs),  np.std(GII_LLs)],
                           "GIII": [np.median(GIII_LLs), np.std(GIII_LLs)]})
    end2 = time.time()
    print("\t\t interface_main: archetype_classify_MC LLs: took {0:.1f} seconds".format(end2 - end1))

    return



#def mcmc_determination(spectrum, mode='COARSE', pool=4):
def mcmc_determination(spectrum, mode='COARSE', burnin_factor=7):

    ### Precondition: must have run archetype_classification
    ### spectrum: spectrum.Spectrum() object

    # mode here means either "coarse" or "fine" , 09/02/2020, J. Yoon
    print("\t * MCMC run mode =  ", mode)
    # spectrum and its info
    print('\t ' + spectrum.get_name().ljust(20) + ":  " + spectrum.get_gravity_class() + " : " + spectrum.get_carbon_mode() + " : " + spectrum.print_KP_bounds())


    ## FOR initial FEH and CFE values, for temp use photometric temp.
    PARAMS = ARCHETYPE_PARAMS[spectrum.get_environ_mode()][spectrum.get_arch_group()]

    #### MAIN MODE BRANCH
 
    if mode=='COARSE':
        ## if it's coarse, then you need the photometric teff and the Sigma/Xi
        print('\t initializing with archetype parameters: ', PARAMS)

        # spectrum.get_photo_temp() returns self.teff_irfm, self.teff_irfm_err
        # so photo_teff[0] and photo_teff[1] respectively.
        photo_teff = spectrum.get_photo_temp()
        # inserted 01/04/2022 for comparison
        #print('Teff : %.0F  [Fe/H] : %.2F   [C/Fe] : %.2F  A(C): %.2F'% (photo_teff[0], PARAMS['FEH'], PARAMS['CFE'], PARAMS['AC']))
        initial = [photo_teff[0], PARAMS['FEH'], PARAMS['CFE']]

        #testing different initial values
        #initial = [4100, PARAMS['FEH'], PARAMS['CFE']]

        ARGS = (spectrum.regions, SYNTH_WAVE, photo_teff[0], photo_teff[1],
                spectrum.get_SN_dict(), spectrum.get_gravity_class())

        initial = np.concatenate([initial,
                                 [spectrum.SN_DICT['CA']['XI_AVG'],
                                  spectrum.SN_DICT['CH']['XI_AVG']]])
        n_step = spectrum.get_MCMC_iterations()



        if spectrum.get_carbon_mode() == "CH+C2":
            print("\t running with carbon mode: CH+C2")

            ### add the beta params
            initial = np.concatenate([initial,
                                     [spectrum.SN_DICT['C2']['XI_AVG']]])
        else: print("\t running with carbon mode: CH only")
        #PARAMS_0 = spectrum.get_mcmc_dict(mode = 'COARSE')
        #print('\t COARSE run result parameters: ', PARAMS_0)
    elif mode == 'REFINE':
        ### In this case we want to use the params determined from the COARSE run
        PARAMS_0 = spectrum.get_mcmc_dict(mode = 'COARSE')
        print('\t initializing with COARSE run result parameters:')
        print(' Teff : %.0F  [Fe/H] : %.2F   [C/Fe] : %.2F   A(C) : %.2F' % (PARAMS_0['TEFF'][0],PARAMS_0['FEH'][0], PARAMS_0['CFE'][0],PARAMS_0['AC'][0]) )

        ARGS = (spectrum.regions, SYNTH_WAVE,
                PARAMS_0, spectrum.get_gravity_class())

        initial = [spectrum.MCMC_COARSE['FEH'][0], spectrum.MCMC_COARSE['CFE'][0]]
        n_step= int(spectrum.get_MCMC_iterations()/4)

    else:
        print("Invalid mode")

    ############################################################################
    ### PREPARE SPECTRA SLICES
    ############################################################################

    ### Select the correct likelihood function
    LL_FUNCTION = LL_FUNCTION_DICT[mode][spectrum.get_carbon_mode()]

    n_cpu = cpu_count()
    print("\t number of cpu = ", n_cpu)

    pos = initial + initial * (2e-2*np.random.rand(64, len(initial))) # Gaussian distribution
    print("\t\t interface_main : pos = ", pos)
    #pos = initial + initial * (np.random.rand(25, len(initial))) # uniform spacing
    nwalkers, ndim = pos.shape
    #bounds = 'default'

    print("\t running for ", n_step, " iterations...")

    with Pool() as pool:
        print(f'Process {current_process().name} started working', flush=True)     
        sampler = emcee.EnsembleSampler(nwalkers, ndim, LL_FUNCTION,
            #moves=[(emcee.moves.DEMove(), 1.0),],
            moves=[(emcee.moves.DEMove(), 0.8),(emcee.moves.DESnookerMove(), 0.2),],
            pool=pool, args=(ARGS))
        start = time.time()


        _ = sampler.run_mcmc(pos, n_step, progress=True)
        end = time.time()
        multi_time = end - start
        print(f'Process {current_process().name} ended working', flush=True) 
        print("\t \t MCMC Multiprocessing took {0:.1f} seconds".format(multi_time))

    # want to print out the latest result from mcmc, 12/13/2021
    #print('\t the latest result from sampler() after MCMC runs:    ', _ )

    spectrum.set_sampler(sampler, mode=mode)

    tau=sampler.get_autocorr_time(quiet=True)
    
    #num_valid_autocorr_time_value = len(tau)-tau.tolist().count(np.nan)
    num_valid_autocorr_time_value = len(tau)-np.isnan(tau).sum()
    print("\t\t interface_main: tau's shape= {}, length ={}, how many nan values = {}".format(tau.shape, len(tau), np.isnan(tau).sum()))
    print("\t\t interface_main: tau = {}, num_valid_autocorr_time_value ={} ".format(tau,num_valid_autocorr_time_value ))
    if num_valid_autocorr_time_value == 0 :
        print("\t\t interface_main: all autocorr_times are Nan!")
        max_auto_corr_time =70 #a random number similar to average value of other maximum autocorr time
        
    elif num_valid_autocorr_time_value == 1  :
        print("\t\t interface_main: all except one dim autocorr_time are Nan")
        for taulist in tau:
            #if taulist != np.nan:
            if not np.isnan(taulist):
                max_auto_corr_time= taulist
        
    else:
        print("\t\t interface_main: n >= 2 in tau array values are vaild numbers ")
        max_auto_corr_time= np.nanmax(tau)
        print("\t\t interface_main: maximum autocorrelation time = ", max_auto_corr_time)

    # Setting burnin by discarding the first n_discard runs. 
    n_discard= int(burnin_factor * max_auto_corr_time) 
    
    # if n_discard is larger than the mcmc iterations, it should be fixed to a random value, 
    # perhaps, discard the first half runs. This can be revisited
    if n_discard >= 0.5 * n_step :
        print("\t\t interface_main: n_discard is larger than the mcmc iterations! Setting n_discard to half the iterations. ") 
        n_discard = int(0.5 * n_step)
    print("\t\t mcmc mode = ", mode)
    mean_acc_fraction= np.mean(sampler.acceptance_fraction)
    print("\t\t interface_main: mean acceptance fraction: {0:.3f}".format(mean_acc_fraction))
    print("\t\t interface_main: max autocorrelation_time = ", max_auto_corr_time)
    # discard the first steps in the chain as burn-in
    print("\t\t interface_main: recommended n_discard = ",n_discard)

    if mode == 'COARSE': 
        spectrum.mcmc_coarse_acc_frac = mean_acc_fraction
        spectrum.mcmc_coarse_tau = max_auto_corr_time
        spectrum.mcmc_coarse_n_discard = n_discard
    else: 
        spectrum.mcmc_refine_acc_frac = mean_acc_fraction
        spectrum.mcmc_refine_tau = max_auto_corr_time
        spectrum.mcmc_refine_n_discard = n_discard

 

    return


def generate_synthetic(spectrum):
    ### Generates the best synth spectrum, given the KDE MCMC params

    """ # when using Devin's interpolator
    NORM_SYNTH_FLUX = INTERPOLATOR[spectrum.get_gravity_class()](spectrum.MCMC_COARSE['TEFF'][0],
                                                            spectrum.MCMC_REFINE['FEH'][0],
                                                            spectrum.MCMC_REFINE['CFE'][0])
    """

    # *****  NEW SYNTH 
    # Here I need GISIC.normalize()
   
    interp_flux = INTERPOLATOR[spectrum.get_gravity_class()](spectrum.MCMC_COARSE['TEFF'][0],
                                                            spectrum.MCMC_REFINE['FEH'][0],
                                                            spectrum.MCMC_REFINE['CFE'][0])
    
    # NORM_SYNTH_FLUX = normalize(SYNTH_WAVE, synth_interp_flux[config.id_start_wave:])
    if np.isfinite(interp_flux).all(): 
        NORM_SYNTH_FLUX =normalize(SYNTH_WAVE, interp_flux[config.id_start_wave:])
        spectrum.set_synth_spectrum(pd.DataFrame({'wave' : SYNTH_WAVE, 'norm' : NORM_SYNTH_FLUX.T}))
    else: 
        print("generate_synthetic: Interpolated synthetic flux is not finite, params = ",spectrum.MCMC_COARSE['TEFF'][0], 
              spectrum.MCMC_REFINE['FEH'][0],spectrum.MCMC_REFINE['CFE'][0])
        print(f"the sequence is {spectrum.get_sequence()} and the star name is {spectrum.get_name()}")
        # somehow the final params could be np.nan due to a slight deviation from the grids. 
        # In that case, the synthetic flux is nan. So we need to get around this problem for 
        spectrum.set_synth_spectrum(pd.DataFrame({'wave' : SYNTH_WAVE, 'norm' : np.nan* np.ones_like(SYNTH_WAVE)}))
    

    return

def estimate_logg(spectrum):
    # interpolate logg value based on the mcmc parameters

    spectrum.logg = GRAV_INTERP[spectrum.get_gravity_class()](spectrum.MCMC_COARSE['TEFF'][0],spectrum.MCMC_REFINE['FEH'][0])
    print(f"\t\t interface_main: logg = {spectrum.logg} " )
    
    #  Only COARSE would work because the size of REFINE dist is different from COARSE. I cannot use teff_coarse and feh_refine for calculating logg.
    samples_COARSE= spectrum.MCMC_COARSE_sampler.get_chain(discard= spectrum.mcmc_coarse_n_discard, thin=1, flat=True)
    teff_dist = samples_COARSE[:,0]
    feh_COARSE_dist = samples_COARSE[:,1]
    logg_COARSE_dist = GRAV_INTERP[spectrum.get_gravity_class()](teff_dist,feh_COARSE_dist)
    spectrum.logg_err = MAD.S_MAD(logg_COARSE_dist)
    logg_err_std = np.std(logg_COARSE_dist)
    logg_err_mad = MAD.MAD(logg_COARSE_dist)
    
    print(f"\t\t interface_main: logg = {spectrum.logg} +/- {spectrum.logg_err} " )
    print(f"\t\t interface_main: logg_std = {logg_err_std}, logg_mad = {logg_err_mad} ")

    return 

def kde_param_reflection(distro):
    ### this version is very susceptible to local maxima...
    ### kde_param tries to ensure correct handling of multimodal distributions

    #### 04/18/22 J. Yoon: I may need to change this part using a new function


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


def generate_kde_params(spectrum, mode, n_thin=1):
    ### main parameter extraction routine following mcmc determination

    ### get chain
    if   mode == 'COARSE':
        #chain = spectrum.MCMC_COARSE_sampler.chain

        # J. Yoon 04/18/2022
        # updated to .get_chain from .chain
        chain = spectrum.MCMC_COARSE_sampler.get_chain(discard= spectrum.mcmc_coarse_n_discard, thin=n_thin, flat=True)

    elif mode == 'REFINE':
        # Yoon 04/18/2022
        # updated to .get_chain from .chain but the resulting arrays are differently storedJ.
        #chain = spectrum.MCMC_REFINE_sampler.chain
        chain = spectrum.MCMC_REFINE_sampler.get_chain(discard= spectrum.mcmc_refine_n_discard, thin=n_thin, flat=True)

    #walkers, iter, ndim = chain.shape
    ndim = chain.shape[1]
    #print("ndim = ", ndim)

    ### Let's use the kde_params
    ### Note: kde is highly susceptible to errors at the boundaries of the grid
    ### I'm going to try a solution involving edge reflection

    results   =  [kde_param_reflection(array) for array in chain.T]


    if ndim == 2:
        dict_keys = ['FEH', 'CFE']

    if ndim == 5:
        dict_keys = ['TEFF', 'FEH', 'CFE', 'XI_CA', 'XI_CH']

    elif ndim == 6:
        dict_keys = ['TEFF', 'FEH', 'CFE', 'XI_CA', 'XI_CH', 'XI_C2']

    #print("dict_keys =  ", dict_keys)
    ### build outputs
    OUTPUT = {key : [results[i]['result'], MAD.S_MAD(chain[:, i])] for i, key in enumerate(dict_keys)}

    OUTPUT['AC'] = [ac.ac(OUTPUT['CFE'][0], OUTPUT['FEH'][0]), np.sqrt(OUTPUT['CFE'][1]**2 + OUTPUT['FEH'][1]**2)]

    KDE_DICT = {key : [element['kde'], element['kde_reflect']] for key, element in zip(dict_keys, results)}


    spectrum.set_mcmc_results(OUTPUT, mode=mode)
    spectrum.set_kde_functions(KDE_DICT, mode=mode)


    return
