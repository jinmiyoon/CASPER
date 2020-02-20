## Author : Devin Whitten
## University of Notre Dame


## Redraft of the synthetic_MLE.py interface, intended for use with the MASTER_spec_interp.pkl

import emcee
from scipy.stats import chisquare
from scipy.stats import beta
from scipy.interpolate import LinearNDInterpolator as NDLinear
from scipy.interpolate import interp1d
from scipy.interpolate import splrep, splev
from scipy.optimize import minimize
import scipy.stats
from scipy.stats import mode
from statsmodels.nonparametric.kde import KDEUnivariate

import corner
import os, sys
import numpy as np
import pickle as pkl
import pandas as pd
import MAD
import EW
import MLE_priors
import MCMC_interface



#### Main MLE function

def run_chi_mcmc(input_spec, initial, mcmc_args):
    ## input_spec : observed spectrum for param determination
    ## inital:  initial values of Teff, [Fe/H], [C/Fe]
    ## mcmc_args should have everything we need for running
    #### need to carry the logg value somehow...

    print("... loading arch libs")
    interp_path = "/Users/MasterD/Google Drive/CCSLab/dev/arch_libs/MASTER_spec_interp.pkl"
    spec_int = pkl.load(open(interp_path, 'rb'))

    mcmc_args.update(MCMC_interface.beta_param_spec(input_spec, hard_var = 0.2))
    #thing = MCMC_interface.beta_param_spec(input_spec)
    #print(thing.keys())
    #################################################################

    ### Check which KP band to use from Beers et al 1999
    print("assigning KP bounds...")

    mcmc_args['KP_bounds'] = EW.get_KP_band(input_spec['wave'], input_spec['flux'])




    if mcmc_args['carbon_mode'] == "CH":
        print("Using CaII + CH LL")
        initial = np.concatenate([initial,
                                 [mcmc_args['CAII']['u'], mcmc_args['CH']['u']]])

        LL_FUNCTION = MCMC_interface.chi_likelihood
        LL_FUNCTION_2 = MCMC_interface.chi_ll_refine   #### Not yet implemented


    elif mcmc_args['carbon_mode'] == "CH+C2":
        print("Using CaII + CH + C2 LL")
        initial = np.concatenate([initial,
                                 [mcmc_args['CAII']['u'], mcmc_args['CH']['u'], mcmc_args['C2']['u']]])

        LL_FUNCTION = MCMC_interface.chi_likelihood_C2
        LL_FUNCTION_2 = MCMC_interface.chi_ll_refine_C2


    mcmc_args['bounds'] = None

    ### initialize mcmc
    pos = initial + initial * (2e-2*np.random.randn(25, len(initial)))

    nwalkers, ndim = pos.shape
    bounds = 'default'



    ###########################################################################
    print("Running coarse sampler:   ", bounds)
    sampler = emcee.EnsembleSampler(nwalkers, ndim,
                                    LL_FUNCTION,
                                    args=(input_spec, spec_int[mcmc_args['class']], mcmc_args, bounds)
                                    )


    _ = sampler.run_mcmc(pos, mcmc_args['iterations'])
    mcmc_args['coarse_sampler'] = sampler


    ###########################################################################
    ## link the function to the mcmc_args


    ############################################################################
    ### Refined parameter search
    ############################################################################

    print('Fine search')
    #first_burn = int(mcmc_args['burnin'] * mcmc_args['coarse_sampler'].chain.shape[1])
    mcmc_args['first_params'] = MCMC_interface.get_mcmc_params(mcmc_args['coarse_sampler'],
                                                               burnin = mcmc_args['burnin'])


    #### set the parameter bounds
    mcmc_args['bounds'] = MCMC_interface.set_param_bounds(mcmc_args['first_params'])


    print(mcmc_args['bounds'])





    ### now reinitialize, just FEH and CFE for now
    pos = np.vstack([np.random.normal(*mcmc_args['first_params']['feh'], 25),
                     np.random.normal(*mcmc_args['first_params']['cfe'], 25)]).T

    nwalkers, ndim = pos.shape


    ### set param_edge function to refined_param_edges

    bounds = 'final'

    print("Running fine sampler:   ", bounds)
    sampler = emcee.EnsembleSampler(nwalkers, ndim,
                                    LL_FUNCTION_2,
                                    args=(input_spec, spec_int[mcmc_args['class']], mcmc_args, bounds)
                                    )

    sampler.run_mcmc(pos, mcmc_args['iterations'])

    mcmc_args['final_sampler'] = sampler

    print("complete")
    return mcmc_args
