################################################################################
### Author: Devin Whitten
### Email: devin.d.whitten@gmail.com
### Institute: University of Notre Dame
################################################################################

#### New interface for the MLE MCMC

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


#### Main Chi^2 MLE MCMC with variable sigma



def ln_chi_square_sigma(flux1, flux2, sigma):
    ### assume the flux arrays are aligned in wavelength

    dof = len(flux1) - 1
    chi = np.square(np.divide(flux1 - flux2, sigma)).sum()

    return (0.5 * dof - 1) * np.log(chi) - 0.5*chi




def teff_lnprior(X, mean, sigma):
    return -np.log(sigma) - 0.5*np.square(np.divide(X - mean, sigma))


def sigma_lnprior(sigma, alpha_value, beta_value):

    return np.log(beta.pdf(sigma, alpha_value, beta_value))

def param_edges(teff, feh, carbon, sigmaCAII, sigmaCH, bounds):

    if (teff < 4000.) or (teff > 5000.):
        return -np.inf

    if (feh < bounds['FEH'][0]) or (feh > bounds['FEH'][1]):
        return -np.inf

    if (carbon < bounds['CARBON'][0]) or (carbon > bounds['CARBON'][1]):
        return -np.inf

    if (sigmaCAII < 0.0) or (sigmaCAII > 1.0):
        return -np.inf

    if (sigmaCH < 0.0)  or (sigmaCH > 1.0):
        return -np.inf

    else:
        return 0.0

def chi_likelihood(theta, obs, interpolator, mcmc_args):

    teff =   theta[0]
    feh  =   theta[1]
    carbon = theta[2]
    sigmaCAII  = theta[3]
    sigmaCH    = theta[4]
    sigmaC2    = theta[5]

    group_bounds = {
                'GI'  : {"T": [4000, 5000], "FEH": [-3.5, -1.0],  'CARBON': [ 7.0, 9.0]},
                'GII' : {"T": [4000, 5000], "FEH": [-4.5, -2.0],  'CARBON': [-1.0, 1.5]},
                'GIII': {"T": [4000, 5000], "FEH": [-4.5, -3.0],  'CARBON': [ 6.0, 7.5]}
    }

    ### generate current spectrum

    synth_function = interp1d(np.arange(3000, 5001, 1),
                              interpolator([teff, feh, carbon])[0],
                              kind = 'linear')

    ## Cut regions
    spec_CAII  = obs[obs['wave'].between(3927.7, 3939.7, inclusive=True)]
    spec_CH    = obs[obs['wave'].between(4222, 4322, inclusive=True)]
    spec_C2    = obs[obs['wave'].between(4715, 4750, inclusive=True)]  # This is the Chrislieb 2001 passband


    LL =  ln_chi_square_sigma(spec_CAII['norm'], synth_function(spec_CAII['wave']), sigmaCAII) + \
          ln_chi_square_sigma(spec_CH['norm'], synth_function(spec_CH['wave']), sigmaCH) + \
          ln_chi_square_sigma(spec_C2['norm'], synth_function(spec_C2['wave']), sigmaC2) + \
          param_edges(teff, feh, carbon, sigmaCAII, sigmaCH, mcmc_args['bounds']) + \
          teff_lnprior(teff, mcmc_args['photo_teff'], mcmc_args['teff_sigma']) + \
          sigma_lnprior(sigmaCAII, mcmc_args['CAII_alpha'], mcmc_args['CAII_beta']) + \
          sigma_lnprior(sigmaCH, mcmc_args['CH_alpha'], mcmc_args['CH_beta']) + \
          sigma_lnprior(sigmaC2, mcmc_args['C2_alpha'], mcmc_args['C2_beta'])

    if np.isfinite(LL):
        return LL

    else:
        return -np.inf


def chi_likelihood_redux(theta, obs, interpolator, mcmc_args):

    teff =   theta[0]
    feh  =   theta[1]
    carbon = theta[2]
    sigmaCAII  = theta[3]
    sigmaCH    = theta[4]


    group_bounds = {
                'GI'  : {"T": [4000, 5000], "FEH": [-3.5, -1.0],  'CARBON': [ 7.0, 9.0]},
                'GII' : {"T": [4000, 5000], "FEH": [-4.5, -2.0],  'CARBON': [-1.0, 1.5]},
                'GIII': {"T": [4000, 5000], "FEH": [-4.5, -3.0],  'CARBON': [ 6.0, 7.5]}
    }

    ### generate current spectrum

    synth_function = interp1d(np.arange(3000, 5001, 1),
                              interpolator([teff, feh, carbon])[0],
                              kind = 'linear')

    ## Cut regions
    spec_CAII  = obs[obs['wave'].between(3927.7, 3939.7, inclusive=True)]
    spec_CH    = obs[obs['wave'].between(4222, 4322, inclusive=True)]


    LL =  ln_chi_square_sigma(spec_CAII['norm'], synth_function(spec_CAII['wave']), sigmaCAII) + \
          ln_chi_square_sigma(spec_CH['norm'], synth_function(spec_CH['wave']), sigmaCH) + \
          param_edges(teff, feh, carbon, sigmaCAII, sigmaCH, mcmc_args['bounds']) + \
          teff_lnprior(teff, mcmc_args['photo_teff'], mcmc_args['teff_sigma']) + \
          sigma_lnprior(sigmaCAII, mcmc_args['CAII_alpha'], mcmc_args['CAII_beta']) + \
          sigma_lnprior(sigmaCH, mcmc_args['CH_alpha'], mcmc_args['CH_beta'])

    if np.isfinite(LL):
        return LL

    else:
        return -np.inf





def get_beta_params(spectrum, bounds):
    ## just return the proper values for alpha and beta for the given spectra.
    ## Poisson uncertainty is assumed for flux bins.ß

    SN = np.divide(1.,np.sqrt(spectrum['flux'][spectrum['wave'].between(bounds[0], bounds[1], inclusive=True)]))

    u = np.median(SN)
    v = np.var(SN)

    print('SN =', np.median(SN))
    print("var(SN)=", np.var(SN))

    alpha_param = ((u**2)/v)*(1 - u) - u
    beta_param  = (1/u - 1) * alpha_param

    return {'alpha': alpha_param, 'beta': beta_param}



def run_chi_mcmc(obs, initial, group_class, mcmc_args):

    ### Group Bounds are still important

    print("... loading arch libs")
    interp_path = "/Users/MasterD/Google Drive/CCSLab/dev/arch_libs/interp/arch_spec_interp.pkl"
    spec_int = pkl.load(open(arch_path, 'rb'))

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

    print("MCMC params")

    mcmc_args['bounds'] = group_bounds[group_class.split("_")[0]]

    print("Computing beta params")
    CAII_BETA = get_beta_params(obs, [3884, 3923])
    CH_BETA   = get_beta_params(obs, [4222, 4322])
    C2_BETA   = get_beta_params(obs, [4500, 4600])

    mcmc_args['CAII_alpha'] = CAII_BETA['alpha']
    mcmc_args['CAII_beta']  = CAII_BETA['beta']

    mcmc_args['CH_alpha']   = CH_BETA['alpha']
    mcmc_args['CH_beta']    = CH_BETA['beta']

    mcmc_args['C2_alpha']   = C2_BETA['alpha']
    mcmc_args['C2_beta']    = C2_BETA['beta']

    pos =  initial + 1e-2*np.random.randn(25, len(initial))
    nwalkers, ndim = pos.shape

    sampler = emcee.EnsembleSampler(nwalkers, ndim, chi_likelihood,
                                    args=(obs, arch_spec_int[group_class], mcmc_args)
                                    )

    _ = sampler.run_mcmc(pos, 1000)

    return sampler, ndim, mcmc_args


def run_chi_mcmc_redux(obs, initial, group_class, mcmc_args):
    # This is just the version of the mcmc without the C2 fitting.
    ### Group Bounds are still important

    print("... loading arch libs")
    arch_path = "/Users/MasterD/Google Drive/CCSLab/dev/arch_libs/interp/arch_spec_interp.pkl"
    arch_spec_int = pkl.load(open(arch_path, 'rb'))

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

    print("MCMC params")

    mcmc_args['bounds'] = group_bounds[group_class.split("_")[0]]

    print("Computing beta params")
    CAII_BETA = get_beta_params(obs, [3884, 3923])
    CH_BETA   = get_beta_params(obs, [4222, 4322])


    mcmc_args['CAII_alpha'] = CAII_BETA['alpha']
    mcmc_args['CAII_beta']  = CAII_BETA['beta']

    mcmc_args['CH_alpha']   = CH_BETA['alpha']
    mcmc_args['CH_beta']    = CH_BETA['beta']


    pos =  initial + 1e-2*np.random.randn(25, len(initial))
    nwalkers, ndim = pos.shape

    sampler = emcee.EnsembleSampler(nwalkers, ndim, chi_likelihood_redux,
                                    args=(obs, arch_spec_int[group_class], mcmc_args)
                                    )

    _ = sampler.run_mcmc(pos, 1500)

    return sampler, ndim, mcmc_args



def get_mcmc_params(SAMPLER, burnin):
    SAMPLES = SAMPLER[0].chain[:, burnin:, :].reshape((-1, SAMPLER[1]))

    MEDIAN = [np.median(array) for array in SAMPLES.T]

    STD =    [MAD.S_MAD(array) for array in SAMPLES.T]

    modes1 = [mode(np.around(row, decimals = rounding))[0][0] for row, rounding in zip(SAMPLES.T, [0,1,1,3,3, 3])]
    modes2 = [mode(np.around(row, decimals = rounding))[0][0] for row, rounding in zip(SAMPLES.T, [0,2,2,3,3, 3])]

    ### Let's use the kde_params
    value2 = [kde_param(row, x0 = x0)['result'] for row, x0 in zip(SAMPLES.T, MEDIAN)]


    OUTPUT = {'teff' :     [value2[0], MAD.S_MAD(SAMPLES[:, 0])],
              'feh' :      [value2[1], MAD.S_MAD(SAMPLES[:, 1])],
              'carbon' :   [value2[2], MAD.S_MAD(SAMPLES[:, 2])],
              'sigmaCAII': [value2[3], MAD.S_MAD(SAMPLES[:, 3])],
              'sigmaCH'  : [value2[4], MAD.S_MAD(SAMPLES[:, 4])],
              'sigmaC2'  : [value2[5], MAD.S_MAD(SAMPLES[:, 5])]}

    return OUTPUT


def interp_histo(counts, edges):
    ### getting kinda wierd, trying to generate a functional pdf from a histogram bin collection


    centers = np.array([(edges[i] + edges[i+1])/2. for i in range(len(edges) -1)])

    x_array = np.arange(min(centers), max(centers), 0.01)
    ### generate interpolated function
    pdf_function = interp1d(centers, counts, kind='cubic')

    ### Let's get that smooth sample
    return scipy.optimize.minimize(lambda x : -1*pdf_function(x), x0 = np.median(centers), bounds = ((centers[0]), (centers[-1]))) #


def kde_param(distribution, x0):


    ### compute kernal density estimation
    KDE = KDEUnivariate(distribution)

    KDE.fit(bw=np.std(distribution)/3.0)

    result = scipy.optimize.minimize(lambda x: -1*KDE.evaluate(x),
    x0 = x0, method='Powell')
    #print(result)

    return {'result' : float(result['x']), 'kde' : KDE}


def bootstrap_params(distro, resolution):
    ##### trying to solve the max bin pdf robustness thing
    bins = int((max(distro) - min(distro))/resolution)

    counts, edges = np.histogram(distro, bins = bins)

    centers = np.array([(edges[i] + edges[i+1])/2. for i in range(len(edges) -1)])

    centers[np.where(counts == max(counts))]

    return centers
