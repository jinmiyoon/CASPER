################################################################################
### Author: Devin Whitten
### Email: devin.d.whitten@gmail.com
### Institute: University of Notre Dame
################################################################################

### Prior functions for the updated MLE procedure

import numpy as np

from scipy.stats import chisquare
from scipy.stats import beta


native_bounds = {
                 "teff" : [4000, 5000],
                 "feh"  : [-4.5, -1.0],
                 "cfe"  : [-0.5, 4.5]
                }

def ln_chi_square_sigma(flux, synth, sigma):
    ### assume the flux arrays are aligned in wavelength
    ### update: doing a transformation on here. sigma is the inverse of the signal-to-noise
    ### the sigma used in the chi square is the noise, so multiply by the current signal flux.
    ### smaller synth, means smaller effective sigma, which should prioritize the centers of the absorption features.

    dof = len(flux) - 1
    chi = np.square(np.divide(flux - synth, sigma*synth)).sum()

    if (chi > 0) and np.isfinite(chi):
        return (0.5 * dof - 1) * np.log(chi) - 0.5*chi

    else:
        return -np.inf


def teff_lnprior(X, mean, sigma):
    return -np.log(sigma) - 0.5*np.square(np.divide(X - mean, sigma))


def sigma_lnprior(sigma, alpha_value, beta_value):

    ### probably want a beta function with more precision.....
    ### here's a quick fix, real fix is to develop a beta.pdf function that incorporates the log.
    prob = beta.pdf(abs(sigma), alpha_value, beta_value)

    if prob >0.0:
        return np.log(prob)

    return -np.inf


def default_param_edges(teff, feh, carbon, sigma_array):
    ### I can implement hard bounds since the MASTER grid doesn't depend on group class
    ### Would be nice if this could handle the C2 case

    if (teff < native_bounds['teff'][0]) or (teff > native_bounds['teff'][1]):
        return -np.inf

    if (feh < native_bounds['feh'][0]) or (feh > native_bounds['feh'][1]):
        return -np.inf

    if (carbon < native_bounds['cfe'][0]) or (carbon >native_bounds['cfe'][1]):
        return -np.inf

    ### same restrictions on sigmaCAII, CH, and C2 so just loop
    for item in sigma_array:
        if (item < 0.0) or (item > 1.0):
            return -np.inf

    else:
        return 0.0

def param_edges(teff, feh, carbon, sigma_array, mcmc_bounds = None, bounds = 'default'):
    ## meant for the refined search, probably a better way to do this..
    ## mcmc_bounds = mcmc_args['first_params']

    if bounds == 'final':
        ## then run with the bounds
        if (teff < mcmc_bounds['teff'][0]) or (teff > mcmc_bounds['teff'][1]):
            return -np.inf

        if (feh < mcmc_bounds['feh'][0]) or (feh > mcmc_bounds['feh'][1]):
            return -np.inf

        if (carbon < mcmc_bounds['cfe'][0]) or (carbon > mcmc_bounds['cfe'][1]):
            return -np.inf

    #### might still escape the grid so...

    return default_param_edges(teff, feh, carbon, sigma_array)

def default_feh_cfe_param_edges(feh, carbon):

    if (feh < native_bounds['feh'][0]) or (feh > native_bounds['feh'][1]):
        return -np.inf

    if (carbon < native_bounds['cfe'][0]) or (carbon >native_bounds['cfe'][1]):
        return -np.inf

    return 0.0

def feh_cfe_param_edges(feh, carbon, mcmc_bounds, bounds = 'default'):
    ## meant for the second iteration MCMC

    if bounds == 'final':
        if (feh < mcmc_bounds['feh'][0]) or (feh > mcmc_bounds['feh'][1]):
            return -np.inf

        if (carbon < mcmc_bounds['cfe'][0]) or (carbon > mcmc_bounds['cfe'][1]):
            return -np.inf

    return default_feh_cfe_param_edges(feh, carbon)
