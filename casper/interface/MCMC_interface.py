################################################################################
### Author: Devin Whitten
### Email: devin.d.whitten@gmail.com
### Institute: University of Notre Dame
################################################################################

#### Main interface for the MCMC procedure

import numpy as np
from scipy.interpolate import interp1d
import scipy
from statsmodels.nonparametric.kde import KDEUnivariate
import MLE_priors
from synthetic_functions import get_interp, normalize
import config


# import synthetic library interpolator 
INTERPOLATOR = get_interp()

def kde_param(distribution, x0):
    ### kde_param tries to ensure correct handling of multimodal distributions

    ### compute kernal density estimation
    KDE = KDEUnivariate(distribution)

    KDE.fit(bw=np.std(distribution)/3.0)

    result = scipy.optimize.minimize(lambda x: -1*KDE.evaluate(x),
    x0 = x0, method='Powell')  ## Powell has been working pretty well.

    return {'result' : float(result['x']), 'kde' : KDE}

# not used currently
def get_beta_params(spectrum, bounds):

    ## just return the proper values for alpha and beta for the given spectra.
    ## Poisson uncertainty is assumed for flux bins.
    ## alpha/beta determine the center and width of the beta function prior used for the S/N estimate.


    SN = np.divide(1.,np.sqrt(spectrum['flux'][spectrum['wave'].between(bounds[0], bounds[1], inclusive=True)]))

    u = np.median(SN)
    v = np.var(SN)

    print('SN =', np.median(SN))
    print("var(SN)=", np.var(SN))

    alpha_param = ((u**2)/v)*(1 - u) - u
    beta_param  = (1/u - 1) * alpha_param

    return {'alpha': alpha_param, 'beta': beta_param, "u": u, 'v':v}

# not used currently
def get_beta_param_bounds(spectrum, left_bounds, right_bounds, hard_var = None):
    ## trying to address the underestimation in the SN for at least CaII,
    ## should really average left and right of the feature

    SN_LEFT  = np.divide(1.,np.sqrt(spectrum['flux'][spectrum['wave'].between(left_bounds[0], left_bounds[1], inclusive=True)]))
    SN_RIGHT = np.divide(1.,np.sqrt(spectrum['flux'][spectrum['wave'].between(right_bounds[0], right_bounds[1], inclusive=True)]))

    u = np.mean([np.median(SN_LEFT), np.median(SN_RIGHT)])

    if hard_var == None:
        v = max([np.var(SN_LEFT), np.var(SN_RIGHT)])

    else:
        v = hard_var * u
        print("Manual SN variance:  ")



    print('SN      = %.3F' % u)
    print("var(SN) = ", v)

    alpha_param = ((u**2)/v)*(1 - u) - u
    beta_param  = (1/u - 1) * alpha_param

    return {'alpha': alpha_param, 'beta': beta_param, "u": u, 'v':v}

# not used currently
def transform_beta(u, v):
    ### quick hack to transform the median and variance to beta distro params
    alpha_param = ((u**2)/v)*(1 - u) - u
    beta_param  = (1/u - 1) * alpha_param

    return alpha_param, beta_param

# not used currently
def beta_param_spec(spectrum, hard_var = None):
    ### To be run in the chi_mcmc.run_chi_mcmc() routine
    ####################################################################

    param_dict = {}

    ### We're underestimating the SN and that's a problem
    CAII_BETA = get_beta_param_bounds(spectrum,
                                      left_bounds = [3884, 3923], right_bounds = [3995, 4045], hard_var=hard_var)

    CH_BETA   = get_beta_param_bounds(spectrum, left_bounds = [4000, 4080], right_bounds = [4440, 4500], hard_var=hard_var)#[4222, 4322])


    C2_BETA   = get_beta_param_bounds(spectrum,
                                      left_bounds = [4500, 4600], right_bounds = [4760, 4820], hard_var=hard_var)


    param_dict['CAII'] = CAII_BETA

    param_dict['CH']   = CH_BETA

    param_dict['C2']   = C2_BETA


    return param_dict


# When using NEW_SYNTH
# interpolating synthetic flux at a given wave range
def interp1d_synth_flux(synth_wave, G_CLASS, teff, feh, carbon):

    if np.isfinite(INTERPOLATOR[G_CLASS]([teff, feh, carbon])).all():
        synth_flux = INTERPOLATOR[G_CLASS]([teff, feh, carbon])[0]
        norm_synth_flux = normalize(synth_wave, synth_flux[config.id_start_wave:config.id_end_wave+1])

        return interp1d(synth_wave, norm_synth_flux, kind = 'linear')
    
    else : print("\t\t MCMC_interface: interp1d_synth_flux: Interpolated synthetic flux is not finite")

def likelihood_params(theta, include_C2 = False):
    if include_C2:
        # params = teff, feh, carbon, XI_CA, XI_CH, XI_C2  
        params  = theta[0], theta[1],theta[2],theta[3], theta[4], theta[5]
    else:
        # params =teff, feh, carbon, XI_CA, XI_CH
        params =  theta[0], theta[1],theta[2],theta[3], theta[4]
        
    return params


def chi_likelihood(theta, observed_spec_regions, synth_wave,
                   photo_teff,
                   photo_teff_unc,
                   SN_DICT, G_CLASS,
                   bounds = 'default'):

    ### This is an important point, that the likelihood needs to accomodate fitting and not fitting the C2 band,
    ### according to the AC value

    teff, feh, carbon, XI_CA, XI_CH = likelihood_params(theta)
    #print("\t\t MCMC_interface: INSIDE chi_likelihood:  ", teff, feh, carbon, XI_CA, XI_CH)

    synth_flux_region = interp1d_synth_flux(synth_wave, G_CLASS, teff, feh, carbon)

    if not synth_flux_region:
        print("\t\t MCMC_interface: chi_likelihood: synth_flux_region (teff={}, feh={}, carbon={})  return -np.inf".format(teff, feh, carbon))
        return -np.inf


    #print("\t\t MCMC_interface: INSIDE chi_likelihood: synthetic wave and norm_flux ", observed_spec_regions['CA']['wave'].values,synth_flux_region(observed_spec_regions['CA']['wave'].values))

    LL =  MLE_priors.ln_chi_square_sigma(observed_spec_regions['CA']['norm'].values, synth_flux_region(observed_spec_regions['CA']['wave'].values), XI_CA) + \
          MLE_priors.ln_chi_square_sigma(observed_spec_regions['CH']['norm'].values, synth_flux_region(observed_spec_regions['CH']['wave'].values), XI_CH) + \
          MLE_priors.teff_lnprior(teff, photo_teff, photo_teff_unc) + \
          MLE_priors.sigma_lnprior(XI_CA, SN_DICT['CA']['alpha'], SN_DICT['CA']['beta']) + \
          MLE_priors.sigma_lnprior(XI_CH, SN_DICT['CH']['alpha'], SN_DICT['CH']['beta']) + \
          MLE_priors.param_edges(teff, feh, carbon, [XI_CA, XI_CH], bounds)

    if np.isfinite(LL):
        return LL

    else:
        print("\t\t MCMC_interface: chi_likelihood return -np.inf")
        return -np.inf




def chi_likelihood_C2(theta, observed_spec_regions, synth_wave,
                   photo_teff,
                   photo_teff_unc,
                   SN_DICT, G_CLASS,
                   bounds = 'default'):

    ## This will get run when/if the AC is above 8

    teff, feh, carbon, XI_CA, XI_CH, XI_C2 = likelihood_params(theta, include_C2=True)
    #print("\t\t MCMC_interface: INSIDE chi_likelihood_C2:  ", teff, feh, carbon, XI_CA, XI_CH, XI_C2)

    synth_flux_region = interp1d_synth_flux(synth_wave, G_CLASS, teff, feh, carbon)

    if not synth_flux_region:
        print("\t\t MCMC_interface: chi_likelihood_C2: synth_flux_region (teff={}, feh={}, carbon={})  return -np.inf".format(teff, feh, carbon))
        return -np.inf

    #print("\t\t MCMC_interface: INSIDE chi_likelihood: synthetic wave and norm_flux ", observed_spec_regions['CA']['wave'].values,synth_flux_region(observed_spec_regions['CA']['wave'].values))

    LL =  MLE_priors.ln_chi_square_sigma(observed_spec_regions['CA']['norm'].values, synth_flux_region(observed_spec_regions['CA']['wave'].values), XI_CA) + \
          0.5*MLE_priors.ln_chi_square_sigma(observed_spec_regions['CH']['norm'].values, synth_flux_region(observed_spec_regions['CH']['wave'].values), XI_CH) + \
          0.5*MLE_priors.ln_chi_square_sigma(observed_spec_regions['C2']['norm'].values, synth_flux_region(observed_spec_regions['C2']['wave'].values), XI_C2) + \
          MLE_priors.teff_lnprior(teff,  photo_teff, photo_teff_unc) + \
          MLE_priors.sigma_lnprior(XI_CA, SN_DICT['CA']['alpha'], SN_DICT['CA']['beta']) + \
          MLE_priors.sigma_lnprior(XI_CH,   SN_DICT['CH']['alpha'], SN_DICT['CH']['beta']) + \
          MLE_priors.sigma_lnprior(XI_C2,   SN_DICT['C2']['alpha'], SN_DICT['C2']['beta']) + \
          MLE_priors.param_edges(teff, feh, carbon, [XI_CA, XI_CH, XI_C2], bounds)

    if np.isfinite(LL):
        return LL

    else:
        return -np.inf



def chi_ll_refine(theta, observed_spec_regions, synth_wave,
                  PARAMS, G_CLASS,
                  bounds = 'default'):
    ## simply [Fe/H] and [C/Fe]
    ## assume the teff, and sigma values are well determined from previous
    ## This will get run when/if the AC is above 8

    teff   = PARAMS['TEFF'][0] #theta[0]
    XI_CA  = PARAMS['XI_CA'][0]#theta[3]
    XI_CH    = PARAMS['XI_CH'][0]

    ################
    feh    = theta[0]
    carbon = theta[1]

    #print("\t\t MCMC_interface: chi_ll_refine: teff = {}, feh = {}, carbon = {}, XI_CA ={}, XI_CH = {}".format(teff, feh, carbon, XI_CA, XI_CH))
    synth_flux_region = interp1d_synth_flux(synth_wave, G_CLASS, teff, feh, carbon)
    if not synth_flux_region:
        print("\t\t MCMC_interface: chi_ll_refine: synth_flux_region (teff={}, feh={}, carbon={})  return -np.inf".format(teff,feh, carbon ))
        return -np.inf

    ### prior needs to change

    LL =  MLE_priors.ln_chi_square_sigma(observed_spec_regions['CA']['norm'].values, synth_flux_region(observed_spec_regions['CA']['wave'].values), XI_CA) + \
          MLE_priors.ln_chi_square_sigma(observed_spec_regions['CH']['norm'].values, synth_flux_region(observed_spec_regions['CH']['wave'].values), XI_CH) + \
          MLE_priors.default_feh_cfe_param_edges(feh, carbon)


    if np.isfinite(LL):
        return LL

    else:
        return -np.inf


def chi_ll_refine_C2(theta, observed_spec_regions, synth_wave,
                     PARAMS, G_CLASS,
                     bounds = 'default'):
    ## simply [Fe/H] and [C/Fe]
    ## assume the teff, and sigma values are well determined from previous
    ## This will get run when/if the AC is above 8

    teff     = PARAMS['TEFF'][0] #theta[0]
    XI_CA    = PARAMS['XI_CA'][0]#theta[3]
    XI_CH    = PARAMS['XI_CH'][0]
    XI_C2    = PARAMS['XI_C2'][0]

    ################
    feh    = theta[0]
    carbon = theta[1]

    #print("\t\t MCMC_interface: chi_ll_refine_C2: teff = {}, feh = {}, carbon = {}, XI_CA ={}, XI_CH = {}, XI_C2 = {}".format(teff, feh, carbon, XI_CA, XI_CH, XI_C2))
    
    synth_flux_region = interp1d_synth_flux(synth_wave, G_CLASS, teff, feh, carbon)

    if not synth_flux_region:
        print("\t\t MCMC_interface: chi_11_refine_C2: synth_flux_region (teff={}, feh={}, carbon={}) return -np.inf".format(teff, feh, carbon))
        return -np.inf


    ### prior needs to change

    LL =  MLE_priors.ln_chi_square_sigma(observed_spec_regions['CA']['norm'].values, synth_flux_region(observed_spec_regions['CA']['wave'].values), XI_CA) + \
          0.5*MLE_priors.ln_chi_square_sigma(observed_spec_regions['CH']['norm'].values, synth_flux_region(observed_spec_regions['CH']['wave'].values), XI_CH) + \
          0.5*MLE_priors.ln_chi_square_sigma(observed_spec_regions['C2']['norm'].values, synth_flux_region(observed_spec_regions['C2']['wave'].values), XI_C2) + \
          MLE_priors.default_feh_cfe_param_edges(feh, carbon)


    if np.isfinite(LL):
        return LL

    else:
        return -np.inf


"""

### get_mcmc_params appears to not be used so deprecated  J. Yoon ###
def get_mcmc_params(SAMPLER, burnin=0.25, return_kde=False):
    print("DEPRECATED FUNCTION")
    #ndim = SAMPLER.get_chain.shape[2]
    #SAMPLES = SAMPLER.get_chain[:, burnin:, :].reshape((-1, ndim))

    try:  ### SAMPLER is chain

        ndim = SAMPLER.shape[2]
        iter = SAMPLER.shape[1]

    except:
        SAMPLER = SAMPLER.get_chain
        ndim = SAMPLER.shape[2]
        iter = SAMPLER.shape[1]



    SAMPLES = SAMPLER[:, int(burnin * iter):, :].reshape((-1, ndim))

    MEDIAN = [np.median(array) for array in SAMPLES.T]

    STD =    [MAD.S_MAD(array) for array in SAMPLES.T]

    #modes1 = [mode(np.around(row, decimals = rounding))[0][0] for row, rounding in zip(SAMPLES.T, [0,1,1,3,3, 3])]
    #modes2 = [mode(np.around(row, decimals = rounding))[0][0] for row, rounding in zip(SAMPLES.T, [0,2,2,3,3, 3])]

    ### Let's use the kde_params
    kde_array = [kde_param(row, x0 = x0)['kde'] for row, x0 in zip(SAMPLES.T, MEDIAN)]
    value2 = [kde_param(row, x0 = x0)['result'] for row, x0 in zip(SAMPLES.T, MEDIAN)]

    if ndim == 2:
        dict_keys = ['feh', 'cfe']

    if ndim == 5:
        dict_keys = ['teff', 'feh', 'cfe', 'XI_CA', 'XI_CH']

    elif ndim == 6:
        dict_keys = ['teff', 'feh', 'cfe', 'XI_CA', 'XI_CH', 'XI_C2']


    OUTPUT = {key : [value2[i], MAD.S_MAD(SAMPLES[:, i])] for i, key in enumerate(dict_keys)}

    OUTPUT['AC'] = [ac.ac(OUTPUT['cfe'][0], OUTPUT['feh'][0]), np.sqrt(OUTPUT['cfe'][1]**2 + OUTPUT['feh'][1]**2)]

    KDE_DICT = {key : kde for key, kde in zip(dict_keys, kde_array)}

    if return_kde:
        return OUTPUT, KDE_DICT

    else:
        return OUTPUT


# not used currently
def get_post_distro(SAMPLER, index=0, burnin=500):
    ### grabs the stuff.
    ndim = SAMPLER.get_chain.shape[2]
    SAMPLES = SAMPLER.get_chain[:, burnin:, :].reshape((-1, ndim))

    return SAMPLES[:, index]



# not used currently
def set_param_bounds(param_dict):
    ############################################################################

    ############################################################################

    bounds = {}

    for key in param_dict.keys():
        bounds[key] = [param_dict[key][0] - 3.0*param_dict[key][1],
                         param_dict[key][0] + 3.0*param_dict[key][1]]

    return bounds

"""
