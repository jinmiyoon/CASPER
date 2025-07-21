# not used currently
import numpy as np


def get_beta_params(spectrum, bounds):
    ## just return the proper values for alpha and beta for the given spectra.
    ## Poisson uncertainty is assumed for flux bins.
    ## alpha/beta determine the center and width of the beta function prior used for the S/N estimate.

    SN = np.divide(1.0, np.sqrt(spectrum["flux"][spectrum["wave"].between(bounds[0], bounds[1], inclusive=True)]))

    u = np.median(SN)
    v = np.var(SN)

    print("SN =", np.median(SN))
    print("var(SN)=", np.var(SN))

    alpha_param = ((u**2) / v) * (1 - u) - u
    beta_param = (1 / u - 1) * alpha_param

    return {"alpha": alpha_param, "beta": beta_param, "u": u, "v": v}


# not used currently
def get_beta_param_bounds(spectrum, left_bounds, right_bounds, hard_var=None):
    ## trying to address the underestimation in the SN for at least CaII,
    ## should really average left and right of the feature

    SN_LEFT = np.divide(
        1.0, np.sqrt(spectrum["flux"][spectrum["wave"].between(left_bounds[0], left_bounds[1], inclusive=True)])
    )
    SN_RIGHT = np.divide(
        1.0, np.sqrt(spectrum["flux"][spectrum["wave"].between(right_bounds[0], right_bounds[1], inclusive=True)])
    )

    u = np.mean([np.median(SN_LEFT), np.median(SN_RIGHT)])

    if hard_var == None:
        v = max([np.var(SN_LEFT), np.var(SN_RIGHT)])

    else:
        v = hard_var * u
        print("Manual SN variance:  ")

    print("SN      = %.3F" % u)
    print("var(SN) = ", v)

    alpha_param = ((u**2) / v) * (1 - u) - u
    beta_param = (1 / u - 1) * alpha_param

    return {"alpha": alpha_param, "beta": beta_param, "u": u, "v": v}


# not used currently
def transform_beta(u, v):
    ### quick hack to transform the median and variance to beta distro params
    alpha_param = ((u**2) / v) * (1 - u) - u
    beta_param = (1 / u - 1) * alpha_param

    return alpha_param, beta_param


# not used currently
def beta_param_spec(spectrum, hard_var=None):
    ### To be run in the chi_mcmc.run_chi_mcmc() routine
    ####################################################################

    param_dict = {}

    ### We're underestimating the SN and that's a problem
    CAII_BETA = get_beta_param_bounds(spectrum, left_bounds=[3884, 3923], right_bounds=[3995, 4045], hard_var=hard_var)

    CH_BETA = get_beta_param_bounds(
        spectrum, left_bounds=[4000, 4080], right_bounds=[4440, 4500], hard_var=hard_var
    )  # [4222, 4322])

    C2_BETA = get_beta_param_bounds(spectrum, left_bounds=[4500, 4600], right_bounds=[4760, 4820], hard_var=hard_var)

    param_dict["CAII"] = CAII_BETA

    param_dict["CH"] = CH_BETA

    param_dict["C2"] = C2_BETA

    return param_dict
