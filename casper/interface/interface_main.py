"""J Yoon 03/11/2022
To use multiprocessing module:
Some builds of NumPy (including the version included with Anaconda) will
automatically parallelize some operations using something like
the MKL linear algebra. This can cause problems when used with
the parallelization methods described here so it can be good to turn that off
(by setting the environment variable OMP_NUM_THREADS=1, for example).
"""

import os
from typing import Any, Dict, Literal

os.environ["OMP_NUM_THREADS"] = "1"

import time
from multiprocessing import Pool, cpu_count, current_process

import emcee
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.optimize import minimize
from statsmodels.nonparametric.kde import KDEUnivariate

from casper.interface import MAD, MCMC_interface, ac, config
from casper.interface.synthetic_functions import CAII_CH_CHI_LH, get_grav_interp, get_interp
from casper.utils.logger_config import setup_logger

logger = setup_logger(__name__)

ARCHETYPE_PARAMS = config.ARCHETYPE_PARAMS

LL_FUNCTION_DICT = {
    "COARSE": {"CH": MCMC_interface.chi_likelihood, "CH+C2": MCMC_interface.chi_likelihood_C2},
    "REFINE": {"CH": MCMC_interface.chi_ll_refine, "CH+C2": MCMC_interface.chi_ll_refine_C2},
}


INTERPOLATOR = get_interp()
GRAV_INTERP = get_grav_interp()

SYNTH_WAVE = config.SYNTH_WAVE


def synth_normalize(spectrum, group: str, temp: float) -> np.ndarray:
    """
    Normalize the synthetic flux for a given stellar spectrum.

    Parameters
    ----------
    spectrum : object
        A spectrum object that includes gravity class (G_CLASS) and mode (MODE).
    group : str
        Archetype group label (e.g., GI, GII, GIII).
    temp : float
        Effective temperature in Kelvin.

    Returns
    -------
    np.ndarray
        Continuum-normalized synthetic flux array in the identification wavelength range
        defined by config.id_start_wave to config.id_end_wave.

    Notes
    -----
    If the interpolated flux contains non-finite values, a warning is printed and
    nothing is returned.
    """

    interp_flux = INTERPOLATOR[spectrum.G_CLASS](
        temp, ARCHETYPE_PARAMS[spectrum.MODE][group]["FEH"], ARCHETYPE_PARAMS[spectrum.MODE][group]["CFE"]
    )
    if np.isfinite(interp_flux).all():
        return normalize_synth_spectrum(SYNTH_WAVE, interp_flux[config.id_start_wave : config.id_end_wave + 1])
    else:
        logger.warning(
            f"Interpolated synthetic flux is not finite, params = {temp}, "
            f"FEH = {ARCHETYPE_PARAMS[spectrum.MODE][group]['FEH']}, "
            f"CFE = {ARCHETYPE_PARAMS[spectrum.MODE][group]['CFE']}"
        )


def archetype_classify_MC(spectrum: object) -> None:
    """
    Classify the archetype group (GI, GII, GIII) for a given spectrum
    using Monte Carlo simulation and log-likelihood comparison.

    Parameters
    ----------
    spectrum : object
        Spectrum object with required attributes including:
        - teff_irfm and teff_irfm_err
        - normalized frame as a DataFrame with "wave" and "norm"
        - KP_bounds and SN_DICT
        - G_CLASS and MODE

    Notes
    -----
    This function:
    - Draws random temperature values from a normal distribution centered on teff_irfm
    - Uses those temperatures to synthesize spectra for GI, GII, and GIII groups
    - Computes log-likelihoods using CAII and CH regions
    - Selects the group with the highest median log-likelihood
    - Stores results in spectrum.LL_DICT and sets spectrum.ARCH_GROUP
    """

    length = 100
    temp_values = np.random.normal(spectrum.teff_irfm, spectrum.teff_irfm_err, length)

    start = time.time()

    GI_NORM_SYNTH = [synth_normalize(spectrum, "GI", temp) for temp in temp_values]
    GII_NORM_SYNTH = [synth_normalize(spectrum, "GII", temp) for temp in temp_values]
    GIII_NORM_SYNTH = [synth_normalize(spectrum, "GIII", temp) for temp in temp_values]

    end1 = time.time()
    logger.info(f"interface_main: archetype_classify_MC SYNTH: took {end1 - start:.1f} seconds")

    GI_LLs = np.array(
        [
            CAII_CH_CHI_LH(
                obs=spectrum.frame,
                synth=pd.DataFrame({"wave": SYNTH_WAVE, "norm": SYNTH}),
                CA_BOUNDS=spectrum.KP_bounds,
                CH_BOUNDS=config.CH_BOUNDS,
                CA_XI=spectrum.SN_DICT["CA"]["XI_AVG"],
                CH_XI=spectrum.SN_DICT["CH"]["XI_AVG"],
            )
            for SYNTH in GI_NORM_SYNTH
        ]
    )

    GII_LLs = np.array(
        [
            CAII_CH_CHI_LH(
                obs=spectrum.frame,
                synth=pd.DataFrame({"wave": SYNTH_WAVE, "norm": SYNTH}),
                CA_BOUNDS=spectrum.KP_bounds,
                CH_BOUNDS=config.CH_BOUNDS,
                CA_XI=spectrum.SN_DICT["CA"]["XI_AVG"],
                CH_XI=spectrum.SN_DICT["CH"]["XI_AVG"],
            )
            for SYNTH in GII_NORM_SYNTH
        ]
    )

    GIII_LLs = np.array(
        [
            CAII_CH_CHI_LH(
                obs=spectrum.frame,
                synth=pd.DataFrame({"wave": SYNTH_WAVE, "norm": SYNTH}),
                CA_BOUNDS=spectrum.KP_bounds,
                CH_BOUNDS=config.CH_BOUNDS,
                CA_XI=spectrum.SN_DICT["CA"]["XI_AVG"],
                CH_XI=spectrum.SN_DICT["CH"]["XI_AVG"],
            )
            for SYNTH in GIII_NORM_SYNTH
        ]
    )

    GI_LLs = GI_LLs[np.isfinite(GI_LLs)]
    GII_LLs = GII_LLs[np.isfinite(GII_LLs)]
    GIII_LLs = GIII_LLs[np.isfinite(GIII_LLs)]

    spectrum.set_group_ll(
        {
            "GI": [np.median(GI_LLs), np.std(GI_LLs)],
            "GII": [np.median(GII_LLs), np.std(GII_LLs)],
            "GIII": [np.median(GIII_LLs), np.std(GIII_LLs)],
        }
    )
    end2 = time.time()
    logger.info(f"interface_main: archetype_classify_MC LLs: took {end2 - end1:.1f} seconds")

    return


def mcmc_determination(spectrum: object, mode: str = "COARSE", burnin_factor: int = 7) -> None:
    """
    Run a Markov Chain Monte Carlo (MCMC) procedure to estimate stellar parameters
    using the input spectrum and configuration mode.

    Parameters
    ----------
    spectrum : object
        Spectrum object that must contain:
        - Spectral regions
        - Photometric temperature and error
        - Signal-to-noise dictionary
        - Gravity class
        - Carbon mode
        - MCMC iteration count
        - Previously determined COARSE parameters (for REFINE mode)

    mode : str, optional
        MCMC run mode, either "COARSE" or "REFINE". Default is "COARSE".

    burnin_factor : int, optional
        Factor to multiply by the maximum autocorrelation time to determine the
        number of burn-in steps to discard. Default is 7.

    Notes
    -----
    The function performs the following:
    - Sets up initial parameters and arguments based on run mode
    - Uses multiprocessing to speed up MCMC sampling with `emcee`
    - Computes autocorrelation time to determine burn-in
    - Stores the MCMC sampler and related diagnostics back into the spectrum object
    """

    logger.info(f"* MCMC run mode = {mode}")

    logger.info(
        f"{spectrum.get_starname()}:  {spectrum.get_gravity_class()} : {spectrum.get_carbon_mode()} : {spectrum.print_KP_bounds()}"
    )

    PARAMS = ARCHETYPE_PARAMS[spectrum.get_environ_mode()][spectrum.get_arch_group()]

    # MAIN MODE BRANCH
    if mode == "COARSE":
        logger.info(f"Initializing with archetype parameters: {PARAMS}")

        photo_teff = spectrum.get_photo_temp()

        initial = [photo_teff[0], PARAMS["FEH"], PARAMS["CFE"]]

        ARGS = (
            spectrum.regions,
            SYNTH_WAVE,
            photo_teff[0],
            photo_teff[1],
            spectrum.get_SN_dict(),
            spectrum.get_gravity_class(),
        )

        initial = np.concatenate([initial, [spectrum.SN_DICT["CA"]["XI_AVG"], spectrum.SN_DICT["CH"]["XI_AVG"]]])
        n_step = spectrum.get_MCMC_iterations()

        if spectrum.get_carbon_mode() == "CH+C2":
            logger.info("Running with carbon mode: CH+C2")

            initial = np.concatenate([initial, [spectrum.SN_DICT["C2"]["XI_AVG"]]])
        else:
            logger.info("Running with carbon mode: CH only")

    elif mode == "REFINE":
        PARAMS_0 = spectrum.get_mcmc_dict(mode="COARSE")
        logger.info("Initializing with COARSE run result parameters:")

        logger.info(
            " Teff : %.0f  [Fe/H] : %.2f   [C/Fe] : %.2f   A(C) : %.2f"
            % (PARAMS_0["TEFF"][0], PARAMS_0["FEH"][0], PARAMS_0["CFE"][0], PARAMS_0["AC"][0])
        )

        ARGS = (spectrum.regions, SYNTH_WAVE, PARAMS_0, spectrum.get_gravity_class())

        initial = [spectrum.MCMC_COARSE["FEH"][0], spectrum.MCMC_COARSE["CFE"][0]]
        n_step = int(spectrum.get_MCMC_iterations() / 4)

    else:
        logger.error("Invalid mode")

    LL_FUNCTION = LL_FUNCTION_DICT[mode][spectrum.get_carbon_mode()]

    n_cpu = cpu_count()
    logger.info(f"Number of CPU cores available: {n_cpu}")

    # Gaussian distribution
    pos = initial + initial * (2e-2 * np.random.rand(64, len(initial)))
    logger.info(f"interface_main: pos = {pos}")

    nwalkers, ndim = pos.shape

    logger.info(f"Running for {n_step} iterations...")

    with Pool() as pool:
        logger.info(f"Process {current_process().name} started working")

        sampler = emcee.EnsembleSampler(
            nwalkers,
            ndim,
            LL_FUNCTION,
            moves=[
                (emcee.moves.DEMove(), 0.8),
                (emcee.moves.DESnookerMove(), 0.2),
            ],
            pool=pool,
            args=(ARGS),
        )
        start = time.time()

        _ = sampler.run_mcmc(pos, n_step, skip_initial_state_check=False, progress=True)
        end = time.time()
        multi_time = end - start
        logger.info(f"Process {current_process().name} ended working")

        logger.info(f"\t \t MCMC Multiprocessing took {multi_time:.1f} seconds")

    spectrum.set_sampler(sampler, mode=mode)

    tau = sampler.get_autocorr_time(quiet=True)

    num_valid_autocorr_time_value = len(tau) - np.isnan(tau).sum()
    logger.info(
        "\t\t interface_main: tau's shape= {}, length ={}, how many nan values = {}".format(
            tau.shape, len(tau), np.isnan(tau).sum()
        )
    )

    logger.info(
        "\t\t interface_main: tau = {}, num_valid_autocorr_time_value ={} ".format(tau, num_valid_autocorr_time_value)
    )

    if num_valid_autocorr_time_value == 0:
        logger.warning("\t\t interface_main: all autocorr_times are Nan!")

        max_auto_corr_time = 70

    elif num_valid_autocorr_time_value == 1:
        logger.warning("\t\t interface_main: all except one dim autocorr_time are Nan")

        for taulist in tau:
            if not np.isnan(taulist):
                max_auto_corr_time = taulist

    else:
        logger.info("\t\t interface_main: n >= 2 in tau array values are vaild numbers ")

        max_auto_corr_time = np.nanmax(tau)
        logger.info(f"\t\t interface_main: maximum autocorrelation time = {max_auto_corr_time}")

    n_discard = int(burnin_factor * max_auto_corr_time)

    if n_discard >= 0.5 * n_step:
        logger.warning(
            "\t\t interface_main: n_discard is larger than the mcmc iterations! Setting n_discard to half the iterations. "
        )

        n_discard = int(0.5 * n_step)
    logger.info(f"\t\t mcmc mode = {mode}")

    mean_acc_fraction = np.mean(sampler.acceptance_fraction)
    logger.info(f"\t\t interface_main: mean acceptance fraction: {mean_acc_fraction:.3f}")

    logger.info(f"\t\t interface_main: max autocorrelation_time = {max_auto_corr_time}")

    logger.info(f"\t\t interface_main: recommended n_discard = {n_discard}")

    if mode == "COARSE":
        spectrum.mcmc_coarse_acc_frac = mean_acc_fraction
        spectrum.mcmc_coarse_tau = max_auto_corr_time
        spectrum.mcmc_coarse_n_discard = n_discard
    else:
        spectrum.mcmc_refine_acc_frac = mean_acc_fraction
        spectrum.mcmc_refine_tau = max_auto_corr_time
        spectrum.mcmc_refine_n_discard = n_discard

    return


def generate_synthetic(spectrum: object) -> None:
    """
    Generate and set the best synthetic spectrum for a star based on MCMC parameters.

    Parameters
    ----------
    spectrum : object
        Spectrum object that must contain:
        - Gravity class
        - MCMC_COARSE["TEFF"] value
        - MCMC_REFINE["FEH"] and ["CFE"] values

    Notes
    -----
    This function:
    - Uses an interpolator to compute synthetic flux values
    - Normalizes the flux using a GISIC-based routine
    - Assigns the normalized synthetic spectrum to the spectrum object
    - Handles invalid or NaN flux cases gracefully by filling with NaNs
    """

    interp_flux = INTERPOLATOR[spectrum.get_gravity_class()](
        spectrum.MCMC_COARSE["TEFF"][0], spectrum.MCMC_REFINE["FEH"][0], spectrum.MCMC_REFINE["CFE"][0]
    )

    if np.isfinite(interp_flux).all():
        NORM_SYNTH_FLUX = normalize_synth_spectrum(
            SYNTH_WAVE, interp_flux[config.id_start_wave : config.id_end_wave + 1]
        )
        spectrum.set_synth_spectrum(pd.DataFrame({"wave": SYNTH_WAVE, "norm": NORM_SYNTH_FLUX.T}))
    else:
        logger.warning(
            f"generate_synthetic: Interpolated synthetic flux is not finite, params = "
            f"{spectrum.MCMC_COARSE['TEFF'][0]}, {spectrum.MCMC_REFINE['FEH'][0]}, {spectrum.MCMC_REFINE['CFE'][0]}"
        )

        logger.warning(f"the sequence is {spectrum.get_sequence()} and the star name is {spectrum.get_starname()}")

        spectrum.set_synth_spectrum(pd.DataFrame({"wave": SYNTH_WAVE, "norm": np.nan * np.ones_like(SYNTH_WAVE)}))

    return


def estimate_logg(spectrum: object) -> None:
    """
    Estimate and assign the surface gravity (logg) and its uncertainty for the input spectrum.

    Parameters
    ----------
    spectrum : object
        Spectrum object containing MCMC parameters and samplers.
        Must include:
        - Gravity class
        - MCMC_COARSE and MCMC_REFINE parameter dictionaries
        - MCMC_COARSE_sampler
        - Burn-in discard value (mcmc_coarse_n_discard)

    Notes
    -----
    - The logg value is interpolated using effective temperature and metallicity.
    - Uncertainty is computed using the scaled MAD from the MCMC COARSE samples.
    - Standard deviation and raw MAD are also printed for reference.
    """

    spectrum.logg = GRAV_INTERP[spectrum.get_gravity_class()](
        spectrum.MCMC_COARSE["TEFF"][0], spectrum.MCMC_REFINE["FEH"][0]
    )
    logger.info(f"\t\t interface_main: logg = {spectrum.logg}")

    samples_COARSE = spectrum.MCMC_COARSE_sampler.get_chain(discard=spectrum.mcmc_coarse_n_discard, thin=1, flat=True)
    teff_dist = samples_COARSE[:, 0]
    feh_COARSE_dist = samples_COARSE[:, 1]
    logg_COARSE_dist = GRAV_INTERP[spectrum.get_gravity_class()](teff_dist, feh_COARSE_dist)
    spectrum.logg_err = MAD.S_MAD(logg_COARSE_dist)
    logg_err_std = np.std(logg_COARSE_dist)
    logg_err_mad = MAD.MAD(logg_COARSE_dist)

    logger.info(f"\t\t interface_main: logg = {spectrum.logg} +/- {spectrum.logg_err}")

    logger.info(f"\t\t interface_main: logg_std = {logg_err_std}, logg_mad = {logg_err_mad}")

    return


def kde_param_reflection(distro: np.ndarray) -> Dict[str, Any]:
    """
    Estimate the peak of a potentially multimodal distribution using kernel density estimation (KDE)
    with reflective padding to reduce edge effects.

    Parameters
    ----------
    distro : np.ndarray
        One-dimensional array representing the parameter distribution. Must contain numeric values.

    Returns
    -------
    Dict[str, Any]
        A dictionary with the following keys:
        - "result": The peak value (mode) estimated from the KDE with reflection.
        - "kde": The KDE fit to the original distribution.
        - "kde_reflect": A scaled interpolation function of the KDE fit to the reflected distribution.

    Notes
    -----
    - This method reduces boundary bias by reflecting the distribution at its minimum and maximum.
    - The same bandwidth is used for both the original and reflected KDE.
    - The Powell optimization method is used to find the maximum of the reflected KDE.
    """

    distro = distro[np.isfinite(distro)]

    MIN, MAX = min(distro), max(distro)
    span = np.linspace(MIN, MAX, 200)

    lower = MIN - abs(distro - MIN)
    upper = MAX + abs(distro - MAX)

    merge = np.concatenate([lower, distro, upper])

    KDE_MAIN = KDEUnivariate(distro)
    KDE_FULL = KDEUnivariate(merge)

    KDE_MAIN.fit(bw=np.std(distro) / 4.0)
    KDE_FULL.fit(bw=np.std(distro) / 4.0)

    scale = np.median(np.divide(KDE_MAIN.evaluate(span), KDE_FULL.evaluate(span)))

    result = minimize(
        lambda x: -1 * KDE_FULL.evaluate(x),
        x0=span[KDE_MAIN.evaluate(span) == max(KDE_MAIN.evaluate(span))],
        method="Powell",
    )

    return {
        "result": float(result["x"]),
        "kde": KDE_MAIN,
        "kde_reflect": interp1d(span, KDE_FULL.evaluate(span) * scale),
    }


def generate_kde_params(spectrum, mode: Literal["COARSE", "REFINE"], n_thin: int = 1) -> None:
    """
    Perform kernel density estimation (KDE) on the MCMC sampler chain to estimate
    parameter values and uncertainties for a given spectrum.

    Parameters
    ----------
    spectrum : Spectrum
        An instance of the Spectrum class containing MCMC sampler results and configuration data.

    mode : Literal["COARSE", "REFINE"]
        Determines whether to use the coarse or refined MCMC sampler chain.

    n_thin : int, optional
        Thinning factor for the MCMC chain. Default is 1 (no thinning).

    Sets
    ----
    - MCMC parameter estimates (e.g., effective temperature, iron to hydrogen ratio,
      carbon to iron ratio, signal-to-noise based uncertainty terms)
    - KDE objects for each parameter to support probability density evaluation

    Notes
    -----
    - For `ndim == 2`, this function estimates iron to hydrogen ratio and carbon to iron ratio.
    - For `ndim == 5` or `6`, it additionally estimates effective temperature and signal-to-noise terms.
    - The abundance of carbon (A(C)) is calculated from the iron and carbon ratios.
    - The KDE is reflected at the edges to reduce boundary artifacts.
    """

    if mode == "COARSE":
        chain = spectrum.MCMC_COARSE_sampler.get_chain(discard=spectrum.mcmc_coarse_n_discard, thin=n_thin, flat=True)

    elif mode == "REFINE":
        chain = spectrum.MCMC_REFINE_sampler.get_chain(discard=spectrum.mcmc_refine_n_discard, thin=n_thin, flat=True)

    ndim = chain.shape[1]

    results = [kde_param_reflection(array) for array in chain.T]

    if ndim == 2:
        dict_keys = ["FEH", "CFE"]

    if ndim == 5:
        dict_keys = ["TEFF", "FEH", "CFE", "XI_CA", "XI_CH"]

    elif ndim == 6:
        dict_keys = ["TEFF", "FEH", "CFE", "XI_CA", "XI_CH", "XI_C2"]

    OUTPUT = {key: [results[i]["result"], MAD.S_MAD(chain[:, i])] for i, key in enumerate(dict_keys)}

    OUTPUT["AC"] = [ac.ac(OUTPUT["CFE"][0], OUTPUT["FEH"][0]), np.sqrt(OUTPUT["CFE"][1] ** 2 + OUTPUT["FEH"][1] ** 2)]

    KDE_DICT = {key: [element["kde"], element["kde_reflect"]] for key, element in zip(dict_keys, results)}

    spectrum.set_mcmc_results(OUTPUT, mode=mode)
    spectrum.set_kde_functions(KDE_DICT, mode=mode)

    return
