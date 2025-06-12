################################################################################
### Author: Devin Whitten, Jinmi Yoon
### Email: devin.d.whitten@gmail.com, jinmi.yoon@gmail.com
################################################################################
# Date: Nov 12, 2016
# This is will serve as the interface for the normalization function.
# So just defining some functions in here.

## Modifying to operate on synthetic spectra
## Jul 15 2020 by Jinmi Yoon
## This routine is under CASPER/interface/


import config
import MAD
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d


###############################
# Spectrum Class Definition
################################
def obtain_flux(data):
    ### This is a catch all function to hopefully properly address fits data format variety
    shape = data.shape

    ##### Case 1
    if len(shape) == 1:
        ### simplest case, just grab array
        return data.flatten()

    else:
        ## grab first row and hope for the best
        return data[0].flatten()


class Spectrum:
    def __init__(self, spec, filename, is_fits=True):
        self.filename = filename
        print("\n... initializing:  ", filename)
        ################################################################################
        if is_fits:
            ## This is a cumbersome attempt to accomodate multiple fits data formats..
            if "CD1_1" in spec[0].header:
                DELTA = "CD1_1"

            elif "CDELT1" in spec[0].header:
                DELTA = "CDELT1"

            else:
                print("I don't know which increment to use!")

            if spec[0].header["CRVAL1"] > 10.0:
                # print("linear wavelength")
                self.wavelength = (np.arange(0, spec[0].header["NAXIS1"], 1) * spec[0].header[DELTA]) + spec[0].header[
                    "CRVAL1"
                ]

            else:
                self.wavelength = np.power(
                    10.0, (spec[0].header["CRVAL1"] + np.arange(0, spec[0].header["NAXIS1"]) * spec[0].header[DELTA])
                )

            self.original_wavelength = self.wavelength

            self.flux = obtain_flux(spec[0].data)
            self.wavelength = np.array(self.wavelength)
            print("spectrum loaded")
            # not to keep fits file open
            spec.close()

            # check endian match
            # self.endian_match = (self.flux.dtype.byteorder == self.wavelength.dtype.byteorder)

            if self.flux.dtype.byteorder == ">":
                print("... correcting endian mismatch")
                self.flux = self.flux.byteswap().view(self.flux.dtype.newbyteorder())

            # print(self.flux.dtype.byteorder == self.wavelength.dtype.byteorder)
        ################################################################################

        else:
            print("\t csv file, ")
            self.spec = spec
            self.flux = self.spec["flux"]
            self.wavelength = np.array(self.spec["wave"], dtype=float)
            self.original_wavelength = self.wavelength

        #### Defined in generate_segments
        self.segments = None
        ####
        self.mad_global = None

        return

    def radial_correction(self, velocity=0.0):
        ### corrects the wavelength shift for given radial velocity
        self.rv = velocity
        # Later, I would use astropy constant for speed of light, Sep 02 2020, J. Yoon
        self.wavelength = self.original_wavelength / ((velocity / 2.99792e5) + 1)

        return

    def ebv_correct(self, row):
        ## Basically if the EBV is finite,
        ## assume that photometry needs to be corrected

        ##-- Jinmi Yoon 06-12-2020
        ## MAke sure the colors used in CASPER are from UKIRT colors.
        ## If you have 2MASS colors, convert them to the UKIRT colors by following
        ## a transformation equation found at https://www.astro.caltech.edu/~jmc/2mass/v3/transformations/
        ## (Ks)2MASS     =    KUKIRT + (0.003 ± 0.004) + (0.004 ± 0.006)(J-K)UKIRT
        ## (J-H)2MASS    =    (1.075 ± 0.013)(J-H)UKIRT + (-0.032 ± 0.006)
        ## (J-Ks)2MASS   =    (1.070 ± 0.008)(J-K)UKIRT + (-0.015 ± 0.006)
        ## (H-Ks)2MASS   =    (1.071 ± 0.026)(H-K)UKIRT + (0.014 ± 0.005)

        self.PHOTO_0 = {key: float(row[key]) for key in ["J-K", "H-K", "H-K", "g-r"]}

        if float(row["EBV_SFD"]) > 0:
            ### Then perform the correction
            print("\t corrected :", self.get_filename())

            self.PHOTO_0["J-K"] = float(row["J-K"]) - (float(config.A_EBV["A_J"]) - float(config.A_EBV["A_K"])) * float(
                row["EBV_SFD"]
            )

            self.PHOTO_0["H-K"] = float(row["H-K"]) - (float(config.A_EBV["A_H"]) - float(config.A_EBV["A_K"])) * float(
                row["EBV_SFD"]
            )

            self.PHOTO_0["g-r"] = float(row["g-r"]) - (float(config.A_EBV["A_g"]) - float(config.A_EBV["A_r"])) * float(
                row["EBV_SFD"]
            )

        else:
            ## THEN ASSUME COLORS ARE ALREADY CORRECTED
            print("\t already corrected:  ", self.get_filename())

    ############################################################
    # def trim_frame(self, bounds= [3000, 5000]):
    def trim_frame(self, bounds=config.WAVE_BOUNDS):
        self.frame = self.frame[self.frame["wave"].between(bounds[0], bounds[1], inclusive="both")]
        return

    def estimate_sn(self):
        #### determines the first guess SN estimates for each region of interest
        #### defines SN_DICT member variable

        self.SN_DICT = {key: [] for key in config.SIDEBANDS.keys()}
        for key in config.SIDEBANDS.keys():
            if (config.SIDEBANDS[key][0][0] > min(self.frame["wave"])) and (
                config.SIDEBANDS[key][1][1] < max(self.frame["wave"])
            ):
                SN_LEFT = np.sqrt(
                    self.frame["flux"][self.frame["wave"].between(*config.SIDEBANDS[key][0], inclusive="both")]
                )
                SN_RIGHT = np.sqrt(
                    self.frame["flux"][self.frame["wave"].between(*config.SIDEBANDS[key][1], inclusive="both")]
                )

                ### Average the left and right config.SIDEBANDS

                self.SN_DICT[key] = {
                    "SN_AVG": np.mean([np.median(SN_LEFT), np.median(SN_RIGHT)]),
                    "SN_STD": max([MAD.S_MAD(SN_LEFT), MAD.S_MAD(SN_RIGHT)]),
                    "XI_AVG": np.mean([np.median(np.divide(1.0, SN_LEFT)), np.median(np.divide(1.0, SN_RIGHT))]),
                    "XI_STD": max([MAD.S_MAD(np.divide(1.0, SN_LEFT)), MAD.S_MAD(np.divide(1.0, SN_RIGHT))]),
                }

                #### parameters for the beta distribution prior
                self.SN_DICT[key]["alpha"] = (
                    (self.SN_DICT[key]["XI_AVG"] ** 2) / np.square(self.SN_DICT[key]["XI_STD"])
                ) * (1 - self.SN_DICT[key]["XI_AVG"]) - self.SN_DICT[key]["XI_AVG"]
                self.SN_DICT[key]["beta"] = (1 / self.SN_DICT[key]["XI_AVG"] - 1) * self.SN_DICT[key]["alpha"]

            else:
                print("band not in wavelength coverage")

                self.SN_DICT[key] = {
                    "SN_AVG": np.nan,
                    "SN_STD": np.nan,
                    "XI_AVG": np.nan,
                    "XI_STD": np.nan,
                    "alpha": np.nan,
                    "beta": np.nan,
                }
        return

    def get_sn(self) -> pd.DataFrame:
        sn_output = pd.DataFrame(
            {
                "SEQUENCE": [self.get_sequence()],
                "FILENAME": [self.get_filename()],
                "SN_AVG_CA": [round(self.SN_DICT["CA"]["SN_AVG"], 0)],
                "SN_STD_CH": [round(self.SN_DICT["CA"]["SN_STD"], 0)],
                "SN_AVG_CA": [round(self.SN_DICT["CH"]["SN_AVG"], 0)],
                "SN_STD_CH": [round(self.SN_DICT["CH"]["SN_STD"], 0)],
                "XI_AVG_CA": [round(self.SN_DICT["CA"]["XI_AVG"], 4)],
                "XI_STD_CA": [round(self.SN_DICT["CA"]["XI_STD"], 4)],
                "XI_AVG_CH": [round(self.SN_DICT["CH"]["XI_AVG"], 4)],
                "XI_STD_CH": [round(self.SN_DICT["CH"]["XI_STD"], 4)],
                "XI_AVG_C2": [round(self.SN_DICT["C2"]["XI_AVG"], 4)],
                "XI_STD_C2": [round(self.SN_DICT["C2"]["XI_STD"], 4)],
            }
        )
        if self.INPUT_CARBON_MODE == "CH+C2":
            sn_c2_output = pd.DataFrame(
                {
                    "XI_C2": [round(self.SN_DICT["C2"]["XI_AVG"], 4)],
                    "XI_C2_ERR": [round(self.SN_DICT["C2"]["XI_STD"], 4)],
                }
            )
            sn_output = pd.concat([sn_output, sn_c2_output], axis=1)
        return sn_output

    #################################################
    ### Total mutators
    def set_params(self, SEQUENCE, STARNAME, CLASS, JK, MODE, INPUT_CARBON_MODE, iter, T_SIGMA, HARD_TEFF):
        # def set_params(self,CLASS, JK, MODE, iter, T_SIGMA, HARD_TEFF):
        self.SEQUENCE = str(SEQUENCE)
        self.STARNAME = str(STARNAME)
        self.G_CLASS = str(CLASS)
        self.JK = JK
        self.MODE = str(MODE)
        self.INPUT_CARBON_MODE = str(INPUT_CARBON_MODE)
        self.MCMC_iterations = iter
        self.T_SIGMA = float(T_SIGMA)
        self.HARD_TEFF = float(HARD_TEFF)
        assert (self.G_CLASS == "GIANT") or (self.G_CLASS == "DWARF"), "Invalid gravity class: {}".format(self.G_CLASS)
        assert (self.MODE == "UFD") or (self.MODE == "HALO"), "Invalid Galactic Environment"
        # assert (self.INPUT_CARBON_MODE =='CH') or (self.INPUT_CARBON_MODE =='CH+C2'), "Invalid Enviornment: {}".format(self.carbon_mode)

        return

    def set_KP_bounds(self, input_bounds):
        ## should be a list
        self.KP_bounds = input_bounds
        return

    # def set_CH_bounds(self, )

    # created this function to provide an option to manually set
    # carbon_mode: CH or CH+C2 modes   J. Yoon
    def set_carbon_mode(self, carbon_mode):
        self.carbon_mode = carbon_mode
        return

    def set_group_ll(self, input_dict):
        self.LL_DICT = input_dict

        LLs = [self.LL_DICT[key][0] for key in self.LL_DICT.keys()]

        GROUP = ["GI", "GII", "GIII"][LLs.index(max(LLs))]

        print("\t " + self.get_filename().ljust(20) + ": ", GROUP, ["%.2F" % val for val in LLs])

        self.ARCH_GROUP = GROUP

        return

    def set_temperature(self, input_temp, sigma):
        ### for use with the calibrate_temperatures function
        ### input_dict:  {"Casagrande":, "Hernandez":, "Bergeat": }
        print(f"\t\t batch.set_temperatue(): temp={input_temp}, sigma={sigma}")
        self.teff_irfm = input_temp
        self.teff_irfm_err = sigma

        """
        # 12/13/2021, J Yoon.
        # I dont understand why Devin wrote this way below.
        # Perhaps, he meant to do something else.

        if hard == True:
            self.teff_irfm = input_temp
            self.teff_irfm_err = sigma
            return

        else:
            self.teff_irfm = input_temp
            self.teff_irfm_err = sigma

        """

        return

    def prepare_regions(self):
        ### prepares the CaII, CH, and C2 regions according to KP_bounds and carbon_mode

        self.regions = {
            "CA": self.frame[self.frame["wave"].between(*self.KP_bounds, inclusive="both")].copy(),
            "CH": self.frame[self.frame["wave"].between(4222, 4322, inclusive="both")].copy(),
        }

        if self.carbon_mode == "CH+C2":
            ### then add the C2 cut
            self.regions["C2"] = self.frame[self.frame["wave"].between(4710, 4750, inclusive="both").copy()]

        return

    def set_temp_frame(self, TEMP_FRAME):
        # print(f"\t\t batch.set_temp_frame(): temp={TEMP_FRAME}")
        self.TEMP_FRAME = TEMP_FRAME
        return

    def set_mcmc_args(self, input_dict=None):
        ## I'll finish if necessary
        if input_dict != None:
            self.mcmc_args = input
        else:
            self.mcmc_args = {}

        return

    def set_mcmc_results(self, input_dict, mode):
        ## I want to anticipate the refined and coarse outputs

        if mode == "COARSE":
            self.MCMC_COARSE = input_dict

        elif mode == "REFINE":
            self.MCMC_REFINE = input_dict

        else:
            print("Invalid mode in set_mcmc_results()")

        return

    def set_sampler(self, input_sampler, mode="COARSE"):
        if mode == "COARSE":
            self.MCMC_COARSE_sampler = input_sampler
        elif mode == "REFINE":
            self.MCMC_REFINE_sampler = input_sampler
        return

    def set_kde_functions(self, input_dict, mode):
        ## I want to anticipate the refined and coarse outputs

        if mode == "COARSE":
            self.KDE_COARSE = input_dict

        elif mode == "REFINE":
            self.KDE_REFINE = input_dict

        else:
            print("Invalid mode in set_mcmc_results()")

        return

    def set_flux(self, input_flux):
        ## Just a hard set function in case of format problems with the fits data section
        # observed flux
        self.flux = input_flux
        return

    def set_norm(self, input_flux):
        ## intended for the external batch normalization
        # observed norm flux
        self.norm = input_flux

        return

    def set_GBAND(self, input):
        ## might be interesting someday..
        self.GBAND_EW = input
        return

    def set_frame(self, wave, flux):
        # Trim the frame within the wave bounds we are interested in.
        # Since the observed wave array size is different from the synthetic wave,
        # we need to set they are the same size. Future work below. FRAME_WAVE should be the same as SYNTH_WAVE
        # new_frame_wave = config.FRAME_WAVE
        # new_frame_flux = interp1d(config.FRAME_WAVE, flux, kind='linear')
        # self.frame = pd.DataFrame({'wave': config.FRAME_WAVE, 'flux': new_frame_flux})

        # setting pandas DF of the original spectra
        self.frame = pd.DataFrame({"wave": wave, "flux": flux})
        return

    # def set_frame_wave(self, input_wave):
    #     self.frame.loc[:, 'wave'] = input_wave
    #     return

    # def set_frame_flux(self, input_flux):
    #     self.frame.loc[:, 'flux'] = input_flux
    #     return

    def set_frame_norm(self, input_norm):
        self.frame.loc[:, "norm"] = input_norm
        return

    def set_frame_cont(self, input_cont):
        self.frame.loc[:, "cont"] = input_cont
        return

    def set_synth_spectrum(self, synth):
        self.synth_spectrum = synth
        return

    def get_sequence(self):
        return "{:s}".format(self.SEQUENCE)

    def get_filename(self):
        return "{:<20}".format(self.filename)

    def get_starname(self):
        return "{:<25}".format(self.STARNAME)

    def get_wave(self):
        # wavelength of observed spectra
        return self.wavelength

    # def get_norm(self):
    #     return self.norm

    def get_flux(self):
        # observed flux
        return self.flux

    def get_frame(self):
        # get pandas frame of wavelength and flux of interest (limited for CASPER analysis)
        return self.frame

    def get_frame_wave(self):
        # get frame (limited) wavelength for CASPER analysis
        return self.frame["wave"]

    def get_frame_flux(self):
        # get frame (limited) flux for CASPER analysis
        return self.frame["flux"]

    def get_frame_norm(self):
        # get frame (limited) normalized flux for CASPER analysis
        return self.norm["norm"]

    def get_gravity_class(self):
        return self.G_CLASS

    def get_logg(self):
        return self.logg.item(), self.logg_err

    def get_carbon_mode(self):
        return self.carbon_mode

    def get_KP_bounds(self):
        return self.KP_bounds

    def get_environ_mode(self):
        return self.MODE

    def get_arch_group(self):
        return self.ARCH_GROUP

    def get_photo_temp(self):
        return self.teff_irfm, self.teff_irfm_err

    def get_rv(self):
        return self.rv

    def get_SN_dict(self):
        return self.SN_DICT

    def get_kde_dict(self):
        if hasattr(self, "KDE_REFINE"):
            return self.KDE_COARSE, self.KDE_REFINE

        return self.KDE_COARSE

    def get_mcmc_dict(self, mode="COARSE"):
        if mode == "COARSE":
            return self.MCMC_COARSE

        elif mode == "REFINE":
            return self.MCMC_REFINE

        elif mode == "BOTH":
            return self.MCMC_COARSE, self.MCMC_REFINE

        else:
            print("Bad mode:  ", mode)
            return np.nan

    def get_MCMC_iterations(self):
        return self.MCMC_iterations

    """
    def get_errors(self):

        self.feh_err = max([self.MCMC_REFINE['FEH'][1], self.MCMC_COARSE['FEH'][1]])
        self.cfe_err = max([self.MCMC_REFINE['CFE'][1], self.MCMC_COARSE['CFE'][1]])
        self.ac_err = max([self.MCMC_REFINE['AC'][1], self.MCMC_COARSE['AC'][1]])

        return
        """

    def get_spectra_row(self):
        output_spectra = pd.DataFrame(
            {
                "OBSERVED_WAVE": [self.frame["wave"].values.tolist()],
                "OBSERVED_FLUX": [self.frame["norm"].values.tolist()],
                "SYNTH_WAVE": [self.synth_spectrum["wave"].values.tolist()],
                "SYNTHETIC_FLUX": [self.synth_spectrum["norm"].values.tolist()],
            }
        )
        return output_spectra

    def get_output_row(self):
        ## simply produces a dataframe row with the desired outputs
        # 01-04-2022 added  a missing suffix ('_UNC') for TEFF_IRFM_UNC
        output = pd.DataFrame(
            {
                "SEQUENCE": [self.get_sequence()],
                "FILENAME": [self.get_filename()],
                "STARNAME": [self.get_starname()],
                "ENV_MODE": [self.get_environ_mode()],
                "CAR_MODE": [self.get_carbon_mode()],
                "CEMP_GRP_TENT": [self.get_arch_group()],
                "GCLASS": [self.get_gravity_class()],
                "TEFF": [round(self.MCMC_COARSE["TEFF"][0], 0)],
                "TEFF_ERR": [round(self.MCMC_COARSE["TEFF"][1], 0)],
                "TEFF_ADT": [round(self.teff_irfm)],
                "TEFF_ADT_ERR": [self.teff_irfm_err],
                "LOGG": [round(self.logg.item(), 2)],
                "LOGG_ERR": [round(self.logg_err, 2)],
                # 'LOGG'     : [round(self.get_logg()[0], 2)],
                # 'LOGG_ERR'     : [round(self.get_logg()[1], 2)],
                "FEH": [round(self.MCMC_REFINE["FEH"][0], 2)],
                "FEH_ERR": [round(max([self.MCMC_REFINE["FEH"][1], self.MCMC_COARSE["FEH"][1]]), 2)],
                "CFE": [round(self.MCMC_REFINE["CFE"][0], 2)],
                "CFE_ERR": [round(max([self.MCMC_REFINE["CFE"][1], self.MCMC_COARSE["CFE"][1]]), 2)],
                "AC": [round(self.MCMC_REFINE["AC"][0], 2)],
                "AC_ERR": [round(max([self.MCMC_REFINE["AC"][1], self.MCMC_COARSE["AC"][1]]), 2)],
                "RV": [round(self.get_rv(), 1)],
                "OBSERVED_WAVE": [self.frame["wave"].values.tolist()],
                "OBSERVED_FLUX": [self.frame["norm"].values.tolist()],
                "SYNTH_WAVE": [self.synth_spectrum["wave"].values.tolist()],
                "SYNTHETIC_FLUX": [self.synth_spectrum["norm"].values.tolist()],
                "XI_CA": [round(self.SN_DICT["CA"]["XI_AVG"], 4)],
                "XI_CA_ERR": [round(self.SN_DICT["CA"]["XI_STD"], 4)],
                "XI_CH": [round(self.SN_DICT["CH"]["XI_AVG"], 4)],
                "XI_CH_ERR": [round(self.SN_DICT["CH"]["XI_STD"], 4)],
            }
        )
        if self.INPUT_CARBON_MODE == "CH+C2":
            c2_output = pd.DataFrame(
                {
                    "XI_C2": [round(self.SN_DICT["C2"]["XI_AVG"], 4)],
                    "XI_C2_ERR": [round(self.SN_DICT["C2"]["XI_STD"], 4)],
                }
            )
            output = pd.concat([output, c2_output], axis=1)

        sn_output = pd.DataFrame(
            {
                "SN_AVG_CA": [round(self.SN_DICT["CA"]["SN_AVG"], 0)],
                "SN_AVG_CH": [round(self.SN_DICT["CH"]["SN_AVG"], 0)],
            }
        )
        output = pd.concat([output, sn_output], axis=1)

        return output

    ###### PRINT METHODS
    def print_KP_bounds(self):
        return str(self.KP_bounds[0]) + " - " + str(self.KP_bounds[1])
