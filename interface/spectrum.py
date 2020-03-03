#Author: Devin Whitten
#Date: Nov 12, 2016
# This is will serve as the interface for the normalization function.
# So just defining some functions in here.

## Modifying to operate on synthetic spectra


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import MAD
import norm_functions
import synthetic_functions
import pandas as pd
import scipy.interpolate as interp
from astropy.table import Table
from scipy.ndimage.filters import gaussian_filter

print('spectrum loaded')
################################
#Spectrum Class Definition
################################
def obtain_flux(data):
    ### This is a catch all function to hopefully properly address fits data format variety
    shape = data.shape

    ##### Case 1
    if len(shape)== 1:
        ### simplest case, just grab array
        return data.flatten()

    else:
        ## grab first row and hope for the best
        return data[0].flatten()

class Spectrum():
    def __init__(self, spec, name, wl_range=[3800, 6200], fits=False):

        self.name = name
        print("... initializing:  ", name)
        #### Use the fits file to generate all necessary info for the spectrum
        #self.spec = spec[spec[0].between(wl_range[0], wl_range[-1], inclusive=True)]
        #self.wavelength = np.power(10. , (self.fits[0].header['CRVAL1'] + np.arange(0, self.fits[0].header['NAXIS1'])*self.fits[0].header['CD1_1']))
################################################################################
        if fits:
            ## This is a cumbersome attempt to accomadate multiple fits data formats..
            print("fits file")
            self.fits = spec

            if 'CD1_1' in spec[0].header:
                DELTA = "CD1_1"

            elif 'CDELT1' in spec[0].header:
                DELTA = 'CDELT1'

            else:
                print("I don't know which increment to use!")

            if self.fits[0].header['CRVAL1'] > 10.:
                print("linear wavelength")
                ###testing purposes, probably this is the way to go
                self.wavelength = (np.arange(0, spec[0].header['NAXIS1'], 1) * spec[0].header[DELTA]) + spec[0].header['CRVAL1']


            else:
                self.wavelength = np.power(10. , (self.fits[0].header['CRVAL1'] + np.arange(0, self.fits[0].header['NAXIS1'])*self.fits[0].header[DELTA]))

            self.original_wavelength = self.wavelength

            self.flux = obtain_flux(self.fits[0].data)
            self.wavelength = np.array(self.wavelength)


            ### check endian match
            #self.endian_match = (self.flux.dtype.byteorder == self.wavelength.dtype.byteorder)

            if self.flux.dtype.byteorder == ">":
                print('... correcting endian mismatch')
                self.flux = self.flux.byteswap().newbyteorder()

            #print(self.flux.dtype.byteorder == self.wavelength.dtype.byteorder)


################################################################################

        else:
            print("\t csv file")
            self.spec = spec
            self.flux = self.spec['flux']
            self.wavelength = np.array(self.spec['wave'], dtype=np.float)
            self.original_wavelength = self.wavelength
        
        #### Defined in generate_segments
        self.segments = None
        ####
        self.mad_global = None

        return



    def radial_correction(self, velocity=0):
        ### corrects the wavelength shift for given radial velocity

        self.wavelength = self.original_wavelength / ((velocity/2.99792e5) + 1)

        return


    ############################################################
    def trim_frame(self, bounds= [3000, 5000]):

        self.frame = self.frame[self.frame['wave'].between(bounds[0], bounds[1], inclusive=True)]
        return



    def estimate_sn(self):

        #### determines the first guess SN estimates for each region of interest
        #### defines SN_DICT member variable

        ### might define SIDEBANDS in another file at some point
        SIDEBANDS = {'CA' : [[3884, 3923], [3995, 4045]],
                     'CH' : [[4000, 4080], [4440, 4500]],
                     'C2' : [[4500, 4600], [4760, 4820]]}

        self.SN_DICT = {key : [] for key in SIDEBANDS.keys()}
        for key in SIDEBANDS.keys():

            if (SIDEBANDS[key][0][0] > min(self.frame['wave'])) and (SIDEBANDS[key][1][1] < max(self.frame['wave'])):

                SN_LEFT  = np.sqrt(self.frame['flux'][self.frame['wave'].between(*SIDEBANDS[key][0], inclusive=True)])
                SN_RIGHT = np.sqrt(self.frame['flux'][self.frame['wave'].between(*SIDEBANDS[key][1], inclusive=True)])

                ### Average the left and right sidebands

                self.SN_DICT[key] = {'SN_AVG' : np.mean([np.median(SN_LEFT), np.median(SN_RIGHT)]),
                                     'SN_STD' : max([MAD.S_MAD(SN_LEFT), MAD.S_MAD(SN_RIGHT)]),
                                     'XI_AVG' : np.mean([np.median(np.divide(1., SN_LEFT)), np.median(np.divide(1. , SN_RIGHT))]),
                                     'XI_STD' : max([MAD.S_MAD(np.divide(1., SN_LEFT)) , MAD.S_MAD(np.divide(1., SN_RIGHT))])}

                #### parameters for the beta distribution prior
                self.SN_DICT[key]['alpha'] = ((self.SN_DICT[key]['XI_AVG']**2)/np.square(self.SN_DICT[key]['XI_STD']))*(1 - self.SN_DICT[key]['XI_AVG']) - self.SN_DICT[key]['XI_AVG']
                self.SN_DICT[key]['beta']  = (1/self.SN_DICT[key]['XI_AVG'] - 1) * self.SN_DICT[key]['alpha']

            else:
                print("band not in wavelength coverage")

                self.SN_DICT[key] = {'SN_AVG' : np.nan,
                                     'SN_STD' : np.nan,
                                     'XI_AVG' : np.nan,
                                     'XI_STD' : np.nan,
                                     'alpha'  : np.nan,
                                     'beta'   : np.nan}


        return




    #################################################
    ### Total mutators
    def set_params(self,CLASS, JK, MODE, iter, T_SIGMA, HARD_TEFF):
        self.gravity_class = str(CLASS)
        self.JK = JK
        self.MODE = str(MODE)
        self.MCMC_iterations = iter
        self.T_SIGMA = float(T_SIGMA)
        self.HARD_TEFF = float(HARD_TEFF)

        assert (self.gravity_class == 'GIANT') or (self.gravity_class == 'DWARF'), "Invalid gravity class: {}".format(self.gravity_class)
        assert (self.MODE == 'UFD') or (self.MODE == 'HALO'), "Invalid Environment"


        return


    def set_KP_bounds(self, input_bounds):
        ## should be a list
        self.KP_bounds = input_bounds
        return

    def set_carbon_mode(self, carbon_mode):
        self.carbon_mode = carbon_mode
        return

    def set_group_ll(self, input_dict):

        self.LL_DICT = input_dict

        LLs = [self.LL_DICT[key][0] for key in self.LL_DICT.keys()]

        GROUP = ['GI', 'GII', 'GIII'][LLs.index(max(LLs))]

        print('\t ' + self.get_name().ljust(20) + ": ", GROUP, ["%.2F" % val for val in LLs])

        self.ARCH_GROUP = GROUP


        return



    def set_temperature(self, input_dict, sigma, hard=False):
        ### for use with the calibrate_temperatures function
        ### input_dict:  {"Casagrande":, "Hernandez":, "Bergeat": }

        if hard == True:
            self.teff_irfm = input_dict
            self.teff_irfm_unc = sigma
            return

        else:
            self.temp_dict = input_dict
            self.teff_irfm = np.mean([self.temp_dict[key] for key in self.temp_dict])
            self.teff_irfm_unc = sigma
        return

    def prepare_regions(self):
        ### prepares the CaII, CH, and C2 regions according to KP_bounds and carbon_mode

        self.regions = {"CA" : self.frame[self.frame['wave'].between(*self.KP_bounds, inclusive=True)].copy(),
                        "CH" : self.frame[self.frame['wave'].between(4222, 4322,      inclusive=True)].copy()}

        if self.carbon_mode == "CH+C2":
            ### then add the C2 cut
            self.regions['C2'] =  self.frame[self.frame['wave'].between(4710, 4750, inclusive=True).copy()]

        return



    def set_mcmc_args(self, input_dict = None):
        ## I'll finish if necessary
        if input_dict != None:
            self.mcmc_args = input
        else:
            self.mcmc_args = {}

        return

    def set_mcmc_results(self, input_dict, mode):
        ## I want to anticipate the refined and coarse outputs

        if mode == 'COARSE':
            self.MCMC_COARSE = input_dict

        elif mode == 'REFINE':
            self.MCMC_REFINE = input_dict

        else:
            print('Invalid mode in set_mcmc_results()')

        return

    def set_sampler(self, input_sampler, mode='COARSE'):
        if mode == 'COARSE':
            self.MCMC_COARSE_sampler = input_sampler
        elif mode == 'REFINE':
            self.MCMC_REFINE_sampler = input_sampler
        return

    def set_kde_functions(self, input_dict, mode):
        ## I want to anticipate the refined and coarse outputs

        if mode == 'COARSE':
            self.KDE_COARSE = input_dict

        elif mode == 'REFINE':
            self.KDE_REFINE = input_dict

        else:
            print('Invalid mode in set_mcmc_results()')

        return

    def set_flux(self, input_flux):
        ## Just a hard set function in case of format problems with the fits data section
        self.flux = input_flux
        return

    def set_norm(self, input_flux):
        ## intended for the external batch normalization
        self.norm = input_flux

        return
    def set_GBAND(self, input):
        ## might be interesting someday..
        self.GBAND_EW = input
        return

    def set_frame(self, wave, flux):
        self.frame = pd.DataFrame({'wave': wave, 'flux': flux})
        return

    def set_frame_wave(self, input_wave):
        self.frame.loc[:, 'wave'] = input_wave
        return

    def set_frame_flux(self, input_flux):
        self.frame.loc[:, 'flux'] = input_flux
        return

    def set_frame_norm(self, input_norm):
        self.frame.loc[:, 'norm'] = input_norm
        return

    def set_frame_cont(self, input_cont):
        self.frame.loc[:, 'cont'] = input_cont
        return



    def set_synth_spectrum(self, synth):
        self.synth_spectrum = synth
        return

    ################################################################
    def get_name(self):
        return "{:<20}".format(self.name)
        #return self.name.ljust(20)

    def get_wave(self):
        return self.wavelength

    def get_norm(self):
        return self.norm

    def get_flux(self):
        return self.flux

    def get_frame(self):
        return self.frame

    def get_frame_wave(self):
        return self.frame['wave']

    def get_frame_flux(self):
        return self.frame['flux']

    def get_frame_norm(self):
        return self.norm['norm']

    def get_gravity_class(self):
        return self.gravity_class

    def get_carbon_mode(self):
        return self.carbon_mode

    def get_KP_bounds(self):
        return self.KP_bounds

    def get_environ_mode(self):
        return self.MODE

    def get_arch_group(self):
        return self.ARCH_GROUP

    def get_photo_temp(self):
        return self.teff_irfm, self.teff_irfm_unc

    def get_SN_dict(self):
        return self.SN_DICT

    def get_kde_dict(self):
        if hasattr(self, 'KDE_REFINE'):
            return self.KDE_COARSE, self.KDE_REFINE

        return self.KDE_COARSE


    def get_mcmc_dict(self, mode='COARSE'):
        if mode == 'COARSE':
            return self.MCMC_COARSE

        elif mode == 'REFINE':
            return self.MCMC_REFINE

        elif mode == 'BOTH':
            return self.MCMC_COARSE, self.MCMC_REFINE

        else:
            print("Bad mode:  ", mode)
            return np.nan

    def get_MCMC_iterations(self):
        return self.MCMC_iterations


    def get_output_row(self):
        ## simply produces a dataframe row with the desired outputs
        return pd.DataFrame({
                        "NAME"     : [self.get_name()],
                        "MODE"     : [self.get_carbon_mode()],
                        'GROUP'    : [self.get_arch_group()],
                        'TEFF'     : [round(self.MCMC_COARSE['TEFF'][0], 0)],
                        'TEFF_ERR' : [round(self.MCMC_COARSE['TEFF'][1], 2)],
                        'TEFF_IRFM': [self.teff_irfm],
                        'TEFF_IRFM': [self.teff_irfm_unc],
                        'FEH'      : [round(self.MCMC_REFINE['FEH'][0], 2)],
                        'FEH_ERR'  : [round(max([self.MCMC_REFINE['FEH'][1], self.MCMC_COARSE['FEH'][1]]), 4)],
                        'CFE'      : [round(self.MCMC_REFINE['CFE'][0], 2)],
                        'CFE_ERR'  : [round(max([self.MCMC_REFINE['CFE'][1], self.MCMC_COARSE['CFE'][1]]), 4)],
                        'AC'       : [round(self.MCMC_REFINE['AC'][0], 2)],
                        'AC_ERR'   : [round(max([self.MCMC_REFINE['AC'][1], self.MCMC_COARSE['AC'][1]]), 4)]
                    })

    ###### PRINT METHODS
    def print_KP_bounds(self):
        return str(self.KP_bounds[0]) + " - " + str(self.KP_bounds[1])
