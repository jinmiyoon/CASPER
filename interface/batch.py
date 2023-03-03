################################################################################
### Author: Devin Whitten,  Jinmi Yoon
### Email: devin.d.whitten@gmail.com, jinmi.yoon@gmail.com
################################################################################
#### this is the class definition for the Batch class.
#### just bundling the Spectrum objects and normalization/analysis routines

import os
import interface_main, spectrum
from astropy.io import fits
from collections import namedtuple
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from texttable import Texttable
import temp_calibrations as TC
import plot_functions
import io_functions
import pandas as pd
import GISIC_C as GISIC
import EW




class Batch():
    #### Main Batch Class
    #def __init__(self, spectra_path, param_path, io_path):
    def __init__(self, io_paths):
        ## load io_path which set inputs and outputs directories and
        ## the file names along with the location where spectra live
        self.io_paths      = io_paths

        return

    def set_io_paths(self):
        print("\nloading io_paths:  ", self.io_paths)
        self.io_params   = eval(open(self.io_paths, 'r').read())

        print("\nsetting io_paths:  ")
        self.param_path  = self.io_params['param_path']
        self.spectra_path  = self.io_params['spectra_dir_path']
        self.output_name  = self.io_params['output_dir_path'] + self.io_params['output_file_name']
        print(" \t\t\t > input setting parameter    :  ", self.param_path)
        print(" \t\t\t > input spectra directory    :  ", self.spectra_path)
        print(" \t\t\t > output directory + filename:  ", self.output_name)

        return

    def load_params(self):
        print("\nloading input params:  ", self.param_path)
        self.param_file  = pd.read_csv(self.param_path)
        print(list(self.param_file.columns))
        # 09-08-2020 J. Yoon
        #'mode' indicate galactic environment, 'HALO' or 'UFD' # I need to change "UFD" to "dSph"
        self.param_file['sequence'] = self.param_file['sequence'].astype(str)
        self.sequence = self.param_file['sequence'].tolist()
        self.param_file['mode'] = self.param_file['mode'].astype(str)
        # 09-08-2020 J. Yoon
        #'class' indicates luminosity (gravity) clas, 'GIANT' or 'DWARF'
        self.param_file['class'] = self.param_file['class'].astype(str)
        # 09-09-2020 J. Yoon
        # I add one more parameter called 'carbon_mode' to freely change its mode for validation.
        self.param_file['carbon_mode'] = self.param_file['carbon_mode'].astype(str) # uncomment when this change needed.
        # 06-11-2021 J. yoon


        return


    def load_spectra(self):
        print("\n ... loading spectra:  ", self.spectra_path)


        ### I only want the spectra in the param file
        self.spectra_names =  self.param_file['name'].tolist()

        
        def spectra_input(pathname, current):
            #print("filepath= ", pathname, current)
            file_ext = current.split('.')[1] 
            if file_ext == 'fits': 
                with fits.open(pathname) as hdu:
                    return spectrum.Spectrum(hdu, name=current, is_fits=True)
            elif file_ext =='csv': 
                return spectrum.Spectrum(pd.read_csv(pathname), name=current, is_fits=False)
            else: 
                raise Exception("Invalid file format extension. Currently only .fits and .csv files are supported")

        self.spectra_array = [spectra_input(self.spectra_path + current, current) for current in self.spectra_names] 

        print("\t\t batch: what is spectra_arry - ", self.spectra_array)
        '''
        else:
            #print("input spectra files are of csv format") 
            self.spectra_array = [spectrum.Spectrum(pd.read_csv(self.spectra_path + current),
                name=current, is_fits=False) for current in self.spectra_names]
        '''
        self.length = len(self.spectra_array)

        return



    def set_params(self):
        ### Need to distribute the parameters across the Spectrum objects
        ### I'll update this as needed
        print("\n... setting spectra parameters")
        for i, row in self.param_file.iterrows():
            spec = self.spectra_array[i]

            assert spec.name == row['name'].strip(), 'Parameter error in calibrate_temperatures()'
            SEQUENCE = row['sequence']
            JK = row['J-K']
            CLASS = row['class'].strip()
            MODE  = row['mode'].strip()
            INPUT_CARBON_MODE = row['carbon_mode'].strip()  # 09-09-2020 J. Yoon
            ITER  = row['MCMC_iter']
            T_SIGMA = row['T_SIGMA']
            HARD_TEFF = row['TEFF_SET']

            #spec.set_params(CLASS = CLASS, JK = JK, MODE=MODE, iter=ITER, T_SIGMA=T_SIGMA, HARD_TEFF=HARD_TEFF)
            # 09-09-2020, 11-13-2021 J. yoon
            spec.set_params(SEQUENCE = SEQUENCE, CLASS = CLASS, JK = JK, MODE=MODE,
                INPUT_CARBON_MODE=INPUT_CARBON_MODE, iter=ITER, T_SIGMA=T_SIGMA, HARD_TEFF=HARD_TEFF)



        return


    def radial_correct(self):
        print("\n... correcting radial velocities")
        #self.sequence = self.param_file['sequence'].tolist()
        for sequence, spec in zip(self.sequence, self.spectra_array):

            radial_velocity = float(self.param_file[self.param_file['sequence'] == sequence]['RV'])
            spec.radial_correction(radial_velocity)
            print('\t For {:s},  RV = {:7.2f} km/s'.format(sequence+': '+spec.name, radial_velocity))
            #print('    correcting RV= %7.3f km/s:  done' %float(self.param_file[self.param_file['name'] == name]['RV']))


    def build_frames(self, bounds = [3880, 4830]):
        ### I'd rather not modify the original wavelength and flux arrays
        ### plus it's nice to work with dataframes, so I'm just gonna dump arrays to member frames
        ### might as well trim the wavelength coverage here to match the synthetic spectra

        print("\n ... build dataframes")
        [spec.set_frame(wave=spec.get_wave(), flux=spec.get_flux()) for spec in self.spectra_array]
        [spec.trim_frame(bounds) for spec in self.spectra_array]

        return


    def normalize(self, default=True):
        print("\n... normalizing spectra batch")
        print("... iterating convolution sigma")
        ### Default specfies whether any GISIC values should be taken from the param_file
        ## for now I'm just going to write the default case
        #for spectrum in self.spectrum:

        if default:
            for spec in self.spectra_array:
                cont_array = []
                ###     July 15 2020 J. Yoon      ###
                #for SIGMA in np.linspace(15, 30, 10): # Devin's original set up

                # Currently best choice with flux_min=80 (GISIC_S.normalize()) I think.
                # but need to be further tested, 09/09/2020 J. Yoon
                # The current setting is flux_min=70 as Devin's original setup.
                # for SIGMA in np.linspace(15, 25, 10): penaltimate best choice

                for SIGMA in np.linspace(25, 35, 10): #
                #for SIGMA in np.linspace(10, 20, 10):
                # this choice is not recommended because it does not capture continuum points well.
                #It even makes C2 band continuum.
                # J.Yoon 10/06/22 update: 
                #best norm param fits (sigma=30, k=1, s=12, cahk=True, band_check=False, flux_min=60, boost=True)
                # GISIC.normalize() defaults kwargs are now set that way execpt, k, so I set it to k=1 here.
                    wave, norm, cont = GISIC.normalize(spec.get_frame_wave(), spec.get_frame_flux(), 
                        sigma = SIGMA, k=1,cahk=True, band_check=False, flux_min=70, boost=True)

                    cont_array.append(cont)

                ########
                ### average the sigma runs together
                cont = np.median(np.array(cont_array), axis=0)

                norm = np.divide(spec.get_frame_flux(), cont)

                if len(norm[norm < 0.0])>1:

                    norm[norm < 0.0] = 1.

                if len(norm[norm >2.0]) >1:

                    norm[norm >2.0] = 1.

                spec.set_frame_norm(norm)
                spec.set_frame_cont(cont)
                print('\t {:20s}'.format(spec.name), ":  okay")

        else:
            print("\t Sorry - can't customize GISIC normalization yet...")


        return

    def ebv_correction(self):
        print("\n... correcting photometry")
        for i, row in self.param_file.iterrows():
            spec = self.spectra_array[i]
            spec.ebv_correct(row)




    def calibrate_temperatures(self, default=True, teff_sigma=250):
        ## Here is where we will use the (J-K)0 values from the param_file
        ## along with the surface gravity class, if known
        ## We'll eventually want to update to override sigma, I'l come back to that

        #print("\n... determining photometric temperature")
        print("\n... determining temperature for archetype classification")

        ### I don't think the spectra_array and the param_file are sorted the same
        ### so I need to be careful

        for i, row in self.param_file.iterrows():

            #io_functions.span_window()

            spec = self.spectra_array[i]

            assert spec.name == row['name'], 'Parameter error in calibrate_temperatures()'
            #print("\t setting photometric temperature sigma: ", spec.T_SIGMA)
            print("\t setting input temperature sigma: ", spec.T_SIGMA) #J.Yoon 02/11/2022

            CLASS = row['class'].strip()

            #### remember that there is a class definition here too

            # Here sets temp with HARD_TEFF if the value exists.
            if np.isfinite(spec.HARD_TEFF):
                spec.set_temp_frame(TC.calibrate_temp_frame(float(spec.PHOTO_0['J-K']),
                                          float(spec.PHOTO_0['g-r']),
                                          CLASS = CLASS))

                spec.TEMP_FRAME.loc['HARD_TEFF', 'VALUE'] = spec.HARD_TEFF # added for extra table column J.Yoon 02/11/2022
                spec.TEMP_FRAME.loc['ADOPTED', 'VALUE'] = spec.HARD_TEFF

                spec.set_temperature(spec.HARD_TEFF, spec.T_SIGMA, hard=True)
                print("\t setting and adopting hard teff:   ", spec.HARD_TEFF)

            #Here sets temp with one of the photometric temps.
            else:
                spec.set_temp_frame(TC.calibrate_temp_frame(float(spec.PHOTO_0['J-K']),
                                          float(spec.PHOTO_0['g-r']),
                                          CLASS = CLASS))
                spec.TEMP_FRAME.loc['HARD_TEFF', 'VALUE'] = np.nan  # added for extra table column J.Yoon 02/11/2022
                spec.set_temperature(spec.TEMP_FRAME.loc['ADOPTED', 'VALUE'], sigma=spec.T_SIGMA)
                print("\t setting and adopting photo teff:   ", spec.TEMP_FRAME.loc['ADOPTED', 'VALUE'] )

        #### NOW ASSEMBLE THE OUTPUT TABLE

        HEADER = ['NAME', 'Bergeat', 'Hernandez', 'Casagrande', 'Fukugita', 'HARD_TEFF', 'ADOPTED']


        output_table = HEADER

        for spec in self.spectra_array:
            row = np.concatenate([[spec.get_name().split(".fits")[0]],[spec.TEMP_FRAME.loc[CURRENT].values[0] for CURRENT in HEADER[1:]]])
            output_table = np.vstack([output_table, row])

        table = Texttable()

        table.add_rows(output_table)

        print(" ------ TEMPERATURES (PHOTOMETRIC + HARD + ADOPTED)-------")
        if len(self.spectra_array) < 30: print(table.draw())
        else : print("Too long table to print; only save the temp table in a file in "+self.io_params['output_dir_path'])
        # save into a file
        print(table.draw(), file=open(self.output_name + "_temp_cal_table.txt", "a"))
        return

    def set_KP_bounds(self):
        ## set the appropriate bandwidth on the CaII index
        print("\n... setting KP bandwidth")
        [spec.set_KP_bounds(EW.get_KP_band(spec)) for spec in self.spectra_array]

        return

    def set_carbon_mode(self):
        ## At some point we'll want to override this using the param file, when desired
        print("\n... setting carbon mode")
        [EW.set_CH_procedure(spec) for spec in self.spectra_array]
        return

    def estimate_sn(self):
        print("\n... estimating S/N")
        [spec.estimate_sn() for spec in self.spectra_array]
        return

    def set_mcmc_args(self):
        print('\n... bulding mcmc_args dict')
        [spec.set_mcmc_args() for spec in self.spectra_array]
        return


    ##### the big ones
    def archetype_classification(self):
        #io_functions.span_window()
        print('\n... determining archetype classification')

        [interface_main.archetype_classify_MC(spec) for spec in self.spectra_array]

        ### prepare output table if it's reasonable
        #if len(self.spectra_array) < 30:

        output_table = ['NAME', "GI", "GII", "GIII"]

        for spec in self.spectra_array:
            row = np.concatenate([[spec.get_name()], [spec.LL_DICT[key][0].round(0) for key in ["GI", "GII", "GIII"]]])
            output_table = np.vstack([output_table, row])

        table = Texttable()
        table.add_rows(output_table)
        print(" ------  ARCHETYPE LIKELIHOODS -------")
        if len(self.spectra_array) < 30: print(table.draw())
        # save into a file
        print(table.draw(), file=open(self.output_name + "_archetype_likelihood_table.txt", "a"))

        return


    # def mcmc_determination(self, pool=20): # I don't see why pool variable is needed here. it was not even used.
    # So I deleted pool variable from mcmc_determination() and interface_main.mcmc_determination()
    #   Also, deleted pool=20 in main.py
    def mcmc_determination(self):
        ### Main iterative method for the mcmc_determination
        #io_functions.span_window()
        print('\n... performing MCMC determinations')

        [spec.prepare_regions() for spec in self.spectra_array]

        #[interface_main.mcmc_determination(spec, mode='COARSE', pool=pool)  for spec in self.spectra_array]
        [interface_main.mcmc_determination(spec, mode='COARSE')  for spec in self.spectra_array]

        print("... performing kde determinations")
        [interface_main.generate_kde_params(spec, mode="COARSE") for spec in self.spectra_array]

        #io_functions.span_window()

        print("... running refined mcmc")
        #[interface_main.mcmc_determination(spec, mode='REFINE', pool=pool)  for spec in self.spectra_array]
        [interface_main.mcmc_determination(spec, mode='REFINE')  for spec in self.spectra_array]

        print("... finalizing kde determinations")
        [interface_main.generate_kde_params(spec, mode='REFINE') for spec in self.spectra_array]

        #io_functions.span_window()
        print("... complete")
        #io_functions.span_window()
        return


    def generate_synthetic(self):

        print("\n... generating synthetic spectra")

        [interface_main.generate_synthetic(spec) for spec in self.spectra_array]

        return

    def generate_plots(self):
        #io_functions.span_window()
        print("\n... generating plots")

        plot_functions.plot_spectra(self)

        print("... generating mcmc trace plots")
        plot_functions.plot_mcmc_trace_array(self)

        print("... generating corner plots")
        plot_functions.plot_corner_array(self)



        return

    def generate_output_files(self):
        #io_functions.span_window()
        print("\n... generating outputs")

        final = pd.concat([spec.get_output_row() for spec in self.spectra_array])
        try:
            final.to_csv( self.output_name + "_out.csv", index=False)

        except:
            final.to_csv(self.output_name + "1_out.csv", index=False)
