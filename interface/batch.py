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
    def __init__(self, spectra_path, param_path, io_path):
        ## pretty much just load the param_file
        ## set the spectra_path

        self.spectra_path = spectra_path
        self.param_path   = param_path
        self.io_path      = io_path


        return

    def load_params(self):
        print("loading params:  ", self.param_path)
        self.param_file  = pd.read_csv(self.param_path)
        self.param_file['mode'] = self.param_file['mode'].astype(str)
        self.param_file['class'] = self.param_file['class'].astype(str)
        print("loading io_params:  ", self.io_path)
        self.io_params   = eval(open(self.io_path, 'r').read())

        return

    def load_spectra(self, is_fits=True):
        print("... loading spectra:  ", self.spectra_path)


        ### I only want the spectra in the param file
        self.spectra_names = self.param_file['name'].tolist()
        if is_fits == True:
            self.spectra_array = [spectrum.Spectrum(fits.open(self.spectra_path + current),name=current, is_fits=fits) for current in self.spectra_names]

        else:
            self.spectra_array = [spectrum.Spectrum(pd.read_csv(self.spectra_path + current),name=current, fits=False) for current in self.spectra_names]
        self.length = len(self.spectra_array)

        return



    def set_params(self):
        ### Need to distribute the parameters across the Spectrum objects
        ### I'll update this as needed
        print("... setting spectra parameters")
        for i, row in self.param_file.iterrows():

            spec = self.spectra_array[i]

            assert spec.name == row['name'], 'Parameter error in calibrate_temperatures()'

            JK = row['J-K']
            CLASS = row['class'].strip()
            MODE  = row['mode'].strip()
            ITER  = row['MCMC_iter']
            T_SIGMA = row['T_SIGMA']
            HARD_TEFF = row['TEFF_SET']
            spec.set_params(CLASS = CLASS, JK = JK, MODE=MODE, iter=ITER, T_SIGMA=T_SIGMA, HARD_TEFF=HARD_TEFF)


        return




    def radial_correct(self):
        print("... correcting radial velocities")
        for name, spec in zip(self.spectra_names, self.spectra_array):


            spec.radial_correction(float(self.param_file[self.param_file['name'] == name]['RV']))
            #print('{:20}'.format(spec.name), ":  okay")


    def build_frames(self, bounds = [3000, 5000]):
        ### I'd rather not modify the original wavelength and flux arrays
        ### plus it's nice to work with dataframes, so I'm just gonna dump arrays to member frames
        ### might as well trim the wavelength coverage here to match the synthetic spectra

        print("... build dataframes")
        [spec.set_frame(wave=spec.get_wave(), flux=spec.get_flux()) for spec in self.spectra_array]
        [spec.trim_frame(bounds) for spec in self.spectra_array]

        return


    def normalize(self, default=True):
        print("... normalizing spectra batch")
        print("... iterating convolution sigma")
        ### Default specfies whether any GISIC values should be taken from the param_file
        ## for now I'm just going to write the default case
        #for spectrum in self.spectrum:

        if default:
            for spec in self.spectra_array:
                cont_array = []
                for SIGMA in np.linspace(15, 30, 10):

                    wave, norm, cont = GISIC.normalize(spec.get_frame_wave(), spec.get_frame_flux(), sigma = SIGMA, k=1)

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
        print("... correcting photometry")
        for i, row in self.param_file.iterrows():
            spec = self.spectra_array[i]
            spec.ebv_correct(row)




    def calibrate_temperatures(self, default=True, teff_sigma=250):
        ## Here is where we will use the (J-K)0 values from the param_file
        ## along with the surface gravity class, if known
        ## We'll eventually want to update to override sigma, I'l come back to that

        print("... determining photometric temperature")

        ### I don't think the spectra_array and the param_file are sorted the same
        ### so I need to be careful

        for i, row in self.param_file.iterrows():

            io_functions.span_window()



            spec = self.spectra_array[i]

            assert spec.name == row['name'], 'Parameter error in calibrate_temperatures()'
            print("\t setting photometric temperature sigma: ", spec.T_SIGMA)

            CLASS = row['class'].strip()

            #### remember that there is a class definition here too
            if np.isfinite(spec.HARD_TEFF):
                print("\t setting hard teff:   ", spec.HARD_TEFF)
                spec.set_temp_frame(TC.calibrate_temp_frame(float(spec.PHOTO_0['J-K']),
                                          float(spec.PHOTO_0['g-r']),
                                          CLASS = CLASS))

                spec.TEMP_FRAME.loc['ADOPTED', 'VALUE'] = spec.HARD_TEFF

                spec.set_temperature(spec.HARD_TEFF, spec.T_SIGMA, hard=True)

            else:
                spec.set_temp_frame(TC.calibrate_temp_frame(float(spec.PHOTO_0['J-K']),
                                          float(spec.PHOTO_0['g-r']),
                                          CLASS = CLASS))

                spec.set_temperature(spec.TEMP_FRAME.loc['ADOPTED', 'VALUE'], sigma=spec.T_SIGMA)


        #### NOW ASSEMBLE THE OUTPUT TABLE

        HEADER = ['NAME', 'Bergeat', 'Hernandez', 'Casagrande', 'Fukugita', 'ADOPTED']

        if len(self.spectra_array) < 30:

            output_table = HEADER

            for spec in self.spectra_array:
                row = np.concatenate([[spec.get_name().split(".fits")[0]],
                                        [spec.TEMP_FRAME.loc[CURRENT].values[0] for CURRENT in HEADER[1:]]
                                        ])

                output_table = np.vstack([output_table, row])

            table = Texttable()

            table.add_rows(output_table)
            print(" ------  PHOTOMETRIC TEMPERATURES -------")
            print(table.draw())

        return

    def set_KP_bounds(self):
        ## set the appropriate bandwidth on the CaII index
        print("... setting KP bandwidth")
        [spec.set_KP_bounds(EW.get_KP_band(spec)) for spec in self.spectra_array]

        return

    def set_carbon_mode(self):
        ## At some point we'll want to override this using the param file, when desired
        print("... setting carbon mode")
        [EW.set_CH_procedure(spec) for spec in self.spectra_array]
        return

    def estimate_sn(self):
        print("... estimating S/N")
        [spec.estimate_sn() for spec in self.spectra_array]
        return

    def set_mcmc_args(self):
        print('... bulding mcmc_args dict')
        [spec.set_mcmc_args() for spec in self.spectra_array]
        return


    ##### the big ones
    def archetype_classification(self):
        io_functions.span_window()
        print('... determining archetype classification')

        [interface_main.archetype_classify_MC(spec) for spec in self.spectra_array]

        ### prepare output table if it's reasonable
        if len(self.spectra_array) < 30:

            output_table = ['NAME', "GI", "GII", "GIII"]

            for spec in self.spectra_array:
                row = np.concatenate([[spec.get_name()], [spec.LL_DICT[key][0].round(0) for key in ["GI", "GII", "GIII"]]])

                output_table = np.vstack([output_table, row])

            table = Texttable()

            table.add_rows(output_table)
            print(" ------  ARCHETYPE LIKELIHOODS -------")
            print(table.draw())




        return


    def mcmc_determination(self, pool=20):
        ### Main iterative method for the mcmc_determination
        io_functions.span_window()
        print('... performing MCMC determinations')

        [spec.prepare_regions() for spec in self.spectra_array]

        [interface_main.mcmc_determination(spec, mode='COARSE', pool=pool)  for spec in self.spectra_array]

        print("... performing kde determinations")
        [interface_main.generate_kde_params(spec, mode="COARSE") for spec in self.spectra_array]

        io_functions.span_window()

        print("... running refined mcmc")
        [interface_main.mcmc_determination(spec, mode='REFINE', pool=pool)  for spec in self.spectra_array]

        print("... finalizing kde determinations")
        [interface_main.generate_kde_params(spec, mode='REFINE') for spec in self.spectra_array]

        io_functions.span_window()
        print("... complete")
        io_functions.span_window()
        return


    def generate_synthetic(self):

        print("... generating synthetic spectra")

        [interface_main.generate_synthetic(spec) for spec in self.spectra_array]

        return

    def generate_plots(self):
        io_functions.span_window()
        print("... generating plots")

        plot_functions.plot_spectra(self)

        print("... generating corner plots")
        plot_functions.plot_corner_array(self)



        return

    def generate_output_files(self):
        io_functions.span_window()
        print("... generating outputs")

        final = pd.concat([spec.get_output_row() for spec in self.spectra_array])
        try:
            final.to_csv("output/" + self.io_params['output_name'] + "_out.csv", index=False)

        except:
            final.to_csv("output/" + self.io_params['output_name'] + "1_out.csv", index=False)
