#### this is the class definition for the Batch class.
#### just bundling the Spectrum objects and normalization/analysis routines
import os
import interface_main, spectrum
from astropy.io import fits
from collections import namedtuple

from matplotlib.backends.backend_pdf import PdfPages
import temp_calibrations
import plot_functions
import pandas as pd
import GISIC
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

    def load_spectra(self):
        print("... loading spectra:  ", self.spectra_path)


        ### I only want the spectra in the param file
        self.spectra_names = self.param_file['name'].tolist()
        self.spectra_array = [spectrum.Spectrum(fits.open(self.spectra_path + current),name=current, fits=True) for current in self.spectra_names]

        self.length = len(self.spectra_array)

        return



    def set_params(self):
        ### Need to distribute the parameters across the Spectrum objects
        ### I'll update this as needed
        print("... setting spectra parameters")
        for i, row in self.param_file.iterrows():

            spec = self.spectra_array[i]

            assert spec.name == row['name'], 'Parameter error in calibrate_temperatures()'

            JK = row['(J-K)0']
            CLASS = row['class'].strip()
            MODE  = row['mode'].strip()
            ITER  = row['MCMC_iter']

            spec.set_params(CLASS = CLASS, JK = JK, MODE=MODE, iter=ITER)


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
        ### Default specfies whether any GISIC values should be taken from the param_file
        ## for now I'm just going to write the default case
        #for spectrum in self.spectrum:

        if default:
            for spec in self.spectra_array:

                wave, norm, cont = GISIC.normalize(spec.get_frame_wave(), spec.get_frame_flux())
                spec.set_frame_norm(norm)
                spec.set_frame_cont(cont)
                print('{:20s}'.format(spec.name), ":  okay")

        else:
            print("\t Sorry - can't customize GISIC normalization yet...")


        return

    def calibrate_temperatures(self, default=True):
        ## Here is where we will use the (J-K)0 values from the param_file
        ## along with the surface gravity class, if known
        ## We'll eventually want to update to override sigma, I'l come back to that

        print("... determining photometric temperature")
        print("\t setting photometric temperature sigma: 250")
        ### I don't think the spectra_array and the param_file are sorted the same
        ### so I need to be careful

        for i, row in self.param_file.iterrows():

            spec = self.spectra_array[i]

            assert spec.name == row['name'], 'Parameter error in calibrate_temperatures()'

            JK = row['(J-K)0']
            CLASS = row['class'].strip()

            #### remember that there is a class definition here too
            spec.set_temperature(temp_calibrations.calibrate_temperatures(float(JK), CLASS = CLASS), sigma=250)

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
        interface_main.span_window()
        print('... determining archetype classification')

        [interface_main.archetype_classify_MC(spec) for spec in self.spectra_array]

        return


    def mcmc_determination(self):
        ### Main iterative method for the mcmc_determination
        interface_main.span_window()
        print('... performing MCMC determinations')

        [spec.prepare_regions() for spec in self.spectra_array]

        [interface_main.mcmc_determination(spec, mode='COARSE')  for spec in self.spectra_array]

        print("... performing kde determinations")
        [interface_main.generate_kde_params(spec, mode="COARSE") for spec in self.spectra_array]

        interface_main.span_window()

        print("... running refined mcmc")
        [interface_main.mcmc_determination(spec, mode='REFINE')  for spec in self.spectra_array]

        print("... finalizing kde determinations")
        [interface_main.generate_kde_params(spec, mode='REFINE') for spec in self.spectra_array]

        interface_main.span_window()
        print("... complete")
        interface_main.span_window()
        return


    def generate_synthetic(self):

        print("... generating synthetic spectra")

        [interface_main.generate_synthetic(spec) for spec in self.spectra_array]

        return

    def generate_plots(self):
        interface_main.span_window()
        print("... generating plots")

        plot_functions.plot_spectra(self)

        return

    def generate_output_files(self):
        interface_main.span_window()
        print("... generating outputs")

        final = pd.concat([spec.get_output_row() for spec in self.spectra_array])

        final.to_csv("output/" + self.io_params['output_name'] + "_out.csv", index=False)
