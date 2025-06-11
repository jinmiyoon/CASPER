################################################################################
### Author: Devin Whitten,  Jinmi Yoon
### Email: devin.d.whitten@gmail.com, jinmi.yoon@gmail.com
################################################################################
#### this is the class definition for the Batch class.
#### just bundling the Spectrum objects and normalization/analysis routines

import time
import os
import interface_main
from spectrum import Spectrum
from astropy.io import fits
import numpy as np
from texttable import Texttable
import temp_calibrations as TC
import plot_functions
import pandas as pd
import GISIC_C as GISIC
import EW
import config

# Add PROJECT_ROOT
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

# Add NPSAVE_DIR and OUTPUT_DIR
NPSAVE_DIR = os.path.join(PROJECT_ROOT, 'npsave')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'outputs')

# Ensure directories exist
os.makedirs(NPSAVE_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

class Batch():
    #### Main Batch Class
    def __init__(self, io_paths):
        self.io_paths = io_paths
        return

    def set_io_paths(self):
        print("\nloading io_paths:  ", self.io_paths)
        self.io_params = eval(open(self.io_paths, 'r').read())

        print("\nsetting io_paths:  ")
        self.param_path = os.path.abspath(os.path.join(PROJECT_ROOT, self.io_params['param_path']))
        self.spectra_path = os.path.abspath(os.path.join(PROJECT_ROOT, self.io_params['spectra_dir_path']))
        self.output_name = os.path.join(OUTPUT_DIR, self.io_params['output_file_name'])
        print(" \t\t\t > input setting parameter    :  ", self.param_path)
        print(" \t\t\t > input spectra directory    :  ", self.spectra_path)
        print(" \t\t\t > output directory + filename:  ", self.output_name)

        return

    def load_params(self):
        print("\nloading input params:  ", self.param_path)
        self.param_file = pd.read_csv(self.param_path)
        print(list(self.param_file.columns))
        self.param_file['sequence'] = self.param_file['sequence'].astype(str)
        self.sequence = self.param_file['sequence'].tolist()
        self.param_file['mode'] = self.param_file['mode'].astype(str)
        self.param_file['class'] = self.param_file['class'].astype(str)
        self.param_file['carbon_mode'] = self.param_file['carbon_mode'].astype(str)
        return

    def load_spectra(self):
        print("\n ... loading spectra:  ", self.spectra_path)
        self.spectra_names = self.param_file['filename'].tolist()

        def spectra_input(pathname, current):
            file_ext = current.split('.')[1]
            if file_ext == 'fits':
                with fits.open(pathname) as hdu:
                    return Spectrum(hdu, filename=current, is_fits=True)
            elif file_ext == 'csv':
                return Spectrum(pd.read_csv(pathname), filename=current, is_fits=False)
            else:
                raise Exception("Invalid file format extension. Currently only .fits and .csv files are supported")

        self.spectra_array = [spectra_input(os.path.join(self.spectra_path, current), current) for current in self.spectra_names]

        print("\t\t batch: what is spectra_array - ", self.spectra_array)
        self.length = len(self.spectra_array)

        return

    def set_params(self):
        print("\n... setting spectra parameters")
        for i, row in self.param_file.iterrows():
            spec = self.spectra_array[i]

            assert spec.filename == row['filename'].strip(), 'Name is not found!'
            SEQUENCE = row['sequence']
            STARNAME = row['starname']
            JK = row['J-K']
            CLASS = row['class'].strip()
            MODE = row['mode'].strip()
            INPUT_CARBON_MODE = row['carbon_mode'].strip()
            ITER = row['MCMC_iter']
            T_SIGMA = row['T_SIGMA']
            HARD_TEFF = row['TEFF_SET']

            spec.set_params(SEQUENCE=SEQUENCE, STARNAME=STARNAME, CLASS=CLASS, JK=JK, MODE=MODE,
                            INPUT_CARBON_MODE=INPUT_CARBON_MODE, iter=ITER, T_SIGMA=T_SIGMA, HARD_TEFF=HARD_TEFF)

        return

    def radial_correct(self):
        print("\n... correcting radial velocities")
        for sequence, spec in zip(self.sequence, self.spectra_array):
            radial_velocity = float(self.param_file[self.param_file['sequence'] == sequence]['RV'])
            spec.radial_correction(radial_velocity)
            print('\t For {:s},  RV = {:7.2f} km/s'.format(sequence + ': ' + spec.filename, radial_velocity))

    def build_frames(self, bounds=config.WAVE_BOUNDS):
        print("\n ... build dataframes")
        [spec.set_frame(wave=spec.get_wave(), flux=spec.get_flux()) for spec in self.spectra_array]
        [spec.trim_frame(bounds) for spec in self.spectra_array]
        return

    def normalize(self, default=True):
        print("\n... normalizing spectra batch")
        print("... iterating convolution sigma")
        start_time = time.time()
        if default:
            for spec in self.spectra_array:
                cont_array = []
                for SIGMA in config.SIGMA:
                    _, norm, cont = GISIC.normalize(spec.get_frame_wave(), spec.get_frame_flux(),
                                                    sigma=SIGMA, k=config.k, cahk=config.cahk,
                                                    band_check=config.cahk, flux_min=config.flux_min, boost=config.boost)

                    cont_array.append(cont)

                cont = np.median(np.array(cont_array), axis=0)
                norm = np.divide(spec.get_frame_flux(), cont)

                norm[norm < 0.0] = 1.
                norm[norm > 2.0] = 1.

                spec.set_frame_norm(norm)
                spec.set_frame_cont(cont)
                print('\t {:20s}'.format(spec.filename), ":  okay")

        else:
            print("\t Sorry - can't customize GISIC normalization yet...")

        print("\t\t batch: Time spent normalizing the observed spectra is {0:.1f}".format(time.time() - start_time))
        return

    def ebv_correction(self):
        print("\n... correcting photometry")
        for i, row in self.param_file.iterrows():
            spec = self.spectra_array[i]
            spec.ebv_correct(row)

    def calibrate_temperatures(self, default=True, teff_sigma=250):
        print("\n... determining temperature for archetype classification")
        for i, row in self.param_file.iterrows():
            spec = self.spectra_array[i]

            assert spec.filename == row['filename'], 'Parameter error in calibrate_temperatures()'
            print("\t setting input temperature sigma: ", spec.T_SIGMA)

            CLASS = row['class'].strip()

            spec.set_temp_frame(TC.calibrate_temp_frame(float(spec.PHOTO_0['J-K']),
                                                        float(spec.PHOTO_0['g-r']),
                                                        CLASS=CLASS))

            if np.isfinite(spec.HARD_TEFF):
                spec.TEMP_FRAME.loc['HARD_TEFF', 'VALUE'] = spec.HARD_TEFF
                spec.TEMP_FRAME.loc['ADOPTED', 'VALUE'] = spec.HARD_TEFF
                spec.set_temperature(spec.HARD_TEFF, spec.T_SIGMA)
                print("\t setting and adopting hard teff:   ", spec.HARD_TEFF)
            else:
                spec.TEMP_FRAME.loc['HARD_TEFF', 'VALUE'] = np.nan
                spec.set_temperature(spec.TEMP_FRAME.loc['ADOPTED', 'VALUE'], sigma=spec.T_SIGMA)
                print("\t setting and adopting photo teff:   ", spec.TEMP_FRAME.loc['ADOPTED', 'VALUE'])

        HEADER = ['NAME', 'Bergeat', 'Hernandez', 'Casagrande', 'Fukugita', 'HARD_TEFF', 'ADOPTED']
        output_table = HEADER

        for spec in self.spectra_array:
            row = np.concatenate([[spec.get_filename().split(".fits")[0]],
                                  [spec.TEMP_FRAME.loc[CURRENT].values[0] for CURRENT in HEADER[1:]]])
            output_table = np.vstack([output_table, row])

        table = Texttable()
        table.add_rows(output_table)

        print(" ------ TEMPERATURES (PHOTOMETRIC + HARD + ADOPTED)-------")
        if len(self.spectra_array) < 30:
            print(table.draw())
        else:
            print("Too long table to print; only save the temp table in a file in " + self.io_params['output_dir_path'])
        
        print(table.draw(), file=open(self.output_name + "_temp_cal_table.txt", "a"))
        return

    def set_KP_bounds(self):
        print("\n... setting KP bandwidth")
        [spec.set_KP_bounds(EW.get_KP_band(spec)) for spec in self.spectra_array]
        return

    def set_carbon_mode(self):
        print("\n... setting carbon mode")
        [EW.set_CH_procedure(spec) for spec in self.spectra_array]
        return

    def estimate_sn(self):
        print("\n... estimating S/N")
        [spec.estimate_sn() for spec in self.spectra_array]
        return

    def get_sn(self):
        snr = pd.concat([spec.get_sn() for spec in self.spectra_array])
        try:
            snr.to_csv(self.output_name + "_snr.csv", index=False)
        except:
            snr.to_csv(self.output_name + "1_snr.csv", index=False)
        return

    def set_mcmc_args(self):
        print('\n... building mcmc_args dict')
        [spec.set_mcmc_args() for spec in self.spectra_array]
        return

    def archetype_classification(self):
        print('\n... determining archetype classification')
        start_time = time.time()

        [interface_main.archetype_classify_MC(spec) for spec in self.spectra_array]

        output_table = ['NAME', "GI", "GII", "GIII"]

        for spec in self.spectra_array:
            row = np.concatenate([[spec.get_filename()], [spec.LL_DICT[key][0].round(0) for key in ["GI", "GII", "GIII"]]])
            output_table = np.vstack([output_table, row])

        table = Texttable()
        table.add_rows(output_table)
        print(" ------  ARCHETYPE LIKELIHOODS -------")
        if len(self.spectra_array) < 30:
            print(table.draw())
        print(table.draw(), file=open(self.output_name + "_archetype_likelihood_table.txt", "a"))
        print("\t\t interface_main: Time spent for archetype classification is {0:.1f}".format(time.time()-start_time))
        return

    def mcmc_determination(self):
        print('\n... performing MCMC determinations')
        start_time = time.time()
        [spec.prepare_regions() for spec in self.spectra_array]

        [interface_main.mcmc_determination(spec, mode='COARSE') for spec in self.spectra_array]

        print("... performing kde determinations")
        [interface_main.generate_kde_params(spec, mode="COARSE") for spec in self.spectra_array]

        print("... running refined mcmc")
        [interface_main.mcmc_determination(spec, mode='REFINE') for spec in self.spectra_array]

        print("... finalizing kde determinations")
        [interface_main.generate_kde_params(spec, mode='REFINE') for spec in self.spectra_array]

        print("\t\t batch: Time spent for mcmc determination is {0:.1f}".format(time.time()-start_time))
        print("... complete")
        return

    def estimate_logg(self):
        print("\n... estimating log g")
        [interface_main.estimate_logg(spec) for spec in self.spectra_array]
        return

    def generate_synthetic(self):
        print("\n... generating synthetic spectra")
        [interface_main.generate_synthetic(spec) for spec in self.spectra_array]
        return

    def generate_plots(self):
        print("... generating corner plots")
        plot_functions.plot_corner_array(self)

        print("... generating mcmc trace plots")
        plot_functions.plot_mcmc_trace_array(self)

        print("\n... generating plots")
        plot_functions.plot_spectra(self)
        return
    def generate_output_files(self):
        print("\n... generating outputs")
        final = pd.concat([spec.get_output_row() for spec in self.spectra_array])

        with open(os.path.join(NPSAVE_DIR, self.io_params['output_file_name'] + 'parameters_output.npy'), 'wb') as f:
            np.save(f, final)

        try:
            final.to_csv(self.output_name + "_out.csv", index=False)
        except:
            final.to_csv(self.output_name + "1_out.csv", index=False)
        return

    def generate_output_spectra(self):
        print("\n... generating output spectra")
        final_spectra = pd.concat([spec.get_spectra_row() for spec in self.spectra_array])

        with open(os.path.join(NPSAVE_DIR, self.io_params['output_file_name'] + 'spectra_output.npy'), 'wb') as f:
            np.save(f, final_spectra)

        try:
            final_spectra.to_csv(self.output_name + "_spectra_output.csv", index=False)
        except:
            final_spectra.to_csv(self.output_name + "spectra_output_1.csv", index=False)
        return
