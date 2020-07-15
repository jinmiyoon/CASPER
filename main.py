################################################################################
### Author: Devin Whitten, revised by Jinmi Yoon
### Email: devin.d.whitten@gmail.com, jinmi.yoon@gmail.com
### Institute: University of Notre Dame
################################################################################

#### To run CASPER, you need to set up paths for spectra and parameters below.
#### Further, you can change output name in io_param.py

### Example path set up for CASPER
#spectra_path  = 'inputs/spectra/'
#param_path    = 'params/param_file_trun.dat'

### set up for spectra and input file for my program stars
spectra_path  = 'inputs/spectra/bf-survey-data/'
param_path    = 'params/g77-61-input.csv'

### io_param_path lets you prepend the output name for parameter file as .csv, casper fit as .pdf file, and cornerplot for mcmc calculations for the best parameters.
io_param_path = 'params/io_param.py'
####

import os, sys

sys.path.append("interface")
import GISIC_C as GISIC
import interface_main
import io_functions
import archetype_interface
import plot_functions
from batch import Batch
import time

print(" Started CASPER and logging!")
sys.stdout=open('output/g77-61/casper-output-log-g77-61-gisic-flux_min80-sigma15_25.txt', 'wt')

start_time = time.time()
print("... initializing spectra batch")
spec_batch = Batch(spectra_path, param_path, io_param_path)

################################################################################
### load spectra + params
spec_batch.load_params()
spec_batch.load_spectra(is_fits=True)
spec_batch.set_params()

io_functions.span_window()

#spec_batch.radial_correct()
spec_batch.build_frames()

io_functions.span_window()

################################################################################
#### Continuum normalization with GISIC
spec_batch.normalize()

################################################################################
#### Preliminaries
spec_batch.set_KP_bounds()
spec_batch.set_carbon_mode()

spec_batch.estimate_sn()
spec_batch.ebv_correction()

################################################################################
#### Main procedures
spec_batch.calibrate_temperatures()

spec_batch.archetype_classification()

spec_batch.mcmc_determination(pool=20)

################################################################################
##### generate output files
spec_batch.generate_synthetic()
spec_batch.generate_plots()
spec_batch.generate_output_files()

print("The total time for this CASPER run is {:.2f}s".format(time.time()-start_time))
