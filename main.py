################################################################################
### Author: Devin Whitten
### Email: devin.d.whitten@gmail.com
### Institute: University of Notre Dame
################################################################################

### CASPER

####
<<<<<<< HEAD
#spectra_path  = 'inputs/spectra/gmos-ft2017a-reduced/'
#spectra_path  = 'inputs/spectra/'
spectra_path  = 'inputs/spectra/mods-phase1-reduced/'
#param_path    = 'params/param_file_trun.dat'
param_path    = 'params/mods-phase1-casper-input.csv'
#param_path    = 'params/param_file_jy1706_colors.dat'
=======
spectra_path  = 'inputs/spectra/'
param_path    = 'params/param_file_trun.dat'


>>>>>>> upstream/master
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
io_functions.print_greeting()

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
