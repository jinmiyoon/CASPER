### Name: Devin D. Whitten
### Email: dwhitten@nd.edu
### Institute: University of Notre Dame

### CCSLab_main.py

####
spectra_path  = 'inputs/spectra/'
param_path    = 'params/param_file_test.dat'
io_param_path = 'params/io_param.py'
####

import os, sys

sys.path.append("interface")
import GISIC_C as GISIC
import interface_main
import archetype_interface
import plot_functions
from batch import Batch




interface_main.print_greeting()

print("... initializing spectra batch")
spec_batch = Batch(spectra_path, param_path, io_param_path)

### load spectra
spec_batch.load_params()
spec_batch.load_spectra(is_fits=True)
spec_batch.set_params()


interface_main.span_window()

#spec_batch.radial_correct()
spec_batch.build_frames()


### normalize with GISIC
interface_main.span_window()


spec_batch.normalize()

#### Preliminaries
spec_batch.set_KP_bounds()
spec_batch.set_carbon_mode()

spec_batch.estimate_sn()
spec_batch.calibrate_temperatures()


##### big functions
spec_batch.archetype_classification()

spec_batch.mcmc_determination(pool=20)

##### generate output files

spec_batch.generate_synthetic()

spec_batch.generate_plots()

spec_batch.generate_output_files()
