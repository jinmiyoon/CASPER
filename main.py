################################################################################
### Author: Devin Whitten, Jinmi Yoon
### Email: devin.d.whitten@gmail.com, jinmi.yoon@gmail.com
### Institute: University of Notre Dame
################################################################################

### you can change output name in io_param.py


###
# To run CASPER, you need to set up paths for input spectra and parameters and
# output directory and the ouput files.
# io_paths lets you prepend the output name for parameter file as .csv,
# casper fit as .pdf file, and cornerplot for mcmc calculations for the best parameters.
io_paths = 'interface/io_paths.py'


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
#sys.stdout=open('output/validation/validation2/rv_correction_test_result-1.txt', 'wt')

start_time = time.time()
print("... initializing spectra batch")

spec_batch = Batch(io_paths)
spec_batch.set_io_paths()

################################################################################
### load spectra + params
spec_batch.load_params()
spec_batch.load_spectra(is_fits=True)
spec_batch.set_params()

io_functions.span_window()

spec_batch.radial_correct()
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

# does this procedure is done for once for initial param for archetype_classification?
spec_batch.calibrate_temperatures()

spec_batch.archetype_classification()

spec_batch.mcmc_determination(pool=20)

################################################################################
##### generate output files
spec_batch.generate_synthetic()
spec_batch.generate_plots()
spec_batch.generate_output_files()

print("The total time for this CASPER run is {:.2f}s".format(time.time()-start_time))
print('\007')
