################################################################################
### Author: Devin Whitten, Jinmi Yoon
### Email: devin.d.whitten@gmail.com, jinmi.yoon@gmail.com
################################################################################

### you can change output name in io_param.py


### J. Yoon
# To run CASPER, you need to set up paths for input spectra and parameters and
# output directory and the ouput files.
# io_paths lets you prepend the output name for parameter file as .csv,
# casper fit as .pdf file, and cornerplot for mcmc calculations for the best parameters.

io_paths = "interface/io_paths.py"


import os
import sys

sys.path.append("./interface")
import time
from multiprocessing import freeze_support

from interface.batch import Batch

# We need the below line to attempt to start a new process before the current
# process has finished its bootstrapping phase. This line allows multiprocessing
# in interface_mcmc.run_mcmc_determination(). J. Yoon 03/11/2022

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

LOG_DIR = os.path.join(PROJECT_ROOT, "logs")
NPSAVE_DIR = os.path.join(PROJECT_ROOT, "npsave")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(NPSAVE_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


if __name__ == "__main__":
    freeze_support()

    start_time = time.time()

    """
    try:
        # Create target Directory
        os.mkdir(dirName)
        print("Directory " , dirName ,  " Created \n ")
    except FileExistsError:
        print("Directory " , dirName ,  " already exists \n")
    """
    # save the CASPER progress printouts in to a log file.

    log_filename = f"casper_run_{time.strftime('%Y-%m-%d-%H-%M-%S')}.log"
    sys.stdout = open(os.path.join(LOG_DIR, log_filename), "wt")
    print("Started CASPER and logging! \n\n")

    print("### CASPER starts now: ###")
    print("\n ... initializing spectra batch")

    # instantiate a Batch object of spectra
    spec_batch = Batch(io_paths)

    # set io path
    spec_batch.set_io_paths()

    ################################################################################
    ### load spectra + params
    spec_batch.load_params()
    print("spectra name:  ", spec_batch.param_file["filename"])
    spec_batch.load_spectra()
    spec_batch.set_params()

    # io_functions.span_window()

    spec_batch.radial_correct()
    spec_batch.build_frames()

    # io_functions.span_window()

    ################################################################################
    #### Continuum normalization with GISIC
    spec_batch.normalize()

    ################################################################################
    #### Preliminaries
    spec_batch.set_KP_bounds()
    spec_batch.set_carbon_mode()

    spec_batch.estimate_sn()
    spec_batch.get_sn()

    # sys.exit()

    spec_batch.ebv_correction()

    ################################################################################
    #### Main procedures

    # is this procedure done for once for initial param for archetype_classification?
    spec_batch.calibrate_temperatures()

    # decide the tentative archetype classification
    spec_batch.archetype_classification()

    # spec_batch.mcmc_determination(pool=20) # no need of pool
    spec_batch.mcmc_determination()

    # interpolate gravity logg from isochrone
    spec_batch.estimate_logg()

    ################################################################################
    ##### generate output files

    # generate synthetic spectra
    spec_batch.generate_synthetic()

    # generate a file of both observed and synthetic spectra for later plot manipulation
    spec_batch.generate_output_spectra()

    # generate an output file of stellar parameters and other parameters
    spec_batch.generate_output_files()

    # generate plots: spectral fit, corner plot, trace plot
    spec_batch.generate_plots()

    print("The total time for this CASPER run is {:.2f}s".format(time.time() - start_time))

    # make a sound when the script run is finished.
    os.system("say beep")
