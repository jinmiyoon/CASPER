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
io_paths = 'interface/io_paths.py'


import os, sys

sys.path.append("./interface")
from batch import Batch
from multiprocessing import freeze_support
import time

# We need the below line to attempt to start a new process before the current
# process has finished its bootstrapping phase. This line allows multiprocessing
# in interface_mcmc.run_mcmc_determination(). J. Yoon 03/11/2022
if __name__ == "__main__":
    freeze_support()

    start_time = time.time()


    # Create directory
    dirName = 'outputs/logs'
    """
    try:
        # Create target Directory
        os.mkdir(dirName)
        print("Directory " , dirName ,  " Created \n ")
    except FileExistsError:
        print("Directory " , dirName ,  " already exists \n")
    """
    # save the CASPER progress printouts in to a log file.
    sys.stdout=open(dirName+'/casper_run_'+time.strftime("%Y-%m-%d-%H:%M:%S")+'.log', 'wt')
    print("Started CASPER and logging! \n\n")

    print("### CAPER starts now: ###")
    print("\n ... initializing spectra batch")

    spec_batch = Batch(io_paths)
    spec_batch.set_io_paths()

    ################################################################################
    ### load spectra + params
    spec_batch.load_params()
    print("spectra name:  ", spec_batch.param_file['name'])
    spec_batch.load_spectra()
    spec_batch.set_params()

    #io_functions.span_window()

    spec_batch.radial_correct()
    spec_batch.build_frames()

    #io_functions.span_window()

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

    # is this procedure done for once for initial param for archetype_classification?
    spec_batch.calibrate_temperatures()

    spec_batch.archetype_classification()

    #spec_batch.mcmc_determination(pool=20) # no need of pool
    spec_batch.mcmc_determination()

    ################################################################################
    ##### generate output files
    spec_batch.generate_synthetic()
    spec_batch.generate_plots()
    spec_batch.generate_output_files()


    print("The total time for this CASPER run is {:.2f}s".format(time.time()-start_time))

    #make a sound when the script run is finished.
    os.system("say beep")
