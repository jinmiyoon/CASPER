import os
import time
from multiprocessing import freeze_support

from casper.interface.batch import Batch
from casper.utils.logger_config import setup_logger

logger = setup_logger(__name__)

io_paths = "interface/io_paths.py"


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

    # log_filename = f"casper_run_{time.strftime('%Y-%m-%d-%H-%M-%S')}.log"
    # sys.stdout = open(os.path.join(LOG_DIR, log_filename), "wt")
    # print("Started CASPER and logging! \n\n")
    logger.info("Started CASPER and logging!")

    # print("### CASPER starts now: ###")
    # print("\n ... initializing spectra batch")
    logger.info("CASPER starts now:")
    logger.info("... initializing spectra batch")

    spec_batch = Batch(io_paths)

    spec_batch.set_io_paths()

    spec_batch.load_params()
    # print("spectra name:  ", spec_batch.param_file["filename"])
    logger.info(f"spectra name: {spec_batch.param_file['filename']}")

    spec_batch.load_spectra()
    spec_batch.set_params()

    spec_batch.radial_correct()
    spec_batch.build_frames()

    # Continuum normalization with GISIC

    spec_batch.normalize()

    # Set preliminary setting
    spec_batch.set_KP_bounds()
    spec_batch.set_carbon_mode()

    spec_batch.estimate_sn()
    spec_batch.get_sn()

    # Extinction correction for color B-V before temperature calibration
    spec_batch.ebv_correction()

    spec_batch.calibrate_temperatures()

    # Classify tentative CEMP Group archetypes (I, II, III) with calibrated temperature
    spec_batch.archetype_classification()

    # Determine the best fit parameters with MCMC
    spec_batch.mcmc_determination()

    # Estimate surface gravity, `logg` using $Y^2$ isochrone
    spec_batch.estimate_logg()

    spec_batch.generate_synthetic()

    spec_batch.generate_output_spectra()

    spec_batch.generate_output_files()

    spec_batch.generate_plots()

    # print("The total time for this CASPER run is {:.2f}s".format(time.time() - start_time))

    total_time = time.time() - start_time
    logger.info(f"The total time for this CASPER run is {total_time:.2f} seconds.")

    os.system("say beep")
