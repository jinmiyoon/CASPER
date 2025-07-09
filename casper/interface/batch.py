import os
import time

import numpy as np
import pandas as pd
from astropy.io import fits
from texttable import Texttable

from casper.interface import EW, config, interface_main, plot_functions
from casper.interface import temp_calibrations as TC
from casper.interface.gisic.normalize import normalize
from casper.interface.spectrum import Spectrum
from casper.utils.logger_config import setup_logger

logger = setup_logger(__name__)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


NPSAVE_DIR = os.path.join(PROJECT_ROOT, "npsave")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")


os.makedirs(NPSAVE_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


class Batch:
    def __init__(self, io_paths: dict[str, str]) -> None:
        """
        Initialize the Batch object with the input/output paths.

        Args:
            io_paths (dict[str, str]): Dictionary containing paths to spectra, parameter files, and output locations.
        """

        self.io_paths = io_paths
        return

    def set_io_paths(self) -> None:
        """
        Load and configure I/O paths from a parameter file.

        This method reads a configuration file specified by `self.io_paths`,
        parses the paths for the parameter file, spectra directory, and output file,
        and sets the corresponding attributes on the object.

        Sets:
            self.io_params (dict): Dictionary containing keys like 'param_path',
                'spectra_dir_path', and 'output_file_name' loaded from the I/O parameter file.
            self.param_path (str): Absolute path to the input parameter file.
            self.spectra_path (str): Absolute path to the directory containing spectra files.
            self.output_name (str): Full output file path including filename, saved under OUTPUT_DIR.
        """

        logger.info(f"\nloading io_paths: {self.io_paths}")

        self.io_params = eval(open(self.io_paths, "r").read())

        logger.info("\nsetting io_paths:")

        self.param_path = os.path.abspath(os.path.join(PROJECT_ROOT, self.io_params["param_path"]))
        self.spectra_path = os.path.abspath(os.path.join(PROJECT_ROOT, self.io_params["spectra_dir_path"]))
        self.output_name = os.path.join(OUTPUT_DIR, self.io_params["output_file_name"])
        logger.info(f"\t\t\t > input setting parameter    : {self.param_path}")

        logger.info(f"\t\t\t > input spectra directory    : {self.spectra_path}")

        logger.info(f"\t\t\t > output directory + filename: {self.output_name}")

        return

    def load_params(self) -> None:
        """
        Load stellar parameter file and process key metadata.

        This method reads the CSV file at `self.param_path` and stores it in `self.param_file`.
        It ensures key columns like 'sequence', 'mode', 'class', and 'carbon_mode' are treated as strings,
        and stores the list of sequences for further use.

        Sets:
            self.param_file (DataFrame): Parsed parameter table from CSV.
            self.sequence (List[str]): List of sequence identifiers from the parameter file.
        """

        logger.info(f"\nloading input params: {self.param_path}")

        self.param_file = pd.read_csv(self.param_path)
        logger.info(f"Input parameter file columns: {list(self.param_file.columns)}")

        self.param_file["sequence"] = self.param_file["sequence"].astype(str)
        self.sequence = self.param_file["sequence"].tolist()
        self.param_file["mode"] = self.param_file["mode"].astype(str)
        self.param_file["class"] = self.param_file["class"].astype(str)
        self.param_file["carbon_mode"] = self.param_file["carbon_mode"].astype(str)
        return

    def load_spectra(self) -> None:
        """
        Load observed spectra based on filenames listed in the parameter file.

        This method constructs the full path to each spectrum file listed in `self.param_file["filename"]`,
        reads the spectrum data (in either FITS or CSV format), and stores it as a list of `Spectrum` objects.

        Sets:
            self.spectra_names (List[str]): List of filenames extracted from the parameter file.
            self.spectra_array (List[Spectrum]): List of Spectrum objects created from the input files.
            self.length (int): Number of loaded spectra.
        """

        logger.info(f"\n ... loading spectra: {self.spectra_path}")

        self.spectra_names = self.param_file["filename"].tolist()

        def spectra_input(pathname: str, current: str) -> Spectrum:
            """
            Load a spectrum from a FITS or CSV file and return it as a Spectrum object.

            Args:
                pathname (str): Full file path to the spectrum file.
                current (str): Filename used to determine the file type and pass metadata.

            Returns:
                Spectrum: A Spectrum object initialized from the file contents.

            Raises:
                Exception: If the file extension is not '.fits' or '.csv'.
            """

            file_ext = current.split(".")[1]
            if file_ext == "fits":
                with fits.open(pathname) as hdu:
                    return Spectrum(hdu, filename=current, is_fits=True)
            elif file_ext == "csv":
                return Spectrum(pd.read_csv(pathname), filename=current, is_fits=False)
            else:
                raise Exception("Invalid file format extension. Currently only .fits and .csv files are supported")

        self.spectra_array = [
            spectra_input(os.path.join(self.spectra_path, current), current) for current in self.spectra_names
        ]

        logger.info(f"\t\t batch: what is spectra_array - {self.spectra_array}")

        self.length = len(self.spectra_array)

        return

    def set_params(self) -> None:
        """
        Assign input parameters from the parameter file to each Spectrum object in the batch.

        Iterates over all rows in the loaded parameter DataFrame and sets key attributes on
        the corresponding Spectrum object, including stellar identifiers, observational class,
        carbon mode, MCMC iteration settings, and temperature constraints.

        Raises:
            AssertionError: If a filename in the parameter file does not match the corresponding Spectrum.
        """

        logger.info("\n... setting spectra parameters")

        for i, row in self.param_file.iterrows():
            spec = self.spectra_array[i]

            assert spec.filename == row["filename"].strip(), "Name is not found!"
            SEQUENCE = row["sequence"]
            STARNAME = row["starname"]
            JK = row["J-K"]
            CLASS = row["class"].strip()
            MODE = row["mode"].strip()
            INPUT_CARBON_MODE = row["carbon_mode"].strip()
            ITER = row["MCMC_iter"]
            T_SIGMA = row["T_SIGMA"]
            HARD_TEFF = row["TEFF_SET"]

            spec.set_params(
                SEQUENCE=SEQUENCE,
                STARNAME=STARNAME,
                CLASS=CLASS,
                JK=JK,
                MODE=MODE,
                INPUT_CARBON_MODE=INPUT_CARBON_MODE,
                iter=ITER,
                T_SIGMA=T_SIGMA,
                HARD_TEFF=HARD_TEFF,
            )

        return

    def radial_correct(self) -> None:
        """
        Apply radial velocity correction to each Spectrum object in the batch.

        For each entry in the parameter file, retrieves the corresponding radial velocity (RV)
        and applies a wavelength shift to the associated Spectrum to correct for Doppler effects.

        Prints the correction applied for each spectrum.

        Raises:
            ValueError: If RV is missing or cannot be converted to float.
        """

        logger.info("\n... correcting radial velocities")

        for sequence, spec in zip(self.sequence, self.spectra_array):
            radial_velocity = float(self.param_file[self.param_file["sequence"] == sequence]["RV"])
            spec.radial_correction(radial_velocity)
            logger.info(f"\t For {sequence + ': ' + spec.filename},  RV = {radial_velocity:7.2f} km/s")

    def build_frames(self, bounds: tuple = config.WAVE_BOUNDS) -> None:
        """
        Construct and trim data frames for each spectrum in the batch.

        For each Spectrum object:
            - Sets its internal frame using its current wavelength and flux data.
            - Trims the frame to the specified wavelength bounds.

        Args:
            bounds (tuple, optional): A (min, max) wavelength range to trim each spectrum to.
                Defaults to config.WAVE_BOUNDS.
        """

        logger.info("\n ... build dataframes")

        [spec.set_frame(wave=spec.get_wave(), flux=spec.get_flux()) for spec in self.spectra_array]
        [spec.trim_frame(bounds) for spec in self.spectra_array]
        return

    def normalize(self, default: bool = True) -> None:
        """
        Normalize all spectra in the batch using GISIC normalization.

        For each spectrum in the batch:
            - Applies GISIC normalization for multiple convolution sigma values.
            - Averages the continuum estimates and uses them to normalize the spectrum.
            - Clips normalized flux values to the range [0.0, 2.0] for stability.
            - Updates the spectrum with the computed normalized flux and continuum.

        Args:
            default (bool, optional): If True, performs standard GISIC normalization.
                Custom normalization is not yet implemented. Defaults to True.
        """

        logger.info("\n... normalizing spectra batch")

        logger.info("... iterating convolution sigma")

        start_time = time.time()
        if default:
            for spec in self.spectra_array:
                cont_array = []
                for SIGMA in config.SIGMA:
                    _, norm, cont = normalize(
                        spec.get_frame_wave(),
                        spec.get_frame_flux(),
                        sigma=SIGMA,
                        k=config.k,
                        cahk=config.cahk,
                        band_check=config.cahk,
                        flux_min=config.flux_min,
                        boost=config.boost,
                    )

                    cont_array.append(cont)

                cont = np.median(np.array(cont_array), axis=0)
                norm = np.divide(spec.get_frame_flux(), cont)

                norm[norm < 0.0] = 1.0
                norm[norm > 2.0] = 1.0

                spec.set_frame_norm(norm)
                spec.set_frame_cont(cont)
                logger.info(f"\t {spec.filename:20s} :  okay")

        else:
            logger.warning("\t Sorry - can't customize GISIC normalization yet...")

        logger.info(f"\t\t batch: Time spent normalizing the observed spectra is {time.time() - start_time:.1f}")

        return

    def ebv_correction(self) -> None:
        """
        Apply E(B-V) reddening correction to each spectrum's photometric data.

        Iterates through the parameter file and applies extinction correction
        to each corresponding spectrum in the batch using its row values.
        """

        logger.info("\n... correcting photometry")

        for i, row in self.param_file.iterrows():
            spec = self.spectra_array[i]
            spec.ebv_correct(row)

    def calibrate_temperatures(self, default: bool = True, teff_sigma: int = 250) -> None:
        """
        Calibrate and assign effective temperatures (Teff) for each spectrum.

        Uses photometric color indices (J-K and g-r) and gravity class to estimate Teff.
        If a hard-set Teff value is available, it overrides the photometric estimate.

        Parameters
        ----------
        default : bool, optional
            Flag for default behavior (not currently used), by default True.
        teff_sigma : int, optional
            Standard deviation to associate with temperature uncertainty, by default 250.

        Side Effects
        ------------
        - Updates the `TEMP_FRAME` for each spectrum with multiple photometric Teff estimates.
        - Sets the adopted temperature (hard or photometric).
        - Writes a temperature summary table to an output file.
        """

        logger.info("\n... determining temperature for archetype classification")

        for i, row in self.param_file.iterrows():
            spec = self.spectra_array[i]

            assert spec.filename == row["filename"], "Parameter error in calibrate_temperatures()"
            logger.info(f"\t setting input temperature sigma: {spec.T_SIGMA}")

            CLASS = row["class"].strip()

            spec.set_temp_frame(
                TC.calibrate_temp_frame(float(spec.PHOTO_0["J-K"]), float(spec.PHOTO_0["g-r"]), CLASS=CLASS)
            )

            if np.isfinite(spec.HARD_TEFF):
                spec.TEMP_FRAME.loc["HARD_TEFF", "VALUE"] = spec.HARD_TEFF
                spec.TEMP_FRAME.loc["ADOPTED", "VALUE"] = spec.HARD_TEFF
                spec.set_temperature(spec.HARD_TEFF, spec.T_SIGMA)
                logger.info(f"\t setting and adopting hard teff: {spec.HARD_TEFF}")

            else:
                spec.TEMP_FRAME.loc["HARD_TEFF", "VALUE"] = np.nan
                spec.set_temperature(spec.TEMP_FRAME.loc["ADOPTED", "VALUE"], sigma=spec.T_SIGMA)
                logger.info(f"\t setting and adopting photo teff: {spec.TEMP_FRAME.loc['ADOPTED', 'VALUE']}")

        HEADER = ["NAME", "Bergeat", "Hernandez", "Casagrande", "Fukugita", "HARD_TEFF", "ADOPTED"]
        output_table = HEADER

        for spec in self.spectra_array:
            row = np.concatenate(
                [
                    [spec.get_filename().split(".fits")[0]],
                    [spec.TEMP_FRAME.loc[CURRENT].values[0] for CURRENT in HEADER[1:]],
                ]
            )
            output_table = np.vstack([output_table, row])

        table = Texttable()
        table.add_rows(output_table)

        logger.info(" ------ TEMPERATURES (PHOTOMETRIC + HARD + ADOPTED)-------")

        if len(self.spectra_array) < 30:
            logger.info("\n" + table.draw())

        else:
            logger.info(
                "Too long table to print; only saving the temp table in a file at " + self.io_params["output_dir_path"]
            )

        print(table.draw(), file=open(self.output_name + "_temp_cal_table.txt", "a"))
        return

    def set_KP_bounds(self) -> None:
        """
        Set the Ca II K line passband (KP) wavelength bounds for each spectrum.

        Uses the `get_KP_band()` function from the EW module to calculate KP boundaries
        and assigns them to each spectrum in the batch.

        Side Effects
        ------------
        - Updates each spectrum's KP_bounds attribute.
        """

        logger.info("\n... setting KP bandwidth")

        [spec.set_KP_bounds(EW.get_KP_band(spec)) for spec in self.spectra_array]
        return

    def set_carbon_mode(self) -> None:
        """
        Set the carbon mode ("CH" or "CH+C2") for each spectrum in the batch.

        Uses the set_CH_procedure() function from the EW module to determine and assign
        the appropriate carbon classification mode based on CH band strength and noise characteristics.

        Side Effects
        ------------
        - Updates each spectrum's carbon_mode attribute.
        """

        logger.info("\n... setting carbon mode")

        [EW.set_CH_procedure(spec) for spec in self.spectra_array]
        return

    def estimate_sn(self) -> None:
        """
        Estimate the signal-to-noise ratio (S/N) for each spectrum in the batch.

        Applies the estimate_sn() method on each Spectrum object in the spectra_array.
        This method computes noise characteristics for relevant wavelength regions
        and stores the S/N statistics in each spectrum's SN_DICT attribute.

        Side Effects
        ------------
        - Updates each spectrum's SN_DICT with estimated S/N values.
        """

        logger.info("\n... estimating S/N")

        [spec.estimate_sn() for spec in self.spectra_array]
        return

    def get_sn(self) -> None:
        """
        Retrieve and save the signal-to-noise ratio (S/N) data for all spectra.

        This method calls get_sn() on each Spectrum object to gather S/N information,
        concatenates the results into a single DataFrame, and writes it to a CSV file.

        Output
        ------
        - Saves a CSV file containing the S/N data to the specified output path.

        Notes
        -----
        - If the first attempt to save fails (e.g., due to file permissions or naming issues),
        it attempts to save the file with an alternate name ending in "1_snr.csv".
        """

        snr = pd.concat([spec.get_sn() for spec in self.spectra_array])
        try:
            snr.to_csv(self.output_name + "_snr.csv", index=False)
        except:
            snr.to_csv(self.output_name + "1_snr.csv", index=False)
        return

    def set_mcmc_args(self) -> None:
        """
        Build and set MCMC argument dictionaries for all spectra.

        This method iterates over each Spectrum object in the batch and
        calls set_mcmc_args(), which prepares the required arguments for
        running MCMC parameter estimation.

        Notes
        -----
        - MCMC arguments typically include spectral regions, synthetic wavelength grids,
        initial temperature estimates, and inverse S/N weights.
        - These are used later during coarse and refined MCMC runs.
        """

        logger.info("\n... building mcmc_args dict")
        [spec.set_mcmc_args() for spec in self.spectra_array]
        return

    def archetype_classification(self) -> None:
        """
        Perform archetype classification for all spectra in the batch.

        This method computes the likelihood of each spectrum belonging to one of the three
        archetype gravity classes (GI, GII, GIII) by calling `archetype_classify_MC()` on
        each Spectrum object. It then generates and prints a table summarizing the results.

        Output
        ------
        - A printed likelihood table displaying likelihood scores for each gravity class per star.
        - A text file saved to disk with suffix "_archetype_likelihood_table.txt".
        - Prints total time spent on classification.

        Notes
        -----
        - Uses the precomputed LL_DICT for likelihood values.
        - Assumes that likelihoods for GI, GII, and GIII have been correctly computed and stored.
        """

        logger.info("\n... determining archetype classification")

        start_time = time.time()

        [interface_main.archetype_classify_MC(spec) for spec in self.spectra_array]

        output_table = ["NAME", "GI", "GII", "GIII"]

        for spec in self.spectra_array:
            row = np.concatenate(
                [[spec.get_filename()], [spec.LL_DICT[key][0].round(0) for key in ["GI", "GII", "GIII"]]]
            )
            output_table = np.vstack([output_table, row])

        table = Texttable()
        table.add_rows(output_table)
        logger.info(" ------  ARCHETYPE LIKELIHOODS -------")

        if len(self.spectra_array) < 30:
            logger.info("\n" + table.draw())

        print(table.draw(), file=open(self.output_name + "_archetype_likelihood_table.txt", "a"))
        logger.info(f"\t\t interface_main: Time spent for archetype classification is {time.time() - start_time:.1f}")

        return

    def mcmc_determination(self) -> None:
        """
        Run MCMC parameter estimation and KDE post-processing for each spectrum in the batch.

        This method:
        1. Prepares spectral regions needed for MCMC.
        2. Performs a coarse MCMC run to estimate initial stellar parameters.
        3. Applies KDE smoothing to the coarse MCMC results.
        4. Performs a refined MCMC run based on the coarse outputs.
        5. Applies KDE smoothing to the refined MCMC results.

        Output
        ------
        - Updates each Spectrum object in `self.spectra_array` with MCMC chains, best-fit values,
        and KDE distributions.
        - Prints progress messages and total time taken.
        """

        logger.info("\n... performing MCMC determinations")
        start_time = time.time()
        [spec.prepare_regions() for spec in self.spectra_array]

        [interface_main.mcmc_determination(spec, mode="COARSE") for spec in self.spectra_array]

        logger.info("... performing kde determinations")
        [interface_main.generate_kde_params(spec, mode="COARSE") for spec in self.spectra_array]

        logger.info("... running refined mcmc")
        [interface_main.mcmc_determination(spec, mode="REFINE") for spec in self.spectra_array]

        logger.info("... finalizing kde determinations")
        [interface_main.generate_kde_params(spec, mode="REFINE") for spec in self.spectra_array]

        logger.info(f"\t\t batch: Time spent for mcmc determination is {time.time() - start_time:.1f}")
        logger.info("... complete")

        return

    def estimate_logg(self) -> None:
        """
        Estimate surface gravity (log g) for each spectrum using coarse MCMC parameters.

        For each spectrum:
        - Interpolates log g from temperature and metallicity using a gravity calibration.
        - Computes uncertainty using the standard deviation and MAD of the sampled distribution.

        Output
        ------
        - Updates each Spectrum object with `logg` and `logg_err`.
        - Prints log g and its uncertainty for each spectrum.
        """

        logger.info("\n... estimating log g")

        [interface_main.estimate_logg(spec) for spec in self.spectra_array]
        return

    def generate_synthetic(self) -> None:
        """
        Generate synthetic spectra for each Spectrum object in the batch.

        For each spectrum:
        - Uses interpolated model flux values based on MCMC-derived parameters.
        - Applies continuum normalization to the synthetic flux.
        - Sets the resulting synthetic spectrum on the Spectrum object.

        Output
        ------
        - Updates each Spectrum with a new `synth_spectrum` DataFrame.
        - Handles edge cases where interpolation fails due to invalid parameters.
        """

        logger.info("\n... generating synthetic spectra")

        [interface_main.generate_synthetic(spec) for spec in self.spectra_array]
        return

    def generate_plots(self) -> None:
        """
        Generate diagnostic and spectral plots for the batch.

        This method produces:
        - Corner plots from MCMC posteriors for each spectrum.
        - Trace plots for MCMC chains to assess convergence.
        - Observed vs. synthetic spectral comparison plots.

        Output
        ------
        Saves plots to output directory as defined in `self.output_name`.
        """
        logger.info("... generating corner plots")

        plot_functions.plot_corner_array(self)

        logger.info("... generating mcmc trace plots")

        plot_functions.plot_mcmc_trace_array(self)

        logger.info("\n... generating plots")

        plot_functions.plot_spectra(self)
        return

    def generate_output_files(self) -> None:
        """
        Generate and save output parameter files for all spectra.

        This method:
        - Collects output parameter rows from each spectrum.
        - Saves the combined DataFrame as a `.npy` binary file in the NPSAVE_DIR.
        - Attempts to write the same DataFrame to a CSV file using the defined output name.

        Output
        ------
        - A `.npy` file for structured numpy access.
        - A `.csv` file for tabular review; backup is saved with "1" appended if the first fails.
        """

        logger.info("\n... generating outputs")

        final = pd.concat([spec.get_output_row() for spec in self.spectra_array])

        with open(os.path.join(NPSAVE_DIR, self.io_params["output_file_name"] + "parameters_output.npy"), "wb") as f:
            np.save(f, final)

        try:
            final.to_csv(self.output_name + "_out.csv", index=False)
        except:
            final.to_csv(self.output_name + "1_out.csv", index=False)
        return

    def generate_output_spectra(self) -> None:
        """
        Generate and save output spectral data for all spectra.

        This method:
        - Collects processed spectral data rows from each spectrum.
        - Saves the combined DataFrame as a `.npy` binary file in the NPSAVE_DIR.
        - Attempts to write the same DataFrame to a CSV file using the defined output name.

        Output
        ------
        - A `.npy` file for structured numpy access.
        - A `.csv` file for tabular review; backup is saved with a suffix if the first attempt fails.
        """

        logger.info("\n... generating output spectra")

        final_spectra = pd.concat([spec.get_spectra_row() for spec in self.spectra_array])

        with open(os.path.join(NPSAVE_DIR, self.io_params["output_file_name"] + "spectra_output.npy"), "wb") as f:
            np.save(f, final_spectra)

        try:
            final_spectra.to_csv(self.output_name + "_spectra_output.csv", index=False)
        except:
            final_spectra.to_csv(self.output_name + "spectra_output_1.csv", index=False)
        return
