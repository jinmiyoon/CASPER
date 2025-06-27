from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from interface import MAD, config


def obtain_flux(data):
    """
    Extract a 1D flux array from FITS-like input data with varying shapes.

    If the input data is 1D, it is flattened and returned directly.
    If the data is multi-dimensional, the first row is selected and flattened.

    Parameters
    ----------
    data : np.ndarray
        Input data array, typically from a FITS file.

    Returns
    -------
    np.ndarray
        A flattened 1D array representing the flux.
    """
    shape = data.shape

    if len(shape) == 1:
        return data.flatten()
    else:
        return data[0].flatten()


class Spectrum:
    def __init__(self, spec: Any, filename: str, is_fits: bool = True) -> None:
        """
        Initialize a spectrum from either a FITS header/data object or a CSV-style DataFrame.

        Parameters
        ----------
        spec : Any
            The spectrum input object.
            - If `is_fits` is True: `spec` should be a FITS object with accessible headers and data.
            - If `is_fits` is False: `spec` should be a pandas DataFrame with "wave" and "flux" columns.
        filename : str
            The name of the input file, used for logging and reference.
        is_fits : bool, optional
            Flag indicating whether the input is a FITS file. If False, CSV-style input is expected.
            Default is True.

        Attributes
        ----------
        self.wavelength : np.ndarray
            Computed wavelength array from FITS header or CSV column.
        self.original_wavelength : np.ndarray
            Copy of the original wavelength array.
        self.flux : np.ndarray
            Flux values corresponding to the wavelengths.
        self.segments : None
            Placeholder for segmented spectrum data (to be initialized later).
        self.mad_global : None
            Placeholder for global median absolute deviation (optional post-processing).
        """
        self.filename = filename
        print("\n... initializing:  ", filename)

        if is_fits:
            if "CD1_1" in spec[0].header:
                DELTA = "CD1_1"

            elif "CDELT1" in spec[0].header:
                DELTA = "CDELT1"

            else:
                print("I don't know which increment to use!")

            if spec[0].header["CRVAL1"] > 10.0:
                self.wavelength = (np.arange(0, spec[0].header["NAXIS1"], 1) * spec[0].header[DELTA]) + spec[0].header[
                    "CRVAL1"
                ]

            else:
                self.wavelength = np.power(
                    10.0, (spec[0].header["CRVAL1"] + np.arange(0, spec[0].header["NAXIS1"]) * spec[0].header[DELTA])
                )

            self.original_wavelength = self.wavelength

            self.flux = obtain_flux(spec[0].data)
            self.wavelength = np.array(self.wavelength)
            print("spectrum loaded")

            spec.close()

            if self.flux.dtype.byteorder == ">":
                print("... correcting endian mismatch")
                self.flux = self.flux.byteswap().view(self.flux.dtype.newbyteorder())

        else:
            print("\t csv file, ")
            self.spec = spec
            self.flux = self.spec["flux"]
            self.wavelength = np.array(self.spec["wave"], dtype=float)
            self.original_wavelength = self.wavelength

        self.segments = None

        self.mad_global = None

        return

    def radial_correction(self, velocity: float = 0.0) -> None:
        """
        Apply radial velocity correction to the wavelength array.

        This function updates the object's wavelength array by shifting it
        to the rest frame using the provided radial velocity.

        Parameters
        ----------
        velocity : float, optional
            Radial velocity in km/s. Default is 0.0 (no correction).

        Updates
        -------
        self.rv : float
            Stores the input radial velocity.
        self.wavelength : np.ndarray
            Corrected wavelength array.
        """
        self.rv = velocity

        self.wavelength = self.original_wavelength / ((velocity / 2.99792e5) + 1)

        return

    def ebv_correct(self, row: pd.Series) -> None:
        """
        Apply reddening correction (E(B-V)) to observed color indices using SFD extinction values.

        This function updates `self.PHOTO_0` with dereddened values for:
        - J-K
        - H-K
        - g-r

        If E(B-V) > 0, the correction is applied using extinction coefficients from config.A_EBV.
        Otherwise, it assumes the values are already corrected.

        Parameters
        ----------
        row : pd.Series
            A pandas row containing the observed color indices:
            - "J-K", "H-K", "g-r"
            and the reddening value:
            - "EBV_SFD"

        Updates
        -------
        self.PHOTO_0 : dict
            A dictionary of dereddened color indices.
        """
        self.PHOTO_0 = {key: float(row[key]) for key in ["J-K", "H-K", "H-K", "g-r"]}

        if float(row["EBV_SFD"]) > 0:
            print("\t corrected :", self.get_filename())

            self.PHOTO_0["J-K"] = float(row["J-K"]) - (float(config.A_EBV["A_J"]) - float(config.A_EBV["A_K"])) * float(
                row["EBV_SFD"]
            )

            self.PHOTO_0["H-K"] = float(row["H-K"]) - (float(config.A_EBV["A_H"]) - float(config.A_EBV["A_K"])) * float(
                row["EBV_SFD"]
            )

            self.PHOTO_0["g-r"] = float(row["g-r"]) - (float(config.A_EBV["A_g"]) - float(config.A_EBV["A_r"])) * float(
                row["EBV_SFD"]
            )

        else:
            print("\t already corrected:  ", self.get_filename())

    def trim_frame(self, bounds: Tuple[float, float] = config.WAVE_BOUNDS) -> None:
        """
        Trim the spectrum frame to the specified wavelength range.

        This function filters `self.frame` to keep only rows where the "wave" column
        falls within the given wavelength bounds (inclusive).

        Parameters
        ----------
        bounds : Tuple[float, float], optional
            Wavelength range as (min_wave, max_wave). Default is config.WAVE_BOUNDS.

        Updates
        -------
        self.frame : pd.DataFrame
            Trimmed DataFrame containing only wavelengths within the specified range.
        """
        self.frame = self.frame[self.frame["wave"].between(bounds[0], bounds[1], inclusive="both")]
        return

    def estimate_sn(self) -> None:
        """
        Estimate the signal-to-noise ratio (S/N) and inverse variance (XI) in sideband regions.

        This function calculates median S/N and XI values, along with their robust spread (using MAD),
        for each spectral region defined in `config.SIDEBANDS`. It also derives shape parameters
        'alpha' and 'beta' for modeling inverse noise variance as a beta distribution.

        Conditions
        ----------
        - The function only evaluates bands if both sideband intervals fall within the wavelength coverage.
        - If a band is out of range, NaNs are stored in the result.

        Updates
        -------
        self.SN_DICT : dict
            A dictionary where each key (e.g., "CA", "CH") maps to a sub-dictionary containing:
            - SN_AVG: average signal-to-noise across both sidebands
            - SN_STD: robust standard deviation (MAD-based) of S/N
            - XI_AVG: average inverse S/N
            - XI_STD: robust std dev of inverse S/N
            - alpha: shape parameter for beta distribution modeling inverse S/N
            - beta: shape parameter for beta distribution modeling inverse S/N
        """
        self.SN_DICT = {key: [] for key in config.SIDEBANDS.keys()}
        for key in config.SIDEBANDS.keys():
            if (config.SIDEBANDS[key][0][0] > min(self.frame["wave"])) and (
                config.SIDEBANDS[key][1][1] < max(self.frame["wave"])
            ):
                SN_LEFT = np.sqrt(
                    self.frame["flux"][self.frame["wave"].between(*config.SIDEBANDS[key][0], inclusive="both")]
                )
                SN_RIGHT = np.sqrt(
                    self.frame["flux"][self.frame["wave"].between(*config.SIDEBANDS[key][1], inclusive="both")]
                )

                self.SN_DICT[key] = {
                    "SN_AVG": np.mean([np.median(SN_LEFT), np.median(SN_RIGHT)]),
                    "SN_STD": max([MAD.S_MAD(SN_LEFT), MAD.S_MAD(SN_RIGHT)]),
                    "XI_AVG": np.mean([np.median(np.divide(1.0, SN_LEFT)), np.median(np.divide(1.0, SN_RIGHT))]),
                    "XI_STD": max([MAD.S_MAD(np.divide(1.0, SN_LEFT)), MAD.S_MAD(np.divide(1.0, SN_RIGHT))]),
                }

                self.SN_DICT[key]["alpha"] = (
                    (self.SN_DICT[key]["XI_AVG"] ** 2) / np.square(self.SN_DICT[key]["XI_STD"])
                ) * (1 - self.SN_DICT[key]["XI_AVG"]) - self.SN_DICT[key]["XI_AVG"]
                self.SN_DICT[key]["beta"] = (1 / self.SN_DICT[key]["XI_AVG"] - 1) * self.SN_DICT[key]["alpha"]

            else:
                print("band not in wavelength coverage")

                self.SN_DICT[key] = {
                    "SN_AVG": np.nan,
                    "SN_STD": np.nan,
                    "XI_AVG": np.nan,
                    "XI_STD": np.nan,
                    "alpha": np.nan,
                    "beta": np.nan,
                }
        return

    def get_sn(self) -> pd.DataFrame:
        """
        Compile a summary of signal-to-noise and inverse S/N statistics for key spectral bands.

        Returns
        -------
        pd.DataFrame
            A single-row DataFrame containing:
            - Sequence and filename
            - SN_AVG and SN_STD for CA and CH bands
            - XI_AVG and XI_STD for CA, CH, and C2 bands
            - XI_C2 and XI_C2_ERR if carbon mode is "CH+C2"
        """
        sn_output = pd.DataFrame(
            {
                "SEQUENCE": [self.get_sequence()],
                "FILENAME": [self.get_filename()],
                "SN_AVG_CA": [round(self.SN_DICT["CA"]["SN_AVG"], 0)],
                "SN_STD_CH": [round(self.SN_DICT["CA"]["SN_STD"], 0)],
                "SN_AVG_CA": [round(self.SN_DICT["CH"]["SN_AVG"], 0)],
                "SN_STD_CH": [round(self.SN_DICT["CH"]["SN_STD"], 0)],
                "XI_AVG_CA": [round(self.SN_DICT["CA"]["XI_AVG"], 4)],
                "XI_STD_CA": [round(self.SN_DICT["CA"]["XI_STD"], 4)],
                "XI_AVG_CH": [round(self.SN_DICT["CH"]["XI_AVG"], 4)],
                "XI_STD_CH": [round(self.SN_DICT["CH"]["XI_STD"], 4)],
                "XI_AVG_C2": [round(self.SN_DICT["C2"]["XI_AVG"], 4)],
                "XI_STD_C2": [round(self.SN_DICT["C2"]["XI_STD"], 4)],
            }
        )
        if self.INPUT_CARBON_MODE == "CH+C2":
            sn_c2_output = pd.DataFrame(
                {
                    "XI_C2": [round(self.SN_DICT["C2"]["XI_AVG"], 4)],
                    "XI_C2_ERR": [round(self.SN_DICT["C2"]["XI_STD"], 4)],
                }
            )
            sn_output = pd.concat([sn_output, sn_c2_output], axis=1)
        return sn_output

    def set_params(
        self,
        SEQUENCE: str,
        STARNAME: str,
        CLASS: str,
        JK: float,
        MODE: str,
        INPUT_CARBON_MODE: str,
        iter: int,
        T_SIGMA: float,
        HARD_TEFF: float,
    ) -> None:
        """
        Set the core stellar parameters and classification attributes for the spectrum.

        Parameters
        ----------
        SEQUENCE : str
            Unique identifier for the spectrum in the sequence list.
        STARNAME : str
            Identifier or name of the observed star.
        CLASS : str
            Stellar gravity class. Must be either "GIANT" or "DWARF".
        JK : float
            J - K color index of the star.
        MODE : str
            Galactic environment mode. Must be "UFD" (ultra-faint dwarf) or "HALO".
        INPUT_CARBON_MODE : str
            Initial carbon mode selection, e.g., "CH" or "CH+C2".
        iter : int
            Number of MCMC iterations to be run for fitting.
        T_SIGMA : float
            Gaussian prior sigma on Teff.
        HARD_TEFF : float
            Fixed Teff value if used as a hard prior.

        Raises
        ------
        AssertionError
            If CLASS is not "GIANT" or "DWARF", or if MODE is not "UFD" or "HALO".
        """
        self.SEQUENCE = str(SEQUENCE)
        self.STARNAME = str(STARNAME)
        self.G_CLASS = str(CLASS)
        self.JK = JK
        self.MODE = str(MODE)
        self.INPUT_CARBON_MODE = str(INPUT_CARBON_MODE)
        self.MCMC_iterations = iter
        self.T_SIGMA = float(T_SIGMA)
        self.HARD_TEFF = float(HARD_TEFF)
        assert (self.G_CLASS == "GIANT") or (self.G_CLASS == "DWARF"), "Invalid gravity class: {}".format(self.G_CLASS)
        assert (self.MODE == "UFD") or (self.MODE == "HALO"), "Invalid Galactic Environment"
        return

    def set_KP_bounds(self, input_bounds: list[float]) -> None:
        """
        Set the KP (Ca II K line) equivalent width integration bounds.

        Parameters
        ----------
        input_bounds : list of float
            A list specifying the wavelength bounds to use for KP index integration.
        """
        self.KP_bounds = input_bounds
        return

    def set_carbon_mode(self, carbon_mode: str) -> None:
        """
        Set the carbon mode used for the spectrum analysis.

        Parameters
        ----------
        carbon_mode : str
            The carbon mode to be applied (e.g., "CH" or "CH+C2").
        """
        self.carbon_mode = carbon_mode
        return

    def set_group_ll(self, input_dict: Dict[str, Tuple[float, Any]]) -> None:
        """
        Set the log-likelihood values and assign the most probable stellar group.

        This function updates the internal LL dictionary, selects the group with the
        highest log-likelihood score, prints the result, and assigns the group label
        (GI, GII, or GIII) to the spectrum.

        Parameters
        ----------
        input_dict : dict
            Dictionary mapping group names (e.g., "GI", "GII", "GIII") to tuples
            containing log-likelihood values and associated data.
        """
        self.LL_DICT = input_dict

        LLs = [self.LL_DICT[key][0] for key in self.LL_DICT.keys()]

        GROUP = ["GI", "GII", "GIII"][LLs.index(max(LLs))]

        print("\t " + self.get_filename().ljust(20) + ": ", GROUP, ["%.2F" % val for val in LLs])

        self.ARCH_GROUP = GROUP

        return

    def set_temperature(self, input_temp: float, sigma: float) -> None:
        """
        Set the effective temperature and its uncertainty.

        This method assigns the input effective temperature and its corresponding
        uncertainty to the spectrum object.

        Parameters
        ----------
        input_temp : float
            The effective temperature value to be set.

        sigma : float
            The uncertainty (standard deviation) associated with the effective temperature.
        """
        print(f"\t\t batch.set_temperatue(): temp={input_temp}, sigma={sigma}")
        self.teff_irfm = input_temp
        self.teff_irfm_err = sigma

        return

    def prepare_regions(self) -> None:
        """
        Prepare wavelength regions for spectral analysis.

        This method segments the spectrum into specific wavelength regions for calcium (CA),
        CH, and optionally C2 molecular bands based on the `carbon_mode`.

        Regions are:
        - CA: Defined by `self.KP_bounds`
        - CH: 4222 angstrom to 4322 angstrom
        - C2: 4710 angstrom to 4750 angstrom (only if carbon_mode is "CH+C2")

        The resulting regions are stored in `self.regions` as a dictionary of DataFrames.
        """
        self.regions = {
            "CA": self.frame[self.frame["wave"].between(*self.KP_bounds, inclusive="both")].copy(),
            "CH": self.frame[self.frame["wave"].between(4222, 4322, inclusive="both")].copy(),
        }

        if self.carbon_mode == "CH+C2":
            self.regions["C2"] = self.frame[self.frame["wave"].between(4710, 4750, inclusive="both").copy()]

        return

    def set_temp_frame(self, TEMP_FRAME: pd.DataFrame) -> None:
        """
        Set the temperature frame for the spectrum.

        This method assigns the given DataFrame of temperature estimates
        to the object's TEMP_FRAME attribute.

        Parameters
        ----------
        TEMP_FRAME : pandas.DataFrame
            A DataFrame containing temperature values from various calibration methods.
        """
        self.TEMP_FRAME = TEMP_FRAME
        return

    def set_mcmc_args(self, input_dict: dict | None = None) -> None:
        """
        Set the arguments used for the Markov Chain Monte Carlo (MCMC) procedure.

        This method stores the MCMC-related arguments into the object's mcmc_args attribute.

        Parameters
        ----------
        input_dict : dict or None, optional
            A dictionary of MCMC arguments. If None, an empty dictionary is assigned.
        """
        if input_dict is not None:
            self.mcmc_args = input
        else:
            self.mcmc_args = {}

        return

    def set_mcmc_results(self, input_dict: dict, mode: str) -> None:
        """
        Set the results from the Markov Chain Monte Carlo (MCMC) runs.

        Depending on the mode provided, stores the MCMC output into either the COARSE or REFINE attribute.

        Parameters
        ----------
        input_dict : dict
            Dictionary containing MCMC results.

        mode : str
            Specifies which stage the results correspond to. Accepts:
            - "COARSE": Stores results in self.MCMC_COARSE
            - "REFINE": Stores results in self.MCMC_REFINE

        Raises
        ------
        Prints a warning if the mode is not recognized.
        """
        if mode == "COARSE":
            self.MCMC_COARSE = input_dict

        elif mode == "REFINE":
            self.MCMC_REFINE = input_dict

        else:
            print("Invalid mode in set_mcmc_results()")

        return

    def set_sampler(self, input_sampler: object, mode: str = "COARSE") -> None:
        """
        Set the MCMC sampler object for a given sampling mode.

        Parameters
        ----------
        input_sampler : object
            The sampler instance containing the MCMC chain and metadata.

        mode : str, optional
            Indicates which sampler to store. Accepts:
            - "COARSE": Stores in self.MCMC_COARSE_sampler
            - "REFINE": Stores in self.MCMC_REFINE_sampler
            Default is "COARSE".
        """
        if mode == "COARSE":
            self.MCMC_COARSE_sampler = input_sampler
        elif mode == "REFINE":
            self.MCMC_REFINE_sampler = input_sampler
        return

    def set_kde_functions(self, input_dict: dict, mode: str) -> None:
        """
        Set the kernel density estimation (KDE) results from MCMC output.

        Parameters
        ----------
        input_dict : dict
            A dictionary containing KDE functions or results for each parameter.
        mode : str
            Indicates which MCMC stage the KDE results belong to.
            Accepted values:
            - "COARSE": store results in self.KDE_COARSE
            - "REFINE": store results in self.KDE_REFINE

        Raises
        ------
        ValueError
            If the provided mode is not "COARSE" or "REFINE".
        """
        if mode == "COARSE":
            self.KDE_COARSE = input_dict

        elif mode == "REFINE":
            self.KDE_REFINE = input_dict

        else:
            print("Invalid mode in set_mcmc_results()")

        return

    def set_flux(self, input_flux: np.ndarray) -> None:
        """
        Set the flux array for the spectrum.

        Parameters
        ----------
        input_flux : np.ndarray
            Array of flux values corresponding to the spectrum's wavelengths.

        Returns
        -------
        None
        """
        self.flux = input_flux
        return

    def set_norm(self, input_flux: np.ndarray) -> None:
        """
        Set the normalized flux array for the spectrum.

        Parameters
        ----------
        input_flux : np.ndarray
            Array of normalized flux values.

        Returns
        -------
        None
        """
        self.norm = input_flux

        return

    def set_GBAND(self, input: float) -> None:
        """
        Set the G-band equivalent width (EW) measurement.

        Parameters
        ----------
        input : float
            The measured G-band equivalent width.

        Returns
        -------
        None
        """
        self.GBAND_EW = input
        return

    def set_frame(self, wave: np.ndarray, flux: np.ndarray) -> None:
        """
        Set the spectral data frame with wavelength and flux.

        Parameters
        ----------
        wave : np.ndarray
            Array of wavelength values.
        flux : np.ndarray
            Array of corresponding flux values.

        Returns
        -------
        None
        """
        self.frame = pd.DataFrame({"wave": wave, "flux": flux})
        return

    def set_frame_norm(self, input_norm: np.ndarray) -> None:
        """
        Add or update the 'norm' column in the spectral data frame.

        Parameters
        ----------
        input_norm : np.ndarray
            Array of normalized flux values to assign to the 'norm' column.

        Returns
        -------
        None
        """
        self.frame.loc[:, "norm"] = input_norm
        return

    def set_frame_cont(self, input_cont: np.ndarray) -> None:
        """
        Add or update the 'cont' column in the spectral data frame.

        Parameters
        ----------
        input_cont : np.ndarray
            Array of continuum flux values to assign to the 'cont' column.

        Returns
        -------
        None
        """
        self.frame.loc[:, "cont"] = input_cont
        return

    def set_synth_spectrum(self, synth: dict) -> None:
        """
        Set the synthetic spectrum for the object.

        Parameters
        ----------
        synth : dict
            Dictionary containing the synthetic spectrum data,
            typically with keys like "wave" and "norm" for wavelength
            and normalized flux arrays.

        Returns
        -------
        None
        """

        self.synth_spectrum = synth
        return

    def get_sequence(self) -> str:
        """
        Retrieve the object's sequence identifier.

        Returns
        -------
        str
            Sequence identifier as a string.
        """
        return "{:s}".format(self.SEQUENCE)

    def get_filename(self) -> str:
        """
        Retrieve the object's filename, padded for formatting.

        Returns
        -------
        str
            The filename as a left-aligned string with a width of 20 characters.
        """

        return "{:<20}".format(self.filename)

    def get_starname(self) -> str:
        """
        Retrieve the object's star name, padded for formatting.

        Returns
        -------
        str
            The star name as a left-aligned string with a width of 25 characters.
        """

        return "{:<25}".format(self.STARNAME)

    def get_wave(self) -> np.ndarray:
        """
        Get the current wavelength array of the spectrum.

        Returns
        -------
        np.ndarray
            Array of wavelength values.
        """

        return self.wavelength

    def get_flux(self) -> np.ndarray:
        """
        Get the observed flux array of the spectrum.

        Returns
        -------
        np.ndarray
            Array of observed flux values.
        """
        return self.flux

    def get_frame(self) -> pd.DataFrame:
        """
        Get the DataFrame containing the wavelength and flux data used for CASPER analysis.

        Returns
        -------
        pd.DataFrame
            DataFrame with "wave" and "flux" (and potentially other) columns.
        """

        return self.frame

    def get_frame_wave(self) -> pd.Series:
        """
        Get the wavelength values from the DataFrame used in CASPER analysis.

        Returns
        -------
        pd.Series
            Series containing the wavelength values from the current frame.
        """

        return self.frame["wave"]

    def get_frame_flux(self) -> pd.Series:
        """
        Get the flux values from the DataFrame used in CASPER analysis.

        Returns
        -------
        pd.Series
            Series containing the flux values from the current frame.
        """

        return self.frame["flux"]

    def get_frame_norm(self) -> pd.Series:
        """
        Get the normalized flux values from the DataFrame used in CASPER analysis.

        Returns
        -------
        pd.Series
            Series containing the normalized flux values from the frame.
        """

        return self.norm["norm"]

    def get_gravity_class(self) -> str:
        """
        Get the gravity classification of the star.

        Returns
        -------
        str
            Gravity class label (e.g., "GIANT" or "DWARF").
        """

        return self.G_CLASS

    def get_logg(self) -> tuple[float, float]:
        """
        Get the surface gravity (logg) and its associated error.

        Returns
        -------
        tuple[float, float]
            A tuple containing:
            - logg : float
                Estimated surface gravity.
            - logg_err : float
                Uncertainty in the surface gravity estimate.
        """

        return self.logg.item(), self.logg_err

    def get_carbon_mode(self) -> str:
        """
        Get the carbon analysis mode assigned to the spectrum.

        Returns
        -------
        str
            The carbon mode used (e.g., "CH" or "CH+C2").
        """

        return self.carbon_mode

    def get_KP_bounds(self) -> tuple[float, float]:
        """
        Get the wavelength bounds used for the Ca II K line analysis (KP).

        Returns
        -------
        tuple[float, float]
            A tuple containing the lower and upper bounds of the KP region in angstroms.
        """
        return self.KP_bounds

    def get_environ_mode(self) -> str:
        """
        Get the galactic environment classification mode.

        Returns
        -------
        str
            The galactic environment mode, either "UFD" (ultra-faint dwarf) or "HALO".
        """

        return self.MODE

    def get_arch_group(self) -> str:
        """
        Get the archival group classification assigned to the star.

        Returns
        -------
        str
            The group classification (e.g., "GI", "GII", or "GIII") based on likelihood analysis.
        """

        return self.ARCH_GROUP

    def get_photo_temp(self) -> tuple[float, float]:
        """
        Get the photometric effective temperature and its associated uncertainty.

        Returns
        -------
        tuple[float, float]
            A tuple containing:
            - teff_irfm (float): The estimated photometric temperature.
            - teff_irfm_err (float): The uncertainty associated with the temperature estimate.
        """

        return self.teff_irfm, self.teff_irfm_err

    def get_rv(self) -> float:
        """
        Get the radial velocity used for wavelength correction.

        Returns
        -------
        float
            The radial velocity in km/s.
        """

        return self.rv

    def get_SN_dict(self) -> dict:
        """
        Get the signal-to-noise (S/N) dictionary computed for spectral sidebands.

        Returns
        -------
        dict
            A dictionary containing S/N statistics for each spectral region (e.g., 'CA', 'CH', 'C2'),
            including average S/N, standard deviation, average XI, XI standard deviation,
            and related parameters.
        """

        return self.SN_DICT

    def get_kde_dict(self) -> dict | tuple[dict, dict]:
        """
        Retrieve the kernel density estimation (KDE) results used in MCMC analysis.

        Returns
        -------
        dict or tuple of dicts
            If only coarse KDE is available, returns the coarse KDE dictionary.
            If both coarse and refined KDEs are present, returns a tuple of (KDE_COARSE, KDE_REFINE).
        """

        if hasattr(self, "KDE_REFINE"):
            return self.KDE_COARSE, self.KDE_REFINE

        return self.KDE_COARSE

    def get_mcmc_dict(self, mode: str = "COARSE") -> dict | tuple[dict, dict] | float:
        """
        Retrieve the MCMC results dictionary for the specified mode.

        Parameters
        ----------
        mode : str, optional
            The mode of MCMC results to retrieve. Options are:
            - "COARSE": Return only the coarse MCMC results.
            - "REFINE": Return only the refined MCMC results.
            - "BOTH": Return a tuple of (coarse, refined) MCMC results.
            Default is "COARSE".

        Returns
        -------
        dict or tuple of dicts or float
            The requested MCMC results dictionary, a tuple of both, or np.nan if the mode is invalid.
        """

        if mode == "COARSE":
            return self.MCMC_COARSE

        elif mode == "REFINE":
            return self.MCMC_REFINE

        elif mode == "BOTH":
            return self.MCMC_COARSE, self.MCMC_REFINE

        else:
            print("Bad mode:  ", mode)
            return np.nan

    def get_MCMC_iterations(self) -> int:
        """
        Retrieve the number of MCMC iterations.

        Returns
        -------
        int
            The number of iterations used in the MCMC process.
        """

        return self.MCMC_iterations

    def get_spectra_row(self) -> pd.DataFrame:
        """
        Retrieve observed and synthetic spectra as a single-row DataFrame.

        This method returns the observed wavelength and normalized flux, along with
        the corresponding synthetic wavelength and normalized flux, all stored as lists
        within a single-row DataFrame.

        Returns
        -------
        pd.DataFrame
            A DataFrame with one row and four columns:
            - "OBSERVED_WAVE": list of observed wavelengths
            - "OBSERVED_FLUX": list of observed normalized flux values
            - "SYNTH_WAVE": list of synthetic wavelengths
            - "SYNTHETIC_FLUX": list of synthetic normalized flux values
        """

        output_spectra = pd.DataFrame(
            {
                "OBSERVED_WAVE": [self.frame["wave"].values.tolist()],
                "OBSERVED_FLUX": [self.frame["norm"].values.tolist()],
                "SYNTH_WAVE": [self.synth_spectrum["wave"].values.tolist()],
                "SYNTHETIC_FLUX": [self.synth_spectrum["norm"].values.tolist()],
            }
        )
        return output_spectra

    def get_output_row(self) -> pd.DataFrame:
        """
        Generate a single-row DataFrame with the final stellar parameter outputs.

        This method consolidates metadata, MCMC-derived parameters, observational data,
        and S/N statistics into a structured output format. Useful for analysis export
        and tracking parameter estimates across runs.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing one row with:
            - Star identifiers, gravity and carbon classification
            - Effective temperature (from IRFM and MCMC) and uncertainties
            - Surface gravity, [Fe/H], [C/Fe], and A(C) with uncertainties
            - Radial velocity, observed and synthetic spectra
            - Signal-to-noise statistics for CA, CH, and optionally C2 regions
        """

        output = pd.DataFrame(
            {
                "SEQUENCE": [self.get_sequence()],
                "FILENAME": [self.get_filename()],
                "STARNAME": [self.get_starname()],
                "ENV_MODE": [self.get_environ_mode()],
                "CAR_MODE": [self.get_carbon_mode()],
                "CEMP_GRP_TENT": [self.get_arch_group()],
                "GCLASS": [self.get_gravity_class()],
                "TEFF": [round(self.MCMC_COARSE["TEFF"][0], 0)],
                "TEFF_ERR": [round(self.MCMC_COARSE["TEFF"][1], 0)],
                "TEFF_ADT": [round(self.teff_irfm)],
                "TEFF_ADT_ERR": [self.teff_irfm_err],
                "LOGG": [round(self.logg.item(), 2)],
                "LOGG_ERR": [round(self.logg_err, 2)],
                "FEH": [round(self.MCMC_REFINE["FEH"][0], 2)],
                "FEH_ERR": [round(max([self.MCMC_REFINE["FEH"][1], self.MCMC_COARSE["FEH"][1]]), 2)],
                "CFE": [round(self.MCMC_REFINE["CFE"][0], 2)],
                "CFE_ERR": [round(max([self.MCMC_REFINE["CFE"][1], self.MCMC_COARSE["CFE"][1]]), 2)],
                "AC": [round(self.MCMC_REFINE["AC"][0], 2)],
                "AC_ERR": [round(max([self.MCMC_REFINE["AC"][1], self.MCMC_COARSE["AC"][1]]), 2)],
                "RV": [round(self.get_rv(), 1)],
                "OBSERVED_WAVE": [self.frame["wave"].values.tolist()],
                "OBSERVED_FLUX": [self.frame["norm"].values.tolist()],
                "SYNTH_WAVE": [self.synth_spectrum["wave"].values.tolist()],
                "SYNTHETIC_FLUX": [self.synth_spectrum["norm"].values.tolist()],
                "XI_CA": [round(self.SN_DICT["CA"]["XI_AVG"], 4)],
                "XI_CA_ERR": [round(self.SN_DICT["CA"]["XI_STD"], 4)],
                "XI_CH": [round(self.SN_DICT["CH"]["XI_AVG"], 4)],
                "XI_CH_ERR": [round(self.SN_DICT["CH"]["XI_STD"], 4)],
            }
        )
        if self.INPUT_CARBON_MODE == "CH+C2":
            c2_output = pd.DataFrame(
                {
                    "XI_C2": [round(self.SN_DICT["C2"]["XI_AVG"], 4)],
                    "XI_C2_ERR": [round(self.SN_DICT["C2"]["XI_STD"], 4)],
                }
            )
            output = pd.concat([output, c2_output], axis=1)

        sn_output = pd.DataFrame(
            {
                "SN_AVG_CA": [round(self.SN_DICT["CA"]["SN_AVG"], 0)],
                "SN_AVG_CH": [round(self.SN_DICT["CH"]["SN_AVG"], 0)],
            }
        )
        output = pd.concat([output, sn_output], axis=1)

        return output

    def print_KP_bounds(self) -> str:
        """
        Return the Ca II K region wavelength bounds as a formatted string.

        Returns
        -------
        str
            A string representation of the Ca II K region bounds, formatted as
            'lower_bound - upper_bound'.
        """

        return str(self.KP_bounds[0]) + " - " + str(self.KP_bounds[1])
