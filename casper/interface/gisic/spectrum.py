from typing import List, Tuple

import numpy as np
import pandas as pd
import scipy.interpolate as interp
from astropy.table import Table
from scipy.ndimage.filters import gaussian_filter

from casper.interface.gisic import norm_functions
from casper.interface.gisic.segment import Segment
from casper.utils.logger_config import setup_logger

logger = setup_logger(__name__)


class Spectrum:
    def __init__(self, wavelength: np.ndarray, flux: np.ndarray) -> None:
        """
        Initialize a Spectrum object with wavelength and flux data.

        Parameters
        ----------
        wavelength : np.ndarray
            The wavelength values of the spectrum.

        flux : np.ndarray
            The flux values corresponding to each wavelength.

        Sets
        ----
        self.wavelength : np.ndarray
            Input wavelength array.

        self.flux : np.ndarray
            Input flux array.

        self.segments : Optional[List]
            Will store segmentation data for continuum fitting (initialized to None).

        self.mad_global : Optional[float]
            Global MAD (Median Absolute Deviation) for the entire spectrum (initialized to None).
        """

        self.wavelength = wavelength

        self.flux = flux

        self.segments = None
        self.mad_global = None

        return

    def generate_segments(self, bins: int = 25, lower: int = 70) -> None:
        """
        Divide the spectrum into segments and compute statistics for each.

        Parameters
        ----------
        bins : int, optional
            Number of equal-sized segments to divide the spectrum into. Default is 25.

        lower : int, optional
            Lower percentile threshold for flux clipping during statistics calculation. Default is 70.

        Sets
        ----
        self.segments : List[Segment]
            A list of Segment objects representing subdivided portions of the spectrum.
            The first and last segments are marked as edges.

        Each segment will also compute its own local statistics (e.g., MAD, median flux).
        """

        self.segments = [
            Segment(wl, flux)
            for wl, flux in zip(np.array_split(self.wavelength, bins), np.array_split(self.flux, bins))
        ]

        self.segments[0].is_edge("left")
        self.segments[-1].is_edge("right")

        [segment.get_statistics(lower) for segment in self.segments]
        return

    def generate_inflection_segments(
        self,
        sigma: int = 30,
        width: int = 30,
        cahk: bool = False,
        cahkwidth: int = 2,
        band_check: bool = False,
        flux_min: float = 70,
    ) -> None:
        """
        Identify inflection segments in the spectrum based on second derivative zero crossings.

        Parameters
        ----------
        sigma : int, optional
            Gaussian smoothing kernel width for the flux array. Default is 30.
        width : int, optional
            Width of each segment around identified inflection points. Default is 30.
        cahk : bool, optional
            If True, forces inflection segments at Ca H (3916Å) and Ca K (3991Å) lines. Default is False.
        cahkwidth : int, optional
            Width around the Ca H&K lines if `cahk` is True. Default is 2.
        band_check : bool, optional
            Whether to skip segments inside known molecular bands. Default is False.
        flux_min : float, optional
            Lower percentile threshold used during segment statistics estimation. Default is 70.

        Sets
        ----
        self.segments : List[Segment]
            List of spectral segments created around inflection points.
        self.frame : pd.DataFrame
            Intermediate DataFrame containing smoothed flux and its derivatives.
        self.ZEROS : List[float]
            List of wavelength values where the second derivative crosses zero.
        """

        # Smooth the flux to reduce high-frequency noise before taking derivatives.
        self.smooth = gaussian_filter(self.flux, sigma=sigma)

        # Compute the first and second derivatives; normalize them so the signs are stable.
        self.d1 = np.gradient(self.smooth)
        self.d2 = np.gradient(self.d1)
        self.d1 = self.d1 / max(self.d1)
        self.d2 = self.d2 / max(self.d2)

        # Assemble the working DataFrame: wavelength, flux, and derivative channels.
        hack = Table(
            [self.wavelength, self.flux, self.d1, self.d2],
            names=("wave", "flux", "d1", "d2"),
            dtype=("f8", "f8", "f8", "f8"),
        )

        self.frame = pd.DataFrame({"wave": hack["wave"], "flux": hack["flux"], "d1": hack["d1"], "d2": hack["d2"]})

        # Identify zero-crossings in the second derivative; these are candidate inflection points.
        self.ZEROS = []
        for i in range(len(self.d2) - 1):
            if self.d2[i] * self.d2[i + 1] < 0.0:
                self.ZEROS.append((self.frame.wave[i] + self.frame.wave[i + 1]) / 2.0)
            elif self.d2[i] == 0.0:
                self.ZEROS.append(self.frame.wave[i])

        # For each interval between adjacent zero crossings, keep the strongest local minimum
        # if the second derivative is predominantly negative across the interval.
        MINIMUMS = []

        for i in range(len(self.ZEROS) - 1):
            SEGMENT = self.frame[self.frame["wave"].between(self.ZEROS[i], self.ZEROS[i + 1], inclusive="both")].copy()

            if len(SEGMENT[SEGMENT["d2"] < 0.0]) > 0.8 * len(SEGMENT["d2"]):
                MIN = SEGMENT[SEGMENT["d2"] == min(SEGMENT["d2"])].copy()
                MIN.loc[:, "size"] = len(SEGMENT)
                MINIMUMS.append(MIN)

        # If requested, force the Ca H and Ca K absorption features into the set of extrema.
        # This is a domain-specific override beyond the generic zero-crossing minima.
        if cahk:
            SEG1 = self.frame[self.frame["wave"].between(3916 - cahkwidth, 3916 + cahkwidth, inclusive="both")]
            SEG2 = self.frame[self.frame["wave"].between(3991 - cahkwidth, 3991 + cahkwidth, inclusive="both")]

            SEG1 = SEG1[SEG1["flux"] == max(SEG1["flux"])].copy()
            SEG2 = SEG2[SEG2["flux"] == max(SEG2["flux"])].copy()

            SEG1.loc[:, "size"] = len(SEG1)
            SEG2.loc[:, "size"] = len(SEG2)
            MIN_CAT = pd.concat(MINIMUMS)
            EXTREMA = pd.concat([MIN_CAT, SEG1, SEG2])
            EXTREMA = EXTREMA.sort_values(by="wave")
            EXTREMA = EXTREMA.iloc[np.unique(EXTREMA["wave"], return_index=True)[1]]

        else:
            EXTREMA = pd.concat(MINIMUMS)

        # Build one Segment for each selected extremum; optionally omit points in known molecular bands.
        self.segments = []
        if band_check:
            for i, row in EXTREMA.iterrows():
                if not norm_functions.in_molecular_band(row["wave"], tol=10):
                    SEGMENT = self.frame[
                        self.frame["wave"].between(
                            row["wave"] - int(width / 2), row["wave"] + int(width / 2), inclusive="both"
                        )
                    ].copy()
                    self.segments.append(Segment(np.array(SEGMENT["wave"]), np.array(SEGMENT["flux"])))
                else:
                    # Skip this extremum because it falls in a known molecular band.
                    pass

        else:
            for i, row in EXTREMA.iterrows():
                SEGMENT = self.frame[
                    self.frame["wave"].between(
                        row["wave"] - int(width / 2), row["wave"] + int(width / 2), inclusive="both"
                    )
                ].copy()
                self.segments.append(Segment(np.array(SEGMENT["wave"]), np.array(SEGMENT["flux"])))

        # Keep the leftmost and rightmost chunks as edge anchors, so the continuum model has fixed endpoints.
        self.segments.insert(
            0, Segment(np.array(self.frame["wave"].iloc[0:width]), np.array(self.frame["flux"].iloc[0:width]))
        )
        self.segments.append(
            Segment(np.array(self.frame["wave"].iloc[-width:]), np.array(self.frame["flux"].iloc[-width:]))
        )

        self.segments[0].is_edge("left")
        self.segments[-1].is_edge("right")

        [segment.get_statistics(flux_min=flux_min) for segment in self.segments]

        return

    def assess_segment_variation(self) -> None:
        """
        Compute variation metrics for each segment based on normalized MAD (Median Absolute Deviation).

        This method evaluates the variability of flux within each segment by collecting normalized
        MAD values. It then calculates global statistics across all segments and derives a
        scaled relative MAD array used for weighting or further processing.

        Sets
        ----
        self.mad_array : np.ndarray
            Normalized MAD values for all segments.

        self.mad_global : float
            Global median of the segment MAD values.

        self.mad_min : float
            Minimum normalized MAD value across segments.

        self.mad_max : float
            Maximum normalized MAD value across segments.

        self.mad_range : float
            Difference between mad_max and mad_min.

        self.mad_relative_array : np.ndarray
            Normalized MAD values scaled between 0 and 1.
        """

        self.mad_array = np.array([segment.mad_normal for segment in self.segments], dtype=float)

        self.mad_global = np.median(self.mad_array)

        self.mad_min, self.mad_max = min(self.mad_array), max(self.mad_array)

        self.mad_range = self.mad_max - self.mad_min

        self.mad_relative_array = np.divide(self.mad_array - self.mad_min, self.mad_range)

        return

    def define_cont_points(self, boost: float) -> None:
        """
        Define continuum points for all segments using a boosted MAD-normalized strategy.

        This method loops over all Segment objects in `self.segments` and calls their
        `define_cont_point()` method. The boost value scales the clipped median flux
        based on the MAD distribution across segments.

        Parameters
        ----------
        boost : float
            A scaling factor applied to adjust the continuum point above the median
            flux, based on signal variation.

        Returns
        -------
        None

        Notes
        -----
        `self.assess_segment_variation()` must be called beforehand to ensure
        `self.mad_min` and `self.mad_range` are initialized.

        """
        [segment.define_cont_point(self.mad_min, self.mad_range, boost=boost) for segment in self.segments]

    def set_segment_midpoints(self) -> np.ndarray:
        """
        Collect and store the midpoint wavelength of each segment.

        This method iterates over all spectral segments and extracts their midpoint values,
        storing them in `self.midpoints`.

        Returns
        -------
        np.ndarray
            Array of midpoint wavelength values for all segments.
        """

        self.midpoints = [segment.midpoint for segment in self.segments]

        return np.array(self.midpoints, dtype=float)

    def set_segment_continuum(self) -> np.ndarray:
        """
        Collect and store the continuum flux point of each segment.

        This method gathers the estimated continuum flux points from each spectral segment
        and stores them in `self.fluxpoints`.

        Returns
        -------
        np.ndarray
            Array of continuum flux values for all segments.
        """

        self.fluxpoints = [segment.continuum_point for segment in self.segments]

        return np.array(self.fluxpoints, dtype=float)

    def add_continuum_point(self, point: Tuple[float, float]) -> None:
        """
        Add a new continuum anchor point and sort existing points by wavelength.

        Parameters
        ----------
        point : tuple of float
            A tuple containing (wavelength, flux) to be added to the continuum points.

        Notes
        -----
        After adding the new point, both `self.midpoints` and `self.fluxpoints` are
        sorted based on the wavelength values in `self.midpoints`.
        """

        self.midpoints.append(point[0])
        self.fluxpoints.append(point[1])

        self.fluxpoints = list(np.array(self.fluxpoints)[np.argsort(self.midpoints)])
        self.midpoints = list(np.array(self.midpoints)[np.argsort(self.midpoints)])
        return

    def remove_point(self, index: List[int]) -> None:
        """
        Remove continuum anchor points at the specified indices.

        Parameters
        ----------
        index : list of int
            A list of indices indicating which points to remove from `midpoints` and `fluxpoints`.

        Notes
        -----
        Indices are assumed to be sorted in increasing order. To avoid indexing errors
        due to shifting during deletion, removal is done in reverse order.
        """

        for i, value in enumerate(index):
            del self.midpoints[value - i]
            del self.fluxpoints[value - i]
        return

    def get_continuum_points(self) -> None:
        """
        Print all continuum anchor points.

        Prints the index, midpoint wavelength, and corresponding flux value
        for each continuum anchor point stored in `self.midpoints` and `self.fluxpoints`.

        Returns
        -------
        None
        """

        for i in range(len(self.midpoints)):
            logger.info(f"{i}: {self.midpoints[i]} {self.fluxpoints[i]}")

    def set_wavelength(self, wavelength: list[float]) -> None:
        """
        Set the list of continuum midpoint wavelengths.

        Parameters
        ----------
        wavelength : list of float
            List of wavelength values representing the continuum midpoints.

        Returns
        -------
        None
        """

        self.midpoints = wavelength

        return

    def set_fluxpoints(self, flux_values: list[float]) -> None:
        """
        Set the list of flux values at continuum midpoints.

        Parameters
        ----------
        flux_values : list of float
            List of flux values corresponding to each continuum midpoint.

        Returns
        -------
        None
        """

        self.fluxpoints = flux_values

        return

    def spline_continuum(self, k: int = 3, s: float = 5.0) -> None:
        """
        Fit a spline to the continuum points and evaluate it across the full wavelength range.

        Parameters
        ----------
        k : int, optional
            Degree of the smoothing spline. Default is 3 (cubic spline).
        s : float, optional
            Smoothing factor. Larger values result in smoother fits. Default is 5.0.

        Returns
        -------
        None
        """

        tck = interp.splrep(self.midpoints, self.fluxpoints, k=k, s=s, quiet=True)

        self.continuum = interp.splev(self.wavelength, tck)

    def normalize(self) -> None:
        """
        Normalize the flux by dividing it by the fitted continuum.

        This method computes the normalized flux (`flux_norm`) by dividing
        the raw flux values by the continuum. Extreme values below 0 or
        above 2 are clipped to 1.0 for stability.

        Returns
        -------
        None
        """

        self.flux_norm = np.divide(self.flux, self.continuum)

        if len(self.flux_norm[self.flux_norm < 0.0]) > 0:
            self.flux_norm[self.flux_norm < 0.0] = 0.0

        if len(self.flux_norm[self.flux_norm > 2.0]) > 0:
            self.flux_norm[self.flux_norm > 2.0] = 1.0
