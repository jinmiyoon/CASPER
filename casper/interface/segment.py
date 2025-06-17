import numpy as np
from numpy.typing import ArrayLike


class Segment:
    def __init__(self, wl: ArrayLike = None, flux: ArrayLike = None):
        """
        Initialize a Segment object with wavelength and flux values.

        Parameters
        ----------
        wl : array_like
            Wavelength values that define the x-axis of the segment.
        flux : array_like
            Flux values corresponding to each wavelength in `wl`, defining the y-axis
            of the segment.

        Attributes
        ----------
        midpoint : float
            The median value of the wavelength array, used to represent the segment's center.
        """
        self.wl = np.array([] if wl is None else wl)
        self.flux = np.array([] if flux is None else flux)
        self.midpoint = np.median(self.wl)

    def is_edge(self, which: str) -> None:
        """
        Set the midpoint of the segment to the edge wavelength.

        This method updates `self.midpoint` to either the first or last
        wavelength in the segment, depending on the specified edge.

        Parameters
        ----------
        which : str
            The edge identifier. Must be one of:
            - "left": set midpoint to the first wavelength
            - "right": set midpoint to the last wavelength

        Raises
        ------
        ValueError
            If `which` is not "left" or "right".
        """

        if which == "left":
            self.midpoint = np.array(self.wl)[0]
        elif which == "right":
            self.midpoint = np.array(self.wl)[-1]
        else:
            print("Error in edge definition")

    def get_statistics(self, lower: float = 85) -> None:
        """
        Compute robust statistics to characterize flux variability and continuum level.

        This method calculates statistical metrics that help define the continuum level
        and assess absorption strength in a spectrum segment. It uses median-based and
        percentile-based techniques to ensure robustness against outliers.

        Parameters
        ----------
        lower : float, optional
            Lower percentile cutoff for clipping and estimating the minimum flux.
            Defaults to 85. A higher value reduces the influence of absorption features.

        Attributes
        ----------
        mad : float
            Median Absolute Deviation (MAD) of the flux values. Serves as a robust measure
            of flux variability.
        flux_med : float
            Median flux value computed from the clipped flux (between `lower` and 98th percentiles).
            Used as a continuum level estimate.
        flux_min : float
            The `lower`-th percentile of the flux distribution. Represents a robust minimum.
        flux_max : float
            The 98th percentile of the flux distribution. Represents a robust maximum.

        Notes
        -----
        A large `mad` value suggests significant absorption features within the segment.
        """
        self.mad = np.median(np.absolute(self.flux - np.median(self.flux)))
        self.flux_med = np.median(
            self.flux[
                np.where((self.flux >= np.percentile(self.flux, lower)) & (self.flux <= np.percentile(self.flux, 98)))
            ]
        )
        self.flux_min = np.percentile(self.flux, lower)
        self.flux_max = np.percentile(self.flux, 98)

    def define_cont_point(self, mad_min: float, mad_range: float) -> None:
        """
        Define a continuum point based on flux variation (MAD normalization).

        This method estimates the continuum point for the segment using a MAD-normalized
        interpolation between the median and the upper flux percentile. The assumption is
        that greater variation in flux (higher MAD) implies stronger absorption features,
        so the continuum estimate should shift closer to the maximum flux.

        Parameters
        ----------
        mad_min : float
            The minimum MAD value across all segments, used to normalize MAD.
        mad_range : float
            The range of MAD values across all segments (max - min), used for scaling.

        Attributes
        ----------
        mad_normal : float
            Normalized MAD value for the current segment, scaled between 0 and 1.
        continuum_point : float
            Estimated continuum flux value for the segment, biased toward the upper
            flux percentile when MAD is high.

        Notes
        -----
        When the flux shows high variability, the continuum is more likely to be near
        the top of the flux range, since absorption features pull the median downward.
        """
        self.mad_normal = (self.mad - mad_min) / mad_range
        self.continuum_point = (self.flux_max - self.flux_med) * self.mad_normal + self.flux_med
