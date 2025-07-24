import numpy as np
from numpy.typing import ArrayLike

from casper.utils.logger_config import setup_logger

logger = setup_logger(__name__)


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
        self.wl = np.array([]) if wl is None else np.array(wl)
        self.flux = np.array([]) if flux is None else np.array(flux)
        self.midpoint = np.median(self.wl)
        print("wl type:", type(self.wl))
        print("flux type:", type(self.flux))

    def is_edge(self, which: str) -> None:
        """
        Set the midpoint of the segment to either the first or last wavelength value,
        depending on whether the segment is at the start or end of the spectrum.

        Parameters
        ----------
        which : str
            Indicates which edge the segment belongs to.
            Must be one of:
            - "left": sets midpoint to the first wavelength.
            - "right": sets midpoint to the last wavelength.

        Raises
        ------
        Prints an error message if `which` is not "left" or "right".
        """

        if which == "left":
            self.midpoint = np.array(self.wl)[0]

        elif which == "right":
            self.midpoint = np.array(self.wl)[-1]

        else:
            logger.error(f"Invalid value for 'which': {which}. Must be 'left' or 'right'.")
            raise ValueError(f"Invalid value for 'which': {which}. Must be 'left' or 'right'.")

    def get_statistics(self, flux_min: float = 70) -> None:
        """
        Compute robust statistical measures of flux variability for the segment.

        Parameters
        ----------
        flux_min : float, optional
            The lower percentile threshold used for clipping the flux distribution.
            Default is 70.

        Sets
        ----
        self.mad : float
            Median Absolute Deviation (MAD) of the flux values.
        self.mad_normal : float
            Normalized MAD, calculated as MAD divided by the median flux.
        self.flux_med : float
            Median flux value computed from the central clipped range
            [flux_min percentile, 98th percentile].
        self.flux_min : float
            Flux value at the `flux_min` percentile.
        self.flux_max : float
            Flux value at the 98th percentile.
        """

        self.mad = np.median(np.absolute(self.flux - np.median(self.flux)))

        if self.mad != 0:
            self.mad_normal = self.mad / np.median(self.flux)
        else:
            self.mad_normal = 0.0

        self.flux_med = np.median(
            self.flux[
                np.where(
                    (self.flux >= np.percentile(self.flux, flux_min)) & (self.flux <= np.percentile(self.flux, 98))
                )
            ]
        )
        self.flux_min = np.percentile(self.flux, flux_min)
        self.flux_max = np.percentile(self.flux, 98)

    def define_cont_point(self, mad_min: float, mad_range: float, boost: bool = True) -> None:
        """
        Define a continuum point based on the flux variability within the segment.

        Parameters
        ----------
        mad_min : float
            Minimum normalized MAD value used as a baseline.
        mad_range : float
            Range used to scale the influence of flux variation on the continuum point.
        boost : bool, optional
            If True, adjust the continuum point toward the segment's maximum flux based
            on relative variation. If False, use the median flux as the continuum point.

        Sets
        ----
        self.mad_relative : float
            Relative MAD used to determine how strongly to bias the continuum point.
        self.continuum_point : float
            Estimated continuum level for this segment.
        """

        self.mad_relative = (self.mad_normal - mad_min) / mad_range

        if boost:
            self.continuum_point = (self.flux_max - self.flux_med) * self.mad_relative + self.flux_med

        else:
            self.continuum_point = self.flux_med

        return
