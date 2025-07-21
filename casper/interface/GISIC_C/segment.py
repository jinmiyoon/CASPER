import numpy as np


class Segment:
    def __init__(self, wl: list = [], flux: list = []):
        """
        Initialize a Segment object with wavelength and flux arrays.

        Parameters
        ----------
        wl : list, optional
            List of wavelength values. Defines the x-axis of the segment.
        flux : list, optional
            List of corresponding flux values for each wavelength. Defines the y-axis of the segment.

        Sets
        ----
        self.wl : list
            The wavelength values.
        self.flux : list
            The flux values.
        self.midpoint : float
            The median wavelength value, used as the segment’s midpoint.
        """

        self.wl = wl
        self.flux = flux
        self.midpoint = np.median(self.wl)

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
            print("Error in edge definition")

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
