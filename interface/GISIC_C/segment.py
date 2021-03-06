### Author: Devin Whitten
### Segment class definition for basic continuum point creation.
import numpy as np




class Segment():
    #### This class will store relavent statistics about the wavelength region,
    #### and determine the best approximation of the continuum point for iterative spline interpolation

    #### Might need to think about how to sense an absorption feature

    def __init__(self, wl=[], flux=[]):
        self.wl = wl
        self.flux = flux

        self.midpoint = np.median(self.wl)

    #### determine basic statistics

    def is_edge(self, which):
        ## Just override the midpoint..
        ## not sure this is the best way or not..
        if which == "left":
            self.midpoint = np.array(self.wl)[0]

        elif which == "right":

            self.midpoint = np.array(self.wl)[-1]

        else:
            print("Error in edge definition")


    def get_statistics(self, flux_min=70):
        ### Scale
        self.mad = np.median(np.absolute(self.flux - np.median(self.flux)))   #An absorption feature would like have a larger MAD


        ### basically percent variation, remove flux impact
        if self.mad != 0:
            self.mad_normal = self.mad/np.median(self.flux)
        else:
            self.mad_normal = 0.0

        ### basic percentile clip in the median
        self.flux_med = np.median(self.flux[np.where((self.flux >= np.percentile(self.flux, flux_min)) & (self.flux <= np.percentile(self.flux, 98)))])

        ### get robust min/max estimate for the flux in segment
        self.flux_min = np.percentile(self.flux, flux_min)
        self.flux_max = np.percentile(self.flux, 98)


    def define_cont_point(self, mad_min, mad_range, boost=True):
        ### Here's the idea, the greater the flux variation in the segment,
        ### the larger the probability of influence from an absorption feature.
        ### there for, for larger variation, we want to bias the flux point assigned
        ### increasingly torwards the maximum flux in the segment

        self.mad_relative = (self.mad_normal - mad_min)/mad_range

        if boost:
            self.continuum_point = (self.flux_max - self.flux_med)*self.mad_relative + self.flux_med

        else:
            self.continuum_point = self.flux_med

        return
