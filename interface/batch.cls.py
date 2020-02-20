#### this is the class definition for the Batch class.
#### just bundling the Spectrum objects and normalization/analysis routines
import interface_main, spectrum

class Batch():
    #### Main Batch Class
    def __init__(self, spectra_path, param_file):

        self.spectrum_array = []

        self.param_file     = []

        return
