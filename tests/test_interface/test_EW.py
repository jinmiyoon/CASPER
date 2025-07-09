import numpy as np

# Import your functions (adjust this path if needed)
from casper.interface.EW import GBAND_QUAD


def test_gband_quad():
    wave = np.linspace(4260, 4320, 1000)  # range is from 4260A to 4320A using 1000 points
    flux = np.ones_like(wave)  # simulates a normalized flat continuum with no abosrption lines
    flux[(wave > 4290) & (wave < 4300)] -= 0.1
    ew, ew_subtract = GBAND_QUAD(wave, flux)
    assert ew > 0
    assert ew_subtract >= 0
