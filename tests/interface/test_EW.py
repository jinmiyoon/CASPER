import numpy as np

from casper.interface.EW import GBAND_QUAD


def test_gband_quad():
    wave = np.linspace(4200, 4350, 1000)  # range is from 4200A to 4350A using 1000 points
    flux = np.ones_like(wave)  # simulates a normalized flat continuum with no absorption lines
    flux[(wave > 4290) & (wave < 4300)] -= 0.1
    ew, ew_subtract = GBAND_QUAD(wave, flux)
    assert ew > 0
