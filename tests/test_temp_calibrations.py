import numpy as np

from casper.interface.temp_calibrations import Hernandez


def test_hernandez_giant():
    teff = Hernandez(JK=0.5, FEH=-2.5, CLASS="GIANT")
    assert isinstance(teff, float)
    assert teff > 0
    assert not np.isnan(teff)


def test_hernandez_dwarf():
    teff = Hernandez(JK=0.5, FEH=-2.5, CLASS="DWARF")
    assert isinstance(teff, float)
    assert teff > 0
    assert not np.isnan(teff)
