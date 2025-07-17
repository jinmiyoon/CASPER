import math

import numpy as np
import pytest

from casper.interface.temp_calibrations import Bergeat, Casagrande, Fukugita, Hernandez


# hernandez
@pytest.mark.parametrize(
    "jk, feh, l_class, expected_t_jk",
    [
        (0.5, -2.5, "GIANT", 5232),
        (1.5, -2.5, "DWARF", np.nan),
        (0.5, -2.5, "Random string", 5232),
        (0.5, -2.5, None, 5232),
    ],
)
def test_hernandez(jk, feh, l_class, expected_t_jk):
    t_jk = Hernandez(JK=jk, FEH=feh, CLASS=l_class)

    if np.isnan(expected_t_jk):
        assert np.isnan(t_jk)
    else:
        assert round(t_jk) == expected_t_jk


# casagrande
@pytest.mark.parametrize(
    "jk, expected_teff",
    [
        (0.5, 5314),  # in range value, rounded expected Teff
        (0.9, np.nan),  # out-of-range, expect np.nan
    ],
)
def test_casagrande(jk, expected_teff):
    teff = Casagrande(JK=jk)  # feh is -2.5

    if np.isnan(expected_teff):
        assert np.isnan(teff)
    else:
        assert round(teff) == expected_teff


# bergeat
@pytest.mark.parametrize(
    "jk, expected_teff", [(1.0, round(10 ** (-0.184 * 1.0 + 3.74))), (2.5, round(10 ** (-0.109 * 2.5 + 3.59)))]
)
def test_bergeat(jk, expected_teff):
    teff = Bergeat(JK=jk)
    print(f"JK={jk}: Teff={teff}, Expected={expected_teff}")
    assert isinstance(teff, float)
    assert round(teff) == expected_teff


# fukugita
@pytest.mark.parametrize(
    "gr, expected_range",
    [
        (0.3, (5000, 7000)),  # Valid case
        (-1.47, (np.nan, np.nan)),  # Divide by zero: np.nan expected
        ("bad", (np.nan, np.nan)),  # Invalid input: np.nan expected
    ],
)
def test_Fukugita(gr, expected_range):
    result = Fukugita(gr)

    if math.isnan(expected_range[0]):
        assert math.isnan(result)
    else:
        assert expected_range[0] < result < expected_range[1]
