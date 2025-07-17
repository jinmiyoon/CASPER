import math

import numpy as np
import pandas as pd
import pytest

from casper.interface.temp_calibrations import Bergeat, Casagrande, Fukugita, Hernandez, calibrate_temp_frame


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


# calibrate_temp_frame
@pytest.mark.parametrize(
    "JK, gr, expected_non_nan_count",
    [
        (0.85, 0.55, 4),  # 3 valid + ADOPTED
        (0.85, np.nan, 3),  # 2 valid + ADOPTED
        (np.nan, 0.55, 2),  # 1 valid + ADOPTED
        (np.nan, np.nan, 0),  # All NaN including ADOPTED
    ],
)
def test_calibrate_temp_frame(JK, gr, expected_non_nan_count):
    df = calibrate_temp_frame(JK, gr)

    assert isinstance(df, pd.DataFrame)
    assert "VALUE" in df.columns
    assert "ADOPTED" in df.index

    # Count how many temperature estimates (including ADOPTED) are not NaN
    non_nan_count = df["VALUE"].notna().sum()

    assert non_nan_count == expected_non_nan_count
