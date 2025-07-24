import pytest

from casper.interface.gisic import in_molecular_band


@pytest.mark.parametrize(
    "wl, expected",
    [
        (4260.0, True),  # in range
        (4800.0, False),  # out of range
    ],
)
def test_in_molecular_band(wl, expected):
    assert in_molecular_band(wl) == expected
