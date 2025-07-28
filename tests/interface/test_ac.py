import pytest

from casper.interface.ac import ac, cfe


def test_ac():
    assert ac(0.0, 0.0) == 8.43
    assert ac(0.5, -1.0) == 7.93
    assert ac(-0.3, -2.2) == pytest.approx(5.93)


def test_cfe():
    assert cfe(8.43, 0.0) == 0.0
    assert cfe(7.93, -0.5) == 0.0
    assert cfe(5.93, -2.2) == pytest.approx(-0.3)
