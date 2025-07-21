import numpy as np
from pytest import approx

from casper.interface.MAD import MAD, S_MAD


def test_mad():
    arr = np.array([1, 2, 3, 4, 5])
    assert MAD(arr) == approx(1.0)


def test_s_mad():
    arr = np.array([1, 2, 3, 4, 5])
    assert S_MAD(arr) == approx(1.4826, rel=1e-4)
