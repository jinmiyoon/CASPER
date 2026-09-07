import numpy as np
from pytest import approx

from casper.interface.MAD import MAD, S_MAD


class TestMAD:
    def test_mad_basic_cases(self):
        """Test MAD on simple known cases"""
        # Single element should have MAD of 0
        assert MAD(np.array([5.0])) == 0.0

        # Constant array should have MAD of 0
        assert MAD(np.array([3.0, 3.0, 3.0])) == 0.0

        # Known value
        arr = np.array([1, 2, 3, 4, 5])
        assert MAD(arr) == approx(1.0)

    def test_s_mad_basic_cases(self):
        """Test scaled MAD on simple known cases"""
        # Single element should have S_MAD of 0
        assert S_MAD(np.array([5.0])) == 0.0

        # Constant array should have S_MAD of 0
        assert S_MAD(np.array([3.0, 3.0, 3.0])) == 0.0

        # Known value
        arr = np.array([1, 2, 3, 4, 5])
        assert S_MAD(arr) == approx(1.4826, rel=1e-4)
