import numpy as np
from numpy.typing import ArrayLike


def MAD(array: ArrayLike) -> float:
    """
    Compute the Median Absolute Deviation (MAD) of a numerical array.

    MAD is a robust measure of statistical dispersion. It is calculated
    as the median of the absolute deviations from the median of the data.

    Parameters
    ----------
    array : ArrayLike
        Input array of numerical values.

    Returns
    -------
    float
        The median absolute deviation of the input array.
    """
    return np.median(np.abs(array - np.median(array)))


def S_MAD(array: ArrayLike) -> float:
    """
    Compute the scaled Median Absolute Deviation (S_MAD) as an estimator of standard deviation.

    This scales the MAD by dividing by 0.6745, which makes it consistent with the
    standard deviation under a normal distribution.

    Parameters
    ----------
    array : ArrayLike
        Input array of numerical values.

    Returns
    -------
    float
        The scaled MAD, used as a robust estimate of standard deviation.
    """
    return MAD(array) / 0.6745
