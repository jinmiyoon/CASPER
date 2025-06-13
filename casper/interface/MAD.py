# MAD : median absolute distribution
"""
Refer to a formular by Iglewicz and Hoaglin (1993)
MAD = med(abs(x-med(x))). If x is normally distributed, it can be shown that

MAD ~ std * quantile(q=0.75) = std * 0.6745.

Therefore,  std = MAD/0.6745 is a consistent estimator of the std

"""

import numpy as np


def MAD(array):
    return np.median(np.abs(array - np.median(array)))


def S_MAD(array):
    return MAD(array) / 0.6745
