"""
Methods for constructing basis and evaluation of Legendre polynomials.
"""
import numpy as np


def legval(n: int, sample: np.array) -> np.array:
    """
    Evaluates the first `n + 1` shifted Legendre polynomials in the given sample.
    """
    y = np.zeros((n + 1, len(sample)))
    y[0] = np.ones_like(sample)

    if n > 0:
        sh_sample = 2 * sample - 1
        y[1] = sh_sample
        for i in range(1, n):
            y[i + 1] = ((2 * i + 1) * sh_sample * y[i] - i * y[i - 1]) / (i + 1)

    return y
