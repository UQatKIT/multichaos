"""
Methods for constructing basis and evaluation of Legendre polynomials.
"""
import numpy as np

from scipy.special import eval_sh_legendre


def legval_up_to_degree(n, sample):
    """Evaluates the first `n` Legendre polynomials in the given sample."""
    y = np.zeros((n + 1, len(sample)))

    y[0] = eval_sh_legendre(0, sample)
    if n > 0:
        y[1] = eval_sh_legendre(1, sample)

    sh_sample = 2 * sample - 1
    for i in range(1, n):
        y[i + 1] = ((2 * i + 1) * sh_sample * y[i] - i * y[i - 1]) / (i + 1)

    return y
