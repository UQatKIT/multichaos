"""
Methods for constructing basis and evaluation of Legendre polynomials.
"""
import numpy as np

from numpy.polynomial.legendre import legval
from numpy.polynomial.polyutils import _gridnd
from scipy.special import binom
from typing import Callable, Union


IndexSet = list[Union[int, tuple[int, ...]]]


def legvalnd(x: np.array, c: np.array) -> np.array:
    """
    Evaluate a multivariate Legendre polynomial at N-D points `x`.
    """
    x = 2 * x - 1  # transform [0, 1] -> [-1, 1]
    if len(x.shape) > 1:
        for i in range(x.shape[-1]):
            c = legval(x[:, i], c) if i == 0 else legval(x[:, i], c, tensor=False)
    else:
        c = legval(x, c)
    return c

def leggridnd(x: list[np.array], c: np.array) -> np.array:
    """
    Evaluate a multivariate Legendre polynomial at the cartesian grid
    given by the 1-D arrays in `x`.
    """
    for i, x_ in enumerate(x):
        x[i] = 2 * x_ - 1  # transform [0, 1] -> [-1, 1]
    return _gridnd(legval, c, *x)

def get_total_degree_index_set(m: int, d: int=2) -> IndexSet:
    """
    Returns the total degree basis index set of order `m` as a list of tuples.
    """
    # TODO: Extend to dimension `d > 2`.
    if d == 1:
        return list(range(m + 1))
    I = []
    for i in range(m + 1):
        for j in range(m - i, -1, -1):
            I.append((i, j))
    return I

def index_set_transformation(I: IndexSet) -> list[np.array]:
    """
    Transform index set `I` into a list of
    coefficient matrices compatible with `legvalnd`.
    """
    I_ = []
    for eta in I:
        c = np.zeros(np.array(eta) + 1)
        c[eta] = 1
        c *= np.sqrt(2 * np.array(eta) + 1).prod()  # normalize polynomials
        I_.append(c)

    return I_

def optimal_density(I: IndexSet, x: np.array) -> np.array:
    """
    Returns the optimal density function defined by `I` evaluated in `x`.
    """
    return sum(legvalnd(x, eta) ** 2 for eta in I) / len(I)

def optimal_weight(I: IndexSet, x: np.array) -> np.array:
    """
    Returns the optimal weight function defined by `I` evaluated in `x`.
    """
    return len(I) / sum(legvalnd(x, eta) ** 2 for eta in I)

def coef_to_func(v: np.array, d: int=2, grid: bool=False) -> Callable:
    """
    Returns the polynomial given by the coefficients `v` w.r.t the
    total degree basis of the corresponding order.
    """
    m = 0
    while binom(m + d, m) != len(v):
        m += 1

    I = get_total_degree_index_set(m, d=d)
    I = index_set_transformation(I)

    if grid:
        func = lambda x: sum(v[i] * leggridnd(d * [x], eta) for i, eta in enumerate(I))
    else:
        func = lambda x: sum(v[i] * legvalnd(x, eta) for i, eta in enumerate(I))
    return func
