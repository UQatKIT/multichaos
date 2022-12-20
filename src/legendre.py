"""
Methods for constructing basis and evaluation of Legendre polynomials.
"""
import numpy as np

from scipy.special import eval_sh_legendre
from typing import Union


IndexSet = list[Union[int, tuple[int, ...]]]


def legvalnd(x: np.array, c: np.array) -> np.array:
    if isinstance(c, int):
        c = [c]
    if x.ndim == 1:
        x = np.expand_dims(x, axis=-1)
    vals = np.zeros((len(c), len(x)))

    for i, k in enumerate(c):
        vals[i] = eval_sh_legendre(k, x[:, i])

    norm = np.sqrt((2 * np.array(c) + 1).prod())
    vals = norm * vals.prod(axis=0)
    return vals

def get_total_degree_index_set(m: int, d: int=2) -> IndexSet:
    """
    Returns the total degree basis index set of order `m` as a list of tuples.
    """
    # TODO: Extend to dimension `d > 2`.
    if d == 1:
        return list(range(m))
    I = []
    for i in range(m + 1):
        for j in range(m - i, -1, -1):
            I.append((i, j))
    return I

def optimal_density(I: IndexSet, x: np.array) -> np.array:
    """
    Returns the optimal density function defined by `I` evaluated in `x`.
    """
    return sum(legvalnd(x, eta) ** 2 for eta in I) / len(I)
