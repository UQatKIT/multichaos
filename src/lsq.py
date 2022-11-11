"""
Weighted polynomial least squares.
"""
import numpy as np

from typing import Callable, Union

from sampling import sample_optimal_distribution
from legendre import index_set_transformation
from legendre import optimal_weight
from legendre import legvalnd, leggridnd


IndexSet = list[Union[int, tuple[int, ...]]]


def get_optimal_sample_size(I: IndexSet, r: float=1.) -> int:
    """
    Returns the sample size that satisfies the
    optimality constraint for a stable projection.
    """
    kappa = (1 - np.log(2)) / (1 + r) / 2
    grid = np.arange(2, 1_000_000)
    ix = np.argmin(np.abs(grid / np.log(grid) - len(I) / kappa))
    return grid[ix]

def assemble_linear_system(I: IndexSet, sample: np.array, f: Callable) -> tuple[np.array, np.array]:
    """
    Assembles the linear system `Gv=c` for the weighted LSQ
    problem using Legendre polynomials.
    """
    I = index_set_transformation(I)
    weights = optimal_weight(I, sample)

    basis_val = np.zeros((len(I), len(sample)))
    for i, eta in enumerate(I):
        basis_val[i] = legvalnd(sample, eta)

    M = (np.sqrt(weights) * basis_val).T / np.sqrt(len(sample))
    G = np.dot(M.T, M)

    f_ = f(sample)
    c = np.mean(weights * f_ * basis_val, axis=1)

    return G, c

class LSQ:
    def __init__(self, I: IndexSet) -> None:
        self.I = I
        self.I_ = index_set_transformation(I)

    def __call__(self, x: Union[np.array, list[np.array]], grid: bool=False) -> np.array:
        try:
            coef = self.coef_
        except:
            raise ValueError("Model was not fitted yet.")
        else:
            if grid:
                return sum(v * leggridnd([x, x], eta) for v, eta in zip(coef, self.I_))
            else:
                return sum(v * legvalnd(x, eta) for v, eta in zip(coef, self.I_))

    def solve(self, f: Callable, N: int=None, sample: np.array=None):
        """
        Calculates the weighted least squares projection of `f`
        onto the polynomial space given by the index set `I` and
        saves the coefficients `v` w.r.t the basis given by `I`.
        """
        if N is None and sample is None:
            N = get_optimal_sample_size(self.I)
            sample = sample_optimal_distribution(self.I, N)
        elif N is not None and sample is None:
            sample = sample_optimal_distribution(self.I, N)
        elif N is not None and sample is not None:
            raise ValueError("Provide either a sample or a number of samples, not both.")

        G, c = assemble_linear_system(self.I, sample, f)

        self.coef_ = np.linalg.solve(G, c)
        self.cond_ = np.linalg.cond(G)

    def l2_error(self, sample: list[np.array], values: np.array) -> float:
        """
        Calculates the L2 error on a grid given by the
        cartesian product of 1-D arrays in `sample`.
        """
        return np.sqrt(((self(sample, grid=True) - values) ** 2).mean())
