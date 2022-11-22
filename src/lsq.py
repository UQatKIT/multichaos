"""
Weighted polynomial least squares.
"""
import numpy as np

from typing import Callable, Literal, Union

from sampling import arcsine, sample_optimal_distribution, sample_arcsine
from legendre import index_set_transformation, optimal_density
from legendre import legvalnd, leggridnd


IndexSet = list[Union[int, tuple[int, ...]]]
SamplingMode = Literal["optimal", "arcsine"]


def get_optimal_sample_size(I: IndexSet, sampling: SamplingMode, r: float=1.) -> int:
    """
    Returns the sample size that satisfies the
    optimality constraint for a stable projection.
    """
    if sampling == "optimal":
        aux = len(I)
    elif sampling == "arcsine":
        C = 1.1
        d = len(I[0]) if len(I) else 1.
        aux = C ** d * len(I)

    kappa = (1 - np.log(2)) / (1 + r) / 2
    grid = np.arange(2, 1_000_000)
    ix = np.argmax((grid / np.log(grid) - aux / kappa) >=0)
    return grid[ix]

def get_sample(I: IndexSet, N: int, sampling: SamplingMode) -> np.array:
    if sampling == "optimal":
        sample = sample_optimal_distribution(I, N)
    elif sampling == "arcsine":
        sample = sample_arcsine((N, len(I[0])))

    return sample

def get_weights(I: IndexSet, sample: np.array, sampling: SamplingMode) -> np.array:
    if sampling == "optimal":
        weights = 1. / optimal_density(index_set_transformation(I), sample)
    elif sampling == "arcsine":
        weights = 1. / arcsine(sample)

    return weights

def assemble_linear_system(I: IndexSet, sample: np.array, f: np.array, weights: np.array) -> tuple[np.array, np.array]:
    """
    Assembles the linear system `Gv=c` for the weighted LSQ
    problem using Legendre polynomials.
    """
    I = index_set_transformation(I)

    basis_val = np.zeros((len(I), len(sample)))
    for i, eta in enumerate(I):
        basis_val[i] = legvalnd(sample, eta)

    M = (np.sqrt(weights) * basis_val).T / np.sqrt(len(sample))
    G = np.dot(M.T, M)
    c = np.mean(weights * f * basis_val, axis=1)

    return G, c

class LSQ:
    def __init__(self, I: IndexSet, sampling: SamplingMode) -> None:
        self.I = I
        self.I_ = index_set_transformation(I)
        self.sampling = sampling
        assert sampling in ["optimal", "arcsine"], \
            ValueError(f"Unkwnown sampling mode '{sampling}'.")

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
        if sample is None:
            if N is None:
                N = get_optimal_sample_size(self.I, self.sampling)
            sample = get_sample(self.I, N, self.sampling)
        else:
            if N is not None:
                raise ValueError("Provide either a sample or a number of samples, not both.")

        weights = get_weights(self.I, sample, self.sampling)
        f = f(sample)
        G, c = assemble_linear_system(self.I, sample, f, weights)

        self.coef_ = np.linalg.solve(G, c)
        self.cond_ = np.linalg.cond(G)

        return self

    def l2_error(self, sample: list[np.array], values: np.array) -> float:
        """
        Calculates the L2 error on a grid given by the
        cartesian product of 1-D arrays in `sample`.
        """
        return np.sqrt(((self(sample, grid=True) - values) ** 2).mean())
