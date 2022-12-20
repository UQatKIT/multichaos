"""
Weighted polynomial least squares.
"""
import numpy as np

from scipy.special import lambertw, eval_sh_legendre
from typing import Callable, Literal, Union
from polynomial_spaces import PolySpace

from sampling import sample_optimal_distribution, sample_arcsine
from legendre import legvalnd
from utils import mse


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
    N = np.ceil(np.real(np.exp(-lambertw(- kappa / aux, k=-1)))).astype(int)

    return N

def get_sample(I: IndexSet, N: int, sampling: SamplingMode) -> np.array:
    if sampling == "optimal":
        sample = sample_optimal_distribution(I, N)
    elif sampling == "arcsine":
        sample = sample_arcsine((N, len(I[0])))

    return sample

def assemble_linear_system(I: IndexSet, sample: np.array, f: np.array) -> tuple[np.array, np.array]:
    """
    Assembles the linear system `Gv=c` for the weighted LSQ
    problem using Legendre polynomials.
    """
    counts = {i: np.unique(col) for i, col in enumerate(np.array(I).T)}
    basis_val_uni = {j: {} for j in counts.keys()}
    for j, ks in counts.items():
        for k in ks:
            basis_val_uni[j][k] = eval_sh_legendre(k, sample[:, j])

    basis_val = np.zeros((len(I), len(sample)))
    for i, eta in enumerate(I):
        prod = 1.
        for j, k in enumerate(eta):
            prod *= basis_val_uni[j][k]
        basis_val[i] = prod

    norms = np.sqrt((2 * np.array(I) + 1).prod(axis=1))
    basis_val *= norms.reshape(-1, 1)

    weights = len(I) / (basis_val ** 2).sum(axis=0)

    M = basis_val * np.sqrt(weights)
    G = np.dot(M, M.T) / len(sample)
    c = np.mean(weights * f * basis_val, axis=1)

    return G, c

class LSQ:
    def __init__(self, poly_space: PolySpace, sampling: SamplingMode) -> None:
        self.poly_space = poly_space
        self.sampling = sampling
        assert sampling in ["optimal", "arcsine"], \
            ValueError(f"Unkwnown sampling mode '{sampling}'.")

    def __call__(self, x: np.array) -> np.array:
        try:
            coef = self.coef_
        except:
            raise ValueError("Model was not fitted yet.")
        else:
            I = self.poly_space.index_set
            return sum(v * legvalnd(x, eta) for v, eta in zip(coef, I))

    def solve(self, f: Union[Callable, np.array], N: int=None, sample: np.array=None):
        """
        Calculates the weighted least squares projection of `f`
        onto the polynomial space given by the index set `I` and
        saves the coefficients `v` w.r.t the basis given by `I`.
        """
        I = self.poly_space.index_set
        if sample is None:
            if N is None:
                N = get_optimal_sample_size(I, self.sampling)
            sample = get_sample(I, N, self.sampling)
        else:
            if N is not None:
                raise ValueError("Provide either a sample or a number of samples, not both.")

        if isinstance(f, Callable):
            f = f(sample)

        G, c = assemble_linear_system(I, sample, f)
        self.coef_ = np.linalg.solve(G, c)

        return self

    def l2_error(self, sample: np.array, values: np.array) -> float:
        """
        Calculates the L2 error on a grid given by the
        cartesian product of 1-D arrays in `sample`.
        """
        return np.sqrt(mse(self(sample), values))
