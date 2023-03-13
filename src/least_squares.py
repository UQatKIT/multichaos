"""
Weighted polynomial least squares.
"""
import numpy as np
import time

from scipy.special import lambertw
from scipy.sparse.linalg import cg
from scipy.sparse.linalg import LinearOperator
from typing import Callable, Literal, Union
from polynomial_spaces import PolySpace

from sampling import sample_optimal_distribution, sample_arcsine
from legendre import legval_up_to_degree
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
        C = 1.0
        d = len(I[0]) if len(I) else 1.
        aux = C ** d * len(I)
    else:
        raise ValueError(f"Unknown sampling mode '{sampling}'.")

    kappa = (1 - np.log(2)) / (1 + r) / 2
    N = np.ceil(np.real(np.exp(-lambertw(- kappa / aux, k=-1)))).astype(int)

    return N

def get_sample(I: IndexSet, N: int, sampling: SamplingMode) -> np.array:
    if sampling == "optimal":
        sample = sample_optimal_distribution(I, N)
    elif sampling == "arcsine":
        sample = sample_arcsine((N, len(I[0])))

    return sample

def evaluate_basis(I: IndexSet, sample: np.array) -> np.array:
    I = np.array(I)

    dim = sample.ndim
    if dim == 1:
        basis_val = legval_up_to_degree(np.max(I), sample)
    else:
        max_idxs = np.max(I, axis=0)
        basis_val_uni = {}
        for j in range(dim):
            basis_val_uni[j] = legval_up_to_degree(max_idxs[j], sample[:, j])

        basis_val = np.zeros((len(I), len(sample)))
        for i, eta in enumerate(I):
            prod = 1.
            for j, k in enumerate(eta):
                prod *= basis_val_uni[j][k]
            basis_val[i] = prod

    norms = np.sqrt((2 * I + 1))
    if dim > 1:
        norms = norms.prod(axis=1)
    basis_val *= norms.reshape(-1, 1)

    return basis_val

def assemble_linear_system(I: IndexSet, sample: np.array, f: np.array) -> tuple[np.array, np.array]:
    """
    Assembles the linear system `Gv=c` for the weighted LSQ
    problem using Legendre polynomials.
    """
    basis_val = evaluate_basis(I, sample)

    weights = len(I) / np.einsum('ij,ij->j', basis_val, basis_val)

    M = basis_val * np.sqrt(weights) / np.sqrt(len(sample))
    c = np.mean(weights * f * basis_val, axis=1)

    return M, c

class SingleLevelLSQ:
    def __init__(self, poly_space: PolySpace, sampling: SamplingMode) -> None:
        self.poly_space = poly_space
        self.sampling = sampling

    def __call__(self, x: np.array) -> np.array:
        try:
            coef = self.coef_
        except:
            raise ValueError("Model was not fitted yet.")
        else:
            I = self.poly_space.index_set
            coef = np.array(list(self.coef_.values()))
            basis = evaluate_basis(I, x)
            return np.dot(coef, basis)

    def fit(self, f: Union[Callable, np.array], N: int=None, sample: np.array=None):
        """
        Calculates the weighted least squares projection of `f`
        onto the polynomial space given by the index set `I` and
        saves the coefficients `v` w.r.t the basis given by `I`.
        """
        start = time.perf_counter()

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

        M, c = assemble_linear_system(I, sample, f)

        m = M.shape[0]
        G = LinearOperator((m, m), matvec=lambda x: np.dot(M, np.dot(M.T, x)))
        v = cg(G, c)[0]

        self.time_ = time.perf_counter() - start

        self.coef_ = dict(zip(I, v))
        self.sample_ = sample
        self.f_vals_ = f

        return self

    def l2_error(self, sample: np.array, values: np.array) -> float:
        """
        Calculates the L2 error on a grid given by the
        cartesian product of 1-D arrays in `sample`.
        """
        return np.sqrt(mse(self(sample), values))
