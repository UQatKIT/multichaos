"""
(Single-level) weighted polynomial least squares.
"""
from __future__ import annotations
import numpy as np
import time

from typing import Callable, Literal, Union

from sampling import sample_optimal_distribution
from sampling import sample_arcsine
from sampling import optimal_sample_size
from legendre import evaluate_basis
from utils import mse


IndexSet = list[Union[int, tuple[int, ...]]]
SamplingMode = Literal["optimal", "arcsine"]


class SingleLevelLSQ:
    def __init__(self, params: dict):
        self.params = params

        self.poly_space = params.get("poly_space", None)
        self.sampling = params.get("sampling", "optimal")
        self.weighted = params.get("weighted", True)
        self.reduce_sample_by = params.get("reduce_sample_by", .0)
        self.verbose = params.get("verbose", True)

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

    def get_sample(self, N: int) -> np.array:
        I = self.poly_space.index_set
        if self.sampling == "optimal":
            sample = sample_optimal_distribution(I, N)
        elif self.sampling == "arcsine":
            d = 1 if isinstance(I[0], int) else len(I[0])
            sample = sample_arcsine((N, d))
        return sample.squeeze()

    def assemble_linear_system(self) -> tuple[np.array, np.array]:
        """
        Assembles the linear system `Gv=c` for the weighted LSQ
        problem using Legendre polynomials.
        """
        I = self.poly_space.index_set
        basis_val = evaluate_basis(I, self.sample_)

        if self.weighted:
            m = self.poly_space.dim()
            weights = m / np.einsum('ij,ij->j', basis_val, basis_val)
        else:
            weights = np.ones(basis_val.shape[1])

        M = basis_val * np.sqrt(weights) / np.sqrt(len(self.sample_))
        c = np.mean(weights * self.f_vals_ * basis_val, axis=1)

        G = np.dot(M, M.T)

        return G, c

    def fit(self, f: Union[Callable, np.array], N: int=None, sample: np.array=None) -> SingleLevelLSQ:
        start = time.perf_counter()

        I = self.poly_space.index_set
        if sample is None:
            if N is None:
                N = optimal_sample_size(I, self.sampling, reduce=self.reduce_sample_by)
            sample = self.get_sample(N)
        else:
            if N is not None:
                raise ValueError("Provide either a sample or a number of samples, not both.")

        if isinstance(f, Callable):
            f = f(sample)

        self.sample_ = sample
        self.f_vals_ = f

        G, c = self.assemble_linear_system()

        self.G = G

        v = np.linalg.solve(G, c)
        self.coef_ = dict(zip(I, v))

        self.time_ = time.perf_counter() - start

        return self

    def l2_error(self, x: np.array, y: np.array) -> float:
        return np.sqrt(mse(self(x), y))
