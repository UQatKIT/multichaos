"""
(Multilevel) weighted polynomial least squares projection.
"""
import numpy as np
import time

from tqdm.notebook import trange
from typing import Callable, Union, Optional

from src.sampling import sample_optimal_distribution
from src.legendre import get_total_degree_index_set, index_set_transformation, optimal_weight, legvalnd, coef_to_func


IndexSet = list[Union[int, tuple[int, ...]]]


class ML_LSQ_Projector:
    def __init__(self, params: dict) -> None:
        self.beta_s = params["beta_s"]
        self.beta_w = params["beta_w"]
        self.gamma = params["gamma"]
        self.sigma = params["sigma"]
        self.alpha = params["alpha"]
        self.set_asymptotics()
        pass

    def set_asymptotics(self) -> tuple[float, float]:
        """
        Calculates two asymptotic parameters for the work of the multilevel algorithm.
        """
        beta_s, beta_w, gamma, sigma, alpha = (
            self.beta_s, self.beta_w, self.gamma, self.sigma, self.alpha
        )
        c = gamma / beta_s - sigma / alpha
        theta = beta_s / beta_w
        lambda_ = theta * gamma / beta_s + (1 - theta) * sigma / alpha if c > 0 else sigma / alpha
        if c < 0:
            t = 2
        elif c == 0:
            t = 3 + sigma / alpha
        elif c > 0 and beta_s == beta_w:
            t = 1
        elif c > 0 and beta_s > beta_w:
            t = 2
        else:
            raise ValueError("Not valid combination of assumption constants.")
        
        self.asymptotics = lambda_, t
        return

    def get_number_of_levels(self, eps: float) -> int:
        """
        Returns the number of levels `L` required to achieve the
        theoretical residual and work bounds given by the tolerance `eps`.
        """
        beta_s, beta_w, gamma, sigma, alpha = (
            self.beta_s, self.beta_w, self.gamma, self.sigma, self.alpha
        )
        c = gamma / beta_s - sigma / alpha
        
        if c < 0:
            L = - (sigma + alpha) / alpha * np.log(eps)
        elif c > 0:
            L = - (gamma + beta_s) / beta_w * np.log(eps)
        else:
            grid = np.linspace(1, 100, 1000)
            ix = np.argmin(np.abs(np.exp(-grid * alpha / (sigma + alpha)) * (grid + 1) - eps))
            L = grid[ix]

        return int(np.ceil(L))
    
    def get_tuning_params(self, L: int, scale: float=.0) -> tuple[np.array, np.array, np.array]:
        """
        Returns the tuning parameters for the multilevel algorithm,
        given problem-dependent constants for specific assumptions.
        """
        beta_s, beta_w, gamma, sigma, alpha = (
            self.beta_s, self.beta_w, self.gamma, self.sigma, self.alpha
        )
        delta = (beta_w - beta_s) / (gamma + beta_s) / alpha
        M = np.exp(delta * L) if gamma / beta_s > sigma / alpha else 1
        
        zero_to_L = np.arange(L + 1)
        mk = M * np.exp(zero_to_L / (sigma + alpha))
        nl = np.exp(zero_to_L / (gamma + beta_s))
        
        mk = np.array(list(map(int, np.ceil(mk))))
        nl = np.array(list(map(int, np.ceil(nl))))
        
        kappa = (1 - np.log(2)) / (1 + L) / 2
        aux = (1 + scale) * mk ** sigma
        grid_N = np.arange(2, 1_000_000)
        Ns = []
        for c in aux:   
            ix = np.argmin(np.abs(grid_N / np.log(grid_N) - c / kappa))
            Ns.append(grid_N[ix])
        Ns = np.array(Ns)

        self.mk = mk
        self.nl = nl
        self.Ns = Ns

        return mk, nl, Ns
    
    def total_work(self, L: int) -> float:
        """
        Returns the (theoretical) computational total work of the multilevel algorithm.
        """
        phi = np.exp(self.gamma / (self.gamma + self.beta_s))
        work = self.Ns[0] + (phi - 1) * (self.Ns[1:] * phi ** np.arange(L)).sum()

        return work
    
    def assemble_linear_system(self, I: IndexSet, x: np.array, f: Callable) -> tuple[np.array, np.array]:
        """
        Assembles the linear system `Gv=c` for the weighted LSQ
        problem using Legendre polynomials.
        """
        I = index_set_transformation(I)
        weights = optimal_weight(I, x)

        basis_val = np.zeros((len(I), len(x)))
        for i, eta in enumerate(I):
            basis_val[i] = legvalnd(x, eta)

        M = (np.sqrt(weights) * basis_val).T / np.sqrt(len(x))
        G = np.dot(M.T, M)

        f_ = f(x)
        c = np.mean(weights * f_ * basis_val, axis=1)

        return G, c

    def projection(self, grid: bool=False) -> Callable:
        funcs = [coef_to_func(v, grid=grid) for v in self.coef]
        func = lambda x: sum(f(x) for f in funcs)
        return func
    
    def l2_error(self, sample, values):
        err = np.sqrt(((self.projection(grid=True)(sample) - values) ** 2).mean())
        return err

    def lsq_projection(self, I: IndexSet, N: int, f: Callable, sample: Optional[np.array]=None) -> np.array:
        """
        Calculates the weighted least-squares projection of the function `f`
        using `N` random samples from the optimal distribution. 
        Returns the coefficient vector `v` w.r.t the basis given by `I`.
        """
        if sample is None:
            sample = sample_optimal_distribution(I, N)
        G, c = self.assemble_linear_system(I, sample, f)
        v = np.linalg.solve(G, c)
        return v


    def project(self, f: Callable, eps: float) -> np.array:
        """
        Returns the coefficients `v` w.r.t the total degree basis on each level `l = 0, ..., L`.
        """
        s_total = time.perf_counter()

        L = self.get_number_of_levels(eps)
        mk, nl, Ns = self.get_tuning_params(L)
        work = self.total_work(L)

        print(f"eps = {eps:.2e}, work = {work:.2e}, L = {L}")

        I = get_total_degree_index_set(mk[-1])
        sample = sample_optimal_distribution(I, max(Ns))

        self.coef = []
        for l in trange(L + 1):
            s = time.perf_counter()

            m = mk[L - l]
            I = get_total_degree_index_set(m)

            N = Ns[L - l]
            sample_l = sample[:N]

            f_l = f(nl[l])
            f_l__ = f_l_[:N] if l > 0 else np.zeros(N)  # f_{-1} := 0
            f_l_ = f_l(sample_l)

            v = self.lsq_projection(I, N, lambda _: f_l_ - f_l__, sample=sample_l)
            self.coef.append(v)

            e = time.perf_counter()

            print(f"Level {l} | m={m}, n={nl[l]}, N={N}, time={e - s:.3f}s")

        e_total = time.perf_counter()

        print(f"TOTAL TIME: {e_total - s_total:.3f}s")

        return
