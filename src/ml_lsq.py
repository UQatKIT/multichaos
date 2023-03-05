"""
Multilevel weighted polynomial least squares.
"""
import numpy as np
import time

from scipy.special import lambertw
from typing import Callable, Literal

from sampling import sample_optimal_distribution
from polynomial_spaces import TensorProduct
from polynomial_spaces import TotalDegree
from polynomial_spaces import HyperbolicCross
from lsq import LSQ
from utils import mse


PolySpace = Literal[
    "TP",   # tensor product
    "TD",   # total degree
    "HC",   # hyperbolic cross
]


def unpack_parameters(params: dict) -> tuple[float]:
    beta = params["beta"]
    gamma = params["gamma"]
    sigma = params["sigma"]
    alpha = params["alpha"]
    return beta, gamma, sigma, alpha

def set_asymptotics(params: dict) -> tuple[float, float]:
    """
    Calculates two asymptotic parameters for the work of the multilevel algorithm.
    """
    beta, gamma, sigma, alpha = unpack_parameters(params)

    c = gamma / beta - sigma / alpha
    l = max(gamma / beta, sigma / alpha)

    if c > 0:
        t = 1
    elif c < 0:
        t = 2
    else:
        t = 3 + sigma / alpha

    return l, t

def get_number_of_levels(eps: float, params: dict) -> int:
    """
    Returns the number of levels `L` required to achieve the
    theoretical residual and work bounds given by the tolerance `eps`.
    """
    beta, gamma, sigma, alpha = unpack_parameters(params)

    c = gamma / beta - sigma / alpha
    if c > 0:
        L = - (gamma + beta) / beta * np.log(eps)
    elif c < 0:
        L = - (sigma + alpha) / alpha * np.log(eps)
    else:
        grid = np.linspace(1, 100, 1000)
        ix = np.argmin(np.abs(np.exp(-grid * alpha / (sigma + alpha)) * (grid + 1) - eps))
        L = grid[ix]

    L = int(np.ceil(L))

    return L

def get_tuning_params(L: int, params: dict) -> tuple[np.array, np.array, np.array]:
    """
    Returns the tuning parameters for the multilevel algorithm,
    given problem-dependent constants for specific assumptions.
    """
    beta, gamma, sigma, alpha = unpack_parameters(params)

    mk = np.exp(np.arange(L + 1) / (sigma + alpha))
    nl = np.exp(np.arange(L + 1) / (gamma + beta))

    r = L
    kappa = (1 - np.log(2)) / (1 + r) / 2
    aux = mk ** sigma
    Ns = np.ceil(np.real(np.exp(-lambertw(-kappa / aux, k=-1))))

    mk = np.ceil(mk).astype(int)
    nl = np.ceil(nl).astype(int)

    return mk, nl, Ns

def theoretical_total_work(L: int, Ns: np.array, params: dict) -> float:
    """
    Returns the (theoretical) computational total work of the multilevel algorithm.
    """
    beta, gamma, sigma, alpha = unpack_parameters(params)

    phi = np.exp(gamma / (gamma + beta))
    work = Ns[0] + (phi - 1) * (Ns[1:] * phi ** np.arange(L)).sum()

    return work

def print_start(eps: float, L: int, mk: np.array, nl: np.array, Ns: np.array, reduce: float, reuse_sample: bool) -> list[int]:
    w = [max(map(len, map(str, arr))) for arr in [mk, nl, Ns]]
    verboseprint(f"eps = {eps:.2e}, L = {L}, reduce = {100 * reduce}%, reuse_sample = {reuse_sample}")
    verboseprint(f"{'Level':>5} | {'m':>{w[0]}} | {'n':>{w[1]}} | {'N':>{w[2]}} | {'time':>7}")
    verboseprint("-" * (24 + sum(w)))
    return w

def print_level(l: int, m: int, n: int, N: int, time: float, w: list[int]):
    verboseprint(f"{l:>5} | {m:>{w[0]}} | {n:>{w[1]}} | {N:>{w[2]}} | {time:>6.3f}s")

def print_end(time):
    verboseprint(f"TOTAL TIME: {time:.3f}s\n")


class ML_LSQ:
    def __init__(self, params: dict, dim: int, poly_space: PolySpace) -> None:
        self.params_ = params
        self.dim_ = dim

        if poly_space == "TP":
            self.poly_space = TensorProduct
        elif poly_space == "TD":
            self.poly_space = TotalDegree
        elif poly_space == "HC":
            self.poly_space = HyperbolicCross
        else:
            raise ValueError(
                "{poly_space} is not a valid polynomial space. Should be one of ['TP', 'TD', 'HC']."
                )

        self.asymptotics_ = set_asymptotics(self.params_)

    def __call__(self, x: np.array) -> np.array:
        try:
            projectors = self.projectors_
        except:
            raise ValueError("Model was not fitted yet.")
        else:
            return sum(p(x) for p in projectors)

    def solve(self, f: Callable, eps: float, reduce_sample_by: float=0., reuse_sample: bool=False, verbose: bool=True) -> np.array:
        """
        Returns the coefficients `v` w.r.t the total degree basis on each level `l = 0, ..., L`.
        """
        global verboseprint
        verboseprint = print if verbose else lambda *a, **k: None

        s_total = time.perf_counter()

        L = get_number_of_levels(eps, self.params_)
        mk, nl, Ns = get_tuning_params(L, self.params_)
        Ns = np.ceil((1 - reduce_sample_by) * Ns).astype(int)

        self.L = L
        self.mk = mk
        self.nl = nl
        self.Ns = Ns

        w = print_start(eps, L, mk, nl, Ns, reduce_sample_by, reuse_sample)

        if reuse_sample:
            I = self.poly_space(max(mk), d=self.dim_).index_set
            sample = sample_optimal_distribution(I, max(Ns))

        self.projectors_ = []
        for l in range(L + 1):
            s = time.perf_counter()

            m = mk[L - l]
            N = Ns[L - l]
            n = nl[l]

            V_m = self.poly_space(m, d=self.dim_)
            level_l_projector = LSQ(V_m, sampling="optimal")
            if not reuse_sample:
                f_l = f(n)
                f_l_ = f(nl[l - 1]) if l > 0 else lambda _: 0  # f_{-1} := 0
                f_ = lambda g: f_l(g) - f_l_(g)
                level_l_projector.solve(f_, N=N)
            else:
                sample_l = sample[:N]
                f_l = f(n)
                f_l__ = f_l_[:N] if l > 0 else np.zeros(N)  # f_{-1} := 0
                f_l_ = f_l(sample_l)
                level_l_projector.solve(f_l_ - f_l__, sample=sample_l)

            self.projectors_.append(level_l_projector)

            e = time.perf_counter()

            print_level(l, m, n, N, e - s, w)

        e_total = time.perf_counter()
        print_end(e_total - s_total)

        return self

    def l2_error(self, sample: np.array, values: np.array) -> float:
        """
        Calculates the L2 error on a grid given by the
        cartesian product of 1-D arrays in `sample`.
        """
        return np.sqrt(mse(self(sample), values))