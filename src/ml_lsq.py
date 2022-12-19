"""
Multilevel weighted polynomial least squares.
"""
import numpy as np
import time

from scipy.special import lambertw
from typing import Callable, Union

from sampling import sample_optimal_distribution
from legendre import get_total_degree_index_set
from lsq import LSQ


def unpack_parameters(params: dict) -> tuple[float]:
    beta_s = params["beta_s"]
    beta_w = params["beta_w"]
    gamma = params["gamma"]
    sigma = params["sigma"]
    alpha = params["alpha"]
    return beta_s, beta_w, gamma, sigma, alpha

def set_asymptotics(params: dict) -> tuple[float, float]:
    """
    Calculates two asymptotic parameters for the work of the multilevel algorithm.
    """
    beta_s, beta_w, gamma, sigma, alpha = unpack_parameters(params)

    c = gamma / beta_s - sigma / alpha
    theta = beta_s / beta_w
    l = theta * gamma / beta_s + (1 - theta) * sigma / alpha if c > 0 else sigma / alpha

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

    return l, t

def get_number_of_levels(eps: float, params: dict) -> int:
    """
    Returns the number of levels `L` required to achieve the
    theoretical residual and work bounds given by the tolerance `eps`.
    """
    beta_s, beta_w, gamma, sigma, alpha = unpack_parameters(params)

    c = gamma / beta_s - sigma / alpha
    if c < 0:
        L = - (sigma + alpha) / alpha * np.log(eps)
    elif c > 0:
        L = - (gamma + beta_s) / beta_w * np.log(eps)
    else:
        grid = np.linspace(1, 100, 1000)
        ix = np.argmin(np.abs(np.exp(-grid * alpha / (sigma + alpha)) * (grid + 1) - eps))
        L = grid[ix]

    L = int(np.ceil(L))

    return L

def get_tuning_params(L: int, params: dict, scale: float=1e-2) -> tuple[np.array, np.array, np.array]:
    """
    Returns the tuning parameters for the multilevel algorithm,
    given problem-dependent constants for specific assumptions.
    """
    beta_s, beta_w, gamma, sigma, alpha = unpack_parameters(params)

    delta = (beta_w - beta_s) / (gamma + beta_s) / alpha
    M = np.exp(delta * L) if gamma / beta_s > sigma / alpha else 1
    
    zero_to_L = np.arange(L + 1)
    mk = M * np.exp(zero_to_L / (sigma + alpha))
    nl = np.exp(zero_to_L / (gamma + beta_s))

    r = L
    kappa = (1 - np.log(2)) / (1 + r) / 2
    aux = (1 + scale) * mk ** sigma
    Ns = np.ceil(np.real(np.exp(-lambertw(-kappa / aux, k=-1))))

    mk = np.ceil(mk).astype(int)
    nl = np.ceil(nl).astype(int)

    return mk, nl, Ns

def theoretical_total_work(L: int, Ns: np.array, params: dict) -> float:
    """
    Returns the (theoretical) computational total work of the multilevel algorithm.
    """
    beta_s, _, gamma, _, _ = unpack_parameters(params)

    phi = np.exp(gamma / (gamma + beta_s))
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
    def __init__(self, params: dict) -> None:
        self.params_ = params
        self.asymptotics_ = set_asymptotics(self.params_)

    def __call__(self, x: Union[np.array, list[np.array]], grid: bool=False) -> np.array:
        try:
            projectors = self.projectors_
        except:
            raise ValueError("Model was not fitted yet.")
        else:
            if grid:
                return sum(p(x, grid=True) for p in projectors)
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

        w = print_start(eps, L, mk, nl, Ns, reduce_sample_by, reuse_sample)

        if reuse_sample:
            I = get_total_degree_index_set(max(mk))
            sample = sample_optimal_distribution(I, max(Ns))

        self.projectors_ = []
        for l in range(L + 1):
            s = time.perf_counter()

            m = mk[L - l]
            I = get_total_degree_index_set(m)
            N = Ns[L - l]
            n = nl[l]

            level_l_projector = LSQ(I, sampling="optimal")
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

    def l2_error(self, sample: list[np.array], values: np.array) -> float:
        """
        Calculates the L2 error on a grid given by the
        cartesian product of 1-D arrays in `sample`.
        """
        return np.sqrt(((self(sample, grid=True) - values) ** 2).mean())
