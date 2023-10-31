"""
Multilevel weighted polynomial least squares.
"""
from __future__ import annotations
import numpy as np
import time

from collections import defaultdict
from scipy.special import lambertw
from typing import Union, Optional

from sampling import sample_optimal_distribution
from legendre import call
from polynomial_spaces import PolySpace
from least_squares import SingleLevelLSQ
from utils import mse
from utils import unit_to_domain


class MultiLevelLSQ:
    def __init__(self, problem: dict, params: dict):
        self.problem = problem
        self.params = params

        self.beta = problem["beta"]
        self.gamma = problem["gamma"]
        self.sigma = problem["sigma"]
        self.alpha = problem["alpha"]

        self.response = problem.get("response", None)
        self.domain = problem.get("domain", None)
        self.dim = len(self.domain)

        self.reuse_sample = params.get("reuse_sample", False)
        self.reduce_sample_by = params.get("reduce_sample_by", .0)
        self.poly_space = params.get("poly_space", "TD")
        self.verbose = params.get("verbose", True)

        self.C_m = params.get("C_m", 1.)
        self.C_n = params.get("C_n", 1.)

        self.asymptotics()

    def asymptotics(self):
        b, g, s, a = self.beta, self.gamma, self.sigma, self.alpha
        case = g / b - s / a
        self.lambda_ = max(g / b, s / a)
        if case > 0:
            self.t = 1
        elif case < 0:
            self.t = 2
        else:
            self.t = 3 + s / a

    def number_of_levels(self, eps: float) -> int:
        b, g, s, a = self.beta, self.gamma, self.sigma, self.alpha
        case = g / b - s / a
        if case > 0:
            L = - (g + b) / b * np.log(eps)
        elif case < 0:
            L = - (s + a) / a * np.log(eps)
        else:
            grid = np.linspace(1, 100, 1000)
            ix = np.argmin(np.abs(np.exp(-grid * a / (s + a)) * (grid + 1) - eps))
            L = grid[ix]
        self.L = np.ceil(L).astype(int)
        return self.L

    def tuning_params(self, L: int):
        b, g, s, a = self.beta, self.gamma, self.sigma, self.alpha

        self.mk = self.C_m * np.exp(np.arange(L + 1) / (s + a))
        self.nl = self.C_n * np.exp(np.arange(L + 1) / (g + b))

        self.Ns = self.sample_sizes(self.mk)

        self.mk = np.ceil(self.mk).astype(int)
        self.nl = np.ceil(self.nl).astype(int)

        return self.mk, self.nl

    def sample_sizes(self, mk):
        r = 1
        kappa = (1 - np.log(2)) / (1 + r) / 2
        aux = mk ** self.sigma
        Ns = np.real(np.exp(-lambertw(-kappa / aux, k=-1)))
        Ns = np.ceil(((1 - self.reduce_sample_by) * Ns)).astype(int)
        return Ns

    def work(self) -> float:
        gamma = self.gamma
        Ns = self.Ns
        nl = self.nl
        if self.reuse_sample:
            work = (Ns[::-1] * nl ** gamma).sum()
        else:
            work = Ns[-1] * nl[0] ** gamma + (Ns[:-1][::-1] * (nl[1:] ** gamma + nl[:-1] ** gamma)).sum()
        return work

    def log(self, level: Union[str, int], runtime: Optional[float]=None):
        if level == "start":
            self.v_print = print if self.verbose else lambda *a, **k: None
            w = [max(map(len, map(str, arr))) for arr in [self.mk, self.nl, self.Ns]]
            self.v_print(
                f"eps = {self.eps:.2e}, L = {self.L}, reduce = {100 * self.reduce_sample_by}%, reuse_sample = {self.reuse_sample}"
            )
            self.v_print(f"{'Level':>5} | {'m':>{w[0]}} | {'n':>{w[1]}} | {'N':>{w[2]}} | {'time':>7}")
            self.v_print("-" * (24 + sum(w)))
            self.print_widths = w
        elif level == "end":
            self.v_print(f"Total runtime: {runtime:.3f}s\n")
        else:
            w = self.print_widths
            m = self.mk[self.L - level]
            n = self.nl[level]
            N = self.Ns[self.L - level]
            self.v_print(f"{level:>5} | {m:>{w[0]}} | {n:>{w[1]}} | {N:>{w[2]}} | {runtime:>6.3f}s")

    def combine_coefs(self):
        coef = defaultdict(float)
        for d in [p.coef_ for p in self.projectors]:
            for index, v in d.items():
                coef[index] += v
        self.coef_ = dict(coef)

    def fit(self, eps: float) -> MultiLevelLSQ:
        start = time.perf_counter()

        self.eps = eps

        L = self.number_of_levels(eps)
        mk, nl = self.tuning_params(L)
        Ns = self.Ns

        self.log(level="start")

        if self.reuse_sample:
            I = PolySpace(self.poly_space, max(mk), self.dim).index_set
            sample = sample_optimal_distribution(I, max(Ns))
            sample = unit_to_domain(sample, self.domain)

        self.projectors = []
        self.times = []
        for l in range(L + 1):
            start_l = time.perf_counter()

            m = mk[L - l]
            n = nl[l]
            N = Ns[L - l]

            V = PolySpace(self.poly_space, m, self.dim)
            projector = SingleLevelLSQ(
                {
                    "poly_space": V,
                    "domain": self.domain,
                }
            )
            if not self.reuse_sample:
                f_l = self.response(n)
                f_l_ = self.response(nl[l - 1]) if l > 0 else lambda _: 0
                f_ = lambda g: f_l(g) - f_l_(g)
                projector.fit(f_, N=N)
            else:
                sample_l = sample[:N]
                f_l = self.response(n)
                f_l__ = f_l_[:N] if l > 0 else np.zeros(N)
                f_l_ = f_l(sample_l)
                projector.fit(f_l_ - f_l__, sample=sample_l)

            self.projectors.append(projector)

            self.times.append(time.perf_counter() - start_l)
            self.log(l, runtime=self.times[l])

        self.combine_coefs()
        self.log("end", runtime=time.perf_counter() - start)

        return self

    def __call__(self, x: np.array) -> np.array:
        try:
            return call(self.coef_, x, self.domain)
        except:
            raise ValueError("Model was not fitted yet.")

    def l2_error(self, x: np.array, y: np.array) -> float:
        return np.sqrt(mse(self(x), y))
