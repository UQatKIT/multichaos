"""
Adaptive multilevel weighted polynomial least squares.
"""
from __future__ import annotations
import numpy as np
import time

from collections import defaultdict
from itertools import product
from typing import Optional, Union

from polynomial_spaces import PolySpace
from sampling import sample_arcsine
from least_squares import get_optimal_sample_size
from least_squares import evaluate_basis
from least_squares import SingleLevelLSQ
from utils import mse


IndexSet = list[Union[int, tuple[int, ...]]]

class AdaptiveLSQ:
    def __init__(self, problem: dict, params: dict):
        self.problem = problem
        self.params = params

        self.response = problem.get("response", None)
        self.dim = problem.get("dim", None)

        self.C_n = params.get("C_n", 1.)
        self.n_pow = params.get("n_pow", 2.)

        self.reduce_sample_by = params.get("reduce_sample_by", .0)
        self.verbose = params.get("verbose", True)

        self.reparametrization()

    def reparametrization(self):
        self.n = lambda l: int(self.C_n * np.ceil(self.n_pow ** l))

    def is_downward_closed(self, I: IndexSet) -> bool:
        """Checks if the given index set `I` is downward closed."""
        I_set = set(I)
        for eta in I:
            for j, k in enumerate(eta):
                eta_ = list(eta)
                if k - 1 >= 0:
                    eta_[j] -= 1
                    if tuple(eta_) not in I_set:
                        return False
        return True

    def get_lower_neighbours(self, eta: tuple[int]):
        """
        Returns the lower neighbours of `eta`, where a neighbour is defined
        as a tuple that differs in exactly one entry by one.
        """
        for i, k in enumerate(eta):
            if k - 1 >= 0:
                eta_down = eta[:i] + (k - 1,) + eta[i+1:]
                yield eta_down

    def get_upper_neighbours(self, eta: tuple[int, ...]):
        """
        Returns the upper neighbours of `eta`, where a neighbour is defined
        as a tuple that differs in exactly one entry by one.
        """
        for i, k in enumerate(eta):
            eta_up = eta[:i] + (k + 1,) + eta[i+1:]
            yield eta_up

    def separate_index(self, eta: tuple[int, ...]) -> tuple[tuple[int, ...], int]:
        """Separates an index `(k, l)` into `k` and `l`."""
        return eta[:-1], eta[-1]

    def get_power_of_two_index_set(self, eta: tuple[int, ...]) -> IndexSet:
        """
        Returns all multi-indices `k` s.t. the condition
        `2 ^ k - 1 <= eta < 2 ^ (k+1) - 1` is fulfilled.
        """
        low = tuple(int(np.ceil(2 ** k)) - 1 for k in eta)
        high = tuple(int(np.ceil(2 ** (k + 1))) - 1 for k in eta)
        I = list(product(*[range(l, h) for (l, h) in zip(low, high)]))
        return I

    def slice_index_set(self, I: IndexSet, l: int) -> IndexSet:
        """Returns all multi-indices `(k, s)` in `I` for which `s = l`."""
        return [eta for eta in I if self.separate_index(eta)[1] == l]

    def sum_times(self, times: dict[int, list[tuple[float, int]]]) -> dict[int, float]:
        """
        Calculates the mean of computation times given by
        a list of tuples of times and number of samples.
        """
        sums_ = {n: np.sum(t, axis=0) for n, t in times.items()}
        means_ = {n: t / reps for n, (t, reps) in sums_.items()}
        return means_

    def polynomial_space_V(self, I, l):
        """
        Constructs the polynomial space `V` which is a partition given by
        exponential index sets defined by the slice `I_l`.
        """
        V = []
        for eta in self.slice_index_set(I, l):
            k, _ = self.separate_index(eta)
            V.extend(self.get_power_of_two_index_set(k))
        return V

    def log(self, level: Union[str, int], runtime: Optional[float]=None):
        if level == "start_constructing":
            print("Constructing multi-index set ... ", end="", flush=True)
        elif level == "end_constructing":
            print(f"Done in {runtime:.3f}s")
        elif level == "start":
            self.v_print = print if self.verbose else lambda *a, **k: None
            w = [max(map(len, map(str, arr))) for arr in [self.mk, self.nl, self.Ns]]
            w[0] = max(w[0], len("dim(V)"))
            self.v_print(
                f"n_steps = {self.n_steps}, L = {self.L}, reduce = {100 * self.reduce_sample_by}%"
            )
            self.v_print(f"{'Level':>5} | {'dim(V)':>{w[0]}} | {'n':>{w[1]}} | {'N':>{w[2]}} | {'time':>7}")
            self.v_print("-" * (24 + sum(w)))
            self.print_widths = w
        elif level == "end":
            self.v_print(f"Total runtime: {runtime:.3f}s\n")
        else:
            w = self.print_widths
            m = self.mk[level]
            n = self.nl[level]
            N = self.Ns[self.L - level]
            self.v_print(f"{level:>5} | {m:>{w[0]}} | {n:>{w[1]}} | {N:>{w[2]}} | {runtime:>6.3f}s")

    def estimate_gain(self, index):
        """
        Estimates the gain of the index `(k, l)` by calculating the
        norm of the projection of `f_l - f_{l-1}` onto `P_k`.
        """
        neighbours = list(self.get_lower_neighbours(index))

        gain = 0
        for nbr in neighbours:
            if nbr in self.gains:
                gain += self.gains[nbr]
                continue

            k, l = self.separate_index(nbr)

            V = self.polynomial_space_V(self.I, l)
            N = get_optimal_sample_size(V, sampling="arcsine")
            N = int((1 - self.reduce_sample_by) * N)

            if N > len(self.sample):
                self.sample = np.vstack(
                    (self.sample, sample_arcsine((N - len(self.sample), self.dim)))
                )

            evals = self.f_vals.get(l, np.empty(0))
            times = self.f_times.get(l, np.empty((0, 2)))

            if N > len(evals):
                s = time.perf_counter()
                new_evals = self.response(self.n(l))(self.sample[len(evals):N])
                e = time.perf_counter()

                self.f_vals[l] = np.hstack((evals, new_evals))
                self.f_times[l] = np.vstack((times, np.array([e - s, N - len(evals)])))

            sample = self.sample[:N]
            f = self.f_vals[l][:N] - self.f_vals[l-1][:N] if l > 0 else self.f_vals[l][:N]
            space = PolySpace("TD", m=1, d=1)
            space.index_set = V

            model = SingleLevelLSQ(space, sampling="arcsine").fit(f, sample=sample.squeeze())

            under_consideration = self.get_power_of_two_index_set(k)
            self.gains[nbr] = np.linalg.norm(
                [v for eta, v in model.coef_.items() if eta in under_consideration]
            )
            gain += self.gains[nbr]
        gain /= len(neighbours)

        return gain

    def estimate_work(self, index):
        """
        Estimates the work of the index `(k, l)` by measuring the time of
        computing `f_l - f_{l-1}` and the number of new samples needed.
        """
        k, l = self.separate_index(index)

        if l not in self.f_times:
            s = time.perf_counter()
            new_evals = self.response(self.n(l))(self.sample[0].reshape(1, -1))
            e = time.perf_counter()

            self.f_vals[l] = new_evals
            self.f_times[l] = np.expand_dims(np.array([e - s, 1]), axis=0)

        times = self.sum_times(self.f_times)

        V = self.polynomial_space_V(self.I, l)
        n_add = get_optimal_sample_size(V + self.get_power_of_two_index_set(k), sampling="arcsine")
        n =  get_optimal_sample_size(V, sampling="arcsine")

        work = times[l] + times[l - 1] if l > 0 else times[l]
        work *= n_add - n

        return work

    def construct_multi_index_set(self):
        """
        Constructs a problem-dependent, downward closed multi-index set `I` (see Haji-Ali et al.).
        """
        self.I = [(0,) * (self.dim + 1)]
        A = [(0,) * i + (1,) + (0,) * (self.dim - i) for i in range(self.dim + 1)]

        self.sample = np.empty((0, self.dim))
        self.f_vals = {}
        self.f_times = {}

        self.gains = {}
        self.ratios = {}

        for _ in range(self.n_steps):
            ratios = {}
            for eta in A:
                gain = self.estimate_gain(eta)
                work = self.estimate_work(eta)

                ratios[eta] = gain / work
                self.ratios[eta] = gain / work

            best_eta = max(ratios, key=ratios.get)
            self.I.append(best_eta)

            A.remove(best_eta)
            for nbr in self.get_upper_neighbours(best_eta):
                if self.is_downward_closed(self.I + [nbr]):
                    A.append(nbr)

        return

    def combine_coefs(self):
        coef = defaultdict(float)
        for d in [p.coef_ for p in self.projectors]:
            for index, v in d.items():
                coef[index] += v
        self.coef_ = dict(coef)

    def fit(self, n_steps: int=None, I: IndexSet=None) -> AdaptiveLSQ:
            start = time.perf_counter()

            self.n_steps = n_steps

            if I is None:
                self.log(level="start_constructing")
                s = time.perf_counter()
                self.construct_multi_index_set()
                self.time_constructing = time.perf_counter() - s
                self.log(level="end_constructing", runtime=self.time_constructing)
            else:
                self.I = I

            self.L = max(eta[-1] for eta in self.I)

            self.Vk = [self.polynomial_space_V(self.I, l) for l in range(self.L + 1)]
            self.mk = list(map(len, self.Vk))
            self.nl = list(map(self.n, range(self.L + 1)))
            self.Ns = [
                int((1. - self.reduce_sample_by) * get_optimal_sample_size(V, sampling="arcsine")) for V in self.Vk
            ]

            self.log(level="start")

            self.projectors = []
            self.times = []
            for l in range(self.L + 1):
                start_l = time.perf_counter()

                V = self.Vk[l]
                N = self.Ns[l]

                if N > len(self.sample):
                    self.sample = np.vstack(
                        (self.sample, sample_arcsine((N - len(self.sample), self.dim)))
                    )
                for l_ in [l, l - 1] if l > 0 else [l]:
                    evals = self.f_vals.get(l_, np.empty(0))
                    if N > evals.size:
                        new_evals = self.response(self.n(l_))(self.sample[evals.size:N])
                        self.f_vals[l_] = np.hstack((evals, new_evals))

                sample = self.sample[:N]
                f_ = self.f_vals[l][:N] - self.f_vals[l-1][:N] if l > 0 else self.f_vals[l][:N]
                dummy_space = PolySpace("TP", m=1, d=1)
                dummy_space.index_set = V

                projector = SingleLevelLSQ(dummy_space, sampling="arcsine").fit(f_, sample=sample.squeeze())
                self.projectors.append(projector)

                self.times.append(time.perf_counter() - start_l)
                self.log(l, runtime=self.times[l])

            self.combine_coefs()
            self.log("end", runtime=time.perf_counter() - start)

            return self

    def __call__(self, x: np.array) -> np.array:
        try:
            coef = self.coef_
        except:
            raise ValueError("Model was not fitted yet.")
        else:
            I = list(self.coef_.keys())
            coef = np.array(list(self.coef_.values()))
            basis = evaluate_basis(I, x)
            return np.dot(coef, basis)

    def l2_error(self, x: np.array, y: np.array) -> float:
        return np.sqrt(mse(self(x), y))
