import numpy as np
import time

from itertools import product
from typing import Callable, Union

from polynomial_spaces import PolySpace
from sampling import sample_arcsine
from least_squares import get_optimal_sample_size
from least_squares import SingleLevelLSQ
from utils import mse


IndexSet = list[Union[int, tuple[int, ...]]]


def is_downward_closed(I: IndexSet) -> bool:
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

def get_lower_neighbours(eta: tuple[int, ...]):
    """
    Returns the lower neighbours of `eta`, where a neighbour is defined
    as a tuple that differs in exactly one entry by one.
    """
    for i, k in enumerate(eta):
        if k - 1 >= 0:
            eta_down = eta[:i] + (k - 1,) + eta[i+1:]
            yield eta_down

def get_upper_neighbours(eta: tuple[int, ...]):
    """
    Returns the upper neighbours of `eta`, where a neighbour is defined
    as a tuple that differs in exactly one entry by one.
    """
    for i, k in enumerate(eta):
        eta_up = eta[:i] + (k + 1,) + eta[i+1:]
        yield eta_up

def seperate_index(eta: tuple[int, ...]) -> tuple[tuple[int, ...], int]:
    """Seperates an index `(k, l)` into `k` and `l`."""
    return eta[:-1], eta[-1]

def get_power_of_two_index_set(eta: tuple[int, ...]) -> IndexSet:
    """
    Returns all multi-indices `k` s.t. the condition
    `2 ^ k - 1 <= eta < 2 ^ (k+1) - 1` is fulfilled.
    """
    low = tuple(int(np.ceil(2 ** k)) - 1 for k in eta)
    high = tuple(int(np.ceil(2 ** (k + 1))) - 1 for k in eta)
    I = list(product(*[range(l, h) for (l, h) in zip(low, high)]))
    return I

def slice_index_set(I: IndexSet, l: int) -> IndexSet:
    """Returns all multi-indices `(k, s)` in `I` for which `s = l`."""
    return [eta for eta in I if seperate_index(eta)[1] == l]

def sum_times(times: dict[int, list[tuple[float, int]]]) -> dict[int, float]:
    """
    Calculates the mean of computation times given by
    a list of tuples of times and number of samples.
    """
    sums_ = {n: np.sum(t, axis=0) for n, t in times.items()}
    means_ = {n: t / reps for n, (t, reps) in sums_.items()}
    return means_

def polynomial_space_V(I, l):
    """
    Constructs the polynomial space `V` which is a partition given by
    exponential index sets defined by the slice `I_l`.
    """
    V = []
    for eta in slice_index_set(I, l):
        k, _ = seperate_index(eta)
        V.extend(get_power_of_two_index_set(k))
    return V

class AdaptiveLSQ:
    def __init__(self, f=None, n_steps=None, d=None):
        self.f = f
        self.d = d
        self.n_steps = n_steps
        pass

    def __call__(self, x: np.array) -> np.array:
        try:
            projectors = self.projectors_
        except:
            raise ValueError("Model was not fitted yet.")
        else:
            return sum(p(x) for p in projectors)

    def estimate_gain(self, index):
        """
        Estimates the gain of the index `(k, l)` by calculating the
        norm of the projection of `f_l - f_{l-1}` onto `P_k`.
        """
        neighbours = list(get_lower_neighbours(index))

        gain = 0
        for nbr in neighbours:
            if nbr in self.gains:
                gain += self.gains[nbr]
                continue

            k, l = seperate_index(nbr)

            V = polynomial_space_V(self.I, l)
            N = get_optimal_sample_size(V, sampling="arcsine")
            N = int((1 - self.reduce) * N)

            if N > len(self.sample):
                self.sample = np.vstack(
                    (self.sample, sample_arcsine((N - len(self.sample), self.d)))
                )

            evals = self.f_vals.get(l, np.empty(0))
            times = self.f_times.get(l, np.empty((0, 2)))

            if N > len(evals):
                s = time.perf_counter()
                new_evals = self.f(l)(self.sample[len(evals):N])
                e = time.perf_counter()

                self.f_vals[l] = np.hstack((evals, new_evals))
                self.f_times[l] = np.vstack((times, np.array([e - s, N - len(evals)])))

            sample = self.sample[:N]
            f = self.f_vals[l][:N] - self.f_vals[l-1][:N] if l > 0 else self.f_vals[l][:N]
            space = PolySpace("TD", m=1, d=1)
            space.index_set = V

            model = SingleLevelLSQ(space, sampling="arcsine").fit(f, sample=sample.squeeze())

            under_consideration = get_power_of_two_index_set(k)
            self.gains[nbr] = np.linalg.norm(
                [v for eta, v in zip(V, model.coef_) if eta in under_consideration]
            )
            gain += self.gains[nbr]
        gain /= len(neighbours)

        return gain

    def estimate_work(self, index):
        """
        Estimates the work of the index `(k, l)` by measuring the time of
        computing `f_l - f_{l-1}` and the number of new samples needed.
        """
        k, l = seperate_index(index)

        if l not in self.f_times:
            s = time.perf_counter()
            new_evals = self.f(l)(self.sample[0].reshape(1, -1))
            e = time.perf_counter()

            self.f_vals[l] = new_evals
            self.f_times[l] = np.expand_dims(np.array([e - s, 1]), axis=0)

        times = sum_times(self.f_times)

        V = polynomial_space_V(self.I, l)
        n_add = get_optimal_sample_size(V + get_power_of_two_index_set(k), sampling="arcsine")
        n =  get_optimal_sample_size(V, sampling="arcsine")

        work = times[l] + times[l - 1] if l > 0 else times[l]
        work *= n_add - n

        return work

    def construct_multi_index_set(self):
        """
        Constructs a problem-dependent, downward closed multi-index set `I` (see Haji-Ali et al.).
        """
        self.I = [(0,) * (self.d + 1)]
        A = [(0,) * i + (1,) + (0,) * (self.d - i) for i in range(self.d + 1)]

        self.sample = np.empty((0, self.d))
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
            for nbr in get_upper_neighbours(best_eta):
                if is_downward_closed(self.I + [nbr]):
                    A.append(nbr)

        return

    def print_start(self, mk: np.array, nl: np.array, Ns: np.array, reduce=.9) -> list[int]:
        w = [max(map(len, map(str, arr))) for arr in [mk, nl, Ns]]
        w[0] = max(w[0], len("dim(V)"))
        verboseprint(f"n_steps = {self.n_steps}, L = {self.L}, reduce = {100 * reduce}%")
        verboseprint(f"{'Level':>5} | {'dim(V)':>{w[0]}} | {'n':>{w[1]}} | {'N':>{w[2]}}")
        verboseprint("-" * (14 + sum(w)))
        return w

    def print_level(self, l: int, m: int, n: int, N: int, w: list[int]):
        verboseprint(f"{l:>5} | {m:>{w[0]}} | {n:>{w[1]}} | {N:>{w[2]}}")

    def print_end(self, time):
        verboseprint(f"TOTAL TIME: {time:.3f}s\n")

    def fit(self, f: Callable, n_steps: int, d: int, reduce: float=0., verbose: bool=True):
        self.f = f
        self.d = d
        self.n_steps = n_steps
        self.reduce = reduce
        self.verbose = verbose

        global verboseprint
        verboseprint = print if self.verbose else lambda *a, **k: None

        s_total = time.perf_counter()

        self.construct_multi_index_set()

        self.L = max(eta[-1] for eta in self.I)

        Vk = [polynomial_space_V(self.I, l) for l in range(self.L + 1)]
        self.mk = [len(V) for V in Vk]
        self.nl = [10 * 2 ** l for l in range(self.L + 1)]
        self.Ns = [get_optimal_sample_size(V, sampling="arcsine") for V in Vk]
        self.Ns = [int((1 - self.reduce) * N) for N in self.Ns]

        w = self.print_start(self.mk, self.nl, self.Ns)

        self.projectors_ = []
        for l in range(self.L + 1):
            V = Vk[l]
            N = self.Ns[l]

            if N > len(self.sample):
                self.sample = np.vstack(
                    (self.sample, sample_arcsine((N - len(self.sample), d)))
                )

            for l_ in [l, l - 1] if l > 0 else [l]:
                evals = self.f_vals.get(l_, np.empty(0))
                if N > evals.size:
                    new_evals = f(l_)(self.sample[evals.size:N])
                    self.f_vals[l_] = np.hstack((evals, new_evals))

            sample = self.sample[:N]
            f_ = self.f_vals[l][:N] - self.f_vals[l-1][:N] if l > 0 else self.f_vals[l][:N]
            space = PolySpace("TD", m=1, d=1)
            space.index_set = V

            level_l_projector = SingleLevelLSQ(space, sampling="arcsine").fit(f_, sample=sample.squeeze())
            self.projectors_.append(level_l_projector)

            self.print_level(l, self.mk[l], self.nl[l], N, w)

        e_total = time.perf_counter()
        self.print_end(e_total - s_total)

        return self

    def l2_error(self, sample: np.array, values: np.array) -> float:
        """
        Calculates the L2 error on a grid given by the
        cartesian product of 1-D arrays in `sample`.
        """
        return np.sqrt(mse(self(sample), values))