from __future__ import annotations
import numpy as np
import time

from itertools import product
from typing import Callable, Union

from sampling import sample_arcsine
from lsq import get_optimal_sample_size
from lsq import LSQ


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

def get_neighbours(eta: tuple[int, ...], extending=False) -> IndexSet:
    """
    Returns the neighbours of `eta`, where a neighbour is defined
    as a tuple that differs in exactly one entry by one.
    """
    neighbours = []
    for i, k in enumerate(eta):
        eta_up, eta_down = list(eta), list(eta)
        eta_up[i] += 1
        neighbours.append(tuple(eta_up))
        if not extending:
            if k - 1 >= 0:
                eta_down[i] -= 1
                neighbours.append(tuple(eta_down))
    return neighbours

def admissible_multi_indices(I: IndexSet) -> IndexSet:
    """
    Returns the admissible multi-indices of `I`,
    the neighbours s.t. the index set is still downward closed.
    """
    indices = []
    for eta in I:
        neighbours = get_neighbours(eta, extending=True)
        for neighbour in neighbours:
            if neighbour in I:
                continue
            I_ = I.copy()
            I_.append(neighbour)
            if is_downward_closed(I_):
                indices.append(neighbour)
    return list(set(indices))

def seperate_index(eta: tuple[int, ...]) -> tuple[tuple[int, ...], int]:
    """Seperates an index `(k, l)` into `k` and `l`."""
    return eta[:-1], eta[-1]

def get_power_of_two_index_set(eta: tuple[int, ...]) -> IndexSet:
    """
    Returns all multi-indices `k` s.t. the condition
    `2 ^ k - 1 <= eta < 2 ^ (k+1) - 1` is fulfilled.
    """
    low = tuple(2 ** k - 1 for k in eta)
    high = tuple(2 ** (k + 1) - 1 for k in eta)
    I = list(product(*[range(l, h) for (l, h) in zip(low, high)]))
    return I


def slice_index_set(I: IndexSet, l: int) -> IndexSet:
    """Returns all multi-indices `(k, s)` in `I` for which `s = l`."""
    return [eta for eta in I if seperate_index(eta)[1] == l]

def estimate_work(f: Callable, d: int) -> tuple[float, int]:
    """Measures the computation time for 50 evaluations of `f`."""
    reps = 50
    x = np.random.rand(reps, d)

    s = time.perf_counter()
    _ = f(x)
    e = time.perf_counter()

    return e - s, reps

def sum_times(times: dict[int, list[tuple[float, int]]]) -> dict[int, float]:
    """
    Calculates the mean of computation times given by
    a list of tuples of times and number of samples.
    """
    sums_ = {n: [sum(tup) for tup in zip(*t)] for n, t in times.items()}
    means_ = {n: t / n_elem for n, (t, n_elem) in sums_.items()}
    return means_

def construct_multi_index_set(f: Callable, n_steps: int, d: int) -> IndexSet:
    """
    Constructs a problem-dependent, downward closed multi-index set `I` (see Haji-Ali et al.).
    """
    I = [(0,) * (d + 1)]

    func_evals = {}
    func_eval_times = {}
    sample = np.empty((0, d))

    for _ in range(n_steps):
        A = admissible_multi_indices(I)
        ratios = {}
        for eta in A:

            # GAIN ESTIMATION
            gain = 0
            neighbours = list(set(I).intersection(set(get_neighbours(eta))))
            for neighbour in neighbours:
                k, l = seperate_index(neighbour)
                under_consideration = get_power_of_two_index_set(k)

                V = []
                for eta_ in slice_index_set(I, l):
                    k_, _ = seperate_index(eta_)
                    V.extend(get_power_of_two_index_set(k_))

                N = get_optimal_sample_size(V, sampling="arcsine")

                if N > len(sample):
                    sample = np.vstack((sample, sample_arcsine((N - len(sample), d))))

                if l in func_evals:
                    evals = func_evals[l]
                    if N > len(evals):
                        s = time.perf_counter()
                        new_evals = f(l)(sample[len(evals):N])
                        e = time.perf_counter()
                        func_evals[l] = np.hstack((evals, new_evals))
                        func_eval_times[l].append((e - s, N - len(evals)))
                else:
                    s = time.perf_counter()
                    new_evals = f(l)(sample[:N])
                    e = time.perf_counter()
                    func_evals[l] = new_evals
                    func_eval_times[l] = [(e - s, N)]

                f_ = lambda _: func_evals[l][:N] - func_evals[l - 1][:N] if l > 0 else func_evals[l][:N]
                model = LSQ(V, sampling="arcsine").solve(f_, sample=sample[:N])

                gain += np.linalg.norm(
                    [v for eta, v in zip(model.I, model.coef_) if eta in under_consideration]
                )
            gain /= len(neighbours)

            # WORK ESTIMATION
            k, l = seperate_index(eta)
            if l not in func_eval_times:
                func_eval_times[l] = [estimate_work(f(l), d)]

            times = sum_times(func_eval_times)

            I_l = slice_index_set(I, l)
            n_add = get_optimal_sample_size(I_l + [k], sampling="arcsine")
            n =  get_optimal_sample_size(I_l, sampling="arcsine")

            work = times[l] - times[l - 1] if l > 0 else times[l]
            work *= n_add - n

            ratios[eta] = gain / work

        I.append(max(ratios, key=ratios.get))

    return I

class AD_ML_LSQ:
    def __init__(self):
        self.I = None

    def __call__(self, x: np.array, grid: bool=False) -> np.array:
        try:
            projectors = self.projectors_
        except:
            raise ValueError("Model was not fitted yet.")
        else:
            if grid:
                return sum(p(x, grid=True) for p in projectors)
            else:
                return sum(p(x) for p in projectors)

    def solve(self, f: Callable, n_steps: int, d: int) -> AD_ML_LSQ:
        if self.I is None:
            self.I = construct_multi_index_set(f, n_steps, d)

        L = max(eta[-1] for eta in self.I)

        self.projectors_ = []
        for l in range(L + 1):
            V = []
            for eta in slice_index_set(self.I, l):
                k, _ = seperate_index(eta)
                V.extend(get_power_of_two_index_set(k))

            N = get_optimal_sample_size(V, sampling="arcsine")
            sample = sample[:N] if l > 0 else sample_arcsine((N, d))

            f_l = f(l)
            f_l__ = f_l_[:N] if l > 0 else np.zeros(N)  # f_{-1} := 0
            f_l_ = f_l(sample)
            f_ = lambda _: f_l_ - f_l__

            level_l_projector = LSQ(V, sampling="arcsine").solve(f_, sample=sample)
            self.projectors_.append(level_l_projector)

        return self

    def l2_error(self, sample: list[np.array], values: np.array) -> float:
        """
        Calculates the L2 error on a grid given by the
        cartesian product of 1-D arrays in `sample`.
        """
        return np.sqrt(((self(sample, grid=True) - values) ** 2).mean())
