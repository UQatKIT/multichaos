import numpy as np
import time

from itertools import product
from typing import Callable

from lsq import get_optimal_sample_size
from lsq import LSQ

from tqdm.notebook import tqdm


def is_downward_closed(I) -> bool:
    I_set = set(I)
    for eta in I:
        for j, k in enumerate(eta):
            eta_ = list(eta)
            if k - 1 >= 0:
                eta_[j] -= 1
                if tuple(eta_) not in I_set:
                    return False
    return True

def get_neighbours(eta, extending=False):
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

def admissible_multi_indices(I):
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

def seperate_index(eta):
    return eta[:-1], eta[-1]

def get_power_of_two_index_set(eta):
    low = tuple(2 ** k - 1 for k in eta)
    high = tuple(2 ** (k + 1) - 1 for k in eta)
    I = list(product(*[range(l, h) for (l, h) in zip(low, high)]))
    return I

def slice_index_set(I, l):
    return [eta for eta in I if seperate_index(eta)[1] == l]

def estimate_work(f, l, d):
    f_ = lambda x: f(l)(x) - f(l-1)(x) if l > 0 else f(l)(x)
    x = np.random.rand(1, d)
    reps = 10
    s = time.perf_counter()
    for _ in range(reps):
        f_(x)
    e = time.perf_counter()

    return (e - s) / reps

def construct_multi_index_set(f: Callable, n_steps: int, d: int):
    I = [(0,) * (d + 1)]

    for i in range(n_steps):
        print(f"------- STEP {i} -------")
        print(f"Current I: {I}")
        A = admissible_multi_indices(I)
        print(f"A = {A}")
        print(f"Number of admissible multi-indices: {len(A)}")
        ratios = {}
        for eta in tqdm(A):

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

                f_l = f(l)
                f_l_ = f(l - 1) if l > 0 else lambda _: 0
                f_ = lambda x: f_l(x) - f_l_(x)
                model = LSQ(V).solve(f_)

                gain += np.linalg.norm(
                    [v for eta, v in zip(model.I, model.coef_) if eta in under_consideration]
                )
            gain /= len(neighbours)

            # WORK ESTIMATION
            k, l = seperate_index(eta)
            n_ = get_optimal_sample_size(slice_index_set(I, l) + [k])
            n =  get_optimal_sample_size(slice_index_set(I, l))
            est_work = estimate_work(f, l, d)
            work = (n_ - n) * est_work

            ratios[eta] = gain / work

        I.append(max(ratios, key=ratios.get))

    return I

def adaptive_ml_algorithm(f: Callable, n_steps: int, d: int):
    print("Start constructing multi-index set ...")
    I = construct_multi_index_set(f, n_steps, d)

    L = max(eta[-1] for eta in I)

    projectors = []
    for l in range(L + 1):
        print(f"LEVEL l = {l}")
        V = []
        for eta in slice_index_set(I, l):
            k, _ = seperate_index(eta)
            V.extend(get_power_of_two_index_set(k))
        f_ = lambda x: f(l)(x) - f(l - 1)(x) if l > 0 else f(l)(x)
        level_l_projector = LSQ(V).solve(f_)
        projectors.append(level_l_projector)

    return projectors
