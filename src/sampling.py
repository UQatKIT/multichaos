"""
Sampling methods for the weighted polynomial least squares projection.
"""
import numpy as np

from collections import defaultdict

from scipy.special import eval_sh_legendre
from scipy.special import lambertw
from typing import Literal, Union
from utils import domain_to_unit


IndexSet = list[Union[int, tuple[int, ...]]]
SamplingMode = Literal["optimal", "arcsine"]


def arcsine(x: np.array, domain: list[tuple]) -> np.array:
    x = domain_to_unit(x, domain)
    aux = 1 / np.sqrt(x * (1 - x)) / np.pi
    return aux.prod(axis=1) if len(x.shape) > 1 else aux

def sample_arcsine(size: tuple[int, ...]) -> np.array:
    u = np.random.uniform(-np.pi/2, np.pi/2, size=size)
    return (np.sin(u) + 1) / 2

def rejection_sampling(n: int, size: int) -> np.array:
    """
    Samples from the distribution with squared
    univariate Lebesgue density of degree `n`.
    """
    samples = []

    while len(samples) < size:
        arcs = sample_arcsine(size)
        aux = (2 * n + 1) * eval_sh_legendre(n, arcs) ** 2 / 4 / np.e / arcsine(arcs, [(0, 1)])
        U = np.random.uniform(size=size)
        samples.extend(arcs[U <= aux])

    return np.array(samples[:size])

def sample_optimal_distribution_uni(I: IndexSet, size: int) -> np.array:
    """
    Samples from the optimal distribution in the univariate case.
    """
    choices = np.random.choice(I, size)
    count_choices = np.bincount(choices)
    
    samples = []
    for n, count in enumerate(count_choices):
        samples.append(rejection_sampling(n, count))

    return np.hstack(samples)

def sample_optimal_distribution(I: IndexSet, size: int) -> np.array:
    """
    Samples from the optimal distribution.
    """
    d = 1 if isinstance(I[0], (int, np.int64)) else len(I[0])
    if d == 1:
        return sample_optimal_distribution_uni(I, size)
    
    choices = np.random.randint(len(I), size=size)
    count_choices = np.bincount(choices)
    
    count_degrees = defaultdict(int)
    for eta, count in zip(I, count_choices):
        for k in eta:
            count_degrees[k] += count

    samples_uni = {}
    for k, count in count_degrees.items():
        samples_uni[k] = rejection_sampling(k, count)

    ind = np.insert(count_choices.cumsum(), 0, 0)

    samples = np.zeros((size, d))
    for i in np.nonzero(count_choices)[0]:
        for j, k in enumerate(I[i]):
            samples[ind[i]:ind[i + 1], j] = samples_uni[k][-count_choices[i]:]
            samples_uni[k] = np.delete(samples_uni[k], range(-count_choices[i], 0))

    return samples

def optimal_sample_size(I: IndexSet, sampling: SamplingMode, r: float=1., reduce: float=0.) -> int:
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
    N = np.ceil(np.real(np.exp(-lambertw(- kappa / aux, k=-1))))
    N *= 1 - reduce

    return int(N)
