"""
Sampling methods for the weighted polynomial least squares projection.
"""
import numpy as np

from scipy.special import eval_sh_legendre
from typing import Union


IndexSet = list[Union[int, tuple[int, ...]]]


def arcsine(x: np.array) -> np.array:
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
        aux = (2 * n + 1) * eval_sh_legendre(n, arcs) ** 2 / 4 / np.e / arcsine(arcs)
        U = np.random.uniform(size=size)
        samples.extend(arcs[U <= aux])

    return np.array(samples[:size])

def sample_optimal_distribution_uni(I: IndexSet, size: int) -> np.array:
    """
    Samples from the optimal distribution in the univariate case.
    """
    choices = np.random.choice(I, size)
    count_choices = np.bincount(choices)
    
    samples = np.zeros(size)
    for n, count in enumerate(count_choices):
        sample = rejection_sampling(n, count)
        start, end = count_choices[:n].sum(), count_choices[:n+1].sum()
        samples[start:end] = sample
    
    return samples

def sample_optimal_distribution(I: IndexSet, size: int) -> np.array:
    """
    Samples from the optimal distribution.
    """
    d = 1 if isinstance(I[0], int) else len(I[0])
    if d == 1:
        return sample_optimal_distribution_uni(I, size)
    
    choices = np.random.randint(len(I), size=size)
    count_choices = np.bincount(choices)
    
    count_degrees = {}
    for eta, count in zip(I, count_choices):
        for k in eta:
            if k in count_degrees:
                count_degrees[k] += count
            else:
                count_degrees[k] = count
    
    samples_uni = {}
    for k, count in count_degrees.items():
        samples_uni[k] = rejection_sampling(k, count)
    
    samples = np.zeros((size, d))
    for i, count in enumerate(count_choices):
        if count == 0:
            continue
        start, end = count_choices[:i].sum(), count_choices[:i+1].sum()
        for j, k in enumerate(I[i]):
            samples[start:end, j] = samples_uni[k][-count_choices[i]:]
            samples_uni[k] = np.delete(samples_uni[k], range(-count_choices[i], 0))

    return samples
