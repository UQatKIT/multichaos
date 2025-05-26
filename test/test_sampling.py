import pytest

import numpy as np

from typing import Union

from src.multichaos.legendre import evaluate_basis
from src.multichaos.index_set import generate_index_set
from src.multichaos.sampling import sample_arcsine
from src.multichaos.sampling import sample_optimal_distribution
from src.multichaos.sampling import arcsine
from src.multichaos.sampling import optimal_sample_size


IndexSet = list[Union[int, tuple[int, ...]]]

input_dim = 2
max_degree = 10
index_set = generate_index_set("TD", max_degree, input_dim)
n_samples = int(1e4)

samples = [
    sample_arcsine((n_samples, input_dim)),
    sample_optimal_distribution(index_set, n_samples),
]
densities = [
    lambda x: arcsine(x),
    lambda x: optimal_density(index_set, x),
]

def optimal_density(index_set: IndexSet, x: np.array) -> np.array:
    """
    Returns the optimal density function defined by `I` evaluated in `x`.
    """
    return (evaluate_basis(index_set, x) ** 2).sum(axis=1) / len(index_set) / 2 ** input_dim

def test_optimal_sample_1d():
    index_set_1d = np.arange(10)
    sample = sample_optimal_distribution(index_set_1d, n_samples)
    assert sample.shape == (n_samples,)

def test_optimal_sample_size_vary_risk():
    dim = len(index_set)
    risks = np.linspace(0.01, .99, 10)
    sample_sizes = optimal_sample_size(dim, risk=risks)
    assert all(sample_sizes >= dim)
    assert all(sample_sizes[i] <= sample_sizes[i+1] for i in range(len(sample_sizes) - 1))

def test_optimal_sample_size_zero_dim():
    dims = np.array([10, 11, 0, 13, 0])
    sample_sizes = optimal_sample_size(dims)
    assert sample_sizes[[2, 4]].all() == 0

@pytest.mark.parametrize("sample", samples)
def test_size(sample):
    assert sample.shape == (n_samples, input_dim)

@pytest.mark.parametrize("sample", samples)
def test_samples_in_correct_interval(sample):
    assert ((-1 < sample) & (sample < 1)).all()

@pytest.mark.parametrize("sample, density", zip(samples, densities))
def test_density(sample, density):
    vals, xedges, yedges = np.histogram2d(*sample.T, density=True)
    xmidpoints = (xedges[:-1] + xedges[1:]) / 2
    ymidpoints = (yedges[:-1] + yedges[1:]) / 2

    cart_prod = np.dstack(np.meshgrid(xmidpoints, ymidpoints))
    cart_prod = cart_prod[1:-1, 1:-1].reshape(-1, 2)
    vals = vals[1:-1, 1:-1]

    assert np.allclose(density(cart_prod), vals.reshape(-1), atol=1e-1)
