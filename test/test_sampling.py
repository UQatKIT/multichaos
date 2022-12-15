import pytest

import numpy as np

from src.legendre import get_total_degree_index_set
from src.legendre import optimal_density
from src.sampling import sample_arcsine
from src.sampling import sample_optimal_distribution
from src.sampling import arcsine


dim = 2
m = 10
I = get_total_degree_index_set(m, d=dim)
size = (100_000, dim)

samples = [
    sample_arcsine(size),
    sample_optimal_distribution(I, size[0]),
]
densities = [
    arcsine,
    lambda x: optimal_density(I, x),
]

@pytest.mark.parametrize("sample", samples)
def test_size(sample):
    assert sample.shape == size

@pytest.mark.parametrize("sample", samples)
def test_samples_in_unit_interval(sample):
    assert ((0 < sample) & (sample < 1)).all()

@pytest.mark.parametrize("sample, density", zip(samples, densities))
def test_density(sample, density):
    vals, xedges, yedges = np.histogram2d(*sample.T, density=True)
    xmidpoints = (xedges[:-1] + xedges[1:]) / 2
    ymidpoints = (yedges[:-1] + yedges[1:]) / 2

    cart_prod = np.dstack(np.meshgrid(xmidpoints, ymidpoints))
    cart_prod = cart_prod[1:-1, 1:-1].reshape(-1, 2)
    vals = vals[1:-1, 1:-1]

    assert np.allclose(density(cart_prod), vals.reshape(-1), atol=1e-1)
