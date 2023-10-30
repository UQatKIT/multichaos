import pytest

import numpy as np

from typing import Union

from src.legendre import evaluate_basis
from src.polynomial_spaces import PolySpace
from src.sampling import sample_arcsine
from src.sampling import sample_optimal_distribution
from src.sampling import arcsine


IndexSet = list[Union[int, tuple[int, ...]]]

d = 2
m = 10
I = PolySpace("TD", m, d).index_set
size = (100_000, 2)

samples = [
    sample_arcsine(size),
    sample_optimal_distribution(I, size[0]),
]
densities = [
    arcsine,
    lambda x: optimal_density(I, x),
]

def optimal_density(I: IndexSet, x: np.array) -> np.array:
    """
    Returns the optimal density function defined by `I` evaluated in `x`.
    """
    return (evaluate_basis(I, x) ** 2).sum(axis=0) / len(I)

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
