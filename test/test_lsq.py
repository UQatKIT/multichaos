import pytest

import numpy as np

from src.sampling import sample_optimal_distribution
from src.legendre import get_total_degree_index_set
from src.lsq import get_optimal_sample_size
from src.lsq import assemble_linear_system
from src.lsq import LSQ


@pytest.fixture(scope="session")
def m():
    return 6

@pytest.fixture(scope="session")
def I(m):
    return get_total_degree_index_set(m)

@pytest.fixture(scope="session")
def G(I):
    N = get_optimal_sample_size(I, sampling="optimal")
    sample = sample_optimal_distribution(I, size=N)
    f = np.ones(len(sample))
    G, _ = assemble_linear_system(I, sample, f)
    return G

@pytest.fixture(scope="session")
def x():
    return np.linspace(0, 1, 100)

def test_get_optimal_sample_size(I, m):
    r = 1
    N = get_optimal_sample_size(I, sampling="optimal", r=r)
    kappa = (1 - np.log(2)) / (1 + r) / 2
    assert kappa * N / np.log(N) >= m

def test_assemble_linear_system(G, I):
    assert G.shape == (len(I), len(I))
    assert np.allclose(G, G.T)

def test_system_matrix(G, I):
    assert np.linalg.norm(G - np.eye(len(I)), ord=2) <= .5
    assert np.linalg.cond(G) <= 3.

@pytest.mark.parametrize("deg", [0, 1, 3, 5])
def test_lsq_polynomial_fit(deg, x):
    I = get_total_degree_index_set(deg)
    f = lambda x: (x ** deg).sum(axis=1)
    model = LSQ(I, sampling="optimal").solve(f)
    x = np.random.rand(1000, 2)
    err = np.sqrt(((model(x) - f(x)) ** 2).mean())

    assert np.isclose(err, 0.)
