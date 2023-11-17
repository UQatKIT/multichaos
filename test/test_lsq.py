import pytest

import numpy as np

from src.sampling import sample_optimal_distribution
from src.sampling import optimal_sample_size
from src.polynomial_spaces import PolySpace
from src.least_squares import SingleLevelLSQ


@pytest.fixture(scope="session")
def m():
    return 6

@pytest.fixture(scope="session")
def I(m):
    return PolySpace("TD", m, d=2).index_set

@pytest.fixture(scope="session")
def M(I):
    space = PolySpace("TP", m=1, d=2)
    space.index_set = I
    model = SingleLevelLSQ({
        "poly_space": space,
        "domain": [(0, 1), (0, 1)],
    })

    N = optimal_sample_size(I, sampling="optimal")
    sample = sample_optimal_distribution(I, size=N)
    f = np.ones(len(sample))

    model.sample_ = sample
    model.f_vals_ = f
    M, _ = model.assemble_linear_system()

    return M

@pytest.fixture(scope="session")
def x():
    return np.linspace(0, 1, 100)

def test_get_optimal_sample_size(I, m):
    r = 1
    N = optimal_sample_size(I, sampling="optimal", r=r)
    kappa = (1 - np.log(2)) / (1 + r) / 2
    assert kappa * N / np.log(N) >= m

def test_assemble_linear_system(M, I):
    N = optimal_sample_size(I, sampling="optimal")
    assert M.shape == (N, len(I))

def test_system_matrix(M, I):
    r = 1
    N = optimal_sample_size(I, sampling="optimal", r=r)
    G = M.T @ M / N
    assert np.linalg.norm(G - np.eye(len(I)), ord=2) <= .5
    assert np.linalg.cond(G) <= 3.

@pytest.mark.parametrize("deg", [0, 1, 3, 5])
def test_lsq_polynomial_fit(deg, x):
    V = PolySpace("TD", deg, d=2)
    f = lambda x: (x ** deg).sum(axis=1)

    model = SingleLevelLSQ({
        "poly_space": V,
        "domain": [(0, 1), (0, 1)]
    }).fit(f)
    x = np.random.rand(1000, 2)
    err = np.sqrt(((model(x) - f(x)) ** 2).mean())

    assert np.isclose(err, 0.)
