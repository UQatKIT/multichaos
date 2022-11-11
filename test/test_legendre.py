import pytest

import numpy as np

from numpy.polynomial.polynomial import polyval
from scipy.special import binom

from src.sampling import transform
from src.legendre import legvalnd
from src.legendre import leggridnd
from src.legendre import get_total_degree_index_set
from src.legendre import index_set_transformation
from src.legendre import optimal_density
from src.legendre import optimal_weight


legendre_ = [
    lambda x: np.ones_like(x),
    lambda x: x,
    lambda x: 1/2 * (3 * x ** 2 - 1),
    lambda x: 1/2 * (5 * x ** 3 - 3 * x),
]

@pytest.mark.parametrize("dim", [2, 3, 4, 5])
def test_leggvalnd(dim):
    x = np.random.random((dim, 10))
    y = polyval(transform(x), [1., 2., 3.])
    c1d = np.array([2., 2., 2.])
    c = np.einsum(*sum([(c1d, [i]) for i in range(dim)], ()))
    vals = legvalnd(x.T, c).flatten()
    expected = y.prod(axis=0)

    assert np.allclose(vals, expected)

@pytest.mark.parametrize("dim", [2, 3, 4, 5])
def test_leggridnd(dim):
    x = np.random.random((dim, 10))
    y = polyval(transform(x), [1., 2., 3.])
    c1d = np.array([2., 2., 2.])
    c = np.einsum(*sum([(c1d, [i]) for i in range(dim)], ()))
    vals = leggridnd([*x], c).flatten()
    expected = np.einsum(*sum([(y_, [i]) for i, y_ in enumerate(y)], ())).flatten()

    assert np.allclose(vals, expected)

def test_get_total_degree_index_set():
    m = 2
    res = get_total_degree_index_set(m)
    expected = [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (0, 2)]
    assert set(res) == set(expected)

def test_get_total_degree_index_set_size():
    m = 2
    I = get_total_degree_index_set(m)
    res = len(I)
    expected = binom(m + 2, 2)
    assert res == expected

def test_index_set_transformation():
    I = [(3, 1, 2), (1, 0, 2)]
    x = np.random.rand(100, 3)

    expected = 0
    for eta in I:
        expected += np.sqrt(2 * np.array(eta) + 1).prod() * np.array(
            [legendre_[ix](transform(x[:, i])) for i, ix in enumerate(eta)]
        ).prod(0)

    I = index_set_transformation(I)
    res = sum(legvalnd(x, eta) for eta in I)

    assert np.allclose(res, expected)

def test_optimal_density_integral_one():
    m = 5
    I = get_total_degree_index_set(m, d=1)
    I = index_set_transformation(I)
    x = np.linspace(0, 1, 10_000)
    res = optimal_density(I, x)
    integral = np.trapz(res, x=x)

    assert np.isclose(integral, 1.)

def test_optimal_density_inverse_weights():
    m = 5
    I = get_total_degree_index_set(m, d=1)
    I = index_set_transformation(I)
    x = np.linspace(0, 1, 10_000)
    res = optimal_density(I, x)
    expected = 1. / optimal_weight(I, x)

    assert np.allclose(res, expected)

@pytest.mark.parametrize("I", [[1, 2, 3], [(0, 0), (0, 1), (1, 0)]])
def test_legendre_orthonormal(I):
    if isinstance(I[0], int): # 1D
        x = np.linspace(0, 1, 10_0000)
        dx = x[1] - x[0]
    else: # ND
        x_ = np.linspace(0, 1, 1000)
        x = np.dstack(np.meshgrid(x_, x_)).reshape(-1, 2)
        dx = (x_[1] - x_[0]) ** 2

    I = index_set_transformation(I)
    vals = np.array([legvalnd(x, eta) for eta in I])
    matrix = dx * (vals @ vals.T)

    assert np.allclose(matrix, np.eye(len(I)), atol=1e-2)
