import pytest

import numpy as np

from src.multichaos.legendre import evaluate_basis
from src.multichaos.utils import cartesian_product


@pytest.mark.parametrize("index_set", [
    np.array([0, 1, 2, 3]).reshape(-1, 1),
    np.array([
        [0, 0],
        [0, 1],
        [1, 0]
    ])
])
def test_legendre_orthonormal(index_set):
    """
    Tests whether the legendre polynomials are
    orthonormal for some index sets.
    """
    if index_set.ndim == 1: # 1D
        dim = 1
        x = np.linspace(-1, 1, int(1e6))
        dx = x[1] - x[0]
    else: # ND
        dim = 2
        x_ = np.linspace(-1, 1, int(1e3))
        x = cartesian_product(x_, x_)
        dx = (x_[1] - x_[0]) ** 2

    vals = evaluate_basis(index_set, x)
    matrix = (vals.T @ vals) * dx / 2 ** dim

    assert np.allclose(matrix, np.eye(len(index_set)), atol=1e-1)
