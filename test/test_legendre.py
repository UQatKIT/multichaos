import pytest

import numpy as np

from src.legendre import legvalnd


@pytest.mark.parametrize("I", [[1, 2, 3], [(0, 0), (0, 1), (1, 0)]])
def test_legendre_orthonormal(I):
    if isinstance(I[0], int): # 1D
        x = np.linspace(0, 1, 10_0000)
        dx = x[1] - x[0]
    else: # ND
        x_ = np.linspace(0, 1, 1000)
        x = np.dstack(np.meshgrid(x_, x_)).reshape(-1, 2)
        dx = (x_[1] - x_[0]) ** 2

    vals = np.array([legvalnd(x, eta) for eta in I])
    matrix = dx * (vals @ vals.T)

    assert np.allclose(matrix, np.eye(len(I)), atol=1e-2)
