import pytest

import matplotlib.pyplot as plt
import numpy as np

from src.polynomial_spaces import index_set


poly_spaces = ["TP", "TD", "HC"]
expected_index_sets = [
    [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (0, 2), (1, 2), (2, 1), (2, 2)],
    [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (0, 2)],
    [(0, 0), (0, 1), (1, 0), (2, 0), (0, 2)],
]

@pytest.mark.parametrize(
    "poly_space, expected",
    zip(poly_spaces, expected_index_sets)
)
def test_index_sets(poly_space, expected):
    m = 2
    res = index_set(poly_space, m)
    assert set(res) == set(expected)

@pytest.mark.parametrize("poly_space", poly_spaces)
def test_plot_index_sets(poly_space):
    m = 16
    res = index_set(poly_space, m)

    plt.scatter(*np.array(res).T, c="k")
    plt.title(poly_space)
    plt.axis("square")
    plt.show()
