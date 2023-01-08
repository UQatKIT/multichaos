import pytest

from src.polynomial_spaces import TensorProduct, TotalDegree, HyperbolicCross


poly_spaces = ["TP", "TD", "HC"]
poly_spaces_dict = {
    "TP": TensorProduct,
    "TD": TotalDegree,
    "HC": HyperbolicCross,
}
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
    m, d = 2, 2
    space = poly_spaces_dict[poly_space](m, d)
    res = space.index_set
    assert set(res) == set(expected)

@pytest.mark.parametrize("poly_space", poly_spaces)
def test_plot_index_sets(poly_space):
    m = 16

    d = 2
    space = poly_spaces_dict[poly_space](m, d)
    space.plot_index_set()

    d = 3
    space = poly_spaces_dict[poly_space](m, d)
    space.plot_index_set()
