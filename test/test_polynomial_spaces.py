import pytest

from src.polynomial_spaces import PolySpace
import matplotlib.pylab as plt


PLOT = False

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
    """Tests if TP, TD, and HC index sets match with expected ones."""
    m, d = 2, 2
    space = PolySpace(poly_space, m, d)
    res = space.index_set
    assert set(res) == set(expected)

if PLOT:
    @pytest.mark.parametrize("poly_space", poly_spaces)
    def test_plot_index_sets(poly_space):
        """Plots TP, TD, and HC index sets in 2 and 3 dimensions."""
        m = 16

        d = 2
        space = PolySpace(poly_space, m, d)
        space.plot_index_set()

        d = 3
        space = PolySpace(poly_space, m, d)
        space.plot_index_set()
