import pytest

from src.multichaos.index_set import generate_index_set


index_set_types = ["TP", "TD", "HC"]
expected_index_sets = [
    [[0, 0], [0, 1], [1, 0], [1, 1], [2, 0], [0, 2], [1, 2], [2, 1], [2, 2]],
    [[0, 0], [0, 1], [1, 0], [1, 1], [2, 0], [0, 2]],
    [[0, 0], [0, 1], [1, 0], [2, 0], [0, 2]]
]

@pytest.mark.parametrize(
    "index_set_type, expected_index_set",
    zip(index_set_types, expected_index_sets)
)
def test_index_sets(index_set_type, expected_index_set):
    """Tests if TP, TD, and HC index sets match with expected ones."""
    max_degree = 2
    dim = 2
    index_set = generate_index_set(index_set_type, max_degree, dim)

    assert set(map(frozenset, index_set)) == set(map(frozenset, expected_index_set))