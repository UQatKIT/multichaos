r"""Multi-index sets and related functions for polynomial spaces.

This module contains functions for creating multi-index sets.
Mathematically, a multi-index set $\Lambda \subset \mathbb{N}_0^d$
is a set of multi-indices $\lambda = (\lambda_1, \ldots, \lambda_d)$, where
$\lambda_i \in \mathbb{N}_0$ for all $i = 1, \ldots, d$.
Using these multi-indices, we can define tensorized basis functions

$$
\begin{equation}
    P_\lambda(x) = \prod_{i=1}^d P_{\lambda_i}(x_i), \quad x \in \mathbb{R}^d,
\end{equation}
$$

and span polynomial spaces conveniently via $\text{span}\{P_\lambda: \lambda \in \Lambda\}$.
The module supports three types of multi-index sets:
tensor product (TP), total degree (TD), and hyperbolic cross (HC).
It also contains utility functions for manipulating and checking admissibility
of index sets.

Functions:
    cartesian_product: Returns the Cartesian product of input arrays.
    generate_index_set: Generates a multi-index set of a given type.
    get_lower_neighbours: Returns the lower neighbours of an index.
    get_upper_neighbours: Returns the upper neighbours of an index.
    is_downward_closed: Checks if the given multi-index set is downward closed.
    separate_index: Separates an adaptive multi-index.
    slice_index_set: Slices an adaptive index set along a level.
    binary_expansion: Generates the binary expansion of an index.
    union_binary_expansions: Generates a union of binary expansions of an sliced index set.
"""
import numpy as np

from itertools import product
from typing import Literal, Generator


def cartesian_product(*arrays: tuple[np.ndarray]) -> np.ndarray:
    r"""Returns the Cartesian product of input arrays.

    Args:
        arrays (tuple): Arrays to be combined.

    Returns:
        (np.ndarray): Cartesian product of input arrays.
    """
    la = len(arrays)
    arr = np.empty([len(a) for a in arrays] + [la])
    for i, a in enumerate(np.ix_(*arrays)):
        arr[...,i] = a
    return arr.reshape(-1, la)

def generate_index_set(
        index_set_type: Literal["TP", "TD", "HC"],
        max_degree: int,
        dim: int
    ) -> np.ndarray:
    r"""Generates a multi-index set of a given type.

    The supported types are tensor product (TP), total degree (TD), and hyperbolic cross (HC).

    Args:
        index_set_type (Literal["TP", "TD", "HC]): Type of the index set.
        max_degree (int): Maximum degree of the index set.
        dim (int): Dimension of the index set.

    Returns:
        (np.ndarray): Multi-index set
    """
    x = np.arange(max_degree + 1)
    index_set = cartesian_product(*([x] * dim)).astype(int)

    if index_set_type == "TP":
        return index_set
    elif index_set_type == "TD":
        ix = np.where(np.sum(index_set, axis=1) <= max_degree)
    elif index_set_type == "HC":
        ix = np.where(np.prod(index_set + 1, axis=1) <= max_degree + 1)

    return index_set[ix]

def get_lower_neighbours(index: list) -> Generator[list, None, None]:
    """Returns the lower neighbours of `index`.

    A lower neighbour is defined as a tuple that differs
    in exactly one entry by subtracting one.

    Args:
        index (list): Multi-index to find lower neighbours for.
    
    Yields:
        (list): Generator of lower neighbours.
    """
    for i, k in enumerate(index):
        if k - 1 >= 0:
            lower_index = index[:i] + (k - 1,) + index[i+1:]
            yield lower_index

def get_upper_neighbours(index: list) -> Generator[list, None, None]:
    """Returns the upper neighbours of `index`.

    An upper neighbour is defined as a tuple that differs
    in exactly one entry by adding one.
    
    Args:
        index (list): Multi-index to find upper neighbours for.
    
    Yields:
        (list): Generator of upper neighbours.
    """
    for i, k in enumerate(index):
        upper_index = index[:i] + (k + 1,) + index[i+1:]
        yield upper_index

def is_downward_closed(index_set: np.ndarray) -> bool:
    r"""Checks if the given multi-index set is downward closed.

    A multi-index set $\Lambda \subset \mathbb{N}_0^d$ is downward closed if
    $\lambda \in \Lambda \Rightarrow \mu \in \Lambda$ for all $\mu \in \mathbb{N}_0^d$
    for which $\mu \leq \lambda$ componentwise.

    Args:
        index_set (np.ndarray): Multi-index set to be checked.

    Returns:
        (bool): True if the index set is downward closed, False otherwise.
    """
    index_set_tuples = {tuple(index) for index in index_set}
    for index in index_set:
        for j, k in enumerate(index):
            index_ = list(index)
            if k - 1 >= 0:
                index_[j] -= 1
                if tuple(index_) not in index_set_tuples:
                    return False
    return True

def separate_index(index: list) -> tuple[list, int]:
    r"""Separates an adaptive multi-index.

    This function simply separates an index $(k, l) = \lambda \in \mathbb{N}_0^{d + 1}$
    into $k \in \mathbb{N}_0^d$ and $l \in \mathbb{N}_0$.

    Args:
        index (list): Multi-index to be separated.

    Returns:
        (tuple): Separated multi-index.
    """
    return index[:-1], index[-1]

def slice_index_set(index_set: list, l: int) -> list:
    r"""Slices an adaptive index set along a level.

    Given an adaptive index set $\Lambda \subset \mathbb{N}_0^{d + 1}$ and level $l \in \mathbb{N}_0$,
    this returns $\{(k, l') \in \Lambda : l' = l\}$.

    Args:
        index_set (list): Adaptive index set to be sliced.
        l (int): Level to slice along.
    
    Returns:
        (np.ndarray): Sliced index set
    """
    return [index for index in index_set if separate_index(index)[1] == l]

def binary_expansion(index: list) -> list:
    r"""Generates the binary expansion of an index.

    If `index` denotes $\lambda \in \mathbb{R}^d$, this returns all multi-indices $k \in \mathbb{R}^d$ for which it holds

    $$
    2^k \leq \lambda < 2^{k+1} - 1.
    $$

    The size comparison is element-wise.

    Args:
        index (list): Multi-index to be checked.
    
    Returns:
        (list): List of multi-indices that satisfy the condition.
    """
    low = tuple(int(np.ceil(2 ** k)) - 1 for k in index)
    high = tuple(int(np.ceil(2 ** (k + 1))) - 1 for k in index)
    index_set = list(product(*[range(l, h) for (l, h) in zip(low, high)]))
    return index_set

def union_binary_expansions(index_set: list, l: int) -> list:
    """Generates a union of binary expansions of a sliced index set.
    
    This returns the index set given by the binary expansion
    of all indices in the level $l$-sliced index set.
    """
    union = []
    for index in slice_index_set(index_set, l):
        k, _ = separate_index(index)
        union.extend(binary_expansion(k))
    return union