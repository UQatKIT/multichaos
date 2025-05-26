"""
Utility functions.
"""
import numpy as np

from collections import defaultdict

from . import single_level

def cartesian_product(*arrays):
    """Compute the Cartesian product of input arrays.

    Args:
        *arrays (list[np.ndarray]): Input arrays.
    
    Returns:
        (np.ndarray): Cartesian product of input arrays.
    """
    la = len(arrays)
    dtype = np.result_type(*arrays)
    arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(np.ix_(*arrays)):
        arr[...,i] = a
    return arr.reshape(-1, la, order="C")

def aggregate_coefficients(
        level_estimators: list[single_level.PCE]
) -> tuple[np.ndarray, np.ndarray]:
    """Aggregate coefficients from a list of level estimators.
    
    Args:
        level_estimators (list[single_level.PCE]): List of level estimators.
    """
    coef_dict = defaultdict(float)

    for pce in level_estimators:
        coefficients = pce.coefficients
        multi_index_set = pce.index_set

        for i, index in enumerate(multi_index_set):
            index_tuple = tuple(index)
            coef_dict[index_tuple] += coefficients[i]

    index_set = np.array(list(coef_dict.keys()))
    coefficients = np.array(list(coef_dict.values()))

    return index_set, coefficients
