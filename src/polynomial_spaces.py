"""
Multivariate polynomial basis index sets.
"""
import matplotlib.pyplot as plt
import numpy as np

from typing import Literal, Union

from utils import cartesian_product


IndexSet = list[Union[int, tuple[int, ...]]]
PolynomialSpace = Literal[
    "TP",   # tensor product
    "TD",   # total degree
    "HC",   # hyperbolic cross
]

def index_set(kind: PolynomialSpace, m: int, d: int=2) -> IndexSet:
    """
    Returns the index set of the given kind and order `m`.
    """
    x = np.arange(m + 1)
    I = cartesian_product(*([x] * d))

    if kind == "TP":
        pass
    elif kind == "TD":
        ix = np.where(np.sum(I, axis=1) <= m)
        I = I[ix]
    elif kind == "HC":
        ix = np.where(np.prod(I + 1, axis=1) <= m + 1)
        I = I[ix]

    I = list(zip(*I.T))
    return I

class PolySpace:
    def __init__(self, m, d):
        self.m = m
        self.d = d

    def set_index_set(self, poly_type: PolynomialSpace):
        self.index_set = index_set(poly_type, self.m, self.d)
        self.dim = len(self.index_set)

    def plot_index_set(self):
        I = np.array(self.index_set)
        if I.shape[-1] == 2:
            plt.scatter(*np.array(I).T)
            plt.axis("square")
        elif I.shape[-1] == 3:
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            ax.scatter(*np.array(I).T)
        plt.show()

class TensorProduct(PolySpace):
    def __init__(self, m, d):
        super().__init__(m, d)
        self.set_index_set("TP")

class TotalDegree(PolySpace):
    def __init__(self, m, d):
        super().__init__(m, d)
        self.set_index_set("TD")

class HyperbolicCross(PolySpace):
    def __init__(self, m, d):
        super().__init__(m, d)
        self.set_index_set("HC")
