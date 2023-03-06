"""
Multivariate polynomial basis index sets.
"""
import matplotlib.pyplot as plt
import numpy as np

from typing import Literal

from utils import cartesian_product


PolynomialSpace = Literal[
    "TP",   # tensor product
    "TD",   # total degree
    "HC",   # hyperbolic cross
]

class PolySpace:
    def __init__(self, kind: PolynomialSpace, m: int, d: int):
        self.kind = kind
        if kind not in ["TP", "TD", "HC"]:
            raise ValueError(
                "{kind} is not a valid polynomial space. Should be one of ['TP', 'TD', 'HC']."
                )
        self.m = m
        self.d = d

        self.set_index_set()
        self.dim = len(self.index_set)

    def set_index_set(self):
        if self.d == 1:
            self.index_set = list(range(self.m + 1))
            return

        x = np.arange(self.m + 1)
        I = cartesian_product(*([x] * self.d))

        if self.kind == "TP":
            pass
        elif self.kind == "TD":
            ix = np.where(np.sum(I, axis=1) <= self.m)
            I = I[ix]
        elif self.kind == "HC":
            ix = np.where(np.prod(I + 1, axis=1) <= self.m + 1)
            I = I[ix]

        self.index_set = list(zip(*I.T))

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
