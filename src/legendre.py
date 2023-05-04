"""
Methods for constructing basis and evaluation of Legendre polynomials.
"""
import numpy as np

from typing import Union


IndexSet = list[Union[int, tuple[int, ...]]]


def legval(n: int, sample: np.array) -> np.array:
    """
    Evaluates the first `n + 1` shifted Legendre polynomials in the given sample.
    """
    y = np.zeros((n + 1, len(sample)))
    y[0] = np.ones_like(sample)

    if n > 0:
        sh_sample = 2 * sample - 1
        y[1] = sh_sample
        for i in range(1, n):
            y[i + 1] = ((2 * i + 1) * sh_sample * y[i] - i * y[i - 1]) / (i + 1)

    return y

def evaluate_basis(I: IndexSet, sample: np.array) -> np.array:
    I = np.array(I)
    if I.shape[-1] == 1:
        I = I.squeeze(-1)

    d = 1 if isinstance(I[0], (int, np.int64)) else len(I[0])
    if d == 1:
        basis_val = legval(np.max(I), sample)[I]
    else:
        max_idxs = np.max(I, axis=0)
        basis_val_uni = {}
        for j in range(d):
            basis_val_uni[j] = legval(max_idxs[j], sample[:, j])

        basis_val = np.zeros((len(I), len(sample)))
        for i, eta in enumerate(I):
            prod = 1.
            for j, k in enumerate(eta):
                prod *= basis_val_uni[j][k]
            basis_val[i] = prod

    norms = np.sqrt(2 * I + 1)
    if d > 1:
        norms = norms.prod(axis=1)
    basis_val *= norms.reshape(-1, 1)

    return basis_val

def call(coef: dict, sample: np.array) -> np.array:
    I = list(coef.keys())
    c = list(coef.values())

    I = np.array(I)
    if I.shape[-1] == 1:
        I = I.squeeze(-1)

    d = 1 if isinstance(I[0], (int, np.int64)) else len(I[0])
    if d == 1:
        basis_val = legval(np.max(I), sample)[I]
        norms = np.sqrt(2 * I_ + 1).reshape(-1, 1)
        basis_val *= norms
        out = np.dot(c, basis_val)
    else:
        max_idxs = np.max(I, axis=0)
        basis_val_uni = {}
        for j in range(d):
            basis_val_uni[j] = legval(max_idxs[j], sample[:, j])

        # Chosen s.t. each chunks takes up ~ 1GB of memory
        chunk_size = int(10 ** 9 / 8 / len(sample))
        iters = len(I) // chunk_size
        rem = len(I) % chunk_size

        out = 0.
        for x in range(iters):
            basis_val = np.zeros((chunk_size, len(sample)))
            I_ = I[x * chunk_size: (x + 1) * chunk_size]
            for i, eta in enumerate(I_):
                prod = 1.
                for j, k in enumerate(eta):
                    prod *= basis_val_uni[j][k]
                basis_val[i] = prod

            norms = np.sqrt(2 * I_ + 1)
            if d > 1:
                norms = norms.prod(axis=1)
            basis_val *= norms.reshape(-1, 1)

            out += np.dot(c[x * chunk_size: (x + 1) * chunk_size], basis_val)

        if rem > 0:
            basis_val = np.zeros((rem, len(sample)))
            I_ = I[-rem:]
            for i, eta in enumerate(I_):
                prod = 1.
                for j, k in enumerate(eta):
                    prod *= basis_val_uni[j][k]
                basis_val[i] = prod

            norms = np.sqrt(2 * I_ + 1)
            if d > 1:
                norms = norms.prod(axis=1)
            basis_val *= norms.reshape(-1, 1)

            out += np.dot(c[-rem:], basis_val)

    return out
