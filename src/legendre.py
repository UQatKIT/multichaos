"""
Methods for constructing basis and evaluation of Legendre polynomials.
"""
import numpy as np

from numpy.polynomial.legendre import legder
from numpy.polynomial.legendre import legint

from typing import Union


IndexSet = list[Union[int, tuple[int, ...]]]


def legval(n: int, sample: np.array, domain: tuple) -> np.array:
    """
    Evaluates the first `n + 1` shifted Legendre polynomials in the given sample.
    """
    y = np.zeros((n + 1, len(sample)))
    y[0] = np.ones_like(sample)

    if n > 0:
        # [a, b] to [-1, 1]
        a, b = domain
        sh_sample = 2 * (sample - a) / (b - a) - 1

        y[1] = sh_sample
        for i in range(1, n):
            y[i + 1] = ((2 * i + 1) * sh_sample * y[i] - i * y[i - 1]) / (i + 1)

    return y

def evaluate_basis(I: IndexSet, sample: np.array, domain: list[tuple]) -> np.array:
    I = np.array(I)
    if I.shape[-1] == 1:
        I = I.squeeze(-1)

    d = 1 if isinstance(I[0], (int, np.int64)) else len(I[0])
    if d == 1:
        basis_val = legval(np.max(I), sample, domain[0])[I]
    else:
        max_idxs = np.max(I, axis=0)
        basis_val_uni = {}
        for j in range(d):
            basis_val_uni[j] = legval(max_idxs[j], sample[:, j], domain[j])

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

def call(coef: dict, sample: np.array, domain: list[tuple]) -> np.array:
    I = list(coef.keys())
    c = list(coef.values())

    I = np.array(I)
    if I.shape[-1] == 1:
        I = I.squeeze(-1)

    d = 1 if isinstance(I[0], (int, np.int64)) else len(I[0])
    if d == 1:
        basis_val = legval(np.max(I), sample, domain[0])[I]
        norms = np.sqrt(2 * I + 1).reshape(-1, 1)
        basis_val *= norms
        out = np.dot(c, basis_val)
    else:
        max_idxs = np.max(I, axis=0)
        basis_val_uni = {}
        for j in range(d):
            basis_val_uni[j] = legval(max_idxs[j], sample[:, j], domain[j])

        # Chosen s.t. each chunk takes up ~1GB of memory
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

def transform_coefs(coef: dict) -> np.array:
    shape = np.array(list(coef.keys())).max(axis=0)
    c = np.zeros(shape + 1)

    for index, value in coef.items():
        c[index] = value

    return c

def deriv(coef: dict, sample: np.array, axis: int, domain: list[tuple]) -> np.array:
    c = transform_coefs(coef)

    I = np.array([ix for ix, _ in np.ndenumerate(c)])
    norm = np.sqrt(2 * I + 1).prod(axis=1)
    c *= norm.reshape(c.shape)
    new_c = legder(c, scl=2, axis=axis)

    I = [ix for ix, _ in np.ndenumerate(new_c)]
    norm = np.sqrt(2 * np.array(I) + 1).prod(axis=1)
    new_c = new_c.reshape(-1) / norm

    new_coef = dict(zip(I, new_c))

    return call(new_coef, sample, domain)

def integ(coef: dict, sample: np.array, axis: int, domain: list[tuple]) -> np.array:
    c = transform_coefs(coef)

    I = np.array([ix for ix, _ in np.ndenumerate(c)])
    norm = np.sqrt(2 * I + 1).prod(axis=1)
    c *= norm.reshape(c.shape)
    new_c = legint(c, scl=.5, axis=axis, lbnd=-1)

    I = [ix for ix, _ in np.ndenumerate(new_c)]
    norm = np.sqrt(2 * np.array(I) + 1).prod(axis=1)
    new_c = new_c.reshape(-1) / norm

    new_coef = dict(zip(I, new_c))

    return call(new_coef, sample, domain)
