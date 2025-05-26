"""Legendre polynomials.

This module provides functions to evaluate the orthogonal
Legendre polynomials on $[-1, 1]^d$ using tensorized basis functions.

Functions:
    legval: Evaluates Legendre polynomials up to degree $n$.
    evaluate_basis: Evaluates a tensorized Legendre basis.
"""
import numpy as np


def legval(n: int, sample: np.ndarray) -> np.ndarray:
    r"""Evaluates Legendre polynomials up to degree $n$.

    This evaluates the Legendre polynomials $P_0, \ldots, P_n$
    up to degree $n \in \mathbb{N}_0$ in the given sample.
    The polynomials are orthonormalized with respect to
    the uniform weighting on $[-1, 1]$, such that

    $$
    \begin{equation}
        \int_{-1}^1 P_i(x) P_j(x) \,  \frac{dx}{2} = \delta_{ij}.
    \end{equation}
    $$

    Args:
        n (int): Number of polynomials to evaluate.
        sample (np.ndarray): Sample points with shape `(n_samples,)`.

    Returns:
        (np.ndarray): Evaluations of shape `(n_samples, n + 1)`.
    """
    sample = sample.squeeze()

    y = np.zeros((len(sample), n + 1))
    y[:, 0] = 1
    if n == 0:
        return y

    y[:, 1] = sample
    for i in range(1, n):
        y[:, i + 1] = ((2 * i + 1) * sample * y[:, i] - i * y[:, i - 1]) / (i + 1)
    
    orthonormalizer = np.sqrt(2 * np.arange(n + 1) + 1)
    y *= orthonormalizer
    return y

def evaluate_basis(index_set: np.ndarray, sample: np.ndarray) -> np.ndarray:
    r"""Evaluates a tensorized Legendre basis.

    Evaluates the tensorized Legendre basis defined by the index set on the given sample.
    For an index $\lambda \in \Lambda$ for some multi-index set $\Lambda \subset \mathbb{N}_0^d$, the corresponding basis function is given by
    
    $$
    \begin{equation}
        P_\lambda (x) := \prod_{j=1}^d P_{\lambda_j} (x_j),     \quad \text{for} \quad x \in [-1, 1]^d,
    \end{equation}
    $$

    where $P_n$ denotes the $n$-th univariate Legendre polynomial.

    Args:
        index_set (np.ndarray): Index set $\Lambda$ of shape `(n_basis, d)`.
        sample (np.ndarray): Sample points with shape `(n_samples, d)`.
    
    Returns:
        (np.ndarray): Evaluations of the basis on the sample with shape `(n_samples, n_basis)`.
    
    """
    max_idxs = np.max(index_set, axis=0)
    basis_val_uni = {}
    for j in range(index_set.shape[1]):
        basis_val_uni[j] = legval(max_idxs[j], sample[:, j])

    basis_val = np.zeros((len(sample), len(index_set)))
    for i, index in enumerate(index_set):
        prod = 1.
        for j, k in enumerate(index):
            prod *= basis_val_uni[j][:, k]
        basis_val[:, i] = prod

    return basis_val