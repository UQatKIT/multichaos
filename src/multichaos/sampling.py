R"""Sampling routines for optimal weighted least squares.

This module provides routines for sampling from the optimal sampling distribution.
Given a polynomial space $V:= \text{span} \{P_\lambda: \lambda \in \Lambda\}$
defined by a multi-index set $\Lambda \subset \mathbb{N}_0^d$,
there exists an optimal sampling distribution $\nu = \nu(V)$
that minimizes the number of samples required for the weighted least squares estimator onto $V$.
This module also contains routines for arcsine sampling and determining the optimal sample size
for quasi-optimality of the weighted least squares estimator.

Functions:
    arcsine: Arcsine density function.
    sample_arcsine: Samples from the arcsine distribution.
    sample_squared_legendre: Samples from the distribution with squared Legendre density.
    optimal_sample_size: Optimal sample size for quasi-optimality.
    sample_optimal_distribution: Samples from the optimal distribution.
"""

import numpy as np

from collections import defaultdict

from . import legendre


def arcsine(x: np.ndarray) -> np.ndarray:
    r"""Arcsine density function.

    This evaluates the arcsine density function

    $$
    \begin{equation}
    p_d(x) = \prod_{i=1}^d \frac{1}{\pi \sqrt{1- x_i^2}}, \quad x \in (-1, 1)^d.
    \end{equation}
    $$

    Args:
        x (np.ndarray): Input array.
    
    Returns:
        (np.ndarray): Arcsine density function evaluated at `x`.
    """
    y = 1 / np.pi / np.sqrt(1 - x ** 2)
    return y.prod(axis=1) if x.ndim > 1 else y

def sample_arcsine(size: tuple[int, ...]) -> np.ndarray:
    """Samples from the arcsine distribution.

    This is based on the fact that if $U \sim \mathcal{U}(-\pi, \pi)$,
    then $X = \sin(U) \sim p_1$ follows the arcsine distribution.

    Args:
        size (tuple): Size of the sample.
    
    Returns:
        (np.ndarray): Sample from the arcsine distribution with shape `size`.
    """
    u = np.random.uniform(-np.pi, np.pi, size=size)
    return np.sin(u)

def sample_squared_legendre(n: int, size: int) -> np.ndarray:
    r"""Samples from the distribution with squared Legendre density.

    This function samples from the distribution with squared Legendre density $P_n^2$,
    where $P_n$ is the Legendre polynomial of degree $n$.
    This is done using rejection sampling with arcsine proposals, since

    $$
    \begin{equation}
        |P_n(x)| \leq 2 p_1(x)^{1/2}, \quad x \in (-1, 1), \,\, \forall n \in \mathbb{N}_0.
    \end{equation}
    $$

    Args:
        n (int): Degree of Legendre polynomial.
        size (int): Number of samples.
    
    Returns:
        (np.ndarray): Sample from the square Legendre density with shape `size`.
    """
    samples = []
    while len(samples) < size:
        props = sample_arcsine(size)
        leg_evals = legendre.legval(n, props)[:, -1] ** 2
        aux = leg_evals / arcsine(props) / 4
        U = np.random.uniform(size=size)
        samples.extend(props[U <= aux])

    return np.array(samples[:size])

def optimal_sample_size(dim: float, risk: float=0) -> int:
    r"""Optimal sample size for quasi-optimality.

    Determines the optimal number of samples $N$ required to ensure
    quasi-optimality of a weighted least squares estimator
    onto a polynomial space $V$.
    For $p \in (0, 1)$ and $\theta = \frac{1}{2}(1 - \log 2)$, the condition

    $$
    \begin{equation}
    N \geq \frac{m}{\theta} \log \left(\frac{2m}{1 - p}\right),
    \end{equation}
    $$

    where $m = \text{dim}(V)$, ensures that

    $$
    \begin{equation}
    \mathbb{P}(\Vert \mathbf{G} - \mathbf{I}\Vert \leq 1/2) \geq p,
    \end{equation}
    $$

    where $\mathbf{G} \in \mathbb{R}^{m \times m}$ denotes the Gramian least squares
    and $\mathbf{I} \in \mathbb{R}^{m \times m}$ the identity matrix.

    Args:
        dim (float): Dimension of the polynomial space used for projection.
        risk (float): Probability $p$ for quasi-optimality to hold.

    Returns:
        int: Optimal sample size.
    """
    safe_dim = np.clip(dim, 1e-10, None)
    theta = (1 - np.log(2)) / 2
    N = safe_dim / theta * np.log(2 * safe_dim / (1 - risk))
    return np.where(dim == 0, 0, np.ceil(N).astype(int))

def sample_optimal_distribution(index_set: np.ndarray, size: int) -> np.ndarray:
    r"""Samples from the optimal distribution.

    Given a polynomial subspace $\text{span} \{P_\lambda: \lambda \in \Lambda\} \subset L^2([-1, 1]^d)$
    defined by an index set $\Lambda \subset \mathbb{N}_0^d$,
    the density of the optimal distribution is defined by

    $$
    \begin{equation}
        \frac{1}{|\Lambda|} \sum_{\lambda \in \Lambda} P_\lambda^2.
    \end{equation}
    $$

    Args:
        index_set (np.ndarray): Index set defining the polynomial subspace of shape `(n_basis, dim)`.
        size (int): Number of samples.
    
    Returns:
        (np.ndarray): Samples with shape `(size, dim)`.
    """
    dim = 1 if index_set.ndim == 1 else index_set.shape[1]
    if dim == 1:
        choices = np.random.choice(np.array(index_set).squeeze(), size)
        count_choices = np.bincount(choices)
        
        samples = []
        for n, count in enumerate(count_choices):
            samples.append(sample_squared_legendre(n, count))

        return np.hstack(samples)
    
    choices = np.random.randint(len(index_set), size=size)
    count_choices = np.bincount(choices)
    
    count_degrees = defaultdict(int)
    for eta, count in zip(index_set, count_choices):
        for k in eta:
            count_degrees[k] += count

    samples_uni = {}
    for k, count in count_degrees.items():
        samples_uni[k] = sample_squared_legendre(k, count)

    ind = np.insert(count_choices.cumsum(), 0, 0)

    samples = np.zeros((size, dim))
    for i in np.nonzero(count_choices)[0]:
        for j, k in enumerate(index_set[i]):
            samples[ind[i]:ind[i + 1], j] = samples_uni[k][-count_choices[i]:]
            samples_uni[k] = np.delete(samples_uni[k], range(-count_choices[i], 0))

    return samples