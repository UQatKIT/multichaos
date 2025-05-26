r"""
Polynomial chaos expansion based on optimal least squares.

This module provides a class for polynomial chaos expansion (PCE) based on
optimal weighted least squares regression.
Given a response function $Q: \Omega \subseteq \mathbb{R}^d \to \mathbb{R}^k$
and a polynomial subspace $V:= \text{span} \{P_\lambda: \lambda \in \Lambda\} \subseteq L^2_\mu(\Omega)$,
we seek for the best approximation 

$$
\begin{equation}
    \arg \min_{v \in V} \|Q - v\|_{L^2_\mu(\Omega)} := \Pi_V Q.
\end{equation}
$$

We discretize this problem and solve it using an importance sampling
enhanced version of unweighted least squares.
In particular, we utilize the seminorm

$$
\begin{equation}
    \|Q-v\|_N^2 = \sum_{i=1}^N \left(\frac{d \mu}{d \nu}\right)(\omega_i) |Q(\omega_i) - v(\omega_i)|^2,
\end{equation}
$$

where $\mu$ denotes the reference and $\nu$ the (potentially different) sampling measure.
The Radon-Nikodym weighting $\frac{d \mu}{d \nu}$ is necessary to ensure unbiasedness of the estimator.
The coefficients of the best-approximation $\Pi_V Q$ then satisfy the
normal equations

$$
\begin{equation}
    \mathbf{G} \mathbf{c} = \mathbf{q},
\end{equation}
$$

where $\mathbf{G} \in\mathbb{R}^{m \times m}$ with $m = \text{dim}(V)$
denotes the Gramian matrix with entries $\mathbf{G}_{ij} = \langle P_{\lambda_i}, P_{\lambda_j} \rangle_N$,
and $\mathbf{q} \in \mathbb{R}^{m \times k}$ the right-hand side with $\mathbf{q}_{i, \cdot} = \langle Q, P_{\lambda_i} \rangle_N$.

By sampling from the [optimal distribution][multichaos.sampling.sample_optimal_distribution]
$\nu = \nu(V)$, one can show that the number of samples $N$ required for a stable approximation
reduces from $\mathcal{O}(m^2 \log m)$ to $\mathcal{O}(m \log m)$.

Note:
    Currently, only the Lebesgue measure $\mu$ is supported, which corresponds to a
    uniform distribution on $\Omega$ with corresponding Legendre orthogonal polynomials.
    Uncertainty modeling is based on [OpenTURNS](https://openturns.github.io/www/),
    which allows an easy extension to other distributions and polynomial families.

Classes:
    PCE: Polynomial chaos expansion class.
"""
import time

import numpy as np

from . import base
from . import legendre
from . import sampling


class PCE(base.BasePCE):
    """Polynomial chaos expansion based on optimal least squares.

    This class implements a polynomial chaos expansion
    using optimal weighted least squares.

    Attributes:
        dist (ot.Distribution): Input distribution.
        index_set (np.ndarray): Multi-index set.
        response (callable): Response function.
        data_in (np.ndarray): Input data points.
        data_out (np.ndarray): Output data points.
        n_samples (int): Number of samples.
        sampling (str): Sampling method.
        data_matrix (np.ndarray): Data matrix.
        weights (np.ndarray): Weights.
        coefficients (np.ndarray): PCE coefficients.
        output_dim (int): Output dimension.
        fitting_time (float): Fitting time.
        sampling_time (float): Sampling time.
        evaluation_time (float): Evaluation time.
        condition_number (float): Condition number of data matrix.
    """
    def __init__(
            self,
            dist,
            index_set,
            response=None,
            data_in=None,
            data_out=None,
            n_samples=None,
    ):
        """Initialize PCE object.

        Args:
            dist (ot.Distribution): Input distribution.
            index_set (np.ndarray): Multi-index set.
            response (callable, optional): Response function.
            data_in (np.ndarray, optional): Input data points.
            data_out (np.ndarray, optional): Output data points.
            n_samples (int, optional): Number of samples.
        """
        super().__init__(dist, index_set)
        self.response = response
        self.data_in = data_in
        self.data_out = data_out
        self.n_samples = n_samples

    def set_data(self, data_in=None, data_out=None, n_samples=None):
        """Sets data for the model.

        This method sets input and output data based on the provided arguments.

        1. If `data_in` and `data_out` are provided, then these are set as data.
        2. If only `n_samples` is provided, the method samples `n_samples` points from the optimal distribution.
        3. If none of the arguments are provided, the method samples the [number of required samples][multichaos.sampling.optimal_sample_size]
        from the optimal distribution such that quasi-optimality is met..

        Args:
            data_in (np.ndarray, optional): Input data points.
            data_out (np.ndarray, optional): Output data points.
            n_samples (int, optional): Number of samples to draw from the optimal distribution.
        
        Raises:
            ValueError: If the data is not within the domain of the input distribution.
        """
        if data_in is None or data_out is None:
            self.sampling = "optimal"
            self.n_samples = n_samples or sampling.optimal_sample_size(
                self.n_basis, risk=self.risk
            )
            start_time_sampling = time.perf_counter()
            data_in = sampling.sample_optimal_distribution(
                self.index_set, self.n_samples
            )
            end_time_sampling = time.perf_counter()
            self.data_in = self.tf_inv(data_in)
            start_time_evaluating = time.perf_counter()
            self.data_out = self.response(
                self.data_in.squeeze()
            )
            end_time_evaluating = time.perf_counter()

            self.sampling_time = end_time_sampling - start_time_sampling
            self.evaluation_time = end_time_evaluating - start_time_evaluating
        else:
            self.sampling = "arcsine"
            self.n_samples = len(data_in)
            self.data_in = data_in
            self.data_out = data_out

        if self.data_out.ndim == 1:
            self.data_out = self.data_out.reshape(-1, 1)

        self.output_dim = self.data_out.shape[1]

    def compute_data_matrix(self):
        r"""Computes the data matrix.

        The data matrix is computed by evaluating the basis functions
        $\{P_\lambda\}_{\lambda \in \Lambda}$ on the input data $\{\omega_i\}_{i=1}^N$.
        The data matrix is thus given by $\mathbf{M} \in \mathbb{R}^{N \times |\Lambda|}$
        with entries $\mathbf{M}_{ij} = P_{\lambda_j}(\omega_i)$,
        where the the multi-indices $\lambda_j$ are enumerated using `self.enumeration`.
        The resulting data matrix has shape `(n_samples, n_basis)`.

        """
        data_in = self.tf(self.data_in)
        self.data_matrix = legendre.evaluate_basis(self.index_set, data_in)

    def compute_weights(self):
        r"""Computes the optimal least squares weights.

        The optimal least squares weights are given as the inverse of the sample distibution.
        This ensures unbiasedness of the least squares estimator.
        When sampling from a different $\nu \ll \mu$ than the underlying reference measure $\mu$,
        the weights are computed as $w_i = \rho(\omega_i)^{-1}$,
        where $\rho = \frac{d \nu}{d \mu}$ denotes the Radon-Nikodym derivative.
        """
        if self.sampling == "optimal":
            self.weights = self.n_basis / np.sum(self.data_matrix ** 2, axis=1)
        elif self.sampling == "arcsine":
            self.weights = 1 / sampling.arcsine(self.tf(self.data_in))
        elif self.sampling == "uniform":
            self.weights = np.ones(self.n_samples)

    def compute_coefficients(self, solve_iteratively=False):
        r"""Computes the PCE coefficients.

        This method computes the PCE coefficients using weighted least squares regression.
        Depending on `solve_iteratively`, the least squares problem is solved either iteratively
        or directly using the normal equations.

        The coefficients are of the shape `(n_basis, output_dim)`.

        Args:
            solve_iteratively (bool, optional): Flag indicating whether to solve the least squares problem iteratively.
        """
        self.set_data(self.data_in, self.data_out, self.n_samples)
        self.compute_data_matrix()
        self.compute_weights()

        start_time_fitting = time.perf_counter()

        rhs = np.sqrt(self.weights[:, None] / self.n_samples) * self.data_out
        lhs = np.sqrt(self.weights[:, None] / self.n_samples) * self.data_matrix

        if solve_iteratively:
            self.coefficients, _, _, s = np.linalg.lstsq(lhs, rhs, rcond=None)
            self.condition_number = s[0] / s[-1]
        else:   
            self.coefficients = np.linalg.solve(lhs.T @ lhs, lhs.T @ rhs)

        end_time_fitting = time.perf_counter()
        self.fitting_time = end_time_fitting - start_time_fitting
