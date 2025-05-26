r"""
Base class for Polynomial Chaos Expansion (PCE) models.

This module provides the base class for Polynomial Chaos Expansion (PCE) models.
The class provides the basic functionality for constructing and evaluating PCE models.

Classes:
    BasePCE: Base class for Polynomial Chaos Expansion (PCE) models.
"""

import numpy as np
import openturns as ot

from . import legendre


class BasePCE:
    r"""Base class for Polynomial Chaos Expansion (PCE) models.

    This class provides the basic functionality for constructing and evaluating PCE models.
    The class is designed to be inherited by other PCE models.

    Attributes:
        dist (ot.Distribution): Input probability distribution.
        index_set (np.ndarray): Index set of the PCE.
        n_basis (int): Number of basis functions in the PCE.
        risk (float): [Quasi-optimality probability][multichaos.sampling.optimal_sample_size].
        coefficients (np.ndarray): Coefficients of the PCE.
        input_dim (int): Input dimension.
        output_dim (int): Output dimension.
        tf (callable): Transformation function.
        tf_inv (callable): Inverse transformation function.
    """
    def __init__(self, dist, index_set=None, risk=0):
        """Initialize Base PCE object.

        Args:
            dist (ot.Distribution): Input probability distribution.
            index_set (np.ndarray): Index set of the PCE.
            risk (float): [Quasi-optimality probability][multichaos.sampling.optimal_sample_size].
        """
        self.dist = dist
        self.index_set = index_set
        self.risk = risk

        self.n_basis = len(index_set) if index_set is not None else None
        self.set_transformation()

    def set_transformation(self):
        r"""Set the transformation functions.
        
        This method sets up the necessary components for transforming the input
        distribution to the underlying measure of the PCE.
        """
        self.input_dim = self.dist.getDimension()
        self.enumeration = ot.LinearEnumerateFunction(self.input_dim)
        self.polynomial_family = [
            ot.StandardDistributionPolynomialFactory(
                self.dist.getMarginal([i]))
            for i in range(self.input_dim)]
        self.product_basis = ot.OrthogonalProductPolynomialFactory(
            self.polynomial_family, self.enumeration)
        self.transformation = ot.DistributionTransformation(
            self.dist, self.product_basis.getMeasure())
        self.tf = lambda x: np.array(self.transformation(x))
        self.tf_inv = lambda x: np.array(self.transformation.inverse()(x))
    
    def __call__(self, data_in):
        """Evaluate the PCE at the given input data.

        Args:
            data_in (np.ndarray): Input data points.
        """
        data_in = self.tf(data_in)
        eval_matrix = legendre.evaluate_basis(self.index_set, data_in)
        return eval_matrix.dot(self.coefficients)

    def compute_mean(self):
        r"""Compute the mean of the PCE.

        This returns the mean of the PCE $P = \sum_{\lambda \in \Lambda} c_\lambda P_\lambda$,
        which, due to orthonormality, is given by $\mathbb{E}[P] = c_\mathbf{0}$.

        Returns:
            (float): Mean of the PCE.
        """
        return self.coefficients[0]
    
    def compute_variance(self):
        r"""Compute the variance of the PCE.

        This returns the variance of the PCE $P = \sum_{\lambda \in \Lambda} c_\lambda P_\lambda$,
        which, due to orthonormality, is given by
        $\mathbb{V}[P] = \sum_{\lambda \in \Lambda \setminus \{\mathbf{0}\}} c_\lambda^2$.

        Returns:
            (float): Variance of the PCE.
        """
        return np.sum(self.coefficients[1:] ** 2, axis=0)

    def l2_error(self, data_in, data_out):
        r"""Compute the $L^2$ error of the PCE.

        This method computes the root mean squared error as an estimate to the
        $L^2$ error of the PCE with respect to the given data, i.e.

        $$
        \begin{equation}
        \|Q - \hat{Q}\|_{L^2} \approx \left( \frac{1}{N} \sum_{i=1}^N (Q_i - \hat{Q}(\omega_i))^2 \right)^{1/2},
        \end{equation}
        $$

        where $\{(\omega_i, Q_i)\}_{i=1}^N$ denotes the data and $\hat{Q}$ the PCE model.

        Args:
            data_in (np.ndarray): Input data points of shape `(n_samples, input_dim)`.
            data_out (np.ndarray): Output data points of shape `(n_samples, output_dim)`.

        Returns:
            (float): Root mean squared error.
        """
        if data_out.ndim != 2:
            raise ValueError(f"Expected a 2D array, but got {data_out.ndim}D.")
        pred = self(data_in)
        err = ((pred - data_out) ** 2).sum(axis=1).mean()
        return np.sqrt(err)