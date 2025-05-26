r"""Multilevel polynomial chaos expansion.

This module provides a class for multilevel polynomial chaos expansion (PCE).
Given a response surface $Q: \Omega \rightarrow \mathbb{R}^k$,
we aim to approximate it using a sequence of approximative response surfaces $Q_n: \Omega \rightarrow \mathbb{R}^k$.
For a fixed level $L \in \mathbb{N}$, consider a sequence of increasing discretization parameters $\{n_0, \ldots, n_L\} \in \mathbb{N}^{L+1}$.
Consider the telescopic sum

$$
\begin{equation}
    Q \approx Q_{n_L} = Q_{n_0} + \sum_{l=0}^L Q_{n_l} - Q_{n_{l-1}}.
\end{equation}
$$

The multilevel idea is now to approximate each term in the sum using an independent [optimally weighted PCE][multichaos.single_level].
For this, similary, consider a sequence of increasing polynomial space parameters $\{m_0, \ldots, m_L\} \in \mathbb{N}^{L+1}$,
and define the multilevel PCE as

$$
\begin{equation}
    \hat{Q} := \Pi_{V_{m_L}} Q_{n_0} + \sum_{l=0}^L \Pi_{V_{m_{L-l}}} (Q_{n_l} - Q_{n_{l-1}}),
\end{equation}
$$

where $V_m$ denotes the polynomial space associated to parameter $m$.

Classes:
    MultilevelPCE: Multilevel PCE class.
"""
import numpy as np

from . import base
from . import index_set
from . import sampling
from . import single_level
from . import utils


class MultilevelPCE(base.BasePCE):
    """Multilevel polynomial chaos expansion.

    This class implements a multilevel polynomial chaos expansion
    using optimal weighted least squares.

    Attributes:
        dist (ot.Distribution): Input distribution.
        response (callable): Response function.
        index_set_type (str): Index set type.
        rates (dict): Rates of convergence.
        tol (float): Error tolerance.
        C_n (int): Constant for discretization tuning.
        C_m (int): Constant for polynomial space tuning.
        data_in (np.ndarray): Input data points.
        data_out (np.ndarray): Output data points.
    """
    def __init__(
            self,
            dist,
            response,
            index_set_type,
            rates,
            tol,
            C_n=1,
            C_m=1,
            data_in=None,
            data_out=None,
    ):
        """Initialize Multilevel PCE object.
        
        Args:
            dist (ot.Distribution): Input distribution.
            response (callable): Response function.
            index_set_type (str): Index set type.
            rates (dict): Rates of convergence.
            tol (float): Error tolerance.
            C_n (int, optional): Constant for discretization tuning.
            C_m (int, optional): Constant for polynomial space tuning.
            data_in (list[np.ndarray], optional): Input data points.
            data_out (list[np.ndarray], optional): Output data points.
        """
        super().__init__(dist)
        self.response = response
        self.index_set_type = index_set_type
        self.rates = rates
        self.tol = tol
        self.C_n = C_n
        self.C_m = C_m

        self.verbose = False

        self.set_problem_rates(**rates)
        self.set_asymptotic_cost_rates()
        self.initialize_levels(tol=tol)
        self.set_data(data_in, data_out)

    def set_problem_rates(self, beta: float, gamma: float, alpha: float, sigma: float):
        r"""Sets the rates of convergence for the problem.

        Given a response $Q$ and approximative responses $Q_n$ for $n \in \mathbb{N}$,
        e.g., those obtained using a PDE solver with $n \in \mathbb{N}$
        discretization points, we make the following assumptions:

        - $\|Q - Q_n\| \lesssim n^{-\beta}$
        - $\text{Work}(Q_n) \lesssim n^{\gamma}$
        - $\inf_{v \in V_m} \|Q - v\| \lesssim m^{-\alpha}$
        - $\text{dim}(V_m) \lesssim m^{\sigma}$

        Note:
            If some of these rates are unknown for the problem at hand,
            the user can consider the [apative PCE][multichaos.adaptive].
    
        Args:
            beta (float): Rate for discretization error.
            gamma (float): Rate for work per sample.
            alpha (float): Rate for polynomial approximability.
            sigma (float): Rate for polynomial dimensionality.
        """
        self.beta = beta
        self.gamma = gamma
        self.sigma = sigma
        self.alpha = alpha

    def set_asymptotic_cost_rates(self):
        r"""Sets the asymptotic cost rates for the problem.

        Given the [problem rates][multichaos.multilevel.MultilevelPCE.set_problem_rates]
        $\beta, \gamma, \alpha, \sigma > 0$,
        the costs for the ML-PCE to reach an error $\epsilon > 0$ are given by

        $$
        \begin{equation}
        \text{Work} ( \hat{Q}) \leq \epsilon^{- \lambda} \log (\epsilon^{-1})^t \log \log (\epsilon^{-1}),
        \end{equation}
        $$

        where $\lambda = \max \left(\gamma / \beta,\, \sigma / \alpha\right)$ and

        $$
        t = \begin{cases}
            2, & \gamma / \beta < \sigma / \alpha, \\
            3 + \sigma / \alpha, & \gamma / \beta = \sigma / \alpha, \\
            1, & \gamma / \beta > \sigma / \alpha.
            \end{cases}
        $$
        """
        b, g, s, a = self.beta, self.gamma, self.sigma, self.alpha
        case = g / b - s / a
        l = max(g / b, s / a)
        if case > 0:
            t = 1
        elif case < 0:
            t = 2
        else:
            t = 3 + s / a
        self.asymptotic_cost_rates = (l, t)

    def set_number_of_levels(self, tol: float) -> int:
        r"""Sets the number of levels for the ML-PCE.

        Given an error tolerance $\epsilon > 0$, the number of levels $L \in \mathbb{N}$
        is chosen such that for the $L^2$ error $\|\hat{Q} - Q\|_{L^2} \leq \epsilon$.

        Args:
            tol (float): Error tolerance.
        """
        self.tol = tol
        b, g, s, a = self.beta, self.gamma, self.sigma, self.alpha
        case = g / b - s / a
        if case > 0:
            L = - (g + b) / b * np.log(tol)
        elif case < 0:
            L = - (s + a) / a * np.log(tol)
        else:
            grid = np.linspace(1, 100, 1000)
            ix = np.argmin(np.abs(np.exp(-grid * a / (s + a)) * (grid + 1) - tol))
            L = grid[ix]

        return np.ceil(L).astype(int)
        # return int(L)

    def set_tuning_params(self, C_n: int=1, C_m: int=1):
        r"""Sets the tuning parameters for the ML-PCE.

        The discretization $n_l$ and polynomial space parameters $m_k$ are set as
        $n_l := C_n \exp (l / (\beta + \gamma))$ and $m_k := C_m \exp (k / (\alpha + \sigma))$
        for $k, l \in \{0, \ldots, L\}$.

        Args:
            C_n (int): Constant for discretization tuning.
            C_m (int): Constant for polynomial space tuning.

        Note:
            The constants $C_n$ and $C_m$ should be chosen sufficiently large.
        
        Returns:
            (tuple[np.ndarray, np.ndarray]): Tuning parameters for sample size and polynomial space.
        """
        b, g, s, a = self.beta, self.gamma, self.sigma, self.alpha
        L = self.number_of_levels

        nl = C_n * np.exp(np.arange(L + 1) / (g + b))
        mk = C_m * np.exp(np.arange(L + 1) / (s + a))

        return mk, nl

    def set_sample_sizes(self):
        """Sets the optimal sample size for each level.

        The [optimal sample size][multichaos.sampling.optimal_sample_size]
        is computed using the approximative dimension $m_k^\sigma$
        for each of the polynomial subspaces $V_{m_k}$ for $k \in \{0, \ldots, L\}$.

        Returns:
            (np.ndarray): Sample sizes for each level.
        """
        sample_sizes = sampling.optimal_sample_size(
            self.mk ** self.sigma,
            risk=self.risk
        )
        return sample_sizes

    def initialize_levels(self, tol=None, L=None):
        """Initializes a PCE object on each level.

        If the number of levels $L \in \mathbb{N}$ is not provided, it is computed
        using the error tolerance $\epsilon > 0$.
        This method initializes a PCE object for each level $l \in \{0, \ldots, L\}$
        with the corresponding tuning parameters $(n_l, m_l)$
        and stores them in the `level_estimators` list attribute.

        Args:
            eps (float): Error tolerance.
            L (int): Number of levels.
        """
        self.number_of_levels = L or self.set_number_of_levels(tol)
        self.mk, self.nl = self.set_tuning_params(self.C_n, self.C_m)
        self.sample_sizes = self.set_sample_sizes()

        # rounding up tuning parameters to integers
        # only now because computed `sample_sizes` using float `mk` previously
        self.mk = np.ceil(self.mk).astype(int)
        self.nl = np.ceil(self.nl).astype(int)

        # initialize estimator for each level
        self.level_estimators = []
        for l in range(self.number_of_levels + 1):
            m = self.mk[::-1][l]
            multi_index_set = index_set.generate_index_set(
                self.index_set_type,
                m,
                self.input_dim,
            )
            pce = single_level.PCE(
                dist=self.dist,
                index_set=multi_index_set
            )
            self.level_estimators.append(pce)

    def set_data(self, data_in=None, data_out=None):
        """Sets the data for each level PCE.

        For level $l=0$, the response $Q_{n_0}$ and for the remaining
        $l \in \{1, \ldots, L\}$, the response differences $Q_{n_l} - Q_{n_{l-1}}$
        are evaluated in the optimal number of samples $N_l$.
        """
        if data_in is None or data_out is None:
            for l, pce in enumerate(self.level_estimators):
                n = self.nl[l]
                N = self.sample_sizes[::-1][l]

                f_l = self.response(n)
                f_l_ = self.response(self.nl[l - 1]) if l > 0 else lambda _: 0
                f_ = lambda x: f_l(x) - f_l_(x)

                pce.response = f_
                pce.set_data(n_samples=N)
        else:
            for l, pce in enumerate(self.level_estimators):
                pce.set_data(data_in[l], data_out[l])
    
        self.output_dim = self.level_estimators[0].output_dim

        if data_in is None or data_out is None:
            self.sampling_times = np.array([pce.sampling_time for pce in self.level_estimators])
            self.evaluation_times = np.array([pce.evaluation_time for pce in self.level_estimators])

    def compute_coefficients(self):
        """Computes the coefficients.

        This method computes the PCE coefficients by running the
        [`compute_coefficients`][multichaos.single_level.PCE.compute_coefficients]
        method of each level estimator. The coefficients are then aggregated
        using the [`aggregate_coefficients`][multichaos.utils.aggregate_coefficients] function.
        """
        for pce in self.level_estimators:
            pce.compute_coefficients()
        self.fitting_times = np.array([pce.fitting_time for pce in self.level_estimators])
        self.index_set, self.coefficients = utils.aggregate_coefficients(
            self.level_estimators
        )