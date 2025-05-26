r"""Adaptive multilevel polynomial chaos expansion.

This module provides a class for adaptive multilevel polynomial chaos expansion (PCE).

If the [problem rates][multichaos.multilevel.MultilevelPCE.set_problem_rates] required for the
[`MultilevelPCE`][multichaos.multilevel.MultilevelPCE] class are not known,
the following adaptive algorithm can be used.
For an adaptive algorithm, suitable discretizations of approximative responses
$\{Q_0, \ldots, Q_L\}$ and polynomial spaces $\{V_0, \ldots, V_L\}$ 
must be found for each level $l \in \{0, \ldots, L$, for some a-priori unknown $L \in \mathbb{N}$.
Then, as for non-adaptive multilevel PCE, the telescopic sum

$$
\begin{equation}
    \hat{Q}_{AD} := \Pi_{V_L} Q_0 + \sum_{l=0}^L \Pi_{V_{L-l}} (Q_l - Q_{l-1}),
\end{equation}
$$

can be employed. The [adaptive method](www.esaim-m2an.org/articles/m2an/abs/2020/02/m2an170180/m2an170180.html)
considered here is based on the iterative construction of a [downward-closed][multichaos.index_set.is_downward_closed]
multi-index set $I \subset \mathbb{N}_0^{d+1}$, which optimizes the ratio between gain and work of multi-indices in each iteration.
Note that, compared to the index-sets considered so far, $I$ contains multi-indices
with one additional dimension. The first $d$ entries of each multi-index in $I$ still correspond to the
polynomial space, while the last entry is used to distinguish between levels.
Therefore, we refer to $I$ as the *multi-level index set*.
This resulting set $I$ then defines the polynomial spaces and
functions to be used on each level by slicing along the last additional dimension
and expanding multi-indices using binary expansion.

Note:
    Instead of a tolerance $\epsilon > 0$, the adaptive algorithm is
    initialized with a fixed number of iteration steps.

Classes:
    AdaptivePCE: Adaptive multilevel PCE class.
"""

import time
import numpy as np

from . import base
from . import index_set
from . import sampling
from . import single_level
from . import utils


class AdaptivePCE(base.BasePCE):
    """Adaptive multilevel polynomial chaos expansion.
    
    This class implements an adaptive multilevel polynomial chaos expansion
    using optimal weighted least squares.
    
    Attributes:
        dist (ot.Distribution): Input distribution.
        response (callable): Response function.
        n_steps (int): Number of adaptive steps.
        multi_level_index_set (np.ndarray): Multi-level index set.
        C_n (int): Constant for discretization tuning.
        n_pow (int): Power for adaptive sample size.
        n (callable): Adaptive sample size function.
        samples (np.ndarray): Samples for all levels.
        evals (dict): Evaluation data for all levels.
        eval_times (dict): Evaluation times for all levels.
        gains (dict): Estimated gains for all indices.
        ratios (dict): Estimated ratios for all indices.
        construct_time (list): Time for each adaptive step.
        number_of_levels (int): Number of levels.
        Vk (list): Multi-index sets for each level.
        mk (np.ndarray): Number of multi-indices for each level.
        nl (np.ndarray): Sample sizes for each level.
        sample_sizes (np.ndarray): Optimal sample sizes for each level.
        level_estimators (list): PCE estimators for each level.
        output_dim (int): Output dimension.
        evaluation_times (dict): Average evaluation times for each level.
        fitting_times (np.ndarray): Fitting times for each level.
    """
    def __init__(
        self,
        dist,
        response,
        n_steps,
        C_n=1,
    ):
        """Initialize AdaptivePCE object.

        Args:
            dist (ot.Distribution): Input distribution.
            response (callable): Response function.
            n_steps (int): Number of adaptive steps.
            C_n (int, optional): Constant for discretization tuning.
        """
        super().__init__(dist)
        self.response = response
        self.n_steps = n_steps
        self.C_n = C_n

        self.discretization()
        self.construct_index_set()
        self.initialize_levels()
        self.set_data()

    def discretization(self):
        r"""Sets the discretization function.
        
        This method sets the discretization function as
        $n_l = C_n 2^l$ for level $l \in \mathbb{N}_0$.
        """
        self.n = lambda l: int(self.C_n * np.ceil(2 ** l))

    def generate_additional_samples(self, n_samples: int):
        """Generate additional samples.
        
        This method generates additional samples using the arcsine distribution
        and stacks them in the `samples` attribute.

        Args:
            n_samples (int): Number of additional samples.
        """
        samples = sampling.sample_arcsine((n_samples, self.input_dim))
        samples = self.tf_inv(samples)
        self.samples = np.vstack((self.samples, samples))

    def get_num_evals(self, level: int) -> int:
        """Return the total number of evaluations for a level.
        
        Return the current total number of stored evaluations
        in `self.evals[level]` for a given `level`.
        
        Args:
            level (int): Level.
        
        Returns:
            (int): Number of evaluations.
        """
        return self.evals[level].shape[0] if level in self.evals else 0

    def evaluate_samples(self, level: int, n_samples: int):
        """Evaluate samples for a level.
        
        This method evaluates `n_samples` samples for a given `level`
        and stores the evaluations in `self.evals[level]`.
        Additionally, the evaluation time is stored in `self.eval_times[level]`.

        Args:
            level (int): Level.
            n_samples (int): Number of samples.
        """
        s = time.perf_counter()
        level_response = self.response(self.n(level))
        n_evals = self.get_num_evals(level)
        samples = self.samples[n_evals:n_evals + n_samples]
        new_evals = level_response(samples)
        e = time.perf_counter()

        if level not in self.evals:
            self.evals[level] = new_evals
        else:
            self.evals[level] = np.concatenate([self.evals[level], new_evals])

        if level not in self.eval_times:
            self.eval_times[level] = (e - s, n_samples)
        else:
            total_time, count = self.eval_times[level]
            total_time += e - s
            count += n_samples
            self.eval_times[level] = (total_time, count)

    def estimate_gain(self, index: tuple[int]) -> float:
        """Estimate the gain of a multi-level index.

        This method estimates the gain of a multi-level index $(k, l) \in \mathbb{N}_0^{d+1}$,
        where $k \in \mathbb{N}_0^d$ and $l \in \mathbb{N}_0$, by estimating
        the norm of the projection of $Q_l - Q_{l-1}$ onto $P_k$,
        the [binary expansion][multichaos.index_set.binary_expansion] of $k$.

        Args:
            index (tuple[int]): Multi-level index with shape `(self.input_dim + 1,)`.
        
        Returns:
            (float): Estimated gain.
        """
        neighbours = list(index_set.get_lower_neighbours(index))

        gain = 0
        for nbr in neighbours:
            if nbr in self.gains:
                gain += self.gains[nbr]
                continue

            k, l = index_set.separate_index(nbr)

            V = index_set.union_binary_expansions(self.multi_level_index_set, l)
            n_samples = sampling.optimal_sample_size(len(V), risk=self.risk)

            if n_samples > len(self.samples):
                self.generate_additional_samples(n_samples - len(self.samples))

            n_evals = self.get_num_evals(l)
            if n_samples > n_evals:
                self.evaluate_samples(l, n_samples - n_evals)

            sample = self.samples[:n_samples]
            f = self.evals[l][:n_samples] - self.evals[l-1][:n_samples] if l > 0 else self.evals[l][:n_samples]

            pce = single_level.PCE(
                dist=self.dist,
                index_set=np.array(V),
                data_in=sample,
                data_out=f,
            )
            pce.compute_coefficients()

            under_consideration = index_set.binary_expansion(k)

            idx = [V.index(ix) for ix in under_consideration]

            aux = pce.coefficients[idx]
            self.gains[nbr] = np.linalg.norm(aux)
            gain += self.gains[nbr]
        gain /= len(neighbours)

        return gain

    def estimate_work(self, index: tuple[int]) -> float:
        """Estimate the work of a multi-level index.

        This method estimates the work of a multi-level index $(k, l) \in \mathbb{N}_0^{d+1}$,
        where $k \in \mathbb{N}_0^d$ and $l \in \mathbb{N}_0$, by estimating
        the time needed to evaluate one sample of $Q_l - Q_{l-1}$
        times the number of new samples needed.

        Args:
            index (tuple[int]): Multi-level index with shape `(self.input_dim + 1,)`.

        Returns:
            (float): Estimated work.
        """
        k, l = index_set.separate_index(index)

        if l not in self.eval_times:
            self.evaluate_samples(l, n_samples=1)

        time_per_level = {
            level: total_time / count for level, (total_time, count) in self.eval_times.items()
        }

        V = index_set.union_binary_expansions(self.multi_level_index_set, l)
        n_add = sampling.optimal_sample_size(
            len(V + index_set.binary_expansion(k)),
            risk=self.risk,
        )
        n = sampling.optimal_sample_size(
            len(V),
            risk=self.risk,
        )

        work = time_per_level[l] + time_per_level[l - 1] if l > 0 else time_per_level[l]
        work *= n_add - n

        return work

    def construct_index_set(self):
        """Construct the multi-level index set.

        This method constructs a problem-dependent, downward closed multi-level index set.
        """
        self.multi_level_index_set = [(0,) * (self.input_dim + 1)]
        admissible_indices = [
            (0,) * i + (1,) + (0,) * (self.input_dim - i)
            for i in range(self.input_dim + 1)
        ]

        self.samples = np.empty((0, self.input_dim))
        self.evals = {}
        self.eval_times = {}
        self.gains = {}
        self.ratios = {}

        self.construct_time = []
        for _ in range(self.n_steps):
            s = time.perf_counter()

            ratios = {}
            for index in admissible_indices:
                gain = self.estimate_gain(index)
                work = self.estimate_work(index)

                ratios[index] = gain / work
                self.ratios[index] = gain / work

            max_index = max(ratios, key=ratios.get)
            self.multi_level_index_set.append(max_index)

            admissible_indices.remove(max_index)
            for nbr in index_set.get_upper_neighbours(max_index):
                if index_set.is_downward_closed(self.multi_level_index_set + [nbr]):
                    admissible_indices.append(nbr)

            self.construct_time.append(time.perf_counter() - s)

        self.evaluation_times = {
            level: total_time / count for level, (total_time, count) in self.eval_times.items()
        }

    def initialize_levels(self):
        """Initializes a PCE object on each level.

        This method uses the `multi_level_index_set` attribute to initialize a
        PCE object for each level $l \in \{0, \ldots, L\}$
        and stores them in the `level_estimators` list attribute.
        """
        self.number_of_levels = max(idx[-1] for idx in self.multi_level_index_set)
        # multi index sets for each level
        self.Vk = [
            np.array(index_set.union_binary_expansions(self.multi_level_index_set, l))
            for l in range(self.number_of_levels + 1)
        ]
        self.mk = np.array(list(map(len, self.Vk)))
        self.nl = np.array(list(map(self.n, range(self.number_of_levels + 1))))
        self.sample_sizes = np.array([
            sampling.optimal_sample_size(len(V), risk=self.risk)
            for V in self.Vk
        ])

        # initialize estimator for each level
        self.level_estimators = []
        for l in range(self.number_of_levels + 1):
            pce = single_level.PCE(
                dist=self.dist,
                index_set=self.Vk[l],
            )
            self.level_estimators.append(pce)

    def set_data(self):
        """Sets the data for each level PCE.

        For level $l=0$, the response $Q_{n_0}$ and for the remaining
        $l \in \{1, \ldots, L\}$, the response differences $Q_{n_l} - Q_{n_{l-1}}$
        are stored in the PCE objects.
        """
        # at this point, additional samples could be needed for some levels
        for l in range(self.number_of_levels + 1):
            n_samples = self.sample_sizes[l]

            # generate more samples if needed
            if n_samples > len(self.samples):
                self.generate_additional_samples(n_samples - len(self.samples))

            # evaluate more samples if needed
            for l_ in [l, l - 1] if l > 0 else [l]:
                n_evals = self.get_num_evals(l_)
                if n_samples > n_evals:
                    self.evaluate_samples(l_, n_samples - n_evals)

        for l, pce in enumerate(self.level_estimators):
            n_samples = self.sample_sizes[l]

            sample = self.samples[:n_samples]
            f_ = self.evals[l][:n_samples] - self.evals[l-1][:n_samples] if l > 0 else self.evals[l][:n_samples]
            pce.set_data(
                data_in=sample,
                data_out=f_,
            )

        self.output_dim = self.level_estimators[0].output_dim

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