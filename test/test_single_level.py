import pytest

import numpy as np
import openturns as ot

from src.multichaos.sampling import sample_optimal_distribution
from src.multichaos.sampling import optimal_sample_size
from src.multichaos.index_set import generate_index_set
from src.multichaos.single_level import PCE


@pytest.fixture(scope="session")
def type():
    return "TD"

@pytest.fixture(scope="session")
def max_degree():
    return 6

@pytest.fixture(scope="session")
def index_set(type, max_degree):
    return generate_index_set(type, max_degree, dim=2)

@pytest.fixture(scope="session")
def data_matrix(index_set):
    dist = ot.JointDistribution(2 * [ot.Uniform(-1, 1)])

    N = optimal_sample_size(len(index_set), risk=.9)
    sample = sample_optimal_distribution(index_set, size=N)
    f = np.ones(len(sample))

    pce = PCE(
        dist=dist,
        index_set=index_set,
        data_in=sample,
        data_out=f
    )
    pce.set_data(pce.data_in, pce.data_out, pce.n_samples)
    pce.compute_data_matrix()
    pce.compute_weights()

    return np.sqrt(pce.weights[:, None] / N) * pce.data_matrix

def test_assemble_data_matrix(data_matrix, index_set):
    N = optimal_sample_size(len(index_set), risk=.9)
    assert data_matrix.shape == (N, len(index_set))

def test_gram_matrix(data_matrix, index_set):
    dim = 2
    G = data_matrix.T @ data_matrix / 2 ** dim
    print(np.linalg.cond(G))
    assert np.linalg.cond(G) <= 3.
    assert np.linalg.norm(G - np.eye(len(index_set)), ord=2) <= .5


@pytest.mark.parametrize("deg", [0, 1, 2, 3])
def test_polynomial_fit(deg):
    response = lambda x: (x ** deg).sum(axis=1)
    dist = ot.JointDistribution(2 * [ot.Uniform(-1, 1)])
    index_set = generate_index_set("TD", deg, dim=2)

    pce = PCE(
        dist=dist,
        index_set=index_set,
        response=response
    )
    pce.compute_coefficients()

    sample = np.array(dist.getSampleByQMC(size=100))
    err = pce.l2_error(sample, response(sample).reshape(-1, 1))

    assert np.isclose(err, 0.)

def test_polynomial_fit_vector():
    deg = 2
    response = lambda x: x ** np.arange(1, deg + 1)
    dist = ot.JointDistribution(2 * [ot.Uniform(-1, 1)])
    index_set = generate_index_set("TD", deg, dim=2)

    pce = PCE(
        dist=dist,
        index_set=index_set,
        response=response
    )
    pce.compute_coefficients()

    sample = np.array(dist.getSampleByQMC(size=100))
    err = pce.l2_error(sample, response(sample))

    assert np.isclose(err, 0.)