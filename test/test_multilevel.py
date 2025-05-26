import pytest

import numpy as np
import openturns as ot

from src.multichaos.multilevel import MultilevelPCE


@pytest.fixture(scope="session")
def response():
    return lambda x: x.sum(axis=1)

@pytest.fixture(scope="session")
def approx_response():
    return lambda n: \
        lambda x: (1 + 1 / n) * x.sum(axis=1)

@pytest.fixture(scope="session")
def dist():
    return ot.JointDistribution(2 * [ot.Uniform(-1, 1)])

@pytest.fixture(scope="session")
def data(dist, response):
    sample = np.array(dist.getSampleByQMC(size=100))
    truth = response(sample).reshape(-1, 1)
    return sample, truth

@pytest.mark.parametrize("eps", [1., .1, .05])
def test_error_under_tolerance(eps, dist, approx_response, data):
    rates = {
        "beta": 1,
        "gamma": 1,
        "sigma": 2,
        "alpha": 4
    }
    ml_pce = MultilevelPCE(
        dist=dist,
        response=approx_response,
        index_set_type="TP",
        rates=rates,
        tol=eps
    )
    ml_pce.compute_coefficients()
    assert ml_pce.l2_error(*data) < eps