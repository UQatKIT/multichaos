import pytest

import numpy as np
import openturns as ot
import time

from src.multichaos.index_set import is_downward_closed
from src.multichaos.adaptive import AdaptivePCE


@pytest.fixture(scope="session")
def response():
    return lambda x: x.sum(axis=1)

@pytest.fixture(scope="session")
def approx_response():
    def helper(n, x):
        time.sleep(1e-10 * n)
        return (1 + 1 / n) * x.sum(axis=1)
    return lambda n: lambda x: helper(n, x)

@pytest.fixture(scope="session")
def response_vector():
    return lambda x: x

@pytest.fixture(scope="session")
def approx_response_vector():
    def helper(n, x):
        time.sleep(1e-10 * n)
        return (1 + 1 / n) * x
    return lambda n: lambda x: helper(n, x)

@pytest.fixture(scope="session")
def dist():
    return ot.JointDistribution(2 * [ot.Uniform(-1, 1)])

@pytest.fixture(scope="session")
def data(dist, response):
    sample = np.array(dist.getSampleByQMC(size=100))
    truth = response(sample).reshape(-1, 1)
    return sample, truth

@pytest.fixture(scope="session")
def data_vector(dist, response_vector):
    sample = np.array(dist.getSampleByQMC(size=100))
    truth = response_vector(sample)
    return sample, truth

@pytest.fixture(scope="session")
def ad_pce(approx_response, dist):
    ad_pce = AdaptivePCE(
        dist=dist,
        response=approx_response,
        n_steps=100,
    )
    return ad_pce

@pytest.fixture(scope="session")
def ad_pce_vector(approx_response_vector, dist):
    ad_pce_vector = AdaptivePCE(
        dist=dist,
        response=approx_response_vector,
        n_steps=10,
    )
    return ad_pce_vector

def test_index_set_size(ad_pce):
    assert len(ad_pce.multi_level_index_set) == ad_pce.n_steps + 1

def test_index_set_downward_closed(ad_pce):
    assert is_downward_closed(ad_pce.multi_level_index_set)

def test_index_set_downward_closed_vector(ad_pce_vector):
    assert is_downward_closed(ad_pce_vector.multi_level_index_set)

def test_l2_error(ad_pce, data):
    ad_pce.compute_coefficients()
    assert ad_pce.l2_error(*data) < 1e-4