"""
Test prior module
"""
import numpy as np
import pytest
from regm import GaussianPrior, UniformPrior


@pytest.mark.parametrize(('mean', 'sd', 'size'),
                         [(np.zeros(5), 1.0, None),
                          (0.0, np.ones(5), None),
                          (0.0, 1.0, 5),
                          (None, None, 5)])
def test_gaussian(mean, sd, size):
    gaussian = GaussianPrior(mean=mean, sd=sd, size=size)
    assert gaussian.size == 5


@pytest.mark.parametrize(('lb', 'ub', 'size'),
                         [(np.zeros(5), 1.0, None),
                          (0.0, np.ones(5), None),
                          (0.0, 1.0, 5),
                          (None, None, 5)])
def test_uniform(lb, ub, size):
    uniform = UniformPrior(lb=lb, ub=ub, size=size)
    assert uniform.size == 5
