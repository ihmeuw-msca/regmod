"""
Test prior module
"""
import numpy as np
import pytest
from xspline import XSpline
from regmod import GaussianPrior, UniformPrior, SplineGaussianPrior, SplineUniformPrior


@pytest.fixture
def spline():
    return XSpline(knots=np.linspace(0.0, 1.0, 5), degree=3)


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


@pytest.mark.parametrize("order", [0, 1, 2])
@pytest.mark.parametrize("size", [10])
@pytest.mark.parametrize("domain_type", ["rel", "abs"])
def test_spline_gaussian(order, domain_type, size, spline):
    prior = SplineGaussianPrior(
        mean=0.0,
        sd=1.0,
        order=order,
        size=size,
        domain_type=domain_type
    )

    mat = prior.get_mat(spline)
    assert mat.shape == (prior.size, spline.num_spline_bases)


@pytest.mark.parametrize("order", [0, 1, 2])
@pytest.mark.parametrize("size", [10])
@pytest.mark.parametrize("domain_type", ["rel", "abs"])
def test_spline_uniform(order, domain_type, size, spline):
    prior = SplineUniformPrior(
        lb=0.0,
        ub=1.0,
        order=order,
        size=size,
        domain_type=domain_type
    )

    mat = prior.get_mat(spline)
    assert mat.shape == (prior.size, spline.num_spline_bases)
