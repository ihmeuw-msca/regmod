"""
Test Tobit Model
"""
import numpy as np
import pandas as pd
import pytest

from regmod.models import TobitModel
from regmod.variable import Variable


@pytest.fixture
def df():
    n = 100
    x = np.random.normal(size=n)
    y = np.random.normal(size=n)
    return pd.DataFrame({"x": x, "y": y, "z": np.where(y > 0, y, 0)})


@pytest.fixture
def param_specs():
    specs = {
        "mu": {"variables": [Variable("x"), Variable("intercept")]},
        "sigma": {"variables": [Variable("intercept")]},
    }
    return specs


@pytest.fixture
def model(param_specs):
    model = TobitModel(y="z", param_specs=param_specs)
    return model


def test_jax_inv_link(param_specs):
    """User-supplied inv_link functions replaced with JAX versions."""
    param_specs["mu"]["inv_link"] = "identity"
    param_specs["sigma"]["inv_link"] = "exp"
    model = TobitModel(y="z", param_specs=param_specs)
    assert model.params[0].inv_link.name == "identity_jax"
    assert model.params[1].inv_link.name == "exp_jax"


def test_neg_obs(df, param_specs):
    """ValueError if data contains negative observations."""
    with pytest.raises(ValueError, match="requires non-negative observations"):
        model = TobitModel(y="y", param_specs=param_specs)
        model._parse(df)


def test_vcov_output(df, model):
    """New get_vcov method matches old version."""
    model.fit(df)
    coef = model.coef
    data = model._parse(df)

    # Old version
    H = model.hessian(data, coef)
    eig_vals, eig_vecs = np.linalg.eig(H)
    inv_H = (eig_vecs / eig_vals).dot(eig_vecs.T)
    J = model.jacobian2(data, coef)
    vcov_old = inv_H.dot(J)
    vcov_old = inv_H.dot(vcov_old.T)

    # New version
    vcov_new = model.get_vcov(data, coef)

    assert np.allclose(vcov_old, vcov_new)


def test_pred_values(df, model):
    """Predicted mu_censored >= 0 and sigma > 0."""
    model.fit(df)
    df_pred = model.predict(df)
    assert np.all(df_pred["mu_censored"] >= 0)
    assert np.all(df_pred["sigma"] > 0)


def test_model_no_variables():
    num_obs = 5
    df = pd.DataFrame(
        {
            "obs": np.random.rand(num_obs) * 10,
            "offset": np.ones(num_obs),
        }
    )
    model = TobitModel(
        y="obs", param_specs={"mu": {"offset": "offset"}, "sigma": {"offset": "offset"}}
    )
    data = model._parse(df)
    coef = np.array([])
    grad = model.gradient(data, coef)
    hessian = model.hessian(data, coef)
    assert grad.size == 0
    assert hessian.size == 0

    model.fit(df)
    assert model.result == "no parameter to fit"


def test_model_one_variable():
    num_obs = 5
    df = pd.DataFrame(
        {
            "obs": np.random.rand(num_obs) * 10,
            "offset": np.ones(num_obs),
        }
    )
    model = TobitModel(
        y="obs",
        param_specs={
            "sigma": {"offset": "offset"},
            "mu": {"variables": [Variable("intercept")]},
        },
    )
    model.fit(df)
    assert model.coef.size == 1
