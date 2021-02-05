"""
Function module
"""
from dataclasses import dataclass, field
from typing import Callable

import numpy as np


@dataclass
class SmoothFunction:
    name: str
    fun: Callable = field(repr=False)
    dfun: Callable = field(repr=False)
    d2fun: Callable = field(default=None, repr=False)


def identity_fun(x):
    return x


def identity_dfun(x):
    return 1.0 if np.isscalar(x) else np.ones(len(x))


def identity_d2fun(x):
    return 0.0 if np.isscalar(x) else np.zeros(len(x))


def quad_fun(x):
    return x**2


def quad_dfun(x):
    return 2.0*x


def quad_d2fun(x):
    return 2.0 if np.isscalar(x) else np.full(len(x), 2.0)


def exp_fun(x):
    return np.exp(x)


def exp_dfun(x):
    return np.exp(x)


def exp_d2fun(x):
    return np.exp(x)


def expit_fun(x):
    neg_indices = x < 0
    z = np.exp(-np.sqrt(x*x))
    y = 1/(1 + z)
    if np.isscalar(x):
        if neg_indices:
            y = 1 - y
    else:
        y[neg_indices] = 1 - y[neg_indices]
    return y


def expit_dfun(x):
    z = np.exp(-np.sqrt(x*x))
    y = z/(1 + z)**2
    return y


def expit_d2fun(x):
    neg_indices = x < 0
    z = np.exp(-np.sqrt(x*x))
    y = z*(z - 1)/(z + 1)**3
    if np.isscalar(x):
        if neg_indices:
            y = -y
    else:
        y[neg_indices] = -y[neg_indices]
    return y


def log_fun(x):
    return np.log(x)


def log_dfun(x):
    return 1/x


def log_d2fun(x):
    return -1/x**2


def logit_fun(x):
    return np.log(x/(1 - x))


def logit_dfun(x):
    return 1/(x*(1 - x))


def logit_d2fun(x):
    return (2*x - 1)/(x*(1 - x))**2


fun_list = [
    SmoothFunction(name="identity", fun=identity_fun, dfun=identity_dfun, d2fun=identity_d2fun),
    SmoothFunction(name="quad", fun=quad_fun, dfun=quad_dfun, d2fun=quad_d2fun),
    SmoothFunction(name="exp", fun=exp_fun, dfun=exp_dfun, d2fun=exp_d2fun),
    SmoothFunction(name="expit", fun=expit_fun, dfun=expit_dfun, d2fun=expit_d2fun),
    SmoothFunction(name="log", fun=log_fun, dfun=log_dfun, d2fun=log_d2fun),
    SmoothFunction(name="logit", fun=logit_fun, dfun=logit_dfun, d2fun=logit_d2fun),
]


fun_dict = {
    fun.name: fun
    for fun in fun_list
}
