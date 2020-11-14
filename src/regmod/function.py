"""
Function module
"""
from typing import Callable
from dataclasses import dataclass, field
import numpy as np


@dataclass
class SmoothFunction:
    name: str
    fun: Callable = field(repr=False)
    dfun: Callable = field(repr=False)
    d2fun: Callable = field(default=None, repr=False)


fun_list = [
    SmoothFunction(name="identity",
                   fun=lambda x: x,
                   dfun=lambda x: 1.0 if np.isscalar(x) else np.ones(len(x)),
                   d2fun=lambda x: 0.0 if np.isscalar(x) else np.zeros(len(x))),
    SmoothFunction(name="quad",
                   fun=lambda x: x**2,
                   dfun=lambda x: 2*x,
                   d2fun=lambda x: 2.0 if np.isscalar(x) else np.full(len(x), 2.0)),
    SmoothFunction(name="exp",
                   fun=np.exp,
                   dfun=np.exp,
                   d2fun=np.exp),
    SmoothFunction(name="expit",
                   fun=lambda x: 1/(1 + np.exp(-x)),
                   dfun=lambda x: np.exp(-x)/(1 + np.exp(-x))**2,
                   d2fun=lambda x: np.exp(-x)*(np.exp(-x) - 1)/(np.exp(-x) + 1)**3),
    SmoothFunction(name="log",
                   fun=np.log,
                   dfun=lambda x: 1/x,
                   d2fun=lambda x: -1/x**2),
    SmoothFunction(name="logit",
                   fun=lambda x: np.log(x/(1 - x)),
                   dfun=lambda x: 1/(x*(1 - x)),
                   d2fun=lambda x: (2*x - 1)/(x*(1 - x))**2),
]

fun_dict = {
    fun.name: fun
    for fun in fun_list
}
