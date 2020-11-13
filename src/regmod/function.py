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
    SmoothFunction(name="exp",
                   fun=np.exp,
                   dfun=np.exp,
                   d2fun=np.exp),
    SmoothFunction(name="expit",
                   fun=lambda x: 1/(1 + np.exp(-x)),
                   dfun=lambda x: np.exp(-x)/(1 + np.exp(-x))**2,
                   d2fun=lambda x: np.exp(-x)*(np.exp(-x) - 1)/(np.exp(-x) + 1)**3),
]

fun_dict = {
    fun.name: fun
    for fun in fun_list
}
