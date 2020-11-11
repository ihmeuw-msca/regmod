"""
Utility classes and functions
"""
from typing import Any
import numpy as np


def default_vec_factory(vec: Any, size: int, default_value: float,
                        vec_name: str = 'vec') -> np.ndarray:
    if vec is None:
        vec = np.repeat(default_value, size)
    elif np.isscalar(vec):
        vec = np.repeat(vec, size)
    else:
        vec = np.asarray(vec)
        check_size(vec, size, vec_name=vec_name)

    return vec


def check_size(vec: Any, size: int, vec_name: str = 'vec'):
    assert len(vec) == size, f"{vec_name} must length {size}."
