"""
Prior module
"""
from typing import List, Any
from collections.abc import Iterable
from dataclasses import dataclass, field
import numpy as np
from .utils import default_vec_factory


@dataclass
class Prior:
    size: int = field(default=None, repr=False)

    def process_size(self, vecs: List[Any]):
        if self.size is None:
            sizes = [len(vec) for vec in vecs if isinstance(vec, Iterable)]
            sizes.append(1)
            self.size = max(sizes)

        if self.size <= 0 or not isinstance(self.size, int):
            raise ValueError("Size of the prior must be a positive integer.")


@dataclass
class GaussianPrior(Prior):
    mean: np.ndarray = None
    sd: np.ndarray = None

    def __post_init__(self):
        self.process_size([self.mean, self.sd])
        self.mean = default_vec_factory(self.mean, self.size, 0.0, vec_name='mean')
        self.sd = default_vec_factory(self.sd, self.size, np.inf, vec_name='sd')
        assert all(self.sd > 0.0), "Standard deviation must be all positive."


@dataclass
class UniformPrior(Prior):
    lb: np.ndarray = None
    ub: np.ndarray = None

    def __post_init__(self):
        self.process_size([self.lb, self.ub])
        self.lb = default_vec_factory(self.lb, self.size, -np.inf, vec_name='lb')
        self.ub = default_vec_factory(self.ub, self.size, np.inf, vec_name='ub')
        assert all(self.lb <= self.ub), "Lower bounds must be less or equal than upper bounds."
