"""
Variable module
"""
from __future__ import annotations
from typing import List, Union
from collections.abc import Iterable
from dataclasses import dataclass, field
from copy import deepcopy
import numpy as np
from .data import Data
from .prior import Prior, GaussianPrior, UniformPrior, SplinePrior


@dataclass
class Variable:
    name: str
    priors: List[Prior] = field(default_factory=list, repr=False)
    gprior: Prior = field(default=None, init=False, repr=False)
    uprior: Prior = field(default=None, init=False, repr=False)

    def __post_init__(self):
        self.process_priors()

    def process_priors(self):
        for prior in self.priors:
            if isinstance(prior, SplinePrior):
                continue
            if isinstance(prior, GaussianPrior):
                if self.gprior is not None and self.gprior != prior:
                    raise ValueError("Can only provide one Gaussian prior.")
                self.gprior = prior
                assert self.gprior.size == self.size, \
                    "Gaussian prior size not match."
            elif isinstance(prior, UniformPrior):
                if self.uprior is not None and self.uprior != prior:
                    raise ValueError("Can only provide one Uniform prior.")
                self.uprior = prior
                assert self.uprior.size == self.size, \
                    "Uniform prior size not match."
            else:
                raise ValueError("Unknown prior type.")

    @property
    def size(self):
        return 1

    def add_priors(self, priors: Union[Prior, List[Prior]]):
        if not isinstance(priors, list):
            priors = [priors]
        self.priors.extend(priors)
        self.process_priors()

    def rm_priors(self, indices: Union[int, List[int], List[bool]]):
        if isinstance(indices, int):
            indices = [indices]
        else:
            assert isinstance(indices, Iterable), \
                "Indies must be int, List[int], or List[bool]."
        if all([not isinstance(index, bool) and isinstance(index, int)
                for index in indices]):
            indices = [i in indices for i in range(len(self.priors))]
        assert all([isinstance(index, bool) for index in indices]), \
            "Index type not consistent."
        assert len(indices) == len(self.priors), \
            "Index size not match with number of priors."
        self.priors = [self.priors[i] for i, index in enumerate(indices)
                       if not index]
        self.gprior = None
        self.uprior = None
        self.process_priors()

    def get_mat(self, data: Data) -> np.ndarray:
        return data.get_covs(self.name)

    def get_gvec(self) -> np.ndarray:
        if self.gprior is None:
            gvec = np.repeat([[0.0], [np.inf]], self.size, axis=1)
        else:
            gvec = np.vstack([self.gprior.mean, self.gprior.sd])
        return gvec

    def get_uvec(self) -> np.ndarray:
        if self.uprior is None:
            uvec = np.repeat([[-np.inf], [np.inf]], self.size, axis=1)
        else:
            uvec = np.vstack([self.uprior.lb, self.uprior.ub])
        return uvec

    def copy(self) -> Variable:
        return deepcopy(self)
