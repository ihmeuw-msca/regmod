"""
Variable module
"""
from collections.abc import Iterable
from copy import deepcopy
from dataclasses import dataclass, field
from typing import List, Union

import numpy as np
from xspline import XSpline

from regmod.data import Data
from regmod.prior import (Prior, GaussianPrior, UniformPrior,
                          LinearPrior, LinearGaussianPrior, LinearUniformPrior,
                          SplinePrior, SplineGaussianPrior, SplineUniformPrior)
from regmod.utils import SplineSpecs


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
            if isinstance(prior, LinearPrior):
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

    def check_data(self, data: Data):
        if self.name not in data.col_covs:
            raise ValueError(f"Data do not contain column {self.name}")

    @property
    def size(self) -> int:
        return 1

    def reset_priors(self):
        self.gprior = None
        self.uprior = None

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
        self.reset_priors()
        self.process_priors()

    def get_mat(self, data: Data) -> np.ndarray:
        self.check_data(data)
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

    def copy(self) -> "Variable":
        return deepcopy(self)


@dataclass
class SplineVariable(Variable):
    spline: XSpline = field(default=None, repr=False)
    spline_specs: SplineSpecs = field(default=None, repr=False)
    linear_gpriors: List[LinearPrior] = field(default_factory=list, repr=False)
    linear_upriors: List[LinearPrior] = field(default_factory=list, repr=False)

    def __post_init__(self):
        if (self.spline is None) and (self.spline_specs is None):
            raise ValueError("At least one of spline and spline_specs is not None.")
        self.process_priors()

    def check_data(self, data: Data):
        super().check_data(data)
        if self.spline is None:
            cov = data.get_cols(self.name)
            self.spline = self.spline_specs.create_spline(cov)
            for prior in self.linear_upriors + self.linear_gpriors:
                if isinstance(prior, SplinePrior):
                    prior.attach_spline(self.spline)

    def process_priors(self):
        for prior in self.priors:
            if isinstance(prior, (SplineGaussianPrior, LinearGaussianPrior)):
                self.linear_gpriors.append(prior)
            elif isinstance(prior, (SplineUniformPrior, LinearUniformPrior)):
                self.linear_upriors.append(prior)
            elif isinstance(prior, GaussianPrior):
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
    def size(self) -> int:
        if self.spline is not None:
            n = self.spline.num_spline_bases
        else:
            n = self.spline_specs.num_spline_bases
        return n

    def reset_priors(self):
        self.gprior = None
        self.uprior = None
        self.linear_gpriors = list()
        self.linear_upriors = list()

    def get_mat(self, data: Data) -> np.ndarray:
        self.check_data(data)
        cov = data.get_cols(self.name)
        return self.spline.design_mat(cov, l_extra=True, r_extra=True)

    def get_linear_uvec(self) -> np.ndarray:
        if not self.linear_upriors:
            uvec = np.empty((2, 0))
        else:
            uvec = np.hstack([
                np.vstack([prior.lb, prior.ub])
                for prior in self.linear_upriors
            ])
        return uvec

    def get_linear_gvec(self) -> np.ndarray:
        if not self.linear_gpriors:
            gvec = np.empty((2, 0))
        else:
            gvec = np.hstack([
                np.vstack([prior.mean, prior.sd])
                for prior in self.linear_gpriors
            ])
        return gvec

    def get_linear_umat(self, data: Data = None) -> np.ndarray:
        if not self.linear_upriors:
            umat = np.empty((0, self.size))
        else:
            if self.spline is None:
                assert data is not None, "Must check data to create spline first."
                self.check_data(data)
            umat = np.vstack([
                prior.mat for prior in self.linear_upriors
            ])
        return umat

    def get_linear_gmat(self, data: Data = None) -> np.ndarray:
        if not self.linear_gpriors:
            gmat = np.empty((0, self.size))
        else:
            if self.spline is None:
                assert data is not None, "Must check data to create spline first."
                self.check_data(data)
            gmat = np.vstack([
                prior.mat for prior in self.linear_gpriors
            ])
        return gmat
