"""
Variable module
"""
from collections.abc import Iterable
from copy import deepcopy
from dataclasses import dataclass, field
from typing import List, Optional, Union

import numpy as np
import pandas as pd
from xspline import XSpline

from regmod.prior import (GaussianPrior, LinearGaussianPrior, LinearPrior,
                          LinearUniformPrior, Prior, SplineGaussianPrior,
                          SplinePrior, SplineUniformPrior, UniformPrior)
from regmod.utils import SplineSpecs


@dataclass
class Variable:
    """Variable class is in charge of storing information of variable including
    name and priors, and accessing data in the data frame. Name correspondes to
    column name in the data frame and priors are used to compute the likelihood.

    Parameters
    ----------
    name : str
        Name of the variable corresponding to the column name in the data frame.
    priors : List[Prior], optional
        A list of priors for the variable. Default is an empty list.

    Attributes
    ----------
    size
    name : str
        Name of the variable corresponding to the column name in the data frame.
    priors : List[Prior]
        A list of priors for the variable.
    gprior : GaussianPrior
        Direct Gaussian prior in `priors`.
    uprior : UniformPrior
        Direct Uniform prior in `priors`.

    Methods
    -------
    process_priors()
        Check the prior type and extract `gprior` and `uprior`.
    check_data(data)
        Check if the data contains the column name `name`.
    reset_prior()
        Reset direct priors.
    add_priors(priors)
        Add priors.
    rm_priors(indices)
        Remove priors.
    get_mat(data)
        Get design matrix.
    get_gvec()
        Get direct Gaussian prior vector.
    get_uvec()
        Get direct Uniform prior vector.
    copy()
        Copy current instance.

    Notes
    -----
    In the future, this class will be combined with the SplineVariable.
    """

    name: str
    priors: List[Prior] = field(default_factory=list, repr=False)
    gprior: GaussianPrior = field(default=None, init=False, repr=False)
    uprior: UniformPrior = field(default=None, init=False, repr=False)

    def __post_init__(self):
        self.process_priors()

    def process_priors(self):
        """Check the prior type and extract `gprior` and `uprior`.

        Raises
        ------
        AssertionError
            Raised if direct Gaussian prior size not match.
        AssertionError
            Raised if direct Uniform prior size not match.
        ValueError
            Raised when any prior in the list is not an instance of Prior.
        """
        for prior in self.priors:
            if isinstance(prior, LinearPrior):
                continue
            if isinstance(prior, GaussianPrior):
                if self.gprior is not None:
                    self.priors.remove(self.gprior)
                self.gprior = prior
                assert self.gprior.size == self.size, \
                    "Gaussian prior size not match."
            elif isinstance(prior, UniformPrior):
                if self.uprior is not None:
                    self.priors.remove(self.uprior)
                self.uprior = prior
                assert self.uprior.size == self.size, \
                    "Uniform prior size not match."
            else:
                raise ValueError("Unknown prior type.")

    def check_data(self, df: pd.DataFrame):
        """Check if the data contains the column name `name`.

        Parameters
        ----------
        df
            Data frame that contains the covariate.

        Raises
        ------
        ValueError
            Raised if data doesn't contain column name `self.name`.

        """
        if self.name not in df.columns and self.name != "intercept":
            raise ValueError(f"Data do not contain column {self.name}")

    @property
    def size(self) -> int:
        """Size of the variable."""
        return 1

    def reset_priors(self) -> None:
        """Reset direct priors."""
        self.gprior = None
        self.uprior = None

    def add_priors(self, priors: Union[Prior, List[Prior]]) -> None:
        """Add priors.

        Parameters
        ----------
        priors : Union[Prior, List[Prior]]
            Priors to be added.
        """
        if not isinstance(priors, list):
            priors = [priors]
        self.priors.extend(priors)
        self.process_priors()

    def rm_priors(self, indices: Union[int, List[int], List[bool]]) -> None:
        """Remove priors.

        Parameters
        ----------
        indices : Union[int, List[int], List[bool]]
            Indicies of the priors that need to be removed. Indicies come in the
            forms of integer, list of integers or list of booleans. When it is
            integer or list of integers, it requires the integer is within the
            bounds `[0, len(self.priors))`. When it is booleans, it requires the
            list have the same length with `self.priors`.

        Raises
        ------
        AssertionError
            Raised when `indices` has the wrong type.
        AssertionError
            Raised when `indices` is a list with mixed types.
        AssertionError
            Raised when `indices` is list of booleans but with different length
            compare to `self.priors`.
        """
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

    def get_mat(self, df: pd.DataFrame) -> np.ndarray:
        """Get design matrix.

        Parameters
        ----------
        data
            Data frame contains the covariate.

        Returns
        -------
        np.ndarray
            Design matrix.

        """
        self.check_data(df)
        if self.name == "intercept":
            return np.ones((df.shape[0], 1))
        return df[[self.name]].to_numpy()

    def get_gvec(self) -> np.ndarray:
        """Get direct Gaussian prior vector.

        Returns
        -------
        np.ndarray
            Direct Gaussian prior vector.
        """
        if self.gprior is None:
            gvec = np.repeat([[0.0], [np.inf]], self.size, axis=1)
        else:
            gvec = np.vstack([self.gprior.mean, self.gprior.sd])
        return gvec

    def get_uvec(self) -> np.ndarray:
        """Get direct Uniform prior vector.

        Returns
        -------
        np.ndarray
            Direct Uniform prior vector.
        """
        if self.uprior is None:
            uvec = np.repeat([[-np.inf], [np.inf]], self.size, axis=1)
        else:
            uvec = np.vstack([self.uprior.lb, self.uprior.ub])
        return uvec

    def copy(self) -> "Variable":
        """Copy current instance.

        Returns
        -------
        Variable
            Current instance.
        """
        return deepcopy(self)


@dataclass
class SplineVariable(Variable):
    """Spline variable that store information of variable with splines.

    Parameters
    ----------
    spline : XSpline, optional
        Spline object that in charge of creating design matrix. Default to be
        `None`. `spline` and `spline_specs` cannot be `None` at the same time.
    spline_specs : SplineSpecs, optional
        Spline settings used to create spline object. Recommend to use only when
        use `knots_type={'rel_domain', 'rel_freq'}. Default to be `None`.
    linear_gpriors : List[LinearPrior], optional
        A list of linear Gaussian priors usually for shape priors of the spline.
        Default to be an empty list.
    linear_upriors : List[LinearPrior], optional
        A list of linear Uniform priors usually for shape priors of the spline.
        spline. Default to be an empty list.

    Attributes
    ----------
    spline : XSpline
        Spline object that in charge of creating design matrix.
    spline_specs : SplineSpecs
        Spline settings used to create spline object.
    linear_gpriors : List[LinearPrior]
        A list of linear Gaussian priors usually for shape priors of the spline.
    linear_upriors : List[LinearPrior]
        A list of linear Uniform priors usually for shape priors of the spline.

    Methods
    -------
    check_data(data)
        Check if the data contains the column name `name`. And create the spline
        object, if only `spline_specs` is provided.
    process_priors()
        Check the prior type and extract `gprior`, `uprior`, `linear_gpriors`
        and `linear_upriors`.
    reset_priors()
        Reset direct and linear priors.
    get_mat(data)
        Get design matrix.
    get_linear_gvec()
        Get linear Gaussian prior vector.
    get_linear_uvec()
        Get linear Uniform prior vector.
    get_linear_gmat(data)
        Get linear Gaussian prior design matrix.
    get_linear_umat(data)
        Get linear Uniform prior design matrix.
    """

    spline: XSpline = field(default=None, repr=False)
    spline_specs: SplineSpecs = field(default=None, repr=False)
    linear_gpriors: List[LinearPrior] = field(default_factory=list, repr=False)
    linear_upriors: List[LinearPrior] = field(default_factory=list, repr=False)

    def __post_init__(self):
        if (self.spline is None) and (self.spline_specs is None):
            raise ValueError("At least one of spline and spline_specs is not None.")
        self.process_priors()

    def check_data(self, df: pd.DataFrame):
        """Check if the data contains the column name `name`. And create the
        spline object, if only `spline_specs` is provided.

        Parameters
        ----------
        df
            Data frame that contains the covariate.

        """
        super().check_data(df)
        if self.spline is None:
            cov = df[self.name].to_numpy()
            self.spline = self.spline_specs.create_spline(cov)
            for prior in self.linear_upriors + self.linear_gpriors:
                if isinstance(prior, SplinePrior):
                    prior.attach_spline(self.spline)

    def process_priors(self):
        """Check the prior type and extract `gprior`, `uprior`, `linear_gpriors`
        and `linear_upriors`.

        Raises
        ------
        AssertionError
            Raised if direct Gaussian prior size not match.
        AssertionError
            Raised if direct Uniform prior size not match.
        ValueError
            Raised when any prior in the list is not an instance of Prior.
        """
        for prior in self.priors:
            if isinstance(prior, (SplineGaussianPrior, LinearGaussianPrior)):
                self.linear_gpriors.append(prior)
            elif isinstance(prior, (SplineUniformPrior, LinearUniformPrior)):
                self.linear_upriors.append(prior)
            elif isinstance(prior, GaussianPrior):
                if self.gprior is not None:
                    self.priors.remove(self.gprior)
                self.gprior = prior
                assert self.gprior.size == self.size, \
                    "Gaussian prior size not match."
            elif isinstance(prior, UniformPrior):
                if self.uprior is not None:
                    self.priors.remove(self.uprior)
                self.uprior = prior
                assert self.uprior.size == self.size, \
                    "Uniform prior size not match."
            else:
                raise ValueError("Unknown prior type.")

    @property
    def size(self) -> int:
        """Size of the variable."""
        if self.spline is not None:
            n = self.spline.num_spline_bases
        else:
            n = self.spline_specs.num_spline_bases
        return n

    def reset_priors(self):
        """Reset direct and linear priors."""
        self.gprior = None
        self.uprior = None
        self.linear_gpriors = list()
        self.linear_upriors = list()

    def get_mat(self, df: pd.DataFrame) -> np.ndarray:
        """Get design matrix.

        Parameters
        ----------
        df
            Data frame that contains the covariate.

        Returns
        -------
        np.ndarray
            Design matrix.

        """
        self.check_data(df)
        cov = df[self.name].to_numpy()
        return self.spline.design_mat(cov, l_extra=True, r_extra=True)

    def get_linear_uvec(self) -> np.ndarray:
        """Get linear Uniform prior vector.

        Returns
        -------
        np.ndarray
            Linear uniform prior vector.
        """
        if not self.linear_upriors:
            uvec = np.empty((2, 0))
        else:
            uvec = np.hstack([
                np.vstack([prior.lb, prior.ub])
                for prior in self.linear_upriors
            ])
        return uvec

    def get_linear_gvec(self) -> np.ndarray:
        """Get linear Gaussian prior vector.

        Returns
        -------
        np.ndarray
            Linear Gaussian prior vector.
        """
        if not self.linear_gpriors:
            gvec = np.empty((2, 0))
        else:
            gvec = np.hstack([
                np.vstack([prior.mean, prior.sd])
                for prior in self.linear_gpriors
            ])
        return gvec

    def get_linear_umat(self, df: Optional[pd.DataFrame] = None) -> np.ndarray:
        """Get linear Uniform prior design matrix.

        Parameters
        ----------
        data : Data, optional
            Data object that provides the covariate. Default to be `None`.

        Raises
        ------
        AssertionError
            Raised when both `data` and `self.spline` are `None`.

        Returns
        -------
        np.ndarray:
            Linear Uniform prior design matrix.
        """
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

    def get_linear_gmat(self, df: Optional[pd.DataFrame] = None) -> np.ndarray:
        """Get linear Gaussian prior design matrix.

        Parameters
        ----------
        data : Data
            Data object that provides the covariate.

        Raises
        ------
        AssertionError
            Raised when both `data` and `self.spline` are `None`.

        Returns
        -------
        np.ndarray:
            Linear Gaussian prior design matrix.
        """
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
