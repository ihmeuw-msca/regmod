"""
Prior module
"""
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any, List

import numpy as np
from xspline import XSpline

from regmod.utils import default_vec_factory


@dataclass
class Prior:
    """Prior information for the variables, it is used to construct the
    liklihood and solve the optimization problem.

    Parameters
    ----------
    size : Optional[int], optional
        Size of variable. Default is `None`. When it is `None`, size is inferred
        from the vector information of the prior.

    Attributes
    ----------
    size : int
        Size of variable

    Methods
    -------
    process_size(vecs)
        Infer and validate size from given vector information.
    """

    size: int = None

    def process_size(self, vecs: List[Any]):
        """Infer and validate size from given vector information.

        Parameters
        ----------
        vecs : List[Any]
            Vector infromation of the prior. For Gaussian prior it will be mean
            and standard deviation. For Uniform prior it will be lower and upper
            bounds.

        Raises
        ------
        ValueError
            Raised when size is not positive or integer.
        """
        if self.size is None:
            sizes = [len(vec) for vec in vecs if isinstance(vec, Iterable)]
            sizes.append(1)
            self.size = max(sizes)

        if self.size <= 0 or not isinstance(self.size, int):
            raise ValueError("Size of the prior must be a positive integer.")


@dataclass
class GaussianPrior(Prior):
    """Gaussian prior information.

    Parameters
    ----------
    size : Optional[int], optional
        Size of variable. Default is `None`. When it is `None`, size is inferred
        from the vector information of the prior.
    mean : Union[float, np.ndarray], default=0
        Mean of the Gaussian prior. Default is 0. If it is a scalar, it will be
        extended to an array with `self.size`.
    sd : Union[float, np.ndarray], default=np.inf
        Standard deviation of the Gaussian prior. Default is `np.inf`. If it is
        a scalar, it will be extended to an array with `self.size`.

    Attributes
    ----------
    size : int
        Size of the variable.
    mean : ndarray
        Mean of the Gaussian prior.
    sd : ndarray
        Standard deviation of the Gaussian prior.

    Raises
    ------
    ValueError
        Raised when size of mean vector doesn't match.
    ValueError
        Raised when size of the standard deviation vector doesn't match.
    ValueError
        Raised when any value in standard deviation vector is non-positive.
    """

    mean: np.ndarray = field(default=0.0, repr=False)
    sd: np.ndarray = field(default=np.inf, repr=False)

    def __post_init__(self):
        self.process_size([self.mean, self.sd])
        if np.isscalar(self.mean):
            self.mean = np.repeat(self.mean, self.size)
        if np.isscalar(self.sd):
            self.sd = np.repeat(self.sd, self.size)
        self.mean = np.asarray(self.mean)
        self.sd = np.asarray(self.sd)
        if self.mean.size != self.size:
            raise ValueError("Mean vector size not matching.")
        if self.sd.size != self.size:
            raise ValueError("Standard deviation vector size not matching.")
        if any(self.sd <= 0.0):
            raise ValueError("Standard deviation must be all positive.")


@dataclass
class UniformPrior(Prior):
    """Uniform prior information.

    Parameters
    ----------
    size : Optional[int], optional
        Size of variable. Default is `None`. When it is `None`, size is inferred
        from the vector information of the prior.
    lb : Union[float, np.ndarray], default=-np.inf
        Lower bound of Uniform prior. Default is `-np.inf`. If it is a scalar,
        it will be extended to an array with `self.size`.
    ub : Union[float, np.ndarray], default=np.inf
        Upper bound of the Uniform prior. Default is `np.inf`. If it is a
        scalar,it will be extended to an array with `self.size`.

    Attributes
    ----------
    size : int
        Size of the variable.
    lb : ndarray
        Lower bound of Uniform prior.
    ub : ndarray
        Upper bound of Uniform prior.

    Raises
    ------
    ValueError
        Raised when size of the lower bound vector doesn't match.
    ValueError
        Raised when size of the upper bound vector doesn't match.
    ValueError
        Raised if lower bound is greater than upper bound.
    """

    lb: np.ndarray = field(default=-np.inf, repr=False)
    ub: np.ndarray = field(default=np.inf, repr=False)

    def __post_init__(self):
        self.process_size([self.lb, self.ub])
        if np.isscalar(self.lb):
            self.lb = np.repeat(self.lb, self.size)
        if np.isscalar(self.ub):
            self.ub = np.repeat(self.ub, self.size)
        self.lb = np.asarray(self.lb)
        self.ub = np.asarray(self.ub)
        if self.lb.size != self.size:
            raise ValueError("Lower bound vector size not matching.")
        if self.ub.size != self.size:
            raise ValueError("Upper bound vector size not matching.")
        if any(self.lb > self.ub):
            ValueError("Lower bounds must be less or equal than upper bounds.")


@dataclass
class LinearPrior:
    mat: np.ndarray = field(default_factory=lambda: np.empty(shape=(0, 1)),
                            repr=False)
    size: int = None

    def __post_init__(self):
        if self.size is None:
            self.size = self.mat.shape[0]
        else:
            assert self.size == self.mat.shape[0], "`mat` and `size` not match."

    def is_empty(self) -> bool:
        return self.mat.size == 0.0


@dataclass
class SplinePrior(LinearPrior):
    size: int = 100
    order: int = 0
    domain_lb: float = field(default=0.0, repr=False)
    domain_ub: float = field(default=1.0, repr=False)
    domain_type: str = field(default="rel", repr=False)

    def __post_init__(self):
        assert self.domain_lb <= self.domain_ub, \
            "Domain lower bound must be less or equal than upper bound."
        assert self.domain_type in ["rel", "abs"], \
            "Domain type must be 'rel' or 'abs'."
        if self.domain_type == "rel":
            assert self.domain_lb >= 0.0 and self.domain_ub <= 1.0, \
                "Using relative domain, bounds must be numbers between 0 and 1."

    def attach_spline(self, spline: XSpline) -> np.ndarray:
        knots_lb = spline.knots[0]
        knots_ub = spline.knots[-1]
        if self.domain_type == "rel":
            points_lb = knots_lb + (knots_ub - knots_lb)*self.domain_lb
            points_ub = knots_lb + (knots_ub - knots_lb)*self.domain_ub
        else:
            points_lb = self.domain_lb
            points_ub = self.domain_ub
        points = np.linspace(points_lb, points_ub, self.size)
        self.mat = spline.design_dmat(points, order=self.order,
                                      l_extra=True, r_extra=True)
        super().__post_init__()


@dataclass
class LinearGaussianPrior(LinearPrior, GaussianPrior):
    def __post_init__(self):
        LinearPrior.__post_init__(self)
        GaussianPrior.__post_init__(self)


@dataclass
class LinearUniformPrior(LinearPrior, UniformPrior):
    def __post_init__(self):
        LinearPrior.__post_init__(self)
        UniformPrior.__post_init__(self)


@dataclass
class SplineGaussianPrior(SplinePrior, GaussianPrior):
    def __post_init__(self):
        SplinePrior.__post_init__(self)
        GaussianPrior.__post_init__(self)


@dataclass
class SplineUniformPrior(SplinePrior, UniformPrior):
    def __post_init__(self):
        SplinePrior.__post_init__(self)
        UniformPrior.__post_init__(self)
