"""
Utility classes and functions
"""
from typing import List, Any
from dataclasses import dataclass
import numpy as np
from xspline import XSpline


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


@dataclass
class SplineSpecs:
    knots: np.ndarray
    degree: int = 3
    l_linear: bool = False
    r_linear: bool = False
    knots_type: str = "abs"

    def __post_init__(self):
        assert self.knots_type in ["abs", "rel_domain", "rel_freq"], \
            "Knots type must be one of 'abs', 'rel_domain' or 'rel_freq'."

    @property
    def num_spline_bases(self) -> int:
        inner_knots = self.knots[int(self.l_linear):
                                 len(self.knots) - int(self.r_linear)]
        return len(inner_knots) - 1 + self.degree

    def create_spline(self, vec: np.ndarray = None) -> XSpline:
        if self.knots_type == "abs":
            knots = self.knots
        else:
            assert vec is not None, \
                "Using relative knots, must provide a vector to finalize knots."
            if self.knots_type == "rel_domain":
                lb = np.min(vec)
                ub = np.max(vec)
                knots = lb + self.knots*(ub - lb)
            else:
                knots = np.quantile(vec, self.knots)

        return XSpline(knots, self.degree,
                       l_linear=self.l_linear,
                       r_linear=self.r_linear)


def sizes_to_sclices(sizes: np.array) -> List[slice]:
    ends = np.cumsum(sizes)
    starts = np.insert(ends, 0, 0)[:-1]
    return [slice(*pair) for pair in zip(starts, ends)]
