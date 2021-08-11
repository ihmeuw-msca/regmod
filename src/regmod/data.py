"""
Data Module
"""
from dataclasses import dataclass, field
from typing import Dict, List, Union, Optional

import numpy as np
from numpy import ndarray
from pandas import DataFrame


@dataclass
class Data:
    """Data class used for validate and accessing data in `pd.DataFrame`.

    Parameters
    ----------
    col_obs : Optional[Union[str, List[str]]], optional
        Column name(s) for observation. Default to be `None`.
    col_covs : List[str], optional
        Column names for covariates. Default to be an empty list.
    col_weights : str, default="weights"
        Column name for weights. Default to be `'weights'`. If `col_weights` is
        not in the data frame, a column with name `col_weights` will be added to
        the data frame filled with 1.
    col_offset : str, default="offset"
        Column name for weights. Default to be `'offset'`. If `col_offset`
        is not in the data frame, a column with name `col_offset` will be added
        to the data frame filled with 0.
    df : pd.DataFrame, optional
        Data frame for the object. Default is an empty data frame.

    Attributes
    ----------
    obs
    covs
    weights
    offset
    trim_weights
    num_obs
    col_obs : Optional[Union[str, List[str]]]
        Column name for observation, can be a single string, a list of string or
        `None`. When it is `None` you cannot access property `obs`.
    col_covs : List[str]
        A list of column names for covariates.
    col_weights : str
        Column name for weights. `weights` can be used in the likelihood
        computation. Values of `weights` are required be between 0 and 1.
        `col_weights` defaultly is set to be `'weights'`. If `col_weights` is
        not in the data frame, a column with name `col_weights` will be added to
        the data frame filled with 1.
    col_offset : str
        Column name for offset. Same as `weights`, `offset` can be used in
        computing likelihood. `offset` need to be pre-transformed according to
        link function of the parameters. `col_offset` defaultly is set to be
        `'offset'`. If `col_offset` is not in the data frame, a column with name
        `col_offset` will be added to the data frame filled with 0.
    df : pd.DataFrame
        Data frame for the object. Default is an empty data frame.
    cols : List[str]
        All the relevant columns, including, `col_obs` (if not `None`),
        `col_covs`, `col_weights`, `col_offset` and `'trim_weights'`.

    Methods
    -------
    is_empty()
        Return `True` when `self.df` is empty.
    check_cols()
        Validate if all `self.cols` are in `self.df`.
    parse_df(df=None)
        Subset `df` with `self.cols`.
    fill_df()
        Automatically add columns `'intercept'`, `col_weights`, `col_offset` and
        `'trim_weights'`, if they are not present in the `self.df`.
    detach_df()
        Set `self.df` to a empty data frame.
    attach_df(df)
        Validate `df` and set `self.df=df`.
    copy(with_df=False)
        Copy `self` to a new instance of the class.
    get_cols(cols)
        Accessing columns in `self.df`.
    get_covs(col_covs)
        Accessing covariates in `self.df`.

    Notes
    -----
    * This class should be replaced by a subclass of a more general dataclass
    * `get_covs` seems very redundant should only keep `get_cols`.
    """

    col_obs: Optional[Union[str, List[str]]] = None
    col_covs: List[str] = field(default_factory=list)
    col_weights: str = "weights"
    col_offset: str = "offset"
    df: DataFrame = field(default_factory=DataFrame)

    def __post_init__(self):
        self.col_covs = list(set(self.col_covs).union({'intercept'}))
        self.cols = self.col_covs + [self.col_weights, self.col_offset, "trim_weights"]
        if self.col_obs is not None:
            if isinstance(self.col_obs, str):
                self.cols.insert(0, self.col_obs)
            else:
                self.cols = self.col_obs + self.cols

        if self.is_empty():
            self.df = DataFrame(columns=self.cols)
        else:
            self.parse_df()
            self.fill_df()
            self.check_cols()

    def is_empty(self) -> bool:
        return self.num_obs == 0

    def check_cols(self):
        for col in self.cols:
            if col not in self.df.columns:
                raise ValueError(f"Missing columnn {col}.")

    def parse_df(self, df: DataFrame = None):
        df = self.df if df is None else df
        self.df = df.loc[:, df.columns.isin(self.cols)].copy()

    def fill_df(self):
        if "intercept" not in self.df.columns:
            self.df["intercept"] = 1.0
        if self.col_weights not in self.df.columns:
            self.df[self.col_weights] = 1.0
        if self.col_offset not in self.df.columns:
            self.df[self.col_offset] = 0.0
        if self.col_obs is not None:
            cols = self.col_obs if isinstance(self.col_obs, list) else [self.col_obs]
            for col in cols:
                if col not in self.df.columns:
                    self.df[col] = np.nan
        self.df["trim_weights"] = 1.0

    def detach_df(self):
        self.df = DataFrame(columns=self.cols)

    def attach_df(self, df: DataFrame):
        self.parse_df(df)
        self.fill_df()
        self.check_cols()

    def copy(self, with_df=False) -> "Data":
        df = self.df.copy() if with_df else DataFrame(columns=self.cols)
        return Data(self.col_obs,
                    self.col_covs,
                    self.col_weights,
                    self.col_offset,
                    df)

    def get_cols(self, cols: Union[List[str], str]) -> ndarray:
        return self.df[cols].to_numpy()

    @property
    def num_obs(self) -> int:
        return self.df.shape[0]

    @property
    def obs(self) -> ndarray:
        if self.col_obs is None:
            raise ValueError("This data object does not contain observations.")
        return self.get_cols(self.col_obs)

    @property
    def covs(self) -> Dict[str, ndarray]:
        return self.df[self.col_covs].to_dict(orient="list")

    @property
    def weights(self) -> ndarray:
        return self.get_cols(self.col_weights)

    @property
    def offset(self) -> ndarray:
        return self.get_cols(self.col_offset)

    @property
    def trim_weights(self) -> ndarray:
        return self.get_cols("trim_weights")

    @trim_weights.setter
    def trim_weights(self, weights: Union[float, ndarray]):
        if np.any(weights < 0.0) or np.any(weights > 1.0):
            raise ValueError("trim_weights has to be between 0 and 1.")
        self.df["trim_weights"] = weights

    def get_covs(self, col_covs: Union[List[str], str]) -> ndarray:
        if not isinstance(col_covs, list):
            col_covs = [col_covs]
        return self.get_cols(col_covs)

    def __repr__(self):
        return f"Data(num_obs={self.num_obs})"
