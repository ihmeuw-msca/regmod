"""
Data Module
"""
from dataclasses import dataclass, field
from typing import Dict, List, Union

import numpy as np
from numpy import ndarray
from pandas import DataFrame


@dataclass
class Data:
    col_obs: Union[str, List[str]] = None
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
