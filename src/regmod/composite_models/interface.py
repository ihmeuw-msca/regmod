"""
Node Model
"""
from typing import Dict

from pandas import DataFrame


class ModelInterface:
    """
    Abstract class that encode the behavior of the model interface
    """

    def __init__(self, name: str):
        self.name = name

    def set_data(self,
                 df: DataFrame,
                 col_value: str = None,
                 col_label: str = None):
        raise NotImplementedError

    def get_data(self, col_label: str = None) -> DataFrame:
        raise NotImplementedError

    def fit(self, **fit_options):
        raise NotImplementedError

    def predict(self,
                df: DataFrame = None,
                col_value: str = None,
                col_label: str = None) -> DataFrame:
        raise NotImplementedError

    def set_prior(self, priors: Dict, masks: Dict = None):
        raise NotImplementedError

    def get_posterior(self) -> Dict:
        raise NotImplementedError

    def subset_df(self,
                  df: DataFrame,
                  col_label: str = None,
                  copy: bool = False) -> DataFrame:
        if col_label is not None and col_label in df.columns:
            df = df[df[col_label] == self.name]
        if copy:
            df = df.copy()
        return df

    def get_col_value(self, col_value: str = None) -> str:
        if col_value is not None:
            return col_value
        return f"{self.name}_pred"

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name})"
