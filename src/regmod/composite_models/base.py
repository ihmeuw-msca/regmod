"""
Base Model
"""
from copy import deepcopy
from typing import Callable, Dict, List

import numpy as np
from numpy import ndarray
from pandas import DataFrame
from regmod.composite_models.node import NodeModel
from regmod.data import Data
from regmod.function import fun_dict
from regmod.models import GaussianModel, PoissonModel
from regmod.prior import GaussianPrior
from regmod.utils import sizes_to_sclices
from regmod.variable import Variable


class BaseModel(NodeModel):
    """
    Base Model, a simple wrapper around the stats model.
    """

    def __init__(self,
                 name: str,
                 data: Data,
                 variables: List[Variable],
                 mtype: str = "gaussian"):

        super().__init__(name)

        self.mtype = mtype
        model_constructor_name = f"get_{self.mtype}_model"
        link_fun_constructor_name = f"get_{self.mtype}_link_fun"
        if not (model_constructor_name in dir(self) and
                link_fun_constructor_name in dir(self)):
            raise ValueError(f"Not support {self.mtype} model.")
        self.get_model = getattr(self, model_constructor_name)
        self.link_fun = getattr(self, link_fun_constructor_name)()

        self.data = deepcopy(data)
        self.variables = deepcopy(variables)
        self.variable_names = [v.name for v in variables]

        self.model = None

    def get_gaussian_model(self) -> GaussianModel:
        return GaussianModel(self.data,
                             param_specs={"mu": {"variables": self.variables,
                                                 "use_offset": True}})

    @staticmethod
    def get_gaussian_link_fun() -> Callable:
        return fun_dict[
            GaussianModel.default_param_specs["mu"]["inv_link"]
        ].inv_fun

    @staticmethod
    def get_poisson_link_fun() -> Callable:
        return fun_dict[
            PoissonModel.default_param_specs["lam"]["inv_link"]
        ].inv_fun

    def get_poisson_model(self) -> PoissonModel:
        return PoissonModel(self.data,
                            param_specs={"lam": {"variables": self.variables,
                                                 "use_offset": True}})

    def get_data(self, col_label: str = None) -> DataFrame:
        df = self.data.df.copy()
        if col_label is not None:
            df[col_label] = self.name
        return df

    def set_data(self,
                 df: DataFrame,
                 col_value: str = None,
                 col_label: str = None):
        df = self.subset_df(df, col_label, copy=True)
        df = self.add_offset(df, col_value, copy=False)
        if df.shape[0] == 0:
            raise ValueError("Attempt to use empty dataframe.")
        self.data.attach_df(df)

    def add_offset(self,
                   df: DataFrame,
                   col_value: str = None,
                   copy: bool = False) -> DataFrame:
        if col_value is not None and col_value in df.columns:
            df[self.data.col_offset] = self.link_fun(df[col_value].values)
        if copy:
            df = df.copy()
        return df

    def fit(self, **fit_options):
        if self.model is None:
            self.model = self.get_model()
        self.model.fit(**fit_options)

    def predict(self,
                df: DataFrame = None,
                col_value: str = None,
                col_label: str = None):
        if df is None:
            df = self.get_data()
        col_value = self.get_col_value(col_value)

        df = self.subset_df(df, col_label, copy=True)
        df = self.add_offset(df, col_value, copy=False)

        pred_data = self.model.data.copy()
        pred_data.attach_df(df)

        df[col_value] = self.model.params[0].get_param(
            self.model.opt_coefs, pred_data
        )
        return df

    def set_prior(self,
                  priors: Dict[str, List],
                  masks: Dict[str, ndarray] = None):
        for name, prior in priors.items():
            index = self.variable_names.index(name)
            if masks is not None and name in masks:
                prior.sd *= masks[name]
            self.variables[index].add_priors(prior)
        self.model = self.get_model()

    def get_posterior(self) -> Dict:
        if self.model.opt_coefs is None:
            raise AttributeError("Please fit the model first.")
        mean = self.model.opt_coefs
        # use minimum standard deviation of the posterior distribution
        sd = np.maximum(0.1, np.sqrt(np.diag(self.model.opt_vcov)))
        slices = sizes_to_sclices([v.size for v in self.variables])
        return {
            v.name: GaussianPrior(mean=mean[slices[i]], sd=sd[slices[i]])
            for i, v in enumerate(self.variables)
        }

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name}, mtype={self.mtype})"
