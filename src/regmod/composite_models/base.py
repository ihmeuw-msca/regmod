"""
Base Model
"""
from typing import Dict, List
from copy import deepcopy
import numpy as np
from numpy import ndarray
from pandas import DataFrame
from regmod.data import Data
from regmod.variable import Variable
from regmod.models import GaussianModel, PoissonModel
from regmod.prior import GaussianPrior
from regmod.utils import sizes_to_sclices
from regmod.composite_models import NodeModel


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
        if model_constructor_name not in dir(self):
            raise ValueError(f"Not support {self.mtype} model.")
        self.model_constructor = getattr(self, model_constructor_name)

        self.data = deepcopy(data)
        self.variables = deepcopy(variables)
        self.variable_names = [v.name for v in variables]

        self.model = self.model_constructor()

    def get_gaussian_model(self) -> GaussianModel:
        return GaussianModel(self.data,
                             param_specs={"mu": {"variables": self.variables,
                                                 "use_offset": True}})

    def get_poisson_model(self) -> PoissonModel:
        return PoissonModel(self.data,
                            param_specs={"lam": {"variables": self.variables,
                                                 "use_offset": True}})

    def get_data(self, col_label: str = None) -> DataFrame:
        df = self.model.data.df.copy()
        if col_label is not None:
            df[col_label] = self.name
        return df

    def set_data(self, df: DataFrame, col_label: str = None):
        df = self.subset_df(df, col_label, copy=True)
        if df.shape[0] == 0:
            raise ValueError("Attempt to use empty dataframe.")
        self.model.data.df = df

    def add_offset(self,
                   df: DataFrame,
                   col_value: str,
                   col_label: str = None) -> DataFrame:
        df = self.subset_df(df, col_label, copy=True)
        df[self.model.data.col_offset] = self.model.params[0].inv_link.inv_fun(
            df[col_value].values
        )
        return df

    def fit(self, **fit_options):
        self.model.fit(**fit_options)

    def predict(self,
                df: DataFrame = None,
                col_value: str = None,
                col_label: str = None):
        if df is None:
            df = self.get_data()
        df = self.subset_df(df, col_label, copy=True)
        col_value = f"{self.name}_pred" if col_value is None else col_value
        pred_data = self.model.data.copy()
        pred_data.df = df
        df[col_value] = self.model.params[0].get_param(
            self.model.opt_coefs,
            pred_data
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
        self.model = self.model_constructor()

    def get_posterior(self) -> Dict:
        if self.model.opt_coefs is None:
            raise AttributeError("Please fit the model first.")
        mean = self.model.opt_coefs
        sd = np.sqrt(np.diag(self.model.opt_vcov))
        slices = sizes_to_sclices([v.size for v in self.variables])
        return {
            v.name: GaussianPrior(mean=mean[slices[i]], sd=sd[slices[i]])
            for i, v in enumerate(self.variables)
        }

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name}, mtype={self.mtype})"
