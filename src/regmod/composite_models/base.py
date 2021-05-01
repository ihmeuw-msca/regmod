"""
Base Model
"""
from typing import Dict, List
import numpy as np
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

        super().__init__(name, data.df)

        self.mtype = mtype
        model_constructor_name = f"get_{self.mtype}_model"
        if model_constructor_name not in dir(self):
            raise ValueError(f"Not support {self.mtype} model.")
        self.model_constructor = getattr(self, model_constructor_name)

        self.data = data
        self.variables = variables
        self.variable_names = [v.name for v in variables]

        self.model = self.model_constructor()

    def get_gaussian_model(self) -> GaussianModel:
        return GaussianModel(self.data,
                             param_specs={"mu": {"variables": self.variables}})

    def get_poisson_model(self) -> PoissonModel:
        return PoissonModel(self.data,
                            param_specs={"lam": {"variables": self.variables}})

    def set_data(self, df: DataFrame) -> DataFrame:
        self.df = df
        self.model.data.df = df

    def set_offset(self, df: DataFrame, col: str):
        df[self.model.data.col_offset] = self.model.params[0].inv_link.inv_fun(
            df[col].values
        )
        return df

    def fit(self, **fit_options):
        self.model.fit(**fit_options)

    def predict(self, df: DataFrame = None):
        if df is None:
            df = self.df.copy()
        pred_data = self.model.data.copy()
        pred_data.df = df

        pred = self.model.params[0].get_param(self.model.opt_coefs, pred_data)
        df[f"{self.name}_pred"] = pred
        return df

    def set_prior(self, priors: Dict[str, List]):
        for name, prior in priors.items():
            index = self.variable_names.index(name)
            self.variables[index].add_priors(prior)
        self.model = self.model_constructor()

    def get_posterior(self) -> Dict:
        mean = self.model.opt_coefs
        sd = np.sqrt(np.diag(self.model.opt_vcov))
        slices = sizes_to_sclices([v.size for v in self.variables])
        return {
            name: GaussianPrior(mean=mean[slices[i]], sd=sd[slices[i]])
            for i, name in enumerate(self.variables)
        }
