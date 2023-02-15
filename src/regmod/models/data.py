from typing import Optional

import jax.numpy as jnp
import numpy as np
from jax.numpy import DeviceArray
from msca.linalg.matrix import Matrix, asmatrix
from numpy.typing import NDArray
from pandas import DataFrame
from scipy.linalg import block_diag
from scipy.sparse import csc_matrix

from regmod.parameter import Parameter

NumpyData = dict[str, NDArray | tuple[NDArray, ...]]
JaxData = dict[str, DeviceArray | tuple[DeviceArray, ...]]
MSCAData = dict[str, Matrix | tuple[Matrix, ...]]


def parse_to_numpy(
    df: DataFrame,
    y: str,
    params: tuple[Parameter, ...],
    weights: Optional[str] = None,
    for_fit: bool = True,
) -> NumpyData:
    for param in params:
        param.check_data(df)

    data = {
        "mat": tuple([param.get_mat(df) for param in params]),
        "offset": tuple([param.get_offset(df) for param in params]),
        "uvec": np.hstack([param.get_uvec() for param in params]),
        "gvec": np.hstack([param.get_gvec() for param in params]),
        "linear_uvec": np.hstack([param.get_linear_uvec() for param in params]),
        "linear_gvec": np.hstack([param.get_linear_gvec() for param in params]),
        "linear_umat": block_diag(*[param.get_linear_umat() for param in params]),
        "linear_gmat": block_diag(*[param.get_linear_gmat() for param in params]),
    }

    if for_fit:
        data["y"] = df[y].to_numpy()
        data["weights"] = (
            np.ones(len(df)) if weights is None else df[weights].to_numpy()
        )
        data["trim_weights"] = np.ones(len(df))

    return data


def parse_to_jax(
    df: DataFrame,
    y: str,
    params: tuple[Parameter, ...],
    weights: Optional[str] = None,
    for_fit: bool = True,
) -> JaxData:
    data = parse_to_numpy(df, y, params, weights, for_fit=for_fit)
    for key, value in data.items():
        if isinstance(value, tuple):
            new_value = tuple([jnp.asarray(x) for x in value])
        else:
            new_value = jnp.asarray(value)
        data[key] = new_value
    # TODO: jax model specific combine weight and trim_weights
    if for_fit:
        data["weights"] = data["trim_weights"] * data["weights"]
    return data


def parse_to_msca(
    df: DataFrame,
    y: str,
    params: tuple[Parameter, ...],
    weights: Optional[str] = None,
    for_fit: bool = True,
) -> MSCAData:
    data = parse_to_numpy(df, y, params, weights, for_fit=for_fit)
    mat, uvec, linear_umat, linear_uvec = (
        data["mat"],
        data["uvec"],
        data["linear_umat"],
        data["linear_uvec"],
    )
    # design matrix
    new_mat = []
    for m in mat:
        issparse = m.size == 0 or ((m == 0).sum() / m.size) > 0.95
        if issparse:
            m = csc_matrix(m).astype(np.float64)
        m = asmatrix(m)
        new_mat.append(m)
    new_mat = tuple(new_mat)

    # constraints
    cmat = np.vstack([np.identity(sum(m.shape[1] for m in mat)), linear_umat])
    cvec = np.hstack([uvec, linear_uvec])

    index = ~np.isclose(cmat, 0.0).all(axis=1)
    cmat = cmat[index]
    cvec = cvec[:, index]

    if cmat.size > 0:
        scale = np.abs(cmat).max(axis=1)
        cmat = cmat / scale[:, np.newaxis]
        cvec = cvec / scale

    cmat = np.vstack([-cmat[~np.isneginf(cvec[0])], cmat[~np.isposinf(cvec[1])]])
    cvec = np.hstack([-cvec[0][~np.isneginf(cvec[0])], cvec[1][~np.isposinf(cvec[1])]])
    if issparse:
        cmat = csc_matrix(cmat).astype(np.float64)
    cmat = asmatrix(cmat)

    data["mat"], data["cmat"], data["cvec"] = new_mat, cmat, cvec

    return data
