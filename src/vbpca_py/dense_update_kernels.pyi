import numpy as np
import numpy.typing as npt

def score_update_dense_no_av(
    x_data: npt.NDArray[np.float64],
    loadings: npt.NDArray[np.float64],
    noise_var: float,
    return_covariance: bool = ...,
) -> dict[str, np.ndarray]: ...
def loadings_update_dense_no_sv(
    x_data: npt.NDArray[np.float64],
    scores: npt.NDArray[np.float64],
    prior_prec: npt.NDArray[np.float64],
    noise_var: float,
    return_covariance: bool = ...,
) -> dict[str, np.ndarray]: ...
def score_update_dense_masked_nopattern(
    x_data: npt.NDArray[np.float64],
    mask: npt.NDArray[np.float64],
    loadings: npt.NDArray[np.float64],
    loading_covariances: npt.NDArray[np.float64] | None = ...,
    noise_var: float = ...,
    return_covariances: bool = ...,
) -> dict[str, np.ndarray]: ...
def loadings_update_dense_masked_nopattern(
    x_data: npt.NDArray[np.float64],
    mask: npt.NDArray[np.float64],
    scores: npt.NDArray[np.float64],
    score_covariances: npt.NDArray[np.float64] | None = ...,
    prior_prec: npt.NDArray[np.float64] = ...,
    noise_var: float = ...,
    return_covariances: bool = ...,
) -> dict[str, np.ndarray]: ...
