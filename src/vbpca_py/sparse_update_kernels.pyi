import numpy as np
import numpy.typing as npt

def score_update_sparse_nopattern(
    x_data: npt.NDArray[np.float64],
    x_indices: npt.NDArray[np.int32],
    x_indptr: npt.NDArray[np.int32],
    loadings: npt.NDArray[np.float64],
    loading_covariances: npt.NDArray[np.float64] | None = ...,
    noise_var: float = ...,
    return_covariances: bool = ...,
    num_cpu: int = ...,
) -> dict[str, np.ndarray]: ...
def loadings_update_sparse_nopattern(
    x_data: npt.NDArray[np.float64],
    x_indices: npt.NDArray[np.int32],
    x_indptr: npt.NDArray[np.int32],
    scores: npt.NDArray[np.float64],
    score_covariances: npt.NDArray[np.float64] | None = ...,
    prior_prec: npt.NDArray[np.float64] = ...,
    noise_var: float = ...,
    return_covariances: bool = ...,
    num_cpu: int = ...,
) -> dict[str, np.ndarray]: ...
