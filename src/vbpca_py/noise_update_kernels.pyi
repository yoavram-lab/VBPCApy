import numpy as np
import numpy.typing as npt

def noise_sxv_sum(
    ix: npt.NDArray[np.int32],
    jx: npt.NDArray[np.int32],
    loadings: npt.NDArray[np.float64],
    scores: npt.NDArray[np.float64],
    sv_by_col: npt.NDArray[np.float64],
    loading_covariances: npt.NDArray[np.float64] | None = ...,
    num_cpu: int = ...,
) -> float: ...
