import numpy as np
import numpy.typing as npt

def errpca_pt(
    x_data: npt.NDArray[np.float64],
    x_indices: npt.NDArray[np.int32],
    x_indptr: npt.NDArray[np.int32],
    loadings: npt.NDArray[np.float64],
    scores: npt.NDArray[np.float64],
    num_cpu: int = ...,
) -> dict[str, np.ndarray]: ...
