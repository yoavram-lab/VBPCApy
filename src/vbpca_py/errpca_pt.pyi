from typing import TypedDict

import numpy as np
import numpy.typing as npt

class SparseResidualResult(TypedDict):
    data: npt.NDArray[np.float64]
    indices: npt.NDArray[np.int32]
    indptr: npt.NDArray[np.int32]
    shape: tuple[int, int]

def errpca_pt(
    x_data: npt.NDArray[np.float64],
    x_indices: npt.NDArray[np.int32],
    x_indptr: npt.NDArray[np.int32],
    loadings: npt.NDArray[np.float64],
    scores: npt.NDArray[np.float64],
    num_cpu: int = ...,
) -> SparseResidualResult: ...
def effective_thread_count(requested: int, n_items: int, /) -> int: ...
