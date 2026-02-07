import numpy as np
import numpy.typing as npt

def subtract_mu_from_sparse(
    data: npt.NDArray[np.float64],
    indices: npt.NDArray[np.int32],
    indptr: npt.NDArray[np.int32],
    shape: tuple[int, int],
    mu: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]: ...
