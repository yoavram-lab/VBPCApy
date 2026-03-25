import numpy as np
import numpy.typing as npt

def congruence_transform_stack(
    cov_stack: npt.NDArray[np.float64],
    left: npt.NDArray[np.float64],
    right: npt.NDArray[np.float64],
    num_cpu: int = ...,
) -> npt.NDArray[np.float64]: ...
def weighted_cov_eigh_psd(
    base: npt.NDArray[np.float64],
    cov_stack: npt.NDArray[np.float64],
    weights: npt.NDArray[np.float64],
    normalizer: float,
) -> dict[str, npt.NDArray[np.float64]]: ...
