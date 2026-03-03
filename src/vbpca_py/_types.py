from __future__ import annotations

from typing import TypeAlias

import numpy as np
import scipy.sparse as sp

Array: TypeAlias = np.ndarray
Dense: TypeAlias = np.ndarray
Sparse: TypeAlias = sp.csr_matrix
Matrix: TypeAlias = np.ndarray | sp.csr_matrix
ArrayLike: TypeAlias = Matrix
