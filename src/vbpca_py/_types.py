"""Dtype contracts for numerical working state and compatibility inputs.

Shapes and ownership are runtime contracts. ``InputArray`` is deliberately
unconstrained at conversion/encoding boundaries; numerical working arrays use
``Float64Array``. Legacy observation weights can use float64 or uint8 storage,
while normalized dense observations use ``BoolArray``. SciPy matrix aliases
remain unparameterized for the supported SciPy/stub range; their data and index
buffers are checked/coerced at the native boundaries.
"""

from __future__ import annotations

from typing import TypeAlias

import numpy as np
import numpy.typing as npt
import scipy.sparse as sp

Float64Array: TypeAlias = npt.NDArray[np.float64]
EncodedArray: TypeAlias = npt.NDArray[np.number] | npt.NDArray[np.object_]
FloatingArray: TypeAlias = npt.NDArray[
    np.float16 | np.float32 | np.float64 | np.longdouble
]
BoolArray: TypeAlias = npt.NDArray[np.bool_]
IndexArray: TypeAlias = npt.NDArray[np.int_]
SparseIndexArray: TypeAlias = npt.NDArray[np.int32] | npt.NDArray[np.int64]
CountArray: TypeAlias = npt.NDArray[np.integer] | npt.NDArray[np.floating]
InputArray: TypeAlias = npt.NDArray[np.generic]
InputMatrix: TypeAlias = InputArray | sp.csr_matrix
InputMask: TypeAlias = InputArray | sp.csr_matrix
ObjectArray: TypeAlias = npt.NDArray[np.object_]
DenseMask: TypeAlias = BoolArray | Float64Array | npt.NDArray[np.uint8]
MaskMatrix: TypeAlias = DenseMask | sp.csr_matrix
Array: TypeAlias = Float64Array
Dense: TypeAlias = Float64Array
Sparse: TypeAlias = sp.csr_matrix
Matrix: TypeAlias = Float64Array | Sparse
ArrayLike: TypeAlias = InputArray | Sparse
