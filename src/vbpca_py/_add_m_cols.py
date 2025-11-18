import numpy as np


def _add_m_cols(
    S: np.ndarray,
    Sv: np.ndarray | list[np.ndarray],
    Ic: np.ndarray | list[int],
    n2x: int,
    Isv: np.ndarray | list[int] | None = None,
) -> tuple[np.ndarray, np.ndarray | list[np.ndarray], np.ndarray | list[int] | None]:
    """Expand S and Sv to the original number of columns (n2x).

    This is used at the end of pca_full to reinsert columns that were removed by
    rmempty. Indices are **0-based** throughout (Python style):

    - S has shape (ncomp, n2) and corresponds to columns in Ic.
    - Ic contains 0-based indices of the kept columns (after rmempty) in the
      original n2x columns.
    - Sv can be:
        * list of full covariance matrices (per-column or per-pattern), or
        * 1D ndarray of diagonal variances (length ncomp).
    - Isv (if not None/empty) maps each kept column index to a 0-based index
      into Sv (unique covariance patterns mode).

    Parameters
    ----------
    S:
        Component matrix of shape (ncomp, n2_kept).
    Sv:
        Either a list of covariance matrices or a 1D array of diagonal
        variances (length ncomp).
    Ic:
        Iterable of 0-based column indices (length n2_kept) referring to
        original column positions in [0, n2x).
    n2x:
        Total number of columns in the original data matrix.
    Isv:
        Optional mapping from kept columns to indices into Sv (0-based). If
        provided and non-empty, the returned Isv_new will be a length-n2x
        mapping for all columns.

    Returns:
    -------
    S_new:
        Expanded component matrix of shape (ncomp, n2x).
    Sv_new:
        Expanded covariance structure:
        - list of length n2x (per-column covariance mode), or
        - list of unique covariances (pattern mode) with Isv_new as mapping, or
        - ndarray of shape (ncomp, n2x) in diagonal mode.
    Isv_new:
        Updated mapping from column index to Sv_new index (0-based) in the
        unique-patterns case, or [] / None otherwise.
    """
    S = np.asarray(S)
    if S.ndim != 2:
        raise ValueError("S must be a 2D array of shape (ncomp, n2_kept).")

    ncomp, n2_kept = S.shape

    Ic_arr = np.asarray(Ic, dtype=int)
    if Ic_arr.ndim != 1:
        raise ValueError("Ic must be a 1D array or list of column indices.")

    if Ic_arr.size != n2_kept:
        raise ValueError(
            f"Number of kept columns in Ic ({Ic_arr.size}) must match "
            f"S.shape[1] ({n2_kept})."
        )

    if np.any(Ic_arr < 0) or np.any(Ic_arr >= n2x):
        raise ValueError("Ic contains indices out of range [0, n2x).")

    # Columns that were removed by rmempty (to be filled with defaults).
    all_cols = np.arange(n2x, dtype=int)
    mask_kept = np.zeros(n2x, dtype=bool)
    mask_kept[Ic_arr] = True
    Ic2 = all_cols[~mask_kept]  # missing columns (0-based indices)

    # Expand S to full width with zeros for missing columns.
    S_new = np.zeros((ncomp, n2x), dtype=S.dtype)
    S_new[:, Ic_arr] = S

    # Handle Sv depending on its representation.
    if isinstance(Sv, list):
        # Full covariance matrices (per-column or per-pattern).
        if Isv is None or len(Isv) == 0:
            # Per-column covariance mode: Sv assumed to have length n2_kept,
            # one covariance matrix per kept column.
            if len(Sv) != n2_kept:
                raise ValueError(
                    "In per-column covariance mode (empty Isv), Sv must "
                    "have length equal to the number of kept columns."
                )

            Sv_new: list[np.ndarray] = [None] * n2x  # type: ignore[assignment]
            # Assign learned covariances to their original positions.
            for j, col in enumerate(Ic_arr):
                Sv_new[col] = Sv[j]

            # Fill missing columns with identity covariance.
            for col in Ic2:
                Sv_new[col] = np.eye(ncomp, dtype=S.dtype)

            Isv_new: list[int] | None = []
        else:
            # Unique-pattern mode: Sv is a list of unique covariances and Isv
            # maps kept columns to 0-based Sv indices.
            Isv_arr = np.asarray(Isv, dtype=int)
            if Isv_arr.ndim != 1 or Isv_arr.size != n2_kept:
                raise ValueError(
                    "Isv must be a 1D array/list whose length equals the "
                    "number of kept columns when provided."
                )

            Sv_new = list(Sv)
            # Append identity covariance for newly added columns.
            identity_cov = np.eye(ncomp, dtype=S.dtype)
            Sv_new.append(identity_cov)
            identity_idx = len(Sv_new) - 1  # 0-based index of the new identity

            # Initialize all columns to use the identity covariance.
            Isv_new = [identity_idx] * n2x

            # Preserve learned mapping for kept columns.
            for j, col in enumerate(Ic_arr):
                idx = int(Isv_arr[j])
                if idx < 0 or idx >= len(Sv):
                    raise ValueError("Isv contains index out of range for Sv.")
                Isv_new[col] = idx

    else:
        # Diagonal covariance mode: Sv is a 1D array of length ncomp.
        Sv_arr = np.asarray(Sv)
        if Sv_arr.ndim != 1 or Sv_arr.shape[0] != ncomp:
            raise ValueError(
                "In diagonal covariance mode, Sv must be a 1D array of "
                "length equal to ncomp."
            )

        Sv_new = np.ones((ncomp, n2x), dtype=Sv_arr.dtype)
        Sv_new[:, Ic_arr] = Sv_arr[:, np.newaxis]
        Isv_new = None

    return S_new, Sv_new, Isv_new
