"""Optional plotting utilities for VB-PCA models.

All functions require ``matplotlib`` (install via ``pip install vbpca_py[plot]``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from vbpca_py.estimators import VBPCA

__all__ = [
    "loadings_barplot",
    "scree_plot",
    "variance_explained_plot",
]


def _require_matplotlib() -> tuple[Any, ...]:
    """Import and return ``(plt, Figure, Axes)``.

    Returns:
        A 3-tuple of ``(matplotlib.pyplot, Figure, Axes)``.

    Raises:
        ImportError: If matplotlib is not installed.
    """
    try:
        import matplotlib.pyplot as plt  # noqa: PLC0415
        from matplotlib.axes import Axes as _Axes  # noqa: PLC0415
        from matplotlib.figure import Figure as _Figure  # noqa: PLC0415
    except ImportError:
        msg = (
            "matplotlib is required for plotting utilities. "
            "Install it with:  pip install vbpca_py[plot]"
        )
        raise ImportError(msg) from None
    return plt, _Figure, _Axes


def scree_plot(
    model: VBPCA,
    *,
    ax: Axes | None = None,
    cumulative: bool = True,
) -> Figure:
    """Plot explained variance ratio per component (scree plot).

    Args:
        model: A fitted ``VBPCA`` instance.
        ax: Optional matplotlib ``Axes`` to draw on.  A new figure is
            created when *None*.
        cumulative: If *True*, overlay a cumulative variance line.

    Returns:
        The matplotlib ``Figure`` containing the plot.

    Raises:
        RuntimeError: If the model has not been fitted or lacks
            explained-variance information.
    """
    plt, _, _ = _require_matplotlib()

    evr = model.explained_variance_ratio_
    if evr is None:
        msg = (
            "Model does not have explained_variance_ratio_"
            " (refit with compute_explained_variance?)."
        )
        raise RuntimeError(msg)

    evr = np.asarray(evr, dtype=float)
    n_comp = evr.size
    indices = np.arange(1, n_comp + 1)

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    else:
        fig = ax.get_figure()

    ax.bar(indices, evr, color="steelblue", alpha=0.8, label="Individual")
    if cumulative:
        ax.plot(
            indices,
            np.cumsum(evr),
            "o-",
            color="darkorange",
            label="Cumulative",
        )
    ax.set_xlabel("Component")
    ax.set_ylabel("Explained variance ratio")
    ax.set_title("Scree plot")
    ax.legend()
    ax.set_xticks(indices)
    fig.tight_layout()
    return fig  # type: ignore[no-any-return]


def loadings_barplot(
    model: VBPCA,
    component: int = 0,
    *,
    ax: Axes | None = None,
    top_n: int | None = None,
    feature_names: list[str] | None = None,
) -> Figure:
    """Bar plot of loadings (weights) for a single component.

    Args:
        model: A fitted ``VBPCA`` instance.
        component: Zero-based component index.
        ax: Optional matplotlib ``Axes``.
        top_n: If set, show only the *top_n* features by absolute loading.
        feature_names: Optional labels for the feature axis.

    Returns:
        The matplotlib ``Figure`` containing the plot.

    Raises:
        RuntimeError: If the model has not been fitted.
    """
    plt, _, _ = _require_matplotlib()

    if model.components_ is None:
        msg = "Model not fitted"
        raise RuntimeError(msg)

    loadings = np.asarray(model.components_[:, component], dtype=float)
    n_features = loadings.size

    if feature_names is None:
        feature_names = [f"F{i}" for i in range(n_features)]

    if top_n is not None and top_n < n_features:
        order = np.argsort(np.abs(loadings))[::-1][:top_n]
        loadings = loadings[order]
        feature_names = [feature_names[i] for i in order]

    if ax is None:
        width = max(6, len(feature_names) * 0.35)
        fig, ax = plt.subplots(figsize=(width, 4))
    else:
        fig = ax.get_figure()

    colours = ["steelblue" if v >= 0 else "coral" for v in loadings]
    ax.bar(range(len(loadings)), loadings, color=colours, alpha=0.85)
    ax.set_xticks(range(len(loadings)))
    ax.set_xticklabels(feature_names, rotation=45, ha="right")
    ax.set_ylabel("Loading")
    ax.set_title(f"Component {component} loadings")
    ax.axhline(0, color="grey", linewidth=0.5)
    fig.tight_layout()
    return fig  # type: ignore[no-any-return]


def variance_explained_plot(
    model: VBPCA,
    *,
    ax: Axes | None = None,
) -> Figure:
    """Plot per-component explained variance (absolute, not ratio).

    Args:
        model: A fitted ``VBPCA`` instance.
        ax: Optional matplotlib ``Axes``.

    Returns:
        The matplotlib ``Figure`` containing the plot.

    Raises:
        RuntimeError: If the model has not been fitted or lacks
            explained-variance information.
    """
    plt, _, _ = _require_matplotlib()

    ev = model.explained_variance_
    if ev is None:
        msg = (
            "Model does not have explained_variance_"
            " (refit with compute_explained_variance?)."
        )
        raise RuntimeError(msg)

    ev = np.asarray(ev, dtype=float)
    n_comp = ev.size
    indices = np.arange(1, n_comp + 1)

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    else:
        fig = ax.get_figure()

    ax.bar(indices, ev, color="steelblue", alpha=0.8)
    ax.set_xlabel("Component")
    ax.set_ylabel("Explained variance")
    ax.set_title("Variance explained per component")
    ax.set_xticks(indices)
    fig.tight_layout()
    return fig  # type: ignore[no-any-return]
