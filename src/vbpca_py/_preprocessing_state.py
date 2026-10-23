"""Native fitted-state lifecycle shared by missing-aware preprocessors."""

from __future__ import annotations

from collections.abc import Generator  # noqa: TC003 - runtime annotation resolution
from contextlib import contextmanager

from sklearn.base import TransformerMixin  # type: ignore[import-untyped]
from sklearn.utils.validation import check_is_fitted  # type: ignore[import-untyped]

from ._native import TypedEstimator

_PRIVATE_FIT_STATE = frozenset({"_output_width", "_output_widths", "_plan", "_scaler"})


class PreprocessorState(TransformerMixin, TypedEstimator):  # type: ignore[misc]
    """Real sklearn transformer with transactional, fit-owned learned state."""

    n_features_in_: int | None

    def _clear_fit_state(self) -> None:
        """Release previous or partially fitted arrays and encoding plans."""
        # Invalidate fitted status before releasing any array/encoder payload.
        if "n_features_in_" in vars(self):
            del self.n_features_in_
        for name in tuple(vars(self)):
            if name.endswith("_") or name in _PRIVATE_FIT_STATE:
                delattr(self, name)

    @contextmanager
    def _fitting(self) -> Generator[None, None, None]:
        """Clear obsolete state and discard partial state when fitting fails.

        Yields:
            Control to the existing numerical fit implementation.
        """
        try:
            self._clear_fit_state()
            yield
        except BaseException:
            self._clear_fit_state()
            raise

    def __sklearn_is_fitted__(self) -> bool:  # noqa: PLW3201 - sklearn protocol
        """Recognize fitted feature metadata, including compatible old artifacts.

        Returns:
            Whether a feature count was learned; old unfitted None defaults
            remain distinguishable from successful fits.
        """
        return getattr(self, "n_features_in_", None) is not None

    def _check_fitted(self) -> None:
        """Require a completed fit through sklearn's validation helper."""
        check_is_fitted(self)
