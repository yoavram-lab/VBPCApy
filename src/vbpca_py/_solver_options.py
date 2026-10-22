"""Mutable per-fit state around immutable typed solver configuration.

The mapping surface is an adapter for legacy numerical helpers. Public option
names are fixed by PCAOptions; caches and preflight records have declared fields
and never enter estimator parameters or serialized user keyword dictionaries.
"""

from __future__ import annotations

from collections.abc import Iterator, MutableMapping
from dataclasses import dataclass, fields, replace
from typing import TYPE_CHECKING, cast

from .configuration import PCAOptions, _replace_config

if TYPE_CHECKING:
    from ._noise_statistics import NoiseComplement


_CONFIG_KEYS = tuple(item.name for item in fields(PCAOptions) if item.init)
_CONFIG_FIELDS = frozenset(_CONFIG_KEYS)
_RUNTIME_FIELDS = {
    "_num_cpu_user_set": "num_cpu_user_set",
    "_dense_score_use_complement": "dense_score_use_complement",
    "_dense_loading_use_complement": "dense_loading_use_complement",
    "_noise_complement": "noise_complement",
    "_runtime_preflight": "preflight",
}


@dataclass(slots=True)
class SolverOptions(MutableMapping[str, object]):
    """Typed settings plus state owned by exactly one numerical fit."""

    config: PCAOptions
    num_cpu_user_set: bool = False
    dense_score_use_complement: bool = False
    dense_loading_use_complement: bool = False
    noise_complement: NoiseComplement | None = None
    preflight: list[dict[str, object]] | None = None

    def __getitem__(self, key: str) -> object:
        if key in _CONFIG_FIELDS:
            return cast("object", getattr(self.config, key))
        if key in _RUNTIME_FIELDS:
            if key == "_runtime_preflight" and self.preflight is None:
                raise KeyError(key)
            return cast("object", getattr(self, _RUNTIME_FIELDS[key]))
        raise KeyError(key)

    def __setitem__(self, key: str, value: object) -> None:
        if key in _CONFIG_FIELDS:
            current = getattr(self.config, key)
            if current is value or (
                isinstance(current, str | int | float)
                and isinstance(value, str | int | float)
                and current == value
            ):
                return
            self.config = _replace_config(self.config, {key: value})
        elif key == "_num_cpu_user_set":
            self.num_cpu_user_set = bool(value)
        elif key == "_dense_score_use_complement":
            self.dense_score_use_complement = bool(value)
        elif key == "_dense_loading_use_complement":
            self.dense_loading_use_complement = bool(value)
        elif key == "_noise_complement":
            self.noise_complement = cast("NoiseComplement | None", value)
        elif key == "_runtime_preflight":
            self.preflight = cast("list[dict[str, object]]", value)
        else:
            raise KeyError(key)

    def __delitem__(self, key: str) -> None:
        if key != "_runtime_preflight":
            raise KeyError(key)
        self.preflight = None

    def __iter__(self) -> Iterator[str]:
        return iter((
            *_CONFIG_KEYS,
            *(
                key
                for key in _RUNTIME_FIELDS
                if key != "_runtime_preflight" or self.preflight is not None
            ),
        ))

    def __len__(self) -> int:
        return len(_CONFIG_FIELDS) + len(_RUNTIME_FIELDS) - int(self.preflight is None)

    def copy(self) -> SolverOptions:
        """Copy per-iteration metadata while sharing immutable configuration.

        Returns:
            A separate mutable state record with its own preflight list.
        """
        return replace(
            self, preflight=None if self.preflight is None else list(self.preflight)
        )
