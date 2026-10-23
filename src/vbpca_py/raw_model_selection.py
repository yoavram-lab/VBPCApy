"""Training-only preprocessing for original-variable cell cross-validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import numpy as np
import scipy.sparse as sp

from ._selection_core import (
    CVConfig,
    _aggregate_cv_results,
    _apply_rule,
    _cv_groups,
    _first_minimum_stops,
    _make_element_folds,
    _normalize_components,
    _run_fold,
)
from ._types import BoolArray, Float64Array, InputArray  # noqa: TC001 - runtime annotation resolution
from ._validation import validate_rank
from .preprocessing import AutoEncoder, _ensure_mask

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from .encoding import EncodingSchema


@dataclass(slots=True)
class PreprocessedFold:
    """One raw-table holdout and its training-fitted preprocessing.

    Attributes:
        encoder: Fresh encoder fitted on training observations only.
        training_mask: Samples-by-variables training observation mask.
        heldout_mask: Samples-by-variables held-out observation mask.
        training: Samples-by-encoded-features training matrix.
        held_out: Samples-by-encoded-features probe matrix.
        raw: Raw samples-by-variables values, with unobserved cells masked.
    """

    encoder: AutoEncoder
    training_mask: BoolArray
    heldout_mask: BoolArray
    training: Float64Array
    held_out: Float64Array
    raw: InputArray

    def __post_init__(self) -> None:
        """Require the declared sample axis without inferring an orientation.

        Raises:
            ValueError: Encoded arrays or masks do not match the raw/schema axes.
        """
        expected = (self.raw.shape[0], len(self.schema.feature_groups))
        if (
            self.training.shape != expected
            or self.held_out.shape != expected
            or self.training_mask.shape != self.raw.shape
            or self.heldout_mask.shape != self.raw.shape
        ):
            msg = (
                "prepared fold arrays must use samples-by-features "
                "and aligned raw masks"
            )
            raise ValueError(msg)

    def __getstate__(self) -> dict[str, object]:
        """Record the fold layout explicitly for serialization.

        Returns:
            Named fold state with the native layout version.
        """
        return {
            "layout_version": 1,
            "encoder": self.encoder,
            "training_mask": self.training_mask,
            "heldout_mask": self.heldout_mask,
            "training": self.training,
            "held_out": self.held_out,
            "raw": self.raw,
        }

    def __setstate__(self, state: object) -> None:
        """Restore native folds; old feature-first artifacts must be rebuilt.

        Raises:
            ValueError: The layout marker is absent or incompatible.
        """
        if not isinstance(state, dict) or state.get("layout_version") != 1:
            msg = "Incompatible prepared-fold layout; rebuild pre-0.5 CV folds"
            raise ValueError(msg)
        self.encoder = cast("AutoEncoder", state["encoder"])
        self.training_mask = cast("BoolArray", state["training_mask"])
        self.heldout_mask = cast("BoolArray", state["heldout_mask"])
        self.training = cast("Float64Array", state["training"])
        self.held_out = cast("Float64Array", state["held_out"])
        self.raw = cast("InputArray", state["raw"])
        self.__post_init__()

    @property
    def schema(self) -> EncodingSchema:
        """Fitted metadata for this fold's encoded matrix.

        Raises:
            ValueError: If the encoder no longer has a fitted dense schema.
        """
        schema = self.encoder.encoding_schema_
        if schema is None:
            msg = "fold encoder has no fitted dense encoding schema"
            raise ValueError(msg)
        return schema

    def numeric_scores(self, prediction: Float64Array) -> dict[str, float]:
        """Score each numeric variable in input or ordinal-index units.

        Args:
            prediction: Samples-by-encoded-features predicted means.

        Returns:
            ``rmse_variable_j`` for each continuous or ordinal input column.
            ``NaN`` when the fold holds out no cells of that variable.

        Raises:
            ValueError: Prediction shape is invalid or a held-out numeric
                prediction is not finite.
        """
        if prediction.shape != self.training.shape:
            msg = "prediction must match the samples-by-encoded-features fold shape"
            raise ValueError(msg)
        scores: dict[str, float] = {}
        for j, variable in enumerate(self.schema.variables):
            if variable.kind == "categorical":
                continue
            cells = np.flatnonzero(self.heldout_mask[:, j])
            if variable.kind == "ordinal":
                levels = {
                    level: index for index, level in enumerate(variable.categories)
                }
                truth = np.array([levels[v] for v in self.raw[cells, j]], dtype=float)
            else:
                truth = np.asarray(self.raw[cells, j], dtype=float)
            decoded = variable.decode_numeric(
                prediction[np.ix_(cells, np.asarray(variable.columns, dtype=np.intp))].T
            )
            if cells.size and not np.all(np.isfinite(decoded)):
                msg = "held-out numeric predictions must be finite"
                raise ValueError(msg)
            scores[f"rmse_variable_{j}"] = (
                float(np.sqrt(np.mean((decoded[0] - truth) ** 2)))
                if cells.size
                else float("nan")
            )
        return scores

    def _feature_first_scores(self, prediction: Float64Array) -> dict[str, float]:
        """Adapt the private numerical scorer's axes without a prediction copy.

        Returns:
            Numeric errors in the established input-variable units.
        """
        return self.numeric_scores(prediction.T)


def prepare_raw_cv_folds(
    x: InputArray,
    *,
    encoder_factory: Callable[[], AutoEncoder],
    mask: BoolArray | None = None,
    config: CVConfig | None = None,
) -> list[PreprocessedFold]:
    """Construct raw variable-cell folds and fit each encoder once.

    Args:
        x: Dense samples-by-original-variables table; strings are supported.
        encoder_factory: Return a fresh AutoEncoder with explicit column_types
            and handle_unknown="raise". Optional column_levels must describe
            external vocabularies, rather than levels learned from holdouts.
        mask: Raw observation mask; NaNs are always missing.
        config: Fold count and seed. Encoded groups/schemas are not accepted.

    Returns:
        Deterministic folds retaining training support in each variable/sample.

    Raises:
        ValueError: If inputs/configuration are invalid, an encoder is reused,
            a held-out level is unknown, or training support is insufficient.
    """
    cfg = config or CVConfig()
    _validate_raw_config(x, mask, cfg)
    raw = np.array(x, dtype=object, copy=True)
    observed = _ensure_mask(raw, mask)
    raw[~observed] = np.nan
    if not np.all(observed.any(axis=0)) or not np.all(observed.any(axis=1)):
        msg = "raw CV requires an observation in every variable and sample"
        raise ValueError(msg)
    partition = _make_element_folds(
        np.where(observed.T, 1.0, np.nan), cfg.n_splits, np.random.default_rng(cfg.seed)
    )
    rows, columns = np.nonzero(observed.T)
    folds: list[PreprocessedFold] = []
    for fold_index, (probe, _train) in enumerate(partition):
        held = np.zeros(observed.shape, dtype=bool)
        held[columns[probe], rows[probe]] = True
        encoder = encoder_factory()
        if (
            not isinstance(encoder, AutoEncoder)
            or encoder.column_types is None
            or len(encoder.column_types) != raw.shape[1]
            or encoder.handle_unknown != "raise"
            or any(encoder is fold.encoder for fold in folds)
        ):
            msg = (
                "encoder_factory must return a fresh AutoEncoder with explicit "
                "column_types and handle_unknown='raise'"
            )
            raise ValueError(msg)
        train_mask = observed & ~held
        train_raw = raw.copy()
        train_raw[~train_mask] = np.nan
        encoder.fit(train_raw, mask=train_mask)
        training = np.asarray(
            encoder.transform(train_raw, mask=train_mask), dtype=float
        )
        try:
            held_out = np.asarray(encoder.transform(raw, mask=held), dtype=float)
        except ValueError as error:
            msg = (
                f"fold {fold_index} cannot encode held-out observations; check input "
                "values and, for unseen categorical levels, declare external "
                f"vocabularies with column_levels: {error}"
            )
            raise ValueError(msg) from error
        fold = PreprocessedFold(encoder, train_mask, held, training, held_out, raw)
        _validate_preprocessed_fold(fold, fold_index, cfg)
        folds.append(fold)
    return folds


def _validate_preprocessed_fold(
    fold: PreprocessedFold, index: int, cfg: CVConfig
) -> None:
    schema = fold.schema
    schema.validate(fold.training.shape[1], strict_numeric=True)
    if any(not variable.columns for variable in schema.variables) or not np.all(
        np.isfinite(fold.training).any(axis=1)
    ):
        msg = f"fold {index} has insufficient encoded training support"
        raise ValueError(msg)
    for encoded, mask in (
        (fold.training, fold.training_mask),
        (fold.held_out, fold.heldout_mask),
    ):
        if not np.array_equal(np.isfinite(encoded), mask[:, schema.feature_groups]):
            msg = (
                f"fold {index} must preserve raw observation masks "
                "with finite encoded values"
            )
            raise ValueError(msg)
    _cv_groups(CVConfig(metric=cfg.metric, encoding_schema=schema), fold.held_out.T)


def cross_validate_raw_components(
    x: InputArray,
    *,
    encoder_factory: Callable[[], AutoEncoder],
    mask: BoolArray | None = None,
    components: Iterable[int] | None = None,
    config: CVConfig | None = None,
    **opts: object,
) -> tuple[int, list[dict[str, object]], list[PreprocessedFold]]:
    """Select capacity after fitting preprocessing within each raw cell fold.

    Args:
        x: Dense samples-by-original-variables table.
        encoder_factory: Fresh configured AutoEncoder per fold; see
            prepare_raw_cv_folds for vocabulary and unknown-level requirements.
        mask: Raw observation mask.
        components: Candidate capacities, optionally including zero. Default
            candidates fit every fold's encoded dimensions.
        config: CV metric/rule, fold count and seed; first-minimum early stopping
            reuses the same prepared encoders across candidate capacities.
        **opts: VBPCA fitting options, applied independently in each fold.

    Returns:
        Selected capacity, aggregate CV records and the prepared folds for
        audit. Categorical metrics use original levels; numeric RMSE is reported
        per variable in continuous input units or ordinal level-index units.
        Probe RMS remains in encoded units. Numeric summaries use folds with
        observed holdouts and report their count.

    Raises:
        ValueError: If raw folds cannot be prepared, capacities exceed a fold's
            dimensions, or an external probe conflicts with these holdouts.
    """
    cfg = config or CVConfig()
    rule = cfg.resolved_rule()
    if "xprobe" in opts or opts.get("xprobe_fraction", 0.0) != 0:
        msg = "raw CV constructs its own probes; do not supply xprobe"
        raise ValueError(msg)
    folds = prepare_raw_cv_folds(
        x, encoder_factory=encoder_factory, mask=mask, config=cfg
    )
    max_k = min(min(fold.training.shape) for fold in folds)
    candidates = _normalize_components(
        None
        if components is None
        else [validate_rank(k, (max_k, max_k)) for k in components],
        max_k,
        max_k,
    )
    fit_options = {"compat_mode": "modern", **opts}
    numeric = tuple(
        f"rmse_variable_{j}"
        for j, v in enumerate(folds[0].schema.variables)
        if v.kind != "categorical"
    )
    evaluated = []
    metrics: list[dict[int, dict[str, object]]] = [{} for _ in folds]
    batches = [[k] for k in sorted(candidates)] if cfg.early_stop else [candidates]
    for batch in batches:
        for index, fold in enumerate(folds):
            metrics[index].update(_evaluate_fold(fold, index, batch, cfg, fit_options))
        evaluated.extend(batch)
        _, results = _aggregate_cv_results(evaluated, metrics, cfg.metric, numeric)
        if (
            cfg.early_stop
            and len(results) > 1
            and _first_minimum_stops(results[-2], results[-1], cfg.metric)
        ):
            break
    best, results = _aggregate_cv_results(evaluated, metrics, cfg.metric, numeric)
    return _apply_rule(best, results, rule, cfg.metric), results, folds


def _evaluate_fold(
    fold: PreprocessedFold,
    index: int,
    candidates: list[int],
    config: CVConfig,
    options: dict[str, object],
) -> dict[int, dict[str, object]]:
    base = np.where(np.isfinite(fold.held_out), fold.held_out, fold.training).T
    rows, columns = np.nonzero(np.isfinite(base))
    probes = np.flatnonzero(np.isfinite(fold.held_out.T[rows, columns]))
    schema = fold.schema
    return _run_fold(
        index,
        base,
        rows,
        columns,
        probes,
        k_list=candidates,
        opts={
            **options,
            "verbose": 0,
            "_cell_groups": schema.feature_groups,
            "_encoding_schema": schema,
            "_extra_scorer": fold._feature_first_scores,  # noqa: SLF001
        },
        seed=config.seed,
        n_splits=config.n_splits,
    )


def _validate_raw_config(x: InputArray, mask: BoolArray | None, cfg: CVConfig) -> None:
    if sp.issparse(x) or sp.issparse(mask) or np.ndim(x) != 2:
        msg = "raw CV needs a dense samples-by-variables table and dense mask"
        raise ValueError(msg)
    if (
        cfg.n_splits < 2
        or cfg.feature_groups is not None
        or cfg.encoding_schema is not None
        or cfg.metric not in {"prms", "brier", "log_score"}
    ):
        msg = (
            "raw CV needs at least two folds, a supported metric "
            "and no encoded groups/schema"
        )
        raise ValueError(msg)
