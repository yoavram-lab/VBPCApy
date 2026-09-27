"""Regime-stratified summaries for convergence-detector validation."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np


def _mean(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize endpoint and detector safety for one record subset."""
    policy_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        for row in record["policies"]["rows"]:
            policy_rows[str(row["policy"])].append(row)
    policies: dict[str, Any] = {}
    for name, rows in policy_rows.items():
        evaluable = [row for row in rows if row["evaluable"]]
        excess = [
            float(row["excess_iterations"])
            for row in evaluable
            if row["excess_iterations"] is not None
        ]
        policies[name] = {
            "n": len(rows),
            "n_evaluable": len(evaluable),
            "premature_rate": _mean([float(row["premature"]) for row in evaluable]),
            "late_or_no_stop_rate": _mean([
                float(row["late_or_no_stop"]) for row in evaluable
            ]),
            "median_excess_iterations": (float(np.median(excess)) if excess else None),
        }
    return {
        "n_records": len(records),
        "endpoint_stable_rate": _mean([
            float(record["fidelity"]["endpoint_stable"]) for record in records
        ]),
        "policies": policies,
    }


def summaries_by(
    records: list[dict[str, Any]], cell_field: str
) -> dict[str, dict[str, Any]]:
    """Return summaries stratified by one frozen cell field."""
    values = sorted({str(record["cell"][cell_field]) for record in records})
    return {
        value: summarize_records([
            record for record in records if str(record["cell"][cell_field]) == value
        ])
        for value in values
    }
