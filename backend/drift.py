# PSI thresholds 0.1 / 0.25 are a common rule of thumb. MIN_ROWS is a provisional choice, not a validated value.
"""Data and prediction drift monitoring utilities (Task 11 / Stage 6).

Role 2: Trust Engine Engineer
Provides:
  - psi(train_vals, live_vals, bins=10): Population Stability Index
  - read_request_log(path, include_rotated=True): Safe JSON-lines log reader
  - flag_rate(rows): Refusal rate and verdict distribution counter
  - feature_psi(rows, reference, min_rows=MIN_ROWS): Per-feature drift scanner
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

FEATURES: List[str] = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
MIN_ROWS: int = 30


def psi(
    train_vals: Any,
    live_vals: Any,
    bins: int = 10,
) -> Optional[float]:
    """Calculate the Population Stability Index (PSI) between reference and live values.

    Formula (Section 7.10):
        Quantile bin edges are derived from train_vals with the first and last edges
        set to -inf and +inf. Proportion counts t and l are calculated for train and live,
        both clipped at 1e-6 to avoid zero-division and log(0):
            PSI = sum((l - t) * ln(l / t))

    Rule of thumb (heuristic guide):
        - PSI < 0.1: Stable, negligible distribution shift.
        - 0.1 <= PSI <= 0.25: Moderate shift, warranting monitoring.
        - PSI > 0.25: Significant shift, potential model degradation.
        (Note: these thresholds are an industry rule of thumb, not a mathematically proven cutoff).

    Args:
        train_vals: Reference/training distribution values (sequence or array).
        live_vals: Production/live distribution values (sequence or array).
        bins: Number of quantile bins to construct from reference (default 10).

    Returns:
        float: Computed PSI value, or None if either input has no finite values.
    """
    if train_vals is None or live_vals is None:
        return None

    try:
        t_arr = np.asarray(train_vals, dtype=float).ravel()
        l_arr = np.asarray(live_vals, dtype=float).ravel()
    except (ValueError, TypeError):
        return None

    t_clean = t_arr[np.isfinite(t_arr)]
    l_clean = l_arr[np.isfinite(l_arr)]

    if len(t_clean) == 0 or len(l_clean) == 0:
        return None

    if bins <= 0:
        bins = 10

    quantiles = np.linspace(0, 100, bins + 1)
    edges = np.percentile(t_clean, quantiles)
    edges[0] = -np.inf
    edges[-1] = np.inf

    counts_t, _ = np.histogram(t_clean, bins=edges)
    counts_l, _ = np.histogram(l_clean, bins=edges)

    sum_t = counts_t.sum()
    sum_l = counts_l.sum()

    if sum_t == 0 or sum_l == 0:
        return None

    t = counts_t / sum_t
    l = counts_l / sum_l

    t = np.clip(t, 1e-6, 1.0)
    l = np.clip(l, 1e-6, 1.0)

    total = np.sum((l - t) * np.log(l / t))
    if not np.isfinite(total):
        return None

    return float(total)


def read_request_log(
    path: Optional[Union[str, Path]],
    include_rotated: bool = True,
) -> Dict[str, Any]:
    """Safely read JSON-lines request log files without raising exceptions.

    Reads from `path` and, if include_rotated=True, reads `path.1` first so rows
    stay in chronological order. Files are opened with errors="replace" so corrupt
    byte sequences do not crash reading and only invalidate their specific lines.

    Args:
        path: Path to the current request log file (str, Path, or None).
        include_rotated: Whether to read the rotated archive `.1` file first.

    Returns:
        Dict containing:
            "rows": List of decoded row dictionaries.
            "n_rows": Total valid JSON object rows read.
            "n_bad_lines": Count of skipped malformed or non-object lines.
            "files_read": List of filenames successfully opened and parsed.
            "status": "ok" if n_rows > 0, else "no_data".
    """
    rows: List[Dict[str, Any]] = []
    n_bad_lines = 0
    files_read: List[str] = []

    if path is None:
        return {
            "rows": rows,
            "n_rows": 0,
            "n_bad_lines": 0,
            "files_read": files_read,
            "status": "no_data",
        }

    try:
        base_path = Path(path)
    except (TypeError, ValueError, Exception):
        return {
            "rows": rows,
            "n_rows": 0,
            "n_bad_lines": 0,
            "files_read": files_read,
            "status": "no_data",
        }

    file_targets: List[Path] = []
    if include_rotated:
        try:
            file_targets.append(Path(str(base_path) + ".1"))
        except Exception:
            pass
    file_targets.append(base_path)

    for target in file_targets:
        try:
            if not target.is_file():
                continue
        except (OSError, Exception):
            continue

        try:
            with open(target, "r", encoding="utf-8", errors="replace") as f:
                lines = f.readlines()
            files_read.append(str(target))
        except (OSError, Exception):
            continue

        for line in lines:
            line_str = line.strip()
            if not line_str:
                continue

            try:
                obj = json.loads(line_str)
                if isinstance(obj, dict):
                    rows.append(obj)
                else:
                    n_bad_lines += 1
            except (json.JSONDecodeError, Exception):
                n_bad_lines += 1

    n_rows = len(rows)
    status = "ok" if n_rows > 0 else "no_data"

    return {
        "rows": rows,
        "n_rows": n_rows,
        "n_bad_lines": n_bad_lines,
        "files_read": files_read,
        "status": status,
    }


def flag_rate(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate the refusal (flag) rate and verdict breakdown across request rows.

    Args:
        rows: List of request dictionaries containing a "verdict" field.

    Returns:
        Dict containing:
            "status": "ok" if rows is non-empty, else "no_data".
            "flag_rate": float (refuse count / total) or None if no data.
            "counts": {"recommend": int, "caution": int, "refuse": int}.
            "total": int (total rows inspected).
    """
    if not rows:
        return {
            "status": "no_data",
            "flag_rate": None,
            "counts": {
                "recommend": 0,
                "caution": 0,
                "refuse": 0,
            },
            "total": 0,
        }

    recommend = sum(1 for r in rows if isinstance(r, dict) and r.get("verdict") == "recommend")
    caution = sum(1 for r in rows if isinstance(r, dict) and r.get("verdict") == "caution")
    refuse = sum(1 for r in rows if isinstance(r, dict) and r.get("verdict") == "refuse")
    total = len(rows)

    rate = float(refuse / total) if total > 0 else None

    return {
        "status": "ok",
        "flag_rate": rate,
        "counts": {
            "recommend": recommend,
            "caution": caution,
            "refuse": refuse,
        },
        "total": total,
    }


def _extract_feature_val(row: Dict[str, Any], feature: str) -> Optional[float]:
    """Helper to extract a finite float feature value from a request dictionary."""
    if not isinstance(row, dict):
        return None

    val = row.get(feature)
    if val is None and isinstance(row.get("inputs"), dict):
        val = row["inputs"].get(feature)
    if val is None and isinstance(row.get("features"), dict):
        val = row["features"].get(feature)

    if val is None:
        return None

    try:
        fval = float(val)
        return fval if np.isfinite(fval) else None
    except (ValueError, TypeError):
        return None


def feature_psi(
    rows: Any,
    reference: Any,
    min_rows: int = MIN_ROWS,
) -> Dict[str, Any]:
    """Calculate Population Stability Index (PSI) per feature against reference distributions.

    Args:
        rows: Sequence of request log rows (dictionaries), accepted as list or tuple.
        reference: Dict mapping feature name to array-like of reference/training values.
        min_rows: Minimum count of usable live rows required to report PSI.

    Returns:
        Dict containing:
            "status": "ok" if n_live >= min_rows, "not_enough_data" if n_live < min_rows,
                      or "no_reference" if reference is not a valid dictionary.
            "n_live": Number of valid dict rows inspected.
            "min_rows": Minimum threshold applied.
            "psi": Dict mapping each feature in FEATURES to float or None.
    """
    valid_rows = [r for r in rows if isinstance(r, dict)] if isinstance(rows, (list, tuple)) else []
    n_live = len(valid_rows)

    if not isinstance(reference, dict):
        return {
            "status": "no_reference",
            "n_live": n_live,
            "min_rows": min_rows,
            "psi": {feat: None for feat in FEATURES},
        }

    if n_live < min_rows:
        return {
            "status": "not_enough_data",
            "n_live": n_live,
            "min_rows": min_rows,
            "psi": {feat: None for feat in FEATURES},
        }

    psi_results: Dict[str, Optional[float]] = {}
    for feat in FEATURES:
        if feat not in reference or reference[feat] is None:
            psi_results[feat] = None
            continue

        live_vals = [
            v for r in valid_rows
            if (v := _extract_feature_val(r, feat)) is not None
        ]

        if len(live_vals) < min_rows:
            psi_results[feat] = None
        else:
            psi_results[feat] = psi(reference[feat], live_vals)

    return {
        "status": "ok",
        "n_live": n_live,
        "min_rows": min_rows,
        "psi": psi_results,
    }

