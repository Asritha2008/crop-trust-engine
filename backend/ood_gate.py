"""Generic Out-of-Distribution (OOD) Gate for Crop Recommendation.

Role 2: Trust Engine Engineer.
Provides a flexible, dimension-agnostic OOD gate based on per-crop Ledoit-Wolf
covariance shrinkage Mahalanobis distance.

Key properties:
  1. Accepts an arbitrary feature list (e.g., all 7 features or a 3-feature weather-only subset).
  2. Pure NumPy inference: serialized artifacts store raw numpy arrays (scaler mean, scale,
     crop centroids, precision matrices) and plain floats, ensuring compatibility across
     scikit-learn versions without pickling sklearn estimator objects.
  3. Calibrated threshold profiles derived from out-of-fold (OOF) cross-validation on training rows.
  4. Runs cleanly without web servers or external dependencies.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np


def fit_mahalanobis_gate(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    seed: int = 42,
) -> Dict[str, Any]:
    """Fit per-crop Ledoit-Wolf Mahalanobis detector on training data for ANY feature list.

    Args:
        X: Feature matrix of shape (n_samples, n_features).
        y: Class label array of shape (n_samples,).
        feature_names: Names of features corresponding to columns of X.
        seed: Random seed for reproducibility.

    Returns:
        Dictionary containing pure NumPy and primitive objects:
          - metadata: dict of configuration and provenance
          - scaler: dict with 'mean', 'scale', and 'var' numpy arrays
          - crop_stats: dict of per-crop 'mean', 'cov', 'inv_cov' arrays and 'shrinkage' float
    """
    from sklearn.covariance import LedoitWolf
    from sklearn.preprocessing import StandardScaler

    X_arr = np.asarray(X, dtype=float)
    y_arr = np.asarray(y)
    n_samples, n_features = X_arr.shape

    if len(feature_names) != n_features:
        raise ValueError(
            f"Feature count mismatch: X has {n_features} columns but {len(feature_names)} names provided"
        )

    # 1. Fit standard scaler on X
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_arr)

    # Store scaler parameters as raw numpy arrays
    scaler_dict = {
        "mean": np.asarray(scaler.mean_, dtype=float),
        "scale": np.asarray(scaler.scale_, dtype=float),
        "var": np.asarray(scaler.var_, dtype=float),
    }

    # 2. Fit per-crop Ledoit-Wolf precision matrices
    classes = sorted(list(np.unique(y_arr)))
    crop_stats: Dict[str, Dict[str, Any]] = {}

    for crop in classes:
        mask = (y_arr == crop)
        Xc_scaled = X_scaled[mask]
        lw = LedoitWolf(assume_centered=False).fit(Xc_scaled)
        crop_stats[str(crop)] = {
            "mean": np.asarray(lw.location_, dtype=float),
            "cov": np.asarray(lw.covariance_, dtype=float),
            "inv_cov": np.asarray(lw.precision_, dtype=float),
            "shrinkage": float(lw.shrinkage_),
        }

    metadata = {
        "version": "1.0.0",
        "role": "Role 2: Trust Engine Engineer",
        "algorithm": "per_crop_mahalanobis_ledoit_wolf",
        "features": list(feature_names),
        "n_crops": len(classes),
        "crops": [str(c) for c in classes],
        "train_samples": int(n_samples),
        "random_seed": int(seed),
        "scaling": "standard_scaler_numpy_arrays",
    }

    return {
        "metadata": metadata,
        "scaler": scaler_dict,
        "crop_stats": crop_stats,
    }


def compute_oof_scores(
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    seed: int = 42,
) -> np.ndarray:
    """Compute out-of-fold (OOF) minimum squared Mahalanobis distances on training data.

    Follows the 5-fold stratified cross-validation protocol to eliminate small-sample
    estimation noise on percentiles without leaking validation or test sets.

    Args:
        X: Training feature matrix (n_samples, n_features).
        y: Training class label array (n_samples,).
        n_splits: Number of stratified folds (default 5).
        seed: Random seed for cross-validation splitting.

    Returns:
        Array of shape (n_samples,) containing OOF squared Mahalanobis distances.
    """
    from sklearn.covariance import LedoitWolf
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler

    X_arr = np.asarray(X, dtype=float)
    y_arr = np.asarray(y)
    classes = sorted(list(np.unique(y_arr)))

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof_scores = np.zeros(len(X_arr), dtype=float)

    for tr_idx, oof_idx in skf.split(X_arr, y_arr):
        X_tr_f, y_tr_f = X_arr[tr_idx], y_arr[tr_idx]
        X_oof_f = X_arr[oof_idx]

        sc_f = StandardScaler().fit(X_tr_f)
        X_tr_sc = sc_f.transform(X_tr_f)
        X_oof_sc = sc_f.transform(X_oof_f)

        stats_f = {}
        for crop in classes:
            lw_f = LedoitWolf(assume_centered=False).fit(X_tr_sc[y_tr_f == crop])
            stats_f[crop] = {
                "mean": lw_f.location_,
                "inv_cov": lw_f.precision_,
            }

        for i, x_sc in enumerate(X_oof_sc):
            min_d2 = np.inf
            for crop in classes:
                diff = x_sc - stats_f[crop]["mean"]
                d2 = float(diff @ stats_f[crop]["inv_cov"] @ diff)
                if d2 < min_d2:
                    min_d2 = d2
            oof_scores[oof_idx[i]] = min_d2

    return oof_scores


def compute_threshold_profiles(
    oof_scores: np.ndarray,
    feature_names: List[str],
    seed: int = 42,
    round_digits: int = 2,
) -> Dict[str, Any]:
    """Compute empirical percentile thresholds and build profile configuration.

    Conservative: refuse p97, caution p90
    Balanced:     refuse p99, caution p95
    Bold:         refuse p99.5, caution p97

    Args:
        oof_scores: 1D array of OOF squared Mahalanobis distances.
        feature_names: List of feature names used.
        seed: Random seed used in CV generation.
        round_digits: Decimal places for stored cutoffs.

    Returns:
        Dictionary formatted to match the thresholds.json schema.
    """
    scores = np.asarray(oof_scores, dtype=float)
    n_samples = int(len(scores))

    med = float(np.median(scores))
    p90 = float(np.percentile(scores, 90.0))
    p95 = float(np.percentile(scores, 95.0))
    p97 = float(np.percentile(scores, 97.0))
    p99 = float(np.percentile(scores, 99.0))
    p99_5 = float(np.percentile(scores, 99.5))

    return {
        "version": "1.0.0",
        "method": "empirical_oof_percentiles",
        "detector": "per_crop_mahalanobis_ledoit_wolf",
        "feature_subset": list(feature_names),
        "reference_distribution": {
            "source": "5_fold_stratified_cv_train_oof",
            "n_samples": n_samples,
            "seed": int(seed),
            "median": med,
            "p90": p90,
            "p95": p95,
            "p97": p97,
            "p99": p99,
            "p99_5": p99_5,
        },
        "profiles": {
            "conservative": {
                "description": "High safety margin: refuses earlier on borderline unusual inputs.",
                "ood_refuse_threshold": round(p97, round_digits),
                "ood_refuse_percentile": 97.0,
                "ood_caution_threshold": round(p90, round_digits),
                "ood_caution_percentile": 90.0,
            },
            "balanced": {
                "description": "Recommended default: balanced trade-off between detection of true unknowns and false alarm rate.",
                "ood_refuse_threshold": round(p99, round_digits),
                "ood_refuse_percentile": 99.0,
                "ood_caution_threshold": round(p95, round_digits),
                "ood_caution_percentile": 95.0,
            },
            "bold": {
                "description": "Permissive: minimizes false alarms on familiar crops; reserves refusal for severe anomalies.",
                "ood_refuse_threshold": round(p99_5, round_digits),
                "ood_refuse_percentile": 99.5,
                "ood_caution_threshold": round(p97, round_digits),
                "ood_caution_percentile": 97.0,
            },
        },
        "limitations": [
            f"OOD gate operates strictly over feature subset: {feature_names}.",
            "Thresholds calibrated on 5-fold out-of-fold training distributions.",
            "Higher overlap between crops in lower dimensional subspaces than in 7D joint space.",
        ],
    }


def score_mahalanobis(
    X_input: Union[np.ndarray, Dict[str, float]],
    gate_model: Dict[str, Any],
) -> Union[Tuple[float, List[Tuple[str, float]], np.ndarray], np.ndarray]:
    """Score input(s) using purely NumPy array arithmetic.

    Args:
        X_input: Either a dict mapping feature name -> float, a 1D vector of shape (p,),
                 or a 2D matrix of shape (n, p).
        gate_model: Dictionary containing 'scaler', 'crop_stats', and 'metadata'.

    Returns:
        If single row dict or 1D array:
            Tuple of (min_d2, sorted_crop_distances, scaled_z).
        If 2D array:
            1D array of shape (n,) containing minimum squared Mahalanobis distance for each row.
    """
    scaler_info = gate_model["scaler"]
    features = gate_model["metadata"]["features"]
    crop_stats = gate_model["crop_stats"]

    # Support both dict and sklearn Scaler objects (for backward compatibility)
    if isinstance(scaler_info, dict):
        mean = scaler_info["mean"]
        scale = scaler_info["scale"]
    else:
        mean = getattr(scaler_info, "mean_")
        scale = getattr(scaler_info, "scale_")

    if isinstance(X_input, dict):
        missing = [f for f in features if f not in X_input]
        if missing:
            raise ValueError(f"Input dictionary missing required features: {missing}")
        x = np.array([float(X_input[f]) for f in features], dtype=float)
        is_single = True
    else:
        arr = np.asarray(X_input, dtype=float)
        if arr.ndim == 1:
            x = arr
            is_single = True
        elif arr.ndim == 2:
            x = arr
            is_single = False
        else:
            raise ValueError(f"Invalid input shape {arr.shape}, expected 1D or 2D array")

    if is_single:
        z = (x - mean) / scale
        crop_distances: List[Tuple[str, float]] = []
        for crop, stats in crop_stats.items():
            diff = z - stats["mean"]
            prec = stats["inv_cov"]
            d2 = float(diff.T @ prec @ diff)
            crop_distances.append((crop, d2))

        crop_distances.sort(key=lambda kv: kv[1])
        min_d2 = crop_distances[0][1]
        return min_d2, crop_distances, z
    else:
        Z = (x - mean) / scale
        n_samples = len(Z)
        all_d2 = np.zeros((n_samples, len(crop_stats)), dtype=float)
        for idx, (crop, stats) in enumerate(crop_stats.items()):
            diff = Z - stats["mean"]
            prec = stats["inv_cov"]
            all_d2[:, idx] = np.sum((diff @ prec) * diff, axis=1)
        return np.min(all_d2, axis=1)


class OODGate:
    """Standalone, lightweight OOD Gate runnable in pure Python without web server dependencies.

    Uses NumPy array calculations exclusively during inference.
    """

    SUPPORTED_RISK_PROFILES = ("conservative", "balanced", "bold")

    def __init__(
        self,
        scaler_mean: np.ndarray,
        scaler_scale: np.ndarray,
        crop_stats: Dict[str, Dict[str, Any]],
        oof_reference_scores: np.ndarray,
        thresholds: Dict[str, Any],
        features: List[str],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.scaler_mean = np.asarray(scaler_mean, dtype=float)
        self.scaler_scale = np.asarray(scaler_scale, dtype=float)
        self.crop_stats = crop_stats
        self.oof_reference_scores = np.asarray(oof_reference_scores, dtype=float)
        self.thresholds = thresholds
        self.features = list(features)
        self.metadata = metadata or {}

    @classmethod
    def load(
        cls,
        stats_path: Union[str, Path],
        thresholds_path: Union[str, Path],
    ) -> "OODGate":
        """Load OOD gate from saved joblib and json artifacts.

        Accepts artifacts storing NumPy arrays directly or backward-compatible dicts.
        """
        s_path = Path(stats_path)
        t_path = Path(thresholds_path)

        if not s_path.exists():
            raise FileNotFoundError(f"Missing stats artifact: {s_path}")
        if not t_path.exists():
            raise FileNotFoundError(f"Missing thresholds artifact: {t_path}")

        stats_data = joblib.load(s_path)
        with open(t_path, "r", encoding="utf-8") as f:
            thresholds_data = json.load(f)

        scaler = stats_data["scaler"]
        if isinstance(scaler, dict):
            scaler_mean = scaler["mean"]
            scaler_scale = scaler["scale"]
        else:
            scaler_mean = scaler.mean_
            scaler_scale = scaler.scale_

        crop_stats = stats_data["crop_stats"]
        oof_reference_scores = stats_data["oof_reference_scores"]
        metadata = stats_data.get("metadata", {})
        features = metadata.get("features", thresholds_data.get("feature_subset", []))

        return cls(
            scaler_mean=scaler_mean,
            scaler_scale=scaler_scale,
            crop_stats=crop_stats,
            oof_reference_scores=oof_reference_scores,
            thresholds=thresholds_data,
            features=features,
            metadata=metadata,
        )

    def score(
        self, row: Dict[str, float]
    ) -> Tuple[float, List[Tuple[str, float]], np.ndarray]:
        """Compute squared Mahalanobis distance to nearest crop using NumPy only."""
        missing = [f for f in self.features if f not in row]
        if missing:
            raise ValueError(f"Input dictionary missing required features: {missing}")

        x = np.array([float(row[f]) for f in self.features], dtype=float)
        z = (x - self.scaler_mean) / self.scaler_scale

        crop_distances: List[Tuple[str, float]] = []
        for crop, stats in self.crop_stats.items():
            diff = z - stats["mean"]
            prec = stats["inv_cov"]
            d2 = float(diff.T @ prec @ diff)
            crop_distances.append((crop, d2))

        crop_distances.sort(key=lambda kv: kv[1])
        min_d2 = crop_distances[0][1]
        return min_d2, crop_distances, z

    def assess_ood(
        self,
        row: Dict[str, Any],
        risk: str = "balanced",
    ) -> Dict[str, Any]:
        """Assess input out-of-distribution status.

        Args:
            row: Feature dictionary containing at least all gate features.
            risk: Risk profile ('conservative', 'balanced', or 'bold').

        Returns:
            Dict containing:
              - eligible: bool (True if not flagged for refusal)
              - flagged: bool (True if min_distance > refuse_threshold)
              - percentile: float (empirical rank vs OOF reference distribution)
              - min_distance: float (minimum squared Mahalanobis distance)
              - threshold: float (refuse cutoff for this profile)
              - caution_threshold: float (caution cutoff for this profile)
              - nearest_known: list of 2 nearest crop names [c1, c2]
              - risk_profile: str (normalized risk profile name)
        """
        normalized_risk = str(risk).strip().lower()
        if normalized_risk not in self.SUPPORTED_RISK_PROFILES:
            raise ValueError(
                f"Unsupported risk profile '{risk}'. Expected one of: {self.SUPPORTED_RISK_PROFILES}"
            )

        min_d2, sorted_crops, _ = self.score(row)

        percentile = float(np.mean(self.oof_reference_scores <= min_d2) * 100.0)
        percentile = round(percentile, 1)

        profile_cfg = self.thresholds["profiles"][normalized_risk]
        refuse_t = float(profile_cfg["ood_refuse_threshold"])
        caution_t = float(profile_cfg["ood_caution_threshold"])

        flagged = bool(min_d2 > refuse_t)
        eligible = not flagged

        nearest_known = [sorted_crops[0][0], sorted_crops[1][0]]

        return {
            "eligible": eligible,
            "flagged": flagged,
            "percentile": percentile,
            "min_distance": round(min_d2, 4),
            "threshold": refuse_t,
            "caution_threshold": caution_t,
            "nearest_known": nearest_known,
            "risk_profile": normalized_risk,
        }

    def assess(self, row: Dict[str, Any], risk: str = "balanced") -> Dict[str, Any]:
        """Alias for assess_ood()."""
        return self.assess_ood(row, risk=risk)

