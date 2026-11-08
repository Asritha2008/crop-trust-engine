"""Crop Trust Engine - Core Engine Interface and Stage 2 OOD Implementation.

Role 2: Trust Engine Engineer
Stage 2 implementation integrating empirical Out-of-Distribution (OOD) detection:
  - Input validation: exact 7 required features, finite numeric types, and physical bounds.
  - Empirical per-crop Mahalanobis distance with Ledoit-Wolf precision matrices.
  - Out-of-fold empirical percentile rank against 1,540 training reference scores.
  - Risk profile thresholds (conservative, balanced, bold) from backend/models/thresholds.json.
  - Excess bivariate coupling attribution via 2D sub-covariance Mahalanobis analysis.
  - Pure Stage 2 assessment method: assess_ood(row, risk).
  - Explicit handoff to Stages 3-5 via StagesNotImplementedError on non-refusal inputs.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd

try:
    from backend.noise import (
        apply_noise,
        compute_feature_sigmas,
        load_noise_config,
    )
except ModuleNotFoundError:
    import sys

    backend_dir = Path(__file__).resolve().parent
    if str(backend_dir) not in sys.path:
        sys.path.insert(0, str(backend_dir))
    from noise import apply_noise, compute_feature_sigmas, load_noise_config


class StagesNotImplementedError(NotImplementedError):
    """Raised when an input passes Stage 2 OOD gating but Stages 3-5

    (confidence calibration, stability assessment, and selective classification)
    are not yet integrated.

    Role 3 maps this exception to HTTP 501 Not Implemented with JSON payload:
        {"error": "Stages 3-5 not integrated yet"}
    """

    def __init__(
        self,
        message: str = "Stages 3-5 not integrated yet",
        ood_result: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(message)
        self.message: str = message
        self.ood_result: Optional[Dict[str, Any]] = ood_result


DEFAULT_SERVING_STABILITY_N_SAMPLES: int = 200
DEFAULT_SERVING_STABILITY_SEED: int = 42


class TrustEngine:
    """The Trust Engine coordinates validation, OOD detection, confidence

    calibration, stability checks, and selective classification verdicts.

    In Stage 2, it implements:
      - Stage 1: Input schema, numeric validation, and physical bounds enforcement.
      - Stage 2: Empirical OOD gate (per-crop Mahalanobis with Ledoit-Wolf shrinkage).
      - Stages 3-5: Pending integration of Role 1's calibrated pipeline and stability assessment.
    """

    SUPPORTED_RISK_PROFILES: Tuple[str, ...] = ("conservative", "balanced", "bold")
    REQUIRED_FEATURES: Tuple[str, ...] = (
        "N",
        "P",
        "K",
        "temperature",
        "humidity",
        "ph",
        "rainfall",
    )
    PHYSICAL_BOUNDS: Dict[str, Tuple[float, float]] = {
        "N": (0.0, 200.0),
        "P": (0.0, 200.0),
        "K": (0.0, 300.0),
        "temperature": (-10.0, 60.0),
        "humidity": (0.0, 100.0),
        "ph": (0.0, 14.0),
        "rainfall": (0.0, 1000.0),
    }
    MODEL_VERSION: str = "stub-2026-11-07"
    SERVING_STABILITY_N_SAMPLES: int = DEFAULT_SERVING_STABILITY_N_SAMPLES
    SERVING_STABILITY_SEED: int = DEFAULT_SERVING_STABILITY_SEED
    FEATURE_DISPLAY_NAMES: Dict[str, str] = {
        "N": "Nitrogen (N)",
        "P": "Phosphorus (P)",
        "K": "Potassium (K)",
        "temperature": "Temperature",
        "humidity": "Humidity",
        "ph": "Soil pH",
        "rainfall": "Rainfall",
    }

    def __init__(
        self,
        model_dir: Optional[Union[str, Path]] = None,
        scaler: Any = None,
        crop_stats: Optional[Dict[str, Dict[str, Any]]] = None,
        oof_reference_scores: Optional[np.ndarray] = None,
        thresholds: Optional[Dict[str, Any]] = None,
        pipeline: Any = None,
        classes: Optional[List[str]] = None,
        iforest_data: Optional[Dict[str, Any]] = None,
        class_means: Optional[Dict[str, Any]] = None,
        class_sds: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize TrustEngine instance.

        Args:
            model_dir: Path to directory containing model artifacts.
            scaler: Fitted StandardScaler from training data.
            crop_stats: Dict of per-crop Ledoit-Wolf means, covariances, and precision matrices.
            oof_reference_scores: Array of out-of-fold training Mahalanobis distances.
            thresholds: Calibrated risk thresholds dictionary.
            pipeline: Optional classification pipeline (Role 1).
            classes: Optional list of class names.
            iforest_data: Optional isolation forest artifact dictionary.
            class_means: Optional dict mapping crop to mean feature values.
            class_sds: Optional dict mapping crop to standard deviation feature values.
        """
        self.model_dir: Optional[Path] = Path(model_dir) if model_dir is not None else None
        self.scaler = scaler
        self.crop_stats: Dict[str, Dict[str, Any]] = crop_stats or {}
        self.oof_reference_scores: np.ndarray = (
            oof_reference_scores
            if oof_reference_scores is not None
            else np.array([], dtype=float)
        )
        self.thresholds: Dict[str, Any] = thresholds or {}
        self.pipeline = pipeline
        self.classes: List[str] = classes or []
        self.iforest_data = iforest_data
        self.class_means: Dict[str, Any] = class_means or {}
        self.class_sds: Dict[str, Any] = class_sds or {}
        self.is_stub: bool = scaler is None or not self.crop_stats

    @classmethod
    def _resolve_model_dir(cls, model_dir: Optional[Union[str, Path]]) -> Path:
        """Resolve model directory path across execution contexts."""
        if model_dir is not None:
            p = Path(model_dir)
            if p.exists() and p.is_dir():
                return p
            candidate_backend = Path("backend") / p
            if candidate_backend.exists() and candidate_backend.is_dir():
                return candidate_backend
            if p.parts and p.parts[0] == "backend":
                candidate_sub = Path(*p.parts[1:])
                if candidate_sub.exists() and candidate_sub.is_dir():
                    return candidate_sub
            raise FileNotFoundError(f"Specified model directory does not exist: {model_dir}")

        candidates = [
            Path("backend/models"),
            Path("models"),
            Path(__file__).resolve().parent / "models",
        ]
        for candidate in candidates:
            if candidate.exists() and (candidate / "ood_stats.joblib").exists():
                return candidate

        raise FileNotFoundError(
            "Could not find default model directory containing ood_stats.joblib"
        )

    @classmethod
    def load(cls, model_dir: Optional[Union[str, Path]] = None) -> "TrustEngine":
        """Load and strictly validate engine artifacts from a model directory.

        Loads:
          - ood_stats.joblib (scaler, per-crop Ledoit-Wolf covariance, OOF reference scores)
          - thresholds.json (calibrated verdict cutoffs per risk profile)
          - iforest.joblib (optional Isolation Forest artifact)
          - classes.json (mandatory list of 22 target crop names)
          - crop_pipeline_calibrated.joblib (mandatory calibrated classification pipeline)

        Args:
            model_dir: Path to the model artifacts directory. If None, auto-resolves.

        Returns:
            An initialized TrustEngine instance with verified empirical OOD detection.

        Raises:
            FileNotFoundError: If the directory or mandatory artifacts are not found.
            ValueError: If mandatory artifact schemas or contents are invalid.
        """
        resolved_dir = cls._resolve_model_dir(model_dir)

        # 1. Mandatory OOD stats
        ood_stats_path = resolved_dir / "ood_stats.joblib"
        if not ood_stats_path.exists():
            raise FileNotFoundError(f"Missing required artifact: {ood_stats_path}")

        try:
            ood_data = joblib.load(ood_stats_path)
        except Exception as exc:
            raise ValueError(f"Failed to load {ood_stats_path}: {exc}") from exc

        if not isinstance(ood_data, dict):
            raise ValueError(f"Invalid ood_stats.joblib format in {ood_stats_path}: must be dict")

        for key in ("scaler", "crop_stats", "oof_reference_scores", "metadata"):
            if key not in ood_data:
                raise ValueError(f"Artifact {ood_stats_path} missing required key '{key}'")

        scaler = ood_data["scaler"]
        crop_stats = ood_data["crop_stats"]
        raw_oof_scores = ood_data["oof_reference_scores"]

        # Validate non-empty, finite reference scores
        try:
            oof_reference_scores = np.asarray(raw_oof_scores, dtype=float)
        except (ValueError, TypeError) as exc:
            raise ValueError(
                f"Artifact {ood_stats_path} oof_reference_scores must be numeric array"
            ) from exc

        if oof_reference_scores.size == 0:
            raise ValueError(f"Artifact {ood_stats_path} contains empty oof_reference_scores")
        if not np.all(np.isfinite(oof_reference_scores)):
            raise ValueError(f"Artifact {ood_stats_path} contains NaN or Infinite oof_reference_scores")

        # Validate feature order in metadata
        metadata = ood_data.get("metadata", {})
        metadata_features = metadata.get("features", [])
        if metadata_features != list(cls.REQUIRED_FEATURES):
            raise ValueError(
                f"Feature ordering mismatch in {ood_stats_path}: "
                f"expected {list(cls.REQUIRED_FEATURES)}, got {metadata_features}"
            )

        # Validate crop statistics schema and dimensions
        if not isinstance(crop_stats, dict) or len(crop_stats) != 22:
            raise ValueError(f"Artifact {ood_stats_path} must contain 22 crops, found {len(crop_stats) if isinstance(crop_stats, dict) else type(crop_stats)}")

        for crop, stats in crop_stats.items():
            if not isinstance(stats, dict):
                raise ValueError(f"Crop stats for '{crop}' must be a dictionary")
            for req in ("mean", "cov", "inv_cov"):
                if req not in stats:
                    raise ValueError(f"Crop stats for '{crop}' missing '{req}'")
            if stats["mean"].shape != (7,):
                raise ValueError(f"Crop '{crop}' mean vector has invalid shape {stats['mean'].shape}, expected (7,)")
            if stats["cov"].shape != (7, 7) or stats["inv_cov"].shape != (7, 7):
                raise ValueError(f"Crop '{crop}' covariance matrices must have shape (7, 7)")

        # 2. Mandatory Thresholds
        thresholds_path = resolved_dir / "thresholds.json"
        if not thresholds_path.exists():
            raise FileNotFoundError(f"Missing required artifact: {thresholds_path}")

        try:
            with open(thresholds_path, "r", encoding="utf-8") as f:
                thresholds = json.load(f)
        except Exception as exc:
            raise ValueError(f"Failed to load {thresholds_path}: {exc}") from exc

        if not isinstance(thresholds, dict) or "profiles" not in thresholds:
            raise ValueError(f"Invalid thresholds.json structure in {thresholds_path}: missing 'profiles'")

        profiles = thresholds["profiles"]
        if not isinstance(profiles, dict):
            raise ValueError("thresholds.json 'profiles' must be a dictionary")

        missing_profiles = [p for p in cls.SUPPORTED_RISK_PROFILES if p not in profiles]
        if missing_profiles:
            raise ValueError(
                f"thresholds.json missing required risk profile(s): {', '.join(missing_profiles)}"
            )

        for p in cls.SUPPORTED_RISK_PROFILES:
            p_cfg = profiles[p]
            if not isinstance(p_cfg, dict):
                raise ValueError(f"Profile configuration for '{p}' must be a dictionary")
            if "ood_caution_threshold" not in p_cfg or "ood_refuse_threshold" not in p_cfg:
                raise ValueError(f"Profile '{p}' missing caution or refusal threshold")

            try:
                c_t = float(p_cfg["ood_caution_threshold"])
                r_t = float(p_cfg["ood_refuse_threshold"])
            except (ValueError, TypeError) as exc:
                raise ValueError(f"Thresholds for profile '{p}' must be numeric floats") from exc

            if not np.isfinite(c_t) or not np.isfinite(r_t) or c_t <= 0 or r_t <= 0:
                raise ValueError(f"Thresholds for profile '{p}' must be positive finite numbers")
            if c_t >= r_t:
                raise ValueError(
                    f"Profile '{p}' has invalid thresholds: caution ({c_t}) "
                    f"must be strictly less than refusal ({r_t})"
                )

        # 3. Optional Isolation Forest
        iforest_path = resolved_dir / "iforest.joblib"
        iforest_data = None
        if iforest_path.exists():
            try:
                iforest_data = joblib.load(iforest_path)
            except Exception:
                iforest_data = None

        # 4. Mandatory Target Classes
        classes_path = resolved_dir / "classes.json"
        if not classes_path.exists():
            raise FileNotFoundError(f"Missing required artifact: {classes_path}")

        try:
            with open(classes_path, "r", encoding="utf-8") as f:
                classes = json.load(f)
        except Exception as exc:
            raise ValueError(f"Failed to load {classes_path}: {exc}") from exc

        if not isinstance(classes, list) or len(classes) != 22 or not all(isinstance(c, str) for c in classes):
            raise ValueError(
                f"Artifact {classes_path} must be a list of 22 class name strings"
            )

        # 5. Mandatory Role 1 Calibrated Pipeline
        pipeline_path = resolved_dir / "crop_pipeline_calibrated.joblib"
        if not pipeline_path.exists():
            raise FileNotFoundError(
                f"Missing required calibrated model artifact: {pipeline_path}"
            )

        try:
            pipeline = joblib.load(pipeline_path)
        except Exception as exc:
            raise ValueError(f"Failed to load {pipeline_path}: {exc}") from exc

        if not hasattr(pipeline, "classes_"):
            raise ValueError(
                f"Calibrated model in {pipeline_path} is missing 'classes_' attribute"
            )
        if list(pipeline.classes_) != classes:
            raise ValueError(
                f"Calibrated model classes do not match classes.json: "
                f"expected {classes}, got {list(pipeline.classes_)}"
            )

        # Optimize inference latency on multi-core Windows environments
        # (Verified bitwise identical predictions; avoids worker pool spawning overhead)
        if hasattr(pipeline, "calibrated_classifiers_"):
            for cc in pipeline.calibrated_classifiers_:
                if hasattr(cc, "estimator"):
                    base_est = cc.estimator
                    if hasattr(base_est, "named_steps") and "clf" in base_est.named_steps:
                        base_est.named_steps["clf"].n_jobs = 1
                    elif hasattr(base_est, "n_jobs"):
                        base_est.n_jobs = 1

        # 6. Mandatory Class Statistics (Role 1)
        class_stats_path = resolved_dir / "class_stats.json"
        if not class_stats_path.exists():
            raise FileNotFoundError(f"Missing required artifact: {class_stats_path}")

        try:
            with open(class_stats_path, "r", encoding="utf-8") as f:
                class_stats_data = json.load(f)
        except Exception as exc:
            raise ValueError(f"Failed to load {class_stats_path}: {exc}") from exc

        if not isinstance(class_stats_data, dict):
            raise ValueError(
                f"Invalid class_stats.json structure in {class_stats_path}: must be dict"
            )

        for req_key in ("features", "crops", "class_means", "class_sds"):
            if req_key not in class_stats_data:
                raise ValueError(
                    f"Artifact {class_stats_path} missing required key '{req_key}'"
                )

        features_list = class_stats_data["features"]
        if features_list != list(cls.REQUIRED_FEATURES):
            raise ValueError(
                f"Feature ordering mismatch in {class_stats_path}: "
                f"expected {list(cls.REQUIRED_FEATURES)}, got {features_list}"
            )

        crops_list = class_stats_data["crops"]
        if not isinstance(crops_list, list) or set(crops_list) != set(classes):
            raise ValueError(
                f"Crop set mismatch in {class_stats_path} relative to classes.json: "
                f"expected {set(classes)}, got {set(crops_list) if isinstance(crops_list, list) else type(crops_list)}"
            )

        class_means = class_stats_data["class_means"]
        class_sds = class_stats_data["class_sds"]
        if not isinstance(class_means, dict) or not isinstance(class_sds, dict):
            raise ValueError(
                f"Artifact {class_stats_path} 'class_means' and 'class_sds' must be dictionaries"
            )

        return cls(
            model_dir=resolved_dir,
            scaler=scaler,
            crop_stats=crop_stats,
            oof_reference_scores=oof_reference_scores,
            thresholds=thresholds,
            pipeline=pipeline,
            classes=classes,
            iforest_data=iforest_data,
            class_means=class_means,
            class_sds=class_sds,
        )

    def _validate_input(
        self, row: Any, risk: str
    ) -> Tuple[Dict[str, float], str]:
        """Enforce Stage 1 input validation rules:

        1. Valid risk profile enum.
        2. Dictionary input shape.
        3. All 7 required features present with exact casing.
        4. Values are finite numbers (not booleans, strings, NaN, or Inf).
        5. Values are within physical reality bounds.
        """
        if not isinstance(risk, str):
            raise ValueError(
                f"Invalid risk profile '{risk}'. Risk profile must be a string. "
                f"Supported profiles are: {', '.join(self.SUPPORTED_RISK_PROFILES)}"
            )
        normalized_risk = risk.lower().strip()
        if normalized_risk not in self.SUPPORTED_RISK_PROFILES:
            raise ValueError(
                f"Invalid risk profile '{risk}'. "
                f"Supported profiles are: {', '.join(self.SUPPORTED_RISK_PROFILES)}"
            )

        if not isinstance(row, dict):
            raise ValueError(f"Input row must be a dictionary, got {type(row).__name__}")

        missing_features = [f for f in self.REQUIRED_FEATURES if f not in row]
        if missing_features:
            raise ValueError(
                f"Missing required feature(s): {', '.join(missing_features)}"
            )

        parsed_row: Dict[str, float] = {}
        for feature in self.REQUIRED_FEATURES:
            val = row[feature]
            if isinstance(val, bool):
                raise ValueError(
                    f"Feature '{feature}' must be numeric, got boolean {val!r}"
                )
            try:
                fval = float(val)
            except (ValueError, TypeError) as exc:
                raise ValueError(
                    f"Feature '{feature}' must be numeric, got {val!r}"
                ) from exc

            if np.isnan(fval) or np.isinf(fval):
                raise ValueError(
                    f"Feature '{feature}' must be a finite number, got {fval!r}"
                )

            low, high = self.PHYSICAL_BOUNDS[feature]
            if fval < low or fval > high:
                raise ValueError(
                    f"Feature '{feature}' value {fval} is outside physical bounds [{low}, {high}]"
                )

            parsed_row[feature] = fval

        return parsed_row, normalized_risk

    def _compute_mahalanobis(
        self, row: Dict[str, float]
    ) -> Tuple[float, List[Tuple[str, float]], np.ndarray]:
        """Compute per-crop Mahalanobis distance using Ledoit-Wolf precision matrices.

        Returns:
            min_d2: Minimum squared Mahalanobis distance across all crops.
            sorted_crops: List of (crop_name, distance_sq) sorted ascending.
            z: Scaled input feature vector.
        """
        x_vec = np.array([[row[f] for f in self.REQUIRED_FEATURES]], dtype=float)
        z = self.scaler.transform(x_vec)[0]

        crop_distances: List[Tuple[str, float]] = []
        for crop, stats in self.crop_stats.items():
            diff = z - stats["mean"]
            prec = stats["inv_cov"]
            d2 = float(diff.T @ prec @ diff)
            crop_distances.append((crop, d2))

        crop_distances.sort(key=lambda kv: kv[1])
        min_d2 = crop_distances[0][1]
        return min_d2, crop_distances, z

    def _find_unusual_pair(
        self, z: np.ndarray, nearest_crop: str
    ) -> Optional[List[str]]:
        """Identify which feature pair exhibits significant bivariate divergence.

        Evaluates 2D sub-covariance Mahalanobis distance for all 21 feature pairs relative
        to the nearest crop's distribution, and computes excess bivariate coupling:
            delta_d2 = d2_pair - (z_i^2 + z_j^2)
        where z_i, z_j are standardized marginal distances relative to the nearest crop.

        Selection criteria:
          1. Significant coupling: delta_d2 >= 0.5 (divergence beyond independent marginals).
          2. Statistical significance: d2_pair >= 5.99 (chi-square 2-dof 95th percentile).
          3. Bounded marginals: |z_i| <= 3.5 and |z_j| <= 3.5 (suppresses single univariate spikes).

        Returns:
            [feat_a, feat_b] with the highest excess coupling delta_d2, or None if no pair qualifies.
        """
        if nearest_crop not in self.crop_stats:
            return None

        stats = self.crop_stats[nearest_crop]
        cov = stats["cov"]
        mean = stats["mean"]
        std = np.sqrt(np.maximum(np.diag(cov), 1e-9))
        z_rel = (z - mean) / std

        candidates: List[Dict[str, Any]] = []
        n_feat = len(self.REQUIRED_FEATURES)

        for i in range(n_feat):
            for j in range(i + 1, n_feat):
                sub_cov = cov[np.ix_([i, j], [i, j])]
                try:
                    inv_sub = np.linalg.inv(sub_cov)
                except np.linalg.LinAlgError:
                    continue

                sub_diff = z[[i, j]] - mean[[i, j]]
                d2_pair = float(sub_diff.T @ inv_sub @ sub_diff)
                indep_d2 = float(z_rel[i] ** 2 + z_rel[j] ** 2)
                delta_d2 = d2_pair - indep_d2

                bounded = (abs(z_rel[i]) <= 3.5) and (abs(z_rel[j]) <= 3.5)
                candidates.append({
                    "pair": [self.REQUIRED_FEATURES[i], self.REQUIRED_FEATURES[j]],
                    "d2": d2_pair,
                    "delta_d2": delta_d2,
                    "bounded": bounded,
                })

        # Priority 1: Bounded marginals with significant coupling and chi-square significance
        bounded_candidates = [
            c for c in candidates
            if c["bounded"] and c["d2"] >= 5.99 and c["delta_d2"] >= 0.5
        ]
        if bounded_candidates:
            bounded_candidates.sort(key=lambda c: c["delta_d2"], reverse=True)
            return bounded_candidates[0]["pair"]

        # Priority 2: Unbounded pairs with significant coupling and chi-square significance
        unbounded_candidates = [
            c for c in candidates
            if c["d2"] >= 5.99 and c["delta_d2"] >= 0.5
        ]
        if unbounded_candidates:
            unbounded_candidates.sort(key=lambda c: c["delta_d2"], reverse=True)
            return unbounded_candidates[0]["pair"]

        return None

    def assess_ood(
        self, row: Dict[str, Any], risk: str = "balanced"
    ) -> Dict[str, Any]:
        """Pure Stage 2 OOD assessment method.

        Performs:
          Stage 1: Validation (schema, finite numeric types, physical bounds).
          Stage 2: Empirical OOD gate (per-crop Ledoit-Wolf Mahalanobis distance,
                   OOF percentile rank, risk profile threshold comparison,
                   excess coupling attribution).

        Does NOT:
          - Query the uncalibrated placeholder Random Forest.
          - Fabricate crop probabilities, stability scores, or trust scores.
          - Synthesize final Stage 5 verdicts.

        Args:
            row: Dictionary containing the 7 crop features (N, P, K, temperature,
                 humidity, ph, rainfall).
            risk: Risk appetite profile ('conservative', 'balanced', or 'bold').

        Returns:
            Dict containing pure Stage 2 diagnostics:
              - eligible: bool (True if min_distance <= refusal threshold, eligible for Stages 3-5)
              - flagged: bool (True strictly when min_distance > refusal threshold)
              - percentile: float (empirical rank relative to 1,540 training OOF distances)
              - min_distance: float (minimum squared Mahalanobis distance across 22 crops)
              - threshold: float (refusal threshold for the selected risk profile)
              - caution_threshold: float (caution threshold for the selected risk profile)
              - nearest_known: list of 2 nearest crop names [c1, c2]
              - unusual_pair: [feat_a, feat_b] if flagged and coupled, else None
              - risk_profile: str (normalized risk profile)

        Raises:
            ValueError: If input fails Stage 1 validation or risk profile is unsupported.
        """
        parsed_row, normalized_risk = self._validate_input(row, risk)

        min_d2, sorted_crops, z = self._compute_mahalanobis(parsed_row)

        percentile = float(np.mean(self.oof_reference_scores <= min_d2) * 100.0)
        percentile = round(percentile, 1)

        profile_cfg = self.thresholds["profiles"][normalized_risk]
        ood_refuse_threshold = float(profile_cfg["ood_refuse_threshold"])
        ood_caution_threshold = float(profile_cfg["ood_caution_threshold"])

        flagged = bool(min_d2 > ood_refuse_threshold)
        eligible = not flagged

        nearest_crops = [sorted_crops[0][0], sorted_crops[1][0]]
        unusual_pair = self._find_unusual_pair(z, nearest_crops[0]) if flagged else None

        return {
            "eligible": eligible,
            "flagged": flagged,
            "percentile": percentile,
            "min_distance": round(min_d2, 4),
            "threshold": ood_refuse_threshold,
            "caution_threshold": ood_caution_threshold,
            "nearest_known": nearest_crops,
            "unusual_pair": unusual_pair,
            "risk_profile": normalized_risk,
        }

    def assess_probabilities(self, row: Dict[str, Any]) -> Dict[str, Any]:
        """Stage 3: Calibrated Prediction.

        Evaluates the input features through the calibrated classification model pipeline
        (Isotonic CalibratedClassifierCV wrapping ExtraTreesClassifier).

        Args:
            row: Dictionary containing the 7 canonical features (N, P, K,
                 temperature, humidity, ph, rainfall).

        Returns:
            Dict containing Stage 3 predictions and confidence metrics:
              - top_crop: str (crop name with highest calibrated probability)
              - top_probability: float (calibrated probability of top crop)
              - second_crop: str (crop name with second highest calibrated probability)
              - second_probability: float (calibrated probability of second crop)
              - margin: float (difference between top and second probabilities)
              - top_crops: list of dicts with top 3 crops, e.g. [{"crop": str, "probability": float}]

        Raises:
            RuntimeError: If calibrated model pipeline or classes are not loaded.
            ValueError: If input is invalid, features missing, non-numeric, non-finite,
                        or model outputs invalid probabilities (shape, NaN/inf, negative, sum != 1.0).
        """
        if self.pipeline is None or not self.classes:
            raise RuntimeError("Calibrated model pipeline or classes not loaded")

        if not isinstance(row, dict):
            raise ValueError(f"Input row must be a dictionary, got {type(row).__name__}")

        missing = [f for f in self.REQUIRED_FEATURES if f not in row]
        if missing:
            raise ValueError(f"Missing required feature(s): {', '.join(missing)}")

        feature_values = []
        for feat in self.REQUIRED_FEATURES:
            val = row[feat]
            if isinstance(val, bool):
                raise ValueError(f"Feature '{feat}' must be numeric, got boolean {val!r}")
            try:
                fval = float(val)
            except (ValueError, TypeError) as exc:
                raise ValueError(f"Feature '{feat}' must be numeric, got {val!r}") from exc
            if not np.isfinite(fval):
                raise ValueError(f"Feature '{feat}' must be a finite number, got {fval!r}")
            feature_values.append(fval)

        df_in = pd.DataFrame([feature_values], columns=list(self.REQUIRED_FEATURES))
        probs_2d = self.pipeline.predict_proba(df_in)

        if not isinstance(probs_2d, np.ndarray) or probs_2d.shape != (1, len(self.classes)):
            raise ValueError(
                f"Model predict_proba returned invalid shape {getattr(probs_2d, 'shape', None)}, "
                f"expected (1, {len(self.classes)})"
            )

        probs = probs_2d[0]
        if not np.all(np.isfinite(probs)):
            raise ValueError("Model predicted NaN or Infinite probability values")
        if np.any(probs < 0.0):
            raise ValueError("Model predicted negative probability values")

        prob_sum = float(np.sum(probs))
        if abs(prob_sum - 1.0) > 1e-4:
            raise ValueError(
                f"Model probabilities do not sum to 1.0 (sum={prob_sum:.6f}, tolerance=1e-4)"
            )

        paired = [(float(probs[i]), self.classes[i]) for i in range(len(self.classes))]
        paired.sort(key=lambda x: (-x[0], x[1]))

        top_prob, top_crop = paired[0]
        second_prob, second_crop = paired[1]
        margin = float(top_prob - second_prob)

        top_crops = [
            {"crop": crop, "probability": round(prob, 4)}
            for prob, crop in paired[:3]
        ]

        return {
            "top_crop": top_crop,
            "top_probability": top_prob,
            "second_crop": second_crop,
            "second_probability": second_prob,
            "margin": margin,
            "top_crops": top_crops,
        }

    def assess_stability(
        self,
        row: Dict[str, Any],
        clean_crop: str,
        n_samples: int = 100,
        seed: Optional[int] = 42,
    ) -> Dict[str, Any]:
        """Stage 4: Stability Assessment.

        Evaluates prediction stability under physical measurement noise using the
        approved primary 1-sigma noise profile and physical boundary clipping.

        Calculates the fraction of noisy perturbations whose prediction matches clean_crop:
          stability = matches / n_samples
          flip_rate = 1.0 - stability

        Args:
            row: Dictionary containing the 7 canonical features (N, P, K,
                 temperature, humidity, ph, rainfall).
            clean_crop: The predicted crop label from Stage 3 (top_crop).
            n_samples: Number of noise perturbations to evaluate (default: 100).
            seed: Random seed for reproducible perturbations (default: 42).
                  If None, stochastic perturbation is used.

        Returns:
            Dict containing Stage 4 stability metrics:
              - stability: float (fraction of perturbations matching clean_crop, in [0.0, 1.0])
              - flip_rate: float (fraction of perturbations that flipped away, 1.0 - stability)
              - clean_crop: str (the reference crop evaluated)
              - n_samples: int (total perturbation count evaluated)
              - matches: int (count of perturbations matching clean_crop)

        Raises:
            RuntimeError: If calibrated model pipeline or classes are not loaded.
            ValueError: If clean_crop is invalid/not in classes, n_samples is not a positive integer,
                        row features are missing/non-numeric/non-finite, or model returns
                        invalid prediction shape or unknown labels.
        """
        if self.pipeline is None or not self.classes:
            raise RuntimeError("Calibrated model pipeline or classes not loaded")

        if not isinstance(clean_crop, str) or clean_crop not in self.classes:
            raise ValueError(
                f"Invalid clean_crop {clean_crop!r}. Must be a string belonging to known classes: {self.classes}"
            )

        if isinstance(n_samples, bool) or not isinstance(n_samples, int) or n_samples <= 0:
            raise ValueError(
                f"n_samples must be a positive integer, got {n_samples!r} ({type(n_samples).__name__})"
            )

        if seed is not None and (isinstance(seed, bool) or not isinstance(seed, (int, np.integer))):
            raise ValueError(f"seed must be None or an integer, got {seed!r}")

        if not isinstance(row, dict):
            raise ValueError(f"Input row must be a dictionary, got {type(row).__name__}")

        missing = [f for f in self.REQUIRED_FEATURES if f not in row]
        if missing:
            raise ValueError(f"Missing required feature(s): {', '.join(missing)}")

        feature_values = []
        for feat in self.REQUIRED_FEATURES:
            val = row[feat]
            if isinstance(val, bool):
                raise ValueError(f"Feature '{feat}' must be numeric, got boolean {val!r}")
            try:
                fval = float(val)
            except (ValueError, TypeError) as exc:
                raise ValueError(f"Feature '{feat}' must be numeric, got {val!r}") from exc
            if not np.isfinite(fval):
                raise ValueError(f"Feature '{feat}' must be a finite number, got {fval!r}")
            feature_values.append(fval)

        noisy_arr = apply_noise(
            feature_values,
            n_samples=n_samples,
            seed=seed,
            profile="primary_1sigma",
        )

        if not isinstance(noisy_arr, np.ndarray) or noisy_arr.shape != (n_samples, len(self.REQUIRED_FEATURES)):
            raise ValueError(
                f"Noise perturbation generator returned invalid shape {getattr(noisy_arr, 'shape', None)}, "
                f"expected ({n_samples}, {len(self.REQUIRED_FEATURES)})"
            )

        df_noisy = pd.DataFrame(noisy_arr, columns=list(self.REQUIRED_FEATURES))
        noisy_preds = self.pipeline.predict(df_noisy)

        if not isinstance(noisy_preds, np.ndarray) or noisy_preds.shape != (n_samples,):
            raise ValueError(
                f"Model predict returned invalid shape {getattr(noisy_preds, 'shape', None)}, "
                f"expected ({n_samples},)"
            )

        unknown_labels = set(noisy_preds) - set(self.classes)
        if unknown_labels:
            raise ValueError(
                f"Model predicted unknown crop labels not in classes: {sorted(list(unknown_labels))}"
            )

        matches = int(np.sum(noisy_preds == clean_crop))
        stability = float(matches / n_samples)
        flip_rate = float(1.0 - stability)

        return {
            "stability": stability,
            "flip_rate": flip_rate,
            "clean_crop": clean_crop,
            "n_samples": n_samples,
            "matches": matches,
        }

    def measure_next_feature(
        self, crop_a: str, crop_b: str
    ) -> Tuple[Optional[str], float]:
        """Compute the single feature that best separates two competing crops.

        Follows Handbook Section 7.10:
          pooled = sqrt((sd_a^2 + sd_b^2) / 2)
          gap = abs(mean_a - mean_b) / pooled
        Keeps the largest gap. Strict '>' ensures the earlier feature in
        REQUIRED_FEATURES order wins ties. Any feature whose pooled standard
        deviation is <= 0 or non-finite is skipped.

        Args:
            crop_a: Primary / top crop label.
            crop_b: Competing / second crop label.

        Returns:
            Tuple of (feature_name, gap) or (None, 0.0) if no feature qualifies.

        Raises:
            ValueError: If crop_a == crop_b or either crop is not in classes/class_stats.
            RuntimeError: If class statistics are not loaded.
        """
        if not self.class_means or not self.class_sds:
            raise RuntimeError("Class statistics not loaded in TrustEngine")

        if crop_a == crop_b:
            raise ValueError(
                f"Crops must be distinct to compare, got crop_a={crop_a!r} and crop_b={crop_b!r}"
            )

        if crop_a not in self.class_means or crop_a not in self.class_sds:
            raise ValueError(f"Unknown crop: {crop_a!r}")
        if crop_b not in self.class_means or crop_b not in self.class_sds:
            raise ValueError(f"Unknown crop: {crop_b!r}")

        best_name: Optional[str] = None
        best_gap: float = 0.0

        for j, feat in enumerate(self.REQUIRED_FEATURES):
            if isinstance(self.class_means[crop_a], dict):
                mean_a = float(self.class_means[crop_a][feat])
                mean_b = float(self.class_means[crop_b][feat])
            else:
                mean_a = float(self.class_means[crop_a][j])
                mean_b = float(self.class_means[crop_b][j])

            if isinstance(self.class_sds[crop_a], dict):
                sd_a = float(self.class_sds[crop_a][feat])
                sd_b = float(self.class_sds[crop_b][feat])
            else:
                sd_a = float(self.class_sds[crop_a][j])
                sd_b = float(self.class_sds[crop_b][j])

            pooled_var = (sd_a ** 2 + sd_b ** 2) / 2.0
            pooled = math.sqrt(pooled_var)

            if pooled <= 0.0 or not math.isfinite(pooled):
                continue

            gap = abs(mean_a - mean_b) / pooled
            if gap > best_gap:
                best_gap = float(gap)
                best_name = feat

        if best_name is None:
            return (None, 0.0)
        return (best_name, float(best_gap))

    def assess(self, row: Dict[str, Any], risk: str = "balanced") -> Dict[str, Any]:
        """Assess input row through the full 5-stage Trust Engine.

        Executes:
          Stage 1: Validation (schema, types, physical bounds).
          Stage 2: Empirical OOD gate via assess_ood().
          Stage 3: Calibrated crop probabilities via assess_probabilities().
          Stage 4: Stability assessment under physical noise via assess_stability().
          Stage 5: Verdict determination (recommend / caution / refuse) & trust score.

        Handling:
          - If Stage 2 rejects the input (flagged=True): Returns the standard HTTP 200
            refusal response matching docs/api-contract.md with explicit nulls.
            Stage 3 and Stage 4 are not called.
          - If Stage 2 accepts the input:
            Evaluates probabilities, margin, and stability against the selected risk
            profile thresholds. Emits 'caution' if any trigger fires, otherwise 'recommend'.
            Calculates calibrated trust_score in [0, 100], capped at 60 on caution.

        Args:
            row: Dictionary containing the 7 crop features.
            risk: Risk appetite profile ('conservative', 'balanced', or 'bold').

        Returns:
            Dict matching the frozen API contract with all 9 required keys:
            verdict, top_crops, conformal_set, ood, stability, trust_score,
            explanation, model_version, measure_next.

        Raises:
            RuntimeError: If engine model artifacts are not loaded.
            ValueError: If input is invalid, features missing/out of bounds, or risk unsupported.
        """
        if (
            self.pipeline is None
            or not self.classes
            or self.scaler is None
            or not self.crop_stats
            or not self.thresholds
        ):
            raise RuntimeError("TrustEngine model artifacts not loaded")

        # Stage 1 Validation & Stage 2 OOD assessment
        ood_res = self.assess_ood(row, risk)

        if ood_res["flagged"]:
            nearest_crops = ood_res["nearest_known"]
            unusual_pair = ood_res["unusual_pair"]
            min_d2 = ood_res["min_distance"]
            percentile = ood_res["percentile"]

            if unusual_pair:
                feat_a, feat_b = unusual_pair
                val_a = float(row[feat_a])
                val_b = float(row[feat_b])
                explanation = (
                    f"Input is out-of-distribution (anomaly score {min_d2:.2f}, percentile {percentile:.1f}%). "
                    f"Features '{feat_a}' ({val_a:.1f}) and '{feat_b}' ({val_b:.1f}) exhibit high bivariate divergence "
                    f"relative to {nearest_crops[0]} patterns. No recommendation is made."
                )
            else:
                explanation = (
                    f"Input is out-of-distribution (anomaly score {min_d2:.2f}, percentile {percentile:.1f}%). "
                    f"Environmental and soil conditions differ substantially from known training data. "
                    f"No recommendation is made."
                )

            return {
                "verdict": "refuse",
                "top_crops": [],
                "conformal_set": None,
                "ood": {
                    "flagged": True,
                    "percentile": percentile,
                    "unusual_pair": unusual_pair,
                    "nearest_known": nearest_crops,
                },
                "stability": None,
                "trust_score": 0,
                "measure_next": None,
                "model_version": self.MODEL_VERSION,
                "explanation": explanation,
            }

        # Stage 3: Calibrated Probabilities
        prob_res = self.assess_probabilities(row)
        top_crop = prob_res["top_crop"]
        top_prob = float(prob_res["top_probability"])
        second_crop = prob_res["second_crop"]
        margin = float(prob_res["margin"])
        top_crops = prob_res["top_crops"]

        # Stage 4: Stability Assessment under measurement noise
        stab_res = self.assess_stability(
            row,
            top_crop,
            n_samples=self.SERVING_STABILITY_N_SAMPLES,
            seed=self.SERVING_STABILITY_SEED,
        )
        stability = float(stab_res["stability"])

        # Stage 5: Selective Classification & Thresholds
        normalized_risk = ood_res["risk_profile"]
        profile_cfg = self.thresholds["profiles"][normalized_risk]
        ood_caution_threshold = float(profile_cfg["ood_caution_threshold"])
        min_prob = float(profile_cfg["min_prob"])
        min_margin = float(profile_cfg["min_margin"])
        min_stability = float(profile_cfg["min_stability"])

        min_d2 = float(ood_res["min_distance"])
        percentile = float(ood_res["percentile"])

        # Collect triggers with strict inequalities (exact equality does NOT trigger)
        triggers: List[str] = []
        if min_d2 > ood_caution_threshold:
            triggers.append("ood_caution")
        if top_prob < min_prob:
            triggers.append("low_probability")
        if margin < min_margin:
            triggers.append("low_margin")
        if stability < min_stability:
            triggers.append("low_stability")

        measure_next: Optional[str] = None
        if triggers:
            verdict = "caution"
            reasons: List[str] = []
            if "low_stability" in triggers:
                reasons.append(
                    f"Predictions are sensitive to simulated measurement variations (stability: {stability * 100.0:.1f}%)."
                )
            if "low_probability" in triggers:
                reasons.append(
                    f"Model confidence is low ({top_prob * 100.0:.1f}%)."
                )
            if "low_margin" in triggers:
                reasons.append(
                    f"Margin between top two crops is narrow ({margin * 100.0:.1f}%)."
                )
            if "ood_caution" in triggers:
                reasons.append(
                    f"Input is more unusual than about {percentile:.0f}% of familiar training inputs."
                )
            explanation = " ".join(reasons)

            if "low_margin" in triggers:
                best_feat, _ = self.measure_next_feature(top_crop, second_crop)
                if best_feat is not None:
                    display_name = self.FEATURE_DISPLAY_NAMES.get(best_feat, best_feat)
                    measure_next = (
                        f"{display_name}: it separates {top_crop} from {second_crop} "
                        f"most strongly. Re-test it."
                    )
        else:
            verdict = "recommend"
            explanation = (
                f"High confidence ({top_prob * 100.0:.1f}%) with high stability "
                f"({stability * 100.0:.1f}%) under simulated measurement variations."
            )

        # trust_score: round half up, capped at 60 on caution
        score = int(math.floor(100.0 * min(top_prob, stability) + 0.5))  # round half up
        if verdict == "caution":
            score = min(score, 60)
        score = max(0, min(100, score))

        return {
            "verdict": verdict,
            "top_crops": top_crops,
            "conformal_set": None,
            "ood": {
                "flagged": ood_res["flagged"],
                "percentile": percentile,
                "unusual_pair": None,
                "nearest_known": None,
            },
            "stability": stability,
            "trust_score": score,
            "explanation": explanation,
            "model_version": self.MODEL_VERSION,
            "measure_next": measure_next,
        }

    def _evaluate_candidate_batch(
        self, cand_matrix: np.ndarray
    ) -> Tuple[np.ndarray, List[str]]:
        """Evaluate a batch of candidate feature vectors through Stage 2 and Stage 3.

        Computes:
          1. Empirical per-crop Mahalanobis distance squared using Ledoit-Wolf precision
             matrices, reproducing _compute_mahalanobis bitwise across all candidates.
          2. Top predicted crop per candidate using the calibrated pipeline with exact
             tie-breaking (descending calibrated probability, then alphabetically ascending
             crop name), reproducing assess_probabilities.

        Args:
            cand_matrix: ndarray of shape (N, 7) in REQUIRED_FEATURES column order.

        Returns:
            Tuple of:
              - min_d2_all: ndarray of shape (N,) with minimum d^2 across crops.
              - cand_top_crops: list of length N with the top predicted crop name per candidate.
        """
        if self.scaler is None or not self.crop_stats:
            raise RuntimeError("Stage 2 OOD detector assets (scaler, crop_stats) not loaded")
        if self.pipeline is None or not self.classes:
            raise RuntimeError("Stage 3 Calibrated model assets (pipeline, classes) not loaded")

        # 1. Vectorized Stage 2 Mahalanobis distance
        z_batch = self.scaler.transform(cand_matrix)
        min_d2_all = np.full(cand_matrix.shape[0], np.inf, dtype=np.float64)
        for crop, stats in self.crop_stats.items():
            diff = z_batch - stats["mean"]
            inv_cov = stats["inv_cov"]
            d2_crop = np.sum((diff @ inv_cov) * diff, axis=1)
            min_d2_all = np.minimum(min_d2_all, d2_crop)

        # 2. Vectorized Stage 3 Predict Proba
        cand_df = pd.DataFrame(cand_matrix, columns=list(self.REQUIRED_FEATURES))
        probs_2d = self.pipeline.predict_proba(cand_df)

        cand_top_crops: List[str] = []
        classes = self.classes
        for i in range(cand_matrix.shape[0]):
            probs = probs_2d[i]
            paired = [(float(probs[c]), classes[c]) for c in range(len(classes))]
            paired.sort(key=lambda x: (-x[0], x[1]))
            cand_top_crops.append(paired[0][1])

        return min_d2_all, cand_top_crops

    def counterfactual_report(
        self, row: Dict[str, Any], risk: str = "balanced"
    ) -> Dict[str, Any]:
        """Generate counterfactual / sensitivity analysis for an input row.

        Evaluates decision margins (for Caution / Recommend inputs) or flag-clearing
        thresholds (for Refuse inputs) along each of the 7 feature dimensions.

        For each feature in REQUIRED_FEATURES:
          - Step size = 0.25 * sigma_j (sigma from the noise model's primary_1sigma profile
            evaluated on the original row).
          - Steps k = 1..60 in both 'up' (+k) and 'down' (-k) directions (up to 15.0 sigma).
          - Candidates qualify if:
              (i) strictly within physical bounds for the feature;
              (ii) familiar under Stage 2 (min_d2 <= ood_refuse_threshold);
              (iii) for caution/recommend: top crop differs from base assessment top crop;
                    for refuse: candidate is no longer refused (familiar under Stage 2).
          - Smallest k is reported. Tie-break: if both 'up' and 'down' qualify at the
            exact same step k, 'up' is chosen first by design.

        Args:
            row: Input dictionary containing 7 canonical features.
            risk: Risk profile ('conservative', 'balanced', 'bold'). Default 'balanced'.

        Returns:
            Dict containing:
              - risk: normalized risk profile name
              - verdict: base assessment verdict ('recommend', 'caution', 'refuse')
              - near_tie: bool (margin < profile min_margin; False on refuse)
              - step_fraction: 0.25
              - max_steps: 60
              - decision_margins: list of 7 dicts in FEATURES order (None on refuse)
              - flag_clearing: list of 7 dicts in FEATURES order (None on caution/recommend)
              - messages: list of model-sensitivity/familiarity strings (one per non-null item)
              - note: "These describe the model's sensitivity, not agronomic advice."
        """
        parsed_row, normalized_risk = self._validate_input(row, risk)
        base_res = self.assess(row, risk=normalized_risk)
        verdict = base_res["verdict"]

        profile_cfg = self.thresholds["profiles"][normalized_risk]
        ood_refuse_threshold = float(profile_cfg["ood_refuse_threshold"])
        min_margin = float(profile_cfg["min_margin"])

        if verdict != "refuse":
            orig_top_crop = base_res["top_crops"][0]["crop"]
            orig_margin = float(
                base_res["top_crops"][0]["probability"] - base_res["top_crops"][1]["probability"]
            )
            near_tie = bool(orig_margin < min_margin)
        else:
            orig_top_crop = None
            near_tie = False

        # Load noise configuration and compute 1-sigma per feature for original row
        noise_cfg_all = load_noise_config(
            self.model_dir / "noise_model.json" if self.model_dir else None
        )
        profile_noise_cfg = noise_cfg_all["profiles"]["primary_1sigma"]
        row_vec = np.array([parsed_row[f] for f in self.REQUIRED_FEATURES], dtype=float)
        sigmas_arr = compute_feature_sigmas(row_vec, profile_noise_cfg)
        sigmas = {f: float(sigmas_arr[i]) for i, f in enumerate(self.REQUIRED_FEATURES)}

        step_fraction = 0.25
        max_steps = 60

        # Build candidate matrix: 7 features x 2 directions x 60 steps = 840 candidates
        candidates_list = []
        metadata = []

        for feat_idx, feat in enumerate(self.REQUIRED_FEATURES):
            base_val = parsed_row[feat]
            feat_sigma = sigmas[feat]
            step = step_fraction * feat_sigma
            min_b, max_b = self.PHYSICAL_BOUNDS[feat]

            for dir_name, dir_sign in [("up", +1), ("down", -1)]:
                for k in range(1, max_steps + 1):
                    val = base_val + dir_sign * k * step
                    in_bounds = bool(min_b <= val <= max_b)

                    cand_vec = list(row_vec)
                    cand_vec[feat_idx] = val
                    candidates_list.append(cand_vec)

                    metadata.append({
                        "feat_idx": feat_idx,
                        "feat": feat,
                        "dir_name": dir_name,
                        "dir_sign": dir_sign,
                        "k": k,
                        "from_value": round(base_val, 4),
                        "to_value": round(val, 4),
                        "delta": round(dir_sign * k * step, 4),
                        "delta_sigma": round(dir_sign * k * step_fraction, 4),
                        "in_bounds": in_bounds,
                    })

        cand_matrix = np.array(candidates_list, dtype=float)

        # Single batched call for all 840 candidates
        min_d2_all, cand_top_crops = self._evaluate_candidate_batch(cand_matrix)

        # Lookup index by (feat, k, dir_name)
        lookup = {
            (m["feat"], m["k"], m["dir_name"]): idx
            for idx, m in enumerate(metadata)
        }

        # Display units mapping
        units: Dict[str, str] = {
            "temperature": "°C",
            "humidity": "%",
            "rainfall": " mm",
            "N": "",
            "P": "",
            "K": "",
            "ph": "",
        }

        messages: List[str] = []
        note = "These describe the model's sensitivity, not agronomic advice."

        if verdict in ("caution", "recommend"):
            decision_margins: Optional[List[Dict[str, Any]]] = []
            flag_clearing: Optional[List[Dict[str, Any]]] = None

            for feat in self.REQUIRED_FEATURES:
                base_val = parsed_row[feat]
                unit = units.get(feat, "")
                feat_disp = self.FEATURE_DISPLAY_NAMES.get(feat, feat)

                found_qualifying: Optional[Dict[str, Any]] = None

                # Search k = 1..max_steps. At each k, test "up" then "down" ("up" wins tie)
                for k in range(1, max_steps + 1):
                    for dir_name in ("up", "down"):
                        idx = lookup[(feat, k, dir_name)]
                        m = metadata[idx]

                        in_b = m["in_bounds"]
                        not_refused = bool(min_d2_all[idx] <= ood_refuse_threshold)
                        crop_differs = bool(cand_top_crops[idx] != orig_top_crop)

                        if in_b and not_refused and crop_differs:
                            found_qualifying = {
                                "feature": feat,
                                "direction": m["dir_name"],
                                "delta": m["delta"],
                                "delta_sigma": m["delta_sigma"],
                                "from_value": m["from_value"],
                                "to_value": m["to_value"],
                                "resulting_crop": cand_top_crops[idx],
                            }
                            break
                    if found_qualifying:
                        break

                if found_qualifying:
                    decision_margins.append(found_qualifying)
                    from_str = f"{found_qualifying['from_value']:.2f}{unit}"
                    to_str = f"{found_qualifying['to_value']:.2f}{unit}"
                    msg = (
                        f"Model sensitivity: the top crop would change from {orig_top_crop} to "
                        f"{found_qualifying['resulting_crop']} if {feat_disp} were {to_str} instead of {from_str}."
                    )
                    messages.append(msg)
                else:
                    decision_margins.append({
                        "feature": feat,
                        "direction": None,
                        "delta": None,
                        "delta_sigma": None,
                        "from_value": round(base_val, 4),
                        "to_value": None,
                        "resulting_crop": None,
                        "searched_sigma": 15.0,
                    })
        else:  # verdict == "refuse"
            decision_margins = None
            flag_clearing = []

            for feat in self.REQUIRED_FEATURES:
                base_val = parsed_row[feat]
                unit = units.get(feat, "")
                feat_disp = self.FEATURE_DISPLAY_NAMES.get(feat, feat)

                found_clearing: Optional[Dict[str, Any]] = None

                # Search k = 1..max_steps. At each k, test "up" then "down" ("up" wins tie)
                for k in range(1, max_steps + 1):
                    for dir_name in ("up", "down"):
                        idx = lookup[(feat, k, dir_name)]
                        m = metadata[idx]

                        in_b = m["in_bounds"]
                        not_refused = bool(min_d2_all[idx] <= ood_refuse_threshold)

                        if in_b and not_refused:
                            found_clearing = {
                                "feature": feat,
                                "direction": m["dir_name"],
                                "delta": m["delta"],
                                "delta_sigma": m["delta_sigma"],
                                "from_value": m["from_value"],
                                "to_value": m["to_value"],
                                "new_d2": round(float(min_d2_all[idx]), 4),
                                "top_crop_after": cand_top_crops[idx],
                            }
                            break
                    if found_clearing:
                        break

                if found_clearing:
                    flag_clearing.append(found_clearing)
                    from_str = f"{found_clearing['from_value']:.2f}{unit}"
                    to_str = f"{found_clearing['to_value']:.2f}{unit}"
                    msg = (
                        f"Model familiarity: this input would no longer be treated as unfamiliar if "
                        f"{feat_disp} were {to_str} instead of {from_str}."
                    )
                    messages.append(msg)
                else:
                    flag_clearing.append({
                        "feature": feat,
                        "direction": None,
                        "delta": None,
                        "delta_sigma": None,
                        "from_value": round(base_val, 4),
                        "to_value": None,
                        "new_d2": None,
                        "top_crop_after": None,
                        "searched_sigma": 15.0,
                    })

        return {
            "risk": normalized_risk,
            "verdict": verdict,
            "near_tie": near_tie,
            "step_fraction": step_fraction,
            "max_steps": max_steps,
            "decision_margins": decision_margins,
            "flag_clearing": flag_clearing,
            "messages": messages,
            "note": note,
        }
