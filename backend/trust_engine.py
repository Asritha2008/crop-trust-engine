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
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np


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
          - crop_pipeline.joblib and classes.json (optional Role 1 artifacts)

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

        # 4. Optional Role 1 Baseline Pipeline & Classes
        pipeline_path = resolved_dir / "crop_pipeline.joblib"
        pipeline = None
        if pipeline_path.exists():
            try:
                pipeline = joblib.load(pipeline_path)
            except Exception:
                pipeline = None

        classes_path = resolved_dir / "classes.json"
        classes = None
        if classes_path.exists():
            try:
                with open(classes_path, "r", encoding="utf-8") as f:
                    classes = json.load(f)
            except Exception:
                classes = None

        return cls(
            model_dir=resolved_dir,
            scaler=scaler,
            crop_stats=crop_stats,
            oof_reference_scores=oof_reference_scores,
            thresholds=thresholds,
            pipeline=pipeline,
            classes=classes,
            iforest_data=iforest_data,
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

    def assess(self, row: Dict[str, Any], risk: str = "balanced") -> Dict[str, Any]:
        """Assess input row through the Trust Engine.

        Executes:
          Stage 1: Validation.
          Stage 2: Empirical OOD gate via assess_ood().

        Handling:
          - If Stage 2 rejects the input (flagged=True): Returns the standard HTTP 200
            refusal response matching docs/api-contract.md with explicit nulls.
          - If Stage 2 accepts the input (eligible=True): Raises StagesNotImplementedError
            so Role 3's FastAPI handler maps it to HTTP 501 with payload:
            {"error": "Stages 3-5 not integrated yet"}.
            No fake probabilities, stability, or trust scores are emitted.

        Args:
            row: Dictionary containing the 7 crop features.
            risk: Risk appetite profile ('conservative', 'balanced', or 'bold').

        Returns:
            Dict matching the frozen API contract for refusal.

        Raises:
            StagesNotImplementedError: When input passes Stage 2 but Stages 3-5 are pending.
            ValueError: If input is invalid, features missing/out of bounds, or risk unsupported.
        """
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

        # Input is In-Distribution (passed Stage 2 OOD gate)
        # As agreed with Role 3, hand off cleanly via StagesNotImplementedError
        # mapping to HTTP 501 {"error": "Stages 3-5 not integrated yet"}
        raise StagesNotImplementedError(
            "Stages 3-5 not integrated yet",
            ood_result=ood_res,
        )
