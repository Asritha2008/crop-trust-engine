"""Crop Trust Engine - Core Engine Interface and Stub Implementation.

Role 2: Trust Engine Engineer
Initial Phase 1 stub implementation providing the frozen contract interface and
deterministic mock fixtures for Role 3 (Backend FastAPI) and Role 4 (Frontend UI).

NOTE: All numerical thresholds, percentiles, and probabilities in this file are
temporary illustrative fixtures for software integration only. Real empirical
detectors (Mahalanobis, Isolation Forest) and calibrated thresholds will be
implemented in subsequent tasks using clean training data splits.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union


class TrustEngine:
    """The Trust Engine coordinates validation, OOD detection, confidence

    calibration, stability checks, and selective classification verdicts.

    This class provides the official interface:
      - TrustEngine.load(model_dir): Load artifacts and initialize the engine.
      - assess(row, risk): Assess a single crop input row under a risk profile.
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
    MODEL_VERSION: str = "stub-2026-11-07"

    # =========================================================================
    # TEMPORARY INTEGRATION FIXTURES (PHASE 1 STUB)
    #
    # WARNING: The conditions and numerical cutoffs below are purely illustrative
    # mock triggers designed to return the 3 contract responses (recommend,
    # caution, refuse) deterministically so Role 3 (FastAPI) and Role 4 (UI) can
    # integrate and test their software components immediately.
    #
    # They DO NOT represent trained machine learning models, empirical OOD
    # boundaries (Mahalanobis / Isolation Forest), or calibrated thresholds.
    # Real detectors and thresholds will be fitted in Task 3 and Task 5 using
    # strictly non-leaking training data and out-of-fold cross-validation.
    # =========================================================================
    _MOCK_REFUSE_HUMIDITY_MIN: float = 85.0
    _MOCK_REFUSE_RAINFALL_MAX: float = 35.0
    _MOCK_REFUSE_NITROGEN_MAX: float = 175.0

    _MOCK_CAUTION_TEMP_MIN: float = 32.0
    _MOCK_CAUTION_RAINFALL_MIN: float = 150.0
    _MOCK_CAUTION_PH_MAX: float = 7.5
    _MOCK_CONSERVATIVE_RAINFALL_THRESHOLD: float = 70.0

    def __init__(self, model_dir: Optional[Union[str, Path]] = None) -> None:
        """Initialize TrustEngine instance.

        Args:
            model_dir: Path to directory containing model artifacts.
        """
        self.model_dir: Optional[Path] = Path(model_dir) if model_dir is not None else None
        # Explicit flag indicating this is the Phase 1 development stub
        self.is_stub: bool = True

    @classmethod
    def load(cls, model_dir: Optional[Union[str, Path]] = "models/") -> "TrustEngine":
        """Load engine artifacts from a model directory.

        In the production implementation, this loads:
          - crop_pipeline.joblib (scaler + classifier + calibrator from Role 1)
          - ood_stats.joblib (per-crop covariance and class means)
          - thresholds.json (calibrated verdict cutoffs per risk profile)
          - noise_model.json (agreed measurement uncertainty)

        In this stub implementation, it gracefully initializes without requiring
        saved artifact files on disk, unblocking backend and frontend integration.

        Args:
            model_dir: Path to the model artifacts directory.

        Returns:
            An initialized TrustEngine instance.
        """
        resolved_path = Path(model_dir) if model_dir is not None else None
        return cls(model_dir=resolved_path)

    def assess(self, row: Dict[str, Any], risk: str = "balanced") -> Dict[str, Any]:
        """Assess a single input row and return a structured verdict and metrics.

        Args:
            row: Dictionary containing the 7 crop features (N, P, K, temperature,
                 humidity, ph, rainfall).
            risk: Risk appetite profile ('conservative', 'balanced', or 'bold').

        Returns:
            Dict matching the frozen API contract:
              - verdict: 'recommend' | 'caution' | 'refuse'
              - top_crops: list of dicts with 'crop' and 'probability' (empty on refuse)
              - ood: dict with 'flagged', 'percentile', and optional details
              - stability: float (0..1) or None on refuse
              - trust_score: int (0..100)
              - explanation: human-readable rationale
              - model_version: version identifier string
              - optional: conformal_set, measure_next

        Raises:
            ValueError: If the risk profile is invalid, input is not a dict,
                        or required features are missing / non-numeric.
        """
        # Validate risk profile
        normalized_risk = risk.lower().strip() if isinstance(risk, str) else ""
        if normalized_risk not in self.SUPPORTED_RISK_PROFILES:
            raise ValueError(
                f"Invalid risk profile '{risk}'. "
                f"Supported profiles are: {', '.join(self.SUPPORTED_RISK_PROFILES)}"
            )

        # Validate input features
        if not isinstance(row, dict):
            raise ValueError(f"Input row must be a dictionary, got {type(row).__name__}")

        missing_features = [f for f in self.REQUIRED_FEATURES if f not in row]
        if missing_features:
            raise ValueError(
                f"Missing required feature(s): {', '.join(missing_features)}"
            )

        # Validate feature types
        parsed_row: Dict[str, float] = {}
        for feature in self.REQUIRED_FEATURES:
            val = row[feature]
            try:
                parsed_row[feature] = float(val)
            except (ValueError, TypeError) as exc:
                raise ValueError(
                    f"Feature '{feature}' must be numeric, got {val!r}"
                ) from exc

        # Generate deterministic response from the stub logic
        return self._generate_stub_response(parsed_row, normalized_risk)

    def _generate_stub_response(
        self, row: Dict[str, float], risk: str
    ) -> Dict[str, Any]:
        """Generate deterministic mock responses representing the three verdicts.

        IMPORTANT: These are temporary integration fixtures, NOT empirical ML predictions.

        Routing logic for software integration:
          1. Refusal case fixture:
             - An illustrative impossible feature combination (e.g. humidity > 85% with
               rainfall < 35mm, reflecting Section 5.3 & 7.5 examples).
             - Or extreme nitrogen (N > 175).
          2. Caution case fixture:
             - Illustrative boundary conditions (temperature > 32°C and rainfall > 150mm,
               or pH > 7.5, or conservative risk with low rainfall < 70mm).
          3. Recommend case fixture:
             - Illustrative familiar conditions (all other valid inputs).
        """
        humidity = row["humidity"]
        rainfall = row["rainfall"]
        nitrogen = row["N"]
        temperature = row["temperature"]
        ph = row["ph"]

        # Fixture 1: OOD Refusal Mock Response
        is_ood_mock = (
            (humidity > self._MOCK_REFUSE_HUMIDITY_MIN and rainfall < self._MOCK_REFUSE_RAINFALL_MAX)
            or (nitrogen > self._MOCK_REFUSE_NITROGEN_MAX)
        )
        if is_ood_mock:
            return {
                "verdict": "refuse",
                "top_crops": [],
                "ood": {
                    "flagged": True,
                    "percentile": 99.8,  # Illustrative stub percentile
                    "unusual_pair": (
                        ["humidity", "rainfall"]
                        if (humidity > self._MOCK_REFUSE_HUMIDITY_MIN and rainfall < self._MOCK_REFUSE_RAINFALL_MAX)
                        else ["N", "rainfall"]
                    ),
                    "nearest_known": ["rice", "jute"],
                },
                "stability": None,
                "trust_score": 0,
                "model_version": self.MODEL_VERSION,
                "explanation": (
                    f"[STUB / ILLUSTRATIVE] Humidity {humidity:.1f}% and rainfall {rainfall:.1f} mm "
                    "are individually plausible, but this combination does not appear in the training data. "
                    "No recommendation is made."
                ),
            }

        # Fixture 2: Caution Mock Response (Ambiguity / Close Margin)
        is_caution_mock = (
            (temperature > self._MOCK_CAUTION_TEMP_MIN and rainfall > self._MOCK_CAUTION_RAINFALL_MIN)
            or (ph > self._MOCK_CAUTION_PH_MAX)
            or (risk == "conservative" and rainfall < self._MOCK_CONSERVATIVE_RAINFALL_THRESHOLD)
        )
        if is_caution_mock:
            top_prob = 0.52 if risk != "bold" else 0.58
            second_prob = 0.41 if risk != "bold" else 0.35
            third_prob = round(1.0 - top_prob - second_prob, 2)

            return {
                "verdict": "caution",
                "top_crops": [
                    {"crop": "rice", "probability": top_prob},
                    {"crop": "jute", "probability": second_prob},
                    {"crop": "maize", "probability": third_prob},
                ],
                "conformal_set": ["rice", "jute"],
                "ood": {
                    "flagged": False,
                    "percentile": 63.0,  # Illustrative stub percentile
                },
                "stability": 0.71,      # Illustrative stub stability
                "trust_score": 52,       # Illustrative stub score
                "measure_next": "Potassium (K): it separates rice from jute most strongly. Re-test it.",
                "model_version": self.MODEL_VERSION,
                "explanation": (
                    "[STUB / ILLUSTRATIVE] Two crops are close (rice and jute) and a small change "
                    "in the inputs can flip the answer. Consider verifying potassium levels before planting."
                ),
            }

        # Fixture 3: Recommend Mock Response (Familiar Conditions)
        return {
            "verdict": "recommend",
            "top_crops": [
                {"crop": "rice", "probability": 0.91},
                {"crop": "jute", "probability": 0.05},
                {"crop": "maize", "probability": 0.02},
            ],
            "ood": {
                "flagged": False,
                "percentile": 41.0,  # Illustrative stub percentile
            },
            "stability": 0.94,      # Illustrative stub stability
            "trust_score": 88,       # Illustrative stub score
            "model_version": self.MODEL_VERSION,
            "explanation": (
                "[STUB / ILLUSTRATIVE] Inputs match known rice conditions and are stable under small changes."
            ),
        }
