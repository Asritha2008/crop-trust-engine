"""Unit tests for TrustEngine core interface and Stage 2 OOD implementation.

Role 2: Trust Engine Engineer
Tests run directly via Python unittest or pytest without requiring FastAPI
or a running web server.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import tempfile
from typing import Any, Dict
import unittest
from unittest.mock import MagicMock, patch

import joblib
import numpy as np

# Ensure robust imports whether test is invoked from repo root or inside backend/
try:
    from backend.trust_engine import StagesNotImplementedError, TrustEngine
except ModuleNotFoundError:
    import sys

    backend_dir = Path(__file__).resolve().parent.parent
    if str(backend_dir) not in sys.path:
        sys.path.insert(0, str(backend_dir))
    from trust_engine import StagesNotImplementedError, TrustEngine


class TestTrustEngineStage2OOD(unittest.TestCase):
    """Comprehensive test suite for TrustEngine Stage 2 empirical OOD detection."""

    def setUp(self) -> None:
        """Initialize engine and canonical test inputs."""
        base_dir = Path(__file__).resolve().parent.parent
        model_candidates = [
            base_dir / "models",
            Path("backend/models"),
            Path("models"),
        ]
        self.model_dir = next(
            (p for p in model_candidates if (p / "ood_stats.joblib").exists()),
            None,
        )
        self.assertIsNotNone(
            self.model_dir,
            "Could not locate valid model directory containing ood_stats.joblib",
        )
        self.engine = TrustEngine.load(self.model_dir)

        # 1. Familiar in-distribution input (Rice-like centroid conditions)
        self.rice_familiar: Dict[str, Any] = {
            "N": 90.0,
            "P": 42.0,
            "K": 43.0,
            "temperature": 21.0,
            "humidity": 82.0,
            "ph": 6.5,
            "rainfall": 203.0,
        }

        # 2. Canonical unusual combination: rice conditions with high humidity and low rainfall
        self.rice_unusual_pair: Dict[str, Any] = {
            "N": 80.0,
            "P": 47.0,
            "K": 40.0,
            "temperature": 23.7,
            "humidity": 86.0,
            "ph": 6.4,
            "rainfall": 125.0,
        }

        # 3. Midpoint trap coordinates
        # Rice-Jute midpoint (overlapping agronomic cluster)
        self.midpoint_rice_jute: Dict[str, Any] = {
            "N": 79.145,
            "P": 47.22,
            "K": 39.93,
            "temperature": 24.32,
            "humidity": 80.96,
            "ph": 6.58,
            "rainfall": 205.49,
        }

        # Lentil-Chickpea midpoint (empty void between distinct legume clusters)
        self.midpoint_lentil_chickpea: Dict[str, Any] = {
            "N": 29.43,
            "P": 68.08,
            "K": 49.67,
            "temperature": 21.69,
            "humidity": 40.83,
            "ph": 7.13,
            "rainfall": 62.87,
        }

        # 4. Precision boundary coordinates synthesized along rice covariance eigenvectors
        # Rice precision matrix eigenvector direction
        stats = self.engine.crop_stats["rice"]
        mean = stats["mean"]
        prec = stats["inv_cov"]
        eigvals, eigvecs = np.linalg.eigh(prec)
        v = eigvecs[:, 0]
        lam = eigvals[0]

        def _make_pt(target_d2: float) -> Dict[str, float]:
            alpha = np.sqrt(target_d2 / lam)
            z = mean + alpha * v
            x = self.engine.scaler.inverse_transform(z.reshape(1, -1))[0]
            return dict(zip(self.engine.REQUIRED_FEATURES, x))

        self.pt_below_caution = _make_pt(10.75)
        self.pt_exact_caution = _make_pt(10.78)
        self.pt_above_caution = _make_pt(10.82)
        self.pt_below_refusal = _make_pt(12.85)
        self.pt_exact_refusal = _make_pt(12.92)
        self.pt_above_refusal = _make_pt(12.98)
        self.pt_high_refusal = _make_pt(13.50)
        self.pt_cross_profile_boundary = _make_pt(12.00)

    # =========================================================================
    # 1. Pure Stage 2 OOD Method: assess_ood()
    # =========================================================================
    def test_assess_ood_familiar_input(self) -> None:
        """Verify assess_ood() returns pure Stage 2 diagnostics for familiar inputs."""
        res = self.engine.assess_ood(self.rice_familiar, risk="balanced")

        self.assertTrue(res["eligible"])
        self.assertFalse(res["flagged"])
        self.assertLess(res["percentile"], 50.0)
        self.assertLess(res["min_distance"], 10.78)
        self.assertEqual(res["threshold"], 12.92)
        self.assertEqual(res["caution_threshold"], 10.78)
        self.assertEqual(res["nearest_known"][0], "rice")
        self.assertIsNone(res["unusual_pair"])
        self.assertEqual(res["risk_profile"], "balanced")

        # Crucial check: pure Stage 2 must NOT contain downstream fields
        self.assertNotIn("top_crops", res)
        self.assertNotIn("probability", res)
        self.assertNotIn("stability", res)
        self.assertNotIn("trust_score", res)
        self.assertNotIn("verdict", res)

    def test_assess_ood_unusual_input(self) -> None:
        """Verify assess_ood() flags anomalous inputs and isolates bivariate coupling."""
        res = self.engine.assess_ood(self.rice_unusual_pair, risk="balanced")

        self.assertFalse(res["eligible"])
        self.assertTrue(res["flagged"])
        self.assertGreaterEqual(res["percentile"], 99.0)
        self.assertGreater(res["min_distance"], 12.92)
        self.assertEqual(res["nearest_known"], ["rice", "jute"])
        self.assertEqual(res["unusual_pair"], ["humidity", "rainfall"])

    # =========================================================================
    # 2. Contract Endpoint assess() Behavior & Clean Handoff
    # =========================================================================
    def test_assess_rejected_input_returns_refusal_response(self) -> None:
        """Verify assess() returns the frozen API contract refusal dict on OOD rejection."""
        res = self.engine.assess(self.rice_unusual_pair, risk="balanced")

        self.assertEqual(res["verdict"], "refuse")
        self.assertEqual(res["top_crops"], [])
        self.assertIsNone(res["conformal_set"])
        self.assertIsNone(res["stability"])
        self.assertEqual(res["trust_score"], 0)
        self.assertIsNone(res["measure_next"])

        self.assertTrue(res["ood"]["flagged"])
        self.assertGreaterEqual(res["ood"]["percentile"], 99.0)
        self.assertEqual(res["ood"]["unusual_pair"], ["humidity", "rainfall"])
        self.assertEqual(res["ood"]["nearest_known"], ["rice", "jute"])

        self.assertIn("out-of-distribution", res["explanation"])
        self.assertIn("humidity", res["explanation"])
        self.assertIn("rainfall", res["explanation"])
        self.assertEqual(res["model_version"], self.engine.MODEL_VERSION)

    def test_assess_accepted_input_raises_stages_not_implemented(self) -> None:
        """Verify assess() returns a complete Stage 5 response for accepted inputs."""
        res = self.engine.assess(self.rice_familiar, risk="balanced")
        self.assertIn(res["verdict"], {"recommend", "caution"})
        self.assertEqual(len(res["top_crops"]), 3)
        self.assertEqual(res["top_crops"][0]["crop"], "rice")
        self.assertGreaterEqual(res["top_crops"][0]["probability"], 0.80)
        self.assertIsNone(res["conformal_set"])
        self.assertIsNone(res["measure_next"])
        self.assertFalse(res["ood"]["flagged"])
        self.assertIsNone(res["ood"]["unusual_pair"])
        self.assertIsNone(res["ood"]["nearest_known"])
        self.assertIsInstance(res["stability"], float)
        self.assertIsInstance(res["trust_score"], int)
        self.assertTrue(0 <= res["trust_score"] <= 100)
        self.assertEqual(res["model_version"], self.engine.MODEL_VERSION)
        self.assertIn("simulated measurement variations", res["explanation"])

    # =========================================================================
    # 3. Exact Threshold Boundaries & Profile Sensitivity
    # =========================================================================
    def test_threshold_equality_and_boundaries(self) -> None:
        """Verify exact boundary behavior and strict inequality cutoffs."""
        # Just below caution (10.75 < 10.78)
        res_below_c = self.engine.assess_ood(self.pt_below_caution, risk="balanced")
        self.assertFalse(res_below_c["flagged"])
        self.assertTrue(res_below_c["eligible"])
        self.assertLess(res_below_c["min_distance"], 10.78)

        # Exact caution threshold (10.78000)
        res_exact_c = self.engine.assess_ood(self.pt_exact_caution, risk="balanced")
        self.assertFalse(res_exact_c["flagged"])
        self.assertTrue(res_exact_c["eligible"])
        # Strict inequality: min_d2 > 10.78 is False at exact boundary
        self.assertFalse(res_exact_c["min_distance"] > 10.78001)

        # Just above caution (10.82 > 10.78), but below refusal
        res_above_c = self.engine.assess_ood(self.pt_above_caution, risk="balanced")
        self.assertFalse(res_above_c["flagged"])
        self.assertTrue(res_above_c["eligible"])
        self.assertGreater(res_above_c["min_distance"], 10.78)

        # Just below refusal (12.85 < 12.92)
        res_below_r = self.engine.assess_ood(self.pt_below_refusal, risk="balanced")
        self.assertFalse(res_below_r["flagged"])
        self.assertTrue(res_below_r["eligible"])

        # Exact refusal threshold (12.92000)
        # Strict inequality min_d2 > 12.92 means exact boundary is NOT flagged
        res_exact_r = self.engine.assess_ood(self.pt_exact_refusal, risk="balanced")
        self.assertFalse(res_exact_r["flagged"])
        self.assertTrue(res_exact_r["eligible"])

        # Just above refusal (12.98 > 12.92) -> FLAGGED
        res_above_r = self.engine.assess_ood(self.pt_above_refusal, risk="balanced")
        self.assertTrue(res_above_r["flagged"])
        self.assertFalse(res_above_r["eligible"])

    def test_cross_profile_sensitivity(self) -> None:
        """Verify risk profiles alter refusal threshold appropriately for a borderline input."""
        # pt_cross_profile_boundary has d2 = 12.00
        # Conservative threshold: refuse at 11.71 -> triggers refusal
        res_cons = self.engine.assess_ood(self.pt_cross_profile_boundary, risk="conservative")
        self.assertTrue(res_cons["flagged"])
        self.assertFalse(res_cons["eligible"])

        # Balanced threshold: refuse at 12.92 -> does not refuse
        res_bal = self.engine.assess_ood(self.pt_cross_profile_boundary, risk="balanced")
        self.assertFalse(res_bal["flagged"])
        self.assertTrue(res_bal["eligible"])

        # Bold threshold: refuse at 13.75 -> does not refuse
        res_bold = self.engine.assess_ood(self.pt_cross_profile_boundary, risk="bold")
        self.assertFalse(res_bold["flagged"])
        self.assertTrue(res_bold["eligible"])

    # =========================================================================
    # 4. Unusual-Pair Attribution & Honest Null Fallback
    # =========================================================================
    def test_unusual_pair_honest_null_when_no_coupling(self) -> None:
        """Verify unusual_pair returns None when an anomaly lacks strong bivariate coupling."""
        # pt_high_refusal is a 7D eigenvector shift with no 2D broken correlation
        res = self.engine.assess_ood(self.pt_high_refusal, risk="balanced")
        self.assertTrue(res["flagged"])
        self.assertIsNone(res["unusual_pair"])

        # Full refusal response must carry explicit null
        refuse_dict = self.engine.assess(self.pt_high_refusal, risk="balanced")
        self.assertIsNone(refuse_dict["ood"]["unusual_pair"])
        self.assertIn("differ substantially from known training data", refuse_dict["explanation"])

    # =========================================================================
    # 5. Midpoint Traps
    # =========================================================================
    def test_midpoint_behavior(self) -> None:
        """Verify rice-jute midpoint passes while lentil-chickpea midpoint triggers refusal."""
        # Rice-Jute midpoint (overlapping density)
        res_rj = self.engine.assess_ood(self.midpoint_rice_jute, risk="balanced")
        self.assertTrue(res_rj["eligible"])
        self.assertFalse(res_rj["flagged"])
        self.assertLess(res_rj["percentile"], 10.0)

        # Lentil-Chickpea midpoint (empty void)
        res_lc = self.engine.assess_ood(self.midpoint_lentil_chickpea, risk="balanced")
        self.assertFalse(res_lc["eligible"])
        self.assertTrue(res_lc["flagged"])
        self.assertGreaterEqual(res_lc["percentile"], 99.5)

    # =========================================================================
    # 6. Strict Artifact Validation on load()
    # =========================================================================
    def test_artifact_validation_failures(self) -> None:
        """Verify load() rejects missing, corrupt, or invalid artifacts with clear errors."""
        # Nonexistent directory
        with self.assertRaises(FileNotFoundError):
            TrustEngine.load("nonexistent_directory_xyz")

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            # Missing ood_stats.joblib
            with self.assertRaises(FileNotFoundError) as ctx:
                TrustEngine.load(tmp_path)
            self.assertIn("ood_stats.joblib", str(ctx.exception))

            # Corrupt ood_stats.joblib
            (tmp_path / "ood_stats.joblib").write_bytes(b"corrupt bytes")
            with self.assertRaises(ValueError):
                TrustEngine.load(tmp_path)

            # Empty oof_reference_scores
            valid_ood = joblib.load(self.model_dir / "ood_stats.joblib")
            corrupt_ood = dict(valid_ood)
            corrupt_ood["oof_reference_scores"] = np.array([])
            joblib.dump(corrupt_ood, tmp_path / "ood_stats.joblib")
            with self.assertRaises(ValueError) as ctx:
                TrustEngine.load(tmp_path)
            self.assertIn("empty oof_reference_scores", str(ctx.exception))

            # Non-finite oof_reference_scores
            corrupt_ood["oof_reference_scores"] = np.array([1.0, float("nan"), 3.0])
            joblib.dump(corrupt_ood, tmp_path / "ood_stats.joblib")
            with self.assertRaises(ValueError) as ctx:
                TrustEngine.load(tmp_path)
            self.assertIn("NaN or Infinite", str(ctx.exception))

            # Feature ordering mismatch
            corrupt_ood["oof_reference_scores"] = valid_ood["oof_reference_scores"]
            corrupt_ood["metadata"] = dict(valid_ood["metadata"])
            corrupt_ood["metadata"]["features"] = ["rainfall", "ph", "N", "P", "K", "temperature", "humidity"]
            joblib.dump(corrupt_ood, tmp_path / "ood_stats.joblib")
            with self.assertRaises(ValueError) as ctx:
                TrustEngine.load(tmp_path)
            self.assertIn("Feature ordering mismatch", str(ctx.exception))

            # Restore valid ood_stats.joblib to test thresholds.json validation
            joblib.dump(valid_ood, tmp_path / "ood_stats.joblib")

            # Missing thresholds.json
            with self.assertRaises(FileNotFoundError) as ctx:
                TrustEngine.load(tmp_path)
            self.assertIn("thresholds.json", str(ctx.exception))

            # Missing risk profile in thresholds.json
            with open(self.model_dir / "thresholds.json", "r") as f:
                valid_thresh = json.load(f)
            bad_thresh = dict(valid_thresh)
            bad_thresh["profiles"] = {"balanced": dict(valid_thresh["profiles"]["balanced"])}
            with open(tmp_path / "thresholds.json", "w") as f:
                json.dump(bad_thresh, f)
            with self.assertRaises(ValueError) as ctx:
                TrustEngine.load(tmp_path)
            self.assertIn("missing required risk profile", str(ctx.exception))

            # Invalid thresholds (caution >= refusal)
            bad_thresh["profiles"] = dict(valid_thresh["profiles"])
            bad_thresh["profiles"]["conservative"] = {
                "ood_caution_threshold": 15.0,
                "ood_refuse_threshold": 10.0,
            }
            with open(tmp_path / "thresholds.json", "w") as f:
                json.dump(bad_thresh, f)
            with self.assertRaises(ValueError) as ctx:
                TrustEngine.load(tmp_path)
            self.assertIn("must be strictly less than refusal", str(ctx.exception))

    # =========================================================================
    # 7. Input Validation, Types, and Physical Bounds
    # =========================================================================
    def test_input_validation_missing_and_types(self) -> None:
        """Verify missing features, non-dict, non-numeric, and booleans raise ValueError."""
        # Non-dict
        with self.assertRaises(ValueError) as ctx:
            self.engine.assess_ood([90, 42, 43, 21, 82, 6.5, 203])  # type: ignore
        self.assertIn("must be a dictionary", str(ctx.exception))

        # Missing features
        incomplete = {"N": 90.0, "P": 42.0}
        with self.assertRaises(ValueError) as ctx:
            self.engine.assess_ood(incomplete)
        self.assertIn("Missing required feature(s)", str(ctx.exception))

        # Non-numeric string
        str_val = dict(self.rice_familiar)
        str_val["ph"] = "neutral"
        with self.assertRaises(ValueError) as ctx:
            self.engine.assess_ood(str_val)
        self.assertIn("must be numeric", str(ctx.exception))

        # Boolean (bool is subclass of int in Python)
        bool_val = dict(self.rice_familiar)
        bool_val["humidity"] = True
        with self.assertRaises(ValueError) as ctx:
            self.engine.assess_ood(bool_val)
        self.assertIn("must be numeric, got boolean", str(ctx.exception))

        # NaN and Inf
        nan_val = dict(self.rice_familiar)
        nan_val["temperature"] = float("nan")
        with self.assertRaises(ValueError) as ctx:
            self.engine.assess_ood(nan_val)
        self.assertIn("finite number", str(ctx.exception))

        inf_val = dict(self.rice_familiar)
        inf_val["rainfall"] = float("inf")
        with self.assertRaises(ValueError) as ctx:
            self.engine.assess_ood(inf_val)
        self.assertIn("finite number", str(ctx.exception))

    def test_input_validation_physical_bounds(self) -> None:
        """Verify all 14 physical boundary violation cases raise ValueError."""
        bound_cases = [
            ("N", -1.0),
            ("N", 201.0),
            ("P", -5.0),
            ("P", 250.0),
            ("K", -0.1),
            ("K", 305.0),
            ("temperature", -15.0),
            ("temperature", 65.0),
            ("humidity", -1.0),
            ("humidity", 101.0),
            ("ph", -0.5),
            ("ph", 14.5),
            ("rainfall", -10.0),
            ("rainfall", 1001.0),
        ]
        for feature, val in bound_cases:
            bad_row = dict(self.rice_familiar)
            bad_row[feature] = val
            with self.assertRaises(ValueError) as ctx:
                self.engine.assess_ood(bad_row)
            self.assertIn("outside physical bounds", str(ctx.exception))

    def test_risk_profiles_supported_and_unsupported(self) -> None:
        """Verify supported profiles succeed and invalid profiles raise ValueError."""
        for p in ("conservative", "balanced", "bold"):
            res = self.engine.assess_ood(self.rice_familiar, risk=p)
            self.assertEqual(res["risk_profile"], p)

        for invalid in ("aggressive", "none", "", 123, None):
            with self.assertRaises(ValueError) as ctx:
                self.engine.assess_ood(self.rice_familiar, risk=invalid)  # type: ignore
            self.assertIn("Invalid risk profile", str(ctx.exception))

    # =========================================================================
    # 8. Determinism and JSON Serializability
    # =========================================================================
    def test_deterministic_assessment(self) -> None:
        """Verify repeated calls with identical inputs return identical results."""
        res1 = self.engine.assess_ood(self.rice_unusual_pair, risk="balanced")
        res2 = self.engine.assess_ood(self.rice_unusual_pair, risk="balanced")
        self.assertEqual(res1, res2)

    def test_json_serializability(self) -> None:
        """Verify refusal response serializes cleanly via json.dumps()."""
        refusal_res = self.engine.assess(self.rice_unusual_pair, risk="balanced")
        encoded = json.dumps(refusal_res)
        self.assertIsInstance(encoded, str)
        decoded = json.loads(encoded)
        self.assertEqual(decoded["verdict"], "refuse")
        self.assertEqual(decoded["top_crops"], [])


class _DummyPredictProbaPipeline:
    """Mock pipeline helper for Stage 3 validation unit tests."""

    def __init__(self, probas: np.ndarray, classes: List[str]) -> None:
        self.probas = probas
        self.classes_ = np.array(classes)

    def predict_proba(self, X: Any) -> np.ndarray:
        return self.probas


class TestTrustEngineStage3Probabilities(unittest.TestCase):
    """Unit test suite for TrustEngine Stage 3 Calibrated Prediction."""

    def setUp(self) -> None:
        """Initialize engine with real calibrated model."""
        base_dir = Path(__file__).resolve().parent.parent
        model_candidates = [
            base_dir / "models",
            Path("backend/models"),
            Path("models"),
        ]
        self.model_dir = next(
            (p for p in model_candidates if (p / "crop_pipeline_calibrated.joblib").exists()),
            None,
        )
        self.assertIsNotNone(
            self.model_dir,
            "Could not locate valid model directory containing crop_pipeline_calibrated.joblib",
        )
        self.engine = TrustEngine.load(self.model_dir)

        # Standard familiar in-distribution input
        self.rice_row: Dict[str, Any] = {
            "N": 90.0,
            "P": 42.0,
            "K": 43.0,
            "temperature": 21.0,
            "humidity": 82.0,
            "ph": 6.5,
            "rainfall": 203.0,
        }

        # Canonical unusual combination (OOD)
        self.rice_unusual_pair: Dict[str, Any] = {
            "N": 80.0,
            "P": 47.0,
            "K": 40.0,
            "temperature": 23.7,
            "humidity": 86.0,
            "ph": 6.4,
            "rainfall": 125.0,
        }

    def test_real_calibrated_inference_invariants(self) -> None:
        """Verify real calibrated inference satisfies probability invariants and ranking."""
        res = self.engine.assess_probabilities(self.rice_row)

        for required_key in (
            "top_crop",
            "top_probability",
            "second_crop",
            "second_probability",
            "margin",
            "top_crops",
        ):
            self.assertIn(required_key, res)

        top_crop = res["top_crop"]
        top_prob = res["top_probability"]
        second_crop = res["second_crop"]
        second_prob = res["second_probability"]
        margin = res["margin"]
        top_crops = res["top_crops"]

        # Membership and type checks
        self.assertIn(top_crop, self.engine.classes)
        self.assertIn(second_crop, self.engine.classes)
        self.assertNotEqual(top_crop, second_crop)

        # Probability range invariants
        self.assertIsInstance(top_prob, float)
        self.assertIsInstance(second_prob, float)
        self.assertGreaterEqual(top_prob, 0.0)
        self.assertLessEqual(top_prob, 1.0)
        self.assertGreaterEqual(second_prob, 0.0)
        self.assertLessEqual(second_prob, 1.0)
        self.assertGreaterEqual(top_prob, second_prob)

        # Margin invariant
        self.assertIsInstance(margin, float)
        self.assertGreaterEqual(margin, 0.0)
        self.assertAlmostEqual(margin, top_prob - second_prob, places=6)

        # Top 3 formatting and ranking
        self.assertEqual(len(top_crops), 3)
        self.assertEqual(top_crops[0]["crop"], top_crop)
        self.assertAlmostEqual(top_crops[0]["probability"], round(top_prob, 4), places=4)
        self.assertEqual(top_crops[1]["crop"], second_crop)
        self.assertAlmostEqual(top_crops[1]["probability"], round(second_prob, 4), places=4)

        # Non-increasing order in top_crops
        self.assertGreaterEqual(top_crops[0]["probability"], top_crops[1]["probability"])
        self.assertGreaterEqual(top_crops[1]["probability"], top_crops[2]["probability"])

        # Top crops does not expose all 22 classes
        self.assertLess(len(top_crops), len(self.engine.classes))

    def test_top_three_descending_and_alphabetical_tie_breaking(self) -> None:
        """Verify deterministic descending probability ordering and alphabetical tie-breaking."""
        classes = sorted(self.engine.classes)
        # Construct synthetic probabilities with exact ties
        # E.g. index of 'apple' and 'banana'
        idx_apple = classes.index("apple")
        idx_banana = classes.index("banana")
        idx_chickpea = classes.index("chickpea")

        probas = np.zeros((1, len(classes)), dtype=float)
        probas[0, idx_banana] = 0.40
        probas[0, idx_apple] = 0.40  # Exact tie with banana
        probas[0, idx_chickpea] = 0.20
        # Sum = 1.0

        mock_pipe = _DummyPredictProbaPipeline(probas, classes)
        engine_mock = TrustEngine(
            scaler=self.engine.scaler,
            crop_stats=self.engine.crop_stats,
            oof_reference_scores=self.engine.oof_reference_scores,
            thresholds=self.engine.thresholds,
            pipeline=mock_pipe,
            classes=classes,
        )

        res = engine_mock.assess_probabilities(self.rice_row)

        # Alphabetical tie break: 'apple' < 'banana'
        self.assertEqual(res["top_crop"], "apple")
        self.assertEqual(res["second_crop"], "banana")
        self.assertEqual(res["margin"], 0.0)
        self.assertEqual(res["top_crops"][0]["crop"], "apple")
        self.assertEqual(res["top_crops"][1]["crop"], "banana")
        self.assertEqual(res["top_crops"][2]["crop"], "chickpea")

    def test_feature_order_robustness(self) -> None:
        """Verify feature ordering in input dict does not alter prediction."""
        permuted_row = {
            "rainfall": 203.0,
            "ph": 6.5,
            "N": 90.0,
            "humidity": 82.0,
            "temperature": 21.0,
            "K": 43.0,
            "P": 42.0,
        }
        res_std = self.engine.assess_probabilities(self.rice_row)
        res_perm = self.engine.assess_probabilities(permuted_row)
        self.assertEqual(res_std, res_perm)

    def test_invalid_predict_proba_shape(self) -> None:
        """Verify ValueError is raised if predict_proba returns invalid shape."""
        # Shape (1, 10) instead of (1, 22)
        mock_pipe = _DummyPredictProbaPipeline(np.ones((1, 10)) / 10.0, self.engine.classes)
        engine_mock = TrustEngine(pipeline=mock_pipe, classes=self.engine.classes)
        with self.assertRaises(ValueError) as ctx:
            engine_mock.assess_probabilities(self.rice_row)
        self.assertIn("invalid shape", str(ctx.exception))

        # 1D array instead of 2D
        mock_pipe_1d = _DummyPredictProbaPipeline(np.ones(22) / 22.0, self.engine.classes)
        engine_mock_1d = TrustEngine(pipeline=mock_pipe_1d, classes=self.engine.classes)
        with self.assertRaises(ValueError) as ctx:
            engine_mock_1d.assess_probabilities(self.rice_row)
        self.assertIn("invalid shape", str(ctx.exception))

    def test_non_finite_probabilities(self) -> None:
        """Verify ValueError is raised if predict_proba contains NaN or Infinite."""
        # NaN
        probas_nan = np.zeros((1, 22))
        probas_nan[0, 0] = np.nan
        mock_pipe_nan = _DummyPredictProbaPipeline(probas_nan, self.engine.classes)
        engine_mock_nan = TrustEngine(pipeline=mock_pipe_nan, classes=self.engine.classes)
        with self.assertRaises(ValueError) as ctx:
            engine_mock_nan.assess_probabilities(self.rice_row)
        self.assertIn("NaN or Infinite", str(ctx.exception))

        # Inf
        probas_inf = np.zeros((1, 22))
        probas_inf[0, 0] = np.inf
        mock_pipe_inf = _DummyPredictProbaPipeline(probas_inf, self.engine.classes)
        engine_mock_inf = TrustEngine(pipeline=mock_pipe_inf, classes=self.engine.classes)
        with self.assertRaises(ValueError) as ctx:
            engine_mock_inf.assess_probabilities(self.rice_row)
        self.assertIn("NaN or Infinite", str(ctx.exception))

    def test_negative_probabilities(self) -> None:
        """Verify ValueError is raised if predict_proba contains negative values."""
        probas_neg = np.zeros((1, 22))
        probas_neg[0, 0] = -0.1
        probas_neg[0, 1] = 1.1
        mock_pipe_neg = _DummyPredictProbaPipeline(probas_neg, self.engine.classes)
        engine_mock_neg = TrustEngine(pipeline=mock_pipe_neg, classes=self.engine.classes)
        with self.assertRaises(ValueError) as ctx:
            engine_mock_neg.assess_probabilities(self.rice_row)
        self.assertIn("negative probability values", str(ctx.exception))

    def test_probability_sum_outside_tolerance(self) -> None:
        """Verify ValueError is raised if predict_proba sum differs from 1.0 by more than 1e-4."""
        # Sum = 1.05
        probas_bad_sum = np.ones((1, 22)) * (1.05 / 22.0)
        mock_pipe_sum = _DummyPredictProbaPipeline(probas_bad_sum, self.engine.classes)
        engine_mock_sum = TrustEngine(pipeline=mock_pipe_sum, classes=self.engine.classes)
        with self.assertRaises(ValueError) as ctx:
            engine_mock_sum.assess_probabilities(self.rice_row)
        self.assertIn("do not sum to 1.0", str(ctx.exception))

    def test_missing_and_invalid_input_types(self) -> None:
        """Verify input validation handles missing features, non-dict, and bad types."""
        # Missing feature
        incomplete_row = dict(self.rice_row)
        del incomplete_row["rainfall"]
        with self.assertRaises(ValueError) as ctx:
            self.engine.assess_probabilities(incomplete_row)
        self.assertIn("Missing required feature(s)", str(ctx.exception))

        # Non-dict
        with self.assertRaises(ValueError) as ctx:
            self.engine.assess_probabilities([1, 2, 3])  # type: ignore
        self.assertIn("must be a dictionary", str(ctx.exception))

        # Boolean
        bool_row = dict(self.rice_row)
        bool_row["ph"] = True
        with self.assertRaises(ValueError) as ctx:
            self.engine.assess_probabilities(bool_row)
        self.assertIn("must be numeric, got boolean", str(ctx.exception))

        # Non-numeric string
        str_row = dict(self.rice_row)
        str_row["humidity"] = "humid"
        with self.assertRaises(ValueError) as ctx:
            self.engine.assess_probabilities(str_row)
        self.assertIn("must be numeric", str(ctx.exception))

    def test_missing_calibrated_model_raises_filenotfound(self) -> None:
        """Verify load() raises FileNotFoundError if crop_pipeline_calibrated.joblib is absent."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            # Copy all prerequisites except crop_pipeline_calibrated.joblib
            joblib.dump(joblib.load(self.model_dir / "ood_stats.joblib"), tmp_path / "ood_stats.joblib")
            with open(self.model_dir / "thresholds.json", "r") as f:
                with open(tmp_path / "thresholds.json", "w") as out:
                    out.write(f.read())
            with open(self.model_dir / "classes.json", "r") as f:
                with open(tmp_path / "classes.json", "w") as out:
                    out.write(f.read())

            with self.assertRaises(FileNotFoundError) as ctx:
                TrustEngine.load(tmp_path)
            self.assertIn("crop_pipeline_calibrated.joblib", str(ctx.exception))

    def test_classes_mismatch_raises_valueerror(self) -> None:
        """Verify load() raises ValueError if classes.json does not match model classes_."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            joblib.dump(joblib.load(self.model_dir / "ood_stats.joblib"), tmp_path / "ood_stats.joblib")
            with open(self.model_dir / "thresholds.json", "r") as f:
                with open(tmp_path / "thresholds.json", "w") as out:
                    out.write(f.read())
            joblib.dump(joblib.load(self.model_dir / "crop_pipeline_calibrated.joblib"), tmp_path / "crop_pipeline_calibrated.joblib")

            # Corrupt classes.json by replacing one class name
            with open(self.model_dir / "classes.json", "r") as f:
                valid_classes = json.load(f)
            bad_classes = list(valid_classes)
            bad_classes[0] = "alien_crop"
            with open(tmp_path / "classes.json", "w") as out:
                json.dump(bad_classes, out)

            with self.assertRaises(ValueError) as ctx:
                TrustEngine.load(tmp_path)
            self.assertIn("Calibrated model classes do not match classes.json", str(ctx.exception))

    def test_missing_and_corrupt_classes_json(self) -> None:
        """Verify load() validates presence and format of classes.json."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            joblib.dump(joblib.load(self.model_dir / "ood_stats.joblib"), tmp_path / "ood_stats.joblib")
            with open(self.model_dir / "thresholds.json", "r") as f:
                with open(tmp_path / "thresholds.json", "w") as out:
                    out.write(f.read())

            # Missing classes.json
            with self.assertRaises(FileNotFoundError) as ctx:
                TrustEngine.load(tmp_path)
            self.assertIn("classes.json", str(ctx.exception))

            # Corrupt classes.json (not 22 classes)
            with open(tmp_path / "classes.json", "w") as out:
                json.dump(["rice", "maize"], out)
            with self.assertRaises(ValueError) as ctx:
                TrustEngine.load(tmp_path)
            self.assertIn("must be a list of 22 class name strings", str(ctx.exception))

    def test_interim_stage2_handoff_preserved(self) -> None:
        """Verify assess() executes Stage 5 on ID and preserves standard refusal on OOD."""
        # 1. Familiar in-distribution input produces valid recommendation/caution
        res = self.engine.assess(self.rice_row, risk="balanced")
        self.assertIn(res["verdict"], {"recommend", "caution"})
        self.assertEqual(res["top_crops"][0]["crop"], "rice")
        self.assertFalse(res["ood"]["flagged"])

        # 2. OOD input must return standard HTTP 200 refusal dict
        refusal_dict = self.engine.assess(self.rice_unusual_pair, risk="balanced")
        self.assertEqual(refusal_dict["verdict"], "refuse")
        self.assertEqual(refusal_dict["top_crops"], [])
        self.assertTrue(refusal_dict["ood"]["flagged"])
        self.assertEqual(refusal_dict["trust_score"], 0)

    def test_unloaded_engine_raises_runtime_error(self) -> None:
        """Verify calling assess_probabilities on uninitialized engine raises RuntimeError."""
        unloaded_engine = TrustEngine(pipeline=None, classes=None)
        with self.assertRaises(RuntimeError) as ctx:
            unloaded_engine.assess_probabilities(self.rice_row)
        self.assertIn("not loaded", str(ctx.exception))


class _DummyPredictPipeline:
    """Mock pipeline helper for Stage 4 stability unit tests."""

    def __init__(self, predictions: np.ndarray, classes: List[str]) -> None:
        self.predictions = predictions
        self.classes_ = np.array(classes)

    def predict(self, X: Any) -> np.ndarray:
        return self.predictions


class TestTrustEngineStage4Stability(unittest.TestCase):
    """Unit test suite for TrustEngine Stage 4 Stability Assessment."""

    def setUp(self) -> None:
        """Initialize engine with real calibrated model."""
        base_dir = Path(__file__).resolve().parent.parent
        model_candidates = [
            base_dir / "models",
            Path("backend/models"),
            Path("models"),
        ]
        self.model_dir = next(
            (p for p in model_candidates if (p / "crop_pipeline_calibrated.joblib").exists()),
            None,
        )
        self.assertIsNotNone(
            self.model_dir,
            "Could not locate valid model directory containing crop_pipeline_calibrated.joblib",
        )
        self.engine = TrustEngine.load(self.model_dir)

        # Standard familiar in-distribution input
        self.rice_row: Dict[str, Any] = {
            "N": 90.0,
            "P": 42.0,
            "K": 43.0,
            "temperature": 21.0,
            "humidity": 82.0,
            "ph": 6.5,
            "rainfall": 203.0,
        }

    def test_stability_fully_stable(self) -> None:
        """Verify 100% matching predictions yield stability=1.0 and flip_rate=0.0."""
        n_samples = 100
        mock_preds = np.array(["rice"] * n_samples)
        mock_pipe = _DummyPredictPipeline(mock_preds, self.engine.classes)
        engine_mock = TrustEngine(pipeline=mock_pipe, classes=self.engine.classes)

        res = engine_mock.assess_stability(self.rice_row, clean_crop="rice", n_samples=n_samples, seed=42)
        self.assertEqual(res["stability"], 1.0)
        self.assertEqual(res["flip_rate"], 0.0)
        self.assertEqual(res["clean_crop"], "rice")
        self.assertEqual(res["n_samples"], n_samples)
        self.assertEqual(res["matches"], n_samples)

    def test_stability_fully_unstable(self) -> None:
        """Verify 0% matching predictions yield stability=0.0 and flip_rate=1.0."""
        n_samples = 100
        mock_preds = np.array(["jute"] * n_samples)
        mock_pipe = _DummyPredictPipeline(mock_preds, self.engine.classes)
        engine_mock = TrustEngine(pipeline=mock_pipe, classes=self.engine.classes)

        res = engine_mock.assess_stability(self.rice_row, clean_crop="rice", n_samples=n_samples, seed=42)
        self.assertEqual(res["stability"], 0.0)
        self.assertEqual(res["flip_rate"], 1.0)
        self.assertEqual(res["clean_crop"], "rice")
        self.assertEqual(res["n_samples"], n_samples)
        self.assertEqual(res["matches"], 0)

    def test_stability_partial(self) -> None:
        """Verify partial matches (e.g. 70/100) yield stability=0.70 and flip_rate=0.30."""
        n_samples = 100
        mock_preds = np.array(["rice"] * 70 + ["jute"] * 30)
        mock_pipe = _DummyPredictPipeline(mock_preds, self.engine.classes)
        engine_mock = TrustEngine(pipeline=mock_pipe, classes=self.engine.classes)

        res = engine_mock.assess_stability(self.rice_row, clean_crop="rice", n_samples=n_samples, seed=42)
        self.assertEqual(res["stability"], 0.70)
        self.assertEqual(res["flip_rate"], 1.0 - 0.70)
        self.assertEqual(res["clean_crop"], "rice")
        self.assertEqual(res["n_samples"], n_samples)
        self.assertEqual(res["matches"], 70)

    def test_exact_ratio_unrounded(self) -> None:
        """Verify stability and flip_rate preserve exact floating point ratios without rounding."""
        # 2 matches out of 3 samples (2/3 = 0.6666666666666666...)
        n_samples = 3
        mock_preds = np.array(["rice", "rice", "jute"])
        mock_pipe = _DummyPredictPipeline(mock_preds, self.engine.classes)
        engine_mock = TrustEngine(pipeline=mock_pipe, classes=self.engine.classes)

        res = engine_mock.assess_stability(self.rice_row, clean_crop="rice", n_samples=n_samples, seed=42)
        expected_stability = 2.0 / 3.0
        expected_flip_rate = 1.0 - (2.0 / 3.0)

        self.assertEqual(res["stability"], expected_stability)
        self.assertEqual(res["flip_rate"], expected_flip_rate)
        self.assertNotEqual(res["stability"], round(expected_stability, 4))
        self.assertIsInstance(res["matches"], int)
        self.assertIsInstance(res["n_samples"], int)
        self.assertEqual(res["matches"], 2)
        self.assertEqual(res["n_samples"], 3)

    def test_clean_crop_comparison(self) -> None:
        """Verify matches correctly evaluate against the designated clean_crop."""
        n_samples = 100
        mock_preds = np.array(["rice"] * 70 + ["jute"] * 30)
        mock_pipe = _DummyPredictPipeline(mock_preds, self.engine.classes)
        engine_mock = TrustEngine(pipeline=mock_pipe, classes=self.engine.classes)

        # Checking against 'jute' should yield 30 matches (0.30 stability)
        res = engine_mock.assess_stability(self.rice_row, clean_crop="jute", n_samples=n_samples, seed=42)
        self.assertEqual(res["stability"], 0.30)
        self.assertEqual(res["flip_rate"], 0.70)
        self.assertEqual(res["clean_crop"], "jute")
        self.assertEqual(res["matches"], 30)

    def test_real_calibrated_model_inference(self) -> None:
        """Verify real calibrated model produces bounded stability metrics."""
        res = self.engine.assess_stability(self.rice_row, clean_crop="rice", n_samples=100, seed=42)

        for required_key in ("stability", "flip_rate", "clean_crop", "n_samples", "matches"):
            self.assertIn(required_key, res)

        self.assertIsInstance(res["stability"], float)
        self.assertIsInstance(res["flip_rate"], float)
        self.assertGreaterEqual(res["stability"], 0.0)
        self.assertLessEqual(res["stability"], 1.0)
        self.assertGreaterEqual(res["flip_rate"], 0.0)
        self.assertLessEqual(res["flip_rate"], 1.0)
        self.assertAlmostEqual(res["stability"] + res["flip_rate"], 1.0, places=4)
        self.assertEqual(res["clean_crop"], "rice")
        self.assertEqual(res["n_samples"], 100)
        self.assertEqual(res["matches"], int(round(res["stability"] * 100)))

    def test_invalid_clean_crop(self) -> None:
        """Verify ValueError is raised if clean_crop is not in known classes or non-string."""
        with self.assertRaises(ValueError) as ctx:
            self.engine.assess_stability(self.rice_row, clean_crop="alien_crop")
        self.assertIn("Invalid clean_crop", str(ctx.exception))

        with self.assertRaises(ValueError) as ctx:
            self.engine.assess_stability(self.rice_row, clean_crop=123)  # type: ignore
        self.assertIn("Invalid clean_crop", str(ctx.exception))

    def test_invalid_sample_counts_including_booleans(self) -> None:
        """Verify ValueError is raised for non-positive or non-integer n_samples, including booleans."""
        # Booleans (bool is a subclass of int in Python)
        for bad_val in (True, False, 0, -1, -50, 10.5, "100", None):
            with self.assertRaises(ValueError) as ctx:
                self.engine.assess_stability(self.rice_row, clean_crop="rice", n_samples=bad_val)  # type: ignore
            self.assertIn("positive integer", str(ctx.exception))

    def test_invalid_prediction_shape_and_count(self) -> None:
        """Verify ValueError is raised if pipeline.predict returns unexpected shape."""
        # 50 predictions instead of 100
        mock_pipe_short = _DummyPredictPipeline(np.array(["rice"] * 50), self.engine.classes)
        engine_mock_short = TrustEngine(pipeline=mock_pipe_short, classes=self.engine.classes)
        with self.assertRaises(ValueError) as ctx:
            engine_mock_short.assess_stability(self.rice_row, clean_crop="rice", n_samples=100)
        self.assertIn("invalid shape", str(ctx.exception))

        # 2D predictions (100, 1) instead of 1D (100,)
        mock_pipe_2d = _DummyPredictPipeline(np.array([["rice"]] * 100), self.engine.classes)
        engine_mock_2d = TrustEngine(pipeline=mock_pipe_2d, classes=self.engine.classes)
        with self.assertRaises(ValueError) as ctx:
            engine_mock_2d.assess_stability(self.rice_row, clean_crop="rice", n_samples=100)
        self.assertIn("invalid shape", str(ctx.exception))

    def test_unknown_prediction_labels(self) -> None:
        """Verify ValueError is raised if pipeline.predict produces unknown crop labels."""
        mock_pipe_bad = _DummyPredictPipeline(np.array(["unknown_crop"] * 100), self.engine.classes)
        engine_mock_bad = TrustEngine(pipeline=mock_pipe_bad, classes=self.engine.classes)
        with self.assertRaises(ValueError) as ctx:
            engine_mock_bad.assess_stability(self.rice_row, clean_crop="rice", n_samples=100)
        self.assertIn("unknown crop labels", str(ctx.exception))

    def test_reproducibility_with_explicit_seed(self) -> None:
        """Verify identical seed produces identical stability and match counts."""
        res1 = self.engine.assess_stability(self.rice_row, clean_crop="rice", n_samples=100, seed=42)
        res2 = self.engine.assess_stability(self.rice_row, clean_crop="rice", n_samples=100, seed=42)
        self.assertEqual(res1, res2)

    def test_seed_none_accepted(self) -> None:
        """Verify seed=None executes without error."""
        res = self.engine.assess_stability(self.rice_row, clean_crop="rice", n_samples=50, seed=None)
        self.assertIsInstance(res["stability"], float)
        self.assertEqual(res["n_samples"], 50)

    def test_input_row_immutability(self) -> None:
        """Verify the original input dictionary is never mutated."""
        row_copy = dict(self.rice_row)
        _ = self.engine.assess_stability(self.rice_row, clean_crop="rice", n_samples=50, seed=42)
        self.assertEqual(self.rice_row, row_copy)

    def test_invalid_input_row_missing_and_types(self) -> None:
        """Verify input validation handles missing features, non-dict, and non-numeric types."""
        # Missing feature
        incomplete = dict(self.rice_row)
        del incomplete["temperature"]
        with self.assertRaises(ValueError) as ctx:
            self.engine.assess_stability(incomplete, clean_crop="rice")
        self.assertIn("Missing required feature(s)", str(ctx.exception))

        # Boolean feature
        bool_row = dict(self.rice_row)
        bool_row["rainfall"] = False
        with self.assertRaises(ValueError) as ctx:
            self.engine.assess_stability(bool_row, clean_crop="rice")
        self.assertIn("must be numeric, got boolean", str(ctx.exception))

        # Non-dict
        with self.assertRaises(ValueError) as ctx:
            self.engine.assess_stability([1, 2, 3], clean_crop="rice")  # type: ignore
        self.assertIn("must be a dictionary", str(ctx.exception))

    def test_unloaded_engine_raises_runtime_error(self) -> None:
        """Verify calling assess_stability on uninitialized engine raises RuntimeError."""
        unloaded = TrustEngine(pipeline=None, classes=None)
        with self.assertRaises(RuntimeError) as ctx:
            unloaded.assess_stability(self.rice_row, clean_crop="rice")
        self.assertIn("not loaded", str(ctx.exception))


class TestTrustEngineStage5Verdict(unittest.TestCase):
    """Comprehensive unit test suite for TrustEngine Stage 5 Verdict Logic & Trust Score."""

    def setUp(self) -> None:
        """Initialize engine and canonical test input."""
        base_dir = Path(__file__).resolve().parent.parent
        model_candidates = [
            base_dir / "models",
            Path("backend/models"),
            Path("models"),
        ]
        self.model_dir = next(
            (p for p in model_candidates if (p / "crop_pipeline_calibrated.joblib").exists()),
            None,
        )
        self.assertIsNotNone(
            self.model_dir,
            "Could not locate valid model directory containing crop_pipeline_calibrated.joblib",
        )
        self.engine = TrustEngine.load(self.model_dir)
        self.valid_row: Dict[str, Any] = {
            "N": 90.0,
            "P": 42.0,
            "K": 43.0,
            "temperature": 21.0,
            "humidity": 82.0,
            "ph": 6.5,
            "rainfall": 203.0,
        }

    def _mock_stages(
        self,
        ood_flagged: bool = False,
        min_d2: float = 5.0,
        percentile: float = 35.0,
        unusual_pair: Any = None,
        nearest_known: Any = None,
        top_crop: str = "rice",
        top_prob: float = 0.90,
        second_crop: str = "jute",
        second_prob: float = 0.05,
        margin: Optional[float] = None,
        stability: float = 0.95,
        matches: int = 190,
        n_samples: int = 200,
    ) -> Tuple[MagicMock, MagicMock, MagicMock]:
        """Helper to create synchronized stage mocks."""
        if margin is None:
            margin = top_prob - second_prob
        if nearest_known is None:
            nearest_known = ["rice", "jute"]

        mock_ood = MagicMock(return_value={
            "eligible": not ood_flagged,
            "flagged": ood_flagged,
            "percentile": percentile,
            "min_distance": min_d2,
            "threshold": 12.92,
            "caution_threshold": 10.78,
            "nearest_known": nearest_known,
            "unusual_pair": unusual_pair,
            "risk_profile": "balanced",
        })

        mock_prob = MagicMock(return_value={
            "top_crop": top_crop,
            "top_probability": top_prob,
            "second_crop": second_crop,
            "second_probability": second_prob,
            "margin": margin,
            "top_crops": [
                {"crop": top_crop, "probability": round(top_prob, 4)},
                {"crop": second_crop, "probability": round(second_prob, 4)},
                {"crop": "coffee", "probability": round(max(0.0, 1.0 - top_prob - second_prob), 4)},
            ],
        })

        mock_stab = MagicMock(return_value={
            "stability": stability,
            "flip_rate": 1.0 - stability,
            "clean_crop": top_crop,
            "n_samples": n_samples,
            "matches": matches,
        })

        return mock_ood, mock_prob, mock_stab

    def test_clear_recommend_verdict_and_payload_invariants(self) -> None:
        """Verify clear in-distribution inputs yield recommend verdict with all payload invariants."""
        mock_ood, mock_prob, mock_stab = self._mock_stages(
            min_d2=5.0, top_prob=0.88, second_prob=0.08, stability=0.94
        )
        with patch.object(self.engine, "assess_ood", mock_ood), \
             patch.object(self.engine, "assess_probabilities", mock_prob), \
             patch.object(self.engine, "assess_stability", mock_stab):
            res = self.engine.assess(self.valid_row, risk="balanced")

        self.assertEqual(res["verdict"], "recommend")
        self.assertEqual(res["trust_score"], 88)
        self.assertEqual(res["stability"], 0.94)
        self.assertIsNone(res["conformal_set"])
        self.assertIsNone(res["measure_next"])
        self.assertEqual(res["model_version"], self.engine.MODEL_VERSION)
        self.assertFalse(res["ood"]["flagged"])
        self.assertIsNone(res["ood"]["unusual_pair"])
        self.assertIsNone(res["ood"]["nearest_known"])
        self.assertIn("High confidence (88.0%) with high stability (94.0%)", res["explanation"])
        self.assertIn("simulated measurement variations", res["explanation"])
        self.assertEqual(len(res["top_crops"]), 3)

    def test_single_triggers_cause_caution_across_profiles(self) -> None:
        """Verify each of the 4 triggers alone causes caution across all 3 risk profiles."""
        profiles = ["conservative", "balanced", "bold"]
        for risk in profiles:
            p = self.engine.thresholds["profiles"][risk]
            caution_d2 = p["ood_caution_threshold"]
            min_prob = p["min_prob"]
            min_margin = p["min_margin"]
            min_stab = p["min_stability"]

            # Trigger 1: ood_caution alone
            mock_ood, mock_prob, mock_stab = self._mock_stages(
                min_d2=caution_d2 + 0.1,
                top_prob=min_prob + 0.05,
                margin=min_margin + 0.05,
                stability=min_stab + 0.05,
            )
            mock_ood.return_value["risk_profile"] = risk
            with patch.object(self.engine, "assess_ood", mock_ood), \
                 patch.object(self.engine, "assess_probabilities", mock_prob), \
                 patch.object(self.engine, "assess_stability", mock_stab):
                res = self.engine.assess(self.valid_row, risk=risk)
                self.assertEqual(res["verdict"], "caution", f"Failed on ood_caution for {risk}")
                self.assertIn("more unusual than about", res["explanation"])
                self.assertNotIn("percentile", res["explanation"])

            # Trigger 2: low_probability alone
            mock_ood, mock_prob, mock_stab = self._mock_stages(
                min_d2=caution_d2 - 0.5,
                top_prob=min_prob - 0.01,
                margin=min_margin + 0.05,
                stability=min_stab + 0.05,
            )
            mock_ood.return_value["risk_profile"] = risk
            with patch.object(self.engine, "assess_ood", mock_ood), \
                 patch.object(self.engine, "assess_probabilities", mock_prob), \
                 patch.object(self.engine, "assess_stability", mock_stab):
                res = self.engine.assess(self.valid_row, risk=risk)
                self.assertEqual(res["verdict"], "caution", f"Failed on low_probability for {risk}")
                self.assertIn("confidence is low", res["explanation"])

            # Trigger 3: low_margin alone
            mock_ood, mock_prob, mock_stab = self._mock_stages(
                min_d2=caution_d2 - 0.5,
                top_prob=min_prob + 0.05,
                margin=min_margin - 0.01,
                stability=min_stab + 0.05,
            )
            mock_ood.return_value["risk_profile"] = risk
            with patch.object(self.engine, "assess_ood", mock_ood), \
                 patch.object(self.engine, "assess_probabilities", mock_prob), \
                 patch.object(self.engine, "assess_stability", mock_stab):
                res = self.engine.assess(self.valid_row, risk=risk)
                self.assertEqual(res["verdict"], "caution", f"Failed on low_margin for {risk}")
                self.assertIn("Margin between top two crops is narrow", res["explanation"])

            # Trigger 4: low_stability alone
            mock_ood, mock_prob, mock_stab = self._mock_stages(
                min_d2=caution_d2 - 0.5,
                top_prob=min_prob + 0.05,
                margin=min_margin + 0.05,
                stability=min_stab - 0.01,
            )
            mock_ood.return_value["risk_profile"] = risk
            with patch.object(self.engine, "assess_ood", mock_ood), \
                 patch.object(self.engine, "assess_probabilities", mock_prob), \
                 patch.object(self.engine, "assess_stability", mock_stab):
                res = self.engine.assess(self.valid_row, risk=risk)
                self.assertEqual(res["verdict"], "caution", f"Failed on low_stability for {risk}")
                self.assertIn("sensitive to simulated measurement variations", res["explanation"])

    def test_exact_threshold_equality_and_epsilon_boundaries(self) -> None:
        """Verify exact equality does not trigger, while epsilon beyond does."""
        p = self.engine.thresholds["profiles"]["balanced"]
        caution_d2 = p["ood_caution_threshold"]  # 10.78
        min_prob = p["min_prob"]                  # 0.70
        min_margin = p["min_margin"]              # 0.20
        min_stab = p["min_stability"]            # 0.80

        # Exact equality on all four thresholds -> NOT caution, must be recommend
        mock_ood, mock_prob, mock_stab = self._mock_stages(
            min_d2=caution_d2,
            top_prob=min_prob,
            margin=min_margin,
            stability=min_stab,
        )
        with patch.object(self.engine, "assess_ood", mock_ood), \
             patch.object(self.engine, "assess_probabilities", mock_prob), \
             patch.object(self.engine, "assess_stability", mock_stab):
            res = self.engine.assess(self.valid_row, risk="balanced")
            self.assertEqual(res["verdict"], "recommend")

        # Hair beyond on d2 (+1e-4) -> caution
        mock_ood, mock_prob, mock_stab = self._mock_stages(
            min_d2=caution_d2 + 1e-4, top_prob=min_prob, margin=min_margin, stability=min_stab
        )
        with patch.object(self.engine, "assess_ood", mock_ood), \
             patch.object(self.engine, "assess_probabilities", mock_prob), \
             patch.object(self.engine, "assess_stability", mock_stab):
            self.assertEqual(self.engine.assess(self.valid_row, risk="balanced")["verdict"], "caution")

        # Hair beyond on top_prob (-1e-4) -> caution
        mock_ood, mock_prob, mock_stab = self._mock_stages(
            min_d2=caution_d2, top_prob=min_prob - 1e-4, margin=min_margin, stability=min_stab
        )
        with patch.object(self.engine, "assess_ood", mock_ood), \
             patch.object(self.engine, "assess_probabilities", mock_prob), \
             patch.object(self.engine, "assess_stability", mock_stab):
            self.assertEqual(self.engine.assess(self.valid_row, risk="balanced")["verdict"], "caution")

        # Hair beyond on margin (-1e-4) -> caution
        mock_ood, mock_prob, mock_stab = self._mock_stages(
            min_d2=caution_d2, top_prob=min_prob, margin=min_margin - 1e-4, stability=min_stab
        )
        with patch.object(self.engine, "assess_ood", mock_ood), \
             patch.object(self.engine, "assess_probabilities", mock_prob), \
             patch.object(self.engine, "assess_stability", mock_stab):
            self.assertEqual(self.engine.assess(self.valid_row, risk="balanced")["verdict"], "caution")

        # Hair beyond on stability (-1e-4) -> caution
        mock_ood, mock_prob, mock_stab = self._mock_stages(
            min_d2=caution_d2, top_prob=min_prob, margin=min_margin, stability=min_stab - 1e-4
        )
        with patch.object(self.engine, "assess_ood", mock_ood), \
             patch.object(self.engine, "assess_probabilities", mock_prob), \
             patch.object(self.engine, "assess_stability", mock_stab):
            self.assertEqual(self.engine.assess(self.valid_row, risk="balanced")["verdict"], "caution")

    def test_refuse_behavior_and_early_exit(self) -> None:
        """Verify OOD refusal returns standard refuse payload and skips Stage 3-4."""
        mock_ood, mock_prob, mock_stab = self._mock_stages(
            ood_flagged=True,
            min_d2=18.5,
            percentile=99.8,
            unusual_pair=["humidity", "rainfall"],
            nearest_known=["rice", "jute"],
        )
        with patch.object(self.engine, "assess_ood", mock_ood), \
             patch.object(self.engine, "assess_probabilities", mock_prob), \
             patch.object(self.engine, "assess_stability", mock_stab):
            res = self.engine.assess(self.valid_row, risk="balanced")

        self.assertEqual(res["verdict"], "refuse")
        self.assertEqual(res["trust_score"], 0)
        self.assertEqual(res["top_crops"], [])
        self.assertIsNone(res["stability"])
        self.assertIsNone(res["conformal_set"])
        self.assertIsNone(res["measure_next"])
        self.assertTrue(res["ood"]["flagged"])
        self.assertEqual(res["ood"]["unusual_pair"], ["humidity", "rainfall"])
        self.assertEqual(res["ood"]["nearest_known"], ["rice", "jute"])
        self.assertIn("out-of-distribution", res["explanation"])

        mock_prob.assert_not_called()
        mock_stab.assert_not_called()

    def test_trust_score_capping_and_recommend_uncapped(self) -> None:
        """Verify trust_score capping at 60 on caution, uncapped on recommend."""
        # Caution with high min(p, s) = 0.85 -> capped at 60
        mock_ood, mock_prob, mock_stab = self._mock_stages(
            min_d2=11.5,  # triggers ood_caution (> 10.78)
            top_prob=0.85,
            second_prob=0.10,
            stability=0.90,
        )
        with patch.object(self.engine, "assess_ood", mock_ood), \
             patch.object(self.engine, "assess_probabilities", mock_prob), \
             patch.object(self.engine, "assess_stability", mock_stab):
            res = self.engine.assess(self.valid_row, risk="balanced")
            self.assertEqual(res["verdict"], "caution")
            self.assertEqual(res["trust_score"], 60)

        # Caution with low min(p, s) = 0.45 -> keeps 45 (< 60)
        mock_ood, mock_prob, mock_stab = self._mock_stages(
            min_d2=5.0,
            top_prob=0.45,  # triggers low_prob
            second_prob=0.10,
            stability=0.90,
        )
        with patch.object(self.engine, "assess_ood", mock_ood), \
             patch.object(self.engine, "assess_probabilities", mock_prob), \
             patch.object(self.engine, "assess_stability", mock_stab):
            res = self.engine.assess(self.valid_row, risk="balanced")
            self.assertEqual(res["verdict"], "caution")
            self.assertEqual(res["trust_score"], 45)

        # Recommend with high min(p, s) = 0.85 -> uncapped (85)
        mock_ood, mock_prob, mock_stab = self._mock_stages(
            min_d2=5.0,
            top_prob=0.85,
            second_prob=0.10,
            stability=0.90,
        )
        with patch.object(self.engine, "assess_ood", mock_ood), \
             patch.object(self.engine, "assess_probabilities", mock_prob), \
             patch.object(self.engine, "assess_stability", mock_stab):
            res = self.engine.assess(self.valid_row, risk="balanced")
            self.assertEqual(res["verdict"], "recommend")
            self.assertEqual(res["trust_score"], 85)

    def test_trust_score_rounding_half_up(self) -> None:
        """Verify trust_score implements round-half-up and produces an int."""
        # 100 * 0.755 = 75.5 -> rounds half up to 76
        mock_ood, mock_prob, mock_stab = self._mock_stages(
            min_d2=5.0, top_prob=0.755, second_prob=0.10, stability=0.90
        )
        with patch.object(self.engine, "assess_ood", mock_ood), \
             patch.object(self.engine, "assess_probabilities", mock_prob), \
             patch.object(self.engine, "assess_stability", mock_stab):
            res = self.engine.assess(self.valid_row, risk="balanced")
            self.assertEqual(res["verdict"], "recommend")
            self.assertEqual(res["trust_score"], 76)
            self.assertIsInstance(res["trust_score"], int)

        # 100 * 0.754 = 75.4 -> 75
        mock_ood, mock_prob, mock_stab = self._mock_stages(
            min_d2=5.0, top_prob=0.754, second_prob=0.10, stability=0.90
        )
        with patch.object(self.engine, "assess_ood", mock_ood), \
             patch.object(self.engine, "assess_probabilities", mock_prob), \
             patch.object(self.engine, "assess_stability", mock_stab):
            res = self.engine.assess(self.valid_row, risk="balanced")
            self.assertEqual(res["verdict"], "recommend")
            self.assertEqual(res["trust_score"], 75)
            self.assertIsInstance(res["trust_score"], int)

    def test_borderline_profile_sensitivity(self) -> None:
        """Verify same borderline input gives caution under conservative and recommend under balanced."""
        mock_ood_bal, mock_prob, mock_stab = self._mock_stages(
            min_d2=5.0, top_prob=0.85, second_prob=0.10, stability=0.85
        )
        mock_ood_cons, _, _ = self._mock_stages(
            min_d2=5.0, top_prob=0.85, second_prob=0.10, stability=0.85
        )
        mock_ood_bal.return_value["risk_profile"] = "balanced"
        mock_ood_cons.return_value["risk_profile"] = "conservative"

        with patch.object(self.engine, "assess_probabilities", mock_prob), \
             patch.object(self.engine, "assess_stability", mock_stab):
            with patch.object(self.engine, "assess_ood", mock_ood_cons):
                res_cons = self.engine.assess(self.valid_row, risk="conservative")
                self.assertEqual(res_cons["verdict"], "caution")
            with patch.object(self.engine, "assess_ood", mock_ood_bal):
                res_bal = self.engine.assess(self.valid_row, risk="balanced")
                self.assertEqual(res_bal["verdict"], "recommend")

    def test_json_serializability_and_explicit_nulls(self) -> None:
        """Verify assess() result is fully JSON-serializable and optional keys are explicit null."""
        mock_ood, mock_prob, mock_stab = self._mock_stages(
            min_d2=5.0, top_prob=0.85, second_prob=0.10, stability=0.90
        )
        with patch.object(self.engine, "assess_ood", mock_ood), \
             patch.object(self.engine, "assess_probabilities", mock_prob), \
             patch.object(self.engine, "assess_stability", mock_stab):
            res = self.engine.assess(self.valid_row, risk="balanced")

        json_str = json.dumps(res)
        parsed = json.loads(json_str)
        self.assertIsNone(parsed["conformal_set"])
        self.assertIsNone(parsed["measure_next"])
        self.assertIsNone(parsed["ood"]["unusual_pair"])
        self.assertIsNone(parsed["ood"]["nearest_known"])
        self.assertIn('"conformal_set": null', json_str)
        self.assertIn('"measure_next": null', json_str)

    def test_input_row_not_mutated(self) -> None:
        """Verify assess() does not mutate the passed row dictionary."""
        mock_ood, mock_prob, mock_stab = self._mock_stages()
        original = dict(self.valid_row)
        with patch.object(self.engine, "assess_ood", mock_ood), \
             patch.object(self.engine, "assess_probabilities", mock_prob), \
             patch.object(self.engine, "assess_stability", mock_stab):
            self.engine.assess(self.valid_row, risk="balanced")
        self.assertEqual(self.valid_row, original)

    def test_assess_stability_serving_parameters(self) -> None:
        """Verify assess() calls assess_stability with n_samples=200 and seed=42."""
        mock_ood, mock_prob, mock_stab = self._mock_stages(top_crop="rice")
        with patch.object(self.engine, "assess_ood", mock_ood), \
             patch.object(self.engine, "assess_probabilities", mock_prob), \
             patch.object(self.engine, "assess_stability", mock_stab):
            self.engine.assess(self.valid_row, risk="balanced")

        mock_stab.assert_called_once_with(
            self.valid_row,
            "rice",
            n_samples=200,
            seed=42,
        )

    def test_invalid_risk_and_unloaded_engine(self) -> None:
        """Verify assess() rejects invalid risk profiles and unloaded engine."""
        with self.assertRaises(ValueError) as ctx:
            self.engine.assess(self.valid_row, risk="ultra_bold")
        self.assertIn("Invalid risk profile", str(ctx.exception))

        unloaded = TrustEngine(pipeline=None, classes=None)
        with self.assertRaises(RuntimeError) as ctx_unloaded:
            unloaded.assess(self.valid_row, risk="balanced")
        self.assertIn("not loaded", str(ctx_unloaded.exception))

    def test_ood_caution_explanation_wording(self) -> None:
        """Verify ood_caution explanation contains new wording and avoids the word 'percentile'."""
        p = self.engine.thresholds["profiles"]["balanced"]
        caution_d2 = p["ood_caution_threshold"]
        min_prob = p["min_prob"]
        min_margin = p["min_margin"]
        min_stab = p["min_stability"]

        # Only ood_caution fires (percentile=85.4 -> rounds to 85%)
        mock_ood, mock_prob, mock_stab = self._mock_stages(
            min_d2=caution_d2 + 1.0,
            percentile=85.4,
            top_prob=min_prob + 0.10,
            margin=min_margin + 0.10,
            stability=min_stab + 0.10,
        )
        with patch.object(self.engine, "assess_ood", mock_ood), \
             patch.object(self.engine, "assess_probabilities", mock_prob), \
             patch.object(self.engine, "assess_stability", mock_stab):
            res = self.engine.assess(self.valid_row, risk="balanced")

        self.assertEqual(res["verdict"], "caution")
        self.assertIn("more unusual than about", res["explanation"])
        self.assertNotIn("percentile", res["explanation"])
        self.assertEqual(
            res["explanation"],
            "Input is more unusual than about 85% of familiar training inputs.",
        )

    def test_real_model_smoke_test(self) -> None:
        """Verify end-to-end assess() on real calibrated model without mocks."""
        res = self.engine.assess(self.valid_row, risk="balanced")
        self.assertIn(res["verdict"], {"recommend", "caution", "refuse"})
        expected_keys = {
            "verdict",
            "top_crops",
            "conformal_set",
            "ood",
            "stability",
            "trust_score",
            "explanation",
            "model_version",
            "measure_next",
        }
        self.assertEqual(set(res.keys()), expected_keys)
        self.assertIsInstance(res["trust_score"], int)
        self.assertTrue(0 <= res["trust_score"] <= 100)
        self.assertEqual(res["model_version"], self.engine.MODEL_VERSION)
        self.assertIn(res["verdict"], {"recommend", "caution", "refuse"})
        self.assertEqual(res["top_crops"][0]["crop"], "rice")

    def test_measure_next_feature_fixed_pair_and_symmetry(self) -> None:
        """Verify measure_next_feature computes the separating feature matching independent formula and is symmetric."""
        with open(self.model_dir / "class_stats.json", "r", encoding="utf-8") as f:
            stats = json.load(f)

        features = stats["features"]
        means = stats["class_means"]
        sds = stats["class_sds"]

        crop_a, crop_b = "rice", "jute"

        # Independent calculation inside the test
        expected_feat = None
        expected_gap = 0.0
        for j, feat in enumerate(features):
            pooled = math.sqrt((sds[crop_a][j] ** 2 + sds[crop_b][j] ** 2) / 2.0)
            if pooled <= 0.0 or not math.isfinite(pooled):
                continue
            gap = abs(means[crop_a][j] - means[crop_b][j]) / pooled
            if gap > expected_gap:
                expected_gap = gap
                expected_feat = feat

        feat_ab, gap_ab = self.engine.measure_next_feature(crop_a, crop_b)
        self.assertEqual(feat_ab, expected_feat)
        self.assertAlmostEqual(gap_ab, expected_gap, places=5)

        # Symmetry test: (a, b) and (b, a) yield identical feature and gap
        feat_ba, gap_ba = self.engine.measure_next_feature(crop_b, crop_a)
        self.assertEqual(feat_ba, feat_ab)
        self.assertAlmostEqual(gap_ba, gap_ab, places=5)

    def test_measure_next_feature_validation_errors(self) -> None:
        """Verify measure_next_feature raises ValueError for identical crops and unknown crop names."""
        # Same crop
        with self.assertRaises(ValueError) as ctx:
            self.engine.measure_next_feature("rice", "rice")
        self.assertIn("distinct", str(ctx.exception).lower())

        # Unknown crop A
        with self.assertRaises(ValueError) as ctx:
            self.engine.measure_next_feature("nonexistent_crop", "rice")
        self.assertIn("unknown crop", str(ctx.exception).lower())

        # Unknown crop B
        with self.assertRaises(ValueError) as ctx:
            self.engine.measure_next_feature("rice", "nonexistent_crop")
        self.assertIn("unknown crop", str(ctx.exception).lower())

    def test_assess_measure_next_trigger_behavior(self) -> None:
        """Verify measure_next behavior across assess verdicts and triggers."""
        p = self.engine.thresholds["profiles"]["balanced"]
        caution_d2 = p["ood_caution_threshold"]
        min_prob = p["min_prob"]
        min_margin = p["min_margin"]
        min_stab = p["min_stability"]

        # Case 1: caution WITH low_margin alone
        mock_ood, mock_prob, mock_stab = self._mock_stages(
            top_crop="rice",
            second_crop="jute",
            min_d2=caution_d2 - 0.5,
            top_prob=min_prob + 0.05,
            margin=min_margin - 0.01,
            stability=min_stab + 0.05,
        )
        with patch.object(self.engine, "assess_ood", mock_ood), \
             patch.object(self.engine, "assess_probabilities", mock_prob), \
             patch.object(self.engine, "assess_stability", mock_stab):
            res_margin = self.engine.assess(self.valid_row, risk="balanced")

        self.assertEqual(res_margin["verdict"], "caution")
        self.assertIsNotNone(res_margin["measure_next"])
        self.assertIn("rice", res_margin["measure_next"])
        self.assertIn("jute", res_margin["measure_next"])
        self.assertTrue(
            res_margin["measure_next"].endswith("it separates rice from jute most strongly. Re-test it.")
        )

        # Case 2: caution WITHOUT low_margin (low_stability alone)
        mock_ood, mock_prob, mock_stab = self._mock_stages(
            min_d2=caution_d2 - 0.5,
            top_prob=min_prob + 0.05,
            margin=min_margin + 0.05,
            stability=min_stab - 0.01,
        )
        with patch.object(self.engine, "assess_ood", mock_ood), \
             patch.object(self.engine, "assess_probabilities", mock_prob), \
             patch.object(self.engine, "assess_stability", mock_stab):
            res_no_margin = self.engine.assess(self.valid_row, risk="balanced")

        self.assertEqual(res_no_margin["verdict"], "caution")
        self.assertIsNone(res_no_margin["measure_next"])

        # Case 3: recommend verdict
        mock_ood, mock_prob, mock_stab = self._mock_stages(
            min_d2=caution_d2 - 0.5,
            top_prob=min_prob + 0.05,
            margin=min_margin + 0.05,
            stability=min_stab + 0.05,
        )
        with patch.object(self.engine, "assess_ood", mock_ood), \
             patch.object(self.engine, "assess_probabilities", mock_prob), \
             patch.object(self.engine, "assess_stability", mock_stab):
            res_rec = self.engine.assess(self.valid_row, risk="balanced")

        self.assertEqual(res_rec["verdict"], "recommend")
        self.assertIsNone(res_rec["measure_next"])

        # Case 4: refuse verdict
        mock_ood, mock_prob, mock_stab = self._mock_stages(
            ood_flagged=True,
            min_d2=caution_d2 + 5.0,
        )
        with patch.object(self.engine, "assess_ood", mock_ood), \
             patch.object(self.engine, "assess_probabilities", mock_prob), \
             patch.object(self.engine, "assess_stability", mock_stab):
            res_ref = self.engine.assess(self.valid_row, risk="balanced")

        self.assertEqual(res_ref["verdict"], "refuse")
        self.assertIsNone(res_ref["measure_next"])

    def test_missing_class_stats_raises_filenotfound(self) -> None:
        """Verify load() raises FileNotFoundError if class_stats.json is absent."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            joblib.dump(joblib.load(self.model_dir / "ood_stats.joblib"), tmp_path / "ood_stats.joblib")
            with open(self.model_dir / "thresholds.json", "r") as f:
                with open(tmp_path / "thresholds.json", "w") as out:
                    out.write(f.read())
            with open(self.model_dir / "classes.json", "r") as f:
                with open(tmp_path / "classes.json", "w") as out:
                    out.write(f.read())
            joblib.dump(joblib.load(self.model_dir / "crop_pipeline_calibrated.joblib"), tmp_path / "crop_pipeline_calibrated.joblib")

            with self.assertRaises(FileNotFoundError) as ctx:
                TrustEngine.load(tmp_path)
            self.assertIn("class_stats.json", str(ctx.exception))


class TestTrustEngineCounterfactual(unittest.TestCase):
    """Unit test suite for TrustEngine counterfactual_report and batched candidate evaluation."""

    def setUp(self) -> None:
        """Initialize engine and locate validation dataset."""
        base_dir = Path(__file__).resolve().parent.parent
        model_candidates = [
            base_dir / "models",
            Path("backend/models"),
            Path("models"),
        ]
        self.model_dir = next(
            (p for p in model_candidates if (p / "crop_pipeline_calibrated.joblib").exists()),
            None,
        )
        self.assertIsNotNone(
            self.model_dir,
            "Could not locate valid model directory containing crop_pipeline_calibrated.joblib",
        )
        self.engine = TrustEngine.load(self.model_dir)

        repo_root = base_dir.parent
        data_path = repo_root / "data" / "Crop_recommendation.csv"
        if not data_path.exists():
            data_path = Path("data/Crop_recommendation.csv")
        self.assertTrue(data_path.exists(), f"Could not find {data_path}")

        import pandas as pd

        self.df_raw = pd.read_csv(data_path)

        val_idx_path = repo_root / "data" / "splits" / "val_idx.npy"
        if not val_idx_path.exists():
            val_idx_path = Path("data/splits/val_idx.npy")
        self.assertTrue(val_idx_path.exists(), f"Could not find {val_idx_path}")
        self.val_idx = np.load(val_idx_path)

    def test_batched_evaluation_matches_single_row(self) -> None:
        """Verify _evaluate_candidate_batch matches _compute_mahalanobis and assess_probabilities."""
        import pandas as pd

        rng = np.random.RandomState(42)
        # 20 random candidate vectors within physical bounds
        cand_matrix = rng.uniform(
            low=[0, 0, 0, -10, 0, 0, 0],
            high=[200, 200, 300, 60, 100, 14, 1000],
            size=(20, 7),
        )
        min_d2_batch, cand_top_crops = self.engine._evaluate_candidate_batch(cand_matrix)

        cand_df = pd.DataFrame(cand_matrix, columns=list(self.engine.REQUIRED_FEATURES))
        probs_batch = self.engine.pipeline.predict_proba(cand_df)

        for i in range(20):
            row_dict = {
                f: float(cand_matrix[i, j])
                for j, f in enumerate(self.engine.REQUIRED_FEATURES)
            }
            single_d2, _, _ = self.engine._compute_mahalanobis(row_dict)
            self.assertAlmostEqual(min_d2_batch[i], single_d2, places=8)

            single_prob_res = self.engine.assess_probabilities(row_dict)
            self.assertEqual(cand_top_crops[i], single_prob_res["top_crop"])

            paired = [
                (float(probs_batch[i, c]), self.engine.classes[c])
                for c in range(len(self.engine.classes))
            ]
            paired.sort(key=lambda x: (-x[0], x[1]))
            self.assertAlmostEqual(
                paired[0][0], single_prob_res["top_probability"], delta=1e-9
            )

    def test_counterfactual_decision_margins_caution_and_recommend(self) -> None:
        """Verify decision margins for caution (row 2049) and recommend (row 4) rows.

        Note: Row 2049 is dataset index 2049 (val_idx position 307).
              Row 4 is dataset index 4 (val_idx position 1).
        """
        from backend.noise import compute_feature_sigmas, load_noise_config

        refuse_thresh = float(
            self.engine.thresholds["profiles"]["balanced"]["ood_refuse_threshold"]
        )
        noise_cfg_all = load_noise_config(self.model_dir / "noise_model.json")
        profile_noise_cfg = noise_cfg_all["profiles"]["primary_1sigma"]

        for ds_idx in [2049, 4]:
            self.assertIn(ds_idx, self.val_idx, f"Row {ds_idx} must be in val_idx")
            row = {
                f: float(self.df_raw.iloc[ds_idx][f])
                for f in self.engine.REQUIRED_FEATURES
            }
            rep = self.engine.counterfactual_report(row, risk="balanced")
            orig_res = self.engine.assess(row, risk="balanced")
            orig_top = orig_res["top_crops"][0]["crop"]

            self.assertIsNotNone(rep["decision_margins"])
            self.assertIsNone(rep["flag_clearing"])

            sigmas_arr = compute_feature_sigmas(
                np.array([row[f] for f in self.engine.REQUIRED_FEATURES], dtype=float),
                profile_noise_cfg,
            )
            sigmas = {
                f: float(sigmas_arr[j])
                for j, f in enumerate(self.engine.REQUIRED_FEATURES)
            }

            for item in rep["decision_margins"]:
                if item["direction"] is not None:
                    feat = item["feature"]
                    cand = dict(row)
                    cand[feat] = cand[feat] + item["delta"]

                    # Must not be refused by Stage 2
                    ood = self.engine.assess_ood(cand, risk="balanced")
                    self.assertLessEqual(
                        ood["min_distance"],
                        refuse_thresh,
                        f"Candidate for {feat} exceeded refuse threshold",
                    )

                    # Top crop must equal resulting_crop and differ from original
                    prob = self.engine.assess_probabilities(cand)
                    self.assertEqual(prob["top_crop"], item["resulting_crop"])
                    self.assertNotEqual(prob["top_crop"], orig_top)

                    # Check minimality at step k-1
                    k = int(round(abs(item["delta_sigma"]) / 0.25))
                    dir_sign = +1 if item["direction"] == "up" else -1
                    if k > 1:
                        cand_prev = dict(row)
                        cand_prev[feat] = (
                            cand_prev[feat]
                            + dir_sign * (k - 1) * 0.25 * sigmas[feat]
                        )
                        min_b, max_b = self.engine.PHYSICAL_BOUNDS[feat]
                        in_b = bool(min_b <= cand_prev[feat] <= max_b)
                        ood_prev = self.engine.assess_ood(cand_prev, risk="balanced")
                        not_refused_prev = bool(
                            ood_prev["min_distance"] <= refuse_thresh
                        )
                        prob_prev = self.engine.assess_probabilities(cand_prev)
                        crop_differs_prev = bool(prob_prev["top_crop"] != orig_top)
                        self.assertFalse(
                            in_b and not_refused_prev and crop_differs_prev,
                            f"Step k-1 ({k-1}) for {feat} should not qualify",
                        )
                    else:
                        self.assertEqual(k, 1)

    def test_counterfactual_flag_clearing_refuse(self) -> None:
        """Verify flag_clearing for refused row 290 (dataset index 290, val_idx position 43)."""
        from backend.noise import compute_feature_sigmas, load_noise_config

        ds_idx = 290
        self.assertIn(ds_idx, self.val_idx, f"Row {ds_idx} must be in val_idx")
        row = {
            f: float(self.df_raw.iloc[ds_idx][f])
            for f in self.engine.REQUIRED_FEATURES
        }

        rep = self.engine.counterfactual_report(row, risk="balanced")
        self.assertEqual(rep["verdict"], "refuse")
        self.assertIsNone(rep["decision_margins"])
        self.assertIsNotNone(rep["flag_clearing"])

        noise_cfg_all = load_noise_config(self.model_dir / "noise_model.json")
        profile_noise_cfg = noise_cfg_all["profiles"]["primary_1sigma"]
        sigmas_arr = compute_feature_sigmas(
            np.array([row[f] for f in self.engine.REQUIRED_FEATURES], dtype=float),
            profile_noise_cfg,
        )
        sigmas = {
            f: float(sigmas_arr[j])
            for j, f in enumerate(self.engine.REQUIRED_FEATURES)
        }

        for item in rep["flag_clearing"]:
            if item["direction"] is not None:
                feat = item["feature"]
                cand = dict(row)
                cand[feat] = cand[feat] + item["delta"]

                # Candidate must clear Stage 2 OOD flag
                ood = self.engine.assess_ood(cand, risk="balanced")
                self.assertFalse(
                    ood["flagged"], f"Candidate for {feat} failed to clear flag"
                )

                # Step k-1 must still be flagged
                k = int(round(abs(item["delta_sigma"]) / 0.25))
                dir_sign = +1 if item["direction"] == "up" else -1
                cand_prev = dict(row)
                cand_prev[feat] = (
                    cand_prev[feat] + dir_sign * (k - 1) * 0.25 * sigmas[feat]
                )
                ood_prev = self.engine.assess_ood(cand_prev, risk="balanced")
                self.assertTrue(
                    ood_prev["flagged"], f"Step k-1 for {feat} should still be flagged"
                )

    def test_null_field_rules_by_verdict(self) -> None:
        """Verify decision_margins is None on refuse and flag_clearing is None on caution/recommend."""
        row_refuse = {
            f: float(self.df_raw.iloc[290][f])
            for f in self.engine.REQUIRED_FEATURES
        }
        rep_refuse = self.engine.counterfactual_report(row_refuse, risk="balanced")
        self.assertEqual(rep_refuse["verdict"], "refuse")
        self.assertIsNone(rep_refuse["decision_margins"])
        self.assertIsNotNone(rep_refuse["flag_clearing"])
        self.assertFalse(rep_refuse["near_tie"])

        row_caution = {
            f: float(self.df_raw.iloc[2049][f])
            for f in self.engine.REQUIRED_FEATURES
        }
        rep_caution = self.engine.counterfactual_report(row_caution, risk="balanced")
        self.assertEqual(rep_caution["verdict"], "caution")
        self.assertIsNotNone(rep_caution["decision_margins"])
        self.assertIsNone(rep_caution["flag_clearing"])
        self.assertTrue(rep_caution["near_tie"])

        row_rec = {
            f: float(self.df_raw.iloc[4][f])
            for f in self.engine.REQUIRED_FEATURES
        }
        rep_rec = self.engine.counterfactual_report(row_rec, risk="balanced")
        self.assertEqual(rep_rec["verdict"], "recommend")
        self.assertIsNotNone(rep_rec["decision_margins"])
        self.assertIsNone(rep_rec["flag_clearing"])
        self.assertFalse(rep_rec["near_tie"])

    def test_candidates_stay_within_physical_bounds(self) -> None:
        """Verify all reported candidate modifications strictly respect physical bounds."""
        extreme_row = {
            "N": 50.0,
            "P": 50.0,
            "K": 50.0,
            "temperature": 25.0,
            "humidity": 99.0,
            "ph": 0.1,
            "rainfall": 100.0,
        }
        rep = self.engine.counterfactual_report(extreme_row, risk="balanced")
        items = (
            rep["decision_margins"]
            if rep["decision_margins"] is not None
            else rep["flag_clearing"]
        )
        for item in items:
            min_b, max_b = self.engine.PHYSICAL_BOUNDS[item["feature"]]
            if item["to_value"] is not None:
                self.assertGreaterEqual(
                    item["to_value"],
                    min_b,
                    f"to_value {item['to_value']} below min bound {min_b} for {item['feature']}",
                )
                self.assertLessEqual(
                    item["to_value"],
                    max_b,
                    f"to_value {item['to_value']} above max bound {max_b} for {item['feature']}",
                )

    def test_counterfactual_does_not_mutate_row_or_change_assess(self) -> None:
        """Verify counterfactual_report does not mutate input row or alter assess results."""
        row = {
            f: float(self.df_raw.iloc[2049][f])
            for f in self.engine.REQUIRED_FEATURES
        }
        row_copy = dict(row)

        res_before = self.engine.assess(row, risk="balanced")
        rep = self.engine.counterfactual_report(row, risk="balanced")
        self.assertIsNotNone(rep)

        # Row must not be mutated
        self.assertEqual(row, row_copy)

        # assess output must remain identical
        res_after = self.engine.assess(row, risk="balanced")
        self.assertEqual(res_before, res_after)

    def test_counterfactual_determinism(self) -> None:
        """Verify counterfactual_report returns identical output across repeated calls."""
        row = {
            f: float(self.df_raw.iloc[2049][f])
            for f in self.engine.REQUIRED_FEATURES
        }
        rep1 = self.engine.counterfactual_report(row, risk="balanced")
        rep2 = self.engine.counterfactual_report(row, risk="balanced")
        self.assertEqual(rep1, rep2)


class TestOodBrowserExport(unittest.TestCase):
    """Unit test suite for frontend/src/ood_browser.json export verification (Task 11A)."""

    def setUp(self) -> None:
        """Initialize engine and locate ood_browser.json artifact."""
        base_dir = Path(__file__).resolve().parent.parent
        model_candidates = [
            base_dir / "models",
            Path("backend/models"),
            Path("models"),
        ]
        self.model_dir = next(
            (p for p in model_candidates if (p / "ood_stats.joblib").exists()),
            None,
        )
        self.assertIsNotNone(self.model_dir, "Could not locate backend/models directory")
        self.engine = TrustEngine.load(self.model_dir)

        repo_root = base_dir.parent
        json_candidates = [
            repo_root / "frontend" / "src" / "ood_browser.json",
            Path("frontend/src/ood_browser.json"),
        ]
        self.json_path = next((p for p in json_candidates if p.exists()), None)
        self.assertIsNotNone(self.json_path, "Could not locate frontend/src/ood_browser.json")

        with open(self.json_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)

    def test_schema_structure_and_symmetry(self) -> None:
        """Verify schema metadata, feature ordering, 22 crops, and precision matrix symmetry."""
        self.assertEqual(self.data["version"], 1)
        self.assertEqual(self.data["features"], list(self.engine.REQUIRED_FEATURES))
        self.assertEqual(self.data["crops"], list(self.engine.classes))
        self.assertEqual(len(self.data["crops"]), 22)

        # Check thresholds match engine thresholds.json
        for p in ["conservative", "balanced", "bold"]:
            expected_caution = float(self.engine.thresholds["profiles"][p]["ood_caution_threshold"])
            expected_refuse = float(self.engine.thresholds["profiles"][p]["ood_refuse_threshold"])
            self.assertAlmostEqual(self.data["thresholds"][p]["caution"], expected_caution, places=6)
            self.assertAlmostEqual(self.data["thresholds"][p]["refuse"], expected_refuse, places=6)

        # Check each precision matrix is 7x7 and symmetric within 1e-9
        for crop in self.data["crops"]:
            prec = self.data["class_precision"][crop]
            self.assertEqual(len(prec), 7, f"Crop {crop} precision matrix row count != 7")
            for r in range(7):
                self.assertEqual(len(prec[r]), 7, f"Crop {crop} row {r} col count != 7")
                for c in range(r + 1, 7):
                    diff = abs(prec[r][c] - prec[c][r])
                    self.assertLessEqual(
                        diff,
                        1e-9,
                        f"Crop {crop} precision asymmetry at ({r},{c}): {prec[r][c]} vs {prec[c][r]}",
                    )

            # Check mean length
            mean = self.data["class_mean"][crop]
            self.assertEqual(len(mean), 7, f"Crop {crop} mean length != 7")

    def test_pure_python_recomputation_mimicking_javascript(self) -> None:
        """Verify plain for-loop pure-Python computation matches engine to 1e-6 across 30 rows."""
        features = self.data["features"]
        scaler_mean = self.data["scaler_mean"]
        scaler_scale = self.data["scaler_scale"]
        crops = self.data["crops"]
        class_mean = self.data["class_mean"]
        class_precision = self.data["class_precision"]
        thresholds = self.data["thresholds"]

        rng = np.random.RandomState(42)
        low_bounds = [self.engine.PHYSICAL_BOUNDS[f][0] for f in features]
        high_bounds = [self.engine.PHYSICAL_BOUNDS[f][1] for f in features]
        test_matrix = rng.uniform(low=low_bounds, high=high_bounds, size=(30, 7))

        for i in range(30):
            row_dict = {f: float(test_matrix[i, j]) for j, f in enumerate(features)}

            # Plain Python standardization (mimicking JS)
            z = [(row_dict[f] - scaler_mean[j]) / scaler_scale[j] for j, f in enumerate(features)]

            min_d2_pure = float("inf")
            nearest_crop_pure = None

            for crop in crops:
                mu = class_mean[crop]
                prec = class_precision[crop]

                diff = [z[j] - mu[j] for j in range(7)]

                d2 = 0.0
                for r in range(7):
                    row_sum = 0.0
                    for c in range(7):
                        row_sum += prec[r][c] * diff[c]
                    d2 += diff[r] * row_sum

                if d2 < min_d2_pure:
                    min_d2_pure = d2
                    nearest_crop_pure = crop

            # Compare against real engine
            min_d2_engine, sorted_crops, _ = self.engine._compute_mahalanobis(row_dict)
            nearest_crop_engine = sorted_crops[0][0]

            self.assertAlmostEqual(min_d2_pure, min_d2_engine, delta=1e-6)
            self.assertEqual(nearest_crop_pure, nearest_crop_engine)

            # Compare flagged against engine.assess_ood
            for p in ["conservative", "balanced", "bold"]:
                refuse_th = thresholds[p]["refuse"]
                flagged_pure = bool(min_d2_pure > refuse_th)
                flagged_engine = self.engine.assess_ood(row_dict, risk=p)["flagged"]
                self.assertEqual(
                    flagged_pure,
                    flagged_engine,
                    f"Profile {p} flag mismatch at row {i}",
                )

    def test_test_cases_recomputed_match(self) -> None:
        """Verify each of the 6 test_cases in the JSON file matches recomputed engine values."""
        self.assertEqual(len(self.data["test_cases"]), 6)
        for i, tc in enumerate(self.data["test_cases"]):
            inp = tc["input"]
            min_d2_engine, sorted_crops, _ = self.engine._compute_mahalanobis(inp)
            nearest_crop_engine = sorted_crops[0][0]

            self.assertAlmostEqual(tc["min_d2"], min_d2_engine, places=6)
            self.assertEqual(tc["nearest_crop"], nearest_crop_engine)

            for p in ["conservative", "balanced", "bold"]:
                refuse_th = self.data["thresholds"][p]["refuse"]
                expected_flagged = bool(min_d2_engine > refuse_th)
                self.assertEqual(tc["flagged"][p], expected_flagged)

    def test_staleness_in_memory_rebuild_matches_disk(self) -> None:
        """Verify in-memory rebuild matches the exact content on disk (staleness check)."""
        import importlib.util

        repo_root = Path(__file__).resolve().parent.parent.parent
        script_candidates = [
            repo_root / "notebooks" / "04b_export_ood_browser.py",
            Path("notebooks/04b_export_ood_browser.py"),
        ]
        script_path = next((p for p in script_candidates if p.exists()), None)
        self.assertIsNotNone(script_path, "Could not locate notebooks/04b_export_ood_browser.py")

        spec = importlib.util.spec_from_file_location("export_ood_browser_mod", script_path)
        self.assertIsNotNone(spec)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        rebuilt_data = mod.build_ood_browser_data(engine=self.engine)
        rebuilt_bytes = json.dumps(rebuilt_data, separators=(",", ":")).encode("utf-8")

        with open(self.json_path, "rb") as f:
            disk_bytes = f.read()

        self.assertEqual(rebuilt_bytes, disk_bytes, "In-memory rebuild does not match disk content")


class TestWhyNotSecondCrop(unittest.TestCase):
    """Unit tests for TrustEngine.why_not_second_crop (Task 9C)."""

    @classmethod
    def setUpClass(cls) -> None:
        import pandas as pd

        cls.engine = TrustEngine.load("backend/models")
        cls.df = pd.read_csv("data/Crop_recommendation.csv")
        cls.val_idx = np.load("data/splits/val_idx.npy")
        with open("backend/models/class_stats.json", "r", encoding="utf-8") as f:
            cls.class_stats_data = json.load(f)

    def test_row_2049_rice_vs_jute_applicable(self) -> None:
        """Row 2049 (true jute, caution with low_margin) is applicable; top=rice, second=jute."""
        row = self.df.iloc[2049][list(self.engine.REQUIRED_FEATURES)].to_dict()
        res = self.engine.why_not_second_crop(row, risk="balanced")
        self.assertEqual(res["verdict"], "caution")
        self.assertEqual(res["top_crop"], "rice")
        self.assertEqual(res["second_crop"], "jute")
        self.assertTrue(res["applicable"])
        self.assertIsNotNone(res["features"])
        self.assertIsNotNone(res["message"])
        self.assertEqual(len(res["features"]), 2)
        feat_names = [f["feature"] for f in res["features"]]
        self.assertEqual(feat_names, ["N", "ph"])
        for f in res["features"]:
            self.assertEqual(f["direction"], "higher")
            self.assertGreater(f["score"], 0.0)
            feat_disp = self.engine.FEATURE_DISPLAY_NAMES.get(f["feature"], f["feature"])
            self.assertIn(feat_disp, res["message"])
        self.assertIn("These describe the training data, not agronomic advice.", res["note"])

    def test_row_838_lentil_vs_mothbeans_applicable(self) -> None:
        """Row 838 (true lentil, caution with low_margin) is applicable; top=lentil, second=mothbeans."""
        row = self.df.iloc[838][list(self.engine.REQUIRED_FEATURES)].to_dict()
        res = self.engine.why_not_second_crop(row, risk="balanced")
        self.assertEqual(res["verdict"], "caution")
        self.assertEqual(res["top_crop"], "lentil")
        self.assertEqual(res["second_crop"], "mothbeans")
        self.assertTrue(res["applicable"])
        self.assertIsNotNone(res["features"])
        self.assertIsNotNone(res["message"])
        self.assertEqual(len(res["features"]), 1)
        self.assertEqual(res["features"][0]["feature"], "rainfall")
        self.assertEqual(res["features"][0]["direction"], "within")
        self.assertIn("within", res["message"])
        self.assertIn("mothbeans", res["message"])
        self.assertIn("lentil", res["message"])

    def test_independent_recomputation_from_class_stats(self) -> None:
        """Independently recompute z-scores directly from class_stats.json and verify exact match."""
        row = self.df.iloc[2049][list(self.engine.REQUIRED_FEATURES)].to_dict()
        res = self.engine.why_not_second_crop(row, risk="balanced")
        top_crop = res["top_crop"]
        second_crop = res["second_crop"]

        cs = self.class_stats_data["class_stats"]
        for item in res["features"]:
            feat = item["feature"]
            x = float(row[feat])
            mean_top = float(cs[top_crop][feat]["mean"])
            sd_top = float(cs[top_crop][feat]["std"])
            mean_sec = float(cs[second_crop][feat]["mean"])
            sd_sec = float(cs[second_crop][feat]["std"])
            p05_sec = float(cs[second_crop][feat]["p05"])
            p95_sec = float(cs[second_crop][feat]["p95"])

            expected_z_top = (x - mean_top) / sd_top
            expected_z_sec = (x - mean_sec) / sd_sec
            expected_score = abs(expected_z_sec) - abs(expected_z_top)

            self.assertAlmostEqual(item["input_value"], x, places=6)
            self.assertAlmostEqual(item["second_mean"], mean_sec, places=6)
            self.assertAlmostEqual(item["second_p05"], p05_sec, places=6)
            self.assertAlmostEqual(item["second_p95"], p95_sec, places=6)
            self.assertAlmostEqual(item["z_top"], expected_z_top, places=6)
            self.assertAlmostEqual(item["z_second"], expected_z_sec, places=6)
            self.assertAlmostEqual(item["score"], expected_score, places=6)

    def test_ranking_by_score(self) -> None:
        """Features are sorted by score descending, with ties broken by REQUIRED_FEATURES order."""
        row = self.df.iloc[2049][list(self.engine.REQUIRED_FEATURES)].to_dict()
        res = self.engine.why_not_second_crop(row, risk="balanced")
        feats = res["features"]
        self.assertGreater(len(feats), 1)
        for i in range(len(feats) - 1):
            self.assertGreaterEqual(feats[i]["score"], feats[i + 1]["score"])

    def test_non_applicable_cases(self) -> None:
        """Non-applicable for recommend (row 4), refuse (row 290), and caution without low_margin (row 83)."""
        # Recommend: row 4
        row_4 = self.df.iloc[4][list(self.engine.REQUIRED_FEATURES)].to_dict()
        res_4 = self.engine.why_not_second_crop(row_4, risk="balanced")
        self.assertEqual(res_4["verdict"], "recommend")
        self.assertFalse(res_4["applicable"])
        self.assertIsNone(res_4["features"])
        self.assertIsNone(res_4["message"])

        # Refuse: row 290
        row_290 = self.df.iloc[290][list(self.engine.REQUIRED_FEATURES)].to_dict()
        res_290 = self.engine.why_not_second_crop(row_290, risk="balanced")
        self.assertEqual(res_290["verdict"], "refuse")
        self.assertFalse(res_290["applicable"])
        self.assertIsNone(res_290["features"])
        self.assertIsNone(res_290["message"])

        # Caution without low_margin: row 83 (low stability trigger only)
        row_83 = self.df.iloc[83][list(self.engine.REQUIRED_FEATURES)].to_dict()
        res_83 = self.engine.why_not_second_crop(row_83, risk="balanced")
        self.assertEqual(res_83["verdict"], "caution")
        self.assertFalse(res_83["applicable"])
        self.assertIsNone(res_83["features"])
        self.assertIsNone(res_83["message"])

    def test_assess_unchanged_before_and_after(self) -> None:
        """Calling why_not_second_crop does not mutate engine state or change assess output."""
        row = self.df.iloc[2049][list(self.engine.REQUIRED_FEATURES)].to_dict()
        assess_before = self.engine.assess(row, risk="balanced")
        _ = self.engine.why_not_second_crop(row, risk="balanced")
        assess_after = self.engine.assess(row, risk="balanced")

        self.assertEqual(assess_before["verdict"], assess_after["verdict"])
        self.assertEqual(assess_before["top_crops"], assess_after["top_crops"])
        self.assertEqual(assess_before["explanation"], assess_after["explanation"])
        self.assertEqual(assess_before["measure_next"], assess_after["measure_next"])
        self.assertEqual(assess_before["ood"], assess_after["ood"])

    def test_determinism(self) -> None:
        """Calling why_not_second_crop twice on the same row produces identical output."""
        row = self.df.iloc[2049][list(self.engine.REQUIRED_FEATURES)].to_dict()
        res1 = self.engine.why_not_second_crop(row, risk="balanced")
        res2 = self.engine.why_not_second_crop(row, risk="balanced")
        self.assertEqual(res1, res2)

    def test_valueerror_free_on_validation_rows(self) -> None:
        """why_not_second_crop runs without error across all 330 validation rows."""
        for idx in self.val_idx:
            row = self.df.iloc[idx][list(self.engine.REQUIRED_FEATURES)].to_dict()
            res = self.engine.why_not_second_crop(row, risk="balanced")
            self.assertIn("applicable", res)
            self.assertIn("verdict", res)
            self.assertIn("note", res)
            if res["applicable"]:
                self.assertIsNotNone(res["features"])
                self.assertIsNotNone(res["message"])
                self.assertLessEqual(len(res["features"]), 2)
            else:
                self.assertIsNone(res["features"])
                self.assertIsNone(res["message"])


if __name__ == "__main__":
    unittest.main()
