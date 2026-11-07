"""Unit tests for TrustEngine core interface and Stage 2 OOD implementation.

Role 2: Trust Engine Engineer
Tests run directly via Python unittest or pytest without requiring FastAPI
or a running web server.
"""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
from typing import Any, Dict
import unittest

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
        """Verify assess() hands off OOD-passed inputs cleanly via StagesNotImplementedError."""
        with self.assertRaises(StagesNotImplementedError) as ctx:
            self.engine.assess(self.rice_familiar, risk="balanced")

        # Must have exact string message for Role 3 HTTP 501 handler
        self.assertEqual(str(ctx.exception), "Stages 3-5 not integrated yet")

        # Must inherit from standard NotImplementedError
        self.assertIsInstance(ctx.exception, NotImplementedError)

        # Must carry the verified Stage 2 diagnostics
        ood_res = ctx.exception.ood_result
        self.assertIsNotNone(ood_res)
        self.assertTrue(ood_res["eligible"])
        self.assertFalse(ood_res["flagged"])
        self.assertEqual(ood_res["nearest_known"][0], "rice")

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


if __name__ == "__main__":
    unittest.main()
