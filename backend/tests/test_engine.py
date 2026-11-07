"""Unit tests for TrustEngine core interface and stub implementation.

Role 2: Trust Engine Engineer
Tests run directly via Python unittest or pytest without requiring FastAPI
or a running web server.
"""

import json
import unittest
from pathlib import Path
from typing import Any, Dict

# Ensure robust imports whether test is invoked from repo root or inside backend/
try:
    from backend.trust_engine import TrustEngine
except ModuleNotFoundError:
    import sys

    backend_dir = Path(__file__).resolve().parent.parent
    if str(backend_dir) not in sys.path:
        sys.path.insert(0, str(backend_dir))
    from trust_engine import TrustEngine


class TestTrustEngineStub(unittest.TestCase):
    """Test suite for the initial TrustEngine stub implementation."""

    def setUp(self) -> None:
        """Set up standard test inputs."""
        self.engine = TrustEngine.load("models/")

        # Typical in-distribution rice-like conditions
        self.valid_recommend_row: Dict[str, Any] = {
            "N": 90.0,
            "P": 42.0,
            "K": 43.0,
            "temperature": 21.0,
            "humidity": 82.0,
            "ph": 6.5,
            "rainfall": 203.0,
        }

        # Boundary / ambiguous conditions (high temp & high rainfall)
        self.valid_caution_row: Dict[str, Any] = {
            "N": 80.0,
            "P": 40.0,
            "K": 40.0,
            "temperature": 34.0,
            "humidity": 80.0,
            "ph": 6.8,
            "rainfall": 190.0,
        }

        # Canonical OOD condition (high humidity with near-zero rainfall)
        self.valid_refuse_row: Dict[str, Any] = {
            "N": 90.0,
            "P": 42.0,
            "K": 43.0,
            "temperature": 25.0,
            "humidity": 92.0,
            "ph": 6.5,
            "rainfall": 25.0,
        }

    # 1. Engine initialization & import tests
    def test_engine_initialization_default(self) -> None:
        """Test default engine instantiation and load classmethod."""
        engine = TrustEngine()
        self.assertTrue(engine.is_stub)
        self.assertIsNone(engine.model_dir)

        loaded_engine = TrustEngine.load()
        self.assertTrue(loaded_engine.is_stub)
        self.assertEqual(loaded_engine.model_dir, Path("models/"))

    def test_engine_initialization_custom_dir(self) -> None:
        """Test engine initialization with custom path and string."""
        custom_path = "backend/models/"
        loaded = TrustEngine.load(custom_path)
        self.assertEqual(loaded.model_dir, Path(custom_path))

        loaded_none = TrustEngine.load(None)
        self.assertIsNone(loaded_none.model_dir)

    def test_independent_invocation_without_web_server(self) -> None:
        """Verify engine can be instantiated and invoked completely independently of FastAPI."""
        isolated_engine = TrustEngine.load()
        out = isolated_engine.assess(self.valid_recommend_row)
        self.assertIsInstance(out, dict)
        self.assertIn("verdict", out)
        self.assertEqual(out["verdict"], "recommend")

    # 2. Valid input acceptance tests
    def test_valid_input_acceptance_integer_values(self) -> None:
        """Verify the engine accepts integer values and converts them to floats."""
        int_row = {
            "N": 90,
            "P": 42,
            "K": 43,
            "temperature": 21,
            "humidity": 82,
            "ph": 6,
            "rainfall": 200,
        }
        result = self.engine.assess(int_row, risk="balanced")
        self.assertIsInstance(result, dict)
        self.assertEqual(result["verdict"], "recommend")

    def test_missing_features_raise_value_error(self) -> None:
        """Verify that omitting required features raises a clear ValueError."""
        incomplete_row = {
            "N": 90.0,
            "P": 42.0,
            # missing K, temperature, humidity, ph, rainfall
        }
        with self.assertRaises(ValueError) as ctx:
            self.engine.assess(incomplete_row)
        self.assertIn("Missing required feature(s)", str(ctx.exception))
        self.assertIn("K", str(ctx.exception))

    def test_non_numeric_features_raise_value_error(self) -> None:
        """Verify non-numeric feature values trigger a ValueError."""
        invalid_val_row = dict(self.valid_recommend_row)
        invalid_val_row["rainfall"] = "one hundred mm"
        with self.assertRaises(ValueError) as ctx:
            self.engine.assess(invalid_val_row)
        self.assertIn("must be numeric", str(ctx.exception))

    def test_non_dict_input_raises_value_error(self) -> None:
        """Verify non-dictionary rows trigger a ValueError."""
        with self.assertRaises(ValueError) as ctx:
            self.engine.assess([90, 42, 43, 21, 82, 6.5, 203])  # type: ignore
        self.assertIn("Input row must be a dictionary", str(ctx.exception))

    # 3. Verdict response structure tests
    def test_verdict_recommend_structure(self) -> None:
        """Verify the structure of a 'recommend' verdict response."""
        res = self.engine.assess(self.valid_recommend_row, risk="balanced")

        self.assertEqual(res["verdict"], "recommend")
        self.assertIsInstance(res["top_crops"], list)
        self.assertEqual(len(res["top_crops"]), 3)
        for item in res["top_crops"]:
            self.assertIn("crop", item)
            self.assertIn("probability", item)
            self.assertIsInstance(item["crop"], str)
            self.assertIsInstance(item["probability"], float)

        self.assertIn("ood", res)
        self.assertFalse(res["ood"]["flagged"])
        self.assertIsInstance(res["ood"]["percentile"], float)

        self.assertIsInstance(res["stability"], float)
        self.assertGreaterEqual(res["stability"], 0.0)
        self.assertLessEqual(res["stability"], 1.0)

        self.assertIsInstance(res["trust_score"], int)
        self.assertGreaterEqual(res["trust_score"], 0)
        self.assertLessEqual(res["trust_score"], 100)

        self.assertIsInstance(res["explanation"], str)
        self.assertIn("[STUB / ILLUSTRATIVE]", res["explanation"])
        self.assertIsInstance(res["model_version"], str)

    def test_verdict_caution_structure(self) -> None:
        """Verify the structure of a 'caution' verdict response."""
        res = self.engine.assess(self.valid_caution_row, risk="balanced")

        self.assertEqual(res["verdict"], "caution")
        self.assertIsInstance(res["top_crops"], list)
        self.assertEqual(len(res["top_crops"]), 3)

        self.assertIn("conformal_set", res)
        self.assertIsInstance(res["conformal_set"], list)
        self.assertTrue(len(res["conformal_set"]) >= 1)

        self.assertIn("ood", res)
        self.assertFalse(res["ood"]["flagged"])

        self.assertIsInstance(res["stability"], float)
        self.assertIsInstance(res["trust_score"], int)

        self.assertIn("measure_next", res)
        self.assertIsInstance(res["measure_next"], str)
        self.assertIn("Potassium", res["measure_next"])

        self.assertIsInstance(res["explanation"], str)
        self.assertIn("[STUB / ILLUSTRATIVE]", res["explanation"])

    def test_verdict_refuse_structure(self) -> None:
        """Verify the structure of a 'refuse' verdict response."""
        res = self.engine.assess(self.valid_refuse_row, risk="balanced")

        self.assertEqual(res["verdict"], "refuse")
        self.assertEqual(res["top_crops"], [])

        self.assertIn("ood", res)
        self.assertTrue(res["ood"]["flagged"])
        self.assertEqual(res["ood"]["percentile"], 99.8)
        self.assertIn("unusual_pair", res["ood"])
        self.assertEqual(res["ood"]["unusual_pair"], ["humidity", "rainfall"])
        self.assertIn("nearest_known", res["ood"])

        self.assertIsNone(res["stability"])
        self.assertEqual(res["trust_score"], 0)

        self.assertIsInstance(res["explanation"], str)
        self.assertIn("[STUB / ILLUSTRATIVE]", res["explanation"])
        self.assertIn("No recommendation is made", res["explanation"])

    # 4. Risk profile tests
    def test_supported_risk_profiles(self) -> None:
        """Verify that all three supported risk profiles execute successfully."""
        for profile in ("conservative", "balanced", "bold"):
            res = self.engine.assess(self.valid_recommend_row, risk=profile)
            self.assertIn(res["verdict"], ("recommend", "caution", "refuse"))

    def test_conservative_profile_boundary(self) -> None:
        """Verify conservative profile raises caution on low rainfall boundary."""
        borderline_row = dict(self.valid_recommend_row)
        borderline_row["rainfall"] = 65.0  # low rainfall below conservative threshold

        # Balanced accepts this as recommend
        res_balanced = self.engine.assess(borderline_row, risk="balanced")
        self.assertEqual(res_balanced["verdict"], "recommend")

        # Conservative profile triggers caution
        res_conservative = self.engine.assess(borderline_row, risk="conservative")
        self.assertEqual(res_conservative["verdict"], "caution")

    # 5. Invalid risk profile handling
    def test_invalid_risk_profile_raises_error(self) -> None:
        """Verify that unsupported risk profiles raise a descriptive ValueError."""
        invalid_profiles = ["aggressive", "risky", "none", "", 123]  # type: ignore
        for profile in invalid_profiles:
            with self.assertRaises(ValueError) as ctx:
                self.engine.assess(self.valid_recommend_row, risk=profile)  # type: ignore
            self.assertIn("Invalid risk profile", str(ctx.exception))

    # 6. JSON serializability test
    def test_json_serializability(self) -> None:
        """Verify that all response variants can be serialized directly into JSON."""
        test_rows = [
            (self.valid_recommend_row, "balanced"),
            (self.valid_caution_row, "balanced"),
            (self.valid_refuse_row, "balanced"),
            (self.valid_caution_row, "bold"),
            (self.valid_recommend_row, "conservative"),
        ]
        for row, risk in test_rows:
            response = self.engine.assess(row, risk=risk)
            # Must serialize without raising TypeError or ValueError
            serialized = json.dumps(response)
            self.assertIsInstance(serialized, str)

            # Round-trip deserialization check
            deserialized = json.loads(serialized)
            self.assertEqual(deserialized["verdict"], response["verdict"])
            self.assertEqual(deserialized["trust_score"], response["trust_score"])

    # 7. Determinism test
    def test_deterministic_responses(self) -> None:
        """Verify repeated calls with the same input produce identical responses."""
        res1 = self.engine.assess(self.valid_caution_row, risk="balanced")
        res2 = self.engine.assess(self.valid_caution_row, risk="balanced")
        self.assertEqual(res1, res2)

        refuse1 = self.engine.assess(self.valid_refuse_row, risk="conservative")
        refuse2 = self.engine.assess(self.valid_refuse_row, risk="conservative")
        self.assertEqual(refuse1, refuse2)


if __name__ == "__main__":
    unittest.main()
