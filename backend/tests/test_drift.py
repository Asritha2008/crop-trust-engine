"""Unit tests for backend/drift.py (Role 2: Drift Monitoring Helper).

Tests cover:
  - Population Stability Index (PSI) formula, self-comparison, shift detection, edge cases.
  - Repeated/discrete values and duplicate quantile handling.
  - Safe request log reading (missing, empty, blank, malformed, null stability, rotation order).
  - Flag rate computation and edge cases.
  - Feature PSI calculation and threshold gating.
"""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from backend.drift import (
    FEATURES,
    MIN_ROWS,
    feature_psi,
    flag_rate,
    psi,
    read_request_log,
)


class TestDriftPSI(unittest.TestCase):
    """Test suite for the core psi() function."""

    def test_psi_self(self) -> None:
        """PSI of an array against itself is essentially 0 (below 0.01)."""
        rng = np.random.default_rng(42)
        arr = rng.normal(loc=20.0, scale=5.0, size=500)
        res = psi(arr, arr)
        self.assertIsNotNone(res)
        self.assertLess(res, 0.01)
        self.assertAlmostEqual(res, 0.0, places=6)

    def test_psi_clearly_shifted(self) -> None:
        """PSI of a clearly shifted array is well above 0.25."""
        rng = np.random.default_rng(42)
        train = rng.normal(loc=0.0, scale=1.0, size=1000)
        live = rng.normal(loc=2.5, scale=1.0, size=1000)
        res = psi(train, live)
        self.assertIsNotNone(res)
        self.assertGreater(res, 0.25)

    def test_psi_empty_and_nan(self) -> None:
        """PSI returns None for empty input, None, or all-NaN input."""
        valid = [1.0, 2.0, 3.0, 4.0, 5.0]
        self.assertIsNone(psi([], valid))
        self.assertIsNone(psi(valid, []))
        self.assertIsNone(psi(None, valid))
        self.assertIsNone(psi(valid, None))
        self.assertIsNone(psi([np.nan, np.inf], valid))
        self.assertIsNone(psi(valid, [np.nan, -np.inf]))
        self.assertIsNone(psi([np.nan, np.nan], [np.nan, np.nan]))

    def test_psi_heavily_repeated_values(self) -> None:
        """Heavily repeated values (e.g. 70% zeros) must not crash and return a finite float."""
        rng = np.random.default_rng(42)
        # 70% zeros, 30% continuous positive values
        train = np.concatenate([np.zeros(700), rng.uniform(1.0, 10.0, 300)])
        live_same = np.concatenate([np.zeros(700), rng.uniform(1.0, 10.0, 300)])
        live_shift = np.concatenate([np.zeros(200), rng.uniform(1.0, 10.0, 800)])

        raw_edges = np.percentile(train, np.linspace(0, 100, 11))
        has_duplicate_edges = len(raw_edges) != len(np.unique(raw_edges))
        self.assertTrue(has_duplicate_edges, "Expected duplicate quantile edges from 70% ties")

        res_same = psi(train, live_same)
        res_shift = psi(train, live_shift)

        print(f"\n[test_psi_heavily_repeated_values] Duplicate quantile edges present: {has_duplicate_edges}")
        print(f"[test_psi_heavily_repeated_values] Raw edges: {raw_edges}")
        print(f"[test_psi_heavily_repeated_values] PSI (same distribution): {res_same}")
        print(f"[test_psi_heavily_repeated_values] PSI (shifted distribution): {res_shift}")

        self.assertIsNotNone(res_same)
        self.assertIsNotNone(res_shift)
        self.assertTrue(np.isfinite(res_same))
        self.assertTrue(np.isfinite(res_shift))
        self.assertLess(res_same, 0.05)
        self.assertGreater(res_shift, 0.25)


class TestReadRequestLog(unittest.TestCase):
    """Test suite for read_request_log()."""

    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmpdir.name)

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    def test_missing_file(self) -> None:
        """Missing file contributes nothing, does not crash, and returns status 'no_data'."""
        missing = self.tmp_path / "non_existent.log"
        res = read_request_log(missing, include_rotated=True)
        self.assertEqual(res["status"], "no_data")
        self.assertEqual(res["rows"], [])
        self.assertEqual(res["n_rows"], 0)
        self.assertEqual(res["n_bad_lines"], 0)
        self.assertEqual(res["files_read"], [])

    def test_read_request_log_none(self) -> None:
        """Passing path=None returns status 'no_data' with 0 rows and no exception."""
        res = read_request_log(None)
        self.assertEqual(res["status"], "no_data")
        self.assertEqual(res["rows"], [])
        self.assertEqual(res["n_rows"], 0)
        self.assertEqual(res["n_bad_lines"], 0)
        self.assertEqual(res["files_read"], [])

    def test_invalid_utf8_bytes_preserves_valid_lines(self) -> None:
        """Invalid UTF-8 bytes on one line are replaced; valid lines around it are preserved."""
        log_file = self.tmp_path / "corrupt_bytes.log"
        content_bytes = (
            b'{"id": 1, "verdict": "recommend"}\n'
            b'\xff\xfe\xfd invalid utf-8 byte sequence\n'
            b'{"id": 2, "verdict": "caution"}\n'
        )
        log_file.write_bytes(content_bytes)

        res = read_request_log(log_file, include_rotated=False)
        self.assertEqual(res["status"], "ok")
        self.assertEqual(res["n_rows"], 2)
        self.assertEqual(res["n_bad_lines"], 1)
        self.assertEqual([r["id"] for r in res["rows"]], [1, 2])
        self.assertEqual(len(res["files_read"]), 1)

    def test_empty_file(self) -> None:
        """Empty file contributes 0 rows, returns status 'no_data'."""
        empty = self.tmp_path / "empty.log"
        empty.write_text("", encoding="utf-8")
        res = read_request_log(empty, include_rotated=False)
        self.assertEqual(res["status"], "no_data")
        self.assertEqual(res["n_rows"], 0)
        self.assertEqual(len(res["files_read"]), 1)

    def test_blank_lines(self) -> None:
        """Blank lines are skipped and do not increment n_bad_lines."""
        log_file = self.tmp_path / "blank.log"
        log_file.write_text("\n   \n\t\n", encoding="utf-8")
        res = read_request_log(log_file, include_rotated=False)
        self.assertEqual(res["status"], "no_data")
        self.assertEqual(res["n_rows"], 0)
        self.assertEqual(res["n_bad_lines"], 0)

    def test_malformed_lines(self) -> None:
        """Malformed JSON or non-object JSON increments n_bad_lines without crashing."""
        log_file = self.tmp_path / "malformed.log"
        content = (
            '{"verdict": "recommend", "N": 50}\n'
            'NOT_JSON_LINE\n'
            '{"verdict": "caution", "N": 60}\n'
            '[1, 2, 3]\n'
            '"just a string"\n'
        )
        log_file.write_text(content, encoding="utf-8")
        res = read_request_log(log_file, include_rotated=False)
        self.assertEqual(res["status"], "ok")
        self.assertEqual(res["n_rows"], 2)
        self.assertEqual(res["n_bad_lines"], 3)

    def test_stability_null(self) -> None:
        """JSON null in stability is kept as Python None, not 0."""
        log_file = self.tmp_path / "stability.log"
        log_file.write_text('{"verdict": "refuse", "stability": null}\n', encoding="utf-8")
        res = read_request_log(log_file, include_rotated=False)
        self.assertEqual(res["status"], "ok")
        self.assertEqual(len(res["rows"]), 1)
        self.assertIsNone(res["rows"][0]["stability"])

    def test_rotated_file_order(self) -> None:
        """Rotated file (.1) is read before base file, preserving chronological time order."""
        base_log = self.tmp_path / "app.log"
        rotated_log = self.tmp_path / "app.log.1"

        rotated_log.write_text('{"id": 1, "verdict": "recommend"}\n{"id": 2, "verdict": "caution"}\n', encoding="utf-8")
        base_log.write_text('{"id": 3, "verdict": "recommend"}\n{"id": 4, "verdict": "refuse"}\n', encoding="utf-8")

        res = read_request_log(base_log, include_rotated=True)
        self.assertEqual(res["status"], "ok")
        self.assertEqual(res["n_rows"], 4)
        ids = [r["id"] for r in res["rows"]]
        self.assertEqual(ids, [1, 2, 3, 4])
        self.assertEqual(len(res["files_read"]), 2)

    def test_str_and_path_inputs(self) -> None:
        """Accepts both string and pathlib.Path instances without distinction."""
        log_file = self.tmp_path / "input_test.log"
        log_file.write_text('{"verdict": "recommend"}\n', encoding="utf-8")

        res_path = read_request_log(log_file, include_rotated=False)
        res_str = read_request_log(str(log_file), include_rotated=False)

        self.assertEqual(res_path["n_rows"], 1)
        self.assertEqual(res_str["n_rows"], 1)
        self.assertEqual(res_path["rows"], res_str["rows"])


class TestFlagRate(unittest.TestCase):
    """Test suite for flag_rate()."""

    def test_empty_rows(self) -> None:
        """Empty rows return status 'no_data', rate None, and 0 counts."""
        res = flag_rate([])
        self.assertEqual(res["status"], "no_data")
        self.assertIsNone(res["flag_rate"])
        self.assertEqual(res["total"], 0)
        self.assertEqual(res["counts"]["recommend"], 0)
        self.assertEqual(res["counts"]["caution"], 0)
        self.assertEqual(res["counts"]["refuse"], 0)

    def test_counts_and_rate(self) -> None:
        """Correctly computes refusal share and breakdown counts."""
        rows = [
            {"verdict": "recommend"},
            {"verdict": "recommend"},
            {"verdict": "caution"},
            {"verdict": "refuse"},
        ]
        res = flag_rate(rows)
        self.assertEqual(res["status"], "ok")
        self.assertEqual(res["total"], 4)
        self.assertEqual(res["counts"]["recommend"], 2)
        self.assertEqual(res["counts"]["caution"], 1)
        self.assertEqual(res["counts"]["refuse"], 1)
        self.assertAlmostEqual(res["flag_rate"], 0.25, places=6)


class TestFeaturePSI(unittest.TestCase):
    """Test suite for feature_psi()."""

    def setUp(self) -> None:
        rng = np.random.default_rng(42)
        self.reference = {
            feat: rng.normal(loc=50.0, scale=10.0, size=500)
            for feat in FEATURES
        }

    def test_not_enough_data_below_min_rows(self) -> None:
        """Returns status 'not_enough_data' and None for all feature PSIs when n_live < MIN_ROWS."""
        rows = [{"N": 50.0, "P": 50.0, "K": 50.0, "temperature": 25.0, "humidity": 80.0, "ph": 6.5, "rainfall": 100.0}] * 15
        self.assertLess(len(rows), MIN_ROWS)

        res = feature_psi(rows, self.reference)
        self.assertEqual(res["status"], "not_enough_data")
        self.assertEqual(res["n_live"], 15)
        self.assertEqual(res["min_rows"], MIN_ROWS)
        for feat in FEATURES:
            self.assertIsNone(res["psi"][feat])

    def test_ok_at_or_above_min_rows(self) -> None:
        """Returns status 'ok' and computes finite PSI when n_live >= MIN_ROWS."""
        rng = np.random.default_rng(42)
        rows = [
            {feat: float(rng.normal(loc=50.0, scale=10.0)) for feat in FEATURES}
            for _ in range(35)
        ]
        self.assertGreaterEqual(len(rows), MIN_ROWS)

        res = feature_psi(rows, self.reference)
        self.assertEqual(res["status"], "ok")
        self.assertEqual(res["n_live"], 35)
        for feat in FEATURES:
            self.assertIsNotNone(res["psi"][feat])
            self.assertTrue(np.isfinite(res["psi"][feat]))

    def test_shifted_feature_has_larger_psi(self) -> None:
        """A shifted feature reports higher PSI than an unshifted feature."""
        rng = np.random.default_rng(42)
        rows = []
        for _ in range(100):
            r = {feat: float(rng.normal(loc=50.0, scale=10.0)) for feat in FEATURES}
            # Severe shift in temperature (+30 units)
            r["temperature"] = float(rng.normal(loc=80.0, scale=10.0))
            rows.append(r)

        res = feature_psi(rows, self.reference)
        self.assertEqual(res["status"], "ok")
        psi_n = res["psi"]["N"]
        psi_temp = res["psi"]["temperature"]

        self.assertIsNotNone(psi_n)
        self.assertIsNotNone(psi_temp)
        self.assertGreater(psi_temp, 0.50)
        self.assertGreater(psi_temp, psi_n)

    def test_missing_feature_in_reference(self) -> None:
        """Feature missing from reference returns None while remaining features compute normally."""
        rng = np.random.default_rng(42)
        partial_reference = {
            feat: rng.normal(loc=50.0, scale=10.0, size=200)
            for feat in FEATURES if feat != "ph"
        }
        rows = [
            {feat: float(rng.normal(loc=50.0, scale=10.0)) for feat in FEATURES}
            for _ in range(35)
        ]

        res = feature_psi(rows, partial_reference)
        self.assertEqual(res["status"], "ok")
        self.assertIsNone(res["psi"]["ph"])
        self.assertIsNotNone(res["psi"]["N"])
        self.assertIsNotNone(res["psi"]["temperature"])

    def test_reference_none_list_empty_dict(self) -> None:
        """feature_psi with reference=None, a list, and {} does not raise; None/list return 'no_reference'."""
        rows = [
            {"N": 50.0, "P": 50.0, "K": 50.0, "temperature": 25.0, "humidity": 80.0, "ph": 6.5, "rainfall": 100.0}
            for _ in range(35)
        ]

        # 1. reference is None -> status 'no_reference'
        res_none = feature_psi(rows, None)
        self.assertEqual(res_none["status"], "no_reference")
        self.assertEqual(res_none["n_live"], 35)
        for feat in FEATURES:
            self.assertIsNone(res_none["psi"][feat])

        # 2. reference is a list -> status 'no_reference'
        res_list = feature_psi(rows, ["not", "a", "dict"])  # type: ignore
        self.assertEqual(res_list["status"], "no_reference")
        self.assertEqual(res_list["n_live"], 35)
        for feat in FEATURES:
            self.assertIsNone(res_list["psi"][feat])

        # 3. reference is empty dict {} -> status 'ok', but every feature missing yields None
        res_empty = feature_psi(rows, {})
        self.assertEqual(res_empty["status"], "ok")
        self.assertEqual(res_empty["n_live"], 35)
        for feat in FEATURES:
            self.assertIsNone(res_empty["psi"][feat])

    def test_rows_as_tuple(self) -> None:
        """feature_psi accepts rows as a tuple without error."""
        rows_tuple = tuple(
            {"N": 50.0, "P": 50.0, "K": 50.0, "temperature": 25.0, "humidity": 80.0, "ph": 6.5, "rainfall": 100.0}
            for _ in range(35)
        )
        res = feature_psi(rows_tuple, self.reference)
        self.assertEqual(res["status"], "ok")
        self.assertEqual(res["n_live"], 35)
        for feat in FEATURES:
            self.assertIsNotNone(res["psi"][feat])

    def test_nested_input_keys(self) -> None:
        """Feature extraction handles inputs nested under 'inputs' or 'features' key."""
        rng = np.random.default_rng(42)
        rows = [
            {"inputs": {feat: float(rng.normal(loc=50.0, scale=10.0)) for feat in FEATURES}}
            for _ in range(35)
        ]
        res = feature_psi(rows, self.reference)
        self.assertEqual(res["status"], "ok")
        for feat in FEATURES:
            self.assertIsNotNone(res["psi"][feat])

    def test_no_function_raises(self) -> None:
        """Ensure no function raises on corrupt or unusual inputs."""
        # psi on corrupt inputs
        self.assertIsNone(psi("not an array", "neither"))
        self.assertIsNone(psi([object()], [object()]))

        # read_request_log on invalid paths
        res_log = read_request_log(Path("/invalid/path/that/does/not/exist"))
        self.assertEqual(res_log["status"], "no_data")

        # flag_rate on malformed rows
        res_flag = flag_rate([None, "str", 123, {}])  # type: ignore
        self.assertEqual(res_flag["status"], "ok")

        # feature_psi on empty or malformed reference/rows
        res_psi = feature_psi([{"N": "not_a_number"}], {})
        self.assertEqual(res_psi["status"], "not_enough_data")


if __name__ == "__main__":
    unittest.main()
