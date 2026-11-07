"""Unit tests for physical measurement noise model and perturbation functions.

Role 2: Trust Engine Engineer
Tests 1-sigma calculations, additive floors, physical clipping, reproducibility,
vectorized batch operations, and immutability.
"""

import unittest
from pathlib import Path
from typing import Any, Dict

import numpy as np

# Ensure robust imports whether test is invoked from repo root or inside backend/
try:
    from backend.noise import (
        FEATURE_NAMES,
        PHYSICAL_BOUNDS,
        apply_noise,
        compute_feature_sigmas,
        load_noise_config,
    )
except ModuleNotFoundError:
    import sys

    backend_dir = Path(__file__).resolve().parent.parent
    if str(backend_dir) not in sys.path:
        sys.path.insert(0, str(backend_dir))
    from noise import (
        FEATURE_NAMES,
        PHYSICAL_BOUNDS,
        apply_noise,
        compute_feature_sigmas,
        load_noise_config,
    )


class TestNoiseModel(unittest.TestCase):
    """Test suite for the approved physical measurement-noise implementation."""

    def setUp(self) -> None:
        """Set up canonical test fixtures."""
        self.config = load_noise_config()
        self.primary_cfg = self.config["profiles"]["primary_1sigma"]

        # Canonical baseline row: [N, P, K, temp, humidity, ph, rainfall]
        self.canonical_row = np.array(
            [90.0, 42.0, 43.0, 21.0, 82.0, 6.5, 203.0], dtype=np.float64
        )
        self.canonical_dict: Dict[str, float] = {
            "N": 90.0,
            "P": 42.0,
            "K": 43.0,
            "temperature": 21.0,
            "humidity": 82.0,
            "ph": 6.5,
            "rainfall": 203.0,
        }

    # 1. Configuration loading tests
    def test_load_noise_config_metadata(self) -> None:
        """Verify noise configuration file exists and contains approved metadata."""
        self.assertEqual(self.config["status"], "final_approved")
        self.assertEqual(self.config["default_profile"], "primary_1sigma")
        self.assertIn("Abhishek", self.config["approval"]["approved_by"])
        self.assertIn("primary_1sigma", self.config["profiles"])
        self.assertIn("mild_half_sigma", self.config["profiles"])

    def test_load_noise_config_missing_file_raises_error(self) -> None:
        """Verify FileNotFoundError on nonexistent configuration path."""
        with self.assertRaises(FileNotFoundError):
            load_noise_config("nonexistent/path/noise.json")

    # 2. 1-sigma calculations, relative scaling, and additive floors
    def test_relative_noise_and_additive_floors(self) -> None:
        """Verify relative feature sigmas scale with value and enforce floors."""
        # Case A: Standard high values (ratio dominates floor)
        # N=90 -> 10% = 9.0 (> floor 2.0)
        # P=42 -> 10% = 4.2 (> floor 2.0)
        # K=43 -> 10% = 4.3 (> floor 2.0)
        # Rainfall=203 -> 10% = 20.3 (> floor 5.0)
        sigmas_high = compute_feature_sigmas(self.canonical_row, self.primary_cfg)
        self.assertAlmostEqual(sigmas_high[0], 9.0)
        self.assertAlmostEqual(sigmas_high[1], 4.2)
        self.assertAlmostEqual(sigmas_high[2], 4.3)
        self.assertAlmostEqual(sigmas_high[6], 20.3)

        # Case B: Depleted / low values where additive floor must engage
        # N=10 -> 10% = 1.0 -> floor is 2.0 -> sigma = 2.0
        # Rainfall=20 -> 10% = 2.0 -> floor is 5.0 -> sigma = 5.0
        low_row = np.array([10.0, 12.0, 5.0, 25.0, 50.0, 7.0, 20.0])
        sigmas_low = compute_feature_sigmas(low_row, self.primary_cfg)
        self.assertEqual(sigmas_low[0], 2.0)  # N floor
        self.assertEqual(sigmas_low[1], 2.0)  # P floor (1.2 < 2.0)
        self.assertEqual(sigmas_low[2], 2.0)  # K floor (0.5 < 2.0)
        self.assertEqual(sigmas_low[6], 5.0)  # Rainfall floor (2.0 < 5.0)

    def test_zero_and_near_zero_handling(self) -> None:
        """Verify zero inputs do not cause zero-variance collapse."""
        zero_row = np.zeros(7, dtype=np.float64)
        sigmas_zero = compute_feature_sigmas(zero_row, self.primary_cfg)
        self.assertEqual(sigmas_zero[0], 2.0)  # N floor
        self.assertEqual(sigmas_zero[1], 2.0)  # P floor
        self.assertEqual(sigmas_zero[2], 2.0)  # K floor
        self.assertEqual(sigmas_zero[6], 5.0)  # Rainfall floor

    # 3. Absolute noise calculation
    def test_absolute_noise_features(self) -> None:
        """Verify temperature, humidity, and pH use fixed approved 1-sigma values."""
        # Row 1: normal conditions
        s1 = compute_feature_sigmas(self.canonical_row, self.primary_cfg)
        self.assertEqual(s1[3], 1.5)  # Temperature sigma
        self.assertEqual(s1[4], 5.0)  # Humidity sigma
        self.assertEqual(s1[5], 0.3)  # pH sigma

        # Row 2: extreme conditions (should have identical absolute sigmas)
        extreme_row = np.array([150.0, 100.0, 200.0, 45.0, 15.0, 3.5, 500.0])
        s2 = compute_feature_sigmas(extreme_row, self.primary_cfg)
        self.assertEqual(s2[3], 1.5)
        self.assertEqual(s2[4], 5.0)
        self.assertEqual(s2[5], 0.3)

    # 4. Reproducibility & seed test
    def test_seed_reproducibility(self) -> None:
        """Verify identical inputs and seeds return bitwise identical noisy copies."""
        res1 = apply_noise(self.canonical_row, n_samples=200, seed=42)
        res2 = apply_noise(self.canonical_row, n_samples=200, seed=42)
        np.testing.assert_array_equal(res1, res2)

        # Different seed produces different perturbations
        res3 = apply_noise(self.canonical_row, n_samples=200, seed=99)
        self.assertFalse(np.array_equal(res1, res3))

    # 5. Independent feature perturbations
    def test_independent_feature_perturbations(self) -> None:
        """Verify noise across features is uncorrelated (independent Gaussian noise)."""
        # Generate 5,000 noisy samples of a single row
        noisy_samples = apply_noise(self.canonical_row, n_samples=5000, seed=42)
        perturbations = noisy_samples - self.canonical_row

        # Compute empirical correlation matrix between feature perturbations
        corr_matrix = np.corrcoef(perturbations, rowvar=False)

        # Off-diagonal correlations should be very close to zero
        for i in range(len(FEATURE_NAMES)):
            for j in range(i + 1, len(FEATURE_NAMES)):
                self.assertLess(
                    abs(corr_matrix[i, j]),
                    0.06,
                    f"Features {FEATURE_NAMES[i]} and {FEATURE_NAMES[j]} appear correlated: {corr_matrix[i, j]}",
                )

    # 6. Vectorized / batch operations
    def test_batch_vectorized_2d_input(self) -> None:
        """Verify 2D batch inputs are handled efficiently and correctly."""
        batch = np.array(
            [
                [90.0, 42.0, 43.0, 21.0, 82.0, 6.5, 203.0],
                [80.0, 40.0, 40.0, 34.0, 80.0, 6.8, 190.0],
                [10.0, 15.0, 12.0, 18.0, 30.0, 5.5, 45.0],
            ]
        )
        # Case A: 1 sample per batch row
        res_single = apply_noise(batch, n_samples=1, seed=42)
        self.assertIsInstance(res_single, np.ndarray)
        self.assertEqual(res_single.shape, (3, 7))

        # Case B: Multiple samples per batch row
        res_multi = apply_noise(batch, n_samples=50, seed=42)
        self.assertEqual(res_multi.shape, (50, 3, 7))

    # 7. Physical clipping and clipping stats
    def test_physical_clipping_bounds(self) -> None:
        """Verify that noisy samples strictly respect physical bounds."""
        # Row with values very close to physical limits
        boundary_row = np.array([199.0, 1.0, 298.0, 59.5, 99.0, 13.9, 995.0])

        noisy_boundary, stats = apply_noise(
            boundary_row, n_samples=500, seed=42, return_clip_stats=True
        )

        # Verify all values stay within API physical bounds
        for idx, feat in enumerate(FEATURE_NAMES):
            low, high = PHYSICAL_BOUNDS[feat]
            self.assertTrue(np.all(noisy_boundary[:, idx] >= low))
            self.assertTrue(np.all(noisy_boundary[:, idx] <= high))

        # Verify clipping statistics were captured
        self.assertGreater(stats["total_clipped"], 0)
        self.assertIn("per_feature_clipped", stats)
        self.assertGreaterEqual(stats["per_feature_clipped"]["humidity"], 1)

    # 8. Immutability / no mutation of input
    def test_no_mutation_of_input(self) -> None:
        """Verify original input array or dictionary is never modified in-place."""
        orig_array = np.array([90.0, 42.0, 43.0, 21.0, 82.0, 6.5, 203.0])
        orig_copy = orig_array.copy()

        _ = apply_noise(orig_array, n_samples=100, seed=42)
        np.testing.assert_array_equal(orig_array, orig_copy)

        dict_copy = dict(self.canonical_dict)
        _ = apply_noise(self.canonical_dict, n_samples=1, seed=42)
        self.assertEqual(self.canonical_dict, dict_copy)

    # 9. Dictionary input handling
    def test_dict_input_handling(self) -> None:
        """Verify dictionary inputs return dictionary outputs when n_samples=1."""
        res_dict = apply_noise(self.canonical_dict, n_samples=1, seed=42)
        self.assertIsInstance(res_dict, dict)
        for feat in FEATURE_NAMES:
            self.assertIn(feat, res_dict)
            self.assertIsInstance(res_dict[feat], float)

    # 10. Secondary optional mild profile
    def test_optional_mild_profile(self) -> None:
        """Verify secondary 'mild_half_sigma' profile uses halved standard deviations."""
        sigmas_primary = compute_feature_sigmas(self.canonical_row, self.primary_cfg)
        mild_cfg = self.config["profiles"]["mild_half_sigma"]
        sigmas_mild = compute_feature_sigmas(self.canonical_row, mild_cfg)

        # Temperature: 1.5 primary -> 0.75 mild
        self.assertEqual(sigmas_mild[3], 0.75)
        # Humidity: 5.0 primary -> 2.5 mild
        self.assertEqual(sigmas_mild[4], 2.5)
        # Relative N: 9.0 primary -> 4.5 mild
        self.assertAlmostEqual(sigmas_mild[0], 4.5)


if __name__ == "__main__":
    unittest.main()
