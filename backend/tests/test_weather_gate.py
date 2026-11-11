"""Unit tests for the generic OOD gate and weather-only detector.

Role 2: Trust Engine Engineer.
Tests:
  (a) Fitting with all 7 features reproduces the current engine's OOD scores within tolerance.
  (b) The weather-only gate flags a clearly impossible weather combination and accepts typical rows.
  (c) The gate runs in pure Python without a web server.
  (d) Thresholds are strictly ordered across risk profiles:
      Conservative caution <= Balanced caution <= Bold caution, and
      Conservative refuse <= Balanced refuse <= Bold refuse.
  (e) Arbitrary feature subsets are supported by the fitting function.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np
import pandas as pd
import pytest

from backend.ood_gate import (
    OODGate,
    compute_oof_scores,
    compute_threshold_profiles,
    fit_mahalanobis_gate,
    score_mahalanobis,
)
from backend.trust_engine import TrustEngine

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_PATH = REPO_ROOT / "data" / "Crop_recommendation.csv"
TRAIN_IDX_PATH = REPO_ROOT / "data" / "splits" / "train_idx.npy"
MODELS_DIR = REPO_ROOT / "backend" / "models"


@pytest.fixture(scope="module")
def train_data():
    """Load training data (train_idx only, test set untouched)."""
    df = pd.read_csv(DATA_PATH)
    train_idx = np.load(TRAIN_IDX_PATH)
    return df, train_idx


@pytest.fixture(scope="module")
def weather_gate():
    """Load the serialized weather-only OOD gate."""
    stats_path = MODELS_DIR / "ood_stats_weather.joblib"
    thresholds_path = MODELS_DIR / "thresholds_weather.json"
    return OODGate.load(stats_path, thresholds_path)


class TestWeatherGate:
    """Test suite for weather-only OOD gate and generic detector fitting."""

    def test_reproduction_on_seven_features(self, train_data) -> None:
        """(a) Fitting with all 7 features reproduces current engine OOD scores within tolerance."""
        df, train_idx = train_data
        canonical_features = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]

        X_train = df[canonical_features].values[train_idx]
        y_train = df["label"].values[train_idx]

        # 1. Fit generic gate on all 7 features
        fitted_7d = fit_mahalanobis_gate(
            X=X_train,
            y=y_train,
            feature_names=canonical_features,
            seed=42,
        )

        # 2. Load existing production engine
        engine = TrustEngine.load(MODELS_DIR)

        # 3. Test on 3 diverse sample inputs
        test_rows = [
            # Familiar rice-like conditions
            {
                "N": 90.0, "P": 42.0, "K": 43.0,
                "temperature": 21.0, "humidity": 82.0, "ph": 6.5, "rainfall": 203.0,
            },
            # Unusual rice pair
            {
                "N": 80.0, "P": 47.0, "K": 40.0,
                "temperature": 23.7, "humidity": 86.0, "ph": 6.4, "rainfall": 125.0,
            },
            # Lentil-chickpea midpoint
            {
                "N": 29.43, "P": 68.08, "K": 49.67,
                "temperature": 21.69, "humidity": 40.83, "ph": 7.13, "rainfall": 62.87,
            },
        ]

        for row in test_rows:
            min_d2_engine, sorted_crops_engine, _ = engine._compute_mahalanobis(row)
            min_d2_gate, sorted_crops_gate, _ = score_mahalanobis(row, fitted_7d)

            assert np.isclose(min_d2_gate, min_d2_engine, atol=1e-5), (
                f"Score mismatch: gate={min_d2_gate:.6f}, engine={min_d2_engine:.6f}"
            )
            assert sorted_crops_gate[0][0] == sorted_crops_engine[0][0], (
                f"Nearest crop mismatch: gate={sorted_crops_gate[0][0]}, engine={sorted_crops_engine[0][0]}"
            )

    def test_weather_gate_accepts_typical_and_flags_impossible(
        self, train_data, weather_gate: OODGate
    ) -> None:
        """(b) Weather-only gate accepts real training typical rows and flags impossible combinations."""
        df, train_idx = train_data
        weather_features = ["temperature", "humidity", "rainfall"]

        # 1. Typical weather example: real centroid mean of rice from training data
        rice_mask = (df["label"].values[train_idx] == "rice")
        rice_weather_mean = df[weather_features].values[train_idx][rice_mask].mean(axis=0)
        typical_weather: Dict[str, float] = {
            "temperature": float(rice_weather_mean[0]),
            "humidity": float(rice_weather_mean[1]),
            "rainfall": float(rice_weather_mean[2]),
        }

        # Verify typical row is accepted across all risk profiles
        res_bal = weather_gate.assess_ood(typical_weather, risk="balanced")
        assert not res_bal["flagged"], f"Typical row was unexpectedly flagged: {res_bal}"
        assert res_bal["eligible"] is True
        assert res_bal["min_distance"] < res_bal["caution_threshold"]
        assert res_bal["percentile"] < 50.0
        assert res_bal["nearest_known"][0] == "rice"

        # 2. Impossible weather example: 50C extreme heat + 95% sauna humidity + 5mm drought
        impossible_weather: Dict[str, float] = {
            "temperature": 50.0,
            "humidity": 95.0,
            "rainfall": 5.0,
        }

        # Verify impossible row is flagged across all risk profiles
        for risk in ("conservative", "balanced", "bold"):
            res_imp = weather_gate.assess_ood(impossible_weather, risk=risk)
            assert res_imp["flagged"] is True, f"Impossible row not flagged for risk={risk}: {res_imp}"
            assert res_imp["eligible"] is False
            assert res_imp["min_distance"] > res_imp["threshold"]
            assert res_imp["percentile"] >= 99.0

    def test_gate_runs_without_web_server(self, weather_gate: OODGate) -> None:
        """(c) OODGate operates purely in Python without web framework or server dependencies."""
        sample_input = {
            "temperature": 25.0,
            "humidity": 75.0,
            "rainfall": 150.0,
        }
        res = weather_gate.assess_ood(sample_input, risk="balanced")
        assert isinstance(res, dict)
        assert "eligible" in res
        assert "flagged" in res
        assert "min_distance" in res
        assert "threshold" in res
        assert "caution_threshold" in res
        assert "nearest_known" in res

    def test_threshold_profile_ordering(self, weather_gate: OODGate) -> None:
        """Test 5: Thresholds are ordered across profiles:
        Conservative caution <= Balanced caution <= Bold caution,
        and Conservative refuse <= Balanced refuse <= Bold refuse.
        """
        profiles = weather_gate.thresholds["profiles"]
        c_caution = profiles["conservative"]["ood_caution_threshold"]
        b_caution = profiles["balanced"]["ood_caution_threshold"]
        bold_caution = profiles["bold"]["ood_caution_threshold"]

        c_refuse = profiles["conservative"]["ood_refuse_threshold"]
        b_refuse = profiles["balanced"]["ood_refuse_threshold"]
        bold_refuse = profiles["bold"]["ood_refuse_threshold"]

        # Check caution order across profiles
        assert c_caution <= b_caution <= bold_caution, (
            f"Caution thresholds not monotonically ordered: {c_caution} <= {b_caution} <= {bold_caution}"
        )

        # Check refuse order across profiles
        assert c_refuse <= b_refuse <= bold_refuse, (
            f"Refusal thresholds not monotonically ordered: {c_refuse} <= {b_refuse} <= {bold_refuse}"
        )

        # Check within-profile caution < refuse
        for pname in ("conservative", "balanced", "bold"):
            p = profiles[pname]
            assert p["ood_caution_threshold"] < p["ood_refuse_threshold"], (
                f"Profile {pname} has caution >= refuse: {p}"
            )

    def test_arbitrary_feature_subsets(self, train_data) -> None:
        """Fitting function accepts any arbitrary feature subset."""
        df, train_idx = train_data

        # 2-feature subset
        sub2 = ["temperature", "rainfall"]
        X2 = df[sub2].values[train_idx]
        y = df["label"].values[train_idx]

        model2 = fit_mahalanobis_gate(X2, y, feature_names=sub2)
        assert model2["metadata"]["features"] == sub2
        assert model2["scaler"]["mean"].shape == (2,)

        min_d2, sorted_c, z = score_mahalanobis({"temperature": 25.0, "rainfall": 100.0}, model2)
        assert min_d2 >= 0.0
        assert z.shape == (2,)
        assert len(sorted_c) == 22

    def test_missing_feature_raises_error(self, weather_gate: OODGate) -> None:
        """Missing required feature raises ValueError."""
        incomplete_row = {"temperature": 25.0, "humidity": 75.0}  # rainfall missing
        with pytest.raises(ValueError, match="missing required features"):
            weather_gate.score(incomplete_row)

    def test_unsupported_risk_profile_raises_error(self, weather_gate: OODGate) -> None:
        """Unsupported risk profile name raises ValueError."""
        valid_row = {"temperature": 25.0, "humidity": 75.0, "rainfall": 150.0}
        with pytest.raises(ValueError, match="Unsupported risk profile"):
            weather_gate.assess_ood(valid_row, risk="hyper_aggressive")

