"""Export OOD detector parameters for in-browser familiarity check (Task 11A).

Role 2: Trust Engine Engineer
Generates frontend/src/ood_browser.json:
  - Standardized class means and precision matrices (Ledoit-Wolf) for 22 crops.
  - StandardScaler mean and scale parameters for 7 canonical features.
  - Calibrated OOD caution and refusal thresholds for 3 risk profiles.
  - Deterministic test cases (3 validation rows + 3 synthetic rows) for frontend verification.

Usage:
  python notebooks/04b_export_ood_browser.py
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

# Handle imports whether run from repo root or inside notebooks/
try:
    from backend.trust_engine import TrustEngine
except ModuleNotFoundError:
    import sys

    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from backend.trust_engine import TrustEngine


def build_ood_browser_data(
    engine: Optional[TrustEngine] = None,
    df_raw: Optional[pd.DataFrame] = None,
    seed: int = 2026,
) -> Dict[str, Any]:
    """Construct the complete serializable dictionary for ood_browser.json.

    Args:
        engine: Initialized TrustEngine instance. If None, loaded from backend/models.
        df_raw: Raw DataFrame of Crop_recommendation.csv. If None, loaded from data/.
        seed: Random seed for synthetic test cases (default 2026).

    Returns:
        Ordered dictionary with all metadata, scaler, class parameters, thresholds, and test cases.
    """
    if engine is None:
        base_dir = Path(__file__).resolve().parent.parent
        model_candidates = [
            base_dir / "backend" / "models",
            Path("backend/models"),
            Path("models"),
        ]
        model_dir = next((p for p in model_candidates if (p / "ood_stats.joblib").exists()), None)
        if model_dir is None:
            raise FileNotFoundError("Could not locate backend/models directory")
        engine = TrustEngine.load(model_dir)

    if df_raw is None:
        base_dir = Path(__file__).resolve().parent.parent
        data_candidates = [
            base_dir / "data" / "Crop_recommendation.csv",
            Path("data/Crop_recommendation.csv"),
        ]
        data_path = next((p for p in data_candidates if p.exists()), None)
        if data_path is None:
            raise FileNotFoundError("Could not locate data/Crop_recommendation.csv")
        df_raw = pd.read_csv(data_path)

    features = list(engine.REQUIRED_FEATURES)
    scaler_mean = [float(v) for v in engine.scaler.mean_]
    scaler_scale = [float(v) for v in engine.scaler.scale_]
    crops = list(engine.classes)

    class_mean = {
        crop: [float(v) for v in engine.crop_stats[crop]["mean"]]
        for crop in crops
    }
    class_precision = {
        crop: [[float(v) for v in row] for row in engine.crop_stats[crop]["inv_cov"]]
        for crop in crops
    }

    thresholds: Dict[str, Dict[str, float]] = {}
    for p in ["conservative", "balanced", "bold"]:
        p_cfg = engine.thresholds["profiles"][p]
        thresholds[p] = {
            "caution": float(p_cfg["ood_caution_threshold"]),
            "refuse": float(p_cfg["ood_refuse_threshold"]),
        }

    test_cases: List[Dict[str, Any]] = []

    # 1. Validation rows: dataset indices 290, 2049, 4
    for ds_idx in [290, 2049, 4]:
        raw_s = df_raw.iloc[ds_idx]
        row_dict = {f: float(raw_s[f]) for f in features}
        min_d2, sorted_crops, _ = engine._compute_mahalanobis(row_dict)
        nearest_crop = sorted_crops[0][0]
        flagged = {
            p: bool(min_d2 > thresholds[p]["refuse"])
            for p in ["conservative", "balanced", "bold"]
        }
        test_cases.append({
            "input": row_dict,
            "min_d2": float(min_d2),
            "nearest_crop": nearest_crop,
            "flagged": flagged,
        })

    # 2. Synthetic rows: 3 random inputs from RandomState(seed) uniform in physical bounds
    rng = np.random.RandomState(seed)
    low_bounds = [engine.PHYSICAL_BOUNDS[f][0] for f in features]
    high_bounds = [engine.PHYSICAL_BOUNDS[f][1] for f in features]
    synth_matrix = rng.uniform(low=low_bounds, high=high_bounds, size=(3, 7))

    for i in range(3):
        row_dict = {f: float(synth_matrix[i, j]) for j, f in enumerate(features)}
        min_d2, sorted_crops, _ = engine._compute_mahalanobis(row_dict)
        nearest_crop = sorted_crops[0][0]
        flagged = {
            p: bool(min_d2 > thresholds[p]["refuse"])
            for p in ["conservative", "balanced", "bold"]
        }
        test_cases.append({
            "input": row_dict,
            "min_d2": float(min_d2),
            "nearest_crop": nearest_crop,
            "flagged": flagged,
        })

    return {
        "version": 1,
        "generated_by": "notebooks/04b_export_ood_browser.py",
        "note": "Server stays the final authority. d2 = min over crops of (z-mean)^T P (z-mean), z = (x - scaler_mean)/scaler_scale. Refuse if d2 > refuse; caution if d2 > caution.",
        "features": features,
        "scaler_mean": scaler_mean,
        "scaler_scale": scaler_scale,
        "crops": crops,
        "class_mean": class_mean,
        "class_precision": class_precision,
        "thresholds": thresholds,
        "test_cases": test_cases,
    }


def export_ood_browser(
    output_path: Optional[Union[str, Path]] = None,
    engine: Optional[TrustEngine] = None,
    df_raw: Optional[pd.DataFrame] = None,
) -> Tuple[Path, int, str]:
    """Export ood_browser.json to the target destination.

    Args:
        output_path: Target path for the JSON file. Defaults to frontend/src/ood_browser.json.
        engine: TrustEngine instance (optional).
        df_raw: Raw DataFrame (optional).

    Returns:
        Tuple of (target_path, file_size_bytes, sha256_hex_digest).
    """
    if output_path is None:
        base_dir = Path(__file__).resolve().parent.parent
        output_path = base_dir / "frontend" / "src" / "ood_browser.json"
    else:
        output_path = Path(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = build_ood_browser_data(engine=engine, df_raw=df_raw)
    json_bytes = json.dumps(data, separators=(",", ":")).encode("utf-8")
    sha256_digest = hashlib.sha256(json_bytes).hexdigest()

    with open(output_path, "wb") as f:
        f.write(json_bytes)

    file_size = output_path.stat().st_size
    return output_path, file_size, sha256_digest


def main() -> None:
    """Generate and verify ood_browser.json twice for determinism."""
    print("=== TASK 11A: EXPORT OOD BROWSER JSON ===")
    
    # Run 1
    path1, size1, hash1 = export_ood_browser()
    print(f"Run 1 exported to: {path1}")
    print(f"  File size: {size1:,} bytes ({size1 / 1024:.2f} KB)")
    print(f"  SHA-256:   {hash1}")

    # Run 2
    path2, size2, hash2 = export_ood_browser()
    print(f"\nRun 2 exported to: {path2}")
    print(f"  File size: {size2:,} bytes ({size2 / 1024:.2f} KB)")
    print(f"  SHA-256:   {hash2}")

    assert hash1 == hash2, "Deterministic export failure: SHA-256 mismatch between runs!"
    assert size1 == size2, "File size mismatch between runs!"
    print(f"\nDeterministic export verified: hashes match perfectly ({hash1 == hash2}).")

    with open(path1, "r", encoding="utf-8") as f:
        loaded = json.load(f)
    print(f"Export Summary:")
    print(f"  Version:       {loaded['version']}")
    print(f"  Generated by:  {loaded['generated_by']}")
    print(f"  Features:      {len(loaded['features'])} features: {loaded['features']}")
    print(f"  Crops count:   {len(loaded['crops'])}")
    print(f"  Thresholds:    {list(loaded['thresholds'].keys())}")
    print(f"  Test cases:    {len(loaded['test_cases'])}")
    for i, tc in enumerate(loaded["test_cases"]):
        kind = "val" if i < 3 else "synthetic"
        print(f"    Case {i+1} ({kind}): nearest={tc['nearest_crop']}, d2={tc['min_d2']:.4f}, flagged(balanced)={tc['flagged']['balanced']}")


if __name__ == "__main__":
    main()

