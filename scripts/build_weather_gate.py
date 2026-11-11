"""Build Weather-Only Out-of-Distribution (OOD) Gate Artifacts & Run Validation Evaluation.

Role 2: Trust Engine Engineer.
Builds the weather-only OOD gate (temperature, humidity, rainfall) for handoff
to Role 1 (Tasks 3 and 11; required for bonus ablations E11 and reality check).

Rules:
  - Fits on training split rows ONLY (data/splits/train_idx.npy, 1,540 rows).
  - Validation evaluation runs on VALIDATION rows ONLY (data/splits/val_idx.npy, 330 rows).
  - NEVER loads or references test_idx.npy (zero test-set leakage).
  - Derives thresholds from 5-fold stratified OOF scores recomputed for 3 dimensions.
  - Stores ONLY raw numpy arrays and plain floats (no pickled sklearn estimators).

Outputs:
  - backend/models/ood_stats_weather.joblib
  - backend/models/thresholds_weather.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

# Add repo root to sys.path so backend modules are importable
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from backend.ood_gate import (
    compute_oof_scores,
    compute_threshold_profiles,
    fit_mahalanobis_gate,
    score_mahalanobis,
)


def compute_metrics(known_scores: np.ndarray, ood_scores: np.ndarray) -> tuple[float, float, float]:
    """Compute AUROC and FPR at 95% TPR (5th percentile cutoff of OOD scores)."""
    y_true = np.concatenate([np.zeros(len(known_scores)), np.ones(len(ood_scores))])
    y_scores = np.concatenate([known_scores, ood_scores])

    auroc = float(roc_auc_score(y_true, y_scores))
    cutoff_95 = float(np.percentile(ood_scores, 5.0))
    fpr_95 = float(np.mean(known_scores >= cutoff_95))

    return auroc, fpr_95, cutoff_95


def main() -> None:
    seed = 42
    weather_features = ["temperature", "humidity", "rainfall"]

    data_path = repo_root / "data" / "Crop_recommendation.csv"
    train_idx_path = repo_root / "data" / "splits" / "train_idx.npy"
    val_idx_path = repo_root / "data" / "splits" / "val_idx.npy"
    models_dir = repo_root / "backend" / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    out_stats_path = models_dir / "ood_stats_weather.joblib"
    out_thresholds_path = models_dir / "thresholds_weather.json"

    # 1. Load data and training split ONLY
    print(f"[1/5] Loading training data from {train_idx_path.name}...")
    df = pd.read_csv(data_path)
    train_idx = np.load(train_idx_path)

    X_train = df[weather_features].values[train_idx]
    y_train = df["label"].values[train_idx]

    print(f"      Training samples: {len(X_train)} rows across {len(np.unique(y_train))} crops.")
    print(f"      Feature subset: {weather_features}")

    # 2. Fit detector on training split (pure NumPy serialization)
    print("[2/5] Fitting per-crop Ledoit-Wolf Mahalanobis detector...")
    gate_data = fit_mahalanobis_gate(
        X=X_train,
        y=y_train,
        feature_names=weather_features,
        seed=seed,
    )

    # 3. Compute 5-fold Stratified OOF scores on training split
    print("[3/5] Computing 5-fold Stratified OOF reference scores (1,540 rows)...")
    oof_scores = compute_oof_scores(
        X=X_train,
        y=y_train,
        n_splits=5,
        seed=seed,
    )
    gate_data["oof_reference_scores"] = oof_scores

    # Verify no sklearn objects are present in gate_data
    for k, v in gate_data.items():
        if k == "crop_stats":
            for crop, c_dict in v.items():
                for sub_k, sub_v in c_dict.items():
                    assert not hasattr(sub_v, "fit"), f"sklearn object found in crop_stats[{crop}][{sub_k}]"
        elif k == "scaler":
            assert isinstance(v, dict), "scaler must be a dict of numpy arrays"
            assert "mean" in v and "scale" in v and "var" in v

    # 4. Compute calibrated threshold profiles
    print("[4/5] Calibrating empirical threshold profiles...")
    thresholds_data = compute_threshold_profiles(
        oof_scores=oof_scores,
        feature_names=weather_features,
        seed=seed,
        round_digits=2,
    )

    # Save artifacts
    print(f"\nWriting artifacts:")
    print(f"  -> {out_stats_path}")
    joblib.dump(gate_data, out_stats_path)

    print(f"  -> {out_thresholds_path}")
    with open(out_thresholds_path, "w", encoding="utf-8") as f:
        json.dump(thresholds_data, f, indent=2)

    # Print summary statistics
    ref = thresholds_data["reference_distribution"]
    profiles = thresholds_data["profiles"]

    print("\n" + "=" * 65)
    print("WEATHER-ONLY OOD GATE ARTIFACT SUMMARY")
    print("=" * 65)
    print(f"Features (p=3):        {weather_features}")
    print(f"Training Samples:      {ref['n_samples']}")
    print(f"OOF Median (p50):      {ref['median']:.4f}")
    print(f"OOF p90:               {ref['p90']:.4f}")
    print(f"OOF p95:               {ref['p95']:.4f}")
    print(f"OOF p97:               {ref['p97']:.4f}")
    print(f"OOF p99:               {ref['p99']:.4f}")
    print(f"OOF p99.5:             {ref['p99_5']:.4f}")
    print("-" * 65)
    print("Calibrated Risk Profiles:")
    for pname in ("conservative", "balanced", "bold"):
        p = profiles[pname]
        print(
            f"  {pname.capitalize():13s} -> "
            f"Caution (p{p['ood_caution_percentile']:.1f}): {p['ood_caution_threshold']:.2f} | "
            f"Refusal (p{p['ood_refuse_percentile']:.1f}): {p['ood_refuse_threshold']:.2f}"
        )
    print("=" * 65)

    # 5. Validation Evaluation Run (Validation split only, N=330)
    print("\n[5/5] Running validation-set evaluation (N=330, test set untouched)...")
    val_idx = np.load(val_idx_path)
    X_val = df[weather_features].values[val_idx]
    n_val = len(X_val)

    val_scores = score_mahalanobis(X_val, gate_data)

    # Box-uniform unknowns (bounds from training split, seed 202)
    train_min = X_train.min(axis=0)
    train_max = X_train.max(axis=0)
    rng_box = np.random.default_rng(202)
    box_unknowns = rng_box.uniform(train_min, train_max, size=(n_val, len(weather_features)))
    box_scores = score_mahalanobis(box_unknowns, gate_data)

    # Shuffled-column unknowns (permute validation columns, seed 404)
    rng_shuf = np.random.default_rng(404)
    shuf_unknowns = X_val.copy()
    for col in range(shuf_unknowns.shape[1]):
        rng_shuf.shuffle(shuf_unknowns[:, col])
    shuf_scores = score_mahalanobis(shuf_unknowns, gate_data)

    # Compute AUROC and FPR@95%
    box_auroc, box_fpr95, box_cut = compute_metrics(val_scores, box_scores)
    shuf_auroc, shuf_fpr95, shuf_cut = compute_metrics(val_scores, shuf_scores)

    print("\n" + "=" * 65)
    print("VALIDATION-SET OOD PERFORMANCE (N=330)")
    print("=" * 65)
    print(f"1. Synthetic Box-Uniform (N={n_val}, seed=202):")
    print(f"   - AUROC:         {box_auroc:.4f}")
    print(f"   - FPR @ 95% TPR: {box_fpr95 * 100.0:.2f}% (cutoff d2 = {box_cut:.4f})")
    print()
    print(f"2. Shuffled-Column (N={n_val}, seed=404):")
    print(f"   - AUROC:         {shuf_auroc:.4f}")
    print(f"   - FPR @ 95% TPR: {shuf_fpr95 * 100.0:.2f}% (cutoff d2 = {shuf_cut:.4f})")
    print("=" * 65)


if __name__ == "__main__":
    main()
