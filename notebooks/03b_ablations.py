"""
notebooks/03b_ablations.py
Role 1: ML Lead (team lead)
Handbook Reference: Section 2 (Role 1 Task 8), Section 7.6 (E11: Ablation Experiments)

Deliverables:
1. Leave-One-Feature-Out Ablation (7 features -> drop each individually)
2. Feature Group Subsets (Full 7, Soil-only 4, Weather-only 3, NPK-only 3)
3. Accuracy vs Training Set Size (Learning Curve: 10% to 100% of train rows)
4. Accuracy vs Measurement Noise Level (0.0 to 3.0 sigma across models)
5. Weather-Only Calibrated Pipeline (trained & calibrated on train rows, saved to backend/models/crop_pipeline_weather.joblib)
6. Figures saved to docs/figures/ and results recorded in backend/models/arena_metrics.json

Rules:
- Zero data leakage: fit strictly on TRAIN rows (data/splits/train_idx.npy, 1,540 rows).
- Evaluated strictly on VAL rows (data/splits/val_idx.npy, 330 rows).
- Test set (data/splits/test_idx.npy) remains untouched.
- No invented numbers.
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, log_loss, f1_score, brier_score_loss
from sklearn.model_selection import StratifiedShuffleSplit

# Import project physical noise generator
sys.path.insert(0, os.path.abspath("."))
from backend.noise import (
    FEATURE_NAMES,
    PHYSICAL_BOUNDS,
    compute_feature_sigmas,
    load_noise_config,
    apply_noise,
)

plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['figure.autolayout'] = True
os.makedirs("docs/figures", exist_ok=True)
os.makedirs("backend/models", exist_ok=True)


def compute_ece(probs: np.ndarray, y_true_indices: np.ndarray, n_bins: int = 10) -> float:
    """Compute Expected Calibration Error (ECE) for multi-class predictions."""
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == y_true_indices).astype(float)
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    total_samples = len(confidences)

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper) if i > 0 else (confidences >= bin_lower) & (confidences <= bin_upper)
        bin_size = np.sum(in_bin)
        if bin_size > 0:
            bin_acc = np.mean(accuracies[in_bin])
            bin_conf = np.mean(confidences[in_bin])
            ece += (bin_size / total_samples) * abs(bin_acc - bin_conf)
    return float(ece)


def compute_multiclass_brier(probs: np.ndarray, y_true_indices: np.ndarray, n_classes: int) -> float:
    """Compute multi-class Brier score."""
    n_samples = len(y_true_indices)
    one_hot = np.zeros((n_samples, n_classes))
    one_hot[np.arange(n_samples), y_true_indices] = 1.0
    return float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))


def main():
    print("=" * 80)
    print("TASK 8: E11 FEATURE ABLATIONS & WEATHER-ONLY CALIBRATED PIPELINE")
    print("=" * 80)

    # 1. Load data and frozen split indices
    data_path = "data/Crop_recommendation.csv"
    train_idx_path = "data/splits/train_idx.npy"
    val_idx_path = "data/splits/val_idx.npy"
    test_idx_path = "data/splits/test_idx.npy"

    df = pd.read_csv(data_path)
    train_idx = np.load(train_idx_path)
    val_idx = np.load(val_idx_path)

    assert os.path.exists(test_idx_path), "Test split index must exist."
    print("Data splits loaded:")
    print(f"  - Training rows:   {len(train_idx)} (70 per crop across 22 classes)")
    print(f"  - Validation rows: {len(val_idx)} (15 per crop across 22 classes)")
    print(f"  - Test set status: UNTOUCHED (0 leakage)\n")

    features_all = list(FEATURE_NAMES)  # ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
    target = 'label'
    classes = sorted(df[target].unique())
    n_classes = len(classes)
    class_to_idx = {c: i for i, c in enumerate(classes)}

    X_train_full = df.iloc[train_idx][features_all]
    y_train = df.iloc[train_idx][target]
    y_train_idx = np.array([class_to_idx[c] for c in y_train])

    X_val_full = df.iloc[val_idx][features_all]
    y_val = df.iloc[val_idx][target]
    y_val_idx = np.array([class_to_idx[c] for c in y_val])

    # Load champion calibrated pipeline for baseline reference
    champ_pipeline = joblib.load("backend/models/crop_pipeline_calibrated.joblib")
    champ_clean_preds = champ_pipeline.predict(X_val_full)
    champ_clean_acc = accuracy_score(y_val, champ_clean_preds)
    champ_clean_probs = champ_pipeline.predict_proba(X_val_full)
    champ_clean_ll = log_loss(y_val, champ_clean_probs, labels=classes)
    champ_clean_f1 = f1_score(y_val, champ_clean_preds, average='macro')
    print(f"Champion Model Baseline (Full 7 features, Extra Trees Isotonic):")
    print(f"  Accuracy: {champ_clean_acc:.4%} ({int(round(champ_clean_acc * len(y_val)))}/330)")
    print(f"  Macro-F1: {champ_clean_f1:.4f}")
    print(f"  Log-Loss: {champ_clean_ll:.4f}\n")

    # =========================================================================
    # PART 1: LEAVE-ONE-FEATURE-OUT ABLATION
    # =========================================================================
    print("-" * 80)
    print("PART 1: LEAVE-ONE-FEATURE-OUT ABLATION (TRAIN on 1,540, VAL on 330)")
    print("-" * 80)

    loo_results = []
    # Include baseline with all features
    loo_results.append({
        "dropped_feature": "None (Full 7 features)",
        "features_used": features_all,
        "n_features": 7,
        "accuracy": champ_clean_acc,
        "accuracy_drop": 0.0,
        "macro_f1": champ_clean_f1,
        "log_loss": champ_clean_ll
    })

    for feat in features_all:
        sub_feats = [f for f in features_all if f != feat]
        # Train Extra Trees classifier (champion base model) on train rows
        et_loo = ExtraTreesClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        et_loo.fit(X_train_full[sub_feats], y_train)

        # Calibrate with cv=5 isotonic strictly on train rows
        cal_loo = CalibratedClassifierCV(et_loo, method="isotonic", cv=5)
        cal_loo.fit(X_train_full[sub_feats], y_train)

        preds = cal_loo.predict(X_val_full[sub_feats])
        probs = cal_loo.predict_proba(X_val_full[sub_feats])
        acc = accuracy_score(y_val, preds)
        macro_f1 = f1_score(y_val, preds, average='macro')
        ll = log_loss(y_val, probs, labels=classes)
        drop = champ_clean_acc - acc

        loo_results.append({
            "dropped_feature": feat,
            "features_used": sub_feats,
            "n_features": 6,
            "accuracy": float(acc),
            "accuracy_drop": float(drop),
            "macro_f1": float(macro_f1),
            "log_loss": float(ll)
        })
        print(f"  Dropped: {feat:12s} | Acc = {acc:.4%} ({int(round(acc*330))}/330) | Drop = {drop:+.4%} | Macro-F1 = {macro_f1:.4f} | Log-Loss = {ll:.4f}")

    # Plot Leave-one-out
    loo_df = pd.DataFrame(loo_results[1:]).sort_values("accuracy_drop", ascending=False)
    plt.figure(figsize=(10, 6))
    bars = plt.barh(loo_df['dropped_feature'], loo_df['accuracy_drop'] * 100, color='#2b5c8f', edgecolor='black', alpha=0.85)
    plt.axvline(0, color='black', linestyle='--', linewidth=0.8)
    plt.xlabel("Validation Accuracy Drop (% points)", fontsize=11, fontweight='bold')
    plt.ylabel("Dropped Feature", fontsize=11, fontweight='bold')
    plt.title("Leave-One-Feature-Out Impact on Validation Accuracy\n(Extra Trees Isotonic, 330 Validation Rows)", fontsize=12, fontweight='bold', pad=12)
    plt.grid(axis='x', linestyle=':', alpha=0.6)
    for bar in bars:
        w = bar.get_width()
        plt.text(w + 0.05, bar.get_y() + bar.get_height()/2, f"{w:+.2f}%", va='center', ha='left', fontsize=9, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    fig_loo_path = "docs/figures/ablation_leave_one_out.png"
    plt.savefig(fig_loo_path, dpi=200)
    plt.close()
    print(f"\nSaved leave-one-out figure to: {fig_loo_path}")

    # =========================================================================
    # PART 2: FEATURE GROUP SUBSETS
    # =========================================================================
    print("\n" + "-" * 80)
    print("PART 2: FEATURE GROUP SUBSETS (Full vs Soil-only vs Weather-only vs NPK-only)")
    print("-" * 80)

    # Approved physical noise on validation rows
    noise_cfg = load_noise_config()
    profile_cfg = noise_cfg["profiles"]["primary_1sigma"]
    X_val_noisy_arr = apply_noise(X_val_full.values, n_samples=1, seed=42)
    X_val_noisy_df = pd.DataFrame(X_val_noisy_arr, columns=features_all)

    subsets = {
        "Full (7 features)": features_all,
        "Soil-only (4: N, P, K, ph)": ["N", "P", "K", "ph"],
        "Weather-only (3: temp, humidity, rainfall)": ["temperature", "humidity", "rainfall"],
        "NPK-only (3: N, P, K)": ["N", "P", "K"]
    }

    subset_results = []
    for s_name, s_cols in subsets.items():
        et = ExtraTreesClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        cal = CalibratedClassifierCV(et, method="isotonic", cv=5)
        cal.fit(X_train_full[s_cols], y_train)

        # Clean evaluation
        clean_preds = cal.predict(X_val_full[s_cols])
        clean_probs = cal.predict_proba(X_val_full[s_cols])
        c_acc = accuracy_score(y_val, clean_preds)
        c_f1 = f1_score(y_val, clean_preds, average='macro')
        c_ll = log_loss(y_val, clean_probs, labels=classes)

        # Noisy evaluation
        noisy_preds = cal.predict(X_val_noisy_df[s_cols])
        noisy_probs = cal.predict_proba(X_val_noisy_df[s_cols])
        n_acc = accuracy_score(y_val, noisy_preds)
        n_f1 = f1_score(y_val, noisy_preds, average='macro')
        n_ll = log_loss(y_val, noisy_probs, labels=classes)

        subset_results.append({
            "subset_name": s_name,
            "feature_count": len(s_cols),
            "features": s_cols,
            "clean_accuracy": float(c_acc),
            "clean_macro_f1": float(c_f1),
            "clean_log_loss": float(c_ll),
            "noisy_accuracy": float(n_acc),
            "noisy_macro_f1": float(n_f1),
            "noisy_log_loss": float(n_ll)
        })
        print(f"  {s_name:42s} | Clean Acc: {c_acc:.4%} | Noisy Acc: {n_acc:.4%} | Clean LL: {c_ll:.4f}")

    # Plot Subsets comparison
    sub_df = pd.DataFrame(subset_results)
    plt.figure(figsize=(10, 6))
    x = np.arange(len(sub_df))
    width = 0.35
    plt.bar(x - width/2, sub_df['clean_accuracy'] * 100, width, label='Clean Validation Accuracy', color='#2b5c8f', edgecolor='black')
    plt.bar(x + width/2, sub_df['noisy_accuracy'] * 100, width, label='Noisy Validation Accuracy (1σ)', color='#d95f02', edgecolor='black')
    plt.ylabel("Accuracy (%)", fontsize=11, fontweight='bold')
    plt.title("Feature Group Performance Comparison\n(Extra Trees Isotonic, 330 Val Rows)", fontsize=12, fontweight='bold', pad=12)
    plt.xticks(x, sub_df['subset_name'], rotation=15, ha='right', fontsize=9, fontweight='bold')
    plt.ylim(0, 105)
    plt.legend(frameon=True, facecolor='white', framealpha=0.9)
    plt.grid(axis='y', linestyle=':', alpha=0.6)
    for i in x:
        c_val = sub_df.loc[i, 'clean_accuracy'] * 100
        n_val = sub_df.loc[i, 'noisy_accuracy'] * 100
        plt.text(i - width/2, c_val + 1.2, f"{c_val:.1f}%", ha='center', fontsize=9, fontweight='bold')
        plt.text(i + width/2, n_val + 1.2, f"{n_val:.1f}%", ha='center', fontsize=9, fontweight='bold')
    plt.tight_layout()
    fig_sub_path = "docs/figures/ablation_feature_subsets.png"
    plt.savefig(fig_sub_path, dpi=200)
    plt.close()
    print(f"\nSaved feature subsets figure to: {fig_sub_path}")

    # =========================================================================
    # PART 3: ACCURACY VS TRAINING SET SIZE (LEARNING CURVE)
    # =========================================================================
    print("\n" + "-" * 80)
    print("PART 3: ACCURACY VS TRAINING SET SIZE (10% to 100% of TRAIN rows)")
    print("-" * 80)

    fractions = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    learning_curve_results = []

    for frac in fractions:
        if frac == 1.0:
            sub_train_idx = np.arange(len(X_train_full))
        else:
            sss = StratifiedShuffleSplit(n_splits=1, train_size=frac, random_state=42)
            sub_train_idx, _ = next(sss.split(X_train_full, y_train))

        n_sub = len(sub_train_idx)
        X_sub = X_train_full.iloc[sub_train_idx]
        y_sub = y_train.iloc[sub_train_idx]

        # Extra Trees
        et = ExtraTreesClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        et.fit(X_sub, y_sub)
        train_acc_et = accuracy_score(y_sub, et.predict(X_sub))
        val_acc_et = accuracy_score(y_val, et.predict(X_val_full))

        # Random Forest
        rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X_sub, y_sub)
        train_acc_rf = accuracy_score(y_sub, rf.predict(X_sub))
        val_acc_rf = accuracy_score(y_val, rf.predict(X_val_full))

        learning_curve_results.append({
            "fraction": frac,
            "train_samples": n_sub,
            "samples_per_crop": round(n_sub / 22, 1),
            "et_train_acc": float(train_acc_et),
            "et_val_acc": float(val_acc_et),
            "rf_train_acc": float(train_acc_rf),
            "rf_val_acc": float(val_acc_rf)
        })
        print(f"  Frac {frac*100:4.0f}% ({n_sub:4d} rows, {n_sub/22:4.1f}/crop) | ET Val Acc: {val_acc_et:.4%} | RF Val Acc: {val_acc_rf:.4%}")

    # Plot Learning Curve
    lc_df = pd.DataFrame(learning_curve_results)
    plt.figure(figsize=(10, 6))
    plt.plot(lc_df['train_samples'], lc_df['et_val_acc'] * 100, marker='o', linewidth=2.2, color='#2b5c8f', label='Extra Trees (Val Acc)')
    plt.plot(lc_df['train_samples'], lc_df['rf_val_acc'] * 100, marker='s', linewidth=2.2, color='#1b7837', linestyle='--', label='Random Forest (Val Acc)')
    plt.axhline(champ_clean_acc * 100, color='gray', linestyle=':', label=f'Full Data Val Acc ({champ_clean_acc*100:.2f}%)')
    plt.xlabel("Number of Training Samples (from 1,540 total)", fontsize=11, fontweight='bold')
    plt.ylabel("Validation Accuracy (%)", fontsize=11, fontweight='bold')
    plt.title("Learning Curves: Validation Accuracy vs Training Set Size\n(330 Fixed Validation Rows)", fontsize=12, fontweight='bold', pad=12)
    plt.ylim(85, 101)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(frameon=True, facecolor='white', framealpha=0.9)
    plt.tight_layout()
    fig_lc_path = "docs/figures/ablation_learning_curve.png"
    plt.savefig(fig_lc_path, dpi=200)
    plt.close()
    print(f"\nSaved learning curve figure to: {fig_lc_path}")

    # =========================================================================
    # PART 4: ACCURACY VS MEASUREMENT NOISE LEVEL (0.0 to 3.0 sigma)
    # =========================================================================
    print("\n" + "-" * 80)
    print("PART 4: ACCURACY VS MEASUREMENT NOISE LEVEL (0.0σ to 3.0σ)")
    print("-" * 80)

    # Base models for noise curve
    et_iso = champ_pipeline
    rf_base = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf_iso = CalibratedClassifierCV(rf_base, method="isotonic", cv=5)
    rf_iso.fit(X_train_full, y_train)

    gnb_base = Pipeline([('scaler', StandardScaler()), ('clf', GaussianNB())])
    gnb_iso = CalibratedClassifierCV(gnb_base, method="isotonic", cv=5)
    gnb_iso.fit(X_train_full, y_train)

    noise_sigmas_base = compute_feature_sigmas(X_val_full.values, profile_cfg)
    lower_bounds = np.array([PHYSICAL_BOUNDS[f][0] for f in features_all])
    upper_bounds = np.array([PHYSICAL_BOUNDS[f][1] for f in features_all])

    noise_multipliers = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    noise_curve_results = []

    for mult in noise_multipliers:
        if mult == 0.0:
            X_noisy_arr = X_val_full.values.copy()
        else:
            rng = np.random.default_rng(42)
            perturbation = rng.normal(0.0, mult * noise_sigmas_base, size=X_val_full.shape)
            unclipped = X_val_full.values + perturbation
            X_noisy_arr = np.clip(unclipped, lower_bounds, upper_bounds)

        X_noisy_df = pd.DataFrame(X_noisy_arr, columns=features_all)

        et_acc = accuracy_score(y_val, et_iso.predict(X_noisy_df))
        et_ll = log_loss(y_val, et_iso.predict_proba(X_noisy_df), labels=classes)

        rf_acc = accuracy_score(y_val, rf_iso.predict(X_noisy_df))
        rf_ll = log_loss(y_val, rf_iso.predict_proba(X_noisy_df), labels=classes)

        gnb_acc = accuracy_score(y_val, gnb_iso.predict(X_noisy_df))
        gnb_ll = log_loss(y_val, gnb_iso.predict_proba(X_noisy_df), labels=classes)

        noise_curve_results.append({
            "multiplier": mult,
            "et_isotonic_accuracy": float(et_acc),
            "et_isotonic_log_loss": float(et_ll),
            "rf_isotonic_accuracy": float(rf_acc),
            "rf_isotonic_log_loss": float(rf_ll),
            "gnb_isotonic_accuracy": float(gnb_acc),
            "gnb_isotonic_log_loss": float(gnb_ll)
        })
        print(f"  Noise {mult:3.1f}σ | ET Iso Acc: {et_acc:.4%} (LL {et_ll:.3f}) | RF Iso Acc: {rf_acc:.4%} (LL {rf_ll:.3f}) | GNB Iso Acc: {gnb_acc:.4%} (LL {gnb_ll:.3f})")

    # Plot Noise Sensitivity Curve
    nc_df = pd.DataFrame(noise_curve_results)
    plt.figure(figsize=(10, 6))
    plt.plot(nc_df['multiplier'], nc_df['et_isotonic_accuracy'] * 100, marker='o', linewidth=2.2, color='#2b5c8f', label='Extra Trees (Isotonic)')
    plt.plot(nc_df['multiplier'], nc_df['rf_isotonic_accuracy'] * 100, marker='s', linewidth=2.2, color='#1b7837', label='Random Forest (Isotonic)')
    plt.plot(nc_df['multiplier'], nc_df['gnb_isotonic_accuracy'] * 100, marker='^', linewidth=2.2, color='#7570b3', label='Gaussian NB (Isotonic)')
    plt.axvline(1.0, color='red', linestyle=':', label='Approved Physical 1σ Noise')
    plt.xlabel("Sensor Noise Scale Multiplier (× approved 1σ)", fontsize=11, fontweight='bold')
    plt.ylabel("Validation Accuracy (%)", fontsize=11, fontweight='bold')
    plt.title("Noise Robustness: Model Accuracy vs Sensor Noise Scale\n(330 Validation Rows, seed=42)", fontsize=12, fontweight='bold', pad=12)
    plt.ylim(60, 102)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(frameon=True, facecolor='white', framealpha=0.9)
    plt.tight_layout()
    fig_nc_path = "docs/figures/ablation_noise_sensitivity.png"
    plt.savefig(fig_nc_path, dpi=200)
    plt.close()
    print(f"\nSaved noise sensitivity figure to: {fig_nc_path}")

    # =========================================================================
    # PART 5: WEATHER-ONLY CALIBRATED PIPELINE (PARTIAL-INPUT MODE)
    # =========================================================================
    print("\n" + "-" * 80)
    print("PART 5: WEATHER-ONLY CALIBRATED PIPELINE (PARTIAL-INPUT MODE)")
    print("-" * 80)

    weather_features = ["temperature", "humidity", "rainfall"]
    X_train_weather = X_train_full[weather_features]
    X_val_weather = X_val_full[weather_features]
    X_val_noisy_weather = X_val_noisy_df[weather_features]

    # Train base Extra Trees model strictly on TRAIN rows
    base_et_weather = ExtraTreesClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    base_et_weather.fit(X_train_weather, y_train)

    # Calibrate using Isotonic regression strictly on held-out folds inside TRAIN rows
    cal_weather = CalibratedClassifierCV(base_et_weather, method="isotonic", cv=5)
    cal_weather.fit(X_train_weather, y_train)

    # Clean metrics (before & after)
    uncal_probs_clean = base_et_weather.predict_proba(X_val_weather)
    uncal_preds_clean = base_et_weather.predict(X_val_weather)
    uncal_acc_clean = accuracy_score(y_val, uncal_preds_clean)
    uncal_ll_clean = log_loss(y_val, uncal_probs_clean, labels=classes)
    uncal_ece_clean = compute_ece(uncal_probs_clean, y_val_idx)
    uncal_brier_clean = compute_multiclass_brier(uncal_probs_clean, y_val_idx, n_classes)

    iso_probs_clean = cal_weather.predict_proba(X_val_weather)
    iso_preds_clean = cal_weather.predict(X_val_weather)
    iso_acc_clean = accuracy_score(y_val, iso_preds_clean)
    iso_ll_clean = log_loss(y_val, iso_probs_clean, labels=classes)
    iso_ece_clean = compute_ece(iso_probs_clean, y_val_idx)
    iso_brier_clean = compute_multiclass_brier(iso_probs_clean, y_val_idx, n_classes)

    # Noisy metrics (1-sigma)
    uncal_probs_noisy = base_et_weather.predict_proba(X_val_noisy_weather)
    uncal_preds_noisy = base_et_weather.predict(X_val_noisy_weather)
    uncal_acc_noisy = accuracy_score(y_val, uncal_preds_noisy)
    uncal_ll_noisy = log_loss(y_val, uncal_probs_noisy, labels=classes)
    uncal_ece_noisy = compute_ece(uncal_probs_noisy, y_val_idx)

    iso_probs_noisy = cal_weather.predict_proba(X_val_noisy_weather)
    iso_preds_noisy = cal_weather.predict(X_val_noisy_weather)
    iso_acc_noisy = accuracy_score(y_val, iso_preds_noisy)
    iso_ll_noisy = log_loss(y_val, iso_probs_noisy, labels=classes)
    iso_ece_noisy = compute_ece(iso_probs_noisy, y_val_idx)

    print("Weather-Only Model Performance (330 Val Rows):")
    print(f"  Clean Accuracy:  Uncalibrated = {uncal_acc_clean:.4%} | Calibrated (Isotonic) = {iso_acc_clean:.4%} ({int(round(iso_acc_clean*330))}/330)")
    print(f"  Clean Log-Loss:  Uncalibrated = {uncal_ll_clean:.4f}   | Calibrated (Isotonic) = {iso_ll_clean:.4f}")
    print(f"  Clean ECE:       Uncalibrated = {uncal_ece_clean:.4f}   | Calibrated (Isotonic) = {iso_ece_clean:.4f}")
    print(f"  Noisy Accuracy:  Uncalibrated = {uncal_acc_noisy:.4%} | Calibrated (Isotonic) = {iso_acc_noisy:.4%} ({int(round(iso_acc_noisy*330))}/330)")
    print(f"  Noisy Log-Loss:  Uncalibrated = {uncal_ll_noisy:.4f}   | Calibrated (Isotonic) = {iso_ll_noisy:.4f}")
    print(f"  Noisy ECE:       Uncalibrated = {uncal_ece_noisy:.4f}   | Calibrated (Isotonic) = {iso_ece_noisy:.4f}")

    # Plot Weather-Only Reliability Diagram
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, p, title, ece_val in zip(
        axes,
        [uncal_probs_clean, iso_probs_clean],
        ["Weather-Only Uncalibrated", "Weather-Only Isotonic Calibrated"],
        [uncal_ece_clean, iso_ece_clean]
    ):
        confs = np.max(p, axis=1)
        preds = np.argmax(p, axis=1)
        accs = (preds == y_val_idx).astype(float)
        bins = np.linspace(0, 1, 11)
        bin_accs = []
        bin_confs = []
        for i in range(10):
            in_b = (confs > bins[i]) & (confs <= bins[i+1]) if i > 0 else (confs >= bins[i]) & (confs <= bins[i+1])
            if np.sum(in_b) > 0:
                bin_accs.append(np.mean(accs[in_b]))
                bin_confs.append(np.mean(confs[in_b]))
            else:
                bin_accs.append(np.nan)
                bin_confs.append((bins[i] + bins[i+1])/2)
        ax.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
        ax.plot(bin_confs, bin_accs, 's-', color='#2b5c8f', label=f'Model (ECE = {ece_val:.4f})')
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.set_xlabel("Mean Predicted Confidence", fontsize=10, fontweight='bold')
        ax.set_ylabel("Empirical Accuracy", fontsize=10, fontweight='bold')
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.legend(loc='upper left', frameon=True)
    plt.tight_layout()
    fig_weather_rel_path = "docs/figures/reliability_diagram_weather.png"
    plt.savefig(fig_weather_rel_path, dpi=200)
    plt.close()
    print(f"\nSaved weather reliability diagram to: {fig_weather_rel_path}")

    # Save calibrated weather pipeline with compress=3
    weather_model_path = "backend/models/crop_pipeline_weather.joblib"
    joblib.dump(cal_weather, weather_model_path, compress=3)
    weather_file_size = os.path.getsize(weather_model_path)
    print(f"Exported weather pipeline to: {weather_model_path} ({weather_file_size:,} bytes / {weather_file_size / (1024*1024):.2f} MB)")

    # Verify weather pipeline contract
    loaded_weather_model = joblib.load(weather_model_path)
    test_weather_input = np.array([[25.0, 80.0, 200.0]])
    test_probs = loaded_weather_model.predict_proba(test_weather_input)
    assert np.isclose(np.sum(test_probs), 1.0), "Probabilities must sum to 1.0"
    assert list(loaded_weather_model.classes_) == classes, "Classes must match alphabetical list"
    print("Verification PASSED:")
    print(f"  - Input shape: 3 weather features (temperature, humidity, rainfall)")
    print(f"  - Probability sum: {np.sum(test_probs):.6f}")
    print(f"  - Number of classes: {len(loaded_weather_model.classes_)}")
    print(f"  - Top predicted crop: {loaded_weather_model.classes_[np.argmax(test_probs)]} (p = {np.max(test_probs):.4f})")

    # =========================================================================
    # PART 6: RECORD EXPERIMENT E11 IN arena_metrics.json
    # =========================================================================
    arena_path = "backend/models/arena_metrics.json"
    with open(arena_path, "r", encoding="utf-8") as f:
        arena = json.load(f)

    arena["ablations_experiment_e11"] = {
        "evaluation_samples": len(val_idx),
        "leave_one_feature_out": loo_results,
        "feature_group_subsets": subset_results,
        "training_size_learning_curve": learning_curve_results,
        "noise_scale_sensitivity": noise_curve_results,
        "weather_only_model": {
            "pipeline_path": weather_model_path,
            "features": weather_features,
            "clean_accuracy": float(iso_acc_clean),
            "clean_log_loss": float(iso_ll_clean),
            "clean_ece": float(iso_ece_clean),
            "clean_brier": float(iso_brier_clean),
            "noisy_accuracy": float(iso_acc_noisy),
            "noisy_log_loss": float(iso_ll_noisy),
            "noisy_ece": float(iso_ece_noisy),
            "figures": {
                "reliability_diagram": fig_weather_rel_path
            }
        },
        "figures": {
            "leave_one_out": fig_loo_path,
            "feature_subsets": fig_sub_path,
            "learning_curve": fig_lc_path,
            "noise_sensitivity": fig_nc_path
        }
    }

    # Banned-word validation before writing
    arena_text = json.dumps(arena, indent=2)
    banned_terms = ["ppm", "agronom", "synthetic", "non-overlapping", "kg/ha"]
    matches = {t: arena_text.lower().count(t) for t in banned_terms if arena_text.lower().count(t) > 0}
    assert not matches, f"Found banned terms in arena_metrics: {matches}"

    with open(arena_path, "w", encoding="utf-8") as f:
        f.write(arena_text + "\n")
    print(f"\nRecorded E11 findings in: {arena_path}")

    print("\n" + "=" * 80)
    print("TASK 8 (E11 ABLATIONS & WEATHER PIPELINE) COMPLETED SUCCESSFULLY")
    print("=" * 80)


if __name__ == "__main__":
    main()

