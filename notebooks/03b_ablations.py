"""
notebooks/03b_ablations.py
Role 1: ML Lead (team lead)
Task: E11 Ablations

Rules:
- Train split only for fitting (data/splits/train_idx.npy, 1,540 rows).
- Validation split only for evaluation (data/splits/val_idx.npy, 330 rows).
- Do NOT touch the test set.
- Do NOT overwrite crop_pipeline.joblib or crop_pipeline_calibrated.joblib.
- No agronomic explanations, no "ppm"/"kg/ha"/"synthetic"/"non-overlapping", no rainfall period.
- Report only numbers the code prints. Show raw outputs, not summaries.

Setup:
- Splits: train_idx, val_idx from data/splits/*.npy, seed 42.
- Feature order: N, P, K, temperature, humidity, ph, rainfall.
- Model: CalibratedClassifierCV(ExtraTreesClassifier(n_estimators=100, random_state=42, n_jobs=-1), method="isotonic", cv=5).
- Noise: backend/noise.py, 1-sigma primary profile on val rows only (seeds 42, 43, 44, 45, 46).
- Metrics: accuracy (count/330), macro-F1, log-loss, ECE (10 bins), clean and noisy.

Experiments:
1. Leave-one-feature-out: drop each of 7 features, refit, evaluate (full model as reference row).
2. Feature subsets: soil only (N, P, K, ph), weather only (temperature, humidity, rainfall), full.
3. Learning curve: stratified 10%, 25%, 50%, 75%, 100% of train split (3 subsamples, seeds 42-44).
4. Noise sensitivity: noise scale 0.5, 1.0, 1.5, 2.0 times 1-sigma, 5 seeds each (final model, RF uncalibrated, GNB uncalibrated).
5. Weather-only calibrated model: fit on train, evaluate on val, clean and noisy (5 seeds). Export backend/models/crop_pipeline_weather.joblib (compress=3).
"""

import os
import sys
import time
import json
import joblib
import numpy as np
import pandas as pd

# Set headless Matplotlib backend before importing pyplot
os.environ['MPLBACKEND'] = 'Agg'
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.model_selection import StratifiedShuffleSplit

# Project physical noise module
sys.path.insert(0, os.path.abspath("."))
from backend.noise import (
    FEATURE_NAMES,
    PHYSICAL_BOUNDS,
    compute_feature_sigmas,
    load_noise_config,
    apply_noise,
)

NOISE_SEEDS = [42, 43, 44, 45, 46]
SUBSAMPLE_SEEDS = [42, 43, 44]
N_BINS_ECE = 10


def compute_ece(probs: np.ndarray, y_true_idx: np.ndarray, n_bins: int = 10) -> float:
    """Compute Expected Calibration Error (ECE) for multi-class predictions."""
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == y_true_idx).astype(float)
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    total_samples = len(confidences)

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        if i == 0:
            in_bin = (confidences >= bin_lower) & (confidences <= bin_upper)
        else:
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        bin_size = np.sum(in_bin)
        if bin_size > 0:
            bin_acc = np.mean(accuracies[in_bin])
            bin_conf = np.mean(confidences[in_bin])
            ece += (bin_size / total_samples) * abs(bin_acc - bin_conf)
    return float(ece)


def evaluate_model_on_val(model, X_val_df, y_val, y_val_idx, classes, features_used, noise_sigmas_dict, lower_bounds_dict, upper_bounds_dict):
    """Evaluate model on clean validation set and across 5 noisy seeds."""
    n_val = len(y_val)

    # Clean evaluation
    clean_preds = model.predict(X_val_df)
    clean_probs = model.predict_proba(X_val_df)
    clean_acc = float(accuracy_score(y_val, clean_preds))
    clean_count = int(np.sum(clean_preds == y_val.values))
    clean_f1 = float(f1_score(y_val, clean_preds, average='macro'))
    clean_ll = float(log_loss(y_val, clean_probs, labels=classes))
    clean_ece = compute_ece(clean_probs, y_val_idx, N_BINS_ECE)

    clean_metrics = {
        "accuracy": clean_acc,
        "accuracy_count": clean_count,
        "accuracy_str": f"{clean_count}/{n_val}",
        "macro_f1": clean_f1,
        "log_loss": clean_ll,
        "ece": clean_ece
    }

    # Noisy evaluation across 5 seeds
    noisy_per_seed = []
    sub_sigmas = np.array([noise_sigmas_dict[f] for f in features_used]).T
    sub_lower = np.array([lower_bounds_dict[f] for f in features_used])
    sub_upper = np.array([upper_bounds_dict[f] for f in features_used])

    for seed in NOISE_SEEDS:
        rng = np.random.default_rng(seed)
        noise = rng.normal(0.0, sub_sigmas, size=X_val_df.shape)
        noisy_x = np.clip(X_val_df.values + noise, sub_lower, sub_upper)
        noisy_x_df = pd.DataFrame(noisy_x, columns=features_used)

        n_preds = model.predict(noisy_x_df)
        n_probs = model.predict_proba(noisy_x_df)

        n_acc = float(accuracy_score(y_val, n_preds))
        n_count = int(np.sum(n_preds == y_val.values))
        n_f1 = float(f1_score(y_val, n_preds, average='macro'))
        n_ll = float(log_loss(y_val, n_probs, labels=classes))
        n_ece = compute_ece(n_probs, y_val_idx, N_BINS_ECE)

        noisy_per_seed.append({
            "seed": seed,
            "accuracy": n_acc,
            "accuracy_count": n_count,
            "macro_f1": n_f1,
            "log_loss": n_ll,
            "ece": n_ece
        })

    noisy_accs = [m["accuracy"] for m in noisy_per_seed]
    noisy_counts = [m["accuracy_count"] for m in noisy_per_seed]
    noisy_f1s = [m["macro_f1"] for m in noisy_per_seed]
    noisy_lls = [m["log_loss"] for m in noisy_per_seed]
    noisy_eces = [m["ece"] for m in noisy_per_seed]

    noisy_summary = {
        "seeds": NOISE_SEEDS,
        "accuracy_mean": float(np.mean(noisy_accs)),
        "accuracy_std": float(np.std(noisy_accs)),
        "accuracy_count_mean": float(np.mean(noisy_counts)),
        "accuracy_count_std": float(np.std(noisy_counts)),
        "macro_f1_mean": float(np.mean(noisy_f1s)),
        "macro_f1_std": float(np.std(noisy_f1s)),
        "log_loss_mean": float(np.mean(noisy_lls)),
        "log_loss_std": float(np.std(noisy_lls)),
        "ece_mean": float(np.mean(noisy_eces)),
        "ece_std": float(np.std(noisy_eces)),
        "per_seed": noisy_per_seed
    }

    return clean_metrics, noisy_summary


def main():
    start_total_time = time.time()
    os.makedirs("docs/figures", exist_ok=True)
    os.makedirs("backend/models", exist_ok=True)

    print("=" * 80)
    print("TASK E11 ABLATION EXPERIMENTS (ROLE 1)")
    print("=" * 80)

    # 1. Load splits and verify test isolation
    data_path = "data/Crop_recommendation.csv"
    train_idx_path = "data/splits/train_idx.npy"
    val_idx_path = "data/splits/val_idx.npy"

    df = pd.read_csv(data_path)
    train_idx = np.load(train_idx_path)
    val_idx = np.load(val_idx_path)
    print("Setup:")
    print(f"  - Dataset: {data_path} (2200 rows, 22 crops)")
    print(f"  - Train rows: {len(train_idx)} (data/splits/train_idx.npy)")
    print(f"  - Val rows:   {len(val_idx)} (data/splits/val_idx.npy)")
    print("  - Test set status: LOCKED / UNTOUCHED (0 leakage rule)")
    print(f"  - Noise seeds: {NOISE_SEEDS}\n")

    features_all = list(FEATURE_NAMES)  # ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
    target = 'label'
    classes = sorted(df[target].unique())
    n_classes = len(classes)
    class_to_idx = {c: i for i, c in enumerate(classes)}

    X_train_full = df.iloc[train_idx][features_all]
    y_train = df.iloc[train_idx][target]

    X_val_full = df.iloc[val_idx][features_all]
    y_val = df.iloc[val_idx][target]
    y_val_idx = np.array([class_to_idx[c] for c in y_val])

    # Pre-calculate physical noise sigmas and bounds dictionary for val rows
    noise_cfg = load_noise_config()
    profile_cfg = noise_cfg["profiles"]["primary_1sigma"]
    sigmas_val_full = compute_feature_sigmas(X_val_full.values, profile_cfg)

    noise_sigmas_dict = {f: sigmas_val_full[:, i] for i, f in enumerate(features_all)}
    lower_bounds_dict = {f: PHYSICAL_BOUNDS[f][0] for f in features_all}
    upper_bounds_dict = {f: PHYSICAL_BOUNDS[f][1] for f in features_all}

    # =========================================================================
    # EXPERIMENT 1: LEAVE-ONE-FEATURE-OUT ABLATION
    # =========================================================================
    print("-" * 80)
    print("EXPERIMENT 1: LEAVE-ONE-FEATURE-OUT ABLATION")
    print("-" * 80)
    t0 = time.time()

    # 1. Fit reference full 7-feature model on train rows
    base_et_full = ExtraTreesClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    cal_full = CalibratedClassifierCV(base_et_full, method="isotonic", cv=5)
    cal_full.fit(X_train_full, y_train)

    ref_clean, ref_noisy = evaluate_model_on_val(
        cal_full, X_val_full, y_val, y_val_idx, classes, features_all,
        noise_sigmas_dict, lower_bounds_dict, upper_bounds_dict
    )

    exp1_records = []
    exp1_records.append({
        "dropped_feature": "None (Full 7 features)",
        "features_used": features_all,
        "n_features": 7,
        "clean_accuracy": ref_clean["accuracy"],
        "clean_count": ref_clean["accuracy_count"],
        "clean_macro_f1": ref_clean["macro_f1"],
        "clean_log_loss": ref_clean["log_loss"],
        "clean_ece": ref_clean["ece"],
        "noisy_accuracy_mean": ref_noisy["accuracy_mean"],
        "noisy_accuracy_std": ref_noisy["accuracy_std"],
        "noisy_count_mean": ref_noisy["accuracy_count_mean"],
        "noisy_count_std": ref_noisy["accuracy_count_std"],
        "noisy_macro_f1_mean": ref_noisy["macro_f1_mean"],
        "noisy_macro_f1_std": ref_noisy["macro_f1_std"],
        "noisy_log_loss_mean": ref_noisy["log_loss_mean"],
        "noisy_log_loss_std": ref_noisy["log_loss_std"],
        "noisy_ece_mean": ref_noisy["ece_mean"],
        "noisy_ece_std": ref_noisy["ece_std"],
        "clean_acc_drop": 0.0,
        "noisy_acc_drop": 0.0
    })

    # 2. Drop each of 7 features
    for feat in features_all:
        sub_feats = [f for f in features_all if f != feat]
        base_et = ExtraTreesClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        cal = CalibratedClassifierCV(base_et, method="isotonic", cv=5)
        cal.fit(X_train_full[sub_feats], y_train)

        c_metrics, n_metrics = evaluate_model_on_val(
            cal, X_val_full[sub_feats], y_val, y_val_idx, classes, sub_feats,
            noise_sigmas_dict, lower_bounds_dict, upper_bounds_dict
        )

        c_drop = ref_clean["accuracy"] - c_metrics["accuracy"]
        n_drop = ref_noisy["accuracy_mean"] - n_metrics["accuracy_mean"]

        exp1_records.append({
            "dropped_feature": feat,
            "features_used": sub_feats,
            "n_features": 6,
            "clean_accuracy": c_metrics["accuracy"],
            "clean_count": c_metrics["accuracy_count"],
            "clean_macro_f1": c_metrics["macro_f1"],
            "clean_log_loss": c_metrics["log_loss"],
            "clean_ece": c_metrics["ece"],
            "noisy_accuracy_mean": n_metrics["accuracy_mean"],
            "noisy_accuracy_std": n_metrics["accuracy_std"],
            "noisy_count_mean": n_metrics["accuracy_count_mean"],
            "noisy_count_std": n_metrics["accuracy_count_std"],
            "noisy_macro_f1_mean": n_metrics["macro_f1_mean"],
            "noisy_macro_f1_std": n_metrics["macro_f1_std"],
            "noisy_log_loss_mean": n_metrics["log_loss_mean"],
            "noisy_log_loss_std": n_metrics["log_loss_std"],
            "noisy_ece_mean": n_metrics["ece_mean"],
            "noisy_ece_std": n_metrics["ece_std"],
            "clean_acc_drop": float(c_drop),
            "noisy_acc_drop": float(n_drop)
        })

    exp1_time = time.time() - t0

    # Plot Experiment 1 figure
    fig_loo_path = "docs/figures/ablation_leave_one_out.png"
    loo_plot_df = pd.DataFrame(exp1_records[1:]).sort_values("clean_acc_drop", ascending=False)
    plt.figure(figsize=(10, 6))
    y_pos = np.arange(len(loo_plot_df))
    plt.barh(y_pos - 0.2, loo_plot_df["clean_acc_drop"] * 100, height=0.38, color="#2b5c8f", edgecolor="black", label="Clean Acc Drop (%)")
    plt.barh(y_pos + 0.2, loo_plot_df["noisy_acc_drop"] * 100, height=0.38, color="#d95f02", edgecolor="black", label="Noisy Acc Drop (mean 5 seeds, %)")
    plt.yticks(y_pos, loo_plot_df["dropped_feature"], fontsize=10, fontweight="bold")
    plt.xlabel("Accuracy Drop (percentage points from Full Model)", fontsize=11, fontweight="bold")
    plt.title("Leave-One-Feature-Out Impact on Validation Accuracy\n(Extra Trees Isotonic, 330 Val Rows)", fontsize=12, fontweight="bold", pad=12)
    plt.axvline(0, color="black", linestyle="--", linewidth=0.8)
    plt.grid(axis="x", linestyle=":", alpha=0.6)
    plt.legend(frameon=True, facecolor="white")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(fig_loo_path, dpi=200)
    plt.close()

    # =========================================================================
    # EXPERIMENT 2: FEATURE SUBSETS
    # =========================================================================
    print("-" * 80)
    print("EXPERIMENT 2: FEATURE SUBSETS")
    print("-" * 80)
    t0 = time.time()

    subsets_dict = {
        "Full (7 features)": features_all,
        "Soil only (4: N, P, K, ph)": ["N", "P", "K", "ph"],
        "Weather only (3: temp, humidity, rainfall)": ["temperature", "humidity", "rainfall"]
    }

    exp2_records = []
    for s_name, s_feats in subsets_dict.items():
        if s_name == "Full (7 features)":
            c_metrics, n_metrics = ref_clean, ref_noisy
        else:
            base_et = ExtraTreesClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            cal = CalibratedClassifierCV(base_et, method="isotonic", cv=5)
            cal.fit(X_train_full[s_feats], y_train)
            c_metrics, n_metrics = evaluate_model_on_val(
                cal, X_val_full[s_feats], y_val, y_val_idx, classes, s_feats,
                noise_sigmas_dict, lower_bounds_dict, upper_bounds_dict
            )

        exp2_records.append({
            "subset_name": s_name,
            "features_used": s_feats,
            "n_features": len(s_feats),
            "clean_accuracy": c_metrics["accuracy"],
            "clean_count": c_metrics["accuracy_count"],
            "clean_macro_f1": c_metrics["macro_f1"],
            "clean_log_loss": c_metrics["log_loss"],
            "clean_ece": c_metrics["ece"],
            "noisy_accuracy_mean": n_metrics["accuracy_mean"],
            "noisy_accuracy_std": n_metrics["accuracy_std"],
            "noisy_count_mean": n_metrics["accuracy_count_mean"],
            "noisy_count_std": n_metrics["accuracy_count_std"],
            "noisy_macro_f1_mean": n_metrics["macro_f1_mean"],
            "noisy_macro_f1_std": n_metrics["macro_f1_std"],
            "noisy_log_loss_mean": n_metrics["log_loss_mean"],
            "noisy_log_loss_std": n_metrics["log_loss_std"],
            "noisy_ece_mean": n_metrics["ece_mean"],
            "noisy_ece_std": n_metrics["ece_std"]
        })

    exp2_time = time.time() - t0

    # Plot Experiment 2 figure
    fig_subsets_path = "docs/figures/ablation_feature_subsets.png"
    sub_plot_df = pd.DataFrame(exp2_records)
    plt.figure(figsize=(9, 6))
    x_idx = np.arange(len(sub_plot_df))
    plt.bar(x_idx - 0.18, sub_plot_df["clean_accuracy"] * 100, width=0.35, label="Clean Val Accuracy (%)", color="#2b5c8f", edgecolor="black")
    plt.bar(x_idx + 0.18, sub_plot_df["noisy_accuracy_mean"] * 100, yerr=sub_plot_df["noisy_accuracy_std"] * 100, capsize=4, width=0.35, label="Noisy Val Accuracy (mean +/- std, 5 seeds, %)", color="#d95f02", edgecolor="black")
    plt.ylabel("Validation Accuracy (%)", fontsize=11, fontweight="bold")
    plt.title("Feature Subsets: Clean vs Noisy Validation Accuracy\n(Extra Trees Isotonic, 330 Val Rows)", fontsize=12, fontweight="bold", pad=12)
    plt.xticks(x_idx, sub_plot_df["subset_name"], fontsize=10, fontweight="bold")
    plt.ylim(0, 105)
    plt.grid(axis="y", linestyle=":", alpha=0.6)
    plt.legend(frameon=True, facecolor="white")
    plt.tight_layout()
    plt.savefig(fig_subsets_path, dpi=200)
    plt.close()

    # =========================================================================
    # EXPERIMENT 3: LEARNING CURVE
    # =========================================================================
    print("-" * 80)
    print("EXPERIMENT 3: LEARNING CURVE (10%, 25%, 50%, 75%, 100% of train)")
    print("-" * 80)
    t0 = time.time()

    fractions = [0.10, 0.25, 0.50, 0.75, 1.00]
    exp3_records = []

    for frac in fractions:
        seed_clean_accs, seed_clean_counts = [], []
        seed_clean_f1s, seed_clean_lls, seed_clean_eces = [], [], []

        for sub_seed in SUBSAMPLE_SEEDS:
            if frac == 1.00:
                sub_train_idx = np.arange(len(X_train_full))
            else:
                sss = StratifiedShuffleSplit(n_splits=1, train_size=frac, random_state=sub_seed)
                sub_train_idx, _ = next(sss.split(X_train_full, y_train))

            X_sub = X_train_full.iloc[sub_train_idx]
            y_sub = y_train.iloc[sub_train_idx]

            base_et = ExtraTreesClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            cal = CalibratedClassifierCV(base_et, method="isotonic", cv=5)
            cal.fit(X_sub, y_sub)

            c_preds = cal.predict(X_val_full)
            c_probs = cal.predict_proba(X_val_full)

            seed_clean_accs.append(float(accuracy_score(y_val, c_preds)))
            seed_clean_counts.append(int(np.sum(c_preds == y_val.values)))
            seed_clean_f1s.append(float(f1_score(y_val, c_preds, average='macro')))
            seed_clean_lls.append(float(log_loss(y_val, c_probs, labels=classes)))
            seed_clean_eces.append(compute_ece(c_probs, y_val_idx, N_BINS_ECE))

        n_samples_train = int(round(frac * len(train_idx)))
        exp3_records.append({
            "fraction": frac,
            "train_samples": n_samples_train,
            "samples_per_crop": round(n_samples_train / n_classes, 1),
            "subsample_seeds": SUBSAMPLE_SEEDS,
            "clean_accuracy_mean": float(np.mean(seed_clean_accs)),
            "clean_accuracy_std": float(np.std(seed_clean_accs)),
            "clean_count_mean": float(np.mean(seed_clean_counts)),
            "clean_count_std": float(np.std(seed_clean_counts)),
            "clean_macro_f1_mean": float(np.mean(seed_clean_f1s)),
            "clean_macro_f1_std": float(np.std(seed_clean_f1s)),
            "clean_log_loss_mean": float(np.mean(seed_clean_lls)),
            "clean_log_loss_std": float(np.std(seed_clean_lls)),
            "clean_ece_mean": float(np.mean(seed_clean_eces)),
            "clean_ece_std": float(np.std(seed_clean_eces))
        })

    exp3_time = time.time() - t0

    # Plot Experiment 3 figure
    fig_lc_path = "docs/figures/ablation_learning_curve.png"
    lc_plot_df = pd.DataFrame(exp3_records)
    plt.figure(figsize=(9, 6))
    plt.errorbar(
        lc_plot_df["train_samples"], lc_plot_df["clean_accuracy_mean"] * 100,
        yerr=lc_plot_df["clean_accuracy_std"] * 100, fmt="o-", linewidth=2.2, capsize=4,
        color="#2b5c8f", ecolor="#2b5c8f", label="Val Accuracy (mean +/- std, 3 seeds)"
    )
    plt.xlabel("Training Samples (from 1,540 total train split)", fontsize=11, fontweight="bold")
    plt.ylabel("Validation Accuracy (%)", fontsize=11, fontweight="bold")
    plt.title("Learning Curve: Validation Accuracy vs Training Sample Size\n(Extra Trees Isotonic, 330 Val Rows)", fontsize=12, fontweight="bold", pad=12)
    plt.ylim(85, 101)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(frameon=True, facecolor="white")
    plt.tight_layout()
    plt.savefig(fig_lc_path, dpi=200)
    plt.close()

    # =========================================================================
    # EXPERIMENT 4: NOISE SENSITIVITY
    # =========================================================================
    print("-" * 80)
    print("EXPERIMENT 4: NOISE SENSITIVITY (0.5x, 1.0x, 1.5x, 2.0x)")
    print("-" * 80)
    t0 = time.time()

    noise_scales = [0.5, 1.0, 1.5, 2.0]

    # Models: Final Model (ET Isotonic), Uncalibrated Random Forest, Gaussian NB
    models_exp4 = {
        "Extra Trees (Isotonic)": cal_full,
        "Random Forest (Uncalibrated)": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1).fit(X_train_full, y_train),
        "Gaussian NB (Uncalibrated)": Pipeline([('scaler', StandardScaler()), ('clf', GaussianNB())]).fit(X_train_full, y_train)
    }

    exp4_records = {}

    lower_bounds_arr = np.array([PHYSICAL_BOUNDS[f][0] for f in features_all])
    upper_bounds_arr = np.array([PHYSICAL_BOUNDS[f][1] for f in features_all])

    for m_name, model in models_exp4.items():
        exp4_records[m_name] = []
        for scale in noise_scales:
            scale_accs, scale_counts = [], []
            scale_f1s, scale_lls, scale_eces = [], [], []

            for seed in NOISE_SEEDS:
                rng = np.random.default_rng(seed)
                pert = rng.normal(0.0, scale * sigmas_val_full, size=X_val_full.shape)
                noisy_arr = np.clip(X_val_full.values + pert, lower_bounds_arr, upper_bounds_arr)
                noisy_df = pd.DataFrame(noisy_arr, columns=features_all)

                preds = model.predict(noisy_df)
                probs = model.predict_proba(noisy_df)

                scale_accs.append(float(accuracy_score(y_val, preds)))
                scale_counts.append(int(np.sum(preds == y_val.values)))
                scale_f1s.append(float(f1_score(y_val, preds, average='macro')))
                scale_lls.append(float(log_loss(y_val, probs, labels=classes)))
                scale_eces.append(compute_ece(probs, y_val_idx, N_BINS_ECE))

            exp4_records[m_name].append({
                "noise_scale": scale,
                "noise_seeds": NOISE_SEEDS,
                "accuracy_mean": float(np.mean(scale_accs)),
                "accuracy_std": float(np.std(scale_accs)),
                "accuracy_count_mean": float(np.mean(scale_counts)),
                "accuracy_count_std": float(np.std(scale_counts)),
                "macro_f1_mean": float(np.mean(scale_f1s)),
                "macro_f1_std": float(np.std(scale_f1s)),
                "log_loss_mean": float(np.mean(scale_lls)),
                "log_loss_std": float(np.std(scale_lls)),
                "ece_mean": float(np.mean(scale_eces)),
                "ece_std": float(np.std(scale_eces))
            })

    exp4_time = time.time() - t0

    # Plot Experiment 4 figure
    fig_noise_path = "docs/figures/ablation_noise_sensitivity.png"
    plt.figure(figsize=(9, 6))
    colors = {
        "Extra Trees (Isotonic)": "#2b5c8f",
        "Random Forest (Uncalibrated)": "#1b7837",
        "Gaussian NB (Uncalibrated)": "#7570b3"
    }
    for m_name, records in exp4_records.items():
        scales = [r["noise_scale"] for r in records]
        means = [r["accuracy_mean"] * 100 for r in records]
        stds = [r["accuracy_std"] * 100 for r in records]
        plt.errorbar(scales, means, yerr=stds, fmt="o-", linewidth=2, capsize=4, label=m_name, color=colors[m_name])

    plt.xlabel("Noise Scale Multiplier (relative to 1-sigma profile)", fontsize=11, fontweight="bold")
    plt.ylabel("Validation Accuracy (%)", fontsize=11, fontweight="bold")
    plt.title("Noise Sensitivity: Accuracy vs Noise Scale Multiplier\n(330 Val Rows, 5 seeds: 42-46)", fontsize=12, fontweight="bold", pad=12)
    plt.ylim(65, 100)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(frameon=True, facecolor="white")
    plt.tight_layout()
    plt.savefig(fig_noise_path, dpi=200)
    plt.close()

    # =========================================================================
    # EXPERIMENT 5: WEATHER-ONLY CALIBRATED MODEL
    # =========================================================================
    print("-" * 80)
    print("EXPERIMENT 5: WEATHER-ONLY CALIBRATED MODEL")
    print("-" * 80)
    t0 = time.time()

    weather_features = ["temperature", "humidity", "rainfall"]
    X_train_weather = X_train_full[weather_features]
    X_val_weather = X_val_full[weather_features]

    base_et_weather = ExtraTreesClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    cal_weather = CalibratedClassifierCV(base_et_weather, method="isotonic", cv=5)
    cal_weather.fit(X_train_weather, y_train)

    w_clean, w_noisy = evaluate_model_on_val(
        cal_weather, X_val_weather, y_val, y_val_idx, classes, weather_features,
        noise_sigmas_dict, lower_bounds_dict, upper_bounds_dict
    )

    # Save pipeline to backend/models/crop_pipeline_weather.joblib with compress=3
    weather_model_path = "backend/models/crop_pipeline_weather.joblib"
    joblib.dump(cal_weather, weather_model_path, compress=3)
    weather_file_size = os.path.getsize(weather_model_path)

    # Verification: raw 3-column input, predict_proba sums to 1, classes_ alphabetical (22)
    loaded_weather_model = joblib.load(weather_model_path)
    test_raw_input = np.array([[25.0, 80.0, 200.0]])
    test_raw_probs = loaded_weather_model.predict_proba(test_raw_input)

    assert test_raw_probs.shape[1] == 22, f"Expected 22 classes, got {test_raw_probs.shape[1]}"
    assert np.isclose(np.sum(test_raw_probs), 1.0), "predict_proba must sum to 1"
    assert list(loaded_weather_model.classes_) == classes, "classes_ must be alphabetical (22)"

    # Plot reliability diagram for weather model
    fig_rel_weather_path = "docs/figures/reliability_diagram_weather.png"
    clean_w_probs = cal_weather.predict_proba(X_val_weather)
    confs = np.max(clean_w_probs, axis=1)
    preds = np.argmax(clean_w_probs, axis=1)
    accs = (preds == y_val_idx).astype(float)
    bins = np.linspace(0, 1, 11)
    bin_accs, bin_confs = [], []
    for i in range(10):
        in_b = (confs >= bins[i]) & (confs <= bins[i+1]) if i == 0 else (confs > bins[i]) & (confs <= bins[i+1])
        if np.sum(in_b) > 0:
            bin_accs.append(np.mean(accs[in_b]))
            bin_confs.append(np.mean(confs[in_b]))
        else:
            bin_accs.append(np.nan)
            bin_confs.append((bins[i] + bins[i+1]) / 2)

    plt.figure(figsize=(7, 6))
    plt.plot([0, 1], [0, 1], "k--", label="Perfect Calibration")
    plt.plot(bin_confs, bin_accs, "s-", color="#2b5c8f", label=f"Weather Model (ECE = {w_clean['ece']:.4f})")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel("Mean Predicted Confidence", fontsize=11, fontweight="bold")
    plt.ylabel("Empirical Accuracy", fontsize=11, fontweight="bold")
    plt.title("Reliability Diagram: Weather-Only Calibrated Model\n(Isotonic, 330 Val Rows)", fontsize=12, fontweight="bold", pad=12)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(frameon=True, facecolor="white")
    plt.tight_layout()
    plt.savefig(fig_rel_weather_path, dpi=200)
    plt.close()

    exp5_time = time.time() - t0
    total_time = time.time() - start_total_time

    # =========================================================================
    # RECORD IN arena_metrics.json UNDER ablations_experiment_e11
    # =========================================================================
    arena_path = "backend/models/arena_metrics.json"
    with open(arena_path, "r", encoding="utf-8") as f:
        arena = json.load(f)

    arena["ablations_experiment_e11"] = {
        "evaluation_samples": len(val_idx),
        "protocol": "train split (1540 rows) for fitting; validation (330 rows) for evaluation",
        "noise_profile": "primary_1sigma",
        "noise_seeds": NOISE_SEEDS,
        "experiment_1_leave_one_feature_out": exp1_records,
        "experiment_2_feature_subsets": exp2_records,
        "experiment_3_learning_curve": exp3_records,
        "experiment_4_noise_sensitivity": exp4_records,
        "experiment_5_weather_only_model": {
            "pipeline_file": weather_model_path,
            "features": weather_features,
            "classes": classes,
            "clean_metrics": w_clean,
            "noisy_metrics": w_noisy,
            "file_size_bytes": weather_file_size,
            "figures": {
                "reliability_diagram": fig_rel_weather_path
            }
        },
        "figures": {
            "leave_one_out": fig_loo_path,
            "feature_subsets": fig_subsets_path,
            "learning_curve": fig_lc_path,
            "noise_sensitivity": fig_noise_path,
            "reliability_diagram_weather": fig_rel_weather_path
        },
        "runtimes_seconds": {
            "exp1_leave_one_out": round(exp1_time, 2),
            "exp2_feature_subsets": round(exp2_time, 2),
            "exp3_learning_curve": round(exp3_time, 2),
            "exp4_noise_sensitivity": round(exp4_time, 2),
            "exp5_weather_only": round(exp5_time, 2),
            "total_runtime": round(total_time, 2)
        }
    }

    # Banned-word check on generated JSON
    arena_str = json.dumps(arena, indent=2)
    banned_terms = ["ppm", "agronom", "synthetic", "non-overlapping", "kg/ha"]
    matches = {t: arena_str.lower().count(t) for t in banned_terms if arena_str.lower().count(t) > 0}
    assert not matches, f"Banned words detected: {matches}"

    with open(arena_path, "w", encoding="utf-8") as f:
        f.write(arena_str + "\n")

    # =========================================================================
    # PRINT RAW OUTPUTS (TABLES, METRICS, RUNTIMES, SCANS)
    # =========================================================================
    print("\n" + "=" * 80)
    print("RAW EXECUTION OUTPUTS")
    print("=" * 80)

    print("\n--- RUNTIME REPORT ---")
    print(f"Experiment 1 (Leave-one-out):   {exp1_time:.2f} s")
    print(f"Experiment 2 (Feature subsets): {exp2_time:.2f} s")
    print(f"Experiment 3 (Learning curve):  {exp3_time:.2f} s")
    print(f"Experiment 4 (Noise sens.):     {exp4_time:.2f} s")
    print(f"Experiment 5 (Weather-only):    {exp5_time:.2f} s")
    print(f"Total Runtime:                  {total_time:.2f} s")

    print("\n--- TABLE 1: LEAVE-ONE-FEATURE-OUT ABLATION (330 VAL ROWS) ---")
    df_t1 = pd.DataFrame(exp1_records)
    print(df_t1[[
        "dropped_feature", "clean_accuracy", "clean_count", "clean_macro_f1", "clean_log_loss", "clean_ece",
        "noisy_accuracy_mean", "noisy_accuracy_std", "noisy_macro_f1_mean", "noisy_log_loss_mean", "clean_acc_drop"
    ]].to_string(index=False))

    print("\n--- TABLE 2: FEATURE SUBSETS EVALUATION (330 VAL ROWS) ---")
    df_t2 = pd.DataFrame(exp2_records)
    print(df_t2[[
        "subset_name", "clean_accuracy", "clean_count", "clean_macro_f1", "clean_log_loss", "clean_ece",
        "noisy_accuracy_mean", "noisy_accuracy_std", "noisy_macro_f1_mean", "noisy_log_loss_mean"
    ]].to_string(index=False))

    print("\n--- TABLE 3: LEARNING CURVE EVALUATION (330 VAL ROWS, 3 SUBSAMPLES EACH) ---")
    df_t3 = pd.DataFrame(exp3_records)
    print(df_t3[[
        "fraction", "train_samples", "samples_per_crop",
        "clean_accuracy_mean", "clean_accuracy_std", "clean_count_mean", "clean_count_std",
        "clean_macro_f1_mean", "clean_log_loss_mean", "clean_ece_mean"
    ]].to_string(index=False))

    print("\n--- TABLE 4: NOISE SENSITIVITY (5 SEEDS: 42-46) ---")
    for m_name, records in exp4_records.items():
        print(f"\nModel: {m_name}")
        df_t4 = pd.DataFrame(records)
        print(df_t4[[
            "noise_scale", "accuracy_mean", "accuracy_std", "accuracy_count_mean",
            "macro_f1_mean", "macro_f1_std", "log_loss_mean", "log_loss_std", "ece_mean", "ece_std"
        ]].to_string(index=False))

    print("\n--- TABLE 5: WEATHER-ONLY CALIBRATED MODEL (330 VAL ROWS) ---")
    w_clean_row = {
        "condition": "Clean",
        "accuracy": w_clean["accuracy"],
        "accuracy_count": w_clean["accuracy_count"],
        "macro_f1": w_clean["macro_f1"],
        "log_loss": w_clean["log_loss"],
        "ece": w_clean["ece"]
    }
    w_noisy_row = {
        "condition": "Noisy (5 seeds mean +/- std)",
        "accuracy": f"{w_noisy['accuracy_mean']:.4f} +/- {w_noisy['accuracy_std']:.4f}",
        "accuracy_count": f"{w_noisy['accuracy_count_mean']:.1f} +/- {w_noisy['accuracy_count_std']:.1f}",
        "macro_f1": f"{w_noisy['macro_f1_mean']:.4f} +/- {w_noisy['macro_f1_std']:.4f}",
        "log_loss": f"{w_noisy['log_loss_mean']:.4f} +/- {w_noisy['log_loss_std']:.4f}",
        "ece": f"{w_noisy['ece_mean']:.4f} +/- {w_noisy['ece_std']:.4f}"
    }
    print(pd.DataFrame([w_clean_row, w_noisy_row]).to_string(index=False))

    print("\n--- WEATHER-ONLY MODEL SUMMARY ---")
    print(f"Clean Accuracy: {w_clean['accuracy_count']}/330 ({w_clean['accuracy']:.4%})")
    print(f"Noisy Accuracy: {w_noisy['accuracy_count_mean']:.1f} +/- {w_noisy['accuracy_count_std']:.1f}/330 ({w_noisy['accuracy_mean']:.4%} +/- {w_noisy['accuracy_std']:.4%})")
    print(f"Saved Model File: {weather_model_path} ({weather_file_size:,} bytes / {weather_file_size / (1024*1024):.2f} MB)")
    print(f"Verification: raw 3-column input shape: {test_raw_input.shape}, prob sum = {np.sum(test_raw_probs):.6f}, classes count = {len(loaded_weather_model.classes_)}")

    print("\n--- BANNED-WORD SCAN OF arena_metrics.json ---")
    for t in banned_terms:
        cnt = arena_str.lower().count(t)
        print(f"  {t}: {cnt} matches")


if __name__ == "__main__":
    main()
