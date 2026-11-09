import os
import sys
import json
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import log_loss

# Add project root to sys.path
sys.path.insert(0, os.path.abspath("."))

from backend.noise import apply_noise


def calculate_reliability(probs: np.ndarray, y_true: np.ndarray, classes: np.ndarray, bins: int = 10):
    top_conf = probs.max(axis=1)
    pred_idx = probs.argmax(axis=1)
    pred_class = classes[pred_idx]
    correct = (pred_class == y_true).astype(float)
    n = len(y_true)

    ece = 0.0
    bin_points = []

    for b in range(bins):
        low, high = b / bins, (b + 1) / bins
        in_bin = (top_conf > low) & (top_conf <= high)
        bin_count = int(in_bin.sum())
        if bin_count > 0:
            avg_conf = float(top_conf[in_bin].mean())
            avg_acc = float(correct[in_bin].mean())
            weight = float(bin_count / n)
            ece += weight * abs(avg_acc - avg_conf)
            bin_points.append({
                "bin_low": low,
                "bin_high": high,
                "bin_count": bin_count,
                "avg_confidence": avg_conf,
                "actual_accuracy": avg_acc
            })
        else:
            bin_points.append({
                "bin_low": low,
                "bin_high": high,
                "bin_count": 0,
                "avg_confidence": (low + high) / 2.0,
                "actual_accuracy": (low + high) / 2.0
            })

    return float(ece), bin_points


def main():
    data_path = "data/Crop_recommendation.csv"
    train_idx_path = "data/splits/train_idx.npy"
    val_idx_path = "data/splits/val_idx.npy"

    df = pd.read_csv(data_path)
    train_idx = np.load(train_idx_path)
    val_idx = np.load(val_idx_path)

    features = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
    target = 'label'
    classes = np.sort(df[target].unique())

    X_train = df.iloc[train_idx][features]
    y_train = df.iloc[train_idx][target]

    X_val = df.iloc[val_idx][features]
    y_val = df.iloc[val_idx][target]

    print("Model configs:")
    print("  Random Forest: RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1) -> n_estimators: 100")
    print("  Extra Trees:   ExtraTreesClassifier(n_estimators=100, random_state=42, n_jobs=-1)   -> n_estimators: 100")
    print("  Gaussian NB:   StandardScaler + GaussianNB()                                         -> n_estimators: N/A")
    print("  Calibrations:  Uncalibrated, Sigmoid (cv=5), Isotonic (cv=5)")
    print(f"  Training samples:   {len(X_train)} (train_idx.npy only)")
    print(f"  Validation samples: {len(X_val)} (val_idx.npy only)")

    # Model definition matching 03_calibration.py
    model_defs = [
        ("Random Forest", "Uncalibrated", 100, lambda: Pipeline([('clf', RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1))])),
        ("Random Forest", "Sigmoid", 100, lambda: CalibratedClassifierCV(Pipeline([('clf', RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1))]), method="sigmoid", cv=5)),
        ("Random Forest", "Isotonic", 100, lambda: CalibratedClassifierCV(Pipeline([('clf', RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1))]), method="isotonic", cv=5)),
        ("Extra Trees", "Uncalibrated", 100, lambda: Pipeline([('clf', ExtraTreesClassifier(n_estimators=100, random_state=42, n_jobs=-1))])),
        ("Extra Trees", "Sigmoid", 100, lambda: CalibratedClassifierCV(Pipeline([('clf', ExtraTreesClassifier(n_estimators=100, random_state=42, n_jobs=-1))]), method="sigmoid", cv=5)),
        ("Extra Trees", "Isotonic", 100, lambda: CalibratedClassifierCV(Pipeline([('clf', ExtraTreesClassifier(n_estimators=100, random_state=42, n_jobs=-1))]), method="isotonic", cv=5)),
        ("Gaussian NB", "Uncalibrated", None, lambda: Pipeline([('scaler', StandardScaler()), ('clf', GaussianNB())])),
        ("Gaussian NB", "Sigmoid", None, lambda: CalibratedClassifierCV(Pipeline([('scaler', StandardScaler()), ('clf', GaussianNB())]), method="sigmoid", cv=5)),
        ("Gaussian NB", "Isotonic", None, lambda: CalibratedClassifierCV(Pipeline([('scaler', StandardScaler()), ('clf', GaussianNB())]), method="isotonic", cv=5)),
    ]

    fitted_models = {}
    for base_name, cal_method, n_est, builder in model_defs:
        key = f"{base_name} ({cal_method})"
        model = builder()
        model.fit(X_train, y_train)
        fitted_models[key] = (model, base_name, cal_method, n_est)

    noise_seeds = [42, 43, 44, 45, 46]

    # Pre-generate noisy validation sets per seed so all models see identical data
    noisy_val_sets = {}
    for seed in noise_seeds:
        arr = apply_noise(X_val.values, n_samples=1, seed=seed)
        noisy_val_sets[seed] = pd.DataFrame(arr, columns=features)

    # Evaluate each model on each noisy seed
    results = {}
    for key, (model, base_name, cal_method, n_est) in fitted_models.items():
        per_seed_data = {}
        for seed in noise_seeds:
            X_noisy = noisy_val_sets[seed]
            preds = model.predict(X_noisy)
            probs = model.predict_proba(X_noisy)

            count = int(np.sum(preds == y_val.values))
            acc = float(count / len(y_val))
            ll = float(log_loss(y_val, probs, labels=classes))
            ece_val = calculate_reliability(probs, y_val.values, classes, bins=10)[0]

            per_seed_data[seed] = {
                "accuracy_count": count,
                "accuracy": acc,
                "accuracy_str": f"{count}/330",
                "log_loss": ll,
                "ece": ece_val
            }

        counts = [per_seed_data[s]["accuracy_count"] for s in noise_seeds]
        accs = [per_seed_data[s]["accuracy"] for s in noise_seeds]
        lls = [per_seed_data[s]["log_loss"] for s in noise_seeds]
        eces = [per_seed_data[s]["ece"] for s in noise_seeds]

        results[key] = {
            "base_model": base_name,
            "calibration_method": cal_method,
            "n_estimators": n_est,
            "per_seed": per_seed_data,
            "mean": {
                "accuracy_count": float(np.mean(counts)),
                "accuracy": float(np.mean(accs)),
                "log_loss": float(np.mean(lls)),
                "ece": float(np.mean(eces))
            },
            "std": {
                "accuracy_count": float(np.std(counts)),
                "accuracy": float(np.std(accs)),
                "log_loss": float(np.std(lls)),
                "ece": float(np.std(eces))
            }
        }

    # Print evaluation table
    print("\n" + "=" * 110)
    print(f"{'Model Variant':<32} {'Seed':<6} {'Accuracy (count/330)':<22} {'Log-Loss':<16} {'ECE':<16}")
    print("=" * 110)

    for key, res in results.items():
        for s in noise_seeds:
            d = res["per_seed"][s]
            print(f"{key:<32} {s:<6} {d['accuracy_str']:<8} ({d['accuracy']:.4f})     {d['log_loss']:<16.4f} {d['ece']:<16.4f}")
        m = res["mean"]
        s_dev = res["std"]
        print(f"{key + ' (MEAN +/- STD)':<32} {'ALL':<6} {m['accuracy_count']:.2f} +/- {s_dev['accuracy_count']:.2f} ({m['accuracy']:.4f} +/- {s_dev['accuracy']:.4f})  {m['log_loss']:.4f} +/- {s_dev['log_loss']:.4f}   {m['ece']:.4f} +/- {s_dev['ece']:.4f}")
        print("-" * 110)

    # 3. Paired per-seed differences
    et_iso = results["Extra Trees (Isotonic)"]
    gnb_iso = results["Gaussian NB (Isotonic)"]
    rf_iso = results["Random Forest (Isotonic)"]

    pairs = [
        ("ET isotonic minus GNB isotonic", et_iso, gnb_iso),
        ("ET isotonic minus RF isotonic", et_iso, rf_iso)
    ]

    paired_results = {}

    print("\n" + "=" * 110)
    print("PAIRED PER-SEED DIFFERENCES")
    print("=" * 110)

    for pair_name, m_et, m_other in pairs:
        print(f"\n[{pair_name}]")
        print(f"{'Seed':<6} {'Diff Accuracy (count)':<24} {'Diff Accuracy (float)':<24} {'Diff Log-Loss':<20} {'Diff ECE':<20}")
        print("-" * 94)

        diff_counts = []
        diff_accs = []
        diff_lls = []
        diff_eces = []
        per_seed_diff = {}

        for s in noise_seeds:
            d_et = m_et["per_seed"][s]
            d_ot = m_other["per_seed"][s]

            d_count = d_et["accuracy_count"] - d_ot["accuracy_count"]
            d_acc = d_et["accuracy"] - d_ot["accuracy"]
            d_ll = d_et["log_loss"] - d_ot["log_loss"]
            d_ece = d_et["ece"] - d_ot["ece"]

            diff_counts.append(d_count)
            diff_accs.append(d_acc)
            diff_lls.append(d_ll)
            diff_eces.append(d_ece)

            per_seed_diff[str(s)] = {
                "diff_accuracy_count": d_count,
                "diff_accuracy": d_acc,
                "diff_log_loss": d_ll,
                "diff_ece": d_ece
            }

            print(f"{s:<6} {d_count:<24} {d_acc:<24.4f} {d_ll:<20.4f} {d_ece:<20.4f}")

        m_dcount = float(np.mean(diff_counts))
        s_dcount = float(np.std(diff_counts))
        m_dacc = float(np.mean(diff_accs))
        s_dacc = float(np.std(diff_accs))
        m_dll = float(np.mean(diff_lls))
        s_dll = float(np.std(diff_lls))
        m_dece = float(np.mean(diff_eces))
        s_dece = float(np.std(diff_eces))

        print("-" * 94)
        print(f"{'MEAN':<6} {m_dcount:<24.2f} {m_dacc:<24.4f} {m_dll:<20.4f} {m_dece:<20.4f}")
        print(f"{'STD':<6} {s_dcount:<24.2f} {s_dacc:<24.4f} {s_dll:<20.4f} {s_dece:<20.4f}")

        paired_results[pair_name] = {
            "per_seed": per_seed_diff,
            "mean": {
                "diff_accuracy_count": m_dcount,
                "diff_accuracy": m_dacc,
                "diff_log_loss": m_dll,
                "diff_ece": m_dece
            },
            "std": {
                "diff_accuracy_count": s_dcount,
                "diff_accuracy": s_dacc,
                "diff_log_loss": s_dll,
                "diff_ece": s_dece
            }
        }

    # 4. Save to backend/models/arena_metrics.json
    arena_path = "backend/models/arena_metrics.json"
    with open(arena_path, "r", encoding="utf-8") as f:
        arena_data = json.load(f)

    # Format numbers only
    json_models_data = {}
    for key, res in results.items():
        json_models_data[key] = {
            "n_estimators": res["n_estimators"],
            "per_seed": {
                str(s): {
                    "accuracy_count": res["per_seed"][s]["accuracy_count"],
                    "accuracy": res["per_seed"][s]["accuracy"],
                    "log_loss": res["per_seed"][s]["log_loss"],
                    "ece": res["per_seed"][s]["ece"]
                }
                for s in noise_seeds
            },
            "mean": res["mean"],
            "std": res["std"]
        }

    arena_data["ablations_experiment_e11"]["noise_robustness_9_variants"] = {
        "seeds": noise_seeds,
        "models": json_models_data,
        "paired_differences": paired_results
    }

    with open(arena_path, "w", encoding="utf-8") as f:
        json.dump(arena_data, f, indent=2)

    print("\nUpdated backend/models/arena_metrics.json successfully.")


if __name__ == "__main__":
    main()
