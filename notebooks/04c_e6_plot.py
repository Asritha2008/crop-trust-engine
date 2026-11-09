"""Experiment E6: False-Alarm Rate vs. Detection Rate Trade-off Plot.

Role 2: Trust Engine Engineer
Reproduces Experiment E6 threshold calibration from notebooks/04_ood_eval.ipynb:
  - 5-fold out-of-fold reference scores (1,540 rows)
  - Validation split (val_calib: 220 rows, val_eval: 110 rows)
  - Box-Uniform unknowns (1,000 samples, seed=202)
  - Shuffled-Column unknowns (1,000 samples, seed=404)
  - Verification of 6-row calibration trade-off table
  - Generates docs/figures/e6_false_alarm_vs_detection.png

Usage:
  python notebooks/04c_e6_plot.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler


def main() -> None:
    seed = 42
    np.random.seed(seed)

    canonical_features = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]

    repo_root = Path(__file__).resolve().parent.parent
    data_path = repo_root / "data" / "Crop_recommendation.csv"
    train_idx_path = repo_root / "data" / "splits" / "train_idx.npy"
    val_idx_path = repo_root / "data" / "splits" / "val_idx.npy"
    thresholds_path = repo_root / "backend" / "models" / "thresholds.json"
    figures_dir = repo_root / "docs" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    out_fig_path = figures_dir / "e6_false_alarm_vs_detection.png"

    # 1. Load data and split indices (Zero test leakage: test_idx.npy is never opened)
    df = pd.read_csv(data_path)
    train_idx = np.load(train_idx_path)
    val_idx = np.load(val_idx_path)

    X = df[canonical_features].values
    y = df["label"].values

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    classes = sorted(list(np.unique(y_train)))

    # 2. Fit global scaler on X_train only
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # 3. Fit per-crop Ledoit-Wolf on X_train_scaled only
    crop_stats = {}
    for crop in classes:
        mask = y_train == crop
        Xc_scaled = X_train_scaled[mask]
        lw = LedoitWolf(assume_centered=False).fit(Xc_scaled)
        crop_stats[crop] = {
            "mean": lw.location_,
            "inv_cov": lw.precision_,
        }

    def compute_mahalanobis_scores(X_input_scaled: np.ndarray) -> np.ndarray:
        n_samples = len(X_input_scaled)
        all_d2 = np.zeros((n_samples, len(classes)))
        for idx, crop in enumerate(classes):
            diff = X_input_scaled - crop_stats[crop]["mean"]
            prec = crop_stats[crop]["inv_cov"]
            all_d2[:, idx] = np.sum((diff @ prec) * diff, axis=1)
        return np.min(all_d2, axis=1)

    # 4. Stratified Validation Partitioning (10 calib, 5 eval per crop)
    rng_val = np.random.default_rng(seed=seed)
    calib_idx_list = []
    eval_idx_list = []
    for crop in classes:
        crop_pos = np.where(y_val == crop)[0]
        shuffled = rng_val.permutation(crop_pos)
        calib_idx_list.extend(shuffled[:10])
        eval_idx_list.extend(shuffled[10:])

    eval_sub_idx = np.array(eval_idx_list)
    X_val_eval = X_val[eval_sub_idx]
    X_val_eval_scaled = X_val_scaled[eval_sub_idx]

    # 5. Compute 5-fold Stratified OOF scores on X_train (1,540 rows)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    oof_mahal_scores = np.zeros(len(X_train))
    for tr_fold_idx, oof_fold_idx in skf.split(X_train, y_train):
        X_tr_f, y_tr_f = X_train[tr_fold_idx], y_train[tr_fold_idx]
        X_oof_f = X_train[oof_fold_idx]

        sc_f = StandardScaler().fit(X_tr_f)
        X_tr_f_sc = sc_f.transform(X_tr_f)
        X_oof_f_sc = sc_f.transform(X_oof_f)

        stats_f = {}
        for crop in classes:
            mask = y_tr_f == crop
            lw_f = LedoitWolf(assume_centered=False).fit(X_tr_f_sc[mask])
            stats_f[crop] = {
                "mean": lw_f.location_,
                "inv_cov": lw_f.precision_,
            }

        for i, x_sc in enumerate(X_oof_f_sc):
            min_d = np.inf
            for crop in classes:
                diff = x_sc - stats_f[crop]["mean"]
                d2 = float(diff @ stats_f[crop]["inv_cov"] @ diff)
                if d2 < min_d:
                    min_d = d2
            oof_mahal_scores[oof_fold_idx[i]] = min_d

    # 6. Generate Evaluation Synthetic Unknowns (independent seeds)
    train_min = X_train.min(axis=0)
    train_max = X_train.max(axis=0)

    def generate_box_uniform(n: int, bounds_min: np.ndarray, bounds_max: np.ndarray, s: int) -> np.ndarray:
        rng = np.random.default_rng(s)
        return rng.uniform(bounds_min, bounds_max, size=(n, len(bounds_min)))

    def generate_shuffled_columns(X_source: np.ndarray, n: int, s: int) -> np.ndarray:
        rng = np.random.default_rng(s)
        idx = rng.choice(len(X_source), size=n, replace=True)
        sampled = X_source[idx].copy()
        for col in range(sampled.shape[1]):
            rng.shuffle(sampled[:, col])
        return sampled

    box_eval = generate_box_uniform(1000, train_min, train_max, s=202)
    shuf_eval = generate_shuffled_columns(X_val_eval, 1000, s=404)

    box_eval_scaled = scaler.transform(box_eval)
    shuf_eval_scaled = scaler.transform(shuf_eval)

    m_known = compute_mahalanobis_scores(X_val_eval_scaled)
    m_box = compute_mahalanobis_scores(box_eval_scaled)
    m_shuf = compute_mahalanobis_scores(shuf_eval_scaled)

    # 7. Reproduce candidate percentiles table and verify match
    candidate_percentiles = [90.0, 95.0, 97.0, 98.0, 99.0, 99.5]
    reproduced_rows = []
    for p in candidate_percentiles:
        cutoff_oof = float(np.percentile(oof_mahal_scores, p))
        val_fa = float(np.mean(m_known > cutoff_oof) * 100.0)
        box_det = float(np.mean(m_box > cutoff_oof) * 100.0)
        shuf_det = float(np.mean(m_shuf > cutoff_oof) * 100.0)
        reproduced_rows.append({
            "Percentile": p,
            "Cutoff d2": round(cutoff_oof, 2),
            "Val False Alarm (%)": round(val_fa, 1),
            "Box Detection (%)": round(box_det, 1),
            "Shuffled Detection (%)": round(shuf_det, 1),
        })

    reproduced_df = pd.DataFrame(reproduced_rows)
    print("=== REPRODUCED EXPERIMENT E6 TABLE ===")
    print(reproduced_df.to_string(index=False))

    expected_table = [
        (90.0, 9.53, 9.1, 99.8, 95.6),
        (95.0, 10.78, 5.5, 99.8, 94.2),
        (97.0, 11.71, 2.7, 99.8, 92.7),
        (98.0, 12.18, 2.7, 99.8, 91.6),
        (99.0, 12.92, 2.7, 99.7, 90.1),
        (99.5, 13.75, 1.8, 99.5, 88.0),
    ]

    is_match = True
    for idx, exp in enumerate(expected_table):
        row = reproduced_rows[idx]
        if (
            row["Percentile"] != exp[0]
            or row["Cutoff d2"] != exp[1]
            or row["Val False Alarm (%)"] != exp[2]
            or row["Box Detection (%)"] != exp[3]
            or row["Shuffled Detection (%)"] != exp[4]
        ):
            is_match = False
            print(f"MISMATCH at row {idx}: got {row}, expected {exp}")

    if not is_match:
        print("TABLE VERIFICATION: MISMATCH. Stopping execution per instructions.")
        return

    print("TABLE VERIFICATION: MATCH")

    # 8. Load locked thresholds from backend/models/thresholds.json
    with open(thresholds_path, "r", encoding="utf-8") as f:
        thresholds_data = json.load(f)

    profiles = thresholds_data["profiles"]
    cons_cut = float(profiles["conservative"]["ood_refuse_threshold"])
    bal_cut = float(profiles["balanced"]["ood_refuse_threshold"])
    bold_cut = float(profiles["bold"]["ood_refuse_threshold"])

    # Compute exact coordinates for markers
    profiles_to_plot = [
        ("Conservative", cons_cut, "#2ca02c", "s"),
        ("Balanced", bal_cut, "#1f77b4", "o"),
        ("Bold", bold_cut, "#d62728", "^"),
    ]

    print("\n=== MARKER COORDINATES (backend/models/thresholds.json) ===")
    marker_coords = {}
    for name, cut, color, marker in profiles_to_plot:
        fa = float(np.mean(m_known > cut) * 100.0)
        b_det = float(np.mean(m_box > cut) * 100.0)
        s_det = float(np.mean(m_shuf > cut) * 100.0)
        marker_coords[name] = {
            "cutoff_d2": cut,
            "fa_pct": fa,
            "box_det_pct": b_det,
            "shuf_det_pct": s_det,
        }
        print(f"Profile: {name:12s} | Cutoff d2: {cut:5.2f} | Val FA: {fa:5.2f}% | Box Det: {b_det:5.2f}% | Shuffled Det: {s_det:5.2f}%")

    # 9. Fine grid for ROC-style trade-off curve
    # Percentiles from 80.0 to 99.9 in steps of 0.1
    p_grid = np.linspace(80.0, 99.9, 250)
    curve_fa = []
    curve_box_det = []
    curve_shuf_det = []

    for p in p_grid:
        c = float(np.percentile(oof_mahal_scores, p))
        curve_fa.append(float(np.mean(m_known > c) * 100.0))
        curve_box_det.append(float(np.mean(m_box > c) * 100.0))
        curve_shuf_det.append(float(np.mean(m_shuf > c) * 100.0))

    curve_fa = np.array(curve_fa)
    curve_box_det = np.array(curve_box_det)
    curve_shuf_det = np.array(curve_shuf_det)

    # 10. Create publication-quality figure
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, ax = plt.subplots(figsize=(10, 6.5))

    # Plot trade-off curves
    ax.plot(
        curve_fa,
        curve_box_det,
        label="Box-Uniform Unknowns (N=1,000, seed 202)",
        color="#1f77b4",
        linewidth=2.4,
        alpha=0.9,
    )
    ax.plot(
        curve_fa,
        curve_shuf_det,
        label="Shuffled-Column Unknowns (N=1,000, seed 404)",
        color="#ff7f0e",
        linewidth=2.4,
        alpha=0.9,
    )

    # Plot markers for the 3 chosen thresholds on the Shuffled curve and Box curve
    box_marker_sizes = {
        "Conservative": 110,
        "Balanced": 45,
        "Bold": 75,
    }

    for name, cut, color, marker in profiles_to_plot:
        coords = marker_coords[name]
        fa = coords["fa_pct"]
        s_det = coords["shuf_det_pct"]
        b_det = coords["box_det_pct"]

        # Marker on Shuffled-column curve (primary discriminating boundary)
        ax.scatter(
            [fa],
            [s_det],
            color=color,
            marker=marker,
            s=120,
            zorder=5,
            edgecolors="black",
            linewidths=1.2,
            label=f"{name} Refusal (d²={cut:.2f}): FA={fa:.1f}%, Det={s_det:.1f}%",
        )

        # Subtle marker on Box curve (distinguishable sizes for overlapping coords)
        ax.scatter(
            [fa],
            [b_det],
            color=color,
            marker=marker,
            s=box_marker_sizes[name],
            zorder=6 if name == "Balanced" else 5,
            edgecolors="black",
            linewidths=0.8,
            alpha=0.65,
        )

    # Annotations on the curve with callouts
    # Offset labels to avoid overlap; Bold placed to the right and below inside plot area
    offsets = {
        "Conservative": (14, 10),
        "Balanced": (16, -16),
        "Bold": (20, -36),
    }
    for name, cut, color, marker in profiles_to_plot:
        coords = marker_coords[name]
        dx, dy = offsets[name]
        ax.annotate(
            f"{name} (d²={cut:.2f})\nFA={coords['fa_pct']:.1f}%, Det={coords['shuf_det_pct']:.1f}%",
            xy=(coords["fa_pct"], coords["shuf_det_pct"]),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=9.5,
            fontweight="bold",
            color=color,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=color, lw=1.2, alpha=0.95),
            arrowprops=dict(arrowstyle="->", color=color, lw=1.2),
        )

    ax.set_title(
        "Experiment E6: False-Alarm Rate vs. Detection Rate Trade-off\nPer-Crop Mahalanobis Detector (Ledoit-Wolf) on val_eval (N=110 rows)",
        fontsize=13,
        fontweight="bold",
        pad=14,
    )
    ax.set_xlabel("False Alarm Rate on Known Validation Rows (%) [val_eval, N=110]", fontsize=11, fontweight="semibold")
    ax.set_ylabel("Detection Rate on Manufactured Unknowns (%) [N=1,000]", fontsize=11, fontweight="semibold")

    ax.set_xlim(-0.5, 22.0)
    ax.set_ylim(80.0, 101.5)
    ax.grid(True, linestyle="--", alpha=0.6)

    # Add legend entry explaining the faint box-unknown markers
    import matplotlib.lines as mlines
    faint_proxy = mlines.Line2D(
        [],
        [],
        color="gray",
        marker="o",
        linestyle="None",
        markersize=7,
        alpha=0.65,
        markeredgecolor="black",
        markeredgewidth=0.8,
        label="Faint markers: same three cutoffs on box unknowns",
    )
    handles, labels = ax.get_legend_handles_labels()
    handles.append(faint_proxy)
    labels.append(faint_proxy.get_label())
    ax.legend(handles=handles, labels=labels, loc="lower right", frameon=True, framealpha=0.95, fontsize=9.2)

    # Required caption line
    fig.text(
        0.5,
        0.015,
        "False alarm uses 110 validation rows, so one row is about 0.9 points.",
        ha="center",
        fontsize=10,
        fontstyle="italic",
        color="#333333",
    )

    plt.tight_layout(rect=(0, 0.04, 1, 1))
    plt.savefig(out_fig_path, dpi=200)
    plt.close()

    print(f"\nFigure successfully generated and saved to:\n  Path: {out_fig_path}")
    print(f"  File size: {os.path.getsize(out_fig_path)} bytes")


if __name__ == "__main__":
    main()

