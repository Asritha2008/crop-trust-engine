"""Validation Script: Reproduce Balanced Caution Candidate and Probability Statistics.

Author: Role 2 (Trust Engine Engineer)
Project: Crop Trust Engine
Branch: role2/engine-candidates (Commit bc39719)

Purpose:
  Independently verify and reproduce two key statistical claims regarding Stage 5
  caution responses under the default 'Balanced' risk profile:
    Claim 1: "25 of 38 Balanced caution responses have exactly one crop with probability >= 0.05."
    Claim 2: "21 of 38 Balanced caution responses have a second-highest crop probability < 0.01."

Scientific & Methodological Constraints:
  1. VALIDATION SPLIT ONLY:
     This script evaluates exclusively the 330 validation samples (data/splits/val_idx.npy).
     The test split (data/splits/test_idx.npy) remains strictly quarantined and untouched.
     Evaluating test data for post-hoc threshold analysis or reporting compromises the
     integrity of the held-out test benchmark.
  2. SECOND-HIGHEST PROBABILITY (p2) ACROSS ALL 22 CLASSES:
     p2 represents the second-highest calibrated posterior probability across all 22 crop
     classes, evaluated via engine.assess_probabilities(). It is NOT derived merely from
     the filtered candidate list (which truncates at p >= 0.05). If p1 = 1.0000, all other
     21 classes have p = 0.0000, so p2 = 0.0000.
  3. REAL PRODUCTION ENGINE & PIPELINE:
     Loads the calibrated champion pipeline (crop_pipeline_calibrated.joblib, Extra Trees with
     5-fold isotonic regression) and thresholds.json through TrustEngine.load().
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

# Set up repository root on path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.trust_engine import TrustEngine

# File Paths
DATA_PATH = REPO_ROOT / "data" / "Crop_recommendation.csv"
VAL_IDX_PATH = REPO_ROOT / "data" / "splits" / "val_idx.npy"
MODEL_DIR = REPO_ROOT / "backend" / "models"


def main() -> None:
    print("=" * 80)
    print("REPRODUCIBILITY EVALUATION: BALANCED CAUTION CANDIDATES & PROBABILITIES")
    print("=" * 80)

    # 1. Load original dataset and validation partition only
    if not DATA_PATH.is_file():
        raise FileNotFoundError(f"Dataset not found at: {DATA_PATH}")
    if not VAL_IDX_PATH.is_file():
        raise FileNotFoundError(f"Validation split index file not found at: {VAL_IDX_PATH}")

    df_full = pd.read_csv(DATA_PATH)
    val_idx = np.load(VAL_IDX_PATH)
    total_val_rows = len(val_idx)

    print(f"Dataset loaded: {len(df_full)} total rows from {DATA_PATH}")
    print(f"Validation partition: {total_val_rows} rows from {VAL_IDX_PATH}")
    print("HYGIENE VERIFICATION: Test split (test_idx.npy) is NOT loaded or accessed.")

    # 2. Load the real production calibrated TrustEngine
    print(f"Loading TrustEngine from {MODEL_DIR}...")
    engine = TrustEngine.load(MODEL_DIR)

    # 3. Evaluate each validation row under risk='balanced'
    recommend_count = 0
    caution_count = 0
    refuse_count = 0

    caution_rows: List[Dict[str, Any]] = []

    for idx in val_idx:
        row_dict = {f: float(df_full.iloc[idx][f]) for f in TrustEngine.REQUIRED_FEATURES}
        res = engine.assess(row_dict, risk="balanced")
        verdict = res["verdict"]

        if verdict == "recommend":
            recommend_count += 1
        elif verdict == "caution":
            caution_count += 1
            # Obtain unrounded probability distribution across all 22 classes
            prob_res = engine.assess_probabilities(row_dict)

            # Candidates array populated by Stage 5 for caution responses (unrounded p >= 0.05)
            candidates = res.get("candidates") or []
            n_candidates = len(candidates)

            # Top and second-highest probabilities across all 22 crop classes
            top_crop = prob_res["top_crop"]
            top_prob = float(prob_res["top_probability"])
            second_crop = prob_res["second_crop"]
            second_prob = float(prob_res["second_probability"])

            caution_rows.append({
                "val_idx": int(idx),
                "true_crop": str(df_full.iloc[idx]["label"]),
                "verdict": verdict,
                "top_crop": top_crop,
                "top_prob": top_prob,
                "second_crop": second_crop,
                "second_prob": second_prob,
                "candidates": candidates,
                "n_candidates": n_candidates,
                "has_single_candidate": (n_candidates == 1),
                "second_prob_below_001": (second_prob < 0.01),
                "explanation": res.get("explanation", ""),
            })
        elif verdict == "refuse":
            refuse_count += 1

    # 4. Filter subgroups
    single_cand_rows = [r for r in caution_rows if r["has_single_candidate"]]
    p2_under_001_rows = [r for r in caution_rows if r["second_prob_below_001"]]

    count_single_cand = len(single_cand_rows)
    count_p2_under_001 = len(p2_under_001_rows)

    # 5. Display comprehensive reproducibility report
    print("\n" + "-" * 80)
    print("STAGE 5 VERDICT DISTRIBUTION (Validation Set, Risk='Balanced'):")
    print(f"  Recommend: {recommend_count:3d} ({recommend_count / total_val_rows * 100:.2f}%)")
    print(f"  Caution:   {caution_count:3d} ({caution_count / total_val_rows * 100:.2f}%)")
    print(f"  Refuse:    {refuse_count:3d} ({refuse_count / total_val_rows * 100:.2f}%)")
    print(f"  Total:     {total_val_rows:3d}")
    print("-" * 80)

    print("\n" + "=" * 98)
    print(f"{'Row Idx':<8} | {'True Crop':<11} | {'Top Crop':<11} | {'p1 (Top)':>9} | {'2nd Crop':<11} | {'p2 (2nd)':>9} | {'Cands':>5} | {'p2 < 0.01':>9}")
    print("-" * 98)
    for r in caution_rows:
        flag_p2 = "YES" if r["second_prob_below_001"] else "no"
        print(
            f"{r['val_idx']:<8d} | {r['true_crop']:<11} | {r['top_crop']:<11} | "
            f"{r['top_prob']:9.4f} | {r['second_crop']:<11} | {r['second_prob']:9.4f} | "
            f"{r['n_candidates']:5d} | {flag_p2:>9}"
        )
    print("=" * 98)

    print("\n" + "-" * 80)
    print("SUBGROUP FINDINGS & INDICES:")
    print(f"  Total Balanced Caution Responses: {caution_count}")
    print(f"  1. Responses with exactly 1 candidate (p >= 0.05): {count_single_cand} of {caution_count} ({count_single_cand / caution_count * 100:.2f}%)")
    print(f"     Validation indices ({count_single_cand}): {[r['val_idx'] for r in single_cand_rows]}")
    print(f"  2. Responses with second-highest probability p2 < 0.01: {count_p2_under_001} of {caution_count} ({count_p2_under_001 / caution_count * 100:.2f}%)")
    print(f"     Validation indices ({count_p2_under_001}): {[r['val_idx'] for r in p2_under_001_rows]}")
    print("-" * 80)

    # 6. Explicit Assertions
    print("\nExecuting verification assertions...")
    assert total_val_rows == 330, (
        f"Assertion Failed: Expected 330 validation rows, got {total_val_rows}"
    )
    assert caution_count == 38, (
        f"Assertion Failed: Expected 38 Balanced caution responses, got {caution_count}"
    )
    assert count_single_cand == 25, (
        f"Assertion Failed: Expected 25 caution responses with exactly 1 candidate, got {count_single_cand}"
    )
    assert count_p2_under_001 == 21, (
        f"Assertion Failed: Expected 21 caution responses with p2 < 0.01, got {count_p2_under_001}"
    )

    print("SUCCESS: All 4 assertions PASSED bitwise.")
    print("=" * 80)


if __name__ == "__main__":
    main()

