"""Reference distribution generator for PSI drift monitoring (Task 9 / Stage 6).

Role 1: ML Lead
Generates reference distributions from the training split (1540 rows) across the 7
canonical features in order: N, P, K, temperature, humidity, ph, rainfall.
Writes reference values with the flat 7-key layout to a temporary file
and validates against the committed artifact backend/models/reference_distribution.json.
"""

from pathlib import Path
import json
import tempfile
import numpy as np
import pandas as pd

FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]


def find_repo_root() -> Path:
    """Locate repository root directory."""
    cwd = Path.cwd()
    if (cwd / "data" / "Crop_recommendation.csv").exists():
        return cwd
    if (cwd.parent / "data" / "Crop_recommendation.csv").exists():
        return cwd.parent
    return cwd


def generate_reference_distribution(repo_root: Path, output_path: Path) -> dict:
    """Load training split and CSV, extract 7 features, and save flat JSON."""
    csv_path = repo_root / "data" / "Crop_recommendation.csv"
    train_idx_path = repo_root / "data" / "splits" / "train_idx.npy"

    train_idx = np.load(train_idx_path)
    df = pd.read_csv(csv_path)

    train_df = df.iloc[train_idx]

    ref_dict = {
        feat: [float(val) for val in train_df[feat].tolist()]
        for feat in FEATURES
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(ref_dict, f, indent=2)

    return ref_dict


def main():
    repo_root = find_repo_root()
    committed_path = repo_root / "backend" / "models" / "reference_distribution.json"

    # Write to a temporary file, not over the committed file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp_file:
        tmp_path = Path(tmp_file.name)

    try:
        ref_dict = generate_reference_distribution(repo_root, tmp_path)

        # Print each feature's list length
        print("Feature list lengths:")
        for feat in FEATURES:
            print(f"  {feat}: {len(ref_dict[feat])}")

        # Check against committed file
        if not committed_path.exists():
            print(f"Committed file not found at {committed_path}")
            return

        with open(committed_path, "r", encoding="utf-8") as f:
            committed_dict = json.load(f)

        with open(tmp_path, "r", encoding="utf-8") as f:
            temp_dict = json.load(f)

        identical = (temp_dict == committed_dict)
        print(f"Temporary output is value-for-value identical to committed file: {identical}")

        if not identical:
            for feat in FEATURES:
                match = (temp_dict.get(feat) == committed_dict.get(feat))
                if not match:
                    print(f"  Mismatch in feature: {feat}")
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


if __name__ == "__main__":
    main()

