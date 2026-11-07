"""
notebooks/03_baseline.py
Role 1: ML Lead (team lead)
Handbook Reference: Section 2 (Role 1 Task 3), Section 5.6 (Leakage Prevention)

Build a baseline scikit-learn Pipeline (StandardScaler + RandomForestClassifier):
- Trained ONLY on train_idx (1,540 rows)
- Evaluated ONCE on val_idx (330 rows)
- Test set is left strictly untouched
- Saved as backend/models/crop_pipeline.joblib
- Saves class order to backend/models/classes.json for downstream roles
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def main():
    print("=" * 60)
    print("STEP 3: BASELINE PIPELINE (StandardScaler + RandomForest)")
    print("=" * 60)

    # 1. Load dataset
    data_path = "data/Crop_recommendation.csv"
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}")
    
    df = pd.read_csv(data_path)
    feature_cols = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
    target_col = 'label'

    X = df[feature_cols]
    y = df[target_col]

    # 2. Load split indices
    train_idx_path = "data/splits/train_idx.npy"
    val_idx_path = "data/splits/val_idx.npy"
    
    if not os.path.exists(train_idx_path) or not os.path.exists(val_idx_path):
        raise FileNotFoundError("Split indices not found in data/splits/. Run Step 2 first.")

    train_idx = np.load(train_idx_path)
    val_idx = np.load(val_idx_path)

    # 3. Create train and val splits (Test set untouched!)
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

    print(f"Features: {feature_cols}")
    print(f"Train samples (X_train): {len(X_train)} rows")
    print(f"Validation samples (X_val): {len(X_val)} rows")
    print("Test set status: Untouched (0 leakage rule)")

    # 4. Construct Pipeline: StandardScaler + RandomForestClassifier
    # Using Pipeline ensures feature scaling is fitted strictly on training data
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1))
    ])

    # 5. Fit pipeline on training rows only
    print("\nFitting pipeline on X_train...")
    pipeline.fit(X_train, y_train)
    print("Pipeline fitting complete.")

    # 6. Evaluate once on validation set
    y_val_pred = pipeline.predict(X_val)
    val_accuracy = accuracy_score(y_val, y_val_pred)
    print(f"\nValidation Accuracy: {val_accuracy:.4f} ({val_accuracy * 100:.2f}%)")

    # 7. Extract class ordering
    # Classes are ordered alphabetically by scikit-learn
    classes = list(pipeline.classes_)
    print(f"\nClass order ({len(classes)} classes):")
    print(classes)

    # 8. Save artifacts to backend/models/
    os.makedirs("backend/models", exist_ok=True)
    pipeline_path = "backend/models/crop_pipeline.joblib"
    classes_path = "backend/models/classes.json"

    joblib.dump(pipeline, pipeline_path)
    with open(classes_path, "w", encoding="utf-8") as f:
        json.dump(classes, f, indent=2)

    print(f"\nSaved baseline pipeline to: {pipeline_path} ({os.path.getsize(pipeline_path)} bytes)")
    print(f"Saved class order to:       {classes_path} ({os.path.getsize(classes_path)} bytes)")

    # 9. Verification: Reload saved pipeline from disk and test inference
    reloaded_pipeline = joblib.load(pipeline_path)
    sample_row = X_val.iloc[[0]]
    sample_pred = reloaded_pipeline.predict(sample_row)[0]
    sample_proba = reloaded_pipeline.predict_proba(sample_row)[0]
    top_class_idx = np.argmax(sample_proba)
    top_class_name = reloaded_pipeline.classes_[top_class_idx]
    top_proba = sample_proba[top_class_idx]

    print("\nInference verification on reloaded pipeline:")
    print(f"  Input sample: {sample_row.to_dict(orient='records')[0]}")
    print(f"  Predicted crop: '{sample_pred}' (confidence: {top_proba * 100:.2f}%)")
    assert sample_pred == top_class_name, "Predicted class does not match top probability class!"
    assert reloaded_pipeline.score(X_val, y_val) == val_accuracy, "Reloaded score mismatch!"
    print("  Verification assertion passed: Pipeline is 100% operational for Roles 2 & 3.")
    print("=" * 60)

if __name__ == "__main__":
    main()

