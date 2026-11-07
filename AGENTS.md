# AGENTS.md: shared rules for every AI assistant working on Crop Trust Engine

Read this file first, then the role file the person tells you they are (docs/roles/ROLE<N>_*.md), then only the sections of docs/PROJECT_GUIDE.md that the role file points to. Section numbers refer to the handbook numbering (for example "Section 7.3" is the five stages of the Trust Engine).

## What the project is
A crop recommender (7 inputs: N, P, K, temperature, humidity, ph, rainfall; 22 crops; Kaggle "Crop Recommendation Dataset", 2,200 rows, 100 rows per crop) that knows when not to trust itself. A "Trust Engine" runs five stages: (1) validate, (2) OOD gate, (3) calibrated classifier, (4) stability check, (5) verdict (recommend / caution / refuse) with a reason. Backend FastAPI (REST, JSON), frontend JavaScript (React + Vite recommended), Postman collection, Docker deployment.

## Locked decisions (never change without asking the person)
- Kaggle data for training; 9 classifiers compared properly (stratified repeated k-fold, identical folds).
- Five-stage Trust Engine; three verdicts, each with a reason.
- OOD: per-crop Mahalanobis (Ledoit-Wolf) + Isolation Forest, proven with manufactured unknowns (leave-one-crop-out, box-uniform, shuffled columns).
- Contract first: the JSON shapes below are frozen once the team agrees them.
- A Refuse verdict is HTTP 200. Invalid input (for example ph = 40) is HTTP 422.

## Repo layout and who owns what
- notebooks/01_eda, 02_compare_models, 03_calibration: Role 1
- notebooks/04_ood_eval, 05_stability: Role 2
- backend/trust_engine.py and backend/models/ thresholds.json, noise_model.json, conformal.json, ood_stats.joblib, iforest.joblib: Role 2
- backend/models/ crop_pipeline.joblib, class_stats.json, model_info.json, reference_distribution.json, arena_metrics.json: Role 1
- backend/main.py, backend/tests/, requirements.txt, Dockerfile, .github/, postman/: Role 3
- frontend/ (including frontend/mock/predict_examples.json): Role 4
- docs/: each writer owns their chapter; Role 1 merges the report

Only edit files your role owns. For anything else, write what you need in a note or open a pull request for the owner to review.

## The frozen API contract (summary)
Input (POST /predict?risk=conservative|balanced|bold): N, P, K, temperature, humidity, ph, rainfall (floats). Physical bounds: N 0-200, P 0-200, K 0-300, temperature -10..60, humidity 0-100, ph 0-14, rainfall 0-1000.
Reply fields: verdict ("recommend" | "caution" | "refuse"); top_crops [{crop, probability}] (empty on refuse); ood {flagged, percentile, unusual_pair?, nearest_known?}; stability (0..1, null on refuse); trust_score (0..100, 0 on refuse); explanation (string); model_version (string); optional conformal_set, measure_next.
Routes: GET /health, GET /model-info, POST /predict, POST /predict/batch (CSV, max 1 MB, max 500 rows), POST /explain, GET /arena, GET /monitor (bonus).
The three example replies live in frontend/mock/predict_examples.json.

## Rules you must follow
1. **No leakage.** Split first. Fit scalers, models, calibrators, OOD detectors and thresholds on training or validation data only. The test set is used once, at the end. Use scikit-learn Pipelines.
2. **Never invent results.** Every number you report must come from code that was actually run on this data. Numbers in the guide are placeholders.
3. **Placeholders stay placeholders.** Thresholds (percentiles, 0.70, 0.20, 0.80), noise sizes and risk profiles are starting values to be tuned on validation data.
4. **Plain, explainable code.** The team must be able to explain every line in a viva. No clever one-liners, no unexplained libraries. Add short comments saying why.
5. **Anything the API imports is a .py file**, never a notebook.
6. **Pin versions** (requirements.txt). Save the whole pipeline (scaler + model + calibrator) as one joblib file to avoid training-serving skew.
7. **Tests.** Add a pytest or a saved output for what you build.
8. **Never commit secrets or large data files.** Use environment variables.
9. **Ask before changing** a locked decision, a contract field, or another role's files.
10. **Say what you could not verify.** If you could not run something, say so plainly.

## How to start a session
1. Read this file and the person's role file.
2. Reply with a short summary: the role's tasks, the files it owns, what it needs from others, and what you would build first. Write no code yet.
3. Wait for the person to pick one task. Do that task only, run it, show the output, then stop.

## Git
Branch per task: role<N>/<topic>. Small commits, clear messages. Merge through pull requests with one reviewer (Roles 1 and 2 review each other; Roles 3 and 4 review each other). Keep main runnable.
