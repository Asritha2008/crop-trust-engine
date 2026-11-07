# Role 1: ML Lead (team lead): data, models, calibration

Also read: AGENTS.md. Handbook sections: 2 (Role 1), 3.5, 5 (all), 7.6 (E1 to E4, E11), 8.4 (reality check).

## Mission
Produce the trained, honestly evaluated classifier and every number about it. This is the part the faculty calls most important. You are also the team lead and edit the final report.

## You own (edit freely)
notebooks/01_eda, 02_compare_models, 03_calibration; split index files and seed; backend/models/crop_pipeline.joblib, class_stats.json, model_info.json, reference_distribution.json; the model-side part of arena_metrics.json; the weather-only pipeline; the reality-check experiment; docs/ report merge.

## Tasks, in order (pick one at a time)
1. Confirm the data: df.describe(), duplicates, class balance, and what period the rainfall column covers.
2. Day 2 to 3: stratified train / validation / test splits (suggested 70/15/15, about 1540/330/330) saved as index files with the seed. Commit them: Role 2 is waiting on these.
3. Day 3 to 4: baseline pipeline (scaler + Random Forest) saved as crop_pipeline.joblib so Roles 2 and 3 can build against a real file.
4. E1 EDA: distributions, correlations, per-crop ranges and overlaps, pairwise scatter plots. Find feature pairs that never co-occur (future demo inputs). Write class_stats.json (per-crop means, spreads, percentiles).
5. E2 comparison: Logistic Regression, KNN, SVM (RBF), Decision Tree, Random Forest, Extra Trees, Gradient Boosting or XGBoost, Gaussian Naive Bayes, small MLP. Pipelines; stratified 5-fold repeated 3 times; identical folds; tuning inside folds. Report mean and std of accuracy, macro-F1, log-loss, ECE, times, and accuracy on noisy inputs. Paired significance test (Wilcoxon or McNemar). Choose the final model for a stated reason.
6. E3 class-wise: per-crop precision, recall, F1; 22x22 confusion heatmap; written reasons for confused pairs. List real misclassified rows (demo ambiguous inputs).
7. E4 calibration: CalibratedClassifierCV (cv=5), sigmoid vs isotonic, ECE / log-loss / Brier before and after, reliability diagrams, and the same on noisy copies of the validation set. Save the final pipeline.
8. E11 ablations (bonus), weather-only pipeline (temperature, humidity, rainfall only, same calibration steps).
9. arena_metrics.json (E2, E3, E4, E11), model_info.json, reference_distribution.json (training histograms for PSI).
10. Reality-check experiment (handbook 8.4): 100 to 200 Indian locations, long-term weather from Open-Meteo or NASA POWER, weather-only model and gate, report share refused and why. Never present agreement as accuracy.
11. Report: merge all chapters; write Model Card text with Role 4.

## You hand over
Splits (day 3), baseline pipeline (day 4), final pipeline, class_stats.json, arena_metrics.json, reference_distribution.json, weather-only pipeline.

## You need
The noise model (agree with Role 2 in week 1); Role 2's detector class for the weather-only gate.

## Role-specific rules for the AI
- Leakage is the main risk in this role. Every transform inside a Pipeline; the test set untouched until the very end.
- At about 99% accuracy calibration looks fine by accident; also evaluate on noisy copies of validation data (handbook 5.4, the easy-data trap).
- Never write a number into a report or JSON unless the code that produced it ran.
- The person leading this role is new to APIs and deployment: when a task touches them, explain briefly in plain words.

## First prompt to paste
> I'm Role 1 (ML Lead). Read AGENTS.md, docs/roles/ROLE1_ML_Lead.md, and handbook sections 2 (Role 1), 5 and 7.6. Don't write code yet. Summarise my tasks, the files I own, what I need from others, and what you'd build first. Then wait for me to pick one task.
