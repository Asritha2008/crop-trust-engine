# Progress log
Add two lines after each finished task: what was done, which files changed.

| Date | Role | Task | Files changed | Notes / open issues |
| --- | --- | --- | --- | --- |
| 2026-11-07 | Role 1 | Step 1 Data Audit: executed notebooks/01_eda.ipynb | notebooks/01_eda.ipynb, docs/PROGRESS.md | Verified 2,200 rows, 22 crops x 100 rows, 0 nulls, 0 duplicates, all features within physical bounds |
| 2026-11-07 | Role 1 | Step 2 Stratified splits saved (70/15/15) | data/splits/*, docs/PROGRESS.md | train_idx.npy (1540), val_idx.npy (330), test_idx.npy (330); seed=42; Role 2 unblocked |

## Component tracker
| Component | Owner | Status | Notes |
| --- | --- | --- | --- |
| API contract frozen + mock JSON | Role 3 | Not started | |
| EDA, splits, demo pairs found | Role 1 | In progress | Step 1 (Audit) & Step 2 (Splits) complete |
| 9-model comparison, class-wise analysis | Role 1 | Not started | |
| Calibration | Role 1 | Not started | |
| OOD detectors + manufactured-unknowns evaluation | Role 2 | Not started | |
| Stability + risk-coverage | Role 2 | Not started | |
| trust_engine.py + artefacts | Role 2 | Not started | |
| FastAPI + tests + Postman collection | Role 3 | Not started | |
| Frontend pages | Role 4 | Not started | |
| Deployment (backend, frontend) + live Postman run | Roles 3, 4 | Not started | |
| Chosen extra features | per handbook 8.5 | Not started | |
| Report, slides, demo rehearsal | Role 1 (report), Role 4 (slides, demo) | Not started | |
