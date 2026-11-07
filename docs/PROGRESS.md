# Progress log
Add two lines after each finished task: what was done, which files changed.

| Date | Role | Task | Files changed | Notes / open issues |
| --- | --- | --- | --- | --- |
| 2026-11-07 | Role 4 | Task 1: Scaffold React + Vite app with centralized API config | frontend/*, docs/decisions.md | Mobile-first layout, build-time VITE_API_URL validation, waking server state |
| 2026-11-07 | Role 4 | Task 2: Frozen contract mock harness & API adapter | frontend/mock/predict_examples.json, frontend/src/api/client.js, frontend/src/App.jsx | Recommend/Caution/Refuse fixtures verified; live/mock toggle; 422 physical validation |
| 2026-11-07 | Role 4 | Task 3: Complete Recommend page UI and trust components | frontend/src/components/*, frontend/src/utils/*, frontend/src/App.* | 7 dual slider inputs with typical ranges, verdict banner, trust meter, calibrated top crops, expandable why, download report |
| 2026-11-07 | Role 1 | Step 1 Data Audit: executed notebooks/01_eda.ipynb | notebooks/01_eda.ipynb, docs/PROGRESS.md | Verified 2,200 rows, 22 crops x 100 rows, 0 nulls, 0 duplicates, all features within physical bounds |
| 2026-11-07 | Role 1 | Step 2 Stratified splits saved (70/15/15) | data/splits/*, docs/PROGRESS.md | train_idx.npy (1540), val_idx.npy (330), test_idx.npy (330); seed=42; Role 2 unblocked |
| 2026-11-07 | Role 1 | Step 3 Baseline Pipeline (scaler + RF) | notebooks/03_baseline.py, backend/models/crop_pipeline.joblib, backend/models/classes.json | Trained on train_idx, val accuracy 99.09%, test set untouched. Unblocks Roles 2 & 3. |
| 2026-11-07 | Role 2 | Task 1: Trust Engine interface & stub implementation | backend/trust_engine.py, backend/tests/test_engine.py | Interface (load, assess) ready with deterministic mock replies; 14 unit tests passing. |
| 2026-11-07 | Role 2 | Task 2: Physical noise model proposed & documented | docs/decisions.md, backend/models/noise_model.json | Proposed noise distributions, rationale, floor & clipping rules documented; coordination note prepared for Role 1. |
| 2026-11-07 | Role 2 | Task 2.1: Finalize & freeze approved noise model | backend/models/noise_model.json, backend/noise.py, backend/tests/test_noise.py, docs/decisions.md | Role 1 (Abhishek) approved 1-sigma model and floors. Reusable noise generator with clipping stats implemented; 27 unit tests passing. Split files pending. |

## Component tracker
| Component | Owner | Status | Notes |
| --- | --- | --- | --- |
| API contract frozen + mock JSON | Role 3 | In progress | Draft contract + mock JSON merged in PR #2 |
| EDA, splits, demo pairs found | Role 1 | In progress | Step 1 (Audit) & Step 2 (Splits) complete |
| 9-model comparison, class-wise analysis | Role 1 | Not started | |
| Calibration | Role 1 | Not started | |
| OOD detectors + manufactured-unknowns evaluation | Role 2 | Not started | |
| Stability + risk-coverage | Role 2 | Not started | |
| trust_engine.py + artefacts | Role 2 | In progress (stub ready) | Stub engine and interface published; real detectors pending data splits |
| FastAPI + tests + Postman collection | Role 3 | Not started | |
| Frontend pages | Role 4 | In progress | Tasks 1-3 complete (Scaffold, Mock harness, Recommend page UI) |
| Deployment (backend, frontend) + live Postman run | Roles 3, 4 | Not started | |
| Chosen extra features | per handbook 8.5 | Not started | |
| Report, slides, demo rehearsal | Role 1 (report), Role 4 (slides, demo) | Not started | |