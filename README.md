# Crop Trust Engine
> > **Status: setup in progress. The backend code, tests, Dockerfile, CI workflow and Postman collection described below are not built yet.**

> **A crop recommender that knows when not to trust itself, explains why, and proves that behaviour with measurements.**

---

## 🌾 Overview

Most crop recommenders blindly output a prediction for any input given to them—even physically impossible values or conditions the model has never encountered. 

**Crop Trust Engine** is built around a five-stage **Trust Engine** that evaluates input validity, tests for Out-of-Distribution (OOD) anomalies, computes calibrated probabilities, assesses prediction stability under realistic sensor noise, and issues one of three explicit verdicts:
1. **Recommend** (high confidence, in-distribution, stable under sensor noise)
2. **Caution** (in-distribution but ambiguous or sensitive to minor perturbations)
3. **Refuse** (unfamiliar input or combination never seen in training; HTTP 200 with an explicit explanation)

---

## 🏛️ Architecture: The 5-Stage Trust Engine

```
[ User Input (7 features) ]
           │
           ▼
[ Stage 1: Validate ] ─────────► Fails physical bounds? ──► HTTP 422 Unprocessable Entity
           │ (Pass)
           ▼
[ Stage 2: OOD Gate ] ─────────► Unfamiliar combination? ──► HTTP 200 Refusal (verdict: refuse)
           │ (Familiar)
           ▼
[ Stage 3: Calibrated Classifier ] ──► Honest top-3 probabilities (Platt/Isotonic calibration)
           │
           ▼
[ Stage 4: Stability Check ] ────────► Perturb inputs across sensor noise distributions
           │
           ▼
[ Stage 5: Verdict & Explanation ] ──► Recommend / Caution / Refuse + Plain-English Reason
```

---

## 👥 Team Roles & Ownership

The project is developed concurrently across four non-overlapping lanes:

| Role | Domain | Key Responsibilities & Owned Files |
|---|---|---|
| **Role 1: ML Lead** | Data & Models | `notebooks/01_eda`, `02_compare_models`, `03_calibration`, `crop_pipeline.joblib`, `class_stats.json`, `model_info.json`, `reference_distribution.json`, `arena_metrics.json` |
| **Role 2: Trust Engine** | Trust & OOD Layer | `notebooks/04_ood_eval`, `05_stability`, `backend/trust_engine.py`, `thresholds.json`, `noise_model.json`, `conformal.json`, `ood_stats.joblib`, `iforest.joblib` |
| **Role 3: Backend & DevOps** | API, Tests, Deployment | `backend/main.py`, `backend/tests/test_api.py`, `requirements.txt`, `Dockerfile`, `.github/workflows/`, `postman/`, `docs/api-contract.md`, `README.md` |
| **Role 4: Frontend & UX** | Web Interface & Demo | `frontend/`, `frontend/mock/predict_examples.json`, translations, interactive what-if sliders, Arena and Model Card pages |

---

## 📁 Repository Layout

```
crop-trust-engine/
├── .github/
│   └── workflows/
│       └── tests.yml             # GitHub Actions CI workflow
├── backend/
│   ├── main.py                   # FastAPI application & routes
│   ├── trust_engine.py           # Trust Engine logic (owned by Role 2)
│   ├── models/                   # Serialized pipelines & parameters
│   └── tests/
│       └── test_api.py           # Pytest API test suite
├── docs/
│   ├── api-contract.md           # Frozen API contract specification
│   ├── decisions.md              # Architectural decisions log
│   ├── PROGRESS.md               # Component progress tracker
│   ├── PROJECT_GUIDE.md          # Comprehensive handbook
│   └── roles/                    # Role charters & task lists
├── frontend/
│   ├── mock/
│   │   └── predict_examples.json # Mock responses for offline UI dev
│   └── src/                      # UI components & pages
├── notebooks/                    # Research & evaluation experiments (E1 - E11)
├── postman/                      # Postman collections & environment JSONs
├── .dockerignore                 # Docker ignore file
├── .gitignore                    # Git ignore file
├── Dockerfile                    # Containerization configuration
├── requirements.txt              # Pinned Python dependencies
└── README.md                     # Project documentation
```

---

## 🌿 Git & Branch Protection Workflow

To ensure stability across all four roles:
- **`main` is always runnable.** Direct commits to `main` are restricted.
- **Branch naming convention:** `role<N>/<topic>` (e.g. `role3/api-routes`, `role1/model-compare`).
- **Pull Request Review Pairs:**
  - Role 1 (ML) $\leftrightarrow$ Role 2 (Trust Engine) review each other.
  - Role 3 (Backend/DevOps) $\leftrightarrow$ Role 4 (Frontend) review each other.
- **Contract First:** Any change to frozen interfaces in `docs/api-contract.md` requires consensus from all four roles.

---

## 🚀 Quickstart & Local Development

### Prerequisites
- Python 3.11+
- Node.js 18+ (for frontend)
- Git

### 1. Backend Setup
```bash
# Create and activate virtual environment
python -m venv venv

# Windows:
venv\Scripts\activate
# Linux/macOS:
# source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run FastAPI with live reload
uvicorn backend.main:app --reload --port 8000
```
Interactive API documentation is immediately available at `http://localhost:8000/docs`.

### 2. Frontend Setup
```bash
cd frontend
npm install
npm run dev
```

---

## 🧪 Testing

### Automated Pytest Suite
```bash
pytest backend/tests -v
```

### Postman Test Suite
1. Import `postman/crop-trust-engine.postman_collection.json` into Postman.
2. Set `base_url` to `http://localhost:8000` (or the deployed public URL).
3. Execute the full test suite using Postman Collection Runner.

---

## 🐳 Docker Deployment

Build and run the container locally:
```bash
docker build -t crop-trust-engine .
docker run -p 8000:8000 -e PORT=8000 crop-trust-engine
```

---

## 📖 API Reference Summary

Refer to [`docs/api-contract.md`](docs/api-contract.md) for the complete frozen contract.

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Liveness check (HTTP 200 `{"status": "ok"}`) |
| `GET` | `/model-info` | Pipeline metadata, version, and training characteristics |
| `POST` | `/predict?risk=balanced` | Evaluates single input through 5-stage Trust Engine |
| `POST` | `/predict/batch` | Evaluates CSV file (max 1 MB, max 500 rows) |
| `POST` | `/explain` | Returns the same assessment fields as /predict. Counterfactuals, decision margins and feature contributions are future work. |
| `GET` | `/arena` | Model comparison metrics and calibration statistics |
| `GET` | `/monitor` | Real-time flag rate and Population Stability Index (PSI) |
| `POST` | `/feedback` | User feedback collection (planned addition, pending role approvals) |
| `GET` | `/weather-lookup` | Location-based weather autofill (planned addition, pending role approvals) |

