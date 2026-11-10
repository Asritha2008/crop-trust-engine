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

## API

Full contract specification: [`docs/api-contract.md`](docs/api-contract.md).

### Live URLs
- **Backend:** https://crop-trust-engine.onrender.com (Swagger UI at `/docs`)
- **Frontend:** https://crop-trust-engine-frontend.vercel.app

### Endpoints
- `GET /health`
- `GET /model-info`
- `POST /predict` — optional `risk` query parameter: `conservative`, `balanced`, `bold`
- `POST /predict/batch` — CSV upload, max 50 rows, max 1 MB
- `POST /explain`
- `GET /arena`

> [!NOTE]
> **OPEN (`model_version`):** `/predict` currently returns a placeholder string and `/model-info` returns the real one; Roles 1 and 2 are making them match.

### Required Inputs (Case-Sensitive)
All 7 inputs are required (`float`) with the following physical bounds from `docs/api-contract.md`:

| Field | Bounds |
|---|---|
| `N` | `0.0 <= N <= 200.0` |
| `P` | `0.0 <= P <= 200.0` |
| `K` | `0.0 <= K <= 300.0` |
| `temperature` | `-10.0 <= temperature <= 60.0` |
| `humidity` | `0.0 <= humidity <= 100.0` |
| `ph` | `0.0 <= ph <= 14.0` |
| `rainfall` | `0.0 <= rainfall <= 1000.0` |

### Error Responses
- **422:** `{"errors": [{"field": "...", "message": "..."}]}`
- **400, 404, 405, 413, 429, 500:** `{"error": "..."}`
- **429:** Includes the `Retry-After` header.

### Rate Limits (Per Minute, Set by Env Vars)
- `POST /predict`: 60 (`RATE_LIMIT_PREDICT`)
- `POST /explain`: 30 (`RATE_LIMIT_EXPLAIN`)
- `POST /predict/batch`: 5 (`RATE_LIMIT_BATCH`)
- `GET /model-info`: 60 (`RATE_LIMIT_MODEL_INFO`)
- `GET /arena`: 60 (`RATE_LIMIT_ARENA`)

### Render Deployment Notes
- **Render free tier:** First request after sleep can take about 50 seconds.
- **Batch timing:** Batch of 50 rows measured at about 13.5 to 15.8 seconds on Render; this may improve.

### Run Locally
```bash
# Activate the virtual environment (Windows: venv\Scripts\activate | Linux/macOS: source venv/bin/activate)
uvicorn backend.main:app
pytest
```

