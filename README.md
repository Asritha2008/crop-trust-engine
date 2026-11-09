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

## 📖 API Reference & Contract Specification

Refer to [`docs/api-contract.md`](docs/api-contract.md) for the complete frozen contract specification.

> [!NOTE]
> The live deployment currently runs a placeholder engine, so all response examples below are **`[STUB / ILLUSTRATIVE]`** and real values will differ once the final model artifacts are loaded.

### 🌐 Base URL & Deployment Latency
- **Local:** `http://localhost:8000` (or `http://127.0.0.1:8000`)
- **Live:** `https://crop-trust-engine.onrender.com`
- **Interactive Documentation:** Available at `http://localhost:8000/docs` (Swagger UI) and `http://localhost:8000/redoc`.

> [!NOTE]
> Swagger at `/docs` shows FastAPI's built-in 422 example (`detail`), which differs from the real custom 422 error body (`errors`) documented here.
>
> On Render's free tier, the service spins down after periods of inactivity. The first request may take about 50 seconds to respond while the instance wakes.

### 🌿 Input Features & Physical Bounds
All inputs represent physical environmental and soil characteristics. Validation limits correspond to physical boundaries (not dataset min/max values), allowing the OOD detector to evaluate data distribution. Field names are case-sensitive and must match exactly: `N`, `P`, `K`, `temperature`, `humidity`, `ph`, `rainfall`. All 7 inputs are required in every prediction and explanation request.

| Field Name | Type | Physical Bounds | Units | Notes |
|---|---|---|---|---|
| `N` | `float` | `0.0 <= N <= 200.0` | dataset units (unitless ratios) | Nitrogen in soil |
| `P` | `float` | `0.0 <= P <= 200.0` | dataset units (unitless ratios) | Phosphorus in soil |
| `K` | `float` | `0.0 <= K <= 300.0` | dataset units (unitless ratios) | Potassium in soil |
| `temperature` | `float` | `-10.0 <= temperature <= 60.0` | °C | Ambient temperature |
| `humidity` | `float` | `0.0 <= humidity <= 100.0` | % | Relative humidity |
| `ph` | `float` | `0.0 <= ph <= 14.0` | pH scale | Soil pH (lowercase `ph`) |
| `rainfall` | `float` | `0.0 <= rainfall <= 1000.0` | mm (period not stated) | Total rainfall |

### ⚖️ Error vs. Refusal Semantics (The 422 vs. 200 Rule)
- **HTTP 422 Unprocessable Entity:** Returned strictly when an input violates physical reality or schema definition (e.g. `ph = 40`, negative rainfall, non-numeric values, missing required fields, or an invalid `risk` value). The model and OOD gate are never invoked.
- **HTTP 200 OK (Refusal):** Returned when input values are physically valid individually, but represent an unfamiliar combination never observed in training data (e.g. humidity 92% with 25 mm rainfall). A refusal is a functioning, correct output from the Trust Engine, **not** an HTTP error.

### 🛡️ Error Response Shapes & Rate Limiting
Error responses never return raw Python tracebacks. Error bodies are keyed strictly by status code:

- **HTTP 400 Bad Request, HTTP 404 Not Found, HTTP 405 Method Not Allowed, HTTP 413 Payload Too Large, HTTP 429 Too Many Requests, HTTP 500 Internal Server Error:**
  Use a single top-level `error` string:
  ```json
  {
    "error": "short message"
  }
  ```
  On **HTTP 429 Too Many Requests**, the response body is `{"error": "Rate limit exceeded. Try again in N seconds."}` and includes a `Retry-After: <seconds>` header indicating the integer seconds until the rate limit window resets (exposed via CORS `Access-Control-Expose-Headers: Retry-After`).

- **HTTP 422 Unprocessable Entity:**
  Uses an array of field-level errors under the `errors` key:
  ```json
  {
    "errors": [
      {
        "field": "ph",
        "message": "Input should be less than or equal to 14"
      }
    ]
  }
  ```
  *(For an invalid `risk` query parameter, returns `{"errors": [{"field": "risk", "message": "..."}]}`).*

#### Configurable Rate Limits (Defaults)
- `POST /predict`: 60 requests / minute
- `POST /explain`: 30 requests / minute
- `POST /predict/batch`: 5 requests / minute
- `GET /model-info`: 60 requests / minute
- `GET /health`: Exempt (unlimited)

### 🚀 Live Endpoints

#### 1. `GET /health`
Liveness probe for monitoring, container orchestrators, and cloud hosting health checks.

- **Request:** None
- **Response 200 OK:**
  ```json
  {
    "status": "ok"
  }
  ```

#### 2. `GET /model-info`
Provides pipeline metadata and operational parameters.

- **Request:** None
- **Response 200 OK:**
  ```json
  {
    "model_version": "stub-2026-11-07",
    "trained_on": "<placeholder date / data split>",
    "sklearn_version": "<placeholder scikit-learn version>"
  }
  ```
  *(Note: `"stub-2026-11-07"`, `trained_on`, and `sklearn_version` are placeholders until real metadata is supplied).*

#### 3. `POST /predict`
The primary prediction endpoint running the 5-stage Trust Engine.

- **Query Parameters:**
  - `risk`: (optional string, default `"balanced"`; allowed values: `"conservative"`, `"balanced"`, `"bold"`).
- **Request Body (`application/json`):**
  ```json
  {
    "N": 90.0,
    "P": 42.0,
    "K": 43.0,
    "temperature": 21.0,
    "humidity": 82.0,
    "ph": 6.5,
    "rainfall": 203.0
  }
  ```
- **Response Contract Rules:**
  - `verdict`: Enum `"recommend"` | `"caution"` | `"refuse"`.
  - `top_crops`: Array of `[{"crop": string, "probability": float}]` (up to 3 items, sorted descending; empty `[]` on refuse).
  - `ood`: Object containing:
    - `flagged`: boolean.
    - `percentile`: float; OOD anomaly score percentile relative to training data. Higher means more unusual (e.g. 99.8 means higher than about 99.8% of reference scores). It is NOT a probability. `flagged` is separate.
    - `unusual_pair`: `[string, string] | null` (driving pair if flagged; `null` if not flagged).
    - `nearest_known`: `[string, string] | null` (nearest crops; `null` if not flagged).
  - `stability`: Float between 0.0 and 1.0 (`null` on refuse).
  - `trust_score`: Integer between 0 and 100 (`0` on refuse).
  - `explanation`: Human-readable, plain-English justification string.
  - `model_version`: String identifying the active model artifact (placeholder `"stub-2026-11-07"`).
  - Optional fields: Explicit `null` returned when not applicable (`conformal_set`, `measure_next`, `ood.unusual_pair`, `ood.nearest_known`, and `stability` on refuse).

##### Example 200 Responses:

**A. Recommend (`verdict: "recommend"`) [STUB / ILLUSTRATIVE]**
```json
{
  "verdict": "recommend",
  "top_crops": [
    { "crop": "rice", "probability": 0.91 },
    { "crop": "jute", "probability": 0.05 },
    { "crop": "maize", "probability": 0.02 }
  ],
  "conformal_set": null,
  "ood": {
    "flagged": false,
    "percentile": 41.0,
    "unusual_pair": null,
    "nearest_known": null
  },
  "stability": 0.94,
  "trust_score": 88,
  "measure_next": null,
  "explanation": "[STUB / ILLUSTRATIVE] Inputs match known rice conditions and are stable under small changes.",
  "model_version": "stub-2026-11-07"
}
```

**B. Caution (`verdict: "caution"`) [STUB / ILLUSTRATIVE]**
```json
{
  "verdict": "caution",
  "top_crops": [
    { "crop": "rice", "probability": 0.52 },
    { "crop": "jute", "probability": 0.41 },
    { "crop": "maize", "probability": 0.07 }
  ],
  "conformal_set": ["rice", "jute"],
  "ood": {
    "flagged": false,
    "percentile": 63.0,
    "unusual_pair": null,
    "nearest_known": null
  },
  "stability": 0.71,
  "trust_score": 52,
  "measure_next": "Potassium (K): it separates rice from jute most strongly. Re-test it.",
  "explanation": "[STUB / ILLUSTRATIVE] Two crops are close (rice and jute) and a small change in the inputs can flip the answer. Consider verifying potassium levels before planting.",
  "model_version": "stub-2026-11-07"
}
```

**C. Refuse (`verdict: "refuse"`) [STUB / ILLUSTRATIVE]**
```json
{
  "verdict": "refuse",
  "top_crops": [],
  "conformal_set": null,
  "ood": {
    "flagged": true,
    "percentile": 99.8,
    "unusual_pair": ["humidity", "rainfall"],
    "nearest_known": ["rice", "jute"]
  },
  "stability": null,
  "trust_score": 0,
  "measure_next": null,
  "explanation": "[STUB / ILLUSTRATIVE] Humidity 92.0% and rainfall 25.0 are individually plausible, but this combination does not appear in the training data. No recommendation is made.",
  "model_version": "stub-2026-11-07"
}
```

#### 4. `POST /explain`
Calls the engine on the real input and returns the same response shape as `/predict`. Counterfactuals, decision margins, and feature contributions are future, not promised.

- **Query Parameters:** `risk` (default `"balanced"`)
- **Request Body:** Same 7-field `CropInput` as `/predict`. All 7 fields required.
- **Response 200 OK:** Identical schema to `/predict`.

#### 5. `POST /predict/batch`
Evaluates multiple field rows from an uploaded CSV file.

- **Request:** `multipart/form-data`
  - `file`: CSV file (required, max size `1 MB`).
  - `risk`: Query parameter (`"conservative"`, `"balanced"`, `"bold"`, default: `"balanced"`).
- **Headers / Columns Required:** `N,P,K,temperature,humidity,ph,rainfall`
- **File-Level Validation (Entire Upload Fails):**
  - File size $> 1\text{ MB} \implies$ **HTTP 413 Payload Too Large** (`{"error": "File too large (limit 1 MB)"}`)
  - Empty or header-only file $\implies$ **HTTP 400 Bad Request** (`{"error": "File is empty"}`)
  - Undecodable or malformed file $\implies$ **HTTP 400 Bad Request** (`{"error": "Invalid CSV format"}`)
  - Missing any of the 7 columns $\implies$ **HTTP 400 Bad Request** (`{"error": "Missing column: <name>"}`)
  - Row count $> 50 \implies$ **HTTP 400 Bad Request** (`{"error": "At most 50 rows per upload"}`)
- **Row-Level Processing & Error Handling:**
  - A bad value in one row does not fail the batch; the endpoint returns **HTTP 200 OK** overall.
  - That row's entry in `results` is `{"row_index": N, "error": "..."}`, while valid rows receive their normal prediction object.
  - `count` is the total number of rows in the file, and `results` contains exactly one entry per row in file order (`results.length == count`).

##### Example Batch 200 Response [STUB / ILLUSTRATIVE]:
```json
{
  "model_version": "stub-2026-11-07",
  "count": 3,
  "results": [
    {
      "row_index": 0,
      "verdict": "recommend",
      "top_crops": [
        { "crop": "rice", "probability": 0.91 },
        { "crop": "jute", "probability": 0.05 },
        { "crop": "maize", "probability": 0.02 }
      ],
      "conformal_set": null,
      "ood": {
        "flagged": false,
        "percentile": 41.0,
        "unusual_pair": null,
        "nearest_known": null
      },
      "stability": 0.94,
      "trust_score": 88,
      "measure_next": null,
      "explanation": "[STUB / ILLUSTRATIVE] Inputs match known rice conditions and are stable under small changes.",
      "model_version": "stub-2026-11-07"
    },
    {
      "row_index": 1,
      "error": "ph: Input should be less than or equal to 14"
    },
    {
      "row_index": 2,
      "verdict": "refuse",
      "top_crops": [],
      "conformal_set": null,
      "ood": {
        "flagged": true,
        "percentile": 99.8,
        "unusual_pair": ["humidity", "rainfall"],
        "nearest_known": ["rice", "jute"]
      },
      "stability": null,
      "trust_score": 0,
      "measure_next": null,
      "explanation": "[STUB / ILLUSTRATIVE] Humidity 92.0% and rainfall 25.0 are individually plausible, but this combination does not appear in the training data. No recommendation is made.",
      "model_version": "stub-2026-11-07"
    }
  ]
}
```

### 🔮 Planned Endpoints (Not Built Yet)
The following endpoints are planned additions with draft shapes that may change:
- **`GET /arena`** *(planned, draft shape, may change)*: Serves benchmark comparison data for evaluated classifiers to populate the Model Arena page.
- **`GET /monitor`** *(planned, draft shape, may change)*: Telemetry endpoint for production drift monitoring and Population Stability Index (PSI) tracking.
- **`POST /feedback`** *(planned, draft shape, may change)*: Collects user-reported corrections to support future active learning / human-in-the-loop retraining.

### ⏸️ Endpoints on Hold
- **`GET /weather-lookup`** *(on hold)*: Location-based environmental autofill pre-populating environmental variables. On hold pending rainfall aggregation period and units agreement.

