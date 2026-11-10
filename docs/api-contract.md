# Crop Trust Engine: API Contract Specification

- **Version:** 1.0.0-draft
- **Date:** 2026-11-07
- **Authors:** Author: Role 3. Reviewers: Role 2, Role 4 (pending)
- **Status:** Draft, pending review by Role 2 and Role 4

> [!NOTE]
> Examples are illustrative stub outputs.

---

## 1. Architectural & Protocol Rules

### 1.1 Contract-First Principle
The backend API and frontend client exchange JSON payloads over HTTP. All endpoints adhere strictly to this specification to enable decoupled, parallel development across lanes.

### 1.2 Error vs. Refusal Semantics (The 422 vs. 200 Rule)
- **HTTP 422 Unprocessable Entity:** Returned strictly when an input violates physical reality or schema definition (e.g. `ph = 40`, negative rainfall, non-numeric values, missing required fields, or an invalid `risk` value). The model and OOD gate are never invoked.
- **HTTP 200 OK (Refusal):** Returned when input values are physically valid but represent an Out-of-Distribution (OOD) point or an impossible agronomic combination never observed in training data (e.g. humidity 92% with 25 rainfall). A refusal is a correct, functioning output from the Trust Engine, **not** an HTTP error.

### 1.3 Batch Processing Limits
- **HTTP 413 Payload Too Large:** Returned if the uploaded CSV file exceeds **1 MB** (`1,000,000` bytes).
- **HTTP 400 Bad Request:** Returned if any required column is missing from the CSV header, if the file contains more than **50 rows**, if the file is empty or header-only (`"File is empty"`), or if the file is undecodable or malformed (`"Invalid CSV format"`).

### 1.4 Model Versioning
Every prediction and explanation reply must include a top-level `model_version` string (e.g. `"stub-2026-11-07"`).

> [!NOTE]
> `model_version` `"stub-2026-11-07"` is a placeholder until Role 1 supplies the real value.

### 1.5 Optional Fields Representation
When an optional field does not apply, the API returns an explicit `null` (never omits the key). This applies to `conformal_set`, `measure_next`, `ood.unusual_pair`, `ood.nearest_known`, and `stability` on refuse.

### 1.6 Error Response Shapes by Status Code

Error responses never return raw Python tracebacks:
- **HTTP 400 Bad Request, HTTP 404 Not Found, HTTP 405 Method Not Allowed, HTTP 413 Payload Too Large, HTTP 429 Too Many Requests, HTTP 500 Internal Server Error:**
  Return a single top-level `error` key with a short string message. On HTTP 429, a `Retry-After` header indicates the integer seconds until the rate limit window resets:
Error responses never return raw Python tracebacks. Error bodies are keyed strictly by status code:
- **HTTP 400 Bad Request, HTTP 404 Not Found, HTTP 405 Method Not Allowed, HTTP 413 Payload Too Large, HTTP 500 Internal Server Error:**
  Use a single top-level `error` key with a short string message:
main
  ```json
  {
    "error": "short message"
  }
  ```
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

### 1.7 Rate Limits

Rate limiting is enforced per client address using an in-memory counter. Limits are environment-configurable with the following defaults:
- `POST /predict`: 60 requests / minute (`RATE_LIMIT_PREDICT="60/minute"`)
- `POST /explain`: 30 requests / minute (`RATE_LIMIT_EXPLAIN="30/minute"`)
- `POST /predict/batch`: 5 requests / minute (`RATE_LIMIT_BATCH="5/minute"`)
- `GET /model-info`: 60 requests / minute (`RATE_LIMIT_MODEL_INFO="60/minute"`)
- `GET /arena`: 60 requests / minute (`RATE_LIMIT_ARENA="60/minute"`)
- `GET /health`: Exempt (unlimited)

When a rate limit is exceeded, the server returns HTTP 429 Too Many Requests with the `Retry-After: <seconds>` header and a JSON body `{"error": "Rate limit exceeded. Try again in N seconds."}`. The `Retry-After` header is exposed via CORS (`Access-Control-Expose-Headers: Retry-After`).

---

## 2. Input Features & Validation Bounds

All inputs represent physical environmental and soil characteristics. Validation limits correspond to physical boundaries (not dataset min/max values), allowing the OOD detector to evaluate data distribution. Field names are case-sensitive and must match exactly: N, P, K, temperature, humidity, ph, rainfall. All 7 inputs are required in every request.

| Field Name | Type | Physical Bounds | Units | Notes |
|---|---|---|---|---|
| `N` | `float` | `0.0 <= N <= 200.0` | dataset units (unitless ratios) | Nitrogen in soil |
| `P` | `float` | `0.0 <= P <= 200.0` | dataset units (unitless ratios) | Phosphorus in soil |
| `K` | `float` | `0.0 <= K <= 300.0` | dataset units (unitless ratios) | Potassium in soil |
| `temperature` | `float` | `-10.0 <= temperature <= 60.0` | °C | Ambient temperature |
| `humidity` | `float` | `0.0 <= humidity <= 100.0` | % | Relative humidity |
| `ph` | `float` | `0.0 <= ph <= 14.0` | pH scale | Soil pH (lowercase `ph`) |
| `rainfall` | `float` | `0.0 <= rainfall <= 1000.0` | mm (period not stated) | Total rainfall |

---

## 3. Endpoints Specification

### 3.1 `GET /health`
Liveness probe for monitoring, container orchestrators, and cloud hosting health checks.

- **Request:** None
- **Response 200 OK:**
  ```json
  {
    "status": "ok"
  }
  ```

---

### 3.2 `GET /model-info`
Provides metadata and model selection details loaded at startup from `backend/models/model_info.json` to feed the Model Card page.

> [!NOTE]
> Draft shape. OPEN: The engine returns `model_version` `"stub-2026-11-07"` while `model_info.json` says `"2026-11-09-a"`; Roles 1 and 2 agreed they should match and the engine will read it from the JSON.

- **Request:** None
- **Response 200 OK (Draft Shape):**
  Serves these 11 fields from `model_info.json` (`selection_amendment` is passed through verbatim without shortening or rephrasing):
  ```json
  {
    "model_version": "2026-11-09-a",
    "trained_on": "train split (1540 rows)",
    "n_train_rows": 1540,
    "feature_order": ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"],
    "sklearn_version": "1.7.2",
    "final_model": "Extra Trees",
    "calibration_method": "isotonic",
    "selection_rule": "<selection_rule text from model_info.json>",
    "selection_amendment": "<full amendment text from model_info.json>",
    "date": "2026-11-09",
    "data_fingerprint": {
      "crop_recommendation_csv_sha256": "<sha256>",
      "csv_sha256_note": "<note from model_info.json>",
      "crop_pipeline_calibrated_joblib_sha256": "<sha256>"
    }
  }
  ```
- **Error Response (HTTP 500 Internal Server Error):**
  If `model_info.json` is missing or invalid JSON, returns `{"error": "Model info unavailable"}` with no traceback or file path.

---

### 3.3 `POST /predict`
The primary prediction endpoint running the 5-stage Trust Engine.

#### Query Parameters
- `risk`: (optional string, default `"balanced"`)
  - Allowed values: `"conservative"`, `"balanced"`, `"bold"`
  - An invalid risk value (not `conservative`, `balanced`, or `bold`) returns **HTTP 422 Unprocessable Entity** with `{"errors": [{"field": "risk", "message": "..."}]}`.

#### Request Body (`application/json`)
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

#### Response Contract Rules
1. `verdict`: Enum `"recommend"` | `"caution"` | `"refuse"`.
2. `top_crops`: Array of crop predictions `[{"crop": string, "probability": float}]`. Up to 3 items, sorted by calibrated probability descending. Empty array `[]` on `"refuse"`.
3. `ood`: Object containing:
   - `flagged`: boolean.
   - `percentile`: float; Percentile rank of the input's OOD anomaly score relative to reference scores calculated from training data. Higher means more unusual (99.8 means the score is higher than about 99.8% of reference scores). It is NOT the probability that the input is OOD. The separate boolean flagged says whether the OOD threshold has been crossed.
   - `unusual_pair`: `[string, string] | null` (the pair driving the OOD flag; `null` if not flagged).
   - `nearest_known`: `[string, string] | null` (nearest crops; `null` if not flagged).
4. `stability`: float between `0.0` and `1.0` (agreement rate under noise perturbations); `null` on `"refuse"`.
5. `trust_score`: integer between `0` and `100`; `0` on `"refuse"`.
6. `explanation`: string providing a human-readable, plain-English justification.
7. `model_version`: string identifying the active model artifact (placeholder `"stub-2026-11-07"` until Role 1 supplies the real value).
8. Optional fields rule: When an optional field does not apply, the API returns an explicit `null` (never omits the key). Applies to `conformal_set`, `measure_next`, `ood.unusual_pair`, `ood.nearest_known`, and `stability` on refuse.


#### Example 200 Responses

##### A. Recommend (`verdict: "recommend"`)
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

##### B. Caution (`verdict: "caution"`)
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
*(Note: The `measure_next` example ("Potassium (K) ...") is illustrative only: the real engine now returns text based on real data, so the actual feature name can differ—for example, rainfall for rice vs jute).*

##### C. Refuse (`verdict: "refuse"`)
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

#### Validation Error (HTTP 422 Unprocessable Entity)
Returned when input values fall outside physical boundaries, a required field is missing, or an invalid `risk` value is supplied.

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
*(For an invalid `risk` parameter, returns `{"errors": [{"field": "risk", "message": "..."}]}`).*

---

### 3.4 `POST /predict/batch`
Evaluates multiple field rows from an uploaded CSV file.

> [!NOTE]
> Limit lowered from 500 to 50 because each assessment takes a noticeable fraction of a second and the free host has a slow CPU. It may be raised later after timing on the live server.

- **Request:** `multipart/form-data`
  - `file`: CSV file (required, max size `1 MB`).
  - `risk`: Query parameter (`"conservative"`, `"balanced"`, `"bold"`, default: `"balanced"`). An invalid risk value returns **HTTP 422 Unprocessable Entity** with `{"errors": [{"field": "risk", "message": "..."}]}`.
- **Headers / Columns Required:** `N,P,K,temperature,humidity,ph,rainfall`
- **File-Level Validation (Entire Upload Fails):**
  - File size $> 1\text{ MB} \implies$ **HTTP 413 Payload Too Large** (`{"error": "File too large (limit 1 MB)"}`).
  - Empty or header-only file $\implies$ **HTTP 400 Bad Request** (`{"error": "File is empty"}`).
  - Undecodable or malformed file $\implies$ **HTTP 400 Bad Request** (`{"error": "Invalid CSV format"}`).
  - Missing any of the 7 columns $\implies$ **HTTP 400 Bad Request** (`{"error": "Missing column: <name>"}`).
  - Row count $> 50 \implies$ **HTTP 400 Bad Request** (`{"error": "At most 50 rows per upload"}`).
- **Row-Level Processing & Error Handling:**
  - A bad value in one row does not fail the batch; the endpoint returns **HTTP 200 OK** overall.
  - That row's entry in `results` is `{"row_index": N, "error": "..."}`, while valid rows receive their normal prediction object.
  - `count` is the total number of rows in the file, and `results` contains exactly one entry per row in file order (`results.length == count`).
- **Response 200 OK:**
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
        "error": "Invalid ph: 40.0 exceeds physical limit of 14.0"
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
  *(Note: `"stub-2026-11-07"` is a placeholder until Role 1 supplies the real value).*

---

### 3.5 `POST /explain`
On-demand diagnostic breakdown for a given input.

- **Query Parameters:** `risk` (default `"balanced"`)
- **Request Body:** Same 7-field `CropInput` as `/predict`. All 7 fields required.
- **Response 200 OK:**
  The response is identical to `/predict`, with the same required and optional fields (including `conformal_set` and `measure_next`, `null` when not applicable), based on the caution example:
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
  *(Note: The `measure_next` example ("Potassium (K) ...") is illustrative only: the real engine now returns text based on real data, so the actual feature name can differ—for example, rainfall for rice vs jute. Counterfactuals, decision margins, and feature contributions are NOT promised yet).*

---

### 3.6 `GET /arena`
Serves benchmark comparison, class-wise analysis, and calibration data for the evaluated classifiers to populate the Model Arena page.

> [!NOTE]
> Draft shape. OPEN: Role 1 and Role 4 to confirm.
>
> - **E5 to E8** (OOD detection benchmarks, threshold curves, selective classification, and stability evaluation) are **not included yet**.
> - **E11** (`ablations_experiment_e11`, including the weather-only model) is **withheld until independently verified**.
> - `selection_rationale` is **withheld** because its current text cites E11 results.
> - `class_wise_performance_rf_baseline` is excluded as superseded by the final champion model's `class_wise_performance`.

- **Request:** None
- **Response 200 OK (Draft Shape):**
  Filtered subset of `arena_metrics.json` containing E2 (9-model comparison), E3 (final model class-wise analysis), and E4 (calibration benchmark):
  ```json
  {
    "benchmark_metadata": {},
    "models_summary": {},
    "fold_level_metrics": {},
    "significance_tests": [],
    "champion_model": "Extra Trees (Isotonic)",
    "class_wise_performance": {},
    "calibration_experiment_e4": {}
  }
  ```
  - `benchmark_metadata`: E2 cross-validation setup (date, protocol, random seed, sample/class/feature counts, noise profile).
  - `models_summary`: E2 aggregated mean/std metrics and latencies keyed by each of the 9 classifiers.
  - `fold_level_metrics`: E2 per-fold metric arrays (15 cross-validation folds) keyed by each of the 9 classifiers.
  - `significance_tests`: list of E2 pairwise statistical significance test entries.
  - `champion_model`: string name of the selected production model.
  - `class_wise_performance`: E3 per-crop metrics (22 crops), confused pairs, ambiguous rows, overall accuracy/macro-F1, and confusion matrix figure path for the champion model.
  - `calibration_experiment_e4`: E4 validation calibration benchmark results across 9 base/calibration variants, winner details, selection rule, and reliability diagram figure paths.
- **Error Response (HTTP 500 Internal Server Error):**
  If `arena_metrics.json` is missing or invalid JSON, returns `{"error": "Arena metrics unavailable"}` with no traceback, file paths, or raw JSON error details.

---

### 3.7 `GET /monitor`
Telemetry endpoint for production drift monitoring.

> [!NOTE]
> OPEN / draft: implementation and monitoring data source not finalized; these are initial heuristics (Role 2 and Role 3).

- **Request:** None
- **Response 200 OK:**
  ```json
  {
    "status": "stable",
    "model_version": "stub-2026-11-07",
    "overall_psi": 0.045,
    "thresholds": {
      "stable": 0.10,
      "warning": 0.25
    },
    "features": [
      { "name": "N", "psi": 0.042, "status": "stable" },
      { "name": "P", "psi": 0.038, "status": "stable" },
      { "name": "K", "psi": 0.051, "status": "stable" },
      { "name": "temperature", "psi": 0.021, "status": "stable" },
      { "name": "humidity", "psi": 0.035, "status": "stable" },
      { "name": "ph", "psi": 0.019, "status": "stable" },
      { "name": "rainfall", "psi": 0.082, "status": "stable" }
    ]
  }
  ```
  *(Initial PSI interpretation: below 0.10 stable, 0.10 to 0.25 warning, above 0.25 alert).*

---

### 3.8 `POST /feedback` (planned addition, pending role approvals)
Collects user-reported corrections to support future active learning / human-in-the-loop retraining.

> [!NOTE]
> Draft shape, not specified in the handbook. OPEN: Role 4 to confirm.

- **Request Body:**
  ```json
  {
    "input": {
      "N": 90.0,
      "P": 42.0,
      "K": 43.0,
      "temperature": 21.0,
      "humidity": 82.0,
      "ph": 6.5,
      "rainfall": 203.0
    },
    "predicted_crop": "rice",
    "corrected_crop": "jute",
    "comments": "High soil salinity in this parcel."
  }
  ```
- **Response 200 OK:** `{"status": "recorded"}`

---

### 3.9 `GET /weather-lookup` (planned addition, pending role approvals)
Location-based autofill pre-populating environmental variables from Open-Meteo or NASA POWER.

- **Route:** `GET /weather-lookup?lat={float}&lon={float}`
- **Response 200 OK (Draft):**
  ```json
  {
    "temperature": 24.5,
    "humidity": 68.0,
    "rainfall": 182.0,
    "source": "Open-Meteo"
  }
  ```
- **Status:** *OPEN: Role 1 to confirm rainfall aggregation period and units*. On hold. Do not feed weather-API rainfall into the model until Role 1 and Role 2 decide how to handle the rainfall period mismatch.

