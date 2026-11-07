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
- **HTTP 422 Unprocessable Entity:** Returned strictly when an input violates physical reality or schema definition (e.g. `ph = 40`, negative rainfall, non-numeric values, or missing required fields). The model and OOD gate are never invoked.
- **HTTP 200 OK (Refusal):** Returned when input values are physically valid but represent an Out-of-Distribution (OOD) point or an impossible agronomic combination never observed in training data (e.g. humidity 92% with 25 mm rainfall). A refusal is a correct, functioning output from the Trust Engine, **not** an HTTP error.

### 1.3 Batch Processing Limits
- **HTTP 413 Payload Too Large:** Returned if the uploaded CSV file exceeds **1 MB** (`1,000,000` bytes).
- **HTTP 400 Bad Request:** Returned if any required column is missing from the CSV header or if the file contains more than **500 rows**.

### 1.4 Model Versioning
Every prediction and explanation reply must include a top-level `model_version` string (e.g. `"stub-2026-11-07"`).

> [!NOTE]
> `model_version` `"stub-2026-11-07"` is a placeholder until Role 1 supplies the real value.

---

## 2. Input Features & Validation Bounds

All inputs represent physical environmental and soil characteristics. Validation limits correspond to physical boundaries (not dataset min/max values), allowing the OOD detector to evaluate data distribution. Field names must be strictly lowercase. All 7 inputs are required in every request.

| Field Name | Type | Physical Bounds | Units | Notes |
|---|---|---|---|---|
| `N` | `float` | `0.0 <= N <= 200.0` | not specified in handbook | Nitrogen in soil |
| `P` | `float` | `0.0 <= P <= 200.0` | not specified in handbook | Phosphorus in soil |
| `K` | `float` | `0.0 <= K <= 300.0` | not specified in handbook | Potassium in soil |
| `temperature` | `float` | `-10.0 <= temperature <= 60.0` | °C | Ambient temperature |
| `humidity` | `float` | `0.0 <= humidity <= 100.0` | % | Relative humidity |
| `ph` | `float` | `0.0 <= ph <= 14.0` | pH scale | Soil pH (strictly lowercase `ph`) |
| `rainfall` | `float` | `0.0 <= rainfall <= 1000.0` | mm | Total rainfall |

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
Provides metadata and operational parameters to feed the frontend Model Card page.

> [!NOTE]
> Draft shape, not specified in the handbook. OPEN: Role 1 to confirm.

- **Request:** None
- **Response 200 OK:**
  ```json
  {
    "model_name": "<from model_info.json>",
    "model_version": "stub-2026-11-07",
    "trained_date": "<from model_info.json>",
    "dataset": "<from model_info.json>",
    "features": ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"],
    "calibrator": "<from model_info.json>",
    "ood_method": "<from model_info.json>"
  }
  ```
  *(Note: `"stub-2026-11-07"` is a placeholder until Role 1 supplies the real value).*

---

### 3.3 `POST /predict`
The primary prediction endpoint running the 5-stage Trust Engine.

#### Query Parameters
- `risk`: (optional string, default `"balanced"`)
  - Allowed values: `"conservative"`, `"balanced"`, `"bold"`

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
   - `percentile`: float (Mahalanobis distance percentile relative to training data).
   - `unusual_pair`: `[string, string] | null` (the pair driving the OOD flag; `null` if not flagged).
   - `nearest_known`: `[string, string] | null` (nearest crops; `null` if not flagged).
4. `stability`: float between `0.0` and `1.0` (agreement rate under noise perturbations); `null` on `"refuse"`.
5. `trust_score`: integer between `0` and `100`; `0` on `"refuse"`.
6. `explanation`: string providing a human-readable, plain-English justification.
7. `model_version`: string identifying the active model artifact (placeholder `"stub-2026-11-07"` until Role 1 supplies the real value).
8. Optional keys (`conformal_set`, `measure_next`, `ood.unusual_pair`, `ood.nearest_known`): Always present in response schema; set to `null` when not applicable (the API layer fills them).
   - *Status:* *OPEN: Role 4 to confirm*.

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
  "explanation": "[STUB / ILLUSTRATIVE] Humidity 92.0% and rainfall 25.0 mm are individually plausible, but this combination does not appear in the training data. No recommendation is made.",
  "model_version": "stub-2026-11-07"
}
```

#### Validation Error (HTTP 422 Unprocessable Entity)
Returned when input values fall outside physical boundaries or any required field is missing.

> [!NOTE]
> Draft shape, not specified in the handbook. OPEN: Role 4 to confirm.

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

---

### 3.4 `POST /predict/batch`
Evaluates multiple field rows from an uploaded CSV file.

> [!NOTE]
> Draft shape, not specified in the handbook. OPEN: Role 4 to confirm.

- **Request:** `multipart/form-data`
  - `file`: CSV file (required, max size `1 MB`).
  - `risk`: Query parameter (`"conservative"`, `"balanced"`, `"bold"`, default: `"balanced"`).
- **Headers / Columns Required:** `N,P,K,temperature,humidity,ph,rainfall`
- **Constraints:**
  - File size $> 1\text{ MB} \implies$ **HTTP 413 Payload Too Large** (`"File too large (limit 1 MB)"`).
  - Missing any of the 7 columns $\implies$ **HTTP 400 Bad Request** (`"Missing column: <name>"`).
  - Row count $> 500 \implies$ **HTTP 400 Bad Request** (`"At most 500 rows per upload"`).
- **Response 200 OK:**
  ```json
  {
    "model_version": "stub-2026-11-07",
    "count": 3,
    "results": [
      {
        "row_index": 0,
        "verdict": "recommend",
        "top_crops": [{ "crop": "rice", "probability": 0.91 }],
        "trust_score": 88
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
  The response shape is identical to `POST /predict`, based on the caution example:
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
  *(Note: Counterfactuals, decision margins, and feature contributions are Future, not guaranteed by the engine).*

---

### 3.6 `GET /arena`
Serves benchmark comparison data for all evaluated classifiers to populate the Model Arena page.

> [!NOTE]
> Draft shape, not specified in the handbook. OPEN: Role 1 to confirm.

- **Request:** None
- **Response 200 OK:** Serialized contents of `arena_metrics.json` containing E2 (9-model metrics), E3 (confusion matrices), E4 (ECE & calibration curves), and E11 (ablations).

---

### 3.7 `GET /monitor`
Telemetry endpoint for production drift monitoring.

> [!NOTE]
> Draft shape, not specified in the handbook. OPEN: Role 2 and Role 3 to confirm.

- **Request:** None
- **Response 200 OK:**
  ```json
  {
    "total_requests": 1420,
    "refusal_rate": 0.084,
    "caution_rate": 0.141,
    "psi_by_feature": {
      "N": 0.042,
      "P": 0.038,
      "K": 0.051,
      "temperature": 0.021,
      "humidity": 0.035,
      "ph": 0.019,
      "rainfall": 0.082
    },
    "max_psi": 0.082,
    "drift_status": "stable"
  }
  ```
  *(PSI $< 0.1$: stable; $0.1 \le \text{PSI} \le 0.25$: moderate shift; $> 0.25$: significant shift).*

---

### 3.8 `POST /feedback` (Additive Extra)
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

### 3.9 `GET /weather-lookup` (Planned Additive Helper)
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
- **Status:** *OPEN: Role 1 to confirm rainfall aggregation period and units*.

