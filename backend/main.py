import copy
import csv
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware

from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, ValidationError
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from starlette.exceptions import HTTPException as StarletteHTTPException

logger = logging.getLogger(__name__)

# Ensure backend directory is in sys.path for local and Docker parity
BACKEND_DIR = Path(__file__).resolve().parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from trust_engine import TrustEngine
try:
    from backend.drift import FEATURES, MIN_ROWS, feature_psi, read_request_log
except ImportError:
    from drift import FEATURES, MIN_ROWS, feature_psi, read_request_log
try:
    from backend.request_log import log_request
except ImportError:
    from request_log import log_request

# Configurable rate limits with defaults
RATE_LIMIT_PREDICT = os.getenv("RATE_LIMIT_PREDICT", "60/minute")
RATE_LIMIT_EXPLAIN = os.getenv("RATE_LIMIT_EXPLAIN", "30/minute")
RATE_LIMIT_BATCH = os.getenv("RATE_LIMIT_BATCH", "5/minute")
RATE_LIMIT_MODEL_INFO = os.getenv("RATE_LIMIT_MODEL_INFO", "60/minute")
RATE_LIMIT_ARENA = os.getenv("RATE_LIMIT_ARENA", "60/minute")
RATE_LIMIT_MONITOR = os.getenv("RATE_LIMIT_MONITOR", "60/minute")


def _rate_limit_predict() -> str:
    return os.getenv("RATE_LIMIT_PREDICT", "60/minute")


def _rate_limit_explain() -> str:
    return os.getenv("RATE_LIMIT_EXPLAIN", "30/minute")


def _rate_limit_batch() -> str:
    return os.getenv("RATE_LIMIT_BATCH", "5/minute")


def _rate_limit_model_info() -> str:
    return os.getenv("RATE_LIMIT_MODEL_INFO", "60/minute")


def _rate_limit_arena() -> str:
    return os.getenv("RATE_LIMIT_ARENA", "60/minute")


def _rate_limit_monitor() -> str:
    return os.getenv("RATE_LIMIT_MONITOR", "60/minute")


def get_client_ip(request: Request) -> str:
    """Extract client IP from X-Forwarded-For or client.host for in-memory rate limiting.

    Note: X-Forwarded-For can be faked by a client and the number of proxies on Render is unverified.
    """
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "127.0.0.1"


limiter = Limiter(key_func=get_client_ip, headers_enabled=False)

app = FastAPI(
    title="Crop Trust Engine",
    description="Backend API for Crop Trust Engine",
    version="1.0.0",
)
app.state.limiter = limiter


# CORS setup: localhost on ports 5173 and 3000, live Vercel URL, plus optional FRONTEND_URL
class _AllowedOrigins(list):
    """List of allowed origins that also checks FRONTEND_URL dynamically if set."""

    def __contains__(self, item: object) -> bool:
        if super().__contains__(item):
            return True
        f_url = os.getenv("FRONTEND_URL")
        if f_url:
            cleaned = f_url.strip().rstrip("/")
            if cleaned and item == cleaned:
                return True
        return False


origins = _AllowedOrigins([
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "https://crop-trust-engine-ui.vercel.app",
])

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type"],
    expose_headers=["Retry-After"],
)

# Load engine, arena metrics, and model info once at startup
MODEL_DIR = BACKEND_DIR / "models"
engine = TrustEngine.load(MODEL_DIR)

ARENA_METRICS_PATH = MODEL_DIR / "arena_metrics.json"
ARENA_ALLOWED_KEYS = (
    "benchmark_metadata",
    "models_summary",
    "fold_level_metrics",
    "significance_tests",
    "champion_model",
    "class_wise_performance",
    "calibration_experiment_e4",
)

MODEL_INFO_PATH = MODEL_DIR / "model_info.json"
MODEL_INFO_ALLOWED_KEYS = (
    "model_version",
    "trained_on",
    "n_train_rows",
    "feature_order",
    "sklearn_version",
    "final_model",
    "calibration_method",
    "selection_rule",
    "selection_amendment",
    "date",
    "data_fingerprint",
)

REFERENCE_DIST_PATH = MODEL_DIR / "reference_distribution.json"
PSI_THRESHOLDS = {"stable": 0.10, "warning": 0.25}


def _load_json_allowlist(
    path: Path,
    allowed_keys: Tuple[str, ...],
    label: str,
    require_all: bool = False,
) -> Optional[Dict[str, Any]]:
    """Load a JSON object from disk, returning only allow-listed keys."""
    try:
        with path.open("r", encoding="utf-8") as f:
            raw = json.load(f)
    except FileNotFoundError:
        logger.warning("%s file not found at %s", label, path)
        return None
    except (OSError, ValueError) as exc:
        logger.error("Failed to load %s from %s: %s", label, path, exc)
        return None

    if not isinstance(raw, dict):
        logger.error("Invalid %s format in %s: expected JSON object", label, path)
        return None

    if require_all and any(key not in raw for key in allowed_keys):
        logger.error("Missing required keys in %s at %s", label, path)
        return None

    return {key: raw[key] for key in allowed_keys if key in raw}


def _load_arena_metrics(path: Path) -> Optional[Dict[str, Any]]:
    """Load arena metrics from disk once, returning only allow-listed keys."""
    return _load_json_allowlist(path, ARENA_ALLOWED_KEYS, "Arena metrics")


def _load_model_info(path: Path) -> Optional[Dict[str, Any]]:
    """Load model info from disk once, requiring the 11 allow-listed keys."""
    return _load_json_allowlist(path, MODEL_INFO_ALLOWED_KEYS, "Model info", require_all=True)


def _load_reference_distribution(path: Path) -> Optional[Dict[str, List[float]]]:
    """Load reference distribution lists once at startup without raising."""
    loaded = _load_json_allowlist(
        path,
        tuple(FEATURES),
        "Reference distribution",
        require_all=True,
    )
    if loaded is None:
        return None

    validated: Dict[str, List[float]] = {}
    for feat in FEATURES:
        vals = loaded[feat]
        if not isinstance(vals, list) or not vals:
            logger.error(
                "Invalid reference distribution for '%s' in %s: expected non-empty list",
                feat,
                path,
            )
            return None
        validated[feat] = vals
    return validated


_ARENA_METRICS = _load_arena_metrics(ARENA_METRICS_PATH)
_MODEL_INFO = _load_model_info(MODEL_INFO_PATH)
_REFERENCE_DISTRIBUTION = _load_reference_distribution(REFERENCE_DIST_PATH)


def _psi_status(psi_value: Optional[float]) -> str:
    """Classify a numeric PSI value against provisional drift thresholds."""
    if psi_value is None:
        return "not_enough_data"
    if psi_value < PSI_THRESHOLDS["stable"]:
        return "stable"
    if psi_value <= PSI_THRESHOLDS["warning"]:
        return "warning"
    return "alert"


class CropInput(BaseModel):
    N: float = Field(..., ge=0.0, le=200.0, description="Nitrogen in soil")
    P: float = Field(..., ge=0.0, le=200.0, description="Phosphorus in soil")
    K: float = Field(..., ge=0.0, le=300.0, description="Potassium in soil")
    temperature: float = Field(..., ge=-10.0, le=60.0, description="Ambient temperature (°C)")
    humidity: float = Field(..., ge=0.0, le=100.0, description="Relative humidity (%)")
    ph: float = Field(..., ge=0.0, le=14.0, description="Soil pH scale (lowercase)")
    rainfall: float = Field(..., ge=0.0, le=1000.0, description="Total rainfall")


def format_assessment_response(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Deepcopy engine assessment and normalize optional keys to explicit nulls."""
    res = copy.deepcopy(raw)
    res.setdefault("model_version", getattr(engine, "MODEL_VERSION", "stub-2026-11-07"))
    res.setdefault("conformal_set", None)
    res.setdefault("measure_next", None)
    res.setdefault("stability", None)

    ood = res.get("ood", {})
    ood.setdefault("unusual_pair", None)
    ood.setdefault("nearest_known", None)
    res["ood"] = ood

    return res


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc: RequestValidationError):
    problems = []
    for err in exc.errors():
        field_name = str(err["loc"][-1]) if err["loc"] else "body"
        problems.append({
            "field": field_name,
            "message": err["msg"],
        })
    return JSONResponse(status_code=422, content={"errors": problems})


@app.exception_handler(RateLimitExceeded)
async def rate_limit_exceeded_handler(request: Request, exc: RateLimitExceeded):
    current_limit = getattr(request.state, "view_rate_limit", None)
    retry_after = 60
    if current_limit:
        try:
            window_stats = request.app.state.limiter.limiter.get_window_stats(
                current_limit[0], *current_limit[1]
            )
            reset_in = 1 + window_stats[0]
            retry_after = max(1, int(reset_in - time.time()))
        except Exception:
            logger.warning("Could not calculate dynamic retry_after from window stats")

    response = JSONResponse(
        status_code=429,
        content={"error": f"Rate limit exceeded. Try again in {retry_after} seconds."},
    )
    response.headers["Retry-After"] = str(retry_after)
    return response


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request, exc: StarletteHTTPException):
    return JSONResponse(status_code=exc.status_code, content={"error": exc.detail})


@app.exception_handler(Exception)
async def general_exception_handler(request, exc: Exception):
    return JSONResponse(status_code=500, content={"error": "Internal server error"})


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/model-info")
@limiter.limit(_rate_limit_model_info)
def model_info(request: Request):
    if _MODEL_INFO is None:
        raise HTTPException(status_code=500, detail="Model info unavailable")
    return _MODEL_INFO


@app.get("/arena")
@limiter.limit(_rate_limit_arena)
def arena(request: Request):
    if _ARENA_METRICS is None:
        raise HTTPException(status_code=500, detail="Arena metrics unavailable")
    return _ARENA_METRICS


@app.get("/monitor")
@limiter.limit(_rate_limit_monitor)
def monitor(request: Request):
    if _REFERENCE_DISTRIBUTION is None:
        raise HTTPException(status_code=500, detail="Reference distribution unavailable")

    log_path = Path(os.getenv("REQUEST_LOG_PATH", "logs/requests.jsonl"))
    log_data = read_request_log(log_path, include_rotated=True)
    psi_result = feature_psi(
        log_data["rows"],
        _REFERENCE_DISTRIBUTION,
        min_rows=MIN_ROWS,
    )
    psi_by_feature = psi_result["psi"]

    features = [
        {
            "name": feat,
            "psi": psi_by_feature.get(feat),
            "status": _psi_status(psi_by_feature.get(feat)),
        }
        for feat in FEATURES
    ]

    non_null_psis = [
        val for val in psi_by_feature.values() if val is not None
    ]
    if psi_result["status"] != "ok" or not non_null_psis:
        overall_psi = None
        status = "not_enough_data"
    else:
        overall_psi = max(non_null_psis)
        status = _psi_status(overall_psi)

    return {
        "status": status,
        "model_version": getattr(engine, "MODEL_VERSION", "stub-2026-11-07"),
        "overall_psi": overall_psi,
        "features": features,
        "thresholds": dict(PSI_THRESHOLDS),
    }


@app.post("/predict")
@limiter.limit(_rate_limit_predict)
def predict(
    request: Request,
    data: CropInput,
    risk: Literal["conservative", "balanced", "bold"] = "balanced",
):
    input_dict = data.model_dump()
    t0 = time.perf_counter()
    raw_result = engine.assess(input_dict, risk=risk)
    latency_ms = (time.perf_counter() - t0) * 1000.0
    formatted = format_assessment_response(raw_result)
    log_request(input_dict, formatted, latency_ms)
    return formatted


@app.post("/explain")
@limiter.limit(_rate_limit_explain)
def explain(
    request: Request,
    data: CropInput,
    risk: Literal["conservative", "balanced", "bold"] = "balanced",
):
    raw_result = engine.assess(data.model_dump(), risk=risk)
    return format_assessment_response(raw_result)


REQUIRED_COLUMNS = [
    "N",
    "P",
    "K",
    "temperature",
    "humidity",
    "ph",
    "rainfall",
]
MAX_BATCH_FILE_SIZE = 1_000_000  # 1 MB
MAX_BATCH_ROWS = 50


@app.post("/predict/batch")
@limiter.limit(_rate_limit_batch)
def predict_batch(
    request: Request,
    file: UploadFile = File(...),
    risk: Literal["conservative", "balanced", "bold"] = "balanced",
):
    # Read in bounded chunks to prevent loading oversized payloads into memory
    content_bytes = bytearray()
    while chunk := file.file.read(65536):
        content_bytes.extend(chunk)
        if len(content_bytes) > MAX_BATCH_FILE_SIZE:
            raise HTTPException(status_code=413, detail="File too large (limit 1 MB)")

    try:
        text = content_bytes.decode("utf-8-sig")
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="Invalid CSV format")

    if "\0" in text:
        raise HTTPException(status_code=400, detail="Invalid CSV format")

    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        raise HTTPException(status_code=400, detail="File is empty")

    try:
        reader = csv.reader(lines)
        raw_header = next(reader, None)
    except csv.Error:
        raise HTTPException(status_code=400, detail="Invalid CSV format")

    if not raw_header:
        raise HTTPException(status_code=400, detail="File is empty")

    headers = [col.strip() for col in raw_header]
    for col in REQUIRED_COLUMNS:
        if col not in headers:
            raise HTTPException(status_code=400, detail=f"Missing column: {col}")

    col_indices = {col: headers.index(col) for col in REQUIRED_COLUMNS}

    data_rows = []
    for row in reader:
        if not row or all(cell.strip() == "" for cell in row):
            continue
        data_rows.append(row)

    if not data_rows:
        raise HTTPException(status_code=400, detail="File is empty")

    if len(data_rows) > MAX_BATCH_ROWS:
        raise HTTPException(status_code=400, detail="At most 50 rows per upload")

    results = []
    for idx, row in enumerate(data_rows):
        row_dict = {}
        for col in REQUIRED_COLUMNS:
            col_idx = col_indices[col]
            row_dict[col] = row[col_idx].strip() if col_idx < len(row) else ""

        try:
            validated = CropInput(**row_dict)
        except ValidationError as e:
            problems = [f"{err['loc'][-1]}: {err['msg']}" for err in e.errors()]
            results.append({"row_index": idx, "error": "; ".join(problems)})
            continue

        try:
            input_dict = validated.model_dump()
            t0 = time.perf_counter()
            raw = engine.assess(input_dict, risk=risk)
            row_latency_ms = (time.perf_counter() - t0) * 1000.0
            pred = format_assessment_response(raw)
            pred["row_index"] = idx
            results.append(pred)
            log_request(input_dict, pred, row_latency_ms)
        except Exception as e:
            logger.exception("Error assessing row %d: %s", idx, e)
            results.append({"row_index": idx, "error": "Could not assess this row"})

    return {
        "model_version": getattr(engine, "MODEL_VERSION", "stub-2026-11-07"),
        "count": len(results),
        "results": results,
    }

