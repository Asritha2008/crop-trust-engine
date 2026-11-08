import copy
import csv
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, Literal

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, ValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

logger = logging.getLogger(__name__)

# Ensure backend directory is in sys.path for local and Docker parity
BACKEND_DIR = Path(__file__).resolve().parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from trust_engine import TrustEngine

app = FastAPI(
    title="Crop Trust Engine",
    description="Backend API for Crop Trust Engine",
    version="1.0.0",
)

# CORS setup: localhost on ports 5173 and 3000, plus optional FRONTEND_URL
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
])

frontend_url = os.getenv("FRONTEND_URL")
if frontend_url:
    cleaned_frontend_url = frontend_url.strip().rstrip("/")
    if cleaned_frontend_url and cleaned_frontend_url not in origins:
        origins.append(cleaned_frontend_url)

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type"],
)

# Load engine once at startup
MODEL_DIR = BACKEND_DIR / "models"
engine = TrustEngine.load(MODEL_DIR)


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
def model_info():
    return {
        "model_version": getattr(engine, "MODEL_VERSION", "stub-2026-11-07"),
        "trained_on": "placeholder: Role 1 model_info.json not ready",
        "sklearn_version": "1.7.2",
    }


@app.post("/predict")
def predict(
    data: CropInput,
    risk: Literal["conservative", "balanced", "bold"] = "balanced",
):
    raw_result = engine.assess(data.model_dump(), risk=risk)
    return format_assessment_response(raw_result)


@app.post("/explain")
def explain(
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
def predict_batch(
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
            raw = engine.assess(validated.model_dump(), risk=risk)
            pred = format_assessment_response(raw)
            pred["row_index"] = idx
            results.append(pred)
        except Exception as e:
            logger.exception("Error assessing row %d: %s", idx, e)
            results.append({"row_index": idx, "error": "Could not assess this row"})

    return {
        "model_version": getattr(engine, "MODEL_VERSION", "stub-2026-11-07"),
        "count": len(results),
        "results": results,
    }

