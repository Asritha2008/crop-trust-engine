import json
import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

try:
    import backend.main as main_module
    from backend.main import (
        ARENA_ALLOWED_KEYS,
        MODEL_INFO_ALLOWED_KEYS,
        _load_arena_metrics,
        _load_model_info,
        app,
        limiter,
    )
except ImportError:
    import main as main_module
    from main import (
        ARENA_ALLOWED_KEYS,
        MODEL_INFO_ALLOWED_KEYS,
        _load_arena_metrics,
        _load_model_info,
        app,
        limiter,
    )

client = TestClient(app)


@pytest.fixture(autouse=True)
def disable_rate_limiting_default():
    """Disable rate limiting by default so standard functional tests never hit limits."""
    limiter.enabled = False
    yield
    limiter.reset()


@pytest.fixture
def enable_rate_limiting():
    """Enable rate limiting for specific rate limit verification tests."""
    limiter.enabled = True
    limiter.reset()
    yield
    limiter.reset()
    limiter.enabled = False


@pytest.fixture(autouse=True)
def isolated_request_log(monkeypatch, tmp_path):
    """Ensure no test writes to the project folder logs/ by redirecting to a tmp path."""
    log_file = tmp_path / "test_requests.jsonl"
    monkeypatch.setenv("REQUEST_LOG_PATH", str(log_file))
    return log_file

GOOD_INPUT = {
    "N": 78.0,
    "P": 42.0,
    "K": 42.0,
    "temperature": 20.13017482,
    "humidity": 81.60487287,
    "ph": 7.628472891,
    "rainfall": 262.7173405,
}

CAUTION_INPUT = {
    "N": 31.0,
    "P": 58.0,
    "K": 15.0,
    "temperature": 28.31886863,
    "humidity": 60.19461399,
    "ph": 6.167855382000001,
    "rainfall": 45.36521251,
}

REFUSE_INPUT = {
    "N": 90.0,
    "P": 42.0,
    "K": 43.0,
    "temperature": 21.0,
    "humidity": 92.0,
    "ph": 6.5,
    "rainfall": 25.0,
}


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_model_info():
    expected = json.loads(main_module.MODEL_INFO_PATH.read_text(encoding="utf-8"))
    response = client.get("/model-info")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data["model_version"], str) and data["model_version"]
    assert data["model_version"] == expected["model_version"]
    assert data["sklearn_version"] == expected["sklearn_version"]
    assert data["trained_on"] == expected["trained_on"]


def test_predict_recommend():
    response = client.post("/predict?risk=balanced", json=GOOD_INPUT)
    assert response.status_code == 200
    data = response.json()
    assert data["verdict"] == "recommend"
    assert isinstance(data["top_crops"], list) and len(data["top_crops"]) > 0
    assert data["top_crops"][0]["crop"] == "rice"
    assert data["ood"]["flagged"] is False
    assert "conformal_set" in data
    assert "measure_next" in data
    assert "unusual_pair" in data["ood"]
    assert "nearest_known" in data["ood"]
    assert "model_version" in data


def test_predict_caution():
    response = client.post("/predict?risk=balanced", json=CAUTION_INPUT)
    assert response.status_code == 200
    data = response.json()
    assert data["verdict"] == "caution"
    assert isinstance(data["top_crops"], list) and len(data["top_crops"]) > 0
    assert data["top_crops"][0]["crop"] == "lentil"
    assert data["ood"]["flagged"] is False
    assert "conformal_set" in data
    assert "measure_next" in data
    assert "unusual_pair" in data["ood"]
    assert "nearest_known" in data["ood"]
    assert "model_version" in data


def test_predict_refuse():
    response = client.post("/predict?risk=balanced", json=REFUSE_INPUT)
    assert response.status_code == 200
    data = response.json()
    assert data["verdict"] == "refuse"
    assert data["top_crops"] == []
    assert data["stability"] is None
    assert data["trust_score"] == 0
    assert data["ood"]["flagged"] is True
    assert "conformal_set" in data
    assert "measure_next" in data
    assert "unusual_pair" in data["ood"]
    assert "nearest_known" in data["ood"]
    assert "model_version" in data


def test_predict_all_risk_profiles():
    for risk_profile in ("conservative", "balanced", "bold"):
        response = client.post(f"/predict?risk={risk_profile}", json=GOOD_INPUT)
        assert response.status_code == 200
        data = response.json()
        assert "verdict" in data
        assert isinstance(data["model_version"], str) and data["model_version"]
        assert data["model_version"] == main_module.engine.MODEL_VERSION


def test_predict_invalid_risk_422():
    response = client.post("/predict?risk=extreme", json=GOOD_INPUT)
    assert response.status_code == 422
    data = response.json()
    assert "errors" in data
    assert any(err["field"] == "risk" for err in data["errors"])


def test_predict_invalid_bounds_422():
    bad_ph = dict(GOOD_INPUT, ph=40.0)
    response = client.post("/predict", json=bad_ph)
    assert response.status_code == 422
    data = response.json()
    assert "errors" in data
    assert any(err["field"] == "ph" for err in data["errors"])


def test_predict_missing_feature_422():
    missing_rainfall = dict(GOOD_INPUT)
    del missing_rainfall["rainfall"]
    response = client.post("/predict", json=missing_rainfall)
    assert response.status_code == 422
    data = response.json()
    assert "errors" in data
    assert any(err["field"] == "rainfall" for err in data["errors"])


def test_explain_calls_engine_and_matches_predict_shape():
    # Calling with familiar inputs returns recommend
    res_good = client.post("/explain?risk=balanced", json=GOOD_INPUT)
    assert res_good.status_code == 200
    data_good = res_good.json()
    assert data_good["verdict"] == "recommend"
    assert isinstance(data_good["top_crops"], list) and len(data_good["top_crops"]) > 0
    assert data_good["top_crops"][0]["crop"] == "rice"
    assert data_good["ood"]["flagged"] is False
    assert "model_version" in data_good
    assert "conformal_set" in data_good
    assert "measure_next" in data_good
    assert "unusual_pair" in data_good["ood"]
    assert "nearest_known" in data_good["ood"]

    # Calling with caution inputs returns caution
    res_caution = client.post("/explain?risk=balanced", json=CAUTION_INPUT)
    assert res_caution.status_code == 200
    data_caution = res_caution.json()
    assert data_caution["verdict"] == "caution"
    assert isinstance(data_caution["top_crops"], list) and len(data_caution["top_crops"]) > 0
    assert data_caution["top_crops"][0]["crop"] == "lentil"
    assert data_caution["ood"]["flagged"] is False
    assert "conformal_set" in data_caution
    assert "measure_next" in data_caution
    assert "unusual_pair" in data_caution["ood"]
    assert "nearest_known" in data_caution["ood"]


def test_not_found_returns_custom_error_shape():
    response = client.get("/nonexistent-endpoint")
    assert response.status_code == 404
    data = response.json()
    assert data == {"error": "Not Found"}


def test_cors_allowed_origin():
    response = client.get("/health", headers={"Origin": "http://localhost:5173"})
    assert response.status_code == 200
    assert response.headers.get("access-control-allow-origin") == "http://localhost:5173"


def test_cors_preflight():
    response = client.options(
        "/predict",
        headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "POST",
        },
    )
    assert response.status_code == 200
    assert response.headers.get("access-control-allow-origin") == "http://localhost:5173"


def test_cors_disallowed_origin():
    response = client.get("/health", headers={"Origin": "https://evil.example.com"})
    assert response.status_code == 200
    assert "access-control-allow-origin" not in response.headers


def test_cors_frontend_url_env(monkeypatch):
    monkeypatch.setenv("FRONTEND_URL", "https://crop-trust-engine-frontend.vercel.app")
    response = client.get(
        "/health",
        headers={"Origin": "https://crop-trust-engine-frontend.vercel.app"},
    )
    assert response.status_code == 200
    assert response.headers.get("access-control-allow-origin") == "https://crop-trust-engine-frontend.vercel.app"


# =============================================================================
# POST /predict/batch Test Suite
# =============================================================================

HEADER_LINE = "N,P,K,temperature,humidity,ph,rainfall"
GOOD_CSV_ROW = "78.0,42.0,42.0,20.13017482,81.60487287,7.628472891,262.7173405"
CAUTION_CSV_ROW = "31.0,58.0,15.0,28.31886863,60.19461399,6.167855382000001,45.36521251"
REFUSE_CSV_ROW = "90.0,42.0,43.0,21.0,92.0,6.5,25.0"


def test_predict_batch_mixed_verdicts():
    csv_data = f"{HEADER_LINE}\n{GOOD_CSV_ROW}\n{CAUTION_CSV_ROW}\n{REFUSE_CSV_ROW}\n"
    response = client.post(
        "/predict/batch?risk=balanced",
        files={"file": ("test.csv", csv_data, "text/csv")},
    )
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data["model_version"], str) and data["model_version"]
    assert data["model_version"] == main_module.engine.MODEL_VERSION
    assert data["count"] == 3
    assert len(data["results"]) == 3

    r0 = data["results"][0]
    assert r0["row_index"] == 0
    assert r0["verdict"] == "recommend"
    assert r0["top_crops"][0]["crop"] == "rice"
    assert r0["ood"]["flagged"] is False

    r1 = data["results"][1]
    assert r1["row_index"] == 1
    assert r1["verdict"] == "caution"
    assert r1["top_crops"][0]["crop"] == "lentil"
    assert r1["ood"]["flagged"] is False

    r2 = data["results"][2]
    assert r2["row_index"] == 2
    assert r2["verdict"] == "refuse"
    assert r2["top_crops"] == []
    assert r2["stability"] is None
    assert r2["trust_score"] == 0
    assert r2["ood"]["flagged"] is True


def test_predict_batch_row_level_error():
    bad_ph_row = "78.0,42.0,42.0,20.13017482,81.60487287,40.0,262.7173405"
    csv_data = f"{HEADER_LINE}\n{GOOD_CSV_ROW}\n{bad_ph_row}\n{REFUSE_CSV_ROW}\n"
    response = client.post(
        "/predict/batch",
        files={"file": ("test.csv", csv_data, "text/csv")},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["count"] == 3
    assert data["results"][0]["verdict"] == "recommend"
    assert data["results"][1] == {
        "row_index": 1,
        "error": "ph: Input should be less than or equal to 14",
    }
    assert data["results"][2]["verdict"] == "refuse"


def test_predict_batch_blank_and_non_numeric_cells():
    blank_cell_row = "78.0,42.0,42.0,20.13017482,81.60487287,,262.7173405"
    non_numeric_row = "78.0,42.0,42.0,abc,81.60487287,7.628472891,262.7173405"
    csv_data = f"{HEADER_LINE}\n{blank_cell_row}\n{non_numeric_row}\n{GOOD_CSV_ROW}\n"
    response = client.post(
        "/predict/batch",
        files={"file": ("test.csv", csv_data, "text/csv")},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["count"] == 3

    # Row 0 error has field name "ph"
    assert data["results"][0]["row_index"] == 0
    assert data["results"][0]["error"].startswith("ph:")

    # Row 1 error has field name "temperature"
    assert data["results"][1]["row_index"] == 1
    assert data["results"][1]["error"].startswith("temperature:")

    # Row 2 succeeds
    assert data["results"][2]["row_index"] == 2
    assert data["results"][2]["verdict"] == "recommend"


def test_predict_batch_header_only_returns_400_empty():
    csv_data = f"{HEADER_LINE}\n"
    response = client.post(
        "/predict/batch",
        files={"file": ("empty.csv", csv_data, "text/csv")},
    )
    assert response.status_code == 400
    assert response.json() == {"error": "File is empty"}


def test_predict_batch_trailing_blank_lines():
    csv_data = f"{HEADER_LINE}\n{GOOD_CSV_ROW}\n{REFUSE_CSV_ROW}\n\n   \n\n"
    response = client.post(
        "/predict/batch",
        files={"file": ("test.csv", csv_data, "text/csv")},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["count"] == 2
    assert len(data["results"]) == 2


def test_predict_batch_at_most_50_rows():
    rows = [GOOD_CSV_ROW] * 50
    csv_data = f"{HEADER_LINE}\n" + "\n".join(rows) + "\n"
    response = client.post(
        "/predict/batch",
        files={"file": ("50_rows.csv", csv_data, "text/csv")},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["count"] == 50
    assert len(data["results"]) == 50


def test_predict_batch_exceeds_50_rows_400():
    rows = [GOOD_CSV_ROW] * 51
    csv_data = f"{HEADER_LINE}\n" + "\n".join(rows) + "\n"
    response = client.post(
        "/predict/batch",
        files={"file": ("51_rows.csv", csv_data, "text/csv")},
    )
    assert response.status_code == 400
    assert response.json() == {"error": "At most 50 rows per upload"}


def test_predict_batch_missing_column_400():
    bad_header = "N,P,K,temperature,humidity,ph"  # Missing rainfall
    csv_data = f"{bad_header}\n90.0,42.0,43.0,21.0,82.0,6.5\n"
    response = client.post(
        "/predict/batch",
        files={"file": ("missing_col.csv", csv_data, "text/csv")},
    )
    assert response.status_code == 400
    assert response.json() == {"error": "Missing column: rainfall"}


def test_predict_batch_file_too_large_413():
    # 1 MB + 1 byte
    oversized_data = b"N,P,K,temperature,humidity,ph,rainfall\n" + b"A" * 1_000_001
    response = client.post(
        "/predict/batch",
        files={"file": ("large.csv", oversized_data, "text/csv")},
    )
    assert response.status_code == 413
    assert response.json() == {"error": "File too large (limit 1 MB)"}


def test_predict_batch_extra_columns_and_whitespace_headers():
    header_with_spaces_and_extra = " N , P , K , temperature , humidity , ph , rainfall , extra_notes , id "
    data_row = " 78.0 , 42.0 , 42.0 , 20.13017482 , 81.60487287 , 7.628472891 , 262.7173405 , sunny plot , 101 "
    csv_data = f"{header_with_spaces_and_extra}\n{data_row}\n"
    response = client.post(
        "/predict/batch",
        files={"file": ("extra.csv", csv_data, "text/csv")},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["count"] == 1
    assert data["results"][0]["verdict"] == "recommend"


def test_predict_batch_engine_exception_handling(monkeypatch):
    from backend import main

    original_assess = main.engine.assess

    def mocked_assess(row_dict, risk="balanced"):
        # Raise when rainfall is 45.36521251 (caution row)
        if row_dict.get("rainfall") == 45.36521251:
            raise RuntimeError("Secret internal failure detail that must not leak")
        return original_assess(row_dict, risk=risk)

    monkeypatch.setattr(main.engine, "assess", mocked_assess)

    csv_data = f"{HEADER_LINE}\n{GOOD_CSV_ROW}\n{CAUTION_CSV_ROW}\n{REFUSE_CSV_ROW}\n"
    response = client.post(
        "/predict/batch",
        files={"file": ("test.csv", csv_data, "text/csv")},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["count"] == 3

    # Row 0 succeeds
    assert data["results"][0]["verdict"] == "recommend"

    # Row 1 caught exception
    assert data["results"][1] == {
        "row_index": 1,
        "error": "Could not assess this row",
    }
    # Verify no traceback or internal error leaked in response text
    assert "Secret internal failure detail" not in response.text

    # Row 2 succeeds
    assert data["results"][2]["verdict"] == "refuse"


def test_predict_batch_risk_parameter():
    csv_data = f"{HEADER_LINE}\n{GOOD_CSV_ROW}\n"
    for r in ("conservative", "balanced", "bold"):
        response = client.post(
            f"/predict/batch?risk={r}",
            files={"file": ("test.csv", csv_data, "text/csv")},
        )
        assert response.status_code == 200
        assert response.json()["count"] == 1


def test_predict_batch_invalid_csv_format():
    binary_data = b"\x80\x81\xff\xfe\x00\x01\x02\x03"
    response = client.post(
        "/predict/batch",
        files={"file": ("corrupt.csv", binary_data, "text/csv")},
    )
    assert response.status_code == 400
    assert response.json() == {"error": "Invalid CSV format"}


# =============================================================================
# Structured Request Logging Test Suite
# =============================================================================

EXPECTED_LOG_KEYS = {
    "time",
    "verdict",
    "ood_percentile",
    "stability",
    "latency_ms",
    "model_version",
    "N",
    "P",
    "K",
    "temperature",
    "humidity",
    "ph",
    "rainfall",
}


def test_predict_writes_structured_log_entry(isolated_request_log):
    response = client.post("/predict?risk=balanced", json=GOOD_INPUT)
    assert response.status_code == 200

    assert isolated_request_log.exists()
    lines = isolated_request_log.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1

    entry = json.loads(lines[0])
    assert set(entry.keys()) == EXPECTED_LOG_KEYS
    assert entry["verdict"] == "recommend"
    assert isinstance(entry["model_version"], str) and entry["model_version"]
    assert entry["model_version"] == main_module.engine.MODEL_VERSION
    assert entry["N"] == 78.0
    assert entry["P"] == 42.0
    assert entry["K"] == 42.0
    assert entry["temperature"] == 20.13017482
    assert entry["humidity"] == 81.60487287
    assert entry["ph"] == 7.628472891
    assert entry["rainfall"] == 262.7173405
    assert isinstance(entry["ood_percentile"], (int, float))
    assert isinstance(entry["stability"], (int, float))
    assert 0.0 <= entry["stability"] <= 1.0
    assert isinstance(entry["latency_ms"], float)
    assert entry["latency_ms"] >= 0.0
    assert isinstance(entry["time"], str)


def test_predict_logs_no_personal_data(isolated_request_log):
    response = client.post(
        "/predict?risk=balanced",
        json=GOOD_INPUT,
        headers={
            "User-Agent": "FarmerSpecialBrowser/4.0",
            "X-Forwarded-For": "203.0.113.195",
            "Cookie": "secret_cookie_token=abc123xyz",
        },
    )
    assert response.status_code == 200

    raw_text = isolated_request_log.read_text(encoding="utf-8")
    assert "FarmerSpecialBrowser" not in raw_text
    assert "203.0.113.195" not in raw_text
    assert "secret_cookie_token" not in raw_text

    entry = json.loads(raw_text.strip())
    assert set(entry.keys()) == EXPECTED_LOG_KEYS


def test_predict_batch_logs_each_successful_row(isolated_request_log):
    csv_data = f"{HEADER_LINE}\n{GOOD_CSV_ROW}\n{CAUTION_CSV_ROW}\n{REFUSE_CSV_ROW}\n"
    response = client.post(
        "/predict/batch?risk=balanced",
        files={"file": ("three_rows.csv", csv_data, "text/csv")},
    )
    assert response.status_code == 200

    lines = isolated_request_log.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 3

    entries = [json.loads(line) for line in lines]
    assert entries[0]["verdict"] == "recommend"
    assert entries[1]["verdict"] == "caution"
    assert entries[2]["verdict"] == "refuse"

    for e in entries:
        assert set(e.keys()) == EXPECTED_LOG_KEYS
        assert "three_rows.csv" not in json.dumps(e)


def test_predict_batch_failed_rows_not_logged(isolated_request_log):
    bad_ph_row = "78.0,42.0,42.0,20.13017482,81.60487287,40.0,262.7173405"
    csv_data = f"{HEADER_LINE}\n{GOOD_CSV_ROW}\n{bad_ph_row}\n{REFUSE_CSV_ROW}\n"
    response = client.post(
        "/predict/batch",
        files={"file": ("test.csv", csv_data, "text/csv")},
    )
    assert response.status_code == 200
    assert response.json()["count"] == 3

    lines = isolated_request_log.read_text(encoding="utf-8").strip().splitlines()
    # Row 1 failed validation, so only Row 0 and Row 2 are logged
    assert len(lines) == 2

    entries = [json.loads(line) for line in lines]
    assert entries[0]["verdict"] == "recommend"
    assert entries[1]["verdict"] == "refuse"


def test_explain_not_logged(isolated_request_log):
    response = client.post("/explain?risk=balanced", json=GOOD_INPUT)
    assert response.status_code == 200
    assert not isolated_request_log.exists() or isolated_request_log.stat().st_size == 0


def test_health_and_model_info_not_logged(isolated_request_log):
    res_h = client.get("/health")
    assert res_h.status_code == 200
    res_m = client.get("/model-info")
    assert res_m.status_code == 200
    assert not isolated_request_log.exists() or isolated_request_log.stat().st_size == 0


def test_predict_422_not_logged(isolated_request_log):
    bad_input = dict(GOOD_INPUT, ph=40.0)
    response = client.post("/predict", json=bad_input)
    assert response.status_code == 422
    assert not isolated_request_log.exists() or isolated_request_log.stat().st_size == 0


def test_log_write_failure_does_not_break_predict(monkeypatch, isolated_request_log):
    import backend.request_log

    def failing_append(log_path, line):
        raise OSError("Simulated disk quota failure")

    monkeypatch.setattr(backend.request_log, "_append_to_file", failing_append)

    response = client.post("/predict?risk=balanced", json=GOOD_INPUT)
    assert response.status_code == 200
    data = response.json()
    assert data["verdict"] == "recommend"
    assert isinstance(data["model_version"], str) and data["model_version"]
    assert data["model_version"] == main_module.engine.MODEL_VERSION


def test_logging_respects_custom_path_env(monkeypatch, tmp_path):
    custom_path = tmp_path / "custom_logs_dir" / "custom_requests.jsonl"
    monkeypatch.setenv("REQUEST_LOG_PATH", str(custom_path))

    response = client.post("/predict?risk=balanced", json=GOOD_INPUT)
    assert response.status_code == 200
    assert custom_path.exists()

    lines = custom_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert entry["verdict"] == "recommend"


def test_logging_rotation_at_5mb(isolated_request_log):
    # Seed log file to just over 5 MB
    isolated_request_log.parent.mkdir(parents=True, exist_ok=True)
    with open(isolated_request_log, "w", encoding="utf-8") as f:
        f.write("x" * (5 * 1024 * 1024 + 50) + "\n")

    response = client.post("/predict?risk=balanced", json=GOOD_INPUT)
    assert response.status_code == 200

    backup_path = isolated_request_log.with_name(f"{isolated_request_log.name}.1")
    assert backup_path.exists()
    assert backup_path.stat().st_size >= 5 * 1024 * 1024

    # Active log file should now contain only the newly written entry
    active_lines = isolated_request_log.read_text(encoding="utf-8").strip().splitlines()
    assert len(active_lines) == 1
    entry = json.loads(active_lines[0])
    assert entry["verdict"] == "recommend"


def test_rate_limit_predict_returns_429(enable_rate_limiting):
    headers = {"X-Forwarded-For": "198.51.100.1"}
    # Default RATE_LIMIT_PREDICT is 60/minute; send 60 successful requests
    for _ in range(60):
        resp = client.post("/predict?risk=balanced", json=GOOD_INPUT, headers=headers)
        assert resp.status_code == 200

    # 61st request triggers 429
    res_429 = client.post("/predict?risk=balanced", json=GOOD_INPUT, headers=headers)
    assert res_429.status_code == 429
    data = res_429.json()
    assert "error" in data
    assert data["error"].startswith("Rate limit exceeded. Try again in ")
    assert data["error"].endswith(" seconds.")

    # Retry-After header must be present and a positive integer
    assert "Retry-After" in res_429.headers
    retry_after = int(res_429.headers["Retry-After"])
    assert retry_after > 0
    assert data["error"] == f"Rate limit exceeded. Try again in {retry_after} seconds."


def test_rate_limit_429_cors_headers(enable_rate_limiting):
    headers = {
        "Origin": "http://localhost:5173",
        "X-Forwarded-For": "198.51.100.2",
    }
    for _ in range(60):
        resp = client.post("/predict?risk=balanced", json=GOOD_INPUT, headers=headers)
        assert resp.status_code == 200

    res_429 = client.post("/predict?risk=balanced", json=GOOD_INPUT, headers=headers)
    assert res_429.status_code == 429

    # CORS headers must be present and expose Retry-After
    assert res_429.headers.get("access-control-allow-origin") == "http://localhost:5173"
    exposed = res_429.headers.get("access-control-expose-headers", "")
    assert "Retry-After" in exposed
    assert "Retry-After" in res_429.headers


def test_health_is_never_rate_limited(enable_rate_limiting):
    headers = {"X-Forwarded-For": "198.51.100.3"}
    for _ in range(70):
        resp = client.get("/health", headers=headers)
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}


def test_rate_limit_ip_isolation(enable_rate_limiting):
    ip_a = {"X-Forwarded-For": "198.51.100.10"}
    ip_b = {"X-Forwarded-For": "198.51.100.20"}

    # Exhaust limit for IP A
    for _ in range(60):
        resp = client.post("/predict?risk=balanced", json=GOOD_INPUT, headers=ip_a)
        assert resp.status_code == 200

    resp_a_blocked = client.post("/predict?risk=balanced", json=GOOD_INPUT, headers=ip_a)
    assert resp_a_blocked.status_code == 429

    # IP B is not blocked
    resp_b = client.post("/predict?risk=balanced", json=GOOD_INPUT, headers=ip_b)
    assert resp_b.status_code == 200
    assert resp_b.json()["verdict"] == "recommend"


def test_rate_limited_requests_not_logged(enable_rate_limiting, isolated_request_log):
    headers = {"X-Forwarded-For": "198.51.100.25"}
    for _ in range(60):
        resp = client.post("/predict?risk=balanced", json=GOOD_INPUT, headers=headers)
        assert resp.status_code == 200

    initial_log_lines = isolated_request_log.read_text(encoding="utf-8").strip().splitlines()
    assert len(initial_log_lines) == 60

    # 61st request is rate limited
    res_429 = client.post("/predict?risk=balanced", json=GOOD_INPUT, headers=headers)
    assert res_429.status_code == 429

    # No additional log line is written
    after_log_lines = isolated_request_log.read_text(encoding="utf-8").strip().splitlines()
    assert len(after_log_lines) == 60


def test_rate_limit_batch_works(enable_rate_limiting):
    headers = {"X-Forwarded-For": "198.51.100.30"}
    csv_bytes = b"N,P,K,temperature,humidity,ph,rainfall\n78.0,42.0,42.0,20.13017482,81.60487287,7.628472891,262.7173405\n"

    # Default RATE_LIMIT_BATCH is 5/minute
    for _ in range(5):
        resp = client.post(
            "/predict/batch",
            files={"file": ("batch.csv", csv_bytes, "text/csv")},
            headers=headers,
        )
        assert resp.status_code == 200

    # 6th request triggers 429
    res_429 = client.post(
        "/predict/batch",
        files={"file": ("batch.csv", csv_bytes, "text/csv")},
        headers=headers,
    )
    assert res_429.status_code == 429
    data = res_429.json()
    assert "error" in data
    assert data["error"].startswith("Rate limit exceeded. Try again in ")
    assert "Retry-After" in res_429.headers
    retry_after = int(res_429.headers["Retry-After"])
    assert retry_after > 0


def test_rate_limit_env_configurable(enable_rate_limiting, monkeypatch):
    monkeypatch.setenv("RATE_LIMIT_PREDICT", "2/minute")
    headers = {"X-Forwarded-For": "198.51.100.40"}

    r1 = client.post("/predict?risk=balanced", json=GOOD_INPUT, headers=headers)
    assert r1.status_code == 200
    r2 = client.post("/predict?risk=balanced", json=GOOD_INPUT, headers=headers)
    assert r2.status_code == 200

    r3 = client.post("/predict?risk=balanced", json=GOOD_INPUT, headers=headers)
    assert r3.status_code == 429
    assert "Retry-After" in r3.headers


# =============================================================================
# GET /arena Test Suite
# =============================================================================

EXPECTED_ARENA_KEYS = {
    "benchmark_metadata",
    "models_summary",
    "fold_level_metrics",
    "significance_tests",
    "champion_model",
    "class_wise_performance",
    "calibration_experiment_e4",
}


def test_arena_returns_allowed_keys_and_excludes_withheld():
    response = client.get("/arena")
    assert response.status_code == 200
    data = response.json()
    assert set(data.keys()) == EXPECTED_ARENA_KEYS
    assert set(ARENA_ALLOWED_KEYS) == EXPECTED_ARENA_KEYS
    assert "class_wise_performance_rf_baseline" not in data
    assert "ablations_experiment_e11" not in data
    assert "experiment_5_weather_only_model" not in response.text


def test_arena_excludes_selection_rationale():
    response = client.get("/arena")
    assert response.status_code == 200
    data = response.json()
    assert "selection_rationale" not in data
    assert "selection_rationale" not in response.text


def test_arena_allowlist_drops_unrecognized_keys(tmp_path):
    sample = {
        "benchmark_metadata": {"classes_count": 22},
        "champion_model": "Extra Trees (Isotonic)",
        "selection_rationale": "Cites E11 results",
        "class_wise_performance_rf_baseline": {"ambiguous_rows": []},
        "ablations_experiment_e11": {"experiment_5_weather_only_model": {}},
        "experiment_5_weather_only_model": {"accuracy": 0.5},
        "unexpected_extra_key": "should_not_appear",
    }
    sample_path = tmp_path / "arena_sample.json"
    sample_path.write_text(json.dumps(sample), encoding="utf-8")

    loaded = _load_arena_metrics(sample_path)
    assert loaded == {
        "benchmark_metadata": {"classes_count": 22},
        "champion_model": "Extra Trees (Isotonic)",
    }


def test_arena_missing_file_returns_500_no_traceback(monkeypatch, tmp_path):
    missing_path = tmp_path / "missing_arena_metrics.json"
    assert _load_arena_metrics(missing_path) is None

    monkeypatch.setattr(main_module, "_ARENA_METRICS", None)
    response = client.get("/arena")
    assert response.status_code == 500
    assert response.json() == {"error": "Arena metrics unavailable"}
    assert "Traceback" not in response.text
    assert "FileNotFoundError" not in response.text
    assert "arena_metrics.json" not in response.text
    assert str(tmp_path) not in response.text


def test_arena_invalid_json_returns_500_no_traceback(monkeypatch, tmp_path):
    bad_json_path = tmp_path / "corrupt.json"
    bad_json_path.write_text("{not valid json", encoding="utf-8")
    assert _load_arena_metrics(bad_json_path) is None

    list_json_path = tmp_path / "list.json"
    list_json_path.write_text('["not", "an", "object"]', encoding="utf-8")
    assert _load_arena_metrics(list_json_path) is None

    monkeypatch.setattr(main_module, "_ARENA_METRICS", None)
    response = client.get("/arena")
    assert response.status_code == 500
    assert response.json() == {"error": "Arena metrics unavailable"}
    assert "Traceback" not in response.text
    assert "JSONDecodeError" not in response.text
    assert str(tmp_path) not in response.text


def test_arena_not_logged(isolated_request_log):
    response = client.get("/arena")
    assert response.status_code == 200
    assert not isolated_request_log.exists() or isolated_request_log.stat().st_size == 0


def test_rate_limit_arena_works(enable_rate_limiting):
    headers = {"X-Forwarded-For": "198.51.100.50"}
    # Default RATE_LIMIT_ARENA is 60/minute
    for _ in range(60):
        resp = client.get("/arena", headers=headers)
        assert resp.status_code == 200

    res_429 = client.get("/arena", headers=headers)
    assert res_429.status_code == 429
    data = res_429.json()
    assert "error" in data
    assert data["error"].startswith("Rate limit exceeded. Try again in ")
    assert "Retry-After" in res_429.headers
    assert int(res_429.headers["Retry-After"]) > 0


def test_rate_limit_arena_env_configurable(enable_rate_limiting, monkeypatch):
    monkeypatch.setenv("RATE_LIMIT_ARENA", "2/minute")
    headers = {"X-Forwarded-For": "198.51.100.51"}

    r1 = client.get("/arena", headers=headers)
    assert r1.status_code == 200
    r2 = client.get("/arena", headers=headers)
    assert r2.status_code == 200

    r3 = client.get("/arena", headers=headers)
    assert r3.status_code == 429
    assert "Retry-After" in r3.headers


# =============================================================================
# GET /model-info Real Metadata Test Suite
# =============================================================================

EXPECTED_MODEL_INFO_KEYS = {
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
}


def test_model_info_all_keys_present(tmp_path):
    response = client.get("/model-info")
    assert response.status_code == 200
    data = response.json()
    assert set(data.keys()) == EXPECTED_MODEL_INFO_KEYS
    assert set(MODEL_INFO_ALLOWED_KEYS) == EXPECTED_MODEL_INFO_KEYS

    expected = json.loads(main_module.MODEL_INFO_PATH.read_text(encoding="utf-8"))
    for key in EXPECTED_MODEL_INFO_KEYS:
        assert data[key] == expected[key]

    # Verify allow-list drops extra keys and rejects incomplete payloads
    with_extra = dict(expected, extra_unapproved_field="ignore-me")
    extra_path = tmp_path / "extra_model_info.json"
    extra_path.write_text(json.dumps(with_extra), encoding="utf-8")
    loaded_extra = _load_model_info(extra_path)
    assert loaded_extra is not None
    assert set(loaded_extra.keys()) == EXPECTED_MODEL_INFO_KEYS

    incomplete = dict(expected)
    del incomplete["data_fingerprint"]
    incomplete_path = tmp_path / "incomplete_model_info.json"
    incomplete_path.write_text(json.dumps(incomplete), encoding="utf-8")
    assert _load_model_info(incomplete_path) is None


def test_model_info_missing_file_returns_500_no_traceback(monkeypatch, tmp_path):
    missing_path = tmp_path / "missing_model_info.json"
    assert _load_model_info(missing_path) is None

    monkeypatch.setattr(main_module, "_MODEL_INFO", None)
    response = client.get("/model-info")
    assert response.status_code == 500
    assert response.json() == {"error": "Model info unavailable"}
    assert "Traceback" not in response.text
    assert "FileNotFoundError" not in response.text
    assert "model_info.json" not in response.text
    assert str(tmp_path) not in response.text


def test_model_info_invalid_json_returns_500_no_traceback(monkeypatch, tmp_path):
    bad_json_path = tmp_path / "corrupt_model_info.json"
    bad_json_path.write_text("{not valid json", encoding="utf-8")
    assert _load_model_info(bad_json_path) is None

    list_json_path = tmp_path / "list_model_info.json"
    list_json_path.write_text('["not", "an", "object"]', encoding="utf-8")
    assert _load_model_info(list_json_path) is None

    monkeypatch.setattr(main_module, "_MODEL_INFO", None)
    response = client.get("/model-info")
    assert response.status_code == 500
    assert response.json() == {"error": "Model info unavailable"}
    assert "Traceback" not in response.text
    assert "JSONDecodeError" not in response.text
    assert str(tmp_path) not in response.text


def test_rate_limit_model_info_works(enable_rate_limiting, monkeypatch):
    headers_default = {"X-Forwarded-For": "198.51.100.60"}
    # Default RATE_LIMIT_MODEL_INFO is 60/minute
    for _ in range(60):
        resp = client.get("/model-info", headers=headers_default)
        assert resp.status_code == 200

    res_429 = client.get("/model-info", headers=headers_default)
    assert res_429.status_code == 429
    data = res_429.json()
    assert "error" in data
    assert data["error"].startswith("Rate limit exceeded. Try again in ")
    assert "Retry-After" in res_429.headers
    assert int(res_429.headers["Retry-After"]) > 0

    monkeypatch.setenv("RATE_LIMIT_MODEL_INFO", "2/minute")
    headers_custom = {"X-Forwarded-For": "198.51.100.61"}
    assert client.get("/model-info", headers=headers_custom).status_code == 200
    assert client.get("/model-info", headers=headers_custom).status_code == 200
    res_custom_429 = client.get("/model-info", headers=headers_custom)
    assert res_custom_429.status_code == 429
    assert "Retry-After" in res_custom_429.headers





