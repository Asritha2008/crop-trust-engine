import pytest
from fastapi.testclient import TestClient

try:
    from backend.main import app
except ImportError:
    from main import app

client = TestClient(app)

GOOD_INPUT = {
    "N": 90.0,
    "P": 42.0,
    "K": 43.0,
    "temperature": 21.0,
    "humidity": 82.0,
    "ph": 6.5,
    "rainfall": 203.0,
}

CAUTION_INPUT = {
    "N": 90.0,
    "P": 42.0,
    "K": 43.0,
    "temperature": 35.0,
    "humidity": 82.0,
    "ph": 6.5,
    "rainfall": 180.0,
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
    response = client.get("/model-info")
    assert response.status_code == 200
    data = response.json()
    assert data["model_version"] == "stub-2026-11-07"
    assert data["sklearn_version"] == "1.7.2"
    assert "placeholder" in data["trained_on"]


def test_predict_recommend():
    response = client.post("/predict?risk=balanced", json=GOOD_INPUT)
    assert response.status_code == 200
    data = response.json()
    assert data["verdict"] == "recommend"
    assert len(data["top_crops"]) == 3
    assert data["top_crops"][0]["crop"] == "rice"
    assert data["conformal_set"] is None
    assert data["measure_next"] is None
    assert data["stability"] == 0.94
    assert data["trust_score"] == 88
    assert data["model_version"] == "stub-2026-11-07"
    assert data["ood"]["flagged"] is False
    assert data["ood"]["unusual_pair"] is None
    assert data["ood"]["nearest_known"] is None
    assert "[STUB / ILLUSTRATIVE]" in data["explanation"]


def test_predict_caution():
    response = client.post("/predict?risk=balanced", json=CAUTION_INPUT)
    assert response.status_code == 200
    data = response.json()
    assert data["verdict"] == "caution"
    assert len(data["top_crops"]) == 3
    assert data["conformal_set"] == ["rice", "jute"]
    assert "Potassium" in data["measure_next"]
    assert data["stability"] == 0.71
    assert data["trust_score"] == 52
    assert data["ood"]["flagged"] is False
    assert data["ood"]["unusual_pair"] is None
    assert data["ood"]["nearest_known"] is None
    assert "[STUB / ILLUSTRATIVE]" in data["explanation"]


def test_predict_refuse():
    response = client.post("/predict?risk=balanced", json=REFUSE_INPUT)
    assert response.status_code == 200
    data = response.json()
    assert data["verdict"] == "refuse"
    assert data["top_crops"] == []
    assert data["stability"] is None
    assert data["trust_score"] == 0
    assert data["conformal_set"] is None
    assert data["measure_next"] is None
    assert data["ood"]["flagged"] is True
    assert data["ood"]["unusual_pair"] == ["humidity", "rainfall"]
    assert data["ood"]["nearest_known"] == ["rice", "jute"]
    assert "[STUB / ILLUSTRATIVE]" in data["explanation"]


def test_predict_all_risk_profiles():
    for risk_profile in ("conservative", "balanced", "bold"):
        response = client.post(f"/predict?risk={risk_profile}", json=GOOD_INPUT)
        assert response.status_code == 200
        data = response.json()
        assert "verdict" in data
        assert data["model_version"] == "stub-2026-11-07"


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
    assert "model_version" in data_good
    assert data_good["conformal_set"] is None

    # Calling with caution inputs returns caution
    res_caution = client.post("/explain?risk=balanced", json=CAUTION_INPUT)
    assert res_caution.status_code == 200
    data_caution = res_caution.json()
    assert data_caution["verdict"] == "caution"
    assert data_caution["conformal_set"] == ["rice", "jute"]
    assert data_caution["measure_next"] is not None


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
GOOD_CSV_ROW = "90.0,42.0,43.0,21.0,82.0,6.5,203.0"
CAUTION_CSV_ROW = "90.0,42.0,43.0,35.0,82.0,6.5,180.0"
REFUSE_CSV_ROW = "90.0,42.0,43.0,21.0,92.0,6.5,25.0"


def test_predict_batch_mixed_verdicts():
    csv_data = f"{HEADER_LINE}\n{GOOD_CSV_ROW}\n{CAUTION_CSV_ROW}\n{REFUSE_CSV_ROW}\n"
    response = client.post(
        "/predict/batch?risk=balanced",
        files={"file": ("test.csv", csv_data, "text/csv")},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["model_version"] == "stub-2026-11-07"
    assert data["count"] == 3
    assert len(data["results"]) == 3

    r0 = data["results"][0]
    assert r0["row_index"] == 0
    assert r0["verdict"] == "recommend"
    assert r0["top_crops"][0]["crop"] == "rice"

    r1 = data["results"][1]
    assert r1["row_index"] == 1
    assert r1["verdict"] == "caution"

    r2 = data["results"][2]
    assert r2["row_index"] == 2
    assert r2["verdict"] == "refuse"


def test_predict_batch_row_level_error():
    bad_ph_row = "90.0,42.0,43.0,21.0,82.0,40.0,203.0"
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
    blank_cell_row = "90.0,42.0,43.0,21.0,82.0,,203.0"
    non_numeric_row = "90.0,42.0,43.0,abc,82.0,6.5,203.0"
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
    data_row = " 90.0 , 42.0 , 43.0 , 21.0 , 82.0 , 6.5 , 203.0 , sunny plot , 101 "
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
        # Raise when rainfall is 180.0 (caution row)
        if row_dict.get("rainfall") == 180.0:
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


