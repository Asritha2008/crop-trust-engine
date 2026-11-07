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
