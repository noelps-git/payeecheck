"""Unit tests for api.py — route wiring, validation and error handling.

Levels 4/5 (sentence-transformers, torch) are never hit here: those routes are
exercised with a stubbed matcher so the suite stays offline and fast.
"""
import pytest
from fastapi.testclient import TestClient

import api


@pytest.fixture(scope="module")
def client():
    return TestClient(api.app)


@pytest.fixture
def stub_matchers(monkeypatch):
    """Replace the lazy matcher loader so no ML model is downloaded."""
    def _get_matcher(level):
        if level not in (1, 2, 3, 4, 5):
            raise api.HTTPException(status_code=404,
                                    detail=f"No matcher for level {level}")
        return lambda entered, actual: {"match": "exact", "score": 1.0,
                                        "level": level}
    monkeypatch.setattr(api, "get_matcher", _get_matcher)


def test_health():
    with TestClient(api.app) as client:
        body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["levels_available"] == [1, 2, 3, 4, 5, 6]


def test_root_serves_the_dashboard(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]


def test_root_404s_when_the_ui_file_is_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(api, "_STATIC_DIR", str(tmp_path))
    with TestClient(api.app) as client:
        response = client.get("/")
    assert response.status_code == 404
    assert "UI file not found" in response.json()["detail"]


def test_match_route_delegates_to_the_requested_level(client, stub_matchers):
    body = client.post("/match/2", json={"entered": "SBI",
                                         "actual": "State Bank of India"}).json()
    assert body["level"] == 2


@pytest.mark.parametrize("level", [0, 6, 99])
def test_match_rejects_out_of_range_levels(client, stub_matchers, level):
    response = client.post(f"/match/{level}",
                           json={"entered": "a", "actual": "b"})
    assert response.status_code == 400
    assert "Level must be 1-5" in response.json()["detail"]


def test_match_requires_both_names(client, stub_matchers):
    assert client.post("/match/1", json={"entered": "a"}).status_code == 422


def test_compare_runs_every_level(client, stub_matchers):
    body = client.post("/compare", json={"entered": "SBI",
                                         "actual": "State Bank of India"}).json()
    assert body["entered"] == "SBI"
    assert list(body["results"]) == [f"level_{i}" for i in range(1, 6)]


def test_get_matcher_caches_and_rejects_unknown_levels(monkeypatch):
    monkeypatch.setattr(api, "_matchers_cache", {})
    first = api.get_matcher(1)
    assert api.get_matcher(1) is first
    with pytest.raises(api.HTTPException) as excinfo:
        api.get_matcher(9)
    assert excinfo.value.status_code == 404


def test_attribution_route_returns_the_three_layer_record(client):
    body = client.post("/attribution", json={
        "transaction": {"input_method": "paste", "paste_trust_score": 0.08,
                        "amount": 24000},
        "payee": {"vpa_age_days": 11, "mule_flagged": True,
                  "mule_flag_count": 3, "name_score": 0.4},
        "temporal": {"first_time_payee": True, "amount_vs_avg_ratio": 1.2},
    }).json()
    assert body["verdict"] in ("block", "warn", "pass")
    assert body["total_signals_fired"] == 5
    assert set(body["attribution"]) == {"L1", "L2", "L3"}


def test_score_route_runs_the_full_pipeline(client):
    body = client.post("/score", json={
        "entered_name": "Paytm Support", "actual_name": "Vikram Nair",
        "payee_vpa": "paytm.support@axl", "amount": 45000,
        "input_method": "paste", "vpa_age_days": 4, "prior_tx_count": 0,
        "unique_senders_7d": 0, "mule_flagged": True, "mule_flag_count": 3,
    }).json()
    assert body["tx_id"] == "live"
    assert body["verdict"] == "block"
    assert body["risk_score"] >= 90
    assert "lookalike" in body["module_outputs"]


def test_score_route_applies_field_defaults(client):
    body = client.post("/score", json={
        "entered_name": "Suresh Kumar", "actual_name": "Suresh Kumar",
        "payee_vpa": "suresh.kumar@ybl", "amount": 2500}).json()
    assert body["verdict"] == "pass"
    assert body["module_outputs"]["velocity"]["vpa_age_days"] == 365


def test_score_route_validates_required_fields(client):
    assert client.post("/score", json={"amount": 10}).status_code == 422


def test_resolve_route_returns_graph_matches(client):
    body = client.post("/resolve", json={"name": "Suresh Kumar"}).json()
    assert isinstance(body, dict)


def test_rings_route_respects_min_size(client):
    body = client.get("/rings", params={"min_size": 2}).json()
    assert body["ring_count"] == len(body["rings"])


def test_sandbox_router_is_mounted(client):
    assert client.get("/sandbox/v1/scenarios").status_code == 200


def test_cors_headers_are_permissive(client):
    response = client.get("/health", headers={"Origin": "http://localhost:3000"})
    assert response.headers["access-control-allow-origin"] == "*"
