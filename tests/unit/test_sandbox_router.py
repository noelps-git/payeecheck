"""Unit tests for sandbox/router.py — the deterministic demo environment."""
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from sandbox.router import SCENARIOS, SandboxRequest, _build_response, router


@pytest.fixture(scope="module")
def client():
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def test_scenario_library_covers_the_documented_ids():
    assert set(SCENARIOS) == {
        "clean", "mule", "mule_consortium", "lookalike", "clipboard_scam",
        "biz_individual_mismatch", "name_mismatch_abbr", "fresh_vpa",
        "ring_member", "ring_satellite", "sanction", "seasonal_burst",
        "session_anomaly"}
    for name, scenario in SCENARIOS.items():
        assert scenario["verdict"] in ("block", "warn", "pass"), name
        assert 0 <= scenario["risk_score"] <= 100, name
        assert scenario["signals"] and scenario["description"], name


def test_list_scenarios_endpoint(client):
    body = client.get("/sandbox/v1/scenarios").json()
    assert len(body["scenarios"]) == len(SCENARIOS)
    entry = next(s for s in body["scenarios"] if s["scenario_id"] == "sanction")
    assert entry["verdict"] == "block"
    assert entry["risk_score"] == 97


@pytest.mark.parametrize("scenario_id", sorted(SCENARIOS))
def test_every_scenario_returns_its_fixed_verdict(client, scenario_id):
    body = client.post("/sandbox/v1/payee-checks",
                       json={"scenario_id": scenario_id}).json()
    assert body["sandbox"] is True
    assert body["scenario_id"] == scenario_id
    assert body["risk_verdict"]["verdict"] == SCENARIOS[scenario_id]["verdict"]
    assert body["risk_verdict"]["risk_score"] == \
        SCENARIOS[scenario_id]["risk_score"]
    assert body["payee_check"]["id"] == body["risk_verdict"]["payee_check_id"]
    assert body["payee_check"]["bank_id"] == "sandbox_bank"


def test_scoring_is_deterministic_except_for_ids(client):
    first = client.post("/sandbox/v1/payee-checks",
                        json={"scenario_id": "mule"}).json()
    second = client.post("/sandbox/v1/payee-checks",
                         json={"scenario_id": "mule"}).json()
    for body in (first, second):
        body["payee_check"].pop("id")
        body["payee_check"].pop("created_at")
        body["risk_verdict"].pop("id")
        body["risk_verdict"].pop("payee_check_id")
    assert first == second


def test_unknown_and_missing_scenario_fall_back_to_clean(client):
    unknown = client.post("/sandbox/v1/payee-checks",
                          json={"scenario_id": "not_a_scenario"}).json()
    assert unknown["scenario_id"] == "not_a_scenario"
    assert unknown["risk_verdict"]["risk_score"] == SCENARIOS["clean"]["risk_score"]

    empty = client.post("/sandbox/v1/payee-checks", json={}).json()
    assert empty["scenario_id"] == "clean"

    explicit_null = client.post("/sandbox/v1/payee-checks",
                                json={"scenario_id": None}).json()
    assert explicit_null["scenario_id"] == "clean"


def test_caller_supplied_evidence_overrides_scenario_defaults(client):
    body = client.post("/sandbox/v1/payee-checks", json={
        "scenario_id": "clean", "entered_name": "My Payee",
        "payee_vpa": "my.payee@ybl", "amount": 100.0}).json()
    evidence = body["payee_check"]["evidence"]
    assert evidence["entered_name"] == "My Payee"
    assert evidence["payee_vpa"] == "my.payee@ybl"
    assert evidence["actual_name"] == SCENARIOS["clean"]["actual_name"]


def test_get_scenario_by_id_uses_scenario_defaults(client):
    body = client.get("/sandbox/v1/payee-checks/lookalike").json()
    evidence = body["payee_check"]["evidence"]
    assert evidence["entered_name"] == SCENARIOS["lookalike"]["entered_name"]
    assert evidence["payee_vpa"] == SCENARIOS["lookalike"]["payee_vpa"]
    assert body["risk_verdict"]["verdict"] == "block"


@pytest.mark.parametrize("scenario_id,label", [
    ("clean", "PAYMENT PASSED"),
    ("clipboard_scam", "FRICTION APPLIED"),
    ("sanction", "PAYMENT BLOCKED"),
])
def test_verdict_labels(scenario_id, label):
    body = _build_response(scenario_id, SandboxRequest())
    assert body["risk_verdict"]["verdict_label"] == label
    assert body["risk_verdict"]["verdict_icon"]


def test_ids_are_namespaced_and_timestamped():
    body = _build_response("mule", SandboxRequest())
    assert body["payee_check"]["id"].startswith("sandbox_pchk_mule_")
    assert body["risk_verdict"]["id"].startswith("sandbox_rv_mule_")
    assert body["payee_check"]["created_at"].endswith("Z")
