from unittest.mock import patch
import pytest
from backend.app.models.entities import Case, Prediction, Decision, EvidencePacket, AuditEvent
from backend.app.services.decision_service import DecisionService
from backend.app.services.case_service import CaseService
from backend.app.services.evidence_agent import EvidenceAgent

def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ["ok", "degraded"]
    assert data["database"] == "connected"
    assert data["model"] in ["loaded", "not_found"]

def test_case_creation(client):
    payload = {
        "case_id": "CASE-TEST-001",
        "transaction_amount": 15000.0,
        "dispute_reason": "item_not_received",
        "delivery_confirmed": True,
        "customer_order_count": 5,
        "evidence_items_available": 4,
        "evidence_items_missing": 1,
    }
    response = client.post("/cases", json=payload)
    assert response.status_code == 201
    data = response.json()
    assert data["case_id"] == "CASE-TEST-001"
    assert data["transaction_amount"] == 15000.0
    assert data["status"] == "CREATED"
    assert data["evidence_completeness"] == pytest.approx(0.8, 0.01)

def test_invalid_case_rejection(client):
    # Invalid negative transaction amount
    invalid_payload = {
        "transaction_amount": -500.0,
    }
    response = client.post("/cases", json=invalid_payload)
    assert response.status_code == 422

    # Invalid refund_amount_ratio > 1.0
    invalid_ratio_payload = {
        "transaction_amount": 1000.0,
        "refund_amount_ratio": 1.5,
    }
    response = client.post("/cases", json=invalid_ratio_payload)
    assert response.status_code == 422

def test_predict_endpoint(client):
    payload = {
        "case_id": "CASE-PRED-001",
        "transaction_amount": 25000.0,
        "dispute_reason": "item_not_received",
        "delivery_confirmed": True,
        "evidence_items_available": 5,
        "evidence_items_missing": 1,
    }
    create_res = client.post("/cases", json=payload)
    assert create_res.status_code == 201

    pred_res = client.post("/cases/CASE-PRED-001/predict")
    assert pred_res.status_code == 200
    data = pred_res.json()
    assert data["case_id"] == "CASE-PRED-001"
    assert 0.0 <= data["contest_probability"] <= 1.0
    assert data["risk_level"] in ["HIGH", "MEDIUM", "LOW"]
    assert isinstance(data["reason_codes"], list)
    assert len(data["reason_codes"]) > 0

    # Verify status updated on case
    case_res = client.get("/cases/CASE-PRED-001")
    assert case_res.json()["status"] == "PREDICTED"

def test_decision_endpoint(client):
    payload = {
        "case_id": "CASE-DEC-001",
        "transaction_amount": 30000.0,
        "delivery_confirmed": True,
        "evidence_items_available": 4,
        "evidence_items_missing": 2,
    }
    client.post("/cases", json=payload)

    dec_res = client.post("/cases/CASE-DEC-001/decide")
    assert dec_res.status_code == 200
    data = dec_res.json()
    assert data["case_id"] == "CASE-DEC-001"
    assert data["action"] in ["ACCEPT", "ESCALATE", "CONTEST"]
    assert isinstance(data["expected_recovery"], (int, float))
    assert isinstance(data["expected_value"], (int, float))
    assert isinstance(data["reasoning"], list)

    # Verify status updated on case
    case_res = client.get("/cases/CASE-DEC-001")
    assert case_res.json()["status"] == "DECIDED"

def test_accept_decision(client):
    # Low probability and low transaction amount -> Net expected value < min_net_value
    payload = {
        "case_id": "CASE-ACCEPT-001",
        "transaction_amount": 300.0,
        "delivery_confirmed": False,
        "previous_refunds": 5,
        "previous_disputes": 3,
        "payment_failures": 4,
        "evidence_items_available": 4,
        "evidence_items_missing": 2,
    }
    client.post("/cases", json=payload)
    dec_res = client.post("/cases/CASE-ACCEPT-001/decide")
    assert dec_res.status_code == 200
    assert dec_res.json()["action"] == "ACCEPT"

def test_escalate_decision_guardrail(client):
    # Evidence completeness < 0.50 triggers guardrail ESCALATE
    payload = {
        "case_id": "CASE-ESC-001",
        "transaction_amount": 50000.0,
        "delivery_confirmed": True,
        "evidence_items_available": 1,
        "evidence_items_missing": 5,
    }
    client.post("/cases", json=payload)
    dec_res = client.post("/cases/CASE-ESC-001/decide")
    assert dec_res.status_code == 200
    data = dec_res.json()
    assert data["action"] == "ESCALATE"
    assert data["guardrail_triggered"] is True

def test_contest_decision(client):
    # High amount, strong delivery confirmation, high evidence completeness -> CONTEST
    payload = {
        "case_id": "CASE-CONTEST-001",
        "transaction_amount": 85000.0,
        "dispute_reason": "item_not_received",
        "delivery_confirmed": True,
        "customer_order_count": 12,
        "previous_refunds": 0,
        "previous_disputes": 0,
        "evidence_items_available": 5,
        "evidence_items_missing": 1,
    }
    client.post("/cases", json=payload)
    dec_res = client.post("/cases/CASE-CONTEST-001/decide")
    assert dec_res.status_code == 200
    data = dec_res.json()
    assert data["action"] == "CONTEST"
    assert data["expected_value"] > 0
    assert data["guardrail_triggered"] is False

def test_case_retrieval(client):
    payload = {
        "case_id": "CASE-RET-001",
        "transaction_amount": 20000.0,
        "delivery_confirmed": True,
    }
    client.post("/cases", json=payload)
    client.post("/cases/CASE-RET-001/predict")
    client.post("/cases/CASE-RET-001/decide")

    res = client.get("/cases/CASE-RET-001")
    assert res.status_code == 200
    data = res.json()
    assert data["case_id"] == "CASE-RET-001"
    assert data["latest_prediction"] is not None
    assert data["latest_decision"] is not None

    # Test 404 for non-existent case
    notFound = client.get("/cases/NON-EXISTENT-CASE")
    assert notFound.status_code == 404

def test_audit_retrieval(client):
    case_id = "CASE-AUDIT-001"
    payload = {
        "case_id": case_id,
        "transaction_amount": 10000.0,
        "delivery_confirmed": True,
    }
    client.post("/cases", json=payload)
    client.post(f"/cases/{case_id}/predict")
    client.post(f"/cases/{case_id}/decide")

    res = client.get(f"/cases/{case_id}/audit")
    assert res.status_code == 200
    events = res.json()
    event_types = [e["event_type"] for e in events]
    assert "CASE_CREATED" in event_types
    assert "PREDICTION_GENERATED" in event_types
    assert "DECISION_MADE" in event_types

def test_database_persistence(client, db_session):
    case_id = "CASE-PERSIST-001"
    payload = {
        "case_id": case_id,
        "transaction_amount": 12500.0,
        "dispute_reason": "duplicate",
    }
    client.post("/cases", json=payload)
    client.post(f"/cases/{case_id}/predict")
    client.post(f"/cases/{case_id}/decide")

    # Direct query via SQLAlchemy db_session
    case_in_db = db_session.query(Case).filter(Case.case_id == case_id).first()
    assert case_in_db is not None
    assert case_in_db.transaction_amount == 12500.0
    assert case_in_db.status == "DECIDED"

    predictions = db_session.query(Prediction).filter(Prediction.case_id == case_id).all()
    assert len(predictions) >= 1

    decisions = db_session.query(Decision).filter(Decision.case_id == case_id).all()
    assert len(decisions) >= 1

    audit_records = db_session.query(AuditEvent).filter(AuditEvent.case_id == case_id).all()
    assert len(audit_records) >= 3

def test_ml_integration_failure_handling(client, db_session):
    case_id = "CASE-FAIL-001"
    payload = {
        "case_id": case_id,
        "transaction_amount": 10000.0,
    }
    client.post("/cases", json=payload)

    with patch("backend.app.services.risk_service.predict_case", side_effect=Exception("Model file corrupted")):
        response = client.post(f"/cases/{case_id}/predict")
        assert response.status_code == 500
        assert "Model file corrupted" in response.json()["detail"]

    # Verify PREDICTION_FAILED audit event was recorded
    fail_events = (
        db_session.query(AuditEvent)
        .filter(AuditEvent.case_id == case_id, AuditEvent.event_type == "PREDICTION_FAILED")
        .all()
    )
    assert len(fail_events) == 1

# --- PHASE 3 DECISION AGENT TESTS ---

def test_invalid_llm_action_rejected_by_guardrail(client, db_session):
    case_id = "CASE-LLM-GUARDRAIL"
    payload = {
        "case_id": case_id,
        "transaction_amount": 75000.0,
        "delivery_confirmed": True,
        "evidence_items_available": 1,
        "evidence_items_missing": 5,  # 1/6 = 16.7%
    }
    client.post("/cases", json=payload)
    client.post(f"/cases/{case_id}/predict")

    case_obj = CaseService.get_case(db_session, case_id)
    simulated_llm_output = {
        "recommended_action": "CONTEST",
        "rationale": "High transaction value, merchant should attempt to fight it.",
        "concerns": [],
    }
    decision = DecisionService.run_decision(
        db=db_session,
        case=case_obj,
        llm_response_override=simulated_llm_output,
    )

    assert decision.action == "ESCALATE"
    assert decision.guardrail_triggered is True
    assert any("Guardrail Override" in r for r in decision.reasoning)

    audit = (
        db_session.query(AuditEvent)
        .filter(AuditEvent.case_id == case_id, AuditEvent.event_type == "DECISION_MADE")
        .order_by(AuditEvent.event_timestamp.desc())
        .first()
    )
    assert audit.metadata_payload.get("fallback_used") is True
    assert audit.metadata_payload.get("llm_used") is True

def test_llm_malformed_response_safe_fallback(client, db_session):
    case_id = "CASE-LLM-MALFORMED"
    payload = {
        "case_id": case_id,
        "transaction_amount": 50000.0,
        "delivery_confirmed": True,
        "evidence_items_available": 4,
        "evidence_items_missing": 1,
    }
    client.post("/cases", json=payload)
    client.post(f"/cases/{case_id}/predict")

    case_obj = CaseService.get_case(db_session, case_id)
    malformed_llm_output = {
        "recommended_action": "MAYBE_FIGHT_OR_NOT",
        "rationale": "Unrecognized action output",
    }
    decision = DecisionService.run_decision(
        db=db_session,
        case=case_obj,
        llm_response_override=malformed_llm_output,
    )

    assert decision.action in ["ACCEPT", "ESCALATE", "CONTEST"]
    assert any("Fallback" in r for r in decision.reasoning)

def test_llm_api_failure_safe_fallback(client, db_session):
    case_id = "CASE-LLM-FAIL"
    payload = {
        "case_id": case_id,
        "transaction_amount": 25000.0,
        "delivery_confirmed": True,
        "evidence_items_available": 4,
        "evidence_items_missing": 1,
    }
    client.post("/cases", json=payload)
    client.post(f"/cases/{case_id}/predict")

    case_obj = CaseService.get_case(db_session, case_id)
    with patch("backend.app.services.decision_agent.DecisionAgent._call_llm", side_effect=Exception("API connection timeout")):
        decision = DecisionService.run_decision(db=db_session, case=case_obj)
        assert decision.action in ["ACCEPT", "ESCALATE", "CONTEST"]

def test_reason_codes_preserved_from_risk_engine(client, db_session):
    case_id = "CASE-REASON-VERIFY"
    payload = {
        "case_id": case_id,
        "transaction_amount": 60000.0,
        "dispute_reason": "item_not_received",
        "delivery_confirmed": True,
        "previous_refunds": 0,
        "previous_disputes": 0,
        "evidence_items_available": 5,
        "evidence_items_missing": 1,
    }
    client.post("/cases", json=payload)
    pred_res = client.post(f"/cases/{case_id}/predict")
    assert pred_res.status_code == 200
    model_reason_codes = pred_res.json()["reason_codes"]

    case_obj = CaseService.get_case(db_session, case_id)
    pred_obj = case_obj.predictions[0]
    assert pred_obj.reason_codes == model_reason_codes
    assert "delivery_confirmed" in pred_obj.reason_codes

# --- PHASE 4 EVIDENCE AGENT TESTS ---

def test_contest_case_starts_evidence_workflow(client):
    case_id = "CASE-EVID-CONTEST"
    payload = {
        "case_id": case_id,
        "transaction_amount": 80000.0,
        "dispute_reason": "item_not_received",
        "delivery_confirmed": True,
        "customer_order_count": 8,
        "previous_refunds": 0,
        "previous_disputes": 0,
        "communication_count": 3,
        "evidence_items_available": 5,
        "evidence_items_missing": 1,
    }
    client.post("/cases", json=payload)
    client.post(f"/cases/{case_id}/predict")
    dec_res = client.post(f"/cases/{case_id}/decide")
    assert dec_res.status_code == 200
    assert dec_res.json()["action"] == "CONTEST"

    # Execute Evidence Workflow
    evid_res = client.post(f"/cases/{case_id}/evidence")
    assert evid_res.status_code == 200
    data = evid_res.json()
    assert data["case_id"] == case_id
    assert data["status"] == "READY_FOR_REVIEW"
    assert data["requires_human_review"] is True
    assert 0.0 <= data["evidence_completeness"] <= 1.0
    assert len(data["evidence_items"]) == 6
    assert isinstance(data["response_draft"]["statement"], str)
    assert len(data["response_draft"]["claims"]) > 0

def test_accept_case_rejects_evidence_workflow(client):
    case_id = "CASE-EVID-ACCEPT"
    payload = {
        "case_id": case_id,
        "transaction_amount": 300.0,
        "dispute_reason": "unauthorized",
        "delivery_confirmed": False,
        "previous_refunds": 4,
        "previous_disputes": 2,
        "evidence_items_available": 4,
        "evidence_items_missing": 2,
    }
    client.post("/cases", json=payload)
    client.post(f"/cases/{case_id}/predict")
    dec_res = client.post(f"/cases/{case_id}/decide")
    assert dec_res.json()["action"] == "ACCEPT"

    # Attempting to run evidence workflow on ACCEPT must return 400 Bad Request
    evid_res = client.post(f"/cases/{case_id}/evidence")
    assert evid_res.status_code == 400
    assert "Only CONTEST decisions qualify" in evid_res.json()["detail"]

def test_escalate_case_rejects_evidence_workflow(client):
    case_id = "CASE-EVID-ESC"
    payload = {
        "case_id": case_id,
        "transaction_amount": 60000.0,
        "delivery_confirmed": True,
        "evidence_items_available": 1,
        "evidence_items_missing": 5,
    }
    client.post("/cases", json=payload)
    client.post(f"/cases/{case_id}/predict")
    dec_res = client.post(f"/cases/{case_id}/decide")
    assert dec_res.json()["action"] == "ESCALATE"

    # Attempting to run evidence workflow on ESCALATE must return 400 Bad Request
    evid_res = client.post(f"/cases/{case_id}/evidence")
    assert evid_res.status_code == 400
    assert "Only CONTEST decisions qualify" in evid_res.json()["detail"]

def test_missing_evidence_detected(client, db_session):
    case_id = "CASE-EVID-MISSING"
    payload = {
        "case_id": case_id,
        "transaction_amount": 70000.0,
        "delivery_confirmed": False,  # Missing delivery proof
        "communication_count": 0,     # Missing communication logs
        "customer_order_count": 5,
    }
    client.post("/cases", json=payload)
    case_obj = CaseService.get_case(db_session, case_id)
    assert case_obj is not None

    items = EvidenceAgent.collect_evidence(case_obj)
    missing, _ = EvidenceAgent.validate_evidence(items, case_obj)
    assert "delivery_proof" in missing
    assert "customer_communication" in missing

def test_conflicting_evidence_detected():
    case_id = "CASE-EVID-CONFLICT"
    case_obj = Case(
        case_id=case_id,
        transaction_amount=50000.0,
        dispute_reason="item_not_received",
        delivery_confirmed=True,
        delivery_delay_days=95.0,     # Conflicting delivery delay
        refund_amount_ratio=0.95,     # Conflicting refund ratio
        previous_refunds=4,
        payment_failures=5,           # Conflicting payment retries
        communication_count=0,
    )
    items = EvidenceAgent.collect_evidence(case_obj)
    _, conflicts = EvidenceAgent.validate_evidence(items, case_obj)
    assert len(conflicts) >= 2
    assert any("fulfillment delay is excessively long" in c for c in conflicts)

def test_response_claims_traceability(client):
    case_id = "CASE-TRACEABILITY"
    payload = {
        "case_id": case_id,
        "transaction_amount": 85000.0,
        "dispute_reason": "item_not_received",
        "delivery_confirmed": True,
        "customer_order_count": 10,
        "communication_count": 4,
        "evidence_items_available": 5,
        "evidence_items_missing": 1,
    }
    client.post("/cases", json=payload)
    client.post(f"/cases/{case_id}/predict")
    client.post(f"/cases/{case_id}/decide")
    evid_res = client.post(f"/cases/{case_id}/evidence")
    assert evid_res.status_code == 200
    data = evid_res.json()

    claims = data["response_draft"]["claims"]
    evidence_ids = {item["evidence_id"] for item in data["evidence_items"] if item["status"] == "AVAILABLE"}

    # Assert every single claim cites a valid, available evidence_id
    for claim_obj in claims:
        assert claim_obj["source_evidence_id"] in evidence_ids
        assert len(claim_obj["claim"]) > 10

def test_evidence_persistence_and_retrieval(client, db_session):
    case_id = "CASE-EVID-PERSIST"
    payload = {
        "case_id": case_id,
        "transaction_amount": 90000.0,
        "dispute_reason": "item_not_received",
        "delivery_confirmed": True,
        "customer_order_count": 12,
        "evidence_items_available": 5,
        "evidence_items_missing": 1,
    }
    client.post("/cases", json=payload)
    client.post(f"/cases/{case_id}/predict")
    client.post(f"/cases/{case_id}/decide")
    client.post(f"/cases/{case_id}/evidence")

    # Retrieve via GET /cases/{id}/evidence
    get_res = client.get(f"/cases/{case_id}/evidence")
    assert get_res.status_code == 200
    assert get_res.json()["case_id"] == case_id
    assert get_res.json()["status"] == "READY_FOR_REVIEW"

    # Retrieve via GET /cases/{id} and verify latest_evidence_packet attached
    case_res = client.get(f"/cases/{case_id}")
    assert case_res.status_code == 200
    assert case_res.json()["latest_evidence_packet"] is not None
    assert case_res.json()["status"] == "EVIDENCE_READY"

    # Verify audit event EVIDENCE_PACKET_CREATED was persisted
    audit_res = client.get(f"/cases/{case_id}/audit")
    assert audit_res.status_code == 200
    events = [e["event_type"] for e in audit_res.json()]
    assert "EVIDENCE_PACKET_CREATED" in events

# --- PHASE 5 FULL SYSTEM INTEGRATION & FRONTEND TESTS ---

def test_list_cases_endpoint(client):
    res = client.get("/cases?limit=10&offset=0")
    assert res.status_code == 200
    data = res.json()
    assert isinstance(data, list)

def test_preset_samples_endpoint(client):
    res = client.get("/cases/presets/samples")
    assert res.status_code == 200
    presets = res.json()
    assert len(presets) >= 3
    preset_ids = [p["preset_id"] for p in presets]
    assert "contest_high_value" in preset_ids
    assert "accept_low_value" in preset_ids
    assert "escalate_low_evidence" in preset_ids

def test_frontend_static_serving(client):
    res = client.get("/")
    assert res.status_code == 200
    assert "RazorPay Sentinel" in res.text

