import os
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session
from sqlalchemy import text
from backend.app.core.database import get_db
from backend.app.core.config import settings
from backend.app.schemas.case import (
    CaseCreate,
    CaseResponse,
    PredictionResponse,
    DecisionResponse,
    EvidencePacketResponse,
    AuditEventResponse,
    HealthResponse,
)
from backend.app.services.case_service import CaseService
from backend.app.services.risk_service import RiskService
from backend.app.services.decision_service import DecisionService
from backend.app.services.evidence_agent import EvidenceAgent
from backend.app.services.audit_service import AuditService

router = APIRouter()

@router.get("/health", response_model=HealthResponse)
def health_check(db: Session = Depends(get_db)):
    db_status = "connected"
    try:
        db.execute(text("SELECT 1"))
    except Exception:
        db_status = "disconnected"

    model_status = "loaded"
    model_file = os.path.join(settings.MODELS_DIR, "risk_model.joblib")
    if not os.path.exists(model_file):
        model_status = "not_found"

    return HealthResponse(
        status="ok" if db_status == "connected" and model_status == "loaded" else "degraded",
        database=db_status,
        model=model_status,
    )

@router.get("/cases", response_model=List[CaseResponse])
def list_cases(
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
):
    cases = CaseService.list_cases(db, limit=limit, offset=offset)
    return [CaseService.format_case_response(c) for c in cases]

@router.get("/cases/presets/samples")
def get_preset_samples():
    """Return pre-configured dispute case profiles for 1-click frontend demonstrations."""
    return [
        {
            "preset_id": "contest_high_value",
            "title": "High-Value Contestable Dispute (INR 85,000)",
            "description": "Item not received, delivery confirmed, clean customer history, high evidence completeness.",
            "data": {
                "transaction_amount": 85000.0,
                "dispute_reason": "item_not_received",
                "delivery_confirmed": True,
                "customer_order_count": 12,
                "customer_avg_order_value": 70000.0,
                "previous_refunds": 0,
                "previous_disputes": 0,
                "delivery_delay_days": 1.0,
                "dispute_delay_days": 4.0,
                "communication_count": 3,
                "refund_amount_ratio": 0.0,
                "payment_failures": 0,
                "evidence_items_available": 5,
                "evidence_items_missing": 1,
            },
        },
        {
            "preset_id": "accept_low_value",
            "title": "Low-Value Unviable Dispute (INR 400)",
            "description": "Unauthorized transaction, unconfirmed delivery, low expected recovery below operational cost.",
            "data": {
                "transaction_amount": 400.0,
                "dispute_reason": "unauthorized",
                "delivery_confirmed": False,
                "customer_order_count": 1,
                "customer_avg_order_value": 400.0,
                "previous_refunds": 4,
                "previous_disputes": 2,
                "delivery_delay_days": 0.0,
                "dispute_delay_days": 15.0,
                "communication_count": 0,
                "refund_amount_ratio": 0.85,
                "payment_failures": 3,
                "evidence_items_available": 4,
                "evidence_items_missing": 2,
            },
        },
        {
            "preset_id": "escalate_low_evidence",
            "title": "Deficient Evidence Guardrail Trigger (INR 60,000)",
            "description": "High win probability profile, but evidence completeness < 50% triggers mandatory guardrail escalation.",
            "data": {
                "transaction_amount": 60000.0,
                "dispute_reason": "item_not_received",
                "delivery_confirmed": True,
                "customer_order_count": 6,
                "customer_avg_order_value": 55000.0,
                "previous_refunds": 0,
                "previous_disputes": 0,
                "delivery_delay_days": 2.0,
                "dispute_delay_days": 3.0,
                "communication_count": 0,
                "refund_amount_ratio": 0.0,
                "payment_failures": 0,
                "evidence_items_available": 1,
                "evidence_items_missing": 5,
            },
        },
    ]

@router.post("/cases", response_model=CaseResponse, status_code=status.HTTP_201_CREATED)
def create_case(case_in: CaseCreate, db: Session = Depends(get_db)):
    if case_in.case_id:
        existing = CaseService.get_case(db, case_in.case_id)
        if existing:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Case with ID '{case_in.case_id}' already exists.",
            )
    case = CaseService.create_case(db, case_in)
    return CaseService.format_case_response(case)

@router.get("/cases/{case_id}", response_model=CaseResponse)
def get_case(case_id: str, db: Session = Depends(get_db)):
    case = CaseService.get_case(db, case_id)
    if not case:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Case '{case_id}' not found.",
        )
    return CaseService.format_case_response(case)

@router.post("/cases/{case_id}/predict", response_model=PredictionResponse)
def predict_case_endpoint(case_id: str, db: Session = Depends(get_db)):
    case = CaseService.get_case(db, case_id)
    if not case:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Case '{case_id}' not found.",
        )
    try:
        prediction = RiskService.run_prediction(db, case)
        return prediction
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )

@router.post("/cases/{case_id}/decide", response_model=DecisionResponse)
def decide_case_endpoint(case_id: str, db: Session = Depends(get_db)):
    case = CaseService.get_case(db, case_id)
    if not case:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Case '{case_id}' not found.",
        )
    try:
        decision = DecisionService.run_decision(db, case)
        return decision
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )

@router.post("/cases/{case_id}/evidence", response_model=EvidencePacketResponse)
def generate_evidence_endpoint(case_id: str, db: Session = Depends(get_db)):
    case = CaseService.get_case(db, case_id)
    if not case:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Case '{case_id}' not found.",
        )

    latest_decision = case.decisions[0] if case.decisions else None
    if not latest_decision:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Case '{case_id}' has not been decided yet. Run /decide before generating evidence.",
        )

    if latest_decision.action != "CONTEST":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Evidence workflow cannot be initiated for decision action '{latest_decision.action}'. Only CONTEST decisions qualify.",
        )

    latest_prediction = case.predictions[0] if case.predictions else None

    try:
        packet = EvidenceAgent.build_packet(
            case=case,
            prediction=latest_prediction,
            decision=latest_decision,
        )
        db.add(packet)
        case.status = "EVIDENCE_READY"
        db.commit()
        db.refresh(packet)

        AuditService.log_event(
            db=db,
            event_type="EVIDENCE_PACKET_CREATED",
            case_id=case.case_id,
            metadata_payload={
                "status": packet.status,
                "evidence_completeness": packet.evidence_completeness,
                "items_count": len(packet.evidence_items),
                "missing_count": len(packet.missing_evidence),
                "conflicts_count": len(packet.conflicting_evidence),
            },
        )
        return packet
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )

@router.get("/cases/{case_id}/evidence", response_model=EvidencePacketResponse)
def get_evidence_endpoint(case_id: str, db: Session = Depends(get_db)):
    case = CaseService.get_case(db, case_id)
    if not case:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Case '{case_id}' not found.",
        )

    latest_packet = case.evidence_packets[0] if case.evidence_packets else None
    if not latest_packet:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No evidence packet found for case '{case_id}'.",
        )
    return latest_packet

@router.get("/cases/{case_id}/audit", response_model=List[AuditEventResponse])
def get_case_audit(case_id: str, db: Session = Depends(get_db)):
    case = CaseService.get_case(db, case_id)
    if not case:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Case '{case_id}' not found.",
        )
    events = AuditService.get_events_for_case(db, case_id)
    return events
