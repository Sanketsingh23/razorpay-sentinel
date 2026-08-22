from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, ConfigDict

class CaseCreate(BaseModel):
    case_id: Optional[str] = None
    transaction_amount: float = Field(..., gt=0, description="Dispute transaction amount in INR")
    dispute_reason: str = Field(default="item_not_received", description="Dispute reason category")
    delivery_confirmed: bool = False
    customer_order_count: int = Field(default=1, ge=0)
    customer_avg_order_value: Optional[float] = Field(default=None, ge=0.0)
    previous_refunds: int = Field(default=0, ge=0)
    previous_disputes: int = Field(default=0, ge=0)
    delivery_delay_days: float = Field(default=0.0, ge=0.0)
    dispute_delay_days: float = Field(default=0.0, ge=0.0)
    communication_count: int = Field(default=0, ge=0)
    refund_amount_ratio: float = Field(default=0.0, ge=0.0, le=1.0)
    payment_failures: int = Field(default=0, ge=0)
    evidence_items_available: int = Field(default=0, ge=0)
    evidence_items_missing: int = Field(default=0, ge=0)
    evidence_completeness: Optional[float] = Field(default=None, ge=0.0, le=1.0)

class PredictionResponse(BaseModel):
    case_id: str
    contest_probability: float
    risk_level: str
    reason_codes: List[str]
    positive_factors: List[str] = []
    negative_factors: List[str] = []
    model_version: str
    prediction_timestamp: datetime

    model_config = ConfigDict(from_attributes=True)

class DecisionResponse(BaseModel):
    case_id: str
    action: str
    expected_recovery: float
    expected_value: float
    evidence_completeness: float
    guardrail_triggered: bool
    reasoning: List[str] = []
    policy_version: str
    decision_timestamp: datetime

    model_config = ConfigDict(from_attributes=True)

class EvidenceItem(BaseModel):
    evidence_id: str
    type: str
    source: str
    status: str
    summary: str
    relevance: str

    model_config = ConfigDict(from_attributes=True)

class ResponseClaim(BaseModel):
    claim: str
    source_evidence_id: str

class DisputeResponseDraft(BaseModel):
    statement: str
    claims: List[ResponseClaim] = []

class EvidencePacketResponse(BaseModel):
    id: int
    case_id: str
    status: str
    evidence_completeness: float
    evidence_items: List[EvidenceItem] = []
    missing_evidence: List[str] = []
    conflicting_evidence: List[str] = []
    response_draft: DisputeResponseDraft
    requires_human_review: bool
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)

class AuditEventResponse(BaseModel):
    id: int
    case_id: Optional[str]
    event_type: str
    event_timestamp: datetime
    metadata_payload: Dict[str, Any] = {}

    model_config = ConfigDict(from_attributes=True)

class CaseResponse(BaseModel):
    case_id: str
    transaction_amount: float
    dispute_reason: str
    delivery_confirmed: bool
    customer_order_count: int
    customer_avg_order_value: float
    previous_refunds: int
    previous_disputes: int
    delivery_delay_days: float
    dispute_delay_days: float
    communication_count: int
    refund_amount_ratio: float
    payment_failures: int
    evidence_items_available: int
    evidence_items_missing: int
    evidence_completeness: float
    status: str
    created_at: datetime
    updated_at: datetime
    latest_prediction: Optional[PredictionResponse] = None
    latest_decision: Optional[DecisionResponse] = None
    latest_evidence_packet: Optional[EvidencePacketResponse] = None

    model_config = ConfigDict(from_attributes=True)

class HealthResponse(BaseModel):
    status: str
    database: str
    model: str
