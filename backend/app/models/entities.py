from datetime import datetime, timezone
from sqlalchemy import Column, String, Integer, Float, Boolean, DateTime, ForeignKey, JSON, Text
from sqlalchemy.orm import relationship
from backend.app.core.database import Base

def utc_now():
    return datetime.now(timezone.utc)

class Case(Base):
    __tablename__ = "cases"

    case_id = Column(String(64), primary_key=True, index=True)
    transaction_amount = Column(Float, nullable=False)
    dispute_reason = Column(String(64), default="item_not_received")
    delivery_confirmed = Column(Boolean, default=False)
    customer_order_count = Column(Integer, default=1)
    customer_avg_order_value = Column(Float, default=0.0)
    previous_refunds = Column(Integer, default=0)
    previous_disputes = Column(Integer, default=0)
    delivery_delay_days = Column(Float, default=0.0)
    dispute_delay_days = Column(Float, default=0.0)
    communication_count = Column(Integer, default=0)
    refund_amount_ratio = Column(Float, default=0.0)
    payment_failures = Column(Integer, default=0)
    evidence_items_available = Column(Integer, default=0)
    evidence_items_missing = Column(Integer, default=0)
    evidence_completeness = Column(Float, default=0.0)
    status = Column(String(32), default="CREATED")
    created_at = Column(DateTime(timezone=True), default=utc_now)
    updated_at = Column(DateTime(timezone=True), default=utc_now, onupdate=utc_now)

    predictions = relationship("Prediction", back_populates="case", cascade="all, delete-orphan", order_by="desc(Prediction.prediction_timestamp)")
    decisions = relationship("Decision", back_populates="case", cascade="all, delete-orphan", order_by="desc(Decision.decision_timestamp)")
    evidence_packets = relationship("EvidencePacket", back_populates="case", cascade="all, delete-orphan", order_by="desc(EvidencePacket.created_at)")
    audit_events = relationship("AuditEvent", back_populates="case", cascade="all, delete-orphan", order_by="desc(AuditEvent.event_timestamp)")

class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    case_id = Column(String(64), ForeignKey("cases.case_id", ondelete="CASCADE"), nullable=False, index=True)
    model_version = Column(String(64), nullable=False)
    contest_probability = Column(Float, nullable=False)
    risk_level = Column(String(16), nullable=False)
    reason_codes = Column(JSON, default=list)
    positive_factors = Column(JSON, default=list)
    negative_factors = Column(JSON, default=list)
    prediction_timestamp = Column(DateTime(timezone=True), default=utc_now)

    case = relationship("Case", back_populates="predictions")

class Decision(Base):
    __tablename__ = "decisions"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    case_id = Column(String(64), ForeignKey("cases.case_id", ondelete="CASCADE"), nullable=False, index=True)
    action = Column(String(16), nullable=False)
    expected_recovery = Column(Float, nullable=False)
    expected_value = Column(Float, nullable=False)
    evidence_completeness = Column(Float, nullable=False)
    guardrail_triggered = Column(Boolean, default=False)
    reasoning = Column(JSON, default=list)
    policy_version = Column(String(32), default="v1.0")
    decision_timestamp = Column(DateTime(timezone=True), default=utc_now)

    case = relationship("Case", back_populates="decisions")

class EvidencePacket(Base):
    __tablename__ = "evidence_packets"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    case_id = Column(String(64), ForeignKey("cases.case_id", ondelete="CASCADE"), nullable=False, index=True)
    status = Column(String(32), default="READY_FOR_REVIEW")
    evidence_completeness = Column(Float, nullable=False)
    evidence_items = Column(JSON, default=list)
    missing_evidence = Column(JSON, default=list)
    conflicting_evidence = Column(JSON, default=list)
    response_draft = Column(JSON, default=dict)
    requires_human_review = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), default=utc_now)

    case = relationship("Case", back_populates="evidence_packets")

class AuditEvent(Base):
    __tablename__ = "audit_events"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    case_id = Column(String(64), ForeignKey("cases.case_id", ondelete="CASCADE"), nullable=True, index=True)
    event_type = Column(String(64), nullable=False, index=True)
    event_timestamp = Column(DateTime(timezone=True), default=utc_now)
    metadata_payload = Column(JSON, default=dict)

    case = relationship("Case", back_populates="audit_events")
