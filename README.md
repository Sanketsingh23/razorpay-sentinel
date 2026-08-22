# RazorPay Sentinel

> **Autonomous Dispute Risk Intelligence & Decision Orchestration System**  
> An intelligent risk management platform that predicts dispute win probabilities, evaluates net financial recovery economics, enforces deterministic policy guardrails, and orchestrates end-to-end evidence response packages.

[![Tests](https://img.shields.io/badge/Tests-26%2F26%20Passed-success?style=flat-square)](https://github.com/Sanketsingh23/razorpay-sentinel)
[![Python](https://img.shields.io/badge/Python-3.9+-blue?style=flat-square&logo=python)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-1.0.0-009688?style=flat-square&logo=fastapi)](https://fastapi.tiangolo.com/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-16-336791?style=flat-square&logo=postgresql)](https://www.postgresql.org/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=flat-square&logo=docker)](https://www.docker.com/)
[![Render](https://img.shields.io/badge/Deployment-Live%20on%20Render-46E3B7?style=flat-square&logo=render)](https://razorpay-sentinel.onrender.com)

**Live Deployed Application:** [https://razorpay-sentinel.onrender.com](https://razorpay-sentinel.onrender.com)  
**Interactive API Documentation:** [https://razorpay-sentinel.onrender.com/docs](https://razorpay-sentinel.onrender.com/docs)  
**GitHub Repository:** [https://github.com/Sanketsingh23/razorpay-sentinel](https://github.com/Sanketsingh23/razorpay-sentinel)

---

## 1. Problem Statement

In online commerce, payment chargebacks and disputes create a massive operational and financial drain on merchants:

1. **Uninformed Concessions:** Merchants frequently concede winnable, high-value disputes due to lack of timely triage and evidence assembly bandwidth.
2. **Wasted Operational Costs:** Contesting a dispute incurs fixed partner handling fees and merchant labor (~₹500+). Contesting low-probability or low-value disputes results in negative net financial recovery.
3. **The "Probability Fallacy":** Simply predicting dispute success ($P(\text{success})$) is insufficient for business operations. A dispute with an 80% win rate on a ₹200 transaction yields an expected recovery of ₹160, which is unviable against a ₹500 operational contesting cost.
4. **Evidence Deficits:** Contesting high-value claims without essential carrier proof or order confirmation leads to guaranteed losses and penalties.
5. **Audit Gaps:** Financial decisions require strict traceability, immutable change logs, and factual claim verification without generative AI hallucination.

---

## 2. Solution Overview

**RazorPay Sentinel** solves dispute triage by combining calibrated supervised machine learning with deterministic financial policy governance:

```text
Dispute Case ──► ML Risk Engine ──► Decision Agent ──► Evidence Agent (if CONTEST) ──► Audit Trail
                  (P(success) +       (Economics +        (6 Evidence Items +             (Immutable
                   Reason Codes)       Guardrails)         Traceable Draft)                Forensics)
```

### Core Architectural Principle: ML Predicts, Policy Governs
The machine learning model **never makes the authoritative business decision**. It produces a well-calibrated win probability $P(\text{success})$ and signed mathematical feature contributions. The separate, deterministic **Decision Agent** applies financial recovery formulas, operational expense thresholds, and evidence completeness guardrails to select the final action: **ACCEPT**, **ESCALATE**, or **CONTEST**.

---

## 3. Key Features

- **Calibrated Contest Probability:** Supervised Logistic Regression model optimized for reliable probability estimates (Brier Score `0.1195`, Expected Calibration Error `0.0198`).
- **Mathematical Reason Codes:** Explains model predictions using signed feature contributions ($c_j = \beta_j \cdot z_j$) with zero LLM hallucination.
- **Economic Decision Policy:** Calculates Expected Recovery and Net Expected Value after deducting fixed operational contesting costs.
- **Evidence Completeness Guardrails:** Mandatory policy rule that forces `ESCALATE` if evidence completeness falls below 50%, regardless of model probability.
- **3-Way Authoritative Actions:**
  - `ACCEPT`: For unviable recovery amounts or unwinnable dispute profiles.
  - `ESCALATE`: For uncertain probability zones or deficient evidence profiles requiring manual investigation.
  - `CONTEST`: For high win probability, complete evidence, and positive net financial recovery.
- **Automated Evidence Assembly:** Collects and validates 6 structured evidence types (`AVAILABLE`, `MISSING`, `CONFLICTING`, `INVALID`).
- **Claim-Traceable Response Drafting:** Generates structured dispute contest statements where every factual claim explicitly cites an available source evidence ID (`[EVID-xxx]`).
- **Forensic Audit Timeline:** Immutable, chronological logging of all lifecycle transitions with full metadata payloads.
- **1-Click Demonstration Presets:** Pre-loaded UI profiles demonstrating high-value contests, low-value accepts, and guardrail escalations.
- **Dual-Mode Backend Architecture:** Fully persistent PostgreSQL backend with zero-dependency SQLite fallback for testing.
- **Production Cloud Deployment:** Containerized with Docker and live on Render.

---

## 4. System Architecture

```mermaid
flowchart TD
    subgraph Client ["Frontend & Client Layer"]
        UI["Interactive Dashboard (Vanilla HTML/CSS/JS)"]
        Swagger["FastAPI Swagger UI (/docs)"]
    end

    subgraph API ["FastAPI Backend Layer"]
        Routes["REST API Routes (/cases, /predict, /decide, /evidence, /audit)"]
        CaseService["Case Service"]
        AuditService["Audit Service"]
    end

    subgraph Intelligence ["Sentinel Risk & Decision Intelligence"]
        RiskEngine["Phase 1: ML Risk Engine (Logistic Regression)"]
        DecisionAgent["Phase 3: Decision Policy & Guardrails"]
        EvidenceAgent["Phase 4: Evidence & Dispute Response Agent"]
    end

    subgraph Persistence ["Persistence & Forensics"]
        DB[(PostgreSQL Database)]
        AuditLog[(Immutable Audit Log)]
    end

    UI -->|REST API Requests| Routes
    Swagger -->|API Requests| Routes
    Routes --> CaseService
    CaseService --> RiskEngine
    RiskEngine -->|P(success) + Reason Codes| DecisionAgent
    DecisionAgent -->|ACCEPT / ESCALATE / CONTEST| CaseService
    CaseService -->|If CONTEST| EvidenceAgent
    EvidenceAgent -->|Evidence Packet + Traceable Draft| CaseService
    CaseService --> DB
    CaseService --> AuditService
    AuditService --> AuditLog
    AuditLog --> DB
```

---

## 5. Decision Engine & Economic Policy

The **Decision Agent** (`ml/decision_policy.py` and `backend/app/services/decision_service.py`) operates on clear financial and risk formulas:

### 1. Mathematical Formulas
$$\text{Expected Recovery} = \text{Transaction Amount} \times P(\text{success})$$
$$\text{Net Expected Value} = \text{Expected Recovery} - \text{Operational Cost}$$

### 2. Policy Configuration Parameters
| Parameter | Value | Description |
| :--- | :---: | :--- |
| `contest_threshold` | **0.65** | Minimum $P(\text{success})$ required to consider contesting |
| `accept_threshold` | **0.30** | Maximum $P(\text{success})$ below which dispute is accepted |
| `evidence_threshold` | **0.50** | Minimum evidence completeness ratio (50%) required to contest |
| `contest_operational_cost` | **₹500.00** | Fixed handling and partner operational cost per contest |
| `min_net_value` | **₹100.00** | Minimum Net Expected Value required to justify a contest |

### 3. Decision Rules
1. **Evidence Guardrail (Top Priority):**  
   If $\text{Evidence Completeness} < 0.50 \implies \mathbf{ESCALATE}$ *(Guardrail triggered; prevents submitting incomplete claims).*
2. **Economic Viability Rule:**  
   If $\text{Net Expected Value} < \text{₹100} \implies \mathbf{ACCEPT}$ *(Prevents spending ₹500 to recover ₹300).*
3. **Low Probability Rule:**  
   If $P(\text{success}) < 0.30 \implies \mathbf{ACCEPT}$ *(Concedes unwinnable disputes).*
4. **Contest Rule:**  
   If $P(\text{success}) \ge 0.65$ **AND** $\text{Net Expected Value} \ge \text{₹100}$ **AND** $\text{Evidence Completeness} \ge 0.50 \implies \mathbf{CONTEST}$.
5. **Uncertainty Rule:**  
   If $0.30 \le P(\text{success}) < 0.65 \implies \mathbf{ESCALATE}$ *(Requires manual human review).*

---

## 6. ML Risk Engine

The Risk Engine (`ml/predict.py`) evaluates dispute characteristics to estimate $P(\text{success})$ and isolate key driving factors.

### Model Selection & Validation
Evaluated on a 10,000-sample synthetic dispute dataset (`data/disputes.csv`) split into Stratified Train (6,000) / Validation (2,000) / Test (2,000) sets:

| Evaluation Metric | Logistic Regression | XGBoost | Selected Model | Why Chosen |
| :--- | :---: | :---: | :---: | :--- |
| **Brier Score $\downarrow$** | **0.1195** | 0.1284 | **Logistic Regression** | Superior probability calibration |
| **Expected Calibration Error (ECE) $\downarrow$** | **0.0198** | 0.0358 | **Logistic Regression** | Probabilities reflect true win frequencies |
| **Cost-Sensitive Decision Loss (Val) $\downarrow$** | **₹2,767,449** | ₹3,035,470 | **Logistic Regression** | Minimizes false contests & missed recoveries |
| **ROC-AUC $\uparrow$** | **0.9019** | 0.8923 | **Logistic Regression** | Strong ranking ability across thresholds |
| **PR-AUC $\uparrow$** | **0.9065** | 0.8954 | **Logistic Regression** | Superior precision at high recall |

### Input Features (19 Total)
- **Financial Signals:** `transaction_amount`, `customer_avg_order_value`, `refund_amount_ratio`
- **Customer History:** `customer_order_count`, `previous_refunds`, `previous_disputes`, `payment_failures`, `communication_count`
- **Fulfillment & Timeline:** `delivery_confirmed`, `delivery_delay_days`, `dispute_delay_days`
- **Evidence Signals:** `evidence_items_available`, `evidence_items_missing`, `evidence_completeness`
- **Dispute Categories (One-Hot):** `reason_item_not_received`, `reason_unauthorized`, `reason_defective`, `reason_duplicate`, `reason_other`

### Mathematical Reason Code Generation
For each feature $j$, its signed standardized contribution is computed:
$$c_j = \beta_j \cdot z_j = \beta_j \cdot \left(\frac{x_j - \mu_j}{\sigma_j}\right)$$
Top positive contributions are surfaced as **Supporting Factors** (`+`), and top negative contributions as **Risk Factors** (`-`).

---

## 7. Evidence & Dispute Workflow

When the Decision Agent selects **CONTEST**, the Evidence Agent (`backend/app/services/evidence_agent.py`) executes:

1. **Extraction & Status Validation:** Evaluates 6 structured evidence types:
   - `EVID-001`: Delivery Proof (`case.delivery_confirmed`)
   - `EVID-002`: Order Confirmation (`case.transaction_amount`)
   - `EVID-003`: Customer Account History (`case.customer_order_count`)
   - `EVID-004`: Refund Records (`case.previous_refunds`, `case.refund_amount_ratio`)
   - `EVID-005`: Customer Communication (`case.communication_count`)
   - `EVID-006`: Payment Verification (`case.payment_failures`)
2. **Conflict Detection:** Identifies conflicting signals (e.g. delivery marked confirmed but fulfillment delay > 60 days).
3. **Deterministic Completeness:** Computes $N_{\text{valid}} / 6.0$.
4. **Claim Traceability:** Every factual sentence in the dispute response draft cites an explicit evidence source:
   ```text
   Carrier delivery confirmation on record with 1 days fulfillment delay. [EVID-001]
   Order transaction record for INR 85,000.00 under category 'item_not_received'. [EVID-002]
   Customer account history verified: 12 previous orders, 0 prior disputes. [EVID-003]
   ```
5. **Human Review Boundary:** Every evidence packet is flagged `requires_human_review = True` and marked `READY_FOR_REVIEW`. Sentinel never writes directly to live payment networks without human approval.

---

## 8. Auditability & Event Logging

Every lifecycle transition is immutably recorded in the `audit_events` table (`backend/app/services/audit_service.py`):

| Event Type | Trigger Point | Forensic Payload Data |
| :--- | :--- | :--- |
| `CASE_CREATED` | `POST /cases` | Initial case parameters, amounts, customer flags |
| `PREDICTION_GENERATED` | `POST /cases/{id}/predict` | $P(\text{success})$, risk level, signed reason codes, model version |
| `DECISION_MADE` | `POST /cases/{id}/decide` | Action (`ACCEPT`/`ESCALATE`/`CONTEST`), expected recovery, net value, guardrail status |
| `EVIDENCE_PACKET_CREATED` | `POST /cases/{id}/evidence` | Evidence item statuses, completeness %, draft claims, citations |

---

## 9. Demonstration Scenarios

The web dashboard includes 1-click demonstration presets:

```text
┌───────────────────────────────────────────────────────────────────────────────────┐
│ 1. High-Value Contestable Dispute (₹85,000)                                       │
│    • Item Not Received, Delivery Confirmed, 12 Orders, 0 Disputes, 5/6 Evidence    │
│    • P(success) = 96.7% | Risk: HIGH | Net Expected Value = +₹81,720              │
│    • Result: CONTEST ──► Generates 6-item evidence packet & traceable draft        │
├───────────────────────────────────────────────────────────────────────────────────┤
│ 2. Low-Value Unviable Dispute (₹400)                                              │
│    • Unauthorized, Unconfirmed Delivery, 4 Refunds, 2 Disputes                    │
│    • P(success) = 4.0% | Risk: LOW | Net Expected Value = -₹484                   │
│    • Result: ACCEPT ──► Avoids wasting ₹500 operational cost                       │
├───────────────────────────────────────────────────────────────────────────────────┤
│ 3. Deficient Evidence Guardrail Trigger (₹60,000)                                 │
│    • High win probability profile, but only 1/6 evidence items available (16.7%)  │
│    • P(success) = 87.9% | Net Expected Value = +₹52,240                           │
│    • Result: ESCALATE ──► Guardrail overrides positive economics to prevent error │
└───────────────────────────────────────────────────────────────────────────────────┘
```

---

## 10. Technology Stack

- **Backend API:** Python 3.9+, FastAPI, Pydantic v2, Uvicorn
- **Database & ORM:** PostgreSQL 16, SQLAlchemy 2.0, psycopg2-binary (SQLite in-memory for testing)
- **Machine Learning:** scikit-learn, joblib, pandas, numpy
- **Frontend:** Vanilla HTML5, CSS3 (Custom design tokens, glassmorphism), JavaScript (Fetch API)
- **Testing & Quality:** pytest, anyio, httpx
- **Containerization & Hosting:** Docker (python:3.9-slim), Docker Compose, Render Web Service

---

## 11. Project Structure

```text
RazorPay Sentinel/
├── backend/
│   ├── app/
│   │   ├── api/routes.py            # REST API endpoints (/health, /cases, /predict, /decide, /evidence, /audit)
│   │   ├── core/config.py           # App settings & environment variable handling
│   │   ├── core/database.py         # SQLAlchemy engine, sessionmaker & init_db
│   │   ├── models/entities.py       # Database entities (Case, Prediction, Decision, EvidencePacket, AuditEvent)
│   │   ├── schemas/case.py          # Pydantic request/response schemas
│   │   └── services/                # Business services (case, risk, decision, evidence, audit)
│   ├── requirements.txt             # Backend dependencies
│   └── tests/                       # 26 automated unit & integration tests
├── data/
│   ├── disputes.csv                 # 10,000-sample synthetic dataset
│   └── generate_dataset.py          # Dataset generation script
├── frontend/
│   ├── css/style.css                # Custom theme & responsive layout
│   ├── index.html                   # Dashboard UI
│   └── js/app.js                    # Client-side API orchestration
├── ml/
│   ├── decision_policy.py           # Deterministic decision policy & guardrails
│   ├── evaluate.py                  # Model evaluation suite
│   ├── predict.py                   # Prediction interface & mathematical reason codes
│   └── train.py                     # ML training pipeline
├── models/                          # Serialized ML artifacts (.joblib, metadata.json)
├── reports/                         # ROC/PR curves, calibration diagrams, evaluation JSONs
├── Dockerfile                       # Production container definition
├── docker-compose.yml               # Multi-container orchestration
├── .env.example                     # Environment template
└── README.md
```

---

## 12. Local Setup & Development

### Prerequisites
- Python 3.9+
- PostgreSQL (optional; in-memory SQLite is used automatically if `DATABASE_URL` is omitted in testing)

### 1. Clone & Setup Environment
```bash
git clone https://github.com/Sanketsingh23/razorpay-sentinel.git
cd "RazorPay Sentinel"

python -m venv venv
# Windows:
.\venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

pip install -r backend/requirements.txt
pip install -r ml/requirements.txt
```

### 2. Configure Environment Variables
```bash
cp .env.example .env
```
Default `.env` settings:
```ini
DATABASE_URL=postgresql://postgres:password@localhost:5432/razorpay_sentinel
API_HOST=0.0.0.0
API_PORT=8000
ENVIRONMENT=development
LOG_LEVEL=INFO

# Optional Phase 3 LLM Reasoning (defaults to False; deterministic fallback operates with 100% functionality):
LLM_ENABLED=false
LLM_API_KEY=
LLM_MODEL=gemini-1.5-flash
```

### 3. Run the Application
```bash
uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload
```
- Open Dashboard: **`http://localhost:8000`**
- Interactive Swagger API: **`http://localhost:8000/docs`**

---

## 13. Docker Deployment

### Run with Docker
```bash
# Build the container
docker build -t razorpay-sentinel .

# Run container (listens on port 8000)
docker run -p 8000:8000 -e DATABASE_URL="postgresql://user:pass@host:5432/db" razorpay-sentinel
```

### Run with Docker Compose
```bash
docker compose up -d --build
```
This orchestrates:
- PostgreSQL 16 service with persistent volume storage.
- RazorPay Sentinel backend + static dashboard on `http://localhost:8000`.

---

## 14. Production Deployment on Render

RazorPay Sentinel is deployed live on **Render** as a Docker Web Service.

- **Production URL:** [https://razorpay-sentinel.onrender.com](https://razorpay-sentinel.onrender.com)
- **Deployment Branch:** `master`
- **Dynamic Port Binding:** The `Dockerfile` utilizes `CMD ["sh", "-c", "uvicorn backend.app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]` to dynamically bind to Render's injected `$PORT` environment variable.
- **Production Health Monitoring:** `GET /health` verifies database connectivity and model artifact loading.

---

## 15. Automated Testing & Verification

The repository maintains an automated test suite covering API contracts, risk model integration, decision guardrails, evidence assembly, and error handling:

```bash
python -m pytest backend/tests -v
```

### Test Suite Summary (26/26 Passed)
```text
backend/tests/test_api.py::test_health_endpoint PASSED                   [  3%]
backend/tests/test_api.py::test_case_creation PASSED                     [  7%]
backend/tests/test_api.py::test_invalid_case_rejection PASSED            [ 11%]
backend/tests/test_api.py::test_predict_endpoint PASSED                  [ 15%]
backend/tests/test_api.py::test_decision_endpoint PASSED                 [ 19%]
backend/tests/test_api.py::test_accept_decision PASSED                   [ 23%]
backend/tests/test_api.py::test_escalate_decision_guardrail PASSED       [ 26%]
backend/tests/test_api.py::test_contest_decision PASSED                  [ 30%]
backend/tests/test_api.py::test_case_retrieval PASSED                    [ 34%]
backend/tests/test_api.py::test_audit_retrieval PASSED                   [ 38%]
backend/tests/test_api.py::test_database_persistence PASSED              [ 42%]
backend/tests/test_api.py::test_ml_integration_failure_handling PASSED   [ 46%]
backend/tests/test_api.py::test_invalid_llm_action_rejected_by_guardrail PASSED [ 50%]
backend/tests/test_api.py::test_llm_malformed_response_safe_fallback PASSED [ 53%]
backend/tests/test_api.py::test_llm_api_failure_safe_fallback PASSED     [ 57%]
backend/tests/test_api.py::test_reason_codes_preserved_from_risk_engine PASSED [ 61%]
backend/tests/test_api.py::test_contest_case_starts_evidence_workflow PASSED [ 65%]
backend/tests/test_api.py::test_accept_case_rejects_evidence_workflow PASSED [ 69%]
backend/tests/test_api.py::test_escalate_case_rejects_evidence_workflow PASSED [ 73%]
backend/tests/test_api.py::test_missing_evidence_detected PASSED         [ 76%]
backend/tests/test_api.py::test_conflicting_evidence_detected PASSED     [ 80%]
backend/tests/test_api.py::test_response_claims_traceability PASSED      [ 84%]
backend/tests/test_api.py::test_evidence_persistence_and_retrieval PASSED [ 88%]
backend/tests/test_api.py::test_list_cases_endpoint PASSED               [ 92%]
backend/tests/test_api.py::test_preset_samples_endpoint PASSED           [ 96%]
backend/tests/test_api.py::test_frontend_static_serving PASSED           [100%]

============================= 26 passed in 7.36s ==============================
```

Standalone ML regression verification:
```bash
python ml/predict.py
python ml/decision_policy.py
```

---

## 16. Step-by-Step Walkthrough of an Example Decision

Here is how Sentinel processes a high-value dispute:

```text
1. Case Input:
   • Transaction Amount: ₹85,000 | Reason: item_not_received
   • Delivery: Confirmed (1 day delay) | Customer Orders: 12 | Prior Disputes: 0
   • Available Evidence: 5 items | Missing Evidence: 1 item

2. Stage 01 (ML Risk Engine):
   • Standardizes features and applies Logistic Regression weights.
   • Computes P(success) = 0.967 (96.7% win probability).
   • Assigns Risk Level: HIGH (winnability level).
   • Extracts top positive reason codes: ['delivery_confirmed', 'dispute_reason_item_not_received', 'no_previous_disputes'].

3. Stage 03 (Decision Agent):
   • Expected Recovery = ₹85,000 × 0.967 = ₹82,195.
   • Net Expected Value = ₹82,195 - ₹500 (operational cost) = ₹81,695.
   • Checks Evidence Completeness: 5/6 = 83.3% >= 50% threshold (Guardrail passed).
   • Checks Net Value: ₹81,695 >= ₹100 threshold (Economic viability passed).
   • Checks Probability: 0.967 >= 0.65 threshold (Contest zone passed).
   • Authoritative Action Selected: CONTEST.

4. Stage 04 (Evidence Agent):
   • Validates 6 evidence items (5 AVAILABLE, 1 MISSING).
   • Assembles factual contest response statement.
   • Links 3 verified claims directly to [EVID-001], [EVID-002], and [EVID-003].
   • Marks packet READY_FOR_REVIEW with requires_human_review = True.

5. Stage 05 (Audit Trail):
   • Logs CASE_CREATED, PREDICTION_GENERATED, DECISION_MADE, EVIDENCE_PACKET_CREATED.
```

---

## 17. Design Principles & Engineering Decisions

1. **Probabilistic Assessment vs. Deterministic Governance:** ML calculates uncertainty; business policy and economics decide actions.
2. **Economic Viability First:** Eliminates "cost-blind" contesting. Transactions where operational expense exceeds expected recovery are automatically accepted.
3. **Hard Guardrails Overrule Optimization:** Incomplete evidence triggers mandatory escalation, preventing reputational damage and penalties from submitting deficient claims.
4. **Zero AI Hallucination:** Reason codes originate from mathematical feature weights ($c_j = \beta_j \cdot z_j$), and response claims strictly cite validated evidence IDs (`[EVID-xxx]`).
5. **Human-in-the-Loop Safeguard:** Dispute responses are compiled and validated for human approval, maintaining security boundaries without unguarded network submissions.

---

## 18. Known Limitations & Future Scope

- **Synthetic Training Baseline:** Trained on a 10,000-sample synthetic dataset designed around chargeback patterns. In a production integration, the model would be retrained on live historical merchant disputes.
- **Submission Boundary:** Prepared packets require human merchant approval and do not execute automated write operations to banking networks.
- **Single-Merchant Scope:** Currently optimized for single-merchant dispute triage, with architectural readiness for multi-tenant gateway deployments.

---

## 19. Razorpay Contest Context

Built for the **Razorpay Contest**, RazorPay Sentinel demonstrates:
- **Intelligent Dispute Triage:** Automated classification of disputes into profitable vs. loss-making paths.
- **Financial Engineering:** Recovery optimization combining probability, transaction scale, and handling expenses.
- **Explainability:** Transparent, auditable mathematical factor extraction.
- **Production Readiness:** Full-stack integration with FastAPI, PostgreSQL persistence, clean REST APIs, automated test coverage, Docker packaging, and live cloud deployment on Render.

---

## 20. License

This project was developed for the Razorpay Contest submission. All rights reserved.
