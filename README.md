# RazorPay Sentinel

Autonomous Dispute Risk Intelligence & Decision Orchestration Platform.

---

## What it does
RazorPay Sentinel analyzes incoming payment disputes, calculates calibrated win probabilities, computes net expected financial recovery, evaluates evidence completeness against strict guardrails, and automates 3-way decision orchestration: **ACCEPT**, **ESCALATE**, or **CONTEST**. For cases decided as **CONTEST**, Sentinel gathers evidence, validates item statuses, ensures claim traceability, and drafts a dispute contest response for human review.

---

## Problem
Online merchants lose millions to payment chargebacks and disputes. Manual dispute evaluation is slow, inconsistent, and often results in contesting unwinnable disputes (incurring wasted operational costs) or conceding high-value, fully winnable disputes.

---

## Solution
RazorPay Sentinel introduces a multi-stage decision pipeline combining:
1. Supervised Machine Learning for calibrated win probabilities and feature explanations.
2. Deterministic policy guardrails and economic thresholds.
3. Automated evidence collection, conflict validation, and claim-traceable dispute response drafting.
4. An interactive single-page dashboard with a forensic audit timeline.

---

## Architecture

```text
               DISPUTE CASE
                    │
                    ▼
         ┌─────────────────────┐
         │ Phase 1: Risk Engine│  ──► Calibrated P(success) + Reason Codes
         └──────────┬──────────┘
                    │
                    ▼
        ┌───────────────────────┐
        │Phase 3: Decision Agent│  ──► Economic Thresholds & Deterministic Guardrails
        └───────────┬───────────┘
                    │
          ┌─────────┼─────────┐
          ▼         ▼         ▼
       ACCEPT    ESCALATE   CONTEST
                              │
                              ▼
                   ┌─────────────────────┐
                   │Phase 4: Evidence Ag.│ ──► 6 Evidence Items + Traceable Draft
                   └──────────┬──────────┘
                              │
                              ▼
                   ┌─────────────────────┐
                   │    Human Review     │
                   └─────────────────────┘
```

---

## Phase 1 — Risk Engine
- **Model:** Logistic Regression selected over XGBoost based on multi-criteria evaluation (Brier Score `0.1195`, ECE `0.0198`, ROC-AUC `0.9019`).
- **Mathematical Reason Codes:** Signed feature contributions ($c_j = \beta_j \cdot z_j$) — no generative AI hallucination.
- **Operating Threshold:** Calibrated at $0.40$ (Precision $81.3\%$, Recall $93.1\%$).

---

## Phase 2 — Backend + PostgreSQL
- **Framework:** FastAPI with SQLAlchemy 2.0.
- **Database:** PostgreSQL storing cases, predictions, decisions, evidence packets, and audit logs.
- **Persistence:** Full state preservation across restarts and automated migrations.

---

## Phase 3 — Decision Agent
- **3-Way Actions:** `ACCEPT` (unviable economics or low win rate), `ESCALATE` (uncertain win rate or deficient evidence guardrail), `CONTEST` (high win rate + positive net recovery + complete evidence).
- **Authoritative Guardrails:** Deterministic rules strictly override any external recommendation.
- **Safe Fallback:** Operates with 100% functionality even without an external LLM API key.

---

## Phase 4 — Evidence Agent
- **Collection & Validation:** Analyzes 6 structured evidence types (`delivery_proof`, `order_confirmation`, `customer_account_history`, `refund_records`, `customer_communication`, `payment_verification`).
- **Validation Statuses:** `AVAILABLE`, `MISSING`, `CONFLICTING`, `INVALID`.
- **Claim Traceability:** Every factual claim in the dispute response draft explicitly cites an available `source_evidence_id` (`[EVID-xxx]`).

---

## Phase 5 — Frontend Integration
- **Dashboard:** Interactive dark-mode web application mounted at `/`.
- **1-Click Demonstration Presets:** Pre-configured profiles for CONTEST, ACCEPT, and ESCALATE scenarios.
- **Forensic Audit Timeline:** Chronological tracking of all case lifecycle events.

---

## Technology Stack
- **Backend:** Python 3.9+, FastAPI, SQLAlchemy, Pydantic, Uvicorn, psycopg2-binary
- **ML / Data:** scikit-learn, joblib, pandas, numpy
- **Frontend:** Modern Vanilla HTML5, CSS3 (Glassmorphism & custom design system), JavaScript (Fetch API)
- **Database:** PostgreSQL 16
- **Deployment:** Docker, Docker Compose

---

## Project Structure
```text
RazorPay Sentinel/
├── backend/
│   ├── app/
│   │   ├── api/routes.py          # REST API endpoints
│   │   ├── core/config.py         # App configuration & environment settings
│   │   ├── core/database.py       # SQLAlchemy engine & session setup
│   │   ├── models/entities.py     # Database entities (Case, Prediction, Decision, EvidencePacket, AuditEvent)
│   │   ├── schemas/case.py        # Pydantic validation schemas
│   │   └── services/              # Domain services (risk, decision, evidence, audit)
│   ├── requirements.txt
│   └── tests/                     # 26 automated unit and integration tests
├── data/
│   ├── disputes.csv               # Synthetic development dataset (10,000 samples)
│   └── generate_dataset.py        # Dataset generator
├── frontend/
│   ├── css/style.css              # Custom styling
│   ├── index.html                 # Dashboard layout
│   └── js/app.js                  # Client application logic
├── ml/
│   ├── decision_policy.py         # Core decision policy logic
│   ├── evaluate.py                # Model evaluation suite
│   ├── predict.py                 # Prediction interface
│   └── train.py                   # Model training pipeline
├── models/                        # Serialized ML artifacts
├── Dockerfile                     # Production container definition
├── docker-compose.yml             # Container orchestration
└── README.md
```

---

## Local Setup

1. **Clone & Navigate:**
   ```bash
   git clone <repo_url>
   cd "RazorPay Sentinel"
   ```

2. **Create Virtual Environment & Install Dependencies:**
   ```bash
   python -m venv venv
   # Windows:
   .\venv\Scripts\activate
   # Linux/macOS:
   source venv/bin/activate

   pip install -r backend/requirements.txt
   pip install -r ml/requirements.txt
   ```

---

## Environment Variables

Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```

Configurable options:
```ini
DATABASE_URL=postgresql://postgres:password@localhost:5432/razorpay_sentinel
API_HOST=0.0.0.0
API_PORT=8000
ENVIRONMENT=development

# Optional Phase 3 LLM Reasoning:
LLM_ENABLED=false
LLM_API_KEY=
LLM_MODEL=gemini-1.5-flash
```

---

## Running Backend

Start the FastAPI application:
```bash
uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload
```

---

## Running Frontend

The frontend is served directly by the FastAPI backend at the root path:
- Open your browser to: **[http://localhost:8000](http://localhost:8000)**

---

## Running Tests

Run the complete test suite (26/26 tests):
```bash
python -m pytest backend/tests -v
```

Run ML Phase 1 standalone regression checks:
```bash
python ml/predict.py
python ml/decision_policy.py
```

---

## API Documentation

- **Interactive Swagger UI:** `http://localhost:8000/docs`
- **ReDoc UI:** `http://localhost:8000/redoc`

Key endpoints:
- `GET /health`: Health check (DB and model readiness)
- `GET /cases`: List recent dispute cases
- `GET /cases/presets/samples`: Retrieve 1-click demonstration presets
- `POST /cases`: Create new dispute case
- `GET /cases/{case_id}`: Retrieve case details with latest predictions and decisions
- `POST /cases/{case_id}/predict`: Run ML Risk Engine
- `POST /cases/{case_id}/decide`: Run Decision Agent
- `POST /cases/{case_id}/evidence`: Generate Evidence Packet & Dispute Draft
- `GET /cases/{case_id}/evidence`: Retrieve Evidence Packet
- `GET /cases/{case_id}/audit`: Retrieve chronological audit timeline

---

## Deployment

### Containerized Deployment (Docker Compose)
```bash
docker compose up -d --build
```
This orchestrates:
- PostgreSQL 16 with a persistent data volume.
- FastAPI backend + Static frontend accessible at `http://localhost:8000`.

### Cloud Deployment (Render, Railway, Fly.io, AWS ECS)
Set environment variable `DATABASE_URL` pointing to your managed PostgreSQL instance.

---

## Known Limitations
1. **Network Submission Boundary:** In accordance with security specifications, Sentinel prepares dispute packets for human approval and does not execute automated live write operations directly to payment gateways.
2. **LLM Reasoning:** Live Gemini reasoning is optional; if unconfigured, deterministic rules and template drafting execute with zero degradation.
