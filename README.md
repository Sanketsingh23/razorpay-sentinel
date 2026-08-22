# RazorPay Sentinel — AI Risk Manager for Intelligent Payment & Dispute Decisions

> **Project Core Definition:**  
> RazorPay Sentinel is an AI Risk Manager that analyzes payment disputes, estimates the probability and expected financial value of contesting them, evaluates evidence quality, and chooses **ACCEPT**, **ESCALATE**, or **CONTEST**. When **CONTEST** is selected, a downstream Evidence/Dispute Agent executes the contest workflow.

---

## 1. System Architecture

```text
Transaction / Dispute
        ↓
Context / Feature Extraction
        ↓
Risk Model (Supervised ML)
        ↓
P(successful contest) + Model-Derived Reason Codes
        ↓
Decision Policy (Expected Value + Evidence Guardrails)
   ┌────────────┼────────────┐
 ACCEPT      ESCALATE     CONTEST
   ↓            ↓            ↓
Refund        Human     Evidence/Dispute Agent
Process       Review         ↓
                         Evidence Packet
                             ↓
                         AI Draft Response
                             ↓
                         Human Approval & Submission
```

### Key Architectural Separation
1. **Risk Manager (Sentinel):** Decides *what action to take* based on probability, evidence quality, expected financial value, and business rules.
2. **Dispute/Evidence Agent:** Downstream *execution capability* activated **only** when the decision is `CONTEST`.
3. **ML Risk Model:** Predicts $P(\text{successful contest})$ and produces model-derived feature contributions. It **never** directly decides the action.

---

## 2. Phase 1 Status: Risk Intelligence Core

> **IMPORTANT DISCLAIMER:**  
> All metrics reported below are **SYNTHETIC DEVELOPMENT METRICS** derived from a 10,000-sample synthetic dataset (`data/disputes.csv`). They do **not** represent production Razorpay performance benchmarks.

### Summary of Deliverables
- [x] **Model Comparison:** Evaluated L2-regularized Logistic Regression vs. XGBoost on Stratified Train (6,000) / Validation (2,000) / Test (2,000) splits.
- [x] **Multi-Criteria Model Selection:** Selected **Logistic Regression** based on superior calibration (Brier, ECE), cost-sensitive decision performance, and discrimination.
- [x] **Calibration Analysis:** Verified predicted probabilities correspond to observed frequencies (Brier: `0.1244`, ECE: `0.0192`).
- [x] **Threshold Optimization:** Swept probability thresholds (0.05 to 0.80) on validation data; selected **0.40** (Precision $\ge 0.70$, Recall $\ge 0.75$, F1 = 0.8681).
- [x] **Cost-Sensitive Evaluation:** Formally modeled false contest costs, missed recovery opportunity costs, and operational expenses.
- [x] **Model-Derived Reason Codes:** Built mathematical feature explanation engine ($c_j = \beta_j \cdot z_j$) — **no LLM hallucination**.
- [x] **Prediction Interface:** Implemented stable, reusable `predict_case(features_dict)` for Phase 2 integration.
- [x] **Reports & Visualizations:** Generated ROC/PR curves, reliability diagrams, threshold sweeps, and JSON reports in `reports/`.

---

## 3. Model Comparison & Selection Results

### Multi-Criteria Validation Performance

| Criterion | Evaluation Metric | Logistic Regression | XGBoost | Winner | Rationale |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **1. Calibration (Primary)** | Brier Score $\downarrow$ | **0.1195** | 0.1284 | **LR** | Lower mean squared probability error |
| **2. Reliability (Primary)** | ECE $\downarrow$ | **0.0198** | 0.0358 | **LR** | Predictions closely match true win rates |
| **3. Decision Economics** | Expected Cost (Val) $\downarrow$ | **INR 2,767,449** | INR 3,035,470 | **LR** | Minimizes wasted effort & missed recoveries |
| **4. Discrimination** | ROC-AUC $\uparrow$ | **0.9019** | 0.8923 | **LR** | Superior ranking ability across thresholds |
| **5. Precision-Recall** | PR-AUC $\uparrow$ | **0.9065** | 0.8954 | **LR** | Superior precision at high-recall operating points |

**Selected Model:** **Logistic Regression** (5/5 evaluation criteria won).

---

## 4. Threshold Optimization (Validation Set)

Threshold optimization was performed strictly on the **validation set** (2,000 samples).

| Threshold | Precision | Recall | F1 Score | TP | FP | FN | TN | Total Cost (INR) | Net Recovery (INR) | Status |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| 0.05 | 0.6249 | 0.9928 | 0.7670 | 1106 | 664 | 8 | 222 | 772,211 | 27,399,960 | Reject (38% wasted contests) |
| 0.10 | 0.6759 | 0.9829 | 0.8010 | 1095 | 525 | 19 | 361 | 1,066,010 | 27,111,661 | Sub-optimal precision |
| 0.20 | 0.7378 | 0.9623 | 0.8352 | 1072 | 381 | 42 | 505 | 1,938,073 | 26,251,098 | Viable |
| 0.30 | 0.7815 | 0.9506 | 0.8578 | 1059 | 296 | 55 | 590 | 2,209,883 | 25,985,788 | Strong |
| **0.40** | **0.8133** | **0.9309** | **0.8681** | **1037** | **238** | **77** | **648** | **2,767,449** | **25,439,222** | **SELECTED OPERATING POINT** |
| 0.50 | 0.8315 | 0.8995 | 0.8642 | 1002 | 203 | 112 | 683 | 3,616,862 | 24,607,309 | Conservative |
| 0.60 | 0.8470 | 0.8501 | 0.8486 | 947 | 171 | 167 | 715 | 5,275,491 | 22,976,180 | Higher missed recoveries |
| 0.70 | 0.8735 | 0.7684 | 0.8176 | 856 | 124 | 258 | 762 | 6,903,547 | 21,393,624 | High precision / Low recall |
| 0.80 | 0.9041 | 0.6176 | 0.7339 | 688 | 73 | 426 | 813 | 11,644,436 | 16,736,735 | Overly restrictive |

### Threshold Selection Rationale
A naive cost-minimizing threshold (0.05–0.06) was rejected because it yields Precision = 62.5% (meaning ~38% of contested disputes are lost, leading to massive operational friction and merchant dissatisfaction). 
**Threshold 0.40** was chosen as the optimal balance:
- **Precision = 81.33%** (over 80% of contested cases are won).
- **Recall = 93.09%** (captures 93% of all recoverable dispute funds).
- **F1 Score = 0.8681** (highest balanced F1 among viable candidates).

---

## 5. Cost-Sensitive Evaluation

### Explicit Cost Assumptions
- **False Contest Cost ($C_{\text{FP}}$):** `INR 1,000` (wasted evidence compilation effort, partner processing fees, merchant operational time).
- **Missed Recovery Cost ($C_{\text{FN}}$):** `1.0 × transaction_amount` (direct revenue lost when an ACCEPT decision is made on a winnable case).
- **Contest Operational Cost ($C_{\text{ops}}$):** `INR 500` per contested case.
- **Escalation Cost ($C_{\text{esc}}$):** `INR 200` per case sent to human review.

### Business Outcomes at Selected Threshold (0.40)
- **Contest Rate:** 63.7% of incoming disputes automated for contest.
- **Gross Recovery:** INR 26,195,722.
- **Net Realized Recovery (after operational & false contest costs):** INR 25,439,222.
- **Wasted Contest Cost:** INR 238,000.
- **Missed Recovery Cost:** INR 2,529,449.

---

## 6. Final Held-Out Test Set Performance

The selected model and operating threshold (0.40) were evaluated **once** on the untouched held-out test set (2,000 samples).

| Metric | Held-Out Test Result | Interpretation |
| :--- | :---: | :--- |
| **Model** | **Logistic Regression (L2)** | Selected candidate |
| **Operating Threshold** | **0.40** | Validation-tuned |
| **Precision** | **79.03%** | 4 out of 5 contested disputes won |
| **Recall** | **93.36%** | Catches 93.4% of winnable dispute revenue |
| **F1 Score** | **0.8560** | Strong balanced classification score |
| **ROC-AUC** | **0.8964** | Excellent ranking discrimination |
| **PR-AUC** | **0.8986** | High precision across recall spectrum |
| **Brier Score** | **0.1244** | Well-calibrated continuous probabilities |
| **Expected Calibration Error (ECE)** | **0.0192 (1.92%)** | Highly reliable probability alignment |
| **Confusion Matrix** | **TP: 1040, FP: 276, FN: 74, TN: 610** | 2,000 test cases total |

---

## 7. Model-Derived Reason Codes

Reason codes are computed directly from the model's standardized log-odds contributions:
$$\Delta_j = \beta_j \cdot \frac{x_j - \mu_j}{\sigma_j}$$

- If $\Delta_j > 0$: feature pulls the case towards **CONTEST** success (positive driver).
- If $\Delta_j < 0$: feature pulls the case away from **CONTEST** success (negative driver / risk factor).

### Top Feature Weights in Logistic Regression

| Feature | Direction | Coefficient ($\beta$) | Impact on Win Probability |
| :--- | :---: | :---: | :--- |
| `delivery_confirmed` | Positive | `+1.5820` | Strongest single predictor of winning dispute |
| `previous_disputes` | Negative | `-0.5367` | Serial dispute history strongly reduces win rate |
| `previous_refunds` | Negative | `-0.5040` | Frequent refund claims correlate with abusive chargebacks |
| `reason_item_not_received` | Positive | `+0.3726` | Winnable with proof of delivery |
| `reason_unauthorized` | Negative | `-0.3256` | Harder to contest without biometric/OTP logs |
| `refund_amount_ratio` | Negative | `-0.2841` | High refund percentage relative to orders reduces win rate |

---

## 8. Prediction Interface (`predict_case`)

Located at `ml/predict.py`.

### Example Input
```python
from ml.predict import predict_case

case_input = {
    "transaction_amount": 84999,
    "delivery_confirmed": True,
    "previous_disputes": 0,
    "previous_refunds": 1,
}

result = predict_case(case_input)
```

### Example Output
```json
{
  "contest_probability": 0.9321,
  "risk_level": "HIGH",
  "reason_codes": [
    "delivery_confirmed",
    "dispute_reason_item_not_received",
    "no_previous_disputes",
    "low_refund_ratio",
    "high_transaction_amount_scrutiny"
  ],
  "positive_factors": [
    "+ Delivery confirmed",
    "+ Dispute reason: Item Not Received (winnable with delivery proof)",
    "+ No previous dispute history",
    "+ Low customer refund ratio (5.0%)",
    "+ Low previous refund activity (1 prior refund)"
  ],
  "negative_factors": [
    "- High-value transaction (INR 84,999) requires rigorous evidence",
    "- Minimal customer communication (2 logs)",
    "- Minimal customer order history (5 orders)"
  ],
  "model_type": "Logistic Regression",
  "recalibrated": false,
  "optimal_threshold": 0.4,
  "data_disclaimer": "Prediction based on model trained with SYNTHETIC DEVELOPMENT DATA.",
  "note": "This is P(success) only. Use decision_policy.decide() for ACCEPT/ESCALATE/CONTEST."
}
```

---

## 9. Decision Policy Layer (`decision_policy.py`)

Separate from ML model. Evaluates:
1. $P(\text{success})$ from risk model
2. Evidence Completeness Guardrail ($\ge 50\%$)
3. Net Expected Value ($E[\text{Recovery}] - C_{\text{ops}} \ge \text{INR } 100$)

```text
Decision Rules:
- If Evidence Completeness < 50%                     -> ESCALATE (Guardrail triggered)
- If P(success) < 0.30 OR Net Expected Value < INR 100 -> ACCEPT
- If P(success) >= 0.65 AND Net Expected Value >= 100  -> CONTEST
- If 0.30 <= P(success) < 0.65 (Uncertainty zone)     -> ESCALATE
```

---

## 10. Phase 1 Sign-Off Status

| Requirement | Specification | Status |
| :--- | :--- | :---: |
| 1. Synthetic Dataset Generator | 10k cases with realistic correlations | **Complete** |
| 2. Baseline Model & Comparison | Logistic Regression vs XGBoost | **Complete** |
| 3. Calibration & Reliability | Brier < 0.13, ECE < 0.03 | **Complete** |
| 4. Threshold Optimization | Validated threshold sweep (0.40) | **Complete** |
| 5. Cost-Sensitive Modeling | Explicit $C_{\text{FP}}$, $C_{\text{FN}}$, $C_{\text{ops}}$, $C_{\text{esc}}$ | **Complete** |
| 6. Model-Derived Reason Codes | Signed coefficient contributions | **Complete** |
| 7. Reusable `predict_case()` Interface | Python function & JSON schema | **Complete** |
| 8. Evaluation Artifacts & Reports | Curves, matrices, JSON summaries | **Complete** |

**Phase 1 is COMPLETE.**

---

## 11. Phase 2: Backend API & Database

The backend exposes the Risk Engine and Decision Policy via a clean REST API and manages persistence for cases, predictions, decisions, and audit history.

### Architecture Overview

```text
Client (REST)
    ↓
FastAPI Backend (backend/app/main.py)
    ↓
┌───────────────────────────────────────────────┐
│ Case Service    │ Creates & retrieves cases   │
│ Risk Service    │ Calls ml.predict            │
│ Decision Service│ Calls ml.decision_policy    │
│ Audit Service   │ Logs state & error history  │
└───────────────────────┬───────────────────────┘
                        ↓
            PostgreSQL Database / SQLite
```

### Setup & Configuration

1. **Install Dependencies:**
   ```bash
   pip install -r backend/requirements.txt
   ```

2. **Environment Variables:**
   Copy `.env.example` to `.env`:
   ```bash
   cp .env.example .env
   ```
   Variables:
   - `DATABASE_URL`: PostgreSQL connection string (e.g. `postgresql://postgres:postgres@localhost:5432/razorpay_sentinel`)
   - `API_HOST`: Bind host (default: `0.0.0.0`)
   - `API_PORT`: Bind port (default: `8000`)
   - `ENVIRONMENT`: `development` or `production`

3. **Running PostgreSQL (Local or Docker):**
   ```bash
   # If using Docker:
   docker run --name sentinel-postgres -e POSTGRES_PASSWORD=postgres -e POSTGRES_DB=razorpay_sentinel -p 5432:5432 -d postgres:16
   ```

4. **Start FastAPI Backend:**
   ```bash
   uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload
   ```

### API Endpoints

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/health` | Health check (verifies DB connection and model readiness) |
| `POST` | `/cases` | Create a new case and log audit event |
| `GET` | `/cases/{case_id}` | Retrieve case details, latest prediction, and latest decision |
| `POST` | `/cases/{case_id}/predict` | Run Phase 1 Risk Engine on case data and store prediction |
| `POST` | `/cases/{case_id}/decide` | Run Phase 1 Decision Policy on prediction and store decision |
| `GET` | `/cases/{case_id}/audit` | Retrieve chronological audit history for the case |

### Running Tests

Run the complete backend test suite:
```bash
python -m pytest backend/tests -v
```

---

## 12. Phase 3: Decision Agent

The **Decision Agent** is the reasoning and orchestration layer between the Phase 1 Risk Engine and the Phase 4 Evidence/Dispute Agent.

### Key Architectural Separation

```text
               CASE
                │
                ▼
         ┌─────────────┐
         │ Risk Engine │ (predict_case -> P(success) + Model-Derived Reason Codes)
         └──────┬──────┘
                │
                ▼
        ┌────────────────┐
        │ Decision Agent │ (Evaluates structured context, guardrails, economics, LLM)
        └───────┬────────┘
                │
      ┌─────────┼─────────┐
      ▼         ▼         ▼
   ACCEPT    ESCALATE   CONTEST
                          │
                          ▼
                Phase 4 Evidence Agent
```

1. **Risk Engine Responsibility:** Predicts contest-success probability ($P(\text{success})$), risk level, and model-derived reason codes from signed logistic regression coefficients.
2. **Decision Agent Responsibility:** Gathers structured case context, evaluates financial economics, applies deterministic guardrails, incorporates structured LLM reasoning (with safe fallback), and validates final actions (`ACCEPT`, `ESCALATE`, `CONTEST`).

### Authoritative Deterministic Guardrails

The LLM is subordinate to deterministic business guardrails:
- **Evidence Completeness Guardrail:** If `evidence_completeness < 50%`, the action is strictly forced to `ESCALATE`.
- **Expected Value Constraint:** If $E[\text{Net Recovery}] < \text{INR } 100$ or $P(\text{success}) < 0.30$, the action is `ACCEPT`.
- **Zero Fabrication:** The LLM cannot invent transaction facts, evidence, or reason codes.
- **Safe Fallback:** If the LLM produces an invalid action, malformed response, or fails, the agent cleanly falls back to the deterministic policy.

### LLM Configuration (Optional)

Configure via environment variables:
```bash
LLM_ENABLED=true
LLM_API_KEY=your_gemini_api_key
LLM_MODEL=gemini-1.5-flash
```
*Note: Automated tests run in deterministic/mock mode without requiring an external LLM API key.*

---

## 13. Phase 4: Evidence & Dispute Response Agent

The **Evidence & Dispute Response Agent** activates when Phase 3 determines a `CONTEST` action. It organizes case evidence, validates item statuses, computes deterministic evidence completeness, verifies claim-to-evidence traceability, and drafts a factual dispute contest response for human review.

### Architecture & Data Flow

```text
CONTEST Decision (from Phase 3)
      │
      ▼
POST /cases/{case_id}/evidence
      │
      ├─► 1. Validate Decision is CONTEST (reject ACCEPT/ESCALATE)
      ├─► 2. Collect structured Evidence Items (EVID-001 to EVID-006)
      ├─► 3. Validate Evidence Status (AVAILABLE, MISSING, CONFLICTING, INVALID)
      ├─► 4. Calculate Deterministic Evidence Completeness (N_valid / N_total)
      ├─► 5. Generate Evidence Packet (Status: READY_FOR_REVIEW)
      ├─► 6. Draft Dispute Contest Response (Every claim cites [EVID-xxx])
      ├─► 7. Persist Evidence Packet & Log Audit Event (EVIDENCE_PACKET_CREATED)
      └─► 8. Surface for Human Review
```

### Evidence Types & Traceability

| Evidence ID | Evidence Type | Case Signal Source | Validation Rule |
| :--- | :--- | :--- | :--- |
| `EVID-001` | `delivery_proof` | `delivery_confirmed`, `delivery_delay_days` | Flagged `CONFLICTING` if delay > 60 days |
| `EVID-002` | `order_confirmation` | `transaction_amount`, `dispute_reason` | Flagged `INVALID` if amount <= 0 |
| `EVID-003` | `customer_account_history`| `customer_order_count`, `previous_disputes`| Categorized `AVAILABLE` |
| `EVID-004` | `refund_records` | `previous_refunds`, `refund_amount_ratio` | Flagged `CONFLICTING` if refund ratio > 80% |
| `EVID-005` | `customer_communication` | `communication_count` | Flagged `MISSING` if communication count is 0 |
| `EVID-006` | `payment_verification` | `payment_failures` | Flagged `CONFLICTING` if failures > 3 |

### Strict Non-Hallucination Boundary
- **Claim Traceability**: Every sentence in `response_draft.claims` cites a verified, available `source_evidence_id`.
- **Zero Fact Invention**: No external dates, arbitrary amounts, or simulated customer statements are fabricated.
- **Human Review Safeguard**: Dispute submissions are NEVER automatically dispatched to payment networks or gateways without human approval.

---

## 14. Phase 5: Full System Integration, Frontend & Deployment

RazorPay Sentinel is fully integrated as an end-to-end demonstrable and containerized dispute risk management platform.

### Running the Frontend Dashboard

1. **Local Execution:**
   ```bash
   uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload
   ```
   Open your browser to [http://localhost:8000](http://localhost:8000) to access the interactive web dashboard.

2. **Containerized Deployment (Docker Compose):**
   ```bash
   docker compose up -d --build
   ```
   - Orchestrates PostgreSQL 16 database and Sentinel FastAPI application.
   - Access the dashboard at `http://localhost:8000`.
   - Access Swagger API documentation at `http://localhost:8000/docs`.

### Features of the Frontend Dashboard
- **1-Click Demonstration Presets**: Pre-loaded contestable, unviable accept, and guardrail escalate profiles.
- **Visual Risk Engine Progression**: Dynamic probability meter, signed reason codes, positive/negative factor analysis.
- **Decision Policy Visualizer**: Expected recovery calculations, net value thresholds, guardrail trigger notifications.
- **Interactive Evidence Inspector**: 6 validated evidence item cards, claim citation links (`[EVID-xxx]`), and dispute contest drafts.
- **Forensic Audit Timeline**: Real-time event tracking with JSON metadata inspection.




