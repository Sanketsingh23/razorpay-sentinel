# RazorPay Sentinel — Project Handoff / Master Specification

## Purpose of this file

This document is the single source of truth for continuing the **RazorPay Sentinel** project in a new AI coding chat.

**Project title:** RazorPay Sentinel  
**Subtitle:** AI Risk Manager for Intelligent Payment & Dispute Decisions

Attach this document **and the three PNG architecture diagrams** to the new chat.

---

# 1. One-sentence project definition

**RazorPay Sentinel is an AI Risk Manager that analyzes a payment/dispute, estimates the probability and expected value of contesting it, evaluates evidence quality, and chooses ACCEPT, ESCALATE, or CONTEST; when CONTEST is selected, an evidence/dispute agent executes the downstream workflow.**

---

# 2. The key product distinction

We deliberately moved away from building only an AI Dispute Responder.

### Old idea

> "A dispute arrived. Automatically gather evidence and fight it."

That overlaps heavily with an existing Razorpay Dispute Responder capability.

### New idea

> "A dispute/payment arrived. What is the smartest action to take?"

The product's core intelligence is **decision-making**:

```text
Analyze case
    ↓
Predict contest success
    ↓
Evaluate evidence
    ↓
Evaluate expected financial/operational value
    ↓
ACCEPT / ESCALATE / CONTEST
```

The old dispute-responder functionality is still present, but only as a **downstream execution capability** of the CONTEST branch.

### Core differentiation

**Risk Manager = decides what should happen.**

**Dispute Agent = executes the contest workflow when the Risk Manager chooses CONTEST.**

---

# 3. Product behavior

A case enters the system.

## Step 1 — Context Engine

Collect/derive:

- transaction amount
- payment information
- customer history
- order information
- delivery information
- previous refunds
- previous disputes
- communications
- dispute reason
- evidence availability

## Step 2 — Risk Engine

A supervised ML model predicts:

- contest-success probability
- risk level
- explainable reason codes

Example:

```text
Contest probability: 0.91
Risk level: HIGH

Reasons:
+ Delivery confirmed
+ No previous dispute pattern
+ High-value transaction
+ Strong evidence availability
```

The probability must come from the ML model.

The reason codes must be derived from real model features/contributions. Do not let the LLM invent them.

## Step 3 — Decision Agent / Policy

The decision layer considers:

- ML probability
- evidence completeness
- expected recovery
- contest/operational cost
- business rules
- model reliability
- uncertainty

Possible actions:

### ACCEPT

Use when contesting is not worthwhile or the case is weak.

### ESCALATE

Use when:

- evidence is insufficient
- records conflict
- model reliability is poor
- the case falls outside automation boundaries

### CONTEST

Use when:

- probability of successful contest is sufficiently high
- evidence is sufficiently complete
- expected value is positive
- the case satisfies business rules

## Step 4 — Contest Agent

Only for CONTEST:

```text
Retrieve evidence
↓
Check evidence completeness
↓
Assemble evidence packet
↓
Draft response
↓
Human approval
↓
Submission
```

The agent must use explicit tools and must never invent evidence.

## Step 5 — Outcome Logger

Log:

- case ID
- features
- model probability
- reason codes
- decision
- evidence completeness
- agent actions
- human decision
- human override
- final outcome
- timestamps

## Step 6 — Feedback

Resolved outcomes can become labelled data for periodic model retraining.

Do NOT claim online learning unless it is actually implemented.

---

# 4. High-level architecture

See `01_high_level_architecture.png`.

```text
Transaction / Dispute
        ↓
Context Engine
        ↓
Risk Engine
        ↓
Decision Agent / Policy
   ↙          ↓          ↘
ACCEPT     ESCALATE     CONTEST
  ↓           ↓            ↓
Refund      Human       Evidence Agent
Process     Review           ↓
  ↓                        Evidence
Outcome Logger             Packet
                              ↓
                         AI Draft
                              ↓
                       Human Approval
                              ↓
                           Submit
                              ↓
                        Outcome Logger
```

---

# 5. Contest/evidence architecture

See `02_contest_agent.png`.

The agent should have explicit tools such as:

- `get_transaction()`
- `get_order()`
- `get_delivery_status()`
- `get_customer_history()`
- `get_refund_history()`
- `get_dispute_history()`
- `get_customer_messages()`
- `calculate_evidence_completeness()`
- `generate_evidence_packet()`
- `draft_dispute_response()`

Show observable tool activity in the UI.

Example:

```text
Case received
↓
Retrieved order
↓
Retrieved delivery
↓
Retrieved customer history
↓
Retrieved refund history
↓
Evidence completeness = 92%
↓
Evidence packet generated
↓
Draft response generated
↓
Human approval required
```

Do NOT expose hidden chain-of-thought. Show only tool calls, outputs, decisions, evidence, and system events.

---

# 6. ML architecture

See `03_ml_feedback_loop.png`.

## Initial target

The ML target should represent:

> **Probability that the merchant successfully contests the dispute.**

This is better than simply predicting "fraud."

Conceptually:

```text
P(successful contest | available evidence)
```

Then the decision policy can combine it with economics and evidence.

Example:

```text
Dispute amount = ₹85,000
P(success) = 0.91

Expected recovery
= ₹85,000 × 0.91
= ₹77,350

If contesting cost is low relative to expected recovery:
→ CONTEST
```

Another example:

```text
Dispute amount = ₹1,200
P(success) = 0.12

Expected recovery
= ₹144

If operational cost exceeds expected recovery:
→ ACCEPT
```

These are examples, not hardcoded production thresholds.

---

# 7. ML features

Development dataset can contain:

- transaction amount
- customer order count
- previous refunds
- previous disputes
- delivery confirmed
- delivery delay
- dispute delay
- customer average order value
- communication count
- refund amount
- payment failures
- evidence items available
- evidence items missing
- dispute reason
- final outcome

Potential future features:

- merchant category
- item category
- device/account signals
- payment method
- geographic consistency
- order cancellation history
- fulfillment type
- historical contest rate
- historical customer behavior

Only add features when they have a clear purpose and data source.

---

# 8. ML evaluation

Use:

```text
Training
    ↓
Validation
    ↓
Held-out Test
```

Metrics:

- Precision
- Recall
- F1
- ROC-AUC
- PR-AUC
- confusion matrix
- false positives
- false negatives
- false-positive cost
- false-negative cost

Do not fabricate or exaggerate metrics.

The project should clearly distinguish:

**Synthetic development metrics**

from

**Real/approved benchmark metrics**

---

# 9. Explainability

The risk model must explain its prediction.

Example:

```text
CONTEST PROBABILITY
91%

REASON CODES
✓ Delivery confirmed
✓ Strong evidence availability
✓ No previous dispute pattern
✓ High-value transaction
```

Use actual model-derived feature contributions/importance.

Possible methods:

- Logistic Regression coefficients
- SHAP
- permutation importance
- model-specific feature importance

The LLM should summarize evidence, not manufacture model explanations.

---

# 10. Decision policy

Do not blindly hardcode arbitrary thresholds.

The decision layer should be tunable and validated.

Conceptually:

```text
Low expected value
→ ACCEPT

High expected value + strong evidence
→ CONTEST

Uncertain / conflicting / incomplete
→ ESCALATE
```

Evidence completeness should be a meaningful guardrail.

Example:

```text
P(success) = 0.91
Evidence = 92%
→ likely CONTEST

P(success) = 0.91
Evidence = 40%
→ ESCALATE
```

Do not equate probability with confidence. They are different concepts.

---

# 11. Human-in-the-loop

Do not make the system blindly autonomous.

For CONTEST:

```text
AI prepares
↓
Human reviews
↓
Approve / Edit / Reject
↓
Submit
```

For uncertain cases:

```text
AI
↓
Human queue
↓
Human decision
```

Human overrides must be logged.

---

# 12. Honest exception handling

The system should know when NOT to automate.

Examples:

```text
CASE #1938
Evidence conflict:
Delivery provider record disagrees with merchant record.

→ ESCALATE
```

```text
CASE #2014
Insufficient customer history.

→ ESCALATE
```

This is a feature, not a failure.

---

# 13. Frontend requirements

Build a professional dashboard later.

Pages/components:

1. Dashboard
2. Case list
3. Case detail
4. Risk assessment
5. Evidence packet
6. Agent execution trace
7. Human approval
8. Exception queue
9. Model evaluation
10. Audit trail
11. Outcome analytics

Example case screen:

```text
CASE #1042

Transaction: ₹84,999
Customer: CUST-1042
Dispute reason: Item not received

Contest probability: 91%
Risk: HIGH

Reason codes:
✓ Delivery confirmed
✓ Strong evidence
✓ No previous dispute pattern

Evidence completeness: 92%

Decision: CONTEST

[View Evidence]
[Edit Response]
[Approve & Submit]
[Escalate]

Agent Trace
→ Order retrieved
→ Delivery retrieved
→ Customer history retrieved
→ Evidence packet generated
```

---

# 14. What-if simulation

Potential differentiating feature.

Allow users to change important inputs and recalculate using the actual model.

Examples:

```text
Current:
Delivery confirmed = YES
Risk = actual model output

Change:
Delivery confirmed = NO

→ Recalculate model
→ Show new probability
```

Do not fake these values.

---

# 15. Razorpay integration

Razorpay should be integrated meaningfully.

Potential integration points:

- payment/order context
- disputes
- evidence
- contest/accept actions
- submission

Before implementing specific API behavior, verify current official Razorpay documentation and Buildathon requirements.

Never put Razorpay secret keys in frontend code or GitHub.

Use environment variables.

---

# 16. Tech stack

Preferred:

### Frontend
- React
- Tailwind CSS

### Backend
- Python/FastAPI or Node.js/Express

Choose one backend and stay consistent. FastAPI is a strong option because the ML model is Python-based.

### Database
- PostgreSQL

### ML
- Python
- pandas
- numpy
- scikit-learn
- SHAP
- XGBoost/LightGBM if justified

### AI
- LLM API

Avoid unnecessary:

- microservices
- Kubernetes
- vector databases
- multi-agent swarms
- LLM fine-tuning
- complex infrastructure

The project should be understandable and deployable.

---

# 17. Deployment requirement

Deployment is a core requirement.

Target architecture:

```text
User
 ↓
Deployed React Frontend
 ↓
Deployed Backend API
 ├── PostgreSQL
 ├── ML Risk Engine
 ├── AI Agent
 └── Razorpay APIs
```

Build locally first, then deploy.

Every component should be deployable.

---

# 18. GitHub requirement

Create a public/private GitHub repository depending on the submission requirement.

Recommended structure:

```text
razorpay-sentinel/
├── frontend/
├── backend/
├── ml/
├── data/
├── models/
├── docs/
│   ├── architecture/
│   ├── screenshots/
│   └── demo/
├── tests/
├── .env.example
├── .gitignore
├── README.md
├── docker-compose.yml
└── LICENSE
```

README must eventually contain:

- project overview
- problem statement
- architecture
- tech stack
- ML methodology
- evaluation
- agent workflow
- API documentation
- local setup
- environment variables
- deployment
- demo URL
- screenshots
- limitations

Never commit:

- API keys
- passwords
- database credentials
- Razorpay secrets
- LLM API keys
- private customer data

---

# 19. Development roadmap

## Phase 1 — Risk Intelligence Core

Current phase.

Build:

- development dataset
- baseline model
- stronger model comparison
- held-out evaluation
- calibration
- threshold optimization
- explainability
- `predict_case()` interface

## Phase 2 — Backend + Database

Build:

- FastAPI/Express backend
- PostgreSQL
- case schema
- prediction API
- decision API
- audit schema

## Phase 3 — Decision Agent + Tools

Build:

- decision policy
- tool interface
- evidence completeness
- expected-value calculation
- observable agent trace

## Phase 4 — Contest / Evidence Agent

Build:

- evidence retrieval
- evidence packet
- response generation
- human approval
- submission flow

## Phase 5 — Human Review + Audit

Build:

- review queue
- exception queue
- override handling
- audit trail

## Phase 6 — Frontend

Build the dashboard and case-detail UX around real backend APIs.

## Phase 7 — Razorpay Integration

Integrate verified APIs/capabilities.

## Phase 8 — Testing + Security

Test:

- model
- API
- agent
- authorization
- input validation
- failure handling
- secrets

## Phase 9 — Deployment

Deploy frontend/backend/database and configure secrets.

## Phase 10 — GitHub + Demo + Submission

Finalize:

- README
- architecture
- screenshots
- demo
- metrics
- deployment URL
- repository
- presentation

---

# 20. MVP vs optional features

## MVP

Must work:

- ML risk prediction
- reason codes
- ACCEPT / ESCALATE / CONTEST
- evidence completeness
- contest evidence workflow
- human approval
- audit trail
- held-out evaluation
- deployed application
- GitHub repository

## Strong differentiators

- expected-value decisioning
- what-if simulation
- transparent agent trace
- exception intelligence
- model calibration
- cost-sensitive threshold selection
- outcome feedback loop

## Do NOT build unless time allows

- multi-agent swarm
- sophisticated RAG
- fine-tuned LLM
- real-time online learning
- unnecessary microservices

---

# 21. Demo story

Show three cases.

### Case 1 — ACCEPT

```text
Low contest probability
Weak/poor economics
↓
ACCEPT
↓
Close/refund
↓
Log outcome
```

### Case 2 — CONTEST

```text
High contest probability
Strong evidence
High expected recovery
↓
CONTEST
↓
Evidence agent
↓
Evidence packet
↓
AI response
↓
Human approval
↓
Submit
```

### Case 3 — ESCALATE

```text
Uncertain / conflicting evidence
↓
ESCALATE
↓
Human review
↓
Human decision
↓
Log outcome
```

Then show:

- Precision
- Recall
- F1
- PR-AUC
- automation rate
- escalation rate
- exception count
- outcomes

---

# 22. Core product story

The final pitch should be:

> Existing systems can automate dispute responses. RazorPay Sentinel decides whether a dispute should be fought in the first place, based on predicted contest success, evidence quality, and expected value. When the optimal action is to contest, its evidence agent executes the downstream workflow with human oversight.

Core loop:

```text
PREDICT
   ↓
EXPLAIN
   ↓
EVALUATE
   ↓
DECIDE
   ↓
ACT
   ↓
LEARN
```

---

# 23. Engineering principles

The new AI coding chat must follow these rules:

1. Do not blindly agree with architectural suggestions.
2. Challenge technically weak assumptions.
3. Do not fabricate metrics.
4. Do not fabricate evidence.
5. Do not use an LLM as the primary ML classifier.
6. Keep ML prediction separate from LLM orchestration.
7. Keep decision policy separate from both.
8. Do not expose hidden chain-of-thought.
9. Use human escalation for uncertainty.
10. Do not commit secrets.
11. Build locally before deploying.
12. Keep GitHub updated throughout development.
13. Test each phase before moving on.
14. Prefer simple, explainable architecture.
15. Do not build the frontend around fake data once backend APIs exist.
16. Verify current Razorpay APIs and Buildathon rules before integration.

---

# 24. Current state

We have started Phase 1 with a synthetic development dataset and a baseline Logistic Regression model.

Current development artifacts:

- `data/disputes.csv`
- `ml/train.py`
- `ml/predict.py`
- `ml/decision_policy.py`
- `ml/requirements.txt`
- `models/risk_model.joblib`
- `reports/metrics.json`
- `README.md`

The existing baseline is **not final**.

It is only a starting point.

The next work should improve and validate the ML core before moving to backend development.

---

# 25. FIRST TASK FOR THE NEW CHAT

Do NOT start by rebuilding the whole application.

Read this specification and the architecture diagrams.

Then:

1. Confirm your understanding of RazorPay Sentinel.
2. Identify contradictions or weak assumptions.
3. Inspect the current Phase 1 artifacts if provided.
4. Review the ML target definition.
5. Propose the correct Phase 1 ML experiment plan.
6. Compare baseline Logistic Regression with at least one stronger model.
7. Define how thresholds should be selected.
8. Define the cost-sensitive decision framework.
9. Define the explainability approach.
10. Then implement Phase 1 incrementally.

Do not proceed to the backend until Phase 1 has a defensible evaluation.

---

# 26. Critical distinction to remember

```text
NOT THIS:

Dispute
  ↓
AI
  ↓
Response

BUT THIS:

Payment / Dispute
       ↓
Risk Intelligence
       ↓
Expected Value + Evidence
       ↓
Decision
 ┌─────┼─────┐
 ↓     ↓     ↓
ACCEPT ESCALATE CONTEST
             ↓
       Dispute Agent
             ↓
          Execute
```

**The decision engine is the product.**

**The dispute responder is an execution capability inside the product.**
