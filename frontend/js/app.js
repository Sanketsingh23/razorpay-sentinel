// RazorPay Sentinel Dashboard Application Logic

let activeCase = null;
let activePrediction = null;
let activeDecision = null;
let activeEvidence = null;

document.addEventListener('DOMContentLoaded', async () => {
  setupEventListeners();
  await checkSystemHealth();
  await loadPresetSamples();
  await refreshCaseList();
});

// Setup DOM Event Listeners
function setupEventListeners() {
  document.getElementById('btn-run-predict').addEventListener('click', handleRunPredict);
  document.getElementById('btn-run-decide').addEventListener('click', handleRunDecide);
  document.getElementById('btn-run-evidence').addEventListener('click', handleRunEvidence);
  document.getElementById('btn-refresh-audit').addEventListener('click', () => {
    if (activeCase) loadCaseAudit(activeCase.case_id);
  });

  // Modal events
  const modal = document.getElementById('modal-create-case');
  document.getElementById('btn-new-case').addEventListener('click', () => modal.classList.remove('hidden'));
  document.getElementById('modal-close-btn').addEventListener('click', () => modal.classList.add('hidden'));
  document.getElementById('btn-cancel-modal').addEventListener('click', () => modal.classList.add('hidden'));
  document.getElementById('form-create-case').addEventListener('submit', handleCreateCaseSubmit);

  // Case dropdown change
  document.getElementById('case-selector-dropdown').addEventListener('change', async (e) => {
    const caseId = e.target.value;
    if (caseId) {
      await loadCaseById(caseId);
    }
  });
}

// 1. Health Check
async function checkSystemHealth() {
  const statusDot = document.querySelector('.status-dot');
  const statusText = document.getElementById('system-status-text');
  try {
    const res = await fetch('/health');
    const data = await res.json();
    if (data.status === 'ok') {
      statusDot.className = 'status-dot online';
      statusText.textContent = `Online (DB: ${data.database}, Model: ${data.model})`;
    } else {
      statusDot.className = 'status-dot';
      statusText.textContent = `Degraded (DB: ${data.database}, Model: ${data.model})`;
    }
  } catch (err) {
    statusDot.className = 'status-dot error';
    statusText.textContent = 'Backend Offline';
  }
}

// 2. Load Preset Demonstration Cases
async function loadPresetSamples() {
  try {
    const res = await fetch('/cases/presets/samples');
    const presets = await res.json();
    const container = document.getElementById('preset-grid');
    container.innerHTML = '';

    presets.forEach(preset => {
      const card = document.createElement('div');
      card.className = 'preset-card';
      card.innerHTML = `
        <div class="preset-card-header">
          <span class="preset-title">${preset.title}</span>
          <span class="badge badge-case">${preset.preset_id.toUpperCase()}</span>
        </div>
        <p class="preset-desc">${preset.description}</p>
        <div class="preset-card-footer">
          <span>Click to Load Case &rarr;</span>
          <span>INR ${preset.data.transaction_amount.toLocaleString()}</span>
        </div>
      `;
      card.addEventListener('click', async () => {
        await createAndSelectCase(preset.data);
      });
      container.appendChild(card);
    });
  } catch (err) {
    console.error('Failed to load presets:', err);
  }
}

// 3. Refresh Case Dropdown List
async function refreshCaseList() {
  try {
    const res = await fetch('/cases?limit=30');
    const cases = await res.json();
    const dropdown = document.getElementById('case-selector-dropdown');
    dropdown.innerHTML = '<option value="">-- Choose an existing dispute case --</option>';

    cases.forEach(c => {
      const opt = document.createElement('option');
      opt.value = c.case_id;
      opt.textContent = `${c.case_id} — ₹${c.transaction_amount.toLocaleString()} (${c.dispute_reason}, ${c.status})`;
      dropdown.appendChild(opt);
    });
  } catch (err) {
    console.error('Failed to refresh cases list:', err);
  }
}

// 4. Create Case Submit
async function handleCreateCaseSubmit(e) {
  e.preventDefault();
  const caseIdInput = document.getElementById('input-case-id').value.trim();
  const payload = {
    case_id: caseIdInput || undefined,
    transaction_amount: parseFloat(document.getElementById('input-amount').value),
    dispute_reason: document.getElementById('input-reason').value,
    delivery_confirmed: document.getElementById('input-delivery').checked,
    customer_order_count: parseInt(document.getElementById('input-orders').value) || 1,
    previous_disputes: parseInt(document.getElementById('input-prev-disputes').value) || 0,
    previous_refunds: parseInt(document.getElementById('input-prev-refunds').value) || 0,
    refund_amount_ratio: parseFloat(document.getElementById('input-refund-ratio').value) || 0.0,
    evidence_items_available: parseInt(document.getElementById('input-evid-avail').value) || 4,
    evidence_items_missing: parseInt(document.getElementById('input-evid-miss').value) || 2,
  };

  document.getElementById('modal-create-case').classList.add('hidden');
  await createAndSelectCase(payload);
}

// Helper: POST /cases and set as active
async function createAndSelectCase(payload) {
  try {
    const res = await fetch('/cases', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    if (!res.ok) {
      const err = await res.json();
      alert(`Error creating case: ${err.detail || res.statusText}`);
      return;
    }
    const created = await res.json();
    await refreshCaseList();
    await loadCaseById(created.case_id);
  } catch (err) {
    console.error('Failed to create case:', err);
  }
}

// 5. Load Case Details by ID
async function loadCaseById(caseId) {
  try {
    const res = await fetch(`/cases/${caseId}`);
    if (!res.ok) return;
    const caseData = await res.json();
    setActiveCase(caseData);
  } catch (err) {
    console.error('Failed to load case:', err);
  }
}

// Render Active Case UI State
function setActiveCase(caseData) {
  activeCase = caseData;
  activePrediction = caseData.latest_prediction || null;
  activeDecision = caseData.latest_decision || null;
  activeEvidence = caseData.latest_evidence_packet || null;

  // Header Banner
  document.getElementById('active-case-id').textContent = caseData.case_id;
  document.getElementById('active-case-status').textContent = caseData.status;
  document.getElementById('active-case-date').textContent = new Date(caseData.created_at).toLocaleString();

  // Facts Grid
  document.getElementById('fact-amount').textContent = `₹${caseData.transaction_amount.toLocaleString()}`;
  document.getElementById('fact-reason').textContent = caseData.dispute_reason;
  document.getElementById('fact-delivery').textContent = caseData.delivery_confirmed ? '✓ Confirmed' : '✗ Unconfirmed';
  document.getElementById('fact-delivery').style.color = caseData.delivery_confirmed ? '#34d399' : '#f87171';
  document.getElementById('fact-orders').textContent = `${caseData.customer_order_count} (Avg: ₹${caseData.customer_avg_order_value.toLocaleString()})`;
  document.getElementById('fact-history').textContent = `${caseData.previous_refunds} refunds / ${caseData.previous_disputes} disputes`;
  document.getElementById('fact-evidence').textContent = `${caseData.evidence_items_available} avail / ${caseData.evidence_items_missing} miss (${(caseData.evidence_completeness * 100).toFixed(0)}%)`;

  // Buttons & Stages setup
  document.getElementById('btn-run-predict').disabled = false;
  
  if (activePrediction) {
    renderPrediction(activePrediction);
    document.getElementById('btn-run-decide').disabled = false;
  } else {
    resetPredictionUI();
    document.getElementById('btn-run-decide').disabled = true;
  }

  if (activeDecision) {
    renderDecision(activeDecision);
    if (activeDecision.action === 'CONTEST') {
      document.getElementById('btn-run-evidence').disabled = false;
    } else {
      document.getElementById('btn-run-evidence').disabled = true;
    }
  } else {
    resetDecisionUI();
    document.getElementById('btn-run-evidence').disabled = true;
  }

  if (activeEvidence) {
    renderEvidence(activeEvidence);
  } else {
    resetEvidenceUI();
  }

  loadCaseAudit(caseData.case_id);
}

// 6. Predict Execution
async function handleRunPredict() {
  if (!activeCase) return;
  const btn = document.getElementById('btn-run-predict');
  btn.disabled = true;
  btn.textContent = 'Running Model...';

  try {
    const res = await fetch(`/cases/${activeCase.case_id}/predict`, { method: 'POST' });
    if (!res.ok) {
      const err = await res.json();
      alert(`Prediction Error: ${err.detail}`);
      return;
    }
    const pred = await res.json();
    activePrediction = pred;
    renderPrediction(pred);
    document.getElementById('btn-run-decide').disabled = false;
    loadCaseAudit(activeCase.case_id);
  } catch (err) {
    console.error(err);
  } finally {
    btn.disabled = false;
    btn.textContent = 'Run Risk Model';
  }
}

function renderPrediction(pred) {
  document.getElementById('risk-empty-state').classList.add('hidden');
  document.getElementById('risk-results').classList.remove('hidden');

  const probPct = (pred.contest_probability * 100).toFixed(1);
  document.getElementById('risk-prob-value').textContent = `${probPct}%`;
  document.getElementById('risk-prob-bar').style.width = `${probPct}%`;

  const badge = document.getElementById('risk-level-badge');
  badge.textContent = `RISK LEVEL: ${pred.risk_level}`;
  badge.className = `badge badge-${pred.risk_level === 'HIGH' ? 'success' : pred.risk_level === 'MEDIUM' ? 'warning' : 'danger'}`;

  // Reason code tags
  const tagsContainer = document.getElementById('reason-codes-tags');
  tagsContainer.innerHTML = '';
  (pred.reason_codes || []).forEach(rc => {
    const tag = document.createElement('span');
    tag.className = 'reason-tag';
    tag.textContent = rc;
    tagsContainer.appendChild(tag);
  });

  // Positive & negative factors
  const posList = document.getElementById('positive-factors-list');
  posList.innerHTML = '';
  (pred.positive_factors || []).forEach(pf => {
    const li = document.createElement('li');
    li.textContent = `+ ${pf}`;
    posList.appendChild(li);
  });

  const negList = document.getElementById('negative-factors-list');
  negList.innerHTML = '';
  (pred.negative_factors || []).forEach(nf => {
    const li = document.createElement('li');
    li.textContent = `- ${nf}`;
    negList.appendChild(li);
  });
}

function resetPredictionUI() {
  document.getElementById('risk-empty-state').classList.remove('hidden');
  document.getElementById('risk-results').classList.add('hidden');
}

// 7. Decision Execution
async function handleRunDecide() {
  if (!activeCase) return;
  const btn = document.getElementById('btn-run-decide');
  btn.disabled = true;
  btn.textContent = 'Evaluating Policy...';

  try {
    const res = await fetch(`/cases/${activeCase.case_id}/decide`, { method: 'POST' });
    if (!res.ok) {
      const err = await res.json();
      alert(`Decision Error: ${err.detail}`);
      return;
    }
    const dec = await res.json();
    activeDecision = dec;
    renderDecision(dec);
    
    if (dec.action === 'CONTEST') {
      document.getElementById('btn-run-evidence').disabled = false;
    } else {
      document.getElementById('btn-run-evidence').disabled = true;
    }
    loadCaseAudit(activeCase.case_id);
  } catch (err) {
    console.error(err);
  } finally {
    btn.disabled = false;
    btn.textContent = 'Run Decision Policy';
  }
}

function renderDecision(dec) {
  document.getElementById('decision-empty-state').classList.add('hidden');
  document.getElementById('decision-results').classList.remove('hidden');

  const badge = document.getElementById('decision-action-badge');
  badge.textContent = dec.action;
  badge.className = `action-badge action-${dec.action}`;

  document.getElementById('dec-expected-recovery').textContent = `₹${Math.round(dec.expected_recovery).toLocaleString()}`;
  document.getElementById('dec-net-value').textContent = `₹${Math.round(dec.expected_value).toLocaleString()}`;
  
  const guardrailStatus = document.getElementById('dec-guardrail-status');
  if (dec.guardrail_triggered) {
    guardrailStatus.textContent = '⚠️ GUARDRAIL TRIGGERED (Forced Escalate)';
    guardrailStatus.style.color = '#fbbf24';
  } else {
    guardrailStatus.textContent = '✓ Standard Policy Rules Passed';
    guardrailStatus.style.color = '#34d399';
  }

  const list = document.getElementById('decision-reasoning-list');
  list.innerHTML = '';
  (dec.reasoning || []).forEach(r => {
    const li = document.createElement('li');
    li.textContent = r;
    list.appendChild(li);
  });
}

function resetDecisionUI() {
  document.getElementById('decision-empty-state').classList.remove('hidden');
  document.getElementById('decision-results').classList.add('hidden');
}

// 8. Evidence Execution
async function handleRunEvidence() {
  if (!activeCase) return;
  const btn = document.getElementById('btn-run-evidence');
  btn.disabled = true;
  btn.textContent = 'Assembling Evidence...';

  try {
    const res = await fetch(`/cases/${activeCase.case_id}/evidence`, { method: 'POST' });
    if (!res.ok) {
      const err = await res.json();
      alert(`Evidence Workflow Error: ${err.detail}`);
      return;
    }
    const evid = await res.json();
    activeEvidence = evid;
    renderEvidence(evid);
    loadCaseAudit(activeCase.case_id);
  } catch (err) {
    console.error(err);
  } finally {
    btn.disabled = false;
    btn.textContent = 'Generate Evidence Packet';
  }
}

function renderEvidence(evid) {
  document.getElementById('evidence-empty-state').classList.add('hidden');
  document.getElementById('evidence-results').classList.remove('hidden');

  const statusBadge = document.getElementById('evid-packet-status');
  statusBadge.textContent = evid.status;
  statusBadge.className = `badge badge-${evid.status === 'READY_FOR_REVIEW' ? 'success' : 'warning'}`;

  document.getElementById('evid-completeness-val').textContent = `${(evid.evidence_completeness * 100).toFixed(1)}%`;

  // Render 6 Evidence Items
  const grid = document.getElementById('evidence-cards-grid');
  grid.innerHTML = '';
  (evid.evidence_items || []).forEach(item => {
    const card = document.createElement('div');
    card.className = 'evidence-item-card';
    card.innerHTML = `
      <div class="evid-card-header">
        <span class="evid-code">${item.evidence_id}</span>
        <span class="badge badge-${item.status === 'AVAILABLE' ? 'success' : item.status === 'MISSING' ? 'danger' : 'warning'}">${item.status}</span>
      </div>
      <div class="evid-title">${item.type.replace(/_/g, ' ').toUpperCase()}</div>
      <div class="evid-summary">${item.summary}</div>
      <div class="text-subtle" style="font-size:0.7rem;">Source: ${item.source} | Relevance: ${item.relevance}</div>
    `;
    grid.appendChild(card);
  });

  // Statement text
  document.getElementById('draft-statement-text').textContent = evid.response_draft.statement || 'No statement drafted.';

  // Claims Table
  const claimsTable = document.getElementById('claims-table');
  claimsTable.innerHTML = '';
  (evid.response_draft.claims || []).forEach(c => {
    const row = document.createElement('div');
    row.className = 'claim-row';
    row.innerHTML = `
      <span class="claim-source-badge">${c.source_evidence_id}</span>
      <span>${c.claim}</span>
    `;
    claimsTable.appendChild(row);
  });
}

function resetEvidenceUI() {
  document.getElementById('evidence-empty-state').classList.remove('hidden');
  document.getElementById('evidence-results').classList.add('hidden');
}

// 9. Load Audit Events
async function loadCaseAudit(caseId) {
  try {
    const res = await fetch(`/cases/${caseId}/audit`);
    if (!res.ok) return;
    const events = await res.json();
    const timeline = document.getElementById('audit-timeline');
    
    if (events.length === 0) {
      timeline.innerHTML = '<p class="text-subtle">No audit events recorded yet.</p>';
      return;
    }

    timeline.innerHTML = '';
    events.forEach(evt => {
      const item = document.createElement('div');
      item.className = 'timeline-event';
      item.innerHTML = `
        <div class="timeline-event-header">
          <span class="timeline-type">${evt.event_type}</span>
          <span class="timeline-time">${new Date(evt.event_timestamp).toLocaleTimeString()}</span>
        </div>
        <div class="timeline-meta">${JSON.stringify(evt.metadata_payload, null, 2)}</div>
      `;
      timeline.appendChild(item);
    });
  } catch (err) {
    console.error('Failed to load audit:', err);
  }
}
