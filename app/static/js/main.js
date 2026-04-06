/**
 * main.js — TruePresence Dashboard
 *
 * Handles:
 *  - Workflow step bar highlighting (steps 1-4)
 *  - Script launch + real-time status polling
 *  - Console output with line cap (300 lines)
 *  - Desktop-hint banner for GUI scripts
 *  - Active card glow for currently running step
 *  - Toast notification system
 *  - Stats card refresh from /api/stats
 *  - Ctrl+Shift+S to stop process
 */

'use strict';

// ── Script → workflow mapping ─────────────────────────────────────────────
// Maps script name → { step number, card ID, shows native window? }
const SCRIPT_META = {
  'get_faces':         { step: 1, cardId: 'card-get-faces', nativeWindow: true  },
  'extract_features':  { step: 2, cardId: 'card-extract',   nativeWindow: false },
  'features_extraction_to_csv.py': { step: 2, cardId: 'card-extract', nativeWindow: false },
  'attendance':        { step: 3, cardId: 'card-attendance', nativeWindow: true  },
};

// ── Toast system ──────────────────────────────────────────────────────────
const toastContainer = document.getElementById('toast-container');

function showToast(message, type = 'info', durationMs = 3500) {
  if (!toastContainer) return;
  const icons = { success: 'fa-check-circle', error: 'fa-exclamation-circle', info: 'fa-info-circle', warning: 'fa-exclamation-triangle' };
  const toast = document.createElement('div');
  toast.className = `toast ${type}`;
  toast.innerHTML = `<i class="fas ${icons[type] || icons.info}"></i><span>${message}</span>`;
  toastContainer.appendChild(toast);
  setTimeout(() => {
    toast.classList.add('fade-out');
    setTimeout(() => toast.remove(), 400);
  }, durationMs);
}

// ── Stats refresh ─────────────────────────────────────────────────────────
function refreshStats() {
  fetch('/api/stats')
    .then(r => r.json())
    .then(data => {
      setEl('stat-total',   data.total   ?? '—');
      setEl('stat-today',   data.today   ?? '—');
      setEl('stat-valid',   data.valid   ?? '—');
      setEl('stat-invalid', data.invalid ?? '—');
      const dot = document.getElementById('db-status-dot');
      const lbl = document.getElementById('db-status-label');
      if (dot) dot.className = `db-dot ${data.db_ok ? 'ok' : 'err'}`;
      if (lbl) lbl.textContent = data.db_ok ? 'DB Connected' : 'DB Unavailable';
    })
    .catch(() => { /* silently degrade */ });
}

function setEl(id, value) {
  const el = document.getElementById(id);
  if (el) el.textContent = value;
}

// ── DOM refs ──────────────────────────────────────────────────────────────
const getFacesBtn        = document.getElementById('get-faces-btn');
const extractFeaturesBtn = document.getElementById('extract-features-btn');
const attendanceBtn      = document.getElementById('attendance-btn');
const stopBtn            = document.getElementById('stop-btn');
const statusDot          = document.querySelector('.status-dot');
const statusText         = document.getElementById('status-text');
const currentProcessEl   = document.getElementById('current-process');
const outputConsole      = document.getElementById('output-console');
const desktopHint        = document.getElementById('desktop-hint');

let statusInterval = null;
let _activeScript  = null;
const MAX_LINES    = 300;

// ── Workflow UI helpers ───────────────────────────────────────────────────

function setWorkflowStep(scriptName, active) {
  const meta = SCRIPT_META[scriptName];
  if (!meta) return;

  // Highlight the right step pill
  document.querySelectorAll('.wf-step').forEach(el => el.classList.remove('active'));
  const stepEl = document.getElementById(`wf-step-${meta.step}`);
  if (stepEl && active) stepEl.classList.add('active');

  // Glow the active card
  document.querySelectorAll('.wf-card').forEach(el => el.classList.remove('active-card'));
  const cardEl = document.getElementById(meta.cardId);
  if (cardEl && active) cardEl.classList.add('active-card');

  // Show desktop hint for GUI scripts
  if (desktopHint) {
    desktopHint.style.display = (active && meta.nativeWindow) ? 'flex' : 'none';
  }
}

function markStepDone(scriptName) {
  const meta = SCRIPT_META[scriptName];
  if (!meta) return;
  const stepEl = document.getElementById(`wf-step-${meta.step}`);
  if (stepEl) {
    stepEl.classList.remove('active');
    stepEl.classList.add('done');
    // Replace number with checkmark
    const numEl = stepEl.querySelector('.wf-num');
    if (numEl) numEl.innerHTML = '<i class="fas fa-check" style="font-size:0.7rem"></i>';
  }
}

// ── Button helpers ────────────────────────────────────────────────────────
function setButtonsEnabled(enabled) {
  const btns = [getFacesBtn, extractFeaturesBtn, attendanceBtn].filter(Boolean);
  btns.forEach(btn => {
    btn.disabled      = !enabled;
    btn.style.opacity = enabled ? '1' : '0.5';
    btn.style.cursor  = enabled ? 'pointer' : 'not-allowed';
  });
}

// ── Console ───────────────────────────────────────────────────────────────
function trimConsole(text) {
  const lines = text.split('\n');
  return lines.length > MAX_LINES
    ? '...[output trimmed to last 300 lines]...\n' + lines.slice(-MAX_LINES).join('\n')
    : text;
}

// ── Script runner ─────────────────────────────────────────────────────────
function runScript(scriptName) {
  _activeScript = scriptName;
  setButtonsEnabled(false);
  if (outputConsole) outputConsole.textContent = `▶ Starting ${scriptName}...\n`;
  setWorkflowStep(scriptName, true);

  fetch('/run_script', {
    method:  'POST',
    headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
    body:    'script=' + encodeURIComponent(scriptName),
  })
  .then(r => r.json())
  .then(data => {
    if (data.status === 'success') {
      showToast(`Started: ${scriptName}`, 'success');
      const meta = SCRIPT_META[scriptName];
      if (meta?.nativeWindow) {
        showToast('Native window opened on your desktop — check your Dock/taskbar', 'info', 6000);
      }
      startStatusPolling();
    } else {
      showToast(data.message || 'Failed to start process', 'error', 5000);
      setButtonsEnabled(true);
      setWorkflowStep(scriptName, false);
    }
  })
  .catch(err => {
    showToast('Network error — could not reach server', 'error');
    setButtonsEnabled(true);
    setWorkflowStep(scriptName, false);
    console.error('runScript error:', err);
  });
}

// ── Stop ──────────────────────────────────────────────────────────────────
function stopProcess() {
  fetch('/stop_script', { method: 'POST' })
  .then(r => r.json())
  .then(data => {
    const ok = data.status === 'success';
    showToast(ok ? 'Process stopped' : (data.message || 'Stop failed'), ok ? 'info' : 'error');
  })
  .catch(() => showToast('Network error — could not stop process', 'error'));
}

// ── Status polling ────────────────────────────────────────────────────────
function startStatusPolling() {
  if (statusInterval) clearInterval(statusInterval);
  checkStatus();
  statusInterval = setInterval(checkStatus, 1200);
}

function checkStatus() {
  fetch('/status')
  .then(r => r.json())
  .then(updateStatusUI)
  .catch(() => { /* suppress polling noise */ });
}

function updateStatusUI(data) {
  const running = Boolean(data.running);

  // Update status dot + text
  if (statusDot) statusDot.className = `status-dot ${running ? 'active' : 'inactive'}`;
  if (statusText) statusText.textContent = running ? 'Running…' : 'Ready — click an action to begin';

  // Script name label
  if (currentProcessEl) {
    if (data.script) {
      currentProcessEl.textContent = running
        ? `▶ ${data.script}`
        : `✓ ${data.script} (completed)`;
    } else {
      currentProcessEl.textContent = '';
    }
  }

  // Stop button
  if (stopBtn) stopBtn.style.display = running ? 'inline-flex' : 'none';
  setButtonsEnabled(!running);

  // Workflow step highlighting
  if (running && data.script) setWorkflowStep(data.script, true);
  if (!running && _activeScript) {
    markStepDone(_activeScript);
    setWorkflowStep(_activeScript, false);
    document.querySelectorAll('.wf-card').forEach(el => el.classList.remove('active-card'));
    if (desktopHint) desktopHint.style.display = 'none';
    _activeScript = null;
    refreshStats();
  }

  // Output console
  if (outputConsole && data.output) {
    outputConsole.textContent = trimConsole(data.output);
    outputConsole.scrollTop = outputConsole.scrollHeight;
  }

  // Stop polling when process ends
  if (!running && statusInterval) {
    clearInterval(statusInterval);
    statusInterval = null;
  }
}

// ── Event listeners ───────────────────────────────────────────────────────
if (getFacesBtn)        getFacesBtn.addEventListener('click',        () => runScript('get_faces'));
if (extractFeaturesBtn) extractFeaturesBtn.addEventListener('click', () => runScript('extract_features'));
if (attendanceBtn)      attendanceBtn.addEventListener('click',      () => runScript('attendance'));
if (stopBtn)            stopBtn.addEventListener('click',            stopProcess);

document.addEventListener('keydown', e => {
  if (e.ctrlKey && e.shiftKey && e.key === 'S') { e.preventDefault(); stopProcess(); }
});

// ── Init ──────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  refreshStats();
  checkStatus();
  setInterval(refreshStats, 30_000);
});
