// ============================================================
// Hash-based Client-side Router
// ============================================================

import * as DB from './db.js';
import { renderDashboard } from './dashboard.js';
import { hydrateUnit } from './progress.js';

const contentEl = () => document.getElementById('content');
const breadcrumbEl = () => document.getElementById('breadcrumb');

// Unit metadata
export const UNITS = [
  { id: 1, name: 'Foundations of Language Modeling', weeks: [1, 2], color: 'violet', icon: '\u{1F4D6}' },
  { id: 2, name: 'Scaling Laws & L(N)', weeks: [3], color: 'cyan', icon: '\u{1F4CA}' },
  { id: 3, name: 'Efficient Architectures', weeks: [4, 5, 6], color: 'magenta', icon: '\u{1F9E9}' },
  { id: 4, name: 'Tokenization', weeks: [7], color: 'amber', icon: '\u{1F524}' },
  { id: 5, name: 'Optimization', weeks: [8, 9, 10], color: 'lime', icon: '\u{26A1}' },
  { id: 6, name: 'Quantization & Compression', weeks: [11, 12, 13], color: 'red', icon: '\u{1F5DC}' },
  { id: 7, name: 'Evaluation Methods', weeks: [14], color: 'blue', icon: '\u{1F3AF}' },
  { id: 8, name: 'Systems & Performance', weeks: [15], color: 'orange', icon: '\u{1F680}' },
  { id: 9, name: 'Integration & Strategy', weeks: [16], color: 'violet', icon: '\u{1F3C6}' },
];

// Unit content cache
const _cache = {};

// Parse current hash into route object
export function parseRoute() {
  const hash = window.location.hash.slice(1) || '/dashboard';
  const parts = hash.split('/').filter(Boolean);

  if (parts[0] === 'unit' && parts[1]) {
    const unitId = parseInt(parts[1], 10);
    const weekId = parts[2] === 'week' && parts[3] ? parseInt(parts[3], 10) : null;
    return { view: 'unit', unitId, weekId };
  }
  if (parts[0] === 'flashcards') return { view: 'flashcards' };
  if (parts[0] === 'settings') return { view: 'settings' };
  return { view: 'dashboard' };
}

// Navigate to a hash route
export function navigate(hash) {
  window.location.hash = hash;
}

// Set breadcrumb
function setBreadcrumb(parts) {
  const el = breadcrumbEl();
  if (!el) return;
  el.innerHTML = parts.map((p, i) => {
    if (i < parts.length - 1) {
      return `<a href="${p.href}">${p.label}</a><span class="sep">/</span>`;
    }
    return `<span>${p.label}</span>`;
  }).join('');
}

// Update sidebar active state
function updateSidebarActive(route) {
  document.querySelectorAll('.nav-item').forEach(el => el.classList.remove('active'));
  document.querySelectorAll('.nav-week').forEach(el => el.classList.remove('active'));
  document.querySelectorAll('.nav-weeks').forEach(el => el.classList.remove('expanded'));

  if (route.view === 'dashboard') {
    document.querySelector('[data-nav="dashboard"]')?.classList.add('active');
  } else if (route.view === 'flashcards') {
    document.querySelector('[data-nav="flashcards"]')?.classList.add('active');
  } else if (route.view === 'settings') {
    document.querySelector('[data-nav="settings"]')?.classList.add('active');
  } else if (route.view === 'unit') {
    const unitNav = document.querySelector(`[data-nav="unit-${route.unitId}"]`);
    if (unitNav) {
      unitNav.classList.add('active');
      const weeks = unitNav.nextElementSibling;
      if (weeks) weeks.classList.add('expanded');
    }
    if (route.weekId) {
      document.querySelector(`[data-nav="week-${route.weekId}"]`)?.classList.add('active');
    }
  }
}

// Load unit content
async function loadUnit(unitId, weekId) {
  const unit = UNITS.find(u => u.id === unitId);
  if (!unit) {
    contentEl().innerHTML = '<div class="coming-soon"><div class="icon-lock">?</div><h2>Unit not found</h2></div>';
    return;
  }

  const crumbs = [{ label: 'Dashboard', href: '#/dashboard' }];
  crumbs.push({ label: `Unit ${unitId}`, href: `#/unit/${unitId}` });
  if (weekId) crumbs.push({ label: `Week ${weekId}`, href: `#/unit/${unitId}/week/${weekId}` });
  setBreadcrumb(crumbs);

  // Try loading the unit HTML fragment
  if (!_cache[unitId]) {
    try {
      const resp = await fetch(`units/unit-${unitId}.html`);
      if (resp.ok) {
        _cache[unitId] = await resp.text();
      }
    } catch (e) {
      // File not available
    }
  }

  if (_cache[unitId]) {
    contentEl().innerHTML = _cache[unitId];
    await hydrateUnit(unitId);
    // Scroll to week if specified
    if (weekId) {
      const weekEl = document.querySelector(`[data-week="${weekId}"]`);
      if (weekEl) {
        weekEl.scrollIntoView({ behavior: 'smooth', block: 'start' });
      }
    }
  } else {
    // Coming soon placeholder
    const weekList = unit.weeks.map(w => `<li>Week ${w}</li>`).join('');
    contentEl().innerHTML = `
      <div class="coming-soon">
        <div class="icon-lock">${unit.icon}</div>
        <h2>Unit ${unitId}: ${unit.name}</h2>
        <p>This unit is coming soon.</p>
        <div style="text-align: left; max-width: 300px; margin: var(--space-lg) auto;">
          <p style="color: var(--pg-text-muted); margin-bottom: var(--space-sm);">Covers:</p>
          <ul style="color: var(--pg-text-dim); list-style: none;">${weekList}</ul>
        </div>
      </div>
    `;
  }
}

// Load flashcards view
async function loadFlashcards() {
  setBreadcrumb([
    { label: 'Dashboard', href: '#/dashboard' },
    { label: 'Flashcards', href: '#/flashcards' }
  ]);

  const today = new Date().toISOString().slice(0, 10);
  const due = await DB.getCardsForReview(today);
  const all = await DB.getAllFlashcards();

  if (all.length === 0) {
    contentEl().innerHTML = `
      <div class="coming-soon">
        <div class="icon-lock">\u{1F0CF}</div>
        <h2>Flashcards</h2>
        <p>No flashcards yet. Complete Unit 1 to unlock your first deck.</p>
      </div>
    `;
    return;
  }

  contentEl().innerHTML = `
    <div class="content-inner">
      <div class="unit-header">
        <h1>Flashcard Review</h1>
        <div class="unit-meta">
          <span class="badge badge-violet">${due.length} due today</span>
          <span class="badge badge-cyan">${all.length} total cards</span>
        </div>
      </div>
      <div id="flashcard-area">
        ${due.length > 0 ? renderFlashcard(due, 0) : '<p style="color: var(--pg-text-muted);">No cards due for review today. Check back tomorrow.</p>'}
      </div>
    </div>
  `;

  if (due.length > 0) {
    initFlashcardReview(due);
  }
}

function renderFlashcard(cards, idx) {
  const card = cards[idx];
  return `
    <div class="card" id="fc-card" style="min-height: 200px; cursor: pointer; text-align: center; display: flex; flex-direction: column; align-items: center; justify-content: center;">
      <p style="color: var(--pg-text-dim); font-size: var(--font-size-xs); margin-bottom: var(--space-md);">Card ${idx + 1} / ${cards.length}</p>
      <div id="fc-front" style="font-size: var(--font-size-lg);">${card.front}</div>
      <div id="fc-back" style="display: none; font-size: var(--font-size-md); color: var(--pg-accent); margin-top: var(--space-lg);">${card.back}</div>
      <p id="fc-hint" style="color: var(--pg-text-dim); font-size: var(--font-size-xs); margin-top: var(--space-lg);">Click to reveal</p>
      <div id="fc-rating" style="display: none; margin-top: var(--space-lg); display: none; gap: var(--space-sm);">
        <button class="btn btn-ghost" data-quality="1">Again</button>
        <button class="btn btn-ghost" data-quality="3">Good</button>
        <button class="btn btn-primary" data-quality="5">Easy</button>
      </div>
    </div>
  `;
}

function initFlashcardReview(cards) {
  let idx = 0;
  let revealed = false;

  const area = document.getElementById('flashcard-area');
  if (!area) return;

  area.addEventListener('click', async (e) => {
    const card = cards[idx];
    const quality = e.target.dataset?.quality;

    if (quality) {
      await DB.updateCardReview(card.cardId, parseInt(quality, 10));
      idx++;
      if (idx >= cards.length) {
        area.innerHTML = '<div class="card" style="text-align: center; padding: var(--space-2xl);"><h3 style="color: var(--pg-lime);">Review complete!</h3><p style="color: var(--pg-text-muted); margin-top: var(--space-md);">All cards reviewed for today.</p></div>';
        return;
      }
      revealed = false;
      area.innerHTML = renderFlashcard(cards, idx);
      return;
    }

    if (!revealed) {
      const back = document.getElementById('fc-back');
      const hint = document.getElementById('fc-hint');
      const rating = document.getElementById('fc-rating');
      if (back) back.style.display = 'block';
      if (hint) hint.style.display = 'none';
      if (rating) { rating.style.display = 'flex'; rating.style.justifyContent = 'center'; }
      revealed = true;
    }
  });
}

// Load settings view
async function loadSettings() {
  setBreadcrumb([
    { label: 'Dashboard', href: '#/dashboard' },
    { label: 'Settings', href: '#/settings' }
  ]);

  const overall = await DB.getOverallProgress();
  contentEl().innerHTML = `
    <div class="content-inner">
      <div class="unit-header">
        <h1>Settings</h1>
      </div>
      <div style="display: flex; flex-direction: column; gap: var(--space-lg);">
        <div class="card">
          <div class="card-header"><h3 class="card-title">Data</h3></div>
          <p style="color: var(--pg-text-muted); margin-bottom: var(--space-md);">${overall.completed} items completed, ${overall.total} total tracked</p>
          <div style="display: flex; gap: var(--space-sm); flex-wrap: wrap;">
            <button class="btn btn-ghost" id="btn-export">Export Data</button>
            <button class="btn btn-ghost" id="btn-import">Import Data</button>
            <button class="btn btn-ghost" id="btn-reset" style="color: var(--pg-danger);">Reset All Data</button>
          </div>
          <input type="file" id="import-file" accept=".json" style="display: none;">
        </div>
      </div>
    </div>
  `;

  document.getElementById('btn-export')?.addEventListener('click', async () => {
    const data = await DB.exportAll();
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = `pgolf-curriculum-backup-${new Date().toISOString().slice(0, 10)}.json`;
    a.click();
  });

  document.getElementById('btn-import')?.addEventListener('click', () => {
    document.getElementById('import-file')?.click();
  });

  document.getElementById('import-file')?.addEventListener('change', async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    const text = await file.text();
    const data = JSON.parse(text);
    await DB.importAll(data);
    alert('Data imported successfully. Reloading...');
    window.location.reload();
  });

  document.getElementById('btn-reset')?.addEventListener('click', async () => {
    if (confirm('This will permanently delete all your progress, notes, and saved code. Continue?')) {
      await DB.clearAll();
      alert('All data has been reset. Reloading...');
      window.location.reload();
    }
  });
}

// Main route handler
async function handleRoute() {
  const route = parseRoute();
  updateSidebarActive(route);
  await DB.setSetting('lastVisited', window.location.hash);

  const scroll = document.querySelector('.content-scroll');
  if (scroll) scroll.scrollTop = 0;

  switch (route.view) {
    case 'dashboard':
      setBreadcrumb([{ label: 'Dashboard', href: '#/dashboard' }]);
      await renderDashboard(contentEl());
      break;
    case 'unit':
      await loadUnit(route.unitId, route.weekId);
      break;
    case 'flashcards':
      await loadFlashcards();
      break;
    case 'settings':
      await loadSettings();
      break;
    default:
      setBreadcrumb([{ label: 'Dashboard', href: '#/dashboard' }]);
      await renderDashboard(contentEl());
  }
}

// Initialize router
export async function initRouter() {
  await DB.ready;

  // Restore last visited route if no hash
  if (!window.location.hash) {
    const last = await DB.getSetting('lastVisited', '#/dashboard');
    window.location.hash = last;
  }

  window.addEventListener('hashchange', handleRoute);
  await handleRoute();
}
