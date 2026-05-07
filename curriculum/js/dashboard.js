// ============================================================
// Dashboard View
// ============================================================

import * as DB from './db.js';
import { UNITS, navigate } from './router.js';

// Unit color mapping
const UNIT_COLORS = {
  violet: '#6c5ce7',
  cyan: '#00cec9',
  magenta: '#e84393',
  amber: '#fdcb6e',
  lime: '#00b894',
  red: '#d63031',
  blue: '#0984e3',
  orange: '#e17055',
};

function progressRingSVG(percent, size = 48, stroke = 4, color = '#6c5ce7') {
  const r = (size - stroke) / 2;
  const circ = 2 * Math.PI * r;
  const offset = circ - (percent / 100) * circ;
  return `
    <svg width="${size}" height="${size}" viewBox="0 0 ${size} ${size}">
      <circle cx="${size / 2}" cy="${size / 2}" r="${r}" class="ring-bg" stroke-width="${stroke}" />
      <circle cx="${size / 2}" cy="${size / 2}" r="${r}" class="ring-fill"
        stroke="${color}" stroke-width="${stroke}"
        stroke-dasharray="${circ}" stroke-dashoffset="${offset}" />
    </svg>
  `;
}

export async function renderDashboard(container) {
  const overall = await DB.getOverallProgress();
  const today = new Date().toISOString().slice(0, 10);
  const dueCards = await DB.getCardsForReview(today);
  const allCards = await DB.getAllFlashcards();

  // Compute per-unit progress
  const unitProgress = [];
  for (const unit of UNITS) {
    const prog = await DB.getUnitProgress(unit.id);
    unitProgress.push({ ...unit, ...prog });
  }

  const completedUnits = unitProgress.filter(u => u.percent === 100).length;
  const inProgressUnits = unitProgress.filter(u => u.percent > 0 && u.percent < 100).length;

  container.innerHTML = `
    <div class="content-inner">
      <div class="unit-header">
        <h1>Parameter Golf Curriculum</h1>
        <div class="unit-meta">
          <span class="badge badge-violet">9 Units</span>
          <span class="badge badge-cyan">16 Weeks</span>
          <span class="badge badge-amber">${overall.percent}% Complete</span>
        </div>
      </div>

      <!-- Stats Row -->
      <div class="stats-grid">
        <div class="stat-card">
          <div class="stat-value" style="color: var(--pg-violet);">${overall.completed}</div>
          <div class="stat-label">Items Done</div>
        </div>
        <div class="stat-card">
          <div class="stat-value" style="color: var(--pg-lime);">${completedUnits}</div>
          <div class="stat-label">Units Complete</div>
        </div>
        <div class="stat-card">
          <div class="stat-value" style="color: var(--pg-amber);">${inProgressUnits}</div>
          <div class="stat-label">In Progress</div>
        </div>
        <div class="stat-card">
          <div class="stat-value" style="color: var(--pg-cyan);">${dueCards.length}</div>
          <div class="stat-label">Cards Due</div>
        </div>
      </div>

      <!-- Unit Grid -->
      <div class="section-header" style="margin-top: var(--space-xl);">
        <span class="section-icon">\u{1F5FA}</span>
        <h3>Unit Map</h3>
      </div>
      <div class="hex-grid">
        ${unitProgress.map(u => {
          const color = UNIT_COLORS[u.color] || UNIT_COLORS.violet;
          const statusClass = u.percent === 100 ? 'complete' : u.percent > 0 ? 'in-progress' : '';
          return `
            <div class="hex-card ${statusClass}" data-unit="${u.id}" style="--hex-color: ${color};">
              <div class="hex-unit-num" style="color: ${color};">${u.id}</div>
              <div class="hex-unit-name">${u.name}</div>
              <div class="hex-progress">${u.total > 0 ? `${u.percent}%` : 'Not started'}</div>
            </div>
          `;
        }).join('')}
      </div>

      <!-- Overall Progress Bar -->
      <div class="section-header">
        <span class="section-icon">\u{1F4CA}</span>
        <h3>Overall Progress</h3>
      </div>
      <div class="card">
        <div style="display: flex; align-items: center; gap: var(--space-lg);">
          <div class="progress-ring" style="flex-shrink: 0;">
            ${progressRingSVG(overall.percent, 80, 6, '#6c5ce7')}
            <span class="ring-text" style="font-size: var(--font-size-md);">${overall.percent}%</span>
          </div>
          <div style="flex: 1;">
            <div style="background: var(--pg-border); border-radius: var(--radius-full); height: 8px; overflow: hidden;">
              <div style="background: linear-gradient(90deg, var(--pg-violet), var(--pg-cyan)); width: ${overall.percent}%; height: 100%; border-radius: var(--radius-full); transition: width var(--transition-slow);"></div>
            </div>
            <p style="color: var(--pg-text-muted); font-size: var(--font-size-sm); margin-top: var(--space-sm);">
              ${overall.completed} of ${overall.total} items completed
            </p>
          </div>
        </div>
      </div>

      <!-- Unit Breakdown Table -->
      <div class="section-header">
        <span class="section-icon">\u{1F4CB}</span>
        <h3>Unit Breakdown</h3>
      </div>
      <div class="card" style="padding: 0; overflow: hidden;">
        <table style="width: 100%; border-collapse: collapse; font-size: var(--font-size-sm);">
          <thead>
            <tr style="border-bottom: 1px solid var(--pg-border); background: var(--pg-bg-elevated);">
              <th style="padding: var(--space-sm) var(--space-md); text-align: left; color: var(--pg-text-muted);">Unit</th>
              <th style="padding: var(--space-sm) var(--space-md); text-align: left; color: var(--pg-text-muted);">Weeks</th>
              <th style="padding: var(--space-sm) var(--space-md); text-align: right; color: var(--pg-text-muted);">Progress</th>
              <th style="padding: var(--space-sm) var(--space-md); text-align: right; color: var(--pg-text-muted);">Done</th>
            </tr>
          </thead>
          <tbody>
            ${unitProgress.map(u => {
              const color = UNIT_COLORS[u.color] || UNIT_COLORS.violet;
              return `
                <tr style="border-bottom: 1px solid var(--pg-border); cursor: pointer;" data-unit="${u.id}">
                  <td style="padding: var(--space-sm) var(--space-md);">
                    <span style="color: ${color}; margin-right: var(--space-xs);">${u.icon}</span>
                    ${u.name}
                  </td>
                  <td style="padding: var(--space-sm) var(--space-md); color: var(--pg-text-dim);">
                    ${u.weeks.length > 1 ? `${u.weeks[0]}-${u.weeks[u.weeks.length - 1]}` : u.weeks[0]}
                  </td>
                  <td style="padding: var(--space-sm) var(--space-md); text-align: right;">
                    <div style="display: inline-block; width: 60px; background: var(--pg-border); border-radius: var(--radius-full); height: 6px; overflow: hidden; vertical-align: middle;">
                      <div style="background: ${color}; width: ${u.percent}%; height: 100%; border-radius: var(--radius-full);"></div>
                    </div>
                  </td>
                  <td style="padding: var(--space-sm) var(--space-md); text-align: right; color: var(--pg-text-muted);">
                    ${u.total > 0 ? `${u.completed}/${u.total}` : '--'}
                  </td>
                </tr>
              `;
            }).join('')}
          </tbody>
        </table>
      </div>

      ${allCards.length > 0 ? `
      <!-- Flashcard Status -->
      <div class="section-header">
        <span class="section-icon">\u{1F0CF}</span>
        <h3>Flashcard Review</h3>
      </div>
      <div class="card" style="cursor: pointer;" id="flashcard-dash-card">
        <div style="display: flex; align-items: center; justify-content: space-between;">
          <div>
            <p style="font-size: var(--font-size-md); font-weight: 700;">${dueCards.length} cards due today</p>
            <p style="color: var(--pg-text-muted); font-size: var(--font-size-sm);">${allCards.length} total cards in your deck</p>
          </div>
          <button class="btn btn-primary">Start Review</button>
        </div>
      </div>
      ` : ''}
    </div>
  `;

  // Attach click handlers for hex cards and table rows
  container.querySelectorAll('[data-unit]').forEach(el => {
    el.addEventListener('click', () => {
      const unitId = el.dataset.unit;
      navigate(`#/unit/${unitId}`);
    });
  });

  // Flashcard card click
  document.getElementById('flashcard-dash-card')?.addEventListener('click', () => {
    navigate('#/flashcards');
  });
}
