// ============================================================
// Progress Computation & Unit Hydration
// ============================================================

import * as DB from './db.js';
import { renderQuiz } from './quiz-engine.js';

// Hydrate a loaded unit: attach checkbox handlers, load saved state, notes
export async function hydrateUnit(unitId) {
  // Hydrate all check-items (topics, readings, exercises)
  const checkItems = document.querySelectorAll('.check-item');
  for (const item of checkItems) {
    const checkbox = item.querySelector('input[type="checkbox"]');
    if (!checkbox) continue;

    const weekId = parseInt(item.closest('[data-week]')?.dataset.week || '0', 10);
    const itemType = item.dataset.type || 'topic';
    const itemId = item.dataset.id || '';

    // Load saved state
    const progress = await DB.getItemProgress(unitId, weekId, itemType, itemId);
    if (progress?.completed) {
      checkbox.checked = true;
      item.classList.add('completed');
    }

    // Attach change handler
    checkbox.addEventListener('change', async () => {
      const checked = checkbox.checked;
      await DB.setItemComplete(unitId, weekId, itemType, itemId, checked);
      item.classList.toggle('completed', checked);
      updateSidebarProgress();
    });
  }

  // Hydrate all note areas
  const noteAreas = document.querySelectorAll('.note-area');
  for (const area of noteAreas) {
    const weekId = parseInt(area.closest('[data-week]')?.dataset.week || '0', 10);
    const itemId = area.dataset.noteKey || '';
    const itemType = area.dataset.noteType || 'note';

    // Load saved note
    const progress = await DB.getItemProgress(unitId, weekId, itemType, itemId);
    if (progress?.notes) {
      area.textContent = progress.notes;
    }

    // Auto-save on blur
    let saveTimer;
    area.addEventListener('input', () => {
      clearTimeout(saveTimer);
      saveTimer = setTimeout(async () => {
        await DB.saveNote(unitId, weekId, itemType, itemId, area.textContent);
      }, 500);
    });
  }

  // Hydrate collapsible sections
  const collapsibles = document.querySelectorAll('.collapsible');
  for (const c of collapsibles) {
    const header = c.querySelector('.collapsible-header');
    if (header) {
      header.addEventListener('click', () => {
        c.classList.toggle('open');
      });
    }
  }

  // Hydrate quiz containers
  const quizContainers = document.querySelectorAll('[data-quiz]');
  for (const container of quizContainers) {
    const quizId = container.dataset.quiz;
    if (quizId) await renderQuiz(container, quizId);
  }

  // Initialize flashcards for this unit
  try {
    const resp = await fetch('data/flashcards.json');
    const cards = await resp.json();
    const unitCards = cards.filter(c => c.unitId === unitId);
    if (unitCards.length > 0) {
      await DB.initFlashcards(unitCards);
    }
  } catch (e) {
    // Flashcard data not available yet
  }

  // Render KaTeX math
  if (window.renderMathInElement) {
    renderMathInElement(document.getElementById('content'), {
      delimiters: [
        { left: '$$', right: '$$', display: true },
        { left: '$', right: '$', display: false },
      ],
      throwOnError: false
    });
  }
}

// Update sidebar progress badges
export async function updateSidebarProgress() {
  for (let unitId = 1; unitId <= 9; unitId++) {
    const badge = document.querySelector(`[data-nav="unit-${unitId}"] .nav-badge`);
    if (!badge) continue;

    const progress = await DB.getUnitProgress(unitId);
    if (progress.total === 0) {
      badge.textContent = '--';
      badge.closest('.nav-item')?.classList.remove('complete', 'in-progress');
    } else {
      badge.textContent = `${progress.percent}%`;
      const navItem = badge.closest('.nav-item');
      navItem?.classList.remove('complete', 'in-progress');
      if (progress.percent === 100) navItem?.classList.add('complete');
      else if (progress.percent > 0) navItem?.classList.add('in-progress');
    }
  }
}
