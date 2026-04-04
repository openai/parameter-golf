// ============================================================
// Quiz Engine — Rendering, scoring, and persistence
// ============================================================

import * as DB from './db.js';

let _quizData = null;

async function loadQuizData() {
  if (_quizData) return _quizData;
  try {
    const resp = await fetch('data/quizzes.json');
    _quizData = await resp.json();
  } catch (e) {
    _quizData = {};
  }
  return _quizData;
}

export async function renderQuiz(container, quizId) {
  const data = await loadQuizData();
  const quiz = data[quizId];
  if (!quiz) {
    container.innerHTML = '<p style="color: var(--pg-text-dim);">Quiz not available yet.</p>';
    return;
  }

  // Check for previous attempts
  const history = await DB.getQuizHistory(quiz.weekId, quiz.quizId);
  const bestScore = history.length > 0 ? Math.max(...history.map(h => h.score)) : null;

  container.innerHTML = `
    <div class="card" id="quiz-${quizId}">
      <div class="card-header">
        <h3 class="card-title">${quiz.title}</h3>
        ${bestScore !== null ? `<span class="badge badge-lime">Best: ${bestScore}/${quiz.questions.length}</span>` : ''}
      </div>
      <div id="quiz-questions-${quizId}">
        ${quiz.questions.map((q, i) => `
          <div class="quiz-question" data-qidx="${i}" style="margin-bottom: var(--space-lg); padding-bottom: var(--space-md); border-bottom: 1px solid var(--pg-border);">
            <p style="font-weight: 600; margin-bottom: var(--space-sm); color: var(--pg-text);">
              ${i + 1}. ${q.question}
            </p>
            <div class="quiz-options" style="display: flex; flex-direction: column; gap: var(--space-xs);">
              ${q.options.map((opt, j) => `
                <label class="check-item" style="cursor: pointer; border: 1px solid transparent; border-radius: var(--radius-md);" data-qidx="${i}" data-oidx="${j}">
                  <input type="radio" name="quiz-${quizId}-q${i}" value="${j}" style="appearance: auto; -webkit-appearance: auto; width: 16px; height: 16px; margin-top: 3px; accent-color: var(--pg-violet);">
                  <span class="check-label">${opt}</span>
                </label>
              `).join('')}
            </div>
            <div class="quiz-feedback" data-qidx="${i}" style="display: none; margin-top: var(--space-sm); padding: var(--space-sm) var(--space-md); border-radius: var(--radius-md); font-size: var(--font-size-sm);"></div>
          </div>
        `).join('')}
      </div>
      <button class="btn btn-primary" id="quiz-submit-${quizId}">Submit Answers</button>
      <div id="quiz-result-${quizId}" style="margin-top: var(--space-md); display: none;"></div>
    </div>
  `;

  // Submit handler
  document.getElementById(`quiz-submit-${quizId}`)?.addEventListener('click', async () => {
    const answers = {};
    let score = 0;

    quiz.questions.forEach((q, i) => {
      const selected = document.querySelector(`input[name="quiz-${quizId}-q${i}"]:checked`);
      const selectedIdx = selected ? parseInt(selected.value, 10) : -1;
      answers[q.id] = selectedIdx;

      const feedback = document.querySelector(`.quiz-feedback[data-qidx="${i}"]`);
      const options = document.querySelectorAll(`label[data-qidx="${i}"]`);

      if (selectedIdx === q.answer) {
        score++;
        if (feedback) {
          feedback.style.display = 'block';
          feedback.style.background = 'rgba(0, 184, 148, 0.1)';
          feedback.style.color = 'var(--pg-lime)';
          feedback.textContent = 'Correct! ' + q.explanation;
        }
        options.forEach((opt, j) => {
          if (j === q.answer) opt.style.borderColor = 'var(--pg-lime)';
        });
      } else {
        if (feedback) {
          feedback.style.display = 'block';
          feedback.style.background = 'rgba(214, 48, 49, 0.1)';
          feedback.style.color = 'var(--pg-red-soft)';
          feedback.textContent = (selectedIdx === -1 ? 'Not answered. ' : 'Incorrect. ') + q.explanation;
        }
        options.forEach((opt, j) => {
          if (j === q.answer) opt.style.borderColor = 'var(--pg-lime)';
          if (j === selectedIdx) opt.style.borderColor = 'var(--pg-red)';
        });
      }
    });

    // Save attempt
    await DB.saveQuizAttempt(quiz.weekId, quiz.quizId, score, quiz.questions.length, answers);

    // Show result
    const resultEl = document.getElementById(`quiz-result-${quizId}`);
    if (resultEl) {
      resultEl.style.display = 'block';
      const pct = Math.round((score / quiz.questions.length) * 100);
      const color = pct >= 80 ? 'var(--pg-lime)' : pct >= 60 ? 'var(--pg-amber)' : 'var(--pg-red)';
      resultEl.innerHTML = `
        <div style="font-size: var(--font-size-lg); font-weight: 700; color: ${color};">
          ${score} / ${quiz.questions.length} (${pct}%)
        </div>
      `;
    }

    // Disable submit
    const btn = document.getElementById(`quiz-submit-${quizId}`);
    if (btn) { btn.disabled = true; btn.textContent = 'Submitted'; btn.style.opacity = '0.5'; }

    // Disable radio buttons
    document.querySelectorAll(`input[name^="quiz-${quizId}"]`).forEach(r => r.disabled = true);
  });
}
