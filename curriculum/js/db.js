// ============================================================
// IndexedDB Abstraction Layer — Parameter Golf Curriculum
// ============================================================

const DB_NAME = 'PGolfCurriculum';
const DB_VERSION = 1;

let _db = null;

function openDB() {
  return new Promise((resolve, reject) => {
    if (_db) { resolve(_db); return; }
    const req = indexedDB.open(DB_NAME, DB_VERSION);
    req.onupgradeneeded = (e) => {
      const db = e.target.result;

      if (!db.objectStoreNames.contains('progress')) {
        const s = db.createObjectStore('progress', { keyPath: 'id' });
        s.createIndex('byUnit', 'unitId', { unique: false });
        s.createIndex('byWeek', 'weekId', { unique: false });
        s.createIndex('byCompleted', 'completed', { unique: false });
      }

      if (!db.objectStoreNames.contains('exercises')) {
        db.createObjectStore('exercises', { keyPath: 'exerciseId' });
      }

      if (!db.objectStoreNames.contains('quizScores')) {
        const qs = db.createObjectStore('quizScores', { keyPath: 'id', autoIncrement: true });
        qs.createIndex('byQuiz', ['weekId', 'quizId'], { unique: false });
      }

      if (!db.objectStoreNames.contains('flashcards')) {
        const fc = db.createObjectStore('flashcards', { keyPath: 'cardId' });
        fc.createIndex('byNextReview', 'nextReview', { unique: false });
        fc.createIndex('byUnit', 'unitId', { unique: false });
      }

      if (!db.objectStoreNames.contains('sessions')) {
        const ss = db.createObjectStore('sessions', { keyPath: 'id', autoIncrement: true });
        ss.createIndex('byDate', 'startedAt', { unique: false });
      }

      if (!db.objectStoreNames.contains('settings')) {
        db.createObjectStore('settings', { keyPath: 'key' });
      }
    };
    req.onsuccess = () => { _db = req.result; resolve(_db); };
    req.onerror = () => reject(req.error);
  });
}

// Generic helpers
function tx(storeName, mode = 'readonly') {
  return _db.transaction(storeName, mode).objectStore(storeName);
}

function reqP(request) {
  return new Promise((resolve, reject) => {
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
}

function txP(storeName, mode, fn) {
  return new Promise((resolve, reject) => {
    const transaction = _db.transaction(storeName, mode);
    const store = transaction.objectStore(storeName);
    fn(store);
    transaction.oncomplete = () => resolve();
    transaction.onerror = () => reject(transaction.error);
  });
}

// ============================================================
// Progress
// ============================================================

function progressId(unitId, weekId, itemType, itemId) {
  return `${unitId}-${weekId}-${itemType}-${itemId}`;
}

export async function setItemComplete(unitId, weekId, itemType, itemId, completed) {
  await openDB();
  const id = progressId(unitId, weekId, itemType, itemId);
  const store = tx('progress', 'readwrite');
  const existing = await reqP(store.get(id));
  const record = existing || { id, unitId, weekId, itemType, itemId, notes: '', timeSpentMs: 0 };
  record.completed = completed;
  record.completedAt = completed ? new Date().toISOString() : null;
  await reqP(tx('progress', 'readwrite').put(record));
}

export async function getItemProgress(unitId, weekId, itemType, itemId) {
  await openDB();
  return reqP(tx('progress').get(progressId(unitId, weekId, itemType, itemId)));
}

export async function saveNote(unitId, weekId, itemType, itemId, notes) {
  await openDB();
  const id = progressId(unitId, weekId, itemType, itemId);
  const store = tx('progress', 'readwrite');
  const existing = await reqP(store.get(id));
  const record = existing || { id, unitId, weekId, itemType, itemId, completed: false, completedAt: null, timeSpentMs: 0 };
  record.notes = notes;
  await reqP(tx('progress', 'readwrite').put(record));
}

export async function getWeekProgress(unitId, weekId) {
  await openDB();
  const all = await getAllFromIndex('progress', 'byUnit', unitId);
  const weekItems = all.filter(r => r.weekId === weekId);
  const completed = weekItems.filter(r => r.completed).length;
  return { completed, total: weekItems.length, percent: weekItems.length ? Math.round((completed / weekItems.length) * 100) : 0 };
}

export async function getUnitProgress(unitId) {
  await openDB();
  const all = await getAllFromIndex('progress', 'byUnit', unitId);
  const completed = all.filter(r => r.completed).length;
  return { completed, total: all.length, percent: all.length ? Math.round((completed / all.length) * 100) : 0 };
}

export async function getOverallProgress() {
  await openDB();
  const all = await getAllFromStore('progress');
  const completed = all.filter(r => r.completed).length;
  return { completed, total: all.length, percent: all.length ? Math.round((completed / all.length) * 100) : 0 };
}

// ============================================================
// Exercises
// ============================================================

export async function saveExerciseCode(exerciseId, code, language = 'python') {
  await openDB();
  const store = tx('exercises', 'readwrite');
  const existing = await reqP(store.get(exerciseId));
  const record = existing || { exerciseId, unitId: 0, weekId: 0, completed: false, attempts: 0 };
  record.code = code;
  record.language = language;
  record.lastModified = new Date().toISOString();
  await reqP(tx('exercises', 'readwrite').put(record));
}

export async function getExerciseCode(exerciseId) {
  await openDB();
  return reqP(tx('exercises').get(exerciseId));
}

export async function setExerciseComplete(exerciseId, completed) {
  await openDB();
  const store = tx('exercises', 'readwrite');
  const existing = await reqP(store.get(exerciseId));
  if (existing) {
    existing.completed = completed;
    await reqP(tx('exercises', 'readwrite').put(existing));
  }
}

// ============================================================
// Quiz Scores
// ============================================================

export async function saveQuizAttempt(weekId, quizId, score, maxScore, answers) {
  await openDB();
  const store = tx('quizScores', 'readwrite');
  await reqP(store.add({
    weekId, quizId, score, maxScore, answers,
    takenAt: new Date().toISOString(),
    timeSpentMs: 0
  }));
}

export async function getQuizHistory(weekId, quizId) {
  await openDB();
  const all = await getAllFromIndex('quizScores', 'byQuiz', [weekId, quizId]);
  return all.sort((a, b) => b.takenAt.localeCompare(a.takenAt));
}

// ============================================================
// Flashcards (SM-2)
// ============================================================

export async function initFlashcards(cards) {
  await openDB();
  const store = tx('flashcards', 'readwrite');
  for (const card of cards) {
    const existing = await reqP(store.get(card.cardId));
    if (!existing) {
      await reqP(tx('flashcards', 'readwrite').put({
        ...card,
        easeFactor: 2.5,
        interval: 0,
        repetitions: 0,
        nextReview: new Date().toISOString().slice(0, 10),
        lastReviewed: null
      }));
    }
  }
}

export async function getCardsForReview(today) {
  await openDB();
  const all = await getAllFromStore('flashcards');
  return all.filter(c => c.nextReview <= today);
}

export async function updateCardReview(cardId, quality) {
  await openDB();
  const store = tx('flashcards', 'readwrite');
  const card = await reqP(store.get(cardId));
  if (!card) return;

  // SM-2 algorithm
  if (quality < 3) {
    card.repetitions = 0;
    card.interval = 0;
  } else {
    if (card.repetitions === 0) card.interval = 1;
    else if (card.repetitions === 1) card.interval = 6;
    else card.interval = Math.round(card.interval * card.easeFactor);
    card.repetitions++;
  }

  card.easeFactor = Math.max(1.3, card.easeFactor + (0.1 - (5 - quality) * (0.08 + (5 - quality) * 0.02)));
  const next = new Date();
  next.setDate(next.getDate() + card.interval);
  card.nextReview = next.toISOString().slice(0, 10);
  card.lastReviewed = new Date().toISOString();

  await reqP(tx('flashcards', 'readwrite').put(card));
}

export async function getAllFlashcards() {
  await openDB();
  return getAllFromStore('flashcards');
}

// ============================================================
// Sessions
// ============================================================

let _currentSession = null;

export async function startSession(unitId, weekId) {
  await openDB();
  _currentSession = { unitId, weekId, startedAt: new Date().toISOString(), activeTimeMs: 0 };
}

export async function endSession() {
  if (!_currentSession) return;
  await openDB();
  _currentSession.endedAt = new Date().toISOString();
  const store = tx('sessions', 'readwrite');
  await reqP(store.add(_currentSession));
  _currentSession = null;
}

export async function getTotalStudyTime() {
  await openDB();
  const all = await getAllFromStore('sessions');
  return all.reduce((sum, s) => sum + (s.activeTimeMs || 0), 0);
}

// ============================================================
// Settings
// ============================================================

export async function setSetting(key, value) {
  await openDB();
  await reqP(tx('settings', 'readwrite').put({ key, value }));
}

export async function getSetting(key, defaultValue = null) {
  await openDB();
  const result = await reqP(tx('settings').get(key));
  return result ? result.value : defaultValue;
}

// ============================================================
// Export / Import / Reset
// ============================================================

export async function exportAll() {
  await openDB();
  const data = {};
  const storeNames = ['progress', 'exercises', 'quizScores', 'flashcards', 'sessions', 'settings'];
  for (const name of storeNames) {
    data[name] = await getAllFromStore(name);
  }
  return data;
}

export async function importAll(data) {
  await openDB();
  for (const [name, records] of Object.entries(data)) {
    if (!_db.objectStoreNames.contains(name)) continue;
    await txP(name, 'readwrite', (store) => {
      store.clear();
      for (const record of records) {
        store.put(record);
      }
    });
  }
}

export async function clearAll() {
  await openDB();
  const storeNames = ['progress', 'exercises', 'quizScores', 'flashcards', 'sessions', 'settings'];
  for (const name of storeNames) {
    await txP(name, 'readwrite', (store) => store.clear());
  }
}

// ============================================================
// Internal helpers
// ============================================================

function getAllFromStore(storeName) {
  return reqP(tx(storeName).getAll());
}

function getAllFromIndex(storeName, indexName, key) {
  return reqP(tx(storeName).index(indexName).getAll(key));
}

// Initialize DB on import
export const ready = openDB();
