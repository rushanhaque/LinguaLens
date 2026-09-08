/**
 * store.js — Single source of truth.
 *
 * Everything the app remembers lives here and is mirrored to localStorage
 * under one versioned key. Views never touch storage directly; they read
 * `state` and call actions, then re-render on the `change` event.
 */

import {
  itemMeta, objId, idsInGroup, GROUP_KEYS, TOTAL_ITEMS, isCameraItem
} from '../data/vocab.js';
import { LANG_CODES } from '../data/languages.js';
import { ACHIEVEMENTS, levelFor, XP } from '../data/achievements.js';

const KEY = 'lemma.v1';
const SCHEMA = 4;
// The app was called LinguaLens through v3 and v4. Both stores are still read
// on first launch so a rename never costs anyone their streak.
const LEGACY_KEYS = ['lingualens.v4', 'lingualens.v3'];

/* ── Defaults ─────────────────────────────────────────────────────────── */

const defaultSettings = () => ({
  targetLang: 'es',
  theme: 'light',             // 'auto' | 'light' | 'dark'
  accent: 'sage',
  confidence: 0.6,            // model score floor
  maxDetections: 6,           // labels drawn at once
  detectHz: 8,                // detection passes per second (render stays 60)
  modelBase: 'auto',          // 'auto' | 'mobilenet_v2' | 'lite_mobilenet_v2'
  focusMode: true,            // name the centred subject via the classifier
  showPhonetics: true,
  showGender: true,
  showConfidence: false,
  speakOnDiscover: false,
  speechRate: 0.85,
  haptics: true,
  sound: true,
  quizMode: false,
  showColors: true,           // read dominant colour off detected objects
  colorPhrases: true,         // render colour + noun as an agreeing phrase
  phraseFrame: 'this_is',
  dailyGoal: 15,              // reviews per day
  onboarded: false,
  reviewStyle: 'mixed'        // 'mixed' | 'choice' | 'listen' | 'type' | 'recall'
});

const defaultProgress = () => ({
  // Keyed by language code, then by vocab item id ('obj:apple' / 'lex:mother').
  // { seen, discoveredAt, lastSeen, correct, wrong, srs:{reps,lapses,ease,interval,due} }
});

const defaultMeta = () => ({
  xp: 0,
  streak: { current: 0, best: 0, lastDay: null },
  history: {},           // 'YYYY-MM-DD' → reviews completed that day
  snapshots: 0,
  badges: [],            // unlocked achievement ids
  bestRun: 0,
  totalReviews: 0,
  totalCorrect: 0,
  createdAt: Date.now()
});

/* ── State ────────────────────────────────────────────────────────────── */

export const state = {
  settings: defaultSettings(),
  progress: defaultProgress(),
  meta: defaultMeta(),
  // Runtime-only — never persisted.
  runtime: {
    ready: false,
    detecting: true,
    fps: 0,
    live: [],            // current confirmed detections
    cameraFacing: 'environment',
    torch: false,
    zoom: 1,
    modelName: ''
  }
};

/* ── Persistence ──────────────────────────────────────────────────────── */

let saveTimer = null;

function load() {
  let raw;
  try { raw = localStorage.getItem(KEY); } catch { return; }   // private mode
  if (raw) {
    try {
      const data = JSON.parse(raw);
      if (data.schema === SCHEMA) {
        Object.assign(state.settings, data.settings || {});
        state.progress = normaliseProgress(data.progress || {});
        Object.assign(state.meta, data.meta || {});
        return;
      }
    } catch { /* corrupt payload — fall through to the migrations */ }
  }
  if (migrateV3()) return;
  migratePrototype();
}

/**
 * v3 stored progress keyed by bare COCO class name. v4 namespaces every entry
 * so lexicon words can share the same store, so the keys are rewritten in
 * place. Nobody loses a streak to a schema bump.
 */
function migrateV3() {
  let raw = null;
  for (const key of LEGACY_KEYS) {
    try { raw = localStorage.getItem(key); } catch { return false; }
    if (raw) break;
  }
  if (!raw) return false;
  try {
    const data = JSON.parse(raw);
    Object.assign(state.settings, data.settings || {});
    state.progress = normaliseProgress(data.progress || {});
    Object.assign(state.meta, data.meta || {});
    saveNow();
    return true;
  } catch { return false; }
}

/** Carry across vocabulary from the v1/v2 prototype so nobody loses progress. */
function migratePrototype() {
  let legacy;
  try { legacy = localStorage.getItem('ll_vocab'); } catch { return; }
  if (!legacy) return;
  try {
    const vocab = JSON.parse(legacy);
    const lang = state.settings.targetLang;
    state.progress[lang] = state.progress[lang] || {};
    for (const cls of Object.keys(vocab)) {
      const id = objId(cls);
      if (!itemMeta(id)) continue;
      state.progress[lang][id] = blankWord(vocab[cls].discovered || Date.now());
      state.progress[lang][id].seen = vocab[cls].count || 1;
    }
    save();
  } catch { /* ignore malformed legacy data */ }
}

/**
 * Rewrite any un-namespaced keys to item ids and drop entries whose word no
 * longer exists, so a stale record can never crash a render.
 */
function normaliseProgress(progress) {
  const out = {};
  for (const [lang, words] of Object.entries(progress || {})) {
    if (!words || typeof words !== 'object') continue;
    out[lang] = {};
    for (const [key, rec] of Object.entries(words)) {
      if (!rec || typeof rec !== 'object') continue;
      const id = key.includes(':') ? key : objId(key);
      if (!itemMeta(id)) continue;
      // Merge rather than overwrite if both forms somehow exist.
      const prev = out[lang][id];
      out[lang][id] = prev && prev.discoveredAt <= rec.discoveredAt ? prev : withDefaults(rec);
    }
  }
  return out;
}

/** Fill in any field a older record may predate. */
function withDefaults(rec) {
  const base = blankWord(rec.discoveredAt || Date.now());
  return {
    ...base,
    ...rec,
    srs: { ...base.srs, ...(rec.srs || {}) }
  };
}

function persist() {
  try {
    localStorage.setItem(KEY, JSON.stringify({
      schema: SCHEMA,
      settings: state.settings,
      progress: state.progress,
      meta: state.meta
    }));
  } catch { /* quota or private mode — the session still works in memory */ }
}

/** Debounced write; detection loops touch state many times per second. */
export function save() {
  clearTimeout(saveTimer);
  saveTimer = setTimeout(persist, 400);
}

export function saveNow() { clearTimeout(saveTimer); persist(); }

/* ── Events ───────────────────────────────────────────────────────────── */

const bus = new EventTarget();

export function on(type, fn) {
  bus.addEventListener(type, fn);
  return () => bus.removeEventListener(type, fn);
}

export function emit(type, detail) {
  bus.dispatchEvent(new CustomEvent(type, { detail }));
}

/* ── Settings ─────────────────────────────────────────────────────────── */

export function setSetting(key, value) {
  if (state.settings[key] === value) return;
  state.settings[key] = value;
  save();
  emit('settings', { key, value });
  emit('change');
}

/* ── Word records ─────────────────────────────────────────────────────── */

function blankWord(now = Date.now()) {
  return {
    discoveredAt: now,
    lastSeen: now,
    seen: 0,
    correct: 0,
    wrong: 0,
    srs: { reps: 0, lapses: 0, ease: 2.5, interval: 0, due: now }
  };
}

export function langProgress(lang = state.settings.targetLang) {
  if (!state.progress[lang]) state.progress[lang] = {};
  return state.progress[lang];
}

export function wordRecord(id, lang = state.settings.targetLang) {
  return langProgress(lang)[id] || null;
}

/** Create the record for an item if it does not exist yet. */
export function ensureRecord(id, lang = state.settings.targetLang) {
  const words = langProgress(lang);
  if (!words[id]) words[id] = blankWord();
  return words[id];
}

/**
 * Record that an object was seen through the camera.
 * @param {string} cls  COCO class name
 * @returns {boolean} true when this is the first sighting (a "discovery").
 */
export function recordSighting(cls, lang = state.settings.targetLang) {
  const id = objId(cls);
  if (!itemMeta(id)) return false;
  const words = langProgress(lang);
  const isNew = !words[id];
  if (isNew) {
    words[id] = blankWord();
    addXP(XP.discover);
  }
  words[id].seen++;
  words[id].lastSeen = Date.now();
  save();
  if (isNew) emit('discovered', { id, cls, lang });
  return isNew;
}

/** Mark a non-camera word as started, so lexicon packs can enter the queue. */
export function recordStudied(id, lang = state.settings.targetLang) {
  if (!itemMeta(id)) return false;
  const words = langProgress(lang);
  const isNew = !words[id];
  if (isNew) {
    words[id] = blankWord();
    addXP(XP.discover);
    emit('discovered', { id, lang });
  }
  save();
  return isNew;
}

/* ── XP, streaks, history ─────────────────────────────────────────────── */

export function todayKey(d = new Date()) {
  return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, '0')}-${String(d.getDate()).padStart(2, '0')}`;
}

function dayDiff(a, b) {
  const ms = new Date(b + 'T00:00:00') - new Date(a + 'T00:00:00');
  return Math.round(ms / 86400000);
}

export function addXP(amount) {
  const before = levelFor(state.meta.xp).level;
  state.meta.xp += amount;
  const after = levelFor(state.meta.xp).level;
  if (after > before) emit('levelup', { level: after });
  save();
}

/** Mark today as active, advancing or resetting the streak. */
export function touchStreak() {
  const today = todayKey();
  const s = state.meta.streak;
  if (s.lastDay === today) return s.current;
  const gap = s.lastDay ? dayDiff(s.lastDay, today) : Infinity;
  s.current = gap === 1 ? s.current + 1 : 1;
  s.lastDay = today;
  s.best = Math.max(s.best, s.current);
  addXP(XP.streakDay);
  emit('streak', { current: s.current });
  save();
  return s.current;
}

export function logReview(correct) {
  const today = todayKey();
  state.meta.history[today] = (state.meta.history[today] || 0) + 1;
  state.meta.totalReviews++;
  if (correct) state.meta.totalCorrect++;
  addXP(correct ? XP.reviewCorrect : XP.reviewWrong);
  touchStreak();
  save();
}

export function logSnapshot() {
  state.meta.snapshots++;
  save();
}

/** Keep the longest unbroken run of correct answers ever achieved. */
export function recordBestRun(run) {
  if (run > state.meta.bestRun) { state.meta.bestRun = run; save(); }
}

/* ── Derived stats ────────────────────────────────────────────────────── */

export function getStats() {
  const lang = state.settings.targetLang;
  const words = langProgress(lang);
  const keys = Object.keys(words);

  const mastered = keys.filter((k) => words[k].srs.reps >= 5 && words[k].srs.interval >= 21).length;

  const languagesUsed = LANG_CODES.filter(
    (l) => state.progress[l] && Object.keys(state.progress[l]).length > 0
  ).length;

  const clearedCategories = GROUP_KEYS.filter((group) => {
    const ids = idsInGroup(group);
    return ids.length > 0 && ids.every((id) => words[id]);
  }).length;

  const reviews = state.meta.totalReviews;
  const objectsFound = keys.filter(isCameraItem).length;

  return {
    lang,
    discovered: keys.length,
    total: TOTAL_ITEMS,
    objectsFound,
    mastered,
    reviews,
    accuracy: reviews ? state.meta.totalCorrect / reviews : 0,
    streak: state.meta.streak.current,
    bestStreak: state.meta.streak.best,
    xp: state.meta.xp,
    level: levelFor(state.meta.xp),
    languagesUsed,
    clearedCategories,
    snapshots: state.meta.snapshots,
    bestRun: state.meta.bestRun,
    todayReviews: state.meta.history[todayKey()] || 0,
    dailyGoal: state.settings.dailyGoal
  };
}

/** Per-language discovery counts, for the Progress view. */
export function languageBreakdown() {
  return LANG_CODES.map((code) => ({
    code,
    count: state.progress[code] ? Object.keys(state.progress[code]).length : 0,
    total: TOTAL_ITEMS
  })).sort((a, b) => b.count - a.count);
}

/** Re-test every badge; returns the ids newly unlocked this call. */
export function refreshBadges() {
  const stats = getStats();
  const fresh = [];
  for (const a of ACHIEVEMENTS) {
    if (state.meta.badges.includes(a.id)) continue;
    let passed = false;
    try { passed = a.test(stats); } catch { passed = false; }
    if (passed) { state.meta.badges.push(a.id); fresh.push(a); }
  }
  if (fresh.length) { save(); fresh.forEach((a) => emit('badge', a)); }
  return fresh;
}

/* ── Data management ──────────────────────────────────────────────────── */

export function exportData() {
  return JSON.stringify({
    app: 'Lemma',
    schema: SCHEMA,
    exportedAt: new Date().toISOString(),
    settings: state.settings,
    progress: state.progress,
    meta: state.meta
  }, null, 2);
}

/**
 * Replace all stored data from a previously exported file.
 * @throws when the payload is not a Lemma export.
 */
export function importData(json) {
  const data = JSON.parse(json);
  if (!['Lemma', 'LinguaLens'].includes(data.app) || !data.progress) {
    throw new Error('Not a Lemma backup file.');
  }
  // Exports from before the rename are still accepted; normaliseProgress
  // rewrites their keys into the current id space.
  if (data.schema > SCHEMA) throw new Error('This backup was made by a newer version of Lemma.');
  Object.assign(state.settings, defaultSettings(), data.settings || {});
  state.progress = normaliseProgress(data.progress);
  Object.assign(state.meta, defaultMeta(), data.meta || {});
  saveNow();
  emit('change');
}

export function resetProgress(lang) {
  if (lang) delete state.progress[lang];
  else {
    state.progress = defaultProgress();
    state.meta = defaultMeta();
  }
  saveNow();
  emit('change');
}

export function resetEverything() {
  state.settings = defaultSettings();
  state.progress = defaultProgress();
  state.meta = defaultMeta();
  saveNow();
  emit('change');
}

load();
