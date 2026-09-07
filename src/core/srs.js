/**
 * srs.js — Spaced repetition scheduling (SM-2, lightly adapted).
 *
 * Grades are the three the UI exposes:
 *   again → forgotten, reset the interval and drop ease
 *   hard  → recalled with effort, small interval growth
 *   good  → clean recall, full interval growth
 *
 * Intervals are in days; `due` is an absolute timestamp so a card stays due
 * across sessions and time-zone changes.
 */

import { state, langProgress, save, logReview, emit } from './store.js';
import { itemMeta, idsInGroup, ALL_ITEM_IDS, isCameraItem } from '../data/vocab.js';

const DAY = 86400000;
const MIN_EASE = 1.3;
const MAX_EASE = 3.0;

/** A card counts as mastered once it has survived several reps at 3+ weeks. */
export function isMastered(rec) {
  return rec && rec.srs.reps >= 5 && rec.srs.interval >= 21;
}

/** Mastery on a 0–5 scale, for the dot indicator. */
export function masteryLevel(rec) {
  if (!rec) return 0;
  const i = rec.srs.interval;
  if (i >= 60) return 5;
  if (i >= 21) return 4;
  if (i >= 7) return 3;
  if (i >= 3) return 2;
  if (i >= 1) return 1;
  return 0;
}

/** Interval each grade would produce, in days — shown on the grade buttons. */
export function previewIntervals(rec) {
  const s = rec ? rec.srs : { reps: 0, ease: 2.5, interval: 0 };
  return {
    again: 0,
    hard: s.reps === 0 ? 1 : Math.max(1, Math.round(s.interval * 1.2)),
    good: s.reps === 0 ? 1 : s.reps === 1 ? 3 : Math.max(1, Math.round(s.interval * s.ease))
  };
}

/**
 * Apply a grade to a card and reschedule it.
 * @param {string} id    vocab item id ('obj:apple' / 'lex:mother')
 * @param {'again'|'hard'|'good'} grade_
 * @param {string} [lang]
 */
export function grade(id, grade_, lang = state.settings.targetLang) {
  const words = langProgress(lang);
  if (!words[id]) {
    words[id] = {
      discoveredAt: Date.now(), lastSeen: Date.now(), seen: 0, correct: 0, wrong: 0,
      srs: { reps: 0, lapses: 0, ease: 2.5, interval: 0, due: Date.now() }
    };
  }
  const rec = words[id];
  const s = rec.srs;
  const correct = grade_ !== 'again';

  if (grade_ === 'again') {
    s.reps = 0;
    s.lapses++;
    s.interval = 0;
    s.ease = Math.max(MIN_EASE, s.ease - 0.20);
    s.due = Date.now() + 10 * 60000;          // back in ten minutes
    rec.wrong++;
  } else {
    const prev = s.interval;
    s.reps++;
    if (grade_ === 'hard') {
      s.ease = Math.max(MIN_EASE, s.ease - 0.15);
      s.interval = s.reps === 1 ? 1 : Math.max(1, Math.round(prev * 1.2));
    } else {
      s.ease = Math.min(MAX_EASE, s.ease + 0.10);
      s.interval = s.reps === 1 ? 1 : s.reps === 2 ? 3 : Math.max(1, Math.round(prev * s.ease));
    }
    s.due = Date.now() + s.interval * DAY;
    rec.correct++;
  }

  logReview(correct);
  save();
  emit('graded', { id, lang, grade: grade_, correct });
  return { correct, interval: s.interval };
}

/* ── Queue building ───────────────────────────────────────────────────── */

/** Cards past their due date, soonest first. */
export function dueCards(lang = state.settings.targetLang) {
  const words = langProgress(lang);
  const now = Date.now();
  return Object.keys(words)
    .filter((id) => itemMeta(id) && words[id].srs.due <= now)
    .sort((a, b) => words[a].srs.due - words[b].srs.due);
}

export function dueCount(lang = state.settings.targetLang) {
  return dueCards(lang).length;
}

/** Discovered but never reviewed — the "new" pile. */
export function newCards(lang = state.settings.targetLang) {
  const words = langProgress(lang);
  return Object.keys(words).filter((id) => itemMeta(id) && words[id].srs.reps === 0);
}

/**
 * Build a study session.
 *
 * @param {object}  opts
 * @param {string}  [opts.lang]
 * @param {number}  [opts.limit=20]      max cards
 * @param {string}  [opts.group]         restrict to one category or pack
 * @param {boolean} [opts.includeUndiscovered=false]
 *        Allow words never encountered yet. Off by default for the mixed
 *        queue — the camera-first loop is the point of the app — but on for
 *        decks the learner explicitly opens.
 */
export function buildSession(opts = {}) {
  const {
    lang = state.settings.targetLang,
    limit = 20,
    group = null,
    includeUndiscovered = false
  } = opts;

  const words = langProgress(lang);
  const scope = group ? new Set(idsInGroup(group)) : null;
  const inScope = (id) => !!itemMeta(id) && (!scope || scope.has(id));

  const due = dueCards(lang).filter(inScope);
  const dueSet = new Set(due);
  const fresh = newCards(lang).filter((id) => inScope(id) && !dueSet.has(id));

  let pool = [...due, ...fresh];

  if (pool.length < limit && includeUndiscovered) {
    const seen = new Set(pool);
    const rest = (scope ? [...scope] : ALL_ITEM_IDS)
      .filter((id) => !words[id] && !seen.has(id))
      .sort((a, b) => (itemMeta(a)?.lvl || 1) - (itemMeta(b)?.lvl || 1));
    pool = pool.concat(rest);
  }

  if (pool.length < limit) {
    // Top up with the least-recently-reviewed known cards so a short session
    // is still a full session.
    const seen = new Set(pool);
    const filler = Object.keys(words)
      .filter((id) => inScope(id) && !seen.has(id))
      .sort((a, b) => words[a].srs.due - words[b].srs.due);
    pool = pool.concat(filler);
  }

  return pool.slice(0, limit);
}

/** Distractor options for a multiple-choice question. */
export function distractors(id, count = 3) {
  const target = itemMeta(id);
  if (!target) return [];
  const sameGroup = idsInGroup(target.group).filter((x) => x !== id);
  // Same-group distractors are harder and more useful; fall back to items of
  // the same kind (object vs lexicon) before going wider, so a "numbers"
  // question is never answered with a picture of a sofa.
  const sameKind = ALL_ITEM_IDS.filter(
    (x) => x !== id && !sameGroup.includes(x) && isCameraItem(x) === isCameraItem(id)
  );
  const picked = shuffle(sameGroup).slice(0, count);
  if (picked.length < count) picked.push(...shuffle(sameKind).slice(0, count - picked.length));
  if (picked.length < count) {
    const rest = ALL_ITEM_IDS.filter((x) => x !== id && !picked.includes(x));
    picked.push(...shuffle(rest).slice(0, count - picked.length));
  }
  return picked;
}

export function shuffle(arr) {
  const a = [...arr];
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
}

/** Human-readable schedule, e.g. "in 3 days" / "now". */
export function formatDue(ms) {
  const delta = ms - Date.now();
  if (delta <= 0) return 'now';
  const mins = Math.round(delta / 60000);
  if (mins < 60) return `${mins}m`;
  const hours = Math.round(mins / 60);
  if (hours < 24) return `${hours}h`;
  const days = Math.round(hours / 24);
  if (days < 30) return `${days}d`;
  return `${Math.round(days / 30)}mo`;
}
