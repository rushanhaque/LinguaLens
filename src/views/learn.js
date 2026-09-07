/**
 * learn.js — Study hub and review sessions.
 *
 * Two states share the tab: a hub (due count, decks, vocabulary list) and an
 * active session. Sessions support four question types; 'mixed' rotates
 * between them, which is both more effective and less monotonous than drilling
 * one format.
 */

import { DICT, translate, CATEGORIES, ALL_CLASSES, classesInCategory } from '../data/dictionary.js';
import { LANGUAGES, genderLabel, citationForm } from '../data/languages.js';
import { state, langProgress, wordRecord, refreshBadges, getStats, recordBestRun } from '../core/store.js';
import { buildSession, grade, dueCount, previewIntervals, distractors, shuffle, masteryLevel, formatDue } from '../core/srs.js';
import { speak, hasVoice } from '../core/speech.js';
import { cue, haptic, unlockAudio } from '../core/feedback.js';
import { $, $$, el, esc, toast, confirmAction } from '../ui/kit.js';
import { icon } from '../ui/icons.js';
import { openWordSheet } from './wordSheet.js';

const MODES = ['recall', 'choice', 'listen', 'type'];

let root = null;
let session = null;

export function initLearn(container) {
  root = container;
  renderHub();
}

export function refreshLearn() {
  if (session) return;          // never yank a session out from under the user
  renderHub();
}

/* ══ Hub ════════════════════════════════════════════════════════════════ */

function renderHub() {
  session = null;
  const lang = state.settings.targetLang;
  const meta = LANGUAGES[lang];
  const words = langProgress(lang);
  const known = Object.keys(words);
  const due = dueCount(lang);
  const stats = getStats();
  const goalPct = Math.min(1, stats.todayReviews / Math.max(1, stats.dailyGoal));

  root.innerHTML = `
    <header class="view-header">
      <div class="grow">
        <div class="footnote">${esc(meta.flag)} ${esc(meta.name)}</div>
        <h1 class="large-title">Learn</h1>
      </div>
      <button class="icon-btn" id="learn-lang" aria-label="Change language">${icon('globe')}</button>
    </header>

    <div class="scroll grow view-scroll">
      <section class="card card-pad">
        <div class="row between" style="margin-bottom:var(--s-3)">
          <div>
            <div class="section-label" style="padding:0">Today's goal</div>
            <div class="title-3 tabular">${stats.todayReviews} / ${stats.dailyGoal} reviews</div>
          </div>
          <div class="row gap-1 accent-text" style="font-weight:700">
            ${icon('flame')}<span class="tabular">${stats.streak}</span>
          </div>
        </div>
        <div class="bar"><div class="bar-fill" style="width:${goalPct * 100}%"></div></div>
      </section>

      <button class="card card-pad row between" id="start-due" style="width:100%;text-align:left">
        <div class="row gap-3">
          <div class="list-row-icon" style="background:color-mix(in srgb,var(--accent) 20%,transparent);color:var(--accent)">
            ${icon('cards')}
          </div>
          <div>
            <div class="headline">${due ? `${due} card${due === 1 ? '' : 's'} due` : 'All caught up'}</div>
            <div class="footnote">${due
              ? 'Review now to keep them in memory'
              : known.length ? 'Study ahead, or go find new words' : 'Point the camera at objects to collect words'}</div>
          </div>
        </div>
        <span class="btn btn-primary btn-sm">${due ? 'Review' : 'Study'}</span>
      </button>

      <section>
        <div class="section-label">Decks by category</div>
        <div class="deck-grid">${deckCards(words)}</div>
      </section>

      <section>
        <div class="row between" style="align-items:baseline">
          <div class="section-label">Vocabulary · ${known.length}/${ALL_CLASSES.length}</div>
          ${known.length ? `<button class="btn-plain btn-sm" id="sort-toggle">Sort</button>` : ''}
        </div>
        <div class="group" id="vocab-list">${vocabRows(words, known)}</div>
      </section>
    </div>`;

  $('#learn-lang').addEventListener('click', () => import('./camera.js').then((m) => m.openLanguagePicker()));
  $('#start-due').addEventListener('click', () => startSession({}));
  $('#sort-toggle')?.addEventListener('click', cycleSort);

  $$('[data-deck]', root).forEach((n) =>
    n.addEventListener('click', () => startSession({ category: n.dataset.deck, includeUndiscovered: true })));
  $$('[data-word]', root).forEach((n) =>
    n.addEventListener('click', () => openWordSheet(n.dataset.word)));
}

let sortMode = 'recent';
const SORTS = ['recent', 'alpha', 'mastery'];

function cycleSort() {
  sortMode = SORTS[(SORTS.indexOf(sortMode) + 1) % SORTS.length];
  haptic('select');
  toast(`Sorted by ${sortMode}`);
  renderHub();
}

function deckCards(words) {
  return Object.entries(CATEGORIES).map(([key, cat]) => {
    const all = classesInCategory(key);
    const have = all.filter((c) => words[c]).length;
    return `
      <button class="deck-card" data-deck="${key}" style="--deck-color:${cat.color}">
        <span class="em">${cat.em}</span>
        <span class="name">${esc(cat.label)}</span>
        <span class="meta">${have}/${all.length} discovered</span>
        <span class="bar" style="margin-top:2px"><span class="bar-fill"
          style="width:${all.length ? (have / all.length) * 100 : 0}%;background:${cat.color}"></span></span>
      </button>`;
  }).join('');
}

function vocabRows(words, known) {
  if (!known.length) {
    return `<div class="empty">
      <div class="empty-em">🔍</div>
      <div class="empty-title">No words yet</div>
      <div class="empty-text">Open the Camera tab and point it at things around you. Every object you find is added here automatically.</div>
    </div>`;
  }
  const lang = state.settings.targetLang;
  const sorted = [...known].sort((a, b) => {
    if (sortMode === 'alpha') return a.localeCompare(b);
    if (sortMode === 'mastery') return masteryLevel(words[b]) - masteryLevel(words[a]);
    return words[b].discoveredAt - words[a].discoveredAt;
  });

  return sorted.map((cls) => {
    const t = translate(cls, lang);
    if (!t) return '';
    const rec = words[cls];
    const g = genderLabel(lang, t.gender);
    return `
      <button class="list-row is-tappable vocab-row" data-word="${esc(cls)}">
        <div class="list-row-icon">${esc(DICT[cls].em)}</div>
        <div class="list-row-body">
          <div class="word" ${LANGUAGES[lang].rtl ? 'dir="rtl"' : ''}>${esc(t.word)}
            ${g ? `<span class="gender-tag" style="margin-left:6px">${esc(g)}</span>` : ''}</div>
          <div class="gloss">${esc(cls)}${rec.srs.reps ? ' · next ' + esc(formatDue(rec.srs.due)) : ''}</div>
        </div>
        <div class="mastery">${[1, 2, 3, 4, 5].map((i) =>
          `<i class="${i <= masteryLevel(rec) ? 'on' : ''}"></i>`).join('')}</div>
      </button>`;
  }).join('');
}

/* ══ Session ════════════════════════════════════════════════════════════ */

function startSession(opts) {
  unlockAudio();
  const lang = state.settings.targetLang;
  const queue = buildSession({ lang, limit: 20, ...opts });

  if (!queue.length) {
    toast('Nothing to study here yet — go find some words!', { emoji: '🔍' });
    return;
  }

  session = {
    lang,
    queue,
    index: 0,
    correct: 0,
    run: 0,
    bestRun: 0,
    answered: false,
    category: opts.category || null
  };
  haptic('medium');
  renderQuestion();
}

function endSession() {
  const s = session;
  session = null;
  if (!s) { renderHub(); return; }

  const total = s.queue.length;
  const perfect = s.correct === total;
  recordBestRun(s.bestRun);
  const fresh = refreshBadges();
  cue(perfect ? 'levelup' : 'correct', 'success');

  root.innerHTML = `
    <div class="review-stage" style="justify-content:center;align-items:center;text-align:center;gap:var(--s-4)">
      <div style="font-size:64px">${perfect ? '🏆' : s.correct / total >= 0.7 ? '🎉' : '💪'}</div>
      <h2 class="large-title">${perfect ? 'Perfect session' : 'Session complete'}</h2>
      <p class="subhead">${s.correct} of ${total} correct${s.bestRun > 2 ? ` · best run ${s.bestRun}` : ''}</p>
      <div class="stat-grid" style="width:100%;max-width:420px;margin-top:var(--s-2)">
        <div class="stat-tile"><div class="k">Accuracy</div><div class="v">${Math.round(s.correct / total * 100)}%</div></div>
        <div class="stat-tile"><div class="k">Streak</div><div class="v">${getStats().streak}</div></div>
      </div>
      ${fresh.length ? `<div class="row wrap gap-2 center" style="margin-top:var(--s-2)">
        ${fresh.map((b) => `<span class="chip is-on">${b.em} ${esc(b.title)}</span>`).join('')}</div>` : ''}
      <div class="col gap-2" style="width:min(340px,100%);margin-top:var(--s-4)">
        <button class="btn btn-primary btn-block btn-lg" id="again">Study more</button>
        <button class="btn btn-block" id="done">Done</button>
      </div>
    </div>`;

  $('#again').addEventListener('click', () => startSession(s.category ? { category: s.category, includeUndiscovered: true } : {}));
  $('#done').addEventListener('click', renderHub);
}

function currentMode() {
  const pref = state.settings.reviewStyle;
  if (pref !== 'mixed') return pref;
  // Rotate deterministically so a session always covers every format.
  let mode = MODES[session.index % MODES.length];
  if (mode === 'listen' && !hasVoice(session.lang)) mode = 'choice';
  return mode;
}

function renderQuestion() {
  const s = session;
  if (s.index >= s.queue.length) { endSession(); return; }

  const cls = s.queue[s.index];
  const t = translate(cls, s.lang);
  if (!t) { s.index++; renderQuestion(); return; }

  const mode = currentMode();
  s.answered = false;
  s.mode = mode;

  root.innerHTML = `
    <header class="view-header" style="padding-bottom:0">
      <button class="icon-btn" id="quit" aria-label="End session">${icon('close')}</button>
      <div class="grow bar" style="margin-bottom:6px">
        <div class="bar-fill" style="width:${(s.index / s.queue.length) * 100}%"></div>
      </div>
      <div class="footnote tabular" style="margin-bottom:4px">${s.index + 1}/${s.queue.length}</div>
    </header>
    <div class="review-stage">
      <div id="card-slot" class="grow" style="display:flex;flex-direction:column;min-height:0"></div>
      <div id="answer-slot"></div>
    </div>`;

  $('#quit').addEventListener('click', async () => {
    if (s.index === 0 || await confirmAction({
      title: 'End session?',
      message: `You've answered ${s.index} of ${s.queue.length} cards. Progress on answered cards is saved.`,
      confirmLabel: 'End session'
    })) { session = null; renderHub(); }
  });

  const slot = $('#card-slot');
  const answers = $('#answer-slot');

  if (mode === 'recall') renderRecall(slot, answers, cls, t);
  else if (mode === 'choice') renderChoice(slot, answers, cls, t);
  else if (mode === 'listen') renderListen(slot, answers, cls, t);
  else renderType(slot, answers, cls, t);
}

function advance(correct, gradeKey, cls) {
  const s = session;
  if (s.answered) return;
  s.answered = true;

  grade(cls, gradeKey, s.lang);
  if (correct) { s.correct++; s.run++; s.bestRun = Math.max(s.bestRun, s.run); }
  else s.run = 0;

  cue(correct ? 'correct' : 'wrong', correct ? 'success' : 'error');
  setTimeout(() => { s.index++; renderQuestion(); }, correct ? 520 : 1150);
}

/* ── Mode: recall (self-graded, the classic SRS flow) ─────────────────── */

function renderRecall(slot, answers, cls, t) {
  const rtl = LANGUAGES[session.lang].rtl;
  const card = el('div', { class: 'flashcard' });
  card.innerHTML = `
    <div class="em">${esc(DICT[cls].em)}</div>
    <div class="prompt">How do you say</div>
    <div class="term">${esc(cls)}</div>
    <div class="sub">in ${esc(LANGUAGES[session.lang].name)}?</div>
    <div id="reveal" class="hidden" style="margin-top:var(--s-4)">
      <div class="term accent-text" ${rtl ? 'dir="rtl"' : ''}>${esc(t.word)}</div>
      ${t.phonetic ? `<div class="phon">${esc(t.phonetic)}</div>` : ''}
      ${genderLabel(session.lang, t.gender) ? `<div style="margin-top:var(--s-2)">
        <span class="gender-tag">${esc(genderLabel(session.lang, t.gender))}</span></div>` : ''}
    </div>`;
  slot.append(card);

  const show = el('button', { class: 'btn btn-primary btn-block btn-lg', text: 'Show answer' });
  answers.append(show);

  show.addEventListener('click', () => {
    $('#reveal').classList.remove('hidden');
    haptic('light');
    speak(citationForm(session.lang, t.word, t.gender), session.lang);
    answers.innerHTML = '';
    answers.append(gradeButtons(cls));
  });
}

function gradeButtons(cls) {
  const iv = previewIntervals(wordRecord(cls, session.lang));
  const row = el('div', { class: 'grade-row' });
  row.innerHTML = `
    <button class="grade-btn" data-grade="again">Again<span class="when">&lt; 10 min</span></button>
    <button class="grade-btn" data-grade="hard">Hard<span class="when">${iv.hard}d</span></button>
    <button class="grade-btn" data-grade="good">Good<span class="when">${iv.good}d</span></button>`;
  $$('[data-grade]', row).forEach((b) =>
    b.addEventListener('click', () => advance(b.dataset.grade !== 'again', b.dataset.grade, cls)));
  return row;
}

/* ── Mode: multiple choice ────────────────────────────────────────────── */

function renderChoice(slot, answers, cls, t) {
  const rtl = LANGUAGES[session.lang].rtl;
  const card = el('div', { class: 'flashcard' });
  card.innerHTML = `
    <div class="prompt">What does this mean?</div>
    <div class="term" ${rtl ? 'dir="rtl"' : ''}>${esc(t.word)}</div>
    ${state.settings.showPhonetics && t.phonetic ? `<div class="phon">${esc(t.phonetic)}</div>` : ''}`;
  slot.append(card);

  const options = shuffle([cls, ...distractors(cls, 3)]);
  answers.append(choiceList(options, cls, (c) => `${DICT[c].em}|${c}`));
}

function choiceList(options, correctCls, fmt) {
  const wrap = el('div', { class: 'choices' });
  options.forEach((c) => {
    const [em, label] = fmt(c).split('|');
    const btn = el('button', { class: 'choice' });
    btn.innerHTML = `<span class="em">${esc(em)}</span><span class="txt">${esc(label)}</span>
                     <span class="choice-mark"></span>`;
    btn.addEventListener('click', () => {
      if (session.answered) return;
      const right = c === correctCls;
      $$('.choice', wrap).forEach((b) => {
        const isTarget = b === btn;
        if (isTarget && right) { b.classList.add('is-correct'); b.querySelector('.choice-mark').textContent = '✓'; }
        else if (isTarget) { b.classList.add('is-wrong'); b.querySelector('.choice-mark').textContent = '✕'; }
        else b.classList.add('is-muted');
      });
      if (!right) {
        // Always reveal the right answer — a wrong guess should still teach.
        const idx = options.indexOf(correctCls);
        const target = $$('.choice', wrap)[idx];
        target?.classList.remove('is-muted');
        target?.classList.add('is-correct');
        if (target) target.querySelector('.choice-mark').textContent = '✓';
        $('.flashcard')?.classList.add('is-wrong');
      }
      advance(right, right ? 'good' : 'again', correctCls);
    });
    wrap.append(btn);
  });
  return wrap;
}

/* ── Mode: listening ──────────────────────────────────────────────────── */

function renderListen(slot, answers, cls, t) {
  const card = el('div', { class: 'flashcard' });
  card.innerHTML = `
    <div class="prompt">Listen and choose</div>
    <button class="icon-btn icon-btn-lg" id="replay"
            style="width:88px;height:88px;background:var(--accent);color:var(--accent-contrast)"
            aria-label="Play the word again">${icon('speaker')}</button>
    <div class="sub">Tap to hear it again</div>`;
  slot.append(card);

  const say = () => speak(citationForm(session.lang, t.word, t.gender), session.lang);
  $('#replay', card).addEventListener('click', () => { haptic('light'); say(); });
  setTimeout(say, 320);

  const options = shuffle([cls, ...distractors(cls, 3)]);
  answers.append(choiceList(options, cls, (c) => {
    const tr = translate(c, session.lang);
    return `${DICT[c].em}|${tr ? tr.word : c}`;
  }));
}

/* ── Mode: typing ─────────────────────────────────────────────────────── */

function renderType(slot, answers, cls, t) {
  const card = el('div', { class: 'flashcard' });
  card.innerHTML = `
    <div class="em">${esc(DICT[cls].em)}</div>
    <div class="prompt">Type it in ${esc(LANGUAGES[session.lang].name)}</div>
    <div class="term">${esc(cls)}</div>
    <div id="type-feedback" class="sub"></div>`;
  slot.append(card);

  const input = el('input', {
    class: 'type-input', type: 'text', autocomplete: 'off', autocapitalize: 'off',
    autocorrect: 'off', spellcheck: 'false', placeholder: 'Your answer',
    'aria-label': `Type the ${LANGUAGES[session.lang].name} word for ${cls}`
  });
  if (LANGUAGES[session.lang].rtl) input.setAttribute('dir', 'rtl');

  const submit = el('button', { class: 'btn btn-primary btn-block btn-lg', text: 'Check', style: 'margin-top:var(--s-3)' });
  const skip = el('button', { class: 'btn btn-block', text: "I don't know", style: 'margin-top:var(--s-2)' });

  const check = () => {
    if (session.answered) return;
    const right = normalise(input.value) === normalise(t.word);
    const fb = $('#type-feedback', card);
    fb.innerHTML = right
      ? `<span style="color:var(--green);font-weight:650">Correct</span>`
      : `<span style="color:var(--red);font-weight:650">${esc(t.word)}</span>`;
    input.blur();
    if (!right) card.classList.add('is-wrong');
    speak(citationForm(session.lang, t.word, t.gender), session.lang);
    advance(right, right ? 'good' : 'again', cls);
  };

  submit.addEventListener('click', check);
  skip.addEventListener('click', () => {
    if (session.answered) return;
    $('#type-feedback', card).innerHTML = `<span class="accent-text" style="font-weight:650">${esc(t.word)}</span>`;
    speak(citationForm(session.lang, t.word, t.gender), session.lang);
    advance(false, 'again', cls);
  });
  input.addEventListener('keydown', (e) => { if (e.key === 'Enter') check(); });

  answers.append(input, submit, skip);
  setTimeout(() => input.focus({ preventScroll: true }), 120);
}

/** Compare answers case-, accent- and punctuation-insensitively. */
function normalise(s) {
  return String(s || '')
    .trim()
    .toLowerCase()
    .normalize('NFD')
    .replace(/[̀-ͯ]/g, '')     // strip combining accents
    .replace(/[^\p{L}\p{N}\s]/gu, '')    // strip punctuation, keep any script
    .replace(/\s+/g, ' ');
}

export function isSessionActive() { return !!session; }
