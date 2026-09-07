/**
 * learn.js — Study hub and review sessions.
 *
 * Two states share the tab: a hub (due count, search, decks, vocabulary list)
 * and an active session. Sessions support five question types; 'mixed' rotates
 * between them, which is both more effective and less monotonous than drilling
 * one format.
 */

import { LANGUAGES, genderLabel, citationForm, buildPhrase, PHRASE_LABELS, glossFor } from '../data/languages.js';
import {
  itemMeta, translateItem, idsInGroup, GROUPS, GROUP_KEYS,
  searchItems, TOTAL_ITEMS, SOURCE_OBJ
} from '../data/vocab.js';
import { state, langProgress, wordRecord, refreshBadges, getStats, recordBestRun } from '../core/store.js';
import {
  buildSession, grade, dueCount, previewIntervals, distractors, shuffle,
  masteryLevel, formatDue
} from '../core/srs.js';
import { speak, hasVoice } from '../core/speech.js';
import { cue, haptic, unlockAudio } from '../core/feedback.js';
import { $, $$, el, esc, toast, confirmAction } from '../ui/kit.js';
import { icon } from '../ui/icons.js';
import { openWordSheet } from './wordSheet.js';

const MODES = ['recall', 'choice', 'listen', 'cloze', 'type'];

let root = null;
let session = null;
let sortMode = 'recent';
let searchQuery = '';
const SORTS = ['recent', 'alpha', 'mastery'];

export function initLearn(container) {
  root = container;
  renderHub();
}

export function refreshLearn() {
  if (session) return;          // never yank a session out from under the user
  renderHub();
}

export function isSessionActive() { return !!session; }

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

  const cameraGroups = GROUP_KEYS.filter((g) => GROUPS[g].source === SOURCE_OBJ);
  const coreGroups = GROUP_KEYS.filter((g) => GROUPS[g].source !== SOURCE_OBJ);

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
              : known.length ? 'Study ahead, or open a deck below' : 'Open a deck, or find words with the camera'}</div>
          </div>
        </div>
        <span class="btn btn-primary btn-sm">${due ? 'Review' : 'Study'}</span>
      </button>

      <section>
        <div class="search-wrap">
          ${icon('search')}
          <input class="search-input" id="vocab-search" type="search" placeholder="Search ${TOTAL_ITEMS} words…"
                 value="${esc(searchQuery)}" autocomplete="off" spellcheck="false"
                 aria-label="Search vocabulary">
          ${searchQuery ? `<button class="icon-btn search-clear" id="search-clear" aria-label="Clear search">${icon('close')}</button>` : ''}
        </div>
        <div id="search-results"></div>
      </section>

      <div id="hub-body">
        <section>
          <div class="section-label">Find these with your camera</div>
          <div class="deck-grid">${deckCards(cameraGroups, words)}</div>
        </section>

        <section>
          <div class="section-label">Core vocabulary</div>
          <div class="deck-grid">${deckCards(coreGroups, words)}</div>
        </section>

        <section>
          <div class="row between" style="align-items:baseline">
            <div class="section-label">Your words · ${known.length}/${TOTAL_ITEMS}</div>
            ${known.length ? '<button class="btn-plain btn-sm" id="sort-toggle">Sort</button>' : ''}
          </div>
          <div class="group" id="vocab-list">${vocabRows(words, known)}</div>
        </section>
      </div>
    </div>`;

  $('#learn-lang').addEventListener('click', () => import('./camera.js').then((m) => m.openLanguagePicker()));
  $('#start-due').addEventListener('click', () => startSession({}));
  $('#sort-toggle')?.addEventListener('click', cycleSort);

  $$('[data-deck]', root).forEach((n) =>
    n.addEventListener('click', () => startSession({ group: n.dataset.deck, includeUndiscovered: true })));
  $$('[data-word]', root).forEach((n) =>
    n.addEventListener('click', () => openWordSheet(n.dataset.word)));

  wireSearch();
}

function wireSearch() {
  const input = $('#vocab-search', root);
  if (!input) return;
  let timer = null;
  input.addEventListener('input', () => {
    clearTimeout(timer);
    timer = setTimeout(() => { searchQuery = input.value; paintSearch(); }, 120);
  });
  $('#search-clear', root)?.addEventListener('click', () => {
    searchQuery = '';
    input.value = '';
    paintSearch();
    input.focus();
  });
  if (searchQuery) paintSearch();
}

function paintSearch() {
  const out = $('#search-results', root);
  const hub = $('#hub-body', root);
  if (!out || !hub) return;

  const q = searchQuery.trim();
  const clear = $('#search-clear', root);
  if (clear) clear.hidden = !q;

  if (!q) { out.innerHTML = ''; hub.hidden = false; return; }
  hub.hidden = true;

  const lang = state.settings.targetLang;
  const hits = searchItems(q, lang, 60);
  if (!hits.length) {
    out.innerHTML = `<div class="empty">
      <div class="empty-em">🔎</div>
      <div class="empty-title">No match for “${esc(q)}”</div>
      <div class="empty-text">Try the English word, or the ${esc(LANGUAGES[lang].name)} one.</div>
    </div>`;
    return;
  }

  const words = langProgress(lang);
  out.innerHTML = `<div class="group" style="margin-top:var(--s-3)">${hits.map(({ id, meta, tr }) => {
    const g = genderLabel(lang, tr.gender);
    const rec = words[id];
    return `
      <button class="list-row is-tappable" data-word="${esc(id)}">
        <div class="list-row-icon">${esc(tr.em)}</div>
        <div class="list-row-body">
          <div class="word" ${LANGUAGES[lang].rtl ? 'dir="rtl"' : ''}>${esc(tr.word)}
            ${g ? `<span class="gender-tag" style="margin-left:6px">${esc(g)}</span>` : ''}</div>
          <div class="gloss">${esc(meta.en)}${GROUPS[meta.group] ? ' · ' + esc(GROUPS[meta.group].label) : ''}</div>
        </div>
        ${rec ? `<div class="mastery">${[1, 2, 3, 4, 5].map((i) =>
          `<i class="${i <= masteryLevel(rec) ? 'on' : ''}"></i>`).join('')}</div>`
        : `<div class="list-row-chevron">${icon('chevronRight')}</div>`}
      </button>`;
  }).join('')}</div>`;

  $$('[data-word]', out).forEach((n) =>
    n.addEventListener('click', () => openWordSheet(n.dataset.word)));
}

function cycleSort() {
  sortMode = SORTS[(SORTS.indexOf(sortMode) + 1) % SORTS.length];
  haptic('select');
  toast(`Sorted by ${sortMode}`);
  renderHub();
}

function deckCards(groups, words) {
  return groups.map((key) => {
    const g = GROUPS[key];
    const ids = idsInGroup(key);
    const have = ids.filter((id) => words[id]).length;
    return `
      <button class="deck-card" data-deck="${esc(key)}" style="--deck-color:${g.color}">
        <span class="em">${g.em}</span>
        <span class="name">${esc(g.label)}</span>
        <span class="meta">${have}/${ids.length} started</span>
        <span class="bar" style="margin-top:2px"><span class="bar-fill"
          style="width:${ids.length ? (have / ids.length) * 100 : 0}%;background:${g.color}"></span></span>
      </button>`;
  }).join('');
}

function vocabRows(words, known) {
  if (!known.length) {
    return `<div class="empty">
      <div class="empty-em">🌱</div>
      <div class="empty-title">Nothing started yet</div>
      <div class="empty-text">Open a deck above to begin, or point the camera at something and Lemma will add it here.</div>
    </div>`;
  }
  const lang = state.settings.targetLang;
  const sorted = [...known].sort((a, b) => {
    if (sortMode === 'alpha') {
      return (itemMeta(a)?.en || '').localeCompare(itemMeta(b)?.en || '');
    }
    if (sortMode === 'mastery') return masteryLevel(words[b]) - masteryLevel(words[a]);
    return words[b].discoveredAt - words[a].discoveredAt;
  });

  return sorted.map((id) => {
    const meta = itemMeta(id);
    const t = translateItem(id, lang);
    if (!meta || !t) return '';
    const rec = words[id];
    const g = genderLabel(lang, t.gender);
    return `
      <button class="list-row is-tappable vocab-row" data-word="${esc(id)}">
        <div class="list-row-icon">${esc(t.em)}</div>
        <div class="list-row-body">
          <div class="word" ${LANGUAGES[lang].rtl ? 'dir="rtl"' : ''}>${esc(t.word)}
            ${g ? `<span class="gender-tag" style="margin-left:6px">${esc(g)}</span>` : ''}</div>
          <div class="gloss">${esc(meta.en)}${rec.srs.reps ? ' · next ' + esc(formatDue(rec.srs.due)) : ''}</div>
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
    toast('Nothing to study here yet', { emoji: '🔍' });
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
    group: opts.group || null
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

  $('#again').addEventListener('click', () =>
    startSession(s.group ? { group: s.group, includeUndiscovered: true } : {}));
  $('#done').addEventListener('click', renderHub);
}

function currentMode(id) {
  const pref = state.settings.reviewStyle;
  let mode = pref !== 'mixed' ? pref : MODES[session.index % MODES.length];
  // Fall back when a mode cannot be presented honestly for this card.
  if (mode === 'listen' && !hasVoice(session.lang)) mode = 'choice';
  if (mode === 'cloze' && !clozeFor(id)) mode = 'choice';
  return mode;
}

/** A cloze needs a sentence that actually contains the target word. */
function clozeFor(id) {
  const t = translateItem(id, session.lang);
  const meta = itemMeta(id);
  if (!t || !meta) return null;
  // Only nouns slot cleanly into the sentence frames.
  if (meta.group === 'verbs' || meta.group === 'questions' ||
      meta.group === 'adjectives' || meta.group === 'greetings') return null;

  const frames = Object.keys(PHRASE_LABELS);
  for (const frame of shuffle(frames)) {
    const sentence = buildPhrase(session.lang, t.word, t.gender, frame);
    if (sentence && sentence.includes(t.word)) {
      return { sentence, frame, word: t.word };
    }
  }
  return null;
}

function renderQuestion() {
  const s = session;
  if (s.index >= s.queue.length) { endSession(); return; }

  const id = s.queue[s.index];
  const meta = itemMeta(id);
  const t = translateItem(id, s.lang);
  if (!meta || !t) { s.index++; renderQuestion(); return; }

  const mode = currentMode(id);
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

  if (mode === 'recall') renderRecall(slot, answers, id, meta, t);
  else if (mode === 'choice') renderChoice(slot, answers, id, meta, t);
  else if (mode === 'listen') renderListen(slot, answers, id, meta, t);
  else if (mode === 'cloze') renderCloze(slot, answers, id, meta, t);
  else renderType(slot, answers, id, meta, t);
}

function advance(correct, gradeKey, id) {
  const s = session;
  if (s.answered) return;
  s.answered = true;

  grade(id, gradeKey, s.lang);
  if (correct) { s.correct++; s.run++; s.bestRun = Math.max(s.bestRun, s.run); }
  else s.run = 0;

  cue(correct ? 'correct' : 'wrong', correct ? 'success' : 'error');
  setTimeout(() => { s.index++; renderQuestion(); }, correct ? 520 : 1150);
}

/* ── Mode: recall (self-graded, the classic SRS flow) ─────────────────── */

function renderRecall(slot, answers, id, meta, t) {
  const rtl = LANGUAGES[session.lang].rtl;
  const card = el('div', { class: 'flashcard' });
  card.innerHTML = `
    <div class="em">${esc(t.em)}</div>
    <div class="prompt">How do you say</div>
    <div class="term">${esc(meta.en)}</div>
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
    answers.append(gradeButtons(id));
  });
}

function gradeButtons(id) {
  const iv = previewIntervals(wordRecord(id, session.lang));
  const row = el('div', { class: 'grade-row' });
  row.innerHTML = `
    <button class="grade-btn" data-grade="again">Again<span class="when">&lt; 10 min</span></button>
    <button class="grade-btn" data-grade="hard">Hard<span class="when">${iv.hard}d</span></button>
    <button class="grade-btn" data-grade="good">Good<span class="when">${iv.good}d</span></button>`;
  $$('[data-grade]', row).forEach((b) =>
    b.addEventListener('click', () => advance(b.dataset.grade !== 'again', b.dataset.grade, id)));
  return row;
}

/* ── Mode: multiple choice ────────────────────────────────────────────── */

function renderChoice(slot, answers, id, meta, t) {
  const rtl = LANGUAGES[session.lang].rtl;
  const card = el('div', { class: 'flashcard' });
  card.innerHTML = `
    <div class="prompt">What does this mean?</div>
    <div class="term" ${rtl ? 'dir="rtl"' : ''}>${esc(t.word)}</div>
    ${state.settings.showPhonetics && t.phonetic ? `<div class="phon">${esc(t.phonetic)}</div>` : ''}`;
  slot.append(card);

  const options = shuffle([id, ...distractors(id, 3)]);
  answers.append(choiceList(options, id, (x) => {
    const m = itemMeta(x);
    return { em: m ? m.em : '•', label: m ? m.en : x };
  }));
}

function choiceList(options, correctId, fmt, onAnswer = null) {
  const wrap = el('div', { class: 'choices' });
  options.forEach((optId) => {
    const { em, label } = fmt(optId);
    const btn = el('button', { class: 'choice' });
    btn.innerHTML = `<span class="em">${esc(em)}</span><span class="txt">${esc(label)}</span>
                     <span class="choice-mark"></span>`;
    btn.addEventListener('click', () => {
      if (session.answered) return;
      const right = optId === correctId;
      $$('.choice', wrap).forEach((b) => {
        const isTarget = b === btn;
        if (isTarget && right) { b.classList.add('is-correct'); b.querySelector('.choice-mark').textContent = '✓'; }
        else if (isTarget) { b.classList.add('is-wrong'); b.querySelector('.choice-mark').textContent = '✕'; }
        else b.classList.add('is-muted');
      });
      if (!right) {
        // Always reveal the right answer — a wrong guess should still teach.
        const idx = options.indexOf(correctId);
        const target = $$('.choice', wrap)[idx];
        target?.classList.remove('is-muted');
        target?.classList.add('is-correct');
        if (target) target.querySelector('.choice-mark').textContent = '✓';
        $('.flashcard')?.classList.add('is-wrong');
      }
      if (onAnswer) onAnswer(right);
      advance(right, right ? 'good' : 'again', correctId);
    });
    wrap.append(btn);
  });
  return wrap;
}

/* ── Mode: listening ──────────────────────────────────────────────────── */

function renderListen(slot, answers, id, meta, t) {
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

  const options = shuffle([id, ...distractors(id, 3)]);
  answers.append(choiceList(options, id, (x) => {
    const tr = translateItem(x, session.lang);
    const m = itemMeta(x);
    return { em: m ? m.em : '•', label: tr ? tr.word : (m ? m.en : x) };
  }));
}

/* ── Mode: cloze (fill the gap in a real sentence) ────────────────────── */

function renderCloze(slot, answers, id, meta, t) {
  const cloze = clozeFor(id);
  if (!cloze) { renderChoice(slot, answers, id, meta, t); return; }

  const rtl = LANGUAGES[session.lang].rtl;
  const blanked = cloze.sentence.replace(cloze.word, '<span class="cloze-gap">?</span>');

  const card = el('div', { class: 'flashcard' });
  card.innerHTML = `
    <div class="em">${esc(t.em)}</div>
    <div class="prompt">Complete the sentence</div>
    <div class="cloze-sentence" ${rtl ? 'dir="rtl"' : ''}>${blanked}</div>
    <div class="sub">${esc(glossFor(cloze.frame, meta.en))}</div>`;
  slot.append(card);

  const options = shuffle([id, ...distractors(id, 3)]);
  answers.append(choiceList(options, id, (x) => {
    const tr = translateItem(x, session.lang);
    const m = itemMeta(x);
    return { em: m ? m.em : '•', label: tr ? tr.word : (m ? m.en : x) };
  }, () => {
    // Fill the gap in either case — a wrong guess should still show the answer.
    const gap = $('.cloze-gap', card);
    if (!gap) return;
    gap.textContent = cloze.word;
    gap.classList.add('is-filled');
    speak(cloze.sentence, session.lang);
  }));
}

/* ── Mode: typing ─────────────────────────────────────────────────────── */

function renderType(slot, answers, id, meta, t) {
  const card = el('div', { class: 'flashcard' });
  card.innerHTML = `
    <div class="em">${esc(t.em)}</div>
    <div class="prompt">Type it in ${esc(LANGUAGES[session.lang].name)}</div>
    <div class="term">${esc(meta.en)}</div>
    <div id="type-feedback" class="sub"></div>`;
  slot.append(card);

  const input = el('input', {
    class: 'type-input', type: 'text', autocomplete: 'off', autocapitalize: 'off',
    autocorrect: 'off', spellcheck: 'false', placeholder: 'Your answer',
    'aria-label': `Type the ${LANGUAGES[session.lang].name} word for ${meta.en}`
  });
  if (LANGUAGES[session.lang].rtl) input.setAttribute('dir', 'rtl');

  const submit = el('button', { class: 'btn btn-primary btn-block btn-lg', text: 'Check', style: 'margin-top:var(--s-3)' });
  const skip = el('button', { class: 'btn btn-block', text: "I don't know", style: 'margin-top:var(--s-2)' });

  const check = () => {
    if (session.answered) return;
    const right = normalise(input.value) === normalise(t.word);
    const fb = $('#type-feedback', card);
    fb.innerHTML = right
      ? '<span style="color:var(--green);font-weight:650">Correct</span>'
      : `<span style="color:var(--red);font-weight:650">${esc(t.word)}</span>`;
    input.blur();
    if (!right) card.classList.add('is-wrong');
    speak(citationForm(session.lang, t.word, t.gender), session.lang);
    advance(right, right ? 'good' : 'again', id);
  };

  submit.addEventListener('click', check);
  skip.addEventListener('click', () => {
    if (session.answered) return;
    $('#type-feedback', card).innerHTML =
      `<span class="accent-text" style="font-weight:650">${esc(t.word)}</span>`;
    speak(citationForm(session.lang, t.word, t.gender), session.lang);
    advance(false, 'again', id);
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
    .replace(/[̀-ͯ]/g, '')               // strip combining accents
    .replace(/[^\p{L}\p{N}\s]/gu, '')     // strip punctuation, keep any script
    .replace(/\s+/g, ' ');
}
