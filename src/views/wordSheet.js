/**
 * wordSheet.js — The word detail sheet.
 *
 * Opened from an AR label, a detected card, the vocabulary list or search.
 * Shows the target word with its gender and phonetics, generated example
 * sentences, the learner's own history with the word, and every other language
 * side by side.
 *
 * When the camera has read a colour off the object, the sheet also shows the
 * agreeing colour phrase — the one grammar point the camera can teach directly.
 */

import { LANGUAGES, LANG_CODES, PHRASE_LABELS, buildPhrase, genderLabel, citationForm, buildColorPhrase, glossFor }
  from '../data/languages.js';
import { itemMeta, translateItem, GROUPS, isCameraItem, SOURCE_COL, objId } from '../data/vocab.js';
import { colorForms, COLORS } from '../data/colors.js';
import { DICT } from '../data/dictionary.js';
import { state, wordRecord } from '../core/store.js';
import { masteryLevel, formatDue } from '../core/srs.js';
import { speak, hasVoice } from '../core/speech.js';
import { openSheet, esc, $$, relativeDay } from '../ui/kit.js';
import { icon } from '../ui/icons.js';
import { cue } from '../core/feedback.js';

/**
 * @param {string} id         vocab item id
 * @param {string} [colorKey] colour the camera read, if any
 */
export function openWordSheet(id, colorKey = null) {
  const meta = itemMeta(id);
  if (!meta) return;
  const lang = state.settings.targetLang;
  const t = translateItem(id, lang);
  if (!t) return;

  const body = openSheet({ title: '' });
  body.innerHTML = renderSheet({ meta, t, lang, colorKey });

  $$('[data-speak]', body).forEach((btn) => {
    btn.addEventListener('click', () => {
      cue('tap', 'light');
      speak(btn.dataset.speak, btn.dataset.lang || lang);
    });
  });
}

function renderSheet({ meta, t, lang, colorKey }) {
  const gTag = genderLabel(lang, t.gender);
  const group = GROUPS[meta.group];
  const voiceOk = hasVoice(lang);
  const rec = wordRecord(meta.id, lang);
  const rtl = LANGUAGES[lang].rtl;
  const citation = citationForm(lang, t.word, t.gender);

  const hero = `
    <div class="word-hero">
      <div class="em">${esc(t.em)}</div>
      <div class="grow">
        <div class="foreign" ${rtl ? 'dir="rtl"' : ''}>${esc(t.word)}</div>
        <div class="english">${esc(meta.en)}</div>
        ${state.settings.showPhonetics && t.phonetic ? `<div class="phon">${esc(t.phonetic)}</div>` : ''}
      </div>
      <button class="icon-btn icon-btn-lg" data-speak="${esc(citation)}" data-lang="${lang}"
              aria-label="Pronounce ${esc(t.word)}"
              ${voiceOk ? '' : 'disabled title="No voice installed for this language"'}>
        ${icon('speaker')}
      </button>
    </div>`;

  const tags = `
    <div class="row wrap gap-2" style="margin-bottom:var(--s-5)">
      ${gTag ? `<span class="gender-tag">${esc(gTag)}</span>` : ''}
      ${group ? `<span class="chip">${esc(group.em)} ${esc(group.label)}</span>` : ''}
      ${isCameraItem(meta.id) ? `<span class="chip">${icon('camera')} Findable</span>` : ''}
      ${rec ? `<span class="chip">Seen ${rec.seen || 0}×</span>`
            : '<span class="chip">Not started</span>'}
    </div>`;

  /* Colour phrase — only shown when the camera actually saw a colour. */
  let colorBlock = '';
  if (colorKey && COLORS[colorKey]) {
    const forms = colorForms(colorKey, lang);
    const phrase = buildColorPhrase(lang, t.word, t.gender, forms);
    if (phrase && forms) {
      colorBlock = `
        <div class="section-label">Colour you're looking at</div>
        <div class="group" style="margin-bottom:var(--s-5)">
          <div class="phrase-row">
            <span class="color-dot" style="background:${COLORS[colorKey].swatch}"></span>
            <div class="body">
              <div class="target" ${rtl ? 'dir="rtl"' : ''}>${esc(phrase)}</div>
              <div class="gloss">the ${esc(colorKey)} ${esc(meta.en)} · adjective agrees with
                ${esc(gTag || 'the noun')}</div>
            </div>
            <button class="icon-btn" data-speak="${esc(phrase)}" data-lang="${lang}"
                    aria-label="Play phrase" ${voiceOk ? '' : 'disabled'}>${icon('speaker')}</button>
          </div>
          <div class="phrase-row">
            <span class="color-dot" style="background:${COLORS[colorKey].swatch}"></span>
            <div class="body">
              <div class="target">${esc(forms.cite)}</div>
              <div class="gloss">${esc(colorKey)}${forms.ph ? ' · ' + esc(forms.ph) : ''}</div>
            </div>
            <button class="icon-btn" data-speak="${esc(forms.cite)}" data-lang="${lang}"
                    aria-label="Play colour" ${voiceOk ? '' : 'disabled'}>${icon('speaker')}</button>
          </div>
        </div>`;
    }
  }

  /* A colour is an adjective, not a noun — running it through the noun frames
     would produce "This is a red." Show how it actually behaves instead:
     the same colour against nouns of each gender the language distinguishes. */
  if (meta.source === SOURCE_COL) {
    return renderColorSheet({ meta, t, lang, gTag, group, voiceOk, rec, rtl, hero, tags });
  }

  /* Example sentences, generated by the grammar engine rather than stored.
     "How much is the person?" is nonsense, so animate nouns skip that frame. */
  const animate = meta.group === 'people' || meta.group === 'animal' || meta.group === 'family';
  const frames = Object.keys(PHRASE_LABELS).filter((f) => !(animate && f === 'how_much'));

  const phrases = frames.map((frame) => {
    const sentence = buildPhrase(lang, t.word, t.gender, frame);
    return `
      <div class="phrase-row">
        <div class="body">
          <div class="target" ${rtl ? 'dir="rtl"' : ''}>${esc(sentence)}</div>
          <div class="gloss">${esc(glossFor(frame, meta.en))}</div>
        </div>
        <button class="icon-btn" data-speak="${esc(sentence)}" data-lang="${lang}"
                aria-label="Play sentence" ${voiceOk ? '' : 'disabled'}>${icon('speaker')}</button>
      </div>`;
  }).join('');

  const progress = rec ? `
    <div class="section-label">Your progress</div>
    <div class="group" style="margin-bottom:var(--s-5)">
      <div class="list-row">
        <div class="list-row-body"><div class="list-row-title">Mastery</div></div>
        <div class="mastery">${[1, 2, 3, 4, 5].map((i) =>
          `<i class="${i <= masteryLevel(rec) ? 'on' : ''}"></i>`).join('')}</div>
      </div>
      <div class="list-row">
        <div class="list-row-body"><div class="list-row-title">Next review</div></div>
        <div class="list-row-value">${rec.srs.reps ? esc(formatDue(rec.srs.due)) : 'not started'}</div>
      </div>
      <div class="list-row">
        <div class="list-row-body"><div class="list-row-title">Accuracy</div></div>
        <div class="list-row-value">${rec.correct + rec.wrong
          ? Math.round(rec.correct / (rec.correct + rec.wrong) * 100) + '%'
          : '—'}</div>
      </div>
      <div class="list-row">
        <div class="list-row-body"><div class="list-row-title">First seen</div></div>
        <div class="list-row-value">${esc(relativeDay(rec.discoveredAt))}</div>
      </div>
    </div>` : '';

  /* Every other language, so the sheet doubles as a mini phrasebook. */
  const others = LANG_CODES.filter((l) => l !== lang).map((l) => {
    const o = translateItem(meta.id, l);
    if (!o) return '';
    const lm = LANGUAGES[l];
    const g = genderLabel(l, o.gender);
    return `
      <button class="list-row is-tappable" data-speak="${esc(citationForm(l, o.word, o.gender))}" data-lang="${l}">
        <div class="list-row-icon" style="background:transparent;font-size:20px">${esc(lm.flag)}</div>
        <div class="list-row-body">
          <div class="list-row-title" ${lm.rtl ? 'dir="rtl"' : ''}>${esc(o.word)}
            ${g ? `<span class="gender-tag" style="margin-left:6px">${esc(g)}</span>` : ''}</div>
          <div class="list-row-sub">${esc(lm.name)}${o.phonetic ? ' · ' + esc(o.phonetic) : ''}</div>
        </div>
        <div class="list-row-chevron">${icon('speaker')}</div>
      </button>`;
  }).join('');

  return `
    ${hero}
    ${tags}
    ${!voiceOk ? `<p class="footnote" style="margin-bottom:var(--s-4)">
      ${icon('info')} No ${esc(LANGUAGES[lang].name)} speech voice is installed on this device — text is still shown.
    </p>` : ''}
    ${colorBlock}
    <div class="section-label">Use it in a sentence</div>
    <div class="group" style="margin-bottom:var(--s-5)">${phrases}</div>
    ${progress}
    <div class="section-label">In other languages</div>
    <div class="group">${others}</div>
    <div style="height:var(--s-4)"></div>`;
}

/* ── Colour sheet ────────────────────────────────────────────────────────
   Demonstration nouns, one per gender the language marks, so the learner sees
   the adjective actually change rather than being told that it does. */

const DEMO_NOUNS = ['car', 'apple', 'book', 'bed', 'dog', 'cup'];

function renderColorSheet({ meta, t, lang, group, voiceOk, rec, rtl, hero, tags }) {
  const forms = colorForms(meta.key, lang);
  const seenGenders = new Set();
  const rows = [];

  for (const cls of DEMO_NOUNS) {
    if (!DICT[cls]) continue;
    const noun = translateItem(objId(cls), lang);
    if (!noun || noun.gender === 'p') continue;
    // One example per distinct gender; languages without gender get one row.
    const key = noun.gender || 'none';
    if (seenGenders.has(key)) continue;
    const phrase = buildColorPhrase(lang, noun.word, noun.gender, forms);
    if (!phrase) continue;
    seenGenders.add(key);
    const gl = genderLabel(lang, noun.gender);
    rows.push(`
      <div class="phrase-row">
        <span class="color-dot" style="background:${COLORS[meta.key].swatch}"></span>
        <div class="body">
          <div class="target" ${rtl ? 'dir="rtl"' : ''}>${esc(phrase)}</div>
          <div class="gloss">the ${esc(meta.key)} ${esc(cls)}${gl ? ` · ${esc(gl)}` : ''}</div>
        </div>
        <button class="icon-btn" data-speak="${esc(phrase)}" data-lang="${lang}"
                aria-label="Play phrase" ${voiceOk ? '' : 'disabled'}>${icon('speaker')}</button>
      </div>`);
  }

  const formRows = [];
  if (forms.m && forms.f && forms.m !== forms.f) {
    formRows.push(['Masculine', forms.m], ['Feminine', forms.f]);
    if (forms.n && forms.n !== forms.m) formRows.push(['Neuter', forms.n]);
  } else if (forms.attr && forms.attr !== forms.cite) {
    formRows.push(['On its own', forms.cite], ['Before a noun', forms.attr]);
  } else if (forms.m && forms.f && forms.m === forms.f) {
    formRows.push(['All genders', forms.cite]);
  }

  const progress = rec ? `
    <div class="section-label">Your progress</div>
    <div class="group" style="margin-bottom:var(--s-5)">
      <div class="list-row">
        <div class="list-row-body"><div class="list-row-title">Mastery</div></div>
        <div class="mastery">${[1, 2, 3, 4, 5].map((i) =>
          `<i class="${i <= masteryLevel(rec) ? 'on' : ''}"></i>`).join('')}</div>
      </div>
      <div class="list-row">
        <div class="list-row-body"><div class="list-row-title">Next review</div></div>
        <div class="list-row-value">${rec.srs.reps ? esc(formatDue(rec.srs.due)) : 'not started'}</div>
      </div>
    </div>` : '';

  const others = LANG_CODES.filter((l) => l !== lang).map((l) => {
    const o = colorForms(meta.key, l);
    if (!o) return '';
    const lm = LANGUAGES[l];
    return `
      <button class="list-row is-tappable" data-speak="${esc(o.cite)}" data-lang="${l}">
        <div class="list-row-icon" style="background:transparent;font-size:20px">${esc(lm.flag)}</div>
        <div class="list-row-body">
          <div class="list-row-title" ${lm.rtl ? 'dir="rtl"' : ''}>${esc(o.cite)}</div>
          <div class="list-row-sub">${esc(lm.name)}${o.ph ? ' · ' + esc(o.ph) : ''}</div>
        </div>
        <div class="list-row-chevron">${icon('speaker')}</div>
      </button>`;
  }).join('');

  return `
    ${hero}
    ${tags}
    ${formRows.length ? `
      <div class="section-label">How it changes</div>
      <div class="group" style="margin-bottom:var(--s-5)">
        ${formRows.map(([label, form]) => `
          <div class="list-row">
            <div class="list-row-body"><div class="list-row-title">${esc(label)}</div></div>
            <div class="list-row-value" ${rtl ? 'dir="rtl"' : ''}>${esc(form)}</div>
          </div>`).join('')}
      </div>` : ''}
    ${rows.length ? `
      <div class="section-label">In use</div>
      <div class="group" style="margin-bottom:var(--s-5)">${rows.join('')}</div>` : ''}
    ${progress}
    <div class="section-label">In other languages</div>
    <div class="group">${others}</div>
    <div style="height:var(--s-4)"></div>`;
}
